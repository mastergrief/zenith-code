from __future__ import annotations

import ast
import inspect
import statistics
import time
from typing import Any, Callable

import pytest
import torch

from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    CLASSIFIER_S3A_VECTOR_PARITY_OK_COST_BOUNDED,
    S3A_EXPLICIT_NON_CLAIMS,
    W6_PACKED_MAX,
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
    clip_then_pack_w6_tensor,
    emit_s3a_classifier_receipt,
    pack_w6,
    pack_w6_tensor,
    strict_roundtrip_w6_tensor,
    unpack_w6,
    unpack_w6_tensor,
)

S3A_HOT_FUNCTIONS: tuple[str, ...] = (
    "pack_w6_tensor",
    "unpack_w6_tensor",
    "strict_roundtrip_w6_tensor",
)

S3A_TENSOR_SHAPES: tuple[tuple[int, ...], ...] = ((3, 3), (128,), (12288,))

S3A_COST_RATIO_BOUND = 10.0
S3A_COST_NUMEL = 12288
S3A_WARMUP_ITERS = 3
S3A_MEASURED_ITERS = 7

S3A_COST_POPULATIONS: tuple[dict[str, Any], ...] = (
    {"label": "tiny_fixture", "numel": 9, "shape": (3, 3)},
    {"label": "phase3_12k_class", "numel": 12288, "shape": (12288,)},
    {"label": "upper_bound_smoke", "numel": 262144, "shape": (262144,)},
)

S3A_LAST_COST_MODEL_RECEIPT: dict[str, Any] | None = None


def _scalar_roundtrip_tensor(acc: torch.Tensor) -> torch.Tensor:
    flat = acc.flatten().tolist()
    roundtripped = [unpack_w6(pack_w6(int(v))) for v in flat]
    return torch.tensor(roundtripped, dtype=torch.int16).reshape(acc.shape)


def _full_domain_values() -> list[int]:
    return list(range(W6_SIGNED_MIN, W6_SIGNED_MAX + 1))


def _make_domain_tensor(shape: tuple[int, ...]) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    values = (_full_domain_values() * ((numel // len(_full_domain_values())) + 1))[:numel]
    return torch.tensor(values, dtype=torch.int16).reshape(shape)


def _inspect_hot_function_source(name: str) -> str:
    from calm.hrm_text_158.native_full_stack import narrow_accumulator_codec as codec_mod

    fn = getattr(codec_mod, name)
    return inspect.getsource(fn)


def _assert_no_python_lane_loops(source: str, *, fn_name: str) -> None:
    forbidden_tokens = (".tolist()",)
    for token in forbidden_tokens:
        assert token not in source, f"{fn_name} must not use {token}"

    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.For):
            raise AssertionError(f"{fn_name} must not use Python for-loops over lanes")
        if isinstance(node, ast.While):
            raise AssertionError(f"{fn_name} must not use while-loops over lanes")
        if isinstance(node, ast.comprehension):
            raise AssertionError(f"{fn_name} must not use comprehensions over lanes")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            if node.func.attr == "item":
                raise AssertionError(f"{fn_name} must not call .item() in hot path")


def test_c1_bit_exact_vs_scalar_full_domain_and_shapes() -> None:
    for value in _full_domain_values():
        scalar_packed = pack_w6(value)
        tensor_packed = pack_w6_tensor(torch.tensor(value, dtype=torch.int16))
        assert int(tensor_packed.item()) == scalar_packed
        scalar_unpacked = unpack_w6(scalar_packed)
        tensor_unpacked = unpack_w6_tensor(tensor_packed.reshape(()))
        assert int(tensor_unpacked.item()) == scalar_unpacked

    for shape in S3A_TENSOR_SHAPES:
        acc = _make_domain_tensor(shape)
        packed = pack_w6_tensor(acc)
        unpacked = unpack_w6_tensor(packed)
        scalar_roundtrip = _scalar_roundtrip_tensor(acc)
        assert torch.equal(unpacked, scalar_roundtrip)
        assert torch.equal(strict_roundtrip_w6_tensor(acc), scalar_roundtrip)


def test_c2_shape_dtype_device_invariants() -> None:
    for shape in S3A_TENSOR_SHAPES:
        acc = _make_domain_tensor(shape)
        packed = pack_w6_tensor(acc)
        unpacked = unpack_w6_tensor(packed)
        roundtripped = strict_roundtrip_w6_tensor(acc)

        assert packed.shape == acc.shape
        assert unpacked.shape == acc.shape
        assert roundtripped.shape == acc.shape
        assert packed.dtype == torch.int16
        assert unpacked.dtype == torch.int16
        assert roundtripped.dtype == torch.int16
        assert not packed.is_cuda
        assert not unpacked.is_cuda
        assert not roundtripped.is_cuda


def test_c3_strict_out_of_domain_reject() -> None:
    with pytest.raises(ValueError, match="pack_w6_tensor requires all values"):
        pack_w6_tensor(torch.tensor([32], dtype=torch.int16))
    with pytest.raises(ValueError, match="pack_w6_tensor requires all values"):
        pack_w6_tensor(torch.tensor([-32], dtype=torch.int16))
    with pytest.raises(ValueError, match="pack_w6_tensor requires all values"):
        pack_w6_tensor(torch.tensor([100, -100], dtype=torch.int16))

    with pytest.raises(ValueError, match="unpack_w6_tensor requires packed lanes"):
        unpack_w6_tensor(torch.tensor([64], dtype=torch.int16))
    with pytest.raises(ValueError, match="unpack_w6_tensor requires packed lanes"):
        unpack_w6_tensor(torch.tensor([-1], dtype=torch.int16))

    in_domain = torch.tensor([0, 31, -31], dtype=torch.int16)
    packed = pack_w6_tensor(in_domain)
    assert torch.all(packed >= 0)
    assert torch.all(packed <= W6_PACKED_MAX)


def test_c4_mandatory_static_source_inspection() -> None:
    for fn_name in S3A_HOT_FUNCTIONS:
        source = _inspect_hot_function_source(fn_name)
        _assert_no_python_lane_loops(source, fn_name=fn_name)


def _median_wall_seconds(fn: Callable[[], None], *, warmup: int, measured: int) -> float:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(measured):
        start = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - start)
    return float(statistics.median(samples))


def measure_s3a_cost_model_population(
    *,
    shape: tuple[int, ...],
    numel: int,
    label: str,
    warmup: int = S3A_WARMUP_ITERS,
    measured: int = S3A_MEASURED_ITERS,
) -> dict[str, Any]:
    acc = _make_domain_tensor(shape)

    def vectorized_roundtrip() -> None:
        strict_roundtrip_w6_tensor(acc)

    def scalar_roundtrip() -> None:
        _scalar_roundtrip_tensor(acc)

    vec_median_s = _median_wall_seconds(
        vectorized_roundtrip,
        warmup=warmup,
        measured=measured,
    )
    scalar_median_s = _median_wall_seconds(
        scalar_roundtrip,
        warmup=warmup,
        measured=measured,
    )
    vectorized_torch_ms_median = vec_median_s * 1000.0
    scalar_tolist_baseline_ms_median = scalar_median_s * 1000.0
    cost_ratio = (
        vectorized_torch_ms_median / scalar_tolist_baseline_ms_median
        if scalar_tolist_baseline_ms_median > 0.0
        else float("inf")
    )
    return {
        "label": label,
        "numel": numel,
        "shape": list(shape),
        "warmup_iterations": warmup,
        "measured_iterations": measured,
        "scalar_tolist_baseline_ms_median": scalar_tolist_baseline_ms_median,
        "vectorized_torch_ms_median": vectorized_torch_ms_median,
        "cost_ratio": cost_ratio,
        "hot_path_uses_tolist": False,
    }


def build_s3a_cost_model_receipt() -> dict[str, Any]:
    populations = [
        measure_s3a_cost_model_population(
            label=str(pop["label"]),
            numel=int(pop["numel"]),
            shape=tuple(pop["shape"]),
        )
        for pop in S3A_COST_POPULATIONS
    ]
    gate = next(pop for pop in populations if pop["label"] == "phase3_12k_class")
    cost_ratio_pass = float(gate["cost_ratio"]) <= S3A_COST_RATIO_BOUND
    return {
        "warmup_iterations": S3A_WARMUP_ITERS,
        "measured_iterations": S3A_MEASURED_ITERS,
        "populations": populations,
        "gate_numel": S3A_COST_NUMEL,
        "scalar_tolist_baseline_ms_median": gate["scalar_tolist_baseline_ms_median"],
        "vectorized_torch_ms_median": gate["vectorized_torch_ms_median"],
        "cost_ratio": gate["cost_ratio"],
        "cost_ratio_bound": S3A_COST_RATIO_BOUND,
        "cost_ratio_pass": cost_ratio_pass,
    }


def test_c5_cost_model_warmup3_measured7_median() -> None:
    global S3A_LAST_COST_MODEL_RECEIPT

    receipt = build_s3a_cost_model_receipt()
    S3A_LAST_COST_MODEL_RECEIPT = receipt

    gate = next(
        pop for pop in receipt["populations"] if pop["label"] == "phase3_12k_class"
    )
    assert receipt["warmup_iterations"] == S3A_WARMUP_ITERS
    assert receipt["measured_iterations"] == S3A_MEASURED_ITERS
    assert receipt["gate_numel"] == S3A_COST_NUMEL
    assert receipt["scalar_tolist_baseline_ms_median"] == pytest.approx(
        gate["scalar_tolist_baseline_ms_median"]
    )
    assert receipt["vectorized_torch_ms_median"] == pytest.approx(
        gate["vectorized_torch_ms_median"]
    )
    assert receipt["cost_ratio"] == pytest.approx(gate["cost_ratio"])
    assert receipt["cost_ratio_pass"] is True
    assert float(receipt["cost_ratio"]) <= S3A_COST_RATIO_BOUND


def test_s3a_classifier_receipt_ok() -> None:
    cost_model = build_s3a_cost_model_receipt()
    receipt = emit_s3a_classifier_receipt(
        parity_pass=True,
        static_inspection_pass=True,
        cost_ratio_at_12288=float(cost_model["cost_ratio"]),
        cost_ratio_pass=bool(cost_model["cost_ratio_pass"]),
        cost_model_receipt=cost_model,
    )
    assert receipt["primary_classifier"] == CLASSIFIER_S3A_VECTOR_PARITY_OK_COST_BOUNDED
    assert receipt["explicit_non_claims"] == list(S3A_EXPLICIT_NON_CLAIMS)
    assert receipt["cost_model_receipt"]["gate_numel"] == S3A_COST_NUMEL
    assert "scalar_tolist_baseline_ms_median" in receipt["cost_model_receipt"]
    assert "vectorized_torch_ms_median" in receipt["cost_model_receipt"]


def test_clip_then_pack_w6_tensor_replay_matches_scalar() -> None:
    acc = torch.tensor([100, -100, 0, 31, -31], dtype=torch.int16)
    packed = clip_then_pack_w6_tensor(acc)
    expected = torch.tensor(
        [pack_w6(v) for v in [31, -31, 0, 31, -31]],
        dtype=torch.int16,
    )
    assert torch.equal(packed, expected)
