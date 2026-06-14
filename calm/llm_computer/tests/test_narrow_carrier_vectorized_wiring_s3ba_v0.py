from __future__ import annotations

import ast
import inspect
import os

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.narrow_accumulator_codec import (
    W6_SIGNED_MAX,
    W6_SIGNED_MIN,
    pack_w6,
    strict_roundtrip_w6_tensor,
    unpack_w6,
)
from calm.hrm_text_158.native_full_stack.narrow_carrier_trainer_integration import (
    CLASSIFIER_S3BA_VECTOR_WIRING_OK,
    RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV,
    S3BA_EXPLICIT_NON_CLAIMS,
    apply_trainer_boundary_narrow_carrier,
    assert_no_packed_w6_state_leak,
    count_int16_vs_w6_crossing_mismatches,
    emit_s3ba_classifier_receipt,
    narrow_carrier_w6_enabled,
    strict_roundtrip_int16_value_through_trainer_boundary,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    carry_self_update_row,
    crosses_threshold,
)

S3BA_TENSOR_SHAPES: tuple[tuple[int, ...], ...] = ((3, 3), (1, 3), (128,))


def _tiny_state() -> BoundedDeltaTensorState:
    q = torch.tensor([[0, 1, -1]], dtype=torch.int8)
    acc = torch.tensor([[5, -9, 21]], dtype=torch.int16)
    return make_bounded_tensor_state("tiny.proj", q, 1.0, acc)


def _tiny_fixture_rows() -> list[tuple[int, int, int]]:
    return [
        (5, 3, 0),
        (-9, 2, 1),
        (21, -4, -1),
        (10, 0, 0),
        (-10, 0, 0),
    ]


def _full_domain_values() -> list[int]:
    return list(range(W6_SIGNED_MIN, W6_SIGNED_MAX + 1))


def _make_domain_tensor(shape: tuple[int, ...], *, device: torch.device) -> torch.Tensor:
    numel = 1
    for dim in shape:
        numel *= dim
    values = (_full_domain_values() * ((numel // len(_full_domain_values())) + 1))[:numel]
    return torch.tensor(values, dtype=torch.int16, device=device).reshape(shape)


def _scalar_roundtrip_tensor(acc: torch.Tensor) -> torch.Tensor:
    flat = acc.flatten().tolist()
    roundtripped = [
        strict_roundtrip_int16_value_through_trainer_boundary(int(v)) for v in flat
    ]
    return torch.tensor(roundtripped, dtype=torch.int16, device=acc.device).reshape(
        acc.shape
    )


def _boundary_source() -> str:
    from calm.hrm_text_158.native_full_stack import (
        narrow_carrier_trainer_integration as integration_mod,
    )

    return inspect.getsource(integration_mod.apply_trainer_boundary_narrow_carrier)


def _assert_no_python_lane_loops(source: str, *, fn_name: str) -> None:
    if ".tolist()" in source:
        raise AssertionError(f"{fn_name} must not use .tolist()")
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


@pytest.fixture(autouse=True)
def _clear_narrow_carrier_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, raising=False)


def test_w1_boundary_hot_path_no_tolist() -> None:
    source = _boundary_source()
    _assert_no_python_lane_loops(source, fn_name="apply_trainer_boundary_narrow_carrier")
    assert "strict_roundtrip_w6_tensor" in source


def test_w2_cpu_regression_bit_identical_vs_prior_s2_scalar_roundtrip() -> None:
    for value in _full_domain_values():
        scalar = strict_roundtrip_int16_value_through_trainer_boundary(value)
        tensor = apply_trainer_boundary_narrow_carrier(
            torch.tensor(value, dtype=torch.int16),
            enabled=True,
        )
        assert int(tensor.item()) == scalar

    for shape in S3BA_TENSOR_SHAPES:
        acc = _make_domain_tensor(shape, device=torch.device("cpu"))
        expected = _scalar_roundtrip_tensor(acc)
        actual = apply_trainer_boundary_narrow_carrier(acc, enabled=True)
        assert torch.equal(actual, expected)


def test_w3_device_dtype_shape_invariants() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        for shape in S3BA_TENSOR_SHAPES:
            acc = _make_domain_tensor(shape, device=device)
            out = apply_trainer_boundary_narrow_carrier(acc, enabled=True)
            assert out.shape == acc.shape
            assert out.dtype == torch.int16
            assert out.device == acc.device


def test_w4_flag_off_identity_cpu_and_cuda() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        acc = _make_domain_tensor((1, 3), device=device)
        assert narrow_carrier_w6_enabled() is False
        assert torch.equal(apply_trainer_boundary_narrow_carrier(acc), acc)
        assert torch.equal(apply_trainer_boundary_narrow_carrier(acc, enabled=False), acc)


def test_w5_int16_oracle_vs_w6_crossing_parity() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        for pre_acc, vote, q_level in _tiny_fixture_rows():
            new_acc_16 = carry_self_update_row(pre_acc, vote, width=16)
            new_acc_6 = carry_self_update_row(pre_acc, vote, width=6)
            oracle_cross = crosses_threshold(
                new_acc_16,
                current_q_level=q_level,
                threshold_abs=10,
            )
            carrier_tensor = apply_trainer_boundary_narrow_carrier(
                torch.tensor([[new_acc_6]], dtype=torch.int16, device=device),
                enabled=True,
            )
            carrier_cross = crosses_threshold(
                int(carrier_tensor.item()),
                current_q_level=q_level,
                threshold_abs=10,
            )
            assert oracle_cross == carrier_cross

    assert count_int16_vs_w6_crossing_mismatches(_tiny_fixture_rows(), enabled=True) == 0


def test_w6_strict_out_of_domain_reject_no_silent_clip() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))

    for device in devices:
        acc = torch.tensor([[100, -100]], dtype=torch.int16, device=device)
        with pytest.raises(ValueError, match="pack_w6 requires value"):
            apply_trainer_boundary_narrow_carrier(acc, enabled=True)


def test_w7_no_packed_w6_state_leak() -> None:
    state = _tiny_state()
    shadow = state.exact_accumulator_shadow
    assert shadow is not None

    os.environ[RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV] = "1"
    try:
        _ = state.vote_update_state()
    finally:
        os.environ.pop(RUN_NARROW_CARRIER_W6_TRAINER_INTEGRATION_ENV, None)

    assert_no_packed_w6_state_leak(state.to_schema_dict(parity_check=False))


def test_s3ba_classifier_receipt_ok() -> None:
    receipt = emit_s3ba_classifier_receipt(
        static_inspection_pass=True,
        cpu_regression_pass=True,
        parity_mismatch_count=0,
        flag_off_identity_pass=True,
        no_packed_state_leak_pass=True,
    )
    assert receipt["primary_classifier"] == CLASSIFIER_S3BA_VECTOR_WIRING_OK
    assert receipt["explicit_non_claims"] == list(S3BA_EXPLICIT_NON_CLAIMS)


def test_strict_roundtrip_w6_tensor_cuda_matches_scalar_oracle() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    acc = torch.tensor([[-31, 0, 31]], dtype=torch.int16, device="cuda")
    out = strict_roundtrip_w6_tensor(acc)
    expected = torch.tensor(
        [unpack_w6(pack_w6(int(v))) for v in [-31, 0, 31]],
        dtype=torch.int16,
        device="cuda",
    ).reshape(acc.shape)
    assert torch.equal(out, expected)
    assert out.device == acc.device
