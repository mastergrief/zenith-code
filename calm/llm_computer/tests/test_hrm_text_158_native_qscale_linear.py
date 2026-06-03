"""Phase-1 int8-levels qscale bridge tests.

CPU/reference tests prove the public contract and c1353fd5 scalar-scale
broadcast. CUDA tests are deferred until gpu:0 is available; skipped CUDA tests
are not a kernel-done receipt.
"""
from __future__ import annotations

import os

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.qscale_linear import (
    INT8_LEVELS_TRANSITIONAL_NOTE,
    QScaleLinearConfig,
    QScaleWeightFormat,
    QScaleWeightState,
    qscale_linear_reference,
    qscale_linear_triton,
    validate_qscale_weight_state,
)


PRODUCTION_SHAPES = [
    ("gqkv_proj", 512, 2048),
    ("o_proj", 512, 512),
    ("gate_up_proj", 512, 3072),
    ("down_proj", 1536, 512),
]

RUN_GPU_QSCALE = os.environ.get("HRM_TEXT_158_RUN_GPU_QSCALE") == "1"


cuda = pytest.mark.skipif(
    (not RUN_GPU_QSCALE) or (not torch.cuda.is_available()),
    reason=(
        "int8-levels qscale GPU receipt deferred; set "
        "HRM_TEXT_158_RUN_GPU_QSCALE=1 only inside a granted gpu:0 lane"
    ),
)


def _make_q(out_features: int, in_features: int, *, device: str = "cpu", seed: int = 17) -> torch.Tensor:
    gen = torch.Generator(device=device).manual_seed(seed)
    levels = torch.tensor([-1, 0, 1], dtype=torch.int8, device=device)
    idx = torch.randint(0, 3, (out_features, in_features), generator=gen, device=device)
    return levels[idx].contiguous()


class _DisableCudaTf32:
    """Match the Triton IEEE dot path during deferred GPU correctness checks."""

    def __enter__(self):
        if torch.cuda.is_available():
            self._old_matmul_tf32 = torch.backends.cuda.matmul.allow_tf32
            self._old_cudnn_tf32 = torch.backends.cudnn.allow_tf32
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        return self

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = self._old_matmul_tf32
            torch.backends.cudnn.allow_tf32 = self._old_cudnn_tf32
        return False


@pytest.mark.parametrize("name,in_features,out_features", PRODUCTION_SHAPES)
@pytest.mark.parametrize("input_shape", ["2d", "3d"])
def test_reference_matches_torch_linear_scalar_scale_on_production_shapes(name, in_features, out_features, input_shape):
    q = _make_q(out_features, in_features, seed=len(name))
    scale = torch.tensor(0.125, dtype=torch.float32)
    bias = torch.linspace(-0.05, 0.05, out_features, dtype=torch.float32)
    state = QScaleWeightState(q_levels=q, scale=scale, format=QScaleWeightFormat.INT8_LEVELS)
    if input_shape == "2d":
        x = torch.randn(2, in_features, dtype=torch.float32)
    elif input_shape == "3d":
        x = torch.randn(1, 2, in_features, dtype=torch.float32)
    else:
        raise AssertionError(input_shape)

    actual = qscale_linear_reference(x, state, bias)
    expected = F.linear(x, q.to(torch.float32) * scale, bias)

    assert actual.shape == (*x.shape[:-1], out_features)
    torch.testing.assert_close(actual, expected, atol=0.0, rtol=0.0)


def test_int8_levels_format_is_explicitly_transitional_and_pack_ready():
    q = _make_q(4, 3)
    state = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(0.25, dtype=torch.float32),
        format="int8_levels",
    )

    assert state.normalized_format == QScaleWeightFormat.INT8_LEVELS
    assert state.is_transitional_int8_levels
    assert "not packed sub-2-bit" in state.format_note
    assert "transitional" in INT8_LEVELS_TRANSITIONAL_NOTE
    assert {fmt.value for fmt in QScaleWeightFormat} == {
        "int8_levels",
        "packed_2bit",
        "packed_ternary",
    }


def test_future_packed_formats_are_named_but_not_silently_implemented():
    q = _make_q(4, 3)
    state = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(0.25, dtype=torch.float32),
        format=QScaleWeightFormat.PACKED_TERNARY,
    )

    with pytest.raises(NotImplementedError, match="pack-ready"):
        validate_qscale_weight_state(state)


def test_rejects_raw_fp_master_tensor():
    fp_master = torch.randn(4, 3, dtype=torch.float32)
    state = QScaleWeightState(
        q_levels=fp_master,
        scale=torch.tensor(0.25, dtype=torch.float32),
        format="int8_levels",
    )

    with pytest.raises(ValueError, match="FP master tensors are not accepted"):
        qscale_linear_reference(torch.randn(2, 3, dtype=torch.float32), state)


def test_rejects_non_ternary_int8_levels():
    q = torch.tensor([[0, 1, 2], [-1, 0, 1]], dtype=torch.int8)
    state = QScaleWeightState(
        q_levels=q,
        scale=torch.tensor(0.25, dtype=torch.float32),
        format="int8_levels",
    )

    with pytest.raises(ValueError, match="ternary int8"):
        validate_qscale_weight_state(state)


def test_rejects_non_scalar_scale_because_c1353fd5_uses_per_tensor_scale():
    state = QScaleWeightState(
        q_levels=_make_q(4, 3),
        scale=torch.ones(4, dtype=torch.float32),
        format="int8_levels",
    )

    with pytest.raises(ValueError, match="per-tensor scalar scale"):
        qscale_linear_reference(torch.randn(2, 3, dtype=torch.float32), state)


def test_config_keeps_tuning_parameters_abstract():
    cfg = QScaleLinearConfig(block_m=8, block_n=16, block_k=32, num_warps=4)
    cfg.validate()

    with pytest.raises(ValueError, match="block_m"):
        QScaleLinearConfig(block_m=0).validate()


@cuda
@pytest.mark.parametrize("name,in_features,out_features", PRODUCTION_SHAPES)
@pytest.mark.parametrize("input_shape", ["2d_m32", "3d_b8_s32"])
def test_triton_allclose_vs_reference_on_production_shapes(name, in_features, out_features, input_shape):
    q = _make_q(out_features, in_features, device="cuda", seed=len(name))
    scale = torch.tensor(0.125, dtype=torch.float32, device="cuda")
    bias = torch.linspace(-0.05, 0.05, out_features, dtype=torch.float32, device="cuda")
    state = QScaleWeightState(q_levels=q, scale=scale, format="int8_levels")
    gen = torch.Generator(device="cuda").manual_seed(1337)
    if input_shape == "2d_m32":
        x = torch.randn(32, in_features, generator=gen, device="cuda", dtype=torch.float32)
    elif input_shape == "3d_b8_s32":
        x = torch.randn(8, 32, in_features, generator=gen, device="cuda", dtype=torch.float32)
    else:
        raise AssertionError(input_shape)

    with _DisableCudaTf32():
        actual = qscale_linear_triton(x, state, bias)
        expected = qscale_linear_reference(x, state, bias)

    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)


@cuda
def test_triton_perf_and_memory_receipt_scaffold():
    name, in_features, out_features = PRODUCTION_SHAPES[0]
    q = _make_q(out_features, in_features, device="cuda", seed=len(name))
    scale = torch.tensor(0.125, dtype=torch.float32, device="cuda")
    state = QScaleWeightState(q_levels=q, scale=scale, format="int8_levels")
    gen = torch.Generator(device="cuda").manual_seed(2026)
    x = torch.randn(32, in_features, generator=gen, device="cuda", dtype=torch.float32)

    torch.cuda.reset_peak_memory_stats()
    with _DisableCudaTf32():
        qscale_linear_triton(x, state)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(10):
            y = qscale_linear_triton(x, state)
        end.record()
        torch.cuda.synchronize()
        triton_ms = start.elapsed_time(end) / 10.0

        start.record()
        for _ in range(10):
            baseline = F.linear(x, q.to(torch.float32) * scale)
        end.record()
        torch.cuda.synchronize()
        baseline_ms = start.elapsed_time(end) / 10.0
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()

    torch.testing.assert_close(y, baseline, atol=1e-4, rtol=1e-4)
    assert triton_ms > 0.0
    assert baseline_ms > 0.0
    assert peak_allocated > 0
    assert peak_reserved >= peak_allocated
    assert q.dtype == torch.int8
    assert state.is_transitional_int8_levels
    print(
        "int8_levels_qscale_gpu_receipt_scaffold "
        f"shape={name} triton_ms={triton_ms:.4f} baseline_materialize_ms={baseline_ms:.4f} "
        f"peak_allocated={peak_allocated} peak_reserved={peak_reserved}"
    )
