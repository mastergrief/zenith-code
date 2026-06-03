"""GPU-native q*scale linear bridge for the Phase-1 native stack.

This is intentionally named an int8-levels bridge, not a packed ternary storage
kernel. It targets the live c1353fd5 authoritative representation:
q:int8 ternary levels plus a frozen per-tensor fp32 scale.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
import torch.nn.functional as F

try:  # Keep CPU/reference tests importable when Triton is unavailable.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only on non-Triton hosts.
    triton = None
    tl = None


INT8_LEVELS_TRANSITIONAL_NOTE = (
    "int8_levels is a transitional qscale bridge format: it removes FP-master "
    "forward materialization, but it is not packed sub-2-bit ternary storage."
)


class QScaleWeightFormat(str, Enum):
    """Pack-ready weight-state format names for native qscale kernels."""

    INT8_LEVELS = "int8_levels"
    PACKED_2BIT = "packed_2bit"
    PACKED_TERNARY = "packed_ternary"


@dataclass(frozen=True)
class QScaleWeightState:
    """Persistent native qscale weight state for one bulk projection."""

    q_levels: torch.Tensor
    scale: torch.Tensor
    format: QScaleWeightFormat | str = QScaleWeightFormat.INT8_LEVELS

    @property
    def normalized_format(self) -> QScaleWeightFormat:
        try:
            return QScaleWeightFormat(self.format)
        except ValueError as exc:
            valid = ", ".join(fmt.value for fmt in QScaleWeightFormat)
            raise ValueError(f"unknown qscale weight format {self.format!r}; valid={valid}") from exc

    @property
    def is_transitional_int8_levels(self) -> bool:
        return self.normalized_format == QScaleWeightFormat.INT8_LEVELS

    @property
    def format_note(self) -> str:
        if self.is_transitional_int8_levels:
            return INT8_LEVELS_TRANSITIONAL_NOTE
        return "future packed qscale format; not implemented in this slice"


@dataclass(frozen=True)
class QScaleLinearConfig:
    """Triton launch knobs kept abstract until the terminal S1 sizing lands."""

    block_m: int = 16
    block_n: int = 32
    block_k: int = 32
    num_warps: int = 4
    validate_levels: bool = True

    def validate(self) -> None:
        for name, value in (
            ("block_m", self.block_m),
            ("block_n", self.block_n),
            ("block_k", self.block_k),
            ("num_warps", self.num_warps),
        ):
            if not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer, got {value!r}")


if triton is not None:

    @triton.jit
    def _int8_levels_qscale_linear_kernel(
        X,
        Q,
        SCALE,
        BIAS,
        OUT,
        M: tl.constexpr,
        N: tl.constexpr,
        K: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)

        for k0 in range(0, K, BLOCK_K):
            k = k0 + offs_k
            x = tl.load(
                X + offs_m[:, None] * K + k[None, :],
                mask=(offs_m[:, None] < M) & (k[None, :] < K),
                other=0.0,
            ).to(tl.float32)
            q = tl.load(
                Q + offs_n[None, :] * K + k[:, None],
                mask=(offs_n[None, :] < N) & (k[:, None] < K),
                other=0,
            ).to(tl.float32)
            acc += tl.dot(x, q, input_precision="ieee")

        scale = tl.load(SCALE).to(tl.float32)
        acc *= scale
        if HAS_BIAS:
            bias = tl.load(BIAS + offs_n, mask=offs_n < N, other=0.0).to(tl.float32)
            acc += bias[None, :]

        tl.store(
            OUT + offs_m[:, None] * N + offs_n[None, :],
            acc,
            mask=(offs_m[:, None] < M) & (offs_n[None, :] < N),
        )

else:
    _int8_levels_qscale_linear_kernel = None


def validate_qscale_weight_state(
    state: QScaleWeightState,
    *,
    validate_levels: bool = True,
) -> tuple[torch.Tensor, torch.Tensor, QScaleWeightFormat]:
    """Validate the live c1353fd5 q:int8 + scalar-scale representation."""

    weight_format = state.normalized_format
    if weight_format != QScaleWeightFormat.INT8_LEVELS:
        raise NotImplementedError(
            f"{weight_format.value} is pack-ready but not implemented in this slice; "
            "use format='int8_levels' for the transitional bridge"
        )

    q_levels = state.q_levels
    scale = state.scale
    if q_levels.dtype.is_floating_point:
        raise ValueError(
            "FP master tensors are not accepted by the int8_levels qscale bridge; "
            "pass persistent q:int8 levels plus frozen_scale:fp32"
        )
    if q_levels.dtype != torch.int8:
        raise ValueError(f"q_levels must have dtype torch.int8, got {q_levels.dtype}")
    if q_levels.ndim != 2:
        raise ValueError(f"q_levels must be a 2-D (out_features, in_features) tensor, got {tuple(q_levels.shape)}")
    if scale.dtype != torch.float32:
        raise ValueError(f"scale must have dtype torch.float32, got {scale.dtype}")
    if scale.numel() != 1:
        raise ValueError(
            "scale must be the live c1353fd5 per-tensor scalar scale; "
            f"got shape {tuple(scale.shape)} with numel={scale.numel()}"
        )
    if scale.device != q_levels.device:
        raise ValueError(f"scale device {scale.device} must match q_levels device {q_levels.device}")
    if not bool(torch.isfinite(scale).all().item()):
        raise ValueError("scale must be finite")
    if float(scale.detach().reshape(()).item()) <= 0.0:
        raise ValueError("scale must be positive")
    if validate_levels:
        allowed = torch.tensor([-1, 0, 1], dtype=torch.int8, device=q_levels.device)
        if not bool(torch.isin(q_levels, allowed).all().item()):
            raise ValueError("q_levels must contain only ternary int8 levels {-1, 0, +1}")
    return q_levels, scale, weight_format


def _validate_linear_inputs(
    input: torch.Tensor,
    q_levels: torch.Tensor,
    bias: Optional[torch.Tensor],
) -> None:
    if input.dtype != torch.float32:
        raise ValueError(f"input must be torch.float32 for this fp32 MVP bridge, got {input.dtype}")
    if input.ndim < 2:
        raise ValueError(f"input must have at least 2 dims (..., in_features), got {tuple(input.shape)}")
    if input.shape[-1] != q_levels.shape[1]:
        raise ValueError(
            f"input last dim {input.shape[-1]} must match q_levels in_features {q_levels.shape[1]}"
        )
    if input.device != q_levels.device:
        raise ValueError(f"input device {input.device} must match q_levels device {q_levels.device}")
    if bias is None:
        return
    if bias.dtype != torch.float32:
        raise ValueError(f"bias must be torch.float32, got {bias.dtype}")
    if bias.device != input.device:
        raise ValueError(f"bias device {bias.device} must match input device {input.device}")
    if bias.shape != (q_levels.shape[0],):
        raise ValueError(f"bias shape must be ({q_levels.shape[0]},), got {tuple(bias.shape)}")


def qscale_linear_reference(
    input: torch.Tensor,
    state: QScaleWeightState,
    bias: Optional[torch.Tensor] = None,
    *,
    validate_levels: bool = True,
) -> torch.Tensor:
    """Reference bridge: F.linear(x, q.float() * scalar_scale, bias)."""

    q_levels, scale, _ = validate_qscale_weight_state(state, validate_levels=validate_levels)
    _validate_linear_inputs(input, q_levels, bias)
    weight = q_levels.to(torch.float32) * scale.to(torch.float32)
    return F.linear(input, weight, bias)


def qscale_linear_triton(
    input: torch.Tensor,
    state: QScaleWeightState,
    bias: Optional[torch.Tensor] = None,
    *,
    config: QScaleLinearConfig = QScaleLinearConfig(),
) -> torch.Tensor:
    """Triton int8-levels qscale bridge, without materializing q*scale weight."""

    if _int8_levels_qscale_linear_kernel is None:
        raise RuntimeError("qscale_linear_triton requires Triton")
    config.validate()
    q_levels, scale, _ = validate_qscale_weight_state(
        state,
        validate_levels=config.validate_levels,
    )
    _validate_linear_inputs(input, q_levels, bias)
    if input.device.type != "cuda":
        raise ValueError("qscale_linear_triton requires CUDA input/q_levels/scale tensors")

    in_features = int(q_levels.shape[1])
    out_features = int(q_levels.shape[0])
    input_2d = input.reshape(-1, in_features).contiguous()
    q_contig = q_levels.contiguous()
    scale_contig = scale.reshape(()).contiguous()
    bias_contig = bias.contiguous() if bias is not None else None
    out_2d = torch.empty(
        (input_2d.shape[0], out_features),
        device=input.device,
        dtype=torch.float32,
    )
    grid = (
        triton.cdiv(input_2d.shape[0], config.block_m),
        triton.cdiv(out_features, config.block_n),
    )
    _int8_levels_qscale_linear_kernel[grid](
        input_2d,
        q_contig,
        scale_contig,
        bias_contig if bias_contig is not None else out_2d,
        out_2d,
        int(input_2d.shape[0]),
        out_features,
        in_features,
        HAS_BIAS=bias_contig is not None,
        BLOCK_M=config.block_m,
        BLOCK_N=config.block_n,
        BLOCK_K=config.block_k,
        num_warps=config.num_warps,
    )
    return out_2d.reshape(*input.shape[:-1], out_features)
