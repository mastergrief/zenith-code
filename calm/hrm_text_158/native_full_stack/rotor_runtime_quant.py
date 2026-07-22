"""Rotor runtime quantization facade for HRM-Text-1.58 transient surfaces.

Phase 0 of the ternary-rotor lane (plan: zenith-code
`.claude/MEMORY/ternary-rotor.md`). Provides:

1. A line-faithful torch port of the turbo2/turbo3 rotated scalar quantizer
   validated on the Bonsai-27B/GTX-1070 lane (reference implementation:
   prismml-llamacpp `ggml/src/ggml-turbo-quant.c`, branch `q1_75-planarquant`).
   Mechanism per 128-value rotation group:
     L2-normalize -> signed FWHT (fixed seed-42 sign vectors) ->
     nearest-centroid Lloyd-Max (2- or 3-bit) -> fp16 corrected group norm.
   Exposed as a fake-quant round trip (`rotor_fake_quant`) for tolerance
   screens on runtime surfaces; no packed storage is materialized.

2. The scale-inclusive bits ledger (`rotor_bits_ledger`) — the accounting
   authority for every claim from this lane. "Sub-2" is scale-inclusive
   (< 2.0 bpw) per `ternary_hybrid_stack.md`; turbo2 as shipped is
   2.125 bpw and the ledger must say so.

This module is measurement tooling only: it makes no training-state,
readiness, or sub2 claim. `full_sub2_runtime_readiness` remains the sole
readiness authority.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

# Rotation group size — equals HRM-Text-1.58 head_dim; hidden=512 = 4 groups.
ROTOR_GROUP = 128

# Checker surface names this lane targets (kept in sync with
# full_sub2_runtime_readiness; import avoided to keep this facade standalone).
ROTOR_TARGET_SURFACES = (
    "activations_residuals",
    "attention_kv_attention_buffers",
    "backward_saved_tensors_transients",
)

# ---------------------------------------------------------------------------
# Constants ported verbatim from ggml-turbo-quant.c (seed-42 sign vectors,
# Lloyd-Max centroids for N(0, 1/128)).
# ---------------------------------------------------------------------------

_S1 = [
    -1, 1, 1, -1, -1, 1, -1, 1, -1, -1, 1, 1, 1, 1, 1, 1,
    1, -1, 1, -1, 1, -1, -1, 1, 1, 1, -1, 1, 1, -1, -1, -1,
    -1, 1, 1, -1, 1, 1, -1, 1, -1, 1, 1, -1, -1, 1, -1, 1,
    1, 1, 1, -1, -1, -1, -1, -1, 1, -1, 1, 1, 1, 1, -1, 1,
    -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, -1, -1, -1, 1, 1,
    1, -1, -1, 1, 1, 1, -1, -1, 1, 1, -1, 1, 1, -1, 1, -1,
    -1, 1, 1, -1, 1, -1, 1, -1, 1, 1, 1, 1, -1, 1, -1, 1,
    1, -1, 1, 1, -1, -1, -1, -1, -1, 1, 1, -1, 1, 1, -1, 1,
]

_S2 = [
    1, 1, 1, 1, -1, 1, 1, -1, 1, -1, -1, -1, 1, -1, -1, -1,
    1, 1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, -1, 1, 1, 1,
    1, 1, -1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, -1, -1, -1, -1, -1, 1, 1,
    1, -1, 1, -1, -1, -1, -1, 1, -1, 1, -1, 1, -1, -1, 1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, -1, -1, -1, 1, -1, -1, 1, -1,
    1, -1, 1, 1, 1, -1, -1, 1, -1, 1, -1, 1, 1, -1, -1, 1,
    -1, 1, -1, 1, 1, -1, 1, -1, 1, -1, -1, -1, -1, -1, 1, -1,
]

_CENTROIDS_2BIT = [-0.133462, -0.039994, 0.039994, 0.133462]

_CENTROIDS_3BIT = [
    -0.190685, -0.117832, -0.065717, -0.021460,
    0.021460, 0.065717, 0.117832, 0.190685,
]

_INV_SQRT_128 = 0.08838834764831845

# 3-level Lloyd-Max for N(0, 1/128): a = 1.224006/sqrt(128) (classic 1.224σ
# symmetric ternary quantizer; derived by fixed-point iteration t=a/2,
# a=φ(t)/Q(t)). This is the sub-2 route: 4-level codes are 2.0 bits flat, so
# NO scale packing clears <2.0 — only 3-level codes base-3 packed with the
# Q1_75 geometry (two 13-byte halves per 128 codes) go under.
_CENTROIDS_TERNARY = [-0.108188, 0.0, 0.108188]

# Keys: 2/3 = turbo2/turbo3 index widths; "ternary" = 3-level polar codes.
_CENTROIDS_BY_BITS = {2: _CENTROIDS_2BIT, 3: _CENTROIDS_3BIT,
                      "ternary": _CENTROIDS_TERNARY}


def _signed_fwht(x: torch.Tensor) -> torch.Tensor:
    """Forward signed WHT over the last dim (must be ROTOR_GROUP).

    y = D2 * (1/sqrt(128)) * H * D1 * x, matching turbo_cpu_fwht.
    Orthonormal: preserves L2 norm exactly (up to float error).
    """
    s1 = torch.tensor(_S1, dtype=x.dtype, device=x.device)
    s2 = torch.tensor(_S2, dtype=x.dtype, device=x.device)
    y = x * s1
    h = 1
    n = x.shape[-1]
    while h < n:
        y = y.reshape(*y.shape[:-1], n // (2 * h), 2, h)
        a = y[..., 0, :]
        b = y[..., 1, :]
        y = torch.stack((a + b, a - b), dim=-2).reshape(*x.shape)
        h *= 2
    return y * (_INV_SQRT_128 * s2)


def _signed_fwht_inverse(y: torch.Tensor) -> torch.Tensor:
    """Inverse of `_signed_fwht`: x = D1 * (1/sqrt(128)) * H * D2 * y."""
    s1 = torch.tensor(_S1, dtype=y.dtype, device=y.device)
    s2 = torch.tensor(_S2, dtype=y.dtype, device=y.device)
    x = y * s2
    h = 1
    n = y.shape[-1]
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack((a + b, a - b), dim=-2).reshape(*y.shape)
        h *= 2
    return x * (_INV_SQRT_128 * s1)


def rotor_fake_quant(x: torch.Tensor, bits: int | str) -> torch.Tensor:
    """turbo2/turbo3 quantize->dequantize round trip (no packed storage).

    Last dim must be a multiple of ROTOR_GROUP (=128). Semantics match
    quantize_row_turbo{2,3}_0_ref + dequantize_row_turbo{2,3}_0:
    group L2 norm -> normalize -> signed FWHT -> nearest centroid ->
    fp16 corrected norm (grp_norm / recon_norm) -> inverse FWHT rescale.
    """
    if bits not in _CENTROIDS_BY_BITS:
        raise ValueError(
            f"unsupported rotor bits: {bits} (supported: 2, 3, 'ternary')")
    if x.shape[-1] % ROTOR_GROUP != 0:
        raise ValueError(
            f"last dim {x.shape[-1]} not a multiple of ROTOR_GROUP={ROTOR_GROUP}"
        )
    orig_dtype = x.dtype
    xf = x.float().reshape(*x.shape[:-1], x.shape[-1] // ROTOR_GROUP, ROTOR_GROUP)

    grp_norm = xf.norm(dim=-1, keepdim=True)
    inv_norm = torch.where(
        grp_norm > 1e-10, 1.0 / grp_norm.clamp(min=1e-30), torch.zeros_like(grp_norm)
    )
    rot = _signed_fwht(xf * inv_norm)

    centroids = torch.tensor(
        _CENTROIDS_BY_BITS[bits], dtype=rot.dtype, device=rot.device
    )
    # Nearest centroid via bucketize on midpoint boundaries — identical to
    # argmin over |x - c| for sorted centroids, without materializing the
    # (..., levels) intermediate (allocator-churn fix for the trainer hook).
    boundaries = (centroids[:-1] + centroids[1:]) / 2
    idx = torch.bucketize(rot, boundaries)
    quant = centroids[idx]

    recon_norm = quant.norm(dim=-1, keepdim=True)
    corrected = torch.where(recon_norm > 1e-10, grp_norm / recon_norm.clamp(min=1e-30), grp_norm)
    # The stored scale is fp16 in the block layout — round through fp16.
    corrected = corrected.to(torch.float16).float()

    out = _signed_fwht_inverse(quant) * corrected
    return out.reshape(x.shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Bits ledger — the scale-inclusive accounting authority for this lane.
# ---------------------------------------------------------------------------

SCALE_BITS_BY_DTYPE = {"fp16": 16, "int8": 8, "fp32": 32}


# ---------------------------------------------------------------------------
# Phase 3b backward-saved codec: quantize-narrow + remat-wide.
# Promoted from the screen script on second use (trainer integration).
# Screen receipts (banked parent, 2026-07-22): remat_only exactly lossless;
# 3-bit narrow med_cos 0.995 / min_cos 0.979 vs FP-saved control (clears the
# prereg bars); blanket quantization without remat parked at min_cos 0.750.
# ---------------------------------------------------------------------------


class SavedTensorRotorCodec:
    """saved_tensors_hooks pack/unpack pair.

    narrow_only=True (the Phase 3b contract): rotor-quantize ONLY 512-wide
    dim-3 float saves — block inputs / residual-stream activations. Wide
    SwiGLU intermediates are handled by checkpoint remat (never stored);
    4-D SDPA saves MUST stay exact (backward recomputes attention scores
    against the exact forward logsumexp; quantizing behind the kernel
    measured med_cos 0.65 with exp-blowup outliers).
    """

    def __init__(self, bits: int | str | None, *, narrow_only: bool = True):
        self.bits = bits
        self.narrow_only = narrow_only
        self.quantized = 0
        self.passed = 0
        self.quantized_values = 0
        self.passed_values = 0

    def _eligible(self, t: torch.Tensor) -> bool:
        if not (torch.is_floating_point(t) and t.dim() == 3
                and t.shape[-1] % 128 == 0):
            return False
        if self.narrow_only and t.shape[-1] != 512:
            return False
        return True

    def pack(self, t: torch.Tensor):
        with torch.no_grad():
            if self.bits is not None and self._eligible(t):
                self.quantized += 1
                self.quantized_values += t.numel()
                return rotor_fake_quant(t.detach(), bits=self.bits)
            self.passed += 1
            if torch.is_floating_point(t):
                self.passed_values += t.numel()
            return t

    def unpack(self, t: torch.Tensor) -> torch.Tensor:
        return t

    def hooks(self):
        """The saved_tensors_hooks context for this codec."""
        return torch.autograd.graph.saved_tensors_hooks(self.pack, self.unpack)


def wrap_swiglu_with_checkpoint(model: torch.nn.Module) -> int:
    """Checkpoint-wrap every SwiGLU so the wide (2*intermediate) tensors are
    recomputed in backward from the saved block input instead of stored.
    Lossless by construction (screen receipt: cos 1.00000 on all params).
    Returns the number of modules wrapped; restore via unwrap_swiglu."""
    from torch.utils.checkpoint import checkpoint
    from calm.hrm_text_158.layers import SwiGLU

    n = 0
    for mod in model.modules():
        if isinstance(mod, SwiGLU) and not hasattr(mod, "_rotor_orig_forward"):
            orig = mod.forward

            def fwd(x, _orig=orig):
                if not (torch.is_grad_enabled() and x.requires_grad):
                    return _orig(x)
                return checkpoint(_orig, x, use_reentrant=False)

            mod._rotor_orig_forward = orig
            mod.forward = fwd
            n += 1
    return n


def unwrap_swiglu(model: torch.nn.Module) -> int:
    from calm.hrm_text_158.layers import SwiGLU

    n = 0
    for mod in model.modules():
        if isinstance(mod, SwiGLU) and hasattr(mod, "_rotor_orig_forward"):
            mod.forward = mod._rotor_orig_forward
            del mod._rotor_orig_forward
            n += 1
    return n


@dataclass(frozen=True)
class RotorBitsLedger:
    surface: str
    n_values: int
    n_groups: int
    code_bits_per_value: float
    sign_plane_bits_per_value: int
    scale_dtype: str
    scale_bits_per_group: int
    payload_bits: int
    scale_bits: int
    total_bits: int
    bpw_scale_inclusive: float
    sub2_scale_inclusive: bool

    def as_dict(self) -> dict:
        return {
            "surface": self.surface,
            "n_values": self.n_values,
            "n_groups": self.n_groups,
            "code_bits_per_value": self.code_bits_per_value,
            "sign_plane_bits_per_value": self.sign_plane_bits_per_value,
            "scale_dtype": self.scale_dtype,
            "scale_bits_per_group": self.scale_bits_per_group,
            "payload_bits": self.payload_bits,
            "scale_bits": self.scale_bits,
            "total_bits": self.total_bits,
            "bpw_scale_inclusive": self.bpw_scale_inclusive,
            "sub2_scale_inclusive": self.sub2_scale_inclusive,
        }


def rotor_bits_ledger(
    n_values: int,
    bits: int,
    *,
    surface: str,
    scale_dtype: str = "fp16",
    group_size: int = ROTOR_GROUP,
) -> RotorBitsLedger:
    """Scale-inclusive bpw ledger for a rotor-quantized tensor.

    `bits` is the total index width (turbo2=2 -> 2-bit codes; turbo3=3 ->
    2-bit codes + 1-bit sign plane, matching the block layout). One scale
    per rotation group. sub2 is strict `< 2.0` scale-inclusive — the lane's
    honest-accounting rule; turbo2/fp16 lands at 2.125 and must NOT pass.
    """
    if n_values <= 0 or n_values % group_size != 0:
        raise ValueError(f"n_values={n_values} not a positive multiple of {group_size}")
    if bits not in _CENTROIDS_BY_BITS:
        raise ValueError(f"unsupported rotor bits: {bits}")
    if scale_dtype not in SCALE_BITS_BY_DTYPE:
        raise ValueError(f"unsupported scale_dtype: {scale_dtype}")
    n_groups = n_values // group_size
    scale_bits_per_group = SCALE_BITS_BY_DTYPE[scale_dtype]
    if bits == "ternary":
        # Base-3 pack, Q1_75 geometry: 128 codes as two 13-byte halves
        # (5 codes/byte, last slot of each half unused) = 26 bytes/group.
        if group_size != 128:
            raise ValueError("ternary pack accounting is defined for "
                             "group_size=128 (Q1_75 geometry)")
        code_bits = 208 / 128  # 1.625 physical
        sign_bits = 0
        payload_bits = n_groups * 208
    else:
        code_bits = 2
        sign_bits = bits - code_bits
        payload_bits = n_values * bits
    scale_bits = n_groups * scale_bits_per_group
    total_bits = payload_bits + scale_bits
    bpw = total_bits / n_values
    return RotorBitsLedger(
        surface=surface,
        n_values=n_values,
        n_groups=n_groups,
        code_bits_per_value=code_bits,
        sign_plane_bits_per_value=sign_bits,
        scale_dtype=scale_dtype,
        scale_bits_per_group=scale_bits_per_group,
        payload_bits=payload_bits,
        scale_bits=scale_bits,
        total_bits=total_bits,
        bpw_scale_inclusive=bpw,
        sub2_scale_inclusive=bpw < 2.0,
    )
