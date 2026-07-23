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

# Constant tensors cached per (device, dtype[, bits]). The pack hook fires
# per saved tensor per forward — rebuilding these from Python lists there
# means a host->device copy + fresh tiny allocation each call, which is the
# allocator-churn/H2D hot-spot that collapsed trainer pace near-full VRAM.
_SIGN_CACHE: dict = {}
_CENTROID_CACHE: dict = {}


def _sign_consts(dtype: torch.dtype, device: torch.device):
    key = (str(device), dtype)
    hit = _SIGN_CACHE.get(key)
    if hit is None:
        hit = (torch.tensor(_S1, dtype=dtype, device=device),
               torch.tensor(_S2, dtype=dtype, device=device))
        _SIGN_CACHE[key] = hit
    return hit


def _centroid_consts(bits, dtype: torch.dtype, device: torch.device):
    key = (str(device), dtype, bits)
    hit = _CENTROID_CACHE.get(key)
    if hit is None:
        centroids = torch.tensor(
            _CENTROIDS_BY_BITS[bits], dtype=dtype, device=device)
        boundaries = (centroids[:-1] + centroids[1:]) / 2
        hit = (centroids, boundaries)
        _CENTROID_CACHE[key] = hit
    return hit


def _signed_fwht(x: torch.Tensor) -> torch.Tensor:
    """Forward signed WHT over the last dim (must be ROTOR_GROUP).

    y = D2 * (1/sqrt(128)) * H * D1 * x, matching turbo_cpu_fwht.
    Orthonormal: preserves L2 norm exactly (up to float error).
    """
    s1, s2 = _sign_consts(x.dtype, x.device)
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
    s1, s2 = _sign_consts(y.dtype, y.device)
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

    # Nearest centroid via bucketize on midpoint boundaries — identical to
    # argmin over |x - c| for sorted centroids, without materializing the
    # (..., levels) intermediate (allocator-churn fix for the trainer hook).
    centroids, boundaries = _centroid_consts(bits, rot.dtype, rot.device)
    idx = torch.bucketize(rot, boundaries)
    quant = centroids[idx]

    recon_norm = quant.norm(dim=-1, keepdim=True)
    corrected = torch.where(recon_norm > 1e-10, grp_norm / recon_norm.clamp(min=1e-30), grp_norm)
    # The stored scale is fp16 in the block layout — round through fp16.
    corrected = corrected.to(torch.float16).float()

    out = _signed_fwht_inverse(quant) * corrected
    return out.reshape(x.shape).to(orig_dtype)


# ---------------------------------------------------------------------------
# Allocation-free pack path (attempt-7 classification: at backward-peak the
# pack hook's ~25 transient allocations per call trigger a WDDM eviction
# storm on a near-full WSL card — FP control at identical recipe never
# stalls; gc_threshold / chunk-size / constant-cache levers all nulled).
# Every intermediate below lives in a cached per-shape workspace and all ops
# are `out=`/in-place; the ONLY allocation per call is the returned saved
# tensor itself. Bit-exact vs rotor_fake_quant (unit-tested via torch.equal).
# ---------------------------------------------------------------------------

_SCALED_SIGN_CACHE: dict = {}


def _scaled_sign_consts(dtype: torch.dtype, device: torch.device):
    """(inv_sqrt*s2, inv_sqrt*s1) — the post-butterfly multipliers, cached so
    the scalar*vector product is not re-materialized per call (reference
    computes the identical product inline)."""
    key = (str(device), dtype)
    hit = _SCALED_SIGN_CACHE.get(key)
    if hit is None:
        s1, s2 = _sign_consts(dtype, device)
        hit = (_INV_SQRT_128 * s2, _INV_SQRT_128 * s1)
        _SCALED_SIGN_CACHE[key] = hit
    return hit


def _fwht_inplace(y: torch.Tensor, tmp: torch.Tensor,
                  s_pre: torch.Tensor, s_post_scaled: torch.Tensor) -> None:
    """In-place signed FWHT over the last dim using a half-size temp buffer.

    Elementwise-identical to `_signed_fwht`/`_signed_fwht_inverse` (same
    a+b / a-b pairs in the same positions), zero allocations.
    """
    y.mul_(s_pre)
    n = y.shape[-1]
    h = 1
    while h < n:
        v = y.view(*y.shape[:-1], n // (2 * h), 2, h)
        a = v[..., 0, :]
        b = v[..., 1, :]
        t = tmp.view(a.shape)
        torch.sub(a, b, out=t)
        a.add_(b)
        b.copy_(t)
        h *= 2
    y.mul_(s_post_scaled)


class _PackWorkspace:
    """Preallocated intermediates for one (max_rows, last_dim) chunk shape."""

    def __init__(self, rows: int, last: int,
                 device: torch.device):
        ng = last // ROTOR_GROUP
        f32 = torch.float32
        self.rows = rows
        self.xf = torch.empty(rows, ng, ROTOR_GROUP, dtype=f32, device=device)
        self.tmp = torch.empty(rows, ng, ROTOR_GROUP // 2, dtype=f32,
                               device=device)
        self.grp = torch.empty(rows, ng, 1, dtype=f32, device=device)
        self.inv = torch.empty(rows, ng, 1, dtype=f32, device=device)
        self.maskb = torch.empty(rows, ng, 1, dtype=torch.bool, device=device)
        self.recon = torch.empty(rows, ng, 1, dtype=f32, device=device)
        self.corr = torch.empty(rows, ng, 1, dtype=f32, device=device)
        self.corr_h = torch.empty(rows, ng, 1, dtype=torch.float16,
                                  device=device)
        self.idx = torch.empty(rows, ng, ROTOR_GROUP, dtype=torch.int64,
                               device=device)


def _fake_quant_ws_chunk(x2d: torch.Tensor, bits, ws: _PackWorkspace,
                         out2d: torch.Tensor) -> None:
    """Workspace fake-quant of a contiguous (rows, last) chunk into out2d.

    Reproduces rotor_fake_quant exactly: same op sequence, same rounding.
    """
    r = x2d.shape[0]
    xf = ws.xf[:r]
    xf.view(r, -1).copy_(x2d)

    grp = ws.grp[:r]
    torch.linalg.vector_norm(xf, 2, dim=-1, keepdim=True, out=grp)

    # inv = where(grp > 1e-10, 1/clamp(grp, 1e-30), 0)
    inv = ws.inv[:r]
    inv.copy_(grp).clamp_(min=1e-30).reciprocal_()
    maskb = ws.maskb[:r]
    torch.gt(grp, 1e-10, out=maskb)
    inv.mul_(maskb)

    s1, s2 = _sign_consts(torch.float32, x2d.device)
    s2_scaled, s1_scaled = _scaled_sign_consts(torch.float32, x2d.device)

    xf.mul_(inv)
    _fwht_inplace(xf, ws.tmp[:r], s1, s2_scaled)

    centroids, boundaries = _centroid_consts(bits, torch.float32, x2d.device)
    idx = ws.idx[:r]
    torch.bucketize(xf, boundaries, out=idx)
    quant = xf  # reuse: overwrite rotated values with their centroids
    torch.index_select(centroids, 0, idx.view(-1),
                       out=quant.view(-1))

    recon = ws.recon[:r]
    torch.linalg.vector_norm(quant, 2, dim=-1, keepdim=True, out=recon)
    # corrected = where(recon > 1e-10, grp/clamp(recon, 1e-30), grp)
    corr = ws.corr[:r]
    corr.copy_(recon).clamp_(min=1e-30)
    torch.div(grp, corr, out=corr)
    torch.gt(recon, 1e-10, out=maskb)
    torch.where(maskb, corr, grp, out=corr)
    # fp16 round-trip of the stored scale
    corr_h = ws.corr_h[:r]
    corr_h.copy_(corr)
    corr.copy_(corr_h)

    _fwht_inplace(quant, ws.tmp[:r], s2, s1_scaled)
    quant.mul_(corr)
    out2d.copy_(quant.view(r, -1))


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

    # Bound the fake-quant transient working set: the FWHT pipeline
    # allocates roughly an order of magnitude of float32 intermediates per
    # call, so quantizing a whole (batch, seq, 512) save in one shot spikes
    # ~GB-scale on a near-full card (the trainer bp_steps>=4 stall class).
    # Chunking over rows is exact — quantization is per-128-group.
    _CHUNK_VALUES = 2_097_152

    def _fake_quant_bounded(self, t: torch.Tensor) -> torch.Tensor:
        flat = t.reshape(-1, t.shape[-1]).contiguous()
        rows = max(1, self._CHUNK_VALUES // t.shape[-1])
        # Workspace path: per-(last_dim, device) preallocated intermediates;
        # the returned `out` is the single allocation of this call.
        if not hasattr(self, "_ws_cache"):
            self._ws_cache = {}
        key = (t.shape[-1], str(t.device))
        ws = self._ws_cache.get(key)
        if ws is None or ws.rows < min(rows, flat.shape[0]):
            ws = _PackWorkspace(rows, t.shape[-1], t.device)
            self._ws_cache[key] = ws
        out = torch.empty_like(flat)
        for i in range(0, flat.shape[0], rows):
            chunk = flat[i:i + rows]
            _fake_quant_ws_chunk(chunk, self.bits, ws, out[i:i + rows])
        return out.reshape(t.shape).to(t.dtype)

    def pack(self, t: torch.Tensor):
        with torch.no_grad():
            if self.bits is not None and self._eligible(t):
                self.quantized += 1
                self.quantized_values += t.numel()
                return self._fake_quant_bounded(t.detach())
            self.passed += 1
            if torch.is_floating_point(t):
                self.passed_values += t.numel()
            # LOAD-BEARING .detach(): returning a saved tensor that still
            # carries grad_fn from pack creates a C++<->Python reference
            # cycle (graph Node -> packed PyObject -> grad_fn -> Node) that
            # neither refcounting nor gc can break -> ~27MB/step CUDA leak
            # (leak-diag receipt: retained (3064,260) logits roots). The
            # detached view shares storage; saved-tensor semantics unchanged.
            return t.detach() if torch.is_tensor(t) else t

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
