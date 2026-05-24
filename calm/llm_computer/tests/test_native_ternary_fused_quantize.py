"""Focused parity test for the libdevice-free fused-quantize round.

Per gabe-greenlit portable kernel patch (chat AUQ 2026-05-24) + codex
plan-gate +1 (msg 1779635421816). Proves
ternary_train_kernel.fused_quantize() rounds half-to-even (matches
torch.round) WITHOUT tl.extra.libdevice, so it JIT-compiles on Triton
versions that don't expose libdevice (triton 3.1.0 / Pascal cu121) and stays
bit-equivalent on those that do (triton 3.6.0).

codex acceptance conditions (msg 1779635421816):
  - half-to-even, NOT half-away-from-zero
  - visible tie cases INSIDE the ternary clamp: q = -0.5, +0.5 and just-off
    ties -0.5001, -0.4999, +0.4999, +0.5001 (these distinguish the two
    rounding modes; +/-1.5 ties are clamp-masked so insufficient alone)
  - run raw parity on both lanes (box triton 3.1.0, 4070 triton 3.6.0)
"""
from __future__ import annotations

import pytest
import torch

cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused_quantize requires CUDA"
)


@cuda
def test_fused_quantize_tie_cases_half_to_even():
    """q exactly on / just off the ±0.5 ties.

    scale=1.0 makes q == w exactly (w*inv_s == w/scale == w, all fp32-exact),
    so this is a bit-equivalence check against torch.round half-to-even AND
    the discriminating proof that we are NOT half-away-from-zero (which would
    map ±0.5 -> ±1 instead of 0).
    """
    from calm.hrm_text_158.ternary_train_kernel import fused_quantize

    q_vals = [
        -1.5, -0.5001, -0.5, -0.4999, 0.0,
        0.4999, 0.5, 0.5001, 1.5,
        -1.0, 1.0, 0.25, -0.75,
    ]
    scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")
    w = torch.tensor(q_vals, dtype=torch.float32, device="cuda")

    out = fused_quantize(w, scale)
    # torch.round is half-to-even; *scale is identity at scale=1.0.
    ref = torch.round(w / scale).clamp(-1.0, 1.0) * scale

    assert torch.equal(out, ref), (
        f"\nq   ={q_vals}\nout ={out.tolist()}\nref ={ref.tolist()}"
    )
    # Explicit half-to-even anchors (would FAIL under half-away-from-zero):
    assert out[q_vals.index(0.5)].item() == 0.0       # round(0.5)=0 (even)
    assert out[q_vals.index(-0.5)].item() == 0.0      # round(-0.5)=0 (even)
    # just-off ties resolve by magnitude
    assert out[q_vals.index(0.5001)].item() == 1.0
    assert out[q_vals.index(0.4999)].item() == 0.0
    assert out[q_vals.index(-0.5001)].item() == -1.0
    assert out[q_vals.index(-0.4999)].item() == 0.0
    # clamp-masked ties still land correctly post-clamp
    assert out[q_vals.index(1.5)].item() == 1.0       # round(1.5)=2 -> clamp 1
    assert out[q_vals.index(-1.5)].item() == -1.0     # round(-1.5)=-2 -> clamp -1


@cuda
def test_fused_quantize_broad_random_parity():
    """Distribution-wide bit-equivalence to the kernel's intended math
    round(w * 1/scale).clamp(-1,1)*scale across several scales + 2-D shapes.

    Mirrors the kernel's multiply-by-inverse (q = w * inv_s) so the test
    isolates the rounding change — the only thing the patch touches.
    """
    from calm.hrm_text_158.ternary_train_kernel import fused_quantize

    torch.manual_seed(17)
    shapes = [(512, 2048), (512, 512), (1536, 512), (64, 64), (1, 1)]
    for shp in shapes:
        w = torch.randn(shp, dtype=torch.float32, device="cuda")
        scale_vals = (w.abs().mean().clamp(min=1e-5).item(), 1.0, 0.5, 2.0)
        for scale_val in scale_vals:
            scale = torch.tensor(scale_val, dtype=torch.float32, device="cuda")
            out = fused_quantize(w, scale)
            inv = 1.0 / scale
            ref = torch.round(w * inv).clamp(-1.0, 1.0) * scale
            assert torch.equal(out, ref), (
                f"shape={shp} scale={scale_val} max_abs_diff="
                f"{(out - ref).abs().max().item():.3e}"
            )
            # output must be ternary in {-scale, 0, +scale}
            allowed = torch.tensor(
                [-scale_val, 0.0, scale_val], device="cuda"
            )
            for u in torch.unique(out):
                assert torch.isclose(u, allowed).any(), (
                    f"non-ternary value {u.item()} (scale={scale_val})"
                )
