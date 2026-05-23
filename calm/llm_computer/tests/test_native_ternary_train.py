"""TTrain-B parity tests for fused-quantize STE training path.

Per codex +1 implement Phase B at msg 1779538337913-2d79fa93. Validation
gate from that msg + per-input-shape audit from msg 1779538370857-fed52af0
(2D + 3D inputs both covered since BitLinear sees (B, S, in) in HRM).

Tests:
  1. Forward value parity vs current path (4 prod shapes × 2D M={1,32}, 3D B={1,8} S={1,32})
  2. Backward gradient parity (grad_master, grad_input, grad_bias) on 3D inputs
  3. One AdamW step parity (master weight bit-close after one update)
  4. Train-mode-only guard (enable_native_train requires .train())
  5. Eval bypass (native flag doesn't fire under .eval())
  6. state_dict invariance (native flag doesn't add/remove keys)
  7. Walker enable/disable count
  8. Cached-inference + native-train mutual exclusion (enabling one clears the other)

Forward tol: atol=1e-5, rtol=1e-5 — same map, different reduction order
in scale.abs().mean(), so tight tol expected.
Grad tol: atol=1e-5, rtol=1e-5.
Optimizer-step tol: atol=1e-4, rtol=1e-4 (one AdamW update amplifies tiny noise).
"""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.bit_linear import (
    BitLinear,
    enable_bitlinears_for_native_train,
    disable_bitlinears_for_native_train,
)


# Production shapes from R1b5 seed=17 ckpt
PRODUCTION_SHAPES = [
    ("gqkv_proj",     512, 2048),
    ("o_proj",        512,  512),
    ("gate_up_proj",  512, 3072),
    ("down_proj",    1536,  512),
]


@pytest.fixture
def device():
    if not torch.cuda.is_available():
        pytest.skip("native-ternary-train requires CUDA + Triton")
    return "cuda"


def _make_pair(in_f: int, out_f: int, device: str, seed: int = 42) -> tuple[BitLinear, BitLinear]:
    """Build two BitLinear modules with identical master weights:
    one in default-path mode, one with native-train enabled.
    """
    torch.manual_seed(seed)
    g = torch.Generator(device=device).manual_seed(seed)
    bl_default = BitLinear(in_f, out_f, bias=False).to(device).to(torch.float32)
    bl_native = BitLinear(in_f, out_f, bias=False).to(device).to(torch.float32)
    # Copy master weights identically
    with torch.no_grad():
        bl_native.weight.copy_(bl_default.weight)
    bl_default.train()
    bl_native.train()
    bl_native.enable_native_train()
    return bl_default, bl_native


# ---------------------------------------------------------------------------- #
# TEST 1 — Forward value parity (2D + 3D inputs)
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=lambda s: s[0])
@pytest.mark.parametrize("input_shape", ["2D_M1", "2D_M32", "3D_B1_S1", "3D_B8_S32"])
def test_forward_value_parity(device, shape, input_shape):
    """Native train forward value bit-close to default path on (B,S,in) and (M,in)."""
    name, in_f, out_f = shape
    bl_default, bl_native = _make_pair(in_f, out_f, device)
    g = torch.Generator(device=device).manual_seed(1337)
    if input_shape == "2D_M1":
        x = torch.randn(1, in_f, generator=g, device=device, dtype=torch.float32)
    elif input_shape == "2D_M32":
        x = torch.randn(32, in_f, generator=g, device=device, dtype=torch.float32)
    elif input_shape == "3D_B1_S1":
        x = torch.randn(1, 1, in_f, generator=g, device=device, dtype=torch.float32)
    elif input_shape == "3D_B8_S32":
        x = torch.randn(8, 32, in_f, generator=g, device=device, dtype=torch.float32)
    else:
        raise AssertionError(input_shape)

    y_default = bl_default(x)
    y_native = bl_native(x)
    assert y_default.shape == y_native.shape, f"shape mismatch: {y_default.shape} vs {y_native.shape}"
    torch.testing.assert_close(y_native, y_default, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------- #
# TEST 2 — Backward gradient parity (3D input, no bias)
# ---------------------------------------------------------------------------- #


@pytest.mark.parametrize("shape", PRODUCTION_SHAPES, ids=lambda s: s[0])
def test_backward_grad_parity_3D(device, shape):
    """grad_master + grad_input bit-close after one backward on 3D input."""
    name, in_f, out_f = shape
    bl_default, bl_native = _make_pair(in_f, out_f, device)
    B, S = 4, 16
    g = torch.Generator(device=device).manual_seed(2024)
    x_default = torch.randn(B, S, in_f, generator=g, device=device,
                            dtype=torch.float32, requires_grad=True)
    x_native = x_default.detach().clone().requires_grad_(True)
    y_default = bl_default(x_default)
    y_native = bl_native(x_native)
    # Same loss across both for fair gradient comparison
    g2 = torch.Generator(device=device).manual_seed(3030)
    grad_out = torch.randn_like(y_default, generator=g2)
    y_default.backward(grad_out)
    y_native.backward(grad_out)
    # Master weight gradient parity (STE identity)
    torch.testing.assert_close(
        bl_native.weight.grad, bl_default.weight.grad,
        atol=1e-5, rtol=1e-5,
    )
    # Input gradient parity (uses quantized fwd weight in both paths)
    torch.testing.assert_close(x_native.grad, x_default.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------- #
# TEST 3 — Backward gradient parity with bias
# ---------------------------------------------------------------------------- #


def test_backward_grad_parity_with_bias(device):
    """Bias grad path: grad_bias = grad_output.sum_over_batch."""
    in_f, out_f = 512, 2048
    torch.manual_seed(11)
    bl_default = BitLinear(in_f, out_f, bias=True).to(device).to(torch.float32)
    bl_native = BitLinear(in_f, out_f, bias=True).to(device).to(torch.float32)
    with torch.no_grad():
        bl_native.weight.copy_(bl_default.weight)
        bl_native.bias.copy_(bl_default.bias)
    bl_default.train()
    bl_native.train()
    bl_native.enable_native_train()

    g = torch.Generator(device=device).manual_seed(101)
    x_d = torch.randn(8, 16, in_f, generator=g, device=device, dtype=torch.float32, requires_grad=True)
    x_n = x_d.detach().clone().requires_grad_(True)
    y_d = bl_default(x_d)
    y_n = bl_native(x_n)
    g2 = torch.Generator(device=device).manual_seed(202)
    go = torch.randn_like(y_d, generator=g2)
    y_d.backward(go)
    y_n.backward(go)
    torch.testing.assert_close(bl_native.bias.grad, bl_default.bias.grad,
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(bl_native.weight.grad, bl_default.weight.grad,
                               atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(x_n.grad, x_d.grad, atol=1e-5, rtol=1e-5)


# ---------------------------------------------------------------------------- #
# TEST 4 — One AdamW step parity
# ---------------------------------------------------------------------------- #


def test_one_adamw_step_parity(device):
    """After one optimizer step, master weights bit-close between paths."""
    in_f, out_f = 512, 2048
    bl_default, bl_native = _make_pair(in_f, out_f, device, seed=77)
    opt_default = torch.optim.AdamW(bl_default.parameters(), lr=1e-3)
    opt_native = torch.optim.AdamW(bl_native.parameters(), lr=1e-3)
    g = torch.Generator(device=device).manual_seed(123)
    x = torch.randn(4, 8, in_f, generator=g, device=device, dtype=torch.float32)
    target = torch.randn(4, 8, out_f, generator=g, device=device, dtype=torch.float32)
    # Default path step
    opt_default.zero_grad()
    y_d = bl_default(x)
    loss_d = ((y_d - target) ** 2).mean()
    loss_d.backward()
    opt_default.step()
    # Native path step (independent module + opt; same input/target)
    opt_native.zero_grad()
    y_n = bl_native(x)
    loss_n = ((y_n - target) ** 2).mean()
    loss_n.backward()
    opt_native.step()
    # Master weights bit-close after one AdamW update
    torch.testing.assert_close(bl_native.weight, bl_default.weight,
                               atol=1e-4, rtol=1e-4)


# ---------------------------------------------------------------------------- #
# TEST 5 — Train-mode-only guard
# ---------------------------------------------------------------------------- #


def test_enable_native_train_requires_train_mode(device):
    bl = BitLinear(512, 512, bias=False).to(device).to(torch.float32)
    bl.eval()
    with pytest.raises(RuntimeError, match="must be called in training mode"):
        bl.enable_native_train()


# ---------------------------------------------------------------------------- #
# TEST 6 — Eval bypass (native flag does NOT fire under .eval())
# ---------------------------------------------------------------------------- #


def test_eval_bypasses_native_train(device):
    """After model.eval(), forward must NOT take the native path even if
    the flag was set in training. The train() override clears it.
    """
    in_f, out_f = 512, 512
    bl = BitLinear(in_f, out_f, bias=False).to(device).to(torch.float32)
    bl.train()
    bl.enable_native_train()
    assert bl._native_train_active is True
    bl.eval()
    assert bl._native_train_active is False, (
        "train(False) must clear _native_train_active per defense-in-depth"
    )
    # Forward in eval should hit default path; no Triton import required
    x = torch.randn(2, in_f, device=device, dtype=torch.float32)
    y = bl(x)
    assert y.shape == (2, out_f)


# ---------------------------------------------------------------------------- #
# TEST 7 — state_dict invariance
# ---------------------------------------------------------------------------- #


def test_state_dict_invariance(device):
    """Enabling/disabling native-train must NOT change state_dict keys or values."""
    bl = BitLinear(512, 512, bias=True).to(device).to(torch.float32)
    bl.train()
    sd_before = {k: v.clone() for k, v in bl.state_dict().items()}
    bl.enable_native_train()
    sd_after_enable = {k: v.clone() for k, v in bl.state_dict().items()}
    bl.disable_native_train()
    sd_after_disable = {k: v.clone() for k, v in bl.state_dict().items()}

    assert set(sd_before.keys()) == set(sd_after_enable.keys()) == set(sd_after_disable.keys()), (
        f"state_dict keys changed: {set(sd_before.keys())} → "
        f"{set(sd_after_enable.keys())} → {set(sd_after_disable.keys())}"
    )
    for k in sd_before:
        torch.testing.assert_close(sd_before[k], sd_after_enable[k], atol=0, rtol=0)
        torch.testing.assert_close(sd_before[k], sd_after_disable[k], atol=0, rtol=0)


# ---------------------------------------------------------------------------- #
# TEST 8 — Walker enable/disable count
# ---------------------------------------------------------------------------- #


def test_walker_enable_disable_count(device):
    """enable_bitlinears_for_native_train returns count of BL modules visited."""
    import torch.nn as nn
    class Wrap(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(64, 128, bias=False)
            self.b = BitLinear(128, 64, bias=False)
            self.c = nn.Linear(64, 64)  # NOT a BitLinear; must be skipped

    w = Wrap().to(device).to(torch.float32)
    w.train()
    n = enable_bitlinears_for_native_train(w)
    assert n == 2, f"expected 2 BitLinear modules enabled, got {n}"
    assert w.a._native_train_active and w.b._native_train_active
    n2 = disable_bitlinears_for_native_train(w)
    assert n2 == 2
    assert not w.a._native_train_active and not w.b._native_train_active


# ---------------------------------------------------------------------------- #
# TEST 9 — Cached-inference + native-train mutual exclusion
# ---------------------------------------------------------------------------- #


def test_native_train_clears_cached_active(device):
    """Enabling native-train clears _cached_active to avoid path collision."""
    bl = BitLinear(512, 512, bias=False).to(device).to(torch.float32)
    # Freeze for inference
    bl.eval()
    bl.freeze_for_inference()
    assert bl._cached_active is True
    # Switch back to train and enable native
    bl.train()
    # train(True) already clears cached_active per existing guard
    assert bl._cached_active is False
    bl.enable_native_train()
    assert bl._native_train_active is True
    # _cached_active should also be False
    assert bl._cached_active is False
