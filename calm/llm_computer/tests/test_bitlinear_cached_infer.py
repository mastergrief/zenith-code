"""T1 (α) parity tests for BitLinear cached-ternary inference path.

Per codex msg 1779528934673-1c8bedf3:
- Bit-equivalence: cached forward equals re-quantize forward at seq_len {1, 16, 32}
  on the 4 representative BitLinear shapes seen in the R1b5 seed=17 probe
  inventory (BL[512x2048], BL[512x512], BL[512x3072], BL[1536x512]).
- Training-bypass guard: under `model.train()`, cached path is NOT used even
  when cache is set — `forward()` must re-quantize.
- Freeze/unfreeze cycle: master weights untouched; cache toggles cleanly.
- train()->eval() transition: cache invalidates on entering train mode
  (defense in depth alongside `self.training` guard in forward).
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from calm.hrm_text_158.bit_linear import BitLinear, freeze_bitlinears_for_inference


SHAPES = [
    (512, 2048),  # gqkv_proj (8 Q heads + 4 KV heads * 128)
    (512, 512),   # o_proj
    (512, 3072),  # gate_up_proj (expansion=3, hidden -> 2*exp*hidden)
    (1536, 512),  # down_proj (exp*hidden -> hidden)
]

SEQ_LENS = [1, 16, 32]


@pytest.fixture
def device():
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.mark.parametrize("in_features,out_features", SHAPES)
@pytest.mark.parametrize("seq_len", SEQ_LENS)
def test_cached_forward_exact_parity(in_features: int, out_features: int,
                                      seq_len: int, device: str):
    """Cached forward must produce bit-identical output to re-quantize forward.

    Bit-identical because cached path stores exactly `w_q * scale` (the
    forward value of the STE expression `w + sg(w_q*scale - w)`) — same
    F.linear call with the same weight tensor.
    """
    torch.manual_seed(42)
    bl = BitLinear(in_features, out_features, bias=False).to(device=device, dtype=torch.float32)
    bl.eval()
    x = torch.randn(1, seq_len, in_features, device=device, dtype=torch.float32)

    # Baseline: re-quantize path
    with torch.no_grad():
        y_baseline = bl(x)

    # Freeze and call again
    bl.freeze_for_inference()
    assert bl._cached_active is True
    assert bl._cached_weight is not None
    with torch.no_grad():
        y_cached = bl(x)

    # Bit-identical: same weight tensor, same F.linear, same activation
    assert torch.equal(y_baseline, y_cached), (
        f"Cached forward differs from re-quantize at shape ({in_features},"
        f"{out_features}) seq_len={seq_len}: max abs diff="
        f"{(y_baseline - y_cached).abs().max().item():.3e}"
    )


@pytest.mark.parametrize("in_features,out_features", SHAPES)
def test_cached_forward_with_bias(in_features: int, out_features: int, device: str):
    """Bias path: cached forward + bias matches re-quantize + bias."""
    torch.manual_seed(7)
    bl = BitLinear(in_features, out_features, bias=True).to(device=device, dtype=torch.float32)
    bl.bias.data = torch.randn_like(bl.bias.data) * 0.1
    bl.eval()
    x = torch.randn(1, 16, in_features, device=device, dtype=torch.float32)
    with torch.no_grad():
        y_baseline = bl(x)
    bl.freeze_for_inference()
    with torch.no_grad():
        y_cached = bl(x)
    assert torch.equal(y_baseline, y_cached)


def test_training_mode_bypasses_cache(device: str):
    """Under `model.train()`, the cached path must NOT be used.

    Two layers of defense:
      1. `forward()` checks `self.training` and re-quantizes when training.
      2. `train(mode=True)` clears `_cached_active=False`.

    Both must agree before cached path runs.
    """
    torch.manual_seed(11)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    bl.eval()
    bl.freeze_for_inference()
    assert bl._cached_active is True

    # Flip to training
    bl.train()
    assert bl._cached_active is False, "train(True) must clear cache active flag"

    # Even if we manually re-set the flag (simulating a bug), forward must
    # still re-quantize under training mode because `self.training` is True.
    bl._cached_active = True
    x = torch.randn(1, 8, 512, device=device, dtype=torch.float32, requires_grad=False)
    # In train mode, forward should NOT use cache regardless of _cached_active.
    # We test this indirectly: mutate master weight and check forward output
    # tracks the new master, not the stale cache.
    bl.weight.data += 1.0
    with torch.no_grad():
        y_after_mutation = bl(x)
    # Compare against a freshly-quantized baseline of the mutated weight
    w_q_ste, _ = bl.quantize_weight()
    y_expected = F.linear(x, w_q_ste, bl.bias)
    assert torch.equal(y_after_mutation, y_expected), (
        "forward() under train mode must re-quantize from CURRENT master, "
        "not return stale cache"
    )


def test_unfreeze_then_freeze_cycle(device: str):
    """Cache can be cleared and re-created without disturbing master."""
    torch.manual_seed(13)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    bl.eval()
    master_before = bl.weight.data.clone()
    bl.freeze_for_inference()
    cached1 = bl._cached_weight.clone()
    bl.unfreeze()
    assert bl._cached_active is False
    assert bl._cached_weight is None
    bl.freeze_for_inference()
    cached2 = bl._cached_weight
    # Master untouched across cycle
    assert torch.equal(bl.weight.data, master_before)
    # Re-freezing on unchanged master yields the same cache tensor
    assert torch.equal(cached1, cached2)


def test_freeze_walker_counts_all_bitlinears(device: str):
    """`freeze_bitlinears_for_inference(module)` walks all BitLinears."""
    import torch.nn as nn

    class Mini(nn.Module):
        def __init__(self):
            super().__init__()
            self.a = BitLinear(64, 64, bias=False)
            self.b = BitLinear(64, 128, bias=False)
            self.linear = nn.Linear(64, 64)  # NOT a BitLinear; should be skipped
            self.c = BitLinear(128, 64, bias=False)

    m = Mini().to(device=device, dtype=torch.float32)
    m.eval()
    n = freeze_bitlinears_for_inference(m)
    assert n == 3
    assert m.a._cached_active is True
    assert m.b._cached_active is True
    assert m.c._cached_active is True


def test_cache_not_in_state_dict(device: str):
    """Cached weight MUST NOT enter state_dict (no .pt schema change)."""
    torch.manual_seed(17)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    bl.eval()
    sd_before = set(bl.state_dict().keys())
    bl.freeze_for_inference()
    sd_after = set(bl.state_dict().keys())
    assert sd_before == sd_after, (
        f"state_dict keys changed after freeze: added "
        f"{sd_after - sd_before}, removed {sd_before - sd_after}"
    )
    # The only key should be 'weight' (bias=False)
    assert sd_after == {"weight"}, f"unexpected state_dict keys: {sd_after}"


def test_cached_path_skips_quantize(device: str):
    """Sanity: changing master weight while cached is active does NOT change
    forward output (proves cache is being used, not re-quantized)."""
    torch.manual_seed(19)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    bl.eval()
    x = torch.randn(1, 8, 512, device=device, dtype=torch.float32)
    bl.freeze_for_inference()
    with torch.no_grad():
        y1 = bl(x)
    # Mutate master directly (bypasses parameter machinery)
    with torch.no_grad():
        bl.weight.data += 1.0
    with torch.no_grad():
        y2 = bl(x)
    # Output should NOT change because cache holds the pre-mutation weight
    assert torch.equal(y1, y2), (
        "cached path failed to bypass quantize step — output tracked "
        "master mutation, meaning forward re-quantized despite _cached_active=True"
    )


def test_freeze_for_inference_raises_in_training_mode(device: str):
    """`freeze_for_inference()` must fail-fast when called in training mode.

    Closes the load-bearing hole codex flagged at msg 1779529701708-b4564ba8:
    if freeze were allowed in training mode, `_cached_active=True` would set
    immediately (the `self.training` forward guard masks it then), but
    subsequent training weight updates would NOT invalidate the cache —
    `train(False)` does not touch `_cached_active`, only `train(True)` clears
    it. The next `.eval()` would then consume a stale cached weight.
    """
    torch.manual_seed(23)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    bl.train()  # explicit training mode
    assert bl.training is True
    with pytest.raises(RuntimeError, match="eval mode"):
        bl.freeze_for_inference()
    # Cache must NOT have been set
    assert bl._cached_active is False
    assert bl._cached_weight is None


def test_train_then_eval_cannot_use_stale_cache(device: str):
    """Train → mutate weights → eval must NOT consume a stale cache.

    Concrete failure scenario codex described:
      1. eval → freeze → cache populated, _cached_active=True
      2. train(True) → cache cleared (defense in depth, _cached_active=False)
      3. master weights mutate (training step)
      4. eval() → _cached_active stays False (train(False) doesn't reactivate)
      5. forward in eval → must re-quantize from CURRENT master

    Without the train(True) cache invalidation, step 4 would consume a
    stale cache. This test exercises the entire round-trip.
    """
    torch.manual_seed(29)
    bl = BitLinear(512, 512, bias=False).to(device=device, dtype=torch.float32)
    x = torch.randn(1, 8, 512, device=device, dtype=torch.float32)

    # 1. eval + freeze
    bl.eval()
    bl.freeze_for_inference()
    assert bl._cached_active is True

    # 2. train mode -> cache cleared by train(True) override
    bl.train()
    assert bl._cached_active is False, "train(True) must clear cache"

    # 3. simulate weight mutation under training
    with torch.no_grad():
        bl.weight.data += 5.0

    # 4. flip back to eval — _cached_active stays False (train(False) is a no-op
    #    on the flag), so forward will re-quantize the MUTATED master
    bl.eval()
    assert bl._cached_active is False, (
        "train(False) must NOT reactivate stale cache flag"
    )

    # 5. forward in eval must match a fresh re-quantize of the mutated master,
    #    NOT the stale pre-mutation cache.
    with torch.no_grad():
        y_actual = bl(x)
    w_q_ste, _ = bl.quantize_weight()
    y_expected_fresh = F.linear(x, w_q_ste, bl.bias)
    assert torch.equal(y_actual, y_expected_fresh), (
        "post-train eval forward consumed stale cache instead of "
        "re-quantizing from current master"
    )
