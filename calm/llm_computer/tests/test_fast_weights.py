"""Unit tests for FastWeightSmall2DTransformer — Round 1 mechanism correctness.

Tests the fast-weights subclass in isolation. The associative-recall
benchmark lives in scripts/experiment_fast_weights.py.
"""

from __future__ import annotations

import torch

from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.fast_weights import (
    FastWeightConfig, FastWeightSmall2DTransformer,
)


def test_disabled_matches_vanilla():
    """With use_fast_weights=False the subclass must match the parent bitwise.

    Regression check that disabling the mechanism restores parent behavior —
    any deviation here means the subclass broke something it shouldn't touch.
    """
    torch.manual_seed(0)
    cfg_v = Small2DConfig(
        vocab_size=32, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
    )
    cfg_f = FastWeightConfig(
        vocab_size=32, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        use_fast_weights=False,
    )
    m_v = Small2DTransformer(cfg_v)
    m_f = FastWeightSmall2DTransformer(cfg_f)
    m_f.load_state_dict(m_v.state_dict())  # strict: no parameter mismatch

    x = torch.randint(0, 32, (3, 7))
    with torch.no_grad():
        out_v = m_v(x)
        out_f = m_f(x)
    assert torch.equal(out_v, out_f), \
        "FastWeightSmall2DTransformer with use_fast_weights=False diverges from parent"


def test_zero_state_single_write():
    """Write one (k, v) pair into empty state, read back with q=k, expect v.

    Verifies the core mechanism: outer(v, k) stored, then W_fast @ k == v.
    Tests _fast_weight_step in isolation without going through the model.
    """
    B, D = 1, 4
    W_fast = torch.zeros(B, D, D)
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0, 0.0]])

    # Step 1: write the pair. Read from empty state is zero (correct).
    W_fast, read_empty = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, torch.zeros(B, D), k, v, lambda_decay=1.0, eta_write=1.0,
    )
    assert torch.allclose(read_empty, torch.zeros(B, D)), \
        "read from empty state should be zero"

    # Step 2: query with q=k, no further writes. Expect read == v.
    _, read_retrieved = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, k, torch.zeros(B, D), torch.zeros(B, D),
        lambda_decay=1.0, eta_write=0.0,
    )
    assert torch.allclose(read_retrieved, v, atol=1e-6), \
        f"expected to retrieve {v}, got {read_retrieved}"


def test_decay_zeros_state():
    """λ<1 with no writes must decay W_fast toward zero."""
    B, D = 1, 4
    W_fast = torch.zeros(B, D, D)
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    # Write once to populate.
    W_fast, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, torch.zeros(B, D), k, v, lambda_decay=1.0, eta_write=1.0,
    )
    initial_norm = W_fast.norm().item()
    assert initial_norm > 0.5, "write should populate state"

    # Decay 20 steps with λ=0.1, no new writes.
    for _ in range(20):
        W_fast, _ = FastWeightSmall2DTransformer._fast_weight_step(
            W_fast,
            torch.zeros(B, D), torch.zeros(B, D), torch.zeros(B, D),
            lambda_decay=0.1, eta_write=0.0,
        )
    final_norm = W_fast.norm().item()
    # 0.1 ** 20 ≈ 1e-20 — should be effectively zero.
    assert final_norm < 1e-10, \
        f"W_fast should decay to near-zero, got norm={final_norm}"


def test_batch_independence():
    """Fast-weight state must not leak across batch elements.

    Runs two independent sequences side-by-side in a batch, then separately,
    and checks outputs match. A batch-coupling bug would make the batched
    run differ from the per-sequence runs.
    """
    torch.manual_seed(0)
    cfg = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        lambda_decay=0.9, eta_write=0.5, use_fast_weights=True,
    )
    model = FastWeightSmall2DTransformer(cfg)
    model.eval()

    x1 = torch.tensor([[1, 2, 3, 4, 5]])
    x2 = torch.tensor([[6, 7, 8, 9, 10]])
    batch = torch.cat([x1, x2], dim=0)

    with torch.no_grad():
        out_batch = model(batch)
        out1 = model(x1)
        out2 = model(x2)
    assert torch.allclose(out_batch[0:1], out1, atol=1e-6), \
        "batch element 0 output differs from standalone run"
    assert torch.allclose(out_batch[1:2], out2, atol=1e-6), \
        "batch element 1 output differs from standalone run"


def test_fast_weights_change_output():
    """Sanity: enabling fast weights must change the output vs disabled.

    If use_fast_weights=True produces the same output as False, the
    mechanism isn't wired into the residual stream.
    """
    torch.manual_seed(0)
    cfg_on = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        lambda_decay=0.9, eta_write=0.5, use_fast_weights=True,
    )
    cfg_off = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        lambda_decay=0.9, eta_write=0.5, use_fast_weights=False,
    )
    m_on = FastWeightSmall2DTransformer(cfg_on)
    m_off = FastWeightSmall2DTransformer(cfg_off)
    m_off.load_state_dict(m_on.state_dict())

    x = torch.randint(0, 16, (2, 7))
    with torch.no_grad():
        out_on = m_on(x)
        out_off = m_off(x)
    assert not torch.allclose(out_on, out_off, atol=1e-4), \
        "fast weights produce identical output to disabled — mechanism not wired in"


if __name__ == "__main__":
    test_disabled_matches_vanilla()
    print("[ok] disabled subclass matches vanilla bitwise")
    test_zero_state_single_write()
    print("[ok] single write + retrieval via _fast_weight_step")
    test_decay_zeros_state()
    print("[ok] λ<1 decays state to zero")
    test_batch_independence()
    print("[ok] per-batch state is independent")
    test_fast_weights_change_output()
    print("[ok] enabled fast weights do change output vs disabled")
