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


def test_delta_rule_overwrites_same_key():
    """Round 4: writing (k, v2) after (k, v1) with delta rule must retrieve v2.
    Without the delta rule, the second write stacks, giving v1+v2."""
    B, D = 1, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v1 = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    v2 = torch.tensor([[0.0, 0.0, 1.0, 0.0]])
    zero = torch.zeros(B, D)

    # --- With delta rule: second write overwrites ---
    W_fast = torch.zeros(B, D, D)
    W_fast, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, zero, k, v1, 1.0, 1.0, use_delta_rule=True,
    )
    _, read_after_1 = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, k, zero, zero, 1.0, 0.0, use_delta_rule=True,
    )
    assert torch.allclose(read_after_1, v1, atol=1e-6), \
        f"after first write, expected {v1}, got {read_after_1}"

    W_fast, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, zero, k, v2, 1.0, 1.0, use_delta_rule=True,
    )
    _, read_after_2 = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, k, zero, zero, 1.0, 0.0, use_delta_rule=True,
    )
    assert torch.allclose(read_after_2, v2, atol=1e-6), \
        f"delta rule should overwrite: expected {v2}, got {read_after_2}"

    # --- Without delta rule: second write stacks ---
    W_no = torch.zeros(B, D, D)
    W_no, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_no, zero, k, v1, 1.0, 1.0,
    )
    W_no, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_no, zero, k, v2, 1.0, 1.0,
    )
    _, read_stacked = FastWeightSmall2DTransformer._fast_weight_step(
        W_no, k, zero, zero, 1.0, 0.0,
    )
    assert torch.allclose(read_stacked, v1 + v2, atol=1e-6), \
        f"without delta, writes stack to {v1+v2}, got {read_stacked}"


def test_write_gate_silences_updates():
    """Round 4: gate=0 silences the update; gate=1 produces the full write."""
    B, D = 1, 4
    k = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    v = torch.tensor([[0.0, 1.0, 0.0, 0.0]])
    zero = torch.zeros(B, D)
    W_fast = torch.zeros(B, D, D)

    # gate=0
    new_W, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, zero, k, v, 1.0, 1.0, write_gate=torch.zeros(B),
    )
    assert torch.allclose(new_W, torch.zeros(B, D, D), atol=1e-6), \
        "gate=0 must fully silence the write"

    # gate=1
    new_W, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, zero, k, v, 1.0, 1.0, write_gate=torch.ones(B),
    )
    expected = torch.einsum("bi,bj->bij", v, k)
    assert torch.allclose(new_W, expected, atol=1e-6), \
        "gate=1 must produce full Schlag write"

    # gate=0.5 scales linearly
    new_W, _ = FastWeightSmall2DTransformer._fast_weight_step(
        W_fast, zero, k, v, 1.0, 1.0, write_gate=torch.full((B,), 0.5),
    )
    assert torch.allclose(new_W, 0.5 * expected, atol=1e-6), \
        "gate should scale update linearly"


def test_gate_mlp_created_only_when_configured():
    """Round 4: gate_mlp module is added ONLY when use_write_gate=True.
    Ensures state_dict compat with parent for the disabled case."""
    cfg_off = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_write_gate=False,
    )
    cfg_on = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8,
        max_len=10, use_write_gate=True, gate_hidden=16,
    )
    m_off = FastWeightSmall2DTransformer(cfg_off)
    m_on = FastWeightSmall2DTransformer(cfg_on)
    assert not hasattr(m_off, "gate_mlp"), \
        "gate_mlp must not exist when use_write_gate=False"
    assert hasattr(m_on, "gate_mlp"), \
        "gate_mlp must exist when use_write_gate=True"
    assert len(m_on.gate_mlp) == cfg_on.n_layers, \
        "one gate MLP per layer"


def test_round4_forward_runs_and_differs():
    """Round 4: model with delta+gate forward runs without error and
    produces output different from plain fast weights."""
    torch.manual_seed(0)
    cfg_plain = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        lambda_decay=0.9, eta_write=0.5, use_fast_weights=True,
    )
    cfg_r4 = FastWeightConfig(
        vocab_size=16, d_model=8, n_heads=4, n_layers=2, d_ffn=8, max_len=10,
        lambda_decay=0.9, eta_write=0.5, use_fast_weights=True,
        use_delta_rule=True, use_write_gate=True, gate_hidden=8,
    )
    m_plain = FastWeightSmall2DTransformer(cfg_plain)
    m_r4 = FastWeightSmall2DTransformer(cfg_r4)
    # Load shared base params so only delta+gate differ the output.
    m_r4.load_state_dict(m_plain.state_dict(), strict=False)

    x = torch.randint(0, 16, (2, 7))
    with torch.no_grad():
        out_plain = m_plain(x)
        out_r4 = m_r4(x)
    assert not torch.allclose(out_plain, out_r4, atol=1e-5), \
        "Round 4 forward should differ from plain fast weights"


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
    test_delta_rule_overwrites_same_key()
    print("[ok] Round 4 delta rule overwrites same-key bindings")
    test_write_gate_silences_updates()
    print("[ok] Round 4 gate silences / scales writes")
    test_gate_mlp_created_only_when_configured()
    print("[ok] Round 4 gate MLPs only exist when use_write_gate=True")
    test_round4_forward_runs_and_differs()
    print("[ok] Round 4 forward runs and output differs from plain FW")
