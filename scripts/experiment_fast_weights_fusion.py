"""Round 2 — fast weights + compiled program on one substrate.

Round 1 proved Schlag-style fast weights work at d_head=2 (99.1% on 3-pair
recall). Round 2 asks: does a fast-weights-enabled layer coexist with a
compiled program on the same substrate without destroying compiled
correctness? This is the "heterogeneous MoE with shared residual context
pool" test — fast-weights expert + compiled-program expert, one forward
pass.

Architecture test, NOT a training test. At d_model=10 (adder_tiny's size),
W_fast is only 100 entries — too small to hold useful bindings. The useful
question at this scale: do fast-weights disturb compiled behavior when
enabled alongside, or do they stay dormant when layer 0 is quiet?

Mirrors experiment_fusion_mvp.py structure:
  Step 1: empty layer 0 + compiled adder layer 1, FW disabled → 16/16 sanity.
  Step 2: empty layer 0 + compiled adder layer 1, FW enabled → should stay
          16/16 (layer 0 projections are zero → q,k,v = 0 → W_fast stays 0 →
          fast_read = 0 → no residual perturbation).
  Step 3: noisy layer 0 + compiled adder layer 1, FW disabled → establish
          noise floor without fast weights.
  Step 4: noisy layer 0 + compiled adder layer 1, FW enabled → does the
          fast-weights mechanism amplify the noise (bad) or absorb it (good)?

If steps 1&2 both give 16/16, the architectural coexistence invariant holds:
fast weights stay silent when their projections are silent. If step 4 matches
or beats step 3, fast-weights doesn't catastrophically interfere.
"""

from __future__ import annotations

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.fast_weights import (
    FastWeightConfig, FastWeightSmall2DTransformer,
)
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


def build_adder_tiny_small2d(target_layer: int = 1, n_layers: int = 2):
    """Same construction as experiment_fusion_mvp.build_adder_at_layer.

    Returns a Small2DTransformer with adder_tiny compiled into `target_layer`
    and all other layers zero-initialized (empty).
    """
    V = 8
    MAX_SUM = 6
    g = GateGraph(vocab_size=V)
    g.add(TokenEmbed(name="own_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias", entries=[(p, 1, 1.0) for p in range(4)]))
    g.add(LookUp(name="copy_a", layer=target_layer,
                  v_source_channels=[0], out_channels=[2]))
    for S in range(MAX_SUM + 1):
        g.add(ReGLU(name=f"step_{S}_hi", layer=target_layer,
                     gate=[(0, 1.0), (2, 1.0), (1, -(S - 1))],
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=1.0))
        g.add(ReGLU(name=f"step_{S}_lo", layer=target_layer,
                     gate=[(0, 1.0), (2, 1.0), (1, -S)],
                     val=[(1, 1.0)],
                     output_channel=3 + S, output_coef=-1.0))
    head = []
    for k in range(MAX_SUM + 1):
        head.append((k, 3 + k, 1.0))
        if k + 1 <= MAX_SUM:
            head.append((k, 3 + k + 1, -1.0))
    g.add(LinearHead(name="onehot", entries=head))

    return compile_program(g, d_model=10, n_heads=5, n_layers=n_layers,
                           d_ffn=14, max_len=4, vocab_size=V)


def lift_to_fast_weights(small_model: Small2DTransformer,
                         use_fast_weights: bool,
                         lambda_decay: float = 0.95,
                         eta_write: float = 0.5) -> FastWeightSmall2DTransformer:
    """Copy a compiled Small2DTransformer into a FastWeightSmall2DTransformer
    with matching weights. FastWeightSmall2DTransformer has no extra
    nn.Parameters so state_dict transfer is exact.
    """
    cfg_s = small_model.config
    cfg_f = FastWeightConfig(
        vocab_size=cfg_s.vocab_size, d_model=cfg_s.d_model,
        n_heads=cfg_s.n_heads, n_layers=cfg_s.n_layers, d_ffn=cfg_s.d_ffn,
        max_len=cfg_s.max_len, use_hard_max=cfg_s.use_hard_max,
        lambda_decay=lambda_decay, eta_write=eta_write,
        use_fast_weights=use_fast_weights,
    )
    m = FastWeightSmall2DTransformer(cfg_f)
    m.load_state_dict(small_model.state_dict())
    return m


def exhaustive_adder(model) -> int:
    """Run all 16 (a, b) ∈ [0,4)² and count correct a+b predictions at pos 1."""
    correct = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            correct += int(got == a + b)
    return correct


def add_noise_to_layer0(model, sigma: float, seed: int = 0):
    """In-place additive Gaussian noise on all layer-0 weight matrices."""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in (model.W_qkv[0].weight, model.W_out[0].weight,
                  model.ff_in[0].weight, model.ff_out[0].weight):
            p.add_(torch.randn(p.shape, generator=gen) * sigma)


def step1_disabled_baseline():
    """Compiled adder layer 1, FW disabled. Must be 16/16 (sanity)."""
    print("\n=== step 1: empty layer 0 + compiled adder layer 1, FW DISABLED ===")
    base = build_adder_tiny_small2d(target_layer=1, n_layers=2)
    model = lift_to_fast_weights(base, use_fast_weights=False)
    acc = exhaustive_adder(model)
    print(f"  {acc}/16  {'PASS' if acc == 16 else 'FAIL (sanity broken)'}")
    return acc


def step2_enabled_dormant():
    """Compiled adder layer 1, FW ENABLED but layer 0 projections are zero.
    Should stay 16/16 — zero projections → q,k,v = 0 → W_fast stays 0 →
    fast_read = 0 → residual unchanged."""
    print("\n=== step 2: empty layer 0 + compiled adder layer 1, FW ENABLED ===")
    base = build_adder_tiny_small2d(target_layer=1, n_layers=2)
    model = lift_to_fast_weights(base, use_fast_weights=True,
                                  lambda_decay=0.95, eta_write=0.5)
    acc = exhaustive_adder(model)
    print(f"  {acc}/16  {'PASS' if acc == 16 else 'FAIL (fast-weights broke dormant case)'}")
    return acc


SIGMAS = (1e-3, 1e-2, 1e-1, 3e-1)
NOISE_SEEDS = tuple(range(8))  # average over 8 noise realizations


def _noisy_mean_acc(use_fast_weights: bool, sigma: float) -> tuple[float, int]:
    """Return (mean_acc, n_fail_cases) over NOISE_SEEDS."""
    total, fail = 0, 0
    for seed in NOISE_SEEDS:
        base = build_adder_tiny_small2d(target_layer=1, n_layers=2)
        model = lift_to_fast_weights(base, use_fast_weights=use_fast_weights,
                                      lambda_decay=0.95, eta_write=0.5)
        add_noise_to_layer0(model, sigma, seed=seed)
        acc = exhaustive_adder(model)
        total += acc
        if acc < 16:
            fail += 1
    mean = total / (16.0 * len(NOISE_SEEDS))
    return mean, fail


def step3_noisy_disabled():
    """Noise on layer 0, FW disabled. Establish degradation baseline.
    Averages over 8 noise realizations per sigma for stability."""
    print("\n=== step 3: noisy layer 0 + compiled adder layer 1, FW DISABLED ===")
    results = {}
    for sigma in SIGMAS:
        mean_acc, n_fail = _noisy_mean_acc(False, sigma)
        results[sigma] = (mean_acc, n_fail)
        print(f"  sigma={sigma:.0e}: {mean_acc:.1%} mean "
              f"({n_fail}/{len(NOISE_SEEDS)} seeds show degradation)")
    return results


def step4_noisy_enabled():
    """Noise on layer 0, FW enabled. Compare to step 3."""
    print("\n=== step 4: noisy layer 0 + compiled adder layer 1, FW ENABLED ===")
    results = {}
    for sigma in SIGMAS:
        mean_acc, n_fail = _noisy_mean_acc(True, sigma)
        results[sigma] = (mean_acc, n_fail)
        print(f"  sigma={sigma:.0e}: {mean_acc:.1%} mean "
              f"({n_fail}/{len(NOISE_SEEDS)} seeds show degradation)")
    return results


def main():
    print("=== Round 2: Fast Weights + Compiled Program Fusion ===")
    print("  substrate: d_model=10  n_heads=5  d_head=2  n_layers=2")
    print("  compiled: adder_tiny at layer 1 (16/16 exhaustive)")
    print("  mechanism: fast weights at layer 0 (d_model×d_model = 100 entries)")

    s1 = step1_disabled_baseline()
    s2 = step2_enabled_dormant()
    s3 = step3_noisy_disabled()
    s4 = step4_noisy_enabled()

    print("\n" + "=" * 60)
    print("Round 2 Results")
    print("=" * 60)
    print(f"\n  step 1 (dormant, FW off):  {s1}/16")
    print(f"  step 2 (dormant, FW on):   {s2}/16")
    print("\n  noise robustness (mean over 8 seeds; higher is better):")
    print("                 FW off     FW on      delta")
    print("                 --------   --------   -----")
    for sigma in sorted(s3.keys()):
        off_acc, _ = s3[sigma]
        on_acc, _ = s4[sigma]
        delta_pp = (on_acc - off_acc) * 100
        print(f"  sigma={sigma:.0e}    {off_acc:6.1%}     {on_acc:6.1%}    {delta_pp:+6.1f} pp")

    print("\n" + "=" * 60)
    # Decision: coexistence holds if step 1 and step 2 both give 16/16.
    # Non-catastrophic-interference holds if step 4 acc >= step 3 acc - tol
    # at every sigma.
    coexist = (s1 == 16) and (s2 == 16)
    tol = 0.05  # 5 pp tolerance for noise averaging
    no_amp = all(s4[sigma][0] >= s3[sigma][0] - tol for sigma in s3.keys())
    if coexist and no_amp:
        print("DECISION: PASS")
        print("  (1) Dormant coexistence holds: FW silent when layer 0 is silent.")
        print("  (2) Fast weights do not catastrophically amplify noise.")
        print("  Architecturally compatible with compiled-program fusion.")
    elif coexist and not no_amp:
        print("DECISION: PARTIAL PASS")
        print("  Dormant coexistence holds, but FW amplifies noise at some sigma.")
        print("  Next round: channel-restricted fast weights (write/read only to "
              "non-compiled channels).")
    else:
        print("DECISION: FAIL")
        print("  Even dormant FW disturbs compiled adder — subclass has a bug.")
    print("=" * 60)


if __name__ == "__main__":
    main()
