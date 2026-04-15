"""Round-21 inter-slot composition — two independently-compiled cards
share a residual channel in one forward pass.

This is the first experiment in genuine compile-time modularity: Card A
and Card B are built by separate functions, each declaring its interface
(channel ownership) via named constants. They meet only at MERGE time,
when both cards' zero-initialized weights are summed into one
`Small2DTransformer`. Because Card A writes channel 9 and Card B reads
channel 9 — and they run on different layers — the composition works
without either card "knowing" about the other.

Shared contract (a minimal interface definition):
    SHARED_CHANNEL_SUM = 9

Card A (layer 0):
    Inputs:  token[0] = a ∈ [0, 7], token[1] = b ∈ [0, 7]
    Outputs: residual[pos 1, channel 9] = a + b

Card B (layer 1):
    Inputs:  residual[:, channel 9] from whoever populated it
    Outputs: head slot 1 if input channel >= threshold, else slot 0

Merging is element-wise weight addition; card A populates only layer 0
slots, card B populates only layer 1 slots, so there's no conflict.

This is NOT the same as `dispatched_v4`:
  * dispatched_v4 = one compiled program authored as a unit, even if
    internally modular (opcode gating).
  * this demo    = TWO compiled programs, built by separate builders,
    composed at link time via a named channel.

If A is replaced later with a different "sum producer" (e.g., a+b+c),
B works unchanged so long as it writes channel 9 with an integer. That's
the modularity thesis.
"""

from __future__ import annotations

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# Shared layout — the interface both cards agree on.
D_MODEL = 16                        # substrate d_model (card-local, merged)
MAX_OPERAND = 7
MAX_SUM = MAX_OPERAND + MAX_OPERAND  # 14
THRESHOLD = 5

# Residual channel layout (the "link contract"):
CH_OWN = 0       # token scalar (standard)
CH_BIAS = 1      # 1 at every position (standard)
CH_COPY_A = 2    # Card A writes: copy-from-pos-0 of token scalar
SHARED_CHANNEL_SUM = 9   # Card A writes a+b here at pos 1. Card B reads.
CH_IS_ABOVE = 10  # Card B writes: step_THRESHOLD(SUM) indicator

VOCAB_SIZE = 16  # covers operand tokens [0, 7] + slot ids


# --- Card A: sum writer ---

def build_card_a(max_len: int = 2) -> Small2DTransformer:
    """Compute a + b at position 1, write to SHARED_CHANNEL_SUM.

    Doesn't know Card B exists. Publishes channel 9 as its output.
    """
    graph = GateGraph(vocab_size=VOCAB_SIZE)
    # token scalar
    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, CH_OWN, float(k)) for k in range(VOCAB_SIZE)],
    ))
    # bias
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, CH_BIAS, 1.0) for p in range(max_len)],
    ))
    # copy_a: LookUp pos 0 -> CH_COPY_A at every query
    graph.add(LookUp(
        name="copy_a", layer=0,
        v_source_channels=[CH_OWN], out_channels=[CH_COPY_A],
    ))
    # sum write: ReLU(1) · (copy_a + own) → CH_SUM  [at pos 1: copy_a=a, own=b → a+b]
    graph.add(ReGLU(
        name="sum_write", layer=0,
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_A, 1.0), (CH_OWN, 1.0)],
        output_channel=SHARED_CHANNEL_SUM,
        output_coef=1.0,
    ))
    # Card A does NOT emit a head — its output is residual channel 9.
    # We install a zero head for compile compatibility.
    graph.add(LinearHead(name="zero_head", entries=[]))

    return compile_program(
        graph,
        d_model=D_MODEL,
        n_heads=D_MODEL // 2,
        n_layers=2,
        d_ffn=4,
        max_len=max_len,
        vocab_size=VOCAB_SIZE,
    )


# --- Card B: threshold reader ---

def build_card_b(max_len: int = 2) -> Small2DTransformer:
    """Read SHARED_CHANNEL_SUM at layer 1, output indicator >= THRESHOLD
    at head slot 1 (vs slot 0 for below).

    Doesn't know Card A exists. Consumes channel 9 as its input.
    """
    graph = GateGraph(vocab_size=VOCAB_SIZE)
    # Card B needs its own tok embedding of zero (doesn't read any token
    # scalar). Leave empty to compile.
    graph.add(TokenEmbed(name="b_tok", entries=[]))
    # Needs a bias channel. If merged with Card A, A's bias write is enough.
    # But for standalone testing, Card B writes its own bias.
    graph.add(PosEmbed(
        name="b_bias",
        entries=[(p, CH_BIAS, 1.0) for p in range(max_len)],
    ))
    # step_THRESHOLD(SUM) = ReLU(SUM - (THRESHOLD-1)) - ReLU(SUM - THRESHOLD)
    # on layer 1. Output → CH_IS_ABOVE.
    graph.add(ReGLU(
        name="step_hi", layer=1,
        gate=[(SHARED_CHANNEL_SUM, 1.0), (CH_BIAS, -(THRESHOLD - 1))],
        val=[(CH_BIAS, 1.0)],
        output_channel=CH_IS_ABOVE,
        output_coef=1.0,
    ))
    graph.add(ReGLU(
        name="step_lo", layer=1,
        gate=[(SHARED_CHANNEL_SUM, 1.0), (CH_BIAS, -THRESHOLD)],
        val=[(CH_BIAS, 1.0)],
        output_channel=CH_IS_ABOVE,
        output_coef=-1.0,
    ))
    # Head: slot 0 = "below" (constant +1 via bias), slot 1 = "above"
    # (CH_IS_ABOVE × +2).
    # Argmax: slot 0 wins (logit 1) if below, slot 1 wins (logit 2) if above.
    graph.add(LinearHead(
        name="threshold_head",
        entries=[
            (0, CH_BIAS, 1.0),         # slot 0 = 1 always
            (1, CH_IS_ABOVE, 2.0),     # slot 1 = 2 if above, 0 if below
        ],
    ))

    return compile_program(
        graph,
        d_model=D_MODEL,
        n_heads=D_MODEL // 2,
        n_layers=2,
        d_ffn=4,
        max_len=max_len,
        vocab_size=VOCAB_SIZE,
    )


# --- Merge: weight addition across the two cards' state_dicts ---

def merge_cards(*cards: Small2DTransformer) -> Small2DTransformer:
    """Sum two or more cards into one Small2DTransformer with the same
    shape. Each card is expected to have ZERO-INITIALIZED weights except
    in its reserved (layer, channel, neuron) rectangle — this is how
    `compile_program` constructs cards. Simple summation then combines
    them without conflict.

    Checks: all cards share config dims. No merge-time conflict detection
    (relies on disjoint weight populations); this is the minimum viable
    demo.
    """
    if not cards:
        raise ValueError("need at least one card")
    base_cfg = cards[0].config
    for c in cards[1:]:
        assert c.config.d_model == base_cfg.d_model, (
            f"d_model mismatch: {c.config.d_model} vs {base_cfg.d_model}"
        )
        assert c.config.n_heads == base_cfg.n_heads
        assert c.config.n_layers == base_cfg.n_layers
        assert c.config.d_ffn == base_cfg.d_ffn
        assert c.config.vocab_size == base_cfg.vocab_size

    merged = Small2DTransformer(Small2DConfig(
        vocab_size=base_cfg.vocab_size,
        d_model=base_cfg.d_model,
        n_heads=base_cfg.n_heads,
        n_layers=base_cfg.n_layers,
        d_ffn=base_cfg.d_ffn,
        max_len=base_cfg.max_len,
        use_hard_max=base_cfg.use_hard_max,
    ))
    with torch.no_grad():
        for p in merged.parameters():
            p.zero_()
        for c in cards:
            for (name_m, p_m), (_, p_c) in zip(
                merged.named_parameters(), c.named_parameters()
            ):
                p_m.add_(p_c)
    return merged


if __name__ == "__main__":
    import itertools

    print("[composed] building Card A (sum writer)...")
    card_a = build_card_a()
    print(f"  params: {card_a.param_count()}")

    print("[composed] building Card B (threshold reader)...")
    card_b = build_card_b()
    print(f"  params: {card_b.param_count()}")

    print("[composed] merging cards (weight addition)...")
    merged = merge_cards(card_a, card_b)
    print(f"  merged params: {merged.param_count()}")

    # Exhaustive: for each (a, b) ∈ [0, 7]², check head predicts
    # slot 1 iff a + b >= THRESHOLD.
    ok = 0
    total = 0
    for a, b in itertools.product(range(MAX_OPERAND + 1), repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            logits = merged(x)
        pred = int(logits[0, 1].argmax().item())
        expected = 1 if a + b >= THRESHOLD else 0
        if pred == expected:
            ok += 1
        total += 1
        if pred != expected:
            print(f"  [✗] a={a}, b={b}: sum={a+b}, "
                  f"pred slot {pred}, expected {expected}")
    print(f"\n[composed] merged substrate: {ok}/{total} "
          f"= {100 * ok / total:.0f}% — "
          f"{'PASS' if ok == total else 'FAIL'}")

    # Also verify: Card A alone doesn't produce meaningful head output
    # (it has a zero head), and Card B alone doesn't compute a sum
    # (it has no ReGLU that writes channel 9). Shows neither is
    # self-sufficient — composition is doing the work.
    x_probe = torch.tensor([[3, 4]], dtype=torch.long)
    with torch.no_grad():
        la = card_a(x_probe)
        lb = card_b(x_probe)
        lmerged = merged(x_probe)
    print(f"\n[composed] sanity — card A alone logits (should be ~zero): "
          f"max |l| = {la.abs().max().item():.3f}")
    print(f"[composed] sanity — card B alone logits for (3,4) → "
          f"slot0={lb[0, 1, 0].item():.3f}, slot1={lb[0, 1, 1].item():.3f}")
    print(f"           (without channel 9 populated, B emits slot 0 = 'below')")
    print(f"[composed] sanity — merged logits for (3,4) → "
          f"slot0={lmerged[0, 1, 0].item():.3f}, slot1={lmerged[0, 1, 1].item():.3f}")
    print(f"           (3+4=7 >= 5, so slot 1 should win in merged)")
