"""Round-6 E2E demo — dispatched_v2 in unified substrate + tied embedding.

Combines the three improvements from rounds 4-5:

  A (gating via opcode dispatch) — one compiled card handles 5 ops
    (GCD, FACTORIAL, IS_PRIME, ADD, MUL). Inactive ops contribute
    exactly zero via the `is_op_k · step_k(...)` pattern.
  B (CALM backends as compiled cards) — 5 backend functions live
    inside ONE substrate-compliant compiled tensor, no training.
  D (tied embedding) — slot range outside the card's vocab is tied
    to `tok.weight` bytes, Gemma-style. Compiled-card head entries
    stay independent; only the "Gemma-would-live-here" slots are tied.

Demo plan:
  1. Build dispatched_v2 card (5 ops).
  2. Build substrate sized to host card + 500 tied "Gemma-style" slots.
  3. Install card at (0, 0, 0, 0).
  4. Populate tied slots' tok embedding with random Gemma-stand-in values.
  5. Tie head to tok for the tied slot range.
  6. Save substrate.state_dict() to .pt.
  7. Reload into fresh substrate with same config.
  8. Exhaustive 5-op test (791 cases).
  9. Tied-head numeric check: head(residual)[tied_range] == residual @ tok[tied_range].T.

Pass = exhaustive PASS + tied-numeric-check bit-exact.
"""

from __future__ import annotations

import itertools
import math
import tempfile
import time
from pathlib import Path

import torch

from calm.llm_computer.card_installer import CardSlot, install_compiled_card
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.programs.dispatched_v2 import (
    ADD_MAX, D_MODEL as CARD_D_MODEL, FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND,
    PRIME_MAX_N, PRIME_MIN_N, VOCAB as CARD_VOCAB,
    build_dispatched_v2, decode_output,
)
from calm.llm_computer.programs.is_prime import _is_prime
from calm.llm_computer.tied_embedding import tie_head_to_tok, tied_logits, verify_tied


TIED_SLOT_COUNT = 500  # "Gemma" placeholder slots above the card's vocab


def build_substrate(card) -> GroupedSmall2DTransformer:
    """Substrate hosting `card` plus `TIED_SLOT_COUNT` extra vocab rows
    that will be tied head ↔ tok (Gemma stand-in)."""
    c = card.config
    n_heads = c.n_heads + 20  # headroom
    d_model = 2 * n_heads
    cfg = GroupedSmall2DConfig(
        vocab_size=c.vocab_size + TIED_SLOT_COUNT,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=c.n_layers,
        d_ffn=c.d_ffn + 50,
        max_len=c.max_len,
        use_hard_max=False,
        layer_modes=tuple(["single"] * c.n_layers),
        layer_hard_max=tuple([True] * c.n_layers),
    )
    sub = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.zero_()
    return sub


def main() -> None:
    t0 = time.time()
    print("[demo] building dispatched_v2 card...")
    card = build_dispatched_v2()
    print(f"  card d_model={card.config.d_model} n_heads={card.config.n_heads} "
          f"n_layers={card.config.n_layers} vocab={card.config.vocab_size} "
          f"params={card.param_count():,}")

    print("[demo] building substrate with tied-region headroom...")
    substrate = build_substrate(card)
    print(f"  substrate d_model={substrate.config.d_model} "
          f"n_heads={substrate.config.n_heads} "
          f"d_ffn={substrate.config.d_ffn} vocab={substrate.config.vocab_size}"
          f" params={substrate.param_count():,}")

    print("[demo] installing card @ slot (0, 0, 0, 0)...")
    install_compiled_card(substrate, card, CardSlot(0, 0, 0, 0))

    # Simulate "Gemma would populate tied slots' tok embedding"
    TIED_LO = card.config.vocab_size
    TIED_HI = substrate.config.vocab_size
    print(f"[demo] populating tied tok region rows [{TIED_LO}, {TIED_HI}) "
          f"with random Gemma-stand-in values...")
    with torch.no_grad():
        substrate.tok.weight[TIED_LO:TIED_HI].normal_(0, 0.02)

    print("[demo] tying head to tok across the tied region...")
    tie_head_to_tok(substrate, tok_range=(TIED_LO, TIED_HI))
    assert verify_tied(substrate, tok_range=(TIED_LO, TIED_HI))

    # Save + reload.
    tmp = Path(tempfile.mkdtemp()) / "unified_v2.pt"
    print(f"[demo] saving substrate to {tmp}...")
    torch.save({
        "state_dict": substrate.state_dict(),
        "config": substrate.config.__dict__,
    }, tmp)
    print(f"  file size: {tmp.stat().st_size / 1e6:.1f} MB")

    print("[demo] reloading into fresh substrate...")
    reloaded = build_substrate(card)
    ckpt = torch.load(tmp, weights_only=True)
    reloaded.load_state_dict(ckpt["state_dict"])

    # --- Exhaustive 5-op test through reloaded substrate ---
    print("\n[demo] EXHAUSTIVE 5-op test (reloaded substrate):")

    def _run(inputs, expected, label):
        x = torch.tensor(inputs, dtype=torch.long)
        with torch.no_grad():
            logits = reloaded(x)
        # Restrict argmax to card's vocab range so tied Gemma-stand-in
        # logits don't contaminate. In the unified architecture Gemma's
        # head routes to Gemma's vocab slots; the card's slots are
        # distinct and argmax-within-range is how the caller picks which
        # card's output to consume.
        card_logits = logits[:, 2, 0:CARD_VOCAB]
        preds = card_logits.argmax(dim=-1).tolist()
        correct = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {correct}/{len(inputs)}")
        return correct, len(inputs)

    gcd_inputs = [(a, b, 0) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    gcd_expected = [math.gcd(a, b) for (a, b, _) in gcd_inputs]
    fact_inputs = [(n, 0, 1) for n in range(FACT_MAX_N + 1)]
    fact_expected = [math.factorial(n) for (n, _, _) in fact_inputs]
    prime_inputs = [(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]
    prime_expected = [_is_prime(n) for (n, _, _) in prime_inputs]
    add_inputs = [(a, b, 3) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    add_expected = [a + b for (a, b, _) in add_inputs]
    mul_inputs = [(a, b, 4) for a, b in itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2)]
    mul_expected = [a * b for (a, b, _) in mul_inputs]

    ok, tot = 0, 0
    for inputs, expected, label in [
        (gcd_inputs, gcd_expected, "GCD      "),
        (fact_inputs, fact_expected, "FACTORIAL"),
        (prime_inputs, prime_expected, "IS_PRIME "),
        (add_inputs, add_expected, "ADD      "),
        (mul_inputs, mul_expected, "MUL      "),
    ]:
        c, n = _run(inputs, expected, label)
        ok += c
        tot += n

    ok_dispatch = ok == tot
    print(f"  total: {ok}/{tot} — {'PASS' if ok_dispatch else 'FAIL'}")

    # --- Tied-region numeric check ---
    print("\n[demo] TIED-region numeric check:")
    # Run forward to get a residual, then compare head(residual)[tied_range]
    # with `residual @ tok[tied_range].T`.
    x = torch.tensor([[0, 0, 3]], dtype=torch.long)  # add 0+0 — any input works
    with torch.no_grad():
        # Forward until just before head — replicate model.forward()'s
        # last-residual-before-head state.
        import torch.nn.functional as F
        B, S = x.shape
        cfg = reloaded.config
        pos_idx = torch.arange(S)
        res = reloaded.tok(x) + reloaded.pos(pos_idx)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        from calm.llm_computer.grouped_attention import (
            grouped_attention_single_head_mode,
        )
        for layer in range(cfg.n_layers):
            qkv = reloaded.W_qkv[layer](res)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
            attn = grouped_attention_single_head_mode(
                qh, kh, vh, mask=mask, scale=1.0,
                hard_max=reloaded._grouped_config.layer_hard_max[layer],
            )
            attn = attn.reshape(B, S, cfg.d_model)
            res = res + reloaded.W_out[layer](attn)
            gate, val = reloaded.ff_in[layer](res).chunk(2, dim=-1)
            res = res + reloaded.ff_out[layer](F.relu(gate) * val)
        head_logits = reloaded.head(res)[:, :, TIED_LO:TIED_HI]
        ref_logits = tied_logits(reloaded, res, tok_range=(TIED_LO, TIED_HI))
    diff = (head_logits - ref_logits).abs().max().item()
    ok_tied = diff < 1e-5
    print(f"  max |head(x)[tied] - residual @ tok[tied].T| = {diff:.2e} — "
          f"{'PASS' if ok_tied else 'FAIL'}")

    # --- Summary ---
    all_ok = ok_dispatch and ok_tied
    t = time.time() - t0
    print(f"\n[demo] OVERALL: {'PASS' if all_ok else 'FAIL'}  (total {t:.1f}s)")
    print("[demo] A (dispatched gating), B (5 CALM ops as cards), "
          "D (tied embedding):")
    print(f"[demo]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    tmp.unlink()


if __name__ == "__main__":
    main()
