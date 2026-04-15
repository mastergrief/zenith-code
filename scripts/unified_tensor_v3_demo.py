"""Round-7 E2E demo — Gemma stand-in + dispatched_v2 + tied head.

Validates the full architectural gap at reduced scale:

  * Gemma-style layers (softmax attention) occupy substrate layers 0..N_G.
  * Compiled card (dispatched_v2) occupies layers N_G..N_G+N_C with
    single-mode + hard_max attention. A compiled card CANNOT share a layer
    with softmax attention — its LookUp logic depends on argmax. The two
    attention modes coexist per-layer via `layer_modes` + `layer_hard_max`.
  * Tied head: `head.weight[gemma_vocab_range] = tok.weight[gemma_vocab_range]`,
    Gemma-style. Compiled card's head entries stay independent (disjoint
    vocab rectangles).

Why not real GGUF bytes here: Gemma 4 E4B at FP32 (d_model=4096, vocab=262144,
n_layers=42) needs ~24GB for weights + ~10GB for embeddings. Real-byte
install is the tq4 path (`test_gemma_byte_installer.py`) — proven on 2
layers but samples residual, can't run the head. This demo proves the
mixed-mode architecture at a scale that fits, using Gemma-shaped random
weights as a stand-in. A tq4-aware card installer (future work) would
let us combine the two.

Demo plan:
  1. Build dispatched_v2 card.
  2. Build substrate: N_G Gemma-style (softmax) layers + N_C card layers.
  3. Install card at layer_off = N_G.
  4. Populate Gemma region (channels 0..D_G, vocab 0..V_G) with random
     Gemma-stand-in weights.
  5. Tie head to tok on Gemma vocab range.
  6. Save/reload/forward.
  7. Verify:
     (a) Gemma residual propagates — non-trivial output on Gemma channels
         after layers 0..N_G.
     (b) Dispatched card still passes 791/791 (its layers run isolated in
         hard_max mode, ignoring Gemma's softmax residual on its channels).
     (c) Tied head: head(x)[:, :, gemma_range] == residual @ tok[gemma_range].T.

Pass = all three.
"""

from __future__ import annotations

import itertools
import math
import tempfile
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.card_installer import CardSlot, install_compiled_card
from calm.llm_computer.grouped_attention import grouped_attention_single_head_mode
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


# Gemma stand-in dimensions (scale that fits in memory; Gemma-shaped)
GEMMA_D_MODEL = 256
GEMMA_VOCAB = 1024
GEMMA_N_LAYERS = 2


def build_substrate(card) -> GroupedSmall2DTransformer:
    """Substrate with N_G softmax layers + card's N_C hard_max layers.

    Layout:
      channels:  [0, GEMMA_D_MODEL)            = Gemma residual slice
                 [GEMMA_D_MODEL, +CARD_D_MODEL) = card residual slice
      vocab:     [0, GEMMA_VOCAB)               = Gemma tokens (tied)
                 [GEMMA_VOCAB, +CARD_VOCAB)     = card tokens (independent head)
      sub-heads: [0, GEMMA_D_MODEL/2)           = Gemma sub-heads
                 [GEMMA_D_MODEL/2, +CARD_N_HEADS) = card sub-heads
      layers:    [0, GEMMA_N_LAYERS)            = softmax (Gemma)
                 [GEMMA_N_LAYERS, +card.n_layers) = hard_max (card)
    """
    c = card.config
    # Need d_model ≥ GEMMA_D_MODEL + card.d_model, rounded to 2*n_heads.
    n_heads = GEMMA_D_MODEL // 2 + c.n_heads + 4   # headroom
    d_model = 2 * n_heads
    d_ffn = 512 + c.d_ffn                           # Gemma FFN + card FFN
    vocab = GEMMA_VOCAB + c.vocab_size
    n_layers = GEMMA_N_LAYERS + c.n_layers
    # Per-layer mode: first N_G are softmax (Gemma-style), rest are hard_max.
    layer_modes = tuple(["single"] * n_layers)
    layer_hard_max = tuple(
        [False] * GEMMA_N_LAYERS + [True] * c.n_layers
    )
    cfg = GroupedSmall2DConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=c.max_len,
        use_hard_max=False,
        layer_modes=layer_modes,
        layer_hard_max=layer_hard_max,
    )
    sub = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.zero_()
    return sub


def populate_gemma_standin(substrate: GroupedSmall2DTransformer) -> None:
    """Fill Gemma's residual slice + Gemma vocab range with random
    weights. Stands in for real GGUF byte install while preserving the
    architectural shape (Gemma occupies a channel/sub-head/vocab rectangle
    in the substrate, the rest is zero-init for the card)."""
    D_G = GEMMA_D_MODEL
    V_G = GEMMA_VOCAB
    SH_G = D_G // 2
    s_cfg = substrate.config
    D_s = s_cfg.d_model
    F_s = s_cfg.d_ffn

    with torch.no_grad():
        # tok/pos: Gemma rows, Gemma channels only
        substrate.tok.weight[:V_G, :D_G].normal_(0, 0.02)
        substrate.pos.weight[:, :D_G].normal_(0, 0.02)

        for l in range(GEMMA_N_LAYERS):
            # W_qkv for Gemma: Q/K/V rectangles all at (sub-heads [0, SH_G),
            # channels [0, D_G)). Shape of substrate.W_qkv[l].weight is
            # (3*D_s, D_s) with Q/K/V stacked.
            qkv = substrate.W_qkv[l].weight
            qkv[:2 * SH_G, :D_G].normal_(0, 0.02)             # Q
            qkv[D_s:D_s + 2 * SH_G, :D_G].normal_(0, 0.02)    # K
            qkv[2 * D_s:2 * D_s + 2 * SH_G, :D_G].normal_(0, 0.02)  # V

            # W_out: (D_s, D_s) — rows [0, D_G), cols [0, 2*SH_G)
            substrate.W_out[l].weight[:D_G, :2 * SH_G].normal_(0, 0.02)

            # ff_in: (2*F_s, D_s) — Gemma occupies first 256 gate rows +
            # first 256 val rows, channels [0, D_G)
            F_G = 256
            ff_in = substrate.ff_in[l].weight
            ff_in[:F_G, :D_G].normal_(0, 0.02)                # gate
            ff_in[F_s:F_s + F_G, :D_G].normal_(0, 0.02)       # val

            # ff_out: (D_s, F_s) — rows [0, D_G), cols [0, F_G)
            substrate.ff_out[l].weight[:D_G, :F_G].normal_(0, 0.02)

        # head: Gemma's slots independently — later tied to tok
        substrate.head.weight[:V_G].normal_(0, 0.02)


def main() -> None:
    t0 = time.time()
    print("[demo] building dispatched_v2 card...")
    card = build_dispatched_v2()
    print(f"  card d_model={card.config.d_model} n_heads={card.config.n_heads} "
          f"n_layers={card.config.n_layers} vocab={card.config.vocab_size}")

    print("[demo] building substrate with mixed-mode layers "
          "(softmax 0..{}, hard_max {}..{})".format(
              GEMMA_N_LAYERS,
              GEMMA_N_LAYERS, GEMMA_N_LAYERS + card.config.n_layers,
          ))
    substrate = build_substrate(card)
    print(f"  substrate d_model={substrate.config.d_model} "
          f"n_heads={substrate.config.n_heads} d_ffn={substrate.config.d_ffn} "
          f"n_layers={substrate.config.n_layers} vocab={substrate.config.vocab_size} "
          f"params={substrate.param_count():,}")
    print(f"  layer_modes={substrate._grouped_config.layer_modes}")
    print(f"  layer_hard_max={substrate._grouped_config.layer_hard_max}")

    print("[demo] populating Gemma stand-in weights...")
    populate_gemma_standin(substrate)

    print(f"[demo] installing dispatched_v2 card at "
          f"(ch_off={GEMMA_D_MODEL}, sh_off={GEMMA_D_MODEL // 2}, "
          f"ffn_off=256, tok_off={GEMMA_VOCAB}, layer_off={GEMMA_N_LAYERS})...")
    install_compiled_card(substrate, card, CardSlot(
        ch_off=GEMMA_D_MODEL,
        sh_off=GEMMA_D_MODEL // 2,
        ffn_off=256,
        tok_off=GEMMA_VOCAB,
        layer_off=GEMMA_N_LAYERS,
    ))

    print("[demo] tying head to tok on Gemma vocab range...")
    tie_head_to_tok(substrate, tok_range=(0, GEMMA_VOCAB))
    assert verify_tied(substrate, tok_range=(0, GEMMA_VOCAB))

    # Save + reload.
    tmp = Path(tempfile.mkdtemp()) / "unified_v3.pt"
    print(f"[demo] saving to {tmp}...")
    torch.save({
        "state_dict": substrate.state_dict(),
        "config": substrate.config.__dict__,
    }, tmp)
    print(f"  file size: {tmp.stat().st_size / 1e6:.1f} MB")

    print("[demo] reloading...")
    reloaded = build_substrate(card)
    ckpt = torch.load(tmp, weights_only=True)
    reloaded.load_state_dict(ckpt["state_dict"])

    # --- (a) Gemma residual propagates ---
    print("\n[demo] CHECK (a) — Gemma residual non-trivial through soft layers")
    x_gemma = torch.randint(0, GEMMA_VOCAB, (1, 3))
    with torch.no_grad():
        logits_full = reloaded(x_gemma)
    gemma_logits = logits_full[0, -1, :GEMMA_VOCAB]
    ok_gemma = gemma_logits.std().item() > 1e-3
    print(f"  Gemma logit range: std={gemma_logits.std():.4f}, "
          f"max={gemma_logits.max():.4f}, min={gemma_logits.min():.4f} — "
          f"{'PASS' if ok_gemma else 'FAIL'}")

    # --- (b) Dispatched card still passes ---
    print("[demo] CHECK (b) — dispatched_v2 exhaustive test (5 ops)")
    # Card tokens in substrate vocab range [GEMMA_VOCAB, GEMMA_VOCAB + CARD_VOCAB)
    def _run(inputs, expected, label):
        shifted = [(a + GEMMA_VOCAB, b + GEMMA_VOCAB, op + GEMMA_VOCAB)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long)
        with torch.no_grad():
            logits = reloaded(x)
        # Restrict argmax to card vocab range
        card_logits = logits[:, 2, GEMMA_VOCAB:GEMMA_VOCAB + CARD_VOCAB]
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

    ok_card, tot = 0, 0
    for inputs, expected, label in [
        (gcd_inputs, gcd_expected, "GCD      "),
        (fact_inputs, fact_expected, "FACTORIAL"),
        (prime_inputs, prime_expected, "IS_PRIME "),
        (add_inputs, add_expected, "ADD      "),
        (mul_inputs, mul_expected, "MUL      "),
    ]:
        c, n = _run(inputs, expected, label)
        ok_card += c
        tot += n
    ok_dispatched = ok_card == tot
    print(f"  dispatched total: {ok_card}/{tot} — "
          f"{'PASS' if ok_dispatched else 'FAIL'}")

    # --- (c) Tied head numerical match ---
    print("[demo] CHECK (c) — tied head numerical match on Gemma vocab range")
    x = torch.randint(0, GEMMA_VOCAB, (1, 3))
    with torch.no_grad():
        B, S = x.shape
        cfg = reloaded.config
        pos_idx = torch.arange(S)
        res = reloaded.tok(x) + reloaded.pos(pos_idx)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
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
        head_logits = reloaded.head(res)[:, :, :GEMMA_VOCAB]
        ref_logits = tied_logits(reloaded, res, tok_range=(0, GEMMA_VOCAB))
    diff = (head_logits - ref_logits).abs().max().item()
    ok_tied = diff < 1e-4
    print(f"  max |head(x)[gemma] - residual @ tok[gemma].T| = {diff:.2e} — "
          f"{'PASS' if ok_tied else 'FAIL'}")

    all_ok = ok_gemma and ok_dispatched and ok_tied
    t = time.time() - t0
    print(f"\n[demo] OVERALL: {'PASS' if all_ok else 'FAIL'}  (total {t:.1f}s)")
    print("[demo] unified substrate: Gemma stand-in + dispatched_v2 card + "
          "tied head:")
    print(f"[demo]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    tmp.unlink()


if __name__ == "__main__":
    main()
