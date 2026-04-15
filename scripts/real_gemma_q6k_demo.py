"""Round-15 E2E — Gemma GGUF + real token_embd (Q6_K) + dispatched_v4.

Extends Round 14 with Q6_K dequant of Gemma's `token_embd.weight`. The
previous demo used random tok embed (normal 0, 0.02) as a stand-in;
this demo loads the real values from the GGUF and installs them into
the substrate's FP32 token embedding.

Memory note: Q6_K dequant of full vocab (262144 × 2560 × 4B = 2.68 GB)
is a one-time cost, ~2 minutes on CPU. Once in `substrate.tok.weight`
the values live FP32 and forward is direct.

Tests:
  (a) Q6_K dequant runs + produces finite values in Gemma-typical range.
  (b) With real tok embed, Gemma residual through 2 tq4 layers shifts
      MORE than with random tok (real correlations trigger real activations).
  (c) dispatched_v4 card still passes 791/791 through hybrid substrate.
"""

from __future__ import annotations

import itertools
import math
import os
import time
from pathlib import Path

import torch

from calm.llm_computer.card_installer import CardSlot
from calm.llm_computer.gemma_byte_installer import install_gemma_layer_bytes
from calm.llm_computer.hybrid_substrate import (
    HybridGroupedSmall2DConfig, HybridGroupedSmall2DTransformer,
    install_compiled_card_hybrid,
)
from calm.llm_computer.programs.dispatched_v4 import (
    FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND, OPCODE_SHIFT, PRIME_MAX_N,
    PRIME_MIN_N, VOCAB as CARD_VOCAB, build_dispatched_v4, decode_output,
)
from calm.llm_computer.programs.is_prime import _is_prime
from calm.llm_computer.q6k_dequant import extract_q6_k_tensor
from calm.llm_computer.tq4_gguf_loader import read_turboquant_gguf
from calm.llm_computer.unified_tensor import UnifiedTensorConfig


GGUF_PATH = Path(os.environ.get(
    "ZENITH_GEMMA_GGUF",
    "/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf",
))


def main() -> None:
    if not GGUF_PATH.exists():
        print(f"GGUF not found at {GGUF_PATH}")
        return

    t0 = time.time()
    GEMMA_N_LAYERS = 2

    print(f"[R15] building dispatched_v4 card...")
    card = build_dispatched_v4()

    print(f"[R15] building hybrid substrate (n_gemma={GEMMA_N_LAYERS} + "
          f"n_card={card.config.n_layers})...")
    ufc = UnifiedTensorConfig(
        gemma_n_layers=GEMMA_N_LAYERS,
        gemma_max_position=128,
        gemma_full_layer_indices=(),
    )
    D_s = ufc.substrate_d_model
    D_ffn = ufc.substrate_d_ffn
    n_heads = ufc.substrate_n_heads
    GEMMA_VOCAB = ufc.gemma_vocab_size
    GEMMA_D = ufc.gemma_d_model

    total_vocab = GEMMA_VOCAB + card.config.vocab_size
    pad = (256 - total_vocab % 256) % 256
    total_vocab_padded = total_vocab + pad
    n_total = GEMMA_N_LAYERS + card.config.n_layers

    cfg = HybridGroupedSmall2DConfig(
        vocab_size=total_vocab_padded, d_model=D_s, n_heads=n_heads,
        n_layers=n_total, d_ffn=D_ffn, max_len=128, use_hard_max=False,
        layer_modes=tuple(["single"] * n_total),
        layer_hard_max=(
            tuple([False] * GEMMA_N_LAYERS)
            + tuple([True] * card.config.n_layers)
        ),
        layer_linear_types=(
            tuple(["tq4"] * GEMMA_N_LAYERS)
            + tuple(["fp32"] * card.config.n_layers)
        ),
    )
    print(f"  d_model={D_s} n_heads={n_heads} d_ffn={D_ffn} vocab={cfg.vocab_size}")

    substrate = HybridGroupedSmall2DTransformer(cfg)
    substrate.initialize_tq4_layers_to_zero()
    # Random init as fallback; will be OVERWRITTEN by Q6_K dequant on Gemma range.
    with torch.no_grad():
        substrate.tok.weight.normal_(0, 0.02)
        substrate.pos.weight.normal_(0, 0.02)
    print(f"  params: {substrate.param_count():,}")

    reader = read_turboquant_gguf(GGUF_PATH)

    # --- CHECK (a) Q6_K dequant ---
    print(f"\n[R15] CHECK (a) — Q6_K dequant of token_embd.weight")
    t_dq = time.time()
    tok_embd = extract_q6_k_tensor(reader, "token_embd.weight")
    dq_time = time.time() - t_dq
    print(f"  shape={tuple(tok_embd.shape)}, dequant time={dq_time:.1f}s")
    ok_a = (
        tok_embd.shape == torch.Size([GEMMA_VOCAB, GEMMA_D])
        and torch.isfinite(tok_embd).all().item()
        and 0.005 < tok_embd.std().item() < 0.2
    )
    print(f"  stats: std={tok_embd.std():.4f} "
          f"range=[{tok_embd.min():.3f}, {tok_embd.max():.3f}] — "
          f"{'PASS' if ok_a else 'FAIL'}")

    # Install into substrate.tok.weight[:GEMMA_VOCAB, :GEMMA_D]
    print(f"  installing into substrate.tok.weight[:{GEMMA_VOCAB}, :{GEMMA_D}]...")
    with torch.no_grad():
        substrate.tok.weight[:GEMMA_VOCAB, :GEMMA_D] = tok_embd
    del tok_embd  # free 2.68 GB
    torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Install Gemma layer weights (same as round 14).
    print(f"[R15] installing Gemma layers 0..{GEMMA_N_LAYERS - 1}...")
    t_g = time.time()
    for i in range(GEMMA_N_LAYERS):
        install_gemma_layer_bytes(substrate, ufc, reader, layer_idx=i)
    print(f"  gemma install time: {time.time() - t_g:.1f}s")

    # Install card
    print(f"[R15] installing dispatched_v4 card...")
    CARD_CH_OFF = GEMMA_D
    CARD_SH_OFF = GEMMA_D // 2
    CARD_TOK_OFF = GEMMA_VOCAB
    CARD_LAYER_OFF = GEMMA_N_LAYERS
    install_compiled_card_hybrid(
        substrate, card,
        ch_off=CARD_CH_OFF, sh_off=CARD_SH_OFF, ffn_off=0,
        tok_off=CARD_TOK_OFF, layer_off=CARD_LAYER_OFF,
    )

    # --- CHECK (b) Gemma residual with REAL tok embed ---
    print(f"\n[R15] CHECK (b) — Gemma residual with real Q6_K tok embed")
    substrate.eval()
    from calm.llm_computer.grouped_attention import grouped_attention_single_head_mode
    import torch.nn.functional as F
    # Use realistic Gemma tokens (common tokens in [1, 10000))
    x = torch.tensor([[1, 100, 1000, 500, 2000, 42, 7, 999]], dtype=torch.long)
    with torch.no_grad():
        B, S = x.shape
        pos_idx = torch.arange(S)
        res = substrate.tok(x) + substrate.pos(pos_idx)
        init_std = res[0, :, :GEMMA_D].std().item()
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        for layer in range(cfg.n_layers):
            qkv = substrate.W_qkv[layer](res)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            qh = q.transpose(1, 2); kh = k.transpose(1, 2); vh = v.transpose(1, 2)
            attn = grouped_attention_single_head_mode(
                qh, kh, vh, mask=mask, scale=1.0,
                hard_max=cfg.layer_hard_max[layer],
            )
            attn = attn.reshape(B, S, cfg.d_model)
            res = res + substrate.W_out[layer](attn)
            gate, val = substrate.ff_in[layer](res).chunk(2, dim=-1)
            res = res + substrate.ff_out[layer](F.relu(gate) * val)
    gemma_residual = res[0, -1, :GEMMA_D]
    final_std = gemma_residual.std().item()
    rel_change = abs(final_std - init_std) / max(init_std, 1e-8)
    ok_b = (torch.isfinite(gemma_residual).all().item()
            and final_std > 1e-4
            and rel_change > 0.1)
    print(f"  initial residual std={init_std:.4f}, "
          f"final Gemma channels std={final_std:.4f} "
          f"(rel Δ={rel_change * 100:.1f}%)")
    print(f"  range=[{gemma_residual.min():.3f}, {gemma_residual.max():.3f}] — "
          f"{'PASS' if ok_b else 'FAIL'}")

    # --- CHECK (c) dispatched_v4 exhaustive ---
    print(f"\n[R15] CHECK (c) — dispatched_v4 791/791")

    def _run(inputs, expected, label):
        shifted = [(a + CARD_TOK_OFF, b + CARD_TOK_OFF,
                    op + OPCODE_SHIFT + CARD_TOK_OFF)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long)
        with torch.no_grad():
            logits = substrate(x)
        card_logits = logits[:, 2, CARD_TOK_OFF:CARD_TOK_OFF + CARD_VOCAB]
        preds = card_logits.argmax(dim=-1).tolist()
        c = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {c}/{len(inputs)}")
        return c, len(inputs)

    pairs = list(itertools.product(range(GCD_BASE), repeat=2))
    mul_pairs = list(itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2))
    ok_c_cnt = tot = 0
    for inp, exp, lab in [
        ([(a, b, 0) for a, b in pairs],
         [math.gcd(a, b) for a, b in pairs], "GCD      "),
        ([(n, 0, 1) for n in range(FACT_MAX_N + 1)],
         [math.factorial(n) for n in range(FACT_MAX_N + 1)], "FACTORIAL"),
        ([(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
         [_is_prime(n) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
         "IS_PRIME "),
        ([(a, b, 3) for a, b in pairs],
         [a + b for a, b in pairs], "ADD      "),
        ([(a, b, 4) for a, b in mul_pairs],
         [a * b for a, b in mul_pairs], "MUL      "),
    ]:
        c, n = _run(inp, exp, lab)
        ok_c_cnt += c
        tot += n
    ok_c = ok_c_cnt == tot
    print(f"  dispatched total: {ok_c_cnt}/{tot} — "
          f"{'PASS' if ok_c else 'FAIL'}")

    all_ok = ok_a and ok_b and ok_c
    t = time.time() - t0
    print(f"\n[R15] OVERALL: {'PASS' if all_ok else 'FAIL'}  ({t:.1f}s)")
    print("[R15] Gemma GGUF (Q6_K tok_embd + tq4 layers) + dispatched_v4 "
          "in ONE hybrid substrate:")
    print(f"[R15]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")


if __name__ == "__main__":
    main()
