"""Round 47.2: per-head ablation at L34 — step-2 composition hub candidate.

Mirror of R17 (L23 arithmetic per-head) on L34, the new peak from R47.1
(mean Δ=-3.17, hurts 10/10 for multi-step a*b+c).

Hypothesis: if multi-step's step-2 (composition / finalization) has a
concentrated attention mechanism, 1-2 of L34's 8 Q-heads will carry
most of the mean Δ. Matches R17's pattern for single-step arithmetic
at L23 (H1 + H4 carried ~95% of L23's signal).

If diffuse (no head Δ < -1.0), L34's role is FFN-locked like R30's
factual recall finding — not compilable at attention level. Pivot to
probing L40 (the other 10/10 late peak) or accept that step-2 is an
FFN mechanism.

L34 is an SWA layer — different from L23 (global). SWA layers share
KV with an earlier layer (here L22 per GemmaConfig.kv_source_layer).
The attention_output wrapper still works the same way — we zero the
head's input slice to attn_output, which ablates that head regardless
of where its KV came from.

Cost: 8 heads × 10 triples + 10 baselines = 90 forwards ≈ 3 min.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

TARGET_LAYER = 34
N_HEADS = 8


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class HeadAblatingWrapper:
    def __init__(self, inner, head_idx, d_head):
        self.inner = inner
        self.head_idx = head_idx
        self.d_head = d_head
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        lo = self.head_idx * self.d_head
        hi = (self.head_idx + 1) * self.d_head
        x_mod = x.clone()
        x_mod[..., lo:hi] = 0
        return self.inner(x_mod)


def forward_with_head_ablation(m, token_ids, ablate_layer, ablate_head,
                                 d_head):
    from calm.llm_computer.gemma_substrate import KVCache
    cfg = m.config
    S = token_ids.shape[1]
    cache = KVCache(cfg.n_layers, device="cuda")

    h = m.token_embd[token_ids].to("cuda") * math.sqrt(cfg.d_model)
    m._per_layer_embd = None
    if m.per_layer_token_embd is not None:
        pl_embd = m.per_layer_token_embd[token_ids] * math.sqrt(cfg.d_per_layer)
        pl_embd = pl_embd.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
        if m.per_layer_model_proj is not None:
            h_proj = h @ m.per_layer_model_proj * (1.0 / math.sqrt(cfg.d_model))
            h_proj = h_proj.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
            if m.per_layer_proj_norm_w is not None:
                h_proj = _rms_norm(h_proj, m.per_layer_proj_norm_w, cfg.rms_norm_eps)
            pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
        m._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

    target = m.layers[ablate_layer]
    original = target.attn_output
    if ablate_head is not None:
        target.attn_output = HeadAblatingWrapper(original, ablate_head,
                                                    d_head)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_output = original


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

# Identical triples as R47.1 for comparability.
TRIPLES = [
    (17, 23, 5),   # 396 '3'
    (47, 19, 23),  # 916 '9'
    (37, 14, 50),  # 568 '5'
    (13, 27, 8),   # 359 '3'
    (21, 38, 15),  # 813 '8'
    (11, 11, 10),  # 131 '1'
    (29, 17, 4),   # 497 '4'
    (32, 25, 7),   # 807 '8'
    (16, 31, 12),  # 508 '5'
    (34, 12, 5),   # 413 '4'
]


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print(f"[r47.2] loading substrate (target L{TARGET_LAYER})...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    attn_q_tgt = m.layers[TARGET_LAYER].attn_q
    q_out = attn_q_tgt.out_features
    d_head = q_out // N_HEADS
    attn_out = m.layers[TARGET_LAYER].attn_output
    print(f"[L{TARGET_LAYER}] attn_q out={q_out}, n_heads={N_HEADS} "
          f"→ d_head={d_head}")
    print(f"[L{TARGET_LAYER}] attn_output in={attn_out.in_features}")

    # Baselines
    print(f"\n=== baseline (no ablation) ===")
    baselines = []
    for a, b, c in TRIPLES:
        prompt = f"{a} times {b} plus {c} equals "
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_head_ablation(
            m, token_ids, TARGET_LAYER, None, d_head)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        argmax = int(logits[0, -1].argmax())
        baselines.append((token_ids, correct_d, base_correct, answer))
        print(f"  {a}×{b}+{c}={answer}: argmax="
              f"{tok.id_to_token.get(argmax, '?')!r}, "
              f"base_correct_logit={base_correct:.2f}")

    # Per-head sweep at target layer
    print(f"\n=== ablate each head of L{TARGET_LAYER} across "
          f"{len(TRIPLES)} triples ===\n")
    head_deltas = torch.zeros(N_HEADS, len(TRIPLES))

    for H in range(N_HEADS):
        for j, (token_ids, correct_d, base_correct, _) in enumerate(
                baselines):
            logits = forward_with_head_ablation(
                m, token_ids, TARGET_LAYER, H, d_head)
            abl_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
            head_deltas[H, j] = abl_correct - base_correct

    # Summary
    print(f"{'head':>5} {'mean_Δcorr':>12} {'std':>8} {'#hurts':>8}  "
          f"(hurts = Δ < -0.5)")
    for H in range(N_HEADS):
        mu = head_deltas[H].mean().item()
        std = head_deltas[H].std().item()
        hurts = int((head_deltas[H] < -0.5).sum().item())
        marker = " ←" if mu < -1.0 or hurts >= 7 else ""
        print(f"H{H:>4} {mu:>+12.3f} {std:>8.3f}   {hurts:>2}/{len(TRIPLES)}"
              f"{marker}")

    sorted_idx = head_deltas.mean(dim=1).argsort()
    print(f"\n  Top 3 load-bearing heads at L{TARGET_LAYER}:")
    for idx in sorted_idx[:3]:
        H = idx.item()
        mean_d = head_deltas[H].mean().item()
        hurts = int((head_deltas[H] < -0.5).sum().item())
        print(f"    H{H}: mean Δ = {mean_d:+.3f}, hurts in "
              f"{hurts}/{len(TRIPLES)}")

    # Full-layer reference (R47.1 found L34 mean=-3.17)
    full_layer_ref = -3.17
    top_head_mean = head_deltas.mean(dim=1).min().item()
    coverage = top_head_mean / full_layer_ref if full_layer_ref else 0
    print(f"\n  L{TARGET_LAYER} full-layer Δ (R47.1): {full_layer_ref:.2f}")
    print(f"  Top head Δ:                 {top_head_mean:.2f}  "
          f"(covers {coverage*100:.0f}% of full-layer)")

    # Gate
    concentrated = [
        H for H in range(N_HEADS)
        if head_deltas[H].mean().item() < -1.0
    ]
    print(f"\n========== R47.2 GATE ==========")
    print(f"  heads with mean Δ < -1.0: {concentrated}")
    if concentrated:
        print(f"  ✓ L{TARGET_LAYER} step-2 circuit has {len(concentrated)} "
              f"concentrated head(s). Compilable at attention level.")
        print(f"    Proceed to R47.3: Q/K/V decomp on top head(s).")
    else:
        print(f"  ~ L{TARGET_LAYER} is DIFFUSE at head level. "
              f"Possibilities:")
        print(f"    - Step-2 circuit is FFN-locked (like R30 factual recall)")
        print(f"    - Try L40 (the other 10/10 peak from R47.1)")
        print(f"    - Multiple heads cooperate (R35-style cooperative circuit)")

    torch.save({
        "head_deltas": head_deltas.cpu(),
        "triples": TRIPLES,
        "target_layer": TARGET_LAYER,
    }, f"/tmp/r47_2_l{TARGET_LAYER}_heads.pt")
    print(f"\n  saved raw data: /tmp/r47_2_l{TARGET_LAYER}_heads.pt")


if __name__ == "__main__":
    sys.exit(main())
