"""Round 47.4: per-head ablation at L24 — the multi-step composition peak.

R47.3 found L24 as the dominant multi-step peak (mean Δ = -17.23,
10/10 hurts) with clean-prompt format that avoids Gemma's copy-c
shortcut. L24 is an SWA layer sharing KV with L22 (step-1 hub).

Hypothesis: if L24 is where Gemma composes (step-1 intermediate + c)
into the final answer, one or two of L24's 8 Q-heads should carry
the load (mirror of R17: L23 H1+H4 carried ~95% of single-step's L23
signal).

Gate: ≥ 1 head with mean Δ < -2.0 (given full-layer is -17.23, a
concentrated head should contribute at least 12% of the signal).

If diffuse (no head < -2.0), L24's role is FFN-based — same typology
as R30/R37 (factual recall, comparison). Pivot to R47.5: attn vs FFN
ablation at L24 to quantify attention vs FFN contribution.

Clean prompt format (same as R47.3): 'What is ({a} * {b}) + {c}? Answer: '
8 heads × 10 triples + 10 baselines = 90 forwards ≈ 3 min.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

TARGET_LAYER = 24
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


def build_prompt(a: int, b: int, c: int) -> str:
    return f"What is ({a} * {b}) + {c}? Answer: "


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print(f"[r47.4] loading substrate (target L{TARGET_LAYER})...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    attn_q_tgt = m.layers[TARGET_LAYER].attn_q
    q_out = attn_q_tgt.out_features
    d_head = q_out // N_HEADS
    attn_out = m.layers[TARGET_LAYER].attn_output
    print(f"[L{TARGET_LAYER}] attn_q out={q_out}, n_heads={N_HEADS} "
          f"→ d_head={d_head}")
    print(f"[L{TARGET_LAYER}] attn_output in={attn_out.in_features}")
    kv_src = m.config.kv_source_layer(TARGET_LAYER, is_swa=True)
    print(f"[L{TARGET_LAYER}] KV source layer = L{kv_src}")

    # Baselines (clean format)
    print(f"\n=== baseline (clean prompt format) ===")
    baselines = []
    n_correct_argmax = 0
    for a, b, c in TRIPLES:
        prompt = build_prompt(a, b, c)
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_head_ablation(
            m, token_ids, TARGET_LAYER, None, d_head)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        is_correct = argmax_tok.lstrip('▁') == correct_d
        if is_correct:
            n_correct_argmax += 1
        baselines.append((token_ids, correct_d, base_correct, answer,
                            is_correct))
        mark = "✓" if is_correct else " "
        print(f"  {mark} {a}×{b}+{c}={answer}: argmax={argmax_tok!r}, "
              f"base_correct_logit={base_correct:.2f}")
    print(f"\n  baseline argmax correct: {n_correct_argmax}/10")

    # Per-head sweep
    print(f"\n=== ablate each head of L{TARGET_LAYER} across "
          f"{len(TRIPLES)} triples ===\n")
    head_deltas = torch.zeros(N_HEADS, len(TRIPLES))

    for H in range(N_HEADS):
        for j, (token_ids, correct_d, base_correct, _, _) in enumerate(
                baselines):
            logits = forward_with_head_ablation(
                m, token_ids, TARGET_LAYER, H, d_head)
            abl_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
            head_deltas[H, j] = abl_correct - base_correct

    print(f"{'head':>5} {'mean_Δcorr':>12} {'std':>8} {'#hurts':>8}  "
          f"(hurts = Δ < -0.5)")
    for H in range(N_HEADS):
        mu = head_deltas[H].mean().item()
        std = head_deltas[H].std().item()
        hurts = int((head_deltas[H] < -0.5).sum().item())
        marker = " ←" if mu < -2.0 or hurts >= 7 else ""
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

    full_layer_ref = -17.23  # R47.3
    top_head_mean = head_deltas.mean(dim=1).min().item()
    coverage = top_head_mean / full_layer_ref if full_layer_ref else 0
    top2_sum = head_deltas.mean(dim=1).sort().values[:2].sum().item()
    top2_cov = top2_sum / full_layer_ref if full_layer_ref else 0
    print(f"\n  L{TARGET_LAYER} full-layer Δ (R47.3): {full_layer_ref:.2f}")
    print(f"  Top head:        {top_head_mean:.2f}  "
          f"(covers {coverage*100:.0f}%)")
    print(f"  Top 2 heads sum: {top2_sum:.2f}  "
          f"(covers {top2_cov*100:.0f}%)")

    # Gate (L24 baseline is -17.23 so threshold is -2.0)
    concentrated = [
        H for H in range(N_HEADS)
        if head_deltas[H].mean().item() < -2.0
    ]
    print(f"\n========== R47.4 GATE ==========")
    print(f"  heads with mean Δ < -2.0: {concentrated}")
    if concentrated:
        print(f"  ✓ L{TARGET_LAYER} multi-step composition circuit has "
              f"{len(concentrated)} concentrated head(s).")
        print(f"    COMPILABLE AT ATTENTION LEVEL.")
        print(f"    Next: R47.5 Q/K/V decomposition on top head(s) to "
              f"localize further.")
    else:
        print(f"  ~ L{TARGET_LAYER} is DIFFUSE at head level.")
        print(f"    Composition signal is FFN-locked or cooperatively")
        print(f"    distributed. Options:")
        print(f"      - R47.5: attn vs FFN ablation at L24 (quantify split)")
        print(f"      - R47.6: top-K heads cooperative test (sum of ablations)")

    torch.save({
        "head_deltas": head_deltas.cpu(),
        "triples": TRIPLES,
        "target_layer": TARGET_LAYER,
        "n_correct_argmax": n_correct_argmax,
    }, f"/tmp/r47_4_l{TARGET_LAYER}_heads.pt")
    print(f"\n  saved raw data: /tmp/r47_4_l{TARGET_LAYER}_heads.pt")


if __name__ == "__main__":
    sys.exit(main())
