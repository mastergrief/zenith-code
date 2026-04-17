"""Round 16: exhaustive ablation sweep across 42 layers × 10 arithmetic
pairs. For each (pair, layer), measure how much ablating that layer
hurts the correct-digit logit. Average across pairs → find layers
that CONSISTENTLY support arithmetic.

Round 14 found L35 flipped '3'→'4' on 17×23. Round 15 showed that's
input-specific. Round 16 asks: averaging across many arithmetic
pairs, which layers' ablation consistently decreases P(correct_digit)?
Those are the load-bearing layers for arithmetic.

Matrix: 42 layers × 10 pairs = 420 forward passes. At ~2s each → ~14 min.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    last = normed[:, -1:, :]
    logits = m.token_embd.output_logits(last)
    return torch.tanh(logits / 30.0) * 30.0


def forward_with_ablation(m, token_ids, ablate_layer=None):
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

    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            h_before = h.clone() if i == ablate_layer else None
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == ablate_layer:
                h = h_before
    return project_to_logits(m, h)


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Matrix: layer × pair → Δ(correct_digit_logit)
    deltas = torch.zeros(m.config.n_layers, len(PAIRS))
    baseline_correct = torch.zeros(len(PAIRS))

    print(f"\n=== baseline forwards ===")
    baselines = []
    for j, (a, b) in enumerate(PAIRS):
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(a * b)[0]
        base_logits = forward_with_ablation(m, token_ids, ablate_layer=None)
        base_correct = base_logits[0, -1, DIGIT_IDS[correct_d]].item()
        baseline_correct[j] = base_correct
        baselines.append((token_ids, correct_d, base_correct))
        base_argmax = int(base_logits[0, -1].argmax())
        print(f"  {a}×{b}={a*b}: base_argmax={tok.id_to_token.get(base_argmax, '?')!r}, "
              f"base_correct_logit={base_correct:.2f}")

    print(f"\n=== ablation sweep: 42 layers × 10 pairs ===")
    for L in range(m.config.n_layers):
        for j, (token_ids, correct_d, base_correct) in enumerate(baselines):
            abl_logits = forward_with_ablation(m, token_ids, ablate_layer=L)
            abl_correct = abl_logits[0, -1, DIGIT_IDS[correct_d]].item()
            deltas[L, j] = abl_correct - base_correct
        if L % 5 == 0:
            mean_d = deltas[L].mean().item()
            print(f"  L{L:>2}: mean Δ(correct) = {mean_d:+.3f}")

    print(f"\n========== LAYER AVERAGES ==========")
    mean_deltas = deltas.mean(dim=1)
    print(f"\n{'L':>3} {'mean_Δcorr':>12} {'std':>8} {'#hurts':>8}   (hurts = Δ < -0.5)")
    for L in range(m.config.n_layers):
        mu = mean_deltas[L].item()
        std = deltas[L].std().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        marker = " ←" if mu < -1.0 or hurts >= 7 else ""
        print(f"{L:>3} {mu:>+12.3f} {std:>8.3f}   {hurts:>2}/10{marker}")

    # Top 10 layers that MOST consistently hurt correct (mean Δ most negative)
    print(f"\n  Layers whose ablation MOST hurts arithmetic on average:")
    sorted_idx = mean_deltas.argsort()[:10]
    for idx in sorted_idx:
        L = idx.item()
        print(f"    L{L:>2}: mean Δ = {mean_deltas[L]:+.3f} "
              f"(hurts in {int((deltas[L] < -0.5).sum())}/10)")


if __name__ == "__main__":
    sys.exit(main())
