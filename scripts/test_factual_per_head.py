"""Round 30: per-head ablation at L11 and L5 for factual recall.

R29 showed L11 is the peak (-1.555) and L5 is co-dominant (-1.180,
10/10 hurts) for factual recall. Which specific Q heads at each of
these layers carry the "subject → associated-fact" lookup?

L11 is a global layer (d_head_q=512). L5 is also global (d_head_q=512).
Same 8-head structure as L23. Apply the R17 pattern.

For each layer L ∈ {5, 11}, ablate each of its 8 Q heads, measure
Δ(correct-capital-token logit) across the 10 country-capital pairs.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
PAIRS = [
    ("France", "Paris"),
    ("Germany", "Berlin"),
    ("Japan", "Tokyo"),
    ("Russia", "Moscow"),
    ("Italy", "Rome"),
    ("Spain", "Madrid"),
    ("China", "Beijing"),
    ("Egypt", "Cairo"),
    ("Canada", "Ottawa"),
    ("Australia", "Canberra"),
]
TARGET_LAYERS = [5, 11]
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
        target.attn_output = HeadAblatingWrapper(original, ablate_head, d_head)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_output = original


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[factual-per-head] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    for L in TARGET_LAYERS:
        q_out = m.layers[L].attn_q.out_features
        d_head = q_out // N_HEADS
        print(f"  L{L}: attn_q out={q_out} → d_head={d_head}")

    # Baselines using argmax as target
    print(f"\n=== baselines ===")
    baselines = []
    for country, capital in PAIRS:
        prompt = f"The capital of {country} is"
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_head_ablation(m, token_ids, TARGET_LAYERS[0], None,
                                              m.layers[TARGET_LAYERS[0]].attn_q.out_features // N_HEADS)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        stripped = argmax_tok.lstrip('▁')
        matches = (stripped.lower() == capital.lower() or
                   capital.lower().startswith(stripped.lower()))
        if matches:
            base_logit = logits[0, -1, argmax].item()
            baselines.append((country, capital, argmax, argmax_tok, token_ids, base_logit))
            print(f"  ✓ {country:>10} → {argmax_tok!r} ({base_logit:+.2f})")

    print(f"\n{len(baselines)}/{len(PAIRS)} clean baselines\n")

    # Per-head sweep for each target layer
    for L in TARGET_LAYERS:
        d_head = m.layers[L].attn_q.out_features // N_HEADS
        print(f"\n=== L{L} (d_head={d_head}) per-head ablation ===")
        deltas = torch.zeros(N_HEADS, len(baselines))
        for H in range(N_HEADS):
            for j, (_, _, tid, _, tids, base) in enumerate(baselines):
                logits = forward_with_head_ablation(m, tids, L, H, d_head)
                deltas[H, j] = logits[0, -1, tid].item() - base
        print(f"  {'head':>5} {'mean_Δ':>10} {'std':>8} {'#hurts':>8}")
        for H in range(N_HEADS):
            mu = deltas[H].mean().item()
            sd = deltas[H].std().item()
            hurts = int((deltas[H] < -0.5).sum().item())
            mark = " ← STRONG" if mu < -1.0 or hurts >= 7 else ""
            print(f"  H{H:>4} {mu:>+10.3f} {sd:>8.3f}  {hurts:>2}/{len(baselines)}{mark}")
        sorted_idx = deltas.mean(dim=1).argsort()
        print(f"\n  L{L} top-3 hurters:")
        for idx in sorted_idx[:3]:
            H = int(idx.item())
            mu = deltas[H].mean().item()
            hurts = int((deltas[H] < -0.5).sum().item())
            print(f"    H{H}: mean_Δ={mu:+.3f}, hurts={hurts}/{len(baselines)}")


if __name__ == "__main__":
    sys.exit(main())
