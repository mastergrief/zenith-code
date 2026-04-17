"""Round 35: per-head at L20 and L31 for counting.

R34 showed L20 is counting's peak (Δ=-3.932, 6/6 hurts) and L31 is
secondary (-2.330, 6/6). L33 and L37 also strong but shared with
induction.

Question: at L20 (the counting-unique peak), which head(s) carry
the signal? If concentrated, it's a new primitive circuit. If
diffuse, it's a distributed counting mechanism.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYERS = [20, 31, 33, 37]
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


def forward_with_head_ablation(m, token_ids, ablate_layer, ablate_head, d_head):
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


def build_counting_prompts(seed=0, n=20):
    random.seed(seed)
    prompts = []
    for _ in range(n):
        length = random.randint(4, 7)
        start = random.randint(1, 9)
        nums = list(range(start, start + length))
        next_num = start + length
        if next_num > 9:
            continue
        prompt = "Count: " + ", ".join(str(x) for x in nums) + ", "
        prompts.append((prompt, next_num))
    return prompts


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[counting-per-head] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    for L in TARGET_LAYERS:
        q_out = m.layers[L].attn_q.out_features
        d_head = q_out // N_HEADS
        print(f"  L{L}: d_head={d_head} ({'GLB' if d_head > 256 else 'SWA'})")

    # Baselines
    prompts = build_counting_prompts()
    print(f"\n=== baselines ===")
    baselines = []
    for prompt, expected in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_head_ablation(
            m, token_ids, TARGET_LAYERS[0], None,
            m.layers[TARGET_LAYERS[0]].attn_q.out_features // N_HEADS)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        stripped = argmax_tok.lstrip('▁')
        if stripped == str(expected):
            base_logit = logits[0, -1, argmax].item()
            baselines.append((prompt, expected, argmax, token_ids, base_logit))
            print(f"  ✓ {prompt!r:>26} → '{stripped}' ({base_logit:+.2f})")

    print(f"\n{len(baselines)}/{len(prompts)} clean\n")

    # Per-head sweep at each layer
    for L in TARGET_LAYERS:
        d_head = m.layers[L].attn_q.out_features // N_HEADS
        print(f"\n=== L{L} (d_head={d_head}) per-head ablation ===")
        deltas = torch.zeros(N_HEADS, len(baselines))
        for H in range(N_HEADS):
            for j, (_, _, tid, tids, base) in enumerate(baselines):
                logits = forward_with_head_ablation(m, tids, L, H, d_head)
                deltas[H, j] = logits[0, -1, tid].item() - base
        print(f"  {'head':>5} {'mean_Δ':>10} {'std':>8} {'#hurts':>8}")
        for H in range(N_HEADS):
            mu = deltas[H].mean().item()
            sd = deltas[H].std().item()
            hurts = int((deltas[H] < -0.5).sum().item())
            mark = " ← STRONG" if mu < -1.0 or hurts >= int(len(baselines) * 0.7) else ""
            print(f"  H{H:>4} {mu:>+10.3f} {sd:>8.3f}  {hurts:>2}/{len(baselines)}{mark}")
        sorted_idx = deltas.mean(dim=1).argsort()
        print(f"\n  L{L} top-3: " + ", ".join(
            f"H{int(i)}:{deltas[i].mean().item():+.3f}"
            for i in sorted_idx[:3]))


if __name__ == "__main__":
    sys.exit(main())
