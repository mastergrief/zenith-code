"""Round 36: comparison sweep — "Which is larger, 5 or 8?" → "8".

Hypothesis: numeric comparison is a primitive circuit distinct from
arithmetic and induction. Or it composes them. Sweep to see.

Format: "Which is larger, A or B? Answer: " → expect max(A, B).
Vary which operand is larger so model can't cheat on positional
preference. Keep operands single-digit so tokenization is clean.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class ZeroReturning:
    def __init__(self, inner):
        self.inner = inner
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        return torch.zeros_like(self.inner(x))


def forward_with_attn_ablation(m, token_ids, ablate_layer):
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

    saved = None
    if ablate_layer is not None:
        tgt = m.layers[ablate_layer]
        saved = tgt.attn_output
        tgt.attn_output = ZeroReturning(saved)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        if saved is not None:
            m.layers[ablate_layer].attn_output = saved


def build_comparison_prompts(seed=0, n=20):
    """Balanced: half have larger operand first, half second."""
    random.seed(seed)
    prompts = []
    for _ in range(n):
        a = random.randint(1, 9)
        b = random.randint(1, 9)
        if a == b:
            continue
        answer = max(a, b)
        # "Which is larger, A or B? Answer: "
        prompt = f"Which is larger, {a} or {b}? Answer: "
        prompts.append((prompt, answer, a, b))
    return prompts


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[comparison-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    prompts = build_comparison_prompts()
    print(f"\n=== sanity: does Gemma do comparison? ===")
    clean = []
    for prompt, expected, a, b in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_attn_ablation(m, token_ids, None)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        stripped = argmax_tok.lstrip('▁')
        ok = stripped == str(expected)
        if ok:
            base_logit = logits[0, -1, argmax].item()
            clean.append((prompt, expected, argmax, token_ids, base_logit))
        mark = "✓" if ok else "✗"
        print(f"  {mark} {prompt!r:>38} → '{expected}' argmax={argmax_tok!r}")

    if len(clean) < 8:
        print(f"\n  only {len(clean)}/{len(prompts)} clean. Sweep anyway if >=5.")
        if len(clean) < 5:
            return 1

    print(f"\n{len(clean)}/{len(prompts)} clean\n")

    n_layers = m.config.n_layers
    deltas = torch.zeros(n_layers, len(clean))
    import time
    t0 = time.time()
    print(f"=== 42-layer sweep × {len(clean)} prompts ===")
    for L in range(n_layers):
        for j, (_, _, tid, tids, base) in enumerate(clean):
            logits = forward_with_attn_ablation(m, tids, L)
            deltas[L, j] = logits[0, -1, tid].item() - base
        if (L + 1) % 10 == 0:
            print(f"  [{L+1}/{n_layers}] {time.time()-t0:.0f}s")

    GLOBAL = {5, 11, 17, 23, 29, 35, 41}
    print(f"\n=== per-layer summary ===\n  {'L':>3} {'mean_Δ':>10} {'std':>8} {'#hurts':>8} type")
    for L in range(n_layers):
        mu = deltas[L].mean().item()
        sd = deltas[L].std().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr = "GLB" if L in GLOBAL else "SWA"
        mark = ""
        if mu < -3.0 or hurts >= int(len(clean) * 0.8):
            mark = " ← STRONG"
        elif mu < -1.0 or hurts >= int(len(clean) * 0.5):
            mark = " ~ moderate"
        print(f"  L{L:>2} {mu:>+10.3f} {sd:>8.3f}  {hurts:>2}/{len(clean)}  {lyr}{mark}")

    sorted_idx = deltas.mean(dim=1).argsort()
    print(f"\n=== top-10 layers hurting comparison ===")
    for rank, idx in enumerate(sorted_idx[:10]):
        L = int(idx.item())
        mu = deltas[L].mean().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr = "GLB" if L in GLOBAL else "SWA"
        print(f"  {rank+1}. L{L:>2} ({lyr})  mean_Δ={mu:+.3f}  hurts={hurts}/{len(clean)}")


if __name__ == "__main__":
    sys.exit(main())
