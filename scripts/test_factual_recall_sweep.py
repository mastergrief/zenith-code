"""Round 29: layer ablation sweep on factual recall — does the arithmetic
methodology generalize to a different Tier-1 circuit?

Arithmetic (R16) localized to L22-L30 cluster, L23 peak, via SWA/global
architecture. Hypothesis: factual recall ("The capital of X is Y") is
also Tier-1 — it requires cross-position info (subject token needs to
be read). Expect similar localization at a global layer.

Measurement: for each layer L ∈ [0, 41], ablate attn_output (zero-return)
so attn contributes nothing, measure Δ(correct-capital-token logit)
relative to baseline across 10 country-capital pairs. Layers with
large negative Δ are load-bearing for factual recall.

Prompt format: "The capital of X is " → should emit first token of X's
capital as next. Capitals chosen to have clean single-token first BPE
representations where possible.

Comparison points:
  Arithmetic R16: L23 mean Δ=-10.18 (10/10 hurts), L22/L24/L26/L28/L29
                  also 10/10 hurts. L35 secondary (-1.50, 9/10).
  Factual recall: where will it cluster? Global layers are 5, 11, 17,
                  23, 29, 35, 41. Literature suggests factual recall
                  lives in middle-to-late FFN layers (ROME/MEMIT work).
                  We'll see.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

# 10 country-capital pairs with clean single-word capitals. We'll
# check the FIRST Gemma BPE token of each capital (that's what the
# model would emit as its "next token" after "is ").
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


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class ZeroReturning:
    """Wrap a linear to return all-zeros of its output shape."""
    def __init__(self, inner):
        self.inner = inner
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        return torch.zeros_like(out)


def forward_with_attn_ablation(m, token_ids, ablate_layer):
    """Run forward with attn_output zero-returned at ablate_layer."""
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


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[factual-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Use Gemma's own baseline argmax as the target token. For correct
    # country-capital pairs Gemma natively emits '▁Paris', '▁Berlin' etc
    # as the next token. If argmax doesn't match the expected capital,
    # skip that pair (wrong fact or unusual tokenization).
    print(f"\n=== baselines (use argmax as target) ===")
    baselines = []
    for country, capital in PAIRS:
        prompt = f"The capital of {country} is"
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_attn_ablation(m, token_ids, None)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        base_logit = logits[0, -1, argmax].item()
        # Require argmax to textually match the capital (modulo leading ▁)
        stripped = argmax_tok.lstrip('▁')
        matches = (stripped.lower() == capital.lower() or
                   capital.lower().startswith(stripped.lower()))
        ok = "✓" if matches else "✗"
        print(f"  {ok} {country:>10} → argmax={argmax_tok!r} "
              f"(logit {base_logit:+.2f})  expected capital={capital!r}")
        if matches:
            baselines.append((country, capital, argmax, argmax_tok, token_ids, base_logit))

    # 42-layer sweep
    n_layers = m.config.n_layers
    deltas = torch.zeros(n_layers, len(baselines))

    print(f"\n=== ablating attn at each layer × {len(baselines)} pairs ===")
    import time
    t0 = time.time()
    for L in range(n_layers):
        for j, (_, _, tid, _, token_ids, base_logit) in enumerate(baselines):
            logits = forward_with_attn_ablation(m, token_ids, L)
            abl_logit = logits[0, -1, tid].item()
            deltas[L, j] = abl_logit - base_logit
        if (L + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"  [{L+1}/{n_layers}] {elapsed:.0f}s")

    # Summary
    print(f"\n=== per-layer summary (attn ablation Δ) ===\n")
    print(f"  {'L':>3} {'mean_Δ':>10} {'std':>8} {'#hurts':>8}  type  marker")
    # Identify global layers: 5, 11, 17, 23, 29, 35, 41
    GLOBAL = {5, 11, 17, 23, 29, 35, 41}
    for L in range(n_layers):
        mu = deltas[L].mean().item()
        sd = deltas[L].std().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr_type = "GLB" if L in GLOBAL else "SWA"
        marker = ""
        if mu < -3.0 or hurts >= 8:
            marker = " ← STRONG"
        elif mu < -1.0 or hurts >= 5:
            marker = " ~ moderate"
        print(f"  L{L:>2} {mu:>+10.3f} {sd:>8.3f} {hurts:>2}/{len(baselines)} {lyr_type}{marker}")

    # Rank top 10 hurters
    sorted_idx = deltas.mean(dim=1).argsort()
    print(f"\n=== top-10 layers hurting factual recall ===")
    for rank, idx in enumerate(sorted_idx[:10]):
        L = int(idx.item())
        mu = deltas[L].mean().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr_type = "GLB" if L in GLOBAL else "SWA"
        print(f"  {rank+1}. L{L:>2} ({lyr_type})  mean_Δ={mu:+.3f}  hurts={hurts}/{len(baselines)}")

    # Save
    torch.save({
        "deltas": deltas,
        "baselines": [(c, cap, tid, ttok, bl) for c, cap, tid, ttok, _, bl in baselines],
    }, "/tmp/r29_factual_sweep.pt")
    print(f"\n[saved] /tmp/r29_factual_sweep.pt")


if __name__ == "__main__":
    sys.exit(main())
