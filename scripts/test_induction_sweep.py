"""Round 31: induction-head layer sweep.

Induction is the canonical mechinterp capability: given a prompt
"A B ... A", the model should predict "B" (continuing the pattern
established earlier). Well-traced in GPT-2 and other models via
Olsson et al. 2022.

Probe: use RANDOM uppercase single-letter pairs (not natural words
that might trigger other mechanisms). Format:
  "Letters: A F T M R A"  → expect "F" (copy of the token AFTER A's
                              earlier occurrence)

For each layer L ∈ [0, 41], ablate attn, measure Δ(correct-letter
logit) across 10 random patterns. Expected: induction heads have
a specific layer that carries most of the signal (not distributed
like factual recall).

If Gemma has a clean induction-head circuit, this should localize
as cleanly as arithmetic did. If not, induction is distributed
(more like factual recall).
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


def build_induction_prompts(tok, n_prompts=15, seq_len=4, seed=0):
    """Build repeated-sequence induction prompts — the gold standard
    induction-head test.

    Format: "A B C D A B C D A B" → model should predict " C" (the
    next letter in the repeated pattern).

    The sequence appears twice in full + starts a third time, giving
    the model a very strong pattern to match. Induction heads should
    fire strongly on this.
    """
    random.seed(seed)
    letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    prompts = []
    for _ in range(n_prompts):
        chosen = random.sample(letters, seq_len)
        # Repeat twice + start of third
        seq = chosen + chosen + chosen[:2]   # A B C D A B C D A B
        prompt = " ".join(seq)
        expected = chosen[2]   # C
        prompts.append((prompt, expected, chosen, seq))
    return prompts


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[induction-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    prompts = build_induction_prompts(tok, n_prompts=20, seq_len=4, seed=0)

    # Filter to prompts where Gemma's baseline argmax is "▁C" (the
    # expected induction answer)
    print(f"\n=== sanity: does Gemma do induction on these prompts? ===")
    clean = []
    for prompt, b, _, seq in prompts:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_attn_ablation(m, token_ids, None)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        # Expected token: "▁B" (leading space + letter)
        expected_tok = "▁" + b
        matches = argmax_tok == expected_tok
        ok = "✓" if matches else "✗"
        print(f"  {ok} {prompt!r}  expected=[{expected_tok!r}] "
              f"argmax={argmax_tok!r} logit={logits[0, -1, argmax].item():+.2f}")
        if matches:
            base_logit = logits[0, -1, argmax].item()
            clean.append((prompt, b, argmax, token_ids, base_logit))

    if len(clean) < 5:
        print(f"\n  only {len(clean)} clean prompts — Gemma may not reliably do "
              f"induction on this format. Try different format?")
        if not clean:
            return 1

    print(f"\n{len(clean)}/{len(prompts)} clean baselines\n")

    # 42-layer attn sweep
    import time
    n_layers = m.config.n_layers
    deltas = torch.zeros(n_layers, len(clean))
    t0 = time.time()
    print(f"=== 42-layer attn ablation sweep × {len(clean)} prompts ===")
    for L in range(n_layers):
        for j, (_, _, tid, tids, base) in enumerate(clean):
            logits = forward_with_attn_ablation(m, tids, L)
            deltas[L, j] = logits[0, -1, tid].item() - base
        if (L + 1) % 10 == 0:
            print(f"  [{L+1}/{n_layers}] {time.time()-t0:.0f}s")

    # Summary
    GLOBAL = {5, 11, 17, 23, 29, 35, 41}
    print(f"\n=== per-layer summary ===\n")
    print(f"  {'L':>3} {'mean_Δ':>10} {'std':>8} {'#hurts':>8} type")
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
    print(f"\n=== top-10 layers hurting induction ===")
    for rank, idx in enumerate(sorted_idx[:10]):
        L = int(idx.item())
        mu = deltas[L].mean().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr = "GLB" if L in GLOBAL else "SWA"
        print(f"  {rank+1}. L{L:>2} ({lyr})  mean_Δ={mu:+.3f}  hurts={hurts}/{len(clean)}")

    torch.save({
        "deltas": deltas,
        "n_clean": len(clean),
    }, "/tmp/r31_induction_sweep.pt")
    print(f"\n[saved] /tmp/r31_induction_sweep.pt")


if __name__ == "__main__":
    sys.exit(main())
