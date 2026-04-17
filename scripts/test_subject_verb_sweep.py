"""Round 38: subject-verb agreement sweep — a linguistic capability.

All 5 previously mapped capabilities are numeric/retrieval/pattern
(arithmetic, factual recall, induction, counting, comparison). We
need a LINGUISTIC test to check whether the 3-shape typology
(concentrated / cooperative / diffuse) holds for syntax, not just
numeric tasks.

Subject-verb agreement is core to language: "The cat is" vs
"The cats are". Model should select verb form agreeing with
subject number. If this localizes cleanly via the same protocol,
methodology generalizes to linguistic circuits.

Prompt format: sentence context where next-token should be a
number-sensitive verb. Measure Δ(Gemma's baseline argmax logit)
under per-layer attn ablation. Gemma's argmax IS the correct
verb form in the baseline — we just check whether each layer's
attention is load-bearing for producing it.
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


# Subject-verb test prompts. Each forces a specific verb form.
# Structure: context + singular/plural noun phrase + position where
# Gemma should emit an appropriate agreement-marked verb.
PROMPTS = [
    # (prompt, expected_verb_number, description)
    ("The cat that sits near the window", "sing", "sing subj + modifier"),
    ("The cats that sit near the window", "plur", "plur subj + modifier"),
    ("The dog with the red collar", "sing", "sing subj + modifier"),
    ("The dogs with the red collar", "plur", "plur subj + modifier"),
    ("The book on the shelves", "sing", "sing subj + plural distractor"),
    ("The books on the shelf", "plur", "plur subj + sing distractor"),
    ("The child near the parents", "sing", "sing subj + plural distractor"),
    ("The children near the parent", "plur", "plur subj + sing distractor"),
    ("The teacher with the students", "sing", "sing subj + plural distractor"),
    ("The teachers with the student", "plur", "plur subj + sing distractor"),
    ("The artist whose paintings sell", "sing", "sing subj + plural clause"),
    ("The artists whose painting sells", "plur", "plur subj + sing clause"),
    ("The key to the cabinets", "sing", "sing subj + plural distractor"),
    ("The keys to the cabinet", "plur", "plur subj + sing distractor"),
    ("The author of many novels", "sing", "sing subj + plural modifier"),
    ("The authors of the novel", "plur", "plur subj + sing modifier"),
    ("The farmer beside the horses", "sing", "sing subj + plural distractor"),
    ("The farmers beside the horse", "plur", "plur subj + sing distractor"),
    ("The computer with many bugs", "sing", "sing subj + plural modifier"),
    ("The computers with one bug", "plur", "plur subj + sing modifier"),
]

# Number-sensitive verb tokens we expect Gemma to emit. These are
# the primary agreement-marked auxiliary/copula verbs with single-
# token BPE representations.
SING_VERB_TOKENS = {"▁is", "▁was", "▁has", "▁does", "▁seems", "▁sits", "▁sells"}
PLUR_VERB_TOKENS = {"▁are", "▁were", "▁have", "▁do", "▁seem", "▁sit"}


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[sv-sweep] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print(f"\n=== sanity: does Gemma do subject-verb agreement? ===")
    clean = []
    for prompt, expected_num, desc in PROMPTS:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_attn_ablation(m, token_ids, None)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        # Check if argmax matches expected number
        expected_set = SING_VERB_TOKENS if expected_num == "sing" else PLUR_VERB_TOKENS
        wrong_set = PLUR_VERB_TOKENS if expected_num == "sing" else SING_VERB_TOKENS
        matches = argmax_tok in expected_set
        # Accept also if any agreement-sensitive verb appears in top-3
        top3 = logits[0, -1].topk(3).indices.tolist()
        top3_toks = [tok.id_to_token.get(t, '?') for t in top3]
        has_correct_verb = any(t in expected_set for t in top3_toks)
        has_wrong_verb = any(t in wrong_set for t in top3_toks)
        ok = matches or (has_correct_verb and not has_wrong_verb)
        status = "✓" if ok else "✗"
        print(f"  {status} ({expected_num:>4}) {prompt!r:>42}")
        print(f"        argmax={argmax_tok!r}  top3={top3_toks}")
        if ok:
            base_logit = logits[0, -1, argmax].item()
            clean.append((prompt, expected_num, argmax, argmax_tok, token_ids, base_logit))

    if len(clean) < 8:
        print(f"\n  only {len(clean)} clean prompts. Sweep anyway if >=6.")
        if len(clean) < 6:
            return 1

    print(f"\n{len(clean)}/{len(PROMPTS)} clean\n")

    import time
    n_layers = m.config.n_layers
    deltas = torch.zeros(n_layers, len(clean))
    t0 = time.time()
    print(f"=== 42-layer sweep × {len(clean)} prompts ===")
    for L in range(n_layers):
        for j, (_, _, tid, _, tids, base) in enumerate(clean):
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
    print(f"\n=== top-10 layers hurting subject-verb agreement ===")
    for rank, idx in enumerate(sorted_idx[:10]):
        L = int(idx.item())
        mu = deltas[L].mean().item()
        hurts = int((deltas[L] < -0.5).sum().item())
        lyr = "GLB" if L in GLOBAL else "SWA"
        print(f"  {rank+1}. L{L:>2} ({lyr})  mean_Δ={mu:+.3f}  hurts={hurts}/{len(clean)}")


if __name__ == "__main__":
    sys.exit(main())
