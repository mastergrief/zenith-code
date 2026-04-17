"""Round 39: per-head at L23, L29, L35 for subject-verb agreement.

R38: three-global-layer pipeline L23→L29→L35 for SV agreement. Is
the circuit concentrated at specific heads (Tier-2-compilable) or
diffuse (FFN-locked, needs different approach)?

L29 is the cleanest (16/16, std 1.56) — start there. L23 and L35
are shared with other capabilities; check whether SV agreement
uses the same heads as arithmetic/comparison or different ones.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYERS = [23, 29, 35]
N_HEADS = 8

PROMPTS = [
    ("The cat that sits near the window", "sing"),
    ("The cats that sit near the window", "plur"),
    ("The dog with the red collar", "sing"),
    ("The dogs with the red collar", "plur"),
    ("The book on the shelves", "sing"),
    ("The books on the shelf", "plur"),
    ("The child near the parents", "sing"),
    ("The children near the parent", "plur"),
    ("The teacher with the students", "sing"),
    ("The teachers with the student", "plur"),
    ("The key to the cabinets", "sing"),
    ("The keys to the cabinet", "plur"),
    ("The author of many novels", "sing"),
    ("The authors of the novel", "plur"),
    ("The farmer beside the horses", "sing"),
    ("The farmers beside the horse", "plur"),
    ("The computer with many bugs", "sing"),
    ("The computers with one bug", "plur"),
]

SING_VERB_TOKENS = {"▁is", "▁was", "▁has", "▁does", "▁seems", "▁sits", "▁sells"}
PLUR_VERB_TOKENS = {"▁are", "▁were", "▁have", "▁do", "▁seem", "▁sit"}


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


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[sv-per-head] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    for L in TARGET_LAYERS:
        q_out = m.layers[L].attn_q.out_features
        d_head = q_out // N_HEADS
        print(f"  L{L}: d_head={d_head}")

    # Baselines with top-3 verb-token check
    print(f"\n=== baselines ===")
    baselines = []
    for prompt, expected_num in PROMPTS:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_head_ablation(
            m, token_ids, TARGET_LAYERS[0], None,
            m.layers[TARGET_LAYERS[0]].attn_q.out_features // N_HEADS)
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        expected_set = SING_VERB_TOKENS if expected_num == "sing" else PLUR_VERB_TOKENS
        wrong_set = PLUR_VERB_TOKENS if expected_num == "sing" else SING_VERB_TOKENS
        top3 = logits[0, -1].topk(3).indices.tolist()
        top3_toks = [tok.id_to_token.get(t, '?') for t in top3]
        ok = argmax_tok in expected_set or (
            any(t in expected_set for t in top3_toks)
            and not any(t in wrong_set for t in top3_toks))
        if ok:
            base_logit = logits[0, -1, argmax].item()
            baselines.append((prompt, expected_num, argmax, argmax_tok, token_ids, base_logit))

    print(f"  {len(baselines)}/{len(PROMPTS)} clean\n")

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
            mark = " ← STRONG" if mu < -1.0 or hurts >= int(len(baselines) * 0.7) else ""
            print(f"  H{H:>4} {mu:>+10.3f} {sd:>8.3f}  {hurts:>2}/{len(baselines)}{mark}")
        sorted_idx = deltas.mean(dim=1).argsort()
        print(f"\n  L{L} top-3: " + ", ".join(
            f"H{int(i)}:{deltas[i].mean().item():+.3f}"
            for i in sorted_idx[:3]))


if __name__ == "__main__":
    sys.exit(main())
