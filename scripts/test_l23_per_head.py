"""Round 17: per-head ablation on L23 (the arithmetic-peak global layer).

L23 has 8 Q-heads. Ablate each head individually by zeroing its slice
of attention output before the attn_output matmul. Measure Δ(correct)
across the 10 arithmetic pairs, find which HEAD carries the load.

If it's one specific head, we've localized arithmetic from 42 layers
→ 1 layer → 1 head. That's ~128 d_head=2 sub-heads or a ~512-d slice
of L23's attention. SAE/feature-inventory work would target that
narrow subspace.
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


class HeadAblatingWrapper:
    """Wraps a linear so inputs in head h's slice are zeroed before passing
    through. Matches MmapTq4Linear's call convention (x @ w)."""
    def __init__(self, inner, head_idx, d_head):
        self.inner = inner
        self.head_idx = head_idx
        self.d_head = d_head
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)  # compat

    def __call__(self, x):
        lo = self.head_idx * self.d_head
        hi = (self.head_idx + 1) * self.d_head
        x_mod = x.clone()
        x_mod[..., lo:hi] = 0
        return self.inner(x_mod)


def forward_with_head_ablation(m, token_ids, ablate_layer, ablate_head,
                                d_head):
    """Run forward with a specific (layer, head) ablated.
    ablate_head=None means no ablation (baseline)."""
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

    # Install the head-ablating wrapper on target layer's attn_output
    target_layer = m.layers[ablate_layer]
    original = target_layer.attn_output
    if ablate_head is not None:
        target_layer.attn_output = HeadAblatingWrapper(original, ablate_head,
                                                         d_head)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target_layer.attn_output = original


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]

TARGET_LAYER = 23
N_HEADS = 8


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[L23-heads] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Detect per-head dim at L23 — global layers have d_head=512
    # Inspect: attn_q.out_features = H * d_head_at_this_layer
    attn_q_23 = m.layers[TARGET_LAYER].attn_q
    q_out = attn_q_23.out_features
    d_head = q_out // N_HEADS
    print(f"[L23] attn_q out_features={q_out}, n_heads={N_HEADS} → d_head={d_head}")
    # attn_output's input is H * d_head → same dim
    attn_out_23 = m.layers[TARGET_LAYER].attn_output
    print(f"[L23] attn_output in_features={attn_out_23.in_features}")

    # Compute baselines
    print(f"\n=== baseline (no ablation) ===")
    baselines = []
    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(a * b)[0]
        logits = forward_with_head_ablation(m, token_ids, TARGET_LAYER, None,
                                              d_head)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        baselines.append((token_ids, correct_d, base_correct))
        argmax = int(logits[0, -1].argmax())
        print(f"  {a}×{b}={a*b}: argmax={tok.id_to_token.get(argmax, '?')!r}, "
              f"base_correct_logit={base_correct:.2f}")

    # Per-head sweep
    print(f"\n=== ablate each head of L23 across {len(PAIRS)} pairs ===\n")
    head_deltas = torch.zeros(N_HEADS, len(PAIRS))

    for H in range(N_HEADS):
        for j, (token_ids, correct_d, base_correct) in enumerate(baselines):
            logits = forward_with_head_ablation(m, token_ids, TARGET_LAYER, H,
                                                 d_head)
            abl_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
            head_deltas[H, j] = abl_correct - base_correct

    # Summary
    print(f"{'head':>5} {'mean_Δcorr':>12} {'std':>8} {'#hurts':>8}  (hurts = Δ < -0.5)")
    for H in range(N_HEADS):
        mu = head_deltas[H].mean().item()
        std = head_deltas[H].std().item()
        hurts = int((head_deltas[H] < -0.5).sum().item())
        marker = " ←" if mu < -1.0 or hurts >= 7 else ""
        print(f"H{H:>4} {mu:>+12.3f} {std:>8.3f}   {hurts:>2}/10{marker}")

    # Rank
    print(f"\n  Heads whose ablation MOST hurts arithmetic on average:")
    sorted_idx = head_deltas.mean(dim=1).argsort()
    for idx in sorted_idx[:5]:
        H = idx.item()
        mean_d = head_deltas[H].mean().item()
        hurts = int((head_deltas[H] < -0.5).sum().item())
        print(f"    H{H}: mean Δ = {mean_d:+.3f}, hurts in {hurts}/10")


if __name__ == "__main__":
    sys.exit(main())
