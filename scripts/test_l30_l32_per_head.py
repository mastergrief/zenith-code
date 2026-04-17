"""Round 25: per-head ablation at L30, L31, L32 — find the fd-writer heads.

R24 showed L30 attn (+0.148), L31 FFN (+0.185), L32 FFN (+0.111) drive fd.
For L30 attn: which of its 8 Q heads carries the arithmetic gather?
For L31 attn (p_ones writer): same question.

Mechanism: adapt R17's HeadAblatingWrapper. For each (layer, head),
zero the head's slice of attn_output's input, measure Δ(fd logit) and
Δ(p_ones logit) relative to the final token on the same 10 arithmetic
pairs from R16-R20. Plus measure Δ via probing on a larger sample if
a head emerges clearly.

This tells us:
  - If one head at L30 dominates fd-gather → that head's 2.6M V-params
    plus L31-L32 FFN neurons are the compilable circuit.
  - If multiple heads contribute → the circuit is distributed,
    shifting R26 to a different strategy.

Cost: 8 heads × 3 layers × 10 pairs = 240 logit measurements + 8×3
forward passes for baselines. Fast (~3 min).
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}
PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]
TARGET_LAYERS = [30, 31, 32]
N_HEADS = 8


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class HeadAblatingWrapper:
    """Zero a specific head's slice of attn_output's input."""
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

    target_layer = m.layers[ablate_layer]
    original = target_layer.attn_output
    if ablate_head is not None:
        target_layer.attn_output = HeadAblatingWrapper(original, ablate_head, d_head)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target_layer.attn_output = original


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l30-l32-heads] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Determine d_head per layer — L30, L32 should be SWA (d_head=256),
    # L29, L35, L41 global (d_head=512). But layer_idx 30-32 alternate...
    # Actually L23 was global; the global layers are 5,11,17,23,29,35,41.
    # So L30,31,32 are all SWA (d_head=256).
    for L in TARGET_LAYERS:
        q_out = m.layers[L].attn_q.out_features
        d_h = q_out // N_HEADS
        print(f"  L{L}: attn_q out={q_out} → d_head={d_h} ({'global' if d_h > 256 else 'SWA'})")

    # Build baselines per pair
    print(f"\n=== baselines (first-digit of product) ===")
    baselines = []
    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        fd_correct = str(a * b)[0]
        p_ones_correct = str((a * b) % 10)
        logits = forward_with_head_ablation(m, token_ids, TARGET_LAYERS[0], None, 256)
        base_fd = logits[0, -1, DIGIT_IDS[fd_correct]].item()
        base_p_ones = logits[0, -1, DIGIT_IDS[p_ones_correct]].item()
        baselines.append((token_ids, fd_correct, base_fd, p_ones_correct, base_p_ones))
        print(f"  {a}×{b}={a*b}  fd='{fd_correct}' logit={base_fd:+.2f}  "
              f"p_ones='{p_ones_correct}' logit={base_p_ones:+.2f}")

    # Per-head ablation
    print(f"\n=== per-head ablation (Δ_fd = abl - base, negative = hurts fd) ===\n")
    for L in TARGET_LAYERS:
        d_head = m.layers[L].attn_q.out_features // N_HEADS
        print(f"\n--- L{L} (d_head={d_head}) ---")
        print(f"  {'head':>5} {'mean_Δfd':>10} {'std':>8} {'#hurts':>8}  "
              f"{'mean_Δp_ones':>14} {'std':>8}")
        fd_deltas = torch.zeros(N_HEADS, len(PAIRS))
        po_deltas = torch.zeros(N_HEADS, len(PAIRS))
        for H in range(N_HEADS):
            for j, (tids, fdc, bfd, poc, bpo) in enumerate(baselines):
                logits = forward_with_head_ablation(m, tids, L, H, d_head)
                fd_deltas[H, j] = logits[0, -1, DIGIT_IDS[fdc]].item() - bfd
                po_deltas[H, j] = logits[0, -1, DIGIT_IDS[poc]].item() - bpo
            mu_fd = fd_deltas[H].mean().item()
            sd_fd = fd_deltas[H].std().item()
            hurts = int((fd_deltas[H] < -0.5).sum().item())
            mu_po = po_deltas[H].mean().item()
            sd_po = po_deltas[H].std().item()
            marker = " ←" if mu_fd < -1.0 or hurts >= 7 else ""
            po_marker = " ★" if mu_po < -1.0 else ""
            print(f"  H{H:>4} {mu_fd:>+10.3f} {sd_fd:>8.3f} {hurts:>2}/10{marker}  "
                  f"{mu_po:>+14.3f} {sd_po:>8.3f}{po_marker}")

        # Top-hurting heads for each metric
        sorted_fd = fd_deltas.mean(dim=1).argsort()
        sorted_po = po_deltas.mean(dim=1).argsort()
        print(f"  fd top-3 hurters:      {[(int(i), round(fd_deltas[i].mean().item(), 3)) for i in sorted_fd[:3]]}")
        print(f"  p_ones top-3 hurters:  {[(int(i), round(po_deltas[i].mean().item(), 3)) for i in sorted_po[:3]]}")


if __name__ == "__main__":
    sys.exit(main())
