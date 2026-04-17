"""Round 18: Q/K vs V decomposition on L23 H1 and H4.

Ablation variants:
  - Ablate V only  (same Q/K pattern, but zero content copied)
  - Ablate Q only  (no attention, uniform softmax)
  - Ablate K only  (same, but Q still computes scores against zeros)
  - Ablate the whole head (Round 17 — baseline for comparison)

If the signal is in V: "V-only ablation" hurts as much as full ablation.
If the signal is in Q/K pattern: V alone doesn't hurt, but Q or K does.

For GQA: Gemma has 8 Q heads mapped to 2 KV groups (4 Q per KV).
Heads 0-3 share one K/V, 4-7 share another. So H1 and H4 are in
different KV groups. Interesting — the 2 arithmetic heads span both
KV groups.
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


class PartialHeadAblation:
    """Wraps attn_q, attn_k, or attn_v to zero one head's slice.
    Mimics MmapTq4Linear's call convention."""
    def __init__(self, inner, head_idx, d_head, n_heads):
        self.inner = inner
        self.head_idx = head_idx
        self.d_head = d_head
        self.n_heads = n_heads
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        # out shape: (..., n_heads * d_head). For Q: n_heads=8; for K,V in GQA: n_heads_kv=2
        # To zero the head_idx-th slice of this output
        lo = self.head_idx * self.d_head
        hi = (self.head_idx + 1) * self.d_head
        out = out.clone()
        out[..., lo:hi] = 0
        return out


def forward_with_partial_ablation(m, token_ids, ablate_layer, ablate_head,
                                    d_head, d_head_kv, n_heads_kv,
                                    mode):
    """mode ∈ {'none', 'Q', 'K', 'V', 'QKV'}"""
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
    saved_q = target.attn_q
    saved_k = target.attn_k
    saved_v = target.attn_v

    if mode != "none":
        # Compute KV group for this Q head. GQA: 8 Q → 2 KV groups
        # of size 4. Q head idx H → KV group H // (n_heads_q / n_heads_kv)
        kv_group = ablate_head // (cfg.n_heads_q // n_heads_kv)
        if mode in ("Q", "QKV"):
            target.attn_q = PartialHeadAblation(saved_q, ablate_head, d_head, cfg.n_heads_q)
        if mode in ("K", "QKV"):
            target.attn_k = PartialHeadAblation(saved_k, kv_group, d_head_kv, n_heads_kv)
        if mode in ("V", "QKV"):
            target.attn_v = PartialHeadAblation(saved_v, kv_group, d_head_kv, n_heads_kv)

    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_q = saved_q
        target.attn_k = saved_k
        target.attn_v = saved_v


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

PAIRS = [
    (17, 23), (34, 12), (47, 19), (13, 27), (21, 38),
    (45, 15), (11, 11), (29, 17), (32, 25), (16, 31),
]

TARGET_LAYER = 23
TARGET_HEADS = [1, 4]


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[qkv-decomp] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    L = TARGET_LAYER
    cfg = m.config
    attn_q_L = m.layers[L].attn_q
    attn_v_L = m.layers[L].attn_v
    q_out = attn_q_L.out_features
    v_out = attn_v_L.out_features
    d_head = q_out // cfg.n_heads_q
    n_heads_kv = cfg.n_heads_kv
    d_head_kv = v_out // n_heads_kv
    print(f"[L{L}] Q head_dim={d_head}  KV n_heads_kv={n_heads_kv} d_head_kv={d_head_kv}")
    print(f"  GQA grouping: Q head 0-3 → KV group 0, Q head 4-7 → KV group 1")

    # Baselines
    baselines = {}
    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(a * b)[0]
        logits = forward_with_partial_ablation(
            m, token_ids, L, -1, d_head, d_head_kv, n_heads_kv, "none")
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        baselines[(a, b)] = (token_ids, correct_d, base_correct)

    # For each head in target heads, for each mode, measure Δ mean
    print(f"\n{'head':>5} {'mode':>6} {'mean_Δcorr':>12} {'std':>8}")
    results = {}
    for H in TARGET_HEADS:
        for mode in ["Q", "K", "V", "QKV"]:
            deltas = []
            for (a, b), (token_ids, correct_d, base) in baselines.items():
                logits = forward_with_partial_ablation(
                    m, token_ids, L, H, d_head, d_head_kv, n_heads_kv, mode)
                abl = logits[0, -1, DIGIT_IDS[correct_d]].item()
                deltas.append(abl - base)
            dt = torch.tensor(deltas)
            results[(H, mode)] = dt
            print(f"H{H:>4} {mode:>6} {dt.mean().item():>+12.3f} {dt.std().item():>8.3f}")

    print(f"\n  Interpretation:")
    print(f"  If V-only ablation ≈ QKV ablation, the arithmetic signal is in V (the copied content).")
    print(f"  If Q or K alone ≈ QKV, the signal is in the attention PATTERN (which positions attend to which).")
    for H in TARGET_HEADS:
        v = results[(H, "V")].mean().item()
        q = results[(H, "Q")].mean().item()
        k = results[(H, "K")].mean().item()
        all_ = results[(H, "QKV")].mean().item()
        print(f"\n  H{H}:")
        print(f"    V-only:  {v:+.2f}  (content)")
        print(f"    Q-only:  {q:+.2f}  (query pattern)")
        print(f"    K-only:  {k:+.2f}  (key pattern)")
        print(f"    QKV all: {all_:+.2f}  (reference)")
        if abs(v) > abs(q) * 1.5 and abs(v) > abs(k) * 1.5:
            print(f"    → Content-dominant (arithmetic signal rides in V)")
        elif abs(q) + abs(k) > abs(v) * 1.5:
            print(f"    → Pattern-dominant (arithmetic signal in Q/K attention pattern)")
        else:
            print(f"    → Mixed (both V and Q/K contribute)")


if __name__ == "__main__":
    sys.exit(main())
