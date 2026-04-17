"""Round 20: per-sub-head ablation on L23 H4 (the arithmetic-V carrier).

R17 localized arithmetic to L23, H1+H4. R18 showed H4's V carries 93%.
R20 narrows further: within H4's 512-d output, is the signal concentrated
in a few d_head=2 sub-head pairs, or smeared across all 256?

Ablation: zero H4's attention-output slice [2i : 2(i+1)] for i in 0..255
before the attn_output matmul. Measure Δ(correct-digit logit) across the
same 10 arithmetic pairs from R16/R17/R18.

Expected outcome: 2-8 sub-heads carry bulk of signal → narrows target
from 2.6M params to ~40K params. Concrete SAE input region for R22.

Cost: 256 sub-heads × 10 pairs = 2560 forward passes ≈ 20-30 min on RTX 4070.
"""

from __future__ import annotations

import math
import os
import sys
import time

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class SubHeadAblatingWrapper:
    """Zeros a 2-dim sub-head slice of H4's attn_output input.

    H4 occupies columns [H4_base : H4_base + d_head] of attn_output's
    input. Sub-head i covers [H4_base + 2i : H4_base + 2(i+1)].
    """
    def __init__(self, inner, sub_lo, sub_hi):
        self.inner = inner
        self.sub_lo = sub_lo
        self.sub_hi = sub_hi
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        x_mod = x.clone()
        x_mod[..., self.sub_lo:self.sub_hi] = 0
        return self.inner(x_mod)


def forward_with_subhead_ablation(m, token_ids, ablate_layer, sub_lo, sub_hi):
    """Run forward with a specific 2-dim slice of attn_output input zeroed.
    Pass sub_lo=None for no ablation (baseline)."""
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
    if sub_lo is not None:
        target_layer.attn_output = SubHeadAblatingWrapper(original, sub_lo, sub_hi)
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
TARGET_HEAD = 4
N_HEADS_Q = 8


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[L23-H4-subhead] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Geometry
    attn_q_23 = m.layers[TARGET_LAYER].attn_q
    q_out = attn_q_23.out_features
    d_head = q_out // N_HEADS_Q  # expect 512 at global L23
    attn_out_23 = m.layers[TARGET_LAYER].attn_output
    print(f"[L{TARGET_LAYER}] attn_q out_features={q_out}, n_heads={N_HEADS_Q} → d_head={d_head}")
    print(f"[L{TARGET_LAYER}] attn_output in_features={attn_out_23.in_features}")
    assert attn_out_23.in_features == N_HEADS_Q * d_head, \
        f"expected attn_output input = {N_HEADS_Q * d_head}, got {attn_out_23.in_features}"

    d_sub = 2
    n_sub = d_head // d_sub  # 256 sub-heads per head (substrate convention)
    h4_base = TARGET_HEAD * d_head
    print(f"[L{TARGET_LAYER} H{TARGET_HEAD}] d_head={d_head} → {n_sub} sub-heads of d_head=2")
    print(f"  H{TARGET_HEAD} input slice: [{h4_base}, {h4_base + d_head})")

    # Baselines
    print(f"\n=== baseline (no ablation) ===")
    baselines = []
    for a, b in PAIRS:
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        correct_d = str(a * b)[0]
        logits = forward_with_subhead_ablation(m, token_ids, TARGET_LAYER, None, None)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        baselines.append((token_ids, correct_d, base_correct))
        argmax = int(logits[0, -1].argmax())
        print(f"  {a}×{b}={a*b}: correct='{correct_d}' base_logit={base_correct:+.2f} "
              f"argmax={tok.id_to_token.get(argmax, '?')!r}")

    # Per-sub-head sweep
    total = n_sub * len(PAIRS)
    print(f"\n=== ablating {n_sub} sub-heads of H{TARGET_HEAD} × {len(PAIRS)} pairs "
          f"= {total} forwards ===\n")
    sh_deltas = torch.zeros(n_sub, len(PAIRS))

    t_start = time.time()
    for i in range(n_sub):
        sub_lo = h4_base + i * d_sub
        sub_hi = h4_base + (i + 1) * d_sub
        for j, (token_ids, correct_d, base_correct) in enumerate(baselines):
            logits = forward_with_subhead_ablation(m, token_ids, TARGET_LAYER,
                                                     sub_lo, sub_hi)
            abl_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
            sh_deltas[i, j] = abl_correct - base_correct
        if (i + 1) % 16 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) * len(PAIRS) / elapsed
            eta = (total - (i + 1) * len(PAIRS)) / rate
            print(f"  [{i+1:>3}/{n_sub}] rate={rate:.1f} fwd/s ETA={eta:.0f}s")

    # Summary
    elapsed = time.time() - t_start
    print(f"\n[done] {total} forwards in {elapsed:.1f}s = {total/elapsed:.1f} fwd/s\n")

    means = sh_deltas.mean(dim=1)
    stds = sh_deltas.std(dim=1)
    hurts = (sh_deltas < -0.5).sum(dim=1)

    # Distribution
    n_real_signal = int((means < -1.0).sum().item())
    n_mild = int(((means >= -1.0) & (means < -0.3)).sum().item())
    n_silent = int((means.abs() < 0.3).sum().item())
    print(f"Distribution across {n_sub} sub-heads:")
    print(f"  mean Δ < -1.0 (real signal):     {n_real_signal}")
    print(f"  -1.0 ≤ mean Δ < -0.3 (mild):      {n_mild}")
    print(f"  |mean Δ| < 0.3    (silent):       {n_silent}")
    print(f"  other:                             {n_sub - n_real_signal - n_mild - n_silent}")

    # Top 20 most damaging sub-heads
    print(f"\nTop 20 sub-heads whose ablation hurts arithmetic MOST:\n")
    print(f"  {'rank':>4} {'sub':>4} {'slice':>14} {'mean_Δ':>10} {'std':>8} {'#hurts':>8}")
    sorted_idx = means.argsort()
    for rank in range(min(20, n_sub)):
        i = int(sorted_idx[rank].item())
        sub_lo = h4_base + i * d_sub
        sub_hi = h4_base + (i + 1) * d_sub
        mu = means[i].item()
        sd = stds[i].item()
        h = int(hurts[i].item())
        marker = " ←" if mu < -1.0 else ""
        print(f"  {rank+1:>4} {i:>4}  [{sub_lo:>4},{sub_hi:>4})"
              f" {mu:>+10.3f} {sd:>8.3f}  {h:>2}/10{marker}")

    # Coverage: cumulative fraction of total |mean_Δ| from top-K sub-heads
    total_damage = -means[means < 0].sum().item()  # positive number
    if total_damage > 0:
        print(f"\nCumulative coverage (how many sub-heads carry what fraction of damage):")
        cum = 0.0
        for k in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
            if k > n_sub:
                break
            top_k = means[sorted_idx[:k]]
            top_k_damage = -top_k[top_k < 0].sum().item()
            frac = top_k_damage / total_damage if total_damage > 0 else 0
            print(f"  top-{k:>3}: {frac:>6.1%} of total negative Δ "
                  f"({top_k_damage:+.2f} / {-total_damage:+.2f})")

    # Save raw data for inspection
    save_path = "/tmp/r20_subhead_deltas.pt"
    torch.save({
        "deltas": sh_deltas,
        "baselines_correct": [b[2] for b in baselines],
        "pairs": PAIRS,
        "target_layer": TARGET_LAYER,
        "target_head": TARGET_HEAD,
        "h4_base": h4_base,
        "d_head": d_head,
        "n_sub": n_sub,
    }, save_path)
    print(f"\n[saved] raw deltas → {save_path}")


if __name__ == "__main__":
    sys.exit(main())
