"""Round 48.1: neuron-level ablation at L24's FFN.

R47.4 null: L24 is diffuse at attention-head level (top head 4% of
-17.23 full-layer). R47 interpretation: composition is FFN-based.

Hypothesis: if the composition signal at L24 is concentrated in a
subset of FFN hidden neurons (d_ffn = 10240), ablating a ≤ 10%-width
chunk of neurons should reproduce ≥ 30% of the full-layer Δ (i.e.,
chunk Δ ≤ -5.17). If so, (a) substrate FFN install + (b) targeted
FFN edit are both viable — compact intervention at a specific neuron
range.

If diffuse at neuron level too (no chunk Δ < -3.0), multi-step
composition in L24 FFN is genuinely distributed across all 10240
neurons — neither FFN install nor ROME-style edit would work cleanly.
Pivot to Route C (compose cards).

Method: wrap layer.ffn_down with a shim that zeros a specific slice
of its input (the gated hidden activation). Scan 20 chunks of 512
neurons each.

Cost: 20 chunks × 10 triples + 10 baselines = 210 forwards ≈ 7 min.
"""

from __future__ import annotations

import math
import os
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")

TARGET_LAYER = 24
CHUNK_SIZE = 512     # 20 chunks × 512 = 10240 = d_ffn


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class FFNNeuronAblator:
    """Wraps ffn_down; zeros input neurons [lo:hi) before the matmul.
    Input to ffn_down is the gated hidden = gelu(gate) * up.
    Zeroing a slice = ablating those hidden neurons."""
    def __init__(self, inner, lo: int, hi: int):
        self.inner = inner
        self.lo = lo
        self.hi = hi
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        x_mod = x.clone()
        x_mod[..., self.lo:self.hi] = 0
        return self.inner(x_mod)


def forward_with_ffn_ablation(m, token_ids, ablate_layer,
                                ablate_lo=None, ablate_hi=None):
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
    original = target.ffn_down
    if ablate_lo is not None:
        target.ffn_down = FFNNeuronAblator(original, ablate_lo, ablate_hi)
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.ffn_down = original


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

TRIPLES = [
    (17, 23, 5),   # 396 '3'
    (47, 19, 23),  # 916 '9'
    (37, 14, 50),  # 568 '5'
    (13, 27, 8),   # 359 '3'
    (21, 38, 15),  # 813 '8'
    (11, 11, 10),  # 131 '1'
    (29, 17, 4),   # 497 '4'
    (32, 25, 7),   # 807 '8'
    (16, 31, 12),  # 508 '5'
    (34, 12, 5),   # 413 '4'
]


def build_prompt(a: int, b: int, c: int) -> str:
    return f"What is ({a} * {b}) + {c}? Answer: "


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print(f"[r48.1] loading substrate (target L{TARGET_LAYER})...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    d_ffn = m.config.d_ffn
    n_chunks = d_ffn // CHUNK_SIZE
    print(f"[L{TARGET_LAYER}] d_ffn={d_ffn}, chunk_size={CHUNK_SIZE}, "
          f"n_chunks={n_chunks}")

    # Baselines
    print(f"\n=== baseline (clean prompt format) ===")
    baselines = []
    n_correct = 0
    for a, b, c in TRIPLES:
        prompt = build_prompt(a, b, c)
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_ffn_ablation(m, token_ids, TARGET_LAYER)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        is_correct = argmax_tok.lstrip('▁') == correct_d
        if is_correct:
            n_correct += 1
        baselines.append((token_ids, correct_d, base_correct, answer))
        print(f"  {'✓' if is_correct else ' '} "
              f"{a}×{b}+{c}={answer}: argmax={argmax_tok!r}, "
              f"base_correct_logit={base_correct:.2f}")
    print(f"\n  baseline argmax correct: {n_correct}/10")

    # Chunk ablation sweep
    print(f"\n=== ablate {n_chunks} chunks × {len(TRIPLES)} triples ===")
    chunk_deltas = torch.zeros(n_chunks, len(TRIPLES))

    for chunk_idx in range(n_chunks):
        lo = chunk_idx * CHUNK_SIZE
        hi = (chunk_idx + 1) * CHUNK_SIZE
        for j, (token_ids, correct_d, base_correct, _) in enumerate(
                baselines):
            logits = forward_with_ffn_ablation(
                m, token_ids, TARGET_LAYER,
                ablate_lo=lo, ablate_hi=hi)
            abl_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
            chunk_deltas[chunk_idx, j] = abl_correct - base_correct
        if chunk_idx % 5 == 0:
            mean_d = chunk_deltas[chunk_idx].mean().item()
            print(f"  chunk {chunk_idx:>2} [{lo:>5}:{hi:>5}]: "
                  f"mean Δ = {mean_d:+.3f}")

    # Summary
    print(f"\n========== L{TARGET_LAYER} FFN CHUNK DELTAS ==========")
    print(f"{'chunk':>6} {'range':>14} {'mean_Δ':>10} {'std':>8} "
          f"{'#hurts':>8}  ({'hurts = Δ < -0.5':>18})")
    mean_all = chunk_deltas.mean(dim=1)
    for chunk_idx in range(n_chunks):
        lo = chunk_idx * CHUNK_SIZE
        hi = (chunk_idx + 1) * CHUNK_SIZE
        mu = mean_all[chunk_idx].item()
        std = chunk_deltas[chunk_idx].std().item()
        hurts = int((chunk_deltas[chunk_idx] < -0.5).sum().item())
        marker = " ←" if mu < -3.0 or hurts >= 8 else ""
        print(f"{chunk_idx:>6} [{lo:>5}:{hi:>5}] {mu:>+10.3f} "
              f"{std:>8.3f}   {hurts:>2}/10{marker}")

    # Rank chunks
    sorted_idx = mean_all.argsort()
    print(f"\n  Top 5 most-load-bearing chunks:")
    for idx in sorted_idx[:5]:
        ci = idx.item()
        lo = ci * CHUNK_SIZE
        hi = (ci + 1) * CHUNK_SIZE
        print(f"    chunk {ci} [{lo:>5}:{hi:>5}]: mean Δ = "
              f"{mean_all[ci]:+.3f}, hurts {int((chunk_deltas[ci] < -0.5).sum())}/10")

    # Concentration ratio
    full_layer_ref = -17.23
    top_chunk = mean_all.min().item()
    top3_sum = mean_all.sort().values[:3].sum().item()
    top5_sum = mean_all.sort().values[:5].sum().item()
    coverage_1 = top_chunk / full_layer_ref if full_layer_ref else 0
    coverage_3 = top3_sum / full_layer_ref if full_layer_ref else 0
    coverage_5 = top5_sum / full_layer_ref if full_layer_ref else 0
    print(f"\n  L{TARGET_LAYER} full-layer Δ (R47.3): {full_layer_ref:.2f}")
    print(f"  Top 1 chunk  (5% of neurons): {top_chunk:+.2f} "
          f"({coverage_1*100:.0f}%)")
    print(f"  Top 3 chunks (15% of neurons): {top3_sum:+.2f} "
          f"({coverage_3*100:.0f}%)")
    print(f"  Top 5 chunks (25% of neurons): {top5_sum:+.2f} "
          f"({coverage_5*100:.0f}%)")

    # Gate
    print(f"\n========== R48.1 GATE ==========")
    concentrated_1 = top_chunk < -3.0
    concentrated_3 = top3_sum < -8.0
    print(f"  Top 1 chunk Δ < -3.0:   "
          f"{'PASS' if concentrated_1 else 'FAIL'} "
          f"({top_chunk:+.2f})")
    print(f"  Top 3 chunks sum <-8.0: "
          f"{'PASS' if concentrated_3 else 'FAIL'} "
          f"({top3_sum:+.2f})")

    if concentrated_1 or concentrated_3:
        print(f"\n  ✓ L24 FFN composition signal has concentrated neurons.")
        print(f"    Routes (a) FFN install + (b) targeted edit are both")
        print(f"    viable. Next: R48.2 zoom into top chunks with finer")
        print(f"    resolution (64-neuron sub-chunks) to narrow further.")
    else:
        print(f"\n  ~ L24 FFN signal is diffuse across neurons too.")
        print(f"    Composition is genuinely distributed — neither (a)")
        print(f"    nor (b) will work with compact intervention.")
        print(f"    Recommend: pivot to Route C (compose compiled cards).")

    torch.save({
        "chunk_deltas": chunk_deltas.cpu(),
        "chunk_size": CHUNK_SIZE,
        "triples": TRIPLES,
        "target_layer": TARGET_LAYER,
    }, f"/tmp/r48_1_l{TARGET_LAYER}_ffn.pt")
    print(f"\n  saved: /tmp/r48_1_l{TARGET_LAYER}_ffn.pt")


if __name__ == "__main__":
    sys.exit(main())
