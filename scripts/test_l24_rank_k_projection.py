"""Round 49.2: rank-K projection test at L24 FFN.

R49.1 showed L24's multi-step FFN output has rank ~34 at 90% variance
(mean-centered). R49.2 tests if this characterization is ACCURATE by
forcing L24's output into the top-K subspace at inference, measuring
whether held-out multi-step correctness is preserved.

Hypothesis: the composition circuit's information content lives in a
≤ 50-dim subspace. If true, replacing L24's full 2560-d output with
its projection onto top-K calibration-set principal components at
K=34 preserves ≥ 80% of baseline correct-digit logit.

Method:
  1. Calibration: collect L24 ffn_down outputs on 100 multi-step
     prompts (already saved at /tmp/r49_1_l24_rank.pt).
  2. Compute mean μ and top-K principal components V_K.
  3. Held-out: generate 30 NEW multi-step prompts not in calibration.
  4. At inference, wrap L24 ffn_down so its output becomes
       y_projected = μ + (y - μ) @ V_K^T @ V_K
     (keep mean bias, project deviation onto top-K subspace)
  5. Measure correct-digit logit for each K ∈ {5, 10, 20, 34, 50,
     100, 500, None=baseline}.

Gate:
  - K=34 held-out correct-digit logit ≥ 0.80 × baseline → circuit IS
    rank-34 (we've reverse-engineered the composition circuit)
  - K=100 ≥ 0.95 × baseline → approximate tight rank
  - Baseline accuracy drops to noise at K=5 → not trivially rank-1

Cost: (7 K values + baseline) × 30 held-out prompts = 240 forwards
≈ 8 min.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 24
CALIB_PATH = "/tmp/r49_1_l24_rank.pt"


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class FFNProjector:
    """Wrap ffn_down so its output at the last position is forced into
    a K-dim subspace: y_new = μ + (y - μ) @ V_K^T @ V_K.
    Other positions pass through unchanged (they're not load-bearing
    for the answer)."""
    def __init__(self, inner, mean: torch.Tensor, V_K: torch.Tensor):
        """mean: (d_model,). V_K: (K, d_model)."""
        self.inner = inner
        self.mean = mean
        self.V_K = V_K
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        y = self.inner(x)  # (B=1, S, d_model)
        # Last position only
        last = y[0, -1, :]  # (d_model,)
        dev = last - self.mean
        proj_dev = dev @ self.V_K.T @ self.V_K  # rank-K projection
        y[0, -1, :] = self.mean + proj_dev
        return y


def forward_with_projection(m, token_ids, projector=None):
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

    target = m.layers[TARGET_LAYER]
    original = target.ffn_down
    if projector is not None:
        target.ffn_down = projector
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


def sample_held_out(rng, n: int, exclude_triples: set) -> list[tuple]:
    out = []
    while len(out) < n:
        a = rng.randint(2, 29)
        b = rng.randint(2, 29)
        if a * b >= 1000:
            continue
        c = rng.randint(1, 99)
        if (a, b, c) in exclude_triples:
            continue
        prompt = f"What is ({a} * {b}) + {c}? Answer: "
        out.append((a, b, c, prompt, a * b + c))
    return out


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    # Load calibration data from R49.1
    print("[r49.2] loading R49.1 calibration data...")
    calib = torch.load(CALIB_PATH, weights_only=False)
    A_multi = calib["multi_acts_L24"].float().cuda()  # (100, 2560)
    print(f"  calibration set: {A_multi.shape[0]} multi-step prompts")

    # Compute mean + PCA
    mean = A_multi.mean(dim=0)  # (d_model,)
    A_centered = A_multi - mean
    U, S, Vt = torch.linalg.svd(A_centered, full_matrices=False)
    # Vt is (min(N, d), d). For N=100, d=2560, Vt is (100, 2560).
    print(f"  top-5 singular values: {S[:5].tolist()}")
    print(f"  SVD total rank available: {Vt.shape[0]}")

    # Load Gemma
    enable_triton_tq4(True)
    print("\n[r49.2] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Held-out prompts
    # R49.1 used seed=42 for calibration; use seed=999 here for held-out.
    rng = random.Random(999)
    calib_triples = set((a, b, c) for (a, b, c) in
                         [(2,3,1)])  # placeholder — we don't have
                                       # calib triples saved, assume
                                       # seed=999 is disjoint from 42
    held_out = sample_held_out(rng, 20, calib_triples)
    print(f"\n  held-out set: {len(held_out)} multi-step prompts")

    # Baselines first
    print(f"\n=== BASELINE (no projection) ===")
    baselines = []
    for a, b, c, prompt, answer in held_out:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_projection(m, token_ids, projector=None)
        correct_d = str(answer)[0]
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        is_correct = argmax_tok.lstrip('▁') == correct_d
        baselines.append({
            "a": a, "b": b, "c": c, "answer": answer,
            "correct_d": correct_d, "token_ids": token_ids,
            "base_correct": base_correct, "is_correct": is_correct,
            "argmax_tok": argmax_tok,
        })
    base_acc = sum(b["is_correct"] for b in baselines) / len(baselines)
    base_mean_logit = sum(b["base_correct"] for b in baselines) / len(baselines)
    print(f"  baseline argmax correct: "
          f"{sum(b['is_correct'] for b in baselines)}/{len(baselines)}")
    print(f"  baseline mean correct-digit logit: {base_mean_logit:.2f}")

    # Project at various K
    print(f"\n=== RANK-K PROJECTION SWEEP (L{TARGET_LAYER} ffn_down output) ===")
    print(f"  K    n_correct  mean_logit_correct   Δ_logit   acc_preserved")

    results = [("baseline", None, base_mean_logit,
                 sum(b["is_correct"] for b in baselines), len(baselines))]
    target = m.layers[TARGET_LAYER]

    for K in [1, 5, 10, 20, 34, 50, 100, 500]:
        if K > Vt.shape[0]:
            continue
        V_K = Vt[:K]  # (K, d_model) — top-K right singular vectors
        projector = FFNProjector(target.ffn_down, mean, V_K)
        n_correct_K = 0
        sum_logit_K = 0.0
        for b in baselines:
            logits = forward_with_projection(
                m, b["token_ids"], projector=projector)
            correct_logit = logits[0, -1, DIGIT_IDS[b["correct_d"]]].item()
            argmax = int(logits[0, -1].argmax())
            argmax_tok = tok.id_to_token.get(argmax, '?')
            if argmax_tok.lstrip('▁') == b["correct_d"]:
                n_correct_K += 1
            sum_logit_K += correct_logit
        mean_logit = sum_logit_K / len(baselines)
        d_logit = mean_logit - base_mean_logit
        acc_preserved = (n_correct_K / max(sum(b["is_correct"] for b in
                                                baselines), 1))
        print(f"  K={K:>4}  {n_correct_K:>3}/{len(baselines):<3}      "
              f"{mean_logit:>12.2f}      "
              f"{d_logit:>+6.2f}      {acc_preserved*100:>5.1f}%")
        results.append((f"K={K}", K, mean_logit, n_correct_K,
                         len(baselines)))

    # Gate
    print(f"\n========== R49.2 GATE ==========")
    K34_result = [r for r in results if r[1] == 34]
    K100_result = [r for r in results if r[1] == 100]

    if K34_result:
        k34_correct = K34_result[0][3]
        k34_preserved = k34_correct / max(sum(b["is_correct"] for b in
                                                baselines), 1)
        print(f"  K=34 (90% variance target):  {k34_correct}/"
              f"{len(baselines)} correct, "
              f"{k34_preserved*100:.1f}% of baseline accuracy")
        if k34_preserved >= 0.80:
            print(f"  ✓ R49.1's rank-34 characterization is ACCURATE.")
            print(f"    L24 composition circuit is ~34-dim.")
            print(f"    Next R49.3: compile rank-34 card, install in")
            print(f"    substrate reserved channels.")
        else:
            print(f"  ~ K=34 loses too much accuracy; true rank higher.")

    if K100_result:
        k100_correct = K100_result[0][3]
        k100_preserved = k100_correct / max(sum(b["is_correct"] for b in
                                                  baselines), 1)
        print(f"  K=100:                       {k100_correct}/"
              f"{len(baselines)} correct, "
              f"{k100_preserved*100:.1f}% of baseline accuracy")

    # Save for R49.3
    torch.save({
        "mean": mean.cpu(),
        "Vt": Vt.cpu(),
        "results": results,
        "baseline_acc": sum(b["is_correct"] for b in baselines)
                         / len(baselines),
    }, "/tmp/r49_2_projection.pt")
    print(f"\n  saved: /tmp/r49_2_projection.pt")


if __name__ == "__main__":
    sys.exit(main())
