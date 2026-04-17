"""Round 49.5: L24 joint-output rank — full-layer compilation target test.

R49.4 showed L24's composition signal is distributed across 3
pathways (attn/ffn/per-layer-embd) with non-additive interactions.
No single pathway is a compact compile target.

R49.5 tests the last cheap hypothesis: maybe the COMBINED L24
contribution is low-rank, even though it's produced by 3 interacting
pathways. If so, we can compile L24 as a single rank-K linear map
replacing the whole layer.

L24's residual contribution = h_after_L24 - h_before_L24.
That's what downstream layers see. If this is rank < 50 across
multi-step prompts, the joint effect IS compressible.

Method:
  1. Calibration: 100 multi-step prompts. Capture L24 contribution
     at last position.
  2. SVD, measure rank@90/95%.
  3. Held-out: replace L24 contribution with rank-K projection at
     inference. Measure held-out accuracy at K = 1, 5, 10, 30,
     100, 500.
  4. Task-relevant rank = smallest K preserving ≥80% baseline.

Gates:
  - PCA rank@90% < 50 → low-rank contributor AT VARIANCE LEVEL
  - Task-relevant rank < 50 at 80% preservation → FULL L24 COMPILABLE
    as a rank-K linear map
  - Task-relevant rank > 500 → L24 is fundamentally high-rank, no
    compact reverse-engineering possible

Cost: 100 calib + 20 baseline + 6 × 20 projection = 240 forwards
≈ 7 min.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 24


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


def forward_with_capture(m, token_ids):
    """Run full forward, capture L24's residual contribution at last pos."""
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

    contribution = None
    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER:
                contribution = (h[0, -1, :] - h_before[0, -1, :]).detach()
    logits = project_to_logits(m, h)
    return logits, contribution


def forward_with_rank_k(m, token_ids, mean, V_K):
    """Forward with L24's last-pos residual contribution replaced by
    its rank-K projection around mean."""
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

    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER:
                contribution = h[0, -1, :] - h_before[0, -1, :]
                dev = contribution - mean
                proj_dev = dev @ V_K.T @ V_K
                # Replace L24's contribution at position -1 with projection
                h[0, -1, :] = h_before[0, -1, :] + mean + proj_dev
    return project_to_logits(m, h)


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}


def sample_multi_step(rng, n):
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        c = rng.randint(1, 99)
        prompt = f"What is ({a} * {b}) + {c}? Answer: "
        out.append((prompt, a * b + c))
    return out


def effective_rank(A, thr=0.9):
    S = torch.linalg.svdvals(A.float())
    energy = (S ** 2).cumsum(0)
    ratio = energy / energy[-1]
    return int((ratio >= thr).nonzero()[0].item()) + 1, S


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r49.5] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Calibration
    rng = random.Random(42)
    calib_prompts = sample_multi_step(rng, 100)
    print(f"\n=== calibration: capture L{TARGET_LAYER} total contribution "
          f"({len(calib_prompts)} prompts) ===")
    contribs = []
    for i, (prompt, _) in enumerate(calib_prompts):
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        _, c = forward_with_capture(m, token_ids)
        contribs.append(c.cpu())
        if i % 25 == 0:
            print(f"  {i+1}/{len(calib_prompts)}")

    A = torch.stack(contribs).float().cuda()  # (100, 2560)
    mean = A.mean(dim=0)
    A_c = A - mean
    U, S, Vt = torch.linalg.svd(A_c, full_matrices=False)
    rank90, _ = effective_rank(A_c, 0.90)
    rank95, _ = effective_rank(A_c, 0.95)
    rank99, _ = effective_rank(A_c, 0.99)
    print(f"\n  L{TARGET_LAYER} joint-contribution rank analysis:")
    print(f"    rank@90% variance: {rank90}")
    print(f"    rank@95% variance: {rank95}")
    print(f"    rank@99% variance: {rank99}")
    print(f"    top-5 singular values: {S[:5].tolist()}")

    # Held-out
    rng_h = random.Random(999)
    held = sample_multi_step(rng_h, 20)

    # Baselines
    print(f"\n=== baseline (20 held-out, no intervention) ===")
    baselines = []
    for prompt, answer in held:
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits, _ = forward_with_capture(m, token_ids)
        correct_d = str(answer)[0]
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        argmax = int(logits[0, -1].argmax())
        argmax_tok = tok.id_to_token.get(argmax, '?')
        is_correct = argmax_tok.lstrip('▁') == correct_d
        baselines.append({
            "prompt": prompt, "answer": answer, "correct_d": correct_d,
            "token_ids": token_ids, "base_correct": base_correct,
            "is_correct": is_correct,
        })
    base_acc = sum(b["is_correct"] for b in baselines)
    base_mean_logit = sum(b["base_correct"] for b in baselines) / len(baselines)
    print(f"  baseline argmax correct: {base_acc}/{len(baselines)}")
    print(f"  baseline mean correct-digit logit: {base_mean_logit:.2f}")

    # Rank-K projections
    print(f"\n=== RANK-K PROJECTION (L{TARGET_LAYER} joint contribution) ===")
    print(f"  K    n_correct  mean_logit   Δ_logit   acc_preserved")
    results = []
    for K in [1, 5, 10, 30, 100, 500]:
        if K > Vt.shape[0]:
            continue
        V_K = Vt[:K].cuda()
        n_correct_K = 0
        sum_logit = 0.0
        for b in baselines:
            logits = forward_with_rank_k(m, b["token_ids"], mean, V_K)
            corr_logit = logits[0, -1, DIGIT_IDS[b["correct_d"]]].item()
            argmax = int(logits[0, -1].argmax())
            argmax_tok = tok.id_to_token.get(argmax, '?')
            if argmax_tok.lstrip('▁') == b["correct_d"]:
                n_correct_K += 1
            sum_logit += corr_logit
        mean_logit = sum_logit / len(baselines)
        d_logit = mean_logit - base_mean_logit
        acc_pres = n_correct_K / max(base_acc, 1)
        results.append((K, n_correct_K, mean_logit, d_logit, acc_pres))
        print(f"  K={K:>4}  {n_correct_K:>3}/{len(baselines):<3}   "
              f"{mean_logit:>8.2f}   {d_logit:>+6.2f}   {acc_pres*100:>5.1f}%")

    # Gate
    print(f"\n========== R49.5 GATE ==========")
    task_rank = None
    for K, n_correct, _, _, acc_pres in results:
        if acc_pres >= 0.80:
            task_rank = K
            break
    if task_rank is not None:
        print(f"  task-relevant rank: ≤ {task_rank}")
        if task_rank <= 50:
            print(f"  ✓ L24 joint output is LOW-RANK and task-preservable")
            print(f"    Compile target: rank-{task_rank} linear map")
            print(f"    x → mean + (x - mean) @ V_K^T @ V_K")
            print(f"    Install as a full-layer replacement at L24.")
        else:
            print(f"  ~ Rank is {task_rank}, not extremely compact but workable")
    else:
        print(f"  ✗ No K reaches 80% preservation. L24 joint output is")
        print(f"    fundamentally high-rank. Reverse-engineering fails.")

    torch.save({
        "calib_contribs": A.cpu(),
        "mean": mean.cpu(),
        "Vt": Vt.cpu(),
        "rank90": rank90, "rank95": rank95, "rank99": rank99,
        "results": results, "base_acc": base_acc,
    }, "/tmp/r49_5_joint.pt")
    print(f"\n  saved: /tmp/r49_5_joint.pt")


if __name__ == "__main__":
    sys.exit(main())
