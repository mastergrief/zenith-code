"""R53.36 — audit R51/R52 tier-3 distillation install boundary.

Hypothesis: R51/R52 tier-3 nulls (session 34) may be csv-style
measurement artifacts — the dual-gate eval measures prefix-match
after student-install but doesn't directly verify that the student
is reproducing L24's contribution as trained. A ~1% install-math
error could make a perfectly-trained student look broken.

Three diagnostic questions:

1. **Student training fidelity**: on a held-out prompt, does the
   student's output approximate Gemma's actual L24 contribution?
   - GT: `contribution = L24(h_before) - h_before`
   - Pred: `student(h_before)`
   - Measure cosine + L2 + per-channel magnitude. If cosine > 0.9
     the student IS reproducing L24; install boundary is suspect.
   - If cosine < 0.3 the student is poorly trained; that's the
     actual failure, no csv-style artifact.

2. **Install boundary correctness**: does Gemma under install
   produce `h_before + student(h_before)` at L24 output?
   - Run Gemma unmodified → capture L24 output `h_native`
   - Run Gemma installed → capture L24 output `h_installed`
   - Verify `h_installed == h_before + student(h_before)`
     to numerical precision. If diff > 1e-4, install math is wrong.

3. **Sequence-length scaling**: capture at multiple S values
   (short, medium, long). If fidelity degrades with S, positional
   encoding is probably misaligned.

Daemon-only:
  bin/gemma-run scripts/r53_36_audit_r51_install.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch


TARGET_LAYER = 24
R51_CKPT = "calm/llm_computer/r51/checkpoints/r51_student.pt"
R52_CKPT = "calm/llm_computer/r51/checkpoints/r52_student_kl.pt"


# Held-out prompts spanning domains — match R51.1 capture taxonomy
PROMPTS = [
    # multi-step arithmetic (tier-3's central target)
    ("multi", "What is 17 times 23 plus 5? Answer: "),
    # single-op arithmetic (training dist)
    ("single", "What is 17 times 23? Answer: "),
    # factual recall
    ("factual", "The capital of France is "),
    # code
    ("code", "def fibonacci(n):\n    if n <= 1:\n"),
]


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def capture_l24_gt(m, tok, prompt: str):
    """Run Gemma unmodified, capture L24's h_before and contribution
    at every position."""
    from calm.llm_computer.gemma_substrate import KVCache
    cfg = m.config
    ids = tok.encode(prompt)
    token_ids = torch.tensor([ids], device="cuda")
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
                h_proj = _rms_norm(h_proj, m.per_layer_proj_norm_w,
                                   cfg.rms_norm_eps)
            pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
        m._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

    h_before = None
    h_after = None
    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone().detach()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER:
                h_after = h.clone().detach()
                break

    contribution = (h_after - h_before).detach()
    return h_before, contribution, h_after


def compare_distributions(gt: torch.Tensor, pred: torch.Tensor, label: str):
    """Cosine per position + L2 norms + per-channel correlation."""
    gt_f = gt[0].float().cpu()       # [S, d_model]
    pr_f = pred[0].float().cpu()
    S = gt_f.shape[0]

    gt_norm = gt_f.norm(dim=-1)
    pr_norm = pr_f.norm(dim=-1)

    cos = (gt_f * pr_f).sum(dim=-1) / (
        gt_norm.clamp_min(1e-8) * pr_norm.clamp_min(1e-8))

    l2_diff = (gt_f - pr_f).norm(dim=-1)

    # Per-channel correlation (over S positions)
    gt_centered = gt_f - gt_f.mean(dim=0, keepdim=True)
    pr_centered = pr_f - pr_f.mean(dim=0, keepdim=True)
    gt_std = gt_f.std(dim=0).clamp_min(1e-8)
    pr_std = pr_f.std(dim=0).clamp_min(1e-8)
    if S > 1:
        ch_corr = (gt_centered * pr_centered).mean(dim=0) / (gt_std * pr_std)
        ch_corr_mean = ch_corr.mean().item()
    else:
        ch_corr_mean = float("nan")

    print(f"  [{label}] S={S}", flush=True)
    print(f"    cosine per-pos: mean={cos.mean():.4f}  "
          f"min={cos.min():.4f}  max={cos.max():.4f}", flush=True)
    print(f"    L2 GT norm:     mean={gt_norm.mean():.4f}  "
          f"range=[{gt_norm.min():.4f}, {gt_norm.max():.4f}]",
          flush=True)
    print(f"    L2 pred norm:   mean={pr_norm.mean():.4f}  "
          f"range=[{pr_norm.min():.4f}, {pr_norm.max():.4f}]",
          flush=True)
    print(f"    L2 diff norm:   mean={l2_diff.mean():.4f}  "
          f"max={l2_diff.max():.4f}", flush=True)
    print(f"    scale ratio:    pred/gt = "
          f"{(pr_norm.mean()/gt_norm.mean().clamp_min(1e-8)):.4f}",
          flush=True)
    print(f"    per-ch corr:    {ch_corr_mean:.4f}", flush=True)

    return {
        "cosine_mean": cos.mean().item(),
        "l2_diff_mean": l2_diff.mean().item(),
        "scale_ratio": (pr_norm.mean() / gt_norm.mean().clamp_min(1e-8)).item(),
    }


def run_audit(m, tok) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

    from calm.llm_computer.r51.install import (
        install_r51_student, load_student_from_checkpoint,
    )

    for ckpt_path, label in [(R51_CKPT, "R51-MSE"), (R52_CKPT, "R52-KL")]:
        print("\n" + "=" * 80, flush=True)
        print(f"[audit] loading {label}: {ckpt_path}", flush=True)
        try:
            student = load_student_from_checkpoint(ckpt_path, device="cuda")
        except FileNotFoundError:
            print(f"  checkpoint missing — skipping {label}", flush=True)
            continue
        print(f"  student params: "
              f"{sum(p.numel() for p in student.parameters()):,}",
              flush=True)

        summaries = []

        for domain, prompt in PROMPTS:
            print(f"\n[{label}] prompt={domain!r}: {prompt[:50]!r}",
                  flush=True)
            ids = tok.encode(prompt)
            S = len(ids)
            print(f"  S={S} tokens", flush=True)

            # 1. Ground truth L24 contribution
            h_before, gt_contrib, h_native = capture_l24_gt(m, tok, prompt)

            # 2. Student prediction
            if S > student.config.max_len:
                print(f"  SKIP: S={S} > student max_len="
                      f"{student.config.max_len}", flush=True)
                continue
            with torch.no_grad():
                pred_contrib = student(h_before.to(
                    device=next(student.parameters()).device,
                    dtype=next(student.parameters()).dtype,
                ))
                pred_contrib = pred_contrib.to(
                    device=gt_contrib.device, dtype=gt_contrib.dtype)

            # 3. Compare distributions (training fidelity)
            summary = compare_distributions(
                gt_contrib, pred_contrib, f"Q1 student vs GT contribution")
            summary["domain"] = domain
            summary["S"] = S
            summaries.append(summary)

            # 4. Install boundary check: install + run, verify L24
            #    output == h_before + student(h_before)
            handle = install_r51_student(m, student, target_layer=TARGET_LAYER)
            try:
                _hb, _gt, h_installed = capture_l24_gt(m, tok, prompt)
                expected = h_before + pred_contrib
                boundary_diff = (h_installed - expected).abs()
                print(f"  [Q2 install boundary check]", flush=True)
                print(f"    L24_installed vs (h_before + student(h_before)):",
                      flush=True)
                print(f"      max abs diff:  {boundary_diff.max():.2e}",
                      flush=True)
                print(f"      mean abs diff: {boundary_diff.mean():.2e}",
                      flush=True)
                if boundary_diff.max() < 1e-4:
                    print(f"      ✓ install math correct", flush=True)
                else:
                    print(f"      ✗ INSTALL MATH BUG — diff > 1e-4",
                          flush=True)
            finally:
                handle.detach()

        # Aggregate
        if summaries:
            print(f"\n[{label}] aggregate:", flush=True)
            cos_mean = sum(s["cosine_mean"] for s in summaries) / len(summaries)
            scale_mean = sum(s["scale_ratio"] for s in summaries) / len(summaries)
            l2_mean = sum(s["l2_diff_mean"] for s in summaries) / len(summaries)
            print(f"  mean cosine (pred vs GT):  {cos_mean:.4f}", flush=True)
            print(f"  mean scale ratio:          {scale_mean:.4f}", flush=True)
            print(f"  mean L2 diff:              {l2_mean:.4f}", flush=True)

            # Verdict
            if cos_mean > 0.8:
                print(f"  → Student reproduces L24 well. "
                      f"If eval still fails, look at install/eval pipeline.",
                      flush=True)
            elif cos_mean > 0.5:
                print(f"  → Student partially reproduces L24. Borderline.",
                      flush=True)
            else:
                print(f"  → Student does NOT reproduce L24. Training is "
                      f"the bottleneck; no install artifact suspected.",
                      flush=True)

    print("\n[audit] DONE", flush=True)


if __name__ == "__main__":
    print("Daemon-only: bin/gemma-run scripts/r53_36_audit_r51_install.py",
          flush=True)
    sys.exit(1)
elif "m" in globals() and "tok" in globals():
    run_audit(m, tok)                                 # noqa: F821
