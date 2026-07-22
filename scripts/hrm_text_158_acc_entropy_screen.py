"""Vote-accumulator entropy screen — the sub-2 branch-decider for the int8 lane.

Question (pre-registered): what is the empirical entropy of the dense vote
accumulator under the REAL update law driven by REAL gradients? The sub-2
persistent budget leaves the acc <= ~0.4 bits/weight (2.0 total - 1.6 packed
q). Two branches:
  (a) H_steady < 0.4 b/w  -> sub-2 is a CODEC problem (entropy coding of the
      existing mechanism suffices; no new dynamics needed);
  (b) H_steady > 0.4 b/w  -> no encoding can save the current mechanism —
      it must FORGET more (leak/decay/sparsify) before carrier work matters.

Design (screen-grade, read-only on all banked artifacts):
- Banked parent `L0c2K2add50s...step00750`, CPU, weights FROZEN (no q updates).
- Real per-step gradients: batch-8 response-only CE over seed-varied samples
  of the A0 exhaustive supports; one backward per step.
- Moves via the lane's OWN `project_s1_gradient_to_moves` (imported, not
  reimplemented); q_levels from the BitLinear quantize law
  (round(w/scale).clamp(-1,1), scale=|w|.mean()).
- Acc update: acc += move (1-quantum sign-pressure votes), W8 clip [-127,127],
  crossing at |acc| >= 10 (two_tier_threshold_semantics.CROSSING_THRESHOLD_ABS).
- Live flip-drain regime is bracketed with three arms (the applied-rate cap
  varies by run):
    armA_backlog   : never drain (saturated-backlog upper bracket)
    armB_drain_all : every crosser resets to 0 (uncapped-flips lower bracket)
    armC_topk      : top-K=1024 crossers by |acc| drain per step (capped, the
                     closest analog of the global-applied-rate-cap regime)
- Every 10 steps: pooled empirical entropy H (bits/weight) over the discrete
  acc distribution, zero-fraction, crosser-fraction, clamp-fraction, max|acc|.

Honest limits (recorded in the receipt): frozen weights (no coupled
weight-motion), marginal entropy is the iid-coding bound (spatial correlation
can only improve it), drain-arm bracket instead of the exact cap schedule.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (  # noqa: E402
    project_s1_gradient_to_moves,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (  # noqa: E402
    CROSSING_THRESHOLD_ABS,
)

CLIP = 127
TOPK_PER_STEP = 1024
SUB2_ACC_BUDGET_BITS = 0.4

BULK_SUFFIXES = ("gqkv_proj.weight", "o_proj.weight",
                 "gate_up_proj.weight", "down_proj.weight")


def _load_probe_and_screen():
    base = os.path.dirname(os.path.abspath(__file__))
    out = []
    for name in ("probe_hrm_text_158", "hrm_text_158_rotor_backward_saved_screen"):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(base, name + ".py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(mod)
    return out


def _entropy_bits(acc: torch.Tensor) -> float:
    vals, counts = torch.unique(acc, return_counts=True)
    p = counts.double() / acc.numel()
    return float(-(p * p.log2()).sum())


def _arm_stats(acc: torch.Tensor) -> dict:
    a = acc.abs()
    return {
        "H_bits_per_weight": _entropy_bits(acc),
        "zero_frac": float((acc == 0).float().mean()),
        "crosser_frac": float((a >= CROSSING_THRESHOLD_ABS).float().mean()),
        "clamp_frac": float((a >= CLIP).float().mean()),
        "max_abs": int(a.max()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--output-json", default=None)
    args = ap.parse_args()

    torch.set_num_threads(8)
    probe, scr = _load_probe_and_screen()

    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )
    import random

    print(f"[acc-H] loading ckpt (CPU): {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    m, tok = probe._build_model_from_ckpt(ckpt, "cpu")
    m.train()

    # Eligible bulk + ternary q_levels per the BitLinear quantize law.
    eligible = {n: p for n, p in m.named_parameters()
                if n.endswith(BULK_SUFFIXES)}
    q_levels = {}
    with torch.no_grad():
        for n, p in eligible.items():
            scale = p.abs().mean().clamp(min=1e-5)
            q_levels[n] = (p / scale).round().clamp(-1, 1).to(torch.int8)
    n_eligible = sum(p.numel() for p in eligible.values())
    print(f"[acc-H] eligible tensors={len(eligible)} weights={n_eligible:,} "
          f"crossing={CROSSING_THRESHOLD_ABS} clip=±{CLIP} "
          f"topK={TOPK_PER_STEP}", flush=True)

    arms = {a: {n: torch.zeros_like(q, dtype=torch.int16)
                for n, q in q_levels.items()}
            for a in ("armA_backlog", "armB_drain_all", "armC_topk")}
    flips = {a: 0 for a in arms}

    pool = [(q, int(e)) for rows in build_exhaustive_supports().values()
            for q, e in rows]
    trajectory = []
    t0 = time.time()
    for step in range(1, args.steps + 1):
        rng = random.Random(1000 + step)
        batch_rows = [{"rung": "mix", "question": q, "expected": e}
                      for q, e in rng.sample(pool, args.batch)]
        _loss, grads = scr._loss_and_grads(
            m, tok, batch_rows, max_seq_len=384, device="cpu", codec=None)

        moves = {n: project_s1_gradient_to_moves(grads[n], q_levels[n])
                 for n in eligible}

        # armA: pure backlog.
        for n, mv in moves.items():
            a = arms["armA_backlog"][n]
            a.add_(mv.to(torch.int16)).clamp_(-CLIP, CLIP)
        # armB: drain every crosser.
        for n, mv in moves.items():
            a = arms["armB_drain_all"][n]
            a.add_(mv.to(torch.int16)).clamp_(-CLIP, CLIP)
            crossed = a.abs() >= CROSSING_THRESHOLD_ABS
            flips["armB_drain_all"] += int(crossed.sum())
            a[crossed] = 0
        # armC: drain global top-K crossers by |acc|.
        flat_abs = []
        for n, mv in moves.items():
            a = arms["armC_topk"][n]
            a.add_(mv.to(torch.int16)).clamp_(-CLIP, CLIP)
            flat_abs.append(a.abs().flatten())
        allabs = torch.cat(flat_abs)
        crosser_idx = torch.nonzero(
            allabs >= CROSSING_THRESHOLD_ABS, as_tuple=False).flatten()
        if crosser_idx.numel():
            k = min(TOPK_PER_STEP, crosser_idx.numel())
            top = crosser_idx[
                allabs[crosser_idx].argsort(descending=True)[:k]]
            flips["armC_topk"] += int(k)
            sel = torch.zeros_like(allabs, dtype=torch.bool)
            sel[top] = True
            off = 0
            for n in moves:
                a = arms["armC_topk"][n]
                nn = a.numel()
                a.view(-1)[sel[off:off + nn]] = 0
                off += nn

        if step % 10 == 0 or step == args.steps:
            snap = {"step": step, "elapsed_s": round(time.time() - t0, 1)}
            for aname, tensors in arms.items():
                pooled = torch.cat([t.flatten() for t in tensors.values()])
                snap[aname] = _arm_stats(pooled)
            snap["flips_cum"] = dict(flips)
            trajectory.append(snap)
            line = " ".join(
                f"{a}: H={snap[a]['H_bits_per_weight']:.4f} "
                f"z={snap[a]['zero_frac']:.3f} "
                f"x={snap[a]['crosser_frac']:.4f} "
                f"clamp={snap[a]['clamp_frac']:.5f} "
                f"max={snap[a]['max_abs']}"
                for a in arms)
            print(f"[acc-H] step {step:4d} ({snap['elapsed_s']}s) {line}",
                  flush=True)

    final = trajectory[-1]
    verdicts = {}
    for a in arms:
        H = final[a]["H_bits_per_weight"]
        verdicts[a] = ("codec_sufficient_sub2"
                       if H < SUB2_ACC_BUDGET_BITS
                       else "mechanism_must_forget_more")
    print(f"[acc-H] === PREREG BRANCHES (budget {SUB2_ACC_BUDGET_BITS} b/w) "
          f"===", flush=True)
    for a, v in verdicts.items():
        print(f"[acc-H]   {a}: H={final[a]['H_bits_per_weight']:.4f} -> {v}",
              flush=True)

    receipt = {
        "screen": "int8_vote_acc_entropy/v1_frozen_weight_bracket",
        "ckpt_path": args.ckpt_path,
        "law": {
            "projection": "project_s1_gradient_to_moves (imported)",
            "vote_quantum": 1,
            "crossing_threshold_abs": CROSSING_THRESHOLD_ABS,
            "clip": CLIP,
            "topk_per_step": TOPK_PER_STEP,
        },
        "limits": [
            "frozen weights (no coupled weight motion)",
            "marginal entropy = iid coding bound",
            "drain bracket, not exact cap schedule",
        ],
        "n_eligible_weights": n_eligible,
        "sub2_acc_budget_bits": SUB2_ACC_BUDGET_BITS,
        "steps": args.steps,
        "batch": args.batch,
        "trajectory": trajectory,
        "branch_verdicts": verdicts,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"[acc-H] receipt -> {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
