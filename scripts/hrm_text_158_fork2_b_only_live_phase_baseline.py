#!/usr/bin/env python3
"""Branch-A live B-only phase baseline (PRE/POST) — frequency-weighted schema.

Drives the production instrumented diagnostic path with diagnostic-only phase
timers. PRE is created exclusively (fail if exists). POST binds pre sha.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.pressure_metric_fork2_phase_timing import (  # noqa: E402
    PhaseTimer,
    assert_pre_immutable,
    frequency_weighted_summary,
    select_dominator_by_window_contribution,
    sha256_file,
    write_json_exclusive,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_warmup_runtime import (  # noqa: E402
    run_one_diagnostic_loop,
)
from calm.hrm_text_158.native_full_stack.screen_execution_loop import (  # noqa: E402
    TOPK_PER_STEP,
)

FUNCTIONAL_HEAD = "50db0b5a1b65220776101e4d24a1048c9f543dae"
DEFAULT_CKPT = (
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c2K2add50s_seed0017_replay80_n12k_lr5e5_pc1p0_"
    "rsL0b1math1r1b2m1idfull1k1to4pin1k5to8pin1L0c1pin1_ceL0c1x3_anchorsv1r3_"
    "from_L0c2K2add120k5to8_step01000_final_step00750.pt"
)
DEFAULT_PRE = (
    "artifacts/acc_entropy/pressure_metric_fork2_b_only_live_phase_baseline_PRE.json"
)
DEFAULT_POST = (
    "artifacts/acc_entropy/pressure_metric_fork2_b_only_live_phase_baseline_POST.json"
)
SEAM_FILES = (
    "calm/hrm_text_158/native_full_stack/screen_execution_loop.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_warmup_runtime.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_fork2_phase_timing.py",
)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _timer_seam_provenance(root: Path) -> dict:
    hashes = {rel: sha256_file(root / rel) for rel in SEAM_FILES}
    # Diff identity vs functional HEAD for seam files only (best-effort).
    try:
        diff = subprocess.check_output(
            ["git", "diff", FUNCTIONAL_HEAD, "--", *SEAM_FILES],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        diff = b""
    return {
        "functional_head_sha": FUNCTIONAL_HEAD,
        "timer_seam_source_hashes": hashes,
        "timer_seam_diff_identity": hashlib.sha256(diff).hexdigest(),
        "timer_seam_diff_bytes": len(diff),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("pre", "post"), required=True)
    ap.add_argument("--ckpt-path", default=DEFAULT_CKPT)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--steps", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--topk", type=int, default=TOPK_PER_STEP)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--pre-out", default=DEFAULT_PRE)
    ap.add_argument("--post-out", default=DEFAULT_POST)
    ap.add_argument("--pre-sha256", default=None, help="Required for --mode post")
    args = ap.parse_args()

    root = _repo_root()
    if not str(args.device).startswith("cuda"):
        print(json.dumps({"status": "INCOMPLETE", "error": "cuda_required"}))
        return 2

    import torch

    if not torch.cuda.is_available():
        print(json.dumps({"status": "INCOMPLETE", "error": "cuda_unavailable"}))
        return 2

    timer = PhaseTimer(enabled=True)
    result = run_one_diagnostic_loop(
        ckpt_path=str(args.ckpt_path),
        device=str(args.device),
        steps=int(args.steps),
        batch=int(args.batch),
        topk=int(args.topk),
        telemetry=True,
        skip_probes=True,
        seed=int(args.seed),
        warmup_enable=True,
        phase_timer=timer,
    )

    try:
        summary = frequency_weighted_summary(
            {k: list(v) for k, v in timer.samples.items()},
            steps=int(args.steps),
        )
    except ValueError as exc:
        payload = {
            "status": "UNPRICEABLE",
            "error": str(exc),
            "raw_sample_keys": sorted(timer.samples.keys()),
            "raw_sample_counts": {k: len(v) for k, v in timer.samples.items()},
        }
        print(json.dumps(payload, indent=2))
        return 3

    dominator = select_dominator_by_window_contribution(summary)
    prov = _timer_seam_provenance(root)
    opportunity = float(dominator["normalized_ms_per_step"])
    payload = {
        "schema": "pressure_metric_fork2_b_only_live_phase_baseline/v2",
        "status": "MEASURED",
        "mode": args.mode,
        "device": str(args.device),
        "steps": int(args.steps),
        "batch": int(args.batch),
        "topk": int(args.topk),
        "seed": int(args.seed),
        "ckpt_path": str(args.ckpt_path),
        "parent_sha": result.get("parent_sha"),
        "close_before_provisional_135ms_not_used": True,
        "timer_diagnostic_only": True,
        "timers_default_off_noop": True,
        **prov,
        **summary,
        "dominator": dominator,
        "single_phase_opportunity_normalized_ms_per_step": opportunity,
        "c2_single_phase_ge_60_opportunity": opportunity >= 60.0,
        "wall_ms_per_step": result.get("wall_ms_per_step"),
    }

    if args.mode == "pre":
        out = root / args.pre_out
        try:
            sha = write_json_exclusive(out, payload)
        except FileExistsError as exc:
            print(json.dumps({"status": "STOP", "error": str(exc)}))
            return 4
        print(
            json.dumps(
                {
                    "status": "MEASURED",
                    "mode": "pre",
                    "path": str(out),
                    "sha256": sha,
                    "dominator": dominator["name"],
                    "normalized_ms_per_step": opportunity,
                    "c2_single_phase_ge_60_opportunity": opportunity >= 60.0,
                },
                indent=2,
            )
        )
        return 0 if opportunity >= 60.0 else 5

    # POST
    if not args.pre_sha256:
        print(json.dumps({"status": "STOP", "error": "--pre-sha256 required for post"}))
        return 4
    pre_path = root / args.pre_out
    assert_pre_immutable(pre_path, str(args.pre_sha256))
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    pre_dom = pre["dominator"]
    post_dom = next(
        r for r in summary["phases"] if r["name"] == pre_dom["name"]
    )
    norm_delta = float(pre_dom["normalized_ms_per_step"]) - float(
        post_dom["normalized_ms_per_step"]
    )
    window_delta = float(pre_dom["window_contribution_ms"]) - float(
        post_dom["window_contribution_ms"]
    )
    payload.update(
        {
            "pre_artifact_sha256": str(args.pre_sha256),
            "pre_dominator_name": pre_dom["name"],
            "pre_dominator_row": pre_dom,
            "post_dominator_row": post_dom,
            "normalized_reduction_ms_per_step": norm_delta,
            "window_contribution_reduction_ms": window_delta,
            "acceptance_normalized_ge_60": norm_delta >= 60.0,
            "acceptance_window_ge_1500": window_delta >= 1500.0,
        }
    )
    out = root / args.post_out
    # POST may be reminted in defect cycles — allow overwrite only for POST.
    out.parent.mkdir(parents=True, exist_ok=True)
    body = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    out.write_text(body, encoding="utf-8")
    sha = hashlib.sha256(body.encode("utf-8")).hexdigest()
    print(
        json.dumps(
            {
                "status": "MEASURED",
                "mode": "post",
                "path": str(out),
                "sha256": sha,
                "pre_artifact_sha256": args.pre_sha256,
                "normalized_reduction_ms_per_step": norm_delta,
                "window_contribution_reduction_ms": window_delta,
                "acceptance_normalized_ge_60": norm_delta >= 60.0,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
