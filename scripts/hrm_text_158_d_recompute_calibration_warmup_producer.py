#!/usr/bin/env python3
"""Produce calibration warmup observations via bounded D-ON probe + parent SHA proof."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.d_recompute_window_calibration_collector import (
    CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
    parent_checkpoint_sha256,
)
from scripts.hrm_text_158_d_recompute_calibration_prepass import default_calibration_policy

PRODUCER_SCHEMA = "hrm_text_158_d_recompute_calibration_warmup_producer_receipt/v0"
DEFAULT_PARENT = (
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
DEFAULT_PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"


def build_calibration_warmup_probe_argv(
    *,
    run_root: Path,
    parent: Path,
    parent_sha256: str,
    warmup_steps: int,
    observations_out: Path,
) -> list[str]:
    warmup_scratch = run_root / "calibration_warmup"
    return [
        "python3",
        "-u",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--allow-gpu-launch",
        "--enable-bounded-delta-probe",
        "--device",
        "cuda",
        "--parent",
        str(parent),
        "--parent-sha256",
        str(parent_sha256),
        "--curriculum-seed",
        "43",
        "--support-order-seed",
        "43",
        "--eligible-scope",
        "all-bitlinear",
        "--batch-size",
        "1",
        "--science-arm",
        "A0_rank_bucket_current_ordering",
        "--global-cap-contract",
        "c1_banked_faithful_long_run_global_cap",
        "--confirmation-envelope",
        "canonical_t10_prereg_v24",
        "--phase",
        "d-recompute-window-feasibility",
        "--emit-progress",
        "--dense-accumulator-w8-clip",
        "--receipt-emit-profile",
        "s3bb_headroom_diagnostic_slim",
        "--d-recompute-window-instrumentation",
        "--d-diagnostic-compact-step-reports",
        "--steps",
        str(int(warmup_steps)),
        "--max-steps-hard",
        str(int(warmup_steps)),
        "--scratch-root",
        str(warmup_scratch),
        "--d-recompute-calibration-warmup-out",
        str(observations_out),
    ]


def run_calibration_warmup_producer(
    *,
    run_root: Path,
    observations_out: Path,
    report_out: Path | None = None,
    parent: Path | None = None,
    parent_sha256: str | None = None,
    warmup_steps: int = 5,
    probe_runner: Any | None = None,
) -> dict[str, Any]:
    parent_path = Path(parent or DEFAULT_PARENT)
    expected_parent_sha = str(parent_sha256 or DEFAULT_PARENT_SHA)
    pre_warmup_sha = parent_checkpoint_sha256(parent_path)
    if pre_warmup_sha != expected_parent_sha:
        raise ValueError(
            f"parent sha256 mismatch before warmup: expected {expected_parent_sha}, "
            f"got {pre_warmup_sha}"
        )

    observations_out = Path(observations_out)
    observations_out.parent.mkdir(parents=True, exist_ok=True)
    argv = build_calibration_warmup_probe_argv(
        run_root=run_root,
        parent=parent_path,
        parent_sha256=expected_parent_sha,
        warmup_steps=int(warmup_steps),
        observations_out=observations_out,
    )
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": ".",
            "HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH": "1",
            "HRM_TEXT_158_RUN_C2_ACQUISITION_PROBE": "1",
            "HRM_TEXT_158_RUN_NARROW_CARRIER_W8_TRAINER_INTEGRATION": "1",
        }
    )
    runner = probe_runner or subprocess.run
    result = runner(argv, check=False, capture_output=True, text=True, env=env)
    if int(result.returncode) != 0:
        raise RuntimeError(
            "calibration warmup probe failed: "
            f"rc={result.returncode} stderr={(result.stderr or '')[-2000:]}"
        )
    if not observations_out.is_file():
        raise FileNotFoundError(
            f"calibration warmup observations not written: {observations_out}"
        )
    payload = json.loads(observations_out.read_text(encoding="utf-8"))
    if str(payload.get("schema_version")) != CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA:
        raise ValueError("warmup observations missing expected schema_version")
    if str(payload.get("pre_warmup_banked_state_sha256")) != pre_warmup_sha:
        raise ValueError("warmup observations pre_warmup_banked_state_sha256 mismatch")

    post_warmup_sha = parent_checkpoint_sha256(parent_path)
    bit_exact_parent_restored = post_warmup_sha == pre_warmup_sha
    receipt = {
        "schema_version": PRODUCER_SCHEMA,
        "run_root": str(run_root),
        "observations_out": str(observations_out),
        "warmup_steps": int(warmup_steps),
        "pre_warmup_banked_state_sha256": pre_warmup_sha,
        "post_warmup_parent_sha256": post_warmup_sha,
        "bit_exact_pre_warmup_parent_restored": bit_exact_parent_restored,
        "calibration_discarded_before_measurement": True,
        "producer": "standalone_warmup_producer_invokes_probe_session_boundary_seam",
        "pass": bit_exact_parent_restored,
    }
    if report_out is not None:
        report_out.parent.mkdir(parents=True, exist_ok=True)
        report_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--observations-out", type=Path, required=True)
    parser.add_argument("--json-report", type=Path, default=None)
    parser.add_argument("--parent", type=Path, default=Path(DEFAULT_PARENT))
    parser.add_argument("--parent-sha256", default=DEFAULT_PARENT_SHA)
    parser.add_argument("--warmup-steps", type=int, default=5)
    args = parser.parse_args(argv)
    receipt = run_calibration_warmup_producer(
        run_root=args.run_root,
        observations_out=args.observations_out,
        report_out=args.json_report,
        parent=args.parent,
        parent_sha256=str(args.parent_sha256),
        warmup_steps=int(args.warmup_steps),
    )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if bool(receipt["pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
