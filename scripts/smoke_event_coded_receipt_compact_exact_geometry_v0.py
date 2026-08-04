#!/usr/bin/env python3
"""Exact-geometry GPU proof: 20-step nondense event-coded screen → receipt under cap.

Binding 2: emits per-key byte census sidecar OUTSIDE bankable receipt.

Replay:
  cd /mnt/c/Users/gabes/projects/claw-code-hrm-text-158
  env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 \\
    HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH=1 \\
    python3 -u -B scripts/smoke_event_coded_receipt_compact_exact_geometry_v0.py \\
      --scratch-root /tmp/ec_receipt_compact_exact_geom_v0
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
PARENT = (
    "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_"
    "rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
BANKABLE_CAP = 10 * 1024 * 1024
TARGET_HEADROOM_MAX = 7 * 1024 * 1024


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--scratch-root",
        type=Path,
        default=Path("/tmp/ec_receipt_compact_exact_geom_v0"),
    )
    args = ap.parse_args(argv)
    scratch: Path = args.scratch_root
    scratch.mkdir(parents=True, exist_ok=True)
    receipt_path = scratch / "receipt.json"
    census_path = scratch / "receipt_key_byte_census.json"
    if receipt_path.exists():
        receipt_path.unlink()

    probe_argv = [
        sys.executable,
        "-u",
        "-B",
        str(PROBE),
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "cuda:0",
        "--parent",
        str(REPO / PARENT),
        "--parent-sha256",
        PARENT_SHA,
        "--curriculum-seed",
        "17",
        "--max-steps-hard",
        "50",
        "--steps",
        "20",
        "--scratch-root",
        str(scratch),
        "--prior-audit-supports",
        "L0b,math_a0",
        "--emit-progress",
        "--phase-heartbeat-seconds",
        "30",
        "--phase-timeout-seconds",
        "120",
        "--total-timeout-seconds",
        "3600",
        "--max-silent-phase-seconds",
        "600",
        "--persistent-accumulator-event-coded-live",
        "--event-coded-sparse-vote-authority",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"

    print(json.dumps({"phase": "probe_start", "argv": probe_argv}, sort_keys=True), flush=True)
    proc = subprocess.run(
        probe_argv,
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    print(
        json.dumps(
            {
                "phase": "probe_end",
                "returncode": proc.returncode,
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if proc.returncode != 0:
        print(
            f"SMOKE_FAIL probe_rc={proc.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return 2
    if not receipt_path.is_file():
        print("SMOKE_FAIL missing_receipt_json", file=sys.stderr, flush=True)
        return 3

    from calm.hrm_text_158.native_full_stack.event_coded_exact_geometry_receipt_validator_v0 import (
        validate_event_coded_exact_geometry_receipt,
    )
    from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
        RECEIPT_BANKABLE_MAX_BYTES,
        census_receipt_key_bytes,
        estimate_receipt_json_bytes,
        find_raw_inline_index_violations,
        validate_bankable_probe_receipt,
    )

    # Pure-validator class → distinct SMOKE_FAIL exit codes (rc8+).
    GEOMETRY_CLASS_RC = {
        "steps_requested": 8,
        "steps_completed": 9,
        "step_reports_coverage": 10,
        "toplevel_event_coded_live": 11,
        "toplevel_sparse_vote_authority": 12,
        "per_step_global_rate_cap": 13,
        "per_step_event_coded_live": 14,
        "gpu_execution_evidence": 15,
        "live_authority": 16,
        "bdgs_corroboration": 17,
    }

    assert RECEIPT_BANKABLE_MAX_BYTES == BANKABLE_CAP
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))

    # Geometry authority = pure per-step validator (NOT last-step-only BDGS).
    geometry_failures = validate_event_coded_exact_geometry_receipt(receipt)
    if geometry_failures:
        first = geometry_failures[0]
        cls = str(first.get("class") or "step_reports_coverage")
        detail = str(first.get("detail") or "")
        rc = int(GEOMETRY_CLASS_RC.get(cls, 18))
        print(
            json.dumps(
                {
                    "phase": "SMOKE_FAIL_geometry",
                    "first_class": cls,
                    "first_detail": detail,
                    "n_failures": len(geometry_failures),
                    "failures": geometry_failures[:20],
                    "rc": rc,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        print(
            f"SMOKE_FAIL geometry class={cls} detail={detail}",
            file=sys.stderr,
            flush=True,
        )
        return rc

    step_reports = receipt.get("step_reports") if isinstance(receipt.get("step_reports"), dict) else {}
    bdgs = (
        receipt.get("bounded_delta_global_summary")
        if isinstance(receipt.get("bounded_delta_global_summary"), dict)
        else {}
    )
    device = str(receipt.get("device") or "")

    size = estimate_receipt_json_bytes(receipt)
    census = census_receipt_key_bytes(receipt)
    census["receipt_path"] = str(receipt_path)
    census["file_bytes"] = receipt_path.stat().st_size
    census["estimate_json_bytes"] = size
    # Per-step authority summary (not last-step-only BDGS).
    per_step_cap = []
    per_step_live = []
    for sk, step in step_reports.items():
        gs = (
            step.get("step_result", {}).get("global_summary", {})
            if isinstance(step, dict)
            else {}
        )
        if isinstance(gs, dict):
            per_step_cap.append(gs.get("global_rate_cap_enabled"))
            per_step_live.append(gs.get("event_coded_live_carrier_enabled"))
    census["geometry"] = {
        "steps_requested": receipt.get("steps_requested"),
        "steps_completed": receipt.get("steps_completed"),
        "step_reports_count": len(step_reports),
        "persistent_accumulator_event_coded_live": receipt.get(
            "persistent_accumulator_event_coded_live"
        ),
        "event_coded_sparse_vote_authority": receipt.get(
            "event_coded_sparse_vote_authority"
        ),
        "per_step_global_rate_cap_enabled_all_false": all(v is False for v in per_step_cap)
        and len(per_step_cap) == 20,
        "per_step_event_coded_live_carrier_all_true": all(v is True for v in per_step_live)
        and len(per_step_live) == 20,
        "bdgs_corroboration_global_rate_cap_enabled": bdgs.get("global_rate_cap_enabled"),
        "bdgs_corroboration_event_coded_live_carrier_enabled": bdgs.get(
            "event_coded_live_carrier_enabled"
        ),
        "device": device,
        "gpu_launched": receipt.get("gpu_launched"),
        "geometry_validator": "validate_event_coded_exact_geometry_receipt",
        "geometry_failures": [],
    }
    census_path.write_text(json.dumps(census, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "phase": "census",
                "census_path": str(census_path),
                "estimate_json_bytes": size,
                "file_bytes": receipt_path.stat().st_size,
                "top5": census.get("top", [])[:5],
                "geometry": census["geometry"],
            },
            sort_keys=True,
        ),
        flush=True,
    )

    raw_violations = find_raw_inline_index_violations(receipt)
    bank_failures = validate_bankable_probe_receipt(receipt)
    if raw_violations:
        print(
            f"SMOKE_FAIL raw_index_violations={raw_violations[:10]}",
            file=sys.stderr,
            flush=True,
        )
        return 4
    if bank_failures:
        print(
            f"SMOKE_FAIL bankable={bank_failures}",
            file=sys.stderr,
            flush=True,
        )
        return 5
    if size > BANKABLE_CAP:
        print(
            f"SMOKE_FAIL size={size} exceeds cap={BANKABLE_CAP}",
            file=sys.stderr,
            flush=True,
        )
        return 6
    if size > TARGET_HEADROOM_MAX:
        print(
            f"SMOKE_FAIL size={size} exceeds headroom_target={TARGET_HEADROOM_MAX} "
            f"(cap still {BANKABLE_CAP}; census at {census_path})",
            file=sys.stderr,
            flush=True,
        )
        return 7

    print(
        json.dumps(
            {
                "phase": "SMOKE_OK",
                "estimate_json_bytes": size,
                "cap": BANKABLE_CAP,
                "headroom_target": TARGET_HEADROOM_MAX,
                "geometry": census["geometry"],
            },
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
