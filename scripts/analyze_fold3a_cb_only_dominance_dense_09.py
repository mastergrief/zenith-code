#!/usr/bin/env python3
"""Fold-3A CPU analysis: crossing-bearing-only dominance on dense-[0..9] receipt."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.s1d7_cb_only_dominance import (
    ANTI_OVERCLAIM_VERBATIM,
    build_fold3a_measurement_receipt,
    evaluate_cb_only_from_ca_confirmation_receipt,
)

DEFAULT_PRIMARY_RECEIPT = (
    "/home/gabe/hrm158_c4s1_phase3_gpu_gate/C4S1_PHASE3_N32_DENSE_09_CA_DECIDER_V1/"
    "prelaunch/callsite_band_counter_ca_confirmation/"
    "callsite_band_counter_ca_confirmation_receipt.json"
)
DEFAULT_WRITE_RECEIPT = (
    "artifacts/measurement_closeout/c4s1d7_fold3a_cb_only_dominance_dense_09_receipt.json"
)
UPSTREAM_CLOSEOUT_SPEC = (
    "artifacts/measurement_closeout/c4s1d7_dense_09_structural_fork_resolution_spec.md"
)
UPSTREAM_CLOSEOUT_RECEIPT = (
    "artifacts/measurement_closeout/c4s1d7_dense_09_structural_fork_resolution_receipt.json"
)
TASK_ID = "1782633464140-b85ec12a"
DISPATCH_MSG_ID = "1783246954375-8e9f37aa"
IMPLEMENT_GATE_MSG_ID = "1783247133497-b80fb0a6"


def _git_head(repo_root: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.strip()


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Evaluate Fold-3A crossing-bearing-only dominance on an existing "
            "CA confirmation receipt (CPU-only, no GPU)."
        )
    )
    ap.add_argument(
        "ca_confirmation_receipt",
        nargs="?",
        type=Path,
        default=Path(DEFAULT_PRIMARY_RECEIPT),
        help="Absolute path to callsite_band_counter_ca_confirmation_receipt.json",
    )
    ap.add_argument(
        "--write-receipt",
        type=Path,
        default=Path(DEFAULT_WRITE_RECEIPT),
        help="Repo-relative path for the fold-3 measurement receipt JSON.",
    )
    ap.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root (default: parent of scripts/).",
    )
    return ap


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    receipt_path = args.ca_confirmation_receipt.resolve()
    if not receipt_path.is_file():
        print(f"error: receipt not found: {receipt_path}", file=sys.stderr)
        return 2

    ca_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    dominance = evaluate_cb_only_from_ca_confirmation_receipt(ca_receipt)

    cross_checks: list[str] = []
    receipt_cb_rows = [
        row
        for row in ca_receipt.get("per_state") or []
        if int(row.get("crossing_indices_len") or 0) > 0
    ]
    if len(receipt_cb_rows) == 1:
        receipt_share = receipt_cb_rows[0].get("per_cb_ca_share")
        if receipt_share is not None:
            computed = dominance["a_plus_c_share"]
            if abs(float(receipt_share) - float(computed)) > 1e-9:
                cross_checks.append(
                    f"a_plus_c_share mismatch: computed={computed} receipt={receipt_share}"
                )

    measurement_receipt = build_fold3a_measurement_receipt(
        ca_confirmation_receipt_path=str(receipt_path),
        ca_confirmation_receipt=ca_receipt,
        dominance_result=dominance,
        git_head=_git_head(repo_root),
        task_id=TASK_ID,
        upstream_closeout_spec_path=UPSTREAM_CLOSEOUT_SPEC,
        upstream_closeout_receipt_path=UPSTREAM_CLOSEOUT_RECEIPT,
        implement_gate_msg_id=IMPLEMENT_GATE_MSG_ID,
        dispatch_msg_id=DISPATCH_MSG_ID,
    )
    measurement_receipt["cross_checks"] = cross_checks
    measurement_receipt["allowed_claim_note"] = (
        "single_cb_support=true marks within-support characterization, "
        "NOT cross-state generalization."
    )
    measurement_receipt["anti_overclaim_verbatim"] = ANTI_OVERCLAIM_VERBATIM

    out_path = (repo_root / args.write_receipt).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(measurement_receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    summary = {
        "terminal_branch": dominance["terminal_branch"],
        "cb_state_count": dominance["cb_state_count"],
        "single_cb_support": dominance["single_cb_support"],
        "excluded_zero_crossing_state_count": dominance[
            "excluded_zero_crossing_state_count"
        ],
        "c_only_dominance_ok": dominance["c_only_dominance_ok"],
        "a_plus_c_share": dominance["a_plus_c_share"],
        "write_receipt": str(out_path.relative_to(repo_root)),
        "cross_checks_failed": cross_checks,
    }
    json.dump(summary, sys.stdout, indent=2, sort_keys=True)
    sys.stdout.write("\n")
    return 1 if cross_checks else 0


if __name__ == "__main__":
    raise SystemExit(main())
