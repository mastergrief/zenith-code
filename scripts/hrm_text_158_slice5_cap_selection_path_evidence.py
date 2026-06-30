#!/usr/bin/env python3
"""Executable cap-selection path evidence for Slice-5 v6e diagnostic re-smoke."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

RECEIPT_SCHEMA = "hrm_text_158_slice5_cap_selection_path_evidence/v1"
ARMS = ("baseline_snapshot_off", "instrumented_snapshot_on")
CAP_SELECTION_JSONL = "sparse_cap_apply_cap_selection_cpu_copy.jsonl"
SPARSE_CAP_JSONL = "sparse_cap_apply.jsonl"

JSONL_MARKER_GPU = "cap_gpu_seam_done"
JSONL_MARKER_CPU_RESIDENT = "cap_reference_cpu_resident_done"
JSONL_MARKER_CPU_SHIM = "cap_reference_cpu_shim_done"
SUMMARY_PATH_GPU = "gpu_seam"
SUMMARY_PATH_CPU_RESIDENT = "cpu_resident_reference"
SUMMARY_PATH_CPU_REFERENCE = "cpu_reference"

ROUTE_GPU = "PATH_GPU_SEAM_EXERCISED"
ROUTE_CPU = "PATH_CPU_RESIDENT_CAP_REFERENCE"
ROUTE_INVALID = "SUBMILESTONE_INSTRUMENTATION_INVALID"


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _sparse_cap_completes(scratch: Path) -> int:
    rows = _read_jsonl(scratch / "liveness_milestones" / SPARSE_CAP_JSONL)
    return sum(1 for row in rows if row.get("milestone_kind") == "phase_complete")


def _summary_cap_selection_path(scratch: Path) -> str | None:
    receipt = _read_json(scratch / "receipt.json")
    for container_key in ("global_summary", "summary", "bounded_delta_global_summary"):
        summary = receipt.get(container_key) or {}
        if isinstance(summary, dict):
            path = summary.get("sparse_cap_submilestone_cap_selection_path")
            if path is not None:
                return str(path)
    return None


def _route_from_marker(
    *,
    marker_kind: str | None,
    summary_path: str | None,
) -> tuple[str, list[str]]:
    failures: list[str] = []
    if marker_kind is None:
        return ROUTE_INVALID, ["cap_selection_marker_absent"]

    if marker_kind == JSONL_MARKER_GPU:
        route = ROUTE_GPU
        allowed_summary = {SUMMARY_PATH_GPU}
    elif marker_kind in {JSONL_MARKER_CPU_RESIDENT, JSONL_MARKER_CPU_SHIM}:
        route = ROUTE_CPU
        allowed_summary = {SUMMARY_PATH_CPU_RESIDENT, SUMMARY_PATH_CPU_REFERENCE, "cpu_shim"}
    else:
        return ROUTE_INVALID, [f"unknown_cap_selection_marker_kind:{marker_kind}"]

    if summary_path is not None and summary_path not in allowed_summary:
        failures.append(
            f"marker_path_mismatch:jsonl={marker_kind},summary={summary_path}"
        )
        return ROUTE_INVALID, failures
    return route, failures


def evaluate_arm(*, scratch: Path) -> dict[str, Any]:
    cap_jsonl = scratch / "liveness_milestones" / CAP_SELECTION_JSONL
    sparse_completes = _sparse_cap_completes(scratch)
    cap_rows = _read_jsonl(cap_jsonl)
    marker_kinds = [str(row.get("milestone_kind")) for row in cap_rows if row.get("milestone_kind")]
    marker_kind = marker_kinds[-1] if marker_kinds else None
    summary_path = _summary_cap_selection_path(scratch)

    failures: list[str] = []
    if sparse_completes > 0 and marker_kind is None:
        failures.append("sparse_cap_complete_but_cap_selection_absent")
        route = ROUTE_INVALID
    else:
        route, route_failures = _route_from_marker(
            marker_kind=marker_kind,
            summary_path=summary_path,
        )
        failures.extend(route_failures)

    return {
        "arm": scratch.name,
        "sparse_cap_phase_complete_count": int(sparse_completes),
        "cap_selection_jsonl_present": cap_jsonl.is_file(),
        "cap_selection_jsonl_milestone_kind": marker_kind,
        "summary_cap_selection_path": summary_path,
        "terminal_route": route,
        "failures": failures,
        "pass": not failures,
    }


def evaluate_cap_selection_path_evidence(*, run_root: Path) -> dict[str, Any]:
    per_arm = [evaluate_arm(scratch=run_root / arm) for arm in ARMS]
    any_sparse_complete = any(row["sparse_cap_phase_complete_count"] > 0 for row in per_arm)
    routes = {row["terminal_route"] for row in per_arm if row["sparse_cap_phase_complete_count"] > 0}
    failures: list[str] = []
    for row in per_arm:
        failures.extend(f"{row['arm']}:{item}" for item in row["failures"])

    if any_sparse_complete and ROUTE_INVALID in routes and len(routes) > 1:
        failures.append("mixed_path_routes_across_arms_with_sparse_complete")

    overall_route = ROUTE_INVALID
    if routes == {ROUTE_GPU}:
        overall_route = ROUTE_GPU
    elif routes <= {ROUTE_CPU, ROUTE_INVALID} and ROUTE_CPU in routes:
        overall_route = ROUTE_CPU
    elif not any_sparse_complete:
        overall_route = ROUTE_INVALID
        failures.append("no_sparse_cap_completion_on_either_arm")

    return {
        "schema": RECEIPT_SCHEMA,
        "run_root": str(run_root),
        "per_arm": per_arm,
        "overall_terminal_route": overall_route,
        "jsonl_marker_kinds_observed": sorted(
            {
                JSONL_MARKER_GPU,
                JSONL_MARKER_CPU_RESIDENT,
                JSONL_MARKER_CPU_SHIM,
            }
        ),
        "summary_path_labels": [
            SUMMARY_PATH_GPU,
            SUMMARY_PATH_CPU_RESIDENT,
            SUMMARY_PATH_CPU_REFERENCE,
        ],
        "pass": not failures,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = evaluate_cap_selection_path_evidence(run_root=args.run_root)
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    out = args.out or (args.run_root / "prelaunch" / "cap_selection_path_evidence_receipt.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    if args.out is None:
        sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
