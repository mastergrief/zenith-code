#!/usr/bin/env python3
"""Stream-extract a bankable Phase A evidence manifest from a live diagnostic run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import ijson


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _step_update_durations(stdout_path: Path) -> dict[int, float]:
    out: dict[int, float] = {}
    with stdout_path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if '"phase": "step_update"' not in line or '"event": "end"' not in line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if obj.get("phase") == "step_update" and obj.get("event") == "end":
                out[int(obj["step"])] = float(obj["duration_seconds"])
    return out


def _stream_r4v_summary(receipt_path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    with receipt_path.open("rb") as f:
        for key, value in ijson.kvitems(f, "r4v_persistent_ledger"):
            if key == "r4v_per_module_acc_rows":
                rows = list(value)
                events = [int(r.get("r4v_actual_events_payload_bytes", 0)) for r in rows]
                backlog = [int(r.get("r4v_actual_backlog_payload_bytes", 0)) for r in rows]
                hot = [int(r.get("r4v_actual_hot_exact_payload_bytes", 0)) for r in rows]
                summary["per_module_acc_rows_summary"] = {
                    "count": len(rows),
                    "events_payload_bytes": {
                        "min": min(events) if events else 0,
                        "max": max(events) if events else 0,
                        "sum": sum(events),
                    },
                    "backlog_payload_bytes": {
                        "min": min(backlog) if backlog else 0,
                        "max": max(backlog) if backlog else 0,
                        "sum": sum(backlog),
                    },
                    "hot_exact_payload_bytes": {
                        "min": min(hot) if hot else 0,
                        "max": max(hot) if hot else 0,
                        "sum": sum(hot),
                    },
                }
            elif key in (
                "r4v_acc_physical_bits_per_weight",
                "r4v_acc_inclusive_physical_bits_per_weight",
                "r4v_budget_physical_bits_per_weight",
            ):
                summary[key] = float(value) if value is not None else None
            elif isinstance(value, (str, int, float, bool)) or value is None:
                summary[key] = value
    return summary


def _stream_terminal_status(receipt_path: Path) -> dict[str, Any]:
    with receipt_path.open("rb") as f:
        return next(ijson.items(f, "terminal_status"))


def build_manifest(
    *,
    run_root: Path,
    run_id: str,
    head_commit: str,
    packet_sha: str,
    baseline_run_id: str,
    classification: str,
) -> dict[str, Any]:
    phase_a = run_root / "phase_a"
    receipt_path = phase_a / "receipt.json"
    stdout_path = phase_a / "probe.stdout.log"
    votes_manifest_path = phase_a / "votes_emit" / "v1" / "manifest.json"
    prehash_path = run_root / "prelaunch" / "parent_checkpoint_prehash.json"
    posthash_path = run_root / "prelaunch" / "parent_checkpoint_posthash.json"
    marker_path = run_root / "prelaunch" / "LAUNCHER_DONE.marker"

    step_update_s = _step_update_durations(stdout_path)
    baseline_root = run_root.parent / f"v4_live_diagnostic_tier1_seed44_43_{baseline_run_id}"
    baseline_stdout = baseline_root / "phase_a" / "probe.stdout.log"
    baseline_step_update_s = (
        _step_update_durations(baseline_stdout) if baseline_stdout.is_file() else {}
    )
    matched_ratios: dict[str, float] = {}
    for step, dur in sorted(step_update_s.items()):
        base = baseline_step_update_s.get(step)
        if base is not None and base > 0:
            matched_ratios[str(step)] = round(dur / base, 4)

    votes_manifest = json.loads(votes_manifest_path.read_text(encoding="utf-8"))
    prehash = json.loads(prehash_path.read_text(encoding="utf-8"))
    posthash = json.loads(posthash_path.read_text(encoding="utf-8"))

    return {
        "schema": "hrm_text_158_v4_phase_a_evidence_manifest/v0",
        "classification": classification,
        "provenance": {
            "run_id": run_id,
            "run_root": str(run_root),
            "head_commit": head_commit,
            "packet_sha": packet_sha,
            "exit_code": int((run_root / "probe.exit_code.txt").read_text().strip()),
            "launcher_done_marker_present": marker_path.is_file(),
            "parent_hash_before": prehash.get("parent_hash_before"),
            "parent_hash_after": posthash.get("parent_hash_after"),
            "parent_hash_unchanged": bool(posthash.get("parent_hash_unchanged")),
        },
        "dynamics": {
            "step_count": int(votes_manifest.get("step_count", 0)),
            "emit_sample_count": int(votes_manifest.get("emit_sample_count", 0)),
            "step_update_duration_seconds_by_step": {
                str(k): round(v, 3) for k, v in sorted(step_update_s.items())
            },
            "baseline_run_id": baseline_run_id,
            "step_update_ratio_vs_baseline_by_matched_step": matched_ratios,
            "final_step_update_seconds": step_update_s.get(max(step_update_s, default=0)),
        },
        "r4v_persistent_ledger": _stream_r4v_summary(receipt_path),
        "terminal_status": _stream_terminal_status(receipt_path),
        "source_artifacts": {
            "receipt_json_bytes": receipt_path.stat().st_size,
            "receipt_json_sha256": _sha256_file(receipt_path),
            "votes_emit_manifest_sha256": votes_manifest.get("manifest_sha256"),
            "extraction_method": "ijson_stream + stdout_line_scan (no whole-file load)",
        },
        "banking_note": (
            "Raw phase_a/receipt.json is runtime-only and non-bankable due to "
            "inline tier-A index arrays in step_reports; this manifest is the "
            "committable evidence surface."
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--head-commit", required=True)
    ap.add_argument("--packet-sha", required=True)
    ap.add_argument("--baseline-run-id", default="2189e72003")
    ap.add_argument(
        "--classification",
        default="B_CLEAN_DIAGNOSTIC+V4_EVENT_CODED_DRAIN_NOT_REDUCIBLE",
    )
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    manifest = build_manifest(
        run_root=args.run_root,
        run_id=args.run_id,
        head_commit=args.head_commit,
        packet_sha=args.packet_sha,
        baseline_run_id=args.baseline_run_id,
        classification=args.classification,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), "bytes": args.output.stat().st_size}))


if __name__ == "__main__":
    main()
