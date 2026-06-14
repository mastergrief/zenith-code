#!/usr/bin/env python3
"""Build chain_terminal_receipt.json for selector_support_consensus_v1 launch."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Baseline thresholds from selector_support_consensus_v1_chain_relaunch_plan real_rss_memory_pressure_read.
A3_CLEAN_RSS_MIN_BYTES = 4_800_000_000
A3_CLEAN_RSS_MAX_BYTES = 6_000_000_000
A3B_HEADROOM_MIN_KIB = 10_000_000
LOW_MEM_AVAILABLE_KIB = 2_097_152
HIGH_RSS_SPIKE_BYTES = 6_500_000_000

_DEATH_BRANCHES = frozenset({"i_tensor_pin", "ii_post_export_region", "iii_b2_regression"})


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    rows: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _merge_per_arm_rows(chain_root: Path) -> list[dict]:
    metrics = { (r.get("label"), r.get("arm")): dict(r) for r in _load_jsonl(chain_root / "per_arm_metrics.jsonl") }
    probes = _load_jsonl(chain_root / "probe_results.jsonl")
    merged: list[dict] = []
    for probe in probes:
        key = (probe.get("label"), probe.get("arm"))
        row = dict(metrics.get(key, {}))
        row.update(probe)
        exit_code = int(probe.get("exit_code", row.get("exit_code", -1)))
        receipt_emitted = bool(probe.get("receipt", probe.get("receipt_emitted", False)))
        rss_bytes = int(row.get("export_rss_peak_bytes") or row.get("rss_peak_bytes") or 0)
        row.setdefault("exit_code", exit_code)
        row.setdefault("wall_s", probe.get("wall_s"))
        row.setdefault("receipt_emitted", receipt_emitted)
        row.setdefault("rss_peak_bytes", rss_bytes)
        row.setdefault("rss_peak_gib", round(rss_bytes / (1024**3), 4) if rss_bytes else 0.0)
        row.setdefault("rss_peak_decimal_gb", round(rss_bytes / 1_000_000_000, 4) if rss_bytes else 0.0)
        merged.append(row)
    return merged


def _classify_tripwire_arm(scratch: Path) -> tuple[str | None, str | None, int | None, bool]:
    """Return (branch, dying_tensor_key, dying_tensor_index, all_tensor_exports_done)."""
    starts: dict[tuple[int | None, str | None], float | None] = {}
    dones: set[tuple[int | None, str | None]] = set()

    def ingest(ev: dict) -> None:
        if ev.get("phase") != "checkpoint_payload":
            return
        event = ev.get("event")
        identity = (ev.get("tensor_index"), ev.get("tensor_key"))
        if event == "checkpoint_tensor_export_start":
            starts[identity] = ev.get("elapsed_since_start_seconds")
        elif event == "checkpoint_tensor_export_done":
            dones.add(identity)

    run_log = scratch / "run.log"
    run_log_text = run_log.read_text(encoding="utf-8", errors="replace") if run_log.is_file() else ""
    if run_log_text:
        for line in run_log_text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                ingest(payload)

    receipt_path = scratch / "receipt.json"
    if receipt_path.is_file():
        try:
            receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
            for ev in receipt.get("phase_telemetry", {}).get("events", []):
                if isinstance(ev, dict):
                    ingest(ev)
        except (json.JSONDecodeError, OSError):
            pass

    if not starts:
        if "LIVENESS_FAILURE" in run_log_text:
            return "iii_b2_regression", None, None, False
        return None, None, None, False

    pending = [k for k in starts if k not in dones]
    all_done = bool(starts) and not pending and len(dones) >= len(starts)
    if pending:
        idx, key = pending[-1]
        return "i_tensor_pin", key, idx, False
    if all_done and not receipt_path.is_file():
        return "ii_post_export_region", None, None, True
    if all_done:
        return None, None, None, True
    return None, None, None, False


def _arm_died_during_checkpoint(row: dict) -> bool:
    if int(row.get("exit_code", 1)) != 0:
        return True
    if not row.get("receipt_emitted"):
        return True
    return row.get("outcome_branch") in _DEATH_BRANCHES


def _classify_site_d_read(
    per_arm: list[dict],
    *,
    tripwire_triggered: bool,
    outcome_branch: str,
) -> tuple[str, str, dict]:
    deciding: dict = {
        "per_arm_export_rss_peak_bytes": [],
        "per_arm_host_mem_min_available_kib_during_checkpoint_payload": [],
        "death_arm_labels": [],
        "max_export_rss_peak_bytes": 0,
        "min_host_mem_available_kib": None,
        "tripwire_triggered": tripwire_triggered,
        "outcome_branch": outcome_branch,
    }
    if not per_arm:
        return "site_d_undetermined", "manual_review", deciding

    for row in per_arm:
        rss = int(row.get("export_rss_peak_bytes") or row.get("rss_peak_bytes") or 0)
        mem = row.get("host_mem_min_available_kib_during_checkpoint_payload")
        deciding["per_arm_export_rss_peak_bytes"].append(
            {"label": row.get("label"), "arm": row.get("arm"), "export_rss_peak_bytes": rss}
        )
        deciding["per_arm_host_mem_min_available_kib_during_checkpoint_payload"].append(
            {
                "label": row.get("label"),
                "arm": row.get("arm"),
                "host_mem_min_available_kib_during_checkpoint_payload": mem,
            }
        )
        if _arm_died_during_checkpoint(row):
            deciding["death_arm_labels"].append(f"{row.get('label')}/{row.get('arm')}")
        if rss > deciding["max_export_rss_peak_bytes"]:
            deciding["max_export_rss_peak_bytes"] = rss
        if mem is not None:
            mem_i = int(mem)
            prev = deciding["min_host_mem_available_kib"]
            deciding["min_host_mem_available_kib"] = mem_i if prev is None else min(prev, mem_i)

    has_rss = deciding["max_export_rss_peak_bytes"] > 0
    has_mem = deciding["min_host_mem_available_kib"] is not None
    death = bool(deciding["death_arm_labels"]) or outcome_branch in _DEATH_BRANCHES

    def pressure_signal() -> bool:
        rss_spike = has_rss and deciding["max_export_rss_peak_bytes"] >= HIGH_RSS_SPIKE_BYTES
        mem_low = has_mem and deciding["min_host_mem_available_kib"] < LOW_MEM_AVAILABLE_KIB
        return rss_spike or mem_low

    if death:
        if not has_rss and not has_mem:
            return "site_d_undetermined", "manual_review", deciding
        if pressure_signal():
            return "site_d_justified", "open_separate_gated_site_d_plan", deciding
        if not has_rss or not has_mem:
            return "site_d_undetermined", "manual_review", deciding
        return "site_d_deferred", "defer", deciding

    if not has_rss or not has_mem:
        return "site_d_undetermined", "manual_review", deciding

    clean_rss_band = A3_CLEAN_RSS_MIN_BYTES <= deciding["max_export_rss_peak_bytes"] <= A3_CLEAN_RSS_MAX_BYTES
    a3b_headroom = deciding["min_host_mem_available_kib"] >= A3B_HEADROOM_MIN_KIB
    if clean_rss_band and a3b_headroom:
        return "site_d_deferred", "defer", deciding
    if has_mem and deciding["min_host_mem_available_kib"] < LOW_MEM_AVAILABLE_KIB:
        return "site_d_justified", "open_separate_gated_site_d_plan", deciding
    return "site_d_undetermined", "manual_review", deciding


def build_receipt(
    *,
    chain_root: Path,
    parent_before: str,
    parent_after: str,
    head: str,
    packet_sha: str,
    tripwire_triggered: bool,
    forced_outcome_branch: str | None,
    post_run_rc: dict[str, int],
    failed_post_run_stage: str | None,
) -> dict:
    transport_pass = False
    transport_sync_requested = False
    transport_manifest = chain_root / "box_artifact_transport.json"
    if transport_manifest.is_file():
        tm = json.loads(transport_manifest.read_text(encoding="utf-8"))
        transport_pass = bool(tm.get("artifact_transport_pass"))
        transport_sync_requested = bool(tm.get("sync_requested"))

    consumer_pass = False
    consumer_status = None
    consumer_manifest = chain_root / "box_consensus_consumer_audit.json"
    if consumer_manifest.is_file():
        cm = json.loads(consumer_manifest.read_text(encoding="utf-8"))
        consumer_pass = bool(cm.get("pass"))
        consumer_status = "pass" if consumer_pass else "fail"

    watcher_entry_status = None
    watcher_entry_issues: list[str] = []
    watcher_pipeline_eligible_entry = False
    watcher_n_flagged = None
    watcher_waived = False
    watcher_manifest_path = chain_root / "box_lane_overlap_manifest.json"
    if watcher_manifest_path.is_file():
        wm = json.loads(watcher_manifest_path.read_text(encoding="utf-8"))
        watcher_waived = bool(wm.get("waived"))
        watcher_n_flagged = wm.get("n_flagged")
        entries = wm.get("entries") or []
        if entries:
            entry = entries[0]
            watcher_entry_status = entry.get("status")
            watcher_entry_issues = list(entry.get("issues") or [])
            watcher_pipeline_eligible_entry = bool(entry.get("pipeline_eligible"))

    per_arm = _merge_per_arm_rows(chain_root)
    for row in per_arm:
        scratch = Path(row.get("scratch_root", ""))
        branch, dying_key, dying_idx, _ = _classify_tripwire_arm(scratch)
        row["dying_tensor_key"] = dying_key
        row["dying_tensor_index"] = dying_idx
        row["outcome_branch"] = branch

    all_arms_pass = bool(per_arm) and all(int(r.get("exit_code", 1)) == 0 and r.get("receipt_emitted") for r in per_arm)
    analyzer_rc = int(post_run_rc.get("analyzer", 0))

    if forced_outcome_branch:
        outcome_branch = forced_outcome_branch
    elif tripwire_triggered:
        outcome_branch = forced_outcome_branch or "v_partial_chain_fail"
        for row in per_arm:
            if row.get("outcome_branch") in {"i_tensor_pin", "ii_post_export_region", "iii_b2_regression"}:
                outcome_branch = str(row["outcome_branch"])
                break
    elif all_arms_pass and transport_pass and consumer_pass and analyzer_rc == 0 and watcher_entry_status in {
        "SERIAL_FALLBACK",
        "OVERLAP",
    }:
        outcome_branch = "vi_serial_plumbing_proven"
    elif all_arms_pass:
        outcome_branch = "vii_post_run_plumbing_fail"
        if failed_post_run_stage is None:
            if not transport_pass:
                failed_post_run_stage = "transport"
            elif analyzer_rc != 0:
                failed_post_run_stage = "analyzer"
            elif not consumer_pass:
                failed_post_run_stage = "consumer"
            elif watcher_entry_status in {"INELIGIBLE", "QUARANTINED_AFTER_CONSUMER_FAIL"}:
                failed_post_run_stage = "watcher"
    else:
        outcome_branch = "v_partial_chain_fail"

    if parent_before != parent_after and forced_outcome_branch is None:
        outcome_branch = "v_parent_hash_drift_block"
        if failed_post_run_stage is None:
            failed_post_run_stage = "parent_drift"

    serial_plumbing_proven = outcome_branch == "vi_serial_plumbing_proven"
    memory_pressure_read_id, site_d_action, site_d_deciding_metrics = _classify_site_d_read(
        per_arm,
        tripwire_triggered=tripwire_triggered,
        outcome_branch=outcome_branch,
    )

    return {
        "schema": "hrm_text_158_consensus_v1_chain_relaunch_receipt/v1",
        "chain_id": chain_root.name,
        "repo_head_sha": head,
        "plan_packet_sha256": packet_sha,
        "parent_hash_before": parent_before,
        "parent_hash_after": parent_after,
        "parent_hash_unchanged": parent_before == parent_after,
        "launch_env_exported": True,
        "code_currency_pass": True,
        "per_arm_rows": per_arm,
        "all_arms_pass": all_arms_pass,
        "tripwire_triggered": tripwire_triggered,
        "outcome_branch": outcome_branch,
        "memory_pressure_read_id": memory_pressure_read_id,
        "site_d_action": site_d_action,
        "site_d_deciding_metrics": site_d_deciding_metrics,
        "failed_post_run_stage": failed_post_run_stage,
        "transport_log_path": str(chain_root / "producer_science_chain.log"),
        "transport_sync_requested": transport_sync_requested,
        "artifact_transport_pass": transport_pass,
        "serial_plumbing_proven": serial_plumbing_proven,
        "watcher_entry_status": watcher_entry_status,
        "watcher_entry_issues": watcher_entry_issues,
        "watcher_pipeline_eligible_per_entry": watcher_pipeline_eligible_entry,
        "watcher_n_flagged": watcher_n_flagged,
        "watcher_waived": watcher_waived,
        "watcher_overlap_manifest_path": str(watcher_manifest_path) if watcher_manifest_path.is_file() else None,
        "consumer_audit_pass": consumer_pass,
        "consumer_terminal_status": consumer_status,
        "analyzer_exit_code": analyzer_rc,
        "post_run_stage_rc": post_run_rc,
        "draft_under_hold_disposition": "plumbing_liveness_smoke_only_not_verdict",
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chain-root", type=Path, required=True)
    ap.add_argument("--parent-before", required=True)
    ap.add_argument("--parent-after", required=True)
    ap.add_argument("--head", required=True)
    ap.add_argument("--packet-sha", required=True)
    ap.add_argument("--tripwire-triggered", choices=("true", "false"), default="false")
    ap.add_argument("--forced-outcome-branch", default="")
    ap.add_argument("--failed-post-run-stage", default="")
    ap.add_argument("--transport-rc", type=int, default=0)
    ap.add_argument("--analyzer-rc", type=int, default=0)
    ap.add_argument("--consumer-rc", type=int, default=0)
    ap.add_argument("--watcher-rc", type=int, default=0)
    args = ap.parse_args(argv)

    post_run_rc = {
        "transport": args.transport_rc,
        "analyzer": args.analyzer_rc,
        "consumer": args.consumer_rc,
        "watcher": args.watcher_rc,
    }
    failed = args.failed_post_run_stage or None
    if failed is None:
        if args.transport_rc != 0:
            failed = "transport"
        elif args.analyzer_rc != 0:
            failed = "analyzer"
        elif args.consumer_rc != 0:
            failed = "consumer"
        elif args.watcher_rc != 0:
            failed = "watcher"

    receipt = build_receipt(
        chain_root=args.chain_root,
        parent_before=args.parent_before,
        parent_after=args.parent_after,
        head=args.head,
        packet_sha=args.packet_sha,
        tripwire_triggered=args.tripwire_triggered == "true",
        forced_outcome_branch=args.forced_outcome_branch or None,
        post_run_rc=post_run_rc,
        failed_post_run_stage=failed,
    )
    out = args.chain_root / "chain_terminal_receipt.json"
    out.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"chain_terminal_receipt": str(out), "outcome_branch": receipt["outcome_branch"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
