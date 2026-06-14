from __future__ import annotations

import json
from pathlib import Path

from scripts.build_consensus_v1_chain_terminal_receipt import (
    _classify_site_d_read,
    _classify_tripwire_arm,
    build_receipt,
)


def _write_chain_fixtures(
    chain_root: Path,
    *,
    per_arm_metrics: list[dict],
    probes: list[dict],
    transport: dict | None = None,
    consumer: dict | None = None,
    watcher: dict | None = None,
) -> None:
    chain_root.mkdir(parents=True, exist_ok=True)
    (chain_root / "per_arm_metrics.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in per_arm_metrics) + "\n",
        encoding="utf-8",
    )
    (chain_root / "probe_results.jsonl").write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in probes) + "\n",
        encoding="utf-8",
    )
    if transport is not None:
        (chain_root / "box_artifact_transport.json").write_text(
            json.dumps(transport, indent=2) + "\n",
            encoding="utf-8",
        )
    if consumer is not None:
        (chain_root / "box_consensus_consumer_audit.json").write_text(
            json.dumps(consumer, indent=2) + "\n",
            encoding="utf-8",
        )
    if watcher is not None:
        (chain_root / "box_lane_overlap_manifest.json").write_text(
            json.dumps(watcher, indent=2) + "\n",
            encoding="utf-8",
        )


def _clean_arm_row(label: str, arm: str, scratch: Path, rss: int, mem_kib: int) -> dict:
    return {
        "label": label,
        "arm": arm,
        "scratch_root": str(scratch),
        "export_rss_peak_bytes": rss,
        "host_mem_min_available_kib_during_checkpoint_payload": mem_kib,
    }


def test_site_d_deferred_clean_a3b_headroom() -> None:
    per_arm = [
        _clean_arm_row("S44_ord44", "on", Path("/tmp/a"), 5_518_536_704, 13_107_200),
    ]
    read_id, action, _ = _classify_site_d_read(per_arm, tripwire_triggered=False, outcome_branch="vi_serial_plumbing_proven")
    assert read_id == "site_d_deferred"
    assert action == "defer"


def test_site_d_justified_death_with_rss_spike() -> None:
    per_arm = [
        {
            "label": "S44_ord44",
            "arm": "on",
            "exit_code": 1,
            "receipt_emitted": False,
            "export_rss_peak_bytes": 7_000_000_000,
            "host_mem_min_available_kib_during_checkpoint_payload": 12_000_000,
        }
    ]
    read_id, action, metrics = _classify_site_d_read(
        per_arm, tripwire_triggered=True, outcome_branch="i_tensor_pin"
    )
    assert read_id == "site_d_justified"
    assert action == "open_separate_gated_site_d_plan"
    assert metrics["death_arm_labels"] == ["S44_ord44/on"]


def test_site_d_undetermined_missing_metrics() -> None:
    read_id, action, _ = _classify_site_d_read([], tripwire_triggered=False, outcome_branch="vi_serial_plumbing_proven")
    assert read_id == "site_d_undetermined"
    assert action == "manual_review"


def test_tripwire_i_tensor_pin_from_run_log(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "run.log").write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "phase": "checkpoint_payload",
                        "event": "checkpoint_tensor_export_start",
                        "tensor_index": 44,
                        "tensor_key": "ord44.weight",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    branch, key, idx, all_done = _classify_tripwire_arm(scratch)
    assert branch == "i_tensor_pin"
    assert key == "ord44.weight"
    assert idx == 44
    assert all_done is False


def test_tripwire_ii_post_export_region_from_run_log(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    events = [
        {
            "phase": "checkpoint_payload",
            "event": "checkpoint_tensor_export_start",
            "tensor_index": 0,
            "tensor_key": "layer0.weight",
        },
        {
            "phase": "checkpoint_payload",
            "event": "checkpoint_tensor_export_done",
            "tensor_index": 0,
            "tensor_key": "layer0.weight",
        },
    ]
    (scratch / "run.log").write_text(
        "\n".join(json.dumps(ev) for ev in events) + "\n",
        encoding="utf-8",
    )
    branch, key, idx, all_done = _classify_tripwire_arm(scratch)
    assert branch == "ii_post_export_region"
    assert key is None
    assert idx is None
    assert all_done is True
    assert not (scratch / "receipt.json").is_file()


def test_tripwire_iii_b2_regression_from_run_log(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    (scratch / "run.log").write_text(
        "probe stderr: LIVENESS_FAILURE during checkpoint_payload\n",
        encoding="utf-8",
    )
    branch, key, idx, all_done = _classify_tripwire_arm(scratch)
    assert branch == "iii_b2_regression"
    assert key is None
    assert idx is None
    assert all_done is False


def test_build_receipt_serial_success(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain"
    scratch = chain_root / "S44_ord44" / "on"
    scratch.mkdir(parents=True)
    metrics = [_clean_arm_row("S44_ord44", "on", scratch, 5_518_536_704, 13_107_200)]
    probes = [
        {
            "label": "S44_ord44",
            "arm": "on",
            "exit_code": 0,
            "receipt": True,
            "scratch_root": str(scratch),
        }
    ]
    _write_chain_fixtures(
        chain_root,
        per_arm_metrics=metrics,
        probes=probes,
        transport={"artifact_transport_pass": True, "sync_requested": False},
        consumer={"pass": True},
        watcher={
            "waived": True,
            "n_flagged": 0,
            "entries": [{"status": "SERIAL_FALLBACK", "issues": [], "pipeline_eligible": False}],
        },
    )
    receipt = build_receipt(
        chain_root=chain_root,
        parent_before="abc",
        parent_after="abc",
        head="abb5535",
        packet_sha="packet",
        tripwire_triggered=False,
        forced_outcome_branch=None,
        post_run_rc={"transport": 0, "analyzer": 0, "consumer": 0, "watcher": 0},
        failed_post_run_stage=None,
    )
    assert receipt["outcome_branch"] == "vi_serial_plumbing_proven"
    assert receipt["memory_pressure_read_id"] == "site_d_deferred"
    assert receipt["site_d_action"] == "defer"
    assert receipt["site_d_deciding_metrics"]["max_export_rss_peak_bytes"] == 5_518_536_704


def test_build_receipt_post_run_fail(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain"
    scratch = chain_root / "S44_ord44" / "on"
    scratch.mkdir(parents=True)
    metrics = [_clean_arm_row("S44_ord44", "on", scratch, 5_000_000_000, 13_107_200)]
    probes = [
        {
            "label": "S44_ord44",
            "arm": "on",
            "exit_code": 0,
            "receipt": True,
            "scratch_root": str(scratch),
        }
    ]
    _write_chain_fixtures(
        chain_root,
        per_arm_metrics=metrics,
        probes=probes,
        transport={"artifact_transport_pass": False, "sync_requested": False},
        consumer={"pass": True},
        watcher={"waived": True, "n_flagged": 0, "entries": [{"status": "SERIAL_FALLBACK", "issues": []}]},
    )
    receipt = build_receipt(
        chain_root=chain_root,
        parent_before="abc",
        parent_after="abc",
        head="abb5535",
        packet_sha="packet",
        tripwire_triggered=False,
        forced_outcome_branch=None,
        post_run_rc={"transport": 1, "analyzer": 0, "consumer": 0, "watcher": 0},
        failed_post_run_stage="transport",
    )
    assert receipt["outcome_branch"] == "vii_post_run_plumbing_fail"
    assert receipt["failed_post_run_stage"] == "transport"


def test_build_receipt_parent_hash_drift(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain"
    scratch = chain_root / "S44_ord44" / "on"
    scratch.mkdir(parents=True)
    metrics = [_clean_arm_row("S44_ord44", "on", scratch, 5_518_536_704, 13_107_200)]
    probes = [
        {
            "label": "S44_ord44",
            "arm": "on",
            "exit_code": 0,
            "receipt": True,
            "scratch_root": str(scratch),
        }
    ]
    _write_chain_fixtures(chain_root, per_arm_metrics=metrics, probes=probes)
    receipt = build_receipt(
        chain_root=chain_root,
        parent_before="before",
        parent_after="after",
        head="abb5535",
        packet_sha="packet",
        tripwire_triggered=False,
        forced_outcome_branch=None,
        post_run_rc={"transport": 0, "analyzer": 0, "consumer": 0, "watcher": 0},
        failed_post_run_stage="parent_drift",
    )
    assert receipt["outcome_branch"] == "v_parent_hash_drift_block"
    assert receipt["parent_hash_unchanged"] is False
    assert receipt["failed_post_run_stage"] == "parent_drift"
