from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
    build_step_log_entry,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    COVERAGE_TIER_REPRESENTATIVE,
    STRESS_TAIL_POLICY_HORIZON_FIXED,
    StratifiedSelectorManifest,
)
from scripts.hrm_text_158_d_recompute_postrun_classifier import (
    CLASSIFIER_D_RECOMPUTE_IN_VIVO_INCONCLUSIVE,
    CLASSIFIER_INPUT_DRIFT_BLOCKED,
    CLASSIFIER_RECEIPT_SCHEMA,
    CLASSIFIER_RECEIPT_SCHEMA_V2,
    PACKET_REVISION,
    PACKET_REVISION_V2,
    REPRODUCTION_MODE,
    REPRODUCTION_MODE_V2,
    emit_d_recompute_window_classifier_receipt,
    helper_script_sha256,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_JSON = (
    REPO_ROOT
    / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v1_replay_commands.json"
)
HELPER_SCRIPT = REPO_ROOT / "scripts/hrm_text_158_d_recompute_postrun_classifier.py"


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")


def _minimal_log_records() -> list[dict]:
    replay = default_production_replay_constants()
    records: list[dict] = []
    for step, (key, vote) in enumerate(
        [
            ("tiny.proj", 1),
            ("tiny.proj", 2),
            ("other.proj", 1),
            ("other.proj", 2),
        ],
        start=1,
    ):
        acc_before = [24 * (step - 1)]
        acc_after = [24 * step]
        records.append(
            build_step_log_entry(
                step=step if step <= 2 else step - 2,
                state_key=key,
                replay_constants=replay,
                acc_before=acc_before,
                acc_after=acc_after,
                q_before=[0],
                q_after=[0],
                vote_lanes=[vote],
                lane_indices=[0],
                cap_order_digest="test",
                applied_order_digest="test",
                vote_source_digest="test",
            )
        )
    return records


def _build_fixture_run_root(base: Path) -> dict:
    scratch = base / "d_recompute_window_diagnostic"
    log_path = scratch / "recompute_window_log.jsonl"
    records = _minimal_log_records()
    _write_jsonl(log_path, records)
    _write_json(
        scratch / "receipt.json",
        {
            "d_recompute_window_instrumentation_enabled": True,
            "d_recompute_window_log_path": str(log_path),
            "steps_completed": 2,
        },
    )
    _write_json(base / "driver_summary.json", {"phase": "d-recompute-window-feasibility"})
    prelaunch = base / "prelaunch"
    for name in (
        "post_confirmation_hygiene_receipt.json",
        "parent_checkpoint_rehash.json",
        "parent_checkpoint_rehash_after_scale_smoke.json",
        "parent_checkpoint_rehash_after_confirmation.json",
    ):
        _write_json(prelaunch / name, {"pass": True, "artifact": name})

    manifest: dict[str, dict] = {}
    for rel in (
        "d_recompute_window_diagnostic/receipt.json",
        "d_recompute_window_diagnostic/recompute_window_log.jsonl",
        "driver_summary.json",
        "prelaunch/post_confirmation_hygiene_receipt.json",
        "prelaunch/parent_checkpoint_rehash.json",
        "prelaunch/parent_checkpoint_rehash_after_scale_smoke.json",
        "prelaunch/parent_checkpoint_rehash_after_confirmation.json",
    ):
        path = base / rel
        entry: dict = {"sha256": _sha256_file(path)}
        if rel.endswith("recompute_window_log.jsonl"):
            entry["jsonl_row_count"] = len(records)
        manifest[rel] = entry
    return {"manifest": manifest}


def _build_temp_packet(tmp_path: Path, manifest: dict[str, dict], *, drift: bool = False) -> Path:
    manifest_copy = dict(manifest)
    if drift:
        first_key = next(iter(manifest_copy))
        manifest_copy[first_key] = dict(manifest_copy[first_key])
        manifest_copy[first_key]["sha256"] = "0" * 64
    packet = {
        "packet_revision": PACKET_REVISION,
        "run_id": "fixture_run",
        "expected_native_input_manifest": manifest_copy,
    }
    packet_path = tmp_path / "packet.json"
    _write_json(packet_path, packet)
    return packet_path


def _build_v2_temp_packet(tmp_path: Path) -> Path:
    packet = {
        "packet_revision": PACKET_REVISION_V2,
        "run_id": "fixture_run_v2",
    }
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    return packet_path


def _make_v2_log_record(
    *,
    step: int,
    state_key: str,
    flip_positions: set[int],
) -> dict:
    lane_indices = [0]
    flip_lanes = [index in flip_positions for index in range(len(lane_indices))]
    entry = build_step_log_entry(
        step=int(step),
        state_key=str(state_key),
        replay_constants=default_production_replay_constants(),
        acc_before=[0],
        acc_after=[1 if flip_lanes[0] else 0],
        q_before=[0],
        q_after=[0],
        vote_lanes=[1],
        lane_indices=lane_indices,
        flip_residual_applied_lanes=flip_lanes,
        flip_direction_lanes=[1 if flip_lanes[0] else None],
        backlog_depth=0,
        global_rate_cap_accepted_count=1,
        global_rate_cap_deferred_count=0,
    )
    entry["schema_version"] = D_RECOMPUTE_WINDOW_SCHEMA_VERSION
    return entry


def _calibrated_selector_manifest() -> StratifiedSelectorManifest:
    body = {
        "schema_version": "hrm_text_158_stratified_selector_manifest/v0",
        "coverage_tier": COVERAGE_TIER_REPRESENTATIVE,
        "selected_key_count": 2,
        "stratum_weights": {"other.proj": 0.5, "tiny.proj": 0.5},
        "manifest_spec": {
            "stress_tail_policy": STRESS_TAIL_POLICY_HORIZON_FIXED,
            "measurement_start_step": 1,
        },
        "entries": [
            {
                "state_key": "tiny.proj",
                "level": "H",
                "layer_idx": 0,
                "role": "attn_q",
                "depth_tercile": "early",
                "numel_band": "small",
                "numel": 1,
                "uniform_lanes": [0],
                "stress_tail_lanes": [0],
                "lane_indices": [0],
                "stratum_weight": 0.5,
            },
            {
                "state_key": "other.proj",
                "level": "H",
                "layer_idx": 1,
                "role": "attn_q",
                "depth_tercile": "early",
                "numel_band": "small",
                "numel": 1,
                "uniform_lanes": [0],
                "stress_tail_lanes": [0],
                "lane_indices": [0],
                "stratum_weight": 0.5,
            },
        ],
    }
    digest = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return StratifiedSelectorManifest.from_dict({**body, "manifest_sha256": digest})


def _build_v2_fixture_run_root(
    base: Path,
    *,
    records: list[dict],
    write_calibrated_manifest: bool = True,
) -> None:
    scratch = base / "d_recompute_window_diagnostic"
    log_path = scratch / "recompute_window_log.jsonl"
    _write_jsonl(log_path, records)
    _write_json(
        scratch / "receipt.json",
        {
            "d_recompute_window_instrumentation_enabled": True,
            "d_recompute_window_log_path": str(log_path),
            "steps_completed": max(int(record["step"]) for record in records),
        },
    )
    _write_json(base / "driver_summary.json", {"phase": "d-recompute-window-feasibility"})
    if write_calibrated_manifest:
        manifest = _calibrated_selector_manifest()
        _write_json(
            base / "prelaunch" / "calibrated_selector_manifest.json",
            manifest.to_dict(),
        )


def test_static_production_postrun_command_shape() -> None:
    replay = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    postrun = replay["postrun_command"]
    assert "bash -c 'bash -c" not in postrun
    assert postrun.count("bash -c") == 1
    assert "--skip-input-drift-check" not in postrun
    assert "hrm_text_158_d_recompute_postrun_classifier.py" in postrun
    assert "PYTHONPATH=. timeout 900" in postrun


def test_emit_timeout_receipt_fallback_fired() -> None:
    with tempfile.TemporaryDirectory(prefix="d_postrun_timeout_") as tmp:
        run_root = Path(tmp)
        proc = subprocess.run(
            [
                sys.executable,
                str(HELPER_SCRIPT),
                "--run-root",
                str(run_root),
                "--emit-timeout-receipt",
                "--timeout-seconds",
                "900",
            ],
            cwd=str(REPO_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        receipt = json.loads((run_root / "classifier_receipt.json").read_text(encoding="utf-8"))
        assert receipt["fallback_fired"] is True
        assert receipt["primary_classifier"] == "OBSERVER_TOO_EXPENSIVE"
        assert receipt["postrun_timeout_classification"] == "OBSERVER_TOO_EXPENSIVE"


def test_input_drift_blocked_nonzero() -> None:
    with tempfile.TemporaryDirectory(prefix="d_postrun_drift_") as tmp:
        run_root = Path(tmp) / "run"
        fixture = _build_fixture_run_root(run_root)
        packet_path = _build_temp_packet(Path(tmp), fixture["manifest"], drift=True)
        proc = subprocess.run(
            [
                sys.executable,
                str(HELPER_SCRIPT),
                "--run-root",
                str(run_root),
                "--packet",
                str(packet_path),
            ],
            cwd=str(REPO_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        receipt_path = run_root / "classifier_receipt.json"
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        assert receipt["primary_classifier"] == CLASSIFIER_INPUT_DRIFT_BLOCKED
        assert "analysis" not in receipt


def test_exact_command_replay_from_replay_json() -> None:
    replay = json.loads(REPLAY_JSON.read_text(encoding="utf-8"))
    postrun_template = replay["postrun_command"]

    with tempfile.TemporaryDirectory(prefix="d_postrun_exact_") as tmp:
        run_root = Path(tmp) / "run"
        fixture = _build_fixture_run_root(run_root)
        packet_path = _build_temp_packet(Path(tmp), fixture["manifest"])

        postrun = postrun_template.replace("{run_root}", str(run_root))
        postrun = postrun.replace(
            "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v1_draft.json",
            str(packet_path),
        )

        proc = subprocess.run(
            ["bash", "-c", postrun],
            cwd=str(REPO_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stdout + proc.stderr
        assert "unexpected EOF" not in (proc.stdout + proc.stderr)
        receipt = json.loads((run_root / "classifier_receipt.json").read_text(encoding="utf-8"))
        assert receipt["fallback_fired"] is False
        assert receipt["packet_revision"] == PACKET_REVISION
        assert receipt["helper_script_sha256"] == helper_script_sha256()
        assert receipt["reproduction_mode"] == "postrun_only_over_native_rev3c_artifacts"
        assert receipt["input_artifact_hashes"]
        assert receipt["primary_classifier"] is not None
        assert "selected_state_keys" in receipt
        assert "numel_by_key" in receipt
        assert "jsonl_row_count" in receipt


def test_helper_script_sha256_self_consistent() -> None:
    assert helper_script_sha256() == _sha256_file(HELPER_SCRIPT)


def _assert_input_drift_blocked_marker(run_root: Path) -> None:
    receipt_path = run_root / "classifier_receipt.json"
    assert receipt_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["primary_classifier"] == CLASSIFIER_INPUT_DRIFT_BLOCKED
    assert receipt["fallback_fired"] is False
    assert "analysis" not in receipt


def test_missing_packet_path_blocked_nonzero() -> None:
    with tempfile.TemporaryDirectory(prefix="d_postrun_missing_packet_") as tmp:
        run_root = Path(tmp) / "run"
        _build_fixture_run_root(run_root)
        missing_packet = Path(tmp) / "does_not_exist_packet.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(HELPER_SCRIPT),
                "--run-root",
                str(run_root),
                "--packet",
                str(missing_packet),
            ],
            cwd=str(REPO_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        _assert_input_drift_blocked_marker(run_root)


def test_no_packet_arg_blocked_nonzero() -> None:
    with tempfile.TemporaryDirectory(prefix="d_postrun_no_packet_") as tmp:
        run_root = Path(tmp) / "run"
        _build_fixture_run_root(run_root)
        proc = subprocess.run(
            [
                sys.executable,
                str(HELPER_SCRIPT),
                "--run-root",
                str(run_root),
            ],
            cwd=str(REPO_ROOT),
            env={**dict(__import__("os").environ), "PYTHONPATH": "."},
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        _assert_input_drift_blocked_marker(run_root)


def test_v2_packet_emit_path_attaches_arc2b_blocks() -> None:
    records = [
        _make_v2_log_record(step=1, state_key="tiny.proj", flip_positions={0}),
        _make_v2_log_record(step=1, state_key="other.proj", flip_positions=set()),
        _make_v2_log_record(step=2, state_key="tiny.proj", flip_positions=set()),
        _make_v2_log_record(step=2, state_key="other.proj", flip_positions=set()),
    ]
    with tempfile.TemporaryDirectory(prefix="d_postrun_v2_emit_") as tmp:
        run_root = Path(tmp) / "run"
        _build_v2_fixture_run_root(run_root, records=records)
        packet_path = _build_v2_temp_packet(Path(tmp))
        receipt = emit_d_recompute_window_classifier_receipt(
            run_root=run_root,
            packet_path=packet_path,
            skip_input_drift_check=True,
        )
        assert receipt["schema"] == CLASSIFIER_RECEIPT_SCHEMA_V2
        assert receipt["packet_revision"] == PACKET_REVISION_V2
        assert receipt["reproduction_mode"] == REPRODUCTION_MODE_V2
        assert receipt["horizon_growth"] is not None
        assert receipt["acc_sizing"] is not None
        assert receipt["in_vivo_validation"] is not None
        assert receipt["arc2b_verdict"] is not None
        assert receipt["primary_classifier"] is not None
        assert "analysis" not in receipt


def test_v2_missing_calibrated_manifest_fail_closed() -> None:
    records = [
        _make_v2_log_record(step=1, state_key="tiny.proj", flip_positions=set()),
        _make_v2_log_record(step=1, state_key="other.proj", flip_positions=set()),
    ]
    with tempfile.TemporaryDirectory(prefix="d_postrun_v2_missing_manifest_") as tmp:
        run_root = Path(tmp) / "run"
        _build_v2_fixture_run_root(run_root, records=records, write_calibrated_manifest=False)
        packet_path = _build_v2_temp_packet(Path(tmp))
        with pytest.raises(FileNotFoundError, match="calibrated_selector_manifest"):
            emit_d_recompute_window_classifier_receipt(
                run_root=run_root,
                packet_path=packet_path,
                skip_input_drift_check=True,
            )


def test_v1_packet_legacy_path_unchanged() -> None:
    with tempfile.TemporaryDirectory(prefix="d_postrun_v1_legacy_") as tmp:
        run_root = Path(tmp) / "run"
        fixture = _build_fixture_run_root(run_root)
        packet_path = _build_temp_packet(Path(tmp), fixture["manifest"])
        receipt = emit_d_recompute_window_classifier_receipt(
            run_root=run_root,
            packet_path=packet_path,
            skip_input_drift_check=True,
        )
        assert receipt["schema"] == CLASSIFIER_RECEIPT_SCHEMA
        assert receipt["packet_revision"] == PACKET_REVISION
        assert receipt["reproduction_mode"] == REPRODUCTION_MODE
        assert "analysis" in receipt
        assert "arc2b_verdict" not in receipt
        assert "horizon_growth" not in receipt
        assert "in_vivo_validation" not in receipt


def test_v2_in_vivo_inconclusive_maps_classifier_verdict() -> None:
    records = []
    for step in (1, 2):
        for key in ("tiny.proj", "other.proj"):
            records.append(
                _make_v2_log_record(step=step, state_key=key, flip_positions={0})
            )
    with tempfile.TemporaryDirectory(prefix="d_postrun_v2_inconclusive_") as tmp:
        run_root = Path(tmp) / "run"
        _build_v2_fixture_run_root(run_root, records=records)
        packet_path = _build_v2_temp_packet(Path(tmp))
        receipt = emit_d_recompute_window_classifier_receipt(
            run_root=run_root,
            packet_path=packet_path,
            skip_input_drift_check=True,
        )
        assert receipt["primary_classifier"] == CLASSIFIER_D_RECOMPUTE_IN_VIVO_INCONCLUSIVE
        assert receipt["in_vivo_validation"]["in_vivo_verdict"] != "DOMINANCE_PROVEN"
