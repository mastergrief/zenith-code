from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    build_step_log_entry,
    default_production_replay_constants,
)
from scripts.hrm_text_158_d_recompute_postrun_classifier import (
    CLASSIFIER_INPUT_DRIFT_BLOCKED,
    PACKET_REVISION,
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
