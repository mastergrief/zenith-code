from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.d_recompute_input_manifest_bind import (
    V2_RECONCILED_ALLOWLIST,
    build_input_manifest,
    compute_spec_sha256,
    load_packet_spec,
)
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
V2_DRAFT = (
    REPO_ROOT
    / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_draft.json"
)
V2_REPLAY = (
    REPO_ROOT
    / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_replay_commands.json"
)
V2_COMMITTED_PACKET_REL = (
    "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v2_draft.json"
)


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


# ---------------------------------------------------------------------------
# STEP 1: two-phase input-manifest bind drift guard (GAP-A/GAP-B integration)
# ---------------------------------------------------------------------------


def _default_v2_records() -> list[dict]:
    return [
        _make_v2_log_record(step=1, state_key="tiny.proj", flip_positions={0}),
        _make_v2_log_record(step=1, state_key="other.proj", flip_positions=set()),
        _make_v2_log_record(step=2, state_key="tiny.proj", flip_positions=set()),
        _make_v2_log_record(step=2, state_key="other.proj", flip_positions=set()),
    ]


def _v2_spec(run_id: str = "fixture_run_v2") -> dict:
    spec = {
        "spec_schema": "hrm_text_158_d_recompute_input_manifest_spec/v0",
        "run_id": run_id,
        "packet_revision": PACKET_REVISION_V2,
        "artifact_allowlist": list(V2_RECONCILED_ALLOWLIST),
        "log_artifact_rel": "d_recompute_window_diagnostic/recompute_window_log.jsonl",
        "diagnostic_receipt_rel": "d_recompute_window_diagnostic/receipt.json",
        "calibrated_selector_manifest_rel": "prelaunch/calibrated_selector_manifest.json",
        "row_count_check": {
            "mode": "equals_steps_x_keys",
            "log_artifact_rel": "d_recompute_window_diagnostic/recompute_window_log.jsonl",
        },
        "selector_internal_sha_cross_check": True,
        "selector_log_key_alignment_check": True,
    }
    spec["spec_sha256"] = compute_spec_sha256(spec)
    return spec


def _v2_packet_with_spec(run_id: str = "fixture_run_v2") -> dict:
    return {
        "packet_revision": PACKET_REVISION_V2,
        "run_id": run_id,
        "expected_native_input_manifest_spec": _v2_spec(run_id),
    }


def _build_full_v2_fixture(base: Path, *, records: list[dict] | None = None) -> dict:
    """run_root carrying ALL v2-allowlisted artifacts, aligned for a clean pass."""
    if records is None:
        records = _default_v2_records()
    scratch = base / "d_recompute_window_diagnostic"
    log_path = scratch / "recompute_window_log.jsonl"
    _write_jsonl(log_path, records)
    selector = _calibrated_selector_manifest()
    _write_json(base / "prelaunch" / "calibrated_selector_manifest.json", selector.to_dict())
    _write_json(
        scratch / "receipt.json",
        {
            "d_recompute_window_instrumentation_enabled": True,
            "d_recompute_window_log_path": str(log_path),
            "steps_completed": max(int(r["step"]) for r in records),
            "d_recompute_selector_manifest_sha256": selector.manifest_sha256,
        },
    )
    for name in (
        "calibration_prepass_receipt.json",
        "scale_smoke_receipt.json",
        "post_confirmation_hygiene_receipt.json",
        "parent_checkpoint_rehash.json",
        "parent_checkpoint_rehash_after_calibration_warmup.json",
        "parent_checkpoint_rehash_after_scale_smoke.json",
        "parent_checkpoint_rehash_after_confirmation.json",
    ):
        _write_json(base / "prelaunch" / name, {"pass": True, "artifact": name})
    return {"run_root": base, "records": records, "selector": selector, "log_path": log_path}


def _run_classifier(run_root: Path, packet_path: Path, input_manifest: Path | None):
    cmd = [
        sys.executable,
        str(HELPER_SCRIPT),
        "--run-root",
        str(run_root),
        "--packet",
        str(packet_path),
    ]
    if input_manifest is not None:
        cmd += ["--input-manifest", str(input_manifest)]
    return subprocess.run(
        cmd,
        cwd=str(REPO_ROOT),
        env={**os.environ, "PYTHONPATH": "."},
        capture_output=True,
        text=True,
    )


def _blocked_receipt(run_root: Path) -> dict:
    receipt = json.loads((run_root / "classifier_receipt.json").read_text(encoding="utf-8"))
    assert receipt["primary_classifier"] == CLASSIFIER_INPUT_DRIFT_BLOCKED
    return receipt


def test_compute_spec_sha256_order_independent_excludes_self() -> None:
    a = {"b": 1, "a": 2, "spec_sha256": "ignored"}
    b = {"a": 2, "b": 1}
    assert compute_spec_sha256(a) == compute_spec_sha256(b)


def test_input_manifest_bind_excludes_nonallowlisted_junk(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    (base / "prelaunch" / "junk_attacker.json").write_text("{}", encoding="utf-8")
    (base / "d_recompute_window_diagnostic" / "extra_present.json").write_text("{}", encoding="utf-8")
    manifest = build_input_manifest(base, _v2_packet_with_spec())
    assert set(manifest["artifacts"].keys()) == set(V2_RECONCILED_ALLOWLIST)
    assert "prelaunch/junk_attacker.json" not in manifest["artifacts"]
    assert "driver_summary.json" not in manifest["artifacts"]


def test_input_manifest_bind_fails_closed_on_missing_artifact(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    (base / "prelaunch" / "scale_smoke_receipt.json").unlink()
    with pytest.raises(ValueError, match="missing_allowlisted_artifacts"):
        build_input_manifest(base, _v2_packet_with_spec())


def test_v2_exact_command_replay_succeeds_without_skip_drift_check(tmp_path: Path) -> None:
    replay = json.loads(V2_REPLAY.read_text(encoding="utf-8"))
    bind_tmpl = replay["postrun_input_manifest_bind_command"]
    postrun_tmpl = replay["postrun_command"]
    assert "--skip-input-drift-check" not in postrun_tmpl

    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, _v2_packet_with_spec())

    bind_cmd = bind_tmpl.replace("{run_root}", str(base)).replace(
        V2_COMMITTED_PACKET_REL, str(packet_path)
    )
    postrun_cmd = postrun_tmpl.replace("{run_root}", str(base)).replace(
        V2_COMMITTED_PACKET_REL, str(packet_path)
    )
    env = {**os.environ, "PYTHONPATH": "."}

    p1 = subprocess.run(["bash", "-c", bind_cmd], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert p1.returncode == 0, p1.stdout + p1.stderr
    assert (base / "prelaunch" / "postrun_input_manifest.json").is_file()

    p2 = subprocess.run(["bash", "-c", postrun_cmd], cwd=str(REPO_ROOT), env=env, capture_output=True, text=True)
    assert p2.returncode == 0, p2.stdout + p2.stderr
    receipt = json.loads((base / "classifier_receipt.json").read_text(encoding="utf-8"))
    assert receipt["primary_classifier"] != CLASSIFIER_INPUT_DRIFT_BLOCKED
    assert receipt["schema"] == CLASSIFIER_RECEIPT_SCHEMA_V2
    for key in (
        "horizon_growth",
        "acc_sizing",
        "in_vivo_validation",
        "arc2b_verdict",
        "final_sizing_verdict",
        "final_verdict_scope",
    ):
        assert key in receipt, key
    log_rel = "d_recompute_window_diagnostic/recompute_window_log.jsonl"
    selector_rel = "prelaunch/calibrated_selector_manifest.json"
    assert receipt["input_artifact_hashes"][log_rel]["sha256"]
    assert receipt["input_artifact_hashes"][log_rel]["jsonl_row_count"] == 4
    assert receipt["input_artifact_hashes"][selector_rel]["selector_internal_manifest_sha256"]


def test_v2_missing_input_manifest_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, _v2_packet_with_spec())
    proc = _run_classifier(base, packet_path, input_manifest=None)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any("missing_input_manifest" in f for f in receipt["drift_failures"])
    assert "analysis" not in receipt


def test_v2_missing_packet_spec_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet_path = tmp_path / "packet_v2_nospec.json"
    _write_json(packet_path, {"packet_revision": PACKET_REVISION_V2, "run_id": "fixture_run_v2"})
    bogus_manifest = tmp_path / "im.json"
    _write_json(bogus_manifest, {"artifacts": {}})
    proc = _run_classifier(base, packet_path, input_manifest=bogus_manifest)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any("missing_packet_spec" in f for f in receipt["drift_failures"])


def test_v2_input_manifest_wrong_spec_sha_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    manifest["spec_sha256"] = "0" * 64
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert "spec_sha256_mismatch" in receipt["drift_failures"]


def test_v2_input_manifest_wrong_run_id_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    manifest["run_id"] = "different_run"
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert "run_id_mismatch" in receipt["drift_failures"]


def test_v2_missing_required_artifact_after_bind_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    (base / "prelaunch" / "scale_smoke_receipt.json").unlink()
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any(
        f == "missing_artifact:prelaunch/scale_smoke_receipt.json" for f in receipt["drift_failures"]
    )


def test_v2_row_count_mismatch_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    log_path = base / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    extra = _make_v2_log_record(step=3, state_key="tiny.proj", flip_positions=set())
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(extra) + "\n")
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any("row_count_mismatch" in f for f in receipt["drift_failures"])


def test_v2_calibrated_selector_manifest_hash_drift_blocked(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    selector_path = base / "prelaunch" / "calibrated_selector_manifest.json"
    payload = json.loads(selector_path.read_text(encoding="utf-8"))
    payload["coverage_tier"] = "MUTATED_AFTER_BIND"
    _write_json(selector_path, payload)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any("selector" in f or "sha256_mismatch" in f for f in receipt["drift_failures"])


def test_v2_manifest_extra_path_outside_allowlist_rejected(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    manifest["artifacts"]["prelaunch/attacker_extra.json"] = {"sha256": "0" * 64}
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert any("manifest_extra_paths" in f for f in receipt["drift_failures"])


def test_v2_replay_postrun_passes_input_manifest_not_skip() -> None:
    replay = json.loads(V2_REPLAY.read_text(encoding="utf-8"))
    assert "postrun_input_manifest_bind_command" in replay
    sequence = replay["launch_sequence"]
    assert "postrun_input_manifest_bind_command" in sequence
    assert sequence.index("postrun_input_manifest_bind_command") < sequence.index("postrun_command")
    assert sequence.index("postrun_input_manifest_bind_command") > sequence.index(
        "parent_checkpoint_rehash_after_confirmation_command"
    )
    postrun = replay["postrun_command"]
    assert "--input-manifest {run_root}/prelaunch/postrun_input_manifest.json" in postrun
    assert "--skip-input-drift-check" not in postrun
    bind = replay["postrun_input_manifest_bind_command"]
    assert "hrm_text_158_d_recompute_input_manifest_bind.py" in bind
    assert "--out {run_root}/prelaunch/postrun_input_manifest.json" in bind


def test_v2_committed_packet_spec_matches_reconciled_constant() -> None:
    packet = json.loads(V2_DRAFT.read_text(encoding="utf-8"))
    spec = load_packet_spec(packet)
    assert tuple(spec["artifact_allowlist"]) == V2_RECONCILED_ALLOWLIST
    assert "driver_summary.json" not in spec["artifact_allowlist"]
    assert compute_spec_sha256(spec) == spec["spec_sha256"]


# ---------------------------------------------------------------------------
# STEP-1b hardening: run_root-local log preference + fail-closed source checks
# ---------------------------------------------------------------------------

_DELETE = object()


def _set_diag_field(run_root: Path, **fields) -> None:
    receipt_path = run_root / "d_recompute_window_diagnostic" / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    for key, value in fields.items():
        if value is _DELETE:
            receipt.pop(key, None)
        else:
            receipt[key] = value
    _write_json(receipt_path, receipt)


def test_v2_reclassify_over_copy_reads_run_root_local_log(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    run_root_local = base / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    # Decoy "source" log with a single key -> if the classifier read the embedded
    # path it would see <2 keys and fail; reading run_root-local sees 2 keys.
    decoy = tmp_path / "source_box" / "recompute_window_log.jsonl"
    _write_jsonl(decoy, [_make_v2_log_record(step=1, state_key="only.proj", flip_positions=set())])
    _set_diag_field(base, d_recompute_window_log_path=str(decoy))
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, _v2_packet_with_spec())
    receipt = emit_d_recompute_window_classifier_receipt(
        run_root=base,
        packet_path=packet_path,
        skip_input_drift_check=True,
    )
    assert receipt["log_path"] == str(run_root_local)
    assert sorted(receipt["selected_state_keys"]) == ["other.proj", "tiny.proj"]
    assert receipt["primary_classifier"] != CLASSIFIER_INPUT_DRIFT_BLOCKED


def test_v2_fallback_to_embedded_log_path_when_run_root_local_absent(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    run_root_local = base / "d_recompute_window_diagnostic" / "recompute_window_log.jsonl"
    moved = tmp_path / "external_box" / "recompute_window_log.jsonl"
    moved.parent.mkdir(parents=True, exist_ok=True)
    moved.write_text(run_root_local.read_text(encoding="utf-8"), encoding="utf-8")
    run_root_local.unlink()
    _set_diag_field(base, d_recompute_window_log_path=str(moved))
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, _v2_packet_with_spec())
    receipt = emit_d_recompute_window_classifier_receipt(
        run_root=base,
        packet_path=packet_path,
        skip_input_drift_check=True,
    )
    assert receipt["log_path"] == str(moved)
    assert sorted(receipt["selected_state_keys"]) == ["other.proj", "tiny.proj"]


def test_v2_row_count_source_absent_fails_closed(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    _set_diag_field(base, steps_completed=_DELETE)
    packet = _v2_packet_with_spec()
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert "row_count_source_missing:steps_completed" in receipt["drift_failures"]


def test_v2_packet_toplevel_vs_spec_run_id_mismatch_fails_closed(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet["run_id"] = "different_top_level_run"
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert "packet_toplevel_run_id_vs_spec_mismatch" in receipt["drift_failures"]


def test_v2_packet_toplevel_vs_spec_revision_mismatch_fails_closed(tmp_path: Path) -> None:
    base = tmp_path / "run"
    _build_full_v2_fixture(base)
    packet = _v2_packet_with_spec()
    packet["packet_revision"] = "v2_rev_other_lifted"
    packet_path = tmp_path / "packet_v2.json"
    _write_json(packet_path, packet)
    manifest = build_input_manifest(base, packet)
    im = tmp_path / "im.json"
    _write_json(im, manifest)
    proc = _run_classifier(base, packet_path, input_manifest=im)
    assert proc.returncode != 0
    receipt = _blocked_receipt(base)
    assert "packet_toplevel_packet_revision_vs_spec_mismatch" in receipt["drift_failures"]
