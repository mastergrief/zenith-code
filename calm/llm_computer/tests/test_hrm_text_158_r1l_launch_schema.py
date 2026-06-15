"""R1-L-SCHEMA CPU tests (R1L-V1..V22) for launch runtime receipt validators."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from io import StringIO
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.activation_relief import (
    R1_CPU_BASE_COMMIT_SHA,
    W6_PARENT_SHA256_PINNED,
    build_launch_runtime_backward_validation_receipt,
    compute_canonical_launch_artifact_sha256,
    compute_proof_env_hash_sha256,
    launch_runtime_backward_receipt_from_dict,
    validate_launch_runtime_backward_artifacts,
    validate_launch_runtime_backward_receipt,
    verify_launch_ancestry_preflight,
)
from calm.hrm_text_158.native_full_stack.activation_relief import (
    _canonical_json_dumps,
)
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS,
    apply_live_r1_backward_wiring_surface_overrides,
    live_r1_backward_wiring_surfaces,
)
from calm.llm_computer.tests.test_hrm_text_158_full_sub2_runtime_readiness import (
    _mint_cpu_wiring_receipt,
    _mint_valid_launch_receipt,
    _post_p1_base_surfaces,
)
from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
    _mint_live_conversion_receipt,
)
from scripts.hrm_text_158_full_sub2_runtime_readiness import main as readiness_cli_main

REPO_ROOT = Path(__file__).resolve().parents[3]


def _manifest_and_env(launch_source: str = R1_CPU_BASE_COMMIT_SHA) -> tuple[dict[str, str], dict[str, str]]:
    manifest = {
        "r1_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-15T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ".",
        "R1L_LAUNCH_RECEIPT_JSON": "/tmp/run/receipts/r1l_launch_runtime_receipt.json",
        "R1L_LAUNCH_LOG": "/tmp/run/logs/r1l_launch.log",
        "R1L_W6_PARENT_PATH": "/tmp/run/artifacts/w6_parent_readonly.pt",
    }
    return manifest, env


def _mint_receipt(**overrides) -> object:
    manifest, env = _manifest_and_env()
    kwargs = dict(
        launch_source_commit_sha=R1_CPU_BASE_COMMIT_SHA,
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=("pytest", "launch"),
        clean_run_dir_sha256="a" * 64,
        w6_parent_path="/tmp/run/artifacts/w6_parent_readonly.pt",
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
        model_config_digest_sha256="b" * 64,
        proof_batch_digest_sha256="c" * 64,
        retained_support_digest_sha256="d" * 64,
        main_baseline_saved_tensor_count=20,
        main_recompute_saved_tensor_count=15,
        main_saved_tensor_payload_bytes_baseline=1000,
        main_saved_tensor_payload_bytes_recompute=800,
        retained_side_in_scope=True,
        retained_side_baseline_saved_tensor_count=18,
        retained_side_recompute_saved_tensor_count=14,
        retained_saved_tensor_payload_bytes_delta=400,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
        log_artifact_sha256=hashlib.sha256(b"r1l launch log bytes").hexdigest(),
    )
    kwargs.update(overrides)
    return build_launch_runtime_backward_validation_receipt(**kwargs)


def _artifact_bytes(receipt) -> tuple[bytes, bytes, bytes]:
    manifest_bytes = _canonical_json_dumps(dict(receipt.launch_manifest_embedded)).encode(
        "utf-8"
    )
    env_bytes = _canonical_json_dumps(dict(receipt.proof_env_embedded)).encode("utf-8")
    log_bytes = b"r1l launch log bytes"
    return manifest_bytes, env_bytes, log_bytes


def test_r1l_v1_valid_launch_receipt_roundtrip():
    receipt = _mint_valid_launch_receipt()
    validate_launch_runtime_backward_receipt(receipt)
    roundtrip = launch_runtime_backward_receipt_from_dict(receipt.to_dict())
    validate_launch_runtime_backward_receipt(roundtrip)


def test_r1l_v2_missing_gpu_field_rejects():
    payload = _mint_valid_launch_receipt().to_dict()
    payload.pop("gpu_uuid")
    with pytest.raises(ValueError, match="gpu_uuid"):
        launch_runtime_backward_receipt_from_dict(payload)


def test_r1l_v3_forged_cuda_threshold_met_rejects():
    receipt = replace(_mint_valid_launch_receipt(), cuda_peak_reduction_threshold_met=False)
    with pytest.raises(ValueError, match="cuda threshold met"):
        validate_launch_runtime_backward_receipt(receipt)


def test_r1l_v4_applier_post_p1_3_to_4_passes():
    receipt = _mint_valid_launch_receipt()
    readiness = live_r1_backward_wiring_surfaces(
        receipt,
        base_surfaces=_post_p1_base_surfaces(),
    )
    assert readiness.sub2_surface_count == 4
    assert readiness.ready_for_pre_full_stack_diagnostic is True
    assert readiness.ready_for_main_science is False


def test_r1l_v5_cpu_wiring_receipt_applier_rejects():
    with pytest.raises(ValueError, match="CPU production autograd wiring"):
        apply_live_r1_backward_wiring_surface_overrides(
            _mint_cpu_wiring_receipt(),
            base_surfaces=_post_p1_base_surfaces(),
        )


def test_r1l_v6_self_hash_tamper_rejects():
    receipt = _mint_valid_launch_receipt()
    tampered = replace(receipt, main_baseline_saved_tensor_count=999)
    with pytest.raises(ValueError, match="canonical_launch_artifact_sha256"):
        validate_launch_runtime_backward_receipt(tampered)


def test_r1l_v8_empty_w6_hash_rejects():
    with pytest.raises(ValueError, match="w6_parent_sha256_before"):
        _mint_receipt(w6_parent_sha256="")


def test_r1l_v9_wrong_w6_hash_rejects():
    with pytest.raises(ValueError, match="w6_parent_sha256_before mismatch"):
        _mint_receipt(w6_parent_sha256="0" * 64)


def test_r1l_v12_env_hash_changes_when_cuda_visible_devices_changes():
    receipt = _mint_valid_launch_receipt()
    env_a = dict(receipt.proof_env_embedded)
    env_b = {**env_a, "CUDA_VISIBLE_DEVICES": "0"}
    assert compute_proof_env_hash_sha256(env_a) != compute_proof_env_hash_sha256(env_b)


def test_r1l_v14_launch_source_descendant_preflight_passes():
    launch_source = verify_launch_ancestry_preflight(repo_root=REPO_ROOT)
    assert launch_source
    receipt = _mint_receipt(launch_source_commit_sha=launch_source)
    validate_launch_runtime_backward_receipt(receipt)


def test_r1l_v15_non_descendant_aborts_at_preflight_not_pure_object():
    with pytest.raises(ValueError, match="not a descendant"):
        verify_launch_ancestry_preflight(
            repo_root=REPO_ROOT,
            launch_source_commit_sha="0000000000000000000000000000000000000000",
        )


def test_r1l_v16_manifest_launch_source_mismatch_rejects():
    receipt = _mint_valid_launch_receipt()
    bad_manifest = dict(receipt.launch_manifest_embedded)
    bad_manifest["launch_source_commit_sha"] = "deadbeef" * 5
    tampered = replace(
        receipt,
        launch_manifest_embedded=bad_manifest,
        launch_source_commit_sha=R1_CPU_BASE_COMMIT_SHA,
    )
    with pytest.raises(ValueError, match="launch_source_commit_sha mismatch"):
        validate_launch_runtime_backward_receipt(tampered)


def test_r1l_v17_wrong_r1_cpu_base_rejects():
    with pytest.raises(ValueError, match="r1_cpu_base_commit_sha mismatch"):
        _mint_receipt(r1_cpu_base_commit_sha="f" * 40)


def test_r1l_v18_artifact_bound_missing_log_rejects():
    receipt = _mint_valid_launch_receipt()
    manifest_bytes, env_bytes, _ = _artifact_bytes(receipt)
    with pytest.raises(ValueError, match="launch log bytes are required"):
        validate_launch_runtime_backward_artifacts(
            receipt,
            launch_manifest_bytes=manifest_bytes,
            env_snapshot_bytes=env_bytes,
            log_bytes=None,
        )


def test_r1l_v19_artifact_bound_log_mismatch_rejects():
    receipt = _mint_valid_launch_receipt()
    manifest_bytes, env_bytes, _ = _artifact_bytes(receipt)
    with pytest.raises(ValueError, match="launch log bytes sha256 mismatch"):
        validate_launch_runtime_backward_artifacts(
            receipt,
            launch_manifest_bytes=manifest_bytes,
            env_snapshot_bytes=env_bytes,
            log_bytes=b"wrong log",
        )


def test_r1l_v20_artifact_bound_manifest_mismatch_rejects():
    receipt = _mint_valid_launch_receipt()
    _, env_bytes, log_bytes = _artifact_bytes(receipt)
    with pytest.raises(ValueError, match="launch manifest bytes sha256 mismatch"):
        validate_launch_runtime_backward_artifacts(
            receipt,
            launch_manifest_bytes=b"{}",
            env_snapshot_bytes=env_bytes,
            log_bytes=log_bytes,
        )


def test_r1l_v21_artifact_bound_env_mismatch_rejects():
    receipt = _mint_valid_launch_receipt()
    manifest_bytes, _, log_bytes = _artifact_bytes(receipt)
    with pytest.raises(ValueError, match="env snapshot proof_env_hash_sha256 mismatch"):
        validate_launch_runtime_backward_artifacts(
            receipt,
            launch_manifest_bytes=manifest_bytes,
            env_snapshot_bytes=b"{}",
            log_bytes=log_bytes,
        )


def test_r1l_v22_attestation_false_rejects():
    with pytest.raises(ValueError, match="ancestry_verified_at_launch_preflight"):
        _mint_receipt(ancestry_verified_at_launch_preflight=False)


def test_r1l_canonical_hash_omits_self_field():
    receipt = _mint_valid_launch_receipt()
    payload = receipt.to_dict()
    assert (
        compute_canonical_launch_artifact_sha256(payload)
        == receipt.canonical_launch_artifact_sha256
    )


def test_r1l_w6_pinned_constant():
    assert W6_PARENT_SHA256_PINNED.startswith("9b4e311a")


def test_r1l_applier_flips_exactly_backward_row():
    receipt = _mint_valid_launch_receipt()
    base = _post_p1_base_surfaces()
    base_by_id = {surface.surface_id: surface for surface in base}
    readiness = live_r1_backward_wiring_surfaces(receipt, base_surfaces=base)
    changed = {
        surface.surface_id
        for surface in readiness.surfaces
        if surface.classification != base_by_id[surface.surface_id].classification
    }
    assert changed == {SURFACE_BACKWARD_SAVED_TENSORS_TRANSIENTS}


def _write_live_r1_cli_inputs(
    tmp_path: Path,
    *,
    write_manifest: bool = True,
    write_env: bool = True,
    write_log: bool = True,
    env_overrides: dict[str, str] | None = None,
    omit_env_keys: tuple[str, ...] = (),
) -> tuple[Path, Path, Path]:
    run_root = tmp_path / "run"
    receipts_dir = run_root / "receipts"
    logs_dir = run_root / "logs"
    receipts_dir.mkdir(parents=True)
    logs_dir.mkdir(parents=True)
    receipt_json_path = receipts_dir / "r1l_launch_runtime_receipt.json"
    log_path = logs_dir / "r1l_launch.log"
    manifest_path = run_root / "launch_manifest.json"
    env_snapshot_path = run_root / "launch_env.json"

    manifest, env_base = _manifest_and_env()
    env = {
        **env_base,
        "R1L_LAUNCH_RECEIPT_JSON": str(receipt_json_path),
        "R1L_LAUNCH_LOG": str(log_path),
    }
    for key in omit_env_keys:
        env.pop(key, None)
    if env_overrides:
        env.update(env_overrides)

    receipt = build_launch_runtime_backward_validation_receipt(
        launch_source_commit_sha=R1_CPU_BASE_COMMIT_SHA,
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=("pytest", "launch"),
        clean_run_dir_sha256="a" * 64,
        w6_parent_path="/tmp/run/artifacts/w6_parent_readonly.pt",
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
        model_config_digest_sha256="b" * 64,
        proof_batch_digest_sha256="c" * 64,
        retained_support_digest_sha256="d" * 64,
        main_baseline_saved_tensor_count=20,
        main_recompute_saved_tensor_count=15,
        main_saved_tensor_payload_bytes_baseline=1000,
        main_saved_tensor_payload_bytes_recompute=800,
        retained_side_in_scope=True,
        retained_side_baseline_saved_tensor_count=18,
        retained_side_recompute_saved_tensor_count=14,
        retained_saved_tensor_payload_bytes_delta=400,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
        log_artifact_sha256=hashlib.sha256(b"r1l launch log bytes").hexdigest(),
    )
    receipt_json_path.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if write_manifest:
        manifest_path.write_text(_canonical_json_dumps(manifest), encoding="utf-8")
    if write_env:
        env_snapshot_path.write_text(_canonical_json_dumps(env), encoding="utf-8")
    if write_log:
        log_path.write_bytes(b"r1l launch log bytes")

    p1_path = tmp_path / "p1_live_conversion.json"
    p1_path.write_text(
        json.dumps(_mint_live_conversion_receipt().to_dict(), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return receipt_json_path, p1_path, tmp_path / "out.json"


def _run_live_r1_cli(
    receipt_json: Path,
    p1_json: Path,
    json_out: Path,
    *,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[int, str]:
    stdout = StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    exit_code = readiness_cli_main(
        [
            "--fixture",
            "live_r1_backward_launch",
            "--live-p1-receipt-json",
            str(p1_json),
            "--r1l-receipt-json",
            str(receipt_json),
            "--json-out",
            str(json_out),
        ]
    )
    return exit_code, stdout.getvalue()


@pytest.mark.parametrize(
    ("write_manifest", "write_env", "write_log", "match"),
    [
        (False, True, True, "launch manifest not found"),
        (True, False, True, "launch env snapshot not found"),
        (True, True, False, "launch log not found"),
    ],
)
def test_cli_live_r1_missing_artifact_file_exits_nonzero(
    tmp_path,
    monkeypatch,
    write_manifest,
    write_env,
    write_log,
    match,
):
    receipt_json, p1_json, json_out = _write_live_r1_cli_inputs(
        tmp_path,
        write_manifest=write_manifest,
        write_env=write_env,
        write_log=write_log,
    )
    with pytest.raises(SystemExit) as exc:
        _run_live_r1_cli(receipt_json, p1_json, json_out, monkeypatch=monkeypatch)
    assert exc.value.code != 0
    assert match in str(exc.value)
    assert not json_out.exists()


def test_cli_live_r1_missing_receipt_env_key_exits_nonzero(tmp_path, monkeypatch):
    receipt_json, p1_json, json_out = _write_live_r1_cli_inputs(
        tmp_path,
        omit_env_keys=("R1L_LAUNCH_RECEIPT_JSON",),
    )
    with pytest.raises(SystemExit) as exc:
        _run_live_r1_cli(receipt_json, p1_json, json_out, monkeypatch=monkeypatch)
    assert exc.value.code != 0
    assert "R1L_LAUNCH_RECEIPT_JSON" in str(exc.value)
    assert not json_out.exists()


def test_cli_live_r1_missing_log_env_key_exits_nonzero(tmp_path, monkeypatch):
    receipt_json, p1_json, json_out = _write_live_r1_cli_inputs(
        tmp_path,
        omit_env_keys=("R1L_LAUNCH_LOG",),
    )
    with pytest.raises(SystemExit) as exc:
        _run_live_r1_cli(receipt_json, p1_json, json_out, monkeypatch=monkeypatch)
    assert exc.value.code != 0
    assert "R1L_LAUNCH_LOG" in str(exc.value)
    assert not json_out.exists()


def test_cli_live_r1_with_all_artifacts_emits_diagnostic_ready(tmp_path, monkeypatch):
    receipt_json, p1_json, json_out = _write_live_r1_cli_inputs(tmp_path)
    exit_code, stdout = _run_live_r1_cli(
        receipt_json,
        p1_json,
        json_out,
        monkeypatch=monkeypatch,
    )
    assert exit_code == 0
    payload = json.loads(stdout)
    assert payload["ready_for_pre_full_stack_diagnostic"] is True
    assert payload["ready_for_main_science"] is False
    assert json_out.is_file()
