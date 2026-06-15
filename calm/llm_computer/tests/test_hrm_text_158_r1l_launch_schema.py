"""R1-L-SCHEMA CPU tests (R1L-V1..V22) for launch runtime receipt validators."""
from __future__ import annotations

import hashlib
import inspect
import json
import runpy
import sys
from dataclasses import dataclass, replace
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.activation_relief import (
    R1_CPU_BASE_COMMIT_SHA,
    R1lLaunchProofAbort,
    R1lLaunchProofMeasurements,
    W6_PARENT_SHA256_PINNED,
    build_launch_runtime_backward_validation_receipt,
    compute_canonical_launch_artifact_sha256,
    compute_proof_env_hash_sha256,
    launch_runtime_backward_receipt_from_dict,
    run_r1l_gpu_launch_proof,
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
W6_PARENT_PATH = (
    REPO_ROOT
    / "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_"
    "anchorsv1r3_from_L0b_final_step01500.pt"
)
FROZEN_LAUNCH_ARGV_TAIL = [
    "--load-from",
    str(W6_PARENT_PATH),
    "--activation-relief-lossless-recompute-launch-proof",
    "--epochs",
    "1",
    "--n-train-cap",
    "8",
    "--batch-size",
    "8",
    "--max-len",
    "384",
    "--hidden-size",
    "512",
    "--n-layers",
    "8",
    "--num-heads",
    "4",
    "--expansion",
    "4",
    "--H-cycles",
    "2",
    "--L-cycles",
    "3",
    "--parent-consistency-weight",
    "1.0",
    "--retained-support",
    "L0b:0.01",
    "--retained-support-batch",
    "8",
    "--seed",
    "17",
]


def _valid_synthetic_measurements(**overrides) -> R1lLaunchProofMeasurements:
    base = R1lLaunchProofMeasurements(
        main_baseline_saved_tensor_count=20,
        main_recompute_saved_tensor_count=15,
        main_saved_tensor_payload_bytes_baseline=1000,
        main_saved_tensor_payload_bytes_recompute=800,
        main_internal_payload_tensor_count=0,
        main_recompute_checkpoint_fired=True,
        retained_side_in_scope=True,
        retained_side_baseline_saved_tensor_count=18,
        retained_side_recompute_saved_tensor_count=14,
        retained_side_internal_payload_tensor_count=0,
        retained_saved_tensor_payload_bytes_delta=400,
        retained_side_recompute_checkpoint_fired=True,
        loss_finite_main=True,
        loss_finite_retained=True,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
    )
    if overrides:
        return replace(base, **overrides)
    return base


def _proof_loader():
    return iter(
        [
            {
                "inputs": torch.zeros((2, 8), dtype=torch.long),
                "labels": torch.zeros((2, 8), dtype=torch.long),
                "sep_positions": torch.zeros((2,), dtype=torch.long),
            }
        ]
    )


def _proof_cfg() -> SimpleNamespace:
    return SimpleNamespace(
        n_layers=8,
        num_heads=4,
        expansion=4,
        H_cycles=2,
        L_cycles=3,
        half_layers=True,
        bp_min_steps=2,
        bp_max_steps=5,
        max_seq_len=384,
    )


def _setup_r1l_launch_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    log_path = tmp_path / "logs" / "launch.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_bytes(b"=== R1-L ancestry preflight ===\nRESULT=PASS\n")
    receipt_path = tmp_path / "receipts" / "r1l_launch_runtime_receipt.json"
    receipt_path.parent.mkdir(parents=True)
    w6_copy = tmp_path / "w6_parent_readonly.pt"
    w6_copy.write_bytes(W6_PARENT_PATH.read_bytes())
    manifest_path = tmp_path / "launch_manifest.json"
    manifest = {
        "r1_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "archive_created_at_utc": "2026-06-15T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
        "CUDA_VISIBLE_DEVICES": "0",
        "R1L_LAUNCH_RECEIPT_JSON": str(receipt_path),
        "R1L_LAUNCH_LOG": str(log_path),
        "R1L_W6_PARENT_PATH": str(w6_copy),
        "TORCH_CUDA_ALLOC_CONF": "",
        "CUBLAS_WORKSPACE_CONFIG": "",
    }
    env_path = tmp_path / "launch_env.json"
    env_path.write_text(json.dumps(env, sort_keys=True), encoding="utf-8")
    monkeypatch.setenv("R1L_ANCESTRY_VERIFIED", "1")
    monkeypatch.setenv("R1L_CLEAN_RUN_DIR_SHA256", "a" * 64)
    monkeypatch.setenv("R1L_LAUNCH_MANIFEST_JSON", str(manifest_path))
    monkeypatch.setenv("R1L_LAUNCH_ENV_JSON", str(env_path))
    monkeypatch.setenv("R1L_LAUNCH_RECEIPT_JSON", str(receipt_path))
    monkeypatch.setenv("R1L_LAUNCH_LOG", str(log_path))
    monkeypatch.setenv("R1L_W6_PARENT_PATH", str(w6_copy))
    return w6_copy


def _run_launch_proof_with_synthetic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    measurements: R1lLaunchProofMeasurements,
    w6_copy: Path | None = None,
):
    if w6_copy is None:
        w6_copy = _setup_r1l_launch_env(monkeypatch, tmp_path)
    monkeypatch.setenv("R1L_GPU_UUID", "gpu-uuid-test")
    return run_r1l_gpu_launch_proof(
        model=object(),
        parent_model=None,
        loader=_proof_loader(),
        device=torch.device("cpu"),
        hidden_size=512,
        cfg=_proof_cfg(),
        active_supports=[
            {
                "name": "L0b",
                "weight": 0.01,
                "hash": "abc123",
                "count": 230,
            }
        ],
        parent_consistency_temp=1.0,
        epochs=1,
        proof_command_argv=("pytest", "launch"),
        w6_parent_path=str(w6_copy),
        gather_retained_parent_response_logits=lambda *_args, **_kwargs: None,
        parent_consistency_kl=lambda *_args, **_kwargs: None,
        parent_consistency_kl_response_positions=lambda *_args, **_kwargs: None,
        cuda_is_available_fn=lambda: True,
        measurement_runner=lambda: measurements,
    )


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
    launch_source = overrides.get("launch_source_commit_sha", R1_CPU_BASE_COMMIT_SHA)
    manifest, env = _manifest_and_env(launch_source=launch_source)
    kwargs = dict(
        launch_source_commit_sha=launch_source,
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


def test_r1li_v0_launch_proof_flag_default_off():
    from scripts.train_hrm_text_158 import train

    default = inspect.signature(train).parameters[
        "activation_relief_lossless_recompute_launch_proof"
    ].default
    assert default is False


def test_r1li_v1_stub_removed_no_exit_before_gpu_mint(
    tmp_path,
    monkeypatch,
    capsys,
):
    import os

    from calm.hrm_text_158.native_full_stack import activation_relief
    from calm.llm_computer.tests.test_hrm_text_158_phase_a_wiring import (
        TINY_ARCH,
        _build_tiny_broad_ckpt_blob,
    )
    from scripts import train_hrm_text_158 as trainer

    parent_path = tmp_path / "parent_R0.pt"
    torch.save(_build_tiny_broad_ckpt_blob(), parent_path)
    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    _setup_r1l_launch_env(monkeypatch, tmp_path)
    manifest = json.loads(
        Path(os.environ["R1L_LAUNCH_MANIFEST_JSON"]).read_text(encoding="utf-8")
    )
    env = json.loads(
        Path(os.environ["R1L_LAUNCH_ENV_JSON"]).read_text(encoding="utf-8")
    )
    log_sha = hashlib.sha256(
        Path(os.environ["R1L_LAUNCH_LOG"]).read_bytes()
    ).hexdigest()

    def _fake_launch_proof(**_kwargs):
        return build_launch_runtime_backward_validation_receipt(
            launch_source_commit_sha=R1_CPU_BASE_COMMIT_SHA,
            launch_manifest_embedded=manifest,
            proof_env_embedded=env,
            proof_command_argv=("pytest",),
            clean_run_dir_sha256="a" * 64,
            w6_parent_path=os.environ["R1L_W6_PARENT_PATH"],
            gpu_name="synthetic-gpu",
            gpu_uuid="gpu-uuid-test",
            driver_version="550.00",
            cuda_version="12.4",
            torch_version=torch.__version__,
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
            log_artifact_sha256=log_sha,
        )

    monkeypatch.setattr(activation_relief, "run_r1l_gpu_launch_proof", _fake_launch_proof)
    trainer.train(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=32,
        curriculum_n_heldout=8,
        load_from=str(parent_path),
        retained_support_profile=[("L0b", 0.01)],
        activation_relief_lossless_recompute_launch_proof=True,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        epochs=1,
        batch_size=2,
        device="cpu",
        n_train_cap=8,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    assert "EXITING before GPU launch mint" not in out
    assert "R1-L launch proof: EXITING before optimizer step" in out


def test_r1li_v2_no_mint_threshold_fail_writes_no_receipt(tmp_path, monkeypatch):
    w6_copy = _setup_r1l_launch_env(monkeypatch, tmp_path)
    receipt_path = tmp_path / "receipts" / "r1l_launch_runtime_receipt.json"
    with pytest.raises(R1lLaunchProofAbort, match="cuda peak reduction below threshold"):
        _run_launch_proof_with_synthetic(
            monkeypatch,
            tmp_path,
            measurements=_valid_synthetic_measurements(
                cuda_peak_allocated_bytes_recompute_median=63 * 1024 * 1024,
            ),
            w6_copy=w6_copy,
        )
    assert not receipt_path.exists()


def test_r1li_v3_no_mint_mechanism_fail_writes_no_receipt(tmp_path, monkeypatch):
    w6_copy = _setup_r1l_launch_env(monkeypatch, tmp_path)
    receipt_path = tmp_path / "receipts" / "r1l_launch_runtime_receipt.json"
    with pytest.raises(R1lLaunchProofAbort, match="main internal payload tensors observed"):
        _run_launch_proof_with_synthetic(
            monkeypatch,
            tmp_path,
            measurements=_valid_synthetic_measurements(
                main_internal_payload_tensor_count=1,
            ),
            w6_copy=w6_copy,
        )
    assert not receipt_path.exists()


def test_r1li_v4_happy_path_mock_cuda_receipt_validates(tmp_path, monkeypatch):
    receipt = _run_launch_proof_with_synthetic(
        monkeypatch,
        tmp_path,
        measurements=_valid_synthetic_measurements(),
    )
    validate_launch_runtime_backward_receipt(receipt)
    assert receipt.w6_parent_sha256_before == W6_PARENT_SHA256_PINNED


def test_r1li_v6_frozen_launch_argv_parses(monkeypatch):
    monkeypatch.chdir(REPO_ROOT)
    monkeypatch.setenv("R1L_ARGV_PARSE_PROBE", "1")
    monkeypatch.setattr(
        sys,
        "argv",
        ["train_hrm_text_158.py", *FROZEN_LAUNCH_ARGV_TAIL],
    )
    with pytest.raises(SystemExit) as exc:
        runpy.run_path(
            str(REPO_ROOT / "scripts/train_hrm_text_158.py"),
            run_name="__main__",
        )
    assert exc.value.code == 0


def test_r1li_v7_cuda_unavailable_aborts(tmp_path, monkeypatch):
    _setup_r1l_launch_env(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match="R1-L launch proof requires CUDA"):
        run_r1l_gpu_launch_proof(
            model=object(),
            parent_model=None,
            loader=_proof_loader(),
            device=torch.device("cpu"),
            hidden_size=512,
            cfg=_proof_cfg(),
            active_supports=[],
            parent_consistency_temp=1.0,
            epochs=1,
            proof_command_argv=("pytest",),
            w6_parent_path=str(tmp_path / "w6_parent_readonly.pt"),
            gather_retained_parent_response_logits=lambda *_a, **_k: None,
            parent_consistency_kl=lambda *_a, **_k: None,
            parent_consistency_kl_response_positions=lambda *_a, **_k: None,
            cuda_is_available_fn=lambda: False,
            measurement_runner=lambda: _valid_synthetic_measurements(
                retained_side_in_scope=False,
                retained_side_baseline_saved_tensor_count=0,
                retained_side_recompute_saved_tensor_count=0,
                retained_side_internal_payload_tensor_count=0,
                retained_saved_tensor_payload_bytes_delta=0,
                retained_side_recompute_checkpoint_fired=False,
                loss_finite_retained=True,
            ),
        )


def test_r1li_v8_ancestry_unset_refuses_mint(tmp_path, monkeypatch):
    _setup_r1l_launch_env(monkeypatch, tmp_path)
    monkeypatch.delenv("R1L_ANCESTRY_VERIFIED", raising=False)
    with pytest.raises(R1lLaunchProofAbort, match="R1L_ANCESTRY_VERIFIED must be 1"):
        run_r1l_gpu_launch_proof(
            model=object(),
            parent_model=None,
            loader=_proof_loader(),
            device=torch.device("cpu"),
            hidden_size=512,
            cfg=_proof_cfg(),
            active_supports=[],
            parent_consistency_temp=1.0,
            epochs=1,
            proof_command_argv=("pytest",),
            w6_parent_path=str(tmp_path / "w6_parent_readonly.pt"),
            gather_retained_parent_response_logits=lambda *_a, **_k: None,
            parent_consistency_kl=lambda *_a, **_k: None,
            parent_consistency_kl_response_positions=lambda *_a, **_k: None,
            cuda_is_available_fn=lambda: True,
            measurement_runner=lambda: _valid_synthetic_measurements(
                retained_side_in_scope=False,
                retained_side_baseline_saved_tensor_count=0,
                retained_side_recompute_saved_tensor_count=0,
                retained_side_internal_payload_tensor_count=0,
                retained_saved_tensor_payload_bytes_delta=0,
                retained_side_recompute_checkpoint_fired=False,
                loss_finite_retained=True,
            ),
        )
