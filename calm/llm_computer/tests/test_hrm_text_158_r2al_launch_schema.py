"""R2-A-L launch runtime receipt validators and applier tests (§2.7)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.activation_relief import (
    build_launch_runtime_backward_validation_receipt,
)
from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
    CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS,
    LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_NON_CLAIMS,
    LaunchRuntimeActivationResidualsValidationReceipt,
    R2A_CPU_BASE_COMMIT_SHA,
    R2alLaunchProofMeasurements,
    build_launch_runtime_activation_residuals_validation_receipt,
    canonicalize_base_sub2_surface_ids,
    launch_runtime_activation_residuals_receipt_from_dict,
    validate_launch_runtime_activation_residuals_receipt,
    validate_r2al_live_base_preflight,
)
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
    RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
    RUNTIME_CLASS_SUB2,
    SURFACE_ACTIVATIONS_RESIDUALS,
    apply_live_activation_residuals_surface_overrides,
    current_repo_scaffold_surfaces,
    live_r1_backward_launch_surfaces,
)
from calm.llm_computer.tests.test_hrm_text_158_full_sub2_runtime_readiness import (
    _mint_r2a_cpu_m1_lossless_equiv_receipt,
    _mint_valid_launch_receipt,
    _post_p1_base_surfaces,
)
from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
    _mint_live_conversion_receipt,
)


def _valid_activation_relief_measurement() -> dict[str, object]:
    return {
        "peak_allocated_bytes": 56 * 1024 * 1024,
        "peak_reserved_bytes": 60 * 1024 * 1024,
        "wall_clock_per_step_seconds": 0.25,
        "max_safe_batch_size": 8,
        "effective_exposure_per_step": 3072,
    }


def _valid_synthetic_measurements(**overrides) -> R2alLaunchProofMeasurements:
    base = R2alLaunchProofMeasurements(
        m1_seam_handle_pack_count=4,
        m1_registered_seam_tensor_full_pack_count=0,
        m1_seam_remat_unpack_recompute_count_total=12,
        m1_saved_tensor_payload_bytes_delta=4096,
        m1_forbidden_closure_tensor_count_total=0,
        m1_registered_seam_tensor_in_closure_count=0,
        m1_no_hidden_bf16_proof_pass=True,
        m1_mechanism_proof_pass=True,
        loss_finite_main=True,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
        cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
        cuda_peak_reserved_bytes_delta_median=0,
        activation_relief_measurement=_valid_activation_relief_measurement(),
    )
    if overrides:
        return replace(base, **overrides)
    return base


def _write_receipt_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _setup_r2al_receipt_paths(tmp_path: Path) -> dict[str, Path]:
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    p1_path = tmp_path / "receipts" / "p1_live_conversion_receipt.json"
    r1l_path = tmp_path / "receipts" / "r1l_launch_runtime_receipt.json"
    _write_receipt_json(p1_path, p1_receipt.to_dict())
    _write_receipt_json(r1l_path, r1l_receipt.to_dict())
    return {"p1_path": p1_path, "r1l_path": r1l_path}


def _mint_valid_r2al_launch_receipt(tmp_path: Path) -> LaunchRuntimeActivationResidualsValidationReceipt:
    paths = _setup_r2al_receipt_paths(tmp_path)
    launch_source = R2A_CPU_BASE_COMMIT_SHA
    manifest = {
        "r2a_cpu_base_commit_sha": R2A_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-19T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ".",
        "R2AL_LAUNCH_RECEIPT_JSON": str(tmp_path / "receipts" / "r2al_launch_runtime_receipt.json"),
        "R2AL_LAUNCH_LOG": str(tmp_path / "logs" / "r2al_launch.log"),
        "R2AL_W6_PARENT_PATH": str(tmp_path / "artifacts" / "w6_parent_readonly.pt"),
        "R2AL_P1_RECEIPT_JSON": str(paths["p1_path"]),
        "R2AL_R1L_RECEIPT_JSON": str(paths["r1l_path"]),
    }
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "r2al_launch.log").write_bytes(b"R2-A-L launch proof log\n")
    return build_launch_runtime_activation_residuals_validation_receipt(
        launch_source_commit_sha=launch_source,
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=("pytest", "r2al-launch"),
        clean_run_dir_sha256="a" * 64,
        w6_parent_path=str(tmp_path / "artifacts" / "w6_parent_readonly.pt"),
        w6_parent_sha256="9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-r2al-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
        model_config_digest_sha256="b" * 64,
        proof_batch_digest_sha256="c" * 64,
        retained_support_digest_sha256=hashlib.sha256(b"[]").hexdigest(),
        p1_receipt=_mint_live_conversion_receipt(),
        r1l_receipt=_mint_valid_launch_receipt(),
        p1_receipt_path=paths["p1_path"],
        r1l_receipt_path=paths["r1l_path"],
        measurements=_valid_synthetic_measurements(),
        log_artifact_sha256=hashlib.sha256(b"R2-A-L launch proof log\n").hexdigest(),
    )


def test_r2al_receipt_roundtrip_and_validator(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    roundtrip = launch_runtime_activation_residuals_receipt_from_dict(receipt.to_dict())
    validate_launch_runtime_activation_residuals_receipt(roundtrip)
    assert tuple(roundtrip.base_sub2_surface_ids) == CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS


def test_r2al_validator_rejects_missing_r1l_base(tmp_path):
    paths = _setup_r2al_receipt_paths(tmp_path)
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ".",
        "R2AL_LAUNCH_RECEIPT_JSON": str(tmp_path / "receipts" / "r2al_launch_runtime_receipt.json"),
        "R2AL_LAUNCH_LOG": str(tmp_path / "logs" / "r2al_launch.log"),
        "R2AL_W6_PARENT_PATH": str(tmp_path / "artifacts" / "w6_parent_readonly.pt"),
        "R2AL_P1_RECEIPT_JSON": str(paths["p1_path"]),
        "R2AL_R1L_RECEIPT_JSON": str(tmp_path / "missing_r1l.json"),
    }
    from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
        load_r2al_base_receipts_from_env,
    )

    with pytest.raises(ValueError, match="R1-L receipt path not found"):
        load_r2al_base_receipts_from_env(env)


def test_r2al_validator_rejects_scaffold_only_base(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    scaffold = current_repo_scaffold_surfaces()
    scaffold_sub2 = canonicalize_base_sub2_surface_ids(
        surface.surface_id
        for surface in scaffold
        if surface.classification == RUNTIME_CLASS_SUB2
    )
    assert scaffold_sub2 != CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS
    forged = replace(
        receipt,
        base_sub2_surface_ids=scaffold_sub2,
        base_sub2_surface_count=len(scaffold_sub2),
    )
    with pytest.raises(ValueError, match="base_sub2_surface_ids must be canonical sorted"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_validator_rejects_wrong_sub2_id_set(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    wrong_ids = (
        "backward_saved_tensors_transients",
        "persistent_qacc_authority",
        "q_sidecar_vote_carrier",
    )
    forged = replace(receipt, base_sub2_surface_ids=wrong_ids, base_sub2_surface_count=3)
    with pytest.raises(ValueError, match="base_sub2_surface_ids must be canonical sorted"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_validator_rejects_activations_already_sub2(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    forged = replace(
        receipt,
        base_activations_residuals_classification=RUNTIME_CLASS_SUB2,
        base_activations_residuals_is_sub2=True,
    )
    with pytest.raises(ValueError, match="base_activations_residuals_is_sub2 must be false"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_applier_changes_exactly_one_surface(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    base = live_r1_backward_launch_surfaces(r1l_receipt, p1_receipt).surfaces
    base_by_id = {surface.surface_id: surface for surface in base}
    flipped = apply_live_activation_residuals_surface_overrides(receipt, base_surfaces=base)
    changed = {
        surface.surface_id
        for surface in flipped
        if surface.classification != base_by_id[surface.surface_id].classification
    }
    assert changed == {SURFACE_ACTIVATIONS_RESIDUALS}
    activations = next(
        surface for surface in flipped if surface.surface_id == SURFACE_ACTIVATIONS_RESIDUALS
    )
    assert activations.classification == RUNTIME_CLASS_SUB2


def test_r2al_applier_rejects_fail_closed_receipt():
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        build_activation_residuals_fail_closed_receipt,
    )
    from calm.llm_computer.tests.test_hrm_text_158_activation_relief import (
        _activation_residual_live_tensor_proof,
    )

    events, zL_init_observation = _activation_residual_live_tensor_proof()
    fixture_receipt = build_activation_residuals_fail_closed_receipt(
        seam_events=events,
        zL_init_observation=zL_init_observation,
    )
    with pytest.raises(ValueError, match="fixture activation/residual fail-closed"):
        apply_live_activation_residuals_surface_overrides(fixture_receipt)


def test_r2al_applier_rejects_r1l_receipt():
    with pytest.raises(ValueError, match="R1-L backward launch receipt"):
        apply_live_activation_residuals_surface_overrides(_mint_valid_launch_receipt())


def test_r2al_applier_rejects_cpu_lossless_equiv_receipt():
    with pytest.raises(ValueError, match="CPU lossless equivalence receipt"):
        apply_live_activation_residuals_surface_overrides(_mint_r2a_cpu_m1_lossless_equiv_receipt())


def test_r2al_live_base_preflight_canonicalizes_sorted_sub2_ids(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    validate_r2al_live_base_preflight(
        receipt,
        p1_receipt=p1_receipt,
        r1l_receipt=r1l_receipt,
        p1_receipt_path=_setup_r2al_receipt_paths(tmp_path)["p1_path"],
        r1l_receipt_path=_setup_r2al_receipt_paths(tmp_path)["r1l_path"],
    )
    base = live_r1_backward_launch_surfaces(r1l_receipt, p1_receipt)
    activations = next(
        surface
        for surface in base.surfaces
        if surface.surface_id == SURFACE_ACTIVATIONS_RESIDUALS
    )
    assert activations.classification == RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC
    assert base.sub2_surface_count == 4


def test_r2al_validator_rejects_pass_a_fail_b_cuda(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    forged = replace(
        receipt,
        cuda_peak_allocated_bytes_recompute_median=receipt.cuda_peak_allocated_bytes_baseline_median,
        cuda_peak_allocated_bytes_delta_median=0,
        cuda_peak_reduction_threshold_met=False,
        gpu_memory_measurement_pass=False,
        launch_runtime_validation_pass=False,
    )
    with pytest.raises(ValueError, match="launch_runtime_validation_pass"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_validator_rejects_r1l_non_claims(tmp_path):
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        LAUNCH_RUNTIME_NON_CLAIMS,
    )

    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    forged = replace(receipt, non_claims=LAUNCH_RUNTIME_NON_CLAIMS)
    with pytest.raises(ValueError, match="non-claims must be exact"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_validator_rejects_r1_only_manifest(tmp_path):
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        R1_CPU_BASE_COMMIT_SHA,
    )

    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    forged_manifest = {
        "r2a_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": R2A_CPU_BASE_COMMIT_SHA,
        "archive_created_at_utc": "2026-06-19T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        compute_launch_manifest_sha256,
    )

    forged = replace(
        receipt,
        launch_manifest_embedded=forged_manifest,
        launch_manifest_sha256=compute_launch_manifest_sha256(forged_manifest),
    )
    with pytest.raises(ValueError, match="r2a_cpu_base_commit_sha"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_applier_rejects_mismatched_p1_receipt_sha(tmp_path):
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        compute_canonical_launch_artifact_sha256,
    )

    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    forged = replace(
        receipt,
        p1_live_conversion_receipt_sha256="0" * 64,
    )
    forged = replace(
        forged,
        canonical_launch_artifact_sha256=compute_canonical_launch_artifact_sha256(
            forged.to_dict()
        ),
    )
    with pytest.raises(ValueError, match="p1_live_conversion_receipt_sha256 mismatch"):
        apply_live_activation_residuals_surface_overrides(forged)
