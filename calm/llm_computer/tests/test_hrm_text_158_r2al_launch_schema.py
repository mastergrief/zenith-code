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
    R2alLaunchProofAbort,
    R2alLaunchProofMeasurements,
    _resolve_r2al_repo_root,
    build_launch_runtime_activation_residuals_validation_receipt,
    canonicalize_base_sub2_surface_ids,
    derive_r2al_live_base_fields,
    launch_runtime_activation_residuals_receipt_from_dict,
    validate_launch_runtime_activation_residuals_receipt,
    validate_r2al_live_base_preflight,
    verify_git_commit_is_ancestor,
    verify_r2al_banked_p1_ancestor_preflight,
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


def _repo_git_root() -> str:
    return str(Path(__file__).resolve().parents[3])


def _write_receipt_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _r2al_env(tmp_path: Path, *, p1_path: Path, r1l_path: Path) -> dict[str, str]:
    return {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": ".",
        "R2AL_GIT_REPO_ROOT": _repo_git_root(),
        "R2AL_LAUNCH_RECEIPT_JSON": str(tmp_path / "receipts" / "r2al_launch_runtime_receipt.json"),
        "R2AL_LAUNCH_LOG": str(tmp_path / "logs" / "r2al_launch.log"),
        "R2AL_W6_PARENT_PATH": str(tmp_path / "artifacts" / "w6_parent_readonly.pt"),
        "R2AL_P1_RECEIPT_JSON": str(p1_path),
        "R2AL_R1L_RECEIPT_JSON": str(r1l_path),
    }


def _setup_r2al_receipt_paths(tmp_path: Path) -> dict[str, Path]:
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    p1_path = tmp_path / "receipts" / "p1_live_conversion_receipt.json"
    r1l_path = tmp_path / "receipts" / "r1l_launch_runtime_receipt.json"
    _write_receipt_json(p1_path, p1_receipt.to_dict())
    _write_receipt_json(r1l_path, r1l_receipt.to_dict())
    return {"p1_path": p1_path, "r1l_path": r1l_path}



def _raw_mint_r2al_receipt_for_validator_tests(
    tmp_path: Path,
) -> LaunchRuntimeActivationResidualsValidationReceipt:
    """TEST-LOCAL only. Constructs an R2-A-L-shaped receipt without the production builder.

    Not a production mint. Explicitly named raw helper so it cannot be mistaken for
    validated production authority. Production path parks with R1_ROW_FLIP_AUTHORITY_UNAVAILABLE.
    """
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        compute_canonical_launch_artifact_sha256,
        compute_gpu_identity_sha256,
        compute_launch_manifest_sha256,
        validate_activation_relief_measurement,
    )
    from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
        AUTHORIZED_R2A_L_SURFACE_TUPLE,
        LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_RECEIPT_SCHEMA_VERSION,
        LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_TARGET_NAME,
        MECHANISM_ID,
        PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS,
        compute_r2al_proof_env_hash_sha256,
        sha256_file_bytes,
        validate_launch_runtime_activation_residuals_receipt,
        verify_r2al_banked_p1_ancestor_preflight,
        _resolve_r2al_repo_root,
    )
    from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        SURFACE_ACTIVATIONS_RESIDUALS,
    )

    paths = _setup_r2al_receipt_paths(tmp_path)
    from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
        _repo_head_sha,
    )

    launch_source = _repo_head_sha()
    manifest = {
        "r2a_cpu_base_commit_sha": R2A_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-19T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = _r2al_env(tmp_path, p1_path=paths["p1_path"], r1l_path=paths["r1l_path"])
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    (tmp_path / "logs" / "r2al_launch.log").write_bytes(b"R2-A-L launch proof log\n")
    measurements = _valid_synthetic_measurements()
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    ancestry_verified_at_launch_preflight = verify_r2al_banked_p1_ancestor_preflight(
        p1_receipt=p1_receipt,
        launch_source_commit_sha=launch_source,
        repo_root=_resolve_r2al_repo_root(proof_env=env),
    )
    validate_activation_relief_measurement(measurements.activation_relief_measurement)
    cuda_delta = (
        measurements.cuda_peak_allocated_bytes_baseline_median
        - measurements.cuda_peak_allocated_bytes_recompute_median
    )
    threshold = max(
        8 * 1024 * 1024,
        int(0.005 * measurements.cuda_peak_allocated_bytes_baseline_median),
    )
    threshold_met = cuda_delta >= threshold
    live_base_preflight_pass = True
    gpu_memory_measurement_pass = threshold_met
    launch_runtime_validation_pass = (
        measurements.m1_mechanism_proof_pass
        and measurements.m1_no_hidden_bf16_proof_pass
        and gpu_memory_measurement_pass
        and live_base_preflight_pass
        and threshold_met
        and measurements.paired_run_count >= 3
        and measurements.loss_finite_main
    )
    gpu_identity_sha256 = compute_gpu_identity_sha256(
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-r2al-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
    )
    # Authority-shaped fields are for R2-A-L *validator* schema tests only.
    # They are not production-minted under Type-1.
    receipt_without_hash = LaunchRuntimeActivationResidualsValidationReceipt(
        schema_version=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_RECEIPT_SCHEMA_VERSION,
        target_name=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_TARGET_NAME,
        proof_kind=PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS,
        mechanism_id=MECHANISM_ID,
        live_readiness_row_flip_authorized=True,
        readiness_row_flip_authorized_surface_names=AUTHORIZED_R2A_L_SURFACE_TUPLE,
        launch_source_commit_sha=launch_source,
        r2a_cpu_base_commit_sha=R2A_CPU_BASE_COMMIT_SHA,
        ancestry_verified_at_launch_preflight=ancestry_verified_at_launch_preflight,
        live_base_preflight_pass=live_base_preflight_pass,
        launch_runtime_validation_pass=launch_runtime_validation_pass,
        m1_mechanism_proof_pass=measurements.m1_mechanism_proof_pass,
        m1_no_hidden_bf16_proof_pass=measurements.m1_no_hidden_bf16_proof_pass,
        gpu_memory_measurement_pass=gpu_memory_measurement_pass,
        launch_manifest_sha256=compute_launch_manifest_sha256(manifest),
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=("pytest", "r2al-launch"),
        proof_env_hash_sha256=compute_r2al_proof_env_hash_sha256(env),
        clean_run_dir_sha256="a" * 64,
        w6_parent_path=str(tmp_path / "checkpoints" / "w6_parent_readonly.pt"),
        w6_parent_sha256_before="9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        w6_parent_sha256_after="9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec",
        gpu_name="synthetic-gpu",
        gpu_uuid="gpu-uuid-r2al-test",
        driver_version="550.00",
        cuda_version="12.4",
        torch_version="2.5.0",
        gpu_identity_sha256=gpu_identity_sha256,
        model_config_digest_sha256="b" * 64,
        proof_batch_digest_sha256="c" * 64,
        retained_support_digest_sha256=hashlib.sha256(b"[]").hexdigest(),
        p1_live_conversion_receipt_sha256=sha256_file_bytes(paths["p1_path"]),
        r1l_launch_runtime_receipt_sha256=sha256_file_bytes(paths["r1l_path"]),
        base_readiness_receipt_sha256="d" * 64,
        base_sub2_surface_count=4,
        base_sub2_surface_ids=CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS,
        base_activations_residuals_classification=RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        base_activations_residuals_is_sub2=False,
        base_ready_for_pre_full_stack_diagnostic=True,
        base_ready_for_main_science=False,
        base_blocker_surface_names=(SURFACE_ACTIVATIONS_RESIDUALS,),
        m1_seam_handle_pack_count=measurements.m1_seam_handle_pack_count,
        m1_registered_seam_tensor_full_pack_count=(
            measurements.m1_registered_seam_tensor_full_pack_count
        ),
        m1_seam_remat_unpack_recompute_count_total=(
            measurements.m1_seam_remat_unpack_recompute_count_total
        ),
        m1_saved_tensor_payload_bytes_delta=measurements.m1_saved_tensor_payload_bytes_delta,
        m1_forbidden_closure_tensor_count_total=(
            measurements.m1_forbidden_closure_tensor_count_total
        ),
        activation_relief_measurement=dict(measurements.activation_relief_measurement),
        paired_run_count=measurements.paired_run_count,
        cuda_peak_allocated_bytes_baseline_median=(
            measurements.cuda_peak_allocated_bytes_baseline_median
        ),
        cuda_peak_allocated_bytes_recompute_median=(
            measurements.cuda_peak_allocated_bytes_recompute_median
        ),
        cuda_peak_allocated_bytes_delta_median=cuda_delta,
        cuda_peak_reduction_threshold_bytes=threshold,
        cuda_peak_reduction_threshold_met=threshold_met,
        cuda_peak_reserved_bytes_delta_median=(
            measurements.cuda_peak_reserved_bytes_delta_median
        ),
        loss_finite_main=measurements.loss_finite_main,
        applier_base_surface_count_sub2=4,
        applier_result_sub2_surface_count=5,
        applier_result_ready_for_main_science=False,
        applier_result_ready_for_pre_full_stack_diagnostic=True,
        applier_flipped_surface_ids=AUTHORIZED_R2A_L_SURFACE_TUPLE,
        log_artifact_sha256=hashlib.sha256(b"R2-A-L launch proof log\n").hexdigest(),
        canonical_launch_artifact_sha256="",
        non_claims=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_NON_CLAIMS,
    )
    # silence unused r1l object (paths already hashed)
    _ = r1l_receipt
    canonical_hash = compute_canonical_launch_artifact_sha256(receipt_without_hash.to_dict())
    receipt = replace(
        receipt_without_hash,
        canonical_launch_artifact_sha256=canonical_hash,
    )
    validate_launch_runtime_activation_residuals_receipt(receipt)
    return receipt


def _mint_valid_r2al_launch_receipt(tmp_path: Path) -> LaunchRuntimeActivationResidualsValidationReceipt:
    # Schema/validator tests use the explicit test-local raw helper.
    # Production builder is exercised separately and must park under Type-1.
    return _raw_mint_r2al_receipt_for_validator_tests(tmp_path)


def test_r2al_receipt_roundtrip_and_validator(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    roundtrip = launch_runtime_activation_residuals_receipt_from_dict(receipt.to_dict())
    validate_launch_runtime_activation_residuals_receipt(roundtrip)
    assert tuple(roundtrip.base_sub2_surface_ids) == CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS


def test_r2al_validator_rejects_missing_r1l_base(tmp_path):
    paths = _setup_r2al_receipt_paths(tmp_path)
    env = _r2al_env(tmp_path, p1_path=paths["p1_path"], r1l_path=tmp_path / "missing_r1l.json")
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


def test_r2al_live_r1_backward_launch_surfaces_row_flip_authority_unavailable():
    """live_r1_backward_launch_surfaces parks under Type-1 (no R1 row-flip authority)."""
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    with pytest.raises(ValueError, match="R1_ROW_FLIP_AUTHORITY_UNAVAILABLE"):
        live_r1_backward_launch_surfaces(r1l_receipt, p1_receipt)


def test_r2al_applier_valid_receipt_row_flip_authority_unavailable(tmp_path):
    """Valid R2-A-L receipt into apply_live_activation_residuals_surface_overrides parks under Type-1."""
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    with pytest.raises(ValueError, match="R1_ROW_FLIP_AUTHORITY_UNAVAILABLE"):
        apply_live_activation_residuals_surface_overrides(receipt)


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


def test_r2al_live_base_preflight_row_flip_authority_unavailable(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    p1_receipt = _mint_live_conversion_receipt()
    r1l_receipt = _mint_valid_launch_receipt()
    with pytest.raises(ValueError, match="R1_ROW_FLIP_AUTHORITY_UNAVAILABLE"):
        validate_r2al_live_base_preflight(
            receipt,
            p1_receipt=p1_receipt,
            r1l_receipt=r1l_receipt,
            p1_receipt_path=_setup_r2al_receipt_paths(tmp_path)["p1_path"],
            r1l_receipt_path=_setup_r2al_receipt_paths(tmp_path)["r1l_path"],
        )


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
        "launch_source_commit_sha": receipt.launch_source_commit_sha,
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


def test_r2al_banked_ancestor_p1_row_flip_authority_unavailable(tmp_path):
    """Live base derive/preflight park under Type-1 (no R1 row-flip authority)."""
    banked_p1_sha = R2A_CPU_BASE_COMMIT_SHA
    p1_receipt = replace(_mint_live_conversion_receipt(), source_commit_sha=banked_p1_sha)
    r1l_receipt = _mint_valid_launch_receipt()
    with pytest.raises(ValueError, match="R1_ROW_FLIP_AUTHORITY_UNAVAILABLE"):
        derive_r2al_live_base_fields(p1_receipt=p1_receipt, r1l_receipt=r1l_receipt)


def test_r2al_non_ancestor_p1_fails_preflight(monkeypatch):
    p1_receipt = replace(
        _mint_live_conversion_receipt(),
        source_commit_sha="deadbeef" * 5,
    )

    def _reject_ancestry(*_args, **_kwargs):
        raise ValueError("P1 source_commit_sha deadbeefdeadbeefdeadbeefdeadbeefdeadbeef is not an ancestor")

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.activation_residuals_launch.verify_git_commit_is_ancestor",
        _reject_ancestry,
    )
    with pytest.raises(ValueError, match="not an ancestor"):
        verify_r2al_banked_p1_ancestor_preflight(
            p1_receipt=p1_receipt,
            launch_source_commit_sha="f" * 40,
        )


def test_r2al_git_repo_root_missing_path_fails_closed(monkeypatch):
    monkeypatch.setenv("R2AL_GIT_REPO_ROOT", "/nonexistent/r2al/git/root/path")
    with pytest.raises(ValueError, match="R2AL_GIT_REPO_ROOT does not exist"):
        _resolve_r2al_repo_root()


def test_r2al_git_repo_root_env_override_uses_provided_cwd(monkeypatch):
    real_repo = _repo_git_root()
    monkeypatch.setenv("R2AL_GIT_REPO_ROOT", real_repo)
    seen: dict[str, object] = {}

    def _fake_run(cmd, cwd=None, **kwargs):
        seen["cwd"] = cwd

        class _Result:
            returncode = 0
            stderr = ""

        return _Result()

    verify_git_commit_is_ancestor(
        "3936d74966f3e9c1b0688d74b56b6aa598ebddcd",
        "dacac0a7557738f678392011dc4c281904782db8",
        subprocess_run=_fake_run,
    )
    assert str(seen["cwd"]) == real_repo


def test_r2al_verify_head_matches_uses_env_root(monkeypatch):
    import subprocess

    real_repo = _repo_git_root()
    launch_source = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=real_repo,
        text=True,
    ).strip()
    monkeypatch.setenv("R2AL_GIT_REPO_ROOT", real_repo)
    p1_receipt = replace(
        _mint_live_conversion_receipt(),
        source_commit_sha=R2A_CPU_BASE_COMMIT_SHA,
    )
    assert verify_r2al_banked_p1_ancestor_preflight(
        p1_receipt=p1_receipt,
        launch_source_commit_sha=launch_source,
        verify_head_matches_launch_source=True,
    )


def test_r2al_proof_env_requires_git_repo_root(tmp_path, monkeypatch):
    from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
        _read_r2al_proof_env_embedded,
    )

    env_path = tmp_path / "proof_env.json"
    env_path.write_text(
        json.dumps(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONPATH": ".",
                "R2AL_LAUNCH_RECEIPT_JSON": str(tmp_path / "receipt.json"),
                "R2AL_LAUNCH_LOG": str(tmp_path / "launch.log"),
                "R2AL_W6_PARENT_PATH": str(tmp_path / "w6.pt"),
                "R2AL_P1_RECEIPT_JSON": str(tmp_path / "p1.json"),
                "R2AL_R1L_RECEIPT_JSON": str(tmp_path / "r1l.json"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("R2AL_LAUNCH_ENV_JSON", str(env_path))
    with pytest.raises(R2alLaunchProofAbort, match="R2AL_GIT_REPO_ROOT"):
        _read_r2al_proof_env_embedded()


def test_r2al_validator_rejects_proof_env_missing_git_repo_root(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    env = dict(receipt.proof_env_embedded)
    del env["R2AL_GIT_REPO_ROOT"]
    forged = replace(receipt, proof_env_embedded=env)
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_validator_rejects_proof_env_empty_git_repo_root(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    env = dict(receipt.proof_env_embedded)
    env["R2AL_GIT_REPO_ROOT"] = ""
    forged = replace(receipt, proof_env_embedded=env)
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        validate_launch_runtime_activation_residuals_receipt(forged)


def test_r2al_receipt_from_dict_rejects_proof_env_missing_git_repo_root(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    payload = receipt.to_dict()
    env = dict(payload["proof_env_embedded"])
    del env["R2AL_GIT_REPO_ROOT"]
    payload["proof_env_embedded"] = env
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        launch_runtime_activation_residuals_receipt_from_dict(payload)


def test_r2al_receipt_from_dict_rejects_proof_env_empty_git_repo_root(tmp_path):
    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    payload = receipt.to_dict()
    env = dict(payload["proof_env_embedded"])
    env["R2AL_GIT_REPO_ROOT"] = "  "
    payload["proof_env_embedded"] = env
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        launch_runtime_activation_residuals_receipt_from_dict(payload)


def test_r2al_artifact_validator_rejects_proof_env_missing_git_repo_root(tmp_path):
    from calm.hrm_text_158.native_full_stack.activation_residuals_launch import (
        validate_launch_runtime_activation_residuals_artifacts,
    )

    receipt = _mint_valid_r2al_launch_receipt(tmp_path)
    manifest_bytes = (
        json.dumps(dict(receipt.launch_manifest_embedded), indent=2, sort_keys=True) + "\n"
    ).encode()
    receipt = replace(
        receipt,
        launch_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )
    env = dict(receipt.proof_env_embedded)
    del env["R2AL_GIT_REPO_ROOT"]
    env_snapshot_bytes = json.dumps(env, sort_keys=True).encode()
    log_bytes = b"R2-A-L launch proof log\n"
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        validate_launch_runtime_activation_residuals_artifacts(
            receipt,
            launch_manifest_bytes=manifest_bytes,
            env_snapshot_bytes=env_snapshot_bytes,
            log_bytes=log_bytes,
        )


def test_r2al_build_receipt_rejects_proof_env_missing_git_repo_root(tmp_path, monkeypatch):
    paths = _setup_r2al_receipt_paths(tmp_path)
    from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
        _repo_head_sha,
    )

    launch_source = _repo_head_sha()
    env = _r2al_env(tmp_path, p1_path=paths["p1_path"], r1l_path=paths["r1l_path"])
    del env["R2AL_GIT_REPO_ROOT"]
    monkeypatch.delenv("R2AL_GIT_REPO_ROOT", raising=False)
    manifest = {
        "r2a_cpu_base_commit_sha": R2A_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-19T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="proof_env_embedded missing required keys.*R2AL_GIT_REPO_ROOT"):
        build_launch_runtime_activation_residuals_validation_receipt(
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

def test_r2al_production_builder_parks_row_flip_authority_unavailable(tmp_path):
    """Production builder must not mint authority-shaped R2-A-L receipt under Type-1."""
    paths = _setup_r2al_receipt_paths(tmp_path)
    from calm.llm_computer.tests.test_hrm_text_158_trainer_sub2_authority_live_checkpoint import (
        _repo_head_sha,
    )

    launch_source = _repo_head_sha()
    manifest = {
        "r2a_cpu_base_commit_sha": R2A_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": "2026-06-19T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = _r2al_env(tmp_path, p1_path=paths["p1_path"], r1l_path=paths["r1l_path"])
    (tmp_path / "logs").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="R1_ROW_FLIP_AUTHORITY_UNAVAILABLE"):
        build_launch_runtime_activation_residuals_validation_receipt(
            launch_source_commit_sha=launch_source,
            launch_manifest_embedded=manifest,
            proof_env_embedded=env,
            proof_command_argv=("pytest", "r2al-launch"),
            clean_run_dir_sha256="a" * 64,
            w6_parent_path=str(tmp_path / "checkpoints" / "w6_parent_readonly.pt"),
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
