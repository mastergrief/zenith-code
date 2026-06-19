"""R2-A-L GPU launch/runtime validation for activation residuals M1 remat."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.activation_relief import (
    W6_PARENT_SHA256_PINNED,
    compute_canonical_launch_artifact_sha256,
    compute_gpu_identity_sha256,
    compute_launch_manifest_sha256,
    validate_activation_relief_measurement,
)
from calm.hrm_text_158.native_full_stack.activation_relief import (
    _canonical_json_sha256,
    _embedded_mapping,
    _require_nonempty_string,
    _string_tuple,
)
from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
    MECHANISM_ID,
    build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5,
)

PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS = (
    "launch_runtime_activation_residuals_validation"
)
AUTHORIZED_R2A_L_SURFACE_TUPLE = ("activations_residuals",)
LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_r2a_launch/v1.gpu_m1_seam_remat_runtime"
)
LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_TARGET_NAME = (
    "r2a_activation_residuals_m1_launch_runtime"
)
CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS = (
    "backward_saved_tensors_transients",
    "dense_int16_persistent_accumulator_absence",
    "persistent_qacc_authority",
    "q_sidecar_vote_carrier",
)
R2A_CPU_BASE_COMMIT_SHA = "860318e3637fd6d82bde051bbf204c58865d55a4"
R2AL_LAUNCH_MANIFEST_EMBEDDED_KEYS = (
    "r2a_cpu_base_commit_sha",
    "launch_source_commit_sha",
    "archive_created_at_utc",
    "archive_method",
)
LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_NON_CLAIMS = (
    "proves activations_residuals sub2 on valid R2-A-L launch/runtime path only",
    "pre_full_stack_diagnostic exception only; NOT ready_for_main_science",
    "does NOT flip backward_saved_tensors_transients / attention_kv_attention_buffers / "
    "optimizer_credit_state / native_kernelized_hot_path",
    "does NOT prove sub2-complete, viability, learning, acquisition, retention, or throughput",
    "does NOT clear zL_init persistent BF16 FP-exception",
    "does NOT mutate banked W6 parent .pt",
    "does NOT authorize full training or optimizer resume",
)
R2AL_LAUNCH_LOG_AT_MINT_BASENAME = "r2al_launch_log_at_mint.log"
R2AL_PROOF_ENV_HASH_KEYS = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "R2AL_LAUNCH_RECEIPT_JSON",
    "R2AL_LAUNCH_LOG",
    "R2AL_W6_PARENT_PATH",
    "R2AL_P1_RECEIPT_JSON",
    "R2AL_R1L_RECEIPT_JSON",
    "R2AL_GIT_REPO_ROOT",
    "TORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
)
REQUIRED_R2AL_PROOF_ENV_KEYS = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "R2AL_GIT_REPO_ROOT",
    "R2AL_LAUNCH_RECEIPT_JSON",
    "R2AL_LAUNCH_LOG",
    "R2AL_W6_PARENT_PATH",
    "R2AL_P1_RECEIPT_JSON",
    "R2AL_R1L_RECEIPT_JSON",
)


class R2alLaunchProofAbort(RuntimeError):
    """Abort R2-A-L launch proof without minting a receipt."""


@dataclass(frozen=True)
class R2alLaunchProofMeasurements:
    m1_seam_handle_pack_count: int
    m1_registered_seam_tensor_full_pack_count: int
    m1_seam_remat_unpack_recompute_count_total: int
    m1_saved_tensor_payload_bytes_delta: int
    m1_forbidden_closure_tensor_count_total: int
    m1_registered_seam_tensor_in_closure_count: int
    m1_no_hidden_bf16_proof_pass: bool
    m1_mechanism_proof_pass: bool
    loss_finite_main: bool
    paired_run_count: int
    cuda_peak_allocated_bytes_baseline_median: int
    cuda_peak_allocated_bytes_recompute_median: int
    cuda_peak_reserved_bytes_delta_median: int
    activation_relief_measurement: dict[str, object]


@dataclass(frozen=True)
class LaunchRuntimeActivationResidualsValidationReceipt:
    """R2-A-L GPU launch/runtime validation receipt (schema v1)."""

    schema_version: str
    target_name: str
    proof_kind: str
    mechanism_id: str
    live_readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    launch_source_commit_sha: str
    r2a_cpu_base_commit_sha: str
    ancestry_verified_at_launch_preflight: bool
    live_base_preflight_pass: bool
    launch_runtime_validation_pass: bool
    m1_mechanism_proof_pass: bool
    m1_no_hidden_bf16_proof_pass: bool
    gpu_memory_measurement_pass: bool
    launch_manifest_sha256: str
    launch_manifest_embedded: Mapping[str, str]
    proof_env_embedded: Mapping[str, str]
    proof_command_argv: tuple[str, ...]
    proof_env_hash_sha256: str
    clean_run_dir_sha256: str
    w6_parent_path: str
    w6_parent_sha256_before: str
    w6_parent_sha256_after: str
    gpu_name: str
    gpu_uuid: str
    driver_version: str
    cuda_version: str
    torch_version: str
    gpu_identity_sha256: str
    model_config_digest_sha256: str
    proof_batch_digest_sha256: str
    retained_support_digest_sha256: str
    p1_live_conversion_receipt_sha256: str
    r1l_launch_runtime_receipt_sha256: str
    base_readiness_receipt_sha256: str
    base_sub2_surface_count: int
    base_sub2_surface_ids: tuple[str, ...]
    base_activations_residuals_classification: str
    base_activations_residuals_is_sub2: bool
    base_ready_for_pre_full_stack_diagnostic: bool
    base_ready_for_main_science: bool
    base_blocker_surface_names: tuple[str, ...]
    m1_seam_handle_pack_count: int
    m1_registered_seam_tensor_full_pack_count: int
    m1_seam_remat_unpack_recompute_count_total: int
    m1_saved_tensor_payload_bytes_delta: int
    m1_forbidden_closure_tensor_count_total: int
    activation_relief_measurement: Mapping[str, object]
    paired_run_count: int
    cuda_peak_allocated_bytes_baseline_median: int
    cuda_peak_allocated_bytes_recompute_median: int
    cuda_peak_allocated_bytes_delta_median: int
    cuda_peak_reduction_threshold_bytes: int
    cuda_peak_reduction_threshold_met: bool
    cuda_peak_reserved_bytes_delta_median: int
    loss_finite_main: bool
    applier_base_surface_count_sub2: int
    applier_result_sub2_surface_count: int
    applier_result_ready_for_main_science: bool
    applier_result_ready_for_pre_full_stack_diagnostic: bool
    applier_flipped_surface_ids: tuple[str, ...]
    log_artifact_sha256: str
    canonical_launch_artifact_sha256: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "proof_kind": self.proof_kind,
            "mechanism_id": self.mechanism_id,
            "live_readiness_row_flip_authorized": self.live_readiness_row_flip_authorized,
            "readiness_row_flip_authorized_surface_names": list(
                self.readiness_row_flip_authorized_surface_names
            ),
            "launch_source_commit_sha": self.launch_source_commit_sha,
            "r2a_cpu_base_commit_sha": self.r2a_cpu_base_commit_sha,
            "ancestry_verified_at_launch_preflight": (
                self.ancestry_verified_at_launch_preflight
            ),
            "live_base_preflight_pass": self.live_base_preflight_pass,
            "launch_runtime_validation_pass": self.launch_runtime_validation_pass,
            "m1_mechanism_proof_pass": self.m1_mechanism_proof_pass,
            "m1_no_hidden_bf16_proof_pass": self.m1_no_hidden_bf16_proof_pass,
            "gpu_memory_measurement_pass": self.gpu_memory_measurement_pass,
            "launch_manifest_sha256": self.launch_manifest_sha256,
            "launch_manifest_embedded": dict(self.launch_manifest_embedded),
            "proof_env_embedded": dict(self.proof_env_embedded),
            "proof_command_argv": list(self.proof_command_argv),
            "proof_env_hash_sha256": self.proof_env_hash_sha256,
            "clean_run_dir_sha256": self.clean_run_dir_sha256,
            "w6_parent_path": self.w6_parent_path,
            "w6_parent_sha256_before": self.w6_parent_sha256_before,
            "w6_parent_sha256_after": self.w6_parent_sha256_after,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "torch_version": self.torch_version,
            "gpu_identity_sha256": self.gpu_identity_sha256,
            "model_config_digest_sha256": self.model_config_digest_sha256,
            "proof_batch_digest_sha256": self.proof_batch_digest_sha256,
            "retained_support_digest_sha256": self.retained_support_digest_sha256,
            "p1_live_conversion_receipt_sha256": self.p1_live_conversion_receipt_sha256,
            "r1l_launch_runtime_receipt_sha256": self.r1l_launch_runtime_receipt_sha256,
            "base_readiness_receipt_sha256": self.base_readiness_receipt_sha256,
            "base_sub2_surface_count": self.base_sub2_surface_count,
            "base_sub2_surface_ids": list(self.base_sub2_surface_ids),
            "base_activations_residuals_classification": (
                self.base_activations_residuals_classification
            ),
            "base_activations_residuals_is_sub2": self.base_activations_residuals_is_sub2,
            "base_ready_for_pre_full_stack_diagnostic": (
                self.base_ready_for_pre_full_stack_diagnostic
            ),
            "base_ready_for_main_science": self.base_ready_for_main_science,
            "base_blocker_surface_names": list(self.base_blocker_surface_names),
            "m1_seam_handle_pack_count": self.m1_seam_handle_pack_count,
            "m1_registered_seam_tensor_full_pack_count": (
                self.m1_registered_seam_tensor_full_pack_count
            ),
            "m1_seam_remat_unpack_recompute_count_total": (
                self.m1_seam_remat_unpack_recompute_count_total
            ),
            "m1_saved_tensor_payload_bytes_delta": self.m1_saved_tensor_payload_bytes_delta,
            "m1_forbidden_closure_tensor_count_total": (
                self.m1_forbidden_closure_tensor_count_total
            ),
            "activation_relief_measurement": dict(self.activation_relief_measurement),
            "paired_run_count": self.paired_run_count,
            "cuda_peak_allocated_bytes_baseline_median": (
                self.cuda_peak_allocated_bytes_baseline_median
            ),
            "cuda_peak_allocated_bytes_recompute_median": (
                self.cuda_peak_allocated_bytes_recompute_median
            ),
            "cuda_peak_allocated_bytes_delta_median": (
                self.cuda_peak_allocated_bytes_delta_median
            ),
            "cuda_peak_reduction_threshold_bytes": self.cuda_peak_reduction_threshold_bytes,
            "cuda_peak_reduction_threshold_met": self.cuda_peak_reduction_threshold_met,
            "cuda_peak_reserved_bytes_delta_median": (
                self.cuda_peak_reserved_bytes_delta_median
            ),
            "loss_finite_main": self.loss_finite_main,
            "applier_base_surface_count_sub2": self.applier_base_surface_count_sub2,
            "applier_result_sub2_surface_count": self.applier_result_sub2_surface_count,
            "applier_result_ready_for_main_science": (
                self.applier_result_ready_for_main_science
            ),
            "applier_result_ready_for_pre_full_stack_diagnostic": (
                self.applier_result_ready_for_pre_full_stack_diagnostic
            ),
            "applier_flipped_surface_ids": list(self.applier_flipped_surface_ids),
            "log_artifact_sha256": self.log_artifact_sha256,
            "canonical_launch_artifact_sha256": self.canonical_launch_artifact_sha256,
            "non_claims": list(self.non_claims),
        }


def sha256_file_bytes(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_r2al_proof_env_hash_sha256(env_embedded: Mapping[str, str]) -> str:
    payload = {key: str(env_embedded.get(key, "")) for key in R2AL_PROOF_ENV_HASH_KEYS}
    return _canonical_json_sha256(payload)


def _proof_env_embedded_mapping(value: object, *, field_name: str) -> dict[str, str]:
    embedded = _embedded_mapping(
        value,
        field_name=field_name,
        required_keys=REQUIRED_R2AL_PROOF_ENV_KEYS,
    )
    empty = [
        key
        for key in REQUIRED_R2AL_PROOF_ENV_KEYS
        if not str(embedded.get(key, "")).strip()
    ]
    if empty:
        raise ValueError(f"{field_name} missing required keys: {', '.join(empty)}")
    return embedded


def _require_nonempty_proof_env_keys(
    env_embedded: Mapping[str, str],
    *,
    field_name: str,
) -> None:
    missing_or_empty = [
        key
        for key in REQUIRED_R2AL_PROOF_ENV_KEYS
        if not str(env_embedded.get(key, "")).strip()
    ]
    if missing_or_empty:
        raise ValueError(
            f"{field_name} missing required keys: {', '.join(missing_or_empty)}"
        )


def _resolve_r2al_repo_root(*, proof_env: Mapping[str, str] | None = None) -> Path:
    raw = ""
    if proof_env is not None:
        raw = str(proof_env.get("R2AL_GIT_REPO_ROOT", "")).strip()
    if not raw:
        raw = os.environ.get("R2AL_GIT_REPO_ROOT", "").strip()
    if raw:
        root = Path(raw)
        if not root.exists():
            raise ValueError(f"R2AL_GIT_REPO_ROOT does not exist: {raw!r}")
        return root.resolve()
    return Path(__file__).resolve().parents[3]


def verify_git_commit_is_ancestor(
    ancestor_sha: str,
    descendant_sha: str,
    *,
    repo_root: Path | None = None,
    subprocess_run: Callable[..., Any] | None = None,
) -> None:
    """Fail-closed: ancestor_sha must be an ancestor of descendant_sha in git."""
    run = subprocess_run or subprocess.run
    root = repo_root or _resolve_r2al_repo_root()
    ancestor = str(ancestor_sha).strip()
    descendant = str(descendant_sha).strip()
    if len(ancestor) != 40 or len(descendant) != 40:
        raise ValueError(
            "git ancestry check requires full 40-char shas "
            f"(ancestor={ancestor!r}, descendant={descendant!r})"
        )
    result = run(
        ["git", "merge-base", "--is-ancestor", ancestor, descendant],
        cwd=root,
        capture_output=True,
        text=True,
    )
    if result.returncode == 0:
        return
    if result.returncode == 1:
        raise ValueError(
            f"P1 source_commit_sha {ancestor} is not an ancestor of "
            f"launch source {descendant}"
        )
    stderr = (result.stderr or "").strip()
    raise ValueError(
        "git ancestry check failed-closed "
        f"(rc={result.returncode}): {stderr or 'missing object/shallow/no-git'}"
    )


def verify_r2al_banked_p1_ancestor_preflight(
    *,
    p1_receipt: Any,
    launch_source_commit_sha: str,
    repo_root: Path | None = None,
    git_is_ancestor_fn: Callable[..., None] | None = None,
    resolve_head_sha_fn: Callable[[], str] | None = None,
    verify_head_matches_launch_source: bool = False,
) -> bool:
    """Verify banked P1 source is an ancestor of launch source during preflight."""
    p1_sha = str(p1_receipt.source_commit_sha).strip()
    launch_sha = str(launch_source_commit_sha).strip()
    if not p1_sha:
        raise ValueError("P1b missing source_commit_sha for ancestry preflight")
    if not launch_sha:
        raise ValueError("R2-A-L launch_source_commit_sha required for ancestry preflight")
    if verify_head_matches_launch_source:
        if resolve_head_sha_fn is None:
            head_result = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repo_root or _resolve_r2al_repo_root(),
                capture_output=True,
                text=True,
                check=False,
            )
            if head_result.returncode != 0:
                stderr = (head_result.stderr or "").strip()
                raise ValueError(
                    "git HEAD resolve failed-closed "
                    f"(rc={head_result.returncode}): {stderr or 'no-git'}"
                )
            head_sha = head_result.stdout.strip()
        else:
            head_sha = str(resolve_head_sha_fn()).strip()
        if head_sha != launch_sha:
            raise ValueError(
                f"HEAD {head_sha} != launch_source_commit_sha {launch_sha}"
            )
    check = git_is_ancestor_fn or verify_git_commit_is_ancestor
    if git_is_ancestor_fn is not None:
        check(p1_sha, launch_sha)
    else:
        check(p1_sha, launch_sha, repo_root=repo_root)
    return True


def canonicalize_base_sub2_surface_ids(
    surface_ids: Sequence[str],
) -> tuple[str, ...]:
    return tuple(sorted(str(item) for item in surface_ids))


def compute_base_readiness_receipt_sha256(base_readiness: Any) -> str:
    return _canonical_json_sha256(base_readiness.to_dict())


def derive_r2al_live_base_fields(
    *,
    p1_receipt: Any,
    r1l_receipt: Any,
) -> dict[str, object]:
    from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        RUNTIME_CLASS_SUB2,
        SURFACE_ACTIVATIONS_RESIDUALS,
        live_r1_backward_launch_surfaces,
    )

    base = live_r1_backward_launch_surfaces(
        r1l_receipt,
        p1_receipt,
        require_source_at_head=False,
    )
    sub2_ids = canonicalize_base_sub2_surface_ids(
        surface.surface_id
        for surface in base.surfaces
        if surface.classification == RUNTIME_CLASS_SUB2
    )
    activations = next(
        surface
        for surface in base.surfaces
        if surface.surface_id == SURFACE_ACTIVATIONS_RESIDUALS
    )
    if activations.classification == RUNTIME_CLASS_SUB2:
        raise ValueError("activations_residuals must not be sub2 before R2-A-L")
    return {
        "base_readiness_receipt_sha256": compute_base_readiness_receipt_sha256(base),
        "base_sub2_surface_count": int(base.sub2_surface_count),
        "base_sub2_surface_ids": sub2_ids,
        "base_activations_residuals_classification": activations.classification,
        "base_activations_residuals_is_sub2": False,
        "base_ready_for_pre_full_stack_diagnostic": bool(
            base.ready_for_pre_full_stack_diagnostic
        ),
        "base_ready_for_main_science": bool(base.ready_for_main_science),
        "base_blocker_surface_names": tuple(base.blocker_surface_names),
    }


def validate_r2al_m1_mechanism_telemetry(telemetry: Mapping[str, object]) -> tuple[bool, bool]:
    mechanism_pass = (
        int(telemetry["registered_seam_tensor_full_pack_count"]) == 0
        and int(telemetry["seam_handle_pack_count"]) >= 4
        and int(telemetry["m1_seam_remat_unpack_recompute_count_total"]) > 0
        and int(telemetry.get("m1_saved_tensor_payload_bytes_delta", 0)) > 0
        and int(telemetry["forbidden_closure_tensor_count_total"]) == 0
    )
    no_hidden = (
        int(telemetry["registered_seam_tensor_in_closure_count"]) == 0
        and int(telemetry["forbidden_closure_tensor_count_total"]) == 0
        and int(telemetry.get("recompute_registration_side_effect_count", 0)) == 0
    )
    return mechanism_pass, no_hidden


def validate_r2al_live_base_preflight(
    receipt: LaunchRuntimeActivationResidualsValidationReceipt,
    *,
    p1_receipt: Any,
    r1l_receipt: Any,
    p1_receipt_path: Path | None = None,
    r1l_receipt_path: Path | None = None,
) -> None:
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        validate_launch_runtime_backward_receipt,
    )
    from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (
        RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC,
        RUNTIME_CLASS_SUB2,
        SURFACE_ACTIVATIONS_RESIDUALS,
        current_repo_scaffold_surfaces,
        live_r1_backward_launch_surfaces,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        validate_trainer_sub2_authority_live_conversion_receipt,
    )

    verify_r2al_banked_p1_ancestor_preflight(
        p1_receipt=p1_receipt,
        launch_source_commit_sha=receipt.launch_source_commit_sha,
        repo_root=_resolve_r2al_repo_root(proof_env=receipt.proof_env_embedded),
    )
    validate_trainer_sub2_authority_live_conversion_receipt(
        p1_receipt,
        require_source_at_head=False,
    )
    validate_launch_runtime_backward_receipt(r1l_receipt)
    if p1_receipt_path is not None:
        if sha256_file_bytes(p1_receipt_path) != receipt.p1_live_conversion_receipt_sha256:
            raise ValueError("p1_live_conversion_receipt_sha256 mismatch")
    if r1l_receipt_path is not None:
        if sha256_file_bytes(r1l_receipt_path) != receipt.r1l_launch_runtime_receipt_sha256:
            raise ValueError("r1l_launch_runtime_receipt_sha256 mismatch")

    base = live_r1_backward_launch_surfaces(
        r1l_receipt,
        p1_receipt,
        require_source_at_head=False,
    )
    if compute_base_readiness_receipt_sha256(base) != receipt.base_readiness_receipt_sha256:
        raise ValueError("base_readiness_receipt_sha256 mismatch")
    if int(base.sub2_surface_count) != 4:
        raise ValueError("live base sub2_surface_count must be 4")
    if not base.ready_for_pre_full_stack_diagnostic:
        raise ValueError("live base must be ready_for_pre_full_stack_diagnostic")
    if base.ready_for_main_science:
        raise ValueError("live base must not be ready_for_main_science")
    if SURFACE_ACTIVATIONS_RESIDUALS not in base.blocker_surface_names:
        raise ValueError("live base blockers must include activations_residuals")

    sub2_ids = canonicalize_base_sub2_surface_ids(
        surface.surface_id
        for surface in base.surfaces
        if surface.classification == RUNTIME_CLASS_SUB2
    )
    if sub2_ids != CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS:
        raise ValueError("live base sub2 surface ids mismatch")
    if tuple(receipt.base_sub2_surface_ids) != CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS:
        raise ValueError("receipt base_sub2_surface_ids must use canonical sorted order")
    if int(receipt.base_sub2_surface_count) != 4:
        raise ValueError("receipt base_sub2_surface_count must be 4")

    activations = next(
        surface
        for surface in base.surfaces
        if surface.surface_id == SURFACE_ACTIVATIONS_RESIDUALS
    )
    if activations.classification != RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC:
        raise ValueError("activations_residuals must be pre_full_stack_diagnostic before flip")
    if receipt.base_activations_residuals_classification != RUNTIME_CLASS_PRE_FULL_STACK_DIAGNOSTIC:
        raise ValueError("receipt base_activations_residuals_classification mismatch")
    if receipt.base_activations_residuals_is_sub2:
        raise ValueError("receipt base_activations_residuals_is_sub2 must be false")
    if not receipt.base_ready_for_pre_full_stack_diagnostic:
        raise ValueError("receipt base_ready_for_pre_full_stack_diagnostic must be true")
    if receipt.base_ready_for_main_science:
        raise ValueError("receipt base_ready_for_main_science must be false")
    if SURFACE_ACTIVATIONS_RESIDUALS not in tuple(receipt.base_blocker_surface_names):
        raise ValueError("receipt base_blocker_surface_names must include activations_residuals")

    scaffold = current_repo_scaffold_surfaces()
    scaffold_sub2 = canonicalize_base_sub2_surface_ids(
        surface.surface_id
        for surface in scaffold
        if surface.classification == RUNTIME_CLASS_SUB2
    )
    if scaffold_sub2 == CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS:
        raise ValueError("scaffold-only base must not satisfy banked live sub2 ids")


def load_r2al_base_receipts_from_env(
    proof_env_embedded: Mapping[str, str],
) -> tuple[Any, Any]:
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        launch_runtime_backward_receipt_from_dict,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        live_conversion_receipt_from_dict,
    )

    p1_path = Path(proof_env_embedded["R2AL_P1_RECEIPT_JSON"])
    r1l_path = Path(proof_env_embedded["R2AL_R1L_RECEIPT_JSON"])
    if not p1_path.is_file():
        raise ValueError(f"P1 receipt path not found: {p1_path}")
    if not r1l_path.is_file():
        raise ValueError(f"R1-L receipt path not found: {r1l_path}")
    p1_payload = json.loads(p1_path.read_text(encoding="utf-8"))
    r1l_payload = json.loads(r1l_path.read_text(encoding="utf-8"))
    if not isinstance(p1_payload, dict) or not isinstance(r1l_payload, dict):
        raise ValueError("P1/R1-L receipt JSON must decode to objects")
    return (
        live_conversion_receipt_from_dict(p1_payload),
        launch_runtime_backward_receipt_from_dict(r1l_payload),
    )


def launch_runtime_activation_residuals_receipt_from_dict(
    payload: Mapping[str, object],
) -> LaunchRuntimeActivationResidualsValidationReceipt:
    measurement = payload.get("activation_relief_measurement")
    if not isinstance(measurement, Mapping):
        raise TypeError("activation_relief_measurement must be a mapping")
    return LaunchRuntimeActivationResidualsValidationReceipt(
        schema_version=_require_nonempty_string(
            payload.get("schema_version"),
            field_name="schema_version",
        ),
        target_name=_require_nonempty_string(payload.get("target_name"), field_name="target_name"),
        proof_kind=_require_nonempty_string(payload.get("proof_kind"), field_name="proof_kind"),
        mechanism_id=_require_nonempty_string(payload.get("mechanism_id"), field_name="mechanism_id"),
        live_readiness_row_flip_authorized=bool(
            payload.get("live_readiness_row_flip_authorized")
        ),
        readiness_row_flip_authorized_surface_names=_string_tuple(
            payload.get("readiness_row_flip_authorized_surface_names"),
            field_name="readiness_row_flip_authorized_surface_names",
        ),
        launch_source_commit_sha=_require_nonempty_string(
            payload.get("launch_source_commit_sha"),
            field_name="launch_source_commit_sha",
        ),
        r2a_cpu_base_commit_sha=_require_nonempty_string(
            payload.get("r2a_cpu_base_commit_sha"),
            field_name="r2a_cpu_base_commit_sha",
        ),
        ancestry_verified_at_launch_preflight=bool(
            payload.get("ancestry_verified_at_launch_preflight")
        ),
        live_base_preflight_pass=bool(payload.get("live_base_preflight_pass")),
        launch_runtime_validation_pass=bool(payload.get("launch_runtime_validation_pass")),
        m1_mechanism_proof_pass=bool(payload.get("m1_mechanism_proof_pass")),
        m1_no_hidden_bf16_proof_pass=bool(payload.get("m1_no_hidden_bf16_proof_pass")),
        gpu_memory_measurement_pass=bool(payload.get("gpu_memory_measurement_pass")),
        launch_manifest_sha256=_require_nonempty_string(
            payload.get("launch_manifest_sha256"),
            field_name="launch_manifest_sha256",
        ),
        launch_manifest_embedded=_embedded_mapping(
            payload.get("launch_manifest_embedded"),
            field_name="launch_manifest_embedded",
            required_keys=R2AL_LAUNCH_MANIFEST_EMBEDDED_KEYS,
        ),
        proof_env_embedded=_proof_env_embedded_mapping(
            payload.get("proof_env_embedded"),
            field_name="proof_env_embedded",
        ),
        proof_command_argv=_string_tuple(
            payload.get("proof_command_argv"),
            field_name="proof_command_argv",
        ),
        proof_env_hash_sha256=_require_nonempty_string(
            payload.get("proof_env_hash_sha256"),
            field_name="proof_env_hash_sha256",
        ),
        clean_run_dir_sha256=_require_nonempty_string(
            payload.get("clean_run_dir_sha256"),
            field_name="clean_run_dir_sha256",
        ),
        w6_parent_path=_require_nonempty_string(
            payload.get("w6_parent_path"),
            field_name="w6_parent_path",
        ),
        w6_parent_sha256_before=_require_nonempty_string(
            payload.get("w6_parent_sha256_before"),
            field_name="w6_parent_sha256_before",
        ),
        w6_parent_sha256_after=_require_nonempty_string(
            payload.get("w6_parent_sha256_after"),
            field_name="w6_parent_sha256_after",
        ),
        gpu_name=_require_nonempty_string(payload.get("gpu_name"), field_name="gpu_name"),
        gpu_uuid=_require_nonempty_string(payload.get("gpu_uuid"), field_name="gpu_uuid"),
        driver_version=_require_nonempty_string(
            payload.get("driver_version"),
            field_name="driver_version",
        ),
        cuda_version=_require_nonempty_string(
            payload.get("cuda_version"),
            field_name="cuda_version",
        ),
        torch_version=_require_nonempty_string(
            payload.get("torch_version"),
            field_name="torch_version",
        ),
        gpu_identity_sha256=_require_nonempty_string(
            payload.get("gpu_identity_sha256"),
            field_name="gpu_identity_sha256",
        ),
        model_config_digest_sha256=_require_nonempty_string(
            payload.get("model_config_digest_sha256"),
            field_name="model_config_digest_sha256",
        ),
        proof_batch_digest_sha256=_require_nonempty_string(
            payload.get("proof_batch_digest_sha256"),
            field_name="proof_batch_digest_sha256",
        ),
        retained_support_digest_sha256=_require_nonempty_string(
            payload.get("retained_support_digest_sha256"),
            field_name="retained_support_digest_sha256",
        ),
        p1_live_conversion_receipt_sha256=_require_nonempty_string(
            payload.get("p1_live_conversion_receipt_sha256"),
            field_name="p1_live_conversion_receipt_sha256",
        ),
        r1l_launch_runtime_receipt_sha256=_require_nonempty_string(
            payload.get("r1l_launch_runtime_receipt_sha256"),
            field_name="r1l_launch_runtime_receipt_sha256",
        ),
        base_readiness_receipt_sha256=_require_nonempty_string(
            payload.get("base_readiness_receipt_sha256"),
            field_name="base_readiness_receipt_sha256",
        ),
        base_sub2_surface_count=int(payload.get("base_sub2_surface_count", 0)),
        base_sub2_surface_ids=_string_tuple(
            payload.get("base_sub2_surface_ids"),
            field_name="base_sub2_surface_ids",
        ),
        base_activations_residuals_classification=_require_nonempty_string(
            payload.get("base_activations_residuals_classification"),
            field_name="base_activations_residuals_classification",
        ),
        base_activations_residuals_is_sub2=bool(
            payload.get("base_activations_residuals_is_sub2")
        ),
        base_ready_for_pre_full_stack_diagnostic=bool(
            payload.get("base_ready_for_pre_full_stack_diagnostic")
        ),
        base_ready_for_main_science=bool(payload.get("base_ready_for_main_science")),
        base_blocker_surface_names=_string_tuple(
            payload.get("base_blocker_surface_names"),
            field_name="base_blocker_surface_names",
        ),
        m1_seam_handle_pack_count=int(payload.get("m1_seam_handle_pack_count", 0)),
        m1_registered_seam_tensor_full_pack_count=int(
            payload.get("m1_registered_seam_tensor_full_pack_count", 0)
        ),
        m1_seam_remat_unpack_recompute_count_total=int(
            payload.get("m1_seam_remat_unpack_recompute_count_total", 0)
        ),
        m1_saved_tensor_payload_bytes_delta=int(
            payload.get("m1_saved_tensor_payload_bytes_delta", 0)
        ),
        m1_forbidden_closure_tensor_count_total=int(
            payload.get("m1_forbidden_closure_tensor_count_total", 0)
        ),
        activation_relief_measurement=dict(measurement),
        paired_run_count=int(payload.get("paired_run_count", 0)),
        cuda_peak_allocated_bytes_baseline_median=int(
            payload.get("cuda_peak_allocated_bytes_baseline_median", 0)
        ),
        cuda_peak_allocated_bytes_recompute_median=int(
            payload.get("cuda_peak_allocated_bytes_recompute_median", 0)
        ),
        cuda_peak_allocated_bytes_delta_median=int(
            payload.get("cuda_peak_allocated_bytes_delta_median", 0)
        ),
        cuda_peak_reduction_threshold_bytes=int(
            payload.get("cuda_peak_reduction_threshold_bytes", 0)
        ),
        cuda_peak_reduction_threshold_met=bool(payload.get("cuda_peak_reduction_threshold_met")),
        cuda_peak_reserved_bytes_delta_median=int(
            payload.get("cuda_peak_reserved_bytes_delta_median", 0)
        ),
        loss_finite_main=bool(payload.get("loss_finite_main")),
        applier_base_surface_count_sub2=int(payload.get("applier_base_surface_count_sub2", 0)),
        applier_result_sub2_surface_count=int(
            payload.get("applier_result_sub2_surface_count", 0)
        ),
        applier_result_ready_for_main_science=bool(
            payload.get("applier_result_ready_for_main_science")
        ),
        applier_result_ready_for_pre_full_stack_diagnostic=bool(
            payload.get("applier_result_ready_for_pre_full_stack_diagnostic")
        ),
        applier_flipped_surface_ids=_string_tuple(
            payload.get("applier_flipped_surface_ids"),
            field_name="applier_flipped_surface_ids",
        ),
        log_artifact_sha256=_require_nonempty_string(
            payload.get("log_artifact_sha256"),
            field_name="log_artifact_sha256",
        ),
        canonical_launch_artifact_sha256=_require_nonempty_string(
            payload.get("canonical_launch_artifact_sha256"),
            field_name="canonical_launch_artifact_sha256",
        ),
        non_claims=_string_tuple(payload.get("non_claims"), field_name="non_claims"),
    )


def build_launch_runtime_activation_residuals_validation_receipt(
    *,
    launch_source_commit_sha: str,
    launch_manifest_embedded: Mapping[str, str],
    proof_env_embedded: Mapping[str, str],
    proof_command_argv: Sequence[str],
    clean_run_dir_sha256: str,
    w6_parent_path: str,
    w6_parent_sha256: str,
    gpu_name: str,
    gpu_uuid: str,
    driver_version: str,
    cuda_version: str,
    torch_version: str,
    model_config_digest_sha256: str,
    proof_batch_digest_sha256: str,
    retained_support_digest_sha256: str,
    p1_receipt: Any,
    r1l_receipt: Any,
    p1_receipt_path: Path,
    r1l_receipt_path: Path,
    measurements: R2alLaunchProofMeasurements,
    log_artifact_sha256: str,
    r2a_cpu_base_commit_sha: str = R2A_CPU_BASE_COMMIT_SHA,
    live_base_preflight_pass: bool = True,
) -> LaunchRuntimeActivationResidualsValidationReceipt:
    manifest = dict(launch_manifest_embedded)
    env = dict(proof_env_embedded)
    _require_nonempty_proof_env_keys(env, field_name="proof_env_embedded")
    if manifest.get("r2a_cpu_base_commit_sha") != r2a_cpu_base_commit_sha:
        raise ValueError("launch manifest r2a_cpu_base_commit_sha mismatch")
    if r2a_cpu_base_commit_sha != R2A_CPU_BASE_COMMIT_SHA:
        raise ValueError("r2a_cpu_base_commit_sha must match R2A_CPU_BASE_COMMIT_SHA")
    ancestry_verified_at_launch_preflight = verify_r2al_banked_p1_ancestor_preflight(
        p1_receipt=p1_receipt,
        launch_source_commit_sha=launch_source_commit_sha,
        repo_root=_resolve_r2al_repo_root(proof_env=env),
    )
    live_base = derive_r2al_live_base_fields(p1_receipt=p1_receipt, r1l_receipt=r1l_receipt)
    cuda_delta = (
        measurements.cuda_peak_allocated_bytes_baseline_median
        - measurements.cuda_peak_allocated_bytes_recompute_median
    )
    threshold = max(
        8 * 1024 * 1024,
        int(0.005 * measurements.cuda_peak_allocated_bytes_baseline_median),
    )
    threshold_met = cuda_delta >= threshold
    validate_activation_relief_measurement(measurements.activation_relief_measurement)
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
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
    )
    receipt_without_hash = LaunchRuntimeActivationResidualsValidationReceipt(
        schema_version=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_RECEIPT_SCHEMA_VERSION,
        target_name=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_TARGET_NAME,
        proof_kind=PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS,
        mechanism_id=MECHANISM_ID,
        live_readiness_row_flip_authorized=True,
        readiness_row_flip_authorized_surface_names=AUTHORIZED_R2A_L_SURFACE_TUPLE,
        launch_source_commit_sha=launch_source_commit_sha,
        r2a_cpu_base_commit_sha=r2a_cpu_base_commit_sha,
        ancestry_verified_at_launch_preflight=ancestry_verified_at_launch_preflight,
        live_base_preflight_pass=live_base_preflight_pass,
        launch_runtime_validation_pass=launch_runtime_validation_pass,
        m1_mechanism_proof_pass=measurements.m1_mechanism_proof_pass,
        m1_no_hidden_bf16_proof_pass=measurements.m1_no_hidden_bf16_proof_pass,
        gpu_memory_measurement_pass=gpu_memory_measurement_pass,
        launch_manifest_sha256=compute_launch_manifest_sha256(manifest),
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=tuple(str(arg) for arg in proof_command_argv),
        proof_env_hash_sha256=compute_r2al_proof_env_hash_sha256(env),
        clean_run_dir_sha256=clean_run_dir_sha256,
        w6_parent_path=w6_parent_path,
        w6_parent_sha256_before=w6_parent_sha256,
        w6_parent_sha256_after=w6_parent_sha256,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
        gpu_identity_sha256=gpu_identity_sha256,
        model_config_digest_sha256=model_config_digest_sha256,
        proof_batch_digest_sha256=proof_batch_digest_sha256,
        retained_support_digest_sha256=retained_support_digest_sha256,
        p1_live_conversion_receipt_sha256=sha256_file_bytes(p1_receipt_path),
        r1l_launch_runtime_receipt_sha256=sha256_file_bytes(r1l_receipt_path),
        base_readiness_receipt_sha256=str(live_base["base_readiness_receipt_sha256"]),
        base_sub2_surface_count=int(live_base["base_sub2_surface_count"]),
        base_sub2_surface_ids=tuple(live_base["base_sub2_surface_ids"]),
        base_activations_residuals_classification=str(
            live_base["base_activations_residuals_classification"]
        ),
        base_activations_residuals_is_sub2=False,
        base_ready_for_pre_full_stack_diagnostic=bool(
            live_base["base_ready_for_pre_full_stack_diagnostic"]
        ),
        base_ready_for_main_science=bool(live_base["base_ready_for_main_science"]),
        base_blocker_surface_names=tuple(live_base["base_blocker_surface_names"]),
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
        log_artifact_sha256=log_artifact_sha256,
        canonical_launch_artifact_sha256="",
        non_claims=LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_NON_CLAIMS,
    )
    validate_r2al_live_base_preflight(
        receipt_without_hash,
        p1_receipt=p1_receipt,
        r1l_receipt=r1l_receipt,
        p1_receipt_path=p1_receipt_path,
        r1l_receipt_path=r1l_receipt_path,
    )
    canonical_hash = compute_canonical_launch_artifact_sha256(receipt_without_hash.to_dict())
    receipt = replace(
        receipt_without_hash,
        canonical_launch_artifact_sha256=canonical_hash,
    )
    validate_launch_runtime_activation_residuals_receipt(receipt)
    return receipt


def validate_launch_runtime_activation_residuals_receipt(
    receipt: LaunchRuntimeActivationResidualsValidationReceipt,
) -> None:
    if receipt.schema_version != LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_RECEIPT_SCHEMA_VERSION:
        raise ValueError("R2-A-L launch receipt schema mismatch")
    if receipt.target_name != LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_TARGET_NAME:
        raise ValueError("R2-A-L launch receipt target mismatch")
    if receipt.proof_kind != PROOF_KIND_LAUNCH_RUNTIME_ACTIVATION_RESIDUALS:
        raise ValueError("R2-A-L launch receipt proof_kind mismatch")
    if receipt.mechanism_id != MECHANISM_ID:
        raise ValueError("R2-A-L launch receipt mechanism_id mismatch")
    if not receipt.live_readiness_row_flip_authorized:
        raise ValueError("R2-A-L launch receipt must authorize live row flip")
    if tuple(receipt.readiness_row_flip_authorized_surface_names) != AUTHORIZED_R2A_L_SURFACE_TUPLE:
        raise ValueError("R2-A-L launch receipt authorized surface tuple mismatch")
    if not receipt.ancestry_verified_at_launch_preflight:
        raise ValueError("R2-A-L launch receipt requires ancestry_verified_at_launch_preflight")
    if not receipt.live_base_preflight_pass:
        raise ValueError("R2-A-L launch receipt requires live_base_preflight_pass")
    if not receipt.launch_runtime_validation_pass:
        raise ValueError("R2-A-L launch receipt requires launch_runtime_validation_pass")
    if not receipt.m1_mechanism_proof_pass:
        raise ValueError("R2-A-L launch receipt requires m1_mechanism_proof_pass")
    if not receipt.m1_no_hidden_bf16_proof_pass:
        raise ValueError("R2-A-L launch receipt requires m1_no_hidden_bf16_proof_pass")
    if not receipt.gpu_memory_measurement_pass:
        raise ValueError("R2-A-L launch receipt requires gpu_memory_measurement_pass")
    manifest_embedded = _embedded_mapping(
        receipt.launch_manifest_embedded,
        field_name="launch_manifest_embedded",
        required_keys=R2AL_LAUNCH_MANIFEST_EMBEDDED_KEYS,
    )
    launch_source = _require_nonempty_string(
        receipt.launch_source_commit_sha,
        field_name="launch_source_commit_sha",
    )
    if launch_source != manifest_embedded["launch_source_commit_sha"]:
        raise ValueError("R2-A-L launch receipt launch_source_commit_sha mismatch")
    if receipt.launch_manifest_sha256 != compute_launch_manifest_sha256(manifest_embedded):
        raise ValueError("R2-A-L launch receipt launch_manifest_sha256 mismatch")
    env_embedded = _proof_env_embedded_mapping(
        receipt.proof_env_embedded,
        field_name="proof_env_embedded",
    )
    if receipt.proof_env_hash_sha256 != compute_r2al_proof_env_hash_sha256(env_embedded):
        raise ValueError("R2-A-L launch receipt proof_env_hash_sha256 mismatch")
    if not receipt.proof_command_argv:
        raise ValueError("R2-A-L launch receipt requires proof_command_argv")
    if receipt.w6_parent_sha256_before != W6_PARENT_SHA256_PINNED:
        raise ValueError("R2-A-L launch receipt w6_parent_sha256_before mismatch")
    if receipt.w6_parent_sha256_after != receipt.w6_parent_sha256_before:
        raise ValueError("R2-A-L launch receipt w6_parent_sha256_after mismatch")
    expected_gpu_identity = compute_gpu_identity_sha256(
        gpu_name=receipt.gpu_name,
        gpu_uuid=receipt.gpu_uuid,
        driver_version=receipt.driver_version,
        cuda_version=receipt.cuda_version,
        torch_version=receipt.torch_version,
    )
    if receipt.gpu_identity_sha256 != expected_gpu_identity:
        raise ValueError("R2-A-L launch receipt gpu_identity_sha256 mismatch")
    validate_activation_relief_measurement(receipt.activation_relief_measurement)
    if receipt.m1_registered_seam_tensor_full_pack_count != 0:
        raise ValueError("R2-A-L launch receipt m1 full_pack must be zero")
    if receipt.m1_seam_handle_pack_count < 4:
        raise ValueError("R2-A-L launch receipt m1 handle_pack must be >= 4")
    if receipt.m1_seam_remat_unpack_recompute_count_total <= 0:
        raise ValueError("R2-A-L launch receipt m1 unpack count must be > 0")
    if receipt.m1_saved_tensor_payload_bytes_delta <= 0:
        raise ValueError("R2-A-L launch receipt m1 payload delta must be > 0")
    if receipt.m1_forbidden_closure_tensor_count_total != 0:
        raise ValueError("R2-A-L launch receipt m1 forbidden_closure must be zero")
    if receipt.paired_run_count < 3:
        raise ValueError("R2-A-L launch receipt requires paired_run_count >= 3")
    expected_threshold = max(
        8 * 1024 * 1024,
        int(0.005 * receipt.cuda_peak_allocated_bytes_baseline_median),
    )
    if receipt.cuda_peak_reduction_threshold_bytes != expected_threshold:
        raise ValueError("R2-A-L launch receipt cuda threshold bytes mismatch")
    threshold_met = (
        receipt.cuda_peak_allocated_bytes_delta_median
        >= receipt.cuda_peak_reduction_threshold_bytes
    )
    if receipt.cuda_peak_reduction_threshold_met != threshold_met:
        raise ValueError("R2-A-L launch receipt cuda threshold met flag mismatch")
    if not receipt.loss_finite_main:
        raise ValueError("R2-A-L launch receipt requires finite main loss")
    if tuple(receipt.base_sub2_surface_ids) != CANONICAL_R2A_L_BASE_SUB2_SURFACE_IDS:
        raise ValueError("R2-A-L launch receipt base_sub2_surface_ids must be canonical sorted")
    if int(receipt.base_sub2_surface_count) != 4:
        raise ValueError("R2-A-L launch receipt base_sub2_surface_count must be 4")
    if receipt.base_activations_residuals_is_sub2:
        raise ValueError("R2-A-L launch receipt base_activations_residuals_is_sub2 must be false")
    if receipt.applier_base_surface_count_sub2 != 4:
        raise ValueError("R2-A-L launch receipt applier base sub2 count must be 4")
    if receipt.applier_result_sub2_surface_count != 5:
        raise ValueError("R2-A-L launch receipt applier result sub2 count must be 5")
    if receipt.applier_result_ready_for_main_science:
        raise ValueError("R2-A-L launch receipt applier must not set ready_for_main_science")
    if not receipt.applier_result_ready_for_pre_full_stack_diagnostic:
        raise ValueError(
            "R2-A-L launch receipt applier must set ready_for_pre_full_stack_diagnostic"
        )
    if tuple(receipt.applier_flipped_surface_ids) != AUTHORIZED_R2A_L_SURFACE_TUPLE:
        raise ValueError("R2-A-L launch receipt applier flipped surface ids mismatch")
    if not _require_nonempty_string(receipt.log_artifact_sha256, field_name="log_artifact_sha256"):
        raise ValueError("R2-A-L launch receipt requires log_artifact_sha256")
    if receipt.r2a_cpu_base_commit_sha != R2A_CPU_BASE_COMMIT_SHA:
        raise ValueError("R2-A-L launch receipt r2a_cpu_base_commit_sha mismatch")
    if manifest_embedded["r2a_cpu_base_commit_sha"] != receipt.r2a_cpu_base_commit_sha:
        raise ValueError("R2-A-L launch manifest embedded r2a_cpu_base_commit_sha mismatch")
    if receipt.non_claims != LAUNCH_RUNTIME_ACTIVATION_RESIDUALS_NON_CLAIMS:
        raise ValueError("R2-A-L launch receipt non-claims must be exact")
    mint_allowed = (
        receipt.launch_runtime_validation_pass
        and receipt.m1_mechanism_proof_pass
        and receipt.m1_no_hidden_bf16_proof_pass
        and receipt.gpu_memory_measurement_pass
        and receipt.live_base_preflight_pass
        and receipt.cuda_peak_reduction_threshold_met
        and receipt.paired_run_count >= 3
    )
    if not mint_allowed:
        raise ValueError("R2-A-L launch receipt mint_allowed conjunct failed")
    recomputed_hash = compute_canonical_launch_artifact_sha256(receipt.to_dict())
    if receipt.canonical_launch_artifact_sha256 != recomputed_hash:
        raise ValueError("R2-A-L launch receipt canonical_launch_artifact_sha256 mismatch")


def validate_launch_runtime_activation_residuals_artifacts(
    receipt: LaunchRuntimeActivationResidualsValidationReceipt,
    *,
    launch_manifest_bytes: bytes,
    env_snapshot_bytes: bytes,
    log_bytes: bytes | None,
) -> None:
    if hashlib.sha256(launch_manifest_bytes).hexdigest() != receipt.launch_manifest_sha256:
        raise ValueError("R2-A-L launch manifest bytes sha256 mismatch")
    try:
        manifest_payload = json.loads(launch_manifest_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("R2-A-L launch manifest bytes are not valid JSON") from exc
    if not isinstance(manifest_payload, dict):
        raise ValueError("R2-A-L launch manifest bytes must decode to an object")
    if str(manifest_payload.get("launch_source_commit_sha")) != receipt.launch_source_commit_sha:
        raise ValueError("R2-A-L launch manifest launch_source_commit_sha mismatch")
    if str(manifest_payload.get("r2a_cpu_base_commit_sha")) != receipt.r2a_cpu_base_commit_sha:
        raise ValueError("R2-A-L launch manifest r2a_cpu_base_commit_sha mismatch")
    try:
        env_payload = json.loads(env_snapshot_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("R2-A-L env snapshot bytes are not valid JSON") from exc
    if not isinstance(env_payload, dict):
        raise ValueError("R2-A-L env snapshot bytes must decode to an object")
    env_embedded = {str(key): str(value) for key, value in env_payload.items()}
    _require_nonempty_proof_env_keys(env_embedded, field_name="proof_env_embedded")
    if compute_r2al_proof_env_hash_sha256(env_embedded) != receipt.proof_env_hash_sha256:
        raise ValueError("R2-A-L env snapshot proof_env_hash_sha256 mismatch")
    if log_bytes is None or not log_bytes:
        raise ValueError("R2-A-L launch log snapshot bytes are required")
    if hashlib.sha256(log_bytes).hexdigest() != receipt.log_artifact_sha256:
        raise ValueError("R2-A-L launch log snapshot bytes sha256 mismatch")


def _read_r2al_launch_manifest_embedded() -> dict[str, str]:
    manifest_path = os.environ.get("R2AL_LAUNCH_MANIFEST_JSON", "").strip()
    if not manifest_path:
        raise R2alLaunchProofAbort("R2AL_LAUNCH_MANIFEST_JSON is required")
    payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise R2alLaunchProofAbort("R2AL launch manifest must decode to an object")
    embedded = {str(key): str(value) for key, value in payload.items()}
    missing = [key for key in R2AL_LAUNCH_MANIFEST_EMBEDDED_KEYS if key not in embedded]
    if missing:
        raise R2alLaunchProofAbort(
            "R2AL launch manifest missing required keys: " + ", ".join(missing)
        )
    if embedded["r2a_cpu_base_commit_sha"] != R2A_CPU_BASE_COMMIT_SHA:
        raise R2alLaunchProofAbort(
            f"R2AL launch manifest r2a_cpu_base_commit_sha mismatch "
            f"(got {embedded['r2a_cpu_base_commit_sha']}, expected {R2A_CPU_BASE_COMMIT_SHA})"
        )
    return embedded


def _read_r2al_proof_env_embedded() -> dict[str, str]:
    env_path = os.environ.get("R2AL_LAUNCH_ENV_JSON", "").strip()
    if not env_path:
        raise R2alLaunchProofAbort("R2AL_LAUNCH_ENV_JSON is required")
    payload = json.loads(Path(env_path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise R2alLaunchProofAbort("R2AL launch env snapshot must decode to an object")
    embedded = {str(key): str(value) for key, value in payload.items()}
    missing = [
        key
        for key in REQUIRED_R2AL_PROOF_ENV_KEYS
        if not str(embedded.get(key, "")).strip()
    ]
    if missing:
        raise R2alLaunchProofAbort(
            "R2AL launch proof env missing required keys: " + ", ".join(missing)
        )
    return embedded


def r2al_launch_log_at_mint_snapshot_path(*, receipt_json_path: str | None = None) -> Path:
    receipt_json = (
        receipt_json_path or os.environ.get("R2AL_LAUNCH_RECEIPT_JSON", "")
    ).strip()
    if not receipt_json:
        raise R2alLaunchProofAbort(
            "R2AL_LAUNCH_RECEIPT_JSON is required for launch log snapshot"
        )
    return Path(receipt_json).resolve().parent / R2AL_LAUNCH_LOG_AT_MINT_BASENAME


def _snapshot_r2al_launch_log_at_mint() -> tuple[Path, str]:
    log_path = Path(os.environ.get("R2AL_LAUNCH_LOG", "").strip())
    if not log_path.is_file():
        raise R2alLaunchProofAbort("R2AL_LAUNCH_LOG is required")
    log_bytes = log_path.read_bytes()
    if not log_bytes:
        raise R2alLaunchProofAbort("R2AL_LAUNCH_LOG must be non-empty")
    snapshot_path = r2al_launch_log_at_mint_snapshot_path()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(log_bytes)
    return snapshot_path, hashlib.sha256(log_bytes).hexdigest()


def _validate_r2al_measurements_for_mint(measurements: R2alLaunchProofMeasurements) -> None:
    if not measurements.m1_mechanism_proof_pass:
        raise R2alLaunchProofAbort("m1 mechanism proof failed")
    if not measurements.m1_no_hidden_bf16_proof_pass:
        raise R2alLaunchProofAbort("m1 no-hidden-bf16 proof failed")
    if not measurements.loss_finite_main:
        raise R2alLaunchProofAbort("main loss non-finite")
    if measurements.paired_run_count < 3:
        raise R2alLaunchProofAbort("paired_run_count must be >= 3")
    cuda_delta = (
        measurements.cuda_peak_allocated_bytes_baseline_median
        - measurements.cuda_peak_allocated_bytes_recompute_median
    )
    threshold = max(
        8 * 1024 * 1024,
        int(0.005 * measurements.cuda_peak_allocated_bytes_baseline_median),
    )
    if cuda_delta < threshold:
        raise R2alLaunchProofAbort(
            f"cuda peak reduction below threshold ({cuda_delta} < {threshold})"
        )
    validate_activation_relief_measurement(measurements.activation_relief_measurement)


def _execute_r2al_gpu_launch_measurement(
    *,
    model: Any,
    loader: Any,
    device: Any,
    hidden_size: int,
    cfg: Any,
    epochs: int,
) -> R2alLaunchProofMeasurements:
    import torch

    proof_batch = next(iter(loader))
    proof_total_steps = max(1, epochs * len(loader))
    extras_base = model.compute_train_extra_args(1, proof_total_steps)

    def _proof_child_batch(batch):
        inputs = batch["inputs"].to(device)
        labels = batch["labels"].to(device)
        sep_positions = batch["sep_positions"].to(device)
        bsz, seq_len = inputs.shape
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=device
        ).unsqueeze(0).expand(bsz, -1)
        return {
            "inputs": inputs,
            "labels": labels,
            "sep_positions": sep_positions,
            "position_ids": position_ids,
        }

    child_batch = _proof_child_batch(proof_batch)
    bsz, seq_len = child_batch["inputs"].shape

    def _collect_payload_bytes(run_fn) -> int:
        total = 0

        def pack_hook(tensor: torch.Tensor):
            nonlocal total
            total += int(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            run_fn()
        return total

    baseline_peaks: list[int] = []
    recompute_peaks: list[int] = []
    reserved_deltas: list[int] = []
    wall_clocks: list[float] = []
    last_telemetry: dict[str, object] = {}
    last_payload_delta = 0

    for _ in range(3):
        torch.cuda.reset_peak_memory_stats(device)
        baseline_payload = _collect_payload_bytes(
            lambda: _run_main_backward(model, child_batch, extras_base)
        )
        baseline_peak = int(torch.cuda.max_memory_allocated(device))
        baseline_reserved = int(torch.cuda.max_memory_reserved(device))

        torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        remat_payload, telemetry = _run_main_backward_with_m1(
            model=model,
            child_batch=child_batch,
            extras_base=extras_base,
        )
        wall_clocks.append(time.perf_counter() - started)
        recompute_peak = int(torch.cuda.max_memory_allocated(device))
        recompute_reserved = int(torch.cuda.max_memory_reserved(device))
        baseline_peaks.append(baseline_peak)
        recompute_peaks.append(recompute_peak)
        reserved_deltas.append(baseline_reserved - recompute_reserved)
        last_telemetry = telemetry
        last_payload_delta = baseline_payload - remat_payload

    def _median_int(values: Sequence[int]) -> int:
        ordered = sorted(int(item) for item in values)
        return ordered[len(ordered) // 2]

    mechanism_pass, no_hidden = validate_r2al_m1_mechanism_telemetry(
        {
            **last_telemetry,
            "m1_saved_tensor_payload_bytes_delta": last_payload_delta,
        }
    )
    activation_relief_measurement = {
        "peak_allocated_bytes": _median_int(recompute_peaks),
        "peak_reserved_bytes": _median_int(
            [baseline_peaks[i] - reserved_deltas[i] for i in range(len(baseline_peaks))]
        ),
        "wall_clock_per_step_seconds": sum(wall_clocks) / len(wall_clocks),
        "max_safe_batch_size": int(bsz),
        "effective_exposure_per_step": int(bsz * seq_len),
    }
    return R2alLaunchProofMeasurements(
        m1_seam_handle_pack_count=int(last_telemetry["seam_handle_pack_count"]),
        m1_registered_seam_tensor_full_pack_count=int(
            last_telemetry["registered_seam_tensor_full_pack_count"]
        ),
        m1_seam_remat_unpack_recompute_count_total=int(
            last_telemetry["m1_seam_remat_unpack_recompute_count_total"]
        ),
        m1_saved_tensor_payload_bytes_delta=last_payload_delta,
        m1_forbidden_closure_tensor_count_total=int(
            last_telemetry["forbidden_closure_tensor_count_total"]
        ),
        m1_registered_seam_tensor_in_closure_count=int(
            last_telemetry["registered_seam_tensor_in_closure_count"]
        ),
        m1_no_hidden_bf16_proof_pass=no_hidden,
        m1_mechanism_proof_pass=mechanism_pass,
        loss_finite_main=True,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=_median_int(baseline_peaks),
        cuda_peak_allocated_bytes_recompute_median=_median_int(recompute_peaks),
        cuda_peak_reserved_bytes_delta_median=_median_int(reserved_deltas),
        activation_relief_measurement=activation_relief_measurement,
    )


def _run_main_backward(model: Any, child_batch: Mapping[str, Any], extras: Mapping[str, Any]) -> None:
    import torch

    model.zero_grad(set_to_none=True)
    _new_carry, loss_main, _metrics = model(None, child_batch, **extras)
    if not torch.isfinite(loss_main):
        raise R2alLaunchProofAbort(f"main loss non-finite: {loss_main.item()}")
    loss_main.backward()


def _run_main_backward_with_m1(
    *,
    model: Any,
    child_batch: Mapping[str, Any],
    extras_base: Mapping[str, Any],
) -> tuple[int, dict[str, object]]:
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(model=model.model)
    extras = {**extras_base, "activation_codec_seam": codec}

    def _collect_payload_bytes(run_fn) -> int:
        import torch

        total = 0

        def pack_hook(tensor: torch.Tensor):
            nonlocal total
            total += int(tensor.numel() * tensor.element_size())
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(pack_hook, lambda tensor: tensor):
            run_fn()
        return total

    payload = _collect_payload_bytes(
        lambda: _run_main_backward_with_codec(model, child_batch, extras, codec)
    )
    telemetry = dict(codec.telemetry())
    return payload, telemetry


def _run_main_backward_with_codec(
    model: Any,
    child_batch: Mapping[str, Any],
    extras: Mapping[str, Any],
    codec: Any,
) -> None:
    with codec.saved_tensor_hook_scope():
        _run_main_backward(model, child_batch, extras)


def run_r2al_gpu_launch_proof(
    *,
    model: Any,
    loader: Any,
    device: Any,
    hidden_size: int,
    cfg: Any,
    epochs: int,
    proof_command_argv: Sequence[str],
    w6_parent_path: str,
    cuda_is_available_fn: Callable[[], bool] | None = None,
    measurement_runner: Callable[[], R2alLaunchProofMeasurements] | None = None,
) -> LaunchRuntimeActivationResidualsValidationReceipt:
    import torch

    _cuda_available = cuda_is_available_fn or torch.cuda.is_available
    if not _cuda_available():
        raise RuntimeError("R2-A-L launch proof requires CUDA")
    if os.environ.get("R2AL_ANCESTRY_VERIFIED") != "1":
        raise R2alLaunchProofAbort("R2AL_ANCESTRY_VERIFIED must be 1")
    if os.environ.get("R2AL_LIVE_BASE_PREFLIGHT") != "PASS":
        raise R2alLaunchProofAbort("R2AL_LIVE_BASE_PREFLIGHT must be PASS")

    w6_path = Path(w6_parent_path)
    if not w6_path.is_file():
        raise R2alLaunchProofAbort(f"w6 parent path not found: {w6_parent_path}")
    w6_before = sha256_file_bytes(w6_path)
    if w6_before != W6_PARENT_SHA256_PINNED:
        raise R2alLaunchProofAbort(
            f"w6 parent sha256 mismatch (got {w6_before}, expected pinned hash)"
        )

    manifest_embedded = _read_r2al_launch_manifest_embedded()
    proof_env_embedded = _read_r2al_proof_env_embedded()
    launch_source_commit_sha = manifest_embedded["launch_source_commit_sha"]
    clean_run_dir_sha256 = os.environ.get("R2AL_CLEAN_RUN_DIR_SHA256", "").strip()
    if not clean_run_dir_sha256:
        raise R2alLaunchProofAbort("R2AL_CLEAN_RUN_DIR_SHA256 is required")

    p1_receipt, r1l_receipt = load_r2al_base_receipts_from_env(proof_env_embedded)
    p1_path = Path(proof_env_embedded["R2AL_P1_RECEIPT_JSON"])
    r1l_path = Path(proof_env_embedded["R2AL_R1L_RECEIPT_JSON"])
    verify_r2al_banked_p1_ancestor_preflight(
        p1_receipt=p1_receipt,
        launch_source_commit_sha=launch_source_commit_sha,
        repo_root=_resolve_r2al_repo_root(proof_env=proof_env_embedded),
    )
    derive_r2al_live_base_fields(p1_receipt=p1_receipt, r1l_receipt=r1l_receipt)

    if measurement_runner is None:
        measurements = _execute_r2al_gpu_launch_measurement(
            model=model,
            loader=loader,
            device=device,
            hidden_size=hidden_size,
            cfg=cfg,
            epochs=epochs,
        )
    else:
        measurements = measurement_runner()

    _validate_r2al_measurements_for_mint(measurements)

    w6_after = sha256_file_bytes(w6_path)
    if w6_after != w6_before:
        raise R2alLaunchProofAbort("w6 parent mutated during launch proof")

    _, log_artifact_sha256 = _snapshot_r2al_launch_log_at_mint()

    model_config_digest_sha256 = _canonical_json_sha256(
        {
            "hidden_size": hidden_size,
            "n_layers": cfg.n_layers,
            "num_heads": cfg.num_heads,
            "expansion": cfg.expansion,
            "H_cycles": cfg.H_cycles,
            "L_cycles": cfg.L_cycles,
            "half_layers": cfg.half_layers,
            "bp_min_steps": cfg.bp_min_steps,
            "bp_max_steps": cfg.bp_max_steps,
            "max_seq_len": cfg.max_seq_len,
        }
    )
    proof_batch = next(iter(loader))
    proof_batch_digest_sha256 = _canonical_json_sha256(
        {
            "inputs_shape": tuple(proof_batch["inputs"].shape),
            "labels_shape": tuple(proof_batch["labels"].shape),
            "sep_positions_shape": tuple(proof_batch["sep_positions"].shape),
        }
    )

    if measurement_runner is not None:
        gpu_name = os.environ.get("R2AL_GPU_NAME", "synthetic-gpu")
        gpu_uuid = os.environ.get("R2AL_GPU_UUID", "gpu-uuid-test")
        driver_version = os.environ.get("R2AL_GPU_DRIVER_VERSION", "550.00")
        cuda_version = os.environ.get("R2AL_CUDA_VERSION", "12.4")
        torch_version = str(torch.__version__)
    else:
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_uuid = os.environ.get("R2AL_GPU_UUID", "").strip() or (
            f"cuda:{torch.cuda.current_device()}:{getattr(props, 'name', gpu_name)}"
        )
        driver_version = str(getattr(torch.version, "cuda", "") or "")
        cuda_version = driver_version
        torch_version = str(torch.__version__)

    return build_launch_runtime_activation_residuals_validation_receipt(
        launch_source_commit_sha=launch_source_commit_sha,
        launch_manifest_embedded=manifest_embedded,
        proof_env_embedded=proof_env_embedded,
        proof_command_argv=tuple(str(arg) for arg in proof_command_argv),
        clean_run_dir_sha256=clean_run_dir_sha256,
        w6_parent_path=w6_parent_path,
        w6_parent_sha256=w6_before,
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version or "unknown",
        cuda_version=cuda_version or "unknown",
        torch_version=torch_version,
        model_config_digest_sha256=model_config_digest_sha256,
        proof_batch_digest_sha256=proof_batch_digest_sha256,
        retained_support_digest_sha256=_canonical_json_sha256([]),
        p1_receipt=p1_receipt,
        r1l_receipt=r1l_receipt,
        p1_receipt_path=p1_path,
        r1l_receipt_path=r1l_path,
        measurements=measurements,
        log_artifact_sha256=log_artifact_sha256,
    )
