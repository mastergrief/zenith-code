"""Lossless activation/saved-tensor relief contract for HRM-Text-1.58.

This slice lands the interface and CPU-provable lossless recompute path only.
Real peak-memory relief and wall-clock tradeoff receipts are deferred to a
gpu:0 resource-lane run because CPU tests cannot measure CUDA activation peaks.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


ACTIVATION_RELIEF_SCHEMA_VERSION = "hrm_text_158_activation_relief/v0.lossless_recompute"
BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_backward_saved_tensors_recompute/v0.saved_tensor_hook"
)
BACKWARD_RECOMPUTE_TARGET_NAME = "step3a1_backward_saved_tensors_recompute"
ACTIVATION_RESIDUALS_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_activation_residuals_fail_closed/v0.live_tensor_seams"
)
ACTIVATION_RESIDUALS_FAIL_CLOSED_TARGET_NAME = (
    "step3a2_activation_residuals_fail_closed"
)
ACTIVATION_RESIDUAL_TARGET_FAMILIES = (
    "recurrent.z_L_update",
    "recurrent.z_H_update",
    "residual.post_attn",
    "residual.post_mlp",
)
ACTIVATION_RESIDUALS_BLOCKED_REASON = (
    "fail-closed activation/residual live-tensor harness only; live BF16/FP "
    "tensor seams are observed and no real sub2/remat/offload/no-hidden-BF16 "
    "proof is present"
)
ZL_INIT_FP_EXCEPTION_CLASSIFICATION = "fp_exception_non_eligible_hrm_tensor"
ZL_INIT_FP_EXCEPTION_REGISTRY_ANCHOR = (
    "calm/hrm_text_158/native_full_stack/fp_exceptions.py:30"
)
ZL_INIT_HRM_SOURCE_ANCHOR = "calm/hrm_text_158/hrm.py:122"

MODE_OFF = "off"
MODE_LOSSLESS_RECOMPUTE = "lossless_recompute"
MODE_LOSSY_ACTIVATION_STORAGE = "lossy_activation_storage"

TIER1_LOSSLESS_RECOMPUTE = "tier1_lossless_recompute"
TIER2_LOSSY_ACTIVATION_STORAGE_DEFERRED = "tier2_lossy_activation_storage_deferred"

TARGET_GRAD_ENABLED_RECURRENCE = "grad_enabled_recurrence"
SAVED_TENSOR_HOOK_PROOF_SOURCE = "torch.autograd.graph.saved_tensors_hooks"
BACKWARD_RECOMPUTE_NO_EXTRA_INTERNAL_PAYLOAD_CLAIM = (
    "no extra stored internal recurrence-block saved payload; boundary z_H/z_L "
    "inputs are accounted under activations_residuals"
)
BACKWARD_RECOMPUTE_NON_CLAIMS = (
    "lossless recompute is a representation/backward-saved-tensor property, not learning, acquisition, retention, or throughput",
    "boundary z_H/z_L activation/residual tensors remain live BF16/FP boundary state and are not sub2 in Step 3A1",
    "attention/KV buffers, optimizer credit state, and native kernel residency remain outside this proof",
    "proof is CPU/small-smoke only and does not launch GPU, write checkpoints, or mutate .pt artifacts",
)
ACTIVATION_RESIDUALS_FAIL_CLOSED_NON_CLAIMS = (
    "activation/residual live-tensor observation is not learning, acquisition, retention, or throughput",
    "observer callbacks returning BF16/FP tensors are blocker evidence, not sub2 credit",
    "this receipt does not cover attention/KV buffers, optimizer credit state, or native hot-path residency",
    "this receipt does not launch GPU, prove CUDA memory relief, write checkpoints, or mutate .pt artifacts",
    "zL_init is cross-referenced as existing non-eligible persistent FP debt, not solved by activation seams",
)

DEFERRED_GPU_MEASUREMENT_NOTE = (
    "lossless equivalence is CPU-provable now; activation-memory relief, "
    "wall-clock/step, max-safe-batch, and exposure/step require a deferred "
    "gpu:0 measurement receipt"
)

REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS = (
    "peak_allocated_bytes",
    "peak_reserved_bytes",
    "wall_clock_per_step_seconds",
    "max_safe_batch_size",
    "effective_exposure_per_step",
)

TERMINAL_DEPENDENT_TUNING_FIELDS = (
    "profile_batch_candidates",
    "profile_sequence_lengths",
    "profile_bp_steps",
    "resource_lane_device",
)


@dataclass(frozen=True)
class ActivationReliefPolicy:
    """Default-off lossless recompute policy.

    `use_reentrant=False` and RNG preservation are mandatory in this slice so
    the checkpointed block remains deterministic and safe for strict equivalence
    tests.
    """

    mode: str = MODE_OFF
    target: str = TARGET_GRAD_ENABLED_RECURRENCE
    use_reentrant: bool = False
    preserve_rng_state: bool = True

    @property
    def enabled(self) -> bool:
        return self.mode == MODE_LOSSLESS_RECOMPUTE

    @property
    def tier(self) -> str:
        if self.mode == MODE_LOSSLESS_RECOMPUTE:
            return TIER1_LOSSLESS_RECOMPUTE
        if self.mode == MODE_LOSSY_ACTIVATION_STORAGE:
            return TIER2_LOSSY_ACTIVATION_STORAGE_DEFERRED
        return MODE_OFF

    def validate(self) -> "ActivationReliefPolicy":
        if self.mode == MODE_LOSSY_ACTIVATION_STORAGE:
            raise NotImplementedError(
                "lossy activation storage is Tier-2/deferred and needs full "
                "acquisition re-validation; this slice implements only "
                "lossless recompute"
            )
        if self.mode not in {MODE_OFF, MODE_LOSSLESS_RECOMPUTE}:
            raise ValueError(f"unknown activation relief mode: {self.mode!r}")
        if self.target != TARGET_GRAD_ENABLED_RECURRENCE:
            raise ValueError(
                f"unsupported activation relief target {self.target!r}; "
                f"expected {TARGET_GRAD_ENABLED_RECURRENCE!r}"
            )
        if self.use_reentrant:
            raise ValueError("lossless recompute requires use_reentrant=False")
        if not self.preserve_rng_state:
            raise ValueError("lossless recompute requires preserve_rng_state=True")
        return self


@dataclass(frozen=True)
class ActivationReliefDecision:
    level: str
    rec_idx: int
    scheduled_grad_enabled: bool
    checkpoint: bool


@dataclass(frozen=True)
class BackwardRecomputeSavedTensorReceipt:
    """Step 3A1 receipt for the gated lossless recompute backward-saved path."""

    schema_version: str
    target_name: str
    policy_mode: str
    target: str
    use_reentrant: bool
    preserve_rng_state: bool
    default_runtime_sub2_claim: bool
    activations_residuals_sub2_claim: bool
    lossless_not_lossy: bool
    kv_cache_side_effects_forbidden: bool
    no_gpu: bool
    no_pt: bool
    no_learning_or_throughput_claim: bool
    H_cycles: int
    L_cycles: int
    bp_steps: int
    checkpointed_grad_enabled_call_count: int
    checkpointed_grad_enabled_calls: tuple[tuple[str, int], ...]
    saved_tensor_hook_source: str
    boundary_tensor_shape: tuple[int, ...]
    boundary_tensor_dtype: str
    observed_boundary_tensor_count: int
    expected_boundary_tensor_count: int
    observed_checkpoint_dummy_tensor_count: int
    expected_checkpoint_dummy_tensor_count: int
    observed_internal_payload_tensor_count: int
    baseline_saved_tensor_count: int
    recompute_saved_tensor_count: int
    no_extra_internal_payload_claim: str
    caveat: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "policy_mode": self.policy_mode,
            "target": self.target,
            "use_reentrant": self.use_reentrant,
            "preserve_rng_state": self.preserve_rng_state,
            "default_runtime_sub2_claim": self.default_runtime_sub2_claim,
            "activations_residuals_sub2_claim": self.activations_residuals_sub2_claim,
            "lossless_not_lossy": self.lossless_not_lossy,
            "kv_cache_side_effects_forbidden": self.kv_cache_side_effects_forbidden,
            "no_gpu": self.no_gpu,
            "no_pt": self.no_pt,
            "no_learning_or_throughput_claim": self.no_learning_or_throughput_claim,
            "H_cycles": self.H_cycles,
            "L_cycles": self.L_cycles,
            "bp_steps": self.bp_steps,
            "checkpointed_grad_enabled_call_count": self.checkpointed_grad_enabled_call_count,
            "checkpointed_grad_enabled_calls": [
                list(call) for call in self.checkpointed_grad_enabled_calls
            ],
            "saved_tensor_hook_source": self.saved_tensor_hook_source,
            "boundary_tensor_shape": list(self.boundary_tensor_shape),
            "boundary_tensor_dtype": self.boundary_tensor_dtype,
            "observed_boundary_tensor_count": self.observed_boundary_tensor_count,
            "expected_boundary_tensor_count": self.expected_boundary_tensor_count,
            "observed_checkpoint_dummy_tensor_count": self.observed_checkpoint_dummy_tensor_count,
            "expected_checkpoint_dummy_tensor_count": self.expected_checkpoint_dummy_tensor_count,
            "observed_internal_payload_tensor_count": self.observed_internal_payload_tensor_count,
            "baseline_saved_tensor_count": self.baseline_saved_tensor_count,
            "recompute_saved_tensor_count": self.recompute_saved_tensor_count,
            "no_extra_internal_payload_claim": self.no_extra_internal_payload_claim,
            "caveat": self.caveat,
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class ActivationResidualLiveTensorFamilyObservation:
    family: str
    observed_count: int
    shapes: tuple[tuple[int, ...], ...]
    dtypes: tuple[str, ...]
    devices: tuple[str, ...]
    requires_grad_values: tuple[bool, ...]
    mechanism: str = "observer_returns_original_tensor"

    def to_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "observed_count": self.observed_count,
            "shapes": [list(shape) for shape in self.shapes],
            "dtypes": list(self.dtypes),
            "devices": list(self.devices),
            "requires_grad_values": list(self.requires_grad_values),
            "mechanism": self.mechanism,
        }


@dataclass(frozen=True)
class ZLInitPersistentNonClaim:
    name: str
    classification: str
    registry_anchor: str
    source_anchor: str
    dtype: str
    shape: tuple[int, ...]
    persistent: bool
    non_claim: str

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "classification": self.classification,
            "registry_anchor": self.registry_anchor,
            "source_anchor": self.source_anchor,
            "dtype": self.dtype,
            "shape": list(self.shape),
            "persistent": self.persistent,
            "non_claim": self.non_claim,
        }


@dataclass(frozen=True)
class ActivationResidualsFailClosedReceipt:
    schema_version: str
    target_name: str
    target_families: tuple[str, ...]
    activations_residuals_sub2_claim: bool
    real_sub2_or_remat_or_offload_mechanism_present: bool
    no_hidden_bf16_authority_proven: bool
    gpu_memory_receipt_present: bool
    lossy_or_compression_claim: bool
    ready_to_flip: bool
    blocked_reason: str
    observed_families: tuple[ActivationResidualLiveTensorFamilyObservation, ...]
    zL_init_non_claim: ZLInitPersistentNonClaim
    smallest_missing_proof: str
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "target_families": list(self.target_families),
            "activations_residuals_sub2_claim": self.activations_residuals_sub2_claim,
            "real_sub2_or_remat_or_offload_mechanism_present": (
                self.real_sub2_or_remat_or_offload_mechanism_present
            ),
            "no_hidden_bf16_authority_proven": self.no_hidden_bf16_authority_proven,
            "gpu_memory_receipt_present": self.gpu_memory_receipt_present,
            "lossy_or_compression_claim": self.lossy_or_compression_claim,
            "ready_to_flip": self.ready_to_flip,
            "blocked_reason": self.blocked_reason,
            "observed_families": [
                observation.to_dict() for observation in self.observed_families
            ],
            "zL_init_non_claim": self.zL_init_non_claim.to_dict(),
            "smallest_missing_proof": self.smallest_missing_proof,
            "non_claims": list(self.non_claims),
        }


def normalize_activation_relief_policy(
    policy: ActivationReliefPolicy | Mapping[str, object] | str | bool | None,
) -> ActivationReliefPolicy:
    """Normalize user-facing policy shapes at the HRM boundary.

    Keeping this at the HRM boundary prevents policy objects from leaking down
    into lower `seq_info` dictionaries as ignored extras.
    """

    if policy is None or policy is False:
        return ActivationReliefPolicy().validate()
    if policy is True:
        return ActivationReliefPolicy(mode=MODE_LOSSLESS_RECOMPUTE).validate()
    if isinstance(policy, ActivationReliefPolicy):
        return policy.validate()
    if isinstance(policy, str):
        return ActivationReliefPolicy(mode=policy).validate()
    if isinstance(policy, Mapping):
        allowed = {"mode", "target", "use_reentrant", "preserve_rng_state"}
        extra = set(policy) - allowed
        if extra:
            raise ValueError(f"unknown activation relief policy fields: {sorted(extra)}")
        return ActivationReliefPolicy(**policy).validate()
    raise TypeError(f"unsupported activation relief policy type: {type(policy).__name__}")


def should_checkpoint_recurrence(
    policy: ActivationReliefPolicy | Mapping[str, object] | str | bool | None,
    *,
    module_training: bool,
    kv_cache_present: bool,
    outer_grad_enabled: bool,
    scheduled_grad_enabled: bool,
) -> bool:
    """Return whether the current H/L recurrence call should be checkpointed."""

    normalized = normalize_activation_relief_policy(policy)
    if not normalized.enabled:
        return False
    if kv_cache_present:
        raise ValueError(
            "activation relief policy is enabled but kv_cache is present; "
            "checkpointing cache-update side effects is forbidden"
        )
    return bool(module_training and outer_grad_enabled and scheduled_grad_enabled)


def recurrence_checkpoint_decisions(
    policy: ActivationReliefPolicy | Mapping[str, object] | str | bool | None,
    *,
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
    module_training: bool = True,
    kv_cache_present: bool = False,
    outer_grad_enabled: bool = True,
) -> tuple[ActivationReliefDecision, ...]:
    """Enumerate checkpoint decisions using the same H/L schedule as HRM."""

    decisions: list[ActivationReliefDecision] = []
    H_bp_steps = min(H_cycles, bp_steps - 1)
    L_bp_steps = bp_steps - H_bp_steps
    for i in range(H_cycles):
        for k in range(i * L_cycles, (i + 1) * L_cycles):
            scheduled = outer_grad_enabled and (k >= H_cycles * L_cycles - L_bp_steps)
            decisions.append(
                ActivationReliefDecision(
                    level="L",
                    rec_idx=k,
                    scheduled_grad_enabled=scheduled,
                    checkpoint=should_checkpoint_recurrence(
                        policy,
                        module_training=module_training,
                        kv_cache_present=kv_cache_present,
                        outer_grad_enabled=outer_grad_enabled,
                        scheduled_grad_enabled=scheduled,
                    ),
                )
            )
        scheduled = outer_grad_enabled and (i >= H_cycles - H_bp_steps)
        decisions.append(
            ActivationReliefDecision(
                level="H",
                rec_idx=i,
                scheduled_grad_enabled=scheduled,
                checkpoint=should_checkpoint_recurrence(
                    policy,
                    module_training=module_training,
                    kv_cache_present=kv_cache_present,
                    outer_grad_enabled=outer_grad_enabled,
                    scheduled_grad_enabled=scheduled,
                ),
            )
        )
    return tuple(decisions)


def _require_int(value: object, *, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an integer, got {type(value).__name__}")
    if value < 0:
        raise ValueError(f"{field_name} must be non-negative, got {value!r}")
    return value


def build_backward_recompute_saved_tensor_receipt(
    *,
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
    saved_tensor_proof: Mapping[str, object],
    policy: ActivationReliefPolicy | Mapping[str, object] | str | bool | None = MODE_LOSSLESS_RECOMPUTE,
    kv_cache_present: bool = False,
) -> BackwardRecomputeSavedTensorReceipt:
    """Build the Step 3A1 receipt from an actual saved-tensor-hook proof."""

    normalized = normalize_activation_relief_policy(policy)
    if not normalized.enabled:
        raise ValueError("Step 3A1 receipt requires enabled lossless recompute")
    checkpointed = tuple(
        (decision.level, decision.rec_idx)
        for decision in recurrence_checkpoint_decisions(
            normalized,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            bp_steps=bp_steps,
            kv_cache_present=kv_cache_present,
        )
        if decision.checkpoint
    )
    checkpointed_count = len(checkpointed)
    expected_boundary_count = checkpointed_count * 2
    expected_dummy_count = checkpointed_count
    observed_boundary_count = _require_int(
        saved_tensor_proof.get("observed_boundary_tensor_count"),
        field_name="observed_boundary_tensor_count",
    )
    observed_dummy_count = _require_int(
        saved_tensor_proof.get("observed_checkpoint_dummy_tensor_count"),
        field_name="observed_checkpoint_dummy_tensor_count",
    )
    observed_internal_payload_count = _require_int(
        saved_tensor_proof.get("observed_internal_payload_tensor_count"),
        field_name="observed_internal_payload_tensor_count",
    )
    baseline_saved_tensor_count = _require_int(
        saved_tensor_proof.get("baseline_saved_tensor_count"),
        field_name="baseline_saved_tensor_count",
    )
    recompute_saved_tensor_count = _require_int(
        saved_tensor_proof.get("recompute_saved_tensor_count"),
        field_name="recompute_saved_tensor_count",
    )
    boundary_shape = tuple(int(dim) for dim in saved_tensor_proof.get("boundary_tensor_shape", ()))
    boundary_dtype = str(saved_tensor_proof.get("boundary_tensor_dtype", ""))
    if not boundary_shape:
        raise ValueError("boundary_tensor_shape must be non-empty")
    if not boundary_dtype.strip():
        raise ValueError("boundary_tensor_dtype must be non-empty")
    if saved_tensor_proof.get("saved_tensor_hook_source") != SAVED_TENSOR_HOOK_PROOF_SOURCE:
        raise ValueError("Step 3A1 proof must come from saved_tensors_hooks")

    receipt = BackwardRecomputeSavedTensorReceipt(
        schema_version=BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION,
        target_name=BACKWARD_RECOMPUTE_TARGET_NAME,
        policy_mode=normalized.mode,
        target=normalized.target,
        use_reentrant=normalized.use_reentrant,
        preserve_rng_state=normalized.preserve_rng_state,
        default_runtime_sub2_claim=False,
        activations_residuals_sub2_claim=False,
        lossless_not_lossy=True,
        kv_cache_side_effects_forbidden=True,
        no_gpu=True,
        no_pt=True,
        no_learning_or_throughput_claim=True,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        bp_steps=bp_steps,
        checkpointed_grad_enabled_call_count=checkpointed_count,
        checkpointed_grad_enabled_calls=checkpointed,
        saved_tensor_hook_source=SAVED_TENSOR_HOOK_PROOF_SOURCE,
        boundary_tensor_shape=boundary_shape,
        boundary_tensor_dtype=boundary_dtype,
        observed_boundary_tensor_count=observed_boundary_count,
        expected_boundary_tensor_count=expected_boundary_count,
        observed_checkpoint_dummy_tensor_count=observed_dummy_count,
        expected_checkpoint_dummy_tensor_count=expected_dummy_count,
        observed_internal_payload_tensor_count=observed_internal_payload_count,
        baseline_saved_tensor_count=baseline_saved_tensor_count,
        recompute_saved_tensor_count=recompute_saved_tensor_count,
        no_extra_internal_payload_claim=BACKWARD_RECOMPUTE_NO_EXTRA_INTERNAL_PAYLOAD_CLAIM,
        caveat=(
            "no stored INTERNAL recurrence-block backward payload via recompute; "
            "this is not a zero-tensors-anywhere claim and does not cover live "
            "z_H/z_L activation/residual boundary state"
        ),
        non_claims=BACKWARD_RECOMPUTE_NON_CLAIMS,
    )
    validate_backward_recompute_saved_tensor_receipt(receipt)
    return receipt


def validate_backward_recompute_saved_tensor_receipt(
    receipt: BackwardRecomputeSavedTensorReceipt,
) -> None:
    if receipt.schema_version != BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("backward recompute receipt schema mismatch")
    if receipt.target_name != BACKWARD_RECOMPUTE_TARGET_NAME:
        raise ValueError("backward recompute receipt target mismatch")
    if receipt.policy_mode != MODE_LOSSLESS_RECOMPUTE:
        raise ValueError("backward recompute receipt requires lossless_recompute mode")
    if receipt.target != TARGET_GRAD_ENABLED_RECURRENCE:
        raise ValueError("backward recompute receipt targets only grad-enabled recurrence")
    if receipt.use_reentrant:
        raise ValueError("backward recompute receipt requires use_reentrant=False")
    if not receipt.preserve_rng_state:
        raise ValueError("backward recompute receipt requires preserved RNG state")
    if receipt.default_runtime_sub2_claim or receipt.activations_residuals_sub2_claim:
        raise ValueError("Step 3A1 cannot claim default runtime or activations_residuals sub2")
    if not (
        receipt.lossless_not_lossy
        and receipt.kv_cache_side_effects_forbidden
        and receipt.no_gpu
        and receipt.no_pt
        and receipt.no_learning_or_throughput_claim
    ):
        raise ValueError("backward recompute receipt is missing required non-claim flags")
    if receipt.observed_boundary_tensor_count != receipt.expected_boundary_tensor_count:
        raise ValueError("saved-tensor proof boundary tensor count mismatch")
    if (
        receipt.observed_checkpoint_dummy_tensor_count
        != receipt.expected_checkpoint_dummy_tensor_count
    ):
        raise ValueError("saved-tensor proof checkpoint dummy tensor count mismatch")
    if receipt.observed_internal_payload_tensor_count != 0:
        raise ValueError("saved-tensor proof observed internal recurrence payload tensors")
    expected_total = (
        receipt.expected_boundary_tensor_count
        + receipt.expected_checkpoint_dummy_tensor_count
    )
    if receipt.recompute_saved_tensor_count != expected_total:
        raise ValueError("recompute saved-tensor total must equal boundary + dummy tensors")
    if receipt.baseline_saved_tensor_count <= receipt.recompute_saved_tensor_count:
        raise ValueError("saved-tensor proof must show recompute reduces HRM-forward saves")
    if (
        receipt.no_extra_internal_payload_claim
        != BACKWARD_RECOMPUTE_NO_EXTRA_INTERNAL_PAYLOAD_CLAIM
    ):
        raise ValueError("backward recompute receipt missing exact internal-payload claim")
    if "not a zero-tensors-anywhere claim" not in receipt.caveat:
        raise ValueError("backward recompute caveat must avoid zero-tensors-anywhere overclaim")
    if "z_H/z_L activation/residual" not in receipt.caveat:
        raise ValueError("backward recompute caveat must preserve activation/residual boundary")


def _require_nonempty_string(value: object, *, field_name: str) -> str:
    if value is None:
        raise ValueError(f"{field_name} must be present")
    text = str(value)
    if not text.strip():
        raise ValueError(f"{field_name} must be non-empty")
    return text


def _shape_tuple(value: object, *, field_name: str) -> tuple[int, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of integer dimensions")
    shape = tuple(int(dim) for dim in value)
    if not shape:
        raise ValueError(f"{field_name} must be non-empty")
    return shape


def _summarize_activation_residual_live_tensor_families(
    seam_events: Sequence[Mapping[str, object]],
) -> tuple[ActivationResidualLiveTensorFamilyObservation, ...]:
    grouped: dict[str, list[Mapping[str, object]]] = {
        family: [] for family in ACTIVATION_RESIDUAL_TARGET_FAMILIES
    }
    for event in seam_events:
        family = event.get("family", event.get("name"))
        if family not in grouped:
            raise ValueError(
                "activation/residual seam event family must be one of exactly "
                f"{ACTIVATION_RESIDUAL_TARGET_FAMILIES!r}; got {family!r}"
            )
        grouped[str(family)].append(event)

    missing = [family for family, events in grouped.items() if not events]
    if missing:
        raise ValueError(
            "activation/residual receipt missing required target families: "
            + ", ".join(missing)
        )

    observations: list[ActivationResidualLiveTensorFamilyObservation] = []
    for family in ACTIVATION_RESIDUAL_TARGET_FAMILIES:
        events = grouped[family]
        observations.append(
            ActivationResidualLiveTensorFamilyObservation(
                family=family,
                observed_count=len(events),
                shapes=tuple(
                    sorted(
                        {
                            _shape_tuple(
                                event.get("shape", ()),
                                field_name=f"{family}.shape",
                            )
                            for event in events
                        }
                    )
                ),
                dtypes=tuple(
                    sorted(
                        {
                            _require_nonempty_string(
                                event.get("dtype", ""),
                                field_name=f"{family}.dtype",
                            )
                            for event in events
                        }
                    )
                ),
                devices=tuple(
                    sorted(
                        {
                            _require_nonempty_string(
                                event.get("device", ""),
                                field_name=f"{family}.device",
                            )
                            for event in events
                        }
                    )
                ),
                requires_grad_values=tuple(
                    sorted({bool(event.get("requires_grad", False)) for event in events})
                ),
            )
        )
    return tuple(observations)


def _zL_init_non_claim_from_observation(
    zL_init_observation: Mapping[str, object],
) -> ZLInitPersistentNonClaim:
    return ZLInitPersistentNonClaim(
        name=_require_nonempty_string(
            zL_init_observation.get("name", "zL_init"),
            field_name="zL_init.name",
        ),
        classification=_require_nonempty_string(
            zL_init_observation.get(
                "classification",
                ZL_INIT_FP_EXCEPTION_CLASSIFICATION,
            ),
            field_name="zL_init.classification",
        ),
        registry_anchor=_require_nonempty_string(
            zL_init_observation.get(
                "registry_anchor",
                ZL_INIT_FP_EXCEPTION_REGISTRY_ANCHOR,
            ),
            field_name="zL_init.registry_anchor",
        ),
        source_anchor=_require_nonempty_string(
            zL_init_observation.get(
                "source_anchor",
                ZL_INIT_HRM_SOURCE_ANCHOR,
            ),
            field_name="zL_init.source_anchor",
        ),
        dtype=_require_nonempty_string(
            zL_init_observation.get("dtype", ""),
            field_name="zL_init.dtype",
        ),
        shape=_shape_tuple(
            zL_init_observation.get("shape", ()),
            field_name="zL_init.shape",
        ),
        persistent=bool(zL_init_observation.get("persistent", False)),
        non_claim=(
            "zL_init is a persistent BF16/FP initial-state buffer covered by "
            "the existing non_eligible_hrm_tensors FP exception; activation "
            "seam observation does not convert or solve it"
        ),
    )


def build_activation_residuals_fail_closed_receipt(
    *,
    seam_events: Sequence[Mapping[str, object]],
    zL_init_observation: Mapping[str, object],
    activations_residuals_sub2_claim: bool = False,
    real_sub2_or_remat_or_offload_mechanism_present: bool = False,
    no_hidden_bf16_authority_proven: bool = False,
    gpu_memory_receipt_present: bool = False,
    lossy_or_compression_claim: bool = False,
    ready_to_flip: bool = False,
    smallest_missing_proof: str = (
        "real activation/residual sub2 representation or lossless remat/offload "
        "mechanism plus GPU memory receipt and no-hidden-BF16 authority proof"
    ),
) -> ActivationResidualsFailClosedReceipt:
    """Build the Step 3A2 fail-closed activation/residual blocker receipt."""

    receipt = ActivationResidualsFailClosedReceipt(
        schema_version=ACTIVATION_RESIDUALS_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION,
        target_name=ACTIVATION_RESIDUALS_FAIL_CLOSED_TARGET_NAME,
        target_families=ACTIVATION_RESIDUAL_TARGET_FAMILIES,
        activations_residuals_sub2_claim=bool(activations_residuals_sub2_claim),
        real_sub2_or_remat_or_offload_mechanism_present=bool(
            real_sub2_or_remat_or_offload_mechanism_present
        ),
        no_hidden_bf16_authority_proven=bool(no_hidden_bf16_authority_proven),
        gpu_memory_receipt_present=bool(gpu_memory_receipt_present),
        lossy_or_compression_claim=bool(lossy_or_compression_claim),
        ready_to_flip=bool(ready_to_flip),
        blocked_reason=ACTIVATION_RESIDUALS_BLOCKED_REASON,
        observed_families=_summarize_activation_residual_live_tensor_families(
            seam_events
        ),
        zL_init_non_claim=_zL_init_non_claim_from_observation(zL_init_observation),
        smallest_missing_proof=_require_nonempty_string(
            smallest_missing_proof,
            field_name="smallest_missing_proof",
        ),
        non_claims=ACTIVATION_RESIDUALS_FAIL_CLOSED_NON_CLAIMS,
    )
    validate_activation_residuals_fail_closed_receipt(receipt)
    return receipt


def validate_activation_residuals_fail_closed_receipt(
    receipt: ActivationResidualsFailClosedReceipt,
) -> None:
    if receipt.schema_version != ACTIVATION_RESIDUALS_FAIL_CLOSED_RECEIPT_SCHEMA_VERSION:
        raise ValueError("activation/residual fail-closed receipt schema mismatch")
    if receipt.target_name != ACTIVATION_RESIDUALS_FAIL_CLOSED_TARGET_NAME:
        raise ValueError("activation/residual fail-closed receipt target mismatch")
    if receipt.target_families != ACTIVATION_RESIDUAL_TARGET_FAMILIES:
        raise ValueError("activation/residual target families must be exactly registered")
    observed_names = tuple(observation.family for observation in receipt.observed_families)
    if observed_names != ACTIVATION_RESIDUAL_TARGET_FAMILIES:
        raise ValueError("activation/residual observed families must match target families")
    for observation in receipt.observed_families:
        if observation.observed_count <= 0:
            raise ValueError(f"{observation.family} must have at least one live tensor")
        if observation.mechanism != "observer_returns_original_tensor":
            raise ValueError("Step 3A2 accepts only observer-returned original tensors")
        if not observation.shapes or not observation.dtypes or not observation.devices:
            raise ValueError(f"{observation.family} is missing tensor metadata")
    zL = receipt.zL_init_non_claim
    if zL.name != "zL_init":
        raise ValueError("zL_init non-claim must identify zL_init")
    if zL.classification != ZL_INIT_FP_EXCEPTION_CLASSIFICATION:
        raise ValueError("zL_init must use the existing non-eligible FP classification")
    if zL.registry_anchor != ZL_INIT_FP_EXCEPTION_REGISTRY_ANCHOR:
        raise ValueError("zL_init must cite the existing fp_exceptions.py registry entry")
    if zL.source_anchor != ZL_INIT_HRM_SOURCE_ANCHOR:
        raise ValueError("zL_init must cite the HRM persistent buffer source anchor")
    if not zL.persistent:
        raise ValueError("zL_init must be classified as a persistent buffer non-claim")
    if "non_eligible_hrm_tensors" not in zL.non_claim:
        raise ValueError("zL_init non-claim must cross-reference non_eligible_hrm_tensors")
    if receipt.lossy_or_compression_claim:
        raise ValueError("lossy/compression wording cannot satisfy activation/residual sub2")
    required_proofs = (
        receipt.real_sub2_or_remat_or_offload_mechanism_present
        and receipt.no_hidden_bf16_authority_proven
        and receipt.gpu_memory_receipt_present
    )
    if receipt.activations_residuals_sub2_claim and not (
        required_proofs and receipt.ready_to_flip
    ):
        raise ValueError(
            "activations_residuals_sub2_claim requires real "
            "sub2/remat/offload/no-hidden-BF16 proof and ready_to_flip=True"
        )
    if receipt.ready_to_flip and not required_proofs:
        raise ValueError("ready_to_flip requires all activation/residual proof gates")
    if not receipt.activations_residuals_sub2_claim and receipt.ready_to_flip:
        raise ValueError("ready_to_flip cannot be true without a sub2 claim")
    if receipt.blocked_reason != ACTIVATION_RESIDUALS_BLOCKED_REASON:
        raise ValueError("activation/residual blocked reason must be exact")
    if receipt.non_claims != ACTIVATION_RESIDUALS_FAIL_CLOSED_NON_CLAIMS:
        raise ValueError("activation/residual receipt non-claims must be exact")


def validate_activation_relief_measurement(
    receipt: Mapping[str, object],
) -> None:
    """Validate that a future relief receipt reports memory and throughput.

    A memory-only receipt is rejected because checkpointing can trade peak memory
    for wall-clock cost. A future win claim needs all first-class fields.
    """

    missing = [
        field
        for field in REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS
        if field not in receipt
    ]
    if missing:
        raise ValueError(
            "activation relief measurement missing required fields: "
            + ", ".join(missing)
        )

    for field in REQUIRED_ACTIVATION_RELIEF_MEASUREMENT_FIELDS:
        value = receipt[field]
        if not isinstance(value, (int, float)):
            raise TypeError(f"{field} must be numeric, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"{field} must be non-negative, got {value!r}")

    if receipt["wall_clock_per_step_seconds"] == 0:
        raise ValueError("wall_clock_per_step_seconds must be > 0")
    if receipt["max_safe_batch_size"] == 0:
        raise ValueError("max_safe_batch_size must be > 0")
    if receipt["effective_exposure_per_step"] == 0:
        raise ValueError("effective_exposure_per_step must be > 0")


TRAINER_BACKWARD_WIRING_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_r1_backward_wiring/v0.cpu_production_autograd"
)
TRAINER_BACKWARD_WIRING_TARGET_NAME = (
    "r1_backward_saved_tensors_production_wiring"
)
PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING = "cpu_production_autograd_wiring"
PROOF_KIND_LAUNCH_RUNTIME_VALIDATION = "launch_runtime_validation"
AUTHORIZED_R1_L_SURFACE_TUPLE = ("backward_saved_tensors_transients",)
TRAINER_BACKWARD_WIRING_NON_CLAIMS = (
    "proves default-off wiring and production-autograd instrumentation only",
    "does not authorize live backward_saved_tensors_transients row flip on current_repo_scaffold",
    "does not substitute for fixture build_backward_recompute_saved_tensor_receipt",
    "does not claim GPU peak-memory relief, learning, acquisition, throughput, or .pt mutation",
    "does not claim activations_residuals sub2",
)

LAUNCH_RUNTIME_BACKWARD_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_r1_backward_launch/v1.gpu_runtime_validation"
)
LAUNCH_RUNTIME_BACKWARD_TARGET_NAME = "r1_backward_saved_tensors_launch_runtime"
R1_CPU_BASE_COMMIT_SHA = "717f6346324388f83126763769c30b9bad53dc45"
W6_PARENT_SHA256_PINNED = (
    "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
)
LAUNCH_RUNTIME_NON_CLAIMS = (
    "pre_full_stack_diagnostic exception only; NOT ready_for_main_science",
    "does NOT prove activations_residuals / attention_kv / optimizer_credit_state / native_kernelized_hot_path sub2",
    "does NOT prove learning, acquisition, retention, or throughput",
    "does NOT mutate banked W6 parent .pt",
    "does NOT authorize full training or optimizer resume",
    "tiny GPU smoke is liveness-only; cannot mint or flip",
)
PROOF_ENV_HASH_KEYS = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "CUDA_VISIBLE_DEVICES",
    "R1L_LAUNCH_RECEIPT_JSON",
    "R1L_LAUNCH_LOG",
    "R1L_W6_PARENT_PATH",
    "TORCH_CUDA_ALLOC_CONF",
    "CUBLAS_WORKSPACE_CONFIG",
)
LAUNCH_MANIFEST_EMBEDDED_KEYS = (
    "r1_cpu_base_commit_sha",
    "launch_source_commit_sha",
    "archive_created_at_utc",
    "archive_method",
)
LAUNCH_LOG_AT_MINT_BASENAME = "launch_log_at_mint.log"
REQUIRED_PROOF_ENV_KEYS = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONPATH",
    "R1L_LAUNCH_RECEIPT_JSON",
    "R1L_LAUNCH_LOG",
    "R1L_W6_PARENT_PATH",
)


@dataclass(frozen=True)
class TrainerBackwardWiringProofReceipt:
    """R1 CPU receipt for production trainer backward-path wiring proof."""

    schema_version: str
    target_name: str
    proof_kind: str
    source_commit_sha: str
    proof_command_argv: tuple[str, ...]
    activation_relief_wiring_proof_flag: bool
    policy_mode: str
    main_path_proven: bool
    retained_side_path_proven: bool
    retained_side_in_scope: bool
    retained_side_skip_reason: str
    main_recompute_checkpoint_fired: bool
    main_baseline_saved_tensor_count: int
    main_recompute_saved_tensor_count: int
    main_internal_payload_tensor_count: int
    retained_side_recompute_checkpoint_fired: bool
    retained_side_baseline_saved_tensor_count: int
    retained_side_recompute_saved_tensor_count: int
    retained_side_internal_payload_tensor_count: int
    default_runtime_sub2_claim: bool
    activations_residuals_sub2_claim: bool
    live_readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    backward_recompute_fixture_receipt_sha256: str
    optimizer_step_called: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "target_name": self.target_name,
            "proof_kind": self.proof_kind,
            "source_commit_sha": self.source_commit_sha,
            "proof_command_argv": list(self.proof_command_argv),
            "activation_relief_wiring_proof_flag": self.activation_relief_wiring_proof_flag,
            "policy_mode": self.policy_mode,
            "main_path_proven": self.main_path_proven,
            "retained_side_path_proven": self.retained_side_path_proven,
            "retained_side_in_scope": self.retained_side_in_scope,
            "retained_side_skip_reason": self.retained_side_skip_reason,
            "main_recompute_checkpoint_fired": self.main_recompute_checkpoint_fired,
            "main_baseline_saved_tensor_count": self.main_baseline_saved_tensor_count,
            "main_recompute_saved_tensor_count": self.main_recompute_saved_tensor_count,
            "main_internal_payload_tensor_count": self.main_internal_payload_tensor_count,
            "retained_side_recompute_checkpoint_fired": (
                self.retained_side_recompute_checkpoint_fired
            ),
            "retained_side_baseline_saved_tensor_count": (
                self.retained_side_baseline_saved_tensor_count
            ),
            "retained_side_recompute_saved_tensor_count": (
                self.retained_side_recompute_saved_tensor_count
            ),
            "retained_side_internal_payload_tensor_count": (
                self.retained_side_internal_payload_tensor_count
            ),
            "default_runtime_sub2_claim": self.default_runtime_sub2_claim,
            "activations_residuals_sub2_claim": self.activations_residuals_sub2_claim,
            "live_readiness_row_flip_authorized": self.live_readiness_row_flip_authorized,
            "readiness_row_flip_authorized_surface_names": list(
                self.readiness_row_flip_authorized_surface_names
            ),
            "backward_recompute_fixture_receipt_sha256": (
                self.backward_recompute_fixture_receipt_sha256
            ),
            "optimizer_step_called": self.optimizer_step_called,
            "non_claims": list(self.non_claims),
        }


@dataclass(frozen=True)
class LaunchRuntimeBackwardValidationReceipt:
    """R1-L GPU launch/runtime validation receipt (schema v1)."""

    schema_version: str
    target_name: str
    proof_kind: str
    live_readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    r1_cpu_base_commit_sha: str
    launch_source_commit_sha: str
    ancestry_verified_at_launch_preflight: bool
    launch_runtime_validation_pass: bool
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
    main_path_proven: bool
    main_recompute_checkpoint_fired: bool
    main_baseline_saved_tensor_count: int
    main_recompute_saved_tensor_count: int
    main_internal_payload_tensor_count: int
    main_saved_tensor_payload_bytes_baseline: int
    main_saved_tensor_payload_bytes_recompute: int
    main_saved_tensor_payload_bytes_delta: int
    retained_side_in_scope: bool
    retained_side_path_proven: bool
    retained_side_recompute_checkpoint_fired: bool
    retained_side_baseline_saved_tensor_count: int
    retained_side_recompute_saved_tensor_count: int
    retained_side_internal_payload_tensor_count: int
    retained_saved_tensor_payload_bytes_delta: int
    paired_run_count: int
    cuda_peak_allocated_bytes_baseline_median: int
    cuda_peak_allocated_bytes_recompute_median: int
    cuda_peak_allocated_bytes_delta_median: int
    cuda_peak_reduction_threshold_bytes: int
    cuda_peak_reduction_threshold_met: bool
    cuda_peak_reserved_bytes_delta_median: int
    loss_finite_main: bool
    loss_finite_retained: bool
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
            "live_readiness_row_flip_authorized": self.live_readiness_row_flip_authorized,
            "readiness_row_flip_authorized_surface_names": list(
                self.readiness_row_flip_authorized_surface_names
            ),
            "r1_cpu_base_commit_sha": self.r1_cpu_base_commit_sha,
            "launch_source_commit_sha": self.launch_source_commit_sha,
            "ancestry_verified_at_launch_preflight": (
                self.ancestry_verified_at_launch_preflight
            ),
            "launch_runtime_validation_pass": self.launch_runtime_validation_pass,
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
            "main_path_proven": self.main_path_proven,
            "main_recompute_checkpoint_fired": self.main_recompute_checkpoint_fired,
            "main_baseline_saved_tensor_count": self.main_baseline_saved_tensor_count,
            "main_recompute_saved_tensor_count": self.main_recompute_saved_tensor_count,
            "main_internal_payload_tensor_count": self.main_internal_payload_tensor_count,
            "main_saved_tensor_payload_bytes_baseline": (
                self.main_saved_tensor_payload_bytes_baseline
            ),
            "main_saved_tensor_payload_bytes_recompute": (
                self.main_saved_tensor_payload_bytes_recompute
            ),
            "main_saved_tensor_payload_bytes_delta": (
                self.main_saved_tensor_payload_bytes_delta
            ),
            "retained_side_in_scope": self.retained_side_in_scope,
            "retained_side_path_proven": self.retained_side_path_proven,
            "retained_side_recompute_checkpoint_fired": (
                self.retained_side_recompute_checkpoint_fired
            ),
            "retained_side_baseline_saved_tensor_count": (
                self.retained_side_baseline_saved_tensor_count
            ),
            "retained_side_recompute_saved_tensor_count": (
                self.retained_side_recompute_saved_tensor_count
            ),
            "retained_side_internal_payload_tensor_count": (
                self.retained_side_internal_payload_tensor_count
            ),
            "retained_saved_tensor_payload_bytes_delta": (
                self.retained_saved_tensor_payload_bytes_delta
            ),
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
            "cuda_peak_reduction_threshold_bytes": (
                self.cuda_peak_reduction_threshold_bytes
            ),
            "cuda_peak_reduction_threshold_met": self.cuda_peak_reduction_threshold_met,
            "cuda_peak_reserved_bytes_delta_median": (
                self.cuda_peak_reserved_bytes_delta_median
            ),
            "loss_finite_main": self.loss_finite_main,
            "loss_finite_retained": self.loss_finite_retained,
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


def _canonical_json_dumps(obj: object) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _canonical_json_sha256(obj: object) -> str:
    return hashlib.sha256(_canonical_json_dumps(obj).encode("utf-8")).hexdigest()


def compute_proof_env_hash_sha256(env_embedded: Mapping[str, str]) -> str:
    payload = {key: str(env_embedded.get(key, "")) for key in PROOF_ENV_HASH_KEYS}
    return _canonical_json_sha256(payload)


def compute_launch_manifest_sha256(manifest_embedded: Mapping[str, str]) -> str:
    return _canonical_json_sha256(dict(manifest_embedded))


def compute_gpu_identity_sha256(
    *,
    gpu_name: str,
    gpu_uuid: str,
    driver_version: str,
    cuda_version: str,
    torch_version: str,
) -> str:
    return _canonical_json_sha256(
        {
            "gpu_name": gpu_name,
            "gpu_uuid": gpu_uuid,
            "driver_version": driver_version,
            "cuda_version": cuda_version,
            "torch_version": torch_version,
        }
    )


def compute_canonical_launch_artifact_sha256(receipt_dict: Mapping[str, object]) -> str:
    payload = dict(receipt_dict)
    payload["canonical_launch_artifact_sha256"] = None
    return _canonical_json_sha256(payload)


def _embedded_mapping(
    value: object,
    *,
    field_name: str,
    required_keys: Sequence[str] | None = None,
) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{field_name} must be a mapping")
    embedded = {str(key): str(item) for key, item in value.items()}
    if required_keys is not None:
        missing = [key for key in required_keys if key not in embedded]
        if missing:
            raise ValueError(f"{field_name} missing required keys: {', '.join(missing)}")
    return embedded


def _string_tuple(value: object, *, field_name: str) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field_name} must be a sequence of strings")
    return tuple(str(item) for item in value)


def launch_runtime_backward_receipt_from_dict(
    payload: Mapping[str, object],
) -> LaunchRuntimeBackwardValidationReceipt:
    return LaunchRuntimeBackwardValidationReceipt(
        schema_version=_require_nonempty_string(
            payload.get("schema_version"),
            field_name="schema_version",
        ),
        target_name=_require_nonempty_string(
            payload.get("target_name"),
            field_name="target_name",
        ),
        proof_kind=_require_nonempty_string(payload.get("proof_kind"), field_name="proof_kind"),
        live_readiness_row_flip_authorized=bool(
            payload.get("live_readiness_row_flip_authorized")
        ),
        readiness_row_flip_authorized_surface_names=_string_tuple(
            payload.get("readiness_row_flip_authorized_surface_names"),
            field_name="readiness_row_flip_authorized_surface_names",
        ),
        r1_cpu_base_commit_sha=_require_nonempty_string(
            payload.get("r1_cpu_base_commit_sha"),
            field_name="r1_cpu_base_commit_sha",
        ),
        launch_source_commit_sha=_require_nonempty_string(
            payload.get("launch_source_commit_sha"),
            field_name="launch_source_commit_sha",
        ),
        ancestry_verified_at_launch_preflight=bool(
            payload.get("ancestry_verified_at_launch_preflight")
        ),
        launch_runtime_validation_pass=bool(payload.get("launch_runtime_validation_pass")),
        launch_manifest_sha256=_require_nonempty_string(
            payload.get("launch_manifest_sha256"),
            field_name="launch_manifest_sha256",
        ),
        launch_manifest_embedded=_embedded_mapping(
            payload.get("launch_manifest_embedded"),
            field_name="launch_manifest_embedded",
            required_keys=LAUNCH_MANIFEST_EMBEDDED_KEYS,
        ),
        proof_env_embedded=_embedded_mapping(
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
        retained_support_digest_sha256=str(payload.get("retained_support_digest_sha256", "")),
        main_path_proven=bool(payload.get("main_path_proven")),
        main_recompute_checkpoint_fired=bool(payload.get("main_recompute_checkpoint_fired")),
        main_baseline_saved_tensor_count=_require_int(
            payload.get("main_baseline_saved_tensor_count"),
            field_name="main_baseline_saved_tensor_count",
        ),
        main_recompute_saved_tensor_count=_require_int(
            payload.get("main_recompute_saved_tensor_count"),
            field_name="main_recompute_saved_tensor_count",
        ),
        main_internal_payload_tensor_count=_require_int(
            payload.get("main_internal_payload_tensor_count"),
            field_name="main_internal_payload_tensor_count",
        ),
        main_saved_tensor_payload_bytes_baseline=_require_int(
            payload.get("main_saved_tensor_payload_bytes_baseline"),
            field_name="main_saved_tensor_payload_bytes_baseline",
        ),
        main_saved_tensor_payload_bytes_recompute=_require_int(
            payload.get("main_saved_tensor_payload_bytes_recompute"),
            field_name="main_saved_tensor_payload_bytes_recompute",
        ),
        main_saved_tensor_payload_bytes_delta=_require_int(
            payload.get("main_saved_tensor_payload_bytes_delta"),
            field_name="main_saved_tensor_payload_bytes_delta",
        ),
        retained_side_in_scope=bool(payload.get("retained_side_in_scope")),
        retained_side_path_proven=bool(payload.get("retained_side_path_proven")),
        retained_side_recompute_checkpoint_fired=bool(
            payload.get("retained_side_recompute_checkpoint_fired")
        ),
        retained_side_baseline_saved_tensor_count=_require_int(
            payload.get("retained_side_baseline_saved_tensor_count"),
            field_name="retained_side_baseline_saved_tensor_count",
        ),
        retained_side_recompute_saved_tensor_count=_require_int(
            payload.get("retained_side_recompute_saved_tensor_count"),
            field_name="retained_side_recompute_saved_tensor_count",
        ),
        retained_side_internal_payload_tensor_count=_require_int(
            payload.get("retained_side_internal_payload_tensor_count"),
            field_name="retained_side_internal_payload_tensor_count",
        ),
        retained_saved_tensor_payload_bytes_delta=_require_int(
            payload.get("retained_saved_tensor_payload_bytes_delta"),
            field_name="retained_saved_tensor_payload_bytes_delta",
        ),
        paired_run_count=_require_int(payload.get("paired_run_count"), field_name="paired_run_count"),
        cuda_peak_allocated_bytes_baseline_median=_require_int(
            payload.get("cuda_peak_allocated_bytes_baseline_median"),
            field_name="cuda_peak_allocated_bytes_baseline_median",
        ),
        cuda_peak_allocated_bytes_recompute_median=_require_int(
            payload.get("cuda_peak_allocated_bytes_recompute_median"),
            field_name="cuda_peak_allocated_bytes_recompute_median",
        ),
        cuda_peak_allocated_bytes_delta_median=_require_int(
            payload.get("cuda_peak_allocated_bytes_delta_median"),
            field_name="cuda_peak_allocated_bytes_delta_median",
        ),
        cuda_peak_reduction_threshold_bytes=_require_int(
            payload.get("cuda_peak_reduction_threshold_bytes"),
            field_name="cuda_peak_reduction_threshold_bytes",
        ),
        cuda_peak_reduction_threshold_met=bool(payload.get("cuda_peak_reduction_threshold_met")),
        cuda_peak_reserved_bytes_delta_median=_require_int(
            payload.get("cuda_peak_reserved_bytes_delta_median"),
            field_name="cuda_peak_reserved_bytes_delta_median",
        ),
        loss_finite_main=bool(payload.get("loss_finite_main")),
        loss_finite_retained=bool(payload.get("loss_finite_retained")),
        applier_base_surface_count_sub2=_require_int(
            payload.get("applier_base_surface_count_sub2"),
            field_name="applier_base_surface_count_sub2",
        ),
        applier_result_sub2_surface_count=_require_int(
            payload.get("applier_result_sub2_surface_count"),
            field_name="applier_result_sub2_surface_count",
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


def verify_launch_ancestry_preflight(
    *,
    repo_root: str | Path,
    launch_source_commit_sha: str | None = None,
    r1_cpu_base_commit_sha: str = R1_CPU_BASE_COMMIT_SHA,
) -> str:
    repo = Path(repo_root)
    source = launch_source_commit_sha
    if source is None:
        source = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            text=True,
        ).strip()
    result = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "merge-base",
            "--is-ancestor",
            r1_cpu_base_commit_sha,
            source,
        ],
        check=False,
    )
    if result.returncode != 0:
        raise ValueError("launch source is not a descendant of R1 CPU base commit")
    return source


class R1lLaunchProofAbort(RuntimeError):
    """Abort R1-L launch proof without minting a receipt."""


@dataclass(frozen=True)
class R1lLaunchProofMeasurements:
    main_baseline_saved_tensor_count: int
    main_recompute_saved_tensor_count: int
    main_saved_tensor_payload_bytes_baseline: int
    main_saved_tensor_payload_bytes_recompute: int
    main_internal_payload_tensor_count: int
    main_recompute_checkpoint_fired: bool
    retained_side_in_scope: bool
    retained_side_baseline_saved_tensor_count: int
    retained_side_recompute_saved_tensor_count: int
    retained_side_internal_payload_tensor_count: int
    retained_saved_tensor_payload_bytes_delta: int
    retained_side_recompute_checkpoint_fired: bool
    loss_finite_main: bool
    loss_finite_retained: bool
    paired_run_count: int
    cuda_peak_allocated_bytes_baseline_median: int
    cuda_peak_allocated_bytes_recompute_median: int
    cuda_peak_reserved_bytes_delta_median: int


def _median_int(values: Sequence[int]) -> int:
    if not values:
        raise ValueError("median requires at least one value")
    ordered = sorted(int(value) for value in values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) // 2


def _saved_tensor_payload_bytes(events: Sequence[Mapping[str, object]]) -> int:
    import torch

    total = 0
    for event in events:
        shape = tuple(event.get("shape", ()))
        dtype_name = str(event.get("dtype", "torch.float32")).replace("torch.", "")
        dtype = getattr(torch, dtype_name, torch.float32)
        nbytes = int(dtype.itemsize)
        for dim in shape:
            nbytes *= int(dim)
        total += nbytes
    return total


def _proof_env_embedded_from_os() -> dict[str, str]:
    return {key: str(os.environ.get(key, "")) for key in PROOF_ENV_HASH_KEYS}


def _read_launch_manifest_embedded() -> dict[str, str]:
    manifest_path = os.environ.get("R1L_LAUNCH_MANIFEST_JSON", "").strip()
    if manifest_path:
        payload = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise R1lLaunchProofAbort("launch manifest JSON must decode to an object")
        return {str(key): str(value) for key, value in payload.items()}
    launch_source = os.environ.get("R1L_LAUNCH_SOURCE_COMMIT_SHA", "").strip()
    if not launch_source:
        raise R1lLaunchProofAbort(
            "R1L_LAUNCH_MANIFEST_JSON or R1L_LAUNCH_SOURCE_COMMIT_SHA is required"
        )
    return {
        "r1_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": launch_source,
        "archive_created_at_utc": os.environ.get("R1L_ARCHIVE_CREATED_AT_UTC", ""),
        "archive_method": os.environ.get("R1L_ARCHIVE_METHOD", "git_archive_HEAD"),
    }


def _read_proof_env_embedded() -> dict[str, str]:
    env_path = os.environ.get("R1L_LAUNCH_ENV_JSON", "").strip()
    if env_path:
        payload = json.loads(Path(env_path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise R1lLaunchProofAbort("launch env JSON must decode to an object")
        embedded = {str(key): str(value) for key, value in payload.items()}
    else:
        embedded = _proof_env_embedded_from_os()
    missing = [
        key
        for key in REQUIRED_PROOF_ENV_KEYS
        if not str(embedded.get(key, "")).strip()
    ]
    if missing:
        raise R1lLaunchProofAbort(
            "launch proof env missing required keys: " + ", ".join(missing)
        )
    return embedded


def launch_log_at_mint_snapshot_path(*, receipt_json_path: str | None = None) -> Path:
    receipt_json = (
        receipt_json_path or os.environ.get("R1L_LAUNCH_RECEIPT_JSON", "")
    ).strip()
    if not receipt_json:
        raise R1lLaunchProofAbort(
            "R1L_LAUNCH_RECEIPT_JSON is required for launch log snapshot"
        )
    return Path(receipt_json).resolve().parent / LAUNCH_LOG_AT_MINT_BASENAME


def _snapshot_launch_log_at_mint() -> tuple[Path, str]:
    log_path = Path(os.environ.get("R1L_LAUNCH_LOG", "").strip())
    if not log_path.is_file():
        raise R1lLaunchProofAbort("R1L_LAUNCH_LOG is required")
    log_bytes = log_path.read_bytes()
    if not log_bytes:
        raise R1lLaunchProofAbort("R1L_LAUNCH_LOG must be non-empty")
    snapshot_path = launch_log_at_mint_snapshot_path()
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot_path.write_bytes(log_bytes)
    return snapshot_path, hashlib.sha256(log_bytes).hexdigest()


def _validate_launch_measurements_for_mint(
    measurements: R1lLaunchProofMeasurements,
) -> None:
    if not measurements.main_recompute_checkpoint_fired:
        raise R1lLaunchProofAbort("main recompute checkpoint did not fire")
    if measurements.main_internal_payload_tensor_count != 0:
        raise R1lLaunchProofAbort("main internal payload tensors observed")
    if (
        measurements.main_baseline_saved_tensor_count
        <= measurements.main_recompute_saved_tensor_count
    ):
        raise R1lLaunchProofAbort("main saved tensor counts invalid")
    if (
        measurements.main_saved_tensor_payload_bytes_baseline
        <= measurements.main_saved_tensor_payload_bytes_recompute
    ):
        raise R1lLaunchProofAbort("main saved tensor payload bytes delta invalid")
    if not measurements.loss_finite_main:
        raise R1lLaunchProofAbort("main loss non-finite")
    if measurements.retained_side_in_scope:
        if not measurements.retained_side_recompute_checkpoint_fired:
            raise R1lLaunchProofAbort("retained recompute checkpoint did not fire")
        if measurements.retained_side_internal_payload_tensor_count != 0:
            raise R1lLaunchProofAbort("retained internal payload tensors observed")
        if (
            measurements.retained_side_baseline_saved_tensor_count
            <= measurements.retained_side_recompute_saved_tensor_count
        ):
            raise R1lLaunchProofAbort("retained saved tensor counts invalid")
        if measurements.retained_saved_tensor_payload_bytes_delta <= 0:
            raise R1lLaunchProofAbort("retained payload bytes delta invalid")
        if not measurements.loss_finite_retained:
            raise R1lLaunchProofAbort("retained loss non-finite")
    if measurements.paired_run_count < 3:
        raise R1lLaunchProofAbort("paired_run_count must be >= 3")
    cuda_delta = (
        measurements.cuda_peak_allocated_bytes_baseline_median
        - measurements.cuda_peak_allocated_bytes_recompute_median
    )
    threshold = max(
        8 * 1024 * 1024,
        int(0.005 * measurements.cuda_peak_allocated_bytes_baseline_median),
    )
    if cuda_delta < threshold:
        raise R1lLaunchProofAbort(
            f"cuda peak reduction below threshold ({cuda_delta} < {threshold})"
        )


def _execute_r1l_gpu_launch_measurement(
    *,
    model: Any,
    parent_model: Any | None,
    loader: Any,
    device: Any,
    hidden_size: int,
    cfg: Any,
    active_supports: Sequence[Mapping[str, Any]],
    parent_consistency_temp: float,
    epochs: int,
    gather_retained_parent_response_logits: Callable[..., Any],
    parent_consistency_kl: Callable[..., Any],
    parent_consistency_kl_response_positions: Callable[..., Any],
) -> R1lLaunchProofMeasurements:
    import torch

    proof_batch = next(iter(loader))
    proof_total_steps = max(1, epochs * len(loader))
    extras_base = model.compute_train_extra_args(1, proof_total_steps)
    extras_policy = {
        **extras_base,
        "activation_relief_policy": MODE_LOSSLESS_RECOMPUTE,
    }

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

    def _collect_saved_tensor_events(run_fn):
        events: list[dict[str, object]] = []

        def pack_hook(tensor: torch.Tensor):
            events.append(
                {
                    "shape": tuple(tensor.shape),
                    "dtype": str(tensor.dtype),
                }
            )
            return tensor

        with torch.autograd.graph.saved_tensors_hooks(
            pack_hook,
            lambda tensor: tensor,
        ):
            run_fn()
        return events

    child_batch = _proof_child_batch(proof_batch)

    def _run_main_backward(extras):
        model.zero_grad(set_to_none=True)
        _new_carry, loss_main, _metrics = model(None, child_batch, **extras)
        if not torch.isfinite(loss_main):
            raise R1lLaunchProofAbort(
                f"main loss non-finite: {loss_main.item()}"
            )
        loss_main.backward()

    main_baseline_events = _collect_saved_tensor_events(
        lambda: _run_main_backward(extras_base)
    )
    main_recompute_events = _collect_saved_tensor_events(
        lambda: _run_main_backward(extras_policy)
    )
    main_path_proof = production_saved_tensor_path_proof_from_events(
        baseline_events=main_baseline_events,
        recompute_events=main_recompute_events,
        hidden_size=hidden_size,
        H_cycles=cfg.H_cycles,
        L_cycles=cfg.L_cycles,
        bp_steps=int(extras_base["bp_steps"]),
    )
    main_checkpoint_fired = recompute_checkpoint_fired(
        H_cycles=cfg.H_cycles,
        L_cycles=cfg.L_cycles,
        bp_steps=int(extras_base["bp_steps"]),
    )

    retained_side_in_scope = bool(active_supports)
    retained_baseline_count = 0
    retained_recompute_count = 0
    retained_internal_payload = 0
    retained_payload_delta = 0
    retained_checkpoint_fired = False
    loss_finite_retained = not retained_side_in_scope

    if retained_side_in_scope:
        _sup = active_supports[0]
        _idx = _sup["sampler"].next_indices()
        _picked = [_sup["cache"][i] for i in _idx]
        s_inputs = torch.stack([p["inputs"] for p in _picked], 0).to(device)
        s_labels = torch.stack([p["labels"] for p in _picked], 0).to(device)
        s_sep = torch.stack([p["sep_position"] for p in _picked], 0).to(device)
        sB, sL = s_inputs.shape
        s_pos = torch.arange(
            sL, dtype=torch.long, device=device
        ).unsqueeze(0).expand(sB, -1)
        s_batch = {
            "inputs": s_inputs,
            "labels": s_labels,
            "sep_positions": s_sep,
            "position_ids": s_pos,
        }

        def _run_retained_backward(extras):
            model.zero_grad(set_to_none=True)
            _sc, _sloss, s_metrics = model(
                None,
                s_batch,
                return_logits=True,
                **extras,
            )
            if _sup.get("parent_response_logits_by_bp") is not None:
                s_parent_response_logits = gather_retained_parent_response_logits(
                    _sup,
                    _idx,
                    int(extras["bp_steps"]),
                    device,
                )
                s_kl = parent_consistency_kl_response_positions(
                    s_metrics["logits"],
                    s_parent_response_logits,
                    s_labels,
                    temp=parent_consistency_temp,
                )
            else:
                if parent_model is None:
                    raise R1lLaunchProofAbort(
                        "retained support requires frozen parent model"
                    )
                with torch.no_grad():
                    _, s_parent_logits = parent_model(
                        None,
                        {
                            "inputs": s_inputs,
                            "sep_positions": s_sep,
                            "position_ids": s_pos,
                        },
                        **extras,
                    )
                s_is_prior = torch.ones(sB, dtype=torch.bool, device=device)
                s_kl = parent_consistency_kl(
                    s_metrics["logits"],
                    s_parent_logits,
                    s_labels,
                    s_is_prior,
                    temp=parent_consistency_temp,
                )
            s_loss = _sup["weight"] * s_kl
            if not torch.isfinite(s_loss):
                raise R1lLaunchProofAbort(
                    f"retained loss non-finite: {s_loss.item()}"
                )
            s_loss.backward()

        retained_baseline_events = _collect_saved_tensor_events(
            lambda: _run_retained_backward(extras_base)
        )
        retained_recompute_events = _collect_saved_tensor_events(
            lambda: _run_retained_backward(extras_policy)
        )
        retained_path_proof = production_saved_tensor_path_proof_from_events(
            baseline_events=retained_baseline_events,
            recompute_events=retained_recompute_events,
            hidden_size=hidden_size,
            H_cycles=cfg.H_cycles,
            L_cycles=cfg.L_cycles,
            bp_steps=int(extras_base["bp_steps"]),
        )
        retained_baseline_count = int(
            retained_path_proof["baseline_saved_tensor_count"]
        )
        retained_recompute_count = int(
            retained_path_proof["recompute_saved_tensor_count"]
        )
        retained_internal_payload = int(
            retained_path_proof["internal_payload_tensor_count"]
        )
        retained_payload_delta = (
            _saved_tensor_payload_bytes(retained_baseline_events)
            - _saved_tensor_payload_bytes(retained_recompute_events)
        )
        retained_checkpoint_fired = recompute_checkpoint_fired(
            H_cycles=cfg.H_cycles,
            L_cycles=cfg.L_cycles,
            bp_steps=int(extras_base["bp_steps"]),
        )
        loss_finite_retained = True

    baseline_peaks: list[int] = []
    recompute_peaks: list[int] = []
    reserved_deltas: list[int] = []
    for _ in range(3):
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        reserved_before = int(torch.cuda.max_memory_reserved(device))
        _run_main_backward(extras_base)
        torch.cuda.synchronize(device)
        baseline_peaks.append(int(torch.cuda.max_memory_allocated(device)))
        reserved_after_base = int(torch.cuda.max_memory_reserved(device))

        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
        _run_main_backward(extras_policy)
        torch.cuda.synchronize(device)
        recompute_peaks.append(int(torch.cuda.max_memory_allocated(device)))
        reserved_after_recompute = int(torch.cuda.max_memory_reserved(device))
        reserved_deltas.append(reserved_after_recompute - reserved_after_base)

    return R1lLaunchProofMeasurements(
        main_baseline_saved_tensor_count=int(
            main_path_proof["baseline_saved_tensor_count"]
        ),
        main_recompute_saved_tensor_count=int(
            main_path_proof["recompute_saved_tensor_count"]
        ),
        main_saved_tensor_payload_bytes_baseline=_saved_tensor_payload_bytes(
            main_baseline_events
        ),
        main_saved_tensor_payload_bytes_recompute=_saved_tensor_payload_bytes(
            main_recompute_events
        ),
        main_internal_payload_tensor_count=int(
            main_path_proof["internal_payload_tensor_count"]
        ),
        main_recompute_checkpoint_fired=main_checkpoint_fired,
        retained_side_in_scope=retained_side_in_scope,
        retained_side_baseline_saved_tensor_count=retained_baseline_count,
        retained_side_recompute_saved_tensor_count=retained_recompute_count,
        retained_side_internal_payload_tensor_count=retained_internal_payload,
        retained_saved_tensor_payload_bytes_delta=retained_payload_delta,
        retained_side_recompute_checkpoint_fired=retained_checkpoint_fired,
        loss_finite_main=True,
        loss_finite_retained=loss_finite_retained,
        paired_run_count=3,
        cuda_peak_allocated_bytes_baseline_median=_median_int(baseline_peaks),
        cuda_peak_allocated_bytes_recompute_median=_median_int(recompute_peaks),
        cuda_peak_reserved_bytes_delta_median=_median_int(reserved_deltas),
    )


def run_r1l_gpu_launch_proof(
    *,
    model: Any,
    parent_model: Any | None,
    loader: Any,
    device: Any,
    hidden_size: int,
    cfg: Any,
    active_supports: Sequence[Mapping[str, Any]],
    parent_consistency_temp: float,
    epochs: int,
    proof_command_argv: Sequence[str],
    w6_parent_path: str,
    gather_retained_parent_response_logits: Callable[..., Any],
    parent_consistency_kl: Callable[..., Any],
    parent_consistency_kl_response_positions: Callable[..., Any],
    cuda_is_available_fn: Callable[[], bool] | None = None,
    measurement_runner: Callable[[], R1lLaunchProofMeasurements] | None = None,
) -> LaunchRuntimeBackwardValidationReceipt:
    import torch

    _cuda_available = cuda_is_available_fn or torch.cuda.is_available
    if not _cuda_available():
        raise RuntimeError("R1-L launch proof requires CUDA")
    if os.environ.get("R1L_ANCESTRY_VERIFIED") != "1":
        raise R1lLaunchProofAbort("R1L_ANCESTRY_VERIFIED must be 1")

    w6_path = Path(w6_parent_path)
    if not w6_path.is_file():
        raise R1lLaunchProofAbort(f"w6 parent path not found: {w6_parent_path}")
    w6_before = hashlib.sha256(w6_path.read_bytes()).hexdigest()
    if w6_before != W6_PARENT_SHA256_PINNED:
        raise R1lLaunchProofAbort(
            f"w6 parent sha256 mismatch (got {w6_before}, expected pinned hash)"
        )

    manifest_embedded = _read_launch_manifest_embedded()
    proof_env_embedded = _read_proof_env_embedded()
    launch_source_commit_sha = manifest_embedded["launch_source_commit_sha"]
    clean_run_dir_sha256 = os.environ.get("R1L_CLEAN_RUN_DIR_SHA256", "").strip()
    if not clean_run_dir_sha256:
        raise R1lLaunchProofAbort("R1L_CLEAN_RUN_DIR_SHA256 is required")

    if measurement_runner is None:
        measurements = _execute_r1l_gpu_launch_measurement(
            model=model,
            parent_model=parent_model,
            loader=loader,
            device=device,
            hidden_size=hidden_size,
            cfg=cfg,
            active_supports=active_supports,
            parent_consistency_temp=parent_consistency_temp,
            epochs=epochs,
            gather_retained_parent_response_logits=gather_retained_parent_response_logits,
            parent_consistency_kl=parent_consistency_kl,
            parent_consistency_kl_response_positions=parent_consistency_kl_response_positions,
        )
    else:
        measurements = measurement_runner()

    _validate_launch_measurements_for_mint(measurements)

    w6_after = hashlib.sha256(w6_path.read_bytes()).hexdigest()
    if w6_after != w6_before:
        raise R1lLaunchProofAbort("w6 parent mutated during launch proof")

    _, log_artifact_sha256 = _snapshot_launch_log_at_mint()

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
    retained_support_digest_sha256 = _canonical_json_sha256(
        [
            {
                "name": support["name"],
                "weight": support["weight"],
                "hash": support["hash"],
                "count": support["count"],
            }
            for support in active_supports
        ]
    )

    if measurement_runner is not None:
        gpu_name = os.environ.get("R1L_GPU_NAME", "synthetic-gpu")
        gpu_uuid = os.environ.get("R1L_GPU_UUID", "gpu-uuid-test")
        driver_version = os.environ.get("R1L_GPU_DRIVER_VERSION", "550.00")
        cuda_version = os.environ.get("R1L_CUDA_VERSION", "12.4")
        torch_version = str(torch.__version__)
    else:
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        gpu_uuid = os.environ.get("R1L_GPU_UUID", "").strip() or (
            f"cuda:{torch.cuda.current_device()}:{getattr(props, 'name', gpu_name)}"
        )
        driver_version = str(getattr(torch.version, "cuda", "") or "")
        cuda_version = driver_version
        torch_version = str(torch.__version__)

    return build_launch_runtime_backward_validation_receipt(
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
        retained_support_digest_sha256=retained_support_digest_sha256,
        main_baseline_saved_tensor_count=measurements.main_baseline_saved_tensor_count,
        main_recompute_saved_tensor_count=measurements.main_recompute_saved_tensor_count,
        main_saved_tensor_payload_bytes_baseline=(
            measurements.main_saved_tensor_payload_bytes_baseline
        ),
        main_saved_tensor_payload_bytes_recompute=(
            measurements.main_saved_tensor_payload_bytes_recompute
        ),
        retained_side_in_scope=measurements.retained_side_in_scope,
        retained_side_baseline_saved_tensor_count=(
            measurements.retained_side_baseline_saved_tensor_count
        ),
        retained_side_recompute_saved_tensor_count=(
            measurements.retained_side_recompute_saved_tensor_count
        ),
        retained_saved_tensor_payload_bytes_delta=(
            measurements.retained_saved_tensor_payload_bytes_delta
        ),
        paired_run_count=measurements.paired_run_count,
        cuda_peak_allocated_bytes_baseline_median=(
            measurements.cuda_peak_allocated_bytes_baseline_median
        ),
        cuda_peak_allocated_bytes_recompute_median=(
            measurements.cuda_peak_allocated_bytes_recompute_median
        ),
        cuda_peak_reserved_bytes_delta_median=(
            measurements.cuda_peak_reserved_bytes_delta_median
        ),
        log_artifact_sha256=log_artifact_sha256,
        ancestry_verified_at_launch_preflight=True,
    )


def build_launch_runtime_backward_validation_receipt(
    *,
    launch_source_commit_sha: str,
    launch_manifest_embedded: Mapping[str, str],
    proof_env_embedded: Mapping[str, str],
    proof_command_argv: Sequence[str],
    clean_run_dir_sha256: str,
    w6_parent_path: str,
    w6_parent_sha256: str = W6_PARENT_SHA256_PINNED,
    gpu_name: str,
    gpu_uuid: str,
    driver_version: str,
    cuda_version: str,
    torch_version: str,
    model_config_digest_sha256: str,
    proof_batch_digest_sha256: str,
    retained_support_digest_sha256: str,
    main_baseline_saved_tensor_count: int,
    main_recompute_saved_tensor_count: int,
    main_saved_tensor_payload_bytes_baseline: int,
    main_saved_tensor_payload_bytes_recompute: int,
    retained_side_in_scope: bool,
    retained_side_baseline_saved_tensor_count: int = 0,
    retained_side_recompute_saved_tensor_count: int = 0,
    retained_saved_tensor_payload_bytes_delta: int = 0,
    paired_run_count: int,
    cuda_peak_allocated_bytes_baseline_median: int,
    cuda_peak_allocated_bytes_recompute_median: int,
    cuda_peak_reserved_bytes_delta_median: int,
    log_artifact_sha256: str,
    applier_base_surface_count_sub2: int = 3,
    applier_result_sub2_surface_count: int = 4,
    ancestry_verified_at_launch_preflight: bool = True,
    r1_cpu_base_commit_sha: str = R1_CPU_BASE_COMMIT_SHA,
) -> LaunchRuntimeBackwardValidationReceipt:
    manifest = dict(launch_manifest_embedded)
    env = dict(proof_env_embedded)
    main_delta = main_saved_tensor_payload_bytes_baseline - main_saved_tensor_payload_bytes_recompute
    cuda_delta = (
        cuda_peak_allocated_bytes_baseline_median
        - cuda_peak_allocated_bytes_recompute_median
    )
    threshold = max(8 * 1024 * 1024, int(0.005 * cuda_peak_allocated_bytes_baseline_median))
    gpu_identity_sha256 = compute_gpu_identity_sha256(
        gpu_name=gpu_name,
        gpu_uuid=gpu_uuid,
        driver_version=driver_version,
        cuda_version=cuda_version,
        torch_version=torch_version,
    )
    receipt_without_hash = LaunchRuntimeBackwardValidationReceipt(
        schema_version=LAUNCH_RUNTIME_BACKWARD_RECEIPT_SCHEMA_VERSION,
        target_name=LAUNCH_RUNTIME_BACKWARD_TARGET_NAME,
        proof_kind=PROOF_KIND_LAUNCH_RUNTIME_VALIDATION,
        live_readiness_row_flip_authorized=True,
        readiness_row_flip_authorized_surface_names=AUTHORIZED_R1_L_SURFACE_TUPLE,
        r1_cpu_base_commit_sha=r1_cpu_base_commit_sha,
        launch_source_commit_sha=launch_source_commit_sha,
        ancestry_verified_at_launch_preflight=ancestry_verified_at_launch_preflight,
        launch_runtime_validation_pass=True,
        launch_manifest_sha256=compute_launch_manifest_sha256(manifest),
        launch_manifest_embedded=manifest,
        proof_env_embedded=env,
        proof_command_argv=tuple(str(arg) for arg in proof_command_argv),
        proof_env_hash_sha256=compute_proof_env_hash_sha256(env),
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
        main_path_proven=True,
        main_recompute_checkpoint_fired=True,
        main_baseline_saved_tensor_count=main_baseline_saved_tensor_count,
        main_recompute_saved_tensor_count=main_recompute_saved_tensor_count,
        main_internal_payload_tensor_count=0,
        main_saved_tensor_payload_bytes_baseline=main_saved_tensor_payload_bytes_baseline,
        main_saved_tensor_payload_bytes_recompute=main_saved_tensor_payload_bytes_recompute,
        main_saved_tensor_payload_bytes_delta=main_delta,
        retained_side_in_scope=retained_side_in_scope,
        retained_side_path_proven=retained_side_in_scope,
        retained_side_recompute_checkpoint_fired=retained_side_in_scope,
        retained_side_baseline_saved_tensor_count=retained_side_baseline_saved_tensor_count,
        retained_side_recompute_saved_tensor_count=retained_side_recompute_saved_tensor_count,
        retained_side_internal_payload_tensor_count=0,
        retained_saved_tensor_payload_bytes_delta=retained_saved_tensor_payload_bytes_delta,
        paired_run_count=paired_run_count,
        cuda_peak_allocated_bytes_baseline_median=cuda_peak_allocated_bytes_baseline_median,
        cuda_peak_allocated_bytes_recompute_median=cuda_peak_allocated_bytes_recompute_median,
        cuda_peak_allocated_bytes_delta_median=cuda_delta,
        cuda_peak_reduction_threshold_bytes=threshold,
        cuda_peak_reduction_threshold_met=cuda_delta >= threshold,
        cuda_peak_reserved_bytes_delta_median=cuda_peak_reserved_bytes_delta_median,
        loss_finite_main=True,
        loss_finite_retained=retained_side_in_scope,
        applier_base_surface_count_sub2=applier_base_surface_count_sub2,
        applier_result_sub2_surface_count=applier_result_sub2_surface_count,
        applier_result_ready_for_main_science=False,
        applier_result_ready_for_pre_full_stack_diagnostic=True,
        applier_flipped_surface_ids=AUTHORIZED_R1_L_SURFACE_TUPLE,
        log_artifact_sha256=log_artifact_sha256,
        canonical_launch_artifact_sha256="",
        non_claims=LAUNCH_RUNTIME_NON_CLAIMS,
    )
    canonical_hash = compute_canonical_launch_artifact_sha256(receipt_without_hash.to_dict())
    receipt = replace(
        receipt_without_hash,
        canonical_launch_artifact_sha256=canonical_hash,
    )
    validate_launch_runtime_backward_receipt(receipt)
    return receipt


def validate_launch_runtime_backward_receipt(
    receipt: LaunchRuntimeBackwardValidationReceipt,
) -> None:
    if receipt.schema_version != LAUNCH_RUNTIME_BACKWARD_RECEIPT_SCHEMA_VERSION:
        raise ValueError("launch runtime receipt schema mismatch")
    if receipt.target_name != LAUNCH_RUNTIME_BACKWARD_TARGET_NAME:
        raise ValueError("launch runtime receipt target mismatch")
    if receipt.proof_kind != PROOF_KIND_LAUNCH_RUNTIME_VALIDATION:
        raise ValueError("launch runtime receipt proof_kind mismatch")
    if not receipt.live_readiness_row_flip_authorized:
        raise ValueError("launch runtime receipt must authorize live row flip")
    authorized = tuple(receipt.readiness_row_flip_authorized_surface_names)
    if authorized != AUTHORIZED_R1_L_SURFACE_TUPLE:
        raise ValueError("launch runtime receipt authorized surface tuple mismatch")
    if receipt.r1_cpu_base_commit_sha != R1_CPU_BASE_COMMIT_SHA:
        raise ValueError("launch runtime receipt r1_cpu_base_commit_sha mismatch")
    launch_source = _require_nonempty_string(
        receipt.launch_source_commit_sha,
        field_name="launch_source_commit_sha",
    )
    manifest_embedded = _embedded_mapping(
        receipt.launch_manifest_embedded,
        field_name="launch_manifest_embedded",
        required_keys=LAUNCH_MANIFEST_EMBEDDED_KEYS,
    )
    if launch_source != manifest_embedded["launch_source_commit_sha"]:
        raise ValueError("launch runtime receipt launch_source_commit_sha mismatch")
    if manifest_embedded["r1_cpu_base_commit_sha"] != receipt.r1_cpu_base_commit_sha:
        raise ValueError("launch manifest embedded r1_cpu_base_commit_sha mismatch")
    if not receipt.ancestry_verified_at_launch_preflight:
        raise ValueError("launch runtime receipt requires ancestry_verified_at_launch_preflight")
    if not receipt.launch_runtime_validation_pass:
        raise ValueError("launch runtime receipt requires launch_runtime_validation_pass")
    if receipt.launch_manifest_sha256 != compute_launch_manifest_sha256(manifest_embedded):
        raise ValueError("launch runtime receipt launch_manifest_sha256 mismatch")
    env_embedded = _embedded_mapping(
        receipt.proof_env_embedded,
        field_name="proof_env_embedded",
    )
    if receipt.proof_env_hash_sha256 != compute_proof_env_hash_sha256(env_embedded):
        raise ValueError("launch runtime receipt proof_env_hash_sha256 mismatch")
    if not receipt.proof_command_argv:
        raise ValueError("launch runtime receipt requires proof_command_argv")
    if receipt.w6_parent_sha256_before != W6_PARENT_SHA256_PINNED:
        raise ValueError("launch runtime receipt w6_parent_sha256_before mismatch")
    if receipt.w6_parent_sha256_after != receipt.w6_parent_sha256_before:
        raise ValueError("launch runtime receipt w6_parent_sha256_after mismatch")
    if not _require_nonempty_string(receipt.w6_parent_path, field_name="w6_parent_path"):
        raise ValueError("launch runtime receipt requires w6_parent_path")
    expected_gpu_identity = compute_gpu_identity_sha256(
        gpu_name=receipt.gpu_name,
        gpu_uuid=receipt.gpu_uuid,
        driver_version=receipt.driver_version,
        cuda_version=receipt.cuda_version,
        torch_version=receipt.torch_version,
    )
    if receipt.gpu_identity_sha256 != expected_gpu_identity:
        raise ValueError("launch runtime receipt gpu_identity_sha256 mismatch")
    if not receipt.main_path_proven or not receipt.main_recompute_checkpoint_fired:
        raise ValueError("launch runtime receipt requires main path mechanism proof")
    if receipt.main_internal_payload_tensor_count != 0:
        raise ValueError("launch runtime receipt main internal payload must be zero")
    if receipt.main_baseline_saved_tensor_count <= receipt.main_recompute_saved_tensor_count:
        raise ValueError("launch runtime receipt main saved tensor counts invalid")
    if receipt.main_saved_tensor_payload_bytes_delta <= 0:
        raise ValueError("launch runtime receipt main payload bytes delta must be > 0")
    if receipt.retained_side_in_scope:
        if not receipt.retained_side_path_proven:
            raise ValueError("launch runtime receipt retained side must be proven when in scope")
        if not receipt.retained_side_recompute_checkpoint_fired:
            raise ValueError(
                "launch runtime receipt retained-side recompute checkpoint must fire"
            )
        if receipt.retained_side_internal_payload_tensor_count != 0:
            raise ValueError("launch runtime receipt retained internal payload must be zero")
        if (
            receipt.retained_side_baseline_saved_tensor_count
            <= receipt.retained_side_recompute_saved_tensor_count
        ):
            raise ValueError("launch runtime receipt retained saved tensor counts invalid")
        if receipt.retained_saved_tensor_payload_bytes_delta <= 0:
            raise ValueError("launch runtime receipt retained payload bytes delta must be > 0")
        if not receipt.loss_finite_retained:
            raise ValueError("launch runtime receipt requires finite retained loss when in scope")
    if receipt.paired_run_count < 3:
        raise ValueError("launch runtime receipt requires paired_run_count >= 3")
    expected_threshold = max(
        8 * 1024 * 1024,
        int(0.005 * receipt.cuda_peak_allocated_bytes_baseline_median),
    )
    if receipt.cuda_peak_reduction_threshold_bytes != expected_threshold:
        raise ValueError("launch runtime receipt cuda threshold bytes mismatch")
    threshold_met = (
        receipt.cuda_peak_allocated_bytes_delta_median
        >= receipt.cuda_peak_reduction_threshold_bytes
    )
    if receipt.cuda_peak_reduction_threshold_met != threshold_met:
        raise ValueError("launch runtime receipt cuda threshold met flag mismatch")
    if not receipt.loss_finite_main:
        raise ValueError("launch runtime receipt requires finite main loss")
    if receipt.applier_base_surface_count_sub2 != 3:
        raise ValueError("launch runtime receipt applier base sub2 count must be 3")
    if receipt.applier_result_sub2_surface_count != 4:
        raise ValueError("launch runtime receipt applier result sub2 count must be 4")
    if receipt.applier_result_ready_for_main_science:
        raise ValueError("launch runtime receipt applier must not set ready_for_main_science")
    if not receipt.applier_result_ready_for_pre_full_stack_diagnostic:
        raise ValueError(
            "launch runtime receipt applier must set ready_for_pre_full_stack_diagnostic"
        )
    if tuple(receipt.applier_flipped_surface_ids) != AUTHORIZED_R1_L_SURFACE_TUPLE:
        raise ValueError("launch runtime receipt applier flipped surface ids mismatch")
    if not _require_nonempty_string(
        receipt.log_artifact_sha256,
        field_name="log_artifact_sha256",
    ):
        raise ValueError("launch runtime receipt requires log_artifact_sha256")
    if receipt.non_claims != LAUNCH_RUNTIME_NON_CLAIMS:
        raise ValueError("launch runtime receipt non-claims must be exact")
    recomputed_hash = compute_canonical_launch_artifact_sha256(receipt.to_dict())
    if receipt.canonical_launch_artifact_sha256 != recomputed_hash:
        raise ValueError("launch runtime receipt canonical_launch_artifact_sha256 mismatch")


def validate_launch_runtime_backward_artifacts(
    receipt: LaunchRuntimeBackwardValidationReceipt,
    *,
    launch_manifest_bytes: bytes,
    env_snapshot_bytes: bytes,
    log_bytes: bytes | None,
) -> None:
    if hashlib.sha256(launch_manifest_bytes).hexdigest() != receipt.launch_manifest_sha256:
        raise ValueError("launch manifest bytes sha256 mismatch")
    try:
        manifest_payload = json.loads(launch_manifest_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("launch manifest bytes are not valid JSON") from exc
    if not isinstance(manifest_payload, dict):
        raise ValueError("launch manifest bytes must decode to an object")
    if str(manifest_payload.get("r1_cpu_base_commit_sha")) != receipt.r1_cpu_base_commit_sha:
        raise ValueError("launch manifest r1_cpu_base_commit_sha mismatch")
    if str(manifest_payload.get("launch_source_commit_sha")) != receipt.launch_source_commit_sha:
        raise ValueError("launch manifest launch_source_commit_sha mismatch")
    try:
        env_payload = json.loads(env_snapshot_bytes.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("env snapshot bytes are not valid JSON") from exc
    if not isinstance(env_payload, dict):
        raise ValueError("env snapshot bytes must decode to an object")
    env_embedded = {str(key): str(value) for key, value in env_payload.items()}
    if compute_proof_env_hash_sha256(env_embedded) != receipt.proof_env_hash_sha256:
        raise ValueError("env snapshot proof_env_hash_sha256 mismatch")
    if log_bytes is None or not log_bytes:
        raise ValueError("launch log snapshot bytes are required")
    if hashlib.sha256(log_bytes).hexdigest() != receipt.log_artifact_sha256:
        raise ValueError("launch log snapshot bytes sha256 mismatch")


def analyze_saved_tensor_hook_events(
    events: Sequence[Mapping[str, object]],
    *,
    boundary_tensor_shape: tuple[int, ...],
    boundary_tensor_dtype: str,
) -> dict[str, int]:
    boundary_count = sum(
        1
        for event in events
        if tuple(event.get("shape", ())) == boundary_tensor_shape
        and str(event.get("dtype", "")) == boundary_tensor_dtype
    )
    dummy_count = sum(1 for event in events if tuple(event.get("shape", ())) == (0,))
    internal_count = len(events) - boundary_count - dummy_count
    return {
        "observed_boundary_tensor_count": boundary_count,
        "observed_checkpoint_dummy_tensor_count": dummy_count,
        "observed_internal_payload_tensor_count": internal_count,
        "saved_tensor_count": len(events),
    }


def production_saved_tensor_path_proof_from_events(
    *,
    baseline_events: Sequence[Mapping[str, object]],
    recompute_events: Sequence[Mapping[str, object]],
    hidden_size: int,
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
) -> dict[str, int]:
    baseline_count = len(baseline_events)
    recompute_count = len(recompute_events)
    if baseline_count <= recompute_count:
        raise ValueError("saved-tensor proof must show recompute reduces saved tensors")

    dummy_count = sum(
        1 for event in recompute_events if tuple(event.get("shape", ())) == (0,)
    )
    checkpointed_count = sum(
        1
        for decision in recurrence_checkpoint_decisions(
            MODE_LOSSLESS_RECOMPUTE,
            H_cycles=H_cycles,
            L_cycles=L_cycles,
            bp_steps=bp_steps,
        )
        if decision.checkpoint
    )
    expected_boundary = checkpointed_count * 2
    expected_dummy = checkpointed_count
    boundary_like = [
        event
        for event in recompute_events
        if len(tuple(event.get("shape", ()))) == 3
        and int(tuple(event.get("shape", ()))[-1]) == hidden_size
    ]
    internal_payload = max(0, len(boundary_like) - expected_boundary)
    if dummy_count != expected_dummy:
        raise ValueError("saved-tensor proof checkpoint dummy tensor count mismatch")
    if len(boundary_like) != expected_boundary:
        raise ValueError("saved-tensor proof boundary tensor count mismatch")
    if internal_payload != 0:
        raise ValueError("saved-tensor proof observed internal recurrence payload tensors")
    return {
        "baseline_saved_tensor_count": baseline_count,
        "recompute_saved_tensor_count": recompute_count,
        "internal_payload_tensor_count": internal_payload,
    }


def saved_tensor_path_proof_from_events(
    *,
    baseline_events: Sequence[Mapping[str, object]],
    recompute_events: Sequence[Mapping[str, object]],
    boundary_tensor_shape: tuple[int, ...],
    boundary_tensor_dtype: str,
) -> dict[str, int]:
    baseline = analyze_saved_tensor_hook_events(
        baseline_events,
        boundary_tensor_shape=boundary_tensor_shape,
        boundary_tensor_dtype=boundary_tensor_dtype,
    )
    recompute = analyze_saved_tensor_hook_events(
        recompute_events,
        boundary_tensor_shape=boundary_tensor_shape,
        boundary_tensor_dtype=boundary_tensor_dtype,
    )
    if recompute["observed_internal_payload_tensor_count"] != 0:
        raise ValueError("saved-tensor proof observed internal recurrence payload tensors")
    if baseline["saved_tensor_count"] <= recompute["saved_tensor_count"]:
        raise ValueError("saved-tensor proof must show recompute reduces saved tensors")
    return {
        "baseline_saved_tensor_count": baseline["saved_tensor_count"],
        "recompute_saved_tensor_count": recompute["saved_tensor_count"],
        "internal_payload_tensor_count": recompute["observed_internal_payload_tensor_count"],
    }


def recompute_checkpoint_fired(
    *,
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
) -> bool:
    decisions = recurrence_checkpoint_decisions(
        MODE_LOSSLESS_RECOMPUTE,
        H_cycles=H_cycles,
        L_cycles=L_cycles,
        bp_steps=bp_steps,
    )
    return any(decision.checkpoint for decision in decisions)


def build_trainer_backward_wiring_proof_receipt(
    *,
    source_commit_sha: str,
    proof_command_argv: Sequence[str],
    H_cycles: int,
    L_cycles: int,
    bp_steps: int,
    main_path_proof: Mapping[str, object],
    retained_side_path_proof: Mapping[str, object] | None = None,
    retained_side_in_scope: bool = True,
    retained_side_skip_reason: str = "",
    backward_recompute_fixture_receipt_sha256: str = "",
) -> TrainerBackwardWiringProofReceipt:
    main_baseline = _require_int(
        main_path_proof.get("baseline_saved_tensor_count"),
        field_name="main_baseline_saved_tensor_count",
    )
    main_recompute = _require_int(
        main_path_proof.get("recompute_saved_tensor_count"),
        field_name="main_recompute_saved_tensor_count",
    )
    main_internal = _require_int(
        main_path_proof.get("internal_payload_tensor_count"),
        field_name="main_internal_payload_tensor_count",
    )
    main_checkpoint_fired = bool(main_path_proof.get("recompute_checkpoint_fired"))
    if main_internal != 0:
        raise ValueError("main path internal payload must be zero")
    if main_baseline <= main_recompute:
        raise ValueError("main path must show baseline > recompute saved tensors")
    if not main_checkpoint_fired:
        raise ValueError("main path recompute checkpoint must fire")

    retained_proven = False
    retained_checkpoint_fired = False
    retained_baseline = 0
    retained_recompute = 0
    retained_internal = 0
    if retained_side_in_scope:
        if retained_side_path_proof is None:
            raise ValueError("retained-side proof required when retained_side_in_scope")
        retained_baseline = _require_int(
            retained_side_path_proof.get("baseline_saved_tensor_count"),
            field_name="retained_side_baseline_saved_tensor_count",
        )
        retained_recompute = _require_int(
            retained_side_path_proof.get("recompute_saved_tensor_count"),
            field_name="retained_side_recompute_saved_tensor_count",
        )
        retained_internal = _require_int(
            retained_side_path_proof.get("internal_payload_tensor_count"),
            field_name="retained_side_internal_payload_tensor_count",
        )
        retained_checkpoint_fired = bool(
            retained_side_path_proof.get("recompute_checkpoint_fired")
        )
        if retained_internal != 0:
            raise ValueError("retained-side internal payload must be zero")
        if retained_baseline <= retained_recompute:
            raise ValueError(
                "retained-side path must show baseline > recompute saved tensors"
            )
        if not retained_checkpoint_fired:
            raise ValueError("retained-side recompute checkpoint must fire")
        retained_proven = True

    receipt = TrainerBackwardWiringProofReceipt(
        schema_version=TRAINER_BACKWARD_WIRING_RECEIPT_SCHEMA_VERSION,
        target_name=TRAINER_BACKWARD_WIRING_TARGET_NAME,
        proof_kind=PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING,
        source_commit_sha=_require_nonempty_string(
            source_commit_sha,
            field_name="source_commit_sha",
        ),
        proof_command_argv=tuple(str(arg) for arg in proof_command_argv),
        activation_relief_wiring_proof_flag=True,
        policy_mode=MODE_LOSSLESS_RECOMPUTE,
        main_path_proven=True,
        retained_side_path_proven=retained_proven,
        retained_side_in_scope=retained_side_in_scope,
        retained_side_skip_reason=str(retained_side_skip_reason),
        main_recompute_checkpoint_fired=main_checkpoint_fired,
        main_baseline_saved_tensor_count=main_baseline,
        main_recompute_saved_tensor_count=main_recompute,
        main_internal_payload_tensor_count=main_internal,
        retained_side_recompute_checkpoint_fired=retained_checkpoint_fired,
        retained_side_baseline_saved_tensor_count=retained_baseline,
        retained_side_recompute_saved_tensor_count=retained_recompute,
        retained_side_internal_payload_tensor_count=retained_internal,
        default_runtime_sub2_claim=False,
        activations_residuals_sub2_claim=False,
        live_readiness_row_flip_authorized=False,
        readiness_row_flip_authorized_surface_names=(),
        backward_recompute_fixture_receipt_sha256=str(
            backward_recompute_fixture_receipt_sha256
        ),
        optimizer_step_called=False,
        non_claims=TRAINER_BACKWARD_WIRING_NON_CLAIMS,
    )
    validate_trainer_backward_wiring_proof_receipt(receipt)
    return receipt


def validate_trainer_backward_wiring_proof_receipt(
    receipt: TrainerBackwardWiringProofReceipt,
) -> None:
    if receipt.schema_version != TRAINER_BACKWARD_WIRING_RECEIPT_SCHEMA_VERSION:
        raise ValueError("trainer backward wiring receipt schema mismatch")
    if receipt.target_name != TRAINER_BACKWARD_WIRING_TARGET_NAME:
        raise ValueError("trainer backward wiring receipt target mismatch")
    if receipt.proof_kind != PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING:
        raise ValueError("trainer backward wiring receipt requires cpu proof kind")
    if not receipt.activation_relief_wiring_proof_flag:
        raise ValueError("trainer backward wiring receipt requires wiring proof flag")
    if receipt.policy_mode != MODE_LOSSLESS_RECOMPUTE:
        raise ValueError("trainer backward wiring receipt requires lossless_recompute")
    if not receipt.main_path_proven:
        raise ValueError("trainer backward wiring receipt requires main path proof")
    if not receipt.main_recompute_checkpoint_fired:
        raise ValueError("trainer backward wiring receipt requires main recompute checkpoint fired")
    if receipt.retained_side_in_scope and not receipt.retained_side_path_proven:
        raise ValueError("retained-side path must be proven when in scope")
    if not receipt.retained_side_in_scope and receipt.retained_side_path_proven:
        raise ValueError("retained-side path cannot be proven when out of scope")
    if (
        receipt.retained_side_in_scope
        and not receipt.retained_side_recompute_checkpoint_fired
    ):
        raise ValueError(
            "trainer backward wiring receipt requires retained-side recompute checkpoint fired"
        )
    if receipt.default_runtime_sub2_claim or receipt.activations_residuals_sub2_claim:
        raise ValueError("CPU wiring receipt cannot claim default runtime or activations sub2")
    if receipt.live_readiness_row_flip_authorized:
        raise ValueError("CPU wiring receipt cannot authorize live readiness row flip")
    if receipt.readiness_row_flip_authorized_surface_names:
        raise ValueError("CPU wiring receipt must keep authorized surface list empty")
    if receipt.optimizer_step_called:
        raise ValueError("CPU wiring receipt must not call optimizer step")
    if receipt.main_internal_payload_tensor_count != 0:
        raise ValueError("main path internal payload must be zero")
    if (
        receipt.retained_side_in_scope
        and receipt.retained_side_internal_payload_tensor_count != 0
    ):
        raise ValueError("retained-side internal payload must be zero")
    if receipt.main_baseline_saved_tensor_count <= receipt.main_recompute_saved_tensor_count:
        raise ValueError("main path must show baseline > recompute saved tensors")
    if (
        receipt.retained_side_in_scope
        and receipt.retained_side_baseline_saved_tensor_count
        <= receipt.retained_side_recompute_saved_tensor_count
    ):
        raise ValueError("retained-side must show baseline > recompute saved tensors")
    if receipt.non_claims != TRAINER_BACKWARD_WIRING_NON_CLAIMS:
        raise ValueError("trainer backward wiring receipt non-claims must be exact")
