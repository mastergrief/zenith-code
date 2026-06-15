"""Lossless activation/saved-tensor relief contract for HRM-Text-1.58.

This slice lands the interface and CPU-provable lossless recompute path only.
Real peak-memory relief and wall-clock tradeoff receipts are deferred to a
gpu:0 resource-lane run because CPU tests cannot measure CUDA activation peaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


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
    """Deferred R1-L receipt shape; synthetic mint allowed in CPU applier tests."""

    proof_kind: str
    live_readiness_row_flip_authorized: bool
    readiness_row_flip_authorized_surface_names: tuple[str, ...]
    source_commit_sha: str
    launch_runtime_validation_pass: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "proof_kind": self.proof_kind,
            "live_readiness_row_flip_authorized": self.live_readiness_row_flip_authorized,
            "readiness_row_flip_authorized_surface_names": list(
                self.readiness_row_flip_authorized_surface_names
            ),
            "source_commit_sha": self.source_commit_sha,
            "launch_runtime_validation_pass": self.launch_runtime_validation_pass,
        }


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


def validate_launch_runtime_backward_receipt(
    receipt: LaunchRuntimeBackwardValidationReceipt,
) -> None:
    if receipt.proof_kind != PROOF_KIND_LAUNCH_RUNTIME_VALIDATION:
        raise ValueError("launch runtime receipt proof_kind mismatch")
    if not receipt.live_readiness_row_flip_authorized:
        raise ValueError("launch runtime receipt must authorize live row flip")
    authorized = tuple(receipt.readiness_row_flip_authorized_surface_names)
    if authorized != AUTHORIZED_R1_L_SURFACE_TUPLE:
        raise ValueError("launch runtime receipt authorized surface tuple mismatch")
    if not _require_nonempty_string(receipt.source_commit_sha, field_name="source_commit_sha"):
        raise ValueError("launch runtime receipt requires source_commit_sha")
    if not receipt.launch_runtime_validation_pass:
        raise ValueError("launch runtime receipt requires launch_runtime_validation_pass")
