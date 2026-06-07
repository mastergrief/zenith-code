"""Lossless activation/saved-tensor relief contract for HRM-Text-1.58.

This slice lands the interface and CPU-provable lossless recompute path only.
Real peak-memory relief and wall-clock tradeoff receipts are deferred to a
gpu:0 resource-lane run because CPU tests cannot measure CUDA activation peaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTIVATION_RELIEF_SCHEMA_VERSION = "hrm_text_158_activation_relief/v0.lossless_recompute"
BACKWARD_RECOMPUTE_RECEIPT_SCHEMA_VERSION = (
    "hrm_text_158_backward_saved_tensors_recompute/v0.saved_tensor_hook"
)
BACKWARD_RECOMPUTE_TARGET_NAME = "step3a1_backward_saved_tensors_recompute"

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
