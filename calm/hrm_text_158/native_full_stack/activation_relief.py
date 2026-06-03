"""Lossless activation/saved-tensor relief contract for HRM-Text-1.58.

This slice lands the interface and CPU-provable lossless recompute path only.
Real peak-memory relief and wall-clock tradeoff receipts are deferred to a
gpu:0 resource-lane run because CPU tests cannot measure CUDA activation peaks.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


ACTIVATION_RELIEF_SCHEMA_VERSION = "hrm_text_158_activation_relief/v0.lossless_recompute"

MODE_OFF = "off"
MODE_LOSSLESS_RECOMPUTE = "lossless_recompute"
MODE_LOSSY_ACTIVATION_STORAGE = "lossy_activation_storage"

TIER1_LOSSLESS_RECOMPUTE = "tier1_lossless_recompute"
TIER2_LOSSY_ACTIVATION_STORAGE_DEFERRED = "tier2_lossy_activation_storage_deferred"

TARGET_GRAD_ENABLED_RECURRENCE = "grad_enabled_recurrence"

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
