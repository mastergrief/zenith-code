"""Phase-B contracts for forgotten-accum training-equivalence (U/E/R0/RW).

CPU-facing schema only. GPU 4-arm formal run is a separate later gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping


SCHEMA_VERSION = "forgotten_accum_training_equivalence_contracts/v1"

# Identity pins (re-hash parent at launch)
PARENT_SHA256_FULL = (
    "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
)
PARENT_SHA256_PREFIX = "9b4e311a"
GLOBAL_CAP_CONTRACT = "c1_banked_faithful_long_run_global_cap"
ELIGIBLE_SCOPE = "all-bitlinear"
CARRIER_NONE = "NONE"
BATCH_SEED = 44
SUPPORT_ORDER_SEED = 43
ORDERING_SEED = 17
BATCH_SIZE = 1
RUNWAY_STEPS = 1500
T_CUT = 500
W_REWARM_STEPS = 32
RECIPE = "slow-safe"
SAVE_CADENCE = (250, 500, 750, 1000, 1250, 1500)

PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY = "PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY"
FUTURE_STREAM_MATCHED_BUDGET = "FUTURE_STREAM_MATCHED_BUDGET"
RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0 = "ORDINARY_SELECTOR_SAME_AS_R0"

# Re-export Phase-A dense site token for smoke predicates
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
)


class ArmId(str, Enum):
    U = "U"
    E = "E"
    R0 = "R0"
    RW = "RW"


class ResumePolicy(str, Enum):
    """Sole policy delta across E vs R0/RW on the SHARED serializer/loader."""

    EXACT_PRESERVE = "exact_preserve"
    ZERO_STRIP = "zero_strip"


class FailClosedClass(str, Enum):
    CONTROL_INVALID = "CONTROL_INVALID"
    FORGET_NO_REWARM_BANK_EQUIVALENT = "FORGET_NO_REWARM_BANK_EQUIVALENT"
    FORGET_REWARM_BANK_EQUIVALENT_WITHIN_FROZEN_COST_CAP = (
        "FORGET_REWARM_BANK_EQUIVALENT_WITHIN_FROZEN_COST_CAP"
    )
    FORGET_REWARM_PARTIAL_OR_CUT_DEPENDENT = "FORGET_REWARM_PARTIAL_OR_CUT_DEPENDENT"
    FORGET_REWARM_NOT_TOLERATED = "FORGET_REWARM_NOT_TOLERATED"
    REWARM_ACCOUNTING_INVALID = "REWARM_ACCOUNTING_INVALID"
    BANK_INPUTS_INVALID = "BANK_INPUTS_INVALID"


RESUME_ARMS = (ArmId.E, ArmId.R0, ArmId.RW)


def policy_for_arm(arm: ArmId) -> ResumePolicy | None:
    if arm is ArmId.U:
        return None
    if arm is ArmId.E:
        return ResumePolicy.EXACT_PRESERVE
    if arm in (ArmId.R0, ArmId.RW):
        return ResumePolicy.ZERO_STRIP
    raise ValueError(f"unknown arm {arm}")


def flip_defer_schedule(arm: ArmId, *, post_cut_step_index: int) -> bool:
    """post_cut_step_index is 1-based within post-cut budget (1 == t_cut+1)."""

    if arm is not ArmId.RW:
        return False
    return 1 <= int(post_cut_step_index) <= int(W_REWARM_STEPS)


@dataclass(frozen=True)
class IdentityManifest:
    parent_sha256: str = PARENT_SHA256_FULL
    batch_seed: int = BATCH_SEED
    support_order_seed: int = SUPPORT_ORDER_SEED
    ordering_seed: int = ORDERING_SEED
    batch_size: int = BATCH_SIZE
    eligible_scope: str = ELIGIBLE_SCOPE
    recipe: str = RECIPE
    runway_steps: int = RUNWAY_STEPS
    t_cut: int = T_CUT
    W: int = W_REWARM_STEPS
    global_cap_contract: str = GLOBAL_CAP_CONTRACT
    save_cadence: tuple[int, ...] = SAVE_CADENCE
    future_stream: str = FUTURE_STREAM_MATCHED_BUDGET

    def as_dict(self) -> dict[str, Any]:
        return {
            "parent_sha256": self.parent_sha256,
            "batch_seed": self.batch_seed,
            "support_order_seed": self.support_order_seed,
            "ordering_seed": self.ordering_seed,
            "batch_size": self.batch_size,
            "eligible_scope": self.eligible_scope,
            "recipe": self.recipe,
            "runway_steps": self.runway_steps,
            "t_cut": self.t_cut,
            "W": self.W,
            "global_cap_contract": self.global_cap_contract,
            "save_cadence": list(self.save_cadence),
            "future_stream": self.future_stream,
        }


@dataclass(frozen=True)
class ArmManifest:
    arm: ArmId
    identity: IdentityManifest
    resume_policy: ResumePolicy | None
    flip_application_deferred_during_W: bool
    artifact_root_relpath: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm.value,
            "identity": self.identity.as_dict(),
            "resume_policy": None if self.resume_policy is None else self.resume_policy.value,
            "flip_application_deferred_during_W": bool(
                self.flip_application_deferred_during_W
            ),
            "artifact_root_relpath": self.artifact_root_relpath,
        }


def build_all_arm_manifests(
    *,
    experiment_root: str = "forgotten_accum_training_equivalence",
) -> dict[str, ArmManifest]:
    identity = IdentityManifest()
    out: dict[str, ArmManifest] = {}
    for arm in ArmId:
        out[arm.value] = ArmManifest(
            arm=arm,
            identity=identity,
            resume_policy=policy_for_arm(arm),
            flip_application_deferred_during_W=(arm is ArmId.RW),
            artifact_root_relpath=f"{experiment_root}/arms/{arm.value}",
        )
    return out


@dataclass(frozen=True)
class LedgerFieldSchema:
    required_fields: tuple[str, ...] = (
        "base_packed_q_bpw",
        "mandatory_metadata_bpw",
        "accumulator_persistent_bpw_claimed",
        "resume_seed_schedule_RNG_version_fields_bits",
        "replay_payload_bpw",
        "rewarm_examples_seen",
        "forward_count",
        "backward_count",
        "update_count",
        "gpu_time_seconds",
        "surplus_compute_vs_E",
        "surplus_compute_vs_U",
    )


SMOKE_CPU_PREDICATES = (
    "carrier_must_be_dense_legacy_not_event_coded",
    "forgotten_accum_cap_site_branch_equals_DENSE_LEGACY_CAP_SITE_ID",
    "flip_application_deferred_true_engages_during_W_law",
    "default_false_ordinary_cap_path",
)


__all__ = [
    "CARRIER_NONE",
    "assert_carrier_preflight",

    "SCHEMA_VERSION",
    "ArmId",
    "ResumePolicy",
    "FailClosedClass",
    "IdentityManifest",
    "ArmManifest",
    "LedgerFieldSchema",
    "DENSE_LEGACY_CAP_SITE_ID",
    "PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY",
    "FUTURE_STREAM_MATCHED_BUDGET",
    "build_all_arm_manifests",
    "flip_defer_schedule",
    "policy_for_arm",
    "SMOKE_CPU_PREDICATES",
    "PARENT_SHA256_FULL",
    "GLOBAL_CAP_CONTRACT",
    "T_CUT",
    "W_REWARM_STEPS",
]


def assert_carrier_preflight(
    *,
    live_acc_carrier_selector: str,
    global_cap_contract: str,
    eligible_scope: str,
    event_coded_flags_present: bool = False,
) -> None:
    if live_acc_carrier_selector != CARRIER_NONE:
        raise ValueError(f"PREFLIGHT_REFUSE: carrier must be {CARRIER_NONE!r}")
    if global_cap_contract != GLOBAL_CAP_CONTRACT:
        raise ValueError(f"PREFLIGHT_REFUSE: cap must be {GLOBAL_CAP_CONTRACT!r}")
    if eligible_scope != ELIGIBLE_SCOPE:
        raise ValueError(f"PREFLIGHT_REFUSE: scope must be {ELIGIBLE_SCOPE!r}")
    if event_coded_flags_present:
        raise ValueError("PREFLIGHT_REFUSE: event-coded flags must be ABSENT")
