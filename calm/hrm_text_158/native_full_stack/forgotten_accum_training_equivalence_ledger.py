"""Cost/persistence ledger arithmetic for forgotten-accum training-equivalence."""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    W_REWARM_STEPS,
    ArmId,
    FailClosedClass,
)


LOG2_3 = math.log2(3.0)  # ≈ 1.58496250072


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


@dataclass(frozen=True)
class ArmComputeCounts:
    arm: str
    forward_count: int
    backward_count: int
    update_count: int
    gpu_time_seconds: float
    rewarm_examples_seen: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PersistenceLedger:
    base_packed_q_bpw: float
    mandatory_metadata_bpw: float
    accumulator_persistent_bpw_claimed: float
    resume_seed_schedule_RNG_version_fields_bits: int
    replay_payload_bpw: float
    arms: dict[str, ArmComputeCounts]
    surplus_compute_vs_E: dict[str, float]
    surplus_compute_vs_U: dict[str, float]
    classification: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "base_packed_q_bpw": self.base_packed_q_bpw,
            "mandatory_metadata_bpw": self.mandatory_metadata_bpw,
            "accumulator_persistent_bpw_claimed": self.accumulator_persistent_bpw_claimed,
            "resume_seed_schedule_RNG_version_fields_bits": (
                self.resume_seed_schedule_RNG_version_fields_bits
            ),
            "replay_payload_bpw": self.replay_payload_bpw,
            "arms": {k: v.as_dict() for k, v in self.arms.items()},
            "surplus_compute_vs_E": dict(self.surplus_compute_vs_E),
            "surplus_compute_vs_U": dict(self.surplus_compute_vs_U),
            "classification": self.classification,
        }


def base_packed_q_bpw() -> float:
    return float(LOG2_3)


def build_ledger(
    *,
    arm_counts: Mapping[str, ArmComputeCounts],
    mandatory_metadata_bpw: float = 0.0,
    accumulator_persistent_bpw_claimed: float = 0.0,
    resume_seed_schedule_RNG_version_fields_bits: int = 0,
    replay_payload_bpw: float = 0.0,
    surplus_tolerance: float = 1e-9,
) -> PersistenceLedger:
    schema = LedgerFieldSchema()
    arms = {k: arm_counts[k] for k in sorted(arm_counts)}
    if ArmId.RW.value in arms:
        if int(arms[ArmId.RW.value].rewarm_examples_seen) != int(W_REWARM_STEPS):
            raise AssertionError("RW rewarm_examples_seen must equal W")
    if ArmId.R0.value in arms and int(arms[ArmId.R0.value].rewarm_examples_seen) != 0:
        raise AssertionError("R0 rewarm_examples_seen must be 0")

    surplus_e: dict[str, float] = {}
    surplus_u: dict[str, float] = {}
    e = arms.get(ArmId.E.value)
    u = arms.get(ArmId.U.value)
    for name, counts in arms.items():
        if e is not None:
            surplus_e[name] = float(counts.update_count - e.update_count)
        if u is not None:
            surplus_u[name] = float(counts.update_count - u.update_count)

    classification = None
    if ArmId.RW.value in surplus_e and abs(surplus_e[ArmId.RW.value]) > surplus_tolerance:
        classification = FailClosedClass.REWARM_ACCOUNTING_INVALID.value
    if ArmId.RW.value in surplus_u and abs(surplus_u[ArmId.RW.value]) > surplus_tolerance:
        classification = FailClosedClass.REWARM_ACCOUNTING_INVALID.value
    if float(replay_payload_bpw) != 0.0:
        classification = FailClosedClass.REWARM_ACCOUNTING_INVALID.value
    if float(accumulator_persistent_bpw_claimed) != 0.0:
        # R0/RW claim 0; non-zero claim without proof is invalid for this ledger
        classification = FailClosedClass.REWARM_ACCOUNTING_INVALID.value

    ledger = PersistenceLedger(
        base_packed_q_bpw=base_packed_q_bpw(),
        mandatory_metadata_bpw=float(mandatory_metadata_bpw),
        accumulator_persistent_bpw_claimed=float(accumulator_persistent_bpw_claimed),
        resume_seed_schedule_RNG_version_fields_bits=int(
            resume_seed_schedule_RNG_version_fields_bits
        ),
        replay_payload_bpw=float(replay_payload_bpw),
        arms=arms,
        surplus_compute_vs_E=surplus_e,
        surplus_compute_vs_U=surplus_u,
        classification=classification,
    )
    # schema presence check
    blob = ledger.as_dict()
    for field in schema.required_fields:
        if field in ("forward_count", "backward_count", "update_count", "gpu_time_seconds"):
            # nested under arms
            continue
        if field.startswith("surplus_"):
            continue
        if field not in blob and field not in {
            "rewarm_examples_seen",
        }:
            # top-level or per-arm — accept if per-arm carries it
            if field == "rewarm_examples_seen":
                continue
            if field not in blob:
                raise AssertionError(f"ledger missing field {field}")
    return ledger


__all__ = [
    "LOG2_3",
    "ArmComputeCounts",
    "PersistenceLedger",
    "base_packed_q_bpw",
    "build_ledger",
]
