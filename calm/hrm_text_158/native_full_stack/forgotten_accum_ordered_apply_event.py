"""Ordered apply-event seam — pure schema, snapshot, identity, sequence validator.

STEP-1 only: no probe/ark/learner/runner_contract/science_driver wiring.
All validation summaries are nonclaimable / non-bankable / not runtime_proven.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping, Sequence

VALIDATION_SCHEMA_ID = "forgotten_accum_ordered_apply_event_validation_v1"
ATTACHMENT_KEY = "ordered_apply_event_validation_summary"
PRODUCER_LITERAL = "run_bounded_delta_steps_post_states_rebind"
APPLY_OUTCOME_SUCCESS = "SUCCESS"

IDENTITY_PROJECTION_FIELDS: tuple[str, ...] = (
    "seq",
    "arm_id",
    "optimizer_step_id",
    "apply_outcome",
    "producer",
)
TELEMETRY_EXCLUDED_FROM_EQUALITY: tuple[str, ...] = (
    "q_changed_count",
    "tensor_state_key_count",
)
FULL_PAYLOAD_FIELDS: tuple[str, ...] = IDENTITY_PROJECTION_FIELDS + TELEMETRY_EXCLUDED_FROM_EQUALITY


class OrderedApplyEventLogRefuse(ValueError):
    """Refuse non-empty ordered_apply_event_log before runner work (pure helper)."""


@dataclass(frozen=True)
class ExpectedIdentity:
    """Ark-owned expected sequence identity — never derived from observed events."""

    arm_id: str
    start_step: int
    steps: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "arm_id", str(self.arm_id))
        object.__setattr__(self, "start_step", int(self.start_step))
        object.__setattr__(self, "steps", int(self.steps))
        if self.steps < 0:
            raise ValueError("steps must be >= 0")


def require_empty_ordered_apply_event_log(
    ordered_apply_event_log: list[Any] | None,
) -> None:
    """Pure pre-work gate: None (disabled) OK; exact built-in empty list OK.

    Rejects non-list containers (tuple/custom sequence) before any runner work so
    invalid evidence holders cannot pass empty-len checks and only fail later on
    ``.append``. Ark allocates a fresh built-in ``list`` on the normal path.
    """

    if ordered_apply_event_log is None:
        return
    if type(ordered_apply_event_log) is not list:
        raise OrderedApplyEventLogRefuse(
            "ordered_apply_event_log must be a built-in list or None before "
            f"runner work; got {type(ordered_apply_event_log).__name__}"
        )
    if len(ordered_apply_event_log) != 0:
        raise OrderedApplyEventLogRefuse(
            "ordered_apply_event_log must be empty before runner work; "
            f"got len={len(ordered_apply_event_log)}"
        )


def make_success_apply_event(
    *,
    seq: int,
    arm_id: str,
    optimizer_step_id: int,
    q_changed_count: int,
    tensor_state_key_count: int,
    apply_outcome: str = APPLY_OUTCOME_SUCCESS,
    producer: str = PRODUCER_LITERAL,
) -> dict[str, Any]:
    """Build one SUCCESS apply-event dict (mutable; for runner append / tests)."""

    return {
        "seq": int(seq),
        "arm_id": str(arm_id),
        "optimizer_step_id": int(optimizer_step_id),
        "apply_outcome": str(apply_outcome),
        "q_changed_count": int(q_changed_count),
        "tensor_state_key_count": int(tensor_state_key_count),
        "producer": str(producer),
    }


def _payload_dict(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: event[key] for key in FULL_PAYLOAD_FIELDS if key in event}


def _identity_dict(event: Mapping[str, Any]) -> dict[str, Any]:
    return {key: event[key] for key in IDENTITY_PROJECTION_FIELDS if key in event}


def snapshot_ordered_apply_event_log(
    live_log: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """ONE immutable snapshot: tuple of MappingProxyType payload dicts."""

    frozen: list[Mapping[str, Any]] = []
    for event in live_log:
        payload = _payload_dict(event)
        # Copy then freeze so later live mutation cannot alter snapshot entries.
        frozen.append(MappingProxyType(dict(payload)))
    return tuple(frozen)


def build_expected_identity_projection(
    expected: ExpectedIdentity,
) -> list[dict[str, Any]]:
    """Independent expected identity sequence from ark-owned (arm, start_step, steps)."""

    return [
        {
            "seq": int(i),
            "arm_id": str(expected.arm_id),
            "optimizer_step_id": int(expected.start_step) + int(i),
            "apply_outcome": APPLY_OUTCOME_SUCCESS,
            "producer": PRODUCER_LITERAL,
        }
        for i in range(int(expected.steps))
    ]


def identity_projection_from_snapshot(
    snapshot: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [_identity_dict(event) for event in snapshot]


def full_payload_from_snapshot(
    snapshot: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    return [_payload_dict(event) for event in snapshot]


def _canonical_json_sha256(value: Any) -> str:
    blob = (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def identity_projection_sha256(projection: Sequence[Mapping[str, Any]]) -> str:
    """Equality-gate hash over identity fields only (excludes q/state telemetry)."""

    return _canonical_json_sha256([dict(row) for row in projection])


def full_payload_sha256(payloads: Sequence[Mapping[str, Any]]) -> str:
    """Forensic full-payload hash — NOT the equality gate."""

    return _canonical_json_sha256([dict(row) for row in payloads])


def characterize_dict_same_key_blindness(
    events: Sequence[Mapping[str, Any]],
    *,
    key_field: str = "optimizer_step_id",
) -> dict[Any, Mapping[str, Any]]:
    """Characterization: same-key dict overwrite loses prior events (duplicates vanish)."""

    blind: dict[Any, Mapping[str, Any]] = {}
    for event in events:
        blind[event[key_field]] = event
    return blind


def _authority_flags() -> dict[str, Any]:
    return {
        "claimable": False,
        "bankable": False,
        "forensic_only": True,
        "runtime_proven": False,
    }


def validate_ordered_apply_event_sequence(
    snapshot: Sequence[Mapping[str, Any]],
    expected: ExpectedIdentity,
) -> dict[str, Any]:
    """Validate immutable snapshot against ark-owned expected identity.

    Never consumes a live mutable list. Always returns nonclaimable summary.
    Does not raise on mismatch — caller attaches summary (Option-A UNVERIFIED path).
    """

    expected_proj = build_expected_identity_projection(expected)
    observed_proj = identity_projection_from_snapshot(snapshot)
    expected_ids = [row["optimizer_step_id"] for row in expected_proj]
    observed_ids = [
        int(row["optimizer_step_id"])
        for row in observed_proj
        if "optimizer_step_id" in row
    ]
    exp_counts = Counter(expected_ids)
    obs_counts = Counter(observed_ids)

    missing_count = sum(
        max(0, exp_counts[step_id] - obs_counts.get(step_id, 0))
        for step_id in exp_counts
    )
    duplicate_count = sum(
        max(0, obs_counts[step_id] - exp_counts.get(step_id, 0))
        for step_id in exp_counts
    )
    extra_count = sum(
        count for step_id, count in obs_counts.items() if step_id not in exp_counts
    )
    wrong_arm_count = sum(
        1
        for event in snapshot
        if str(event.get("arm_id", "")) != str(expected.arm_id)
    )
    reorder_detected = list(observed_proj) != list(expected_proj)

    expected_hash = identity_projection_sha256(expected_proj)
    observed_hash = identity_projection_sha256(observed_proj)
    forensic_hash = full_payload_sha256(full_payload_from_snapshot(snapshot))

    expected_count = int(expected.steps)
    observed_count = len(snapshot)
    sequence_exact_ok = (
        missing_count == 0
        and duplicate_count == 0
        and extra_count == 0
        and wrong_arm_count == 0
        and reorder_detected is False
        and expected_hash == observed_hash
        and expected_count == observed_count
    )

    summary: dict[str, Any] = {
        "schema_id": VALIDATION_SCHEMA_ID,
        "arm_id": str(expected.arm_id),
        "start_step": int(expected.start_step),
        "steps": int(expected.steps),
        "expected_count": expected_count,
        "observed_count": observed_count,
        "sequence_exact_ok": bool(sequence_exact_ok),
        "missing_count": int(missing_count),
        "duplicate_count": int(duplicate_count),
        "extra_count": int(extra_count),
        "wrong_arm_count": int(wrong_arm_count),
        "reorder_detected": bool(reorder_detected),
        "expected_identity_projection_sha256": expected_hash,
        "observed_identity_projection_sha256": observed_hash,
        "full_payload_sha256": forensic_hash,
    }
    summary.update(_authority_flags())
    return summary


__all__ = [
    "ATTACHMENT_KEY",
    "APPLY_OUTCOME_SUCCESS",
    "ExpectedIdentity",
    "FULL_PAYLOAD_FIELDS",
    "IDENTITY_PROJECTION_FIELDS",
    "OrderedApplyEventLogRefuse",
    "PRODUCER_LITERAL",
    "TELEMETRY_EXCLUDED_FROM_EQUALITY",
    "VALIDATION_SCHEMA_ID",
    "build_expected_identity_projection",
    "characterize_dict_same_key_blindness",
    "full_payload_sha256",
    "identity_projection_from_snapshot",
    "identity_projection_sha256",
    "make_success_apply_event",
    "require_empty_ordered_apply_event_log",
    "snapshot_ordered_apply_event_log",
    "validate_ordered_apply_event_sequence",
]
