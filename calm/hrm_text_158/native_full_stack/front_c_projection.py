"""Front-C CPU/static density, ledger, and decision-equivalence reducers.

This scaffold consumes compact bounded-delta audit timeline rows and computes
the pre-registered Front-C projection. It deliberately does not build an
accumulator encoder, touch GPU paths, or integrate with the live learner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaInclusiveLedger,
    BoundedDeltaStorageProjection,
    bounded_delta_inclusive_ledger,
    project_bounded_delta_accumulator_bpw,
    validate_bounded_delta_inclusive_ledger,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import (
    BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT,
    BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT,
    BASE3_Q_FORMAT,
    Base3QEntropyLedgerRow,
    validate_base3_q_entropy_ledger,
)


FRONT_C_SCHEMA_VERSION = "hrm_text_158_front_c/v0.cpu_static_projection_scaffold"
FRONT_C_LABEL = "front_c_bounded_delta_density_ledger_decision_equivalence_scaffold"
FRONT_C_PAYLOAD_ONLY_GATING_PHRASE = (
    "Report payload-only density for intuition, but gate only on "
    "overhead-inclusive projected bpw plus zero-drift decision equivalence."
)
FRONT_C_NO_VIABILITY_CLAIM = (
    "no Front-C viability claim until run-derived density + zero-drift evidence lands"
)
COUNT_ONLY_ARTIFACT_REJECTION = (
    "count-only Front-C artifacts cannot compute union/churn; per-surface "
    "identities are required"
)

FrontCIdentity = tuple[str, int]

SURFACE_FIELD_NAMES = (
    "current_magnitude_threshold_keys",
    "active_next_step_keys",
    "ranking_sensitive_exact_keys",
    "global_cap_frontier_keys",
    "backlog_carry_keys",
    "replay_veto_residual_keys",
)


def _bits_per_weight(bits: int | float, eligible_weight_count: int) -> float:
    eligible = int(eligible_weight_count)
    if eligible <= 0:
        raise ValueError("eligible_weight_count must be > 0")
    return float(bits) / float(eligible)


def _p95(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    idx = int(math.ceil(0.95 * float(len(ordered))) - 1)
    idx = max(0, min(idx, len(ordered) - 1))
    return float(ordered[idx])


def _mean(values: Sequence[float]) -> float:
    return float(sum(float(value) for value in values) / float(len(values))) if values else 0.0


def _normalize_identity(value: Any, *, name: str) -> FrontCIdentity:
    if isinstance(value, Mapping):
        state_key = value.get("state_key", value.get("state"))
        flat_index = value.get("flat_index", value.get("index"))
    elif isinstance(value, (list, tuple)) and len(value) == 2:
        state_key, flat_index = value
    else:
        raise ValueError(f"{name} identity must be [state_key, flat_index] or a mapping")

    state = str(state_key)
    if not state:
        raise ValueError(f"{name} state_key must be non-empty")
    index = int(flat_index)
    if index < 0:
        raise ValueError(f"{name} flat_index must be >= 0")
    return state, index


def _normalize_identity_tuple(values: Sequence[Any] | None, *, name: str) -> tuple[FrontCIdentity, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of identities, not a string")
    normalized = {_normalize_identity(value, name=name) for value in values}
    return tuple(sorted(normalized))


def _identity_sha256(identities: Sequence[FrontCIdentity]) -> str:
    h = hashlib.sha256()
    for state_key, flat_index in sorted((str(k), int(i)) for k, i in identities):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(flat_index).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _identity_dicts(identities: Sequence[FrontCIdentity]) -> list[dict[str, int | str]]:
    return [
        {"state_key": state_key, "flat_index": int(flat_index)}
        for state_key, flat_index in sorted(identities)
    ]


def _surface_union(step: "FrontCDecisionSurfaceStep") -> tuple[FrontCIdentity, ...]:
    out: set[FrontCIdentity] = set()
    for field_name in SURFACE_FIELD_NAMES:
        out |= set(getattr(step, field_name))
    return tuple(sorted(out))


@dataclass(frozen=True)
class FrontCDecisionSurfaceStep:
    """One compact Front-C timeline row with per-surface decision identities."""

    step: int
    eligible_weight_count: int
    current_magnitude_threshold_keys: Sequence[Any] = ()
    active_next_step_keys: Sequence[Any] = ()
    ranking_sensitive_exact_keys: Sequence[Any] = ()
    global_cap_frontier_keys: Sequence[Any] = ()
    backlog_carry_keys: Sequence[Any] = ()
    replay_veto_residual_keys: Sequence[Any] = ()

    def __post_init__(self) -> None:
        if int(self.step) < 0:
            raise ValueError("step must be >= 0")
        if int(self.eligible_weight_count) <= 0:
            raise ValueError("eligible_weight_count must be > 0")
        object.__setattr__(self, "step", int(self.step))
        object.__setattr__(self, "eligible_weight_count", int(self.eligible_weight_count))
        for field_name in SURFACE_FIELD_NAMES:
            object.__setattr__(
                self,
                field_name,
                _normalize_identity_tuple(getattr(self, field_name), name=field_name),
            )

    @property
    def decision_relevant_exact_keys(self) -> tuple[FrontCIdentity, ...]:
        return _surface_union(self)

    def to_dict(self) -> dict[str, Any]:
        return {
            "step": int(self.step),
            "eligible_weight_count": int(self.eligible_weight_count),
            **{
                field_name: _identity_dicts(getattr(self, field_name))
                for field_name in SURFACE_FIELD_NAMES
            },
        }


def normalize_front_c_decision_surface_step(
    value: FrontCDecisionSurfaceStep | Mapping[str, Any],
) -> FrontCDecisionSurfaceStep:
    """Normalize a timeline artifact row and reject count-only fake evidence."""

    if isinstance(value, FrontCDecisionSurfaceStep):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("Front-C timeline rows must be mappings or FrontCDecisionSurfaceStep")

    count_fields = tuple(
        sorted(
            str(key)
            for key in value
            if (
                str(key) != "eligible_weight_count"
                and (str(key).endswith("_count") or str(key) == "decision_relevant_exact_count")
            )
        )
    )
    if count_fields:
        raise ValueError(f"{COUNT_ONLY_ARTIFACT_REJECTION}: count_fields={count_fields}")

    return FrontCDecisionSurfaceStep(
        step=int(value["step"]),
        eligible_weight_count=int(value["eligible_weight_count"]),
        **{field_name: value.get(field_name, ()) for field_name in SURFACE_FIELD_NAMES},
    )


@dataclass(frozen=True)
class FrontCStepDensity:
    step: int
    eligible_weight_count: int
    current_magnitude_threshold_count: int
    active_next_step_count: int
    ranking_sensitive_exact_count: int
    global_cap_frontier_count: int
    backlog_carry_count: int
    replay_veto_residual_count: int
    decision_relevant_exact_count: int
    current_magnitude_threshold_density: float
    active_next_step_density: float
    ranking_sensitive_exact_density: float
    global_cap_frontier_density: float
    backlog_carry_density: float
    replay_veto_residual_density: float
    decision_relevant_exact_density: float
    current_magnitude_threshold_indices_sha256: str
    active_next_step_indices_sha256: str
    ranking_sensitive_exact_indices_sha256: str
    global_cap_frontier_indices_sha256: str
    backlog_carry_indices_sha256: str
    replay_veto_residual_indices_sha256: str
    decision_relevant_exact_indices_sha256: str

    def to_dict(self) -> dict[str, int | float | str]:
        return asdict(self)


def measure_front_c_step_density(step: FrontCDecisionSurfaceStep | Mapping[str, Any]) -> FrontCStepDensity:
    normalized = normalize_front_c_decision_surface_step(step)
    eligible = int(normalized.eligible_weight_count)
    counts = {field_name: len(getattr(normalized, field_name)) for field_name in SURFACE_FIELD_NAMES}
    hashes = {
        field_name: _identity_sha256(getattr(normalized, field_name))
        for field_name in SURFACE_FIELD_NAMES
    }
    decision_relevant = normalized.decision_relevant_exact_keys
    return FrontCStepDensity(
        step=int(normalized.step),
        eligible_weight_count=eligible,
        current_magnitude_threshold_count=counts["current_magnitude_threshold_keys"],
        active_next_step_count=counts["active_next_step_keys"],
        ranking_sensitive_exact_count=counts["ranking_sensitive_exact_keys"],
        global_cap_frontier_count=counts["global_cap_frontier_keys"],
        backlog_carry_count=counts["backlog_carry_keys"],
        replay_veto_residual_count=counts["replay_veto_residual_keys"],
        decision_relevant_exact_count=len(decision_relevant),
        current_magnitude_threshold_density=_bits_per_weight(
            counts["current_magnitude_threshold_keys"],
            eligible,
        ),
        active_next_step_density=_bits_per_weight(counts["active_next_step_keys"], eligible),
        ranking_sensitive_exact_density=_bits_per_weight(
            counts["ranking_sensitive_exact_keys"],
            eligible,
        ),
        global_cap_frontier_density=_bits_per_weight(counts["global_cap_frontier_keys"], eligible),
        backlog_carry_density=_bits_per_weight(counts["backlog_carry_keys"], eligible),
        replay_veto_residual_density=_bits_per_weight(counts["replay_veto_residual_keys"], eligible),
        decision_relevant_exact_density=_bits_per_weight(len(decision_relevant), eligible),
        current_magnitude_threshold_indices_sha256=hashes["current_magnitude_threshold_keys"],
        active_next_step_indices_sha256=hashes["active_next_step_keys"],
        ranking_sensitive_exact_indices_sha256=hashes["ranking_sensitive_exact_keys"],
        global_cap_frontier_indices_sha256=hashes["global_cap_frontier_keys"],
        backlog_carry_indices_sha256=hashes["backlog_carry_keys"],
        replay_veto_residual_indices_sha256=hashes["replay_veto_residual_keys"],
        decision_relevant_exact_indices_sha256=_identity_sha256(decision_relevant),
    )


@dataclass(frozen=True)
class FrontCTimelineDensitySummary:
    schema_version: str
    label: str
    step_count: int
    eligible_weight_count: int
    step_densities: tuple[FrontCStepDensity, ...]
    max_decision_relevant_exact_density: float
    p95_decision_relevant_exact_density: float
    union_decision_relevant_exact_count: int
    union_decision_relevant_exact_density: float
    union_decision_relevant_exact_indices_sha256: str
    union_current_magnitude_threshold_count: int
    union_active_next_step_count: int
    union_ranking_sensitive_exact_count: int
    union_global_cap_frontier_count: int
    union_backlog_carry_count: int
    union_replay_veto_residual_count: int
    transition_count: int
    total_entry_count: int
    total_exit_count: int
    max_entry_rate: float
    p95_entry_rate: float
    max_exit_rate: float
    p95_exit_rate: float
    max_churn_rate: float
    p95_churn_rate: float
    mean_churn_rate: float
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["step_densities"] = [density.to_dict() for density in self.step_densities]
        payload["non_claims"] = list(self.non_claims)
        return payload


def measure_front_c_timeline_density(
    steps: Sequence[FrontCDecisionSurfaceStep | Mapping[str, Any]],
) -> FrontCTimelineDensitySummary:
    """Compute max/p95/union/churn over compact decision-surface timeline rows."""

    if not steps:
        raise ValueError("Front-C timeline must contain at least one row")
    normalized = tuple(normalize_front_c_decision_surface_step(step) for step in steps)
    ordered = tuple(sorted(normalized, key=lambda step: step.step))
    if len({step.step for step in ordered}) != len(ordered):
        raise ValueError("Front-C timeline steps must be unique")
    eligible = int(ordered[0].eligible_weight_count)
    if any(int(step.eligible_weight_count) != eligible for step in ordered):
        raise ValueError("Front-C timeline eligible_weight_count must be stable across rows")

    step_densities = tuple(measure_front_c_step_density(step) for step in ordered)
    decision_sets = [set(step.decision_relevant_exact_keys) for step in ordered]
    union_decision = set().union(*decision_sets)
    union_by_surface: dict[str, set[FrontCIdentity]] = {field_name: set() for field_name in SURFACE_FIELD_NAMES}
    for step in ordered:
        for field_name in SURFACE_FIELD_NAMES:
            union_by_surface[field_name] |= set(getattr(step, field_name))

    entry_rates: list[float] = []
    exit_rates: list[float] = []
    churn_rates: list[float] = []
    total_entry = 0
    total_exit = 0
    for before, after in zip(decision_sets, decision_sets[1:]):
        entries = len(after - before)
        exits = len(before - after)
        churn = len(before ^ after)
        total_entry += entries
        total_exit += exits
        entry_rates.append(_bits_per_weight(entries, eligible))
        exit_rates.append(_bits_per_weight(exits, eligible))
        churn_rates.append(_bits_per_weight(churn, eligible))

    densities = [density.decision_relevant_exact_density for density in step_densities]
    return FrontCTimelineDensitySummary(
        schema_version=FRONT_C_SCHEMA_VERSION,
        label=FRONT_C_LABEL,
        step_count=len(ordered),
        eligible_weight_count=eligible,
        step_densities=step_densities,
        max_decision_relevant_exact_density=max(densities),
        p95_decision_relevant_exact_density=_p95(densities),
        union_decision_relevant_exact_count=len(union_decision),
        union_decision_relevant_exact_density=_bits_per_weight(len(union_decision), eligible),
        union_decision_relevant_exact_indices_sha256=_identity_sha256(tuple(union_decision)),
        union_current_magnitude_threshold_count=len(union_by_surface["current_magnitude_threshold_keys"]),
        union_active_next_step_count=len(union_by_surface["active_next_step_keys"]),
        union_ranking_sensitive_exact_count=len(union_by_surface["ranking_sensitive_exact_keys"]),
        union_global_cap_frontier_count=len(union_by_surface["global_cap_frontier_keys"]),
        union_backlog_carry_count=len(union_by_surface["backlog_carry_keys"]),
        union_replay_veto_residual_count=len(union_by_surface["replay_veto_residual_keys"]),
        transition_count=max(0, len(ordered) - 1),
        total_entry_count=total_entry,
        total_exit_count=total_exit,
        max_entry_rate=max(entry_rates) if entry_rates else 0.0,
        p95_entry_rate=_p95(entry_rates),
        max_exit_rate=max(exit_rates) if exit_rates else 0.0,
        p95_exit_rate=_p95(exit_rates),
        max_churn_rate=max(churn_rates) if churn_rates else 0.0,
        p95_churn_rate=_p95(churn_rates),
        mean_churn_rate=_mean(churn_rates),
        raw_arrays_included=False,
        non_claims=(
            "timeline density scaffold only; no sparse accumulator encoder",
            "compact identities/counts/hashes only; no raw per-weight tensors",
            FRONT_C_NO_VIABILITY_CLAIM,
        ),
    )


@dataclass(frozen=True)
class FrontCQPhysicalGateReport:
    q_regime_name: str
    q_format: str
    base3_fixed_payload_bits_per_weight: float
    q_packed_data_bits_per_weight: float
    q_packed_total_bits_per_weight: float
    q_effective_entropy_floor_bits_per_weight: float
    base3_validator_passed: bool
    physical_base3_bounds_passed: bool
    gate_valid: bool
    diagnostic_only: bool
    rejection_reason: str

    def to_dict(self) -> dict[str, int | float | bool | str]:
        return asdict(self)


def front_c_physical_base3_q_gate_report(row: Base3QEntropyLedgerRow) -> FrontCQPhysicalGateReport:
    """Fail closed unless the q row proves physical base3 byte bounds."""

    validator_passed = True
    rejection = ""
    try:
        validate_base3_q_entropy_ledger(row)
    except ValueError as exc:
        validator_passed = False
        rejection = f"base3 q ledger validation failed: {exc}"

    format_ok = row.format == BASE3_Q_FORMAT
    fixed_ok = math.isclose(
        float(row.base3_fixed_payload_bits_per_weight),
        BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT,
        abs_tol=1e-12,
    )
    data_floor_ok = (
        float(row.q_packed_data_bits_per_weight)
        >= BASE3_FIXED_PAYLOAD_BITS_PER_WEIGHT - 1e-12
    )
    physical_bounds = bool(format_ok and fixed_ok and data_floor_ok)
    if validator_passed and not format_ok:
        rejection = "q ledger format is not BASE3_Q_FORMAT"
    elif validator_passed and not fixed_ok:
        rejection = "q ledger fixed payload basis is not physical base3 8/5"
    elif validator_passed and not data_floor_ok:
        rejection = (
            "q_packed_data_bits_per_weight is below physical base3 8/5; "
            "entropy-floor log2(3) is diagnostic-only and cannot gate"
        )

    gate_valid = bool(validator_passed and physical_bounds)
    return FrontCQPhysicalGateReport(
        q_regime_name=str(row.regime_name),
        q_format=str(row.format),
        base3_fixed_payload_bits_per_weight=float(row.base3_fixed_payload_bits_per_weight),
        q_packed_data_bits_per_weight=float(row.q_packed_data_bits_per_weight),
        q_packed_total_bits_per_weight=float(row.q_packed_total_bits_per_weight),
        q_effective_entropy_floor_bits_per_weight=BASE3_EFFECTIVE_TERNARY_ENTROPY_BITS_PER_WEIGHT,
        base3_validator_passed=validator_passed,
        physical_base3_bounds_passed=physical_bounds,
        gate_valid=gate_valid,
        diagnostic_only=not gate_valid,
        rejection_reason="" if gate_valid else rejection,
    )


def require_front_c_physical_base3_q_gate(row: Base3QEntropyLedgerRow) -> FrontCQPhysicalGateReport:
    report = front_c_physical_base3_q_gate_report(row)
    if not report.gate_valid:
        raise ValueError(report.rejection_reason)
    return report


def _normalize_direction_map(values: Any, *, name: str) -> dict[FrontCIdentity, int]:
    if values is None:
        return {}
    out: dict[FrontCIdentity, int] = {}
    items: Sequence[Any]
    if isinstance(values, Mapping):
        items = list(values.items())
        for raw_identity, raw_direction in items:
            identity = _normalize_identity(raw_identity, name=name)
            direction = int(raw_direction)
            if direction not in (-1, 1):
                raise ValueError(f"{name} directions must be -1 or +1")
            out[identity] = direction
        return dict(sorted(out.items()))
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must not be a string")
    items = list(values)
    for item in items:
        if isinstance(item, Mapping):
            direction = int(item["direction"])
            identity = _normalize_identity(item, name=name)
        elif isinstance(item, (list, tuple)) and len(item) == 3:
            identity = _normalize_identity((item[0], item[1]), name=name)
            direction = int(item[2])
        else:
            raise ValueError(f"{name} entries must carry state_key, flat_index, direction")
        if direction not in (-1, 1):
            raise ValueError(f"{name} directions must be -1 or +1")
        out[identity] = direction
    return dict(sorted(out.items()))


def _direction_dicts(values: Mapping[FrontCIdentity, int]) -> list[dict[str, int | str]]:
    return [
        {"state_key": state_key, "flat_index": int(flat_index), "direction": int(direction)}
        for (state_key, flat_index), direction in sorted(values.items())
    ]


def _direction_sha256(values: Mapping[FrontCIdentity, int]) -> str:
    h = hashlib.sha256()
    for (state_key, flat_index), direction in sorted(values.items()):
        h.update(state_key.encode("utf-8"))
        h.update(b":")
        h.update(str(int(flat_index)).encode("utf-8"))
        h.update(b":")
        h.update(str(int(direction)).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass(frozen=True)
class FrontCDecisionPath:
    """Compact decision surface from either the dense oracle or sparse projection."""

    label: str
    q_flip_directions: Any = ()
    accepted_under_global_cap_keys: Sequence[Any] = ()
    deferred_under_global_cap_keys: Sequence[Any] = ()
    backlog_keys: Sequence[Any] = ()
    replay_veto_decision_keys: Sequence[Any] = ()

    def __post_init__(self) -> None:
        if not str(self.label):
            raise ValueError("decision path label must be non-empty")
        object.__setattr__(self, "label", str(self.label))
        object.__setattr__(
            self,
            "q_flip_directions",
            _normalize_direction_map(self.q_flip_directions, name="q_flip_directions"),
        )
        for field_name in (
            "accepted_under_global_cap_keys",
            "deferred_under_global_cap_keys",
            "backlog_keys",
            "replay_veto_decision_keys",
        ):
            object.__setattr__(
                self,
                field_name,
                _normalize_identity_tuple(getattr(self, field_name), name=field_name),
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "q_flip_directions": _direction_dicts(self.q_flip_directions),
            "accepted_under_global_cap_keys": _identity_dicts(self.accepted_under_global_cap_keys),
            "deferred_under_global_cap_keys": _identity_dicts(self.deferred_under_global_cap_keys),
            "backlog_keys": _identity_dicts(self.backlog_keys),
            "replay_veto_decision_keys": _identity_dicts(self.replay_veto_decision_keys),
        }


def normalize_front_c_decision_path(value: FrontCDecisionPath | Mapping[str, Any]) -> FrontCDecisionPath:
    if isinstance(value, FrontCDecisionPath):
        return value
    if not isinstance(value, Mapping):
        raise ValueError("decision path must be a mapping or FrontCDecisionPath")
    return FrontCDecisionPath(
        label=str(value["label"]),
        q_flip_directions=value.get("q_flip_directions", ()),
        accepted_under_global_cap_keys=value.get("accepted_under_global_cap_keys", ()),
        deferred_under_global_cap_keys=value.get("deferred_under_global_cap_keys", ()),
        backlog_keys=value.get("backlog_keys", ()),
        replay_veto_decision_keys=value.get("replay_veto_decision_keys", ()),
    )


@dataclass(frozen=True)
class FrontCDecisionEquivalenceReport:
    schema_version: str
    label: str
    dense_path_label: str
    sparse_path_label: str
    q_flip_identity_changed_count: int
    q_flip_direction_changed_count: int
    accepted_under_global_cap_changed_count: int
    deferred_under_global_cap_changed_count: int
    backlog_key_changed_count: int
    replay_veto_decision_changed_count: int
    dense_q_flip_directions_sha256: str
    sparse_q_flip_directions_sha256: str
    dense_accepted_under_global_cap_sha256: str
    sparse_accepted_under_global_cap_sha256: str
    dense_deferred_under_global_cap_sha256: str
    sparse_deferred_under_global_cap_sha256: str
    dense_backlog_keys_sha256: str
    sparse_backlog_keys_sha256: str
    dense_replay_veto_decisions_sha256: str
    sparse_replay_veto_decisions_sha256: str
    zero_drift: bool
    failed_surfaces: tuple[str, ...]
    raw_arrays_included: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_surfaces"] = list(self.failed_surfaces)
        return payload


def compare_front_c_decision_equivalence(
    dense_path: FrontCDecisionPath | Mapping[str, Any],
    sparse_path: FrontCDecisionPath | Mapping[str, Any],
) -> FrontCDecisionEquivalenceReport:
    """Compare the sparse path against the dense int16 oracle on locked surfaces."""

    dense = normalize_front_c_decision_path(dense_path)
    sparse = normalize_front_c_decision_path(sparse_path)
    dense_q_keys = set(dense.q_flip_directions)
    sparse_q_keys = set(sparse.q_flip_directions)
    q_identity_changed = len(dense_q_keys ^ sparse_q_keys)
    common_q = dense_q_keys & sparse_q_keys
    q_direction_changed = sum(
        1
        for identity in common_q
        if int(dense.q_flip_directions[identity]) != int(sparse.q_flip_directions[identity])
    )
    accepted_changed = len(
        set(dense.accepted_under_global_cap_keys) ^ set(sparse.accepted_under_global_cap_keys)
    )
    deferred_changed = len(
        set(dense.deferred_under_global_cap_keys) ^ set(sparse.deferred_under_global_cap_keys)
    )
    backlog_changed = len(set(dense.backlog_keys) ^ set(sparse.backlog_keys))
    replay_changed = len(
        set(dense.replay_veto_decision_keys) ^ set(sparse.replay_veto_decision_keys)
    )

    failed: list[str] = []
    if q_identity_changed:
        failed.append("q_flip_identity")
    if q_direction_changed:
        failed.append("q_flip_direction")
    if accepted_changed:
        failed.append("accepted_under_global_cap")
    if deferred_changed:
        failed.append("deferred_under_global_cap")
    if backlog_changed:
        failed.append("backlog_keys")
    if replay_changed:
        failed.append("replay_veto_decisions")

    return FrontCDecisionEquivalenceReport(
        schema_version=FRONT_C_SCHEMA_VERSION,
        label=FRONT_C_LABEL,
        dense_path_label=dense.label,
        sparse_path_label=sparse.label,
        q_flip_identity_changed_count=q_identity_changed,
        q_flip_direction_changed_count=q_direction_changed,
        accepted_under_global_cap_changed_count=accepted_changed,
        deferred_under_global_cap_changed_count=deferred_changed,
        backlog_key_changed_count=backlog_changed,
        replay_veto_decision_changed_count=replay_changed,
        dense_q_flip_directions_sha256=_direction_sha256(dense.q_flip_directions),
        sparse_q_flip_directions_sha256=_direction_sha256(sparse.q_flip_directions),
        dense_accepted_under_global_cap_sha256=_identity_sha256(
            dense.accepted_under_global_cap_keys,
        ),
        sparse_accepted_under_global_cap_sha256=_identity_sha256(
            sparse.accepted_under_global_cap_keys,
        ),
        dense_deferred_under_global_cap_sha256=_identity_sha256(
            dense.deferred_under_global_cap_keys,
        ),
        sparse_deferred_under_global_cap_sha256=_identity_sha256(
            sparse.deferred_under_global_cap_keys,
        ),
        dense_backlog_keys_sha256=_identity_sha256(dense.backlog_keys),
        sparse_backlog_keys_sha256=_identity_sha256(sparse.backlog_keys),
        dense_replay_veto_decisions_sha256=_identity_sha256(dense.replay_veto_decision_keys),
        sparse_replay_veto_decisions_sha256=_identity_sha256(sparse.replay_veto_decision_keys),
        zero_drift=not failed,
        failed_surfaces=tuple(failed),
        raw_arrays_included=False,
    )


@dataclass(frozen=True)
class FrontCProjectionReport:
    schema_version: str
    label: str
    q_physical_gate: FrontCQPhysicalGateReport
    timeline_density: FrontCTimelineDensitySummary
    storage_projection: BoundedDeltaStorageProjection
    bounded_delta_inclusive_ledger: BoundedDeltaInclusiveLedger
    decision_equivalence: FrontCDecisionEquivalenceReport
    payload_only_bits_per_weight: float
    payload_only_would_fit_remaining_budget: bool
    payload_only_gate_used: bool
    overhead_inclusive_gate_used: bool
    overhead_inclusive_projected_bpw: float
    decision_guard_passed: bool
    final_gate_passed: bool
    gate_basis_statement: str
    raw_arrays_included: bool
    non_claims: tuple[str, ...]

    @property
    def claimable_physical_sub2_with_decision_guard(self) -> bool:
        return bool(self.final_gate_passed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "label": self.label,
            "q_physical_gate": self.q_physical_gate.to_dict(),
            "timeline_density": self.timeline_density.to_dict(),
            "storage_projection": self.storage_projection.to_dict(),
            "bounded_delta_inclusive_ledger": self.bounded_delta_inclusive_ledger.to_dict(),
            "decision_equivalence": self.decision_equivalence.to_dict(),
            "payload_only_bits_per_weight": float(self.payload_only_bits_per_weight),
            "payload_only_would_fit_remaining_budget": bool(
                self.payload_only_would_fit_remaining_budget
            ),
            "payload_only_gate_used": bool(self.payload_only_gate_used),
            "overhead_inclusive_gate_used": bool(self.overhead_inclusive_gate_used),
            "overhead_inclusive_projected_bpw": float(self.overhead_inclusive_projected_bpw),
            "decision_guard_passed": bool(self.decision_guard_passed),
            "final_gate_passed": bool(self.final_gate_passed),
            "claimable_physical_sub2_with_decision_guard": (
                self.claimable_physical_sub2_with_decision_guard
            ),
            "gate_basis_statement": self.gate_basis_statement,
            "raw_arrays_included": bool(self.raw_arrays_included),
            "non_claims": list(self.non_claims),
        }


def build_front_c_projection_report(
    *,
    timeline_steps: Sequence[FrontCDecisionSurfaceStep | Mapping[str, Any]],
    q_ledger_row: Base3QEntropyLedgerRow,
    dense_decision_path: FrontCDecisionPath | Mapping[str, Any],
    sparse_decision_path: FrontCDecisionPath | Mapping[str, Any],
    value_bits_per_row: int = 16,
    flag_bits_per_row: int = 2,
    tensor_metadata_bits: int = 0,
    bucket_metadata_bits: int = 0,
    scale_metadata_bits: int = 0,
    guardrail_metadata_bits: int = 0,
    event_delta_count: int = 0,
) -> FrontCProjectionReport:
    """Build the locked Front-C ledger, using only overhead-inclusive gates."""

    q_gate = front_c_physical_base3_q_gate_report(q_ledger_row)
    timeline = measure_front_c_timeline_density(timeline_steps)
    decision = compare_front_c_decision_equivalence(dense_decision_path, sparse_decision_path)
    payload_only_bits = int(timeline.union_decision_relevant_exact_count) * int(value_bits_per_row)
    payload_only_bpw = _bits_per_weight(payload_only_bits, timeline.eligible_weight_count)
    projection = project_bounded_delta_accumulator_bpw(
        eligible_weight_count=timeline.eligible_weight_count,
        hot_exact_row_count=timeline.union_decision_relevant_exact_count,
        event_delta_count=event_delta_count,
        backlog_entry_count=timeline.union_backlog_carry_count,
        hot_value_bits_per_row=value_bits_per_row,
        hot_flag_bits_per_row=flag_bits_per_row,
        tensor_metadata_bits=tensor_metadata_bits,
        bucket_metadata_bits=bucket_metadata_bits,
        scale_metadata_bits=scale_metadata_bits,
        guardrail_metadata_bits=guardrail_metadata_bits,
    )
    ledger = bounded_delta_inclusive_ledger(q_ledger_row, projection)
    validate_bounded_delta_inclusive_ledger(ledger)
    payload_only_fits = payload_only_bpw <= float(ledger.remaining_accumulator_budget_bits_per_weight)
    final_gate = bool(q_gate.gate_valid and ledger.claimable_physical_sub2 and decision.zero_drift)
    return FrontCProjectionReport(
        schema_version=FRONT_C_SCHEMA_VERSION,
        label=FRONT_C_LABEL,
        q_physical_gate=q_gate,
        timeline_density=timeline,
        storage_projection=projection,
        bounded_delta_inclusive_ledger=ledger,
        decision_equivalence=decision,
        payload_only_bits_per_weight=payload_only_bpw,
        payload_only_would_fit_remaining_budget=bool(payload_only_fits),
        payload_only_gate_used=False,
        overhead_inclusive_gate_used=True,
        overhead_inclusive_projected_bpw=float(projection.bounded_delta_acc_bits_per_weight),
        decision_guard_passed=bool(decision.zero_drift),
        final_gate_passed=final_gate,
        gate_basis_statement=FRONT_C_PAYLOAD_ONLY_GATING_PHRASE,
        raw_arrays_included=False,
        non_claims=(
            FRONT_C_NO_VIABILITY_CLAIM,
            "payload-only density is intuition-only and is never a gate",
            "q entropy floor log2(3) is diagnostic-only and is never a gate",
            "CPU/static scaffold only; no GPU, .pt, live learner, or sparse-acc hot loop",
        ),
    )


def validate_front_c_projection_report(
    report: FrontCProjectionReport,
    *,
    claimed_front_c_viable: bool = False,
) -> None:
    if report.raw_arrays_included:
        raise ValueError("Front-C reports must not include raw per-weight arrays")
    if report.payload_only_gate_used:
        raise ValueError("payload-only density must never be used as a Front-C gate")
    if not report.overhead_inclusive_gate_used:
        raise ValueError("Front-C must gate on overhead-inclusive projected bpw")
    recomputed = bool(
        report.q_physical_gate.gate_valid
        and report.bounded_delta_inclusive_ledger.claimable_physical_sub2
        and report.decision_equivalence.zero_drift
    )
    if bool(report.final_gate_passed) != recomputed:
        raise ValueError("Front-C final gate must be q physical + ledger + zero-drift conjunctive")
    if report.gate_basis_statement != FRONT_C_PAYLOAD_ONLY_GATING_PHRASE:
        raise ValueError("Front-C gate basis statement must preserve the locked prereg phrase")
    if claimed_front_c_viable and not report.final_gate_passed:
        raise ValueError("Front-C viability cannot be claimed without ledger + zero-drift evidence")
