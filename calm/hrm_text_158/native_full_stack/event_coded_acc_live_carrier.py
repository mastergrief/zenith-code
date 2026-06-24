"""V4-LIVE event-coded accumulator carrier (CPU synthetic / standalone)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    carry_self_update_row,
    crossing_bool_w6,
    encode_post_flip_residual,
)

DEFAULT_WATCH_BAND = 3
DEFAULT_COLD_DEFAULT = 0
DEFAULT_DECAY_NUMERATOR = 1
DEFAULT_DECAY_DENOMINATOR = 1
DEFAULT_VERDICT_NUMEL = 1024


def promotion_carry_threshold(*, threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS) -> int:
    return int(threshold_abs) - int(DEFAULT_WATCH_BAND)


def hot_risk_proxy_indices(
    carries: Mapping[int, int],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> set[int]:
    """Numeric near-threshold proxy for synthetic CPU sweeps (not regret/tie-band)."""

    promote_at = promotion_carry_threshold(threshold_abs=threshold_abs)
    return {int(index) for index, carry in carries.items() if abs(int(carry)) >= promote_at}


@dataclass
class StepSurfaceRecord:
    step_index: int
    crossing_indices: tuple[int, ...]
    applied_indices: tuple[int, ...]
    backlog_indices: tuple[int, ...]
    q_levels: dict[int, int]
    hot_exact_row_count: int
    promotion_count: int
    demotion_on_decay_count: int
    demotion_on_crossing_count: int


@dataclass
class EventCodedAccLiveState:
    logical_numel: int
    cold_default: int = DEFAULT_COLD_DEFAULT
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS
    demotion_band: int = 3
    hot_exact: dict[int, int] = field(default_factory=dict)
    events: list[EventCodedAccEvent] = field(default_factory=list)
    backlog: set[int] = field(default_factory=set)
    q_levels: dict[int, int] = field(default_factory=dict)
    step_records: list[StepSurfaceRecord] = field(default_factory=list)
    dense_accumulator_materialized_numel: int = 0

    def reconstruct_lane(self, flat_index: int) -> int:
        return int(self.hot_exact.get(int(flat_index), int(self.cold_default)))

    def q_level(self, flat_index: int) -> int:
        return int(self.q_levels.get(int(flat_index), 0))

    def _should_promote(
        self,
        flat_index: int,
        *,
        post_carry: int,
        vote_touched: bool,
        hot_risk_proxy: set[int],
    ) -> bool:
        if int(flat_index) in self.hot_exact:
            return True
        if vote_touched:
            return True
        if int(flat_index) in hot_risk_proxy:
            return True
        promote_at = promotion_carry_threshold(threshold_abs=self.threshold_abs)
        return abs(int(post_carry)) >= int(promote_at)

    def _maybe_demote(
        self,
        flat_index: int,
        *,
        post_carry: int,
        fired_crossing: bool,
        hot_risk_proxy: set[int],
        vote_touched: bool,
    ) -> bool:
        if int(flat_index) not in self.hot_exact:
            return False
        if fired_crossing:
            self.hot_exact.pop(int(flat_index), None)
            return True
        if vote_touched:
            return False
        if int(flat_index) in hot_risk_proxy:
            return False
        if int(flat_index) in self.backlog:
            return False
        if abs(int(post_carry)) >= int(self.demotion_band):
            return False
        self.hot_exact.pop(int(flat_index), None)
        return True

    def apply_step(
        self,
        step_index: int,
        *,
        votes: Mapping[int, int],
        hot_risk_override: Iterable[int] | None = None,
    ) -> StepSurfaceRecord:
        vote_map = {int(k): int(v) for k, v in votes.items()}
        touched = set(vote_map)
        pre_carries = {index: self.reconstruct_lane(index) for index in self.hot_exact}
        pre_carries.update({index: self.reconstruct_lane(index) for index in touched})
        proxy = (
            set(int(item) for item in hot_risk_override)
            if hot_risk_override is not None
            else hot_risk_proxy_indices(pre_carries, threshold_abs=self.threshold_abs)
        )
        active = set(self.hot_exact) | touched | proxy
        post_carries: dict[int, int] = {}
        promotion_count = 0
        for flat_index in active:
            pre = self.reconstruct_lane(flat_index)
            vote = int(vote_map.get(flat_index, 0))
            post = carry_self_update_row(
                pre,
                vote,
                decay_numerator=DEFAULT_DECAY_NUMERATOR,
                decay_denominator=DEFAULT_DECAY_DENOMINATOR,
            )
            post_carries[int(flat_index)] = int(post)
            if self._should_promote(
                flat_index,
                post_carry=post,
                vote_touched=flat_index in touched,
                hot_risk_proxy=proxy,
            ):
                if flat_index not in self.hot_exact:
                    promotion_count += 1
                self.hot_exact[int(flat_index)] = int(post)

        crossing_indices: list[int] = []
        applied_indices: list[int] = []
        demotion_on_crossing = 0
        demotion_on_decay = 0
        for flat_index, post in sorted(post_carries.items()):
            q = self.q_level(flat_index)
            fired = crossing_bool_w6(post, q, threshold_abs=self.threshold_abs)
            if fired:
                crossing_indices.append(int(flat_index))
                direction = 1 if int(post) >= 0 else 0
                residual_mag = min(abs(int(post)), int(self.threshold_abs) - 1)
                self.events.append(
                    EventCodedAccEvent(
                        flat_index=int(flat_index),
                        direction=int(direction),
                        residual_mag=int(residual_mag),
                        event_type=1,
                    )
                )
                encode_post_flip_residual(
                    1 if direction else -1,
                    residual_mag,
                    threshold_abs=self.threshold_abs,
                )
                new_q = 1 if int(post) >= 0 else -1
                self.q_levels[int(flat_index)] = int(new_q)
                applied_indices.append(int(flat_index))
                if self._maybe_demote(
                    flat_index,
                    post_carry=post,
                    fired_crossing=True,
                    hot_risk_proxy=proxy,
                    vote_touched=flat_index in touched,
                ):
                    demotion_on_crossing += 1
            elif self._maybe_demote(
                flat_index,
                post_carry=post,
                fired_crossing=False,
                hot_risk_proxy=proxy,
                vote_touched=flat_index in touched,
            ):
                demotion_on_decay += 1
            elif flat_index in self.hot_exact:
                self.hot_exact[int(flat_index)] = int(post)

        record = StepSurfaceRecord(
            step_index=int(step_index),
            crossing_indices=tuple(crossing_indices),
            applied_indices=tuple(applied_indices),
            backlog_indices=tuple(sorted(self.backlog)),
            q_levels=dict(self.q_levels),
            hot_exact_row_count=len(self.hot_exact),
            promotion_count=int(promotion_count),
            demotion_on_decay_count=int(demotion_on_decay),
            demotion_on_crossing_count=int(demotion_on_crossing),
        )
        self.step_records.append(record)
        return record

    def to_checkpoint_payload(self):
        hot_indices = tuple(sorted(self.hot_exact))
        hot_values = tuple(int(self.hot_exact[index]) for index in hot_indices)
        return pack_event_coded_acc_checkpoint_v1(
            logical_numel=int(self.logical_numel),
            events=tuple(self.events),
            backlog_indices=tuple(sorted(self.backlog)),
            hot_exact_indices=hot_indices,
            hot_exact_values=hot_values,
        )


@dataclass(frozen=True)
class DenseOracleState:
    logical_numel: int
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS
    cold_default: int = DEFAULT_COLD_DEFAULT
    accumulators: list[int] = field(default_factory=list)
    q_levels: list[int] = field(default_factory=list)
    step_records: list[StepSurfaceRecord] = field(default_factory=list)

    @classmethod
    def zeros(cls, logical_numel: int) -> DenseOracleState:
        return cls(
            logical_numel=int(logical_numel),
            accumulators=[0] * int(logical_numel),
            q_levels=[0] * int(logical_numel),
        )

    def apply_step(self, step_index: int, *, votes: Mapping[int, int]) -> StepSurfaceRecord:
        vote_map = {int(k): int(v) for k, v in votes.items()}
        crossing_indices: list[int] = []
        applied_indices: list[int] = []
        for flat_index in range(int(self.logical_numel)):
            pre = int(self.accumulators[flat_index])
            vote = int(vote_map.get(flat_index, 0))
            post = carry_self_update_row(
                pre,
                vote,
                decay_numerator=DEFAULT_DECAY_NUMERATOR,
                decay_denominator=DEFAULT_DECAY_DENOMINATOR,
            )
            self.accumulators[flat_index] = int(post)
            q = int(self.q_levels[flat_index])
            if crossing_bool_w6(post, q, threshold_abs=self.threshold_abs):
                crossing_indices.append(int(flat_index))
                self.q_levels[flat_index] = 1 if int(post) >= 0 else -1
                applied_indices.append(int(flat_index))
        record = StepSurfaceRecord(
            step_index=int(step_index),
            crossing_indices=tuple(crossing_indices),
            applied_indices=tuple(applied_indices),
            backlog_indices=(),
            q_levels={index: int(value) for index, value in enumerate(self.q_levels)},
            hot_exact_row_count=int(self.logical_numel),
            promotion_count=0,
            demotion_on_decay_count=0,
            demotion_on_crossing_count=0,
        )
        self.step_records.append(record)
        return record


def _decisive_q_snapshot(record: StepSurfaceRecord) -> dict[int, int]:
    """Compare q only on decisive lanes (crossing/applied), not full-tensor oracle keys."""

    indices = set(record.applied_indices) | set(record.crossing_indices)
    return {int(index): int(record.q_levels.get(int(index), 0)) for index in sorted(indices)}


def observed_surfaces_dict(record: StepSurfaceRecord) -> dict[str, Any]:
    return {
        "crossing_flat_indices": [int(value) for value in record.crossing_indices],
        "applied_flat_indices": [int(value) for value in record.applied_indices],
        "decisive_q_snapshot": {
            str(key): int(value) for key, value in _decisive_q_snapshot(record).items()
        },
    }


def decisive_surface_drift_count(
    carrier_records: Sequence[StepSurfaceRecord],
    oracle_records: Sequence[StepSurfaceRecord],
) -> int:
    drift = 0
    for carrier, oracle in zip(carrier_records, oracle_records):
        if carrier.crossing_indices != oracle.crossing_indices:
            drift += 1
        if carrier.applied_indices != oracle.applied_indices:
            drift += 1
        if _decisive_q_snapshot(carrier) != _decisive_q_snapshot(oracle):
            drift += 1
    return int(drift)


def decisive_surface_drift_details(
    carrier_records: Sequence[StepSurfaceRecord],
    oracle_records: Sequence[StepSurfaceRecord],
) -> list[dict[str, object]]:
    """Per-step mismatch dump for harness root-cause classification."""

    details: list[dict[str, object]] = []
    for carrier, oracle in zip(carrier_records, oracle_records):
        mismatches: list[str] = []
        if carrier.crossing_indices != oracle.crossing_indices:
            mismatches.append("crossing_indices")
        if carrier.applied_indices != oracle.applied_indices:
            mismatches.append("applied_indices")
        if _decisive_q_snapshot(carrier) != _decisive_q_snapshot(oracle):
            mismatches.append("decisive_q_snapshot")
        if mismatches:
            details.append(
                {
                    "step_index": int(carrier.step_index),
                    "mismatch_fields": tuple(mismatches),
                    "carrier_crossing": tuple(carrier.crossing_indices),
                    "oracle_crossing": tuple(oracle.crossing_indices),
                    "carrier_applied": tuple(carrier.applied_indices),
                    "oracle_applied": tuple(oracle.applied_indices),
                    "carrier_q": _decisive_q_snapshot(carrier),
                    "oracle_q": _decisive_q_snapshot(oracle),
                }
            )
    return details
