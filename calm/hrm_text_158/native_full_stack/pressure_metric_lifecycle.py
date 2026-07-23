"""Mutable identity-tracking / deferred-event lifecycle store (PLAN_v6).

Owns: DurableAggregates + PressureTelemetryStore (CPU int32x3 identity state).
Internal-only — never emit raw per-weight arrays in receipts.
Dependency: lifecycle → pressure_metric_telemetry (constants/helpers only via
shared threshold); actually depends only on two_tier + local logic.
Bound by PLAN_v6 sha 346b67d8…; extracted under rev3 re-scope 1784828063166.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS,
)

# Local copies of prereg constants used by survival summary (avoid cycles).
FOLLOW_UP_HORIZON_STEPS = 32
MIN_COHORT_N = 100
GROWING_DEFERRED_SURVIVAL_DELTA = 0.10
STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR = 0.50


@dataclass
class DurableAggregates:
    N_events_evaluable: int = 0
    N_events_censored_insufficient_followup: int = 0
    N_survived_applied_within_H: int = 0
    N_never_applied_within_H: int = 0
    N_events_evaluable_early: int = 0
    N_events_evaluable_late: int = 0
    N_never_applied_within_H_early: int = 0
    N_never_applied_within_H_late: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "N_events_evaluable": self.N_events_evaluable,
            "N_events_censored_insufficient_followup": self.N_events_censored_insufficient_followup,
            "N_survived_applied_within_H": self.N_survived_applied_within_H,
            "N_never_applied_within_H": self.N_never_applied_within_H,
            "N_events_evaluable_early": self.N_events_evaluable_early,
            "N_events_evaluable_late": self.N_events_evaluable_late,
            "N_never_applied_within_H_early": self.N_never_applied_within_H_early,
            "N_never_applied_within_H_late": self.N_never_applied_within_H_late,
        }


@dataclass
class PressureTelemetryStore:
    """Process-local identity trackers + durable aggregates (CPU)."""

    steps: int
    follow_up_horizon: int = FOLLOW_UP_HORIZON_STEPS
    threshold: int = CROSSING_THRESHOLD_ABS
    first_deferral_step: dict[str, torch.Tensor] = field(default_factory=dict)
    applied_after_deferral_step: dict[str, torch.Tensor] = field(default_factory=dict)
    episode_generation: dict[str, torch.Tensor] = field(default_factory=dict)
    aggregates: DurableAggregates = field(default_factory=DurableAggregates)
    per_step_ratios: list[dict[str, Any]] = field(default_factory=list)
    two_tier_threshold_assert_pass: bool = True

    @classmethod
    def from_q_levels(
        cls,
        q_levels: Mapping[str, torch.Tensor],
        *,
        steps: int,
        follow_up_horizon: int = FOLLOW_UP_HORIZON_STEPS,
    ) -> "PressureTelemetryStore":
        store = cls(steps=int(steps), follow_up_horizon=int(follow_up_horizon))
        for n, q in q_levels.items():
            z = torch.zeros_like(q, dtype=torch.int32)
            store.first_deferral_step[n] = z.clone()
            store.applied_after_deferral_step[n] = z.clone()
            store.episode_generation[n] = z.clone()
        return store

    def _cohort_bucket(self, first_step: int) -> str | None:
        steps = int(self.steps)
        H = int(self.follow_up_horizon)
        if first_step > steps - H:
            return None
        mid = steps // 2
        if 1 <= first_step <= mid:
            return "early"
        if mid < first_step <= steps - H:
            return "late"
        return None

    def _close_event(
        self,
        *,
        first_step: int,
        applied_after: int,
        now_step: int,
        reason: str,
    ) -> None:
        steps = int(self.steps)
        H = int(self.follow_up_horizon)
        if first_step > steps - H:
            self.aggregates.N_events_censored_insufficient_followup += 1
            return
        if reason == "window_end" and applied_after == 0 and (now_step - first_step) < H:
            if now_step >= steps and (first_step + H) > steps:
                self.aggregates.N_events_censored_insufficient_followup += 1
                return

        self.aggregates.N_events_evaluable += 1
        bucket = self._cohort_bucket(first_step)
        if bucket == "early":
            self.aggregates.N_events_evaluable_early += 1
        elif bucket == "late":
            self.aggregates.N_events_evaluable_late += 1

        survived = applied_after > 0 and 0 < (applied_after - first_step) <= H
        if survived:
            self.aggregates.N_survived_applied_within_H += 1
        else:
            if applied_after == 0 and (now_step - first_step) >= H:
                self.aggregates.N_never_applied_within_H += 1
                if bucket == "early":
                    self.aggregates.N_never_applied_within_H_early += 1
                elif bucket == "late":
                    self.aggregates.N_never_applied_within_H_late += 1
            elif applied_after > 0 and (applied_after - first_step) > H:
                self.aggregates.N_never_applied_within_H += 1
                if bucket == "early":
                    self.aggregates.N_never_applied_within_H_early += 1
                elif bucket == "late":
                    self.aggregates.N_never_applied_within_H_late += 1
            elif reason in ("horizon_expired", "residual_clear", "residual_restart") and applied_after == 0:
                if (now_step - first_step) >= H:
                    self.aggregates.N_never_applied_within_H += 1
                    if bucket == "early":
                        self.aggregates.N_never_applied_within_H_early += 1
                    elif bucket == "late":
                        self.aggregates.N_never_applied_within_H_late += 1
                else:
                    self.aggregates.N_events_evaluable -= 1
                    if bucket == "early":
                        self.aggregates.N_events_evaluable_early -= 1
                    elif bucket == "late":
                        self.aggregates.N_events_evaluable_late -= 1
                    self.aggregates.N_events_censored_insufficient_followup += 1

    def process_pre_writeback(
        self,
        *,
        candidate_masks: Mapping[str, torch.Tensor],
        applied_masks: Mapping[str, torch.Tensor],
        step: int,
        n_candidates: int,
        n_applied: int,
    ) -> None:
        t = int(step)
        H = int(self.follow_up_horizon)
        for n, cand in candidate_masks.items():
            applied = applied_masks[n]
            deferred = cand & ~applied
            first = self.first_deferral_step[n]
            new_def = deferred & (first == 0)
            if bool(new_def.any()):
                first[new_def] = t

        for n, applied in applied_masks.items():
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            hit = applied & (first > 0) & (after == 0) & (first < t)
            if bool(hit.any()):
                after[hit] = t

        for n in list(self.first_deferral_step.keys()):
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            open_ev = first > 0
            if not bool(open_ev.any()):
                continue
            just_survived = open_ev & (after == t) & ((after - first) <= H) & ((after - first) > 0)
            if bool(just_survived.any()):
                idxs = just_survived.nonzero(as_tuple=False)
                for idx in idxs:
                    idx_t = tuple(int(x) for x in idx.tolist())
                    fs = int(first[idx_t].item())
                    aa = int(after[idx_t].item())
                    self._close_event(first_step=fs, applied_after=aa, now_step=t, reason="survived")
                    first[idx_t] = 0
                    after[idx_t] = 0

            expired = (first > 0) & (after == 0) & ((t - first) >= H)
            if bool(expired.any()):
                idxs = expired.nonzero(as_tuple=False)
                for idx in idxs:
                    idx_t = tuple(int(x) for x in idx.tolist())
                    fs = int(first[idx_t].item())
                    self._close_event(first_step=fs, applied_after=0, now_step=t, reason="horizon_expired")
                    first[idx_t] = 0
                    after[idx_t] = 0

        ratio = float(n_candidates) / float(max(1, n_applied))
        self.per_step_ratios.append(
            {
                "step": t,
                "candidate_crossers_before_cap": int(n_candidates),
                "applied_count": int(n_applied),
                "demand_applied_ratio": ratio,
                "deferred_count": max(0, int(n_candidates) - int(n_applied)),
            }
        )

    def close_before_writeback_resets(
        self,
        *,
        applied_masks: Mapping[str, torch.Tensor],
        step: int,
        residual_zero: Mapping[str, torch.Tensor] | None = None,
        residual_restart: Mapping[str, torch.Tensor] | None = None,
    ) -> None:
        t = int(step)
        for n, applied in applied_masks.items():
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            open_applied = applied & (first > 0)
            if not bool(open_applied.any()):
                continue
            idxs = open_applied.nonzero(as_tuple=False)
            for idx in idxs:
                idx_t = tuple(int(x) for x in idx.tolist())
                fs = int(first[idx_t].item())
                aa = int(after[idx_t].item())
                if fs == 0:
                    continue
                reason = "residual_restart"
                if residual_zero is not None and bool(residual_zero[n][idx_t]):
                    reason = "residual_clear"
                self._close_event(first_step=fs, applied_after=aa, now_step=t, reason=reason)
                first[idx_t] = 0
                after[idx_t] = 0

    def roll_tracker_after_writeback(
        self,
        *,
        applied_masks: Mapping[str, torch.Tensor],
        episode_start_before: Mapping[str, torch.Tensor],
        episode_start_after: Mapping[str, torch.Tensor],
        step: int,
    ) -> None:
        for n, applied in applied_masks.items():
            if not bool(applied.any()):
                continue
            before = episode_start_before[n]
            after = episode_start_after[n]
            changed = applied & (before != after)
            if not bool(changed.any()):
                continue
            self.episode_generation[n][changed] = self.episode_generation[n][changed] + 1
            self.first_deferral_step[n][changed] = 0
            self.applied_after_deferral_step[n][changed] = 0

    def finalize_window(self, *, final_step: int) -> None:
        t = int(final_step)
        for n in list(self.first_deferral_step.keys()):
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            open_ev = first > 0
            if not bool(open_ev.any()):
                continue
            idxs = open_ev.nonzero(as_tuple=False)
            for idx in idxs:
                idx_t = tuple(int(x) for x in idx.tolist())
                fs = int(first[idx_t].item())
                aa = int(after[idx_t].item())
                self._close_event(first_step=fs, applied_after=aa, now_step=t, reason="window_end")
                first[idx_t] = 0
                after[idx_t] = 0

    def survival_summary(self) -> dict[str, Any]:
        agg = self.aggregates
        n_eval = int(agg.N_events_evaluable)
        n_surv = int(agg.N_survived_applied_within_H)
        n_never = int(agg.N_never_applied_within_H)
        frac_surv = n_surv / max(1, n_eval)
        frac_never = n_never / max(1, n_eval)
        early_n = int(agg.N_events_evaluable_early)
        late_n = int(agg.N_events_evaluable_late)
        early_never_frac = (
            float(agg.N_never_applied_within_H_early) / float(early_n) if early_n else None
        )
        late_never_frac = (
            float(agg.N_never_applied_within_H_late) / float(late_n) if late_n else None
        )
        delta = None
        if early_never_frac is not None and late_never_frac is not None:
            delta = float(late_never_frac) - float(early_never_frac)

        klass = "other"
        if n_eval == 0:
            klass = "vacuous"
        elif early_n < MIN_COHORT_N or late_n < MIN_COHORT_N:
            klass = "other"
            if early_n == 0 and late_n == 0:
                klass = "vacuous"
        elif delta is not None:
            if delta >= GROWING_DEFERRED_SURVIVAL_DELTA:
                klass = "growing"
            elif (
                abs(delta) < GROWING_DEFERRED_SURVIVAL_DELTA
                and frac_never >= STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR
                and n_eval >= MIN_COHORT_N
            ):
                klass = "stable_high"
            elif delta <= -GROWING_DEFERRED_SURVIVAL_DELTA:
                klass = "collapsing"

        return {
            **agg.as_dict(),
            "deferred_survival_frac": float(frac_surv),
            "deferred_never_apply_within_H_frac": float(frac_never),
            "deferred_never_apply_within_H_frac_early": early_never_frac,
            "deferred_never_apply_within_H_frac_late": late_never_frac,
            "delta_never_apply": delta,
            "deferred_survival_class": klass,
        }
