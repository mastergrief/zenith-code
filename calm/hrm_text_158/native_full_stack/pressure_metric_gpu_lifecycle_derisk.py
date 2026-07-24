"""Device-resident pressure/metric lifecycle store (production-load path).

Tensor-resident trackers + int64 aggregate carrier with the full
`build_diagnostic_receipt` consumer-facing protocol (`per_step_ratios`,
`survival_summary`, `two_tier_threshold_assert_pass`, `steps`).
One-way imports only: geometry/production lifecycle oracles may be imported;
selection/writeback must NOT import this module's inverse (selection may import lifecycle).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_lifecycle import (
    FOLLOW_UP_HORIZON_STEPS,
    GROWING_DEFERRED_SURVIVAL_DELTA,
    MIN_COHORT_N,
    STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR,
    PressureTelemetryStore,
)

AGG_KEYS = (
    "N_events_evaluable",
    "N_events_censored_insufficient_followup",
    "N_survived_applied_within_H",
    "N_never_applied_within_H",
    "N_events_evaluable_early",
    "N_events_evaluable_late",
    "N_never_applied_within_H_early",
    "N_never_applied_within_H_late",
)


@dataclass
class DeviceLifecycleStore:
    """Tensor-resident lifecycle with full receipt-facing projection."""

    steps: int
    follow_up_horizon: int = FOLLOW_UP_HORIZON_STEPS
    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    first_deferral_step: dict[str, torch.Tensor] = field(default_factory=dict)
    applied_after_deferral_step: dict[str, torch.Tensor] = field(default_factory=dict)
    episode_generation: dict[str, torch.Tensor] = field(default_factory=dict)
    per_step_ratios: list[dict[str, Any]] = field(default_factory=list)
    two_tier_threshold_assert_pass: bool = True
    aggregates_t: torch.Tensor = field(init=False)
    hot_scalar_publishes: int = 0

    def __post_init__(self) -> None:
        self.aggregates_t = torch.zeros(8, dtype=torch.int64, device=self.device)

    @property
    def per_step_demand(self) -> list[dict[str, Any]]:
        """Internal alias only — receipt consumers must use `per_step_ratios`."""
        return self.per_step_ratios

    @classmethod
    def from_arm_shapes(
        cls,
        shapes: Mapping[str, tuple[int, ...]],
        *,
        steps: int,
        device: torch.device | str = "cpu",
        follow_up_horizon: int = FOLLOW_UP_HORIZON_STEPS,
    ) -> "DeviceLifecycleStore":
        dev = torch.device(device)
        store = cls(steps=int(steps), follow_up_horizon=int(follow_up_horizon), device=dev)
        for n, shape in shapes.items():
            z = torch.zeros(tuple(shape), dtype=torch.int32, device=dev)
            store.first_deferral_step[n] = z.clone()
            store.applied_after_deferral_step[n] = z.clone()
            store.episode_generation[n] = z.clone()
        return store

    def aggregates_as_dict(self) -> dict[str, int]:
        vals = self.aggregates_t.detach().cpu().tolist()
        return {k: int(v) for k, v in zip(AGG_KEYS, vals)}

    def _close_events_masked(
        self,
        *,
        first: torch.Tensor,
        after: torch.Tensor,
        close_mask: torch.Tensor,
        now_step: int,
        reason: str,
    ) -> None:
        steps = int(self.steps)
        H = int(self.follow_up_horizon)
        t = int(now_step)
        m = close_mask.bool()
        fs = first
        aa = after
        censor_insuff = m & (fs > (steps - H))
        rem = m & ~censor_insuff
        if reason == "window_end":
            outer_short = rem & (aa == 0) & ((t - fs) < H)
            censor_window = outer_short & (t >= steps) & ((fs + H) > steps)
            rem = rem & ~censor_window
        else:
            censor_window = torch.zeros_like(m)
        evaluable = rem
        mid = steps // 2
        early = evaluable & (fs >= 1) & (fs <= mid)
        late = evaluable & (fs > mid) & (fs <= (steps - H))
        survived = evaluable & (aa > 0) & ((aa - fs) > 0) & ((aa - fs) <= H)
        not_surv = evaluable & ~survived
        never_a = not_surv & (aa == 0) & ((t - fs) >= H)
        rem_ns = not_surv & ~never_a
        never_b = rem_ns & (aa > 0) & ((aa - fs) > H)
        rem_ns2 = rem_ns & ~never_b
        if reason in ("horizon_expired", "residual_clear", "residual_restart"):
            residual_arm = rem_ns2 & (aa == 0)
            never_c = residual_arm & ((t - fs) >= H)
            early_censor_dec = residual_arm & ~never_c
        else:
            never_c = torch.zeros_like(m)
            early_censor_dec = torch.zeros_like(m)
        never = never_a | never_b | never_c
        n_censor = (
            censor_insuff.to(torch.int64).sum()
            + censor_window.to(torch.int64).sum()
            + early_censor_dec.to(torch.int64).sum()
        )
        n_eval = evaluable.to(torch.int64).sum() - early_censor_dec.to(torch.int64).sum()
        n_eval_early = (
            early.to(torch.int64).sum() - (early_censor_dec & early).to(torch.int64).sum()
        )
        n_eval_late = (
            late.to(torch.int64).sum() - (early_censor_dec & late).to(torch.int64).sum()
        )
        n_surv = survived.to(torch.int64).sum()
        n_never = never.to(torch.int64).sum()
        n_never_early = (never & early).to(torch.int64).sum()
        n_never_late = (never & late).to(torch.int64).sum()
        self.aggregates_t[0] += n_eval
        self.aggregates_t[1] += n_censor
        self.aggregates_t[2] += n_surv
        self.aggregates_t[3] += n_never
        self.aggregates_t[4] += n_eval_early
        self.aggregates_t[5] += n_eval_late
        self.aggregates_t[6] += n_never_early
        self.aggregates_t[7] += n_never_late
        first[m] = 0
        after[m] = 0

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
            first[new_def] = t
        hits_by_arm: dict[str, torch.Tensor] = {}
        for n, applied in applied_masks.items():
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            hit = applied & (first > 0) & (after == 0) & (first < t)
            hits_by_arm[n] = hit
            after[hit] = t
        for n in list(self.first_deferral_step.keys()):
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            hit = hits_by_arm.get(n)
            if hit is None:
                hit = torch.zeros_like(first, dtype=torch.bool)
            just_survived = hit & ((t - first) > 0) & ((t - first) <= H)
            self._close_events_masked(
                first=first,
                after=after,
                close_mask=just_survived,
                now_step=t,
                reason="survived",
            )
            if t > H:
                expired = (first > 0) & (after == 0) & ((t - first) >= H)
                self._close_events_masked(
                    first=first,
                    after=after,
                    close_mask=expired,
                    now_step=t,
                    reason="horizon_expired",
                )
        nc = int(n_candidates)
        na = int(n_applied)
        ratio = float(nc) / float(max(1, na))
        self.per_step_ratios.append(
            {
                "step": t,
                "candidate_crossers_before_cap": nc,
                "applied_count": na,
                "demand_applied_ratio": ratio,
                "deferred_count": max(0, nc - na),
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
        """Close open-applied rows before writeback resets.

        residual_clear/residual_restart partition open_applied and share the same
        residual-arm branch in ``_close_events_masked``. Fuse to ONE close pass
        per arm (Branch-A F1; semantics-preserving vs dual-pass partition).
        """
        del residual_restart, residual_zero
        t = int(step)
        for n, applied in applied_masks.items():
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            open_applied = applied & (first > 0)
            self._close_events_masked(
                first=first,
                after=after,
                close_mask=open_applied,
                now_step=t,
                reason="residual_clear",
            )

    def roll_tracker_after_writeback(
        self,
        *,
        applied_masks: Mapping[str, torch.Tensor],
        episode_start_before: Mapping[str, torch.Tensor],
        episode_start_after: Mapping[str, torch.Tensor],
        step: int,
    ) -> None:
        del step
        for n, applied in applied_masks.items():
            before = episode_start_before[n]
            after = episode_start_after[n]
            changed = applied & (before != after)
            self.episode_generation[n][changed] = self.episode_generation[n][changed] + 1
            self.first_deferral_step[n][changed] = 0
            self.applied_after_deferral_step[n][changed] = 0

    def finalize_window(self, *, final_step: int) -> None:
        t = int(final_step)
        for n in list(self.first_deferral_step.keys()):
            first = self.first_deferral_step[n]
            after = self.applied_after_deferral_step[n]
            open_ev = first > 0
            self._close_events_masked(
                first=first,
                after=after,
                close_mask=open_ev,
                now_step=t,
                reason="window_end",
            )

    def survival_summary(self) -> dict[str, Any]:
        """Receipt-facing survival projection — parity with PressureTelemetryStore."""
        agg = self.aggregates_as_dict()
        n_eval = int(agg["N_events_evaluable"])
        n_surv = int(agg["N_survived_applied_within_H"])
        n_never = int(agg["N_never_applied_within_H"])
        frac_surv = n_surv / max(1, n_eval)
        frac_never = n_never / max(1, n_eval)
        early_n = int(agg["N_events_evaluable_early"])
        late_n = int(agg["N_events_evaluable_late"])
        early_never_frac = (
            float(agg["N_never_applied_within_H_early"]) / float(early_n)
            if early_n
            else None
        )
        late_never_frac = (
            float(agg["N_never_applied_within_H_late"]) / float(late_n)
            if late_n
            else None
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
            **agg,
            "deferred_survival_frac": float(frac_surv),
            "deferred_never_apply_within_H_frac": float(frac_never),
            "deferred_never_apply_within_H_frac_early": early_never_frac,
            "deferred_never_apply_within_H_frac_late": late_never_frac,
            "delta_never_apply": delta,
            "deferred_survival_class": klass,
        }


def cpu_store_from_shapes(
    shapes: Mapping[str, tuple[int, ...]],
    *,
    steps: int,
) -> PressureTelemetryStore:
    q_like = {n: torch.zeros(shape, dtype=torch.int8) for n, shape in shapes.items()}
    return PressureTelemetryStore.from_q_levels(q_like, steps=steps)


def run_full_per_step_lifecycle(
    store: DeviceLifecycleStore,
    *,
    candidate_masks: Mapping[str, torch.Tensor],
    applied_masks: Mapping[str, torch.Tensor],
    episode_start_before: Mapping[str, torch.Tensor],
    episode_start_after: Mapping[str, torch.Tensor],
    step: int,
    n_candidates: int,
    n_applied: int,
    residual_zero: Mapping[str, torch.Tensor] | None = None,
) -> None:
    """Production per-step sequence: process → close_before → roll."""
    store.process_pre_writeback(
        candidate_masks=candidate_masks,
        applied_masks=applied_masks,
        step=step,
        n_candidates=n_candidates,
        n_applied=n_applied,
    )
    store.close_before_writeback_resets(
        applied_masks=applied_masks,
        step=step,
        residual_zero=residual_zero,
    )
    store.roll_tracker_after_writeback(
        applied_masks=applied_masks,
        episode_start_before=episode_start_before,
        episode_start_after=episode_start_after,
        step=step,
    )
