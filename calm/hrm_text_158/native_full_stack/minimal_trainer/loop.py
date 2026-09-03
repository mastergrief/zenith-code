#!/usr/bin/env python3
# ADVISOR_ROUTE: 1787660482744-362bf43f + 1787661679613-29749c30 + 1787662064718-ba0850ee + 1787662756505-b67b7ec7 + 1787662818490-9e6356f6 + 1787663294729-ae3398e4 + 1787663396497-a3e41b3b + 1787664407382-bd1d5e9d + 1787664473238-f8ba85b3 + 1787667968372-bc5f49a3 + 1787668333973-45d3c65d + 1787669129125-44516dbf + 1787669631996-db859015 + 1787670631494-80766415 + 1787672249157-2a04e77b + 1787673873550-79467121 + 1787673958027-de5f6a07 + 1787997347232-69ad666d
# ADVISOR_ROUTE: 1788456771491-42f1f60c (threshold_abs threading, task 1788456823866-aa9a873d)
"""Minimal SHARE step loop for freeze_v1 kwargs (S1)."""

# INVARIANT: never import bounded_delta_loop*.py.
# INVARIANT: never import the name run_bounded_delta_steps.

from __future__ import annotations

import sys
import time
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


def _install_scripts_namespace() -> None:
    """Load scripts.* without exec'ing untracked scripts/__init__.py."""
    if "scripts" in sys.modules:
        return
    root = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
    pkg = types.ModuleType("scripts")
    pkg.__path__ = [str(root / "scripts")]  # type: ignore[attr-defined]
    pkg.__package__ = "scripts"
    sys.modules["scripts"] = pkg


_install_scripts_namespace()

from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
    _bounded_delta_vote_step_two_tier_kwargs,
    _compute_ce_weighted_grads,
    _science_global_cap_spec_for_arm,
    _science_local_selection_ordering_mode,
    _weighted_grads_to_science_arm_votes,
    resolve_probe_vote_update_spec,
    resolve_r7_deferred_backlog_vote_step_kwargs,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    default_dry_run_rank_vote_spec,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    seed_initial_deferred_backlog,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    resolve_named_global_cap_spec,
)
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ARM_A0_RANK_BUCKET_CURRENT,
)


PRODUCTION_CANDIDATE_WINDOWS = (1, 10, 50)

NEVER_CANDIDATE_STEP = -1


def _ever_crossed_masks_for_states(
    tensor_states: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Frame-local ever-crossed masks, one bool per eligible-leaf element.

    In-memory only: these never enter tensor_states, state_dict, checkpoints,
    or the returned train-state.
    """
    return {
        str(key): torch.zeros(int(state.q_levels.numel()), dtype=torch.bool)
        for key, state in tensor_states.items()
    }


def _last_candidate_steps_for_states(
    tensor_states: Mapping[str, Any],
) -> dict[str, torch.Tensor]:
    """Frame-local last-candidate-step stamps, one int32 per eligible-leaf element.

    Same lifetime and exclusions as the ever-crossed masks: never in
    tensor_states, state_dict, checkpoints, or the returned train-state.
    """
    return {
        str(key): torch.full(
            (int(state.q_levels.numel()),),
            NEVER_CANDIDATE_STEP,
            dtype=torch.int32,
        )
        for key, state in tensor_states.items()
    }


def _prepared_candidate_indices(
    observation: Mapping[str, Any],
    numel_by_key: Mapping[str, int],
) -> dict[str, torch.Tensor]:
    """Validate the whole observation before any observer state is written.

    Refuses a plan/observer key-set mismatch and any out-of-range candidate
    index, so a later invalid leaf cannot leave earlier leaves mutated.
    """
    plans_by_key = {
        str(key): plan for key, plan in dict(observation["plans_by_key"]).items()
    }
    if set(plans_by_key) != set(numel_by_key):
        raise RuntimeError(
            "candidate observation key mismatch: plans "
            f"{sorted(plans_by_key)} vs observer {sorted(numel_by_key)}"
        )
    prepared: dict[str, torch.Tensor] = {}
    for key, plan in plans_by_key.items():
        idx = plan.candidate_indices.detach().cpu().to(torch.int64).reshape(-1)
        numel = int(numel_by_key[key])
        if int(idx.numel()) > 0:
            low = int(idx.min().item())
            high = int(idx.max().item())
            if low < 0 or high >= numel:
                raise RuntimeError(
                    f"candidate index out of range for leaf {key}: "
                    f"[{low}, {high}] outside [0, {numel - 1}]"
                )
        prepared[key] = idx
    return prepared


def _observe_candidates(
    masks: Mapping[str, torch.Tensor] | None,
    last_candidate_steps: Mapping[str, torch.Tensor] | None,
    observation: Mapping[str, Any],
    step: int,
) -> None:
    """One producer read: OR into the cumulative masks and stamp `step`.

    Both frame-local observer states are written from the same validated
    indices, so neither can be mutated by an observation the other rejects.
    """
    numel_source = masks if masks is not None else last_candidate_steps
    if numel_source is None:
        raise RuntimeError(
            "candidate observation requires at least one frame-local observer state"
        )
    prepared = _prepared_candidate_indices(
        observation,
        {str(key): int(tensor.numel()) for key, tensor in numel_source.items()},
    )
    for key, idx in prepared.items():
        if int(idx.numel()) == 0:
            continue
        if masks is not None:
            masks[key][idx] = True
        if last_candidate_steps is not None:
            last_candidate_steps[key][idx] = int(step)


def _or_accumulate_ever_crossed(
    masks: dict[str, torch.Tensor],
    observation: Mapping[str, Any],
) -> None:
    """OR this step's producer candidate_indices into the cumulative masks."""
    _observe_candidates(masks, None, observation, 0)


def _ever_crossed_emission(masks: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Cumulative fractions plus their denominators; refuses an empty universe."""
    numel_by_key = {key: int(mask.numel()) for key, mask in sorted(masks.items())}
    crossed_by_key = {
        key: int(mask.sum().item()) for key, mask in sorted(masks.items())
    }
    total_numel = sum(numel_by_key.values())
    if total_numel <= 0:
        raise RuntimeError(
            "ever-crossed emission empty-denominator: zero elements across "
            f"{len(numel_by_key)} eligible leaves; refusing to report a fraction"
        )
    return {
        "ever_crossed_numel_by_key": numel_by_key,
        "ever_crossed_fraction_by_key": {
            key: float(crossed_by_key[key]) / float(numel_by_key[key])
            for key in numel_by_key
        },
        "ever_crossed_fraction_total": float(sum(crossed_by_key.values()))
        / float(total_numel),
    }


def _require_fresh_observation(observed_step: Any, step: int) -> None:
    """Refuse to publish a row the observer did not produce on this step.

    A vote-step family that returns before calling the observer would otherwise
    leave the previous step's stamps in place and emit a valid-looking row.
    """
    if observed_step != int(step):
        raise RuntimeError(
            "candidate observer did not fire on step "
            f"{int(step)} (last observed {observed_step}): the vote-step family "
            "did not call the observer; refusing to publish a stale "
            "windowed-candidate row"
        )


def _windowed_candidate_emission(
    last_candidate_steps: Mapping[str, torch.Tensor],
    step: int,
    windows: Sequence[int],
) -> dict[str, Any]:
    """Trailing-window candidate occupancy: count(last_candidate_step >= t-W+1).

    The -1 sentinel is excluded explicitly, so a window reaching below step 0
    cannot count entries that were never candidates.
    """
    numel_by_key = {
        key: int(stamps.numel())
        for key, stamps in sorted(last_candidate_steps.items())
    }
    numel_total = sum(numel_by_key.values())
    if numel_total <= 0:
        raise RuntimeError(
            "windowed-candidate emission empty-denominator: zero elements across "
            f"{len(numel_by_key)} eligible leaves; refusing to report a fraction"
        )
    empty_leaves = sorted(key for key, numel in numel_by_key.items() if numel <= 0)
    if empty_leaves:
        raise RuntimeError(
            "windowed-candidate emission empty-denominator: leaves "
            f"{empty_leaves} have zero elements; refusing to report a fraction"
        )
    rows: dict[str, Any] = {}
    for window in windows:
        if int(window) < 1:
            raise RuntimeError(
                f"windowed-candidate window must be >= 1, got {int(window)}"
            )
        threshold = int(step) - int(window) + 1
        count_by_key = {}
        for key in numel_by_key:
            stamps = last_candidate_steps[key]
            count_by_key[key] = int(
                torch.count_nonzero((stamps >= threshold) & (stamps >= 0)).item()
            )
        count_total = sum(count_by_key.values())
        rows[str(int(window))] = {
            "numel_by_key": dict(numel_by_key),
            "count_by_key": count_by_key,
            "fraction_by_key": {
                key: float(count_by_key[key]) / float(numel_by_key[key])
                for key in numel_by_key
            },
            "numel_total": int(numel_total),
            "count_total": int(count_total),
            "fraction_total": float(count_total) / float(numel_total),
        }
    return rows


def run_loop(
    model: Any,
    batch: Mapping[str, Any],
    tensor_states: Mapping[str, Any],
    eligible_modules: Mapping[str, Any],
    *,
    device: Any,
    steps: int,
    require_q_change: bool = False,
    max_abs_per_tensor: int,
    support_batches: Sequence[Mapping[str, Any]] | None = None,
    r7_deferred_backlog_carry_enabled: bool = False,
    global_cap_contract: str,
    science_arm: str = ARM_A0_RANK_BUCKET_CURRENT,
    start_step: int = 1,
    confirmation_envelope: str | None = None,
    threshold_abs: int = 1,
    two_tier_carry_w6_enabled: bool = False,
    event_coded_sparse_vote_authority: bool = False,
    global_horizon: int | None = None,
    ever_crossed_observer_enabled: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    str,
    int,
    Any,
    Any,
    list[int],
]:
    del batch  # freeze_v1 supplies support_batches
    model.train()
    vote_spec = resolve_probe_vote_update_spec(
        max_abs_per_tensor=int(max_abs_per_tensor),
        confirmation_envelope=confirmation_envelope,
        threshold_abs=int(threshold_abs),
        vote_update_decay_numerator=None,
        vote_update_decay_denominator=None,
    )
    rank_spec = default_dry_run_rank_vote_spec()
    vote_specs = {key: vote_spec for key in tensor_states}
    states = dict(tensor_states)
    carry_backlog = (
        seed_initial_deferred_backlog(None)
        if bool(r7_deferred_backlog_carry_enabled)
        else None
    )
    ever_crossed_masks = (
        _ever_crossed_masks_for_states(states)
        if bool(ever_crossed_observer_enabled)
        else None
    )
    last_candidate_steps = (
        None
        if ever_crossed_masks is None
        else _last_candidate_steps_for_states(states)
    )
    observed_step: list[int | None] = [None]

    def _candidate_observer_for_step(observed_at: int):
        def observe(observation: Mapping[str, Any]) -> None:
            _observe_candidates(
                ever_crossed_masks, last_candidate_steps, observation, observed_at
            )
            observed_step[0] = observed_at

        return observe
    if not support_batches:
        raise RuntimeError("bounded-delta step loop requires at least one support batch")
    step_batches = list(support_batches)
    step_reports: dict[str, Any] = {}
    steps_completed = 0
    bp_horizon = (
        int(global_horizon)
        if global_horizon is not None
        else int(start_step) + int(steps) - 1
    )
    for step in range(int(start_step), int(start_step) + int(steps)):
        t0 = time.perf_counter()
        step_batch = step_batches[(step - 1) % len(step_batches)]["batch"]
        extras = model.compute_train_extra_args(step, max(1, bp_horizon))
        weighted_grads, loss, metrics = _compute_ce_weighted_grads(
            model,
            step_batch,
            states,
            eligible_modules,
            device=device,
            extras=extras,
        )
        step_loss = float(loss)
        del loss, metrics
        sparse_events_by_key: dict[str, Any] = {}
        votes_by_key, vote_pressure_by_key, finite_weighted_grad = (
            _weighted_grads_to_science_arm_votes(
                weighted_grads,
                states,
                rank_spec=rank_spec,
                vote_spec=vote_spec,
                science_arm=str(science_arm),
                sparse_events_out=sparse_events_by_key,
                sparse_construction_only=False,
            )
        )
        del vote_pressure_by_key, finite_weighted_grad
        global_cap_spec = resolve_named_global_cap_spec(
            str(global_cap_contract),
            step=int(step),
        )
        effective_global_cap_spec = _science_global_cap_spec_for_arm(
            global_cap_spec,
            science_arm=str(science_arm),
        )
        two_tier_vote_step_kwargs = _bounded_delta_vote_step_two_tier_kwargs(
            two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
            local_loss_delta_by_key=None,
        )
        step_selection_ordering_mode = _science_local_selection_ordering_mode(
            str(science_arm)
        )
        apply_votes_by_key = (
            None if bool(event_coded_sparse_vote_authority) else votes_by_key
        )
        step_result = apply_bounded_delta_vote_step(
            states,
            apply_votes_by_key,
            vote_specs,
            global_cap_spec=effective_global_cap_spec,
            global_cap_contract_name=(
                str(global_cap_contract)
                if effective_global_cap_spec is not None
                else None
            ),
            local_selection_ordering_mode=step_selection_ordering_mode,
            local_selection_ordering_seed=SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
            local_selection_ordering_step=int(step),
            candidate_sparse_vote_events_by_key=sparse_events_by_key,
            front_c_identity_observer=(
                None
                if ever_crossed_masks is None
                else _candidate_observer_for_step(int(step))
            ),
            **two_tier_vote_step_kwargs,
            **resolve_r7_deferred_backlog_vote_step_kwargs(
                r7_deferred_backlog_carry_enabled=bool(
                    r7_deferred_backlog_carry_enabled
                ),
                carry_backlog=carry_backlog,
            ),
        )
        states = step_result.tensor_states
        if r7_deferred_backlog_carry_enabled:
            carry_backlog = step_result.deferred_backlog
        q_changed_count = int(step_result.global_summary.get("q_changed_count", 0))
        if require_q_change and q_changed_count <= 0:
            raise RuntimeError(
                "bounded-delta step produced no q movement under --require-q-change"
            )
        step_report: dict[str, Any] = {
            "duration_seconds": float(time.perf_counter() - t0),
            "q_changed_count": q_changed_count,
            "loss": step_loss,
        }
        if ever_crossed_masks is not None and last_candidate_steps is not None:
            _require_fresh_observation(observed_step[0], int(step))
            step_report.update(_ever_crossed_emission(ever_crossed_masks))
            step_report["windowed_candidate_step"] = int(step)
            step_report["windowed_candidate_windows"] = _windowed_candidate_emission(
                last_candidate_steps, int(step), PRODUCTION_CANDIDATE_WINDOWS
            )
        step_reports[str(step)] = step_report
        steps_completed = int(step)
    return (
        step_reports,
        {},
        states,
        {},
        "max_steps_completed",
        steps_completed,
        None,
        None,
        [],
    )
