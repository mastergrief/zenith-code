"""Slice-10 Phase-1: C2b_app diagnostic site-emit wiring proofs (emit-only).

Covers WRAP/EMIT-ONLY equivalence, cpu_reference survival/emit-counts,
gpu_seam non-emit of NEW cap_mut marks, multi-state SUM_OF_SIGNED_DELTAS
aggregation, and observer-tax 3x / ABSOLUTE_FLOOR_FALLBACK fixtures.

No classify-run / OWNER bank / mechanism math change.
"""

from __future__ import annotations

import copy
import hashlib
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    _PackedHotTable,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    SLICE10_C2B_APP_CLASSIFY_SITE_IDS,
    SLICE10_C2B_APP_FORBIDDEN_OWNER_SITE_IDS,
    EventCodedVoteUpdateState,
    apply_event_coded_integer_vote_update_from_plan,
    apply_event_coded_vote_and_cap_from_plan,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
    VoteUpdateSpec,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
ADAPTER_PATH = (
    REPO_ROOT
    / "calm/hrm_text_158/native_full_stack/event_coded_vote_update_adapter.py"
)
PROBE_PATH = (
    REPO_ROOT / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
)

REQUIRED_NEW_SITE_IDS = (
    "C2b.S1_vote_first_sync_clone",
    "C2b.S1_vote_first_sync_hot_list",
    "C2b.S1_vote_first_sync_contig",
    "C2b.S1_cap_mut_q_clone",
    "C2b.S1_cap_mut_sync_clone",
    "C2b.S1_cap_mut_sync_hot_list",
    "C2b.S1_cap_mut_sync_contig",
)

C2B_APP_MEDIAN_ANCHOR_GIB = 0.8722820281982422
OWNER_BAR_FRAC = 0.5
ABSOLUTE_FLOOR_GIB = 0.10


def _minimal_plan(*, applied_indices: list[int], numel: int) -> VoteUpdatePlan:
    applied = torch.tensor(applied_indices, dtype=torch.int64)
    empty_i64 = torch.tensor([], dtype=torch.int64)
    empty_i16 = torch.tensor([], dtype=torch.int16)
    empty_i8 = torch.tensor([], dtype=torch.int8)
    return VoteUpdatePlan(
        q_i16=torch.zeros(numel, dtype=torch.int16),
        new_acc_i32=torch.zeros(numel, dtype=torch.int32),
        candidate_indices=applied.clone(),
        pre_veto_selected_indices=applied.clone(),
        applied_indices=applied,
        applied_directions=torch.ones(len(applied_indices), dtype=torch.int8),
        applied_thresholds=torch.full((len(applied_indices),), 10, dtype=torch.int16),
        replay_ce_veto_indices=empty_i64,
        replay_veto_directions=empty_i8,
        replay_veto_thresholds=empty_i16,
        pc_aux_negative_indices=empty_i64,
        pc_aux_veto_indices=empty_i64,
        stats={},
    )


def _make_vote_state(
    *,
    numel: int,
    seed: int,
    hot_count: int,
) -> tuple[EventCodedVoteUpdateState, VoteUpdateInputs, VoteUpdateSpec]:
    rng = np.random.default_rng(int(seed))
    indices = np.sort(rng.choice(numel, size=hot_count, replace=False))
    values = rng.integers(-4, 5, size=hot_count, dtype=np.int16)
    carrier = EventCodedAccLiveState(
        logical_numel=int(numel),
        demotion_band=3,
        _hot=_PackedHotTable.from_arrays(indices, values),
    )
    q_levels = torch.zeros(numel, dtype=torch.int8)
    for flat_index in indices[: min(8, hot_count)]:
        idx_list = list(indices)
        carrier.q_levels[int(flat_index)] = (
            1 if int(values[idx_list.index(flat_index)]) >= 0 else -1
        )
    votes = torch.zeros(numel, dtype=torch.int16)
    vote_active = indices[: min(16, hot_count)]
    for flat_index in vote_active:
        votes[int(flat_index)] = int(rng.integers(1, 12))
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=2,
    )
    state = EventCodedVoteUpdateState(q_levels=q_levels, carrier=carrier)
    inputs = VoteUpdateInputs(votes=votes)
    return state, inputs, spec


def _snapshot(result: Any) -> dict[str, Any]:
    return {
        "carrier_sha256": carrier_content_sha256(result.carrier),
        "q_sha256": hashlib.sha256(
            result.q_levels.detach().cpu().numpy().tobytes()
        ).hexdigest(),
        "live_q": dict(result.carrier.q_levels),
        "hot_exact": dict(result.carrier.hot_exact),
        "q_changed_count": int(result.stats.get("q_changed_count", -1)),
    }


def _collecting_site_emit(marks: list[dict[str, Any]]):
    def site_emit(
        site_id: str,
        suffix: str,
        *,
        origin_file: str = "",
        origin_line: int = 0,
        optimizer_step_index: int = 0,
        state_index: int = -1,
    ) -> None:
        marks.append(
            {
                "site_id": str(site_id),
                "suffix": str(suffix),
                "optimizer_step_index": int(optimizer_step_index),
                "state_index": int(state_index),
                "origin_file": str(origin_file),
                "origin_line": int(origin_line),
            }
        )

    return site_emit


def _pair_count(
    marks: list[dict[str, Any]],
    site_id: str,
    *,
    step: int | None = None,
    state_index: int | None = None,
) -> int:
    pre = 0
    post = 0
    for mark in marks:
        if mark["site_id"] != site_id:
            continue
        if step is not None and int(mark["optimizer_step_index"]) != int(step):
            continue
        if state_index is not None and int(mark["state_index"]) != int(state_index):
            continue
        if mark["suffix"] == "pre":
            pre += 1
        elif mark["suffix"] == "post":
            post += 1
    assert pre == post, (site_id, pre, post, step, state_index)
    return pre


def _sum_of_signed_deltas(
    pairs: list[tuple[int, int, str, float, float]],
) -> dict[tuple[int, str], float]:
    """pairs: (step, state, site_id, pre_rss, post_rss) → (step, site_id) SUM."""

    out: dict[tuple[int, str], float] = defaultdict(float)
    for step, _state, site_id, pre_rss, post_rss in pairs:
        out[(int(step), str(site_id))] += float(post_rss) - float(pre_rss)
    return dict(out)


def _classify_owner_vs_tax(
    *,
    owner_aggregated_step_median: float,
    c2b_app_median: float,
    measured_observer_tax_median: float | None,
) -> dict[str, Any]:
    owner_bar = OWNER_BAR_FRAC * float(c2b_app_median)
    if measured_observer_tax_median is None:
        tax = ABSOLUTE_FLOOR_GIB
        tax_source = "ABSOLUTE_FLOOR_FALLBACK"
    else:
        tax = float(measured_observer_tax_median)
        tax_source = "MEASURED_NOOP_BAND"
    clears_owner_bar = owner_aggregated_step_median >= owner_bar
    clears_3x_tax = owner_aggregated_step_median >= 3.0 * tax
    clears_abs = owner_aggregated_step_median >= ABSOLUTE_FLOOR_GIB
    if clears_owner_bar and clears_3x_tax and clears_abs:
        verdict = "OWNER_NAMED_SITE"
    elif clears_owner_bar and not clears_3x_tax:
        verdict = "INCONCLUSIVE_SPLIT_INSIDE_C2B_APP"
    else:
        verdict = "INCONCLUSIVE_SPLIT_INSIDE_C2B_APP"
    return {
        "verdict": verdict,
        "tax_source": tax_source,
        "tax": tax,
        "owner_bar": owner_bar,
        "clears_owner_bar": clears_owner_bar,
        "clears_3x_tax": clears_3x_tax,
        "clears_abs": clears_abs,
    }


def test_survival_required_site_id_literals_in_adapter() -> None:
    text = ADAPTER_PATH.read_text(encoding="utf-8")
    for site_id in REQUIRED_NEW_SITE_IDS:
        assert f'"{site_id}"' in text, f"missing literal {site_id}"
    assert '"C4.S1a"' in text
    for site_id in REQUIRED_NEW_SITE_IDS:
        assert site_id in SLICE10_C2B_APP_CLASSIFY_SITE_IDS
    assert "C4.S1c_clone" in SLICE10_C2B_APP_FORBIDDEN_OWNER_SITE_IDS
    assert "C4.S1c_contig" in SLICE10_C2B_APP_FORBIDDEN_OWNER_SITE_IDS


def test_probe_allowlist_registers_slice10_site_ids() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_SLICE10_C2B_APP_SITE_IDS,
    )

    for site_id in REQUIRED_NEW_SITE_IDS:
        assert site_id in PROFILE_HOST_RSS_SLICE10_C2B_APP_SITE_IDS
    assert "C4.S1a" in PROFILE_HOST_RSS_SLICE10_C2B_APP_SITE_IDS
    # probe file must also contain the frozenset literal registry
    probe_text = PROBE_PATH.read_text(encoding="utf-8")
    assert "PROFILE_HOST_RSS_SLICE10_C2B_APP_SITE_IDS" in probe_text


def test_wrap_emit_only_vote_cap_q_bit_identical() -> None:
    """Emitter threading must NOT change vote/cap/q math."""

    numel = 256
    applied = [5, 17, 42, 99]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    state0, inputs, spec = _make_vote_state(numel=numel, seed=7, hot_count=32)

    baseline_state = EventCodedVoteUpdateState(
        q_levels=state0.q_levels.clone(),
        carrier=state0.carrier.cow_copy(),
    )
    emit_state = EventCodedVoteUpdateState(
        q_levels=state0.q_levels.clone(),
        carrier=state0.carrier.cow_copy(),
    )
    marks: list[dict[str, Any]] = []
    baseline = apply_event_coded_vote_and_cap_from_plan(
        baseline_state,
        inputs,
        spec,
        plan,
        applied,
        step_index=3,
        lightweight_runtime_stats=True,
        site_emit_enabled=False,
    )
    with_emit = apply_event_coded_vote_and_cap_from_plan(
        emit_state,
        inputs,
        spec,
        plan,
        applied,
        step_index=3,
        lightweight_runtime_stats=True,
        host_allocator_site_emit=_collecting_site_emit(marks),
        site_emit_enabled=True,
        optimizer_step_index=3,
        state_index=0,
    )
    assert _snapshot(baseline) == _snapshot(with_emit)
    assert marks  # emitter actually fired


def test_cpu_reference_emit_counts_per_step_state() -> None:
    numel = 128
    applied = [3, 7, 11]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    state0, inputs, spec = _make_vote_state(numel=numel, seed=11, hot_count=24)
    # Ensure hot q_levels non-empty so hot_list brackets fire.
    assert state0.carrier.q_levels

    marks: list[dict[str, Any]] = []
    apply_event_coded_vote_and_cap_from_plan(
        state0,
        inputs,
        spec,
        plan,
        applied,
        step_index=2,
        lightweight_runtime_stats=True,
        host_allocator_site_emit=_collecting_site_emit(marks),
        site_emit_enabled=True,
        optimizer_step_index=2,
        state_index=0,
    )

    assert _pair_count(marks, "C4.S1a", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_vote_first_sync_clone", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_vote_first_sync_hot_list", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_vote_first_sync_contig", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_clone", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_hot_list", step=2, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_contig", step=2, state_index=0) == 1
    # Vote vs cap IDs remain distinct (no shared site_id across paths).
    assert _pair_count(marks, "C4.S1c_clone", step=2, state_index=0) == 0
    assert _pair_count(marks, "C4.S1c_contig", step=2, state_index=0) == 0


def test_cpu_reference_empty_accepted_skips_cap_mut_marks() -> None:
    numel = 64
    plan = _minimal_plan(applied_indices=[1, 2], numel=numel)
    state0, inputs, spec = _make_vote_state(numel=numel, seed=13, hot_count=12)
    marks: list[dict[str, Any]] = []
    apply_event_coded_vote_and_cap_from_plan(
        state0,
        inputs,
        spec,
        plan,
        [],  # empty accepted
        step_index=1,
        lightweight_runtime_stats=True,
        host_allocator_site_emit=_collecting_site_emit(marks),
        site_emit_enabled=True,
        optimizer_step_index=1,
        state_index=0,
    )
    assert _pair_count(marks, "C4.S1a", step=1, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_vote_first_sync_clone", step=1, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone", step=1, state_index=0) == 0
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_clone", step=1, state_index=0) == 0
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_contig", step=1, state_index=0) == 0


def test_gpu_seam_path_emits_zero_new_cap_mut_marks() -> None:
    """GPU seam uses integer_vote_update (not vote_and_cap) → no NEW C2b cap_mut marks;
    vote-path sync keeps legacy C4.S1c_* when classify_site_prefix is unset.
    """

    numel = 64
    applied = [3, 7]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    state0, inputs, spec = _make_vote_state(numel=numel, seed=19, hot_count=16)
    marks: list[dict[str, Any]] = []
    apply_event_coded_integer_vote_update_from_plan(
        state0,
        inputs,
        spec,
        plan,
        step_index=4,
        lightweight_runtime_stats=True,
        host_allocator_site_emit=_collecting_site_emit(marks),
        site_emit_enabled=True,
        optimizer_step_index=4,
        state_index=1,
        # GPU seam leaves classify_site_prefix unset
    )
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone") == 0
    assert _pair_count(marks, "C2b.S1_cap_mut_sync_clone") == 0
    assert _pair_count(marks, "C2b.S1_vote_first_sync_clone") == 0
    # Legacy GPU marks still fire on vote sync
    assert _pair_count(marks, "C4.S1c_clone", step=4, state_index=1) == 1
    assert _pair_count(marks, "C4.S1a", step=4, state_index=1) == 1


def test_multi_state_sum_of_signed_deltas_aggregation() -> None:
    """>=2 states in one step: aggregate SUM_OF_SIGNED_DELTAS per (step, site_id)."""

    numel = 96
    applied = [2, 5, 8]
    plan = _minimal_plan(applied_indices=applied, numel=numel)
    step = 9
    # Synthetic RSS pairs for two states (same site_id) — proves aggregation unit.
    pairs = [
        (step, 0, "C2b.S1_cap_mut_q_clone", 1.00, 1.20),  # +0.20
        (step, 1, "C2b.S1_cap_mut_q_clone", 1.20, 1.35),  # +0.15
        (step, 0, "C4.S1a", 0.50, 0.55),  # +0.05
        (step, 1, "C4.S1a", 0.55, 0.52),  # -0.03 (signed; do not clamp)
    ]
    agg = _sum_of_signed_deltas(pairs)
    assert abs(agg[(step, "C2b.S1_cap_mut_q_clone")] - 0.35) < 1e-9
    assert abs(agg[(step, "C4.S1a")] - 0.02) < 1e-9
    # Per-state median would undercount / mis-sign vs step-level C2b_app.
    per_state_medians = [0.20, 0.15]
    assert abs(sum(per_state_medians) - agg[(step, "C2b.S1_cap_mut_q_clone")]) < 1e-9

    # Live emit: two states produce distinct state_index pairs for same site_id.
    marks: list[dict[str, Any]] = []
    emit = _collecting_site_emit(marks)
    for state_index, seed in ((0, 21), (1, 22)):
        state, inputs, spec = _make_vote_state(numel=numel, seed=seed, hot_count=20)
        apply_event_coded_vote_and_cap_from_plan(
            state,
            inputs,
            spec,
            plan,
            applied,
            step_index=step,
            lightweight_runtime_stats=True,
            host_allocator_site_emit=emit,
            site_emit_enabled=True,
            optimizer_step_index=step,
            state_index=state_index,
        )
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone", step=step, state_index=0) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone", step=step, state_index=1) == 1
    assert _pair_count(marks, "C2b.S1_cap_mut_q_clone", step=step) == 2


def test_observer_tax_3x_pass_fail_and_absolute_floor_fallback() -> None:
    c2b = C2B_APP_MEDIAN_ANCHOR_GIB
    # Case 1: clears 0.5×C2b_app and 3×tax → OWNER
    pass_case = _classify_owner_vs_tax(
        owner_aggregated_step_median=0.50,
        c2b_app_median=c2b,
        measured_observer_tax_median=0.05,
    )
    assert pass_case["verdict"] == "OWNER_NAMED_SITE"
    assert pass_case["tax_source"] == "MEASURED_NOOP_BAND"
    assert pass_case["clears_3x_tax"] is True

    # Case 2: clears 0.5×C2b_app but fails 3×tax → INCONCLUSIVE
    fail_tax = _classify_owner_vs_tax(
        owner_aggregated_step_median=0.50,
        c2b_app_median=c2b,
        measured_observer_tax_median=0.20,  # 3×tax = 0.60 > 0.50
    )
    assert fail_tax["verdict"] == "INCONCLUSIVE_SPLIT_INSIDE_C2B_APP"
    assert fail_tax["clears_owner_bar"] is True
    assert fail_tax["clears_3x_tax"] is False

    # Case 3: ABSOLUTE_FLOOR_FALLBACK 0.10 replayable when no-op band absent
    fallback = _classify_owner_vs_tax(
        owner_aggregated_step_median=0.50,
        c2b_app_median=c2b,
        measured_observer_tax_median=None,
    )
    assert fallback["tax_source"] == "ABSOLUTE_FLOOR_FALLBACK"
    assert abs(fallback["tax"] - ABSOLUTE_FLOOR_GIB) < 1e-12
    # 3×0.10 = 0.30; 0.50 clears → OWNER under fallback tax
    assert fallback["verdict"] == "OWNER_NAMED_SITE"
    assert fallback["clears_3x_tax"] is True
