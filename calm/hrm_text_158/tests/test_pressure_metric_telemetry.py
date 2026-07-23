"""CPU-static tests for PLAN_v6 pressure_metric_telemetry.

Covers: lifecycle order, observer fidelity (candidates>topk), classifier matrix,
aggregate strict-JSON on real output.
"""
from __future__ import annotations

import json

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_classifier import (
    classify_pressure_metric_family,
    select_family_from_predicates,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_lifecycle import (
    DurableAggregates,
    PressureTelemetryStore,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_readiness import (
    evaluate_readiness,
    validate_trajectory_schemas,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_receipt import (
    build_diagnostic_receipt,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    CROSSING_THRESHOLD_ABS,
    FOLLOW_UP_HORIZON_STEPS,
    GROWING_DEFERRED_SURVIVAL_DELTA,
    HIGH_DEMAND_RATIO,
    HIGH_LCF,
    LABEL_R0,
    LABEL_R1,
    LABEL_R2,
    LABEL_R3,
    LABEL_R4,
    LOW_MODERATE_DEMAND_RATIO_MAX,
    MATERIAL_H_MOTION_BPW,
    MIN_COHORT_N,
    PARENT_SHA256,
    PLAN_SHA256,
    REPRESENTATION_IMMOVABLE_H_DELTA_MAX,
    STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR,
    SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
    active_episode_stats,
    compute_topk_masks_and_counts,
    expected_trajectory_boundaries,
    global_margin_quantiles,
    hash_scale_dict,
    margin_quantiles,
    sanitize_receipt_for_strict_json,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    CROSSING_THRESHOLD_ABS as T_CANON,
)


def _store(n: int = 8, steps: int = 150) -> PressureTelemetryStore:
    q = {"w": torch.zeros(n, dtype=torch.int8)}
    return PressureTelemetryStore.from_q_levels(q, steps=steps)


def test_pre_deferral_application_ignored():
    """Apply before first deferral does not set applied_after or count as survival."""
    st = _store(4, steps=100)
    # step1: apply only (no candidate deferral) — force applied mask without prior deferral
    zeros = torch.zeros(4, dtype=torch.bool)
    applied = torch.tensor([True, False, False, False])
    cand = applied.clone()  # applied implies candidate
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": applied},
        step=1,
        n_candidates=1,
        n_applied=1,
    )
    # No first deferral because deferred = cand & ~applied is empty
    assert int(st.first_deferral_step["w"].sum().item()) == 0
    assert int(st.applied_after_deferral_step["w"].sum().item()) == 0
    # later deferral then should start fresh
    cand2 = torch.tensor([True, False, False, False])
    app2 = torch.tensor([False, False, False, False])
    st.process_pre_writeback(
        candidate_masks={"w": cand2},
        applied_masks={"w": app2},
        step=2,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 2


def test_deferral_then_later_application_counted_lifecycle():
    st = _store(4, steps=100)
    cand = torch.tensor([True, False, False, False])
    none = torch.zeros(4, dtype=torch.bool)
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": none},
        step=5,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 5
    # apply later within H
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": cand},
        step=10,
        n_candidates=1,
        n_applied=1,
    )
    surv = st.survival_summary()
    assert surv["N_survived_applied_within_H"] >= 1
    assert int(st.first_deferral_step["w"][0].item()) == 0  # closed


def test_applied_flip_residual_restart_closes_before_generation_rollover():
    st = _store(2, steps=100)
    cand = torch.tensor([True, False])
    none = torch.zeros(2, dtype=torch.bool)
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": none},
        step=3,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 3
    # about to writeback-apply with residual restart — close BEFORE generation bump
    applied = torch.tensor([True, False])
    n_before = st.aggregates.N_events_evaluable + st.aggregates.N_events_censored_insufficient_followup
    st.close_before_writeback_resets(
        applied_masks={"w": applied},
        step=4,
        residual_zero={"w": torch.tensor([False, False])},
    )
    n_after = st.aggregates.N_events_evaluable + st.aggregates.N_events_censored_insufficient_followup
    assert n_after == n_before + 1
    assert int(st.first_deferral_step["w"][0].item()) == 0
    # then generation rollover
    before = {"w": torch.tensor([3, 0], dtype=torch.int32)}
    after = {"w": torch.tensor([4, 0], dtype=torch.int32)}
    gen0 = int(st.episode_generation["w"][0].item())
    st.roll_tracker_after_writeback(
        applied_masks={"w": applied},
        episode_start_before=before,
        episode_start_after=after,
        step=4,
    )
    assert int(st.episode_generation["w"][0].item()) == gen0 + 1


def test_zero_residual_reset_closes_or_clears():
    st = _store(2, steps=100)
    cand = torch.tensor([True, False])
    none = torch.zeros(2, dtype=torch.bool)
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": none},
        step=2,
        n_candidates=1,
        n_applied=0,
    )
    st.close_before_writeback_resets(
        applied_masks={"w": cand},
        step=3,
        residual_zero={"w": torch.tensor([True, False])},
    )
    assert int(st.first_deferral_step["w"][0].item()) == 0


def test_repeated_episodes_no_state_leak():
    st = _store(2, steps=100)
    cand = torch.tensor([True, False])
    none = torch.zeros(2, dtype=torch.bool)
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": none},
        step=2,
        n_candidates=1,
        n_applied=0,
    )
    st.close_before_writeback_resets(
        applied_masks={"w": cand}, step=3, residual_zero={"w": none}
    )
    st.roll_tracker_after_writeback(
        applied_masks={"w": cand},
        episode_start_before={"w": torch.tensor([2, 0], dtype=torch.int32)},
        episode_start_after={"w": torch.tensor([3, 0], dtype=torch.int32)},
        step=3,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 0
    assert int(st.episode_generation["w"][0].item()) == 1
    # new episode deferral
    st.process_pre_writeback(
        candidate_masks={"w": cand},
        applied_masks={"w": none},
        step=5,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 5


def test_early_late_zero_denominator_guard_cohort():
    # Force class=other when cohort dens too small
    st = _store(1, steps=150)
    st.aggregates = DurableAggregates(
        N_events_evaluable=50,
        N_survived_applied_within_H=10,
        N_never_applied_within_H=40,
        N_events_evaluable_early=40,
        N_events_evaluable_late=10,  # < MIN_COHORT_N
        N_never_applied_within_H_early=30,
        N_never_applied_within_H_late=10,
    )
    surv = st.survival_summary()
    assert surv["deferred_survival_class"] == "other"
    # growing predicates must not fire via classifier R1
    v = classify_pressure_metric_family(
        telemetry_ok=True,
        two_tier_threshold_assert_pass=True,
        paired_determinism_cost_ok=True,
        N_events_evaluable=50,
        mean_ratio=3.0,
        frac_steps_ratio_ge_2=0.9,
        deferred_survival_class="other",
        delta_never_apply=0.5,
        N_events_evaluable_early=40,
        N_events_evaluable_late=10,
        deferred_never_apply_within_H_frac=0.8,
        lcf=0.99,
        H_final=7.0,
        H_step25=5.0,
        retention_ok_flag=True,
    )
    assert v["R1"] is False


def test_observer_fidelity_candidate_gt_topk():
    # 5 crossers, topk=2 → candidates > topk
    # Hand-computed expected masks (durable vs self-referential helper equality).
    T = CROSSING_THRESHOLD_ABS
    acc = {"w": torch.tensor([T + 5, T + 4, T + 3, T + 2, T + 1, 0, 0], dtype=torch.int16)}
    topk = 2
    expected_candidates = torch.tensor([1, 1, 1, 1, 1, 0, 0], dtype=torch.bool)
    expected_applied = torch.tensor([1, 1, 0, 0, 0, 0, 0], dtype=torch.bool)
    cand, applied, n_c, n_a = compute_topk_masks_and_counts(acc, topk=topk)
    assert n_c == 5
    assert n_c > topk
    assert n_a == 2
    assert abs((n_c / max(1, n_a)) - (5 / 2)) < 1e-9
    assert torch.equal(cand["w"], expected_candidates)
    assert torch.equal(applied["w"], expected_applied)
    # separate margin populations
    mq_pre = margin_quantiles(acc["w"], cand["w"])
    mq_app = margin_quantiles(acc["w"], applied["w"])
    assert mq_pre["n"] == 5
    assert mq_app["n"] == 2
    assert mq_pre["p50"] != mq_app["p50"] or mq_pre["n"] != mq_app["n"]
    assert T_CANON == T


def test_global_margin_quantiles_multi_param_never_n_parts():
    """D1: margins are GLOBAL over concatenated |acc|; never degrade to n_parts."""
    T = CROSSING_THRESHOLD_ABS
    # Crossers on both params; topk=2 forces partial application across params.
    # a: margins 10, 1  |  b: margins 8, 5, 2  → 5 candidates, apply top-2 = 10 (a0), 8 (b0)
    acc = {
        "a": torch.tensor([T + 10, T + 1, 0], dtype=torch.int16),
        "b": torch.tensor([T + 8, T + 5, T + 2], dtype=torch.int16),
    }
    topk = 2
    cand, applied, n_c, n_a = compute_topk_masks_and_counts(acc, topk=topk)
    assert n_c == 5
    assert n_a == 2
    assert torch.equal(cand["a"], torch.tensor([1, 1, 0], dtype=torch.bool))
    assert torch.equal(cand["b"], torch.tensor([1, 1, 1], dtype=torch.bool))
    assert torch.equal(applied["a"], torch.tensor([1, 0, 0], dtype=torch.bool))
    assert torch.equal(applied["b"], torch.tensor([1, 0, 0], dtype=torch.bool))

    pre = global_margin_quantiles(acc, cand, threshold=T)
    app = global_margin_quantiles(acc, applied, threshold=T)
    # Exact torch.quantile on pooled margins [10,1,8,5,2] and [10,8]
    assert pre["n"] == 5
    assert app["n"] == 2
    assert abs(pre["p10"] - 1.4) < 1e-5
    assert abs(pre["p50"] - 5.0) < 1e-5
    assert abs(pre["p90"] - 9.2) < 1e-5
    assert abs(app["p10"] - 8.2) < 1e-5
    assert abs(app["p50"] - 9.0) < 1e-5
    assert abs(app["p90"] - 9.8) < 1e-5
    # Emission shape must be quantile tuple, never n_parts collapse
    for payload in (pre, app):
        assert "n_parts" not in payload
        assert set(payload.keys()) == {"p10", "p50", "p90", "n"}
    # Per-param collapse would have been wrong: single-param p50s differ from global
    a_only = margin_quantiles(acc["a"], cand["a"], threshold=T)
    assert a_only["n"] == 2
    assert a_only["p50"] != pre["p50"]


def _base_kwargs(**over):
    kw = dict(
        telemetry_ok=True,
        two_tier_threshold_assert_pass=True,
        paired_determinism_cost_ok=True,
        N_events_evaluable=200,
        mean_ratio=1.0,
        frac_steps_ratio_ge_2=0.0,
        deferred_survival_class="other",
        delta_never_apply=0.0,
        N_events_evaluable_early=100,
        N_events_evaluable_late=100,
        deferred_never_apply_within_H_frac=0.2,
        lcf=0.5,
        H_final=5.0,
        H_step25=5.0,
        retention_ok_flag=True,
    )
    kw.update(over)
    return kw


def test_classifier_matrix_R0_vacuous_missing_telemetry():
    v = classify_pressure_metric_family(**_base_kwargs(telemetry_ok=False))
    assert v["family"] == LABEL_R0
    assert "missing_telemetry" in v["stop_reason"]


def test_classifier_matrix_R0_vacuous_zero_evaluable_events():
    v = classify_pressure_metric_family(**_base_kwargs(N_events_evaluable=0))
    assert v["family"] == LABEL_R0
    assert "denominator_zero" in v["stop_reason"]


def test_classifier_matrix_R0_vacuous_determinism_or_threshold_fail():
    v = classify_pressure_metric_family(**_base_kwargs(two_tier_threshold_assert_pass=False))
    assert v["family"] == LABEL_R0
    v2 = classify_pressure_metric_family(**_base_kwargs(paired_determinism_cost_ok=False))
    assert v2["family"] == LABEL_R0
    v3 = classify_pressure_metric_family(**_base_kwargs(paired_determinism_cost_ok=None))
    assert v3["family"] == LABEL_R0
    assert "missing" in v3["stop_reason"]


def test_classifier_matrix_R1_growing():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=HIGH_DEMAND_RATIO,
            frac_steps_ratio_ge_2=SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
            deferred_survival_class="growing",
            delta_never_apply=GROWING_DEFERRED_SURVIVAL_DELTA,
        )
    )
    assert v["family"] == LABEL_R1


def test_classifier_matrix_R1_stable_high():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=3.0,
            frac_steps_ratio_ge_2=0.8,
            deferred_survival_class="stable_high",
            deferred_never_apply_within_H_frac=STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR,
        )
    )
    assert v["family"] == LABEL_R1


def test_classifier_matrix_R2_metric_mismatch():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=LOW_MODERATE_DEMAND_RATIO_MAX,
            lcf=HIGH_LCF,
            H_final=5.0 + MATERIAL_H_MOTION_BPW,
            H_step25=5.0,
            deferred_survival_class="other",
        )
    )
    assert v["family"] == LABEL_R2


def test_classifier_matrix_R3_representation_unresolved__low_pressure_low_H_motion():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.0,
            lcf=0.5,
            H_final=5.0,
            H_step25=5.0,
            retention_ok_flag=True,
        )
    )
    assert v["family"] == LABEL_R3
    assert "representation_limit" not in v["family"]


def test_classifier_matrix_R4_else_inconclusive():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.5,  # between moderate and high
            frac_steps_ratio_ge_2=0.0,
            lcf=0.5,
            H_final=5.0,
            H_step25=5.0,
            retention_ok_flag=False,
        )
    )
    assert v["family"] == LABEL_R4


def test_boundary_HIGH_DEMAND_RATIO_exact():
    # exactly at boundary with other R1 conjuncts → R1
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=HIGH_DEMAND_RATIO,
            frac_steps_ratio_ge_2=SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
            deferred_survival_class="growing",
        )
    )
    assert v["R1"] is True


def test_boundary_SUSTAINED_HIGH_DEMAND_FRAC_STEPS_exact():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=3.0,
            frac_steps_ratio_ge_2=SUSTAINED_HIGH_DEMAND_FRAC_STEPS,
            deferred_survival_class="growing",
        )
    )
    assert v["R1"] is True


def test_boundary_GROWING_DEFERRED_SURVIVAL_DELTA_exact():
    assert GROWING_DEFERRED_SURVIVAL_DELTA == 0.10


def test_boundary_STABLE_HIGH_FLOOR_exact():
    assert STABLE_HIGH_DEFERRED_NEVER_APPLY_FLOOR == 0.50


def test_boundary_LOW_MODERATE_DEMAND_RATIO_MAX_exact():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=LOW_MODERATE_DEMAND_RATIO_MAX,
            lcf=HIGH_LCF,
            H_final=5.0 + MATERIAL_H_MOTION_BPW,
            H_step25=5.0,
        )
    )
    assert v["family"] == LABEL_R2


def test_boundary_HIGH_LCF_exact():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.0,
            lcf=HIGH_LCF,
            H_final=5.0 + MATERIAL_H_MOTION_BPW,
            H_step25=5.0,
        )
    )
    assert v["family"] == LABEL_R2


def test_boundary_MATERIAL_H_MOTION_BPW_exact():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.0,
            lcf=HIGH_LCF,
            H_final=5.0 + MATERIAL_H_MOTION_BPW,
            H_step25=5.0,
        )
    )
    assert v["family"] == LABEL_R2
    v2 = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.0,
            lcf=HIGH_LCF,
            H_final=5.0 + MATERIAL_H_MOTION_BPW - 1e-6,
            H_step25=5.0,
            retention_ok_flag=True,
        )
    )
    assert v2["family"] != LABEL_R2


def test_boundary_REPRESENTATION_IMMOVABLE_H_DELTA_MAX_exact():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=1.0,
            lcf=0.5,
            H_final=5.0 + REPRESENTATION_IMMOVABLE_H_DELTA_MAX,
            H_step25=5.0,
            retention_ok_flag=True,
        )
    )
    assert v["family"] == LABEL_R3


def test_cohort_underflow_either_denom_lt_MIN_COHORT_N_blocks_R1():
    v = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=3.0,
            frac_steps_ratio_ge_2=0.9,
            deferred_survival_class="growing",
            N_events_evaluable_early=MIN_COHORT_N - 1,
            N_events_evaluable_late=MIN_COHORT_N,
        )
    )
    assert v["R1"] is False


def test_R1_R2_multi_match_prefers_R1():
    # Pure precedence selector: R1=True and R2_raw=True → R1 + multi_match
    v = select_family_from_predicates(r1=True, r2_raw=True, r3=False)
    assert v["family"] == LABEL_R1
    assert v["multi_match"] is True
    assert v["R1"] is True
    assert v["R2"] is True
    # Science path still prefers R1 when only R1 fires
    v2 = classify_pressure_metric_family(
        **_base_kwargs(
            mean_ratio=3.0,
            frac_steps_ratio_ge_2=0.9,
            deferred_survival_class="growing",
            lcf=HIGH_LCF,
            H_final=8.0,
            H_step25=5.0,
        )
    )
    assert v2["family"] == LABEL_R1


def _good_paired_proof():
    return {
        "path": "/tmp/paired_timing.json",
        "sha256": "abc",
        "protocol": "AB_BA_median_of_N",
        "overhead_frac_AB": 0.05,
        "overhead_frac_BA": 0.06,
        "determinism_prefix_match": True,
        "accepted": True,
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "device": "cpu",
        "N": 3,
        "steps": 25,
        "batch": 8,
        "topk": 1024,
        "is_proof": True,
        "two_tier_threshold_assert_pass": True,
    }


def _full_traj(steps: int = 50):
    """Build margin/episode/H/demand trajectories with exact boundaries."""
    bounds = expected_trajectory_boundaries(steps)
    margin = [
        {
            "step": s,
            "residual_margin_pre_cap_crossers": {"p10": 0.0, "p50": 1.0, "p90": 2.0, "n": 2},
            "residual_margin_applied_topk": {"p10": 0.0, "p50": 1.0, "p90": 2.0, "n": 1},
        }
        for s in bounds
    ]
    episode = [
        {
            "step": s,
            "active_episode_count": 2,
            "episode_age_quantiles_p10_p50_p90": {"p10": 1, "p50": 5, "p90": 10, "n": 2},
        }
        for s in bounds
    ]
    H = [{"step": s, "H_bits_per_weight": 3.5 if s == 25 else 4.0} for s in bounds]
    demand = [
        {
            "step": s,
            "candidate_crossers_before_cap": 2,
            "applied_count": 1,
            "demand_applied_ratio": 2.0,
        }
        for s in bounds
    ]
    return margin, episode, H, demand


def _good_bank():
    return {"before": PARENT_SHA256, "after": PARENT_SHA256, "match": True}


def _good_scale():
    return {"before": "scale0", "after": "scale0", "match": True}



def _good_probes():
    return {
        "acq_step0_count": 10,
        "acq_final_count": 10,
        "ret_step0_count": 10,
        "ret_final_count": 10,
        "retention_ok": True,
        "step0_taken_before_train": True,
    }

def _good_route():
    return {"n_fixed_qscale_forwards": 32, "n_bitlinear_dynamic_forwards": 0}


def test_aggregate_receipt_strict_json_real_output():
    st = _store(4, steps=50)
    for t in range(1, 51):
        cand = torch.tensor([True, True, False, False])
        app = torch.tensor([True, False, False, False])
        st.process_pre_writeback(
            candidate_masks={"w": cand},
            applied_masks={"w": app},
            step=t,
            n_candidates=2,
            n_applied=1,
        )
    st.finalize_window(final_step=50)
    margin_traj, episode_traj, H_traj, _demand = _full_traj(50)
    receipt = build_diagnostic_receipt(
        store=st,
        measurements={
            "n_flips": 10,
            "q_changed_count": 5,
            "credited_mass": 100,
            "lifetime_censored_frac": 0.95,
            "p50_flip_lifetime": 12.0,
            "H_bits_per_weight": 4.0,
            "H_trajectory": H_traj,
            "n_applied_drains": 10,
            "margin_trajectory": margin_traj,
            "episode_trajectory": episode_traj,
        },
        probes=_good_probes(),
        paired_determinism_cost_ok=True,
        paired_proof=_good_paired_proof(),
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        expected_parent_sha=PARENT_SHA256,
        steps=50,
        require_probes=True,
    )
    blob = json.dumps(receipt)
    assert "first_deferral_step" not in blob
    assert "episode_generation" not in blob
    assert "margin_trajectory" in receipt["measurements"]
    assert receipt["measurements"]["margin_trajectory"]
    assert "episode_trajectory" in receipt["measurements"]
    payload = sanitize_receipt_for_strict_json(receipt)
    dumped = json.dumps(payload, allow_nan=False)

    def _reject(c):
        raise ValueError(c)

    json.loads(dumped, parse_constant=_reject)
    assert "NaN" not in dumped
    assert receipt["classifier"]["family"] in {
        LABEL_R0, LABEL_R1, LABEL_R2, LABEL_R3, LABEL_R4
    }
    assert receipt["readiness"]["ok"] is True


def test_missing_paired_proof_yields_R0():
    st = _store(2, steps=50)
    st.per_step_ratios.append({"step": 1, "candidate_crossers_before_cap": 2, "applied_count": 1, "demand_applied_ratio": 2.0, "deferred_count": 1})
    margin_traj, episode_traj, H_traj, _ = _full_traj(50)
    receipt = build_diagnostic_receipt(
        store=st,
        measurements={
            "H_bits_per_weight": 4.0,
            "H_trajectory": H_traj,
            "lifetime_censored_frac": 0.9,
            "margin_trajectory": margin_traj,
            "episode_trajectory": episode_traj,
        },
        paired_determinism_cost_ok=True,
        paired_proof=None,
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        expected_parent_sha=PARENT_SHA256,
        steps=50,
        require_probes=False,
    )
    assert receipt["family"] == LABEL_R0
    assert "paired" in receipt["stop_reason"]


def test_missing_H_step25_yields_R0():
    st = _store(2, steps=50)
    st.per_step_ratios.append({"step": 1, "candidate_crossers_before_cap": 2, "applied_count": 1, "demand_applied_ratio": 2.0, "deferred_count": 1})
    margin_traj, episode_traj, _, _ = _full_traj(50)
    receipt = build_diagnostic_receipt(
        store=st,
        measurements={
            "H_bits_per_weight": 4.0,
            "H_trajectory": [{"step": 50, "H_bits_per_weight": 4.0}],
            "lifetime_censored_frac": 0.9,
            "margin_trajectory": margin_traj,
            "episode_trajectory": episode_traj,
        },
        paired_determinism_cost_ok=True,
        paired_proof=_good_paired_proof(),
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        expected_parent_sha=PARENT_SHA256,
        steps=50,
        require_probes=False,
    )
    assert receipt["family"] == LABEL_R0
    assert "H_step25" in receipt["stop_reason"] or "trajectory" in receipt["stop_reason"]


def test_readiness_bank_scale_route_negatives():
    margin, episode, H, demand = _full_traj(50)
    base = dict(
        expected_parent_sha=PARENT_SHA256,
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        paired_proof=_good_paired_proof(),
        paired_determinism_cost_ok=True,
        H_step25=3.5,
        required_telemetry={
            "demand": {"mean_ratio": 1.0},
            "deferred_survival": {"N_events_evaluable": 1},
            "margin_trajectory": margin,
            "episode_trajectory": episode,
            "demand_per_25": demand,
            "H_trajectory": H,
        },
        steps=50,
        require_probes=False,
    )
    assert evaluate_readiness(**base)["ok"] is True
    bad_bank = dict(base)
    bad_bank["banked_sha"] = {"before": "x", "after": "x", "match": True}
    assert evaluate_readiness(**bad_bank)["ok"] is False
    bad_scale = dict(base)
    bad_scale["frozen_scale_sha"] = {"before": "a", "after": "b", "match": False}
    assert evaluate_readiness(**bad_scale)["ok"] is False
    bad_route = dict(base)
    bad_route["route_counters"] = {"n_fixed_qscale_forwards": 0, "n_bitlinear_dynamic_forwards": 0}
    assert evaluate_readiness(**bad_route)["ok"] is False
    bad_paired = dict(base)
    bad_paired["paired_determinism_cost_ok"] = None
    assert evaluate_readiness(**bad_paired)["ok"] is False
    for key in ("demand", "deferred_survival", "margin_trajectory", "episode_trajectory", "demand_per_25", "H_trajectory"):
        bad_tel = dict(base)
        tel = dict(base["required_telemetry"])
        tel[key] = None
        bad_tel["required_telemetry"] = tel
        r = evaluate_readiness(**bad_tel)
        assert r["ok"] is False
        assert key in r["stop_reason"]


def test_active_episode_stats_quantiles():
    acc = {"w": torch.tensor([1.0, 0.0, -2.0, 3.0])}
    ep = {"w": torch.tensor([10, 0, 5, 20], dtype=torch.int32)}
    stats = active_episode_stats(acc, ep, step=30)
    assert stats["active_episode_count"] == 3
    ages = stats["episode_age_quantiles_p10_p50_p90"]
    assert ages["n"] == 3
    assert ages["p50"] is not None


def test_episode_trajectory_boundaries_in_receipt_allowlist():
    st = _store(2, steps=50)
    st.per_step_ratios.append({"step": 25, "candidate_crossers_before_cap": 1, "applied_count": 1, "demand_applied_ratio": 1.0, "deferred_count": 0})
    receipt = build_diagnostic_receipt(
        store=st,
        measurements={
            "H_bits_per_weight": 4.0,
            "H_trajectory": [{"step": 25, "H_bits_per_weight": 3.5}],
            "lifetime_censored_frac": 0.5,
            "margin_trajectory": [],
            "episode_trajectory": [],
        },
        paired_determinism_cost_ok=True,
        paired_proof=_good_paired_proof(),
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        expected_parent_sha=PARENT_SHA256,
        steps=50,
        require_probes=False,
    )
    assert "margin_trajectory" in receipt["measurements"]
    assert "episode_trajectory" in receipt["measurements"]
    assert receipt["family"] == LABEL_R0


def test_schema_only_cli_smoke(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "hrm_text_158_censor_null_pressure_metric_diagnostic.py"
    )
    out = tmp_path / "schema.json"
    r = subprocess.run(
        [
            sys.executable,
            str(script),
            "--schema-only",
            "--steps",
            "5",
            "--device",
            "cpu",
            "--output-json",
            str(out),
        ],
        cwd=str(Path(__file__).resolve().parents[3]),
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr
    data = json.loads(out.read_text())
    assert data["schema_only"] is True
    assert data["plan_sha256"].startswith("346b67d8")


def test_trajectory_schema_truncated_boundaries_R0():
    margin, episode, H, demand = _full_traj(50)
    bad_margin = margin[:-1]
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=bad_margin,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "boundaries" in r["stop_reason"]


def test_trajectory_schema_malformed_row_R0():
    margin, episode, H, demand = _full_traj(50)
    bad = list(margin)
    bad[0] = {"step": 25}
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=bad,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "missing_key" in r["stop_reason"]


def test_hash_scale_dict_detects_mutation():
    scales = {"a": torch.tensor(1.0), "b": torch.tensor(2.0)}
    before = hash_scale_dict(scales)
    scales["a"] = torch.tensor(1.5)
    after = hash_scale_dict(scales)
    assert before != after
    margin, episode, H, demand = _full_traj(50)
    r = evaluate_readiness(
        expected_parent_sha=PARENT_SHA256,
        banked_sha=_good_bank(),
        frozen_scale_sha={"before": before, "after": after, "match": before == after},
        route_counters=_good_route(),
        paired_proof=_good_paired_proof(),
        paired_determinism_cost_ok=True,
        H_step25=3.5,
        required_telemetry={
            "demand": {"mean_ratio": 1.0},
            "deferred_survival": {"N_events_evaluable": 1},
            "margin_trajectory": margin,
            "episode_trajectory": episode,
            "demand_per_25": demand,
            "H_trajectory": H,
        },
        steps=50,
        require_probes=False,
    )
    assert r["ok"] is False
    assert "frozen_scale" in r["stop_reason"]




def test_trajectory_schema_nested_quantile_malformed_R0():
    margin, episode, H, demand = _full_traj(50)
    bad = [dict(r) for r in margin]
    bad[0] = dict(bad[0])
    bad[0]["residual_margin_pre_cap_crossers"] = {"p10": 0.0, "p50": 1.0}  # missing p90/n
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=bad,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "quantile_missing" in r["stop_reason"]


def test_trajectory_schema_nonfinite_H_R0():
    margin, episode, H, demand = _full_traj(50)
    bad_h = [dict(r) for r in H]
    bad_h[0] = dict(bad_h[0])
    bad_h[0]["H_bits_per_weight"] = float("nan")
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=margin,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=bad_h,
    )
    assert r["ok"] is False
    assert "nonfinite" in r["stop_reason"]


def test_trajectory_schema_null_H_refuses():
    """rev5: H_bits_per_weight=None is an instrumentation failure, not valid."""
    margin, episode, H, demand = _full_traj(50)
    bad_h = [dict(r) for r in H]
    bad_h[0] = dict(bad_h[0])
    bad_h[0]["H_bits_per_weight"] = None
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=margin,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=bad_h,
    )
    assert r["ok"] is False
    assert "H_nonfinite" in r["stop_reason"]


def test_trajectory_schema_null_demand_ratio_refuses():
    """rev5: demand_applied_ratio=None must refuse (producer always emits float)."""
    margin, episode, H, demand = _full_traj(50)
    bad = [dict(r) for r in demand]
    bad[0] = dict(bad[0])
    bad[0]["demand_applied_ratio"] = None
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=margin,
        episode_trajectory=episode,
        demand_per_25=bad,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "demand_ratio_nonfinite" in r["stop_reason"]


def test_trajectory_schema_negative_episode_count_refuses():
    margin, episode, H, demand = _full_traj(50)
    bad = [dict(r) for r in episode]
    bad[0] = dict(bad[0])
    bad[0]["active_episode_count"] = -1
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=margin,
        episode_trajectory=bad,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "episode_count_invalid" in r["stop_reason"]


def test_trajectory_schema_episode_count_population_mismatch_refuses():
    """rev5: active count must equal its quantile population n."""
    margin, episode, H, demand = _full_traj(50)
    bad = [dict(r) for r in episode]
    bad[0] = dict(bad[0])
    bad[0]["active_episode_count"] = 7  # quantile n stays 2
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=margin,
        episode_trajectory=bad,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "episode_count_population_mismatch" in r["stop_reason"]


def test_trajectory_schema_margin_count_population_mismatch_refuses():
    """rev5: when a margin row carries n_candidates/n_applied, quantile n must match."""
    margin, episode, H, demand = _full_traj(50)
    bad = [dict(r) for r in margin]
    bad[0] = dict(bad[0])
    bad[0]["n_candidates"] = 99  # pre_cap quantile n stays 2
    bad[0]["n_applied"] = 1
    r = validate_trajectory_schemas(
        steps=50,
        margin_trajectory=bad,
        episode_trajectory=episode,
        demand_per_25=demand,
        H_trajectory=H,
    )
    assert r["ok"] is False
    assert "margin_count_population_mismatch" in r["stop_reason"]


def test_readiness_requires_step0_final_probe_fields():
    margin, episode, H, demand = _full_traj(50)
    base = dict(
        expected_parent_sha=PARENT_SHA256,
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        paired_proof=_good_paired_proof(),
        paired_determinism_cost_ok=True,
        H_step25=3.5,
        required_telemetry={
            "demand": {"mean_ratio": 1.0},
            "deferred_survival": {"N_events_evaluable": 1},
            "margin_trajectory": margin,
            "episode_trajectory": episode,
            "demand_per_25": demand,
            "H_trajectory": H,
        },
        steps=50,
        require_probes=True,
        probes={"acq_step0_count": 1},  # incomplete
    )
    r = evaluate_readiness(**base)
    assert r["ok"] is False
    assert "probes_missing_field" in r["stop_reason"]
    r2 = evaluate_readiness(**{**base, "probes": _good_probes()})
    assert r2["ok"] is True


def test_readiness_rejects_missing_threshold_proof_field():
    bad = dict(_good_paired_proof())
    del bad["two_tier_threshold_assert_pass"]
    margin, episode, H, demand = _full_traj(50)
    r = evaluate_readiness(
        expected_parent_sha=PARENT_SHA256,
        banked_sha=_good_bank(),
        frozen_scale_sha=_good_scale(),
        route_counters=_good_route(),
        paired_proof=bad,
        paired_determinism_cost_ok=True,
        H_step25=3.5,
        required_telemetry={
            "demand": {"mean_ratio": 1.0},
            "deferred_survival": {"N_events_evaluable": 1},
            "margin_trajectory": margin,
            "episode_trajectory": episode,
            "demand_per_25": demand,
            "H_trajectory": H,
        },
        steps=50,
        require_probes=False,
    )
    assert r["ok"] is False
    assert "two_tier" in r["stop_reason"] or "fields_missing" in r["stop_reason"]


def test_expected_boundaries_150():
    assert expected_trajectory_boundaries(150) == [25, 50, 75, 100, 125, 150]
    assert expected_trajectory_boundaries(50) == [25, 50]
