"""CPU-static reducer tests for vote-lifetime screen (rider-2/3 bindings)."""
from __future__ import annotations

import torch

from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    FAMILY_F1,
    FAMILY_F2,
    FAMILY_F3,
    FAMILY_F4,
    accumulate_hazard,
    age_rate_partition,
    apply_drain_resets,
    classify_forgetting_family,
    count_censored_active_episodes,
    empty_hazard_table,
    lifetime_censored_frac,
    update_episode_starts,
)


def test_episode_start_on_zero_to_nonzero():
    prev = torch.tensor([0, 0], dtype=torch.int16)
    new = torch.tensor([1, 0], dtype=torch.int16)
    ep = torch.tensor([0, 0], dtype=torch.int32)
    out = update_episode_starts(prev, new, ep, step=7)
    assert out.tolist() == [7, 0]


def test_episode_reset_on_return_to_zero():
    prev = torch.tensor([3, -2], dtype=torch.int16)
    new = torch.tensor([0, -2], dtype=torch.int16)
    ep = torch.tensor([2, 4], dtype=torch.int32)
    out = update_episode_starts(prev, new, ep, step=9)
    assert out.tolist() == [0, 4]


def test_episode_reset_on_sign_reversal_starts_new():
    prev = torch.tensor([5, -3, 2], dtype=torch.int16)
    new = torch.tensor([-1, -4, 2], dtype=torch.int16)
    ep = torch.tensor([1, 2, 3], dtype=torch.int32)
    out = update_episode_starts(prev, new, ep, step=11)
    assert out.tolist() == [11, 2, 3]


def test_applied_drain_records_lifetime_and_clears():
    acc = torch.tensor([12, 3, -15], dtype=torch.int16)
    ep = torch.tensor([4, 8, 5], dtype=torch.int32)
    drain = torch.tensor([True, False, True])
    new_acc, new_ep, lifetimes = apply_drain_resets(acc, ep, drain, step=14)
    assert new_acc.tolist() == [0, 3, 0]
    assert new_ep.tolist() == [0, 8, 0]
    assert sorted(lifetimes) == [9, 10]  # 14-5=9, 14-4=10


def test_lifetime_censored_frac_exact_half_boundary():
    # n_censored=1, n_flips=1 → 0.50 exactly
    assert lifetime_censored_frac(1, 1) == 0.50
    assert lifetime_censored_frac(0, 0) == 0.0  # max(1,0+0)=1 → 0/1
    assert lifetime_censored_frac(3, 1) == 0.25


def test_post_flip_active_episode_counts_in_censor():
    # Weight flipped earlier; new episode alive at window end.
    final_acc = torch.tensor([0, 4, -2], dtype=torch.int16)
    episode_start = torch.tensor([0, 40, 90], dtype=torch.int32)
    n_censored = count_censored_active_episodes(final_acc, episode_start)
    assert n_censored == 2
    # Suppose 3 flips earlier in window:
    assert lifetime_censored_frac(3, n_censored) == 2 / 5


def test_classifier_r4_vacuous_n_flips():
    out = classify_forgetting_family(
        n_flips=10,
        credited_mass=50_000,
        p50_flip_lifetime=4.0,
        never_convert_frac=0.9,
        age_rate_old=0.0,
        age_rate_young=0.1,
        lifetime_censored_frac_value=0.0,
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "R4_vacuous"


def test_classifier_r4_vacuous_credited_mass():
    out = classify_forgetting_family(
        n_flips=5000,
        credited_mass=100,
        p50_flip_lifetime=4.0,
        never_convert_frac=0.9,
        age_rate_old=0.0,
        age_rate_young=0.1,
        lifetime_censored_frac_value=0.0,
    )
    assert out["family"] == FAMILY_F4
    assert out["predicates"]["R4_vacuous"] is True


def test_classifier_censor_guard_at_exact_0_50():
    out = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=32.0,
        never_convert_frac=0.1,
        age_rate_old=0.1,
        age_rate_young=0.1,
        lifetime_censored_frac_value=0.50,
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "censor_guard"
    assert out["predicates"]["censor_guard"] is True


def test_classifier_boundaries_r1_r2_r3():
    # R1 → F2 at exact thresholds
    r1 = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=8.0,
        never_convert_frac=0.50,
        age_rate_old=0.025,
        age_rate_young=0.1,  # ratio 0.25
        lifetime_censored_frac_value=0.0,
    )
    assert r1["family"] == FAMILY_F2
    assert r1["predicates"]["R1"] is True

    # R2 → F1
    r2 = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=32.0,
        never_convert_frac=0.1,
        age_rate_old=0.06,
        age_rate_young=0.1,  # ratio 0.60
        lifetime_censored_frac_value=0.0,
    )
    assert r2["family"] == FAMILY_F1
    assert r2["predicates"]["R2"] is True

    # R3 → F3
    r3 = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=16.0,
        never_convert_frac=0.70,
        age_rate_old=0.0,
        age_rate_young=0.1,
        lifetime_censored_frac_value=0.0,
    )
    assert r3["family"] == FAMILY_F3
    assert r3["predicates"]["R3"] is True


def test_classifier_multi_match_prefers_f3():
    # Force R1+R3 both true: p50=8, never_convert=0.70, young-dominated ratio
    out = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=8.0,
        never_convert_frac=0.70,
        age_rate_old=0.01,
        age_rate_young=0.1,  # ratio 0.1 <= 0.25 → R1; p50<=16 & never>=0.70 → R3
        lifetime_censored_frac_value=0.0,
    )
    assert out["predicates"]["R1"] is True
    assert out["predicates"]["R3"] is True
    assert out["multi_match"] is True
    assert out["family"] == FAMILY_F3


def test_classifier_no_match_is_f4():
    out = classify_forgetting_family(
        n_flips=5000,
        credited_mass=50_000,
        p50_flip_lifetime=20.0,
        never_convert_frac=0.2,
        age_rate_old=0.05,
        age_rate_young=0.1,
        lifetime_censored_frac_value=0.0,
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "no_rule_match"


def test_hazard_denominator_young_old_partition():
    """Known ages + flip mask → per-bin exposure/flips and aggregate rates."""
    # ages: two young (<=8), three old (>=33); flips on one young + two old
    ages = torch.tensor([1, 8, 33, 40, 65], dtype=torch.int64)
    flip_mask = torch.tensor([True, False, True, True, False])
    table = empty_hazard_table()
    accumulate_hazard(table, ages, flip_mask)

    assert table["0-1"]["exposure"] == 1
    assert table["0-1"]["flips"] == 1
    assert table["5-8"]["exposure"] == 1
    assert table["5-8"]["flips"] == 0
    assert table["33-64"]["exposure"] == 2
    assert table["33-64"]["flips"] == 2
    assert table["65+"]["exposure"] == 1
    assert table["65+"]["flips"] == 0

    old_rate, young_rate = age_rate_partition(table)
    # young bins <=8: flips=1, exposure=2 → 0.5
    # old bins >=33: flips=2, exposure=3 → 2/3
    assert young_rate == 0.5
    assert abs(old_rate - (2.0 / 3.0)) < 1e-12
