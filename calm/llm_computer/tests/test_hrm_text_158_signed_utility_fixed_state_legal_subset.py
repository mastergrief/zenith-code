"""CPU-static tests for D2c8 full-state legal-subset filter."""
from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    ESTIMAND_NAME, LegalSubsetError, TAG_RETAINED, characterize_plans_bidirectional_legal,
    clamp_acc_residual, encode_retained_record, enforce_legal_subset_support_floors,
    filter_plans_bidirectional_legal, is_full_state_bidirectionally_legal,
    payload_has_raw_index_arrays, simulate_public_apply_at_index,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import _canonical_sort, _le_hash
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdatePlan, VoteUpdateState, apply_integer_vote_update_from_frozen_plan,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_legal_subset.py"


@dataclass(frozen=True)
class _Plan:
    q_i16: torch.Tensor
    new_acc_i32: torch.Tensor
    candidate_indices: torch.Tensor
    pre_veto_selected_indices: torch.Tensor
    applied_indices: torch.Tensor
    applied_directions: torch.Tensor
    applied_thresholds: torch.Tensor
    replay_ce_veto_indices: torch.Tensor
    replay_veto_directions: torch.Tensor
    replay_veto_thresholds: torch.Tensor
    pc_aux_negative_indices: torch.Tensor
    pc_aux_veto_indices: torch.Tensor
    stats: dict


def _plan(indices, dirs, *, q=None, new_acc=None, thr=10, dir_dtype=torch.int16, thr_dtype=torch.int32):
    n = len(indices)
    empty = torch.zeros(0, dtype=torch.int64)
    q_t = torch.zeros(4, dtype=torch.int16) if q is None else torch.tensor(q, dtype=torch.int16)
    a_t = torch.zeros(4, dtype=torch.int32) if new_acc is None else torch.tensor(new_acc, dtype=torch.int32)
    return _Plan(
        q_i16=q_t.clone(), new_acc_i32=a_t.clone(),
        candidate_indices=torch.arange(4, dtype=torch.int64),
        pre_veto_selected_indices=torch.tensor(indices, dtype=torch.int64),
        applied_indices=torch.tensor(indices, dtype=torch.int64),
        applied_directions=torch.tensor(dirs, dtype=dir_dtype),
        applied_thresholds=torch.tensor([thr] * n if not isinstance(thr, list) else thr, dtype=thr_dtype),
        replay_ce_veto_indices=empty.clone(), replay_veto_directions=torch.zeros(0, dtype=torch.int16),
        replay_veto_thresholds=torch.zeros(0, dtype=torch.int32),
        pc_aux_negative_indices=empty.clone(), pc_aux_veto_indices=empty.clone(), stats={"flip_count": n},
    )


def _st(q, acc=None):
    return type("S", (), {
        "q_levels": torch.tensor(q, dtype=torch.int8),
        "exact_accumulator_shadow": torch.tensor(acc if acc is not None else [0] * len(q), dtype=torch.int16),
        "frozen_scale": torch.tensor(1.0),
    })()


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 250


def test_claude_geometry_q_pass_acc_fail_then_drop():
    assert is_full_state_bidirectionally_legal(
        prior_q=0, prior_acc=0, plan_q=0, plan_new_acc=10, d=1, thr=10
    ) is False
    qp, ap = simulate_public_apply_at_index(prior_q=0, prior_acc=0, plan_q=0, plan_new_acc=10, d=1, thr=10)
    qm, am = simulate_public_apply_at_index(prior_q=0, prior_acc=0, plan_q=0, plan_new_acc=10, d=-1, thr=10)
    assert (qp, ap) == (1, 0) and (qm, am) == (-1, 9)
    prior = {"k0": _st([0], [0])}
    plans = {"k0": _plan([0], [1], q=[0], new_acc=[10], thr=10)}
    _, rec = characterize_plans_bidirectional_legal(prior, plans)
    assert rec["support_floors"]["pass"] is False and rec["retained_total"] == 0
    with pytest.raises(LegalSubsetError, match="legal_subset_support_degenerate"):
        enforce_legal_subset_support_floors(rec)
    with pytest.raises(LegalSubsetError, match="legal_subset_support_degenerate"):
        filter_plans_bidirectional_legal(prior, plans)


def test_symmetric_new_acc_zero_green_and_hash_order():
    prior = {"k0": _st([0, 0, 0, 0], [0, 0, 0, 0])}
    plans_a = {"k0": _plan([2, 0, 1], [1, 1, 1], thr=10)}
    plans_b = {"k0": _plan([0, 1, 2], [1, 1, 1], thr=10)}
    fa, ra = filter_plans_bidirectional_legal(prior, plans_a)
    fb, rb = filter_plans_bidirectional_legal(prior, plans_b)
    assert ra["estimand"] == ESTIMAND_NAME
    assert ra["retained_stream_sha256"] == rb["retained_stream_sha256"]
    assert ra["applied_plan_index_direction_sha256"] == rb["applied_plan_index_direction_sha256"]
    assert int(fa["k0"].applied_indices.numel()) == 3
    assert torch.equal(fa["k0"].candidate_indices, plans_a["k0"].candidate_indices)
    assert ra["retained_stream_sha256"] == _le_hash(
        TAG_RETAINED, _canonical_sort([("k0".encode(), i, encode_retained_record("k0", i, 1, 10)) for i in (2, 0, 1)])
    )


def test_production_plan_dtypes_fail_closed():
    prior = {"k0": _st([0, 0, 0, 0])}
    with pytest.raises(LegalSubsetError, match="plan_dtype_mismatch:applied_directions"):
        characterize_plans_bidirectional_legal(prior, {"k0": _plan([0], [1], dir_dtype=torch.int64)})
    with pytest.raises(LegalSubsetError, match="plan_dtype_mismatch:applied_thresholds"):
        characterize_plans_bidirectional_legal(prior, {"k0": _plan([0], [1], thr=10, thr_dtype=torch.int64)})
    with pytest.raises(LegalSubsetError, match="plan_dtype_mismatch:applied_thresholds"):
        characterize_plans_bidirectional_legal(prior, {"k0": _plan([0], [1], thr=10, thr_dtype=torch.int16)})


def test_empty_retained_skew_json_and_nonempty_semantics():
    prior = {"k0": _st([0], [0])}
    _, rec = characterize_plans_bidirectional_legal(
        prior, {"k0": _plan([0], [1], q=[0], new_acc=[10], thr=10)})
    assert rec["all_keys_nonempty"] is False
    assert rec["support_floors"]["skew_defined"] is False
    assert rec["support_floors"]["skew_observed"] is None
    import json
    json.dumps(rec, allow_nan=False)
    prior5 = {"k0": _st([0, 0, 0, 0, 0], [0, 0, 0, 0, 0])}
    plan5 = _plan(
        [0, 1, 2, 3, 4], [1, 1, 1, 1, 1],
        q=[0, 0, 0, 0, 0], new_acc=[0, 10, 10, 10, 10], thr=10,
    )
    _, r5 = characterize_plans_bidirectional_legal(prior5, {"k0": plan5})
    assert r5["per_key"]["k0"]["retained_count"] == 1
    assert r5["all_keys_nonempty"] is True
    assert r5["support_floors"]["pass"] is False
    assert r5["support_floors"]["skew_defined"] is True


def test_int32_residual_matches_public_apply_boundaries():
    # Claude + public law: int32 wrap then clamp+int16
    assert clamp_acc_residual(2**31 - 1, -1, 10) == -9
    cases = [
        (0, 0, 0, 0, 1, 10), (0, 0, 0, 10, 1, 10), (0, 0, 0, 0, -1, 10),
        (1, 0, 1, 0, 1, 10), (-1, 0, -1, 0, -1, 10), (0, 5, 0, 5, 1, 10),
    ]
    for prior_q, prior_acc, plan_q, plan_new_acc, d, thr in cases:
        sim_q, sim_a = simulate_public_apply_at_index(
            prior_q=prior_q, prior_acc=prior_acc, plan_q=plan_q, plan_new_acc=plan_new_acc, d=d, thr=thr)
        q = torch.tensor([prior_q, 0, 0, 0], dtype=torch.int8)
        acc = torch.tensor([prior_acc, 0, 0, 0], dtype=torch.int16)
        state = VoteUpdateState(q_levels=q, accumulators=acc)
        empty = torch.zeros(0, dtype=torch.int64)
        plan = VoteUpdatePlan(
            q_i16=torch.tensor([plan_q, 0, 0, 0], dtype=torch.int16),
            new_acc_i32=torch.tensor([plan_new_acc, 0, 0, 0], dtype=torch.int32),
            candidate_indices=torch.arange(4, dtype=torch.int64),
            pre_veto_selected_indices=torch.tensor([0], dtype=torch.int64),
            applied_indices=torch.tensor([0], dtype=torch.int64),
            applied_directions=torch.tensor([d], dtype=torch.int16),
            applied_thresholds=torch.tensor([thr], dtype=torch.int32),
            replay_ce_veto_indices=empty, replay_veto_directions=torch.zeros(0, dtype=torch.int16),
            replay_veto_thresholds=torch.zeros(0, dtype=torch.int32),
            pc_aux_negative_indices=empty, pc_aux_veto_indices=empty, stats={},
        )
        out = apply_integer_vote_update_from_frozen_plan(state, plan)
        assert int(out.q_levels[0]) == sim_q and int(out.accumulators[0]) == sim_a


def test_replay_veto_nonempty_fail_closed():
    prior = {"k0": _st([0, 0, 0, 0])}
    plan = _plan([0], [1], thr=10)
    bad = replace(
        plan,
        replay_ce_veto_indices=torch.tensor([0], dtype=torch.int64),
        replay_veto_directions=torch.tensor([1], dtype=torch.int16),
        replay_veto_thresholds=torch.tensor([10], dtype=torch.int32),
    )
    with pytest.raises(LegalSubsetError, match="replay_veto_nonempty"):
        filter_plans_bidirectional_legal(prior, {"k0": bad})


def test_rank2_applied_and_replay_rejected():
    prior = {"k0": _st([0, 0, 0, 0])}
    bad = _plan([0], [1], thr=10)
    bad_rank = replace(
        bad,
        applied_indices=torch.tensor([[0]], dtype=torch.int64),
        applied_directions=torch.tensor([[1]], dtype=torch.int16),
        applied_thresholds=torch.tensor([[10]], dtype=torch.int32),
    )
    with pytest.raises(LegalSubsetError, match="plan_rank_mismatch:applied_indices"):
        characterize_plans_bidirectional_legal(prior, {"k0": bad_rank})
    bad_replay = replace(
        bad,
        replay_ce_veto_indices=torch.zeros((1, 0), dtype=torch.int64),
        replay_veto_directions=torch.zeros((1, 0), dtype=torch.int16),
        replay_veto_thresholds=torch.zeros((1, 0), dtype=torch.int32),
    )
    with pytest.raises(LegalSubsetError, match="plan_rank_mismatch:replay_ce_veto_indices"):
        characterize_plans_bidirectional_legal(prior, {"k0": bad_replay})
    prior = {"k0": _st([0, 0, 0, 0])}
    with pytest.raises(LegalSubsetError, match="plan_q_i16_not_bound_to_prior_q"):
        filter_plans_bidirectional_legal(prior, {"k0": _plan([0], [1], q=[1, 0, 0, 0], thr=10)})
    assert payload_has_raw_index_arrays({"candidate_indices": [1, 2, 3]}) is True
    assert payload_has_raw_index_arrays({"nested": {"applied_indices": [0]}}) is True
    assert payload_has_raw_index_arrays({"prod_changed_count": 2, "retained_stream_sha256": "a" * 64}) is False
