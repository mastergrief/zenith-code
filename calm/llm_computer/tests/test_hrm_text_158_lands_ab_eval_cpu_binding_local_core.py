"""LANDS-AB CPU binding local suite (IMPLEMENT_v12 split)."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
    LandsAbReducerSchemaError,
    all_true_matrix,
    matrix_with,
    reduce_lands_ab_branch_strict,
)
from calm.llm_computer.tests.lands_ab_eval_test_helpers import (
    base_ok as _base_ok,
    write_real_cpu_row as _write_real_cpu_row,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_EQUIVALENT,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_VACUOUS,
    CANONICAL_CELL_KEYS,
)














def test_dense_moves_must_not_be_derived_from_credit():
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        credit_from_weighted_grad,
        default_dry_run_rank_vote_spec,
        project_s1_gradient_to_moves,
        rank_bucketed_int16_votes,
    )
    g = torch.randn(4, 4)
    q = torch.zeros(4, 4, dtype=torch.int8)
    credit = credit_from_weighted_grad(g)
    moves = project_s1_gradient_to_moves(g, q)
    votes = rank_bucketed_int16_votes(credit, moves, default_dry_run_rank_vote_spec())
    assert votes.dtype == torch.int16


def test_rank_spec_drift_rejected():
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        canonical_acquisition_rank_vote_spec,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
        require_canonical_rank_spec,
        two_branch_dense_votes,
    )
    alt = canonical_acquisition_rank_vote_spec()
    with pytest.raises(ValueError, match="rank_spec"):
        require_canonical_rank_spec(alt)
    with pytest.raises(ValueError, match="rank_spec"):
        two_branch_dense_votes(torch.randn(2, 2), torch.zeros(2, 2, dtype=torch.int8), alt)


def test_twin_state_no_alias_and_prestate_equal():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
        build_twin_states_from_prior,
        logical_int16_accumulator,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        derive_trainer_sub2_authority_states,
        select_trainer_eligible_bitlinears,
    )
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)
        def forward(self, x):
            return self.lin(x)
    prior = derive_trainer_sub2_authority_states(
        select_trainer_eligible_bitlinears(_Tiny(), use_ternary_bulk=True)
    )
    twins = build_twin_states_from_prior(prior)
    for k in prior:
        s, d = twins["sparse"][k], twins["dense"][k]
        assert s.exact_accumulator_shadow is None
        assert d.exact_accumulator_shadow is not None
        assert s.bounded_accumulator is not prior[k].bounded_accumulator
        assert torch.equal(logical_int16_accumulator(s), logical_int16_accumulator(d))


def test_g_cpu_static_ab_seed158_exact_polarities():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
        measure_g_cpu_static_ab,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    obs = measure_g_cpu_static_ab()
    assert obs["synthetic_only"] is False
    cells = recompute_surface_cells_from_primitives(
        gating_row="G_CPU_STATIC_AB",
        metrics=obs["metrics"],
        key_universe=obs["key_universe"],
        fixture_contract_raw_fail=False,
    )
    assert cells == {"s1": True, "s2": True, "s3": True, "s4": True, "s6": True}
    scratch = Path(os.environ.get("LANDS_AB_RUNTIME_SCRATCH", "/tmp/lands_ab_runtime_scratch")) / uuid.uuid4().hex
    path = runtime_scratch_raw_path(scratch_dir=scratch, gating_row="G_CPU_STATIC_AB", run_nonce="v5")
    sha = o_excl_write_json(path, obs)
    assert len(sha) == 64
    with pytest.raises(FileExistsError):
        o_excl_write_json(path, obs)


def test_p1_hash_cotamper_forces_s3_false_or_reject():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import measure_g_cpu_static_ab
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    obs = measure_g_cpu_static_ab()
    bad = dict(obs["metrics"])
    bad["q_match"] = True
    pq = dict(bad["post_q_sha256_by_key"])
    k = next(iter(pq))
    pq[k] = {"sparse": pq[k]["sparse"], "dense": "0" * 64}
    bad["post_q_sha256_by_key"] = pq
    with pytest.raises(ValueError, match="aggregate_vs_primitive_mismatch"):
        recompute_surface_cells_from_primitives(
            gating_row="G_CPU_STATIC_AB",
            metrics=bad,
            key_universe=obs["key_universe"],
            fixture_contract_raw_fail=False,
        )
    bad2 = dict(obs["metrics"])
    del bad2["q_match"]
    pq = dict(bad2["post_q_sha256_by_key"])
    k = next(iter(pq))
    pq[k] = {"sparse": pq[k]["sparse"], "dense": "0" * 64}
    bad2["post_q_sha256_by_key"] = pq
    cells = recompute_surface_cells_from_primitives(
        gating_row="G_CPU_STATIC_AB",
        metrics=bad2,
        key_universe=obs["key_universe"],
        fixture_contract_raw_fail=False,
    )
    assert cells["s3"] is False


def test_production_binding_rejects_count_only_q_hash_mismatch():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_local_update,
    )
    production = {
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 5,
        "q_changed_count": 2,
        "production_post_q_sha256_by_key": {"lin": "a" * 64},
        "production_applied_row_identities_sha256_by_key": {"lin": "b" * 64},
    }
    compare = {
        "sparse_event_count": 5,
        "q_changed_count_sparse": 2,
        "post_q_sha256_by_key": {"lin": {"sparse": "c" * 64, "dense": "c" * 64}},
    }
    bind = bind_production_to_twin_local_update(production=production, compare=compare)
    assert bind["production_sparse_matches_twin"] is False
    assert bind["hash_ok"] is False
    # matching q WITHOUT logical-acc must FAIL (v8 fail-closed)
    production2 = dict(production)
    production2["production_post_q_sha256_by_key"] = {"lin": "c" * 64}
    bind2 = bind_production_to_twin_local_update(production=production2, compare=compare)
    assert bind2["production_sparse_matches_twin"] is False
    assert bind2.get("reason") == "production_logical_acc_not_ro_observable" or bind2.get("logical_acc_ok") is False
    # matching q AND logical-acc pass
    production3 = dict(production2)
    production3["production_post_logical_acc_sha256_by_key"] = {"lin": "d" * 64}
    compare3 = dict(compare)
    compare3["post_logical_acc_sha256_by_key"] = {"lin": {"sparse": "d" * 64, "dense": "d" * 64}}
    bind3 = bind_production_to_twin_local_update(production=production3, compare=compare3)
    assert bind3["production_sparse_matches_twin"] is True
    assert bind3["logical_acc_ok"] is True


def test_probe_q_only_without_logical_acc_rejected():
    """co_lead probe 1: production post-q equal but missing logical-acc → match False."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_local_update,
    )
    production = {
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 4,
        "q_changed_count": 1,
        "production_post_q_sha256_by_key": {"lin": "a" * 64},
        "production_post_logical_acc_sha256_by_key": {},  # missing
        "production_applied_row_identities_sha256_by_key": {"lin": "b" * 64},
    }
    compare = {
        "sparse_event_count": 4,
        "q_changed_count_sparse": 1,
        "post_q_sha256_by_key": {"lin": {"sparse": "a" * 64, "dense": "a" * 64}},
        "post_logical_acc_sha256_by_key": {"lin": {"sparse": "c" * 64, "dense": "c" * 64}},
    }
    bind = bind_production_to_twin_local_update(production=production, compare=compare)
    assert bind["production_sparse_matches_twin"] is False
    assert bind.get("logical_acc_ok") is False


def test_probe_logical_acc_value_mismatch_rejected():
    """co_lead probe 2: q matches, logical-acc VALUES differ → match False."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_local_update,
    )
    production = {
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 4,
        "q_changed_count": 1,
        "production_post_q_sha256_by_key": {"lin": "a" * 64},
        "production_post_logical_acc_sha256_by_key": {"lin": "c" * 64},
        "production_applied_row_identities_sha256_by_key": {"lin": "b" * 64},
    }
    compare = {
        "sparse_event_count": 4,
        "q_changed_count_sparse": 1,
        "post_q_sha256_by_key": {"lin": {"sparse": "a" * 64, "dense": "a" * 64}},
        "post_logical_acc_sha256_by_key": {"lin": {"sparse": "d" * 64, "dense": "d" * 64}},
    }
    bind = bind_production_to_twin_local_update(production=production, compare=compare)
    assert bind["production_sparse_matches_twin"] is False
    assert bind["hash_ok"] is True  # q ok
    assert bind["logical_acc_ok"] is False  # acc mismatch
