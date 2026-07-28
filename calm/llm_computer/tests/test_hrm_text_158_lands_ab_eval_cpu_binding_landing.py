"""CPU landing/oracle / B3 binding tests for LANDS-AB (IMPLEMENT_v10 split)."""
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














def test_b3_binding_requires_p1b_pass_receipt():
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_landing,
    )
    production = {
        "p1b_pass_receipt": False,
        "builder_receipt_pass": False,
        "total_sparse_vote_event_count": 3,
        "q_changed_count": 1,
        "post_update_payload_sha256": "d" * 64,
        "weighted_grad_capture_sha256_by_key": {"lin": "e" * 64},
    }
    compare = {
        "sparse_event_count": 3,
        "q_changed_count_sparse": 1,
        "twin_post_authoritative_state_payload_sha256": "d" * 64,
    }
    cap = {"lin": "e" * 64}
    bind = bind_production_to_twin_landing(
        production=production, compare=compare, capture_wg_sha_by_key=cap
    )
    assert bind["production_sparse_matches_twin"] is False
    production["p1b_pass_receipt"] = True
    production["builder_receipt_pass"] = True
    bind2 = bind_production_to_twin_landing(
        production=production, compare=compare, capture_wg_sha_by_key=cap
    )
    assert bind2["production_sparse_matches_twin"] is True
    # VALUE mismatch on WG fails
    bind3 = bind_production_to_twin_landing(
        production=production, compare=compare, capture_wg_sha_by_key={"lin": "f" * 64}
    )
    assert bind3["production_sparse_matches_twin"] is False
    assert bind3["wg_ok"] is False
    # existence-only payload (no twin) fails
    bind4 = bind_production_to_twin_landing(
        production=production,
        compare={"sparse_event_count": 3, "q_changed_count_sparse": 1},
        capture_wg_sha_by_key=cap,
    )
    assert bind4["production_sparse_matches_twin"] is False
    assert bind4["payload_ok"] is False
    # twin VALUE mismatch fails
    bind5 = bind_production_to_twin_landing(
        production=production,
        compare={
            "sparse_event_count": 3,
            "q_changed_count_sparse": 1,
            "twin_post_authoritative_state_payload_sha256": "0" * 64,
        },
        capture_wg_sha_by_key=cap,
    )
    assert bind5["production_sparse_matches_twin"] is False
    assert bind5["payload_reason"] == "payload_value_mismatch"


def test_fused_only_builder_cannot_mint_s5():
    """Blocker 3 hostile: fused-only named builder cannot set oracle_mode_on_named_site."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        extract_oracle_from_builder_receipt,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        build_trainer_sub2_authority_local_update_receipt,
    )
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)
        def forward(self, x):
            return self.lin(x)
    def loss(m, b):
        return (m(b["x"]) ** 2).mean()
    r = build_trainer_sub2_authority_local_update_receipt(
        model=_Tiny(),
        batch={"x": torch.randn(2, 4)},
        forward_loss_fn=loss,
        use_ternary_bulk=True,
        device=torch.device("cpu"),
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    named = extract_oracle_from_builder_receipt(r)
    assert named.get("resolved_mode") != "oracle_on" or not named.get("events_equal_by_key")
    # fused_only → oracle_mode_on_named_site must be false under v7 observation rule
    oracle_mode_on_named_site = named.get("resolved_mode") == "oracle_on"
    assert oracle_mode_on_named_site is False
    # S5 requires oracle_mode_on_named_site True
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    metrics = {
        "events_equal_by_key": {"lin": True},
        "events_equal_fused_vs_dense_derived": True,
        "independent_two_branch_recompute_ok": True,
        "dense_derived_provenance": "two_branch_parallel_dense_vote_derivation",
        "d1_densify_from_sparse_used": False,
        "sparse_vote_authority_mode": "oracle_on",
        "votes_by_key_applied": None,
        "builder_receipt_pass": True,
        "oracle_mode_on_named_site": False,  # fused-only observation
    }
    with pytest.raises(ValueError, match="oracle_mode_not_on_named_site"):
        recompute_surface_cells_from_primitives(
            gating_row="G_CUDA_ORACLE_B1",
            metrics=metrics,
            key_universe=["lin"],
            fixture_contract_raw_fail=False,
        )


def test_swallowed_exception_cannot_pass_oracle_builder():
    """Builder failure must surface as fixture_contract / not silent S5 true."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_cuda_sites import (
        measure_oracle_at_production_site,
    )
    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)
        def forward(self, x):
            return self.lin(x)
    def loss(m, b):
        return (m(b["x"]) ** 2).mean()
    def bad_runner():
        raise RuntimeError("forced_builder_fail")
    obs = measure_oracle_at_production_site(
        gating_row="G_CUDA_ORACLE_B1",
        model=_Tiny(),
        batch={"x": torch.randn(2, 4)},
        forward_loss_fn=loss,
        device=torch.device("cpu"),
        site_tag="oracle_B1_local",
        production_site="build_trainer_sub2_authority_local_update_receipt",
        site_runner=bad_runner,
    )
    assert obs["fixture_contract_raw_fail"] is True
    assert obs["metrics"].get("builder_exception")
    assert obs["metrics"].get("oracle_mode_on_named_site") is False
    assert obs["measured_surfaces"]["s5"] is False


def test_probe_b3_wg_keyset_only_without_value_equality_rejected():
    """WG key-set-only match without VALUE equality cannot pass."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_landing,
    )
    production = {
        "p1b_pass_receipt": True,
        "builder_receipt_pass": True,
        "total_sparse_vote_event_count": 3,
        "q_changed_count": 1,
        "post_update_payload_sha256": "d" * 64,
        "weighted_grad_capture_sha256_by_key": {"lin": "e" * 64},
    }
    compare = {
        "sparse_event_count": 3,
        "q_changed_count_sparse": 1,
        "twin_post_authoritative_state_payload_sha256": "d" * 64,
    }
    # same keys, different values
    bind = bind_production_to_twin_landing(
        production=production,
        compare=compare,
        capture_wg_sha_by_key={"lin": "0" * 64},
    )
    assert bind["production_sparse_matches_twin"] is False
    assert bind["wg_ok"] is False
    # empty capture fail-closed
    bind2 = bind_production_to_twin_landing(
        production=production, compare=compare, capture_wg_sha_by_key=None
    )
    assert bind2["production_sparse_matches_twin"] is False




def test_named_map_absent_not_rescued_by_generic_path_oracle():
    """Hostile: named events_equal map absent + generic TRUE must NOT rescue S5."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    # named map absent → schema filled with False keys; fused_vs False; fixture fail
    metrics = {
        "events_equal_by_key": {"lin": False},  # named-absent filler, not path_oracle TRUE
        "events_equal_fused_vs_dense_derived": False,
        "independent_two_branch_recompute_ok": True,  # generic recompute true
        "dense_derived_provenance": "two_branch_parallel_dense_vote_derivation",
        "d1_densify_from_sparse_used": False,
        "sparse_vote_authority_mode": "oracle_on",
        "votes_by_key_applied": None,
        "builder_receipt_pass": True,
        "oracle_mode_on_named_site": True,
        "named_events_equal_map_present": False,
        "path_oracle_fallback_used": False,
    }
    cells = recompute_surface_cells_from_primitives(
        gating_row="G_CUDA_ORACLE_B3",
        metrics=metrics,
        key_universe=["lin"],
        fixture_contract_raw_fail=True,
    )
    assert cells["s5"] is False
    # even without fixture fail, False map cannot mint S5
    cells2 = recompute_surface_cells_from_primitives(
        gating_row="G_CUDA_ORACLE_B3",
        metrics=metrics,
        key_universe=["lin"],
        fixture_contract_raw_fail=False,
    )
    assert cells2["s5"] is False
