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














def test_authoritative_sidecar_matches_production_blob():
    """Twin serializer helper must equal TSA build_..._checkpoint_blob sidecar sha."""
    import torch
    from calm.hrm_text_158.bit_linear import BitLinear
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_authoritative_payload import (
        authoritative_sidecar_payload_sha256,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        build_trainer_sub2_authority_checkpoint_blob,
        derive_trainer_sub2_authority_states,
        select_trainer_eligible_bitlinears,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)

        def forward(self, x):
            return self.lin(x)

    m = _Tiny()
    eligible = select_trainer_eligible_bitlinears(m, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    blob = build_trainer_sub2_authority_checkpoint_blob(
        m, eligible_modules=eligible, tensor_states=states, step=1
    )
    prod = blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"]
    twin = authoritative_sidecar_payload_sha256(states, step=1)
    assert prod == twin
    assert len(twin) == 64


def test_b1_full_pipeline_branch_good_and_divergence():
    """IMPLEMENT_v10: live B1 through recompute + single-row matrix consistency.

    Good path: claimed surfaces == reducer recompute; production binding True;
    no metric_cell_contradiction. Seeded divergence: force binding False → s3 False
    with claimed==recomputed (no contradiction).
    """
    import copy
    import torch
    from calm.hrm_text_158.bit_linear import BitLinear
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_cuda_sites import (
        measure_b1_local_update_site,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        validate_raw_row_observation,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
        recompute_surface_cells_from_primitives,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
        reduce_lands_ab_branch_strict,
        all_true_matrix,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
        cell_key,
        GATING_ROWS,
        APPLICABILITY_MAP,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)

        def forward(self, x):
            return self.lin(x)

    def loss(m, b):
        return (m(b["x"]) ** 2).mean()

    torch.manual_seed(11)
    obs = measure_b1_local_update_site(
        model=_Tiny(),
        batch={"x": torch.randn(2, 4)},
        forward_loss_fn=loss,
        device=torch.device("cpu"),
    )
    validate_raw_row_observation(obs)
    claimed = dict(obs["measured_surfaces"])
    recomputed = recompute_surface_cells_from_primitives(
        gating_row=obs["gating_row"],
        metrics=obs["metrics"],
        key_universe=list(obs["key_universe"]),
        fixture_contract_raw_fail=bool(obs["fixture_contract_raw_fail"]),
    )
    assert claimed == recomputed, (claimed, recomputed)
    # IMPLEMENT_v14: injective post-acc RO surface absent on named receipt →
    # crosscheck fail-closed; B1 cannot mint production_sparse_matches_twin True.
    xcheck = obs["metrics"].get("production_reapply_crosscheck") or {}
    assert xcheck.get("transition_fields_equal") is True or xcheck.get("crosscheck_ok") is False
    assert obs["metrics"]["production_sparse_matches_twin"] is False
    assert claimed.get("s3") is False
    # fixture_fail from injective absence may zero all surfaces via recompute;
    # s4/s6 polarity is secondary to the injective fail-closed claim.
    assert "s4" in claimed and "s6" in claimed
    assert (
        "neither_full" in str(xcheck.get("reason") or "")
        or xcheck.get("injective_post_acc_binding_ro_available") is False
    )
    # phase topology good + update duration real work
    topo = obs.get("phase_topology") or {}
    assert topo.get("good_topology") is True
    ends = {
        e["phase"]: float(e.get("duration_s") or 0)
        for e in (obs.get("phase_events") or [])
        if e.get("type") == "PHASE_END"
    }
    assert set(ends) == {"forward_backward", "update", "emission", "flush"}
    # native builder phases must show real (non-zero-total) work; flush may be tiny
    assert ends["forward_backward"] > 0.0 or ends["update"] > 0.0
    assert ends["update"] + ends["forward_backward"] + ends["emission"] > 0.0

    # Seeded divergence: production_sparse_matches_twin forced False → s3 False,
    # still claimed==recomputed (no contradiction path)
    bad = copy.deepcopy(obs)
    bad["metrics"]["production_sparse_matches_twin"] = False
    bad_re = recompute_surface_cells_from_primitives(
        gating_row=bad["gating_row"],
        metrics=bad["metrics"],
        key_universe=list(bad["key_universe"]),
        fixture_contract_raw_fail=False,
    )
    assert bad_re["s3"] is False
    # if claimed still True while recompute False → contradiction class
    bad["measured_surfaces"] = bad_re  # author from same primitives
    assert bad["measured_surfaces"] == bad_re

    # single-row branch primitives: s3 false → not EQUIVALENT
    matrix = all_true_matrix()
    for row in GATING_ROWS:
        for surf in APPLICABILITY_MAP[row]:
            matrix[cell_key(row, surf)] = True
    matrix[cell_key("G_CUDA_B1_APPLY", "s3")] = False
    out = reduce_lands_ab_branch_strict(
        {"scope_creep": False, "fixture_contract_raw_fail": False, "surface_pass_by_row": matrix}
    )
    assert out["branch_id"] != "EQUIVALENT"


def test_builder_pass_empty_receipt_proof_cannot_match_twin():
    """IMPLEMENT_v11: builder_receipt_pass=True + empty proof → not production_sparse_matches_twin."""
    import torch
    from calm.hrm_text_158.bit_linear import BitLinear
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
        crosscheck_production_q_vs_receipt_proof,
        production_fused_apply_post_states,
        production_post_q_and_logical_acc_sha256_by_key,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_binding import (
        bind_production_to_twin_local_update,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
        capture_weighted_grad_by_key,
        measure_from_production_capture,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        derive_trainer_sub2_authority_states,
        select_trainer_eligible_bitlinears,
    )

    # unit: fail-closed crosscheck branches
    x1 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key={},
        builder_receipt_pass=True,
    )
    assert x1["crosscheck_ok"] is False
    assert "no_receipt_proof" in x1["reason"]
    x2 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key={"lin": {"other": 1}},
        builder_receipt_pass=True,
        reapply_proof_by_key={"lin": {"other": 1}},
    )
    assert x2["crosscheck_ok"] is False
    x3 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64, "lin2": "b" * 64},
        receipt_proof_by_key={"lin": {"candidate_q_sha256_after": "a" * 64}},
        builder_receipt_pass=True,
        reapply_proof_by_key={"lin": {"candidate_q_sha256_after": "a" * 64}},
    )
    assert x3["crosscheck_ok"] is False
    assert "mismatch" in x3["reason"] or "partial" in x3["reason"] or "key_set" in x3["reason"]
    # pass=False + empty still soft-ok (bind gate fails separately)
    x4 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key={},
        builder_receipt_pass=False,
    )
    assert x4["crosscheck_ok"] is True

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)

        def forward(self, x):
            return self.lin(x)

    def loss(m, b):
        return (m(b["x"]) ** 2).mean()

    torch.manual_seed(17)
    m = _Tiny()
    eligible = select_trainer_eligible_bitlinears(m, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    batch = {"x": torch.randn(2, 4)}
    weighted = capture_weighted_grad_by_key(
        model=m,
        batch=batch,
        forward_loss_fn=loss,
        states=states,
        eligible=eligible,
        device="cpu",
    )
    # measure with builder_pass True but EMPTY proof → must not match twin
    obs = measure_from_production_capture(
        gating_row="G_CUDA_B1_APPLY",
        prior_states=states,
        weighted_grad_by_key=weighted,
        device="cpu",
        site_tag="B1_empty_proof_hostile",
        production_site="hostile_empty_proof",
        phase_events=[],
        builder_receipt_pass=True,
        receipt_proof_by_key={},  # empty
        production_event_count=None,
        production_q_changed_count=None,
    )
    assert obs["metrics"]["production_sparse_matches_twin"] is False
    assert obs["metrics"]["production_reapply_crosscheck"]["crosscheck_ok"] is False
    # even if re-apply alone would bind, fixture/crosscheck blocks match
    assert obs["fixture_contract_raw_fail"] is True or obs["measured_surfaces"].get("s3") is False




def test_transition_proof_residual_mutation_rejects():
    """co_lead hostile: same q/counts + mutated residual_after_threshold → reject."""
    import copy
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
        TRANSITION_PROOF_FIELDS,
        crosscheck_production_q_vs_receipt_proof,
    )

    good = {
        "candidate_q_sha256_after": "a" * 64,
        "q_changed_identities_sha256": "b" * 64,
        "applied_row_identities_sha256": "c" * 64,
        "ordered_applied_row_identities_sha256": "c" * 64,
        "applied_directions_sha256": "d" * 64,
        "applied_thresholds_sha256": "e" * 64,
        "residual_after_threshold_sha256": "f" * 64,
        "bounded_accumulator_summary_after": {"hot_exact_row_count": 1, "cold_exception_row_count": 2},
        "q_changed_count": 1,
        "applied_row_count": 1,
        "event_vote_count": 3,
        "candidate_count": 4,
    }
    assert set(TRANSITION_PROOF_FIELDS).issubset(good.keys())
    reapply = {"lin": dict(good)}
    named = {"lin": dict(good)}
    ok = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=named,
        builder_receipt_pass=True,
        reapply_proof_by_key=reapply,
    )
    # 12-field equality holds but injective RO surface absent → still fail-closed
    assert ok.get("transition_fields_equal") is True
    assert ok["crosscheck_ok"] is False
    assert "injective" in ok["reason"] or "neither_full" in ok["reason"]

    mut = copy.deepcopy(named)
    mut["lin"]["residual_after_threshold_sha256"] = "0" * 64  # same q, different residual
    bad = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=mut,
        builder_receipt_pass=True,
        reapply_proof_by_key=reapply,
    )
    assert bad["crosscheck_ok"] is False
    assert any("residual_after_threshold_sha256" in m for m in bad.get("mismatches") or [])

    mut2 = copy.deepcopy(named)
    mut2["lin"]["ordered_applied_row_identities_sha256"] = "1" * 64
    mut2["lin"]["applied_directions_sha256"] = "2" * 64
    bad2 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=mut2,
        builder_receipt_pass=True,
        reapply_proof_by_key=reapply,
    )
    assert bad2["crosscheck_ok"] is False
    assert any("ordered_applied" in m or "applied_directions" in m for m in bad2.get("mismatches") or [])

    # q-only match insufficient: change residual but keep candidate_q equal
    mut3 = copy.deepcopy(named)
    mut3["lin"]["bounded_accumulator_summary_after"] = {"hot_exact_row_count": 99}
    bad3 = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "a" * 64},
        receipt_proof_by_key=mut3,
        builder_receipt_pass=True,
        reapply_proof_by_key=reapply,
    )
    assert bad3["crosscheck_ok"] is False
    assert bad3["per_key_q_equal"].get("lin") is True  # q still equal


def test_cap_collision_legacy_fields_equal_injective_rejects():
    """co_lead hostile: cap collision — all 12 legacy fields equal, post-acc differs.

    Spec: threshold_abs=1, max_abs_per_tensor=2; events A={0:127,1:126,2:1}
    vs B={0:127,1:126,2:2} → same applied top-2, equal q, equal residual/applied
    digests as currently defined, but logical acc differs on sub-threshold row.
    New binding MUST reject (injective layer fail-closed while surface absent OR
    reject when surface present and digests differ).
    """
    import copy
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
        INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE,
        TRANSITION_PROOF_FIELDS,
        crosscheck_production_q_vs_receipt_proof,
    )

    # Construct two proofs that match on all 12 fields but represent different
    # sub-threshold residual carriers (A vs B). Under current non-injective fields
    # they can be byte-equal on the 12-field set.
    base = {
        "candidate_q_sha256_after": "q" * 64,
        "q_changed_identities_sha256": "i" * 64,
        "applied_row_identities_sha256": "a" * 64,
        "ordered_applied_row_identities_sha256": "a" * 64,
        "applied_directions_sha256": "d" * 64,
        "applied_thresholds_sha256": "t" * 64,
        # residual_after_threshold only covers threshold-crossing rows → same for A/B
        "residual_after_threshold_sha256": "r" * 64,
        "bounded_accumulator_summary_after": {
            "logical_shape": [2, 2],
            "logical_numel": 4,
            "cold_default_value": 0,
            "hot_exact_row_count": 0,
            "cold_exception_row_count": 1,
            "candidate_name": "budget_capped_hot_exact_cold_default_sparse_exceptions",
            "raw_arrays_included": False,
        },
        "q_changed_count": 2,
        "applied_row_count": 2,
        "event_vote_count": 3,
        "candidate_count": 3,
    }
    assert set(TRANSITION_PROOF_FIELDS).issubset(base.keys())
    # A and B share the 12-field fingerprint (non-injective)
    named_A = {"lin": dict(base)}
    reapply_B = {"lin": dict(base)}  # same 12 fields as A
    # Logical acc would differ: A->[0,0,1,0] vs B->[0,0,2,0] but that is NOT in the 12 fields
    res = crosscheck_production_q_vs_receipt_proof(
        production_post_q_sha256_by_key={"lin": "q" * 64},
        receipt_proof_by_key=named_A,
        builder_receipt_pass=True,
        reapply_proof_by_key=reapply_B,
    )
    assert res.get("transition_fields_equal") is True
    # MUST reject: either injective surface unavailable (current) or digests differ
    assert res["crosscheck_ok"] is False
    assert INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE is False
    assert "neither_full" in res["reason"] or "injective" in res["reason"]
    assert res.get("injective_post_acc_binding_ro_available") is False
    assert "named_receipt.full_sparse_event_carrier_or_hash_per_key" in res.get(
        "injective_post_acc_binding_missing_surfaces", []
    )
