"""Shared LANDS-AB CPU test helpers (IMPLEMENT_v12 dedupe)."""
from __future__ import annotations

import uuid
from pathlib import Path

from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import all_true_matrix


def base_ok(**over):
    p = {
        "scope_creep": False,
        "fixture_contract_raw_fail": False,
        "surface_pass_by_row": all_true_matrix(),
    }
    p.update(over)
    return p


def write_real_cpu_row(scratch: Path):
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_site_measurement import (
        measure_g_cpu_static_ab,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        o_excl_write_json,
        runtime_scratch_raw_path,
    )
    obs = measure_g_cpu_static_ab()
    path = runtime_scratch_raw_path(
        scratch_dir=scratch, gating_row="G_CPU_STATIC_AB", run_nonce=uuid.uuid4().hex[:8]
    )
    sha = o_excl_write_json(path, obs)
    return obs, sha, path


def make_cuda_fixture_fail_obs(gating_row: str, *, key: str = "lin"):
    """Representative CUDA-row raw observation with honest fixture-fail polarity."""
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
        make_raw_row_observation,
    )
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        synthesize_good_topology_events,
    )

    if gating_row.startswith("G_CUDA_ORACLE"):
        surfs = {"s5": False}
        metrics = {
            "events_equal_by_key": {key: False},
            "events_equal_fused_vs_dense_derived": False,
            "independent_two_branch_recompute_ok": True,
            "dense_derived_provenance": "two_branch_parallel_dense_vote_derivation",
            "d1_densify_from_sparse_used": False,
            "sparse_vote_authority_mode": "oracle_on",
            "votes_by_key_applied": None,
            "builder_receipt_pass": False,
            "oracle_mode_on_named_site": True,
        }
    else:
        surfs = {"s3": False, "s4": False, "s6": False}
        metrics = {
            "post_q_sha256_by_key": {key: {"sparse": "a" * 64, "dense": "b" * 64}},
            "post_logical_acc_sha256_by_key": {key: {"sparse": "c" * 64, "dense": "d" * 64}},
            "events_equal_by_key": {key: False},
            "sparse_event_count": 0,
            "q_changed_count_sparse": 0,
            "q_changed_count_dense": 0,
            "s6_geometry": {
                "votes_by_key_applied": None,
                "sparse_vote_authority_only": True,
                "transient_over2_tensors": ["weighted_grad"],
                "oracle_only_absent_on_fused": True,
            },
            "d1_densify_from_sparse_used": False,
            "builder_receipt_pass": False,
            "production_sparse_matches_twin": False,
        }
    return make_raw_row_observation(
        gating_row=gating_row,
        device="cuda",
        measured_surfaces=surfs,
        metrics=metrics,
        key_universe=[key],
        fixture_contract_raw_fail=True,
        synthetic_only=False,
        phase_events=synthesize_good_topology_events(node_id=gating_row),
    )
