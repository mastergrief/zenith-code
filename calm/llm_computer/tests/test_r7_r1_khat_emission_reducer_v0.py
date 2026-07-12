"""CPU-static tests for R1 K_hat emission reducer (IMPL PLAN v1.1)."""
from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack import r7_r1_khat_emission_reducer as core
from calm.hrm_text_158.native_full_stack.r7_b2_table2_trajectory_sufficiency_reducer import (
    OVERALL_INSUFFICIENT,
    OVERALL_INVALID,
    OVERALL_SUFFICIENT,
    B2ReduceResult,
    Companion,
    Dispersion,
    IntegrityChecks,
    IntegrityGate,
    RollingMeanW8,
    RollingWindow,
    Trajectory,
    Verdicts,
)
from calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer import (
    DEFAULT_K_GRID,
    OUTCOME_CANDIDATE_ONLY,
    OUTCOME_FREEZE_OK,
    OUTCOME_INVALID_COMPARISON_INPUT,
    OUTCOME_INVALID_OBSERVATION,
    OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2,
    OUTCOME_NO_CANDIDATE_NONVACUOUS,
    OUTCOME_NO_FREEZE_DISAGREEMENT,
    R1_APPROVED_DIFF_PATHS,
    ROLE_ACCEPTED_PRIMARY,
    ROLE_FRESH_REPLICATE,
    ActivationDeltaProof,
    AnalysisProvenance,
    ObservationProvenance,
    PerKAggregate,
    R1RunEnvelope,
    ScienceSourcePins,
    derive_aggregates,
    derive_cliff,
    evaluate_r1_final_freeze,
    reduce_r1_khat_emission,
    to_json_dict,
    validate_result_consistency,
)

N, W = 32, 8
GRID = DEFAULT_K_GRID
S_SS = tuple(range(22, 33))
ENDS = (29, 30, 31, 32)
PRIMARY_HEAD = "2a2d1a82f52366b1eccc8777e35bcdc276de0231"
SIDECAR_SHA = "2bb10883b00688451afc5fff23712981d44e9b4052e896b9854926202a424124"
NONCE = "20260712T111946Z_8239799d269f"
RECEIPT = "5d21110aa3a8742fefb072a0ce46a0ad8aba087a8e37df0794c10afc5fc5f33b"
AMEND = "3d96fbe8"
PINS = ScienceSourcePins("c41d874a", "f1c10c89", "b86efe9b", "9b4e311a", "783ae6e5")
ANALYSIS = AnalysisProvenance("design", "core", "cli", "test", "analysisHEAD", "lineage")
CLI = "calm.hrm_text_158.native_full_stack.r7_r1_khat_emission_reducer_cli"


def _empty_b2(overall: str, final_four=None) -> B2ReduceResult:
    ic = IntegrityChecks(True, True, True, True, True, True, True, True, True, True)
    ig = IntegrityGate(32, tuple(range(1, 33)), 0, 0, tuple(["OK"] * 32), ic, True)
    if overall == OVERALL_INVALID:
        ig = IntegrityGate(0, (), 0, 0, None, IntegrityChecks(*([False] * 10)), False)
    roll = RollingMeanW8((), (), final_four, None)
    return B2ReduceResult(
        ig, Trajectory((), 0, 0, (), ()), Dispersion(None, None, None, None, None), roll,
        Verdicts("PASS" if overall != OVERALL_INVALID else "FAIL", "PASS", "PASS", "PASS", overall),
        Companion(True, (), "boundary"),
    )


def _counts_for_khat12() -> dict[int, int]:
    # nested nonincreasing, all >=1, K16=0 -> k_hat=12
    return {2: 100, 4: 80, 8: 60, 12: 40, 16: 0}


def make_row(step: int, counts: dict[int, int], *, denom: int = 1000, bad_frac: bool = False,
             extra_k: int | None = None, drop_k: int | None = None, unequal_denom: bool = False,
             closure_false_k: int | None = None) -> dict[str, Any]:
    per_k = {}
    for k in GRID:
        if drop_k is not None and k == drop_k:
            continue
        d = denom + (1 if unequal_denom and k == 16 else 0)
        c = counts[k]
        frac = (c / d) + (0.5 if bad_frac and k == 2 else 0.0)
        per_k[str(k)] = {
            "eligible_count": c,
            "eligible_fraction_of_deferred": frac,
            "eligibility_closure_ok": False if closure_false_k == k else True,
            "current_deferred_candidate_denominator": d,
        }
    if extra_k is not None:
        per_k[str(extra_k)] = {
            "eligible_count": 1, "eligible_fraction_of_deferred": 0.001,
            "eligibility_closure_ok": True, "current_deferred_candidate_denominator": denom,
        }
    return {
        "schema_version": "hrm_text_158_r7_selective_drain_eligibility_census_step_chunk/v1",
        "step": step, "census_status": "OK", "digest_schema": "order_independent_v1_blake2b",
        "raw_arrays_included": False,
        "table1": {"cap_closure_ok": True},
        "table2": {"table2_status": "OK", "materiality_closure_ok": True, "re_candidated_fraction": 0.2},
        "table3": {"per_k": per_k},
    }


def make_rows(counts=_counts_for_khat12(), **kw) -> list[dict[str, Any]]:
    # B2 needs 32 ordinary rows 1..32; Table-3 validated only on S_ss
    rows = []
    for step in range(1, 33):
        rows.append(make_row(step, counts, **kw))
    return rows


def stub_b2(monkeypatch, overall=OVERALL_SUFFICIENT, ends=ENDS):
    ff = tuple(RollingWindow(e, 0.16) for e in ends) if ends is not None else None
    monkeypatch.setattr(core, "reduce_b2_trajectory", lambda *a, **k: _empty_b2(overall, ff))


def obs(role: str, *, nonce: str, sidecar_sha: str, head: str, **kw) -> ObservationProvenance:
    base = dict(
        role=role, launch_gate_msg_id="gate", launch_packet_sha="pkt",
        nonce_or_run_id=nonce, scratch_root="/tmp/root", sidecar_sha256=sidecar_sha,
        sidecar_path="/tmp/side.jsonl", observation_HEAD=head, science_source_pins=PINS,
        argv_semantic_family_digest="argvfam", N=N, W=W, k_grid=GRID,
    )
    base.update(kw)
    return ObservationProvenance(**base)


def primary_obs(**kw) -> ObservationProvenance:
    role = kw.pop("role", ROLE_ACCEPTED_PRIMARY)
    return obs(
        role, nonce=NONCE, sidecar_sha=SIDECAR_SHA, head=PRIMARY_HEAD,
        role_anchor_b2_terminal_receipt_sha256=RECEIPT,
        role_anchor_b2_bookend_amendment=AMEND,
        role_anchor_original_launch_nonce=NONCE,
        role_anchor_accepted_sidecar_sha256=SIDECAR_SHA,
        **kw,
    )


def replicate_obs(**kw) -> ObservationProvenance:
    role = kw.pop("role", ROLE_FRESH_REPLICATE)
    return obs(
        role, nonce="repnonce", sidecar_sha="repside" * 4 + "abcd",
        head="postlandingHEAD0000000000000000000000001",
        role_anchor_replicate_launch_gate_msg_id="launch2",
        role_anchor_replicate_terminal_receipt_sha256="term2",
        **kw,
    )


def env_for(result, role_obs: ObservationProvenance) -> R1RunEnvelope:
    return R1RunEnvelope(role_obs, ANALYSIS, result, result.S_ss)


def proof(primary_head=PRIMARY_HEAD, replicate_head="postlandingHEAD0000000000000000000000001",
          paths=R1_APPROVED_DIFF_PATHS, pins_unchanged=True) -> ActivationDeltaProof:
    return ActivationDeltaProof(primary_head, replicate_head, "r1commit", paths, pins_unchanged, "note")


@pytest.fixture
def candidate(monkeypatch):
    stub_b2(monkeypatch)
    return reduce_r1_khat_emission(make_rows())


def test_candidate_happy(candidate):
    assert candidate.overall == OUTCOME_CANDIDATE_ONLY
    assert candidate.k_hat == 12
    assert candidate.S_ss == S_SS
    assert candidate.cliff.cliff_holds is True
    ok, _ = validate_result_consistency(candidate)
    assert ok


@pytest.mark.parametrize("kw,locus", [
    ({"drop_k": 8}, "k_grid_mismatch"),
    ({"extra_k": 99}, "k_grid_mismatch"),
    ({"unequal_denom": True}, "unequal_denom"),
    ({"bad_frac": True}, "fraction_mismatch"),
    ({"closure_false_k": 4}, "closure_false"),
])
def test_table3_structural_red(monkeypatch, kw, locus):
    stub_b2(monkeypatch)
    r = reduce_r1_khat_emission(make_rows(**kw))
    assert r.overall == OUTCOME_INVALID_OBSERVATION
    assert locus in (r.failure_locus or "")


def test_nesting_breach(monkeypatch):
    stub_b2(monkeypatch)
    counts = {2: 10, 4: 20, 8: 5, 12: 3, 16: 0}  # 2<4 breach
    r = reduce_r1_khat_emission(make_rows(counts))
    assert r.overall == OUTCOME_INVALID_OBSERVATION
    assert r.failure_locus == "nesting_breach"


def test_no_feasible_nonvacuous(monkeypatch):
    stub_b2(monkeypatch)
    counts = {2: 0, 4: 0, 8: 0, 12: 0, 16: 0}
    r = reduce_r1_khat_emission(make_rows(counts))
    assert r.overall == OUTCOME_NO_CANDIDATE_NONVACUOUS
    assert r.k_hat is None


def test_final_four_reds(monkeypatch):
    stub_b2(monkeypatch, ends=None)
    assert reduce_r1_khat_emission(make_rows()).failure_locus == "final_four_available_none"
    stub_b2(monkeypatch, ends=(29, 30, 31))
    assert "len_ne_4" in (reduce_r1_khat_emission(make_rows()).failure_locus or "")
    stub_b2(monkeypatch, ends=(29, 29, 31, 32))
    assert "dup" in (reduce_r1_khat_emission(make_rows()).failure_locus or "")
    stub_b2(monkeypatch, ends=(32, 31, 30, 29))
    assert "increasing" in (reduce_r1_khat_emission(make_rows()).failure_locus or "")


def test_b2_insufficient(monkeypatch):
    stub_b2(monkeypatch, overall=OVERALL_INSUFFICIENT)
    r = reduce_r1_khat_emission(make_rows())
    assert r.overall == OUTCOME_NO_CANDIDATE_INSUFFICIENT_B2
    assert r.k_hat is None


def test_b2_invalid(monkeypatch):
    stub_b2(monkeypatch, overall=OVERALL_INVALID)
    r = reduce_r1_khat_emission(make_rows())
    assert r.overall == OUTCOME_INVALID_OBSERVATION


def test_dto_authority_a_agree(candidate):
    ok, _ = validate_result_consistency(candidate)
    assert ok
    body = to_json_dict(candidate)
    recomputed, *_ = derive_aggregates(candidate.per_step, candidate.k_grid)
    assert body["per_k"] == [
        {"k": a.k, "min_eligible_count": a.min_eligible_count, "max_eligible_count": a.max_eligible_count,
         "min_fraction": a.min_fraction, "mean_fraction": a.mean_fraction, "any_zero": a.any_zero,
         "all_closures_ok": a.all_closures_ok, "feasible": a.feasible}
        for a in recomputed
    ]


def test_dto_authority_b_aggregates_pass_dto_fails(candidate):
    # Make DTO fail feasibility for k_hat while leaving stale feasible aggregates claiming pass
    bad_steps = []
    for s in candidate.per_step:
        counts = list(s.ordered_K_eligible_counts)
        counts[3] = 0  # K12 index
        fracs = tuple(c / s.denominator for c in counts)
        bad_steps.append(replace(s, ordered_K_eligible_counts=tuple(counts), derived_fractions=fracs))
    # keep old aggregates that still claim K12 feasible
    tampered = replace(candidate, per_step=tuple(bad_steps))
    ok, loci = validate_result_consistency(tampered)
    assert not ok
    assert any("per_k" in x or "k_hat" in x or "cliff" in x for x in loci)


def test_dto_authority_c_tampered_aggregates(candidate):
    fake = tuple(
        PerKAggregate(a.k, a.min_eligible_count, a.max_eligible_count, a.min_fraction,
                      a.mean_fraction, a.any_zero, a.all_closures_ok, True)
        for a in candidate.per_k
    )
    tampered = replace(candidate, per_k=fake, cliff=derive_cliff(fake, candidate.k_grid, 16))
    # force mismatch vs per_step-derived
    tampered = replace(tampered, k_hat=16)
    ok, loci = validate_result_consistency(tampered)
    assert not ok
    e1 = env_for(candidate, primary_obs())
    e2 = env_for(tampered, replicate_obs())
    # even with activation delta, consistency on replicate must fail closed
    cmp = evaluate_r1_final_freeze(e1, e2, activation_delta_proof=proof())
    assert cmp.overall == OUTCOME_INVALID_COMPARISON_INPUT
    assert any("consistency_replicate" in x for x in cmp.hard_check_failures)


def test_canonical_json_matches_recompute(candidate):
    body = to_json_dict(candidate)
    aggs, dmin, dmax, dconst = derive_aggregates(candidate.per_step, candidate.k_grid)
    cliff = derive_cliff(aggs, candidate.k_grid, candidate.k_hat)
    assert body["denominator_min"] == dmin
    assert body["cliff"]["cliff_holds"] == cliff.cliff_holds
    assert "k_star" not in body


def test_compare_freeze_ok(candidate):
    e1 = env_for(candidate, primary_obs())
    # replicate needs distinct identity but same science result
    rep = reduce_r1_khat_emission.__wrapped__ if False else candidate  # same result object ok if envelopes differ
    e2 = env_for(candidate, replicate_obs())
    cmp = evaluate_r1_final_freeze(e1, e2, activation_delta_proof=proof())
    assert cmp.overall == OUTCOME_FREEZE_OK
    assert cmp.freeze_eligible_k_hat == 12
    assert "k_star" not in to_json_dict(cmp)


def test_compare_khat_mismatch(monkeypatch, candidate):
    stub_b2(monkeypatch)
    other = reduce_r1_khat_emission(make_rows({2: 100, 4: 80, 8: 60, 12: 0, 16: 0}))
    assert other.k_hat == 8
    cmp = evaluate_r1_final_freeze(env_for(candidate, primary_obs()), env_for(other, replicate_obs()),
                                   activation_delta_proof=proof())
    assert cmp.overall == OUTCOME_NO_FREEZE_DISAGREEMENT


def test_compare_same_nonce(candidate):
    e1 = env_for(candidate, primary_obs())
    e2 = env_for(candidate, replicate_obs(nonce_or_run_id=NONCE))
    cmp = evaluate_r1_final_freeze(e1, e2, activation_delta_proof=proof())
    assert cmp.overall == OUTCOME_INVALID_COMPARISON_INPUT
    assert "nonce_not_distinct" in cmp.hard_check_failures


def test_compare_activation_delta_allow_deny(candidate):
    e1 = env_for(candidate, primary_obs())
    e2 = env_for(candidate, replicate_obs())
    assert evaluate_r1_final_freeze(e1, e2, activation_delta_proof=proof()).overall == OUTCOME_FREEZE_OK
    bad = proof(paths=R1_APPROVED_DIFF_PATHS + ("extra.py",))
    assert evaluate_r1_final_freeze(e1, e2, activation_delta_proof=bad).overall == OUTCOME_INVALID_COMPARISON_INPUT


def test_role_swap(candidate):
    # put primary anchors on replicate role object and vice versa identities
    swapped_primary = replicate_obs(role=ROLE_ACCEPTED_PRIMARY)
    swapped_replicate = primary_obs(role=ROLE_FRESH_REPLICATE)
    cmp = evaluate_r1_final_freeze(env_for(candidate, swapped_primary), env_for(candidate, swapped_replicate),
                                   activation_delta_proof=proof(
                                       primary_head=swapped_primary.observation_HEAD,
                                       replicate_head=swapped_replicate.observation_HEAD,
                                   ))
    assert cmp.overall == OUTCOME_INVALID_COMPARISON_INPUT


def test_insufficient_replicate_nofreeze(monkeypatch, candidate):
    stub_b2(monkeypatch, overall=OVERALL_INSUFFICIENT)
    insuff = reduce_r1_khat_emission(make_rows())
    cmp = evaluate_r1_final_freeze(env_for(candidate, primary_obs()), env_for(insuff, replicate_obs()),
                                   activation_delta_proof=proof())
    assert cmp.overall == OUTCOME_NO_FREEZE_DISAGREEMENT


def test_immutability_and_nonalias(candidate):
    with pytest.raises(Exception):
        candidate.overall = "X"  # type: ignore[misc]
    rows = make_rows()
    before = copy.deepcopy(rows)
    # need stub again
    # reduce already done; check input unchanged pattern on fresh call via monkeypatch in caller
    assert rows == before
    body1 = to_json_dict(candidate)
    body2 = to_json_dict(candidate)
    assert body1 == body2
    assert body1["per_step"] is not candidate.per_step  # list projection not alias


def test_reduce_input_unchanged(monkeypatch):
    stub_b2(monkeypatch)
    rows = make_rows()
    before = copy.deepcopy(rows)
    reduce_r1_khat_emission(rows)
    assert rows == before


def test_cli_exits(tmp_path, monkeypatch, candidate):
    # reduce insufficient -> 0
    stub_b2(monkeypatch, overall=OVERALL_INSUFFICIENT)
    side = tmp_path / "side.jsonl"
    side.write_text("\n".join(json.dumps(r) for r in make_rows()) + "\n")
    # Without stub inside subprocess, use module main directly
    from calm.hrm_text_158.native_full_stack import r7_r1_khat_emission_reducer_cli as cli
    # monkeypatch reduce path via core already stubbed for this process
    rc = cli.main(["reduce", str(side)])
    assert rc == 0
    # compare FREEZE_OK
    e1 = to_json_dict(env_for(candidate, primary_obs()))
    e2 = to_json_dict(env_for(candidate, replicate_obs()))
    p1, p2, pr = tmp_path / "a.json", tmp_path / "b.json", tmp_path / "proof.json"
    p1.write_text(json.dumps(e1)); p2.write_text(json.dumps(e2))
    pr.write_text(json.dumps({
        "primary_observation_HEAD": PRIMARY_HEAD,
        "replicate_observation_HEAD": replicate_obs().observation_HEAD,
        "approved_r1_commit_sha": "c",
        "git_diff_paths": list(R1_APPROVED_DIFF_PATHS),
        "science_pins_unchanged": True,
        "operator_attestation_note": "n",
    }))
    rc = cli.main(["compare", str(p1), str(p2), "--activation-delta-proof", str(pr)])
    assert rc == 0
    # IO -> 2
    assert cli.main(["reduce", str(tmp_path / "missing.jsonl")]) == 2
    # invalid compare (same nonce) -> 3
    e2b = to_json_dict(env_for(candidate, replicate_obs(nonce_or_run_id=NONCE)))
    p2b = tmp_path / "b2.json"; p2b.write_text(json.dumps(e2b))
    assert cli.main(["compare", str(p1), str(p2b), "--activation-delta-proof", str(pr)]) == 3


def test_no_k_star_anywhere(candidate):
    assert "k_star" not in json.dumps(to_json_dict(candidate))
    e1 = env_for(candidate, primary_obs()); e2 = env_for(candidate, replicate_obs())
    cmp = evaluate_r1_final_freeze(e1, e2, activation_delta_proof=proof())
    assert "k_star" not in json.dumps(to_json_dict(cmp))


def test_forbidden_symbols_and_line_budget():
    text = Path(core.__file__).read_text(encoding="utf-8")
    lines = text.splitlines()
    assert len(lines) < 500
    for bad in ("argparse", "pathlib", "sys.exit", "subprocess", "__main__", "k_star"):
        assert bad not in text
