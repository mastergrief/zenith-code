"""Phase-B CPU characterization for forgotten-accum training-equivalence.

Proves A1 E load-only roundtrip authority, R0/RW physical absence + isolation,
default-off runner flag threading, all-else-identical manifests, ledger
arithmetic, and CPU-checkable smoke predicates.
"""
from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_arms import (
    FutureStreamBudget,
    all_else_identical_manifests,
    isolated_arm_roots,
    prove_r0_rw_same_zero_seed,
    resume_arm_from_live_cut,
    rw_flip_defer_flags_for_post_cut_window,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_bank_eval import (
    e_must_match_u_bank,
    evaluate_arm_bank_gate,
    select_earliest_all_clear,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY,
    SMOKE_CPU_PREDICATES,
    T_CUT,
    W_REWARM_STEPS,
    ArmId,
    ResumePolicy,
    build_all_arm_manifests,
    flip_defer_schedule,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ledger import (
    ArmComputeCounts,
    LOG2_3,
    build_ledger,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_zero_seed import (
    EXACT_SHADOW_KEY,
    assert_r0_rw_physical_absence,
    load_resume_artifact,
    pre_cut_source_sha256,
    write_resume_artifact,
    build_resume_artifact,
)

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"


def _live_states():
    q = torch.zeros(6, dtype=torch.int8)
    acc = torch.tensor([0, 11, 0, -12, 0, 7], dtype=torch.int16)
    return {"A": make_bounded_tensor_state("A", q, 1.0, acc)}


def test_all_else_identical_arm_manifests():
    manifests = all_else_identical_manifests()
    assert set(manifests) == {"U", "E", "R0", "RW"}
    ids = [manifests[a]["identity"] for a in manifests]
    assert ids[0] == ids[1] == ids[2] == ids[3]
    assert manifests["E"]["resume_policy"] == ResumePolicy.EXACT_PRESERVE.value
    assert manifests["R0"]["resume_policy"] == ResumePolicy.ZERO_STRIP.value
    assert manifests["RW"]["flip_application_deferred_during_W"] is True
    assert manifests["U"]["resume_policy"] is None


def test_A1_E_serialize_discard_load_roundtrip(tmp_path: Path):
    live = _live_states()
    backlog = {"A": {1: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}}
    source = pre_cut_source_sha256(live, backlog, {"batch_seed": 44})
    pre_shadow = live["A"].exact_accumulator_shadow.clone()
    pre_q = live["A"].q_levels.clone()

    result = resume_arm_from_live_cut(
        arm=ArmId.E,
        live_states=live,
        live_backlog=backlog,
        experiment_root=tmp_path,
        rng_metadata={"batch_seed": 44},
        shared_pre_cut_source_sha256=source,
    )
    assert result.meta["policy"] == ResumePolicy.EXACT_PRESERVE.value
    assert result.meta["pre_cut_source_sha256"] == source
    assert torch.equal(result.tensor_states["A"].exact_accumulator_shadow, pre_shadow)
    assert torch.equal(result.tensor_states["A"].q_levels, pre_q)
    assert result.deferred_backlog == backlog
    # Missing exact shadow fails closed
    roots = isolated_arm_roots(tmp_path)
    bad = build_resume_artifact(
        arm=ArmId.E,
        policy=ResumePolicy.EXACT_PRESERVE,
        tensor_states=_live_states(),
        deferred_backlog={},
        pre_cut_source_sha256_value=source,
    )
    obj = bad.to_json_obj()
    del obj["states"]["A"][EXACT_SHADOW_KEY]
    path = roots[ArmId.E] / "resume_artifact.json"
    path.write_text(__import__("json").dumps(obj), encoding="utf-8")
    with pytest.raises(ValueError, match="CONTROL_INVALID"):
        load_resume_artifact(
            roots[ArmId.E],
            expected_arm=ArmId.E,
            expected_policy=ResumePolicy.EXACT_PRESERVE,
            allowed_artifact_roots=roots,
        )


def test_A1_R0_RW_strip_byte_absence_isolation_and_same_zero(tmp_path: Path):
    live = _live_states()
    backlog = {"A": {2: {"first_step": 3, "last_deferred_step": 3, "defer_count": 1}}}
    source = pre_cut_source_sha256(live, backlog, {})
    r0 = resume_arm_from_live_cut(
        arm=ArmId.R0,
        live_states=live,
        live_backlog=backlog,
        experiment_root=tmp_path,
        shared_pre_cut_source_sha256=source,
    )
    # fresh live for RW (same content)
    live2 = _live_states()
    rw = resume_arm_from_live_cut(
        arm=ArmId.RW,
        live_states=live2,
        live_backlog=backlog,
        experiment_root=tmp_path,
        shared_pre_cut_source_sha256=source,
    )
    assert r0.deferred_backlog == {}
    assert rw.deferred_backlog == {}
    assert prove_r0_rw_same_zero_seed(r0, rw) == PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY

    roots = isolated_arm_roots(tmp_path)
    r0_raw = (roots[ArmId.R0] / "resume_artifact.json").read_bytes()
    rw_raw = (roots[ArmId.RW] / "resume_artifact.json").read_bytes()
    assert_r0_rw_physical_absence(r0_raw)
    assert_r0_rw_physical_absence(rw_raw)
    assert EXACT_SHADOW_KEY.encode() not in r0_raw
    assert EXACT_SHADOW_KEY.encode() not in rw_raw

    # Isolation: R0 loader cannot open E root
    e_live = _live_states()
    resume_arm_from_live_cut(
        arm=ArmId.E,
        live_states=e_live,
        live_backlog={},
        experiment_root=tmp_path,
        shared_pre_cut_source_sha256=source,
    )
    with pytest.raises(PermissionError):
        load_resume_artifact(
            roots[ArmId.E],
            expected_arm=ArmId.R0,
            expected_policy=ResumePolicy.ZERO_STRIP,
            allowed_artifact_roots=roots,
        )


def test_shared_pre_cut_source_identity_across_resume_arms(tmp_path: Path):
    live = _live_states()
    source = pre_cut_source_sha256(live, {}, {"ordering_seed": 17})
    metas = []
    for arm in (ArmId.E, ArmId.R0, ArmId.RW):
        res = resume_arm_from_live_cut(
            arm=arm,
            live_states=_live_states(),
            live_backlog={},
            experiment_root=tmp_path,
            rng_metadata={"ordering_seed": 17},
            shared_pre_cut_source_sha256=source,
        )
        metas.append(res.meta["pre_cut_source_sha256"])
    assert metas[0] == metas[1] == metas[2] == source


def test_RW_flip_defer_schedule_and_future_stream_budget():
    flags = rw_flip_defer_flags_for_post_cut_window()
    assert flags[:W_REWARM_STEPS] == [True] * W_REWARM_STEPS
    assert flags[W_REWARM_STEPS] is False  # W+1 ordinary
    assert flip_defer_schedule(ArmId.R0, post_cut_step_index=1) is False
    budget = FutureStreamBudget(t_cut=T_CUT, W=W_REWARM_STEPS, runway_end=1500)
    budget.assert_matched()
    assert budget.post_cut_train_steps() == 1000
    assert budget.rw_rewarm_step_indices() == tuple(range(1, W_REWARM_STEPS + 1))


def test_ledger_arithmetic_matched_budget():
    counts = {
        "U": ArmComputeCounts("U", 1500, 1500, 1500, 10.0, 0),
        "E": ArmComputeCounts("E", 1500, 1500, 1500, 10.1, 0),
        "R0": ArmComputeCounts("R0", 1000, 1000, 1000, 7.0, 0),
        "RW": ArmComputeCounts("RW", 1000, 1000, 1000, 7.2, W_REWARM_STEPS),
    }
    # post-cut matched: compare surplus on full runway counts equal for U/E;
    # for R0/RW use same update_count so surplus vs each other is 0 when compared via E/U
    # Rebuild with matched post-cut update counts relative to U/E for surplus check:
    counts["R0"] = ArmComputeCounts("R0", 1500, 1500, 1500, 7.0, 0)
    counts["RW"] = ArmComputeCounts("RW", 1500, 1500, 1500, 7.2, W_REWARM_STEPS)
    ledger = build_ledger(arm_counts=counts, replay_payload_bpw=0.0)
    assert abs(ledger.base_packed_q_bpw - LOG2_3) < 1e-12
    assert ledger.classification is None
    assert ledger.surplus_compute_vs_U["RW"] == 0.0

    bad = dict(counts)
    bad["RW"] = ArmComputeCounts("RW", 1500, 1500, 1600, 7.2, W_REWARM_STEPS)
    bad_ledger = build_ledger(arm_counts=bad)
    assert bad_ledger.classification == "REWARM_ACCOUNTING_INVALID"


def test_bank_eval_earliest_all_clear_and_E_matches_U():
    clears = {250: False, 500: True, 750: True, 1500: True}
    assert select_earliest_all_clear(clears) == 500
    u = evaluate_arm_bank_gate(
        arm="U",
        acquire_pct=91.0,
        retain_pct_by_support={"L0b": 90.0, "math_a0": 92.0},
        clears_by_save=clears,
        parent_consistency_ok=True,
        close_sibling_ok=True,
        hashes_diagnostic={"q": "abc"},
    )
    e = evaluate_arm_bank_gate(
        arm="E",
        acquire_pct=91.0,
        retain_pct_by_support={"L0b": 90.0, "math_a0": 92.0},
        clears_by_save=clears,
        parent_consistency_ok=True,
        close_sibling_ok=True,
        hashes_diagnostic={"q": "def"},  # diagnostic — does not veto
    )
    assert u.bank_clear and e.bank_clear
    assert e_must_match_u_bank(u, e)
    assert u.as_dict()["hashes_grant_or_veto"] is False


def test_probe_threads_flip_application_deferred_default_false():
    src = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_bounded_delta_steps":
            fn = node
            break
    assert fn is not None
    defaults = {
        a.arg: d
        for a, d in zip(fn.args.args[::-1], (fn.args.defaults or [])[::-1])
        if isinstance(a, ast.arg)
    }
    # also kwonly
    for a, d in zip(fn.args.kwonlyargs, fn.args.kw_defaults or []):
        if d is not None:
            defaults[a.arg] = d
    assert "flip_application_deferred" in {
        a.arg for a in fn.args.args + fn.args.kwonlyargs
    }
    # default False
    kw_map = dict(zip([a.arg for a in fn.args.kwonlyargs], fn.args.kw_defaults))
    default_node = kw_map.get("flip_application_deferred")
    assert default_node is not None
    assert ast.literal_eval(default_node) is False
    assert "flip_application_deferred=bool(flip_application_deferred)" in src
    assert "apply_bounded_delta_vote_step" in src


def test_smoke_cpu_predicates_defined():
    assert "carrier_must_be_dense_legacy_not_event_coded" in SMOKE_CPU_PREDICATES
    assert DENSE_LEGACY_CAP_SITE_ID.startswith("DENSE_LEGACY_")
    manifests = build_all_arm_manifests()
    assert manifests["RW"].identity.global_cap_contract == (
        "c1_banked_faithful_long_run_global_cap"
    )
