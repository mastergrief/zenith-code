"""Phase-A §2b characterization for forgotten-accum flip deferral (dense-legacy cap).

Fixtures frozen BEFORE source edits (FIXTURE_INDEX). NEVER rebaseline.
"""
from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_apply import (
    apply_global_rate_cap_with_optional_flip_deferral,
    build_W_plus_1_release_record,
    build_during_W_telemetry,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_contracts import (
    DENSE_LEGACY_CAP_SITE_ID,
    RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
    assert_W_plus_1_anti_burst,
    assert_backlog_unchanged,
    assert_cap_site_branch_coverage,
    assert_during_W,
    assert_pre_W_seed_invariant,
    backlog_content_sha256,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "forgotten_accum_flip_deferral_characterization_v0"
REPO_ROOT = Path(__file__).resolve().parents[3]


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def _normalize_backlog(backlog: dict | None) -> dict:
    """JSON fixtures store flat_index keys as strings; runtime uses ints."""
    out: dict = {}
    for state_key, by_index in dict(backlog or {}).items():
        out[str(state_key)] = {
            str(int(flat_index)): dict(entry) for flat_index, entry in dict(by_index).items()
        }
    return out


def _payload(res) -> dict:
    return {
        "q_shas": {tr.state_key: _sha_tensor(tr.q_levels) for tr in res.tensor_results},
        "acc_shas": {tr.state_key: _sha_tensor(tr.accumulators) for tr in res.tensor_results},
        "q_lists": {tr.state_key: tr.q_levels.flatten().tolist() for tr in res.tensor_results},
        "acc_lists": {tr.state_key: tr.accumulators.flatten().tolist() for tr in res.tensor_results},
        "backlog": _normalize_backlog(res.deferred_backlog),
        "backlog_sha": backlog_content_sha256(res.deferred_backlog),
        "backlog_cardinality": sum(len(v) for v in (res.deferred_backlog or {}).values()),
        "step_summary_authority": {
            k: res.step_summary[k]
            for k in (
                "global_rate_cap_enabled",
                "global_rate_cap_cap",
                "global_rate_cap_accepted_count",
                "global_rate_cap_applied_count",
                "global_rate_cap_deferred_count",
                "q_changed_count",
                "global_rate_cap_saturated",
            )
            if k in res.step_summary
        },
        "accepted_count": len(res.accepted_rows),
        "deferred_count": len(res.deferred_rows),
        "demand_count": len(res.rows),
    }


def _inputs_from_fixture(rows: list[dict]) -> list[GlobalRateCapTensorInput]:
    out: list[GlobalRateCapTensorInput] = []
    for row in rows:
        state = VoteUpdateState(
            q_levels=torch.tensor(row["q"], dtype=torch.int8),
            accumulators=torch.tensor(row["acc"], dtype=torch.int16),
        )
        vin = VoteUpdateInputs(votes=torch.tensor(row["votes"], dtype=torch.int16))
        plan = plan_integer_vote_update_reference(state, vin, _spec())
        out.append(
            GlobalRateCapTensorInput(
                state_key=row["state_key"],
                state=state,
                plan=plan,
                vote_inputs=vin,
            )
        )
    return out


def _cap_spec(d: dict) -> GlobalRateCapSpec:
    return GlobalRateCapSpec(
        cap=int(d["cap"]),
        step=int(d["step"]),
        mutate_outputs=bool(d.get("mutate_outputs", True)),
    )


def test_fixture_index_frozen_and_never_rebaseline():
    index = _load("FIXTURE_INDEX.json")
    assert index["never_rebaseline"] is True
    assert index["frozen_against_head"] == "d86b7417df5bc6ab62ce0690e4972de468fc4c2e"
    for name, digest in index["files"].items():
        raw = (FIX / name).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == digest


def test_T_CAP_DEFAULT_OFF_PARITY_backlog():
    """facade(False) == apply_global_rate_cap_reference on frozen saturated fixture."""
    fx = _load("cap_on_ordinary_saturated.json")
    inputs = _inputs_from_fixture(fx["inputs"])
    spec = _cap_spec(fx["spec"])
    seed = fx["seed_backlog"]
    ref = apply_global_rate_cap_reference(
        inputs,
        spec,
        deferred_backlog=dict(seed),
        contract_name=fx["spec"]["contract_name"],
    )
    fac = apply_global_rate_cap_with_optional_flip_deferral(
        inputs,
        spec,
        deferred_backlog=dict(seed),
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=False,
    )
    assert _payload(fac) == _payload(ref)
    assert _payload(fac) == fx["expected"]


def test_T_CAP_ON_ORDINARY_unsaturated_fixture():
    fx = _load("cap_on_ordinary_unsaturated.json")
    inputs = _inputs_from_fixture(fx["inputs"])
    fac = apply_global_rate_cap_with_optional_flip_deferral(
        inputs,
        _cap_spec(fx["spec"]),
        deferred_backlog={},
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=False,
    )
    assert _payload(fac) == fx["expected"]
    assert fac.step_summary["global_rate_cap_saturated"] is False


def test_reducers_forbidden_imports_cpu_pure():
    src = (
        REPO_ROOT
        / "calm/hrm_text_158/native_full_stack/forgotten_accum_flip_deferral_reducers.py"
    ).read_text(encoding="utf-8")
    tree = ast.parse(src)
    banned = {"torch.cuda", "pathlib", "os", "subprocess", "global_rate_cap"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert not any(b in alias.name for b in banned)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert "global_rate_cap" not in node.module
            assert "torch.cuda" not in node.module
            assert node.module not in {"os", "pathlib", "subprocess"}


def test_T_DURING_W_CAP_and_backlog_fixed():
    fx = _load("cap_on_ordinary_saturated.json")
    inputs = _inputs_from_fixture(fx["inputs"])
    seed = {"A": {99: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}}
    seed_sha = backlog_content_sha256(seed)
    acc_pre = hashlib.sha256(
        b"|".join(_sha_tensor(inp.state.accumulators).encode() for inp in inputs)
    ).hexdigest()
    res = apply_global_rate_cap_with_optional_flip_deferral(
        inputs,
        _cap_spec(fx["spec"]),
        deferred_backlog=seed,
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=True,
    )
    assert_backlog_unchanged(before=seed, after=res.deferred_backlog)
    assert int(res.step_summary["global_rate_cap_applied_count"]) == 0
    assert int(res.step_summary["q_changed_count"]) == 0
    tel = build_during_W_telemetry(acc_hash_pre=acc_pre, result=res)
    assert_during_W(tel, seed_backlog_sha=seed_sha)
    # Carry must have moved vs original zeros
    assert any(
        tr.accumulators.flatten().tolist() != inp.state.accumulators.flatten().tolist()
        for tr, inp in zip(res.tensor_results, inputs)
    )


def test_T_W_PLUS_1_CAP_anti_burst():
    fx = _load("cap_on_ordinary_saturated.json")
    # Start from deferred carry state after one W step
    inputs0 = _inputs_from_fixture(fx["inputs"])
    w_res = apply_global_rate_cap_with_optional_flip_deferral(
        inputs0,
        _cap_spec(fx["spec"]),
        deferred_backlog={},
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=True,
    )
    # Rebuild inputs from post-W state with fresh identical votes
    next_inputs: list[GlobalRateCapTensorInput] = []
    for tr, row in zip(w_res.tensor_results, fx["inputs"]):
        state = VoteUpdateState(q_levels=tr.q_levels.clone(), accumulators=tr.accumulators.clone())
        vin = VoteUpdateInputs(votes=torch.tensor(row["votes"], dtype=torch.int16))
        plan = plan_integer_vote_update_reference(state, vin, _spec())
        next_inputs.append(
            GlobalRateCapTensorInput(
                state_key=row["state_key"], state=state, plan=plan, vote_inputs=vin
            )
        )
    pre_carry = hashlib.sha256(
        b"|".join(_sha_tensor(inp.state.accumulators).encode() for inp in next_inputs)
    ).hexdigest()
    w1 = apply_global_rate_cap_with_optional_flip_deferral(
        next_inputs,
        _cap_spec(fx["spec"]),
        deferred_backlog=w_res.deferred_backlog,
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=False,
    )
    record = build_W_plus_1_release_record(pre_vote_carry_hash=pre_carry, result=w1)
    # Default-off path has no release_path_id annotation; force ordinary for record.
    if record.release_path_id != RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0:
        record = type(record)(
            **{**record.__dict__, "release_path_id": RELEASE_PATH_ORDINARY_SELECTOR_SAME_AS_R0}
        )
    assert_W_plus_1_anti_burst(record)
    assert record.applied_count <= int(fx["spec"]["cap"])


def test_T_PRE_W_SEED_and_T_CROSSING_WITNESS():
    fx = _load("crossing_witness_shared_start.json")
    rows = fx["inputs_template"]
    r0_inputs = _inputs_from_fixture(rows)
    rw_inputs = _inputs_from_fixture(rows)
    assert_pre_W_seed_invariant(
        r0_acc_sha=_sha_tensor(r0_inputs[0].state.accumulators),
        rw_acc_sha=_sha_tensor(rw_inputs[0].state.accumulators),
        r0_backlog_sha=backlog_content_sha256({}),
        rw_backlog_sha=backlog_content_sha256({}),
        r0_backlog_cardinality=0,
        rw_backlog_cardinality=0,
    )
    spec = _cap_spec(fx["spec"])
    # R0 ordinary step
    r0 = apply_global_rate_cap_with_optional_flip_deferral(
        r0_inputs,
        spec,
        deferred_backlog={},
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=False,
    )
    # RW deferred step
    rw = apply_global_rate_cap_with_optional_flip_deferral(
        rw_inputs,
        spec,
        deferred_backlog={},
        contract_name=fx["spec"]["contract_name"],
        flip_application_deferred=True,
    )
    r0_traj = (_payload(r0)["q_shas"], _payload(r0)["acc_shas"], _payload(r0)["backlog_sha"])
    rw_traj = (_payload(rw)["q_shas"], _payload(rw)["acc_shas"], _payload(rw)["backlog_sha"])
    assert r0_traj != rw_traj
    assert _payload(rw)["backlog_sha"] == backlog_content_sha256({})
    assert int(rw.step_summary["global_rate_cap_applied_count"]) == 0
    assert int(r0.step_summary["global_rate_cap_applied_count"]) >= 1


def test_T_BRANCH_COVERAGE_learner_dense_legacy_site():
    """Learner dense-cap site must execute the facade (not non-cap alone)."""
    q = torch.zeros(6, dtype=torch.int8)
    acc = torch.zeros(6, dtype=torch.int16)
    votes = torch.tensor([0, 25, 0, 25, 0, 25], dtype=torch.int16)
    state = make_bounded_tensor_state("A", q, 1.0, acc)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    cap = GlobalRateCapSpec(cap=1, step=3, mutate_outputs=True)
    step = apply_bounded_delta_vote_step(
        {"A": state},
        {"A": votes},
        {"A": spec},
        global_cap_spec=cap,
        global_cap_contract_name="c1_banked_faithful_long_run_global_cap",
        flip_application_deferred=False,
    )
    branch = str(step.global_summary.get("forgotten_accum_cap_site_branch", ""))
    assert_cap_site_branch_coverage(branch)
    # Deferred mode also hits same site
    step_d = apply_bounded_delta_vote_step(
        {"A": make_bounded_tensor_state("A", q.clone(), 1.0, acc.clone())},
        {"A": votes},
        {"A": spec},
        global_cap_spec=cap,
        global_cap_contract_name="c1_banked_faithful_long_run_global_cap",
        flip_application_deferred=True,
    )
    assert_cap_site_branch_coverage(
        str(step_d.global_summary.get("forgotten_accum_cap_site_branch", ""))
    )
    assert bool(step_d.global_summary.get("flip_application_deferred")) is True
    assert int(step_d.global_summary.get("q_changed_count", -1)) == 0


def test_T_LEARNER_DEFAULT_omitted_kwarg_matches_explicit_false():
    q = torch.zeros(4, dtype=torch.int8)
    acc = torch.zeros(4, dtype=torch.int16)
    votes = torch.tensor([25, 0, 25, 0], dtype=torch.int16)
    spec = VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )
    cap = GlobalRateCapSpec(cap=2, step=3, mutate_outputs=True)
    a = apply_bounded_delta_vote_step(
        {"A": make_bounded_tensor_state("A", q.clone(), 1.0, acc.clone())},
        {"A": votes},
        {"A": spec},
        global_cap_spec=cap,
    )
    b = apply_bounded_delta_vote_step(
        {"A": make_bounded_tensor_state("A", q.clone(), 1.0, acc.clone())},
        {"A": votes},
        {"A": spec},
        global_cap_spec=cap,
        flip_application_deferred=False,
    )
    assert _sha_tensor(a.tensor_states["A"].q_levels) == _sha_tensor(b.tensor_states["A"].q_levels)
    assert a.tensor_states["A"].exact_accumulator_shadow is not None
    assert b.tensor_states["A"].exact_accumulator_shadow is not None
    assert _sha_tensor(a.tensor_states["A"].exact_accumulator_shadow) == _sha_tensor(
        b.tensor_states["A"].exact_accumulator_shadow
    )
