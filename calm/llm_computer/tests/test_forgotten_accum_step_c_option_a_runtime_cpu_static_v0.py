"""CPU-static tests for Step-C Option-A runtime facade (mocked runner; no model load)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import BoundedDeltaTensorState
from calm.hrm_text_158.native_full_stack import forgotten_accum_step_c_option_a_runtime as R
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import ArmId


class FakeCudaDevice:
    type = "cuda"


class FakeCudaTensor:
    def __init__(self) -> None:
        self.device = FakeCudaDevice()
        self.dtype = torch.float32


class FakeCudaModel:
    device = FakeCudaDevice()


def _cpu_state() -> BoundedDeltaTensorState:
    q = torch.zeros(2, dtype=torch.int8)
    shadow = torch.zeros(2, dtype=torch.int16)
    acc = BoundedDeltaAccumulatorState(
        logical_shape=(2,), cold_default_value=0, hot_exact_indices=(), hot_exact_values=(),
    )
    return BoundedDeltaTensorState(
        state_key="toy", q_levels=q, frozen_scale=torch.tensor(1.0),
        bounded_accumulator=acc, exact_accumulator_shadow=shadow,
        bounded_accumulator_fresh_for_exact_shadow=True, event_coded_live_carrier=None,
    )


def _batch() -> dict:
    t = FakeCudaTensor()
    return {"inputs": t, "labels": t, "sep_positions": t, "position_ids": t}


def test_budgets_and_unadmitted():
    assert sum(1 for _ in Path(R.__file__).open()) <= 280
    with pytest.raises(R.StepCBlock) as ei:
        R.assert_geometry_admitted(2, 4, 2)
    assert ei.value.code == "UNADMITTED_GEOMETRY"
    R.assert_geometry_admitted(2, 4, 1)


def test_i1_precall_planned_cap_sequence():
    calls, ctx = [], R.make_ctx()
    w = R.wrap_runner(lambda *a, **k: calls.append(k.get("steps")) or "ok", ctx)
    for steps, arm in ((4, "U"), (2, "E"), (2, "R0"), (2, "RW")):
        assert w(steps=steps, ordered_apply_event_arm_id=arm) == "ok"
    assert calls == [4, 2, 2, 2] and ctx.planned_physical_update_count == 10
    with pytest.raises(R.StepCBlock) as ei:
        w(steps=1, ordered_apply_event_arm_id="U")
    assert ei.value.code == "PLANNED_PHYSICAL_OVERFLOW" and len(calls) == 4


def test_i1_malformed_steps_before_real():
    calls, ctx = [], R.make_ctx()
    w = R.wrap_runner(lambda *a, **k: calls.append(1), ctx)
    for bad in (0, -1, 1.5, True, "2", None):
        with pytest.raises(R.StepCBlock) as ei:
            w(steps=bad, ordered_apply_event_arm_id="U")
        assert ei.value.code == "STEPS_MALFORMED"
    assert calls == []


def test_i2_injected_clock_phase_budgets():
    t = {"n": 0.0}
    ctx = R.make_ctx(clock=lambda: t["n"])
    t0 = ctx.clock(); t["n"] = 361.0
    with pytest.raises(R.StepCBlock) as ei:
        R.phase_check(ctx, "materialize", t0)
    assert ei.value.code == "PHASE_BUDGET_EXCEEDED"
    t["n"] = 0.0
    ctx2 = R.make_ctx(clock=lambda: t["n"])

    def slow(*a, **k):
        t["n"] += 481.0

    w = R.wrap_runner(slow, ctx2)
    with pytest.raises(R.StepCBlock) as ei2:
        w(steps=4, ordered_apply_event_arm_id="U")
    assert ei2.value.code == "PHASE_BUDGET_EXCEEDED"
    for name, over in (("postflight", 91.0), ("receipt_emission", 61.0), ("arm_E", 301.0)):
        t["n"] = 0.0
        ctx3 = R.make_ctx(clock=lambda: t["n"])
        t0 = ctx3.clock(); t["n"] = over
        with pytest.raises(R.StepCBlock):
            R.phase_check(ctx3, name, t0)
    src = Path(R.__file__).read_text()
    assert "[PROG] {phase} START" in src and "[PROG] {phase} END" in src
    assert "[PROG] cut_fork_serialize_load START" in src
    # materialize/postflight/receipt_emission PROG markers live in Path-injected harness


def test_i3_scratch_allowlist(tmp_path):
    scratch = tmp_path / "scratch"
    assert R.list_scratch_files(scratch) == []
    (scratch / "arms" / "E").mkdir(parents=True)
    (scratch / "arms" / "E" / "extra.txt").write_text("x")
    with pytest.raises(R.StepCBlock) as ei:
        R.run_log_identity_preflight(
            log_path=tmp_path / "a.log", identity_path=tmp_path / "id.json",
            receipt_path=tmp_path / "r.json", monitor_path=tmp_path / "m.json",
            scratch=scratch, repo=tmp_path, parent=tmp_path / "p.pt", check_git=False,
        )
    assert ei.value.code == "ARTIFACT_COLLISION"
    for a in ("E", "R0", "RW"):
        d = scratch / "arms" / a; d.mkdir(parents=True, exist_ok=True)
        (d / "resume_artifact.json").write_text("{}")
    (scratch / "arms" / "E" / "extra.txt").unlink()
    allow = tuple(scratch / "arms" / a / "resume_artifact.json" for a in ("E", "R0", "RW"))
    ctx = R.make_ctx()
    ctx.real_runner_attempt_count = 4
    ctx.planned_physical_update_count = 10
    ctx.cuda_model_forward_backward_completed_count = 10
    ctx.cpu_q_acc_apply_completed_count = 10
    ctx.resume_arm_call_count = 3
    ctx.resume_artifact_load_count = 3
    ctx.model_checkpoint_materialization_call_count = 1
    ctx.parent_checkpoint_deserialize_count = 1
    ctx.observed_keysets = {"U": {1, 2, 3, 4}, "E": {3, 4}, "R0": {3, 4}, "RW": {3, 4}}
    R.finalize_counters(ctx, scratch, allow)
    (scratch / "arms" / "E" / "rogue.pt").write_bytes(b"pt")
    with pytest.raises(R.StepCBlock) as ei2:
        R.finalize_counters(ctx, scratch, allow)
    assert ei2.value.code == "BANKED_ARTIFACT_MUTATION"


def test_i4_preflight_identity_and_collision(tmp_path):
    log, ident, receipt, mon, scratch = (
        tmp_path / "a.log", tmp_path / "id.json", tmp_path / "r.json",
        tmp_path / "m.json", tmp_path / "scratch",
    )
    frozen = R.run_log_identity_preflight(
        log_path=log, identity_path=ident, receipt_path=receipt,
        monitor_path=mon, scratch=scratch, repo=tmp_path, parent=tmp_path / "p.pt",
        check_git=False,
    )
    assert frozen["resolved_path"] == str(log.resolve())
    with pytest.raises(R.StepCBlock) as ei:
        R.run_log_identity_preflight(
            log_path=log, identity_path=tmp_path / "id2.json", receipt_path=receipt,
            monitor_path=mon, scratch=scratch, repo=tmp_path, parent=tmp_path / "p.pt",
            check_git=False,
        )
    assert ei.value.code == "ARTIFACT_COLLISION"


def test_holder_fp_fork_gate_monitor_forge():
    resume = SimpleNamespace(
        meta={"pre_cut_source_sha256": "abc", "policy": "exact_preserve"},
        tensor_states={"k": 1}, deferred_backlog={"b": 2},
    )
    e, bl = {}, {}
    R.mutate_holders_from_resume(
        state_holder=e, backlog_holder=bl, resume=resume, expected_fp="abc", arm="E",
    )
    assert e == {"k": 1}
    with pytest.raises(R.StepCBlock):
        R.mutate_holders_from_resume(
            state_holder={}, backlog_holder={}, resume=resume, expected_fp="nope", arm="E",
        )
    ctx = R.make_ctx()
    with pytest.raises(R.StepCBlock) as ei:
        R.gate_e_requires_fork(ctx)
    assert ei.value.code == "FORK_INCOMPLETE"
    with pytest.raises(R.StepCBlock) as ei2:
        R.refuse_monitor_evidence_forge(Path("/tmp/mon.json"))
    assert ei2.value.code == "HARNESS_MUST_NOT_FORGE_MONITOR_EVIDENCE"


def test_wrong_log_identity_and_keyset(tmp_path):
    log = tmp_path / "a.log"; log.write_text("")
    st = log.stat()
    bad = {
        "resolved_path": str((tmp_path / "WRONG.log").resolve()),
        "st_dev": st.st_dev, "st_ino": st.st_ino, "mode": st.st_mode & 0o777,
    }
    with pytest.raises(R.StepCBlock) as ei:
        R.verify_log_identity(log_path=log, frozen=bad)
    assert ei.value.code == "WRONG_LOG_IDENTITY"
    ctx = R.make_ctx()
    ctx.observed_keysets = {"U": {1}, "E": {3, 4}, "R0": {3, 4}, "RW": {3, 4}}
    ctx.real_runner_attempt_count = 4
    ctx.planned_physical_update_count = 10
    ctx.cuda_model_forward_backward_completed_count = 10
    ctx.cpu_q_acc_apply_completed_count = 10
    ctx.resume_arm_call_count = 3
    ctx.resume_artifact_load_count = 3
    ctx.model_checkpoint_materialization_call_count = 1
    ctx.parent_checkpoint_deserialize_count = 1
    with pytest.raises(R.StepCBlock) as ei2:
        R.finalize_counters(ctx, tmp_path / "empty", ())
    assert ei2.value.code == "STEP_KEYSET_MISMATCH"


def test_cut_fork_zero_updates_and_markers():
    holders_s = {"E": {}, "R0": {}, "RW": {}}
    holders_b = {"E": {}, "R0": {}, "RW": {}}
    event = SimpleNamespace(step=2, states={"s": 1}, carry_backlog=None)

    def ok_resume(**kwargs):
        name = kwargs["arm"].value
        pol = "exact_preserve" if name == "E" else "zero_strip"
        return SimpleNamespace(
            meta={"pre_cut_source_sha256": "fp", "policy": pol},
            tensor_states={"k": name}, deferred_backlog={},
        )

    ctx = R.make_ctx()
    R.run_cut_fork_at_u_step2(
        event=event, ctx=ctx, holders_states=holders_s, holders_backlog=holders_b,
        scratch=Path("/tmp"), resume_fn=ok_resume, pre_cut_fp_fn=lambda *a: "fp",
        arm_id_cls=ArmId,
    )
    assert ctx.fork_complete and ctx.resume_arm_call_count == 3 and ctx.cut_fork_updates_during == 0
    assert "cut_fork_serialize_load" in ctx.phase_durations


def test_record_step_uses_corrected_observer():
    ctx = R.make_ctx()
    model, batch = FakeCudaModel(), _batch()
    states = {"m": _cpu_state()}
    R.record_step(ctx, arm="U", step=1, model=model, batch=batch, states=states)
    assert ctx.cuda_model_forward_backward_completed_count == 1
    bad = dict(batch); bad.pop("inputs")
    with pytest.raises(R.StepCBlock) as ei:
        R.record_step(ctx, arm="U", step=2, model=model, batch=bad, states=states)
    assert ei.value.code == "CUDA_LOOP_FAILURE"
    legacy = {"input_ids": FakeCudaTensor()}
    with pytest.raises(R.StepCBlock) as ei2:
        R.record_step(ctx, arm="U", step=3, model=model, batch=legacy, states=states)
    assert ei2.value.code == "CUDA_LOOP_FAILURE"


def test_compose_hooks_e_requires_fork():
    ctx = R.make_ctx()
    model, batch = FakeCudaModel(), _batch()
    hooks = R.compose_hooks(
        ctx=ctx, model=model, batch=batch,
        holders_states={"E": {}, "R0": {}, "RW": {}},
        holders_backlog={"E": {}, "R0": {}, "RW": {}},
        scratch=Path("/tmp"),
        resume_fn=lambda **k: None, pre_cut_fp_fn=lambda *a: "fp", arm_id_cls=ArmId,
    )
    ev = SimpleNamespace(step=3, states={"m": _cpu_state()}, carry_backlog=None)
    with pytest.raises(R.StepCBlock) as ei:
        hooks["E"](ev)
    assert ei.value.code == "FORK_INCOMPLETE"
    ctx.fork_complete = True
    hooks["E"](ev)
    assert 3 in ctx.observed_keysets["E"]


def test_source_pins_present():
    assert R.SOURCE_PINS["contracts"] == (
        "19ae9024cce39b0edefe0f146036cb618369e1442eee6163ebd827a5d254f2cd"
    )
    for key in ("observer", "reducers", "learner", "accumulator", "event_coded_carrier"):
        assert key in R.SOURCE_PINS and any(k == key for _, k in R.PIN_RELS)
    assert "runtime" not in R.SOURCE_PINS  # external packet/preflight pins runtime
    assert R.PREIMPLEMENTATION_HEAD_BASELINE == (
        "200632f5c1f0ebf64aa0cf8f4b3ed012a3aef6cb"
    )
    assert not hasattr(R, "HEAD_PIN")


def test_expected_head_fail_closed_and_match(tmp_path, monkeypatch):
    repo = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
    monkeypatch.delenv(R.EXPECTED_HEAD_ENV, raising=False)
    with pytest.raises(R.StepCBlock) as ei:
        R.verify_head_origin_pin(repo=repo)
    assert ei.value.code == "HEAD_PIN_UNRESOLVED"
    with pytest.raises(R.StepCBlock) as ei2:
        R.verify_head_origin_pin(repo=repo, expected_head="UNRESOLVED_POST_COMMIT")
    assert ei2.value.code == "HEAD_PIN_UNRESOLVED"
    with pytest.raises(R.StepCBlock) as ei_mal:
        R.verify_head_origin_pin(repo=repo, expected_head="not-a-valid-sha")
    assert ei_mal.value.code == "HEAD_PIN_UNRESOLVED"

    fake_head = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    fake_origin = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"

    def _git_double(cmd, cwd=None, text=False, **kwargs):
        args = [str(x) for x in cmd]
        if "rev-parse" not in args:
            raise AssertionError(f"unexpected subprocess cmd={args!r}")
        if "HEAD" in args:
            out = fake_head
        elif "origin/feature/hrm-text-1.58" in args:
            out = fake_head
        else:
            raise AssertionError(f"unexpected rev-parse cmd={args!r}")
        return out + ("\n" if text else "")

    monkeypatch.setattr(R.subprocess, "check_output", _git_double)
    ok = R.verify_head_origin_pin(repo=repo, expected_head=fake_head)
    assert ok == fake_head
    with pytest.raises(R.StepCBlock) as ei_wrong:
        R.verify_head_origin_pin(repo=repo, expected_head="deadbeef" * 5)
    assert ei_wrong.value.code == "SOURCE_OR_CHECKPOINT_DRIFT"

    def _mismatch_double(cmd, cwd=None, text=False, **kwargs):
        args = [str(x) for x in cmd]
        if "HEAD" in args:
            out = fake_head
        elif "origin/feature/hrm-text-1.58" in args:
            out = fake_origin
        else:
            raise AssertionError(f"unexpected rev-parse cmd={args!r}")
        return out + ("\n" if text else "")

    monkeypatch.setattr(R.subprocess, "check_output", _mismatch_double)
    with pytest.raises(R.StepCBlock) as ei_mm:
        R.verify_head_origin_pin(repo=repo, expected_head=fake_head)
    assert ei_mm.value.code == "SOURCE_OR_CHECKPOINT_DRIFT"


def test_new_module_tamper_refused(tmp_path):
    repo = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
    parent = repo / (
        "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_"
        "rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
    )
    # Temporarily break observer pin expectation via monkeypatch of SOURCE_PINS copy path:
    real = dict(R.SOURCE_PINS)
    R.SOURCE_PINS["observer"] = "0" * 64
    try:
        with pytest.raises(R.StepCBlock) as ei:
            R.verify_source_and_parent_pins(repo=repo, parent=parent)
        assert ei.value.code == "SOURCE_OR_CHECKPOINT_DRIFT"
        assert "observer" in str(ei.value)
    finally:
        R.SOURCE_PINS.clear()
        R.SOURCE_PINS.update(real)
