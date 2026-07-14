"""CPU characterization for forgotten-accum run-arms science driver."""
from __future__ import annotations

from pathlib import Path

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    FailClosedClass,
    PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    assert_carrier_preflight,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
    authoritative_state_fingerprint,
    make_cadence_cut_post_step_hook,
    resolve_flip_application_deferred_for_step,
    run_forgotten_accum_training_equivalence_arms,
    rw_resolved_flags_for_absolute_window,
)


def _states():
    q = torch.zeros(4, dtype=torch.int8)
    acc = torch.tensor([0, 3, 0, -4], dtype=torch.int16)
    return {"A": make_bounded_tensor_state("A", q, 1.0, acc)}


def _fake_runner_factory(resolved_log: list):
    def fake_runner(model, batch, tensor_states, eligible_modules, device=None, steps=1, **kw):
        start = int(kw.get("start_step", 1))
        hook = kw.get("post_step_hook")
        backlog = kw.get("initial_deferred_backlog") or {}
        states = dict(tensor_states)
        flip = bool(kw.get("flip_application_deferred", False))
        schedule = kw.get("flip_application_deferred_schedule")
        for i in range(int(steps)):
            step = start + i
            flag = resolve_flip_application_deferred_for_step(
                step,
                flip_application_deferred=flip,
                flip_application_deferred_schedule=schedule,
            )
            resolved_log.append({"arm_start": start, "step": step, "deferred": flag})
            if hook is not None:
                hook(
                    BoundedDeltaPostStepEvent(
                        step=step, states=states, carry_backlog=dict(backlog)
                    )
                )
        # Behavior-only fake AFTER real-signature bind in driver. Must emit A-EFF stamp
        # matching consumed pins so three-way equality is exercisable on CPU.
        from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
            EFFECTIVE_STAMP_KEY,
            finalize_a_eff_stamp_from_observations,
        )

        horizon = int(kw["global_horizon"])
        stamp = finalize_a_eff_stamp_from_observations(
            horizon_obs=[horizon],
            cap_name_obs=[str(kw["global_cap_contract"])],
            cap_resolved_obs=[True],
            max_abs_per_tensor=int(kw["max_abs_per_tensor"]),
            r7_consumed=bool(kw["r7_deferred_backlog_carry_enabled"]),
            require_q_change_consumed=bool(kw["require_q_change"]),
        )
        updater_config = {EFFECTIVE_STAMP_KEY: stamp}
        return ({}, updater_config, states, {}, "ok", int(steps), None, None, [])

    return fake_runner


def _cpu_saver(*, path: Path, model, event, config, source_pin):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "cpu_fake_cadence_checkpoint/v0",
        "step": int(event.step),
        "source_pin": str(source_pin),
        "fingerprint": authoritative_state_fingerprint(event.states, event.carry_backlog),
        "checkpoint_written": True,
    }
    torch.save(payload, path)
    return path


def test_preflight_refuses_wrong_carrier_cap_scope_event_coded():
    assert_carrier_preflight(
        live_acc_carrier_selector="NONE",
        global_cap_contract="c1_banked_faithful_long_run_global_cap",
        eligible_scope="all-bitlinear",
    )
    for kwargs in (
        {"live_acc_carrier_selector": "EVENT", "global_cap_contract": "c1_banked_faithful_long_run_global_cap", "eligible_scope": "all-bitlinear"},
        {"live_acc_carrier_selector": "NONE", "global_cap_contract": "other", "eligible_scope": "all-bitlinear"},
        {"live_acc_carrier_selector": "NONE", "global_cap_contract": "c1_banked_faithful_long_run_global_cap", "eligible_scope": "subset"},
        {"live_acc_carrier_selector": "NONE", "global_cap_contract": "c1_banked_faithful_long_run_global_cap", "eligible_scope": "all-bitlinear", "event_coded_flags_present": True},
    ):
        try:
            assert_carrier_preflight(**kwargs)
            raise AssertionError("expected refuse")
        except ValueError as exc:
            assert "PREFLIGHT_REFUSE" in str(exc)


def test_driver_one_call_per_arm_rw_schedule_cadence_and_control(tmp_path: Path):
    resolved: list = []
    runner = _fake_runner_factory(resolved)
    t_cut, W, runway = 2, 2, 6
    cadence = (2, 4, 6)
    result = run_forgotten_accum_training_equivalence_arms(
        runner=runner,
        model=object(),
        batch={},
        tensor_states=_states(),
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        t_cut=t_cut,
        runway_steps=runway,
        W=W,
        save_cadence=cadence,
        cadence_saver=_cpu_saver,
        developer_validation=True,
    )
    assert result.status == "OK"
    assert result.science_label is None
    assert result.bank_receipts is None
    assert result.notes.get("bank_section") == "suppressed"
    assert result.notes.get("ledger_claimable") is False
    assert result.arm_call_counts == {"U": 1, "E": 1, "R0": 1, "RW": 1}
    assert sum(1 for r in result.runner_invocations if r["arm"] == "RW") == 1
    assert result.zero_seed_proof == PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY
    # RW schedule on absolute steps 3,4 (W=2) True; step 5 False
    rw_rows = [r for r in resolved if r["arm_start"] == t_cut + 1 and any(
        inv["arm"] == "RW" and inv["start_step"] == t_cut + 1
        for inv in result.runner_invocations
    )]
    # Filter by matching RW invocation window only: last post-cut start is shared;
    # use schedule flags from result must-prove + resolved deferred for RW call.
    rw_inv = next(r for r in result.runner_invocations if r["arm"] == "RW")
    assert rw_inv["has_schedule"] is True
    rw_resolved = [
        r for r in resolved
        if r["step"] >= t_cut + 1 and r["deferred"] in (True, False)
    ]
    # Last post-cut arm is RW — its resolved flags are the trailing post-cut block
    post_cut = [r for r in resolved if r["arm_start"] == t_cut + 1]
    # three post-cut arms × 4 steps = 12; take last 4 as RW
    rw_block = post_cut[-4:]
    by_step = {r["step"]: r["deferred"] for r in rw_block}
    assert by_step[3] is True and by_step[4] is True
    assert by_step[5] is False and by_step[6] is False

    # Formal must-prove window helper (independent of reduced smoke sizes)
    formal = rw_resolved_flags_for_absolute_window(t_cut=500, W=32, through_step=533)
    assert formal[501] and formal[532] and not formal[533]

    # Cadence artifacts loadable + fingerprint equality
    assert result.cadence_fingerprint_pairs
    assert all(a == b for a, b in result.cadence_fingerprint_pairs)
    u_paths = result.cadence_paths_by_arm["U"]
    assert set(int(k) for k in u_paths) >= {2, 4, 6}
    loaded = torch.load(u_paths[2], map_location="cpu", weights_only=False)
    assert loaded["checkpoint_written"] is True
    assert loaded["step"] == 2


def test_driver_control_invalid_when_e_bank_diverges(tmp_path: Path):
    resolved: list = []
    # Bank-eval earliest-all-clear uses formal SAVE_CADENCE keys (not reduced smoke cadence).
    bank = {
        "U": {
            "acquire_pct": 100.0,
            "retain_pct_by_support": {"L0b": 100.0, "math_a0": 100.0},
            "clears_by_save": {250: True, 500: True, 1500: True},
            "parent_consistency_ok": True,
            "close_sibling_ok": True,
        },
        "E": {
            "acquire_pct": 50.0,
            "retain_pct_by_support": {"L0b": 50.0, "math_a0": 50.0},
            "clears_by_save": {250: False, 500: False, 1500: False},
            "parent_consistency_ok": True,
            "close_sibling_ok": True,
        },
        "R0": {
            "acquire_pct": 50.0,
            "retain_pct_by_support": {"L0b": 50.0, "math_a0": 50.0},
            "clears_by_save": {250: False, 500: False, 1500: False},
            "parent_consistency_ok": True,
            "close_sibling_ok": True,
        },
        "RW": {
            "acquire_pct": 50.0,
            "retain_pct_by_support": {"L0b": 50.0, "math_a0": 50.0},
            "clears_by_save": {250: False, 500: False, 1500: False},
            "parent_consistency_ok": True,
            "close_sibling_ok": True,
        },
    }
    result = run_forgotten_accum_training_equivalence_arms(
        runner=_fake_runner_factory(resolved),
        model=object(),
        batch={},
        tensor_states=_states(),
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        t_cut=2,
        runway_steps=6,
        W=2,
        save_cadence=(2, 6),
        cadence_saver=_cpu_saver,
        bank_inputs=bank,
        developer_validation=True,
    )
    assert result.status == "FAILURE"
    assert result.fail_closed_class == FailClosedClass.CONTROL_INVALID.value


def test_driver_preflight_refuse_short_circuits_before_runner(tmp_path: Path):
    calls = []

    def boom(*_a, **_k):
        calls.append(1)
        raise AssertionError("runner must not be called")

    result = run_forgotten_accum_training_equivalence_arms(
        runner=boom,
        model=object(),
        batch={},
        tensor_states=_states(),
        eligible_modules={},
        device="cpu",
        experiment_root=tmp_path,
        live_acc_carrier_selector="EVENT_CODED",
    )
    assert result.status == "REFUSED"
    assert calls == []
    assert result.science_label is None


def test_cadence_hook_non_mutation_fingerprint(tmp_path: Path):
    states = _states()
    backlog = {"A": {0: {"defer_count": 1}}}
    paths: dict[int, Path] = {}
    fps: list[tuple[str, str]] = []
    hook = make_cadence_cut_post_step_hook(
        model=object(),
        arm_root=tmp_path,
        cadence=(1,),
        cut_store={},
        t_cut=99,
        config={},
        source_pin="pin",
        cadence_paths=paths,
        fingerprints_pre_post=fps,
        saver=_cpu_saver,
    )
    pre = authoritative_state_fingerprint(states, backlog)
    hook(BoundedDeltaPostStepEvent(step=1, states=states, carry_backlog=backlog))
    post = authoritative_state_fingerprint(states, backlog)
    assert pre == post == fps[0][0] == fps[0][1]
    assert 1 in paths
    blob = torch.load(paths[1], map_location="cpu", weights_only=False)
    assert blob["fingerprint"] == pre
