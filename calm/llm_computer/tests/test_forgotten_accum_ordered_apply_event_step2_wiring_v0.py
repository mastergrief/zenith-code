"""STEP2 wiring — probe refuse/append placement + ark ownership/anti-stub (CPU)."""
from __future__ import annotations

import inspect
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ATTACHMENT_KEY,
    APPLY_OUTCOME_SUCCESS,
    OrderedApplyEventLogRefuse,
    PRODUCER_LITERAL,
    make_success_apply_event,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
    EFFECTIVE_STAMP_KEY,
    build_forgotten_accum_runner_contract,
)
import calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ark_invoke as ark_mod
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_ark_invoke import (
    invoke_arm_with_a_rk,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
ARK = (
    REPO
    / "calm/hrm_text_158/native_full_stack/forgotten_accum_training_equivalence_ark_invoke.py"
)


def _probe_src() -> str:
    return PROBE.read_text(encoding="utf-8")


def _line(src: str, needle: str, *, start: int = 0) -> int:
    idx = src.find(needle, start)
    assert idx >= 0, needle
    return src.count("\n", 0, idx) + 1


def _eff_updater(*, horizon: int = 4) -> dict:
    c = build_forgotten_accum_runner_contract(runway_steps=horizon)
    return {
        EFFECTIVE_STAMP_KEY: {
            **c.as_pins_dict(),
            "global_cap_resolved_spec_present": True,
            "within_arm_consistent": True,
        }
    }


def _nine_tuple(*, horizon: int = 4):
    return ({}, _eff_updater(horizon=horizon), {}, {}, "ok", 0, None, None, [])


def _appending_runner(*, mutate_after: bool = False, mismatch: bool = False):
    """Spy runner: appends SUCCESS events like probe; returns valid 9-tuple."""

    seen: dict = {}

    def runner(*args, **kwargs):
        log = kwargs.get("ordered_apply_event_log")
        arm_id = kwargs.get("ordered_apply_event_arm_id")
        steps = int(kwargs["steps"])
        start = int(kwargs.get("start_step", 1))
        seen["log_id"] = id(log)
        seen["arm_id"] = arm_id
        seen["kwargs"] = kwargs
        if log is not None:
            n = max(0, steps - 1) if mismatch else steps
            for i in range(n):
                log.append(
                    make_success_apply_event(
                        seq=i,
                        arm_id=str(arm_id),
                        optimizer_step_id=start + i,
                        q_changed_count=0,
                        tensor_state_key_count=1,
                    )
                )
            if mutate_after:
                seen["live_log"] = log
        return _nine_tuple(horizon=int(kwargs["global_horizon"]))

    runner.seen = seen  # type: ignore[attr-defined]
    return runner


def _rk(horizon: int = 4) -> dict:
    return build_forgotten_accum_runner_contract(runway_steps=horizon).as_runner_kwargs()


# --- PROBE boundary -----------------------------------------------------------


def test_probe_refuse_non_list_and_non_empty_before_model_train():
    model = MagicMock()
    model.train.side_effect = AssertionError("model.train must not run")
    with pytest.raises(OrderedApplyEventLogRefuse, match="built-in list"):
        run_bounded_delta_steps(
            model, {}, {}, {}, device="cpu", steps=1,
            require_q_change=False, max_abs_per_tensor=1,
            ordered_apply_event_log=(),  # type: ignore[arg-type]
            ordered_apply_event_arm_id="U",
        )
    model.train.assert_not_called()
    with pytest.raises(OrderedApplyEventLogRefuse, match="must be empty"):
        run_bounded_delta_steps(
            model, {}, {}, {}, device="cpu", steps=1,
            require_q_change=False, max_abs_per_tensor=1,
            ordered_apply_event_log=[{"prefill": True}],
            ordered_apply_event_arm_id="U",
        )
    model.train.assert_not_called()


def test_probe_ast_append_immediately_after_states_rebind_before_downstream():
    src = _probe_src()
    states = "states = step_result.tensor_states"
    append = "ordered_apply_event_log.append("
    carrier = "if carrier_growth_collector is not None:"
    hook = "invoke_post_step_hook("
    require = (
        'raise RuntimeError("bounded-delta step produced no q movement '
        'under --require-q-change")'
    )
    report = '"start_step": int(start_step),'
    i_states = src.find(states)
    i_append = src.find(append, i_states)
    i_carrier = src.find(carrier, i_states)
    i_hook = src.find(hook, i_states)
    i_req = src.find(require, i_states)
    i_report = src.find(report, i_states)
    assert i_states < i_append < i_carrier < i_hook < i_req < i_report
    assert "make_success_apply_event(" in src[i_append : i_append + 400]
    assert PRODUCER_LITERAL in src or "make_success_apply_event" in src


def test_probe_signature_none_disabled_and_nine_tuple_return():
    import typing

    sig = inspect.signature(run_bounded_delta_steps)
    assert sig.parameters["ordered_apply_event_log"].default is None
    assert sig.parameters["ordered_apply_event_arm_id"].default is None
    hints = typing.get_type_hints(run_bounded_delta_steps)
    ret = hints["return"]
    assert typing.get_origin(ret) is tuple
    assert len(typing.get_args(ret)) == 9


def test_probe_bind_real_signature_accepts_private_list_kwargs():
    from calm.hrm_text_158.native_full_stack.forgotten_accum_runner_contract import (
        bind_real_run_bounded_delta_steps,
    )

    log: list = []
    args = (object(), {}, {}, {})
    kw = {
        "device": "cpu",
        "steps": 1,
        "require_q_change": False,
        "max_abs_per_tensor": 4096,
        "global_horizon": 4,
        "global_cap_contract": "c1_banked_faithful_long_run_global_cap",
        "r7_deferred_backlog_carry_enabled": True,
        "ordered_apply_event_log": log,
        "ordered_apply_event_arm_id": "U",
    }
    bound = bind_real_run_bounded_delta_steps(run_bounded_delta_steps, args, kw)
    assert bound.arguments["ordered_apply_event_log"] is log
    assert bound.arguments["ordered_apply_event_arm_id"] == "U"


# --- ARK boundary + anti-stub -------------------------------------------------


def test_ark_fresh_private_list_per_arm_and_runner_appends_success():
    r_u = _appending_runner()
    r_e = _appending_runner()
    receipts: list = []
    inv: list = []
    contract = build_forgotten_accum_runner_contract(runway_steps=4)
    invoke_arm_with_a_rk(
        r_u, object(), {}, {}, {}, "cpu", 2, 1, 4, None, None, False, None,
        _rk(), "U", inv, runner_contract=contract, a_rk_receipts=receipts,
    )
    invoke_arm_with_a_rk(
        r_e, object(), {}, {}, {}, "cpu", 2, 1, 4, None, None, False, None,
        _rk(), "E", inv, runner_contract=contract, a_rk_receipts=receipts,
    )
    assert r_u.seen["log_id"] != r_e.seen["log_id"]
    assert r_u.seen["arm_id"] == "U" and r_e.seen["arm_id"] == "E"
    su = receipts[0][ATTACHMENT_KEY]
    se = receipts[1][ATTACHMENT_KEY]
    assert su["sequence_exact_ok"] is True
    assert se["sequence_exact_ok"] is True
    assert su["arm_id"] == "U" and se["arm_id"] == "E"
    assert su["claimable"] is False and su["runtime_proven"] is False


def test_ark_normal_return_attaches_summary_then_existing_a_rk_fields():
    runner = _appending_runner()
    receipts: list = []
    invoke_arm_with_a_rk(
        runner, object(), {}, {}, {}, "cpu", 3, 5, 4, None, None, False, None,
        _rk(), "U", [], runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
        a_rk_receipts=receipts,
    )
    row = receipts[0]
    assert ATTACHMENT_KEY in row
    assert row["three_way_equality_pass"] is True
    summary = row[ATTACHMENT_KEY]
    assert summary["schema_id"].endswith("_validation_v1")
    assert summary["expected_count"] == 3
    assert summary["observed_count"] == 3
    assert summary["sequence_exact_ok"] is True
    assert summary["bankable"] is False
    assert summary["forensic_only"] is True


def test_ark_summary_attached_and_live_discarded_before_a_rk_check_failure(
    monkeypatch,
):
    """Temporal order: attach+discard precede existing A-RK checks; raise propagates."""

    runner = _appending_runner(mutate_after=True)
    receipts: list = []
    order: list[str] = []

    def boom_horizon(**_kwargs):
        order.append("horizon_check")
        assert len(receipts) == 1
        assert ATTACHMENT_KEY in receipts[0]
        assert receipts[0][ATTACHMENT_KEY]["claimable"] is False
        assert receipts[0][ATTACHMENT_KEY]["sequence_exact_ok"] is True
        # Live list must already be discarded before this existing A-RK check.
        assert runner.seen["live_log"] == []
        raise RuntimeError("forced a_rk horizon failure")

    monkeypatch.setattr(ark_mod, "assert_horizon_matches_runway", boom_horizon)
    with pytest.raises(RuntimeError, match="forced a_rk horizon failure"):
        invoke_arm_with_a_rk(
            runner,
            object(),
            {},
            {},
            {},
            "cpu",
            2,
            1,
            4,
            None,
            None,
            False,
            None,
            _rk(),
            "U",
            [],
            runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
            a_rk_receipts=receipts,
        )
    assert order == ["horizon_check"]
    # Summary remains attached even though the existing A-RK check raised.
    assert len(receipts) == 1
    assert ATTACHMENT_KEY in receipts[0]
    assert "three_way_equality_pass" not in receipts[0]
    assert runner.seen["live_log"] == []


def test_ark_mismatch_nonclaimable_and_exception_no_summary():
    receipts: list = []
    invoke_arm_with_a_rk(
        _appending_runner(mismatch=True),
        object(), {}, {}, {}, "cpu", 3, 1, 4, None, None, False, None,
        _rk(), "U", [], runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
        a_rk_receipts=receipts,
    )
    bad = receipts[0][ATTACHMENT_KEY]
    assert bad["sequence_exact_ok"] is False
    assert bad["claimable"] is False
    assert bad["runtime_proven"] is False

    def boom(*_a, **_k):
        raise RuntimeError("runner exploded")

    receipts2: list = []
    with pytest.raises(RuntimeError, match="runner exploded"):
        invoke_arm_with_a_rk(
            boom, object(), {}, {}, {}, "cpu", 1, 1, 4, None, None, False, None,
            _rk(), "U", [], runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
            a_rk_receipts=receipts2,
        )
    assert receipts2 == []


def test_ark_post_snapshot_live_mutation_cannot_alter_attachment():
    runner = _appending_runner(mutate_after=True)
    receipts: list = []
    invoke_arm_with_a_rk(
        runner, object(), {}, {}, {}, "cpu", 2, 1, 4, None, None, False, None,
        _rk(), "U", [], runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
        a_rk_receipts=receipts,
    )
    before = dict(receipts[0][ATTACHMENT_KEY])
    live = runner.seen["live_log"]
    # Ark discards via clear(); any further append must not change attached summary.
    live.append(
        make_success_apply_event(
            seq=99, arm_id="X", optimizer_step_id=999,
            q_changed_count=0, tensor_state_key_count=0,
        )
    )
    assert receipts[0][ATTACHMENT_KEY] == before
    assert before["observed_count"] == 2
    assert before["sequence_exact_ok"] is True


def test_ark_anti_stub_calls_real_ark_and_real_runner_signature_surface():
    assert invoke_arm_with_a_rk.__module__.endswith(
        "forgotten_accum_training_equivalence_ark_invoke"
    )
    assert run_bounded_delta_steps.__module__.endswith(
        "hrm_text_158_bounded_delta_acquisition_probe"
    ) or "hrm_text_158_bounded_delta_acquisition_probe" in run_bounded_delta_steps.__module__
    src = ARK.read_text(encoding="utf-8")
    assert "validate_ordered_apply_event_sequence" in src
    assert "ordered_apply_event_log: list" in src or "ordered_apply_event_log =" in src
    assert "ATTACHMENT_KEY" in src
    # Behavioral: real ark + real signature bind already exercised above; fake
    # runner cannot manufacture claimable evidence.
    receipts: list = []
    invoke_arm_with_a_rk(
        _appending_runner(),
        object(), {}, {}, {}, "cpu", 1, 1, 4, None, None, False, None,
        _rk(), "FAKE", [], runner_contract=build_forgotten_accum_runner_contract(runway_steps=4),
        a_rk_receipts=receipts,
    )
    s = receipts[0][ATTACHMENT_KEY]
    assert s["sequence_exact_ok"] is True
    assert s["claimable"] is False
    assert s["runtime_proven"] is False
    assert APPLY_OUTCOME_SUCCESS == "SUCCESS"
