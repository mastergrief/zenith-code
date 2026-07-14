"""Characterize three live probe dict-blindness forms — why ordered event log is required.

PROOF SURFACES (independent; each fails if its probe ordering marker moves):
1. same-key overwrite — step_reports[str(step)] = ... collapses duplicates
2. update-without-event — durable states= rebind then require_q_change can raise
   before the success step_reports write
3. event-without-update — headroom step_reports write + break before states= rebind

Tests read probe source/AST order markers only; this module does not wire production.
"""
from __future__ import annotations

import ast
from pathlib import Path

from calm.hrm_text_158.native_full_stack.forgotten_accum_ordered_apply_event import (
    ExpectedIdentity,
    characterize_dict_same_key_blindness,
    make_success_apply_event,
    snapshot_ordered_apply_event_log,
    validate_ordered_apply_event_sequence,
)

_PROBE = (
    Path(__file__).resolve().parents[3]
    / "scripts"
    / "hrm_text_158_bounded_delta_acquisition_probe.py"
)


def _probe_source() -> str:
    return _PROBE.read_text(encoding="utf-8")


def _line_of(source: str, needle: str, *, start: int = 0) -> int:
    idx = source.find(needle, start)
    assert idx >= 0, f"missing probe marker: {needle!r}"
    return source.count("\n", 0, idx) + 1


def test_proof1_same_key_overwrite_blindness_hides_missing_plus_duplicate():
    """PROOF 1 — same-key overwrite/collapse (step_reports dict semantics)."""

    ordered = [
        make_success_apply_event(
            seq=0, arm_id="U", optimizer_step_id=1,
            q_changed_count=0, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=1, arm_id="U", optimizer_step_id=1,
            q_changed_count=1, tensor_state_key_count=1,
        ),
        make_success_apply_event(
            seq=2, arm_id="U", optimizer_step_id=2,
            q_changed_count=0, tensor_state_key_count=1,
        ),
    ]
    blind = characterize_dict_same_key_blindness(ordered)
    assert set(blind) == {1, 2}
    assert len(blind) == 2
    assert len(ordered) == 3

    summary = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(ordered),
        ExpectedIdentity(arm_id="U", start_step=1, steps=3),
    )
    assert summary["observed_count"] == 3
    assert summary["missing_count"] >= 1
    assert summary["duplicate_count"] >= 1
    assert summary["sequence_exact_ok"] is False

    # Live marker: success path assigns step_reports[str(step)] (dict overwrite).
    src = _probe_source()
    assert "step_reports[str(step)] = {" in src


def test_proof1_same_key_overwrite_cannot_prove_sequence_exact_under_adversarial_dup():
    """PROOF 1 companion — dict collapse vs ordered validator on duplicate step id."""

    perfect = [
        make_success_apply_event(
            seq=i, arm_id="U", optimizer_step_id=1 + i,
            q_changed_count=0, tensor_state_key_count=1,
        )
        for i in range(3)
    ]
    assert len(characterize_dict_same_key_blindness(perfect)) == 3
    adversarial = list(perfect) + [
        make_success_apply_event(
            seq=3, arm_id="U", optimizer_step_id=2,
            q_changed_count=9, tensor_state_key_count=9,
        )
    ]
    assert len(characterize_dict_same_key_blindness(adversarial)) == 3
    bad = validate_ordered_apply_event_sequence(
        snapshot_ordered_apply_event_log(adversarial),
        ExpectedIdentity(arm_id="U", start_step=1, steps=3),
    )
    assert bad["observed_count"] == 4
    assert bad["sequence_exact_ok"] is False
    assert bad["duplicate_count"] >= 1


def test_proof2_update_without_event_states_rebind_before_require_q_then_report():
    """PROOF 2 — update-without-event control-flow order in live probe source.

    Durable ``states = step_result.tensor_states`` precedes ``require_q_change``
    raise, which precedes the success ``step_reports[str(step)] =`` write.
    If require_q_change fires, state already rebound but no new report event.
    """

    src = _probe_source()
    states_rebind = "states = step_result.tensor_states"
    require_raise = (
        'raise RuntimeError("bounded-delta step produced no q movement '
        'under --require-q-change")'
    )
    # Headroom path also writes step_reports; success report is the later one
    # that includes start_step (absent from headroom breach report).
    success_report_unique = '"start_step": int(start_step),'

    line_states = _line_of(src, states_rebind)
    line_require = _line_of(src, require_raise)
    # Success report block: first step_reports assign AFTER states rebind that
    # also contains start_step field.
    idx_states = src.find(states_rebind)
    idx_success = src.find(success_report_unique, idx_states)
    assert idx_success > idx_states
    line_success = src.count("\n", 0, idx_success) + 1
    # Nearest preceding step_reports[str(step)] = { before start_step field.
    report_assign = src.rfind("step_reports[str(step)] = {", idx_states, idx_success)
    assert report_assign > idx_states
    line_report = src.count("\n", 0, report_assign) + 1

    assert line_states < line_require < line_report, (
        f"update-without-event order broken: "
        f"states@{line_states} require@{line_require} report@{line_report}"
    )
    # Behavioral fixture: rebound happened; report dict never gained the step key.
    step_reports: dict[str, dict] = {}
    states = {"rebound": True}
    q_changed_count = 0
    require_q_change = True
    if require_q_change and q_changed_count <= 0:
        # mirrors probe: raise before step_reports write
        assert "7" not in step_reports
        assert states.get("rebound") is True
    else:  # pragma: no cover
        step_reports["7"] = {"event": True}
        raise AssertionError("require_q_change path not exercised")


def test_proof3_event_without_update_headroom_report_before_break_skips_states_rebind():
    """PROOF 3 — event-without-update: headroom report+break before states= rebind."""

    src = _probe_source()
    tree = ast.parse(src)
    fn = None
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "run_bounded_delta_steps":
            fn = node
            break
    assert fn is not None

    # Locate headroom terminated branch: step_reports assign, then break, and
    # ensure that branch's break precedes the function-level states rebind line.
    headroom_report = _line_of(
        src,
        'step_reports[str(step)] = {\n                            "loss":',
    )
    # break after headroom breach (steps_completed = step then break)
    headroom_break = _line_of(src, "steps_completed = step\n                        break")
    states_rebind = _line_of(src, "states = step_result.tensor_states")

    assert headroom_report < headroom_break < states_rebind, (
        f"event-without-update order broken: "
        f"report@{headroom_report} break@{headroom_break} states@{states_rebind}"
    )
    # Headroom report uses pre_apply_states (not rebound post-update states).
    attach_marker = "post_update_states=pre_apply_states"
    attach_line = _line_of(src, attach_marker)
    assert headroom_report < attach_line < headroom_break

    # Behavioral fixture: report written without durable rebind.
    step_reports: dict[str, dict] = {}
    states = {"generation": 0}
    step_reports["3"] = {"headroom_breach": True, "q_changed_count": 0}
    # break — never execute states = step_result.tensor_states
    assert "3" in step_reports
    assert states["generation"] == 0
