"""CPU characterization for bounded_delta_runner_hook_contract (BEFORE probe wire).

Strengthening B: nested backlog fixture must be byte/field-equivalent to
canonical private ``_clone_backlog_for_front_c`` (bounded_delta_learner.py:1922).
Mismatch = STOP/re-plan — NEVER re-baseline.
"""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    _clone_backlog_for_front_c,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent,
    clone_deferred_backlog,
    invoke_post_step_hook,
    seed_initial_deferred_backlog,
)


def _nested_backlog_fixture() -> dict[str, dict[int, dict[str, int]]]:
    """≥2 state keys, ≥2 flat indices, mixed fields (co_lead strengthening B)."""

    return {
        "mod.a.weight": {
            7: {"vote": 3, "age": 1, "defer_count": 0},
            19: {"vote": -2, "age": 4, "defer_count": 2, "extra": 9},
        },
        "mod.b.weight": {
            0: {"vote": 1, "age": 0},
            64: {"vote": 0, "age": 8, "defer_count": 1},
        },
    }


def test_clone_deferred_backlog_byte_equivalent_to_front_c_private():
    src = _nested_backlog_fixture()
    facade = clone_deferred_backlog(src)
    canonical = _clone_backlog_for_front_c(src)
    assert facade == canonical
    assert json.dumps(facade, sort_keys=True, separators=(",", ":")) == json.dumps(
        canonical, sort_keys=True, separators=(",", ":")
    )


def test_clone_deferred_backlog_none_and_empty():
    assert clone_deferred_backlog(None) == {}
    assert clone_deferred_backlog({}) == {}
    assert _clone_backlog_for_front_c(None) == clone_deferred_backlog(None)


def test_seed_initial_deferred_backlog_none_is_none():
    assert seed_initial_deferred_backlog(None) is None


def test_seed_initial_defensive_mutation_isolation():
    freeze = _nested_backlog_fixture()
    freeze_copy = copy.deepcopy(freeze)
    seeded = seed_initial_deferred_backlog(freeze)
    assert seeded is not None
    seeded["mod.a.weight"][7]["vote"] = 999
    seeded["mod.a.weight"][99] = {"vote": 1}
    del seeded["mod.b.weight"]
    assert freeze == freeze_copy  # original freeze untouched


def test_clone_deferred_backlog_mutation_isolation():
    src = _nested_backlog_fixture()
    src_copy = copy.deepcopy(src)
    cloned = clone_deferred_backlog(src)
    cloned["mod.a.weight"][7]["age"] = 12345
    assert src == src_copy


def test_invoke_post_step_hook_none_is_noop():
    event = BoundedDeltaPostStepEvent(step=4, states={}, carry_backlog=None)
    invoke_post_step_hook(None, event)  # must not raise


def test_invoke_post_step_hook_propagates_exception_fail_closed():
    def _boom(_event: BoundedDeltaPostStepEvent) -> None:
        raise RuntimeError("hook_boom")

    event = BoundedDeltaPostStepEvent(
        step=16,
        states={"k": object()},
        carry_backlog={"k": {1: {"vote": 1}}},
        step_batch_metadata={"row_ids": ["a"]},
    )
    with pytest.raises(RuntimeError, match="hook_boom"):
        invoke_post_step_hook(_boom, event)


def test_post_step_event_fields_and_live_states_identity():
    states = {"live": object()}
    backlog = _nested_backlog_fixture()
    meta = {"batch_index": 3}
    seen: list[BoundedDeltaPostStepEvent] = []

    def _capture(event: BoundedDeltaPostStepEvent) -> None:
        seen.append(event)

    event = BoundedDeltaPostStepEvent(
        step=28,
        states=states,
        carry_backlog=backlog,
        step_batch_metadata=meta,
    )
    invoke_post_step_hook(_capture, event)
    assert len(seen) == 1
    assert seen[0].step == 28
    assert seen[0].states is states  # LIVE identity documented
    assert seen[0].carry_backlog is backlog
    assert seen[0].step_batch_metadata == meta


def test_hook_contract_imports_forbid_probe_cli_gpu():
    import calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract as mod

    src = Path(mod.__file__).read_text(encoding="utf-8")
    forbidden = ("scripts.", "argparse", "torch.cuda", "ai_room", "fork_b_resume_parity_science_driver")
    for token in forbidden:
        assert token not in src, f"forbidden import/token {token!r} in hook facade"


def test_baseline_pre_edit_anchors_recorded():
    """Pre-edit baseline artifact must exist (characterization-first order)."""

    path = Path(
        "/home/gabe/claw-code-creditdir/transient_fp_credit/"
        "fork_b_runner_hook_baseline_pre_edit_v1.json"
    )
    assert path.is_file(), "missing pre-edit baseline — characterization skipped"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["anchors"]["post_step_hook_ABSENT_pre"] is True
    assert payload["anchors"]["audit_callback_param"] is True


def test_audit_callback_signature_contract_still_documented():
    """audit_callback remains (step, states)->dict; new hooks are separate params."""

    import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe

    sig = inspect.signature(probe.run_bounded_delta_steps)
    audit = sig.parameters["audit_callback"]
    assert "Callable" in str(audit.annotation) or audit.annotation is not inspect._empty
