"""CPU characterization: probe flip schedule default-off + per-step resolve."""
from __future__ import annotations

import ast
import inspect
from pathlib import Path

from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
    make_rw_absolute_flip_schedule,
    resolve_flip_application_deferred_for_step,
    rw_resolved_flags_for_absolute_window,
)

REPO = Path(__file__).resolve().parents[3]
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"


def test_resolve_default_none_matches_bool_path():
    assert resolve_flip_application_deferred_for_step(1) is False
    assert resolve_flip_application_deferred_for_step(1, flip_application_deferred=True) is True
    assert (
        resolve_flip_application_deferred_for_step(
            1, flip_application_deferred=True, flip_application_deferred_schedule=None
        )
        is True
    )


def test_resolve_schedule_overrides_bool():
    sched = lambda step: step == 7  # noqa: E731
    assert resolve_flip_application_deferred_for_step(
        7, flip_application_deferred=False, flip_application_deferred_schedule=sched
    )
    assert not resolve_flip_application_deferred_for_step(
        8, flip_application_deferred=True, flip_application_deferred_schedule=sched
    )


def test_rw_formal_window_exact_501_532_true_533_false():
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        flip_defer_schedule,
        ArmId,
    )

    flags = rw_resolved_flags_for_absolute_window(t_cut=500, W=32, through_step=533)
    assert flags[501] is True and flags[532] is True
    assert flags[533] is False
    assert sum(1 for s, v in flags.items() if v) == 32
    assert 500 not in flags
    # Matches contracts.flip_defer_schedule for formal W
    sched = make_rw_absolute_flip_schedule(t_cut=500, W=32)
    for step in range(501, 534):
        post = step - 500
        assert sched(step) == flip_defer_schedule(ArmId.RW, post_cut_step_index=post)


def test_local_W_schedule_two_step_transition():
    flags = rw_resolved_flags_for_absolute_window(t_cut=2, W=2, through_step=5)
    assert flags == {3: True, 4: True, 5: False}


def test_probe_signature_schedule_default_none_and_default_off_string():
    src = PROBE.read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next(
        n for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "run_bounded_delta_steps"
    )
    kw_map = dict(zip([a.arg for a in fn.args.kwonlyargs], fn.args.kw_defaults))
    # also positional-or-keyword trailing defaults
    pos_defaults = {
        a.arg: d
        for a, d in zip(fn.args.args[::-1], (fn.args.defaults or [])[::-1])
    }
    defaults = {**pos_defaults, **{k: v for k, v in kw_map.items() if v is not None}}
    assert "flip_application_deferred_schedule" in {
        a.arg for a in fn.args.args + fn.args.kwonlyargs
    }
    # default None
    node = kw_map.get("flip_application_deferred_schedule") or defaults.get(
        "flip_application_deferred_schedule"
    )
    assert node is not None
    assert isinstance(node, ast.Constant) and node.value is None
    # Phase-B default-off substring preserved
    assert "flip_application_deferred=bool(flip_application_deferred)" in src
    assert "flip_application_deferred_schedule(int(step))" in src


def test_probe_bind_partial_accepts_schedule_none():
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps

    sig = inspect.signature(run_bounded_delta_steps)
    assert "flip_application_deferred_schedule" in sig.parameters
    assert sig.parameters["flip_application_deferred_schedule"].default is None
    # bind_partial proves kwargs surface without invoking GPU path
    sig.bind_partial(
        flip_application_deferred=False,
        flip_application_deferred_schedule=None,
    )
    sched = make_rw_absolute_flip_schedule(t_cut=500, W=32)
    sig.bind_partial(flip_application_deferred_schedule=sched)
