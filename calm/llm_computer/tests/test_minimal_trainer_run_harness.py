"""Dry orchestration characterization for the moved run harness (slice S1).

Stub producers only: no model, no GPU, no run root, no checkpoint. The property
under test is the single-process admission gate the frozen H2 packet carried and
this module now owns — on a smoke STOP the K=200 phase is unreachable in the same
flow, and on a smoke pass it is entered exactly once.

Task 1788428215079-af9995e7, slice S1. ADVISOR_ROUTE: 1788439962383-bd9b8b20.
"""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.minimal_trainer.run_harness import (
    SMOKE_STEPS,
    orchestrate,
)


def _smoke_stub(admission: str, calls: list[str]):
    def smoke_fn():
        calls.append("smoke")
        return (
            {
                "admission": admission,
                "steps_completed": SMOKE_STEPS if admission == "PASS_TO_K200" else 0,
            },
            "smoke-digest",
        )

    return smoke_fn


def _k200_stub(calls: list[str]):
    def k200_fn():
        calls.append("k200")
        return ({"phase": "k200"}, "terminal-digest")

    return k200_fn


def test_cpu_smoke_stop_leaves_k200_unreachable(capsys):
    calls: list[str] = []
    with pytest.raises(SystemExit) as exc:
        orchestrate(_smoke_stub("STOP_BEFORE_K200", calls), _k200_stub(calls))
    out = capsys.readouterr().out
    assert exc.value.code == 3
    assert calls == ["smoke"]
    assert "[STOP] SMOKE_ADMISSION_FAILED" in out
    assert "SMOKE_ADMITTED" not in out
    assert "RUN_COMPLETE" not in out


def test_cpu_smoke_pass_enters_k200_once(capsys):
    calls: list[str] = []
    rc = orchestrate(_smoke_stub("PASS_TO_K200", calls), _k200_stub(calls))
    out = capsys.readouterr().out
    assert rc == 0
    assert calls == ["smoke", "k200"]
    assert "[PROG] SMOKE_ADMITTED sha256=smoke-digest" in out
    assert "[TERMINAL] RUN_COMPLETE sha256=terminal-digest" in out
    assert "[STOP]" not in out
