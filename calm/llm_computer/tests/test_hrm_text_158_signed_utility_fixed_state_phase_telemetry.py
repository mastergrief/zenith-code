"""CPU-static tests for phase telemetry (PLAN v6 D1)."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_phase_telemetry import (
    PhaseBudgetBreach,
    PhaseBudgetClock,
    phase_marker_pair,
)

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_phase_telemetry.py"


def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 140


def test_phase_marker_pair():
    assert phase_marker_pair("MATERIALIZE") == ("PHASE_MATERIALIZE_BEGIN", "PHASE_MATERIALIZE_END")


def test_clock_markers_and_budget_ok():
    clock = PhaseBudgetClock({"MATERIALIZE": 2.0})
    assert clock.begin("MATERIALIZE") == "PHASE_MATERIALIZE_BEGIN"
    assert clock.end("MATERIALIZE") == "PHASE_MATERIALIZE_END"
    assert clock.markers["PHASE_MATERIALIZE_BEGIN"] is True
    assert clock.markers["PHASE_MATERIALIZE_END"] is True
    assert clock.receipt["phases"]["MATERIALIZE"]["budget_s"] == 2.0


def test_clock_budget_breach():
    clock = PhaseBudgetClock({"CAPTURE": 0.01})
    clock.begin("CAPTURE")
    time.sleep(0.02)
    with pytest.raises(PhaseBudgetBreach, match="phase_budget_breach:CAPTURE"):
        clock.end("CAPTURE")
    assert clock.receipt["breaches"]


def test_double_begin_and_end_without_begin():
    clock = PhaseBudgetClock({"X": 1.0})
    clock.begin("X")
    with pytest.raises(RuntimeError, match="phase_already_open"):
        clock.begin("X")
    clock2 = PhaseBudgetClock({"Y": 1.0})
    with pytest.raises(RuntimeError, match="phase_not_open"):
        clock2.end("Y")
