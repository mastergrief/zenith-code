"""Bounded-steps triage tests (Slice B-DIAG)."""
from __future__ import annotations

from pathlib import Path

from scripts.hrm_text_158_slice5_bounded_steps_triage import triage_bounded_steps

V6D_FIXTURE = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "slice5_re_m4_v6d_2189e72024_corrected_null"
)


def test_v6d_triage_wrapper_budget_too_tight() -> None:
    assert V6D_FIXTURE.is_dir(), f"missing pinned fixture: {V6D_FIXTURE}"
    receipt = triage_bounded_steps(run_root=V6D_FIXTURE, max_steps_hard=3)
    assert receipt["bounded_steps_triage_class"] == "WRAPPER_BUDGET_TOO_TIGHT"
    assert receipt["instrumented_outer_timeout_after_max_steps"] is True
    assert receipt["baseline_sparse_cap_step_stall"] is True
