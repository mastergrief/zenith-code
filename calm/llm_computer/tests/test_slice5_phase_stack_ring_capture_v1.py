"""Ring capture tests — real frames, not false-green exception signatures."""
from __future__ import annotations

import time
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.phase_stack_ring_sampler import (
    PhaseStackRingSampler,
    is_false_green_stack_text,
    stack_text_contains_target_frame,
)

TARGET = "_synthetic_capture_target"


def _run_capture_target_phase(sampler: PhaseStackRingSampler, ring_jsonl: Path) -> None:
    sampler.start("synthetic_budgeted_phase", flush_path=ring_jsonl)
    time.sleep(0.12)
    sampler.stop()


def test_ring_capture_has_real_target_frame(tmp_path: Path) -> None:
    ring_jsonl = tmp_path / "ring.jsonl"
    sampler = PhaseStackRingSampler(interval_seconds=0.05)
    _run_capture_target_phase(sampler, ring_jsonl)
    assert sampler.sample_count > 0
    lines = [
        __import__("json").loads(line)
        for line in ring_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert lines
    stack_text = str(lines[-1]["stack_text"])
    assert not is_false_green_stack_text(stack_text)
    assert "phase_stack_ring_sampler.py" in stack_text
    assert "_run_capture_target_phase" in stack_text


def test_broken_faulthandler_capture_is_false_green() -> None:
    """RED fixture: prior faulthandler-buffer signature must fail the gate."""
    stack_text = (
        "Traceback (most recent call last):\n"
        '  File ".../phase_stack_ring_sampler.py", line 87, in _capture_sample\n'
        "    faulthandler.dump_traceback(file=buffer, all_threads=True)\n"
        "AttributeError: '_StackTraceBuffer' object has no attribute 'fileno'\n"
    )
    assert is_false_green_stack_text(stack_text)
    assert not stack_text_contains_target_frame(stack_text, TARGET)


def test_fixed_capture_passes_real_frame_check(tmp_path: Path) -> None:
    ring_jsonl = tmp_path / "ring.jsonl"
    sampler = PhaseStackRingSampler(interval_seconds=0.05)
    _run_capture_target_phase(sampler, ring_jsonl)
    stack_text = str(sampler.samples()[-1]["stack_text"])
    assert stack_text_contains_target_frame(stack_text, "_run_capture_target_phase")
