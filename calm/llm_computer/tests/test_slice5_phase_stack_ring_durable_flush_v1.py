"""Durable per-sample flush tests for phase stack ring sampler."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.phase_stack_ring_sampler import (
    PhaseStackRingSampler,
)


def test_durable_jsonl_written_before_clean_stop(tmp_path: Path) -> None:
    ring_jsonl = tmp_path / "liveness_stack_ring.jsonl"
    sampler = PhaseStackRingSampler(interval_seconds=0.05)
    sampler.start("synthetic_budgeted_phase", flush_path=ring_jsonl)
    time.sleep(0.12)
    lines_before_stop = sampler.durable_jsonl_line_count()
    assert lines_before_stop >= 1
    sampler.stop()
    final_lines = [
        json.loads(line)
        for line in ring_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(final_lines) >= lines_before_stop
    assert final_lines[0]["stack_text"]
    assert "AttributeError" not in final_lines[0]["stack_text"]


def test_mid_stall_kill_preserves_prior_flushed_lines(tmp_path: Path) -> None:
    """Append-as-captured: lines written before faulthandler kill remain readable."""
    ring_jsonl = tmp_path / "liveness_stack_ring.jsonl"
    sampler = PhaseStackRingSampler(interval_seconds=0.05)
    sampler.start("sparse_cap_apply", flush_path=ring_jsonl)
    time.sleep(0.12)
    captured_before_kill = sampler.durable_jsonl_line_count()
    assert captured_before_kill >= 1
    # Simulate fail-closed kill without clean stop/_exit_phase_stack.
    sampler._stop_event.set()
    sampler._thread = None
    sampler._active_phase = None
    lines = [
        json.loads(line)
        for line in ring_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == captured_before_kill
    assert lines[0]["phase"] == "sparse_cap_apply"
