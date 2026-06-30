"""G1 non-perturbation gate tests for phase stack ring sampler."""
from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.phase_stack_ring_sampler import (
    PhaseStackRingSampler,
)
from scripts.hrm_text_158_slice5_phase_stack_sampler_non_perturbation_gate import (
    G1_SMOKE_INTERVAL_SECONDS,
    G1_SMOKE_PHASE_SECONDS,
    PREREG_EPSILON_SECONDS,
    run_non_perturbation_gate,
)


def test_non_perturbation_gate_passes_with_forced_sample(tmp_path: Path) -> None:
    ring_jsonl = tmp_path / "phase_stack_ring_samples.jsonl"
    receipt = run_non_perturbation_gate(
        epsilon_seconds=PREREG_EPSILON_SECONDS,
        ring_jsonl=ring_jsonl,
    )
    assert receipt["sampler_non_perturbation_pass"] is True
    assert receipt["duration_delta_seconds"] <= PREREG_EPSILON_SECONDS
    assert receipt["sampler_sample_count"] > 0
    assert receipt["ring_jsonl_lines"] > 0
    assert receipt["durable_jsonl_lines_before_stop"] > 0
    assert receipt["stack_has_real_target_frame"] is True
    assert receipt["stack_false_green_exception_signature"] is False
    assert ring_jsonl.is_file()
    lines = [
        json.loads(line)
        for line in ring_jsonl.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == receipt["sampler_sample_count"]
    assert lines[0]["stack_text"]
    assert "_synthetic_phase_duration" in lines[-1]["stack_text"]


def test_non_perturbation_gate_fails_on_exception_only_stack_text(tmp_path: Path) -> None:
    import scripts.hrm_text_158_slice5_phase_stack_sampler_non_perturbation_gate as gate_mod

    class BrokenCaptureSampler(PhaseStackRingSampler):
        def _capture_stack_text(self) -> str:
            return (
                "Traceback (most recent call last):\n"
                '  File ".../phase_stack_ring_sampler.py", line 87, in _capture_sample\n'
                "AttributeError: fileno\n"
            )

    original = gate_mod.PhaseStackRingSampler
    gate_mod.PhaseStackRingSampler = BrokenCaptureSampler
    try:
        receipt = run_non_perturbation_gate(
            epsilon_seconds=PREREG_EPSILON_SECONDS,
            ring_jsonl=tmp_path / "ring.jsonl",
        )
    finally:
        gate_mod.PhaseStackRingSampler = original
    assert receipt["sampler_non_perturbation_pass"] is False
    assert receipt["stack_false_green_exception_signature"] is True
    assert any(
        item in receipt["failures"]
        for item in (
            "stack_text_false_green_exception_signature",
            "stack_text_missing_real_target_frame",
        )
    )


def test_non_perturbation_gate_fails_on_slow_sampler() -> None:
    class SlowCaptureSampler(PhaseStackRingSampler):
        def _capture_sample(self) -> None:
            time.sleep(0.2)
            super()._capture_sample()

    threads_before = __import__("threading").active_count()
    off = time.perf_counter()
    time.sleep(G1_SMOKE_PHASE_SECONDS)
    off_duration = time.perf_counter() - off
    sampler = SlowCaptureSampler(
        ring_capacity=10,
        interval_seconds=G1_SMOKE_INTERVAL_SECONDS,
    )
    sampler.start("synthetic_budgeted_phase")
    start = time.perf_counter()
    time.sleep(G1_SMOKE_PHASE_SECONDS)
    sampler.stop()
    on_duration = time.perf_counter() - start
    delta = abs(on_duration - off_duration)
    threads_after = __import__("threading").active_count()
    assert delta > PREREG_EPSILON_SECONDS
    assert threads_after <= threads_before


def test_non_perturbation_gate_receipt_fails_when_perturbing(tmp_path: Path) -> None:
    import scripts.hrm_text_158_slice5_phase_stack_sampler_non_perturbation_gate as gate_mod

    class SlowCaptureSampler(PhaseStackRingSampler):
        def _capture_sample(self) -> None:
            time.sleep(0.2)
            super()._capture_sample()

    original_cls = gate_mod.PhaseStackRingSampler
    gate_mod.PhaseStackRingSampler = SlowCaptureSampler
    try:
        receipt = run_non_perturbation_gate(
            epsilon_seconds=PREREG_EPSILON_SECONDS,
            ring_jsonl=tmp_path / "ring.jsonl",
        )
    finally:
        gate_mod.PhaseStackRingSampler = original_cls
    assert receipt["sampler_non_perturbation_pass"] is False
    assert receipt["sampler_sample_count"] > 0
    assert any("duration_delta_exceeds_epsilon" in item for item in receipt["failures"])


def test_sampler_stop_raises_on_join_timeout() -> None:
    sampler = PhaseStackRingSampler(interval_seconds=60.0)

    def _block_join() -> None:
        time.sleep(10.0)

    sampler._thread = __import__("threading").Thread(target=_block_join, daemon=True)
    sampler._thread.start()
    sampler._active_phase = "test"
    with pytest.raises(RuntimeError, match="join_timeout"):
        sampler.stop()
