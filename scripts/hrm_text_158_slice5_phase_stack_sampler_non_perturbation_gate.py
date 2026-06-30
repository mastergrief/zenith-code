#!/usr/bin/env python3
"""G1 non-perturbation gate for phase stack ring sampler (Slice B-DIAG)."""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.phase_stack_ring_sampler import (
    PhaseStackRingSampler,
    is_false_green_stack_text,
    stack_text_contains_target_frame,
)

GATE_SCHEMA = "hrm_text_158_slice5_phase_stack_sampler_non_perturbation_gate/v1"
PREREG_EPSILON_SECONDS = 0.050
G1_SMOKE_INTERVAL_SECONDS = 0.05
G1_SMOKE_PHASE_SECONDS = 0.15
PREREG_RING_CAPACITY = 10
G1_TARGET_FRAME = "_synthetic_phase_duration"


def _synthetic_phase_duration(
    *,
    sampler: PhaseStackRingSampler | None,
    phase_seconds: float,
    ring_jsonl: Path | None = None,
) -> tuple[float, int]:
    durable_before_stop = 0
    if sampler is not None:
        sampler.start("synthetic_budgeted_phase", flush_path=ring_jsonl)
    start = time.perf_counter()
    time.sleep(float(phase_seconds))
    if sampler is not None:
        durable_before_stop = sampler.durable_jsonl_line_count()
        sampler.stop()
        if ring_jsonl is not None:
            sampler.flush_jsonl(ring_jsonl)
    return time.perf_counter() - start, int(durable_before_stop)


def run_non_perturbation_gate(
    *,
    epsilon_seconds: float = PREREG_EPSILON_SECONDS,
    ring_jsonl: Path | None = None,
    interval_seconds: float = G1_SMOKE_INTERVAL_SECONDS,
    phase_seconds: float = G1_SMOKE_PHASE_SECONDS,
) -> dict[str, Any]:
    failures: list[str] = []
    threads_before = threading.active_count()
    off_duration, _ = _synthetic_phase_duration(sampler=None, phase_seconds=phase_seconds)
    sampler = PhaseStackRingSampler(
        ring_capacity=PREREG_RING_CAPACITY,
        interval_seconds=float(interval_seconds),
    )
    on_duration, durable_before_stop = _synthetic_phase_duration(
        sampler=sampler,
        phase_seconds=phase_seconds,
        ring_jsonl=ring_jsonl,
    )
    threads_after = threading.active_count()
    delta = abs(on_duration - off_duration)
    if delta > float(epsilon_seconds):
        failures.append(
            f"duration_delta_exceeds_epsilon:{delta:.6f}>{epsilon_seconds:.6f}"
        )
    if sampler.sample_count <= 0:
        failures.append("sampler_sample_count_zero")
    if sampler.thread_alive:
        failures.append("sampler_thread_still_alive_after_stop")
    if threads_after > threads_before:
        failures.append(
            f"thread_count_increased:{threads_before}->{threads_after}"
        )
    ring_lines = 0
    stack_has_real_target_frame = False
    stack_false_green = False
    if ring_jsonl is not None and ring_jsonl.is_file():
        lines = [
            json.loads(line)
            for line in ring_jsonl.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        ring_lines = len(lines)
        if lines:
            last_stack = str(lines[-1].get("stack_text", ""))
            stack_false_green = is_false_green_stack_text(last_stack)
            stack_has_real_target_frame = stack_text_contains_target_frame(
                last_stack,
                G1_TARGET_FRAME,
            )
    if ring_jsonl is not None and ring_lines <= 0:
        failures.append("ring_jsonl_missing_or_empty")
    if durable_before_stop <= 0:
        failures.append("durable_jsonl_lines_before_stop_zero")
    if stack_false_green:
        failures.append("stack_text_false_green_exception_signature")
    if sampler.sample_count > 0 and not stack_has_real_target_frame:
        failures.append("stack_text_missing_real_target_frame")
    return {
        "schema": GATE_SCHEMA,
        "sampler_non_perturbation_pass": not failures,
        "preregistered_epsilon_seconds": float(epsilon_seconds),
        "g1_smoke_interval_seconds": float(interval_seconds),
        "g1_smoke_phase_seconds": float(phase_seconds),
        "threads_before": int(threads_before),
        "threads_after": int(threads_after),
        "off_duration_seconds": round(off_duration, 6),
        "on_duration_seconds": round(on_duration, 6),
        "duration_delta_seconds": round(delta, 6),
        "sampler_sample_count": int(sampler.sample_count),
        "ring_jsonl_lines": int(ring_lines),
        "durable_jsonl_lines_before_stop": int(durable_before_stop),
        "stack_has_real_target_frame": bool(stack_has_real_target_frame),
        "stack_false_green_exception_signature": bool(stack_false_green),
        "g1_target_frame": G1_TARGET_FRAME,
        "failures": failures,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epsilon-seconds",
        type=float,
        default=PREREG_EPSILON_SECONDS,
    )
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--ring-jsonl", type=Path, default=None)
    args = parser.parse_args(argv)
    receipt = run_non_perturbation_gate(
        epsilon_seconds=float(args.epsilon_seconds),
        ring_jsonl=args.ring_jsonl,
    )
    text = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0 if receipt.get("sampler_non_perturbation_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
