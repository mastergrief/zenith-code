"""Non-killing periodic stack ring sampler for budgeted probe phases (Slice B-DIAG)."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque


PHASE_STACK_RING_SAMPLER_ENV = "HRM_TEXT_158_PHASE_STACK_RING_SAMPLER"
RING_SAMPLER_SCHEMA = "hrm_text_158_phase_stack_ring_sample/v1"
FALSE_GREEN_MARKERS = ("_capture_sample", "AttributeError", "fileno")


def phase_stack_ring_sampler_enabled() -> bool:
    return os.environ.get(PHASE_STACK_RING_SAMPLER_ENV) == "1"


def is_false_green_stack_text(stack_text: str) -> bool:
    """True when stack_text is the prior faulthandler-buffer exception signature."""
    text = str(stack_text)
    return "_capture_sample" in text and "AttributeError" in text


def stack_text_contains_target_frame(stack_text: str, target: str) -> bool:
    return str(target) in str(stack_text) and not is_false_green_stack_text(stack_text)


ClockFn = Callable[[], float]


@dataclass
class PhaseStackRingSampler:
    """Daemon sampler: captures stacks to a ring buffer without exiting/killing."""

    ring_capacity: int = 10
    interval_seconds: float = 30.0
    clock: ClockFn = time.perf_counter
    _ring: Deque[dict[str, Any]] = field(default_factory=deque, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _active_phase: str | None = field(default=None, init=False)
    _sample_count: int = field(default=0, init=False)
    _durable_flush_path: Path | None = field(default=None, init=False)

    @property
    def sample_count(self) -> int:
        return int(self._sample_count)

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def durable_flush_path(self) -> Path | None:
        return self._durable_flush_path

    def start(self, phase: str, *, flush_path: Path | None = None) -> None:
        self.stop()
        self._active_phase = str(phase)
        self._durable_flush_path = flush_path
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"phase-stack-ring-{phase}",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            thread = self._thread
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError("phase_stack_ring_sampler_join_timeout")
            self._thread = None
        self._active_phase = None
        self._durable_flush_path = None

    def samples(self) -> list[dict[str, Any]]:
        return list(self._ring)

    def durable_jsonl_line_count(self) -> int:
        path = self._durable_flush_path
        if path is None or not path.is_file():
            return 0
        return sum(
            1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
        )

    def flush_jsonl(self, path: Path) -> None:
        """Append any in-memory ring rows not yet durably flushed (idempotent-safe)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        existing = 0
        if path.is_file():
            existing = sum(
                1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
            )
        with path.open("a", encoding="utf-8") as handle:
            for record in list(self._ring)[existing:]:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=float(self.interval_seconds)):
            self._capture_sample()

    def _capture_stack_text(self) -> str:
        frames = sys._current_frames()
        parts: list[str] = []
        for tid in sorted(frames):
            frame = frames[tid]
            parts.append(f"Thread {tid}:\n")
            parts.append("".join(traceback.format_stack(frame)))
            parts.append("\n")
        return "".join(parts)

    def _append_durable(self, record: dict[str, Any]) -> None:
        path = self._durable_flush_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _capture_sample(self) -> None:
        phase = self._active_phase or "unknown"
        stack_text = self._capture_stack_text()
        self._sample_count += 1
        record = {
            "schema": RING_SAMPLER_SCHEMA,
            "phase": phase,
            "sample_index": int(self._sample_count),
            "elapsed_seconds": float(self.clock()),
            "stack_text": stack_text,
        }
        self._ring.append(record)
        self._append_durable(record)
        while len(self._ring) > int(self.ring_capacity):
            self._ring.popleft()
