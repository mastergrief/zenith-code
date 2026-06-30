"""Non-killing periodic stack ring sampler for budgeted probe phases (Slice B-DIAG)."""
from __future__ import annotations

import faulthandler
import json
import os
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Deque, TextIO


PHASE_STACK_RING_SAMPLER_ENV = "HRM_TEXT_158_PHASE_STACK_RING_SAMPLER"
RING_SAMPLER_SCHEMA = "hrm_text_158_phase_stack_ring_sample/v1"


def phase_stack_ring_sampler_enabled() -> bool:
    return os.environ.get(PHASE_STACK_RING_SAMPLER_ENV) == "1"


ClockFn = Callable[[], float]


@dataclass
class PhaseStackRingSampler:
    """Daemon sampler: dumps stacks to a ring buffer without exiting/killing."""

    ring_capacity: int = 10
    interval_seconds: float = 30.0
    clock: ClockFn = time.perf_counter
    _ring: Deque[dict[str, Any]] = field(default_factory=deque, init=False)
    _thread: threading.Thread | None = field(default=None, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _active_phase: str | None = field(default=None, init=False)
    _sample_count: int = field(default=0, init=False)

    @property
    def sample_count(self) -> int:
        return int(self._sample_count)

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, phase: str) -> None:
        self.stop()
        self._active_phase = str(phase)
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

    def samples(self) -> list[dict[str, Any]]:
        return list(self._ring)

    def flush_jsonl(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            for record in self._ring:
                handle.write(json.dumps(record, sort_keys=True) + "\n")

    def _loop(self) -> None:
        while not self._stop_event.wait(timeout=float(self.interval_seconds)):
            self._capture_sample()

    def _capture_sample(self) -> None:
        phase = self._active_phase or "unknown"
        buffer = _StackTraceBuffer()
        try:
            faulthandler.dump_traceback(file=buffer, all_threads=True)
        except Exception:
            buffer.write("".join(traceback.format_exc()))
        self._sample_count += 1
        record = {
            "schema": RING_SAMPLER_SCHEMA,
            "phase": phase,
            "sample_index": int(self._sample_count),
            "elapsed_seconds": float(self.clock()),
            "stack_text": buffer.getvalue(),
        }
        self._ring.append(record)
        while len(self._ring) > int(self.ring_capacity):
            self._ring.popleft()


class _StackTraceBuffer:
    """Minimal text buffer for faulthandler.dump_traceback."""

    def __init__(self) -> None:
        self._parts: list[str] = []

    def write(self, text: str) -> None:
        self._parts.append(str(text))

    def flush(self) -> None:
        return None

    def getvalue(self) -> str:
        return "".join(self._parts)
