"""W8 launch hygiene helpers — run-log JSONL parsing only (no classifier science)."""
from __future__ import annotations

import json
from pathlib import Path

BOUNDED_STEPS_START_PHASE = "bounded_steps"
BOUNDED_STEPS_START_EVENT = "start"


def count_bounded_steps_starts_in_run_log(run_log: Path) -> int:
    """Count bounded_steps start events in a probe run.log stream.

    run.log is a stdout tee that may contain bare JSON scalars (pretty-printed
    receipt blocks) mixed with JSONL telemetry dict rows. Non-dict values are
    skipped; only dict rows with event=start and phase=bounded_steps count.
    """
    if not run_log.is_file():
        return 0

    starts = 0
    for line in run_log.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(rec, dict):
            continue
        if (
            rec.get("event") == BOUNDED_STEPS_START_EVENT
            and rec.get("phase") == BOUNDED_STEPS_START_PHASE
        ):
            starts += 1
    return starts
