"""Slice-5 Step-2 live carrier byte snapshot emitter (observer-only)."""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)


def initialize_live_carrier_snapshot_log(log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def _carrier_from_tensor_state(state: Any) -> EventCodedAccLiveState | None:
    carrier = getattr(state, "event_coded_live_carrier", None)
    if isinstance(carrier, EventCodedAccLiveState):
        return carrier
    return None


def emit_live_carrier_snapshots_for_probe_step(
    *,
    enabled: bool,
    log_path: Path | None,
    step: int,
    post_update_states: Mapping[str, Any],
) -> int:
    if not enabled or log_path is None:
        return 0
    emitted = 0
    with Path(log_path).open("a", encoding="utf-8") as handle:
        for state_key in sorted(post_update_states):
            carrier = _carrier_from_tensor_state(post_update_states[state_key])
            if carrier is None:
                continue
            carrier.assert_live_carrier_byte_counters_exact()
            row = {
                "step": int(step),
                "state_key": str(state_key),
                "logical_numel": int(carrier.logical_numel),
                **carrier.live_carrier_byte_snapshot(),
                "live_carrier_bytes_exact": True,
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            emitted += 1
    return int(emitted)
