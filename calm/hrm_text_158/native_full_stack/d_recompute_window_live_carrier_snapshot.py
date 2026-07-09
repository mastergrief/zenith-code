"""Slice-5 Step-2 live carrier byte snapshot emitter (observer-only).

Stage A LEN-LOG (+1 1783590889937): also emits passive O(1) q_levels length
fields. NEW q_levels observer is len()-only — never materialize/copy q_levels.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
)

# Pre-registered residual-closure threshold (keys). Comparisons MUST use global.
GLOBAL_Q_LEVELS_DOMINANCE_THRESHOLD = 4_470_000


class DuplicateLiveCarrierObjectIdError(ValueError):
    """Fail-closed: same carrier object appears under multiple state keys in one step."""


class PerModuleQLevelsThresholdCompareError(ValueError):
    """Fail-closed: 4.47M threshold compared against a per-module length only."""


def initialize_live_carrier_snapshot_log(log_path: Path) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("", encoding="utf-8")


def _carrier_from_tensor_state(state: Any) -> EventCodedAccLiveState | None:
    carrier = getattr(state, "event_coded_live_carrier", None)
    if isinstance(carrier, EventCodedAccLiveState):
        return carrier
    return None


def collect_q_levels_length_pass(
    post_update_states: Mapping[str, Any],
) -> tuple[dict[str, int], int]:
    """PRE-ROW PASS: O(1) len per unique carrier; fail-closed on duplicate object ids.

    Returns (per_state_key_len, global_len_q_levels).
    """
    per_key_len: dict[str, int] = {}
    seen_ids: dict[int, str] = {}
    global_sum = 0
    for state_key in sorted(post_update_states):
        carrier = _carrier_from_tensor_state(post_update_states[state_key])
        if carrier is None:
            continue
        carrier_id = id(carrier)
        prior_key = seen_ids.get(carrier_id)
        if prior_key is not None:
            raise DuplicateLiveCarrierObjectIdError(
                f"duplicate live carrier object id within step: "
                f"state_key={state_key!r} shares id with {prior_key!r}"
            )
        seen_ids[carrier_id] = str(state_key)
        # PASSIVE / NO-ALLOC (NEW q_levels observer ONLY): len() only.
        length = len(carrier.q_levels)
        per_key_len[str(state_key)] = int(length)
        global_sum += int(length)
    return per_key_len, int(global_sum)


def assert_global_q_levels_threshold_unit(
    *,
    value: int,
    field_name: str,
    threshold: int = GLOBAL_Q_LEVELS_DOMINANCE_THRESHOLD,
) -> None:
    """UNIT WITNESS: threshold compares must name global_len_q_levels (or explicit sum)."""
    allowed = {
        "global_len_q_levels",
        "global_len_q_levels_sum",
        "explicit_sum_of_per_module_len_q_levels",
    }
    if str(field_name) not in allowed:
        raise PerModuleQLevelsThresholdCompareError(
            f"HARNESS_INVALID: threshold compare field {field_name!r} is not a "
            f"global unit witness (allowed={sorted(allowed)}); "
            f"refusing per-module-only compare against {threshold}"
        )
    # Touch value so callers pass an int; witness is about the field name/unit.
    int(value)


def emit_live_carrier_snapshots_for_probe_step(
    *,
    enabled: bool,
    log_path: Path | None,
    step: int,
    post_update_states: Mapping[str, Any],
) -> int:
    if not enabled or log_path is None:
        return 0
    per_key_len, global_len_q_levels = collect_q_levels_length_pass(post_update_states)
    emitted = 0
    with Path(log_path).open("a", encoding="utf-8") as handle:
        for state_key in sorted(post_update_states):
            carrier = _carrier_from_tensor_state(post_update_states[state_key])
            if carrier is None:
                continue
            carrier.assert_live_carrier_byte_counters_exact()
            key = str(state_key)
            row = {
                "step": int(step),
                "state_key": key,
                "logical_numel": int(carrier.logical_numel),
                **carrier.live_carrier_byte_snapshot(),
                "live_carrier_bytes_exact": True,
                # Stage A LEN-LOG fields (passive):
                "len_q_levels": int(per_key_len[key]),
                "global_len_q_levels": int(global_len_q_levels),
            }
            handle.write(json.dumps(row, sort_keys=True) + "\n")
            emitted += 1
    return int(emitted)
