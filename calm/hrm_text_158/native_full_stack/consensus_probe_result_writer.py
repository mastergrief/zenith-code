"""Probe-results JSONL writer for selector_support_consensus_v0 launch harness."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def coerce_nonneg_int(name: str, value: str | int) -> int:
    """Single robust parse site for launcher-fed numeric fields."""
    if isinstance(value, int):
        if value < 0:
            raise ValueError(f"{name}: negative integer {value!r}")
        return value
    text = str(value).strip()
    if not text or "\n" in text or "\r" in text or not text.isdigit():
        raise ValueError(f"{name}: invalid non-negative integer {value!r}")
    return int(text)


def append_probe_result_jsonl(
    probe_results_path: Path | str,
    *,
    probe_num: str | int,
    label: str,
    arm: str,
    exit_code: str | int,
    wall_s: str | int,
    heartbeats: str | int,
    scratch_root: Path | str,
) -> dict[str, Any]:
    scratch = Path(scratch_root)
    receipt_path = scratch / "receipt.json"
    receipt_exists = receipt_path.is_file()
    steps_completed: Any = "?"
    if receipt_exists:
        try:
            receipt_data = json.loads(receipt_path.read_text(encoding="utf-8"))
            steps_completed = receipt_data.get("steps_completed", "?")
        except (json.JSONDecodeError, OSError):
            steps_completed = "?"

    last_active_phase: dict[str, Any] | None = None
    lap_path = scratch / "last_active_phase.json"
    if lap_path.is_file():
        try:
            loaded = json.loads(lap_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                last_active_phase = loaded
        except (json.JSONDecodeError, OSError):
            last_active_phase = None

    row: dict[str, Any] = {
        "probe_num": coerce_nonneg_int("probe_num", probe_num),
        "label": str(label),
        "arm": str(arm),
        "exit_code": coerce_nonneg_int("exit_code", exit_code),
        "wall_s": coerce_nonneg_int("wall_s", wall_s),
        "receipt": bool(receipt_exists),
        "steps_completed": steps_completed,
        "heartbeats": coerce_nonneg_int("heartbeats", heartbeats),
    }
    if last_active_phase is not None:
        row["last_active_phase"] = last_active_phase

    path = Path(probe_results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
    return row
