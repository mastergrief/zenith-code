"""Bankable receipt compactness guard for probe step_reports surfaces."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

# Mirrors TIER_A_PROBE_RECEIPT_INDEX_SURFACE_KEYS in the probe harness.
TIER_A_INLINE_INDEX_SURFACES: frozenset[str] = frozenset(
    {
        "pre_veto_selected_indices",
        "applied_indices",
        "post_veto_would_apply_pre_cap_indices",
        "replay_ce_veto_indices",
    }
)

RECEIPT_BANKABLE_MAX_BYTES = 10 * 1024 * 1024
RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN = 64


def _sha16(indices: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def summarize_inline_index_surface(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {
            "tier_a_index_surface_omitted": True,
            "value_type": type(value).__name__,
        }
    indices = [int(item) for item in value]
    return {
        "tier_a_index_surface_omitted": True,
        "len": len(indices),
        "applied_flat_indices_hash16": _sha16(indices),
    }


def compact_tensor_stats_for_bankable_receipt(
    tensor_stats: Mapping[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for state_key, stats in tensor_stats.items():
        if not isinstance(stats, dict):
            compact[str(state_key)] = stats
            continue
        row = dict(stats)
        for surface_key in TIER_A_INLINE_INDEX_SURFACES:
            if surface_key not in row:
                continue
            raw = row.pop(surface_key)
            row[f"{surface_key}_summary"] = summarize_inline_index_surface(raw)
        compact[str(state_key)] = row
    return compact


def compact_step_reports_for_bankable_receipt(
    step_reports: Mapping[str, Any],
) -> dict[str, Any]:
    compact: dict[str, Any] = {}
    for step_id, report in step_reports.items():
        if not isinstance(report, dict):
            compact[str(step_id)] = report
            continue
        row = dict(report)
        tensor_stats = row.get("tensor_stats")
        if isinstance(tensor_stats, dict):
            row["tensor_stats"] = compact_tensor_stats_for_bankable_receipt(tensor_stats)
        compact[str(step_id)] = row
    return compact


def compact_probe_receipt_for_banking(receipt: dict[str, Any]) -> dict[str, Any]:
    """Replace raw tier-A index arrays with compact summaries (receipt shape only)."""

    step_reports = receipt.get("step_reports")
    if isinstance(step_reports, dict):
        receipt["step_reports"] = compact_step_reports_for_bankable_receipt(step_reports)
    receipt["receipt_compactness_guard_applied"] = True
    receipt["receipt_compactness_guard_schema"] = (
        "hrm_text_158_probe_receipt_compactness_guard/v0"
    )
    return receipt


def find_raw_inline_index_violations(
    receipt: Mapping[str, Any],
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> list[str]:
    failures: list[str] = []
    step_reports = receipt.get("step_reports")
    if not isinstance(step_reports, dict):
        return failures
    for step_id, report in sorted(step_reports.items(), key=lambda item: str(item[0])):
        if not isinstance(report, dict):
            continue
        tensor_stats = report.get("tensor_stats")
        if not isinstance(tensor_stats, dict):
            continue
        for state_key, stats in tensor_stats.items():
            if not isinstance(stats, dict):
                continue
            for surface_key in TIER_A_INLINE_INDEX_SURFACES:
                if surface_key not in stats:
                    continue
                raw = stats[surface_key]
                if isinstance(raw, list) and len(raw) > max_inline_len:
                    failures.append(
                        f"step={step_id} state={state_key} surface={surface_key} "
                        f"len={len(raw)}"
                    )
    return failures


def estimate_receipt_json_bytes(receipt: Mapping[str, Any]) -> int:
    return len(
        json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode("utf-8")
    )


def validate_bankable_probe_receipt(
    receipt: Mapping[str, Any],
    *,
    max_bytes: int = RECEIPT_BANKABLE_MAX_BYTES,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> list[str]:
    failures = find_raw_inline_index_violations(
        receipt, max_inline_len=max_inline_len
    )
    size_bytes = estimate_receipt_json_bytes(receipt)
    if size_bytes > max_bytes:
        failures.append(
            f"receipt_json_bytes={size_bytes} exceeds bankable cap {max_bytes}"
        )
    return failures
