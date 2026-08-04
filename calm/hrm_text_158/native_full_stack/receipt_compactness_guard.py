"""Bankable receipt compactness guard for probe receipts (recursive)."""
from __future__ import annotations

import copy
import hashlib
import json
import re
from typing import Any, Mapping, MutableMapping, Sequence

import numpy as np

# Legacy alias: kept for importers; recursive visitor uses QUALIFYING set below.
TIER_A_INLINE_INDEX_SURFACES: frozenset[str] = frozenset(
    {
        "pre_veto_selected_indices",
        "applied_indices",
        "post_veto_would_apply_pre_cap_indices",
        "replay_ce_veto_indices",
    }
)

QUALIFYING_RAW_INDEX_SURFACE_KEYS: frozenset[str] = frozenset(
    {
        "pre_veto_selected_indices",
        "applied_indices",
        "post_veto_would_apply_pre_cap_indices",
        "post_veto_applied_indices",
        "replay_ce_veto_indices",
        "global_rate_cap_deferred_indices",
        "global_rate_cap_accepted_indices",
        # Event-coded observed surfaces (name members of the class rule).
        "crossing_flat_indices",
        "applied_flat_indices",
    }
)

# Class rule: any list-of-int leaf under a key matching these suffixes is bankable-raw
# when len > max_inline_len (covers unnamed emitters of the same shape).
_INDEX_KEY_CLASS_RE = re.compile(r"(?:_indices|_flat_indices)$")

IDENTITY_SIGNAL_INDEX_FIELDS: frozenset[str] = frozenset(
    {
        "global_rate_cap_accepted_indices",
        "global_rate_cap_deferred_indices",
        "post_veto_applied_indices",
        "replay_ce_veto_indices",
    }
)

RECEIPT_BANKABLE_MAX_BYTES = 10 * 1024 * 1024
RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN = 64
BANKABLE_INDEX_SURFACE_SUMMARY_SCHEMA = "hrm_text_158_bankable_index_surface_summary/v1"
BANKABLE_Q_SNAPSHOT_SUMMARY_SCHEMA = "hrm_text_158_bankable_q_snapshot_summary/v1"
RECEIPT_COMPACTNESS_GUARD_SCHEMA = "hrm_text_158_probe_receipt_compactness_guard/v1"
DECISIVE_Q_SNAPSHOT_KEY = "decisive_q_snapshot"


class ReceiptCompactnessCollisionError(ValueError):
    """Existing sibling count/hash disagrees with the compacted index list."""


def _sha16(indices: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in indices], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def canonical_int64_index_list_sha256_v1(indices: Sequence[int]) -> str:
    """Full SHA-256 matching global_rate_cap._tensor_sha256(torch.int64 tensor).

    Encoding: sha256(b\"torch.int64\" + str(shape).encode() + int64_le_tobytes).
    Pure (no torch import). Summary hash16 is a different contract and must not
    be treated as equivalent to this digest.
    """

    arr = np.asarray([int(value) for value in indices], dtype=np.int64)
    digest = hashlib.sha256()
    digest.update(b"torch.int64")
    digest.update(str(tuple(arr.shape)).encode("utf-8"))
    digest.update(arr.tobytes())
    return digest.hexdigest()


def _as_int_index_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    indices: list[int] = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            return None
        indices.append(int(item))
    return indices


def _key_matches_index_class(key: str) -> bool:
    return bool(_INDEX_KEY_CLASS_RE.search(str(key)))


def qualifies_as_raw_index_array(
    key: str,
    value: Any,
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> bool:
    """True when this leaf is a bankable-raw integer index array.

    Class rule (primary): key matches ``*_indices`` / ``*_flat_indices`` AND
    value is a list-of-int with len > max_inline_len.

    Named members (QUALIFYING_RAW_INDEX_SURFACE_KEYS) are documented instances of
    the same class, not a closed exclusive set.
    """

    key_s = str(key)
    if key_s not in QUALIFYING_RAW_INDEX_SURFACE_KEYS and not _key_matches_index_class(
        key_s
    ):
        return False
    indices = _as_int_index_list(value)
    if indices is None:
        return False
    return len(indices) > int(max_inline_len)


def qualifies_as_raw_q_snapshot(key: str, value: Any) -> bool:
    """Event-coded decisive_q_snapshot: flat→q dict that must not stay raw."""

    if str(key) != DECISIVE_Q_SNAPSHOT_KEY:
        return False
    if not isinstance(value, dict) or not value:
        return False
    # Require int-like keys/values (stringified flat indices are normal).
    for k, v in value.items():
        try:
            int(k)
        except (TypeError, ValueError):
            return False
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)):
            return False
    return True


def summarize_inline_index_surface(value: Any) -> dict[str, Any]:
    if not isinstance(value, list):
        return {
            "schema": BANKABLE_INDEX_SURFACE_SUMMARY_SCHEMA,
            "schema_version": BANKABLE_INDEX_SURFACE_SUMMARY_SCHEMA,
            "tier_a_index_surface_omitted": True,
            "value_type": type(value).__name__,
        }
    indices = [int(item) for item in value]
    count = len(indices)
    return {
        "schema": BANKABLE_INDEX_SURFACE_SUMMARY_SCHEMA,
        "schema_version": BANKABLE_INDEX_SURFACE_SUMMARY_SCHEMA,
        "tier_a_index_surface_omitted": True,
        "count": count,
        "len": count,
        "dtype": "int",
        "shape": [count],
        "order_sensitive_content_hash16": _sha16(indices),
        "applied_flat_indices_hash16": _sha16(indices),
    }


def summarize_decisive_q_snapshot(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        return {
            "schema": BANKABLE_Q_SNAPSHOT_SUMMARY_SCHEMA,
            "tier_a_index_surface_omitted": True,
            "value_type": type(value).__name__,
        }
    items = sorted((int(k), int(v)) for k, v in value.items())
    keys = [k for k, _ in items]
    vals = [v for _, v in items]
    return {
        "schema": BANKABLE_Q_SNAPSHOT_SUMMARY_SCHEMA,
        "tier_a_index_surface_omitted": True,
        "count": len(items),
        "key_order_sensitive_hash16": _sha16(keys),
        "value_order_sensitive_hash16": _sha16(vals),
        "pair_order_sensitive_hash16": _sha16(
            [x for pair in items for x in pair]
        ),
    }


def _emit_identity_siblings(row: MutableMapping[str, Any], key: str, indices: Sequence[int]) -> None:
    count_key = f"{key}_count"
    sha_key = f"{key}_sha256"
    expected_count = int(len(indices))
    expected_sha = canonical_int64_index_list_sha256_v1(indices)

    if count_key in row:
        existing_count = row[count_key]
        if int(existing_count) != expected_count:
            raise ReceiptCompactnessCollisionError(
                f"{count_key} mismatch: existing={existing_count!r} expected={expected_count}"
            )
    else:
        row[count_key] = expected_count

    if sha_key in row:
        existing_sha = str(row[sha_key])
        if existing_sha != expected_sha:
            raise ReceiptCompactnessCollisionError(
                f"{sha_key} mismatch: existing={existing_sha!r} expected={expected_sha}"
            )
        if len(existing_sha) != 64:
            raise ReceiptCompactnessCollisionError(
                f"{sha_key} must be full 64-hex digest, got len={len(existing_sha)}"
            )
    else:
        row[sha_key] = expected_sha


def _compact_list_node(items: list[Any], *, max_inline_len: int) -> None:
    """Recurse into nested lists/dicts so list-of-list containers cannot escape."""

    for item in items:
        if isinstance(item, dict):
            _compact_mapping_node(item, max_inline_len=max_inline_len)
        elif isinstance(item, list):
            _compact_list_node(item, max_inline_len=max_inline_len)


def _scan_list_node(
    items: list[Any],
    *,
    max_inline_len: int,
    path: str,
    failures: list[str],
) -> None:
    """Independent scanner: same nested-list recursion as the transform."""

    for index, item in enumerate(items):
        child_path = f"{path}[{index}]"
        if isinstance(item, dict):
            _scan_mapping_node(
                item, max_inline_len=max_inline_len, path=child_path, failures=failures
            )
        elif isinstance(item, list):
            _scan_list_node(
                item, max_inline_len=max_inline_len, path=child_path, failures=failures
            )


def _compact_mapping_node(
    node: MutableMapping[str, Any],
    *,
    max_inline_len: int,
) -> None:
    for key in list(node.keys()):
        value = node[key]
        key_s = str(key)
        if qualifies_as_raw_index_array(key_s, value, max_inline_len=max_inline_len):
            indices = _as_int_index_list(value)
            assert indices is not None
            if key_s in IDENTITY_SIGNAL_INDEX_FIELDS:
                _emit_identity_siblings(node, key_s, indices)
            node.pop(key)
            node[f"{key_s}_summary"] = summarize_inline_index_surface(indices)
            continue
        if qualifies_as_raw_q_snapshot(key_s, value):
            node.pop(key)
            node[f"{key_s}_summary"] = summarize_decisive_q_snapshot(value)
            continue
        if isinstance(value, dict):
            _compact_mapping_node(value, max_inline_len=max_inline_len)
        elif isinstance(value, list):
            _compact_list_node(value, max_inline_len=max_inline_len)


def _scan_mapping_node(
    node: Mapping[str, Any],
    *,
    max_inline_len: int,
    path: str,
    failures: list[str],
) -> None:
    for key, value in node.items():
        key_s = str(key)
        child_path = f"{path}.{key_s}" if path else key_s
        if qualifies_as_raw_index_array(key_s, value, max_inline_len=max_inline_len):
            indices = _as_int_index_list(value)
            assert indices is not None
            failures.append(f"{child_path} len={len(indices)}")
            continue
        if qualifies_as_raw_q_snapshot(key_s, value):
            failures.append(f"{child_path} decisive_q_snapshot_count={len(value)}")
            continue
        if isinstance(value, dict):
            _scan_mapping_node(
                value, max_inline_len=max_inline_len, path=child_path, failures=failures
            )
        elif isinstance(value, list):
            _scan_list_node(
                value,
                max_inline_len=max_inline_len,
                path=child_path,
                failures=failures,
            )


def compact_tensor_stats_for_bankable_receipt(
    tensor_stats: Mapping[str, Any],
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> dict[str, Any]:
    """Compat helper: compact one tensor_stats mapping via the recursive visitor."""

    compact = copy.deepcopy(dict(tensor_stats))
    _compact_mapping_node(compact, max_inline_len=max_inline_len)
    return compact


def compact_step_reports_for_bankable_receipt(
    step_reports: Mapping[str, Any],
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> dict[str, Any]:
    """Compat helper: compact step_reports via the recursive visitor."""

    compact = copy.deepcopy(dict(step_reports))
    _compact_mapping_node(compact, max_inline_len=max_inline_len)
    return compact


def compact_probe_receipt_for_banking(
    receipt: dict[str, Any],
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> dict[str, Any]:
    """Replace over-threshold raw index arrays with summaries (transform-on-copy).

    Mutates ``receipt`` to the compacted form for probe call-site compatibility,
    but builds the transform on a deep copy so the caller's pre-compact nested
    raw lists remain intact if they retained separate references.
    """

    compacted = copy.deepcopy(receipt)
    _compact_mapping_node(compacted, max_inline_len=max_inline_len)
    compacted["receipt_compactness_guard_applied"] = True
    compacted["receipt_compactness_guard_schema"] = RECEIPT_COMPACTNESS_GUARD_SCHEMA
    receipt.clear()
    receipt.update(compacted)
    return receipt


def find_raw_inline_index_violations(
    receipt: Mapping[str, Any],
    *,
    max_inline_len: int = RECEIPT_BANKABLE_MAX_INLINE_INDEX_LEN,
) -> list[str]:
    failures: list[str] = []
    if isinstance(receipt, dict):
        _scan_mapping_node(
            receipt, max_inline_len=max_inline_len, path="", failures=failures
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


def census_receipt_key_bytes(
    receipt: Mapping[str, Any],
    *,
    max_depth: int = 4,
    top_n: int = 40,
) -> dict[str, Any]:
    """Bounded per-key byte census OUTSIDE bankable emission (sizes only).

    Does not include raw array contents — only path, type, optional len, and
    compact-json byte size of each node up to max_depth.
    """

    rows: list[dict[str, Any]] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > max_depth:
            return
        try:
            size = len(
                json.dumps(node, separators=(",", ":"), sort_keys=True, default=str).encode(
                    "utf-8"
                )
            )
        except (TypeError, ValueError):
            size = -1
        entry: dict[str, Any] = {
            "path": path or "$",
            "type": type(node).__name__,
            "json_bytes": size,
        }
        if isinstance(node, (list, dict, str)):
            entry["len"] = len(node)
        rows.append(entry)
        if isinstance(node, dict) and depth < max_depth:
            for key, value in node.items():
                child = f"{path}.{key}" if path else str(key)
                walk(value, child, depth + 1)
        elif isinstance(node, list) and depth < max_depth:
            # Do not expand large raw lists element-wise (would re-materialize).
            if len(node) <= 8 and all(isinstance(x, dict) for x in node):
                for index, item in enumerate(node):
                    walk(item, f"{path}[{index}]", depth + 1)

    walk(receipt, "", 0)
    rows.sort(key=lambda r: int(r.get("json_bytes") or 0), reverse=True)
    return {
        "schema": "hrm_text_158_receipt_key_byte_census/v1",
        "total_json_bytes": estimate_receipt_json_bytes(receipt)
        if isinstance(receipt, Mapping)
        else -1,
        "n_nodes_visited": len(rows),
        "top": rows[: int(top_n)],
    }
