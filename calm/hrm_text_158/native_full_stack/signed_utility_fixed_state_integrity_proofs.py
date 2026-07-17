"""Causal storage isolation + terminal precedence (PLAN v6 D1)."""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

INTEGRITY = "UNVERIFIED_INTEGRITY_OR_EXECUTION"
ASYMMETRY = "UNVERIFIED_ASYMMETRIC_INTERVENTION"
PRESENT = "SIGNED_CREDIT_SIGNAL_PRESENT_UNPROVEN"
NULL_OR_HARMFUL = "SIGNED_CREDIT_SIGNAL_NULL_OR_HARMFUL"


class IntegrityProofError(RuntimeError):
    pass


def _get(obj: Any, key: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _tensor_storage_span(t: Any) -> tuple[int, int] | None:
    """Return [data_ptr, data_ptr+nbytes) for the whole untyped storage, or None if empty."""
    if t is None:
        return None
    try:
        if int(t.numel()) == 0 and int(t.untyped_storage().nbytes()) == 0:
            return None
        storage = t.untyped_storage()
        ptr = int(storage.data_ptr())
        nbytes = int(storage.nbytes())
    except Exception as exc:  # noqa: BLE001 — surface as integrity failure upstream
        raise IntegrityProofError(f"storage_span_unavailable:{type(t).__name__}:{exc}") from exc
    if nbytes <= 0:
        return None
    return ptr, ptr + nbytes


def _collect_tensors(obj: Any, out: list[Any]) -> None:
    if obj is None:
        return
    if hasattr(obj, "untyped_storage") and hasattr(obj, "numel"):
        out.append(obj)
        return
    if isinstance(obj, Mapping):
        for v in obj.values():
            _collect_tensors(v, out)
        return
    for attr in (
        "q_levels",
        "frozen_scale",
        "exact_accumulator_shadow",
        "bounded_accumulator",
        "event_coded_live_carrier",
    ):
        if hasattr(obj, attr):
            _collect_tensors(getattr(obj, attr), out)
    for attr in ("values", "storage", "data", "payload"):
        if hasattr(obj, attr):
            _collect_tensors(getattr(obj, attr), out)


def enumerate_mutable_storage_spans(state_obj: Any) -> list[dict[str, Any]]:
    tensors: list[Any] = []
    _collect_tensors(state_obj, tensors)
    spans: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for t in tensors:
        span = _tensor_storage_span(t)
        if span is None:
            continue
        key = (span[0], span[1])
        if key in seen:
            continue
        seen.add(key)
        spans.append({"data_ptr": span[0], "end_ptr": span[1], "nbytes": span[1] - span[0]})
    return spans


def within_state_alias_topology(state_obj: Any) -> dict[str, Any]:
    """Record which tensors share the same underlying storage inside one state object."""
    tensors: list[Any] = []
    _collect_tensors(state_obj, tensors)
    by_ptr: dict[int, list[str]] = {}
    named = []
    for i, t in enumerate(tensors):
        span = _tensor_storage_span(t)
        label = f"t{i}:{getattr(t, 'dtype', '?')}:{tuple(getattr(t, 'shape', ()))}"
        named.append(label)
        if span is None:
            continue
        by_ptr.setdefault(span[0], []).append(label)
    aliases = {str(ptr): labs for ptr, labs in by_ptr.items() if len(labs) > 1}
    return {"tensor_labels": named, "shared_storage_groups": aliases}


def _spans_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def assert_zero_cross_arm_storage_overlap(
    arms: Mapping[str, Any],
    *,
    base: Any | None = None,
) -> dict[str, Any]:
    """Reject any shared underlying mutable storage across arms/base (whole untyped storage)."""
    labeled: list[tuple[str, tuple[int, int]]] = []
    for arm_name, state in arms.items():
        for span in enumerate_mutable_storage_spans(state):
            labeled.append((str(arm_name), (int(span["data_ptr"]), int(span["end_ptr"]))))
    if base is not None:
        for span in enumerate_mutable_storage_spans(base):
            labeled.append(("__base__", (int(span["data_ptr"]), int(span["end_ptr"]))))
    for i in range(len(labeled)):
        for j in range(i + 1, len(labeled)):
            a_name, a_span = labeled[i]
            b_name, b_span = labeled[j]
            if a_name == b_name:
                continue  # within-state aliases allowed
            if _spans_overlap(a_span, b_span):
                raise IntegrityProofError(
                    f"cross_arm_storage_overlap:{a_name}:{a_span}:{b_name}:{b_span}"
                )
    return {"ok": True, "span_count": len(labeled), "arms": sorted(arms)}


def reject_shallow_shared_storage_fork(original: Any, candidate: Any) -> None:
    """Fail if candidate shares any whole untyped storage with original."""
    assert_zero_cross_arm_storage_overlap({"original": original, "candidate": candidate})


def _tup(v: Any) -> Any:
    if v is None:
        return None
    if isinstance(v, (list, tuple)):
        return tuple(v)
    return v


def _bounded_accumulator_authority(acc: Any) -> dict[str, Any]:
    if acc is None:
        return {"absent": True}
    return {
        "logical_shape": _tup(_get(acc, "logical_shape", ())),
        "cold_default_value": _get(acc, "cold_default_value"),
        "hot_exact_indices": _tup(_get(acc, "hot_exact_indices", ())),
        "hot_exact_values": _tup(_get(acc, "hot_exact_values", ())),
        "cold_exception_indices": _tup(_get(acc, "cold_exception_indices", ())),
        "cold_exception_values": _tup(_get(acc, "cold_exception_values", ())),
        "candidate_name": _get(acc, "candidate_name"),
        "raw_arrays_included": bool(_get(acc, "raw_arrays_included", False)),
    }


def non_tensor_authority_manifest(state_obj: Any) -> dict[str, Any]:
    """Canonical non-tensor authority for BoundedDeltaTensorState (and mapping stand-ins)."""
    carrier = _get(state_obj, "event_coded_live_carrier")
    return {
        "state_key": _get(state_obj, "state_key"),
        "bounded_accumulator_fresh_for_exact_shadow": _get(
            state_obj, "bounded_accumulator_fresh_for_exact_shadow"
        ),
        "bounded_accumulator_rebuild_hot_exact_indices": _tup(
            _get(state_obj, "bounded_accumulator_rebuild_hot_exact_indices")
        ),
        "bounded_accumulator_rebuild_cold_default_value": _get(
            state_obj, "bounded_accumulator_rebuild_cold_default_value"
        ),
        "bounded_accumulator": _bounded_accumulator_authority(
            _get(state_obj, "bounded_accumulator")
        ),
        "event_coded_live_carrier_present": carrier is not None,
    }


def hash_arm_state_manifest(state_obj: Any) -> str:
    spans = enumerate_mutable_storage_spans(state_obj)
    parts: list[str] = [json.dumps(non_tensor_authority_manifest(state_obj), sort_keys=True)]
    tensors: list[Any] = []
    _collect_tensors(state_obj, tensors)
    for t in tensors:
        try:
            payload = t.detach().cpu().contiguous().numpy().tobytes()
            parts.append(hashlib.sha256(payload).hexdigest())
        except Exception:
            span = _tensor_storage_span(t)
            parts.append(json.dumps(span))
    parts.append(json.dumps(spans, sort_keys=True))
    return hashlib.sha256("|".join(parts).encode()).hexdigest()


def untouched_sentinel_report(
    *,
    before: Mapping[str, str],
    after: Mapping[str, str],
    required_unchanged: Sequence[str],
) -> dict[str, Any]:
    drifted = []
    for key in required_unchanged:
        if before.get(key) != after.get(key):
            drifted.append(key)
    if drifted:
        raise IntegrityProofError(f"untouched_sentinel_drift:{drifted}")
    return {"ok": True, "checked": list(required_unchanged)}


def terminal_precedence_classify(
    *,
    integrity_failure: bool,
    asymmetry_failure: bool,
    empty_applied: bool,
    science_classifier: str | None = None,
) -> str:
    """Integrity beats asymmetry; only then empty-applied; only then science."""
    if integrity_failure:
        return INTEGRITY
    if asymmetry_failure:
        return ASYMMETRY
    if empty_applied:
        return INTEGRITY
    if science_classifier in (PRESENT, NULL_OR_HARMFUL):
        return science_classifier
    if science_classifier is None:
        return INTEGRITY
    raise IntegrityProofError(f"unknown_science_classifier:{science_classifier}")


def canonical_result_forbidden(failure_class: str) -> bool:
    return failure_class in {
        "preflight_execution_receipt",
        "OOM",
        "timeout_exit_124",
        "crash_before_terminal_classifier",
        "smoke_namespace",
    }


__all__ = [
    "ASYMMETRY",
    "INTEGRITY",
    "IntegrityProofError",
    "NULL_OR_HARMFUL",
    "PRESENT",
    "assert_zero_cross_arm_storage_overlap",
    "canonical_result_forbidden",
    "enumerate_mutable_storage_spans",
    "hash_arm_state_manifest",
    "non_tensor_authority_manifest",
    "reject_shallow_shared_storage_fork",
    "terminal_precedence_classify",
    "untouched_sentinel_report",
    "within_state_alias_topology",
]
