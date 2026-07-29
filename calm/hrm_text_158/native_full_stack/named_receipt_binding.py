"""Pure named-receipt binding facade (DW_INJECTIVE PLAN_v7).

Emission-only helpers: canonical collision-resistant sparse event-map bindings,
serializable oracle projections, streaming decode-sha emission, and path-map
assembly for resolve → B1/B2/B3 consumers. No training-loop / GPU / filesystem.
"""
from __future__ import annotations

import hashlib
import time
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    bounded_accumulator_decoded_sha256,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents

__all__ = [
    "sparse_event_map_binding_sha256",
    "build_sparse_event_binding_by_key",
    "build_sparse_event_count_by_key",
    "logical_shape_by_key_from_q_levels",
    "emit_candidate_bounded_decode_sha256_after",
    "oracle_only_serializable_projection",
    "build_named_receipt_path_bindings",
    "require_finite_nonnegative_interval",
    "require_lowercase_sha256_hex",
    "validate_named_receipt_evidence_maps",
]


def require_lowercase_sha256_hex(value: Any, *, field: str) -> str:
    """Fail-closed: exact lowercase 64-char hex digest."""
    if not isinstance(value, str):
        raise ValueError(f"{field} must be str 64-hex, got {type(value).__name__}")
    if len(value) != 64:
        raise ValueError(f"{field} must be 64-hex, got len={len(value)}")
    for ch in value:
        if ch not in "0123456789abcdef":
            raise ValueError(f"{field} must be lowercase 64-hex, got {value!r}")
    return value


def _events_to_sorted_unique_pairs(
    events: SparseVoteEvents | Mapping[int, int],
    *,
    logical_shape: tuple[int, ...],
) -> list[tuple[int, int]]:
    if not logical_shape:
        raise ValueError("logical_shape must be non-empty")
    numel = 1
    for dim in logical_shape:
        d = int(dim)
        if d <= 0:
            raise ValueError(f"logical_shape dims must be > 0, got {logical_shape}")
        numel *= d

    if isinstance(events, SparseVoteEvents):
        n = events.event_count()
        if n == 0:
            pairs = []
        else:
            idxs = [int(x) for x in events.indices.tolist()]
            vals = [int(x) for x in events.values.tolist()]
            pairs = list(zip(idxs, vals))
    elif isinstance(events, Mapping):
        pairs = [(int(k), int(v)) for k, v in events.items()]
    else:
        raise TypeError(f"events must be SparseVoteEvents or Mapping[int,int], got {type(events)}")

    # fail-closed validation
    seen: set[int] = set()
    for index, value in pairs:
        if index in seen:
            raise ValueError(f"duplicate sparse event index {index} (fail-closed; no last-wins)")
        seen.add(index)
        if index < 0 or index >= numel:
            raise ValueError(f"sparse event index {index} out of range for numel={numel}")
        if value == 0:
            raise ValueError("sparse event values must be non-zero")
        if value < -32768 or value > 32767:
            raise ValueError("sparse event values must fit int16")

    pairs.sort(key=lambda item: item[0])
    return pairs


def sparse_event_map_binding_sha256(
    events: SparseVoteEvents | Mapping[int, int],
    *,
    logical_shape: tuple[int, ...],
) -> str:
    """Canonical collision-resistant binding under stated normalization.

    Not mathematical injectivity. Preimage: shape prefix + sorted unique i=v lines.
    """
    shape = tuple(int(d) for d in logical_shape)
    pairs = _events_to_sorted_unique_pairs(events, logical_shape=shape)
    h = hashlib.sha256()
    h.update(b"shape=")
    h.update(",".join(str(d) for d in shape).encode("utf-8"))
    h.update(b"\n")
    for index, value in pairs:
        h.update(f"{index}={value}\n".encode("utf-8"))
    return h.hexdigest()


def build_sparse_event_binding_by_key(
    sparse_events_by_key: Mapping[str, SparseVoteEvents | Mapping[int, int]],
    *,
    logical_shape_by_key: Mapping[str, tuple[int, ...]],
) -> dict[str, str]:
    if set(sparse_events_by_key) != set(logical_shape_by_key):
        raise ValueError(
            "sparse_events_by_key and logical_shape_by_key must have identical keys "
            f"(events={sorted(sparse_events_by_key)} shapes={sorted(logical_shape_by_key)})"
        )
    out: dict[str, str] = {}
    for key in sorted(sparse_events_by_key):
        out[str(key)] = sparse_event_map_binding_sha256(
            sparse_events_by_key[key],
            logical_shape=tuple(int(d) for d in logical_shape_by_key[key]),
        )
    return out


def build_sparse_event_count_by_key(
    sparse_events_by_key: Mapping[str, SparseVoteEvents | Mapping[int, int]],
) -> dict[str, int]:
    out: dict[str, int] = {}
    for key in sorted(sparse_events_by_key):
        events = sparse_events_by_key[key]
        if isinstance(events, SparseVoteEvents):
            out[str(key)] = int(events.event_count())
        else:
            out[str(key)] = sum(1 for v in events.values() if int(v) != 0)
    return out


def logical_shape_by_key_from_q_levels(
    q_levels_by_key: Mapping[str, Any],
) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    for key in sorted(q_levels_by_key):
        q = q_levels_by_key[key]
        shape = tuple(int(d) for d in q.shape)
        if not shape:
            raise ValueError(f"q_levels for {key!r} has empty shape")
        out[str(key)] = shape
    return out


def emit_candidate_bounded_decode_sha256_after(
    state: BoundedDeltaAccumulatorState,
) -> str:
    """Single S2 authority: streaming BDA helper (no dense materialization)."""
    digest = bounded_accumulator_decoded_sha256(state)
    return require_lowercase_sha256_hex(
        digest, field="bounded_accumulator_decoded_sha256"
    )


def oracle_only_serializable_projection(
    oracle_only: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    """JSON-safe oracle map fields only; None under fused ABSENCE.

    Fail-closed: events_equal_fused_vs_dense_derived must be present on the real
    source object (never derived from the per-key map).
    """
    if oracle_only is None:
        return None
    if "events_equal_by_key" not in oracle_only:
        raise ValueError("oracle_only missing events_equal_by_key")
    if "events_equal_fused_vs_dense_derived" not in oracle_only:
        raise ValueError(
            "oracle_only missing events_equal_fused_vs_dense_derived "
            "(fail-closed; do not derive from per-key map)"
        )
    raw = dict(oracle_only.get("events_equal_by_key") or {})
    events_equal_by_key = {str(k): bool(v) for k, v in sorted(raw.items())}
    fused_vs = oracle_only["events_equal_fused_vs_dense_derived"]
    if not isinstance(fused_vs, (bool, int)):
        raise ValueError(
            "events_equal_fused_vs_dense_derived must be bool-like, "
            f"got {type(fused_vs).__name__}"
        )
    return {
        "events_equal_by_key": events_equal_by_key,
        "events_equal_fused_vs_dense_derived": bool(fused_vs),
        "dense_reference_tagged": "oracle_only",
    }


def validate_named_receipt_evidence_maps(
    evidence: Mapping[str, Any],
    *,
    resolved_mode: str,
    require_oracle_only_key: bool = True,
    allow_legacy_without_named_evidence: bool = False,
) -> None:
    """Fail-closed checks for binding/count/shape/timing/oracle evidence maps.

    Used by landing-receipt validation on DESERIALIZED receipts and by B1/B2 assembly.
    Default: named evidence REQUIRED. Legacy pre-DW stubs may opt in via
    allow_legacy_without_named_evidence=True (auditable; never default on production).
    """
    binding = evidence.get("sparse_event_map_binding_sha256_by_key")
    counts = evidence.get("sparse_event_count_by_key")
    shapes = evidence.get("sparse_event_logical_shape_by_key")
    present = [
        isinstance(binding, Mapping),
        isinstance(counts, Mapping),
        isinstance(shapes, Mapping),
    ]
    if not any(present):
        if allow_legacy_without_named_evidence:
            return
        raise ValueError(
            "named-receipt evidence maps required "
            "(binding/count/shape absent; set allow_legacy_without_named_evidence=True "
            "only for audited pre-DW fixtures)"
        )
    if not all(present):
        raise ValueError(
            "named-receipt evidence requires binding/count/shape maps together "
            "(partial presence forbidden)"
        )
    if not (set(binding) == set(counts) == set(shapes)):
        raise ValueError(
            "named-receipt evidence key-set mismatch "
            f"binding={sorted(binding)} count={sorted(counts)} shape={sorted(shapes)}"
        )
    for key, digest in sorted(binding.items()):
        require_lowercase_sha256_hex(
            digest, field=f"sparse_event_map_binding_sha256_by_key[{key}]"
        )
    for key, count in sorted(counts.items()):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError(
                f"sparse_event_count_by_key[{key}] must be nonnegative int, got {count!r}"
            )
    for key, shape in sorted(shapes.items()):
        if not isinstance(shape, (list, tuple)) or not shape:
            raise ValueError(
                f"sparse_event_logical_shape_by_key[{key}] must be non-empty list/tuple"
            )
        for dim in shape:
            if not isinstance(dim, int) or isinstance(dim, bool) or dim <= 0:
                raise ValueError(
                    f"sparse_event_logical_shape_by_key[{key}] dims must be positive ints"
                )
    interval = evidence.get("s1_binding_interval_seconds_diagnostic")
    if interval is None:
        interval = evidence.get("s1_binding_interval_seconds")
    require_finite_nonnegative_interval(
        interval, field="s1_binding_interval_seconds_diagnostic"
    )
    mode = str(resolved_mode)
    has_oracle = "oracle_only" in evidence and evidence.get("oracle_only") is not None
    if mode == "oracle_on":
        if require_oracle_only_key and not has_oracle:
            raise ValueError("oracle_on requires oracle_only evidence map")
        if has_oracle:
            oo = evidence["oracle_only"]
            if not isinstance(oo, Mapping):
                raise ValueError("oracle_only must be a mapping")
            if "events_equal_by_key" not in oo:
                raise ValueError("oracle_only missing events_equal_by_key")
            if "events_equal_fused_vs_dense_derived" not in oo:
                raise ValueError("oracle_only missing events_equal_fused_vs_dense_derived")
            if oo.get("dense_reference_tagged") != "oracle_only":
                raise ValueError(
                    "oracle_only.dense_reference_tagged must be exact literal "
                    f"'oracle_only', got {oo.get('dense_reference_tagged')!r}"
                )
            eq = oo["events_equal_by_key"]
            if not isinstance(eq, Mapping):
                raise ValueError("events_equal_by_key must be a mapping")
            if set(eq) != set(binding):
                raise ValueError(
                    "oracle events_equal_by_key key-set must match binding keys"
                )
            for k, v in eq.items():
                # exact bool only — reject int/str (gate-2 D4)
                if type(v) is not bool:
                    raise ValueError(
                        f"events_equal_by_key[{k}] must be bool, got {type(v).__name__}"
                    )
            fused_vs = oo["events_equal_fused_vs_dense_derived"]
            if type(fused_vs) is not bool:
                raise ValueError(
                    "events_equal_fused_vs_dense_derived must be bool, "
                    f"got {type(fused_vs).__name__}"
                )
    else:
        if has_oracle:
            raise ValueError("fused_only ABSENCE violated: oracle_only present")


def require_finite_nonnegative_interval(value: Any, *, field: str) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be finite float, got {value!r}") from exc
    if v != v or v == float("inf") or v == float("-inf"):  # NaN/inf
        raise ValueError(f"{field} must be finite, got {v!r}")
    if v < 0.0:
        raise ValueError(f"{field} must be >= 0, got {v}")
    return v


def build_named_receipt_path_bindings(
    *,
    sparse_events_by_key: Mapping[str, SparseVoteEvents | Mapping[int, int]],
    logical_shape_by_key: Mapping[str, tuple[int, ...]],
    oracle_only: Mapping[str, Any] | None,
    resolved_mode: str,
) -> dict[str, Any]:
    """Produce path maps + diagnostic S1 interval (timed around this block).

    Returns keys for resolve path object. Interval is diagnostic-only.
    """
    t0 = time.perf_counter()
    binding = build_sparse_event_binding_by_key(
        sparse_events_by_key, logical_shape_by_key=logical_shape_by_key
    )
    counts = build_sparse_event_count_by_key(sparse_events_by_key)
    shapes = {
        str(k): [int(d) for d in logical_shape_by_key[k]]
        for k in sorted(logical_shape_by_key)
    }
    if not (set(binding) == set(counts) == set(shapes) == {str(k) for k in sparse_events_by_key}):
        raise ValueError("named receipt binding key-set invariant failed")

    mode = str(resolved_mode)
    if mode == "oracle_on":
        serializable = oracle_only_serializable_projection(oracle_only)
        if serializable is None or not serializable.get("events_equal_by_key"):
            # allow empty events_equal only if zero keys overall
            if sparse_events_by_key and (
                serializable is None or "events_equal_by_key" not in serializable
            ):
                raise ValueError("oracle_on requires serializable events_equal_by_key")
        if serializable is not None:
            eq_keys = set(serializable["events_equal_by_key"])
            if eq_keys != set(binding):
                raise ValueError(
                    "oracle events_equal_by_key key-set must match binding keys "
                    f"(eq={sorted(eq_keys)} binding={sorted(binding)})"
                )
    else:
        serializable = None

    elapsed = float(time.perf_counter() - t0)
    require_finite_nonnegative_interval(elapsed, field="s1_binding_interval_seconds")

    out: dict[str, Any] = {
        "sparse_event_map_binding_sha256_by_key": binding,
        "sparse_event_count_by_key": counts,
        "sparse_event_logical_shape_by_key": shapes,
        "s1_binding_interval_seconds": elapsed,
        "oracle_only_serializable": serializable,
    }
    return out
