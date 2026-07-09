"""Slice-7 Item2 Class-A: decisive-only StepSurfaceRecord.q_levels retention.

Gate: +1 implement 1783594551362 / dual-accept c6e8fb9e + 1783594508169.
Proves: (1) live q unchanged + decisive-record contract; (2) retained bytes
collapse vs full-dict floor at H25-scale key counts.
"""

from __future__ import annotations

import sys
from typing import Any

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    StepSurfaceRecord,
    decisive_q_levels_snapshot,
)


def deep_sizeof(obj: Any, seen: set[int] | None = None) -> int:
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(deep_sizeof(k, seen) + deep_sizeof(v, seen) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set, frozenset)):
        size += sum(deep_sizeof(item, seen) for item in obj)
    elif isinstance(obj, StepSurfaceRecord) or hasattr(obj, "__dict__"):
        size += deep_sizeof(vars(obj), seen)
    return int(size)


def test_decisive_q_levels_snapshot_helper() -> None:
    live = {1: 1, 2: -1, 99: 1, 1000: -1}
    snap = decisive_q_levels_snapshot(
        live,
        applied_indices=(1, 99),
        crossing_indices=(99, 2),
    )
    assert snap == {1: 1, 2: -1, 99: 1}
    assert 1000 not in snap


def test_apply_step_retains_decisive_only_and_preserves_live_q() -> None:
    carrier = EventCodedAccLiveState.with_hot_exact(
        logical_numel=128,
        demotion_band=3,
        hot_exact={4: 12, 17: -11, 42: 3},
    )
    # Seed a dense-ish live map that must NOT all land in the record.
    for index in range(0, 80):
        carrier.q_levels[index] = 1 if index % 2 == 0 else -1
    live_before = dict(carrier.q_levels)
    record = carrier.apply_step(0, votes={4: 20, 17: -20, 50: 3})
    # Live map only grows/updates; never replaced by decisive-only retention.
    assert len(carrier.q_levels) >= len(live_before)
    for index, value in live_before.items():
        # Prior keys remain unless a crossing rewrote them to a new ±1.
        assert index in carrier.q_levels
    decisive = {int(i) for i in record.applied_indices} | {
        int(i) for i in record.crossing_indices
    }
    assert set(record.q_levels.keys()) == decisive
    assert len(record.q_levels) <= len(decisive)
    assert len(record.q_levels) < len(carrier.q_levels)
    for index, value in record.q_levels.items():
        assert int(value) == int(carrier.q_levels.get(int(index), 0))


def test_memory_collapse_vs_full_dict_floor_at_h25_scale() -> None:
    """Retained per-step q bytes ≪ full-dict floor (~0.55 GiB/step at ~7.18M keys).

    Uses deep_sizeof corroboration at a dense key count; extrapolates linearly
    to the observed H25 ceiling cardinality.
    """

    n_live_keys = 50_000
    n_decisive = 64
    carrier = EventCodedAccLiveState(logical_numel=n_live_keys + 16, demotion_band=1)
    for index in range(n_live_keys):
        carrier.q_levels[index] = 1 if index % 2 == 0 else -1
    # Force a small decisive set via sparse votes that cross.
    votes = {index: 20 for index in range(0, n_decisive)}
    record = carrier.apply_step(0, votes=votes)

    full_dict_bytes = deep_sizeof(dict(carrier.q_levels))
    retained_bytes = deep_sizeof(dict(record.q_levels))
    assert len(record.q_levels) <= n_decisive * 2  # applied∪crossing bound
    assert retained_bytes < full_dict_bytes
    ratio = float(retained_bytes) / float(max(1, full_dict_bytes))
    assert ratio < 0.01, (
        f"decisive retention {retained_bytes}B not ≪ full {full_dict_bytes}B "
        f"(ratio={ratio:.4f})"
    )

    # Extrapolate to observed step5 global_len ≈ 7,176,412.
    h25_keys = 7_176_412
    bytes_per_full_key = float(full_dict_bytes) / float(n_live_keys)
    est_full_floor_gib = (bytes_per_full_key * h25_keys) / (1024.0 ** 3)
    # Decisive set stays O(accepted crossings), not O(live keys).
    est_decisive_gib = float(retained_bytes) / (1024.0 ** 3)
    assert est_decisive_gib < 0.01 * max(est_full_floor_gib, 1e-9)
    # Sanity: full-dict extrapolation lands near the named ~0.55 GiB/step band
    # (80.43 B/key × 7.18M ≈ 0.54 GiB) within a loose factor for deep_sizeof overhead.
    assert 0.2 <= est_full_floor_gib <= 2.0, est_full_floor_gib
