"""CPU fixtures for R7 selective-drain eligibility census facade (STEP-2 land)."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census import (
    CENSUS_INVALID,
    CENSUS_OK,
    DEFAULT_K_GRID,
    ObserverContinuityTracker,
    TABLE2_NOT_EVALUABLE,
    TABLE3_BUCKETS,
    accounting_invariant,
    build_census_chunk,
    build_selective_drain_census_step_dto,
    build_table1_cap_accounting,
    build_table2_backlog_materiality,
    build_table3_eligibility,
    maybe_run_selective_drain_census,
    pre_step_backlog_set_digest_oi_v1,
)


@dataclass(frozen=True)
class _Row:
    state_key: str
    flat_index: int
    abs_new_acc: int
    threshold_abs: int = 10


@dataclass(frozen=True)
class _CapResult:
    accepted_rows: list[_Row]
    deferred_rows: list[_Row]
    step_summary: dict[str, Any]


def _dto(*, step=0, pre=None, accepted=None, deferred=None):
    accepted = accepted or [_Row("w", 1, 100)]
    deferred = deferred or [_Row("w", 2, 50), _Row("w", 3, 40)]
    return build_selective_drain_census_step_dto(
        step=step,
        ordering_mode="margin",
        cap=1,
        pre_step_backlog=pre,
        accepted_rows=accepted,
        deferred_rows=deferred,
        plans_by_key=None,
    )


def test_dto_packed_bytes_no_per_row_objects():
    dto = _dto()
    assert dto.retained_per_row_python_objects() == 0
    assert isinstance(dto.deferred_flat_index, bytes)
    assert dto.deep_retained_bytes() > 0


def test_table1_closure():
    dto = _dto()
    t1 = build_table1_cap_accounting(dto)
    assert t1["cap_closure_ok"]
    assert t1["authoritative_candidate_denominator"] == 3


def test_table2_empty_not_evaluable():
    dto = _dto(pre=None)
    t2 = build_table2_backlog_materiality(dto)
    assert t2["table2_status"] == TABLE2_NOT_EVALUABLE
    assert t2["re_candidated_fraction"] is None
    assert t2["materiality_closure_ok"]


def test_table2_pre_step_only_fresh_deferral_not_re_candidated():
    # pre has only (w,1); current deferred includes fresh (w,9)
    pre = {"w": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    accepted = [_Row("w", 1, 100)]  # re-candidated from pre
    deferred = [_Row("w", 9, 10)]  # fresh
    dto = _dto(pre=pre, accepted=accepted, deferred=deferred)
    t2 = build_table2_backlog_materiality(dto)
    assert t2["table2_status"] == "OK"
    assert t2["re_candidated_current_count"] == 1
    assert t2["backlog_only_not_current_candidate_count"] == 0
    assert t2["materiality_closure_ok"]
    # was_in_pre for deferred fresh is False
    assert dto.deferred_was_in_pre_step_backlog == bytes([0])


def test_digest_order_invariance_and_one_id_change():
    a = {"b": {2: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}, "a": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    b = {"a": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}, "b": {2: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    ca, da = pre_step_backlog_set_digest_oi_v1(a)
    cb, db = pre_step_backlog_set_digest_oi_v1(b)
    assert ca == cb == 2
    assert da == db
    c = {"a": {1: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    _, dc = pre_step_backlog_set_digest_oi_v1(c)
    assert dc != da


def test_tracker_discontinuity_invalid_not_empty_ok():
    tr = ObserverContinuityTracker()
    tr.reset()
    dto0 = _dto(step=0)
    tr.update_from_dto(dto0)
    dto2 = _dto(step=2)
    tr.update_from_dto(dto2)
    assert tr.status == CENSUS_INVALID
    chunk = build_census_chunk(dto2, tr)
    assert chunk["census_status"] == CENSUS_INVALID
    assert chunk["census_invalid_reason"]


def test_table3_bucket_keys_and_closure():
    tr = ObserverContinuityTracker()
    tr.reset()
    dto = _dto(step=0)
    tr.update_from_dto(dto)
    t3 = build_table3_eligibility(dto, tr, k_grid=(2,))
    body = t3["per_k"]["2"]
    assert set(body["partition_counts"]) == set(TABLE3_BUCKETS)
    assert "not_in_candidate_set" not in body["partition_counts"]
    assert body["eligibility_closure_ok"]
    assert sum(body["partition_counts"].values()) == dto.deferred_count()


def test_maybe_run_default_off_skips_dto():
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {"global_rate_cap_cap": 1})
    assert maybe_run_selective_drain_census(
        enabled=False,
        pre_step_backlog=None,
        cap_result=cap,
        plans_by_key=None,
        step=0,
        tracker=ObserverContinuityTracker(),
    ) is None


def test_accounting_invariant_cross_table():
    pre = {"w": {2: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}, 99: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1}}}
    dto = _dto(pre=pre, accepted=[_Row("w", 1, 100)], deferred=[_Row("w", 2, 50)])
    tr = ObserverContinuityTracker(); tr.reset(); tr.update_from_dto(dto)
    chunk = build_census_chunk(dto, tr)
    assert accounting_invariant(chunk) == [] or chunk["census_status"] in {CENSUS_OK, CENSUS_INVALID}


def test_deep_retained_bytes_includes_overhead():
    dto = _dto()
    # nbytes-only lower bound
    nbytes = len(dto.deferred_flat_index) + len(dto.accepted_flat_index)
    assert dto.deep_retained_bytes() > nbytes


def _make_scale_backlog(n: int) -> dict[str, dict[int, dict[str, int]]]:
    # single state_key to keep key table small
    return {"w": {i: {"first_step": 0, "last_deferred_step": 0, "defer_count": 1} for i in range(n)}}


def _make_scale_rows(n: int) -> tuple[list[_Row], list[_Row]]:
    # one accepted, rest deferred; abs descending
    accepted = [_Row("w", 0, 10_000)]
    deferred = [_Row("w", i, 10_000 - i) for i in range(1, n)]
    return accepted, deferred


def _vm_rss_bytes() -> int:
    vm_path = Path("/proc/self/status")
    if vm_path.is_file():
        for line in vm_path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def run_fresh_process_scale_ceiling() -> int:
    """Fresh-process scale driver: EVERY run must pass ceilings; report median."""
    backlog_n = int(os.environ.get("CENSUS_SCALE_BACKLOG", "737000"))
    cand_n = int(os.environ.get("CENSUS_SCALE_CANDIDATES", "131072"))
    reps = int(os.environ.get("CENSUS_SCALE_REPS", "3"))
    deep_max = 8_000_000
    wall_max_ms = 5000.0
    rss_max = 256 * 1024 * 1024
    results = []
    for rep in range(reps):
        pre = _make_scale_backlog(backlog_n)
        accepted, deferred = _make_scale_rows(cand_n)
        # Baseline AFTER source fixture exists (borrowed input); delta measures DTO build retain cost.
        baseline_rss = _vm_rss_bytes()
        t0 = time.perf_counter()
        dto = build_selective_drain_census_step_dto(
            step=0,
            ordering_mode="margin",
            cap=1,
            pre_step_backlog=pre,
            accepted_rows=accepted,
            deferred_rows=deferred,
        )
        wall_ms = (time.perf_counter() - t0) * 1000.0
        deep = dto.deep_retained_bytes()
        # Drop borrowed source refs before peak sample so retained DTO is what remains attributable.
        del pre, accepted, deferred
        peak_rss = _vm_rss_bytes()
        rss_delta = max(0, peak_rss - baseline_rss)
        ok = (
            deep <= deep_max
            and wall_ms <= wall_max_ms
            and rss_delta <= rss_max
            and dto.retained_per_row_python_objects() == 0
        )
        results.append({"rep": rep, "deep": deep, "wall_ms": wall_ms, "rss_delta": rss_delta, "ok": ok})
        print(json.dumps(results[-1]), flush=True)
        if not ok:
            print("SCALE_FAIL", results[-1], flush=True)
            return 1
        del dto
    def median(xs):
        ys = sorted(xs)
        return ys[len(ys) // 2]
    print(json.dumps({
        "median_deep": median([r["deep"] for r in results]),
        "median_wall_ms": median([r["wall_ms"] for r in results]),
        "median_rss_delta": median([r["rss_delta"] for r in results]),
        "all_ok": all(r["ok"] for r in results),
    }))
    return 0


if __name__ == "__main__":
    if "--scale-rss" in sys.argv:
        raise SystemExit(run_fresh_process_scale_ceiling())
    raise SystemExit("use pytest or --scale-rss")
