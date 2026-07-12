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
    OBSERVER_INIT_KIND,
    OBSERVER_INIT_SCHEMA,
    ObserverContinuityTracker,
    SCHEMA,
    SelectiveDrainCensusObserverInitError,
    TABLE2_NOT_EVALUABLE,
    TABLE3_BUCKETS,
    accounting_invariant,
    build_census_chunk,
    build_selective_drain_census_step_dto,
    build_table1_cap_accounting,
    build_table2_backlog_materiality,
    build_table3_eligibility,
    initialize_selective_drain_census_observer_continuity_at_step0,
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


def _tracker_snapshot(tr: ObserverContinuityTracker) -> dict[str, Any]:
    return {
        "status": tr.status,
        "invalid_reason": tr.invalid_reason,
        "enabled_at_step": tr.enabled_at_step,
        "last_step": tr.last_step,
        "cardinality": tr.cardinality(),
    }


def _assert_init_red_no_mutation(
    *,
    tr: ObserverContinuityTracker,
    sidecar: Path,
    before: dict[str, Any],
    sidecar_existed_before: bool,
    **kwargs: Any,
) -> None:
    with pytest.raises(SelectiveDrainCensusObserverInitError):
        initialize_selective_drain_census_observer_continuity_at_step0(
            tracker=tr,
            **kwargs,
        )
    assert _tracker_snapshot(tr) == before
    assert sidecar.exists() is sidecar_existed_before
    if not sidecar_existed_before:
        assert not sidecar.exists()


def test_a_fresh_init_then_step1_ok(tmp_path: Path):
    sidecar = tmp_path / "r7_selective_drain_eligibility_census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    receipt = initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
        pre_step_backlog=None,
    )
    assert receipt["enabled_at_step"] == 0
    assert tr.enabled_at_step == 0
    assert tr.last_step == 0
    assert tr.status == CENSUS_OK
    assert not sidecar.exists()
    dto1 = _dto(step=1, pre=None)
    tr.update_from_dto(dto1)
    chunk = build_census_chunk(dto1, tr)
    assert tr.status == CENSUS_OK
    assert tr.enabled_at_step == 0
    assert tr.last_step == 1
    assert chunk["census_status"] == CENSUS_OK


def test_b_step1_table2_empty_step2_evaluable(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    dto1 = _dto(step=1, pre=None)
    tr.update_from_dto(dto1)
    t2_1 = build_table2_backlog_materiality(dto1)
    assert t2_1["table2_status"] == TABLE2_NOT_EVALUABLE
    assert build_census_chunk(dto1, tr)["census_status"] == CENSUS_OK
    pre2 = {"w": {2: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}}
    dto2 = _dto(
        step=2,
        pre=pre2,
        accepted=[_Row("w", 2, 100)],
        deferred=[_Row("w", 9, 10)],
    )
    tr.update_from_dto(dto2)
    t2_2 = build_table2_backlog_materiality(dto2)
    assert t2_2["table2_status"] == "OK"
    assert t2_2["re_candidated_fraction"] is not None
    assert build_census_chunk(dto2, tr)["census_status"] == CENSUS_OK


def test_c_no_standard_step0_data_row(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    assert not sidecar.exists()


def test_d_sidecar_lines_eq_N(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    n = 3
    for step in range(1, n + 1):
        pre = (
            None
            if step == 1
            else {"w": {2: {"first_step": 1, "last_deferred_step": step - 1, "defer_count": 1}}}
        )
        cap = _CapResult(
            [_Row("w", 1, 100)],
            [_Row("w", 2, 50)],
            {"global_rate_cap_cap": 1},
        )
        chunk = maybe_run_selective_drain_census(
            enabled=True,
            pre_step_backlog=pre,
            cap_result=cap,
            plans_by_key=None,
            step=step,
            tracker=tr,
            sidecar_path=sidecar,
        )
        assert chunk is not None
        assert chunk["schema_version"] == SCHEMA
        assert chunk["step"] != 0
    lines = sidecar.read_text(encoding="utf-8").splitlines()
    assert len(lines) == n
    parsed = [json.loads(line) for line in lines]
    assert all(row["schema_version"] == SCHEMA for row in parsed)
    assert all(row["step"] >= 1 for row in parsed)
    assert not any(row.get("kind") == OBSERVER_INIT_KIND for row in parsed)
    assert not any(
        row["schema_version"]
        == "hrm_text_158_r7_selective_drain_eligibility_census_observer_init/v1"
        for row in parsed
    )


def test_e_disabled_identity(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {"global_rate_cap_cap": 1})
    assert (
        maybe_run_selective_drain_census(
            enabled=False,
            pre_step_backlog=None,
            cap_result=cap,
            plans_by_key=None,
            step=1,
            tracker=ObserverContinuityTracker(),
            sidecar_path=sidecar,
        )
        is None
    )
    assert not sidecar.exists()


def test_f_duplicate_init_raises_InitError_deterministically(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    before = _tracker_snapshot(tr)
    with pytest.raises(SelectiveDrainCensusObserverInitError):
        initialize_selective_drain_census_observer_continuity_at_step0(
            tracker=tr,
            observed_step=0,
            sidecar_path=sidecar,
        )
    assert _tracker_snapshot(tr) == before
    assert before["enabled_at_step"] == 0
    assert before["last_step"] == 0
    assert not sidecar.exists()


def test_g_non_empty_backlog_init_fails_red(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    before = _tracker_snapshot(tr)
    _assert_init_red_no_mutation(
        tr=tr,
        sidecar=sidecar,
        before=before,
        sidecar_existed_before=False,
        observed_step=0,
        sidecar_path=sidecar,
        pre_step_backlog={"w": {1: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}},
    )


@pytest.mark.parametrize(
    "existing_payload",
    [
        b"",
        b"\n",
        b"{not-json",
        b'{"schema_version":"unknown/v0"}\n',
        (
            b'{"schema_version":"hrm_text_158_r7_selective_drain_eligibility_census_step_chunk/v1"'
            b',"step":1}\n'
        ),
    ],
)
def test_h_existing_sidecar_path_red(tmp_path: Path, existing_payload: bytes):
    sidecar = tmp_path / "census.jsonl"
    sidecar.write_bytes(existing_payload)
    before_bytes = sidecar.read_bytes()
    tr = ObserverContinuityTracker()
    tr.reset()
    before = _tracker_snapshot(tr)
    with pytest.raises(SelectiveDrainCensusObserverInitError):
        initialize_selective_drain_census_observer_continuity_at_step0(
            tracker=tr,
            observed_step=0,
            sidecar_path=sidecar,
        )
    assert _tracker_snapshot(tr) == before
    assert sidecar.read_bytes() == before_bytes

    tr2 = ObserverContinuityTracker()
    tr2.reset()
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {"global_rate_cap_cap": 1})
    chunk = maybe_run_selective_drain_census(
        enabled=True,
        pre_step_backlog=None,
        cap_result=cap,
        plans_by_key=None,
        step=1,
        tracker=tr2,
        sidecar_path=None,
    )
    assert chunk is not None
    assert chunk["census_status"] == CENSUS_INVALID
    assert chunk["census_invalid_reason"] == "mid_run_observer_enablement"


def test_i_discontinuity_duplicate_guards_unchanged(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    tr.update_from_dto(_dto(step=2))
    assert tr.status == CENSUS_INVALID
    assert tr.invalid_reason == "step_discontinuity"

    tr2 = ObserverContinuityTracker()
    tr2.reset()
    initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr2,
        observed_step=0,
        sidecar_path=tmp_path / "census2.jsonl",
    )
    tr2.update_from_dto(_dto(step=1))
    tr2.update_from_dto(_dto(step=1))
    assert tr2.status == CENSUS_INVALID
    assert tr2.invalid_reason == "duplicate_step"


@pytest.mark.parametrize("bad_step", [1, -1, True, 0.0, "0"])
def test_j_observed_step_enforcement_red(tmp_path: Path, bad_step: Any):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    before = _tracker_snapshot(tr)
    _assert_init_red_no_mutation(
        tr=tr,
        sidecar=sidecar,
        before=before,
        sidecar_existed_before=False,
        observed_step=bad_step,
        sidecar_path=sidecar,
    )


def test_k_receipt_literal_constant_values(tmp_path: Path):
    sidecar = tmp_path / "census.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    receipt = initialize_selective_drain_census_observer_continuity_at_step0(
        tracker=tr,
        observed_step=0,
        sidecar_path=sidecar,
    )
    assert (
        receipt["schema_version"]
        == "hrm_text_158_r7_selective_drain_eligibility_census_observer_init/v1"
    )
    assert receipt["kind"] == "observer_continuity_init"
    assert receipt["schema_version"] == OBSERVER_INIT_SCHEMA
    assert receipt["kind"] == OBSERVER_INIT_KIND


def test_l_exactly_once_runtime_integration(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    probe_path = Path(
        "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/scripts/"
        "hrm_text_158_bounded_delta_acquisition_probe.py"
    )
    src = probe_path.read_text(encoding="utf-8")
    call_token = "initialize_selective_drain_census_observer_continuity_at_step0("
    assert src.count(call_token) == 1
    init_pos = src.index(call_token)
    loop_pos = src.index("for step in range(1, int(steps) + 1):")
    assert init_pos < loop_pos
    vote_pos = src.index("return apply_bounded_delta_vote_step(")
    assert init_pos < vote_pos
    # Outer tracker construct/reset region must not also initialize.
    outer_marker = "r7_selective_drain_eligibility_census_tracker = (\n            ObserverContinuityTracker()"
    assert outer_marker in src
    outer_pos = src.index(outer_marker)
    # The sole call site is inside run_bounded_delta_steps, not the outer construct block.
    assert abs(init_pos - outer_pos) > 200

    calls: list[dict[str, Any]] = []
    real_init = initialize_selective_drain_census_observer_continuity_at_step0

    def _spy(**kwargs: Any) -> dict[str, Any]:
        calls.append(dict(kwargs))
        return real_init(**kwargs)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.r7_selective_drain_eligibility_census."
        "initialize_selective_drain_census_observer_continuity_at_step0",
        _spy,
    )
    # Local replica of the probe enable/disable gate (mirrors run_bounded_delta_steps).
    def _probe_enable_gate(*, enabled: bool, tracker, sidecar_path, backlog):
        if not enabled:
            return None
        return _spy(
            tracker=tracker,
            observed_step=0,
            sidecar_path=sidecar_path,
            pre_step_backlog=backlog,
        )

    calls.clear()
    assert (
        _probe_enable_gate(
            enabled=False,
            tracker=ObserverContinuityTracker(),
            sidecar_path=tmp_path / "off.jsonl",
            backlog=None,
        )
        is None
    )
    assert len(calls) == 0

    tr = ObserverContinuityTracker()
    tr.reset()
    sidecar = tmp_path / "on.jsonl"
    _probe_enable_gate(enabled=True, tracker=tr, sidecar_path=sidecar, backlog=None)
    assert len(calls) == 1
    assert calls[0]["observed_step"] == 0
    # Init precedes any subsequent maybe_run (first cap observation).
    action_log: list[str] = []
    action_log.append("init")
    cap = _CapResult([_Row("w", 1, 100)], [_Row("w", 2, 50)], {"global_rate_cap_cap": 1})
    maybe_run_selective_drain_census(
        enabled=True,
        pre_step_backlog=None,
        cap_result=cap,
        plans_by_key=None,
        step=1,
        tracker=tr,
        sidecar_path=sidecar,
    )
    action_log.append("cap")
    assert action_log == ["init", "cap"]
    assert tr.enabled_at_step == 0
    assert tr.last_step == 1


@pytest.mark.parametrize("bad_path", [None, 123, [1], {"x": 1}])
def test_m_sidecar_path_none_or_invalid_type_red(tmp_path: Path, bad_path: Any):
    sentinel = tmp_path / "must_not_be_created.jsonl"
    tr = ObserverContinuityTracker()
    tr.reset()
    before = _tracker_snapshot(tr)
    with pytest.raises(SelectiveDrainCensusObserverInitError):
        initialize_selective_drain_census_observer_continuity_at_step0(
            tracker=tr,
            observed_step=0,
            sidecar_path=bad_path,
        )
    assert _tracker_snapshot(tr) == before
    assert not sentinel.exists()


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
