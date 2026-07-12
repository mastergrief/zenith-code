"""B2 Table-2 trajectory sufficiency reducer (pure post-hoc CPU core; no IO/CLI)."""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.receipt_compactness_guard import (
    find_raw_inline_index_violations,
)

# Hardcoded literals — must byte-equal census module (drift-guarded in tests only).
SCHEMA = "hrm_text_158_r7_selective_drain_eligibility_census_step_chunk/v1"
OBSERVER_INIT_SCHEMA = "hrm_text_158_r7_selective_drain_eligibility_census_observer_init/v1"
DIGEST_SCHEMA = "order_independent_v1_blake2b"
TABLE2_NOT_EVALUABLE = "NOT_EVALUABLE_EMPTY_PRE_STEP_BACKLOG"
TABLE2_OK = "OK"
CENSUS_OK = "OK"

OVERALL_SUFFICIENT = "SUFFICIENT_TO_CHARACTERIZE"
OVERALL_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
OVERALL_INVALID = "INVALID_OBSERVATION"

CLAIM_BOUNDARY = "observation/characterization only; NOT K/drain/materiality selection"
@dataclass(frozen=True, slots=True)
class IntegrityChecks:
    exact_n: bool
    unique_ordered_1_to_n: bool
    no_step0: bool
    no_init_rows: bool
    all_status_ok: bool
    blake2b: bool
    raw_arrays_false: bool
    closures: bool
    raw_index_violations_empty: bool
    all_dicts: bool

    def all_pass(self) -> bool:
        return all((self.exact_n, self.unique_ordered_1_to_n, self.no_step0, self.no_init_rows,
                    self.all_status_ok, self.blake2b, self.raw_arrays_false, self.closures,
                    self.raw_index_violations_empty, self.all_dicts))
@dataclass(frozen=True, slots=True)
class IntegrityGate:
    ordinary_schema_line_count: int
    observed_steps_sorted: tuple[int, ...]
    step0_row_count: int
    observer_init_row_count: int
    per_chunk_census_status: tuple[str, ...] | None
    integrity_checks: IntegrityChecks
    passed: bool
@dataclass(frozen=True, slots=True)
class TrajectoryRow:
    step: int
    re_candidated_fraction: float | None
    pre_step_backlog_unique_count: int
    re_candidated_current_count: int
    table2_status: str
@dataclass(frozen=True, slots=True)
class Trajectory:
    rows: tuple[TrajectoryRow, ...]
    n_null: int
    n_evaluable: int
    null_steps: tuple[int, ...]
    evaluable_steps: tuple[int, ...]
@dataclass(frozen=True, slots=True)
class Dispersion:
    min: float | None
    max: float | None
    mean: float | None
    median: float | None
    sample_std: float | None
@dataclass(frozen=True, slots=True)
class RollingWindow:
    end_step: int
    mean: float
@dataclass(frozen=True, slots=True)
class RollingMeanW8:
    available_windows: tuple[RollingWindow, ...]
    unavailable_end_steps: tuple[int, ...]
    final_four_available: tuple[RollingWindow, ...] | None
    terminal_deltas: tuple[float, ...] | None
@dataclass(frozen=True, slots=True)
class Verdicts:
    S0_numeric_domain: str
    S1_min_evaluable: str
    S2_tail_bound: str
    S3_continuity: str
    overall: str
@dataclass(frozen=True, slots=True)
class Companion:
    table1_denom_health: bool
    table3_per_k_last_or_summary: tuple[tuple[str, Any], ...]
    claim_boundary: str
@dataclass(frozen=True, slots=True)
class B2ReduceResult:
    integrity_gate: IntegrityGate
    trajectory: Trajectory
    dispersion_evaluable_only: Dispersion
    rolling_mean_W8: RollingMeanW8
    verdicts: Verdicts
    companion: Companion

    def to_json_dict(self) -> dict[str, Any]:
        return to_json_dict(self)
def to_json_dict(result: B2ReduceResult) -> dict[str, Any]:
    ig, ic = result.integrity_gate, result.integrity_gate.integrity_checks
    tr, disp, roll = result.trajectory, result.dispersion_evaluable_only, result.rolling_mean_W8
    ver, comp = result.verdicts, result.companion
    def _wins(ws):
        return [{"end_step": w.end_step, "mean": w.mean} for w in ws]
    return {
        "integrity_gate": {
            "ordinary_schema_line_count": ig.ordinary_schema_line_count,
            "observed_steps_sorted": list(ig.observed_steps_sorted),
            "step0_row_count": ig.step0_row_count,
            "observer_init_row_count": ig.observer_init_row_count,
            "per_chunk_census_status": None if ig.per_chunk_census_status is None else list(ig.per_chunk_census_status),
            "integrity_checks": {
                "exact_n": ic.exact_n, "unique_ordered_1_to_n": ic.unique_ordered_1_to_n,
                "no_step0": ic.no_step0, "no_init_rows": ic.no_init_rows,
                "all_status_OK": ic.all_status_ok, "blake2b": ic.blake2b,
                "raw_arrays_false": ic.raw_arrays_false, "closures": ic.closures,
                "raw_index_violations_empty": ic.raw_index_violations_empty, "all_dicts": ic.all_dicts,
            },
            "pass": ig.passed,
        },
        "trajectory": {
            "rows": [{"step": r.step, "re_candidated_fraction": r.re_candidated_fraction,
                      "pre_step_backlog_unique_count": r.pre_step_backlog_unique_count,
                      "re_candidated_current_count": r.re_candidated_current_count,
                      "table2_status": r.table2_status} for r in tr.rows],
            "n_null": tr.n_null, "n_evaluable": tr.n_evaluable,
            "null_steps": list(tr.null_steps), "evaluable_steps": list(tr.evaluable_steps),
        },
        "dispersion_evaluable_only": {
            "min": disp.min, "max": disp.max, "mean": disp.mean,
            "median": disp.median, "sample_std": disp.sample_std,
        },
        "rolling_mean_W8": {
            "available_windows": _wins(roll.available_windows),
            "unavailable_end_steps": list(roll.unavailable_end_steps),
            "final_four_available": None if roll.final_four_available is None else _wins(roll.final_four_available),
            "terminal_deltas": None if roll.terminal_deltas is None else list(roll.terminal_deltas),
        },
        "verdicts": {
            "S0_numeric_domain": ver.S0_numeric_domain, "S1_min_evaluable": ver.S1_min_evaluable,
            "S2_tail_bound": ver.S2_tail_bound, "S3_continuity": ver.S3_continuity, "overall": ver.overall,
        },
        "companion": {
            "table1_denom_health": comp.table1_denom_health,
            "table3_per_k_last_or_summary": dict(comp.table3_per_k_last_or_summary),
            "claim_boundary": comp.claim_boundary,
        },
    }
def _empty_dispersion() -> Dispersion:
    return Dispersion(None, None, None, None, None)
def _empty_rolling() -> RollingMeanW8:
    return RollingMeanW8((), (), None, None)
def _invalid_result(
    *,
    integrity: IntegrityGate,
    s0: str,
    s3: str,
    trajectory: Trajectory | None = None,
    companion: Companion | None = None,
) -> B2ReduceResult:
    tr = trajectory or Trajectory((), 0, 0, (), ())
    comp = companion or Companion(False, (), CLAIM_BOUNDARY)
    return B2ReduceResult(
        integrity_gate=integrity,
        trajectory=tr,
        dispersion_evaluable_only=_empty_dispersion(),
        rolling_mean_W8=_empty_rolling(),
        verdicts=Verdicts(
            S0_numeric_domain=s0,
            S1_min_evaluable="FAIL",
            S2_tail_bound="FAIL",
            S3_continuity=s3,
            overall=OVERALL_INVALID,
        ),
        companion=comp,
    )
def _is_canonical_number(value: Any) -> bool:
    return type(value) is int or type(value) is float
def _row_closures_ok(row: Mapping[str, Any]) -> bool:
    t1 = row.get("table1") or {}
    if not bool(t1.get("cap_closure_ok", False)):
        return False
    t2 = row.get("table2") or {}
    status = str(t2.get("table2_status") or "")
    frac = t2.get("re_candidated_fraction")
    if status == TABLE2_OK:
        if not bool(t2.get("materiality_closure_ok", False)):
            return False
    elif status == TABLE2_NOT_EVALUABLE:
        if frac is not None:
            return False
    t3 = row.get("table3") or {}
    per_k = t3.get("per_k") or {}
    if not isinstance(per_k, Mapping) or not per_k:
        return False
    for _k, body in per_k.items():
        if not isinstance(body, Mapping):
            return False
        if not bool(body.get("eligibility_closure_ok", False)):
            return False
    return True
def _build_integrity(rows: Sequence[Mapping[str, Any]], *, N: int) -> IntegrityGate:
    all_dicts = all(isinstance(r, Mapping) for r in rows)
    step0_row_count = 0
    observer_init_row_count = 0
    ordinary: list[Mapping[str, Any]] = []
    for r in rows:
        if not isinstance(r, Mapping):
            continue
        schema = r.get("schema_version")
        step = r.get("step")
        if step == 0 or step == 0.0:
            step0_row_count += 1
        if schema == OBSERVER_INIT_SCHEMA:
            observer_init_row_count += 1
            continue
        if schema == SCHEMA:
            ordinary.append(r)

    steps: list[int] = []
    statuses: list[str] = []
    for r in ordinary:
        try:
            steps.append(int(r.get("step")))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            steps.append(-1)
        statuses.append(str(r.get("census_status") or ""))

    expected = list(range(1, N + 1))
    exact_n = len(ordinary) == N
    unique_ordered = steps == expected
    no_step0 = step0_row_count == 0
    no_init = observer_init_row_count == 0
    all_status_ok = bool(ordinary) and all(s == CENSUS_OK for s in statuses)
    blake2b = all(r.get("digest_schema") == DIGEST_SCHEMA for r in ordinary)
    raw_false = all(r.get("raw_arrays_included") is False for r in ordinary)
    closures = all(_row_closures_ok(r) for r in ordinary) if ordinary else False
    raw_empty = all(
        len(find_raw_inline_index_violations(dict(r))) == 0 for r in ordinary
    ) if ordinary else False

    checks = IntegrityChecks(
        exact_n=exact_n,
        unique_ordered_1_to_n=unique_ordered,
        no_step0=no_step0,
        no_init_rows=no_init,
        all_status_ok=all_status_ok,
        blake2b=blake2b,
        raw_arrays_false=raw_false,
        closures=closures,
        raw_index_violations_empty=raw_empty,
        all_dicts=all_dicts and len(rows) == len(list(rows)),
    )
    # all_dicts: every element of input rows is a Mapping
    checks = IntegrityChecks(
        exact_n=checks.exact_n,
        unique_ordered_1_to_n=checks.unique_ordered_1_to_n,
        no_step0=checks.no_step0,
        no_init_rows=checks.no_init_rows,
        all_status_ok=checks.all_status_ok,
        blake2b=checks.blake2b,
        raw_arrays_false=checks.raw_arrays_false,
        closures=checks.closures,
        raw_index_violations_empty=checks.raw_index_violations_empty,
        all_dicts=all(isinstance(r, Mapping) for r in rows),
    )
    return IntegrityGate(
        ordinary_schema_line_count=len(ordinary),
        observed_steps_sorted=tuple(sorted(steps)),
        step0_row_count=step0_row_count,
        observer_init_row_count=observer_init_row_count,
        per_chunk_census_status=tuple(statuses) if ordinary else None,
        integrity_checks=checks,
        passed=checks.all_pass(),
    )
def _dispersion(values: Sequence[float]) -> Dispersion:
    n = len(values)
    if n == 0:
        return _empty_dispersion()
    if n == 1:
        sole = float(values[0])
        return Dispersion(sole, sole, sole, sole, None)
    return Dispersion(
        min(values),
        max(values),
        float(statistics.mean(values)),
        float(statistics.median(values)),
        float(statistics.stdev(values)),
    )
def _table3_summary(row: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
    t3 = row.get("table3") or {}
    per_k = t3.get("per_k") or {}
    if not isinstance(per_k, Mapping):
        return ()
    items: list[tuple[str, Any]] = []
    for k in sorted(per_k.keys(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
        body = per_k[k]
        if not isinstance(body, Mapping):
            items.append((str(k), None))
            continue
        items.append(
            (
                str(k),
                {
                    "eligible_count": body.get("eligible_count"),
                    "eligible_fraction_of_deferred": body.get(
                        "eligible_fraction_of_deferred"
                    ),
                },
            )
        )
    return tuple(items)
def reduce_b2_trajectory(
    rows: Sequence[Mapping[str, Any]],
    *,
    N: int = 32,
    W: int = 8,
    eps: float = 0.05,
    s1_min_evaluable: int = 16,
) -> B2ReduceResult:
    """Pure sufficiency reducer over ordinary census step chunks."""
    integrity = _build_integrity(rows, N=N)
    if not integrity.passed:
        s3 = "PASS" if integrity.integrity_checks.all_status_ok else "FAIL"
        return _invalid_result(integrity=integrity, s0="FAIL", s3=s3)

    # Integrity passed ⇒ ordinary rows are exactly steps 1..N in order.
    ordinary = [r for r in rows if isinstance(r, Mapping) and r.get("schema_version") == SCHEMA]
    ordinary_sorted = sorted(ordinary, key=lambda r: int(r["step"]))

    traj_rows: list[TrajectoryRow] = []
    null_steps: list[int] = []
    evaluable_steps: list[int] = []
    evaluable_fracs: dict[int, float] = {}

    for r in ordinary_sorted:
        step = int(r["step"])
        t2 = r.get("table2") or {}
        status = str(t2.get("table2_status") or "")
        frac_raw = t2.get("re_candidated_fraction")
        pre_n = int(t2.get("pre_step_backlog_unique_count") or 0)
        re_c = int(t2.get("re_candidated_current_count") or 0)

        is_null = status == TABLE2_NOT_EVALUABLE or frac_raw is None
        if is_null:
            null_steps.append(step)
            traj_rows.append(
                TrajectoryRow(step, None, pre_n, re_c, status or TABLE2_NOT_EVALUABLE)
            )
            continue

        # S0 — claimed evaluable
        if not _is_canonical_number(frac_raw):
            tr = Trajectory(
                tuple(traj_rows),
                len(null_steps),
                len(evaluable_steps),
                tuple(null_steps),
                tuple(evaluable_steps),
            )
            return _invalid_result(
                integrity=integrity,
                s0="FAIL",
                s3="PASS",
                trajectory=tr,
                companion=Companion(
                    True,
                    _table3_summary(ordinary_sorted[-1]),
                    CLAIM_BOUNDARY,
                ),
            )
        frac = float(frac_raw)
        if not math.isfinite(frac) or frac < 0.0 or frac > 1.0:
            tr = Trajectory(
                tuple(traj_rows),
                len(null_steps),
                len(evaluable_steps),
                tuple(null_steps),
                tuple(evaluable_steps),
            )
            return _invalid_result(
                integrity=integrity,
                s0="FAIL",
                s3="PASS",
                trajectory=tr,
                companion=Companion(
                    True,
                    _table3_summary(ordinary_sorted[-1]),
                    CLAIM_BOUNDARY,
                ),
            )
        evaluable_steps.append(step)
        evaluable_fracs[step] = frac
        traj_rows.append(TrajectoryRow(step, frac, pre_n, re_c, status))

    # Finish remaining steps already in traj_rows from loop — all covered.
    trajectory = Trajectory(
        tuple(traj_rows),
        len(null_steps),
        len(evaluable_steps),
        tuple(null_steps),
        tuple(evaluable_steps),
    )

    values = [evaluable_fracs[s] for s in evaluable_steps]
    dispersion = _dispersion(values)

    available: list[RollingWindow] = []
    unavailable: list[int] = []
    eval_set = set(evaluable_steps)
    for end in range(W, N + 1):
        window_steps = range(end - W + 1, end + 1)
        if all(s in eval_set for s in window_steps):
            mean = float(statistics.mean(evaluable_fracs[s] for s in window_steps))
            available.append(RollingWindow(end, mean))
        else:
            unavailable.append(end)

    if len(available) < 4:
        s2 = "INSUFFICIENT_WINDOWS"
        final_four = None
        deltas = None
        s2_pass = False
    else:
        final_four = tuple(available[-4:])
        deltas = tuple(
            abs(final_four[i].mean - final_four[i - 1].mean) for i in range(1, 4)
        )
        s2_pass = all(d <= eps for d in deltas)
        s2 = "PASS" if s2_pass else "FAIL"

    rolling = RollingMeanW8(
        tuple(available),
        tuple(unavailable),
        final_four,
        deltas,
    )

    s1_pass = trajectory.n_evaluable >= int(s1_min_evaluable)
    s1 = "PASS" if s1_pass else "FAIL"
    s3 = "PASS"
    s0 = "PASS"

    if s1_pass and s2_pass and s3 == "PASS":
        overall = OVERALL_SUFFICIENT
    else:
        overall = OVERALL_INSUFFICIENT

    denom_health = all(
        int((r.get("table1") or {}).get("authoritative_candidate_denominator") or 0) >= 1
        for r in ordinary_sorted
    )

    return B2ReduceResult(
        integrity_gate=integrity,
        trajectory=trajectory,
        dispersion_evaluable_only=dispersion,
        rolling_mean_W8=rolling,
        verdicts=Verdicts(s0, s1, s2, s3, overall),
        companion=Companion(
            denom_health,
            _table3_summary(ordinary_sorted[-1]),
            CLAIM_BOUNDARY,
        ),
    )
