"""CPU-static fixtures for B2 Table-2 trajectory sufficiency reducer."""
from __future__ import annotations

import copy
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from calm.hrm_text_158.native_full_stack.r7_b2_reducer_cli import (
    load_and_reduce,
    load_sidecar_jsonl,
    main,
)
from calm.hrm_text_158.native_full_stack.r7_b2_table2_trajectory_sufficiency_reducer import (
    CENSUS_OK,
    DIGEST_SCHEMA,
    OBSERVER_INIT_SCHEMA,
    OVERALL_INSUFFICIENT,
    OVERALL_INVALID,
    OVERALL_SUFFICIENT,
    SCHEMA,
    TABLE2_NOT_EVALUABLE,
    TABLE2_OK,
    reduce_b2_trajectory,
    to_json_dict,
)

N = 32
W = 8
REPO = Path(__file__).resolve().parents[3]
CLI_MOD = (
    REPO / "calm/hrm_text_158/native_full_stack/r7_b2_reducer_cli.py"
)


def _per_k() -> dict[str, Any]:
    return {
        str(k): {
            "eligible_count": 0,
            "eligible_fraction_of_deferred": 0.0,
            "eligibility_closure_ok": True,
        }
        for k in (2, 4, 8, 12, 16)
    }


def make_ok_row(
    step: int,
    fraction: float | None,
    *,
    table2_status: str | None = None,
    census_status: str = CENSUS_OK,
    denom: int = 131072,
    pre_n: int | None = None,
    re_c: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if table2_status is None:
        table2_status = TABLE2_NOT_EVALUABLE if fraction is None else TABLE2_OK
    if pre_n is None:
        pre_n = 0 if fraction is None else 1000
    if re_c is None:
        if fraction is None:
            re_c = 0
        else:
            re_c = int(round(float(fraction) * pre_n))
    only = pre_n - re_c if fraction is not None else 0
    row: dict[str, Any] = {
        "schema_version": SCHEMA,
        "step": step,
        "census_status": census_status,
        "digest_schema": DIGEST_SCHEMA,
        "raw_arrays_included": False,
        "table1": {
            "accepted_current_count": 512,
            "deferred_current_count": denom - 512,
            "authoritative_candidate_denominator": denom,
            "cap_closure_ok": True,
        },
        "table2": {
            "table2_status": table2_status,
            "pre_step_backlog_unique_count": pre_n,
            "re_candidated_current_count": re_c,
            "backlog_only_not_current_candidate_count": only,
            "re_candidated_fraction": fraction,
            "materiality_closure_ok": True,
        },
        "table3": {"per_k": _per_k()},
    }
    if extra:
        row.update(extra)
    return row


def make_series(
    fractions: list[float | None],
    *,
    n: int = N,
) -> list[dict[str, Any]]:
    assert len(fractions) == n
    return [make_ok_row(i + 1, fractions[i]) for i in range(n)]


def _stable_tail_fractions(
    *,
    n: int = N,
    null_prefix: int = 1,
    early: float = 0.40,
    late: float = 0.42,
) -> list[float | None]:
    """Build fractions with null prefix then stable-enough terminal tail."""
    out: list[float | None] = [None] * null_prefix
    remaining = n - null_prefix
    # ramp then flat late band so final-four W8 means have |Δ|<=0.05
    for i in range(remaining):
        # keep late values tightly clustered
        if i < remaining - 12:
            out.append(early + 0.001 * (i % 3))
        else:
            out.append(late + 0.001 * ((i % 2) * 0.5))
    assert len(out) == n
    return out


def _early_stable_late_drift_fractions(*, n: int = N) -> list[float | None]:
    """Early flat plateau, then a steep late ramp so terminal W8 means drift > eps."""
    out: list[float | None] = [None]
    for i in range(1, n):
        step = i + 1
        if step <= 20:
            out.append(0.30)
        else:
            # steps 21..32: climb so consecutive W8 means shift by >0.05
            # slide delta = (v_t - v_{t-8})/8; with per-step +0.08 => delta=0.08
            out.append(min(0.99, 0.30 + 0.08 * (step - 20)))
    return out


def test_schema_literals_match_census_module() -> None:
    from calm.hrm_text_158.native_full_stack import (
        r7_selective_drain_eligibility_census as census,
    )
    from calm.hrm_text_158.native_full_stack import (
        r7_b2_table2_trajectory_sufficiency_reducer as red,
    )

    assert red.SCHEMA == census.SCHEMA
    assert red.OBSERVER_INIT_SCHEMA == census.OBSERVER_INIT_SCHEMA
    assert red.DIGEST_SCHEMA == census.DIGEST_SCHEMA
    assert red.TABLE2_NOT_EVALUABLE == census.TABLE2_NOT_EVALUABLE
    assert red.CENSUS_OK == census.CENSUS_OK


def test_d_clean_stable_tail_sufficient() -> None:
    rows = make_series(_stable_tail_fractions())
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_SUFFICIENT
    assert result.verdicts.S0_numeric_domain == "PASS"
    assert result.verdicts.S1_min_evaluable == "PASS"
    assert result.verdicts.S2_tail_bound == "PASS"
    assert result.verdicts.S3_continuity == "PASS"
    assert result.integrity_gate.passed is True
    assert result.trajectory.n_evaluable >= 16
    assert result.rolling_mean_W8.final_four_available is not None
    assert len(result.rolling_mean_W8.final_four_available) == 4


def test_happy_path_32_clean_sufficient_or_shaped() -> None:
    test_d_clean_stable_tail_sufficient()


def test_a_early_stable_then_late_drift_insufficient() -> None:
    rows = make_series(_early_stable_late_drift_fractions())
    result = reduce_b2_trajectory(rows)
    assert result.integrity_gate.passed is True
    assert result.verdicts.S0_numeric_domain == "PASS"
    assert result.verdicts.S3_continuity == "PASS"
    assert result.verdicts.S2_tail_bound == "FAIL"
    assert result.verdicts.overall == OVERALL_INSUFFICIENT


def test_b_terminal_null_gap_insufficient() -> None:
    # Contiguous evaluable streak too short for 4 W8 windows after a mid/late null gap.
    # steps: 1 null, 2..11 evaluable (len=10 => at most 3 W8 windows), 12..32 null.
    fracs: list[float | None] = [None]
    fracs.extend([0.41] * 10)
    fracs.extend([None] * (N - 11))
    assert len(fracs) == N
    rows = make_series(fracs)
    result = reduce_b2_trajectory(rows)
    assert result.integrity_gate.passed is True
    assert result.verdicts.overall == OVERALL_INSUFFICIENT
    assert result.verdicts.S2_tail_bound == "INSUFFICIENT_WINDOWS"


@pytest.mark.parametrize(
    "bad",
    [float("nan"), float("inf"), -0.1, 1.1, True],
    ids=["nan", "inf", "neg", "gt1", "bool_true"],
)
def test_c_nan_inf_oor_invalid(bad: Any) -> None:
    fracs: list[float | None] = _stable_tail_fractions()
    # put bad value at an evaluable step (step 2)
    rows = make_series(fracs)
    rows[1]["table2"]["re_candidated_fraction"] = bad
    rows[1]["table2"]["table2_status"] = TABLE2_OK
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_INVALID
    assert result.verdicts.S0_numeric_domain == "FAIL"


def test_e_truncated_lt_N_invalid() -> None:
    rows = make_series(_stable_tail_fractions())[:16]
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_INVALID
    assert result.integrity_gate.passed is False


def test_f_missing_step_invalid() -> None:
    rows = make_series(_stable_tail_fractions())
    del rows[10]  # remove step 11
    # renumber? no — leave gap by keeping other steps; after delete we have 31 rows
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_INVALID


def test_g_duplicate_step_invalid() -> None:
    rows = make_series(_stable_tail_fractions())
    rows[5] = make_ok_row(5, 0.4)  # duplicate step 5, was step 6
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_INVALID


def test_h_out_of_order_invalid() -> None:
    rows = make_series(_stable_tail_fractions())
    rows[0], rows[1] = rows[1], rows[0]
    result = reduce_b2_trajectory(rows)
    assert result.verdicts.overall == OVERALL_INVALID


def test_i_n_evaluable_0_insufficient_null_stats() -> None:
    rows = make_series([None] * N)
    result = reduce_b2_trajectory(rows)
    assert result.integrity_gate.passed is True
    assert result.verdicts.S0_numeric_domain == "PASS"
    assert result.verdicts.S3_continuity == "PASS"
    assert result.trajectory.n_evaluable == 0
    d = result.dispersion_evaluable_only
    assert d.min is None and d.max is None and d.mean is None
    assert d.median is None and d.sample_std is None
    assert result.rolling_mean_W8.available_windows == ()
    assert result.verdicts.overall == OVERALL_INSUFFICIENT


def test_j_n_evaluable_1_insufficient_degenerate_stats() -> None:
    fracs: list[float | None] = [None] * N
    fracs[5] = 0.41  # sole evaluable at step 6
    rows = make_series(fracs)
    result = reduce_b2_trajectory(rows)
    assert result.integrity_gate.passed is True
    assert result.trajectory.n_evaluable == 1
    d = result.dispersion_evaluable_only
    assert d.min == d.max == d.mean == d.median == 0.41
    assert d.sample_std is None
    assert result.rolling_mean_W8.available_windows == ()
    assert result.verdicts.overall == OVERALL_INSUFFICIENT


def test_loader_readonly_roundtrip(tmp_path: Path) -> None:
    rows = make_series(_stable_tail_fractions())
    path = tmp_path / "census.jsonl"
    raw = "".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows)
    path.write_text(raw, encoding="utf-8")
    before = path.read_bytes()
    loaded = load_sidecar_jsonl(path)
    via_load = load_and_reduce(path)
    via_direct = reduce_b2_trajectory(rows)
    assert via_load == via_direct
    assert loaded == rows
    assert path.read_bytes() == before


def test_result_toplevel_immutable() -> None:
    result = reduce_b2_trajectory(make_series(_stable_tail_fractions()))
    with pytest.raises(Exception):
        result.verdicts = result.verdicts  # type: ignore[misc]


def test_result_nested_immutable() -> None:
    result = reduce_b2_trajectory(make_series(_stable_tail_fractions()))
    with pytest.raises(Exception):
        result.verdicts.overall = "X"  # type: ignore[misc]
    with pytest.raises(Exception):
        result.trajectory.rows.append(result.trajectory.rows[0])  # type: ignore[attr-defined]


def test_input_rows_unchanged() -> None:
    rows = make_series(_stable_tail_fractions())
    before = copy.deepcopy(rows)
    _ = reduce_b2_trajectory(rows)
    assert rows == before


def test_reduce_deterministic_equal() -> None:
    rows = make_series(_stable_tail_fractions())
    r1 = reduce_b2_trajectory(rows)
    r2 = reduce_b2_trajectory(rows)
    assert r1 == r2
    assert to_json_dict(r1) == to_json_dict(r2)


def test_result_does_not_alias_input() -> None:
    rows = make_series(_stable_tail_fractions())
    result = reduce_b2_trajectory(rows)
    # trajectory rows are new objects; mutating input must not affect result
    rows[1]["table2"]["re_candidated_fraction"] = 0.99
    assert result.trajectory.rows[1].re_candidated_fraction != 0.99


def _run_cli(args: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(CLI_MOD), *args],
        cwd=str(cwd or REPO),
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "PYTHONPATH": str(REPO)},
    )


def test_cli_exit_0_sufficient(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    rows = make_series(_stable_tail_fractions())
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    proc = _run_cli([str(path)])
    assert proc.returncode == 0
    body = json.loads(proc.stdout.strip().splitlines()[-1])
    assert body["verdicts"]["overall"] == OVERALL_SUFFICIENT


def test_cli_exit_0_insufficient(tmp_path: Path) -> None:
    path = tmp_path / "ins.jsonl"
    rows = make_series([None] * N)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    proc = _run_cli([str(path)])
    assert proc.returncode == 0
    body = json.loads(proc.stdout.strip().splitlines()[-1])
    assert body["verdicts"]["overall"] == OVERALL_INSUFFICIENT


def test_cli_exit_3_invalid_emits_body(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    rows = make_series(_stable_tail_fractions())[:10]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    proc = _run_cli([str(path)])
    assert proc.returncode == 3
    body = json.loads(proc.stdout.strip().splitlines()[-1])
    assert body["verdicts"]["overall"] == OVERALL_INVALID


def test_cli_exit_2_io_or_usage(tmp_path: Path) -> None:
    missing = tmp_path / "nope.jsonl"
    proc = _run_cli([str(missing)])
    assert proc.returncode == 2
    proc2 = _run_cli([])
    assert proc2.returncode == 2


def test_main_exit_codes_inprocess(tmp_path: Path) -> None:
    path = tmp_path / "ok.jsonl"
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in make_series(_stable_tail_fractions())),
        encoding="utf-8",
    )
    assert main([str(path)]) == 0
    assert main([]) == 2
