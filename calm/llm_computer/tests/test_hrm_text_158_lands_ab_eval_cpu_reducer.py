"""CPU reducer tests for LANDS-AB evaluation (IMPLEMENT_v5 seam split).

Totality suite imports the REAL production reducer — no duplicated test oracle.
"""
from __future__ import annotations

import ast
import hashlib
import importlib
import inspect
import itertools
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack import lands_ab_eval_branch_reducer as reducer_mod
from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
    LandsAbReducerSchemaError,
    all_true_matrix,
    derive_state,
    matrix_with,
    reduce_lands_ab_branch,
    reduce_lands_ab_branch_strict,
    select_branch,
    validate_primitive_input,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_DIVERGENT_ORACLE_LIVE,
    BRANCH_EQUIVALENT,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_SCOPE_CREEP,
    BRANCH_VACUOUS,
    CANONICAL_CELL_KEYS,
    FORBIDDEN_INPUT_KEYS,
    PRIORITY_ORDER,
)


def _base_ok(**over):
    p = {
        "scope_creep": False,
        "fixture_contract_raw_fail": False,
        "surface_pass_by_row": all_true_matrix(),
    }
    p.update(over)
    return p


# ---------------------------------------------------------------------------
# Schema hostiles — forbidden / unknown / missing keys
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fk", sorted(FORBIDDEN_INPUT_KEYS))
def test_reducer_rejects_forbidden_key_in_input(fk):
    raw = _base_ok()
    raw[fk] = False
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(raw)
    out = reduce_lands_ab_branch(raw)
    assert out["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL
    assert out["ok"] is False


def test_reducer_rejects_s1_pass_in_input():
    raw = _base_ok(s1_pass=False)
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(raw)


def test_reducer_rejects_s2_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(s2_pass=True))


def test_reducer_rejects_s3_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(s3_pass=True))


def test_reducer_rejects_s4_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(s4_pass=True))


def test_reducer_rejects_s5_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(s5_pass=True))


def test_reducer_rejects_s6_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(s6_pass=True))


def test_reducer_rejects_site_s4_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(site_s4_pass={}))


def test_reducer_rejects_gating_row_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(gating_row_pass={}))


def test_reducer_rejects_vacuous_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(vacuous=True))


def test_reducer_rejects_divergent_event_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(divergent_event=True))


def test_reducer_rejects_divergent_apply_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(divergent_apply=True))


def test_reducer_rejects_divergent_oracle_live_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(divergent_oracle_live=True))


def test_reducer_rejects_gating_rows_all_pass_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(gating_rows_all_pass=True))


def test_reducer_rejects_fixture_contract_fail_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(fixture_contract_fail=True))


def test_reducer_rejects_branch_id_in_input():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(branch_id=BRANCH_EQUIVALENT))


def test_reducer_rejects_missing_surface_cell():
    m = all_true_matrix()
    del m[CANONICAL_CELL_KEYS[0]]
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(
            {
                "scope_creep": False,
                "fixture_contract_raw_fail": False,
                "surface_pass_by_row": m,
            }
        )


def test_reducer_rejects_extra_surface_cell():
    m = all_true_matrix()
    m["EXTRA/s9"] = True
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(
            {
                "scope_creep": False,
                "fixture_contract_raw_fail": False,
                "surface_pass_by_row": m,
            }
        )


def test_reducer_rejects_unknown_field():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(not_a_field=True))


def test_reducer_rejects_non_bool_cell():
    m = all_true_matrix()
    m[CANONICAL_CELL_KEYS[0]] = 1  # type: ignore[assignment]
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(
            {
                "scope_creep": False,
                "fixture_contract_raw_fail": False,
                "surface_pass_by_row": m,
            }
        )


def test_reducer_rejects_missing_primitive():
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input({"scope_creep": False, "surface_pass_by_row": all_true_matrix()})


# ---------------------------------------------------------------------------
# Positive branch cases from primitives (REAL reducer)
# ---------------------------------------------------------------------------


def test_reducer_equivalent_requires_all_cells_true_and_no_scope_fixture():
    out = reduce_lands_ab_branch_strict(_base_ok())
    assert out["branch_id"] == BRANCH_EQUIVALENT
    d = out["derived"]
    assert d["gating_rows_all_pass"] is True
    assert all(d[f"s{i}_pass"] for i in range(1, 7))


def test_reducer_every_branch_positive_case_from_row_matrix_primitives():
    cases = {
        BRANCH_SCOPE_CREEP: _base_ok(scope_creep=True),
        BRANCH_FIXTURE_CONTRACT_FAIL: _base_ok(fixture_contract_raw_fail=True),
        BRANCH_VACUOUS: _base_ok(
            surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s4": False})
        ),
        BRANCH_DIVERGENT_EVENT: _base_ok(
            surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s1": False})
        ),
        BRANCH_DIVERGENT_APPLY: _base_ok(
            surface_pass_by_row=matrix_with(**{"G_CUDA_B1_APPLY/s3": False})
        ),
        BRANCH_DIVERGENT_ORACLE_LIVE: _base_ok(
            surface_pass_by_row=matrix_with(**{"G_CUDA_ORACLE_B1/s5": False})
        ),
        BRANCH_EQUIVALENT: _base_ok(),
    }
    for expected, raw in cases.items():
        out = reduce_lands_ab_branch_strict(raw)
        assert out["branch_id"] == expected, (expected, out)


def test_reducer_multi_hit_priority_from_row_matrix():
    # scope beats everything
    raw = _base_ok(
        scope_creep=True,
        fixture_contract_raw_fail=True,
        surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s1": False, "G_CPU_STATIC_AB/s4": False}),
    )
    assert reduce_lands_ab_branch_strict(raw)["branch_id"] == BRANCH_SCOPE_CREEP
    # fixture beats vacuous / divergent
    raw = _base_ok(
        fixture_contract_raw_fail=True,
        surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s1": False, "G_CPU_STATIC_AB/s4": False}),
    )
    assert reduce_lands_ab_branch_strict(raw)["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL
    # vacuous before divergent_event when both would fire? vacuous is s4; divergent_event is s1
    # priority: vacuous > divergent_event
    raw = _base_ok(
        surface_pass_by_row=matrix_with(
            **{"G_CPU_STATIC_AB/s1": False, "G_CPU_STATIC_AB/s4": False}
        )
    )
    assert reduce_lands_ab_branch_strict(raw)["branch_id"] == BRANCH_VACUOUS
    # divergent_event before divergent_apply
    raw = _base_ok(
        surface_pass_by_row=matrix_with(
            **{"G_CPU_STATIC_AB/s1": False, "G_CUDA_B1_APPLY/s3": False}
        )
    )
    assert reduce_lands_ab_branch_strict(raw)["branch_id"] == BRANCH_DIVERGENT_EVENT


def test_reducer_s6_only_false_yields_fixture_contract_fail_branch():
    raw = _base_ok(
        surface_pass_by_row=matrix_with(**{"G_CPU_STATIC_AB/s6": False})
    )
    out = reduce_lands_ab_branch_strict(raw)
    assert out["derived"]["s6_pass"] is False
    assert out["derived"]["fixture_contract_fail"] is True
    assert out["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL


def test_reducer_s6_false_never_reaches_equivalent():
    for cell in [k for k in CANONICAL_CELL_KEYS if k.endswith("/s6")]:
        raw = _base_ok(surface_pass_by_row=matrix_with(**{cell: False}))
        out = reduce_lands_ab_branch_strict(raw)
        assert out["branch_id"] != BRANCH_EQUIVALENT
        assert out["branch_id"] == BRANCH_FIXTURE_CONTRACT_FAIL


# ---------------------------------------------------------------------------
# Totality suite — REAL reducer, not a test-side reimplementation of branch law
# beyond calling production derive_state/select_branch
# ---------------------------------------------------------------------------


def test_reducer_derivation_map_total_pure_and_or_no_partial():
    """derive_state + select_branch are total pure functions over validated primitives."""
    # sample corners of matrix space
    samples = [all_true_matrix()]
    for k in CANONICAL_CELL_KEYS:
        samples.append(matrix_with(**{k: False}))
    # two-false combos
    keys = list(CANONICAL_CELL_KEYS)
    for a, b in ((keys[0], keys[5]), (keys[3], keys[10]), (keys[-1], keys[2])):
        samples.append(matrix_with(**{a: False, b: False}))
    for m in samples:
        for scope in (False, True):
            for fix in (False, True):
                prim = {
                    "scope_creep": scope,
                    "fixture_contract_raw_fail": fix,
                    "surface_pass_by_row": m,
                }
                d = derive_state(prim)
                branch, reasons = select_branch(d)
                assert branch in PRIORITY_ORDER
                assert isinstance(reasons, list)
                assert branch is not None


def test_reducer_derived_state_exhaustive_partition_exactly_one_branch():
    """Enumerate derived-flag space (~2^7) via synthetic derived dicts through select_branch."""
    flag_names = [
        "scope_creep",
        "fixture_contract_fail",
        "vacuous",
        "divergent_event",
        "divergent_apply",
        "divergent_oracle_live",
    ]
    # also need s1..s6 and gating_rows_all_pass for EQUIVALENT path
    seen = set()
    for bits in itertools.product([False, True], repeat=len(flag_names)):
        flags = dict(zip(flag_names, bits))
        # s_pass: if divergent_event then s1 or s2 false etc — construct consistent enough
        # for select_branch which only reads the flags + s*_pass + gating_rows_all_pass
        for s_all in (False, True):
            for g_all in (False, True):
                derived = {
                    **flags,
                    "s1_pass": s_all and not flags["divergent_event"],
                    "s2_pass": s_all and not flags["divergent_event"],
                    "s3_pass": s_all and not flags["divergent_apply"],
                    "s4_pass": s_all and not flags["vacuous"],
                    "s5_pass": s_all and not flags["divergent_oracle_live"],
                    "s6_pass": s_all and not flags["fixture_contract_fail"],
                    "gating_rows_all_pass": g_all,
                    "site_s4_pass": {},
                    "gating_row_pass": {},
                }
                branch, _ = select_branch(derived)
                assert branch in PRIORITY_ORDER
                assert branch is not None
                seen.add(branch)
    # all priority branches reachable in this synthetic space
    assert BRANCH_SCOPE_CREEP in seen
    assert BRANCH_FIXTURE_CONTRACT_FAIL in seen
    assert BRANCH_VACUOUS in seen
    assert BRANCH_DIVERGENT_EVENT in seen
    assert BRANCH_DIVERGENT_APPLY in seen
    assert BRANCH_DIVERGENT_ORACLE_LIVE in seen
    assert BRANCH_EQUIVALENT in seen


def test_reducer_derived_state_exhaustive_priority_order():
    """When multiple flags true, higher-priority branch wins (via production select_branch)."""
    # scope + all others
    d = {
        "scope_creep": True,
        "fixture_contract_fail": True,
        "vacuous": True,
        "divergent_event": True,
        "divergent_apply": True,
        "divergent_oracle_live": True,
        "s1_pass": False,
        "s2_pass": False,
        "s3_pass": False,
        "s4_pass": False,
        "s5_pass": False,
        "s6_pass": False,
        "gating_rows_all_pass": False,
        "site_s4_pass": {},
        "gating_row_pass": {},
    }
    assert select_branch(d)[0] == BRANCH_SCOPE_CREEP
    d["scope_creep"] = False
    assert select_branch(d)[0] == BRANCH_FIXTURE_CONTRACT_FAIL
    d["fixture_contract_fail"] = False
    # still vacuous
    assert select_branch(d)[0] == BRANCH_VACUOUS


def test_receipt_persists_primitives_and_all_derived_hits():
    out = reduce_lands_ab_branch_strict(_base_ok())
    assert out["primitives"] is not None
    assert set(out["primitives"]["surface_pass_by_row"]) == set(CANONICAL_CELL_KEYS)
    d = out["derived"]
    for k in (
        "s1_pass",
        "s2_pass",
        "s3_pass",
        "s4_pass",
        "s5_pass",
        "s6_pass",
        "vacuous",
        "divergent_event",
        "divergent_apply",
        "divergent_oracle_live",
        "fixture_contract_fail",
        "gating_rows_all_pass",
        "site_s4_pass",
        "gating_row_pass",
    ):
        assert k in d
    assert out["branch_id"] == BRANCH_EQUIVALENT


def test_receipt_rejects_caller_authored_branch_id():
    """Caller cannot inject branch_id into input — schema fail-closed."""
    with pytest.raises(LandsAbReducerSchemaError):
        validate_primitive_input(_base_ok(branch_id=BRANCH_EQUIVALENT))


def test_reducer_module_has_no_io_gpu_imports():
    src = Path(inspect.getfile(reducer_mod)).read_text()
    tree = ast.parse(src)
    banned = {"torch", "subprocess", "socket", "urllib", "requests", "http", "os", "pathlib"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                assert root not in banned, alias.name
        if isinstance(node, ast.ImportFrom) and node.module:
            root = node.module.split(".")[0]
            # allow calm.hrm_text_158.native_full_stack.lands_ab_eval_schema only
            assert root in {"calm", "typing", "__future__"} or root not in banned
