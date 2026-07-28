"""Pure LANDS-AB branch reducer (PLAN_v6 R4).

No IO, no GPU, no wall clock, no global mutation.
Primitives-only input → derived aggregates + exactly one terminal branch.
"""
from __future__ import annotations

from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    APPLICABILITY_MAP,
    BRANCH_DIVERGENT_APPLY,
    BRANCH_DIVERGENT_EVENT,
    BRANCH_DIVERGENT_ORACLE_LIVE,
    BRANCH_EQUIVALENT,
    BRANCH_FIXTURE_CONTRACT_FAIL,
    BRANCH_SCOPE_CREEP,
    BRANCH_VACUOUS,
    CANONICAL_CELL_KEYS,
    FORBIDDEN_INPUT_KEYS,
    GATING_ROWS,
    PRIORITY_ORDER,
    REQUIRED_TOP_LEVEL_KEYS,
    SURFACE_S4_SITES,
    cell_key,
)


class LandsAbReducerSchemaError(ValueError):
    """Schema-invalid reducer input → maps to FIXTURE-CONTRACT-FAIL class."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.branch_class = BRANCH_FIXTURE_CONTRACT_FAIL


def _require_bool(name: str, value: Any) -> bool:
    if type(value) is not bool:  # reject numpy/int bool-ish
        raise LandsAbReducerSchemaError(f"non_bool:{name}")
    return value


def validate_primitive_input(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and return a normalized primitive dict. Raises LandsAbReducerSchemaError."""
    if not isinstance(raw, Mapping):
        raise LandsAbReducerSchemaError("input_not_mapping")
    keys = set(raw.keys())
    unknown = keys - REQUIRED_TOP_LEVEL_KEYS
    if unknown:
        # any forbidden derived key or other unknown
        bad = sorted(unknown)
        raise LandsAbReducerSchemaError(f"unknown_or_forbidden_field:{bad}")
    missing = REQUIRED_TOP_LEVEL_KEYS - keys
    if missing:
        raise LandsAbReducerSchemaError(f"missing_required:{sorted(missing)}")

    # also reject if forbidden keys somehow nested — top-level only here
    for fk in FORBIDDEN_INPUT_KEYS:
        if fk in raw:
            raise LandsAbReducerSchemaError(f"forbidden_input_key:{fk}")

    scope_creep = _require_bool("scope_creep", raw["scope_creep"])
    fixture_contract_raw_fail = _require_bool(
        "fixture_contract_raw_fail", raw["fixture_contract_raw_fail"]
    )
    matrix = raw["surface_pass_by_row"]
    if not isinstance(matrix, Mapping):
        raise LandsAbReducerSchemaError("surface_pass_by_row_not_mapping")
    mkeys = set(matrix.keys())
    expected = set(CANONICAL_CELL_KEYS)
    if mkeys != expected:
        missing_c = sorted(expected - mkeys)
        extra_c = sorted(mkeys - expected)
        raise LandsAbReducerSchemaError(
            f"surface_pass_by_row_key_set_mismatch:missing={missing_c}:extra={extra_c}"
        )
    norm_matrix: dict[str, bool] = {}
    for k in CANONICAL_CELL_KEYS:
        norm_matrix[k] = _require_bool(f"surface_pass_by_row[{k}]", matrix[k])

    return {
        "scope_creep": scope_creep,
        "fixture_contract_raw_fail": fixture_contract_raw_fail,
        "surface_pass_by_row": norm_matrix,
    }


def _and_cells(matrix: Mapping[str, bool], keys: list[str]) -> bool:
    return all(matrix[k] for k in keys)


def derive_state(primitives: Mapping[str, Any]) -> dict[str, Any]:
    """Derive aggregates from validated primitives. Pure total AND/OR."""
    matrix: Mapping[str, bool] = primitives["surface_pass_by_row"]
    scope_creep = bool(primitives["scope_creep"])
    fixture_raw = bool(primitives["fixture_contract_raw_fail"])

    # surface aggregates
    s_pass: dict[str, bool] = {}
    for surf in ("s1", "s2", "s3", "s4", "s5", "s6"):
        keys = [cell_key(row, surf) for row, surfs in APPLICABILITY_MAP.items() if surf in surfs]
        s_pass[surf] = _and_cells(matrix, keys)

    site_s4 = {
        site: bool(matrix[cell_key(site, "s4")]) for site in SURFACE_S4_SITES
    }

    gating_row_pass: dict[str, bool] = {}
    for row in GATING_ROWS:
        keys = [cell_key(row, s) for s in APPLICABILITY_MAP[row]]
        gating_row_pass[row] = _and_cells(matrix, keys)

    gating_rows_all_pass = all(gating_row_pass[r] for r in GATING_ROWS)

    s1_pass = s_pass["s1"]
    s2_pass = s_pass["s2"]
    s3_pass = s_pass["s3"]
    s4_pass = s_pass["s4"]
    s5_pass = s_pass["s5"]
    s6_pass = s_pass["s6"]

    # S6 false closes no-branch hole into FIXTURE-CONTRACT-FAIL
    fixture_contract_fail = bool(fixture_raw or (not s6_pass))
    vacuous = not s4_pass
    divergent_event = (not s1_pass) or (not s2_pass)
    divergent_apply = not s3_pass
    divergent_oracle_live = not s5_pass

    return {
        "s1_pass": s1_pass,
        "s2_pass": s2_pass,
        "s3_pass": s3_pass,
        "s4_pass": s4_pass,
        "s5_pass": s5_pass,
        "s6_pass": s6_pass,
        "site_s4_pass": site_s4,
        "gating_row_pass": gating_row_pass,
        "gating_rows_all_pass": gating_rows_all_pass,
        "fixture_contract_fail": fixture_contract_fail,
        "vacuous": vacuous,
        "divergent_event": divergent_event,
        "divergent_apply": divergent_apply,
        "divergent_oracle_live": divergent_oracle_live,
        "scope_creep": scope_creep,
    }


def select_branch(derived: Mapping[str, Any]) -> tuple[str, list[str]]:
    """Priority reduce to exactly one branch. Returns (branch_id, reason_codes)."""
    reasons: list[str] = []
    if derived["scope_creep"]:
        reasons.append("scope_creep")
        return BRANCH_SCOPE_CREEP, reasons
    if derived["fixture_contract_fail"]:
        reasons.append("fixture_contract_fail")
        return BRANCH_FIXTURE_CONTRACT_FAIL, reasons
    if derived["vacuous"]:
        reasons.append("vacuous")
        return BRANCH_VACUOUS, reasons
    if derived["divergent_event"]:
        reasons.append("divergent_event")
        return BRANCH_DIVERGENT_EVENT, reasons
    if derived["divergent_apply"]:
        reasons.append("divergent_apply")
        return BRANCH_DIVERGENT_APPLY, reasons
    if derived["divergent_oracle_live"]:
        reasons.append("divergent_oracle_live")
        return BRANCH_DIVERGENT_ORACLE_LIVE, reasons
    # EQUIVALENT requires all s1..s6 and gating_rows_all_pass
    all_s = all(
        derived[k]
        for k in ("s1_pass", "s2_pass", "s3_pass", "s4_pass", "s5_pass", "s6_pass")
    )
    if all_s and derived["gating_rows_all_pass"]:
        reasons.append("all_gating_predicates_true")
        return BRANCH_EQUIVALENT, reasons
    # Unreachable if derivation total — fail-closed escape
    reasons.append("unreachable_without_equivalent_predicates")
    return BRANCH_FIXTURE_CONTRACT_FAIL, reasons


def reduce_lands_ab_branch(raw_input: Mapping[str, Any]) -> dict[str, Any]:
    """Public pure reducer entrypoint.

    Returns a receipt-shaped dict with primitives, derived, branch_id, reason_codes.
    Schema errors surface as branch_id=FIXTURE-CONTRACT-FAIL with schema_error set
    (callers may also catch LandsAbReducerSchemaError for strict hostiles).
    """
    try:
        primitives = validate_primitive_input(raw_input)
    except LandsAbReducerSchemaError as exc:
        return {
            "ok": False,
            "schema_error": str(exc),
            "primitives": None,
            "derived": None,
            "branch_id": BRANCH_FIXTURE_CONTRACT_FAIL,
            "reason_codes": ["schema_invalid", str(exc)],
        }

    derived = derive_state(primitives)
    branch_id, reasons = select_branch(derived)
    return {
        "ok": True,
        "schema_error": None,
        "primitives": primitives,
        "derived": derived,
        "branch_id": branch_id,
        "reason_codes": reasons,
        "priority_order": list(PRIORITY_ORDER),
    }


def reduce_lands_ab_branch_strict(raw_input: Mapping[str, Any]) -> dict[str, Any]:
    """Like reduce_lands_ab_branch but re-raises schema errors (for hostiles)."""
    primitives = validate_primitive_input(raw_input)
    derived = derive_state(primitives)
    branch_id, reasons = select_branch(derived)
    return {
        "ok": True,
        "schema_error": None,
        "primitives": primitives,
        "derived": derived,
        "branch_id": branch_id,
        "reason_codes": reasons,
        "priority_order": list(PRIORITY_ORDER),
    }


def all_true_matrix() -> dict[str, bool]:
    return {k: True for k in CANONICAL_CELL_KEYS}


def matrix_with(**overrides: bool) -> dict[str, bool]:
    m = all_true_matrix()
    for k, v in overrides.items():
        if k not in m:
            raise KeyError(k)
        m[k] = v
    return m
