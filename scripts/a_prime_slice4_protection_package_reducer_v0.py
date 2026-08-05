"""A′ slice-4 Rung-2 reducer: branch classification only (cycle-5 seam).

Plan v5 + addendum. Imports schema for admission; no schema primitives redefined.
Dependency: reducer → schema; schema imports nothing local.
"""
from __future__ import annotations

import json
from typing import Any, Mapping

from scripts.a_prime_slice4_protection_package_schema_v0 import (
    BASELINE_STRICT,
    COUNT_DOMAIN_MAX,
    FINAL_STRICT_DEN,
    FROZEN_OUT_TERMINAL_SHA256,
    HORIZONS,
    PARENT_SHA_EXPECTED,
    REQUIRED_CLAIM_BOUNDARY,
    REQUIRED_OUT_AUTHORITY,
    REQUIRED_PACKAGE_BINDING,
    REQUIRED_RUN_GEOMETRY,
    START_SURVIVOR_DENOMINATORS,
    SUPPORT_ROWS_EXPECTED,
    admit_count_table,
    build_out_authority,
    check_count_domain,
    check_package_binding,
    check_parent_pins,
    check_per_step_global_horizon,
    check_run_geometry,
    extract_final_strict_count,
    extract_package_binding,
    extract_support_counts,
    is_exact_dict,
    is_exact_int,
    sha256_hex,
)

# re-export for tests/classifier stability
__all__ = [
    "HORIZONS",
    "START_SURVIVOR_DENOMINATORS",
    "SUPPORT_ROWS_EXPECTED",
    "FROZEN_OUT_TERMINAL_SHA256",
    "PARENT_SHA_EXPECTED",
    "REQUIRED_RUN_GEOMETRY",
    "REQUIRED_PACKAGE_BINDING",
    "REQUIRED_CLAIM_BOUNDARY",
    "REQUIRED_OUT_AUTHORITY",
    "COUNT_DOMAIN_MAX",
    "EFFECT_VALUES",
    "SUPPORT_VALUES",
    "OUT_TOTALS",
    "OUT_SUPPORT",
    "check_package_binding",
    "extract_final_strict_count",
    "extract_package_binding",
    "sha256_hex",
    "classify_from_counts",
    "classification_core",
    "bind_and_classify_package",
    "is_exact_bool",
    "is_exact_int",
    "is_exact_str",
    "is_exact_list",
    "is_exact_dict",
]

from scripts.a_prime_slice4_protection_package_schema_v0 import (  # noqa: E402
    is_exact_bool,
    is_exact_list,
    is_exact_str,
)

TOTAL_ROWS = 1485
GATED_TOTAL_FLOOR = 1337
OWN_LOSS_MAX = 0.20
OWN_LOSS_COCOLLAPSE = 0.90
LIFT_MIN = 500
LIFT_TOTAL_FLOOR = 582
NULL_TOL_TOTAL = 30
NULL_TOL_SUPPORT: dict[str, int] = {"L0b": 8, "math_a0": 25}
OUT_TOTALS: dict[int, int] = {10: 938, 20: 82, 50: 62}
OUT_SUPPORT: dict[int, dict[str, int]] = {
    10: {"L0b": 122, "math_a0": 816},
    20: {"L0b": 15, "math_a0": 67},
    50: {"L0b": 15, "math_a0": 47},
}
EFFECT_VALUES = ("CLEARS", "PREVENTS", "DELAYS", "NULL", "OTHER")
SUPPORT_VALUES = ("BOTH_HELD", "L0B_ONLY", "MATH_ONLY", "REDISTRIBUTED", "NO_MIX")


def own_loss(final: int, baseline: int) -> float:
    if not is_exact_int(baseline) or baseline <= 0:
        raise ValueError("baseline must be positive exact int")
    if not is_exact_int(final):
        raise ValueError("final must be exact int")
    return (baseline - final) / baseline


def package_effect_profile(counts: Mapping[int, Mapping[str, int]]) -> str:
    t = {n: counts[n]["L0b"] + counts[n]["math_a0"] for n in HORIZONS}
    ol = own_loss(counts[20]["L0b"], START_SURVIVOR_DENOMINATORS["L0b"])
    om = own_loss(counts[20]["math_a0"], START_SURVIVOR_DENOMINATORS["math_a0"])
    lift = t[20] - OUT_TOTALS[20]
    both_le = ol <= OWN_LOSS_MAX and om <= OWN_LOSS_MAX
    both_ge = ol >= OWN_LOSS_COCOLLAPSE and om >= OWN_LOSS_COCOLLAPSE
    if t[10] >= GATED_TOTAL_FLOOR and t[20] >= GATED_TOTAL_FLOOR and both_le:
        return "CLEARS"
    if both_le or t[20] >= GATED_TOTAL_FLOOR:
        return "PREVENTS"
    left = t[10] >= GATED_TOTAL_FLOOR or lift >= LIFT_MIN
    right = both_ge or t[20] < LIFT_TOTAL_FLOOR
    if left and right:
        return "DELAYS"
    total_null = all(abs(t[n] - OUT_TOTALS[n]) <= NULL_TOL_TOTAL for n in HORIZONS)
    per_ok = all(
        abs(counts[n][s] - OUT_SUPPORT[n][s]) <= NULL_TOL_SUPPORT[s]
        for n in HORIZONS
        for s in ("L0b", "math_a0")
    )
    if total_null and per_ok and both_ge:
        return "NULL"
    return "OTHER"


def support_response_profile(counts: Mapping[int, Mapping[str, int]]) -> str:
    t = {n: counts[n]["L0b"] + counts[n]["math_a0"] for n in HORIZONS}
    ol = own_loss(counts[20]["L0b"], START_SURVIVOR_DENOMINATORS["L0b"])
    om = own_loss(counts[20]["math_a0"], START_SURVIVOR_DENOMINATORS["math_a0"])
    total_null = all(abs(t[n] - OUT_TOTALS[n]) <= NULL_TOL_TOTAL for n in HORIZONS)
    per_ok = all(
        abs(counts[n][s] - OUT_SUPPORT[n][s]) <= NULL_TOL_SUPPORT[s]
        for n in HORIZONS
        for s in ("L0b", "math_a0")
    )
    if total_null and not per_ok:
        return "REDISTRIBUTED"
    if ol <= OWN_LOSS_MAX and om >= 0.50:
        return "L0B_ONLY"
    if om <= OWN_LOSS_MAX and ol >= 0.50:
        return "MATH_ONLY"
    if ol <= OWN_LOSS_MAX and om <= OWN_LOSS_MAX:
        return "BOTH_HELD"
    return "NO_MIX"


def successor_for(effect: str, support: str) -> str:
    if support in ("L0B_ONLY", "MATH_ONLY", "REDISTRIBUTED"):
        return "Rung-1 densify before mechanism (MIXED-support precedence override)"
    if effect == "NULL" and support == "NO_MIX":
        return "Rung-1 densify next"
    if effect in ("CLEARS", "PREVENTS") and support in ("BOTH_HELD", "NO_MIX"):
        return "DECOMPOSE replay-veto vs PC-veto (NOT Rung-1 by default)"
    if effect == "DELAYS" and support in ("BOTH_HELD", "NO_MIX"):
        return (
            "DECOMPOSE or tighter package; Rung-1 optional only if geometry ambiguity named"
        )
    return "classify residual; no mechanism mint"


def classify_from_counts(counts: Mapping[Any, Any]) -> dict[str, Any]:
    admitted = admit_count_table(counts)
    effect = package_effect_profile(admitted)
    support = support_response_profile(admitted)
    branch = f"{effect}__{support}"
    totals = {n: admitted[n]["L0b"] + admitted[n]["math_a0"] for n in HORIZONS}
    ol = own_loss(admitted[20]["L0b"], START_SURVIVOR_DENOMINATORS["L0b"])
    om = own_loss(admitted[20]["math_a0"], START_SURVIVOR_DENOMINATORS["math_a0"])
    lift = totals[20] - OUT_TOTALS[20]
    return {
        "branch": branch,
        "package_effect_profile": effect,
        "support_response_profile": support,
        "successor": successor_for(effect, support),
        "survivor_denominators": dict(START_SURVIVOR_DENOMINATORS),
        "support_rows_expected": dict(SUPPORT_ROWS_EXPECTED),
        "counts": {str(n): dict(admitted[n]) for n in HORIZONS},
        "totals": {str(n): totals[n] for n in HORIZONS},
        "out_totals": {str(n): OUT_TOTALS[n] for n in HORIZONS},
        "out_support_counts": {str(n): dict(OUT_SUPPORT[n]) for n in HORIZONS},
        "N20_own_loss": {"L0b": ol, "math_a0": om},
        "lift_N20": lift,
        "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
        "instrument_fail": False,
        "reasons": [],
    }


def classification_core(cls: Mapping[str, Any]) -> dict[str, Any]:
    """Full load-bearing projection — types preserved; includes instrument evidence + source_shas."""
    if not isinstance(cls, Mapping):
        raise ValueError("cls_not_mapping")
    counts_raw = cls.get("counts")
    if not isinstance(counts_raw, Mapping):
        raise ValueError("counts_missing")
    counts: dict[str, dict[str, Any]] = {}
    for n in HORIZONS:
        key = str(n)
        if key not in counts_raw:
            raise ValueError(f"counts_missing_N{n}")
        entry = counts_raw[key]
        if not isinstance(entry, Mapping):
            raise ValueError(f"counts_N{n}_not_mapping")
        counts[key] = {"L0b": entry.get("L0b"), "math_a0": entry.get("math_a0")}
    boundary = cls.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("claim_boundary_missing")
    out_auth = cls.get("out_authority")
    src = cls.get("source_shas")
    return {
        "branch": cls.get("branch"),
        "package_effect_profile": cls.get("package_effect_profile"),
        "support_response_profile": cls.get("support_response_profile"),
        "successor": cls.get("successor"),
        "survivor_denominators": dict(cls.get("survivor_denominators") or {}),
        "support_rows_expected": dict(cls.get("support_rows_expected") or {}),
        "counts": counts,
        "totals": dict(cls["totals"]) if isinstance(cls.get("totals"), Mapping) else cls.get("totals"),
        "out_totals": (
            dict(cls["out_totals"])
            if isinstance(cls.get("out_totals"), Mapping)
            else cls.get("out_totals")
        ),
        "out_support_counts": (
            dict(cls["out_support_counts"])
            if isinstance(cls.get("out_support_counts"), Mapping)
            else cls.get("out_support_counts")
        ),
        "N20_own_loss": (
            dict(cls["N20_own_loss"])
            if isinstance(cls.get("N20_own_loss"), Mapping)
            else cls.get("N20_own_loss")
        ),
        "lift_N20": cls.get("lift_N20"),
        "claim_boundary": dict(boundary),
        "package_binding": (
            dict(cls["package_binding"])
            if isinstance(cls.get("package_binding"), Mapping)
            else cls.get("package_binding")
        ),
        "out_authority": dict(out_auth) if isinstance(out_auth, Mapping) else out_auth,
        "source_shas": dict(src) if isinstance(src, Mapping) else src,
        "instrument_fail": cls.get("instrument_fail"),
        "reasons": list(cls["reasons"]) if isinstance(cls.get("reasons"), list) else cls.get("reasons"),
    }


def bind_and_classify_package(
    package_receipt_bytes_by_n: Mapping[int, bytes],
    *,
    out_terminal: Mapping[str, Any] | None = None,
    out_terminal_sha256: str | None = None,
    require_frozen_out_terminal_sha: bool = True,
) -> dict[str, Any]:
    reasons: list[str] = []
    if require_frozen_out_terminal_sha:
        if out_terminal is None:
            reasons.append("out_terminal_missing")
        if out_terminal_sha256 is None:
            reasons.append("out_terminal_sha_missing")
        elif out_terminal_sha256 != FROZEN_OUT_TERMINAL_SHA256:
            reasons.append(f"out_terminal_sha_mismatch:{out_terminal_sha256}")
        if out_terminal is not None:
            if out_terminal.get("branch") != "NONMONOTONE_OR_MULTI_CLIFF":
                reasons.append(f"out_terminal_branch={out_terminal.get('branch')!r}")
            if out_terminal.get("terminal_authority") != "manifest+marker":
                reasons.append(
                    f"out_terminal_authority={out_terminal.get('terminal_authority')!r}"
                )
            if out_terminal.get("synthetic") is not False:
                reasons.append(f"out_terminal_synthetic={out_terminal.get('synthetic')!r}")

    bound_shas: dict[str, str] = {}
    count_table: dict[int, dict[str, int]] = {}
    package_binding_report: dict[str, Any] = {}

    for n in HORIZONS:
        raw = package_receipt_bytes_by_n.get(n)
        if raw is None:
            reasons.append(f"missing_package_receipt_bytes:N{n}")
            continue
        bound_shas[f"package/N{n}"] = sha256_hex(raw)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            reasons.append(f"parse_fail:N{n}:{e}")
            continue
        if not is_exact_dict(obj):
            reasons.append(f"N{n}:receipt_not_dict")
            continue
        sc = obj.get("steps_completed")
        if not is_exact_int(sc) or sc != n:
            reasons.append(f"N{n}:steps_completed={sc!r}")
        reasons.extend(check_parent_pins(obj, n))
        reasons.extend(check_run_geometry(obj, n))
        reasons.extend(check_per_step_global_horizon(obj, n))
        binding = extract_package_binding(obj)
        if n == min(HORIZONS):
            package_binding_report = binding
        reasons.extend([f"N{n}:{r}" for r in check_package_binding(binding)])
        try:
            l0 = extract_support_counts(obj, "L0b")
            m = extract_support_counts(obj, "math_a0")
        except Exception as e:
            reasons.append(f"N{n}:count_extract:{e}")
            continue
        domain_fail = check_count_domain(l0, m, n)
        if domain_fail:
            reasons.extend(domain_fail)
            continue
        count_table[n] = {"L0b": l0, "math_a0": m}

    out_authority_report: dict[str, Any] | None = None
    if out_terminal is not None and out_terminal_sha256 is not None:
        bound_shas["out/terminal"] = out_terminal_sha256
        out_authority_report = build_out_authority(out_terminal, out_terminal_sha256)

    if reasons or set(count_table) != set(HORIZONS):
        return {
            "branch": "INSTRUMENT_OR_BIND_FAIL",
            "package_effect_profile": None,
            "support_response_profile": None,
            "successor": "instrument repair only",
            "instrument_fail": True,
            "reasons": reasons,
            "source_shas": bound_shas,
            "package_binding": package_binding_report,
            "out_authority": out_authority_report,
            "counts": {str(n): dict(count_table[n]) for n in sorted(count_table)},
            "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
        }

    result = classify_from_counts(count_table)
    result["source_shas"] = bound_shas
    result["package_binding"] = package_binding_report
    result["out_authority"] = out_authority_report
    result["instrument_fail"] = False
    result["reasons"] = []
    return result
