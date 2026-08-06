"""Pure reducer: A′ slice-4 Rung-6 count-standardization (PLAN v6). No CLI/IO."""

from __future__ import annotations

import math
import re
from fractions import Fraction
from typing import Any, Mapping

from scripts.a_prime_slice4_count_standardization_schema_v0 import (
    BRANCH_BOTH_AXES_OR_INTERACTION,
    BRANCH_BOUNDARY_TIE,
    BRANCH_NEITHER_AXIS_SELECTS,
    BRANCH_RATE_PROFILE_SELECTS,
    BRANCH_STANDARDIZATION_BIND_FAIL,
    BRANCH_WEIGHT_PROFILE_SELECTS,
    CLAIM_BOUNDARY_REQUIRED,
    CLAIM_CEILING_SENTENCE_B,
    CLAIM_SILENT_ALLOWLIST,
    COMPONENTS,
    COMPONENTS_SET,
    FROZEN_COUNTS,
    KITAGAWA_LIVE_EXACT,
    LABEL_MIXED,
    LABEL_PERSISTENT,
    LABEL_TRANSIENT,
    PLAN_REVISION_BINDING,
    PRODUCT,
    PUBLISHED_D2_LABELS_RAW,
    SCHEMA_ID,
    SUCCESSOR_MAPPING,
    SUPPORTS,
    THRESHOLD_PERSISTENT,
    THRESHOLD_TRANSIENT,
)


class ComponentSetBindError(ValueError):
    """Exact {R0,R1b4v2} component-set preflight failed."""

    def __init__(self, reasons: list):
        super().__init__(";".join(reasons))
        self.reasons = list(reasons)


def normalize_published_label(raw: str) -> str:
    s = str(raw)
    return s[2:] if s.startswith("E_") else s


def label_q(q: Fraction) -> str:
    if not isinstance(q, Fraction):
        raise TypeError("label_q requires Fraction")
    if q >= THRESHOLD_PERSISTENT:
        return LABEL_PERSISTENT
    if q <= THRESHOLD_TRANSIENT:
        return LABEL_TRANSIENT
    return LABEL_MIXED


def _frac(n: int, d: int) -> Fraction:
    if d == 0:
        raise ZeroDivisionError("zero denominator")
    return Fraction(int(n), int(d))


def _as_int(x: Any, name: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int):
        raise TypeError(f"{name} must be int")
    return x


def preflight_component_sets(c1_raw: Mapping) -> list:
    """Exact component set == {R0,R1b4v2} per support before indexing."""
    reasons: list = []
    if not isinstance(c1_raw, Mapping):
        return ["c1_raw_not_mapping"]
    for s in SUPPORTS:
        if s not in c1_raw:
            reasons.append(f"missing_support:{s}")
            continue
        block = c1_raw[s]
        if not isinstance(block, Mapping):
            reasons.append(f"support_not_mapping:{s}")
            continue
        observed = frozenset(block.keys())
        if observed != COMPONENTS_SET:
            reasons.append(f"component_set_mismatch:{s}:{sorted(observed)}")
    return reasons


def _cell_counts(cell: Mapping) -> dict:
    if "N50" in cell:
        return {
            "N50": _as_int(cell["N50"], "N50"),
            "present_N20": _as_int(cell["present_N20"], "present_N20"),
            "absent_N20": _as_int(cell["absent_N20"], "absent_N20"),
        }
    return {
        "N50": _as_int(cell["|B50|_row_ids"], "N50"),
        "present_N20": _as_int(
            cell["present_at_package_N20_row_id_intersection"], "present_N20"
        ),
        "absent_N20": _as_int(
            cell["absent_from_package_N20_row_id_difference"], "absent_N20"
        ),
    }


def extract_counts_from_c1_raw(c1_raw: Mapping) -> dict:
    """Extract after component-set preflight (no silent-drop)."""
    reasons = preflight_component_sets(c1_raw)
    if reasons:
        raise ComponentSetBindError(reasons)
    return {s: {c: _cell_counts(c1_raw[s][c]) for c in COMPONENTS} for s in SUPPORTS}


def live_shaped_counts() -> dict:
    return {s: {c: dict(FROZEN_COUNTS[s][c]) for c in COMPONENTS} for s in SUPPORTS}


def live_shaped_aggregates() -> dict:
    return {s: dict(FROZEN_COUNTS[s]["aggregate"]) for s in SUPPORTS}


def live_shaped_published_d2() -> dict:
    return dict(PUBLISHED_D2_LABELS_RAW)


def weights_and_rates(counts: Mapping):
    w: dict = {}
    p: dict = {}
    for s in SUPPORTS:
        total = sum(counts[s][c]["N50"] for c in COMPONENTS)
        if total == 0:
            raise ZeroDivisionError(f"zero support total N50 for {s}")
        w[s], p[s] = {}, {}
        for c in COMPONENTS:
            n = counts[s][c]["N50"]
            if n == 0:
                raise ZeroDivisionError(f"zero N50 for {s}/{c}")
            w[s][c] = _frac(n, total)
            p[s][c] = _frac(counts[s][c]["present_N20"], n)
    return w, p


def q_cell(w: Mapping, p: Mapping, w_profile: str, r_profile: str) -> Fraction:
    return sum((w[w_profile][c] * p[r_profile][c] for c in COMPONENTS), Fraction(0))


def kitagawa_tables(
    qLL: Fraction, qLM: Fraction, qML: Fraction, qMM: Fraction
) -> dict:
    delta_q = qMM - qLL
    l0b_share, l0b_rate = qML - qLL, qMM - qML
    math_share, math_rate = qMM - qLM, qLM - qLL
    sym_share = (l0b_share + math_share) / 2
    sym_rate = (l0b_rate + math_rate) / 2
    tables = {
        "L0b": {"share_term": l0b_share, "rate_term": l0b_rate, "delta_q": delta_q},
        "math_a0": {
            "share_term": math_share,
            "rate_term": math_rate,
            "delta_q": delta_q,
        },
        "symmetric_average": {
            "share_term": sym_share,
            "rate_term": sym_rate,
            "delta_q": delta_q,
        },
    }
    for base, t in tables.items():
        if t["share_term"] + t["rate_term"] != t["delta_q"]:
            raise AssertionError(f"kitagawa identity fail for {base}")
    return {
        "tables_by_base": tables,
        "delta_q": delta_q,
        "delta_R_derived": -delta_q,
        "delta_R_terms_sign_negated": {
            "L0b": {"share_term": -l0b_share, "rate_term": -l0b_rate},
            "math_a0": {"share_term": -math_share, "rate_term": -math_rate},
            "symmetric_average": {"share_term": -sym_share, "rate_term": -sym_rate},
        },
    }


def integer_margins(aggregates: Mapping) -> dict:
    out: dict = {}
    for s in SUPPORTS:
        n = _as_int(aggregates[s]["N50"], "agg N50")
        a = _as_int(aggregates[s]["absent_N20"], "agg absent")
        out[s] = a - int(math.ceil(Fraction(7, 10) * n))
    return out


def diagnostics(counts: Mapping, w: Mapping) -> dict:
    dens = {s: {c: counts[s][c]["N50"] for c in COMPONENTS} for s in SUPPORTS}
    lattice = {
        s: {c: _frac(1, counts[s][c]["N50"]) for c in COMPONENTS} for s in SUPPORTS
    }
    return {
        "branch_authority": "NONE",
        "raw_denominators": dens,
        "lattice_steps": lattice,
        "near_equal_weights_exact": abs(w["L0b"]["R0"] - w["math_a0"]["R0"]),
        "near_equal_weights_expected": Fraction(3, 184),
    }


def _ws_norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip())


def affirmative_attribution_fires(text: str) -> bool:
    """Exact generated-string contract (BLOCK 1786008170381-b1d3a7b2).

    Silent iff empty/whitespace OR ws-normalized text is in
    schema.CLAIM_SILENT_ALLOWLIST (ceilings, PLAN approved phrasing, successors,
    branch names, IDENTITY_OK__ composites, prohibition calibrations).
    Every other non-empty value FIRES. No open-ended NL / lexical grammar.
    """
    t = text.strip()
    if not t:
        return False
    allow = {_ws_norm(x) for x in CLAIM_SILENT_ALLOWLIST}
    return _ws_norm(t) not in allow


def _preflight_counts_keys(counts: Mapping) -> list:
    reasons: list = []
    for s in SUPPORTS:
        if s not in counts:
            reasons.append(f"missing_support:{s}")
            continue
        observed = frozenset(counts[s].keys())
        if observed != COMPONENTS_SET:
            reasons.append(f"component_set_mismatch:{s}:{sorted(observed)}")
    return reasons


def _bind_fail_reasons(
    counts: Mapping,
    aggregates: Mapping,
    published_d2_raw: Mapping,
    labels: Mapping,
) -> list:
    reasons = _preflight_counts_keys(counts)
    if reasons:
        return reasons
    for s in SUPPORTS:
        total = sum_p = sum_a = 0
        for c in COMPONENTS:
            n = counts[s][c]["N50"]
            if n == 0:
                reasons.append(f"zero_denominator:{s}:{c}")
            total += n
            sum_p += counts[s][c]["present_N20"]
            sum_a += counts[s][c]["absent_N20"]
        if total == 0:
            reasons.append(f"zero_support_total:{s}")
        agg = aggregates[s]
        if sum_p != agg["present_N20"]:
            reasons.append(
                f"recomposition_present_N20:{s}:{sum_p}!={agg['present_N20']}"
            )
        if sum_a != agg["absent_N20"]:
            reasons.append(
                f"recomposition_absent_N20:{s}:{sum_a}!={agg['absent_N20']}"
            )
        if total != agg["N50"]:
            reasons.append(f"recomposition_N50:{s}:{total}!={agg['N50']}")
        pub = normalize_published_label(published_d2_raw[s])
        if labels[f"diag_{s}"] != pub:
            reasons.append(
                f"diagonal_label_mismatch:{s}:{labels[f'diag_{s}']}!={pub}"
            )
    return reasons


def _bind_fail_terminal(reasons: list) -> dict:
    return {
        "schema": SCHEMA_ID,
        "product": PRODUCT,
        "plan_revision_binding": PLAN_REVISION_BINDING,
        "primary": BRANCH_STANDARDIZATION_BIND_FAIL,
        "composite_terminal": BRANCH_STANDARDIZATION_BIND_FAIL,
        "terminal_kind": BRANCH_STANDARDIZATION_BIND_FAIL,
        "terminal_reasons": reasons,
        "claim_boundary": dict(CLAIM_BOUNDARY_REQUIRED),
        "successor": SUCCESSOR_MAPPING[BRANCH_STANDARDIZATION_BIND_FAIL],
        "branch_authority_secondary": "NONE",
    }


def _terminal_shell(
    primary: str,
    *,
    reasons: list,
    cells: dict,
    cell_labels: dict,
    claim_boundary: dict,
    composite: str | None = None,
) -> dict:
    return {
        "schema": SCHEMA_ID,
        "product": PRODUCT,
        "plan_revision_binding": PLAN_REVISION_BINDING,
        "primary": primary,
        "composite_terminal": composite if composite is not None else primary,
        "terminal_kind": primary,
        "terminal_reasons": reasons,
        "cells": {k: str(v) for k, v in cells.items()},
        "cell_labels": cell_labels,
        "claim_boundary": claim_boundary,
        "successor": SUCCESSOR_MAPPING[primary],
        "branch_authority_secondary": "NONE",
    }


def standardize_from_counts(
    counts: Mapping,
    aggregates: Mapping,
    published_d2_raw: Mapping,
    *,
    force_boundary_q: Fraction | None = None,
    force_boundary_cell: str | None = None,
) -> dict:
    """Core pure standardization (force_boundary_* for BOUNDARY_TIE only)."""
    claim_boundary = dict(CLAIM_BOUNDARY_REQUIRED)
    pre = _preflight_counts_keys(counts)
    if pre:
        return _bind_fail_terminal(pre)
    try:
        w, p = weights_and_rates(counts)
        qLL = q_cell(w, p, "L0b", "L0b")
        qLM = q_cell(w, p, "L0b", "math_a0")
        qML = q_cell(w, p, "math_a0", "L0b")
        qMM = q_cell(w, p, "math_a0", "math_a0")
    except ZeroDivisionError as e:
        return _bind_fail_terminal([f"zero_denominator:{e}"])

    cells = {"wL_rL": qLL, "wL_rM": qLM, "wM_rL": qML, "wM_rM": qMM}
    if force_boundary_q is not None and force_boundary_cell is not None:
        cells[force_boundary_cell] = force_boundary_q
    cell_labels = {k: label_q(v) for k, v in cells.items()}
    labels = {
        **cell_labels,
        "diag_L0b": cell_labels["wL_rL"],
        "diag_math_a0": cell_labels["wM_rM"],
    }
    bind_reasons = _bind_fail_reasons(counts, aggregates, published_d2_raw, labels)
    if bind_reasons:
        return _terminal_shell(
            BRANCH_STANDARDIZATION_BIND_FAIL,
            reasons=bind_reasons,
            cells=cells,
            cell_labels=cell_labels,
            claim_boundary=claim_boundary,
        )

    tie_cells = [
        k
        for k, v in cells.items()
        if v == THRESHOLD_TRANSIENT or v == THRESHOLD_PERSISTENT
    ]
    if tie_cells:
        return _terminal_shell(
            BRANCH_BOUNDARY_TIE,
            reasons=[f"boundary_tie:{k}:{cells[k]}" for k in tie_cells],
            cells=cells,
            cell_labels=cell_labels,
            claim_boundary=claim_boundary,
        )

    lab_LL, lab_LM = cell_labels["wL_rL"], cell_labels["wL_rM"]
    lab_ML, lab_MM = cell_labels["wM_rL"], cell_labels["wM_rM"]
    rate_selects = (lab_LL == lab_ML) and (lab_LM == lab_MM)
    weight_selects = (lab_LL == lab_LM) and (lab_ML == lab_MM)
    if rate_selects and not weight_selects:
        primary = BRANCH_RATE_PROFILE_SELECTS
    elif weight_selects and not rate_selects:
        primary = BRANCH_WEIGHT_PROFILE_SELECTS
    elif (not rate_selects) and (not weight_selects):
        primary = BRANCH_BOTH_AXES_OR_INTERACTION
    else:
        primary = BRANCH_NEITHER_AXIS_SELECTS

    kita = kitagawa_tables(qLL, qLM, qML, qMM)
    diag = diagnostics(counts, w)
    return {
        "schema": SCHEMA_ID,
        "product": PRODUCT,
        "plan_revision_binding": PLAN_REVISION_BINDING,
        "primary": primary,
        "composite_terminal": f"IDENTITY_OK__{primary}",
        "terminal_kind": primary,
        "terminal_reasons": [],
        "rate_selects": rate_selects,
        "weight_selects": weight_selects,
        "cells": {k: str(v) for k, v in cells.items()},
        "cells_fraction": cells,
        "cell_labels": cell_labels,
        "published_d2_normalized": {
            s: normalize_published_label(published_d2_raw[s]) for s in SUPPORTS
        },
        "kitagawa": {
            "branch_authority": "NONE",
            "tables_by_base": {
                b: {
                    "share_term": str(t["share_term"]),
                    "rate_term": str(t["rate_term"]),
                    "delta_q": str(t["delta_q"]),
                }
                for b, t in kita["tables_by_base"].items()
            },
            "tables_by_base_fraction": kita["tables_by_base"],
            "delta_q": str(kita["delta_q"]),
            "delta_R_derived": str(kita["delta_R_derived"]),
            "delta_R_terms_sign_negated": {
                b: {kk: str(vv) for kk, vv in t.items()}
                for b, t in kita["delta_R_terms_sign_negated"].items()
            },
        },
        "integer_margins": integer_margins(aggregates),
        "diagnostics": {
            "branch_authority": "NONE",
            "raw_denominators": diag["raw_denominators"],
            "lattice_steps": {
                s: {c: str(v) for c, v in row.items()}
                for s, row in diag["lattice_steps"].items()
            },
            "near_equal_weights_exact": str(diag["near_equal_weights_exact"]),
        },
        "claim_boundary": claim_boundary,
        "successor": SUCCESSOR_MAPPING[primary],
        "claim_ceiling_sentence_b": CLAIM_CEILING_SENTENCE_B,
        "branch_authority_secondary": "NONE",
    }


def standardize_live_shaped() -> dict:
    return standardize_from_counts(
        live_shaped_counts(), live_shaped_aggregates(), live_shaped_published_d2()
    )


def standardize_from_c1_raw(
    c1_raw: Mapping,
    aggregates: Mapping,
    published_d2_raw: Mapping,
) -> dict:
    """Live extraction: preflight exact set, then standardize."""
    try:
        counts = extract_counts_from_c1_raw(c1_raw)
    except ComponentSetBindError as e:
        return _bind_fail_terminal(e.reasons)
    return standardize_from_counts(counts, aggregates, published_d2_raw)


def kitagawa_live_exact_pins() -> dict:
    return KITAGAWA_LIVE_EXACT
