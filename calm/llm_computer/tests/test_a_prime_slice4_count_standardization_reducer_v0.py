"""STEP-1 tests for Rung-6 count-standardization reducer (PLAN v6)."""

from __future__ import annotations

import ast
import inspect
from fractions import Fraction
from pathlib import Path

import scripts.a_prime_slice4_count_standardization_reducer_v0 as reducer
import scripts.a_prime_slice4_count_standardization_schema_v0 as schema

PLAN_BINDING = (
    "PLAN_v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900"
)
REPO = Path(__file__).resolve().parents[3]


def _counts():
    return reducer.live_shaped_counts()


def _aggs():
    return reducer.live_shaped_aggregates()


def _pub():
    return reducer.live_shaped_published_d2()


def test_plan_revision_binding_full_sha_equality():
    assert schema.PLAN_REVISION_BINDING == PLAN_BINDING
    out = reducer.standardize_live_shaped()
    assert out["plan_revision_binding"] == PLAN_BINDING
    assert out["plan_revision_binding"] == schema.PLAN_REVISION_BINDING


def test_live_shaped_four_cells_exact():
    out = reducer.standardize_live_shaped()
    fr = out["cells_fraction"]
    assert fr["wL_rL"] == Fraction(3, 8)
    assert fr["wL_rM"] == Fraction(85, 336)
    assert fr["wM_rL"] == Fraction(44, 115)
    assert fr["wM_rM"] == Fraction(6, 23)


def test_live_shaped_rate_profile_selects():
    out = reducer.standardize_live_shaped()
    assert out["rate_selects"] is True
    assert out["weight_selects"] is False
    assert out["primary"] == schema.BRANCH_RATE_PROFILE_SELECTS
    assert out["composite_terminal"] == (
        f"IDENTITY_OK__{schema.BRANCH_RATE_PROFILE_SELECTS}"
    )


def test_diagonal_labels_reproduce_published():
    out = reducer.standardize_live_shaped()
    assert out["cell_labels"]["wL_rL"] == "MIXED"
    assert out["cell_labels"]["wM_rM"] == "TRANSIENT"
    assert out["published_d2_normalized"]["L0b"] == "MIXED"
    assert out["published_d2_normalized"]["math_a0"] == "TRANSIENT"


def test_bind_fail_component_set_mismatch():
    counts = _counts()
    counts["L0b"]["R1_EXTRA"] = {"N50": 1, "present_N20": 0, "absent_N20": 1}
    # also need to remove proper set — add extra key so set != {R0,R1b4v2}
    out = reducer.standardize_from_counts(counts, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("component_set_mismatch" in r for r in out["terminal_reasons"])


def test_bind_fail_zero_denominator():
    counts = _counts()
    counts["L0b"]["R0"]["N50"] = 0
    out = reducer.standardize_from_counts(counts, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("zero_denominator" in r for r in out["terminal_reasons"])


def test_bind_fail_recomposition_present():
    counts = _counts()
    aggs = _aggs()
    aggs["L0b"]["present_N20"] = 99
    out = reducer.standardize_from_counts(counts, aggs, _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("recomposition_present_N20" in r for r in out["terminal_reasons"])


def test_bind_fail_recomposition_absent():
    counts = _counts()
    aggs = _aggs()
    aggs["math_a0"]["absent_N20"] = 0
    out = reducer.standardize_from_counts(counts, aggs, _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("recomposition_absent_N20" in r for r in out["terminal_reasons"])


def test_bind_fail_recomposition_n50():
    counts = _counts()
    aggs = _aggs()
    aggs["L0b"]["N50"] = 99
    out = reducer.standardize_from_counts(counts, aggs, _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("recomposition_N50" in r for r in out["terminal_reasons"])


def test_bind_fail_diagonal_label_mismatch():
    pub = _pub()
    pub["L0b"] = "E_TRANSIENT"  # published says TRANSIENT but diag is MIXED
    out = reducer.standardize_from_counts(_counts(), _aggs(), pub)
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("diagonal_label_mismatch" in r for r in out["terminal_reasons"])


def test_boundary_tie_3_10():
    out = reducer.standardize_from_counts(
        _counts(),
        _aggs(),
        _pub(),
        force_boundary_q=Fraction(3, 10),
        force_boundary_cell="wL_rM",
    )
    assert out["primary"] == schema.BRANCH_BOUNDARY_TIE
    assert any("boundary_tie:wL_rM" in r for r in out["terminal_reasons"])


def test_boundary_tie_7_10():
    out = reducer.standardize_from_counts(
        _counts(),
        _aggs(),
        _pub(),
        force_boundary_q=Fraction(7, 10),
        force_boundary_cell="wM_rL",
    )
    assert out["primary"] == schema.BRANCH_BOUNDARY_TIE


def test_weight_selects_only_fixture():
    # Same component rates on both supports (p_L0b,c == p_math,c), different
    # weights → q depends only on w → weight swap changes label, rate swap
    # preserves → WEIGHT_PROFILE_SELECTS.
    # p: R0=0.9 PERSISTENT-ish component, R1=0.0 TRANSIENT component.
    # wL heavy on R0 → q≈0.9 PERSISTENT; wM heavy on R1 → q≈0.1 TRANSIENT.
    counts = {
        "L0b": {
            "R0": {"N50": 9, "present_N20": 9, "absent_N20": 0},  # p=1
            "R1b4v2": {"N50": 1, "present_N20": 0, "absent_N20": 1},  # p=0
        },
        "math_a0": {
            "R0": {"N50": 1, "present_N20": 1, "absent_N20": 0},  # p=1 (same rates)
            "R1b4v2": {"N50": 9, "present_N20": 0, "absent_N20": 9},  # p=0
        },
    }
    # qLL=qLM=0.9*1+0.1*0=0.9 PERSISTENT; qML=qMM=0.1*1+0.9*0=0.1 TRANSIENT
    aggs = {
        "L0b": {"N50": 10, "present_N20": 9, "absent_N20": 1},
        "math_a0": {"N50": 10, "present_N20": 1, "absent_N20": 9},
    }
    pub = {"L0b": "E_PERSISTENT", "math_a0": "E_TRANSIENT"}
    out = reducer.standardize_from_counts(counts, aggs, pub)
    assert out["weight_selects"] is True
    assert out["rate_selects"] is False
    assert out["primary"] == schema.BRANCH_WEIGHT_PROFILE_SELECTS


def test_neither_all_four_labels_equal():
    # All p identical → all q equal → rate and weight both select → NEITHER
    counts = {
        "L0b": {
            "R0": {"N50": 4, "present_N20": 2, "absent_N20": 2},
            "R1b4v2": {"N50": 4, "present_N20": 2, "absent_N20": 2},
        },
        "math_a0": {
            "R0": {"N50": 6, "present_N20": 3, "absent_N20": 3},
            "R1b4v2": {"N50": 6, "present_N20": 3, "absent_N20": 3},
        },
    }
    aggs = {
        "L0b": {"N50": 8, "present_N20": 4, "absent_N20": 4},
        "math_a0": {"N50": 12, "present_N20": 6, "absent_N20": 6},
    }
    pub = {"L0b": "E_MIXED", "math_a0": "E_MIXED"}
    out = reducer.standardize_from_counts(counts, aggs, pub)
    assert out["rate_selects"] is True
    assert out["weight_selects"] is True
    assert out["primary"] == schema.BRANCH_NEITHER_AXIS_SELECTS


def test_both_axes_interaction_checkerboard():
    # Checkerboard labels: need NOT rate_selects and NOT weight_selects.
    # Live-shaped is rate_selects. Construct opposite pattern:
    # lab_LL=MIXED, lab_LM=TRANSIENT, lab_ML=TRANSIENT, lab_MM=MIXED
    # → rate: LL!=ML and LM!=MM → not rate; weight: LL!=LM and ML!=MM → not weight.
    # Achieve via asymmetric rates and weights carefully.
    # Simpler: reuse force is not available for labels; craft counts.
    # p_L0b: R0=0.5, R1=0.5 → q any w with r=L0b = 0.5 MIXED
    # That forces lab_LL=lab_ML=MIXED → rate_selects partial.
    # Try:
    # L0b rates: R0 high present, R1 low; math rates inverse; weights differ.
    counts = {
        "L0b": {
            "R0": {"N50": 10, "present_N20": 8, "absent_N20": 2},  # p=0.8 PERSISTENT
            "R1b4v2": {"N50": 10, "present_N20": 1, "absent_N20": 9},  # p=0.1 TRANSIENT
        },
        "math_a0": {
            "R0": {"N50": 10, "present_N20": 1, "absent_N20": 9},  # p=0.1 TRANSIENT
            "R1b4v2": {"N50": 10, "present_N20": 8, "absent_N20": 2},  # p=0.8 PERSISTENT
        },
    }
    # w L0b = 0.5/0.5; w math = 0.5/0.5 — same weights then q only depends on r
    # → weight_selects true. Need different weights.
    counts = {
        "L0b": {
            "R0": {"N50": 9, "present_N20": 9, "absent_N20": 0},  # p=1 PERSISTENT
            "R1b4v2": {"N50": 1, "present_N20": 0, "absent_N20": 1},  # p=0 TRANSIENT
        },
        "math_a0": {
            "R0": {"N50": 1, "present_N20": 0, "absent_N20": 1},  # p=0 TRANSIENT
            "R1b4v2": {"N50": 9, "present_N20": 9, "absent_N20": 0},  # p=1 PERSISTENT
        },
    }
    # wL: R0=0.9, R1=0.1; wM: R0=0.1, R1=0.9
    # qLL = 0.9*1 + 0.1*0 = 0.9 PERSISTENT
    # qLM = 0.9*0 + 0.1*1 = 0.1 TRANSIENT
    # qML = 0.1*1 + 0.9*0 = 0.1 TRANSIENT
    # qMM = 0.1*0 + 0.9*1 = 0.9 PERSISTENT
    # rate_selects: LL==ML? P!=T no; weight: LL==LM? P!=T no → BOTH interaction
    aggs = {
        "L0b": {"N50": 10, "present_N20": 9, "absent_N20": 1},
        "math_a0": {"N50": 10, "present_N20": 9, "absent_N20": 1},
    }
    pub = {"L0b": "E_PERSISTENT", "math_a0": "E_PERSISTENT"}
    out = reducer.standardize_from_counts(counts, aggs, pub)
    assert out["rate_selects"] is False
    assert out["weight_selects"] is False
    assert out["primary"] == schema.BRANCH_BOTH_AXES_OR_INTERACTION


def test_kitagawa_l0b_exact():
    out = reducer.standardize_live_shaped()
    t = out["kitagawa"]["tables_by_base_fraction"]["L0b"]
    assert t["share_term"] == Fraction(7, 920)
    assert t["rate_term"] == Fraction(-14, 115)
    assert t["share_term"] + t["rate_term"] == Fraction(-21, 184)


def test_kitagawa_math_exact():
    out = reducer.standardize_live_shaped()
    t = out["kitagawa"]["tables_by_base_fraction"]["math_a0"]
    assert t["share_term"] == Fraction(61, 7728)
    assert t["rate_term"] == Fraction(-41, 336)
    assert t["share_term"] + t["rate_term"] == Fraction(-21, 184)


def test_kitagawa_symmetric_exact():
    out = reducer.standardize_live_shaped()
    t = out["kitagawa"]["tables_by_base_fraction"]["symmetric_average"]
    assert t["share_term"] == Fraction(599, 77280)
    assert t["rate_term"] == Fraction(-9419, 77280)
    assert t["share_term"] + t["rate_term"] == Fraction(-21, 184)
    assert out["kitagawa"]["delta_q"] == str(Fraction(-21, 184))


def test_delta_r_derived_negation():
    out = reducer.standardize_live_shaped()
    assert out["kitagawa"]["delta_R_derived"] == str(Fraction(21, 184))
    # sign-negated terms
    for base in ("L0b", "math_a0", "symmetric_average"):
        pos = out["kitagawa"]["tables_by_base_fraction"][base]
        neg = out["kitagawa"]["delta_R_terms_sign_negated"][base]
        assert Fraction(neg["share_term"]) == -pos["share_term"]
        assert Fraction(neg["rate_term"]) == -pos["rate_term"]


def test_integer_margins():
    out = reducer.standardize_live_shaped()
    assert out["integer_margins"]["L0b"] == -1
    assert out["integer_margins"]["math_a0"] == 0


def test_n10_fields_do_not_alter_branch():
    counts = _counts()
    # inject N10-like noise keys that extractor ignores when using short keys
    for s in counts:
        for c in counts[s]:
            counts[s][c]["present_N10"] = 99
            counts[s][c]["N10"] = 99
    out = reducer.standardize_from_counts(counts, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_RATE_PROFILE_SELECTS


def test_near_equal_weights_and_lattice():
    out = reducer.standardize_live_shaped()
    assert out["diagnostics"]["near_equal_weights_exact"] == str(Fraction(3, 184))
    assert out["diagnostics"]["lattice_steps"]["L0b"]["R0"] == str(Fraction(1, 5))
    assert out["diagnostics"]["lattice_steps"]["L0b"]["R1b4v2"] == str(Fraction(1, 3))
    assert out["diagnostics"]["lattice_steps"]["math_a0"]["R0"] == str(Fraction(1, 14))
    assert out["diagnostics"]["lattice_steps"]["math_a0"]["R1b4v2"] == str(
        Fraction(1, 9)
    )


def test_no_forbidden_imports_in_reducer_source():
    src = Path(inspect.getsourcefile(reducer)).read_text()
    assert "shared_component" not in src
    assert "support_split_residual_densify" not in src
    assert "residual_classification" not in src


def test_schema_exports_thresholds_and_branches():
    assert schema.THRESHOLD_TRANSIENT == Fraction(3, 10)
    assert schema.THRESHOLD_PERSISTENT == Fraction(7, 10)
    assert schema.BRANCH_RATE_PROFILE_SELECTS in schema.FIRST_MATCH_ORDER
    assert len(schema.FIRST_MATCH_ORDER) == 6


def test_claim_boundary_eight_keys_all_true():
    out = reducer.standardize_live_shaped()
    cb = out["claim_boundary"]
    assert set(cb.keys()) == set(schema.CLAIM_BOUNDARY_KEY_SET)
    assert len(cb) == 8
    assert all(cb[k] is True for k in schema.CLAIM_BOUNDARY_KEY_SET)


def test_affirmative_attribution_checker():
    """Exact allowlist contract (BLOCK 1786008170381 + correction 1786008310888).

    Silent iff empty/ws or member of schema.CLAIM_SILENT_ALLOWLIST (ceilings,
    approved phrasing, successors, branches, IDENTITY_OK__*, prohibition
    calibrations). Every other non-empty string FIRES. No open-ended grammar.
    """
    # (a) every allowlist member silent — denominator from the artifact
    for s in schema.CLAIM_SILENT_ALLOWLIST:
        assert reducer.affirmative_attribution_fires(s) is False, s[:80]
    # (e) empty/whitespace silent
    assert reducer.affirmative_attribution_fires("") is False
    assert reducer.affirmative_attribution_fires("   ") is False
    # (b) one-token mutation of approved[0] and of sentence B fires
    mut0 = schema.APPROVED_PHRASING_EXAMPLES[0].replace("selects", "selectsX", 1)
    mut_b = schema.CLAIM_CEILING_SENTENCE_B + " x"
    assert reducer.affirmative_attribution_fires(mut0) is True
    assert reducer.affirmative_attribution_fires(mut_b) is True
    # prohibition calibrations silent; one-token mutations of two fire (exactness)
    for s in schema.PROHIBITION_CALIBRATION_STRINGS:
        assert reducer.affirmative_attribution_fires(s) is False, s[:80]
    mut_p0 = schema.PROHIBITION_CALIBRATION_STRINGS[0] + " x"
    mut_p1 = schema.PROHIBITION_CALIBRATION_STRINGS[1].replace(
        "weights", "weightsX", 1
    )
    assert reducer.affirmative_attribution_fires(mut_p0) is True
    assert reducer.affirmative_attribution_fires(mut_p1) is True
    # (c) affirmative fire set incl. lead-to/explain
    fire = [
        "rates cause the label split",
        "rates lead to the label split",
        "weights explain the label split",
        "rates cause the label split; no claim about mechanism",
        "the label split is caused by rates",
        "no claim about mechanism but rates cause the split",
        "without claiming mechanism the rates cause the split",
        "no claim about the mechanism, and the rates cause the split",
        "no claim about the mechanism, so the rates cause the split",
        "no claim about the mechanism, yet the rates cause the split",
        "rates determine individual fate",
        "weights drive transfer",
        "the labeler follows the weight profile",
        "the rate profile determines which individual survives",
        "paraphrase of a successor string still fires",
    ]
    for s in fire:
        assert reducer.affirmative_attribution_fires(s) is True, s[:80]


def test_successor_text_rate_branch():
    out = reducer.standardize_live_shaped()
    assert out["successor"] == schema.SUCCESSOR_MAPPING[schema.BRANCH_RATE_PROFILE_SELECTS]


def test_no_float_compare_in_decision_helpers():
    src = Path(inspect.getsourcefile(reducer)).read_text()
    tree = ast.parse(src)
    # Disallow float() constructor in module for decision path hygiene.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            assert node.func.id != "float", "float() call in reducer"


def test_line_caps_step1_files():
    files = [
        REPO / "scripts/a_prime_slice4_count_standardization_schema_v0.py",
        REPO / "scripts/a_prime_slice4_count_standardization_reducer_v0.py",
        REPO
        / "calm/llm_computer/tests/test_a_prime_slice4_count_standardization_reducer_v0.py",
    ]
    for f in files:
        n = len(f.read_text().splitlines())
        assert n < 500, f"{f} has {n} lines"


def _long_key_c1_raw():
    """Pinned Rung-5 long-key C1_profile.raw shape (live extraction path)."""
    return {
        "L0b": {
            "R0": {
                "|B50|_row_ids": 5,
                "present_at_package_N20_row_id_intersection": 1,
                "absent_from_package_N20_row_id_difference": 4,
            },
            "R1b4v2": {
                "|B50|_row_ids": 3,
                "present_at_package_N20_row_id_intersection": 2,
                "absent_from_package_N20_row_id_difference": 1,
            },
        },
        "math_a0": {
            "R0": {
                "|B50|_row_ids": 14,
                "present_at_package_N20_row_id_intersection": 1,
                "absent_from_package_N20_row_id_difference": 13,
            },
            "R1b4v2": {
                "|B50|_row_ids": 9,
                "present_at_package_N20_row_id_intersection": 5,
                "absent_from_package_N20_row_id_difference": 4,
            },
        },
    }


def test_long_key_good_path_rate_branch():
    out = reducer.standardize_from_c1_raw(_long_key_c1_raw(), _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_RATE_PROFILE_SELECTS
    assert out["cells_fraction"]["wL_rL"] == Fraction(3, 8)


def test_long_key_extra_component_bind_fail():
    raw = _long_key_c1_raw()
    raw["L0b"]["R1_EXTRA"] = {
        "|B50|_row_ids": 1,
        "present_at_package_N20_row_id_intersection": 0,
        "absent_from_package_N20_row_id_difference": 1,
    }
    out = reducer.standardize_from_c1_raw(raw, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("component_set_mismatch" in r for r in out["terminal_reasons"])
    # extract itself also raises controlled error (never silent-drop)
    try:
        reducer.extract_counts_from_c1_raw(raw)
        assert False, "expected ComponentSetBindError"
    except reducer.ComponentSetBindError as e:
        assert any("component_set_mismatch" in r for r in e.reasons)


def test_long_key_missing_component_bind_fail_no_keyerror():
    raw = _long_key_c1_raw()
    del raw["L0b"]["R1b4v2"]
    out = reducer.standardize_from_c1_raw(raw, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("component_set_mismatch" in r for r in out["terminal_reasons"])
    try:
        reducer.extract_counts_from_c1_raw(raw)
        assert False, "expected ComponentSetBindError"
    except reducer.ComponentSetBindError as e:
        assert any("component_set_mismatch" in r for r in e.reasons)
    except KeyError:
        assert False, "KeyError must not escape missing-component path"


def test_short_key_missing_component_via_standardize_from_counts():
    counts = _counts()
    del counts["math_a0"]["R1b4v2"]
    out = reducer.standardize_from_counts(counts, _aggs(), _pub())
    assert out["primary"] == schema.BRANCH_STANDARDIZATION_BIND_FAIL
    assert any("component_set_mismatch" in r for r in out["terminal_reasons"])


def test_frozen_count_pins_match_schema():
    assert schema.FROZEN_COUNTS["L0b"]["R0"]["N50"] == 5
    assert schema.FROZEN_COUNTS["math_a0"]["R1b4v2"]["present_N20"] == 5
    assert schema.PUBLISHED_D2_LABELS_RAW["L0b"] == "E_MIXED"


def test_normalize_published_label():
    assert reducer.normalize_published_label("E_MIXED") == "MIXED"
    assert reducer.normalize_published_label("MIXED") == "MIXED"


def test_label_q_boundaries():
    assert reducer.label_q(Fraction(7, 10)) == "PERSISTENT"
    assert reducer.label_q(Fraction(3, 10)) == "TRANSIENT"
    assert reducer.label_q(Fraction(1, 2)) == "MIXED"
    assert reducer.label_q(Fraction(0)) == "TRANSIENT"
    assert reducer.label_q(Fraction(1)) == "PERSISTENT"
