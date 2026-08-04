"""F1–F4 pure core tests for A′ slice1 fidelity."""
from __future__ import annotations

from scripts.a_prime_slice1_fidelity_core import (
    DEFAULT_PINNED_SUPPORTS,
    FINAL_BRANCHES,
    branch_matches_rc,
    classify_branch,
    extract_prior_rates,
    parse_strict_exact_fraction,
)


def _receipt(fracs: dict[str, str]) -> dict:
    pins = DEFAULT_PINNED_SUPPORTS
    per = {}
    proofs = {}
    for name, pin in pins.items():
        proofs[name] = {
            "support_hash16": pin["expected_hash16"],
            "expected_count": pin["expected_count"],
        }
        per[name] = {
            "support_hash16": pin["expected_hash16"],
            "support_rows_expected": pin["expected_count"],
            "final": {"strict_exact": fracs[name]},
        }
    return {
        "prior_audit": {
            "enabled": True,
            "per_support": per,
            "support_proofs": proofs,
        }
    }


def test_F1_frac_and_row_weighted_aggregate():
    """Fails on wrong aggregate or silent pin miss."""
    assert parse_strict_exact_fraction("230/230") == (230, 230)
    assert parse_strict_exact_fraction("bad") is None
    r = extract_prior_rates(
        _receipt({"L0b": "230/230", "math_a0": "1200/1255"}),
        pinned_supports=DEFAULT_PINNED_SUPPORTS,
    )
    assert r["ok"] is True
    assert r["aggregate_count"] == 230 + 1200
    assert r["aggregate_total"] == 230 + 1255
    assert abs(r["aggregate_exact_rate"] - (1430 / 1485)) < 1e-12


def test_F2_missing_pin_instrument_gap():
    """Fails if fail-open on missing pin."""
    r = extract_prior_rates(
        {"prior_audit": {"enabled": True, "per_support": {}}},
        pinned_supports=DEFAULT_PINNED_SUPPORTS,
    )
    assert r["ok"] is False
    assert r["pin_errors"]
    branch, _ = classify_branch(
        dense_prior=r, nondense_prior=r, delta_collapse=0.1
    )
    assert branch == "INSTRUMENT_GAP"


def test_F3_classify_threshold():
    """Fails if threshold off-by."""
    dense = extract_prior_rates(
        _receipt({"L0b": "230/230", "math_a0": "1255/1255"}),
        pinned_supports=DEFAULT_PINNED_SUPPORTS,
    )
    nondense_ok = extract_prior_rates(
        _receipt({"L0b": "220/230", "math_a0": "1200/1255"}),
        pinned_supports=DEFAULT_PINNED_SUPPORTS,
    )
    nondense_bad = extract_prior_rates(
        _receipt({"L0b": "100/230", "math_a0": "800/1255"}),
        pinned_supports=DEFAULT_PINNED_SUPPORTS,
    )
    b1, d1 = classify_branch(
        dense_prior=dense, nondense_prior=nondense_ok, delta_collapse=0.1
    )
    assert b1 == "PAIRED_ACHIEVED_FIDELITY_AT_N"
    b2, d2 = classify_branch(
        dense_prior=dense, nondense_prior=nondense_bad, delta_collapse=0.1
    )
    assert b2 == "FIDELITY_COLLAPSE"
    assert d2 is not None and d2 > 0.1


def test_F4_final_branches_fail_closed():
    """Fails if unknown branch treated as final."""
    assert "NOT_A_BRANCH" not in FINAL_BRANCHES
    assert branch_matches_rc("INSTRUMENT_GAP", 2)
    assert not branch_matches_rc("INSTRUMENT_GAP", 0)
    assert not branch_matches_rc("UNKNOWN", 0)
