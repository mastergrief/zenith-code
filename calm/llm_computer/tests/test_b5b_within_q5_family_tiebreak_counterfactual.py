from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.activation_credit_ceiling_audit import (
    B5B_BRANCH_CALIBRATION_FAILURE,
    B5B_BRANCH_HARNESS_OR_INPUT_FAIL,
    B5B_BRANCH_RAW_ONLY_RECOVERS,
    B5B_BRANCH_TIEBREAK_BASELINE_REPRO,
    B5B_BRANCH_TIEBREAK_STILL_COLLAPSES,
    B5B_TASK_ID,
    KNOWN_BRANCH4_AUC_TOLERANCE,
    KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS,
    SEED29_LABEL,
    SEED43_LABEL,
    load_activation_credit_ceiling_audit_receipt,
    run_b5b_within_q5_family_tiebreak_counterfactual,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK,
    ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP,
    ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL,
)

_REPO_ROOT = Path(__file__).resolve().parents[3]
_BRANCH4_SEED43 = (
    _REPO_ROOT
    / ".run_logs/b7b_screen_b1_replay_sources_1781001257348/source_00_94057581e3d1239a.json"
)
_BRANCH4_SEED29 = (
    _REPO_ROOT
    / ".run_logs/b7b_screen_b1_replay_sources_1781001257348/source_01_0071b56bcfee5235.json"
)


def _approx_equal(left: float, right: float) -> bool:
    return abs(float(left) - float(right)) <= KNOWN_BRANCH4_AUC_TOLERANCE


@pytest.mark.skipif(
    not _BRANCH4_SEED43.is_file() or not _BRANCH4_SEED29.is_file(),
    reason="historical branch4 replay receipts unavailable",
)
def test_b5b_reproduces_baseline_anchors_on_branch4_fixtures() -> None:
    result = run_b5b_within_q5_family_tiebreak_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    assert result["task_id"] == B5B_TASK_ID
    assert result["harness_ok"] is True
    assert result["baseline_anchor_reproduced"] is True
    assert (
        B5B_BRANCH_TIEBREAK_BASELINE_REPRO
        in result["branch_classifier"]["all_applicable_branches"]
    )
    baseline = result["variant_metrics"][ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK]
    assert _approx_equal(
        baseline[SEED43_LABEL]["receipt_family_compressed_auc"],
        KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED43_LABEL],
    )
    assert _approx_equal(
        baseline[SEED29_LABEL]["receipt_family_compressed_auc"],
        KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS[SEED29_LABEL],
    )


@pytest.mark.skipif(
    not _BRANCH4_SEED43.is_file() or not _BRANCH4_SEED29.is_file(),
    reason="historical branch4 replay receipts unavailable",
)
def test_b5b_status_quo_counterfactual_matches_receipt_family_metrics() -> None:
    seed43_receipt = load_activation_credit_ceiling_audit_receipt(
        _BRANCH4_SEED43,
        seed_label=SEED43_LABEL,
    )
    seed29_receipt = load_activation_credit_ceiling_audit_receipt(
        _BRANCH4_SEED29,
        seed_label=SEED29_LABEL,
    )
    result = run_b5b_within_q5_family_tiebreak_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    baseline = result["variant_metrics"][ACTIVATION_CREDIT_TIEBREAK_KEY_CURRENT_RANK]
    seed43_family = seed43_receipt.family_metrics_by_id["F_taylor_benefit_q5"][
        "within_band_pairwise_auc_report_only"
    ]
    seed29_family = seed29_receipt.family_metrics_by_id["F_taylor_benefit_q5"][
        "within_band_pairwise_auc_report_only"
    ]
    assert _approx_equal(
        baseline[SEED43_LABEL]["receipt_family_compressed_auc"],
        float(seed43_family),
    )
    assert _approx_equal(
        baseline[SEED29_LABEL]["receipt_family_compressed_auc"],
        float(seed29_family),
    )


@pytest.mark.skipif(
    not _BRANCH4_SEED43.is_file() or not _BRANCH4_SEED29.is_file(),
    reason="historical branch4 replay receipts unavailable",
)
def test_b5b_branch_classifier_emits_science_branch_on_branch4_fixtures() -> None:
    result = run_b5b_within_q5_family_tiebreak_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    primary = result["branch_classifier"]["primary_branch"]
    assert primary in {
        B5B_BRANCH_TIEBREAK_STILL_COLLAPSES,
        B5B_BRANCH_RAW_ONLY_RECOVERS,
        B5B_BRANCH_CALIBRATION_FAILURE,
    }
    assert result["invariants"]["uses_raw_continuous_inside_bucket"] is False
    ternary = result["variant_metrics"][
        ACTIVATION_CREDIT_TIEBREAK_KEY_TERNARY_ELIGIBILITY_ORDINAL
    ]
    assert ternary["uses_raw_continuous_inside_bucket"] is False
    raw = result["variant_metrics"][ACTIVATION_CREDIT_TIEBREAK_KEY_RAW_ELIGIBILITY_FP]
    assert raw["uses_raw_continuous_inside_bucket"] is True
    assert raw["role"] == "label_leak_upper_bound_diagnostic_only"
    assert result["branch_classifier"]["sub2_win"] is False


def test_b5b_harness_fail_on_missing_receipt(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    result = run_b5b_within_q5_family_tiebreak_counterfactual(
        seed43_receipt_path=missing,
        seed29_receipt_path=missing,
    )
    assert result["harness_ok"] is False
    assert (
        result["branch_classifier"]["primary_branch"] == B5B_BRANCH_HARNESS_OR_INPUT_FAIL
    )


def test_b5b_raw_only_tagged_non_win_when_raw_recovers_more_than_ternary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
  # Synthetic path: if raw recovers but ternary does not, primary must be RAW_ONLY not sub2 win.
    from calm.hrm_text_158.native_full_stack import activation_credit_ceiling_audit as audit

    def fake_run(
        *,
        seed43_receipt_path: str | Path,
        seed29_receipt_path: str | Path,
    ) -> dict[str, object]:
        branch = audit._b5b_emit_branch_classifier(
            harness_ok=True,
            baseline_reproduced=True,
            calibration_failure=False,
            ternary_recovers=False,
            q5_recovers=False,
            raw_recovers=True,
            ordinal_only_recovers=False,
        )
        return {
            "branch_classifier": branch,
            "invariants": {"uses_raw_continuous_inside_bucket": False},
        }

    monkeypatch.setattr(audit, "run_b5b_within_q5_family_tiebreak_counterfactual", fake_run)
    result = audit.run_b5b_within_q5_family_tiebreak_counterfactual(
        seed43_receipt_path=tmp_path / "a.json",
        seed29_receipt_path=tmp_path / "b.json",
    )
    assert result["branch_classifier"]["primary_branch"] == B5B_BRANCH_RAW_ONLY_RECOVERS
    assert result["branch_classifier"]["sub2_win"] is False
    assert (
        B5B_BRANCH_RAW_ONLY_RECOVERS
        in result["branch_classifier"]["explicit_non_win_branches"]
    )
