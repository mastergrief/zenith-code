from __future__ import annotations

from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.activation_credit_ceiling_audit import (
    H1_ARM_BASELINE_TAYLOR_Q5,
    H1_ARM_SNR_Q5,
    H1_ARM_TOPOLOGY_BLOCK128,
    H1_BRANCH_GT2_Q5_OR_L5_ONLY_RECOVERS_DIAGNOSTIC,
    H1_BRANCH_HARNESS_OR_CALIBRATION_FAIL,
    H1_BRANCH_NO_BUCKET_ARM_RECOVERS,
    H1_BRANCH_SUB2_L3_BUCKET_RECOVERS,
    H1_COUNTERFACTUAL_BUCKET_FIELD,
    H1_PREREGISTERED_ARM_SPECS,
    KNOWN_BRANCH4_AUC_TOLERANCE,
    KNOWN_BRANCH4_RECEIPT_FAMILY_AUC_EXPECTATIONS,
    SEED29_LABEL,
    SEED43_LABEL,
    load_activation_credit_ceiling_audit_receipt,
    run_h1_bucket_swap_counterfactual,
)
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ACTIVATION_CREDIT_PRIMARY_FAMILY_ID,
    ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    _activation_credit_family_key,
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


def test_h1_preregistered_arm_count_and_ephemeral_field_name() -> None:
    assert len(H1_PREREGISTERED_ARM_SPECS) == 9
    assert H1_COUNTERFACTUAL_BUCKET_FIELD == "h1_counterfactual_bucket_bin"


def test_native_family_registry_unchanged_raises_on_unknown_id() -> None:
    row = {
        "taylor_benefit_q5_bin": 0,
        "snr_q5_bin": 0,
        "diagfisher_q5_bin": 0,
        "topology_row_block_128": 0,
    }
    with pytest.raises(ValueError, match="unsupported activation-credit family_id"):
        _activation_credit_family_key(row, family_id="F_counterfactual_unknown")


@pytest.mark.skipif(
    not _BRANCH4_SEED43.is_file() or not _BRANCH4_SEED29.is_file(),
    reason="historical branch4 replay receipts unavailable",
)
def test_h1_baseline_reproduces_branch4_anchors() -> None:
    result = run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    assert result["harness_ok"] is True
    baseline = result["arm_metrics"][H1_ARM_BASELINE_TAYLOR_Q5]
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
def test_h1_receipt_native_arms_match_receipt_family_metrics() -> None:
    seed43_receipt = load_activation_credit_ceiling_audit_receipt(
        _BRANCH4_SEED43,
        seed_label=SEED43_LABEL,
    )
    seed29_receipt = load_activation_credit_ceiling_audit_receipt(
        _BRANCH4_SEED29,
        seed_label=SEED29_LABEL,
    )
    result = run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    taylor = result["arm_metrics"][H1_ARM_BASELINE_TAYLOR_Q5]
    snr = result["arm_metrics"][H1_ARM_SNR_Q5]
    assert _approx_equal(
        taylor[SEED43_LABEL]["receipt_family_compressed_auc"],
        float(
            seed43_receipt.family_metrics_by_id[ACTIVATION_CREDIT_PRIMARY_FAMILY_ID][
                "within_band_pairwise_auc_report_only"
            ]
        ),
    )
    assert _approx_equal(
        snr[SEED29_LABEL]["receipt_family_compressed_auc"],
        float(
            seed29_receipt.family_metrics_by_id[ACTIVATION_CREDIT_SNR_Q5_ABLATION_FAMILY_ID][
                "within_band_pairwise_auc_report_only"
            ]
        ),
    )


@pytest.mark.skipif(
    not _BRANCH4_SEED43.is_file() or not _BRANCH4_SEED29.is_file(),
    reason="historical branch4 replay receipts unavailable",
)
def test_h1_branch4_diagnostic_emits_expected_classifier_shape() -> None:
    result = run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=_BRANCH4_SEED43,
        seed29_receipt_path=_BRANCH4_SEED29,
    )
    classifier = result["branch_classifier"]
    assert "winning_arm_ids" in classifier
    assert "winning_arm_persistent_bits" in classifier
    assert "sub2_bucket_recovery" in classifier
    assert classifier["sub2_bucket_recovery"] is False
    assert result["topology_negative_control_clears"] is False
    assert classifier["primary_branch"] in {
        H1_BRANCH_NO_BUCKET_ARM_RECOVERS,
        H1_BRANCH_HARNESS_OR_CALIBRATION_FAIL,
    }
    for arm_id, metrics in result["arm_metrics"].items():
        assert metrics["recovery_tier"]
        assert metrics["uses_raw_continuous_inside_bucket"] is False
        assert arm_id in {spec.arm_id for spec in H1_PREREGISTERED_ARM_SPECS}


def test_h1_topology_negative_control_routes_to_harness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calm.hrm_text_158.native_full_stack import activation_credit_ceiling_audit as audit

    def fake_run(
        *,
        seed43_receipt_path: str | Path,
        seed29_receipt_path: str | Path,
    ) -> dict[str, object]:
        branch = audit._h1_emit_branch_classifier(
            harness_ok=True,
            calibration_failure=False,
            topology_negative_control_clears=True,
            sub2_l3_winning_arm_ids=(),
            gt2_only_winning_arm_ids=(),
        )
        return {"branch_classifier": branch, "topology_negative_control_clears": True}

    monkeypatch.setattr(audit, "run_h1_bucket_swap_counterfactual", fake_run)
    result = audit.run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=Path("a.json"),
        seed29_receipt_path=Path("b.json"),
    )
    assert (
        result["branch_classifier"]["primary_branch"]
        == H1_BRANCH_HARNESS_OR_CALIBRATION_FAIL
    )


def test_h1_gt2_diagnostic_branch_not_sub2_win(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calm.hrm_text_158.native_full_stack import activation_credit_ceiling_audit as audit

    def fake_run(
        *,
        seed43_receipt_path: str | Path,
        seed29_receipt_path: str | Path,
    ) -> dict[str, object]:
        branch = audit._h1_emit_branch_classifier(
            harness_ok=True,
            calibration_failure=False,
            topology_negative_control_clears=False,
            sub2_l3_winning_arm_ids=(),
            gt2_only_winning_arm_ids=[H1_ARM_SNR_Q5],
        )
        return {"branch_classifier": branch}

    monkeypatch.setattr(audit, "run_h1_bucket_swap_counterfactual", fake_run)
    result = audit.run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=Path("a.json"),
        seed29_receipt_path=Path("b.json"),
    )
    assert (
        result["branch_classifier"]["primary_branch"]
        == H1_BRANCH_GT2_Q5_OR_L5_ONLY_RECOVERS_DIAGNOSTIC
    )
    assert result["branch_classifier"]["sub2_bucket_recovery"] is False


def test_h1_sub2_l3_branch_sets_sub2_bucket_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from calm.hrm_text_158.native_full_stack import activation_credit_ceiling_audit as audit

    def fake_run(
        *,
        seed43_receipt_path: str | Path,
        seed29_receipt_path: str | Path,
    ) -> dict[str, object]:
        branch = audit._h1_emit_branch_classifier(
            harness_ok=True,
            calibration_failure=False,
            topology_negative_control_clears=False,
            sub2_l3_winning_arm_ids=["ARM_TAYLOR_L3"],
            gt2_only_winning_arm_ids=(),
        )
        return {"branch_classifier": branch}

    monkeypatch.setattr(audit, "run_h1_bucket_swap_counterfactual", fake_run)
    result = audit.run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=Path("a.json"),
        seed29_receipt_path=Path("b.json"),
    )
    assert (
        result["branch_classifier"]["primary_branch"]
        == H1_BRANCH_SUB2_L3_BUCKET_RECOVERS
    )
    assert result["branch_classifier"]["sub2_bucket_recovery"] is True


def test_h1_harness_fail_on_missing_receipt(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"
    result = run_h1_bucket_swap_counterfactual(
        seed43_receipt_path=missing,
        seed29_receipt_path=missing,
    )
    assert result["harness_ok"] is False
    assert (
        result["branch_classifier"]["primary_branch"]
        == H1_BRANCH_HARNESS_OR_CALIBRATION_FAIL
    )
