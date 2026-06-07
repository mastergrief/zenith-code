from __future__ import annotations

from calm.hrm_text_158.native_full_stack.sub2_native_birth_movement_contract import (
    tierb_lane1_hybrid_movement_report,
    tierb_lane1_hybrid_movement_success_contract,
    tierb_lane1_hybrid_movement_terminal_semantics_contract,
)
from calm.hrm_text_158.native_full_stack.sub2_native_birth_scaffold import (
    HYBRID_MOVEMENT_CONTRACT_SCOPE,
    RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT,
)


def _audit_event(
    *,
    step: int,
    strict_exact_count: int,
    support_row_count: int = 120,
    baseline_native_init: bool,
    nonimproving_audit_streak: int = 0,
    last_improving_step: int | None = None,
    passed: bool = True,
) -> dict[str, object]:
    return {
        "step": step,
        "strict_exact": f"{strict_exact_count}/{support_row_count}",
        "strict_exact_count": strict_exact_count,
        "strict_exact_total": support_row_count,
        "support_row_count": support_row_count,
        "baseline_native_init": baseline_native_init,
        "nonimproving_audit_streak": nonimproving_audit_streak,
        "last_improving_step": last_improving_step,
        "pass": passed,
    }


def _lane_summary(
    *,
    history: list[dict[str, object]],
    q_changed_total: int,
    stop_reasons: list[str] | None = None,
) -> dict[str, object]:
    return {
        "lane": "lane1",
        "role": "full_hybrid_qacc_activation_codec",
        "support_exact_audit_history": history,
        "q_changed_total": q_changed_total,
        "stop_conditions_triggered": [
            {"reason": reason} for reason in (stop_reasons or [])
        ],
    }


def test_movement_success_contract_carries_named_sidecar_stop_surfaces():
    contract = tierb_lane1_hybrid_movement_success_contract(
        mode="applied_crossing_direction_plus_4bit_residual",
        steps=200,
        support_name="L0c2-K2-addition-120",
        support_row_count=120,
        support_audit_every_steps=25,
        support_audit_batch_size=8,
        support_plateau_warmup_step=100,
        support_plateau_patience_audits=4,
    )

    assert contract["contract_scope"] == HYBRID_MOVEMENT_CONTRACT_SCOPE
    assert contract["required_terminal_report"] == "lane1_hybrid_movement_report"
    assert contract["terminal_taxonomy"] == [
        "support_exact_moved",
        "flat_exact_plateau",
        "hard_fail",
    ]
    assert contract["persistent_authority_contract"]["runtime_state_authority"] == (
        RUNTIME_STATE_AUTHORITY_SUB2_PERSISTENT_HYBRID_DENSE_TRANSIENT_CREDIT
    )
    assert "persistent_sidecar_budget_fail" in contract["stop_conditions"]
    assert "persistent_sidecar_state_authority_fail" in contract["stop_conditions"]


def test_movement_terminal_semantics_exit_codes_are_frozen():
    semantics = tierb_lane1_hybrid_movement_terminal_semantics_contract(
        steps=200,
        support_target_strict_exact=120,
        support_audit_every_steps=25,
        support_plateau_warmup_step=100,
        support_plateau_patience_audits=4,
    )

    assert semantics["success_exit_code"] == 0
    assert semantics["failure_exit_code"] == 2
    assert "persistent_sidecar_budget_fail" in semantics["hard_fail_when"]
    assert "persistent_sidecar_state_authority_fail" in semantics["hard_fail_when"]


def test_movement_report_classifies_support_exact_moved_with_exit_zero():
    report = tierb_lane1_hybrid_movement_report(
        _lane_summary(
            history=[
                _audit_event(step=0, strict_exact_count=5, baseline_native_init=True, last_improving_step=0),
                _audit_event(step=1, strict_exact_count=7, baseline_native_init=False, last_improving_step=1),
            ],
            q_changed_total=16,
        ),
        validation_tiny=False,
        tiny_non_citable_label="tiny",
        support_name="L0c2-K2-addition-120",
        support_target_strict_exact=120,
    )

    assert report["classification"] == "support_exact_moved"
    assert report["baseline_step"] == 0
    assert report["best_strict_exact_delta"] == 2
    assert report["expected_runtime_exit_code"] == 0
    assert report["pass"] is True


def test_movement_report_classifies_flat_exact_plateau_with_exit_two():
    report = tierb_lane1_hybrid_movement_report(
        _lane_summary(
            history=[
                _audit_event(step=0, strict_exact_count=5, baseline_native_init=True, last_improving_step=0),
                _audit_event(
                    step=25,
                    strict_exact_count=5,
                    baseline_native_init=False,
                    nonimproving_audit_streak=4,
                    last_improving_step=0,
                ),
            ],
            q_changed_total=12,
        ),
        validation_tiny=False,
        tiny_non_citable_label="tiny",
        support_name="L0c2-K2-addition-120",
        support_target_strict_exact=120,
    )

    assert report["classification"] == "flat_exact_plateau"
    assert report["expected_runtime_exit_code"] == 2
    assert report["pass"] is False


def test_movement_report_classifies_hard_fail_with_exit_two():
    report = tierb_lane1_hybrid_movement_report(
        _lane_summary(
            history=[
                _audit_event(step=0, strict_exact_count=5, baseline_native_init=True, last_improving_step=0),
                _audit_event(step=1, strict_exact_count=6, baseline_native_init=False, last_improving_step=1),
            ],
            q_changed_total=8,
            stop_reasons=["persistent_sidecar_budget_fail"],
        ),
        validation_tiny=False,
        tiny_non_citable_label="tiny",
        support_name="L0c2-K2-addition-120",
        support_target_strict_exact=120,
    )

    assert report["classification"] == "hard_fail"
    assert report["hard_fail_reasons"] == ["persistent_sidecar_budget_fail"]
    assert report["expected_runtime_exit_code"] == 2
    assert report["pass"] is False


def test_movement_report_classifies_support_audit_only_validation_tiny():
    report = tierb_lane1_hybrid_movement_report(
        _lane_summary(
            history=[
                _audit_event(step=0, strict_exact_count=5, baseline_native_init=True, last_improving_step=0),
                _audit_event(step=1, strict_exact_count=5, baseline_native_init=False, last_improving_step=0),
            ],
            q_changed_total=4,
        ),
        validation_tiny=True,
        tiny_non_citable_label="cpu_tiny_only",
        support_name="L0c2-K2-addition-120",
        support_target_strict_exact=120,
    )

    assert report["classification"] == "support_audit_only_validation_tiny"
    assert report["label"] == "cpu_tiny_only"
    assert report["measurement_pass"] is True
    assert report["expected_runtime_exit_code"] == 0
    assert report["pass"] is True
