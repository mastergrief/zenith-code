from __future__ import annotations

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    SIZING_VERDICT_RECOMMENDED_LAW,
    SIZING_VERDICT_SIZED_NOT_SUB2,
    VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_in_vivo_bound_validator import (
    IN_VIVO_DOMINANCE_PROVEN,
    IN_VIVO_EXCEEDS,
    VERDICT_SCOPE_IN_VIVO_VALIDATED,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_postrun_pipeline import (
    merge_postrun_verdict,
    run_postrun_arc2b_analysis,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_postrun_pipeline import (
    run_postrun_arc2b_analysis,
)
from calm.llm_computer.tests.test_d_recompute_window_in_vivo_bound_validator_v0 import (
    _make_record,
    _minimal_manifest,
)


def test_merge_downgrades_envelope_recommended_law_without_in_vivo() -> None:
    merged = merge_postrun_verdict(
        horizon_growth={"growth_branch": "PLATEAU_SIZED"},
        acc_sizing={
            "sizing_verdict": SIZING_VERDICT_RECOMMENDED_LAW,
            "strict_less_than_budget": True,
            "verdict_scope": VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
        },
        in_vivo_validation={
            "in_vivo_verdict": IN_VIVO_EXCEEDS,
            "verdict_scope": VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
        },
    )
    assert merged["final_sizing_verdict"] == SIZING_VERDICT_SIZED_NOT_SUB2
    assert merged["recommended_law_eligible"] is False


def test_merge_allows_recommended_law_only_with_in_vivo_validated() -> None:
    merged = merge_postrun_verdict(
        horizon_growth={"growth_branch": "PLATEAU_SIZED"},
        acc_sizing={
            "sizing_verdict": SIZING_VERDICT_RECOMMENDED_LAW,
            "strict_less_than_budget": True,
            "verdict_scope": VERDICT_SCOPE_ENVELOPE_MODEL_ONLY,
        },
        in_vivo_validation={
            "in_vivo_verdict": IN_VIVO_DOMINANCE_PROVEN,
            "verdict_scope": VERDICT_SCOPE_IN_VIVO_VALIDATED,
        },
    )
    assert merged["final_sizing_verdict"] == SIZING_VERDICT_RECOMMENDED_LAW
    assert merged["final_verdict_scope"] == VERDICT_SCOPE_IN_VIVO_VALIDATED
    assert merged["recommended_law_eligible"] is True


def test_pipeline_returns_compact_arc2b_block() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=1, state_key="key.a", lane_indices=[0], flip_positions={0}),
        _make_record(step=1, state_key="key.b", lane_indices=[0], flip_positions=set()),
        _make_record(step=2, state_key="key.a", lane_indices=[0], flip_positions=set()),
        _make_record(step=2, state_key="key.b", lane_indices=[0], flip_positions=set()),
    ]
    result = run_postrun_arc2b_analysis(
        records,
        manifest=manifest,
        numel_for_bpw=16,
        sizing_horizon_h=10,
    )
    assert "horizon_growth_summary" in result
    assert "acc_sizing_summary" in result
    assert "in_vivo_validation" in result
    assert "arc2b_verdict" in result
    assert "lane_indices" not in str(result)


def test_arc2b_verdict_unchanged_with_quantile_block() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=step, state_key="key.a", lane_indices=[0], flip_positions={0})
        for step in range(1, 101)
    ] + [
        _make_record(step=step, state_key="key.b", lane_indices=[0], flip_positions=set())
        for step in range(1, 101)
    ]
    result = run_postrun_arc2b_analysis(
        records,
        manifest=manifest,
        numel_for_bpw=16,
        sizing_horizon_h=100,
    )
    assert "quantile_acc_sizing" in result
    assert result["final_sizing_verdict"] == "INCONCLUSIVE_BUDGET_MAPPING_MISSING"
    assert result["arc2b_verdict"]["final_sizing_verdict"] == result["final_sizing_verdict"]


def test_postrun_pipeline_h200_wire_passes_horizons_and_sizing_horizon() -> None:
    manifest = _minimal_manifest()
    records = [
        _make_record(step=step, state_key="key.a", lane_indices=[0], flip_positions={0})
        for step in range(1, 201)
    ] + [
        _make_record(step=step, state_key="key.b", lane_indices=[0], flip_positions=set())
        for step in range(1, 201)
    ]
    result = run_postrun_arc2b_analysis(
        records,
        manifest=manifest,
        numel_for_bpw=16,
        sizing_horizon_h=200,
        horizons=(25, 50, 100, 200),
        classification_horizon_h=200,
    )
    assert result["horizon_growth_summary"]["summaries_by_h"]["200"] is not None
    assert "quantile_acc_sizing" in result
    assert result["quantile_acc_sizing"]["quantile_sub2_candidate"] is False
