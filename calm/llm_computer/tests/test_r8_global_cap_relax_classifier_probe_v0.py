"""CPU fixtures for R8 global-cap-relax classifier and contract."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
    c1_banked_faithful_long_run_global_cap_for_step,
    global_cap_relax_512_for_step,
    named_global_cap_contract_receipt,
    resolve_named_global_cap_spec,
)
from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    build_step_chunk,
)
from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
    MIN_MEASURED_STEPS,
)
from calm.hrm_text_158.native_full_stack.r8_global_cap_relax_classifier_probe import (
    BRANCH_ARTIFACT_INSUFFICIENT,
    BRANCH_CAP_RELAX_DESTABILIZES,
    BRANCH_CAP_WAS_BINDING,
    BRANCH_CARRIER_CAPACITY_FAIL,
    BRANCH_HARNESS_FAIL,
    BRANCH_NOT_CAP_BOUND,
    BRANCH_RELAXATION_INSUFFICIENT,
    BRANCH_SCHEMA_FAIL,
    BRANCH_UNCLASSIFIED,
    R7_BASELINE,
    R7_BASELINE_RUN_ID,
    build_classifier_from_chunks,
    detect_baseline_correct_strict_regressions,
    select_branch,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser


def _summary(
    *,
    step: int = 1,
    candidate: int = 100,
    accepted: int = 40,
    deferred: int = 60,
    age: int = 0,
    accepted_from_prior: int = 0,
) -> dict[str, object]:
    accepted_fresh = accepted - accepted_from_prior
    return {
        "global_rate_cap_enabled": True,
        "global_pre_cap_would_apply_count": candidate,
        "global_rate_cap_accepted_count": accepted,
        "global_rate_cap_deferred_count": deferred,
        "global_rate_cap_cap": 512,
        "global_rate_cap_saturated": candidate > 512,
        "q_changed_count": accepted,
        "deferred_backlog_size": deferred,
        "deferred_backlog_max_age_steps": age,
        "deferred_backlog_max_defer_count": 1 if deferred else 0,
        "accepted_from_prior_deferred_count": accepted_from_prior,
        "accepted_fresh_count": accepted_fresh,
    }


def _chunks_for_profile(
    *,
    steps: int,
    deferred_ratio: float,
    max_age: int,
    pressure_first: int,
    pressure_last: int,
    accepted_from_prior_total: int,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    per_step_prior = accepted_from_prior_total // max(steps, 1)
    for step in range(1, steps + 1):
        candidate = 100
        deferred = int(candidate * deferred_ratio)
        accepted = candidate - deferred
        pressure = pressure_first if step == 1 else pressure_last
        delta = None if step == 1 else pressure - pressure_first
        chunks.append(
            build_step_chunk(
                step=step,
                global_summary=_summary(
                    step=step,
                    deferred=deferred,
                    age=max_age if deferred else 0,
                    accepted_from_prior=per_step_prior,
                ),
                pressure_mass=pressure,
                pressure_mass_delta=delta,
            )
        )
    return chunks


def _metrics_from_chunks(chunks: list[dict[str, object]]) -> dict[str, object]:
    from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
        compute_run_metrics,
    )

    return compute_run_metrics(chunks)


def _prior_audit_with_regression(
    *,
    row_id: str = "math_a0:row_42",
    support: str = "math_a0",
    audit_mismatch: bool = False,
) -> dict[str, object]:
    return {
        "enabled": True,
        "requested_supports": ["L0b", "math_a0", "L0c1"],
        "start_reports": {
            support: {
                "audit_mismatch": audit_mismatch,
                "strict_failure_row_ids": [],
            }
        },
        "final_reports": {
            support: {
                "audit_mismatch": audit_mismatch,
                "strict_failure_row_ids": [row_id],
            }
        },
        "deltas": {
            support: {
                "new_strict_failure_row_ids": [] if audit_mismatch else [row_id],
                "new_parsed_failure_row_ids": ["parsed_only_row"],
            }
        },
    }


def test_global_cap_relax_512_contract_schedule_and_contrast_with_c1():
    for step in (1, 3, 10):
        assert global_cap_relax_512_for_step(step) == 512
    assert c1_banked_faithful_long_run_global_cap_for_step(3) == 256
    receipt = named_global_cap_contract_receipt(GLOBAL_CAP_RELAX_512_CONTRACT_NAME)
    assert receipt["name"] == GLOBAL_CAP_RELAX_512_CONTRACT_NAME
    assert receipt["finite_schedule_source"] == [512]
    resolved = resolve_named_global_cap_spec(GLOBAL_CAP_RELAX_512_CONTRACT_NAME, step=3)
    assert resolved is not None
    assert resolved.cap == 512


def test_r7_baseline_pins_v7_banked_run_not_v5_crash():
    assert R7_BASELINE_RUN_ID.endswith("111514Z_d85208d8")
    assert "095809Z" not in R7_BASELINE_RUN_ID
    assert R7_BASELINE["run_mean_deferred_saturation"] == 0.7984375
    assert R7_BASELINE["run_max_deferred_backlog_max_age_steps"] == 7


def test_argparse_accepts_relax_contract_and_prior_audit_supports():
    parser = build_arg_parser()
    ns = parser.parse_args(
        [
            "--parent",
            "calm/hrm/checkpoints/x.pt",
            "--global-cap-contract",
            GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
            "--prior-audit-supports",
            "L0b,math_a0,L0c1",
        ]
    )
    assert ns.global_cap_contract == GLOBAL_CAP_RELAX_512_CONTRACT_NAME
    assert ns.prior_audit_supports == "L0b,math_a0,L0c1"


def test_audit_regression_helper_strict_only_excludes_audit_mismatch():
    prior = _prior_audit_with_regression()
    regressions = detect_baseline_correct_strict_regressions(prior)
    assert len(regressions) == 1
    assert regressions[0]["row_id"] == "math_a0:row_42"
    mismatch = _prior_audit_with_regression(audit_mismatch=True)
    assert detect_baseline_correct_strict_regressions(mismatch) == []


def test_branch_cap_was_binding():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.35,
        max_age=4,
        pressure_first=10,
        pressure_last=12,
        accepted_from_prior_total=800,
    )
    metrics = _metrics_from_chunks(chunks)
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt={"prior_audit": {"enabled": True, "deltas": {}}},
        prior_audit={"enabled": True, "deltas": {}},
    )
    assert branch["branch"] == BRANCH_CAP_WAS_BINDING


def test_branch_not_cap_bound():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.798,
        max_age=7,
        pressure_first=1,
        pressure_last=20_000_000,
        accepted_from_prior_total=600,
    )
    metrics = _metrics_from_chunks(chunks)
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt=None,
        prior_audit=None,
    )
    assert branch["branch"] == BRANCH_NOT_CAP_BOUND


def test_branch_relaxation_insufficient():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.71,
        max_age=5,
        pressure_first=100,
        pressure_last=120,
        accepted_from_prior_total=650,
    )
    metrics = _metrics_from_chunks(chunks)
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt=None,
        prior_audit=None,
    )
    assert branch["branch"] == BRANCH_RELAXATION_INSUFFICIENT


def test_branch_cap_relax_destabilizes_requires_strict_regression_and_drain():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.35,
        max_age=4,
        pressure_first=10,
        pressure_last=12,
        accepted_from_prior_total=650,
    )
    metrics = _metrics_from_chunks(chunks)
    prior = _prior_audit_with_regression()
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt={"prior_audit": prior},
        prior_audit=prior,
    )
    assert branch["branch"] == BRANCH_CAP_RELAX_DESTABILIZES


def test_max_age_five_overlap_priority_destabilizes_before_was_binding():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.35,
        max_age=5,
        pressure_first=10,
        pressure_last=12,
        accepted_from_prior_total=800,
    )
    metrics = _metrics_from_chunks(chunks)
    prior = _prior_audit_with_regression()
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt={"prior_audit": prior},
        prior_audit=prior,
    )
    assert branch["branch"] == BRANCH_CAP_RELAX_DESTABILIZES


def test_max_age_five_without_regression_can_be_was_binding():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.35,
        max_age=5,
        pressure_first=10,
        pressure_last=12,
        accepted_from_prior_total=800,
    )
    metrics = _metrics_from_chunks(chunks)
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt=None,
        prior_audit=None,
    )
    assert branch["branch"] == BRANCH_CAP_WAS_BINDING


def test_branch_unclassified_emits_failed_thresholds_and_metric_snapshot():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.75,
        max_age=6,
        pressure_first=100,
        pressure_last=120,
        accepted_from_prior_total=650,
    )
    metrics = _metrics_from_chunks(chunks)
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt=None,
        prior_audit=None,
    )
    assert branch["branch"] == BRANCH_UNCLASSIFIED
    assert branch["next_action"] == "manual_review_required"
    assert branch["metric_snapshot"]["steps_observed"] == 10
    assert isinstance(branch["failed_thresholds"], list)
    assert branch["failed_thresholds"]


def test_harness_fail_schema_fail_and_artifact_insufficient():
    metrics = _metrics_from_chunks([])
    assert (
        select_branch(
            harness_fail=True,
            schema_fail=False,
            metrics=metrics,
            diagnostic_receipt=None,
            prior_audit=None,
        )["branch"]
        == BRANCH_HARNESS_FAIL
    )
    assert (
        select_branch(
            harness_fail=False,
            schema_fail=True,
            metrics=metrics,
            diagnostic_receipt=None,
            prior_audit=None,
        )["branch"]
        == BRANCH_SCHEMA_FAIL
    )
    short = _metrics_from_chunks(
        _chunks_for_profile(
            steps=MIN_MEASURED_STEPS - 1,
            deferred_ratio=0.5,
            max_age=1,
            pressure_first=1,
            pressure_last=2,
            accepted_from_prior_total=0,
        )
    )
    assert (
        select_branch(
            harness_fail=False,
            schema_fail=False,
            metrics=short,
            diagnostic_receipt=None,
            prior_audit=None,
        )["branch"]
        == BRANCH_ARTIFACT_INSUFFICIENT
    )


def test_carrier_capacity_fail_from_headroom_breach():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.5,
        max_age=3,
        pressure_first=1,
        pressure_last=2,
        accepted_from_prior_total=0,
    )
    metrics = _metrics_from_chunks(chunks)
    diagnostic = {
        "stop_reason": "headroom_breach",
        "step_reports": {"9": {"step_result": {"headroom_breach": True}}},
    }
    branch = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics=metrics,
        diagnostic_receipt=diagnostic,
        prior_audit=None,
    )
    assert branch["branch"] == BRANCH_CARRIER_CAPACITY_FAIL


def test_build_classifier_from_chunks_includes_baseline_comparison():
    chunks = _chunks_for_profile(
        steps=10,
        deferred_ratio=0.35,
        max_age=4,
        pressure_first=10,
        pressure_last=12,
        accepted_from_prior_total=800,
    )
    receipt = build_classifier_from_chunks(
        chunks=chunks,
        run_root="/tmp/relax_run",
        head_sha256="abc",
        sidecar_path="/tmp/relax_run/diagnostic/r7_cap_defer_pressure_sidecar.jsonl",
    )
    assert receipt["sidecar_source"] == "relax_run_own_diagnostic_sidecar"
    assert receipt["r7_baseline_provenance"]["run_id"] == R7_BASELINE_RUN_ID
    assert receipt["baseline_comparison"]["baseline_run_id"] == R7_BASELINE_RUN_ID


def test_r8_flag_witness_passes_relax_argv(tmp_path: Path):
    from scripts.hrm_text_158_r8_flag_witness import run_flag_witness

    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True)
    argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--parent",
        "calm/hrm/checkpoints/x.pt",
        "--global-cap-contract",
        GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
        "--prior-audit-supports",
        "L0b,math_a0,L0c1",
        "--r7-deferred-backlog-carry",
        "--r7-cap-defer-pressure-instrumentation",
        "--two-tier-carry-w6-enabled",
        "--persistent-accumulator-w6-byte-packed",
        "--persistent-q-ternary-byte-packed",
        "--curriculum-seed",
        "44",
        "--support-order-seed",
        "43",
        "--eligible-scope",
        "all-bitlinear",
        "--steps",
        "10",
    ]
    (prelaunch / "argv_echo.json").write_text(
        json.dumps({"arms": {"diagnostic": {"argv": argv}}}) + "\n",
        encoding="utf-8",
    )
    witness = run_flag_witness(tmp_path)
    assert witness["r8_flag_witness_pass"] is True


def test_r8_flag_witness_fails_c1_banked_or_missing_prior_audit(tmp_path: Path):
    from scripts.hrm_text_158_r8_flag_witness import run_flag_witness

    prelaunch = tmp_path / "prelaunch"
    prelaunch.mkdir(parents=True)
    bad_argv = [
        "python3",
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
        "--parent",
        "calm/hrm/checkpoints/x.pt",
        "--global-cap-contract",
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        "--r7-deferred-backlog-carry",
        "--r7-cap-defer-pressure-instrumentation",
        "--curriculum-seed",
        "44",
        "--support-order-seed",
        "43",
        "--eligible-scope",
        "all-bitlinear",
        "--steps",
        "10",
    ]
    (prelaunch / "argv_echo.json").write_text(
        json.dumps({"arms": {"diagnostic": {"argv": bad_argv}}}) + "\n",
        encoding="utf-8",
    )
    witness = run_flag_witness(tmp_path)
    assert witness["r8_flag_witness_pass"] is False
    assert any("c1" in failure or "prior_audit" in failure for failure in witness["failures"])
