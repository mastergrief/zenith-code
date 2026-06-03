"""CPU/static tests for the HRM-Text native-full-stack Phase-0 scaffold."""
from __future__ import annotations

import re

from calm.hrm_text_158.native_full_stack import (
    ACCEPTANCE_METRICS,
    ACTIVE_HRM_REPO_ROOT,
    ATTRIBUTION_HOOK_POINTS,
    ATTRIBUTION_INTEGRITY_CHECKS,
    FP_EXCEPTION_REGISTRY,
    HISTORICAL_NON_ANCHOR_POINTERS,
    IMPLEMENTATION_STATUS_SKELETON_ONLY,
    LIVE_C1353FD5_OBSERVATIONS,
    LIVE_S1_TRAINER_POINTER,
    PHASE0_LEDGER_ROWS,
    PHASE0_SOURCE_POINTERS,
    PROJECTION_GROUPS,
    SUBSYSTEM_CONTRACTS,
    first_class_metric_names,
)
from calm.hrm_text_158.native_full_stack.fp_exceptions import HIDDEN_FP_LEARNER_FAIL_STATE


def test_source_pointer_is_hashed_read_only_snapshot_anchor() -> None:
    ptr = LIVE_S1_TRAINER_POINTER

    ptr.validate_static()
    assert ACTIVE_HRM_REPO_ROOT == "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"
    assert ptr.sha_kind == "file-content-sha256"
    assert ptr.runtime_dependency is False
    assert ptr.lifecycle == "phase0_snapshot_anchor"
    assert ptr.anchored_as_of == "S1 run 1780347615017-1538f834"
    assert "re-read" in ptr.reanchor_note or "refresh" in ptr.reanchor_note
    assert ptr.expected_sha256 == LIVE_C1353FD5_OBSERVATIONS["sha256"]
    assert re.fullmatch(r"[0-9a-f]{64}", ptr.expected_sha256)
    assert PHASE0_SOURCE_POINTERS == (ptr,)


def test_historical_7206_pointer_is_not_active_anchor() -> None:
    active_hashes = {ptr.expected_sha256 for ptr in PHASE0_SOURCE_POINTERS}
    historical_hashes = {ptr.expected_sha256 for ptr in HISTORICAL_NON_ANCHOR_POINTERS}

    assert "7206be4e7f020526756eceffd82267dbe3da293ba03442ec53350bd0c7e5c28a" in historical_hashes
    assert historical_hashes.isdisjoint(active_hashes)
    assert all(ptr.lifecycle == "historical_non_anchor" for ptr in HISTORICAL_NON_ANCHOR_POINTERS)


def test_subsystem_contracts_stay_at_skeleton_depth() -> None:
    contract_names = {contract.name for contract in SUBSYSTEM_CONTRACTS}

    assert set(PROJECTION_GROUPS) == {
        "attn.gqkv.gate",
        "attn.gqkv.query",
        "attn.gqkv.key",
        "attn.gqkv.value",
        "attn.o",
        "mlp.gate_up.gate",
        "mlp.gate_up.up",
        "mlp.down",
    }
    assert {"authoritative_q_levels", "integer_vote_accumulators", "frozen_scales"} <= contract_names
    assert {"credit_capture_hooks", "authoritative_forward_context", "decode_eos_probe_surface"} <= contract_names
    assert all(contract.implementation_status == IMPLEMENTATION_STATUS_SKELETON_ONLY for contract in SUBSYSTEM_CONTRACTS)
    assert not any("kernel implementation" in contract.native_target_role for contract in SUBSYSTEM_CONTRACTS)


def test_ledger_rows_are_pending_and_make_no_acquisition_claim() -> None:
    assert PHASE0_LEDGER_ROWS
    assert all(row.status == "pending_s1_terminal" for row in PHASE0_LEDGER_ROWS)
    assert all(row.current_b4_measured == "pending_s1_terminal" for row in PHASE0_LEDGER_ROWS)
    assert any(row.subsystem == "decode_eos_smoke" for row in PHASE0_LEDGER_ROWS)
    assert not any("validated acquisition" in row.proof_gate.lower() for row in PHASE0_LEDGER_ROWS)


def test_fp_exception_registry_declares_hidden_fp_fail_state() -> None:
    names = {exception.name for exception in FP_EXCEPTION_REGISTRY}

    assert "frozen_scale_authoritative_state" in names
    assert "structural_bitlinear_fp_master" in names
    assert "current_ttrain_b_fp_master_path" in names
    assert "hidden FP masters" in HIDDEN_FP_LEARNER_FAIL_STATE
    assert all(exception.proof_gate for exception in FP_EXCEPTION_REGISTRY)
    assert any(exception.lifecycle.startswith("sunset") for exception in FP_EXCEPTION_REGISTRY)


def test_attribution_hooks_are_observed_in_live_c1353fd5() -> None:
    obs = LIVE_C1353FD5_OBSERVATIONS

    assert obs["observed_in"] == "live c1353fd5"
    assert obs["expected_grad_enabled_invocation_strata"] == 160
    assert obs["expected_schedule_excluded_no_grad"] == 96
    assert obs["strata_constant_lines"] == "36-37"
    assert obs["strata_assertion_lines"] == "4228-4253"
    assert obs["hook_lines"] == "913-936"
    assert obs["state_lines"] == "494-551"
    assert obs["state_components"] == ("q:int8", "acc:int16", "frozen_scale:float32")
    assert {hook.hook_timing for hook in ATTRIBUTION_HOOK_POINTS} >= {"forward_hook", "full_backward_hook"}
    assert any("decode/EOS" in hook.decode_eos_tie_in for hook in ATTRIBUTION_HOOK_POINTS)
    assert all(check.required for check in ATTRIBUTION_INTEGRITY_CHECKS)


def test_acceptance_metrics_make_iteration_speed_first_class() -> None:
    first_class = set(first_class_metric_names())
    metric_by_name = {metric.name: metric for metric in ACCEPTANCE_METRICS}

    assert {
        "wall_clock_per_step",
        "max_safe_batch",
        "effective_exposure_per_step",
        "time_to_diagnosable_failure",
        "resource_headroom",
        "attribution_integrity",
    } <= first_class
    assert metric_by_name["acquisition_quality"].primary_gate is False
    assert metric_by_name["acquisition_quality"].required_for_first_addition is True
    assert "tracked trend" in metric_by_name["acquisition_quality"].pass_semantics
