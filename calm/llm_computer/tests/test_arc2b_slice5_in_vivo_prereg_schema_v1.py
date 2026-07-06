"""CPU-static schema tests for Arc #2b Slice-5 Step 1 prereg/preflight artifacts."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
DRAFT = (
    REPO
    / "artifacts/consensus_prep/arc2b_slice5_in_vivo_law_validation_prereg_packet_v1_draft.json"
)
PREFLIGHT = (
    REPO
    / "artifacts/measurement_closeout/arc2b_slice5_feasibility_preflight_receipt.json"
)
APPLY_SCRIPT = (
    REPO / "scripts/apply_arc2b_slice5_in_vivo_law_validation_prereg_packet.py"
)
B1_CLASSIFIER_RECEIPT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "d_recompute_window_feasibility_seed43_43_2189e72017/classifier_receipt.json"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def test_prereg_and_preflight_artifacts_exist() -> None:
    assert DRAFT.is_file(), "prereg draft missing; run apply script"
    assert PREFLIGHT.is_file(), "preflight receipt missing; run apply script"


def test_prereg_packet_schema_valid() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        validate_prereg_packet_schema,
    )

    packet = _load(DRAFT)
    failures = validate_prereg_packet_schema(packet)
    assert failures == [], failures


def test_preflight_receipt_schema_valid() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        validate_preflight_receipt_schema,
    )

    receipt = _load(PREFLIGHT)
    failures = validate_preflight_receipt_schema(receipt)
    assert failures == [], failures


def test_offline_b1_contract_forbids_mechanism_terminals() -> None:
    packet = _load(DRAFT)
    contract = packet["offline_b1_contract"]
    assert "D_NEEDS_UPDATE_LAW_REDESIGN" in contract["forbidden_terminal_branches"]
    assert "SLICE5_IN_VIVO_LAW_BOUND" in contract["forbidden_terminal_branches"]
    assert contract["decay_mismatch_diagnostic_only"] is True


def test_adversarial_b1_fixture_never_emits_mechanism_terminal() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        OFFLINE_B1_ALLOWED_TERMINALS,
        STEP2_ONLY_TERMINALS,
        build_adversarial_b1_offline_branch_input,
        classify_arc2b_slice5_in_vivo_branch,
    )

    inputs = build_adversarial_b1_offline_branch_input()
    assert inputs["offline_bracket_decision"] == "REAL_DENSITY_EXCEEDS_SUB2"
    assert inputs["runtime_decay_den"] == 1
    assert inputs["prereg_law_decay_den"] == 2

    result = classify_arc2b_slice5_in_vivo_branch(inputs)
    terminal = result["terminal_branch"]
    assert terminal not in STEP2_ONLY_TERMINALS
    assert terminal in OFFLINE_B1_ALLOWED_TERMINALS
    assert terminal == "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH"


@pytest.mark.skipif(
    not B1_CLASSIFIER_RECEIPT.is_file(),
    reason="B1 classifier receipt not available on this host",
)
def test_b1_classifier_receipt_routes_diagnostic_not_redesign() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        STEP2_ONLY_TERMINALS,
        build_branch_input_from_b1_classifier_receipt,
        classify_arc2b_slice5_in_vivo_branch,
    )

    receipt = json.loads(B1_CLASSIFIER_RECEIPT.read_text(encoding="utf-8"))
    inputs = build_branch_input_from_b1_classifier_receipt(receipt)
    result = classify_arc2b_slice5_in_vivo_branch(inputs)
    assert result["terminal_branch"] not in STEP2_ONLY_TERMINALS


def test_manifest_drift_fail_closed() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        classify_arc2b_slice5_in_vivo_branch,
    )

    inputs = {
        "operational_ok": True,
        "schema_ok": True,
        "evidence_source": "b1_offline_bracket",
        "prereg_law_window_k": 180,
        "prereg_law_decay_num": 1,
        "prereg_law_decay_den": 2,
        "runtime_decay_num": 1,
        "runtime_decay_den": 1,
        "runtime_window_k": 180,
        "recorded_selector_internal_manifest_sha256": "deadbeef",
        "on_disk_selector_manifest_sha256": "cafebabe",
        "manifest_binding_ok": False,
        "log_coverage_ok": True,
        "live_snapshot_present": False,
        "resume_generation": None,
        "offline_bracket_decision": "REAL_DENSITY_EXCEEDS_SUB2",
        "live_carrier_rows": [],
        "eligible_weight_numel": 1000,
        "effective_acc_budget_bpw": 0.4,
        "tolerance_bpw": 0.0,
    }
    result = classify_arc2b_slice5_in_vivo_branch(inputs)
    assert result["terminal_branch"] == "SLICE5_INCONCLUSIVE_INPUT_DRIFT"


def test_step2_live_carrier_pass_and_fail_paths() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        classify_arc2b_slice5_in_vivo_branch,
    )

    base = {
        "operational_ok": True,
        "schema_ok": True,
        "evidence_source": "step2_gpu_live_carrier",
        "prereg_law_window_k": 180,
        "prereg_law_decay_num": 1,
        "prereg_law_decay_den": 2,
        "runtime_decay_num": 1,
        "runtime_decay_den": 2,
        "runtime_window_k": 180,
        "recorded_selector_internal_manifest_sha256": "abc",
        "on_disk_selector_manifest_sha256": "abc",
        "manifest_binding_ok": True,
        "log_coverage_ok": True,
        "live_snapshot_present": True,
        "resume_generation": 0,
        "offline_bracket_decision": None,
        "eligible_weight_numel": 1_000_000,
        "effective_acc_budget_bpw": 0.4,
        "tolerance_bpw": 0.0,
    }
    under_budget_row = {
        "events_bytes": 10,
        "backlog_bytes": 10,
        "hot_exact_bytes": 10,
        "metadata_bytes": 10,
        "live_acc_carrier_bytes_total": 40,
        "live_carrier_bytes_exact": True,
    }
    over_budget_row = {
        "events_bytes": 60_000,
        "backlog_bytes": 0,
        "hot_exact_bytes": 0,
        "metadata_bytes": 0,
        "live_acc_carrier_bytes_total": 60_000,
        "live_carrier_bytes_exact": True,
    }

    pass_result = classify_arc2b_slice5_in_vivo_branch(
        {**base, "live_carrier_rows": [under_budget_row]}
    )
    assert pass_result["terminal_branch"] == "SLICE5_IN_VIVO_LAW_BOUND"

    fail_result = classify_arc2b_slice5_in_vivo_branch(
        {**base, "live_carrier_rows": [over_budget_row]}
    )
    assert fail_result["terminal_branch"] == "D_NEEDS_UPDATE_LAW_REDESIGN"


def test_step2_resume_generation_must_be_exact_zero_for_mechanism_terminal() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        STEP2_ONLY_TERMINALS,
        classify_arc2b_slice5_in_vivo_branch,
    )

    base = {
        "operational_ok": True,
        "schema_ok": True,
        "evidence_source": "step2_gpu_live_carrier",
        "prereg_law_window_k": 180,
        "prereg_law_decay_num": 1,
        "prereg_law_decay_den": 2,
        "runtime_decay_num": 1,
        "runtime_decay_den": 2,
        "runtime_window_k": 180,
        "recorded_selector_internal_manifest_sha256": "abc",
        "on_disk_selector_manifest_sha256": "abc",
        "manifest_binding_ok": True,
        "log_coverage_ok": True,
        "live_snapshot_present": True,
        "offline_bracket_decision": None,
        "eligible_weight_numel": 1_000_000,
        "effective_acc_budget_bpw": 0.4,
        "tolerance_bpw": 0.0,
    }
    under_budget_row = {
        "events_bytes": 10,
        "backlog_bytes": 10,
        "hot_exact_bytes": 10,
        "metadata_bytes": 10,
        "live_acc_carrier_bytes_total": 40,
        "live_carrier_bytes_exact": True,
    }
    over_budget_row = {
        "events_bytes": 60_000,
        "backlog_bytes": 0,
        "hot_exact_bytes": 0,
        "metadata_bytes": 0,
        "live_acc_carrier_bytes_total": 60_000,
        "live_carrier_bytes_exact": True,
    }

    for resume_generation in (None, 1):
        for live_carrier_rows in ([under_budget_row], [over_budget_row]):
            result = classify_arc2b_slice5_in_vivo_branch(
                {
                    **base,
                    "resume_generation": resume_generation,
                    "live_carrier_rows": live_carrier_rows,
                }
            )
            assert result["terminal_branch"] not in STEP2_ONLY_TERMINALS
            assert result["terminal_branch"] == "SLICE5_NO_VERDICT_SCHEMA"


def test_carrier_byte_mapping_and_bpw_formula() -> None:
    from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
        compute_live_acc_bpw,
        live_acc_carrier_bytes_total,
        strict_under_budget,
    )

    snapshot = {
        "events_bytes": 100,
        "backlog_bytes": 200,
        "hot_exact_bytes": 300,
        "metadata_bytes": 16,
        "live_acc_carrier_bytes_total": 616,
        "live_carrier_bytes_exact": True,
    }
    assert live_acc_carrier_bytes_total(snapshot) == 616
    bpw = compute_live_acc_bpw(
        live_acc_carrier_bytes_total=616,
        eligible_weight_numel=12_288,
    )
    assert bpw == pytest.approx(616 * 8 / 12_288)
    assert strict_under_budget(
        observed_bpw=0.399,
        effective_acc_budget_bpw=0.4,
        tolerance_bpw=0.0,
    )
    assert not strict_under_budget(
        observed_bpw=bpw,
        effective_acc_budget_bpw=0.4,
        tolerance_bpw=0.0,
    )


def test_apply_script_self_verify_passes() -> None:
    proc = subprocess.run(
        ["python3", str(APPLY_SCRIPT)],
        cwd=REPO,
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["ok"] is True
    assert payload["deterministic_regen"] is True


def test_json_tool_valid_on_artifacts() -> None:
    for path in (DRAFT, PREFLIGHT):
        proc = subprocess.run(
            ["python3", "-m", "json.tool", str(path)],
            check=False,
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
