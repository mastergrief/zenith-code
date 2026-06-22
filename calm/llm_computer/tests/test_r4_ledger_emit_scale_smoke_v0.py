from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hrm_text_158_r4_ledger_emit_scale_smoke import (
    DEFAULT_LANES_PER_MODULE,
    DEFAULT_MODULES,
    DEFAULT_TOTAL_LANES,
    GEN_C_BANKED_TOTAL_LANE_COUNT,
    GATES,
    MEASURED_PEAK_TOLERANCE_MIB,
    R4_ACC_BPW_TARGET,
    R4_ACC_PAYLOAD_MIB_TARGET,
    R4_INCLUSIVE_BPW_CEILING,
    R4_INCLUSIVE_BPW_NEAR,
    R4_PAYLOAD_MIB_TOLERANCE,
    R4_Q_BPW_TARGET,
    R4_Q_PAYLOAD_MIB_TARGET,
    run_scale_smoke,
)


@pytest.mark.parametrize("modules,lanes_per_module", [(DEFAULT_MODULES, DEFAULT_LANES_PER_MODULE)])
def test_r4_ledger_emit_scale_smoke_tensor_wide(
    tmp_path: Path,
    modules: int,
    lanes_per_module: int,
) -> None:
    metrics = run_scale_smoke(
        output_dir=tmp_path / "r4_scale_smoke",
        modules=modules,
        lanes_per_module=lanes_per_module,
    )
    assert metrics["total_lanes"] == DEFAULT_TOTAL_LANES
    assert metrics["total_lanes"] == GEN_C_BANKED_TOTAL_LANE_COUNT
    assert metrics["scale_matches_gen_c_banked_total_lanes"] is True
    assert metrics["memory_gates_pass"] is True
    assert metrics["all_gates_pass"] is True
    assert metrics["gate_results"]["compact_emit_wall_seconds_lte"] is True
    assert metrics["full_scale_wall_seconds"] <= GATES["compact_emit_wall_seconds_lte"]
    assert metrics["r4_1_checkpoint_serialization_cleared"] is True
    assert metrics["r4_per_module_q_row_count"] == modules
    assert metrics["r4_per_module_acc_row_count"] == modules
    assert metrics["r4_ledger_pass"] is True
    assert metrics["raw_byte_list_hits"] == []

    component = metrics["emit_component_model"]
    assert component["retained_payload_only_mib"] < component["emit_only_transient_upper_bound_mib"]
    assert (
        metrics["compact_emit_measured_peak_rss_mib"]
        <= float(GATES["emit_only_transient_upper_bound_mib_lte"])
        or component["emit_only_transient_upper_bound_mib"]
        <= float(GATES["emit_only_transient_upper_bound_mib_lte"])
    )
    assert (
        metrics["compact_emit_measured_peak_rss_mib"]
        <= component["emit_only_transient_upper_bound_mib"] + MEASURED_PEAK_TOLERANCE_MIB
    )
    assert (
        metrics["legacy_to_compact_projected_rss_ratio"]
        >= GATES["legacy_to_compact_projected_rss_ratio_gte"]
    )

    q_bpw = float(metrics["r4_q_physical_bits_per_weight"])
    acc_bpw = float(metrics["r4_acc_physical_bits_per_weight"])
    inclusive = float(metrics["r4_checkpoint_inclusive_physical_bits_per_weight"])
    assert abs(q_bpw - R4_Q_BPW_TARGET) <= 0.25
    assert abs(acc_bpw - R4_ACC_BPW_TARGET) <= 0.25
    assert inclusive <= R4_INCLUSIVE_BPW_CEILING
    assert abs(inclusive - R4_INCLUSIVE_BPW_NEAR) <= 0.5

    assert abs(float(metrics["r4_actual_q_payload_mib"]) - R4_Q_PAYLOAD_MIB_TARGET) <= R4_PAYLOAD_MIB_TOLERANCE
    assert abs(float(metrics["r4_actual_acc_payload_mib"]) - R4_ACC_PAYLOAD_MIB_TARGET) <= R4_PAYLOAD_MIB_TOLERANCE

    assert metrics["compact_receipt_file_size_mib"] <= GATES["compact_receipt_file_size_mib_lte"]
    assert metrics["gate_results"]["zero_raw_byte_lists_in_receipt"] is True
    assert metrics["gate_results"]["primary_memory_gate_pass"] is True
    assert metrics["gate_results"]["measured_peak_hard_fail_over_512"] is False

    metrics_path = tmp_path / "r4_scale_smoke" / "r4_ledger_emit_scale_smoke_metrics.json"
    receipt_path = tmp_path / "r4_scale_smoke" / "r4_compact_receipt.json"
    assert metrics_path.is_file()
    assert receipt_path.is_file()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "hrm_text_158_r4_ledger_emit_scale_smoke/v1"
    assert "emit_component_model" in loaded
    assert "compact_emit_measured_peak_rss_semantics" in loaded
