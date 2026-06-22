from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.hrm_text_158_r3_ledger_emit_scale_smoke import (
    DEFAULT_LANES_PER_MODULE,
    DEFAULT_MODULES,
    DEFAULT_TOTAL_LANES,
    GEN_C_BANKED_TOTAL_LANE_COUNT,
    GATES,
    MEASURED_PEAK_TOLERANCE_MIB,
    run_scale_smoke,
)


@pytest.mark.parametrize("modules,lanes_per_module", [(DEFAULT_MODULES, DEFAULT_LANES_PER_MODULE)])
def test_r3_ledger_emit_scale_smoke_tensor_wide(
    tmp_path: Path,
    modules: int,
    lanes_per_module: int,
) -> None:
    metrics = run_scale_smoke(
        output_dir=tmp_path / "r3_scale_smoke",
        modules=modules,
        lanes_per_module=lanes_per_module,
    )
    assert metrics["all_gates_pass"] is True
    assert metrics["total_lanes"] == DEFAULT_TOTAL_LANES
    assert metrics["total_lanes"] == GEN_C_BANKED_TOTAL_LANE_COUNT
    assert metrics["scale_matches_gen_c_banked_total_lanes"] is True
    assert metrics["gate_results"]["compact_emit_transient_upper_bound_mib_lte"] is True
    assert metrics["gate_results"]["compact_emit_measured_peak_within_analytic_tolerance"] is True
    assert metrics["gate_results"]["legacy_to_compact_projected_rss_ratio_gte"] is True
    assert (
        metrics["compact_emit_transient_upper_bound_mib"]
        <= GATES["compact_emit_transient_upper_bound_mib_lte"]
    )
    assert (
        metrics["compact_emit_measured_peak_rss_mib"]
        <= metrics["compact_emit_transient_upper_bound_mib"] + MEASURED_PEAK_TOLERANCE_MIB
    )
    assert (
        metrics["legacy_to_compact_projected_rss_ratio"]
        >= GATES["legacy_to_compact_projected_rss_ratio_gte"]
    )
    assert metrics["r3_per_module_payload_row_count"] == modules
    assert "compact_post_emit_rss_delta_mib" in metrics
    assert "compact_emit_measured_peak_rss_mib" in metrics
    metrics_path = tmp_path / "r3_scale_smoke" / "r3_ledger_emit_scale_smoke_metrics.json"
    assert metrics_path.is_file()
    loaded = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == "hrm_text_158_r3_ledger_emit_scale_smoke/v1.2"
    assert "cpython_int_interning_caveat" in loaded["legacy_micro_sample"]
    assert "compact_emit_measured_peak_rss_semantics" in loaded
    assert "compact_post_emit_rss_delta_semantics" in loaded
