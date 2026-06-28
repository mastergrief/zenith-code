from __future__ import annotations

import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.d_recompute_window_receipt_compact import (
    COMPACT_SCHEMA_VERSION,
    D_DIAGNOSTIC_COMPACT_TENSOR_STATS_DROP_KEYS,
    D_DIAGNOSTIC_COMPACT_TENSOR_STATS_KEEP_KEYS,
    D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE,
    DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX,
    DEFAULT_TARGET_TENSOR_STATS_BYTES_PER_STEP,
    compact_d_diagnostic_step_result,
    estimate_step_reports_tensor_stats_bytes,
    extrapolate_h100_byte_projections,
    should_apply_d_diagnostic_receipt_compaction,
)
from calm.hrm_text_158.native_full_stack.s3bb_headroom_telemetry import (
    RECEIPT_EMIT_PROFILE_SLIM,
)


def _bulky_tensor_stats(module_name: str) -> dict:
    return {
        "q_changed_count": 3,
        "replay_ce_veto_count": 0,
        "post_veto_applied_flip_count": 1,
        "applied_flat_indices_hash16": "abcd1234",
        "top8_flat_indices_hash16": "efgh5678",
        "top64_flat_indices_hash16": "ijkl9012",
        "applied_selection_score_p50": 0.12,
        "applied_selection_score_p95": 0.88,
        "applied_selection_score_semantics": "local_loss_delta_margin",
        "cap_window_jaccard_vs_prior_step": 0.5,
        "cap_window_audit_non_authoritative": True,
        "applied_indices": list(range(4096)),
        "pre_veto_selected_indices": list(range(2048)),
        "post_veto_applied_indices": list(range(1024)),
        "replay_ce_veto_indices": [1, 2, 3],
        "applied_selection_scores": [float(i) for i in range(512)],
        "module": module_name,
    }


def _build_step_reports(*, steps: int, modules: int) -> dict[str, dict]:
    reports: dict[str, dict] = {}
    for step in range(1, steps + 1):
        tensor_stats = {
            f"model.layers.{module}.attn.o_proj": _bulky_tensor_stats(str(module))
            for module in range(modules)
        }
        reports[str(step)] = {
            "loss": 1.0,
            "step_result": {
                "schema": "hrm_text_158_c2p0_bounded_delta_step_result/v0.compact",
                "tensor_stats": tensor_stats,
                "global_summary": {"q_changed_count": 3},
            },
        }
    return reports


def test_should_apply_compaction_gate() -> None:
    assert should_apply_d_diagnostic_receipt_compaction(
        phase=D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE,
        receipt_emit_profile=RECEIPT_EMIT_PROFILE_SLIM,
        d_diagnostic_compact_step_reports=True,
    )
    assert not should_apply_d_diagnostic_receipt_compaction(
        phase=D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE,
        receipt_emit_profile=RECEIPT_EMIT_PROFILE_SLIM,
        d_diagnostic_compact_step_reports=False,
    )
    assert not should_apply_d_diagnostic_receipt_compaction(
        phase="other-phase",
        receipt_emit_profile=RECEIPT_EMIT_PROFILE_SLIM,
        d_diagnostic_compact_step_reports=True,
    )


def test_compact_drops_raw_arrays_and_keeps_hashes() -> None:
    step_result = _build_step_reports(steps=1, modules=2)["1"]["step_result"]
    compacted = compact_d_diagnostic_step_result(step_result)
    stats = compacted["tensor_stats"]["model.layers.0.attn.o_proj"]
    for key in D_DIAGNOSTIC_COMPACT_TENSOR_STATS_DROP_KEYS:
        assert key not in stats
    for key in D_DIAGNOSTIC_COMPACT_TENSOR_STATS_KEEP_KEYS:
        assert key in stats
    assert compacted["d_diagnostic_receipt_compact"]["schema_version"] == COMPACT_SCHEMA_VERSION


def test_compacted_receipt_tensor_stats_under_per_step_cap() -> None:
    reports = _build_step_reports(steps=10, modules=32)
    compacted_reports = {
        step: {
            **report,
            "step_result": compact_d_diagnostic_step_result(report["step_result"]),
        }
        for step, report in reports.items()
    }
    raw_bytes = estimate_step_reports_tensor_stats_bytes(reports)
    compact_bytes = estimate_step_reports_tensor_stats_bytes(compacted_reports)
    assert compact_bytes < raw_bytes
    per_step = compact_bytes / 10
    assert per_step <= DEFAULT_TARGET_TENSOR_STATS_BYTES_PER_STEP

    receipt = {
        "schema": "probe-receipt-fixture",
        "step_reports": compacted_reports,
        "steps_completed": 10,
    }
    receipt_bytes = len(json.dumps(receipt, separators=(",", ":"), sort_keys=True).encode("utf-8"))
    projection = extrapolate_h100_byte_projections(
        receipt_bytes=receipt_bytes,
        smoke_steps=10,
        recompute_log_bytes=100_000,
    )
    assert projection["extrapolated_h100_receipt_bytes"] <= DEFAULT_EXTRAPOLATED_H100_RECEIPT_BYTES_MAX
    assert projection["launch_allowed"] is True


def test_extrapolation_math_flags_oversize_smoke() -> None:
    projection = extrapolate_h100_byte_projections(
        receipt_bytes=8_000_000,
        smoke_steps=5,
        recompute_log_bytes=2_000_000,
    )
    assert projection["receipt_bytes_per_step"] == 1_600_000.0
    assert projection["extrapolated_h100_receipt_bytes"] == 160_000_000
    assert projection["pass"]["extrapolated_h100_receipt_bytes"] is False
    assert projection["launch_allowed"] is False


def test_scale_smoke_receipt_harness_on_fixture(tmp_path: Path) -> None:
    run_root = tmp_path / "run"
    diagnostic = run_root / "d_recompute_window_diagnostic"
    diagnostic.mkdir(parents=True)
    reports = _build_step_reports(steps=5, modules=32)
    compacted_reports = {
        step: {
            **report,
            "step_result": compact_d_diagnostic_step_result(report["step_result"]),
        }
        for step, report in reports.items()
    }
    receipt = {
        "schema": "probe-receipt-fixture",
        "step_reports": compacted_reports,
        "steps_completed": 5,
    }
    receipt_path = diagnostic / "receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    log_path = diagnostic / "recompute_window_log.jsonl"
    log_path.write_text('{"step":1,"state_key":"k"}\n' * 50, encoding="utf-8")
    (run_root / "driver_summary.json").write_text(
        json.dumps({"phase": D_RECOMPUTE_WINDOW_FEASIBILITY_PHASE}),
        encoding="utf-8",
    )

    import subprocess
    from unittest.mock import patch

    from scripts.hrm_text_158_d_recompute_scale_smoke_receipt import (
        DEFAULT_MIN_FREE_MEMORY_BYTES,
        build_scale_smoke_receipt,
    )

    def _fixture_nvidia_smi_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0, "0, 2048, 8192\n", "")

    process_dead_proof = {"pgrep_exit_code": 1, "matches": [], "process_dead": True}
    with patch(
        "scripts.hrm_text_158_d_recompute_scale_smoke_receipt._gpu_process_dead",
        return_value=process_dead_proof,
    ):
        smoke_receipt = build_scale_smoke_receipt(
            run_root=run_root,
            smoke_steps=5,
            min_free_memory_bytes=DEFAULT_MIN_FREE_MEMORY_BYTES,
            nvidia_smi_runner=_fixture_nvidia_smi_runner,
        )
    assert smoke_receipt["pass"] is True
    assert smoke_receipt["byte_projection"]["launch_allowed"] is True
