from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pytest
import torch

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe
from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
    SUBPHASE_RESOLVE_FRACTION,
    SUBPHASE_UNMAPPED_FRACTION,
    attribute_host_rss_profile,
    attribute_subphase_rss_profile,
    build_attribution_receipt,
    classify_live_vs_resident_diagnostic,
    extract_run_root,
    reconcile_parent_subphases,
    resolve_mechanism_owner,
)


def test_profile_host_rss_enabled_truthy_values(monkeypatch) -> None:
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    assert probe.profile_host_rss_enabled() is True
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "true")
    assert probe.profile_host_rss_enabled() is True
    monkeypatch.delenv(probe.PROFILE_HOST_RSS_ENV, raising=False)
    assert probe.profile_host_rss_enabled() is False


def test_proc_self_resource_snapshot_includes_rss_fields() -> None:
    snap = probe._proc_self_resource_snapshot()
    assert "pid" in snap
    assert "rss_kib" in snap or "rss_error" in snap


def test_make_host_rss_subphase_emitter_default_off(tmp_path: Path) -> None:
    progress = probe.PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        host_rss_profile_path=None,
    )
    assert progress.make_host_rss_subphase_emitter(step=1) is None


def test_phase_progress_emits_host_rss_profile_marks(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    profile_path = tmp_path / probe.HOST_RSS_PROFILE_JSONL_NAME
    progress = probe.PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        host_rss_profile_path=profile_path,
    )
    with progress.phase("sparse_cap_apply", step=1):
        pass
    rows = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["event"] for row in rows} == {"enter", "exit"}
    assert all(row["phase"] == "sparse_cap_apply" for row in rows)
    assert all("sub_phase" not in row for row in rows)
    assert all("resource_snapshot" in row for row in rows)


def test_subphase_emitter_emits_paired_marks(tmp_path: Path) -> None:
    profile_path = tmp_path / probe.HOST_RSS_PROFILE_JSONL_NAME
    progress = probe.PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        host_rss_profile_path=profile_path,
    )
    emit = progress.make_host_rss_subphase_emitter(step=1)
    assert emit is not None
    emit(
        "enter",
        sub_phase_id="C1_vote_plan_build",
        optimizer_step_index=1,
        allocation_dims={"n_states": 32},
    )
    emit(
        "exit",
        sub_phase_id="C1_vote_plan_build",
        optimizer_step_index=1,
        allocation_dims={"n_states": 32, "total_q_numel": 1000},
    )
    rows = [
        json.loads(line)
        for line in profile_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rows) == 2
    assert {row["event"] for row in rows} == {"enter", "exit"}
    assert all(row["sub_phase"] == "C1_vote_plan_build" for row in rows)
    assert rows[0]["schema"].endswith("/v2")


def test_phase_progress_skips_non_profiled_phases(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    profile_path = tmp_path / probe.HOST_RSS_PROFILE_JSONL_NAME
    progress = probe.PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        host_rss_profile_path=profile_path,
    )
    with progress.phase("load"):
        pass
    assert not profile_path.exists()


def test_attribute_host_rss_profile_picks_sparse_cap_delta() -> None:
    marks = [
        {
            "phase": "step_forward_backward",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1000 * 1024},
        },
        {
            "phase": "step_forward_backward",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 1100 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1100 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 4100 * 1024},
        },
    ]
    result = attribute_host_rss_profile(
        marks,
        wall_totals={"sparse_cap_apply": 90.0, "step_forward_backward": 1.0},
    )
    assert result["dominant_rss_owner"]["phase"] == "sparse_cap_apply"
    assert result["dominant_phase_owner"] == "sparse_cap_apply"
    assert result["culprit_class"] is None
    assert result["culprit_class_status"] == "UNRESOLVED"
    assert result["phase_class_candidate_hint"] == "A"
    assert result["falsified_mechanism"] == "A"
    assert result["next_candidate_class"] == "C"
    assert result["mechanism_owner_status"] == "UNRESOLVED_SUBPHASE_REQUIRED"
    assert result["dominant_wall_owner"]["phase"] == "sparse_cap_apply"


def test_parent_only_marks_do_not_resolve_mechanism_owner() -> None:
    marks = [
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1000 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 9000 * 1024},
        },
    ]
    result = attribute_host_rss_profile(marks)
    assert result["mechanism_owner_status"] == "UNRESOLVED_SUBPHASE_REQUIRED"
    assert result["culprit_class"] is None
    assert result["dominant_subphase_owner"] is None


def test_subphase_reconciliation_resolved_when_dominant_and_plausible() -> None:
    parent_delta = 8.0
    sub_marks = [
        {
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 2000 * 1024},
        },
        {
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 8800 * 1024},
            "allocation_dims": {
                "expected_raw_bytes_q_tensors": int(7.5 * (1024**3)),
                "dtype_q": "torch.int8",
                "n_q_held": 32,
            },
        },
        {
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C2_cap_input_assembly",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 8800 * 1024},
        },
        {
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C2_cap_input_assembly",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 8900 * 1024},
            "allocation_dims": {"expected_raw_bytes_shape_stub": 50_000_000},
        },
    ]
    parent_marks = [
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1000 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 9000 * 1024},
        },
    ]
    result = attribute_host_rss_profile(parent_marks + sub_marks)
    assert result["dominant_subphase_owner"] == "C4_gpu_cap_apply_sync"
    assert result["mechanism_owner_status"] == "RESOLVED"
    assert result["culprit_class"] == "C"
    assert result["reconciliation"]["unmapped_remainder_rss_gib"] is not None


def test_subphase_reconciliation_unmapped_when_remainder_too_large() -> None:
    subphase = attribute_subphase_rss_profile(
        [
            {
                "sub_phase": "C1_vote_plan_build",
                "step": 1,
                "event": "enter",
                "resource_snapshot": {"rss_kib": 1000 * 1024},
            },
            {
                "sub_phase": "C1_vote_plan_build",
                "step": 1,
                "event": "exit",
                "resource_snapshot": {"rss_kib": 1500 * 1024},
                "allocation_dims": {"expected_raw_bytes_shape_stub": 1_000},
            },
        ]
    )
    recon = reconcile_parent_subphases(
        parent_delta_rss_gib=8.0,
        subphase_attribution=subphase,
    )
    assert recon["reconciliation_status"] == "UNMAPPED_OR_UNRESOLVED"
    mechanism = resolve_mechanism_owner(
        parent_delta_rss_gib=8.0,
        subphase_attribution=subphase,
        reconciliation=recon,
    )
    assert mechanism["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"
    assert mechanism["culprit_class"] is None


def test_build_attribution_receipt_separates_phase_and_mechanism_owner(
    tmp_path: Path,
) -> None:
    profile_path = tmp_path / "host_rss_profile.jsonl"
    marks = [
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1100 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 4100 * 1024},
        },
    ]
    profile_path.write_text(
        "\n".join(json.dumps(row) for row in marks) + "\n",
        encoding="utf-8",
    )
    receipt = build_attribution_receipt(run_root=tmp_path, profile_path=profile_path)
    assert receipt["schema"].endswith("/v4")
    assert receipt["dominant_phase_owner"] == "sparse_cap_apply"
    assert receipt["rss_phase_owner_status"] == "RESOLVED"
    assert receipt["mechanism_owner_status"] == "UNRESOLVED_SUBPHASE_REQUIRED"
    assert receipt["culprit_class"] is None
    assert receipt["falsified_mechanism"] == "A"
    assert receipt["next_candidate_class"] == "C"


def test_extract_run_root_from_v6i_abort_fixture() -> None:
    fixture_root = (
        Path(__file__).resolve().parents[3]
        / "calm/llm_computer/tests/fixtures/slice5_v6i_oom_2189e7202a_extract"
    )
    if not fixture_root.is_dir():
        return
    report = extract_run_root(fixture_root)
    assert report["schema"].endswith("/v1")
    assert len(report["arms"]) == 2


def test_build_attribution_receipt_unresolved_without_profile(tmp_path: Path) -> None:
    receipt = build_attribution_receipt(
        run_root=tmp_path,
        profile_path=tmp_path / "missing.jsonl",
    )
    assert receipt["rss_owner_status"] == "UNRESOLVED"
    assert receipt["rss_phase_owner_status"] == "UNRESOLVED"
    assert receipt["mechanism_owner_status"] == "UNRESOLVED_SUBPHASE_REQUIRED"
    assert receipt["dominant_rss_owner"] is None


def test_profile_host_rss_live_resident_default_off(monkeypatch) -> None:
    monkeypatch.delenv(probe.PROFILE_HOST_RSS_LIVE_RESIDENT_ENV, raising=False)
    assert probe.profile_host_rss_live_resident_enabled() is False
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_LIVE_RESIDENT_ENV, "1")
    assert probe.profile_host_rss_live_resident_enabled() is True


def test_classify_live_vs_resident_allocator_retention() -> None:
    marks = [
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_pre_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 10_000 * 1024},
        },
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_post_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 7_500 * 1024},
        },
    ]
    result = classify_live_vs_resident_diagnostic(marks)
    assert result["live_vs_resident_classification"] == "ALLOCATOR_RETENTION"
    assert result["next_fix_type"] == "allocator_trim"
    assert result["trim_delta_rss_gib"] == pytest.approx(2.4414, rel=1e-3)
    assert result["measurement_perturbed"] is True


def test_classify_live_vs_resident_live_allocation() -> None:
    marks = [
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_pre_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 10_000 * 1024},
        },
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_post_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 9_900 * 1024},
        },
    ]
    result = classify_live_vs_resident_diagnostic(marks)
    assert result["live_vs_resident_classification"] == "LIVE_ALLOCATION"
    assert result["next_fix_type"] == "materialization_shape"


def test_attribute_host_rss_profile_merges_diagnostic_verdict() -> None:
    primary = [
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "enter",
            "resource_snapshot": {"rss_kib": 1000 * 1024},
        },
        {
            "phase": "sparse_cap_apply",
            "step": 1,
            "event": "exit",
            "resource_snapshot": {"rss_kib": 9000 * 1024},
        },
    ]
    diagnostic = [
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_pre_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 9000 * 1024},
        },
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "live_resident_post_trim",
            "measurement_perturbed": True,
            "resource_snapshot": {"rss_kib": 6500 * 1024},
        },
    ]
    result = attribute_host_rss_profile(primary, diagnostic_marks=diagnostic)
    assert result["live_vs_resident_classification"] == "ALLOCATOR_RETENTION"
    assert result["live_vs_resident_diagnostic"]["trim_delta_rss_gib"] == pytest.approx(
        2.4414, rel=1e-3
    )


def test_profile_torch_cpu_census_default_off(monkeypatch) -> None:
    monkeypatch.delenv(probe.PROFILE_HOST_RSS_ENV, raising=False)
    monkeypatch.delenv(probe.PROFILE_TORCH_CPU_CENSUS_ENV, raising=False)
    from calm.hrm_text_158.native_full_stack.host_torch_census import (
        profile_torch_cpu_census_enabled,
    )

    assert profile_torch_cpu_census_enabled() is False
    monkeypatch.setenv(probe.PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(probe.PROFILE_TORCH_CPU_CENSUS_ENV, "1")
    assert profile_torch_cpu_census_enabled() is True


def test_torch_cpu_census_dedupes_shared_storage_views() -> None:
    from calm.hrm_text_158.native_full_stack.host_torch_census import (
        torch_cpu_tensor_census,
    )

    base = torch.zeros(1024, dtype=torch.float32)
    view_a = base[:512]
    view_b = base[512:]
    census = torch_cpu_tensor_census(top_k=5)
    assert census["n_cpu_tensors"] >= 2
    assert census["logical_tensor_bytes"] >= int(base.numel() * 4)
    assert census["unique_storage_bytes"] <= census["logical_tensor_bytes"]
    del view_a, view_b, base


def test_attribute_torch_census_resolved_when_unique_storage_reconciles() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_torch_census_profile,
    )

    def _mark(event: str, unique_bytes: int, group: dict[str, Any]) -> dict[str, Any]:
        return {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": event,
            "torch_census": {
                "unique_storage_bytes": unique_bytes,
                "top_groups": [group],
            },
        }

    group = {
        "device": "cpu",
        "dtype": "torch.float32",
        "shape": [1000, 1000],
        "unique_storage_bytes": 8_000_000_000,
        "unique_storage_count": 32,
        "logical_tensor_bytes": 8_000_000_000,
        "tensor_count": 32,
    }
    marks = [
        _mark("census_C3_exit", 1_000_000_000, group),
        _mark("census_C4_enter", 1_500_000_000, group),
        _mark("census_C4_exit", 9_000_000_000, group),
    ]
    result = attribute_torch_census_profile(marks, c4_delta_rss_gib=7.5)
    assert result["dimensional_reconciliation_unique_storage"]["status"] == "PASS"
    assert result["mechanism_owner_status"] == "RESOLVED"
    assert result["culprit_class"] == "C"
    assert result["dominant_allocation"]["dtype"] == "torch.float32"


def test_attribute_torch_census_unmapped_when_reconciliation_fails() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_torch_census_profile,
    )

    marks = [
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C3_exit",
            "torch_census": {"unique_storage_bytes": 1000, "top_groups": []},
        },
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C4_enter",
            "torch_census": {"unique_storage_bytes": 2000, "top_groups": []},
        },
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C4_exit",
            "torch_census": {"unique_storage_bytes": 3000, "top_groups": []},
        },
    ]
    result = attribute_torch_census_profile(marks, c4_delta_rss_gib=7.5)
    assert result["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"
    assert result["culprit_class"] is None
    assert result["next_probe_route"] == "allocator_native_smaps_anonymous"


def test_census_boundary_marks_merge_handoff_and_loop_bytes() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_torch_census_profile,
    )

    marks = [
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C3_exit",
            "torch_census": {"unique_storage_bytes": 1_000_000_000, "top_groups": []},
        },
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C4_enter",
            "torch_census": {"unique_storage_bytes": 6_000_000_000, "top_groups": []},
        },
        {
            "schema": probe.PROFILE_HOST_RSS_CENSUS_SCHEMA,
            "event": "census_C4_exit",
            "torch_census": {"unique_storage_bytes": 6_500_000_000, "top_groups": []},
        },
    ]
    result = attribute_torch_census_profile(marks, c4_delta_rss_gib=7.5)
    boundaries = result["census_boundaries"]
    assert boundaries["handoff_unique_storage_bytes"] == 5_000_000_000
    assert boundaries["loop_unique_storage_bytes"] == 500_000_000
    assert result["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"

