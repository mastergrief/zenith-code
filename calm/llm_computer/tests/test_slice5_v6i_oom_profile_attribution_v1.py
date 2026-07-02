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
    assert receipt["schema"].endswith("/v12")
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


def test_profile_allocator_native_default_off(monkeypatch) -> None:
    monkeypatch.delenv(probe.PROFILE_ALLOCATOR_NATIVE_ENV, raising=False)
    assert probe.profile_allocator_native_enabled() is False


def _allocator_mark(
    event: str,
    *,
    rss_kib: int,
    anonymous_kb: int | None = None,
    uordblks: int | None = None,
    host_active: int | None = None,
    state_index: int | None = None,
    state_bucket: int | None = None,
) -> dict[str, Any]:
    probe_payload: dict[str, Any] = {
        "smaps_rollup": {"anonymous_kb": anonymous_kb},
        "mallinfo2": {"uordblks_bytes": uordblks},
        "cuda_allocator": {
            "host_memory_stats_available": host_active is not None,
            "cuda_gpu_allocated_bytes": 1_000_000_000,
            "cuda_gpu_stats_role": "negative_control_not_host_rss_contributor",
        },
    }
    if host_active is not None:
        probe_payload["cuda_allocator"]["cuda_host_active_bytes_all_current"] = host_active
    mark: dict[str, Any] = {
        "schema": probe.PROFILE_HOST_RSS_ALLOCATOR_SCHEMA,
        "event": event,
        "resource_snapshot": {"rss_kib": rss_kib},
        "allocator_probe": probe_payload,
        "measurement_perturbed": True,
    }
    if state_index is not None:
        mark["state_index"] = state_index
    if state_bucket is not None:
        mark["state_bucket"] = state_bucket
    return mark


def test_allocator_dominance_resolves_cuda_host() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000, anonymous_kb=1000, host_active=100),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=3_000_000,
            anonymous_kb=1100,
            host_active=900_000_000,
            state_index=3,
            state_bucket=0,
        ),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=4_000_000,
            anonymous_kb=1200,
            host_active=1_800_000_000,
            state_index=7,
            state_bucket=1,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=7.5)
    assert result["mechanism_owner_status"] == "RESOLVED"
    assert result["allocation_source"] == "cuda_host_caching_allocator"
    assert result["overlap_accounting"]["remainder_status"] == "overlap_unknown"


def test_allocator_unmapped_when_no_dominance() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000, anonymous_kb=1000),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=2_100_000,
            anonymous_kb=1050,
            state_index=3,
            state_bucket=0,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=0.1)
    assert result["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"


def test_allocator_site_tier_b_no_overclaim_from_bucket_only() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=3_000_000,
            host_active=500_000_000,
            state_index=3,
            state_bucket=0,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=1.0)
    assert result["call_site_status"] == "UNRESOLVED"


def test_host_cache_empty_diagnostic_classification() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        classify_host_cache_empty_diagnostic,
    )

    post = _allocator_mark("allocator_host_cache_post_empty", rss_kib=2_500_000)
    post["allocation_dims"] = {"host_cache_diag": {"status": "ok"}}
    marks = [
        _allocator_mark("allocator_host_cache_pre_empty", rss_kib=4_000_000),
        post,
    ]
    result = classify_host_cache_empty_diagnostic(marks)
    assert result["classification"] == "CUDA_HOST_CACHE_CONFIRMED"
    assert result["empty_cache_status"] == "ok"


def _allocator_site_mark(
    site_id: str,
    suffix: str,
    *,
    rss_kib: int,
    state_index: int,
) -> dict[str, Any]:
    return {
        "schema": probe.PROFILE_HOST_RSS_ALLOCATOR_SITE_SCHEMA,
        "event": f"allocator_site_{site_id}_{suffix}",
        "site_id": site_id,
        "origin_file": "sparse_cap_gpu_seam_adapter.py",
        "origin_line": 522,
        "state_index": state_index,
        "resource_snapshot": {"rss_kib": rss_kib},
        "allocator_probe": {},
        "measurement_perturbed": True,
    }


def test_empty_cache_status_preserved_from_allocation_dims() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        classify_host_cache_empty_diagnostic,
    )

    post = _allocator_mark("allocator_host_cache_post_empty", rss_kib=3_000_000)
    post["allocation_dims"] = {"host_cache_diag": {"status": "ok", "trim_delta_rss_kib": 2}}
    marks = [
        _allocator_mark("allocator_host_cache_pre_empty", rss_kib=3_010_000),
        post,
    ]
    result = classify_host_cache_empty_diagnostic(marks)
    assert result["empty_cache_status"] == "ok"
    assert result["classification"] == "LIVE_RESIDENT"
    assert result["cache_falsified"] is True


def test_host_cache_live_resident_requires_successful_empty_cache() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        classify_host_cache_empty_diagnostic,
    )

    post = _allocator_mark("allocator_host_cache_post_empty", rss_kib=3_000_000)
    post["allocation_dims"] = {"host_cache_diag": {"status": "RuntimeError: unavailable"}}
    marks = [
        _allocator_mark("allocator_host_cache_pre_empty", rss_kib=3_001_000),
        post,
    ]
    result = classify_host_cache_empty_diagnostic(marks)
    assert result["classification"] == "INCONCLUSIVE"
    assert result.get("cache_falsified") is False


def test_state_index_zero_site_parsing() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000),
        _allocator_site_mark("C4.S1", "pre", rss_kib=2_100_000, state_index=0),
        _allocator_site_mark("C4.S1", "post", rss_kib=2_200_000, state_index=0),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=3_000_000,
            anonymous_kb=1200,
            state_index=3,
            state_bucket=0,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=1.0)
    assert len(result["intra_state_site_deltas"]) == 1
    assert result["intra_state_site_deltas"][0]["site_id"] == "C4.S1"


def test_first_bucket_states_in_bucket_is_four() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000, anonymous_kb=1000),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=3_000_000,
            anonymous_kb=1100,
            state_index=3,
            state_bucket=0,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=1.0)
    assert result["allocator_bucket_series"][0]["states_in_bucket"] == 4


def test_tier_c_anonymous_only_no_call_site_overclaim() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_native_profile,
    )

    marks = [
        _allocator_mark("allocator_C4_enter", rss_kib=2_000_000, anonymous_kb=1000),
        _allocator_mark(
            "allocator_C4_after_state",
            rss_kib=3_000_000,
            anonymous_kb=2000,
            state_index=3,
            state_bucket=0,
        ),
    ]
    result = attribute_allocator_native_profile(marks, c4_delta_rss_gib=1.0)
    assert result["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"
    assert result["call_site_status"] == "UNRESOLVED"


def _alloc_hook_mark(event: str, *, rss_kib: int, stats: dict | None = None) -> dict:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA,
    )

    return {
        "schema": PROFILE_HOST_RSS_ALLOC_HOOK_SCHEMA,
        "event": event,
        "resource_snapshot": {"rss_kib": rss_kib},
        "alloc_hook_stats": stats or {},
        "allocator_probe": {"vma_entries": []},
    }


def test_profile_alloc_hook_default_off(monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack import host_alloc_hook_probe as hook_probe

    monkeypatch.delenv(hook_probe.PROFILE_ALLOC_HOOK_ENV, raising=False)
    monkeypatch.delenv("HRM_TEXT_158_PROFILE_HOST_RSS", raising=False)
    assert hook_probe.profile_alloc_hook_enabled() is False


def test_alloc_hook_table_overflow_inconclusive() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_alloc_hook_profile,
    )

    stats = {
        "window_net_bytes": 8_000_000_000,
        "lost_owner_count": 1,
        "table_overflow_count": 1,
        "unknown_free_bytes_bounded": True,
        "unknown_free_unmeasured_count": 0,
        "top_sites": [{"owner_frame": "0xabc", "net_bytes": 8_000_000_000}],
    }
    marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_000_000),
        _alloc_hook_mark("alloc_hook_C4_exit", rss_kib=10_000_000, stats=stats),
    ]
    result = attribute_alloc_hook_profile(marks, c4_delta_rss_gib=7.5)
    assert result["mechanism_owner_status"] == "INCONCLUSIVE"


def test_alloc_hook_unknown_unmeasured_inconclusive() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_alloc_hook_profile,
    )

    stats = {
        "window_net_bytes": 1_000_000,
        "lost_owner_count": 0,
        "table_overflow_count": 0,
        "unknown_free_bytes_bounded": False,
        "unknown_free_unmeasured_count": 2,
        "top_sites": [],
    }
    marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_000_000),
        _alloc_hook_mark("alloc_hook_C4_exit", rss_kib=3_000_000, stats=stats),
    ]
    result = attribute_alloc_hook_profile(marks, c4_delta_rss_gib=1.0)
    assert result["mechanism_owner_status"] == "INCONCLUSIVE"


def test_vma_diff_existing_region() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import diff_vma_entries

    before = [{"start": 100, "end": 200, "rss_kb": 1000, "anonymous_kb": 500, "name": "[heap]"}]
    after = [{"start": 100, "end": 200, "rss_kb": 1300, "anonymous_kb": 800, "name": "[heap]"}]
    result = diff_vma_entries(before, after)
    assert result["top_positive_vma_deltas"][0]["delta_rss_kb"] == 300
    assert result["top_positive_vma_deltas"][0]["is_new_vma"] is False


def test_read_vma_entries_parses_smaps_detail_lines(tmp_path, monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import read_vma_entries

    smaps_text = """00400000-00401000 r-xp 00000000 08:01 1234 /usr/lib/libc.so.6
Size:                  4 kB
Rss:                   4 kB
Private_Dirty:         0 kB
Anonymous:             0 kB
Referenced:            4 kB
7f0000000000-7f0000100000 rw-p 00000000 00:00 0 
Size:               1024 kB
Rss:                 512 kB
Private_Dirty:       512 kB
Anonymous:           512 kB
Referenced:          512 kB
"""
    smaps_path = tmp_path / "smaps"
    smaps_path.write_text(smaps_text, encoding="utf-8")
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.Path",
        lambda p: smaps_path if str(p) == "/proc/self/smaps" else Path(p),
    )
    entries = read_vma_entries()
    assert len(entries) == 2
    assert entries[0]["name"] == "/usr/lib/libc.so.6"
    assert entries[0]["size_kb"] == 4
    assert entries[0]["rss_kb"] == 4
    assert entries[1]["rss_kb"] == 512
    assert entries[1]["anonymous_kb"] == 512


def test_alloc_hook_lock_contention_inconclusive() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_alloc_hook_profile,
    )

    stats = {
        "top_sites": [{"owner_frame": "0x1000", "net_bytes": 8_000_000_000}],
        "window_net_bytes": 8_000_000_000,
        "lost_owner_count": 0,
        "table_overflow_count": 0,
        "lock_contention_drop_count": 42,
        "unknown_free_bytes_bounded": True,
        "unknown_free_unmeasured_count": 0,
    }
    marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_000_000),
        _alloc_hook_mark("alloc_hook_C4_exit", rss_kib=10_000_000, stats=stats),
    ]
    result = attribute_alloc_hook_profile(marks, c4_delta_rss_gib=7.5)
    assert result["mechanism_owner_status"] == "INCONCLUSIVE"
    assert result["lock_contention_drop_count"] == 42


def test_read_vma_entries_rejects_size_detail_as_header(tmp_path, monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import read_vma_entries

    smaps_text = """00400000-00401000 r-xp 00000000 08:01 1234 /usr/lib/libc.so.6
Size:                  4 kB
Rss:                   4 kB
Size:               1024 kB
Rss:                 512 kB
7f0000000000-7f0000100000 rw-p 00000000 00:00 0 
Size:               1024 kB
"""
    smaps_path = tmp_path / "smaps"
    smaps_path.write_text(smaps_text, encoding="utf-8")
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.Path",
        lambda p: smaps_path if str(p) == "/proc/self/smaps" else Path(p),
    )
    entries = read_vma_entries()
    assert len(entries) == 2
    assert entries[0]["name"] == "/usr/lib/libc.so.6"
    assert entries[1]["rss_kb"] == 0


def test_positive_control_miss_hook_failure(tmp_path) -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import run_positive_control

    result = run_positive_control(tmp_path / "missing_stats.json")
    assert result["status"] == "HOOK_FAILURE"


def test_alloc_hook_unmapped_anonymous_mmap_status() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_alloc_hook_profile,
    )

    stats = {
        "window_net_bytes": 4_469_030_912,
        "prefault_done": 1,
        "lost_owner_count": 0,
        "table_overflow_count": 0,
        "lock_contention_drop_count": 0,
        "unknown_free_bytes_bounded": True,
        "unknown_free_unmeasured_count": 0,
        "top_sites": [{"owner_frame": "0x54c5b3", "net_bytes": 4_469_030_912}],
    }
    marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark("alloc_hook_C4_exit", rss_kib=10_000_000, stats=stats),
    ]
    result = attribute_alloc_hook_profile(marks, c4_delta_rss_gib=7.33)
    assert result["status"] == "UNMAPPED_ANONYMOUS_MMAP"
    assert result["mechanism_owner_status"] == "UNMAPPED_OR_UNRESOLVED"
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["hook_ran"] is True
    partition = result["allocator_type_partition"]
    assert partition is not None
    assert partition["mmap_net_gib"] > 4.0
    assert partition["non_mmap_remainder_gib"] > 2.5
    assert partition["libc_malloc_interposition_cuda_safe"] is False
    classified = result["classified_null"]
    assert classified["slice_outcome"] == "CLASSIFIED_NULL"
    assert classified["libc_malloc_ldpreload_cuda_incompatible"] is True
    assert "F4" in classified["superseded_folds"]


def test_build_attribution_receipt_auto_detects_alloc_hook_marks(tmp_path) -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        HOST_RSS_PROFILE_JSONL_NAME,
        PROFILE_HOST_RSS_SCHEMA,
    )
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        build_attribution_receipt,
    )

    profile_path = tmp_path / HOST_RSS_PROFILE_JSONL_NAME
    lines = [
        {
            "schema": PROFILE_HOST_RSS_SCHEMA,
            "phase": "sparse_cap_apply",
            "event": "enter",
            "step": 1,
            "resource_snapshot": {"rss_kib": 2_400_000},
        },
        {
            "schema": PROFILE_HOST_RSS_SCHEMA,
            "phase": "sparse_cap_apply",
            "event": "exit",
            "step": 1,
            "resource_snapshot": {"rss_kib": 10_200_000},
        },
        _alloc_hook_mark(
            "alloc_hook_C4_enter",
            rss_kib=2_500_000,
            stats={"window_net_bytes": 0},
        ),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={
                "window_net_bytes": 4_000_000_000,
                "prefault_done": 1,
                "lost_owner_count": 0,
                "table_overflow_count": 0,
                "unknown_free_bytes_bounded": True,
                "unknown_free_unmeasured_count": 0,
                "top_sites": [{"owner_frame": "0xabc", "net_bytes": 4_000_000_000}],
            },
        ),
    ]
    profile_path.write_text(
        "\n".join(json.dumps(row) for row in lines) + "\n",
        encoding="utf-8",
    )
    receipt = build_attribution_receipt(run_root=tmp_path, profile_path=profile_path)
    assert receipt["alloc_hook_attribution"] is not None
    assert receipt["alloc_hook_attribution"]["status"] == "UNMAPPED_ANONYMOUS_MMAP"
    assert receipt["alloc_hook_attribution"]["call_site_status"] == "UNRESOLVED"
    assert receipt["call_site_status"] == "UNRESOLVED"
    assert receipt["classified_null"]["slice_outcome"] == "CLASSIFIED_NULL"
    assert receipt["allocator_type_partition"]["mmap_net_gib"] > 3.5


def _malloc_info_payload(
    *,
    system_current: int,
    total_mmap: int,
) -> dict[str, Any]:
    return {
        "available": True,
        "system_current_bytes": system_current,
        "total_mmap_bytes": total_mmap,
        "glibc_arena_system_or_retained_bytes": system_current,
        "label": "glibc_arena_system_or_retained",
    }


def _allocator_mark_malloc_info(
    event: str,
    *,
    rss_kib: int,
    system_current: int,
    total_mmap: int,
    host_active: int | None = None,
) -> dict[str, Any]:
    cuda: dict[str, Any] = {"host_memory_stats_available": host_active is not None}
    if host_active is not None:
        cuda["cuda_host_active_bytes_all_current"] = host_active
    return {
        "schema": probe.PROFILE_HOST_RSS_ALLOCATOR_SCHEMA,
        "event": event,
        "resource_snapshot": {"rss_kib": rss_kib},
        "allocator_probe": {
            "malloc_info_all_arenas": _malloc_info_payload(
                system_current=system_current,
                total_mmap=total_mmap,
            ),
            "mallinfo2": {"uordblks_bytes": system_current},
            "cuda_allocator": cuda,
        },
        "measurement_perturbed": True,
    }


def _c4_subphase_profile_marks(*, delta_gib: float = 7.33) -> list[dict[str, Any]]:
    enter_kib = 2_500_000
    exit_kib = enter_kib + int(delta_gib * 1024 * 1024)
    return [
        {
            "schema": probe.PROFILE_HOST_RSS_SCHEMA,
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "enter",
            "resource_snapshot": {"rss_kib": enter_kib},
        },
        {
            "schema": probe.PROFILE_HOST_RSS_SCHEMA,
            "parent_phase": "sparse_cap_apply",
            "sub_phase": "C4_gpu_cap_apply_sync",
            "event": "exit",
            "resource_snapshot": {"rss_kib": exit_kib},
        },
    ]


def test_compute_delta_disjoint_partition_delta_not_absolute() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        compute_delta_disjoint_partition,
    )

    enter = _malloc_info_payload(system_current=500_000_000, total_mmap=1_000_000_000)
    exit_ = _malloc_info_payload(system_current=3_500_000_000, total_mmap=5_500_000_000)
    c4_bytes = int(7.33 * (1024**3))
    hook_window = 4_500_000_000
    part = compute_delta_disjoint_partition(
        c4_delta_rss_bytes=c4_bytes,
        hook_window_net_bytes=hook_window,
        malloc_info_enter=enter,
        malloc_info_exit=exit_,
        mmap_hook_catches_glibc_internal=True,
    )
    assert part["delta_glibc_mmap_bytes"] == 4_500_000_000
    assert part["non_glibc_mmap_bytes"] == 0
    assert "negative_non_glibc_mmap" not in part["fail_reasons"]
    absolute_wrong = hook_window - int(exit_["total_mmap_bytes"])
    assert absolute_wrong < 0


def test_allocator_type_partition_self_footprint_exceeded_inconclusive() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_type_partition,
    )

    marks = _c4_subphase_profile_marks()
    hook_marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={"window_net_bytes": 4_000_000_000},
        ),
    ]
    alloc_marks = [
        _allocator_mark_malloc_info(
            "allocator_C4_enter",
            rss_kib=2_500_000,
            system_current=500_000_000,
            total_mmap=1_000_000_000,
        ),
        _allocator_mark_malloc_info(
            "allocator_C4_exit",
            rss_kib=10_000_000,
            system_current=3_000_000_000,
            total_mmap=5_000_000_000,
        ),
    ]
    result = attribute_allocator_type_partition(
        marks=marks,
        alloc_hook_marks=hook_marks,
        allocator_marks=alloc_marks,
        disjointness_probe={"status": "ok", "mmap_hook_catches_glibc_internal": True},
        self_footprint={
            "malloc_info_self_footprint_status": "exceeded",
            "malloc_info_self_footprint_bytes": 8_000_000,
        },
    )
    assert result["allocator_type_owner_status"] == "INCONCLUSIVE"
    assert result["reason"] == "malloc_info_self_footprint_exceeded"


def test_allocator_type_partition_cross_run_not_resolved() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_type_partition,
    )

    marks = _c4_subphase_profile_marks()
    hook_marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={"window_net_bytes": 4_000_000_000},
        ),
    ]
    alloc_marks = [
        _allocator_mark_malloc_info(
            "allocator_C4_enter",
            rss_kib=2_500_000,
            system_current=500_000_000,
            total_mmap=1_000_000_000,
        ),
        _allocator_mark_malloc_info(
            "allocator_C4_exit",
            rss_kib=10_000_000,
            system_current=6_500_000_000,
            total_mmap=5_000_000_000,
        ),
    ]
    result = attribute_allocator_type_partition(
        marks=marks,
        alloc_hook_marks=hook_marks,
        allocator_marks=alloc_marks,
        disjointness_probe={"status": "ok", "mmap_hook_catches_glibc_internal": True},
        self_footprint={"malloc_info_self_footprint_status": "ok", "malloc_info_self_footprint_bytes": 0},
        cross_run_reconcile_caveat=True,
    )
    assert result["allocator_type_owner_status"] == "INCONCLUSIVE"
    assert result["cross_run_reconcile_caveat"] is True


def test_allocator_type_partition_disjointness_ambiguous_inconclusive() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_type_partition,
    )

    marks = _c4_subphase_profile_marks()
    hook_marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={"window_net_bytes": 4_000_000_000},
        ),
    ]
    alloc_marks = [
        _allocator_mark_malloc_info(
            "allocator_C4_enter",
            rss_kib=2_500_000,
            system_current=500_000_000,
            total_mmap=1_000_000_000,
        ),
        _allocator_mark_malloc_info(
            "allocator_C4_exit",
            rss_kib=10_000_000,
            system_current=3_000_000_000,
            total_mmap=5_000_000_000,
        ),
    ]
    result = attribute_allocator_type_partition(
        marks=marks,
        alloc_hook_marks=hook_marks,
        allocator_marks=alloc_marks,
        disjointness_probe=None,
        self_footprint={"malloc_info_self_footprint_status": "ok", "malloc_info_self_footprint_bytes": 0},
    )
    assert result["allocator_type_owner_status"] == "INCONCLUSIVE"
    assert "disjointness_probe_ambiguous" in result["fail_reasons"]


def test_allocator_type_partition_honest_glibc_label() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_type_partition,
    )

    marks = _c4_subphase_profile_marks()
    hook_marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={"window_net_bytes": 500_000_000},
        ),
    ]
    alloc_marks = [
        _allocator_mark_malloc_info(
            "allocator_C4_enter",
            rss_kib=2_500_000,
            system_current=200_000_000,
            total_mmap=100_000_000,
        ),
        _allocator_mark_malloc_info(
            "allocator_C4_exit",
            rss_kib=10_000_000,
            system_current=6_800_000_000,
            total_mmap=200_000_000,
        ),
    ]
    result = attribute_allocator_type_partition(
        marks=marks,
        alloc_hook_marks=hook_marks,
        allocator_marks=alloc_marks,
        disjointness_probe={"status": "ok", "mmap_hook_catches_glibc_internal": False},
        self_footprint={"malloc_info_self_footprint_status": "ok", "malloc_info_self_footprint_bytes": 0},
    )
    measured = result["partition"]["measured_buckets_bytes"]
    assert "glibc_arena_system_or_retained_bytes" in measured
    assert result["partition"]["delta_glibc_arena_system_or_retained_bytes"] == 6_600_000_000


def test_allocator_type_residual_dominant_without_cuda_host_stats() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        attribute_allocator_type_partition,
    )

    marks = _c4_subphase_profile_marks()
    hook_marks = [
        _alloc_hook_mark("alloc_hook_C4_enter", rss_kib=2_500_000),
        _alloc_hook_mark(
            "alloc_hook_C4_exit",
            rss_kib=10_000_000,
            stats={"window_net_bytes": 200_000_000},
        ),
    ]
    alloc_marks = [
        _allocator_mark_malloc_info(
            "allocator_C4_enter",
            rss_kib=2_500_000,
            system_current=200_000_000,
            total_mmap=100_000_000,
        ),
        _allocator_mark_malloc_info(
            "allocator_C4_exit",
            rss_kib=10_000_000,
            system_current=400_000_000,
            total_mmap=150_000_000,
        ),
    ]
    result = attribute_allocator_type_partition(
        marks=marks,
        alloc_hook_marks=hook_marks,
        allocator_marks=alloc_marks,
        disjointness_probe={"status": "ok", "mmap_hook_catches_glibc_internal": False},
        self_footprint={"malloc_info_self_footprint_status": "ok", "malloc_info_self_footprint_bytes": 0},
    )
    assert result["allocator_type_owner_status"] != "RESOLVED_BY_TYPE"
    assert result["cuda_host_measured"] is False
    if result["allocator_type_owner_status"] == "INFERRED_RESIDUAL_DOMINANT":
        assert result["tier"] == "C"


def test_parse_malloc_info_xml_labels() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import parse_malloc_info_xml

    xml = b"""<malloc version="1"><system type="current" size="12345"/>
    <total type="rest" size="100"/><total type="mmap" size="200"/></malloc>"""
    parsed = parse_malloc_info_xml(xml)
    assert parsed["available"] is True
    assert parsed["label"] == "glibc_arena_system_or_retained"
    assert parsed["glibc_arena_system_or_retained_bytes"] == 12345


def test_repeated_malloc_info_capture_footprint_bounded() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES,
        measure_malloc_info_self_footprint,
        read_malloc_info_all_arenas,
    )

    for _ in range(50):
        captured = read_malloc_info_all_arenas()
        if not captured.get("available"):
            pytest.skip("malloc_info unavailable on host")
    footprint = measure_malloc_info_self_footprint(samples=3)
    assert footprint["malloc_info_self_footprint_status"] == "ok"
    assert int(footprint["malloc_info_self_footprint_bytes"] or 0) <= MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES


def test_measure_malloc_info_self_footprint_counts_total_rest_delta(monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES,
        measure_malloc_info_self_footprint,
    )

    readings = [
        {
            "available": True,
            "system_current_bytes": 121_499_648,
            "total_mmap_bytes": 4_927_488,
            "total_rest_bytes": 844_074,
        },
        {
            "available": True,
            "system_current_bytes": 121_499_648,
            "total_mmap_bytes": 4_927_488,
            "total_rest_bytes": 839_994,
        },
        {
            "available": True,
            "system_current_bytes": 121_499_648,
            "total_mmap_bytes": 4_927_488,
            "total_rest_bytes": 835_914,
        },
    ]
    call_idx = {"i": 0}

    def _fake_read() -> dict:
        row = readings[min(call_idx["i"], len(readings) - 1)]
        call_idx["i"] += 1
        return dict(row)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.read_malloc_info_all_arenas",
        _fake_read,
    )
    footprint = measure_malloc_info_self_footprint(samples=3)
    assert footprint["malloc_info_self_footprint_bytes"] == 844_074 - 835_914
    assert footprint["malloc_info_self_footprint_bytes"] > 0
    assert footprint["malloc_info_self_footprint_status"] == "ok"
    assert footprint["malloc_info_self_footprint_bytes"] <= MALLOC_INFO_SELF_FOOTPRINT_MAX_BYTES


def test_measure_malloc_info_self_footprint_total_rest_exceeded_inconclusive(monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        measure_malloc_info_self_footprint,
    )

    readings = [
        {
            "available": True,
            "system_current_bytes": 100,
            "total_mmap_bytes": 200,
            "total_rest_bytes": 0,
        },
        {
            "available": True,
            "system_current_bytes": 100,
            "total_mmap_bytes": 200,
            "total_rest_bytes": 5_000_000,
        },
    ]
    call_idx = {"i": 0}

    def _fake_read() -> dict:
        row = readings[min(call_idx["i"], len(readings) - 1)]
        call_idx["i"] += 1
        return dict(row)

    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.read_malloc_info_all_arenas",
        _fake_read,
    )
    footprint = measure_malloc_info_self_footprint(samples=2)
    assert footprint["malloc_info_self_footprint_bytes"] == 5_000_000
    assert footprint["malloc_info_self_footprint_status"] == "exceeded"


def test_join_tracked_ranges_file_backed_rss_lt_len_no_rss_inflate() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        join_tracked_ranges_to_vmas,
    )

    live_ranges = [
        {
            "addr": "0x10000000",
            "addr_end": "0x20000000",
            "len": 268_435_456,
            "fd": 7,
            "owner_frame": "0x54c5b3",
        }
    ]
    vma_entries = [
        {
            "start": 0x10000000,
            "end": 0x20000000,
            "name": "/dev/nvidia0",
            "size_kb": 262144,
            "rss_kb": 4096,
            "private_dirty_kb": 4096,
            "anonymous_kb": 0,
            "referenced_kb": 4096,
            "excluded_hook_vma": False,
        }
    ]
    join = join_tracked_ranges_to_vmas(live_ranges, vma_entries)
    cuda = join["buckets"]["cuda_driver"]
    assert cuda["va_bytes"] == 268_435_456
    assert cuda["rss_bytes"] == 4096 * 1024
    assert cuda["rss_bytes"] < cuda["va_bytes"]


def test_attribute_non_glibc_mmap_source_unclassified_over_tolerance_inconclusive() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_non_glibc_mmap_source,
    )

    target = 1_000_000_000
    hook_marks = [
        {
            "event": "alloc_hook_C4_enter",
            "allocator_probe": {"vma_entries": []},
        },
        {
            "event": "alloc_hook_C4_exit",
            "alloc_hook_stats": {
                "window_net_bytes": target,
                "ring_drop_count": 0,
                "table_overflow_count": 0,
                "lost_owner_count": 0,
                "partial_munmap_ambiguity_count": 0,
                "lock_contention_drop_count": 0,
            },
            "live_ranges": [
                {
                    "addr": "0x1000",
                    "addr_end": "0x20000000",
                    "len": 500_000_000,
                    "fd": -1,
                    "owner_frame": "0x54c5b3",
                }
            ],
            "allocator_probe": {"vma_entries": []},
        },
    ]
    result = attribute_non_glibc_mmap_source(
        hook_marks,
        non_glibc_mmap_target_bytes=target,
    )
    assert result["source_tier"] == "C"
    assert result["status"] == "INCONCLUSIVE"
    assert "unclassified_over_tolerance" in result["fail_reasons"]
    assert result["call_site_status"] == "UNRESOLVED"


def test_attribute_non_glibc_mmap_source_call_site_unresolved_when_unmapped() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_non_glibc_mmap_source,
    )

    hook_marks = [
        {"event": "alloc_hook_C4_enter", "allocator_probe": {"vma_entries": []}},
        {
            "event": "alloc_hook_C4_exit",
            "alloc_hook_stats": {
                "window_net_bytes": 0,
                "ring_drop_count": 0,
                "table_overflow_count": 0,
                "lost_owner_count": 0,
                "partial_munmap_ambiguity_count": 0,
                "lock_contention_drop_count": 0,
            },
            "live_ranges": [],
            "allocator_probe": {"vma_entries": []},
        },
    ]
    result = attribute_non_glibc_mmap_source(hook_marks, non_glibc_mmap_target_bytes=1)
    assert result["call_site_status"] == "UNRESOLVED"
    assert result["source_tier"] != "A"


def test_classify_vma_name_fd_positive_blank_name_unknown_fd_not_file_backed() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import classify_vma_name

    assert classify_vma_name("", fd=7) == "unknown_fd"
    assert classify_vma_name("[anon:0x0]", fd=3) == "unknown_fd"
    assert classify_vma_name("", fd=-1) == "anonymous_private"


def test_classify_vma_name_fd_close_before_flush_uses_smaps_name_not_fd() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        classify_vma_name,
        join_tracked_ranges_to_vmas,
    )

    # Stale/reused fd after close; smaps name still identifies CUDA driver mapping.
    assert classify_vma_name("/dev/nvidia0", fd=99) == "cuda_driver"
    live_ranges = [
        {
            "addr": "0x10000000",
            "addr_end": "0x20000000",
            "len": 268_435_456,
            "fd": 99,
            "owner_frame": "0x54c5b3",
        }
    ]
    vma_entries = [
        {
            "start": 0x10000000,
            "end": 0x20000000,
            "name": "/dev/nvidia0",
            "size_kb": 262144,
            "rss_kb": 4096,
            "private_dirty_kb": 4096,
            "anonymous_kb": 0,
            "referenced_kb": 4096,
            "excluded_hook_vma": False,
        }
    ]
    join = join_tracked_ranges_to_vmas(live_ranges, vma_entries)
    assert "cuda_driver" in join["buckets"]
    assert "file_backed" not in join["buckets"]


def test_classify_vma_name_fd_reuse_stale_fd_fail_closed_unknown_fd() -> None:
    from calm.hrm_text_158.native_full_stack.host_alloc_hook_probe import (
        attribute_non_glibc_mmap_source,
        classify_vma_name,
    )

    assert classify_vma_name("", fd=42) == "unknown_fd"
    hook_marks = [
        {"event": "alloc_hook_C4_enter", "allocator_probe": {"vma_entries": []}},
        {
            "event": "alloc_hook_C4_exit",
            "alloc_hook_stats": {
                "window_net_bytes": 1_000_000_000,
                "ring_drop_count": 0,
                "table_overflow_count": 0,
                "lost_owner_count": 0,
                "partial_munmap_ambiguity_count": 0,
                "lock_contention_drop_count": 0,
            },
            "live_ranges": [
                {
                    "addr": "0x1000",
                    "addr_end": "0x2000",
                    "len": 4096,
                    "fd": 42,
                    "owner_frame": "0xdeadbeef",
                }
            ],
            "allocator_probe": {
                "vma_entries": [
                    {
                        "start": 0x1000,
                        "end": 0x2000,
                        "name": "",
                        "size_kb": 4,
                        "rss_kb": 4,
                        "private_dirty_kb": 4,
                        "anonymous_kb": 4,
                        "referenced_kb": 4,
                        "excluded_hook_vma": False,
                    }
                ]
            },
        },
    ]
    result = attribute_non_glibc_mmap_source(
        hook_marks,
        non_glibc_mmap_target_bytes=1_000_000_000,
    )
    assert result["source_tier"] == "C"
    assert result["status"] == "INCONCLUSIVE"
    assert "unknown_fd_classification" in result.get("fail_reasons", [])


def _c4_subphase_marks(delta_gib: float, *, step: int = 1) -> list[dict[str, Any]]:
    enter_kib = 1000 * 1024
    exit_kib = enter_kib + int(delta_gib * 1024 * 1024)
    return [
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "step": step,
            "event": "enter",
            "measurement_perturbed": False,
            "resource_snapshot": {"rss_kib": enter_kib},
        },
        {
            "sub_phase": "C4_gpu_cap_apply_sync",
            "step": step,
            "event": "exit",
            "measurement_perturbed": False,
            "resource_snapshot": {"rss_kib": exit_kib},
        },
    ]


def _triangulation_pair(
    *,
    current_delta: int,
    peak_delta: int,
    arena_base: int = 0,
    arena_exit: int = 0,
) -> list[dict[str, Any]]:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_TRIANGULATION_SCHEMA,
    )

    base_current = 100_000_000
    return [
        {
            "schema": PROFILE_HOST_RSS_TRIANGULATION_SCHEMA,
            "event": "triangulation_C3_exit",
            "tracemalloc": {
                "enabled": True,
                "traced_current_bytes": base_current,
                "traced_peak_bytes": base_current,
                "top_frames": [],
            },
            "debugmallocstats": {
                "available": True,
                "arena_bytes": arena_base,
                "arena_count": 1,
            },
        },
        {
            "schema": PROFILE_HOST_RSS_TRIANGULATION_SCHEMA,
            "event": "triangulation_C4_exit",
            "tracemalloc": {
                "enabled": True,
                "traced_current_bytes": base_current + current_delta,
                "traced_peak_bytes": base_current + peak_delta,
                "top_frames": [
                    {
                        "size_bytes": current_delta,
                        "traceback_key": "line1|line2|line3",
                        "traceback": ["line1", "line2", "line3"],
                    }
                ],
            },
            "debugmallocstats": {
                "available": True,
                "arena_bytes": arena_exit,
                "arena_count": 2,
            },
        },
    ]


def test_triangulation_denominator_invalid_inconclusive() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    delta = TOTAL_C4_REFERENCE_GIB + 2.0
    marks = _c4_subphase_marks(delta)
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
    )
    assert result["fail_closed_terminal"] == "DENOMINATOR_INVALID_INCONCLUSIVE"


def test_triangulation_cross_run_denominator_inconclusive() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks_a = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    marks_a_prime = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB + 3.0)
    result = attribute_python_allocator_triangulation(
        marks_a=marks_a,
        marks_a_prime=marks_a_prime,
        marks_b=marks_a,
    )
    assert result["fail_closed_terminal"] == "INCONCLUSIVE_CROSS_RUN_DENOMINATOR"


def test_triangulation_perturbation_adversary() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks_a = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    marks_b = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB + 3.0)
    result = attribute_python_allocator_triangulation(
        marks_a=marks_a,
        marks_a_prime=marks_a,
        marks_b=marks_b,
        debugmallocstats_preflight={"status": "ok"},
    )
    assert result["fail_closed_terminal"] == "TRACEMALLOC_PERTURBED_INCONCLUSIVE"


def test_triangulation_b_incomplete_tracemalloc_perturbed() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    partial_b = _triangulation_pair(
        current_delta=0,
        peak_delta=0,
    )[:1]
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + partial_b,
        debugmallocstats_preflight={"status": "ok"},
    )
    assert result["fail_closed_terminal"] == "TRACEMALLOC_PERTURBED_INCONCLUSIVE"
    assert result.get("b_run_incomplete") is True


def test_triangulation_arena_stats_unavailable() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    tri = _triangulation_pair(
        current_delta=BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        peak_delta=BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        arena_base=0,
        arena_exit=0,
    )
    tri[0]["debugmallocstats"] = {"available": False}
    tri[1]["debugmallocstats"] = {"available": False}
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + tri,
        debugmallocstats_preflight={"status": "unavailable"},
    )
    assert result["fail_closed_terminal"] == "ARENA_STATS_UNAVAILABLE_INCONCLUSIVE"


def test_triangulation_branch1_rewrite_candidates() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    tri = _triangulation_pair(
        current_delta=BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        peak_delta=BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        arena_base=100,
        arena_exit=100,
    )
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + tri,
        debugmallocstats_preflight={"status": "ok"},
    )
    assert result["classifier_branch"] == "LIVE_PYTHON_OBJECT_CHURN"
    assert result["rewrite_candidate_frames"]
    assert result["fail_closed_terminal"] is None


def test_triangulation_branch2_pymalloc_retention() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    peak = BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES
    current = int(peak * 0.10)
    tri = _triangulation_pair(
        current_delta=current,
        peak_delta=peak,
        arena_base=0,
        arena_exit=int(peak * 0.6),
    )
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + tri,
        debugmallocstats_preflight={"status": "ok"},
    )
    assert result["classifier_branch"] == "PYMALLOC_HIGH_WATER_RETENTION"
    assert result["rewrite_candidate_frames"] is None


def test_triangulation_branch3_untraced_extension() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_python_allocator_triangulation,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    current = int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES * 0.10)
    tri = _triangulation_pair(
        current_delta=current,
        peak_delta=current,
        arena_base=1000,
        arena_exit=2_000_000,
    )
    result = attribute_python_allocator_triangulation(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + tri,
        debugmallocstats_preflight={"status": "ok"},
    )
    assert result["classifier_branch"] == "UNTRACED_PYMEMP_C_EXTENSION"
    assert result["rewrite_candidate_frames"] is None


def test_debugmallocstats_preflight_parseable() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        preflight_debugmallocstats_self_test,
    )

    result = preflight_debugmallocstats_self_test()
    assert result["capture_method"] == "os.dup2_fd2_tempfile"
    assert result["status"] == "ok"


def test_gate_a_parse_debugmallocstats_current_arena_bytes() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        parse_debugmallocstats,
    )

    text = """
# arenas allocated current = 3
# arenas allocated total = 10
# arenas highwater mark = 8
3 arenas * 1048576 bytes/arena = 3,145,728
# bytes in allocated blocks = 1,234,567
# bytes in available blocks = 2,345,678
"""
    parsed = parse_debugmallocstats(text)
    assert parsed["parse_ok"] is True
    assert parsed["arenas_allocated_current"] == 3
    assert parsed["arenas_allocated_total"] == 10
    assert parsed["arenas_highwater_mark"] == 8
    assert parsed["arena_bytes"] == 3_145_728
    assert parsed["bytes_in_allocated_blocks"] == 1_234_567


def _obmalloc_c4_marks(
    *,
    c3_arena: int,
    c4_enter_arena: int,
    c4_exit_arena: int,
    c4_enter_alloc: int,
    c4_exit_alloc: int,
) -> list[dict[str, Any]]:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
    )

    stats = lambda arena, alloc: {
        "available": True,
        "parse_ok": True,
        "arenas_allocated_current": 1,
        "arena_bytes": arena,
        "bytes_in_allocated_blocks": alloc,
    }
    return [
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C3_exit",
            "measurement_perturbed": True,
            "debugmallocstats": stats(c3_arena, 0),
        },
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C4_enter",
            "measurement_perturbed": True,
            "debugmallocstats": stats(c4_enter_arena, c4_enter_alloc),
        },
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C4_exit",
            "measurement_perturbed": True,
            "debugmallocstats": stats(c4_exit_arena, c4_exit_alloc),
        },
    ]


def test_gate_b_obmalloc_reconcile_uses_c4_enter_baseline() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    obmalloc = _obmalloc_c4_marks(
        c3_arena=100,
        c4_enter_arena=1_000_000,
        c4_exit_arena=1_000_000 + BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        c4_enter_alloc=100,
        c4_exit_alloc=int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES * 0.8),
    )
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + obmalloc,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    deltas = result["deltas"]
    assert deltas["arena_bytes_delta_reconcile"] == BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES
    assert deltas["arena_bytes_delta_c3_to_c4_enter"] == 1_000_000 - 100
    assert result["classifier_terminal"] == "OBMALLOC_LIVE_CHURN"


def test_gate_c_obmalloc_self_footprint_inconclusive() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={
            "status": "exceeded",
            "debugmallocstats_self_footprint_status": "exceeded",
            "debugmallocstats_self_footprint_bytes": 9_000_000,
        },
    )
    assert result["fail_closed_terminal"] == "OBMALLOC_SELF_FOOTPRINT_INCONCLUSIVE"


def test_gate_c_obmalloc_arena_stats_unparseable() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        debugmallocstats_preflight={"status": "unavailable"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "ARENA_STATS_UNPARSEABLE_INCONCLUSIVE"


def test_gate_c_obmalloc_observer_perturbed_incomplete_b() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    partial_b = marks + [
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C4_enter",
            "measurement_perturbed": True,
            "debugmallocstats": {"available": True, "arena_bytes": 1, "bytes_in_allocated_blocks": 1},
        }
    ]
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=partial_b,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "OBSERVER_PERTURBED_INCONCLUSIVE"


def test_gate_d_obmalloc_arena_delta_floor_not_obmalloc() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    tiny_delta = 512 * 1024
    obmalloc = _obmalloc_c4_marks(
        c3_arena=0,
        c4_enter_arena=1_000_000,
        c4_exit_arena=1_000_000 + tiny_delta,
        c4_enter_alloc=100,
        c4_exit_alloc=100,
    )
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + obmalloc,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] == "NOT_OBMALLOC_UNTRACED"
    assert result["deltas"]["reconcile_ratio"] == pytest.approx(
        tiny_delta / BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES
    )


def test_gate_f_obmalloc_reconcile_out_of_band_low() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    delta = int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES * 0.30)
    obmalloc = _obmalloc_c4_marks(
        c3_arena=0,
        c4_enter_arena=0,
        c4_exit_arena=delta,
        c4_enter_alloc=0,
        c4_exit_alloc=int(delta * 0.8),
    )
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + obmalloc,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "RECONCILE_OUT_OF_BAND_INCONCLUSIVE"


def test_gate_f_obmalloc_not_obmalloc_untraced_low_ratio() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    delta = int(BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES * 0.10)
    obmalloc = _obmalloc_c4_marks(
        c3_arena=0,
        c4_enter_arena=0,
        c4_exit_arena=delta,
        c4_enter_alloc=0,
        c4_exit_alloc=0,
    )
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + obmalloc,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] == "NOT_OBMALLOC_UNTRACED"
    assert result["fail_closed_terminal"] is None


def test_gate_f_obmalloc_high_water_retention_branch() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_arena_retention,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    delta = BANKED_NON_GLIBC_MMAP_REFERENCE_BYTES
    obmalloc = _obmalloc_c4_marks(
        c3_arena=0,
        c4_enter_arena=0,
        c4_exit_arena=delta,
        c4_enter_alloc=0,
        c4_exit_alloc=int(delta * 0.1),
    )
    result = attribute_obmalloc_arena_retention(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + obmalloc,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] == "OBMALLOC_HIGH_WATER_RETENTION"


def test_gate_e_dual_env_abort_subprocess() -> None:
    import subprocess
    import sys

    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    env[probe.PROFILE_HOST_RSS_ENV] = "1"
    env[probe.PROFILE_TRACEMALLOC_ENV] = "1"
    env[probe.PROFILE_DEBUGMALLOCSTATS_ENV] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import scripts.hrm_text_158_bounded_delta_acquisition_probe as p; "
            "p.assert_profile_tracemalloc_debugmallocstats_mutual_exclusion()",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "profile_env_mutual_exclusion_abort" in proc.stdout


def test_capture_debugmallocstats_fd_cleanup_on_error(monkeypatch) -> None:
    from calm.hrm_text_158.native_full_stack import host_allocator_probe as hap

    monkeypatch.setattr(hap.sys, "_debugmallocstats", lambda: (_ for _ in ()).throw(RuntimeError("boom")), raising=False)
    text, err = hap.capture_debugmallocstats_fd2()
    assert text is None
    assert err is not None


def test_measure_debugmallocstats_self_footprint_ok() -> None:
    from calm.hrm_text_158.native_full_stack.host_allocator_probe import (
        DEBUGMALLOC_SELF_FOOTPRINT_MAX_BYTES,
        measure_debugmallocstats_self_footprint,
    )

    result = measure_debugmallocstats_self_footprint(samples=2)
    assert result["status"] in {"ok", "exceeded", "unavailable"}
    if result["status"] == "ok":
        assert int(result["debugmallocstats_self_footprint_bytes"] or 0) <= DEBUGMALLOC_SELF_FOOTPRINT_MAX_BYTES


def _obmalloc_site_bracket_marks(
    *,
    window_entry_arena: int,
    window_exit_arena: int,
    leaf_deltas: dict[str, int],
    s2_pre_arena: int | None = None,
) -> list[dict[str, Any]]:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        OBMALLOC_SITE_LEAF_SITES,
    )
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
    )

    def _stats(arena: int) -> dict[str, Any]:
        return {
            "available": True,
            "parse_ok": True,
            "arenas_allocated_current": 1,
            "arena_bytes": arena,
            "bytes_in_allocated_blocks": 0,
        }

    marks: list[dict[str, Any]] = []
    base = 10_000_000

    marks.append(
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
            "event": "obmalloc_site_C4.S1_pre",
            "site_id": "C4.S1",
            "state_index": 0,
            "measurement_perturbed": True,
            "debugmallocstats": _stats(window_entry_arena),
        }
    )
    s1_delta = int(leaf_deltas.get("C4.S1", 0))
    marks.append(
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
            "event": "obmalloc_site_C4.S1_post",
            "site_id": "C4.S1",
            "state_index": 0,
            "measurement_perturbed": True,
            "debugmallocstats": _stats(window_entry_arena + s1_delta),
        }
    )

    for site_id in OBMALLOC_SITE_LEAF_SITES:
        if site_id in {"C4.S1", "C4.S2"}:
            continue
        delta = int(leaf_deltas.get(site_id, 0))
        marks.extend(
            [
                {
                    "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
                    "event": f"obmalloc_site_{site_id}_pre",
                    "site_id": site_id,
                    "state_index": 0,
                    "measurement_perturbed": True,
                    "debugmallocstats": _stats(base),
                },
                {
                    "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
                    "event": f"obmalloc_site_{site_id}_post",
                    "site_id": site_id,
                    "state_index": 0,
                    "measurement_perturbed": True,
                    "debugmallocstats": _stats(base + delta),
                },
            ]
        )

    if s2_pre_arena is not None:
        marks.append(
            {
                "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
                "event": "obmalloc_site_C4.S2_pre",
                "site_id": "C4.S2",
                "state_index": 0,
                "measurement_perturbed": True,
                "debugmallocstats": _stats(s2_pre_arena),
            }
        )
    marks.append(
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
            "event": "obmalloc_site_C4.S2_post",
            "site_id": "C4.S2",
            "state_index": 0,
            "measurement_perturbed": True,
            "debugmallocstats": _stats(window_exit_arena),
        }
    )
    return marks


def test_obmalloc_site_brackets_leaf_sum_excludes_aggregate_parent() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        OBMALLOC_SITE_LEAF_SITES,
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_site_brackets,
    )

    window_entry = 100_000_000
    state0_local = 1_000_000_000
    window_exit = window_entry + state0_local
    leaf_deltas = {
        "C4.S1": 600_000_000,
        "C4.S2a": 150_000_000,
        "C4.S2b": 150_000_000,
        "C4.S2c": 100_000_000,
    }
    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    site_marks = _obmalloc_site_bracket_marks(
        window_entry_arena=window_entry,
        window_exit_arena=window_exit,
        leaf_deltas=leaf_deltas,
        s2_pre_arena=window_entry + leaf_deltas["C4.S1"],
    )
    result = attribute_obmalloc_site_brackets(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + site_marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    localization = result["localization"]
    leaf_sum = sum(int(localization["leaf_deltas_bytes"][site]) for site in OBMALLOC_SITE_LEAF_SITES)
    aggregate_s2 = int(localization["aggregate_s2_delta_bytes"])
    assert localization["leaf_sum_bytes"] == leaf_sum
    assert localization["leaf_sum_bytes"] == state0_local
    assert aggregate_s2 == sum(
        leaf_deltas[site] for site in ("C4.S2a", "C4.S2b", "C4.S2c")
    )
    assert localization["leaf_sum_bytes"] + aggregate_s2 > state0_local
    assert localization["unattributed_remainder_bytes"] == 0
    assert result["classifier_terminal"] == "DOMINANT_BRACKET_C4.S1"
    assert result["slice8_rewrite_authorized"] is False


def test_obmalloc_site_brackets_q_loop_growth_lands_in_remainder() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_site_brackets,
    )

    state0_local = 800_000_000
    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
    site_marks = _obmalloc_site_bracket_marks(
        window_entry_arena=2_000_000,
        window_exit_arena=2_000_000 + state0_local,
        leaf_deltas={
            "C4.S1": 0,
            "C4.S2a": 0,
            "C4.S2b": 0,
            "C4.S2c": 0,
        },
    )
    result = attribute_obmalloc_site_brackets(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks + site_marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    localization = result["localization"]
    assert localization["leaf_sum_bytes"] == 0
    assert localization["unattributed_remainder_bytes"] == state0_local
    assert localization["next_lane_rank1_carrier_audit"] is True
    assert result["fail_closed_terminal"] == "BRACKET_REMAINDER_TOO_LARGE"
    assert result["classifier_terminal"] is None
    assert result["slice8_rewrite_authorized"] is False


def test_compute_obmalloc_expanded_sampled_states_n5() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        compute_obmalloc_expanded_sampled_states,
    )

    assert compute_obmalloc_expanded_sampled_states(5) == (0, 1, 3, 4)
    assert compute_obmalloc_expanded_sampled_states(32) == (0, 10, 21, 31)


def test_profile_obmalloc_expanded_default_off_all_profiling(monkeypatch) -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_DEBUGMALLOCSTATS_ENV,
        PROFILE_HOST_RSS_ENV,
        PROFILE_OBMALLOC_EXPANDED_ENV,
        PROFILE_OBMALLOC_SITE_BRACKETS_ENV,
        profile_obmalloc_expanded_enabled,
    )

    monkeypatch.delenv(PROFILE_HOST_RSS_ENV, raising=False)
    monkeypatch.delenv(PROFILE_DEBUGMALLOCSTATS_ENV, raising=False)
    monkeypatch.delenv(PROFILE_OBMALLOC_SITE_BRACKETS_ENV, raising=False)
    monkeypatch.delenv(PROFILE_OBMALLOC_EXPANDED_ENV, raising=False)
    assert profile_obmalloc_expanded_enabled() is False


def test_profile_obmalloc_expanded_default_off_site_brackets_on_expanded_off(
    monkeypatch,
) -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_DEBUGMALLOCSTATS_ENV,
        PROFILE_HOST_RSS_ENV,
        PROFILE_OBMALLOC_EXPANDED_ENV,
        PROFILE_OBMALLOC_SITE_BRACKETS_ENV,
        profile_obmalloc_expanded_enabled,
        profile_obmalloc_site_brackets_enabled,
    )

    monkeypatch.setenv(PROFILE_HOST_RSS_ENV, "1")
    monkeypatch.setenv(PROFILE_DEBUGMALLOCSTATS_ENV, "1")
    monkeypatch.setenv(PROFILE_OBMALLOC_SITE_BRACKETS_ENV, "1")
    monkeypatch.delenv(PROFILE_OBMALLOC_EXPANDED_ENV, raising=False)
    assert profile_obmalloc_site_brackets_enabled() is True
    assert profile_obmalloc_expanded_enabled() is False


def test_obmalloc_site_state_enabled_respects_sampled_states() -> None:
    from calm.hrm_text_158.native_full_stack.sparse_cap_gpu_seam_adapter import (
        _obmalloc_site_state_enabled,
    )

    assert _obmalloc_site_state_enabled(0, sampled_states=None) is True
    assert _obmalloc_site_state_enabled(1, sampled_states=None) is False
    sampled = frozenset({0, 10, 20, 31})
    assert _obmalloc_site_state_enabled(10, sampled_states=sampled) is True
    assert _obmalloc_site_state_enabled(5, sampled_states=sampled) is False


def _obmalloc_expanded_site_marks_for_state(
    *,
    state_index: int,
    leaf_holding_deltas: dict[str, int],
    base_blocks: int = 100_000_000,
) -> list[dict[str, Any]]:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
    )

    def _stats(arena: int, blocks: int) -> dict[str, Any]:
        return {
            "available": True,
            "parse_ok": True,
            "arenas_allocated_current": 1,
            "arena_bytes": arena,
            "bytes_in_allocated_blocks": blocks,
        }

    marks: list[dict[str, Any]] = []
    running_blocks = int(base_blocks)
    running_arena = 10_000_000
    for site_id, delta in leaf_holding_deltas.items():
        pre_blocks = running_blocks
        pre_arena = running_arena
        post_blocks = pre_blocks + int(delta)
        post_arena = pre_arena + max(int(delta), 0)
        marks.extend(
            [
                {
                    "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
                    "event": f"obmalloc_site_{site_id}_pre",
                    "site_id": site_id,
                    "state_index": int(state_index),
                    "obmalloc_expanded": True,
                    "measurement_perturbed": True,
                    "debugmallocstats": _stats(pre_arena, pre_blocks),
                },
                {
                    "schema": PROFILE_HOST_RSS_OBMALLOC_SITE_SCHEMA,
                    "event": f"obmalloc_site_{site_id}_post",
                    "site_id": site_id,
                    "state_index": int(state_index),
                    "obmalloc_expanded": True,
                    "measurement_perturbed": True,
                    "debugmallocstats": _stats(post_arena, post_blocks),
                },
            ]
        )
        running_blocks = post_blocks
        running_arena = post_arena
    return marks


def _obmalloc_expanded_boundary_marks(
    *,
    after_state_blocks: list[int],
) -> list[dict[str, Any]]:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
    )

    def _stats(arena: int, blocks: int) -> dict[str, Any]:
        return {
            "available": True,
            "parse_ok": True,
            "arena_bytes": arena,
            "bytes_in_allocated_blocks": blocks,
        }

    marks: list[dict[str, Any]] = [
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C4_enter",
            "measurement_perturbed": True,
            "debugmallocstats": _stats(1_000_000, 50_000_000),
        }
    ]
    for idx, blocks in enumerate(after_state_blocks):
        marks.append(
            {
                "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
                "event": "obmalloc_C4_after_state",
                "state_index": int(idx * 4 + 3),
                "measurement_perturbed": True,
                "debugmallocstats": _stats(1_000_000 + idx * 1_048_576, int(blocks)),
            }
        )
    marks.append(
        {
            "schema": PROFILE_HOST_RSS_OBMALLOC_SCHEMA,
            "event": "obmalloc_C4_exit",
            "measurement_perturbed": True,
            "debugmallocstats": _stats(10_000_000, int(after_state_blocks[-1])),
        }
    )
    return marks


def test_obmalloc_expanded_event_count_no_duplicate() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        OBMALLOC_EXPANDED_EVENT_COUNT_MAX,
        OBMALLOC_EXPANDED_EVENT_COUNT_TARGET,
        TOTAL_C4_REFERENCE_GIB,
        _count_obmalloc_expanded_enabled_events,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas={
                    "C4.S1": 100_000,
                    "C4.S2a": 50_000,
                    "C4.S2b": 50_000,
                    "C4.S2c": 50_000,
                },
            )
        )
    boundary = _obmalloc_expanded_boundary_marks(
        after_state_blocks=[100_000_000 + i * 50_000_000 for i in range(8)],
    )
    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB) + boundary + site_marks
    counts = _count_obmalloc_expanded_enabled_events(marks)
    assert counts["obmalloc_C4_enter"] == 1
    assert counts["obmalloc_C4_after_state"] == 8
    assert counts["obmalloc_C4_exit"] == 1
    assert counts["site_leaf_bracket"] == 32
    assert counts["total"] == 42

    duplicate_boundary = list(boundary)
    for extra_idx in range(65):
        duplicate_boundary.append(
            {
                "schema": boundary[1]["schema"],
                "event": "obmalloc_C4_after_state",
                "state_index": 100 + extra_idx,
                "measurement_perturbed": True,
                "debugmallocstats": boundary[1]["debugmallocstats"],
            }
        )
    dup_counts = _count_obmalloc_expanded_enabled_events(
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB) + duplicate_boundary + site_marks
    )
    assert dup_counts["total"] > OBMALLOC_EXPANDED_EVENT_COUNT_MAX

    dup_result = attribute_obmalloc_expanded(
        marks_a=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_a_prime=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_b=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB) + duplicate_boundary + site_marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert dup_result["observer_reason"] == "duplicate_obmalloc_emit"

    result = attribute_obmalloc_expanded(
        marks_a=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_a_prime=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_b=marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] is not None


def test_obmalloc_expanded_holder_zero_denom_adversary() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks = _obmalloc_expanded_site_marks_for_state(
        state_index=int(sampled[0]),
        leaf_holding_deltas={
            "C4.S1": 0,
            "C4.S2a": 0,
            "C4.S2b": 0,
            "C4.S2c": 0,
        },
    )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[100_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        sampled_states=sampled[:1],
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "HOLDER_AMBIGUOUS"
    assert result["slice8_rewrite_authorized"] is False


def test_obmalloc_expanded_cancellation_adversary() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks = _obmalloc_expanded_site_marks_for_state(
        state_index=int(sampled[0]),
        leaf_holding_deltas={
            "C4.S1": 1_000_000,
            "C4.S2a": -500_000,
            "C4.S2b": -400_000,
            "C4.S2c": 100_000,
        },
    )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[100_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        sampled_states=sampled[:1],
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "HOLDER_AMBIGUOUS"
    assert result["observer_reason"] == "cancellation_inflation"
    assert result["slice8_rewrite_authorized"] is False


def test_obmalloc_expanded_k2_no_slice8_authorize() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
    )

    sampled = (0, 31)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas={
                    "C4.S1": 600_000_000,
                    "C4.S2a": 100_000_000,
                    "C4.S2b": 100_000_000,
                    "C4.S2c": 100_000_000,
                },
            )
        )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        sampled_states=sampled,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert str(result.get("classifier_terminal", "")).startswith("DOMINANT_HOLDER_BRACKET_")
    assert result["slice8_rewrite_authorized"] is False


def test_obmalloc_expanded_missing_c4_exit_observer_perturbed() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
    )

    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB) + [
        {
            "schema": "hrm_text_158_profile_host_rss_mark/v8",
            "event": "obmalloc_C4_enter",
            "measurement_perturbed": True,
            "debugmallocstats": {"available": True, "arena_bytes": 1, "bytes_in_allocated_blocks": 1},
        }
    ]
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["fail_closed_terminal"] == "OBSERVER_PERTURBED_INCONCLUSIVE"
    assert result["observer_reason"] == "missing_obmalloc_C4_exit"


def test_obmalloc_expanded_dominant_holder_bracket() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas={
                    "C4.S1": 600_000_000,
                    "C4.S2a": 100_000_000,
                    "C4.S2b": 100_000_000,
                    "C4.S2c": 100_000_000,
                },
            )
        )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] == "DOMINANT_HOLDER_BRACKET_C4.S1"
    localization = result["localization"]
    assert localization["representativeness_cleared"] is True


def test_obmalloc_expanded_cross_state_retention_classifier() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas={
                    "C4.S1": 10_000,
                    "C4.S2a": 10_000,
                    "C4.S2b": 10_000,
                    "C4.S2c": 10_000,
                },
            )
        )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(
            after_state_blocks=[
                100_000_000,
                200_000_000,
                300_000_000,
                400_000_000,
                500_000_000,
                600_000_000,
                700_000_000,
                800_000_000,
            ]
        )
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        debugmallocstats_preflight={"status": "ok"},
        self_footprint_preflight={"status": "ok", "debugmallocstats_self_footprint_status": "ok"},
    )
    assert result["classifier_terminal"] == "RETENTION_DOMINANT_CROSS_STATE"
    assert result["slice8_rewrite_authorized"] is False


def _obmalloc_expanded_child_holding_deltas(
    parent_hold: int,
    *,
    reconcile: bool = True,
) -> dict[str, int]:
    if reconcile:
        child_parts = {
            "C4.S1a": parent_hold // 20,
            "C4.S1b": parent_hold // 20,
            "C4.S1c_clone": (parent_hold * 9) // 10,
            "C4.S1c_contig": parent_hold // 50,
            "C4.S1d": parent_hold // 100,
            "C4.S1e": parent_hold // 100,
            "C4.S1f": parent_hold // 100,
        }
        child_sum = sum(child_parts.values())
        child_parts["C4.S1c_clone"] += parent_hold - child_sum
    else:
        child_parts = {
            "C4.S1a": parent_hold // 10,
            "C4.S1b": parent_hold // 10,
            "C4.S1c_clone": parent_hold,
            "C4.S1c_contig": parent_hold // 10,
            "C4.S1d": parent_hold // 10,
            "C4.S1e": parent_hold // 10,
            "C4.S1f": parent_hold // 10,
        }
    return {
        "C4.S1": parent_hold,
        **child_parts,
        "C4.S2a": 50_000,
        "C4.S2b": 50_000,
        "C4.S2c": 50_000,
    }


def _obmalloc_expanded_preflight() -> dict[str, Any]:
    return {
        "debugmallocstats_preflight": {"status": "ok"},
        "self_footprint_preflight": {
            "status": "ok",
            "debugmallocstats_self_footprint_status": "ok",
        },
    }


def test_obmalloc_expanded_invisible_child_mark_dropped_without_consumer_wiring(
    monkeypatch,
) -> None:
    import scripts.hrm_text_158_slice5_v6i_oom_profile_attribution as attribution

    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        OBMALLOC_SITE_LEAF_SITES,
        TOTAL_C4_REFERENCE_GIB,
        _site_bracket_holding_delta_bytes,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    legacy_leaf_sites = ("C4.S1", "C4.S2a", "C4.S2b", "C4.S2c")
    sampled = compute_obmalloc_expanded_sampled_states(32)
    state_idx = int(sampled[0])
    site_marks = _obmalloc_expanded_site_marks_for_state(
        state_index=state_idx,
        leaf_holding_deltas={
            "C4.S1": 100_000,
            "C4.S1c_clone": 80_000,
            "C4.S2a": 50_000,
            "C4.S2b": 50_000,
            "C4.S2c": 50_000,
        },
    )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[100_000_000] * 8)
        + site_marks
    )
    unwired_totals = {
        site_id: _site_bracket_holding_delta_bytes(
            marks,
            site_id,
            state_idx,
            absent_is_zero=True,
        )
        for site_id in legacy_leaf_sites
    }
    assert "C4.S1c_clone" not in unwired_totals
    assert _site_bracket_holding_delta_bytes(
        marks,
        "C4.S1c_clone",
        state_idx,
        absent_is_zero=True,
    ) == 80_000

    site_marks_full: list[dict[str, Any]] = []
    for idx in sampled:
        site_marks_full.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(idx),
                leaf_holding_deltas=_obmalloc_expanded_child_holding_deltas(100_000),
            )
        )
    full_marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[100_000_000] * 8)
        + site_marks_full
    )
    wired = attribute_obmalloc_expanded(
        marks_a=full_marks,
        marks_a_prime=full_marks,
        marks_b=full_marks,
        **_obmalloc_expanded_preflight(),
    )
    assert "C4.S1c_clone" in wired["localization"]["aggregate_holder_pos_bytes"]
    assert wired["localization"]["aggregate_holder_pos_bytes"]["C4.S1c_clone"] > 0
    assert len(OBMALLOC_SITE_LEAF_SITES) == 11
    monkeypatch.setattr(attribution, "OBMALLOC_SITE_LEAF_SITES", legacy_leaf_sites)
    assert "C4.S1c_clone" not in attribution.OBMALLOC_SITE_LEAF_SITES


def test_obmalloc_expanded_legacy_c4s1_s2_leaf_sites_preserved_after_child_extension() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas={
                    "C4.S1": 600_000_000,
                    "C4.S2a": 100_000_000,
                    "C4.S2b": 100_000_000,
                    "C4.S2c": 100_000_000,
                },
            )
        )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        **_obmalloc_expanded_preflight(),
    )
    localization = result["localization"]
    assert localization["child_profile_mode"] is False
    assert result["fail_closed_terminal"] is None
    assert localization["child_dominance_verdict"] == "legacy_child_sites_absent"
    assert result["classifier_terminal"] == "DOMINANT_HOLDER_BRACKET_C4.S1"
    assert localization["aggregate_holder_pos_bytes"]["C4.S1"] == 600_000_000 * len(sampled)


def test_obmalloc_expanded_child_sum_reconciles_parent_c4s1() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    parent_hold = 100_000_000
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas=_obmalloc_expanded_child_holding_deltas(
                    parent_hold,
                    reconcile=True,
                ),
            )
        )
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        **_obmalloc_expanded_preflight(),
    )
    localization = result["localization"]
    assert localization["child_profile_mode"] is True
    assert result["fail_closed_terminal"] is None
    assert localization["child_parent_reconcile_fraction"] <= 0.15
    assert localization["child_dominant_bracket"] == "C4.S1c_clone"
    assert localization["child_dominant_bracket"] != "C4.S1"

    bad_marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + [
            *_obmalloc_expanded_site_marks_for_state(
                state_index=int(sampled[0]),
                leaf_holding_deltas=_obmalloc_expanded_child_holding_deltas(
                    parent_hold,
                    reconcile=False,
                ),
            )
        ]
    )
    bad_result = attribute_obmalloc_expanded(
        marks_a=bad_marks,
        marks_a_prime=bad_marks,
        marks_b=bad_marks,
        sampled_states=(int(sampled[0]),),
        **_obmalloc_expanded_preflight(),
    )
    assert bad_result["fail_closed_terminal"] == "CHILD_PARENT_RECONCILE_FAIL"


def test_obmalloc_expanded_event_count_child_sites() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        OBMALLOC_EXPANDED_EVENT_COUNT_MAX,
        OBMALLOC_EXPANDED_EVENT_COUNT_TARGET,
        OBMALLOC_SITE_LEAF_SITES,
        TOTAL_C4_REFERENCE_GIB,
        _count_obmalloc_expanded_enabled_events,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    site_marks: list[dict[str, Any]] = []
    for state_idx in sampled:
        site_marks.extend(
            _obmalloc_expanded_site_marks_for_state(
                state_index=int(state_idx),
                leaf_holding_deltas=_obmalloc_expanded_child_holding_deltas(100_000),
            )
        )
    boundary = _obmalloc_expanded_boundary_marks(
        after_state_blocks=[100_000_000 + i * 50_000_000 for i in range(8)],
    )
    marks = _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB) + boundary + site_marks
    counts = _count_obmalloc_expanded_enabled_events(marks)
    assert len(OBMALLOC_SITE_LEAF_SITES) == 11
    assert counts["site_leaf_bracket"] == 88
    assert counts["total"] == OBMALLOC_EXPANDED_EVENT_COUNT_TARGET
    assert counts["total"] == 98

    duplicate = list(marks)
    duplicate.extend(
        _obmalloc_expanded_site_marks_for_state(
            state_index=int(sampled[0]),
            leaf_holding_deltas=_obmalloc_expanded_child_holding_deltas(100_000),
        )
    )
    dup_counts = _count_obmalloc_expanded_enabled_events(duplicate)
    assert dup_counts["total"] > OBMALLOC_EXPANDED_EVENT_COUNT_MAX

    dup_result = attribute_obmalloc_expanded(
        marks_a=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_a_prime=_c4_subphase_marks(TOTAL_C4_REFERENCE_GIB),
        marks_b=duplicate,
        **_obmalloc_expanded_preflight(),
    )
    assert dup_result["fail_closed_terminal"] == "OBSERVER_PERTURBED_INCONCLUSIVE"


def test_obmalloc_expanded_missing_child_site_fails_coverage_not_silent_zero() -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        TOTAL_C4_REFERENCE_GIB,
        attribute_obmalloc_expanded,
        compute_obmalloc_expanded_sampled_states,
    )

    sampled = compute_obmalloc_expanded_sampled_states(32)
    state_idx = int(sampled[0])
    holding = _obmalloc_expanded_child_holding_deltas(100_000_000)
    site_marks = _obmalloc_expanded_site_marks_for_state(
        state_index=state_idx,
        leaf_holding_deltas=holding,
    )
    site_marks = [
        row
        for row in site_marks
        if row["event"] != "obmalloc_site_C4.S1d_pre"
    ]
    marks = (
        _c4_subphase_marks(TOTAL_C4_REFERENCE_GIB)
        + _obmalloc_expanded_boundary_marks(after_state_blocks=[200_000_000] * 8)
        + site_marks
    )
    result = attribute_obmalloc_expanded(
        marks_a=marks,
        marks_a_prime=marks,
        marks_b=marks,
        sampled_states=(state_idx,),
        **_obmalloc_expanded_preflight(),
    )
    assert result["guards"]["child_profile_mode"] is True
    assert result["fail_closed_terminal"] == "CHILD_COVERAGE_FAIL"
    assert result["missing_child_site"] == "C4.S1d"


def test_fixture_probe_argv_includes_max_silent_phase_seconds_600_non_tracemalloc(
    tmp_path: Path,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS,
        _fixture_probe_argv,
    )

    cmd = _fixture_probe_argv(tmp_path / "scratch")
    idx = cmd.index("--max-silent-phase-seconds")
    assert cmd[idx + 1] == str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS)
    assert FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS == 600


def test_fixture_probe_argv_tracemalloc_keeps_900(tmp_path: Path) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC,
        _fixture_probe_argv,
    )

    cmd = _fixture_probe_argv(tmp_path / "scratch", tracemalloc=True)
    idx = cmd.index("--max-silent-phase-seconds")
    assert cmd[idx + 1] == str(FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC)
    assert FIXTURE_PROBE_MAX_SILENT_PHASE_SECONDS_TRACEMALLOC == 900


def test_run_subprocess_streaming_to_log_returns_stream_path_no_capture_output(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        _run_subprocess_streaming_to_log,
    )

    log_path = tmp_path / "probe_stream.log"

    class _FakeProc:
        returncode = 0

        def wait(self, timeout=None):
            log_path.write_text("phase_heartbeat tick\n", encoding="utf-8")
            return 0

        def kill(self):
            return None

    def _fake_popen(*_args, **_kwargs):
        assert "capture_output" not in _kwargs
        assert _kwargs.get("stdout") is not None
        return _FakeProc()

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.subprocess.Popen",
        _fake_popen,
    )
    result = _run_subprocess_streaming_to_log(
        ["echo", "ok"],
        cwd=tmp_path,
        env={},
        log_path=log_path,
        timeout=30.0,
    )
    assert result["used_capture_output"] is False
    assert result["probe_stream_log"] == str(log_path)


def test_run_fixture_obmalloc_probe_lane_release_on_timeout(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        _run_fixture_obmalloc_probe,
    )

    release_calls: list[Path] = []

    def _fake_acquire(out_root: Path):
        return {"lane": "held"}

    def _fake_release(out_root: Path):
        release_calls.append(out_root)
        return {"lane": "released"}

    def _fake_streaming(*_args, **_kwargs):
        return {
            "exit_code": 124,
            "probe_stream_log": str(tmp_path / "probe_stream.log"),
            "subprocess_timeout_expired": True,
            "stdout_tail": "",
            "stderr_tail": "",
            "used_capture_output": False,
        }

    monkeypatch.setattr(
        "scripts.hrm_text_158_r7_resource_lane_acquire.acquire_resource_lane",
        _fake_acquire,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_r7_resource_lane_release.release_resource_lane",
        _fake_release,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_subprocess_streaming_to_log",
        _fake_streaming,
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._fixture_obmalloc_env",
        lambda **_kwargs: {},
    )

    payload = _run_fixture_obmalloc_probe(
        tmp_path,
        scratch_name="timeout_case",
        debugmallocstats=True,
    )
    assert release_calls == [tmp_path]
    assert payload["subprocess_timeout_expired"] is True
    assert payload["resource_lane_release"] == {"lane": "released"}


def test_durable_mirror_default_off(tmp_path: Path, monkeypatch) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        run_fixture_obmalloc_expanded_combined,
    )

    def _fake_probe(*_args, **_kwargs):
        return {
            "exit_code": 0,
            "marks": [],
        }

    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution._run_fixture_obmalloc_probe",
        _fake_probe,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.preflight_debugmallocstats_self_test",
        lambda: {"ok": True},
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.host_allocator_probe.measure_debugmallocstats_self_footprint",
        lambda: {"ok": True},
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.attribute_obmalloc_expanded",
        lambda **_kwargs: {"classifier_terminal": "INCONCLUSIVE"},
    )
    monkeypatch.setattr(
        "scripts.hrm_text_158_slice5_v6i_oom_profile_attribution.attribute_c4_retention_owner_census",
        lambda **_kwargs: {"classifier_terminal": "INCONCLUSIVE"},
    )

    payload = run_fixture_obmalloc_expanded_combined(tmp_path, mirror_durable_attribution=False)
    assert "durable_mirror_receipt" not in payload
    assert "durable_artifact_path" not in payload
    assert "combined_attribution_path" not in payload


def test_durable_mirror_opt_in_writes_receipt(tmp_path: Path) -> None:
    from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
        _maybe_mirror_durable_attribution,
    )

    mirror_path = tmp_path / "mirror.json"
    mirror_path.write_text('{"seed": true}\n', encoding="utf-8")
    payload = {"fixture_mode": "fixture_obmalloc_expanded", "exit_code": 0}
    result = _maybe_mirror_durable_attribution(
        payload,
        mirror=True,
        mirror_path=mirror_path,
    )
    receipt = result["durable_mirror_receipt"]
    assert receipt["mirror_path"] == str(mirror_path)
    assert receipt["pre_hash"] is not None
    assert receipt["backup_path"] is not None
    assert receipt["post_hash"] is not None
    assert Path(receipt["backup_path"]).is_file()
    assert result["durable_artifact_path"] == str(mirror_path)