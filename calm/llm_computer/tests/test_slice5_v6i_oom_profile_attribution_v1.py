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
    assert receipt["schema"].endswith("/v7")
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

