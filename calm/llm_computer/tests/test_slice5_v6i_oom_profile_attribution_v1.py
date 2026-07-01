from __future__ import annotations

import json
import os
from pathlib import Path

import torch

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe
from scripts.hrm_text_158_slice5_v6i_oom_profile_attribution import (
    attribute_host_rss_profile,
    build_attribution_receipt,
    extract_run_root,
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
    assert all("resource_snapshot" in row for row in rows)


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
    assert result["dominant_wall_owner"]["phase"] == "sparse_cap_apply"


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
    assert receipt["schema"].endswith("/v2")
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
