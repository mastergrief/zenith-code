"""CPU tests for box-lane code-currency preflight and science-chain watcher (C6 + V*)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.box_lane import (
    EXIT_ARTIFACT_RSYNC_MISMATCH,
    EXIT_CODE_CURRENCY_MISMATCH,
    EXIT_OVERLAP_FAILURE,
    PinnedFile,
    build_code_currency_manifest,
    classify_overlap,
    hash_pinned_files,
    process_science_chain_log,
    sync_pinned_files,
    validate_chain_id,
    validate_receipt_residency,
    verify_artifact_manifest,
    verify_head_triple,
)
from scripts.box_lane_chain_watcher import main as watcher_main
from scripts.box_lane_code_currency_preflight import main as preflight_main


def test_fetch_head_mismatch_fails() -> None:
    issues = verify_head_triple(
        head_now="abc",
        fetch_head="def",
        head_expected="abc",
    )
    assert "FETCH_HEAD_MISMATCH" in issues


def test_head_now_mismatch_fails() -> None:
    issues = verify_head_triple(
        head_now="abc",
        fetch_head="abc",
        head_expected="def",
    )
    assert "HEAD_NOW_MISMATCH" in issues


def test_missing_pinned_file_marks_mismatch(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    rows = hash_pinned_files(repo, [PinnedFile("probe", "scripts/missing.py")])
    assert rows[0]["missing"] is True


def test_remote_sha_mismatch_exit_11(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("print('ok')\n", encoding="utf-8")
    rows = hash_pinned_files(repo, [PinnedFile("probe", str(rel))])

    def bad_remote_sha(_box: str, _remote_rel: str) -> str:
        return "0" * 64

    mismatches, _ = sync_pinned_files(
        repo_root=repo,
        remote_repo="/remote/repo",
        box="box",
        pinned_rows=rows,
        rsync_runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
        remote_sha_runner=bad_remote_sha,
    )
    assert any(m.startswith("remote_sha_mismatch:") for m in mismatches)


def test_dry_run_avoids_ssh_and_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        lambda _root, *args: "deadbeef" if args == ("rev-parse", "HEAD") else "deadbeef",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network command invoked during dry-run")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            "deadbeef",
            "--dry-run",
            "--skip-fetch",
            "--output",
            str(manifest_path),
            "--pinned-manifest",
            str(_write_pinned_manifest(tmp_path, rel)),
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["dry_run"] is True
    assert payload["sync_requested"] is False
    assert payload["remote_repo_root"] == "/home/gabe/claw-code-hrm-158"
    assert payload["code_currency_pass"] is True


def test_default_without_sync_never_networks(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        lambda _root, *args: "deadbeef",
    )

    def fail_if_called(*_args, **_kwargs):
        raise AssertionError("network command invoked without --sync")

    monkeypatch.setattr(subprocess, "run", fail_if_called)
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            "deadbeef",
            "--skip-fetch",
            "--output",
            str(manifest_path),
            "--pinned-manifest",
            str(_write_pinned_manifest(tmp_path, rel)),
        ]
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["sync_requested"] is False
    assert payload["dry_run"] is False


def test_artifact_rsync_mismatch_exit_12() -> None:
    mismatches = verify_artifact_manifest(
        [
            {
                "role": "capture_receipt",
                "producer_sha256": "a" * 64,
                "consumer_sha256": "b" * 64,
            }
        ]
    )
    assert mismatches
    assert EXIT_ARTIFACT_RSYNC_MISMATCH == 12


def test_serial_overlap_watcher_exit_nonzero(tmp_path: Path) -> None:
    log = tmp_path / "producer.log"
    log.write_text(
        "\n".join(
            [
                "capture_complete: chain_id=c1 code_currency_pass=true artifact_sha_verified=true ts=1.0",
                "producer_next_capture_start: chain_id=c1 ts=2.0",
                "consumer_audit_start: chain_id=c1 ts=3.0",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "overlap.json"
    rc = watcher_main([str(log), "--manifest", str(manifest)])
    assert rc == EXIT_OVERLAP_FAILURE
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_flagged"] == 1


def test_overlap_earned_when_consumer_starts_before_next_capture(tmp_path: Path) -> None:
    log = tmp_path / "producer.log"
    log.write_text(
        "\n".join(
            [
                "capture_complete: chain_id=c1 code_currency_pass=true artifact_sha_verified=true ts=1.0",
                "consumer_audit_start: chain_id=c1 ts=1.5",
                "producer_next_capture_start: chain_id=c1 ts=2.0",
                "consumer_terminal: chain_id=c1 status=pass",
            ]
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "overlap.json"
    rc = watcher_main([str(log), "--manifest", str(manifest)])
    assert rc == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_overlap"] == 1
    entry = payload["entries"][0]
    assert entry["status"] == "OVERLAP"
    assert entry["overlap_seconds"] == pytest.approx(0.5)
    assert entry["pipeline_eligible"] is True
    assert entry["verdict_eligible"] is True


def test_cpu_receipt_rejects_hot_loop_laundering() -> None:
    issues = validate_receipt_residency(
        {
            "compute_lane": "cpu_trace_analytics",
            "device_residency_claim": True,
            "hot_loop_residency_claim": False,
            "device_residency_not_hot_loop_residency": True,
        }
    )
    assert "cpu_phase_claims_device_or_hot_loop" in issues


def test_cuda_hot_loop_requires_native_kernelized_hot_path_receipt() -> None:
    issues = validate_receipt_residency(
        {
            "compute_lane": "cuda_probe",
            "device_residency_claim": True,
            "hot_loop_residency_claim": True,
            "device_residency_not_hot_loop_residency": True,
        }
    )
    assert "hot_loop_claim_without_native_kernelized_hot_path_receipt" in issues

    ok = validate_receipt_residency(
        {
            "compute_lane": "cuda_probe",
            "device_residency_claim": True,
            "hot_loop_residency_claim": True,
            "native_kernelized_hot_path_receipt": "chain_a/native_kernelized_hot_path_receipt.json",
            "device_residency_not_hot_loop_residency": True,
        }
    )
    assert not ok


def test_n_plus_one_quarantine_on_consumer_failure() -> None:
    lines = [
        "capture_complete: chain_id=c1 code_currency_pass=true artifact_sha_verified=true ts=1.0",
        "consumer_audit_start: chain_id=c1 ts=1.2",
        "producer_next_capture_start: chain_id=c2 ts=1.5",
        "consumer_terminal: chain_id=c1 status=fail",
    ]
    states = process_science_chain_log(lines)
    assert states["c2"].quarantined is True
    verdict = classify_overlap(states["c2"])
    assert verdict.status == "QUARANTINED_AFTER_CONSUMER_FAIL"
    assert verdict.verdict_eligible is False
    assert verdict.pipeline_eligible is False


def test_preflight_missing_file_returns_exit_11(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    manifest_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        lambda _root, *args: "deadbeef",
    )
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            "deadbeef",
            "--dry-run",
            "--skip-fetch",
            "--output",
            str(manifest_path),
            "--pinned-manifest",
            str(_write_pinned_manifest(tmp_path, Path("scripts/missing.py"))),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH


def _write_pinned_manifest(tmp_path: Path, rel: Path) -> Path:
    manifest = tmp_path / "pinned.json"
    manifest.write_text(
        json.dumps({"files": [{"role": "probe", "rel_path": str(rel).replace("\\", "/")}]}),
        encoding="utf-8",
    )
    return manifest


def test_build_manifest_records_chain_roots(tmp_path: Path) -> None:
    local_root = tmp_path / "creditdir" / "chain_a"
    remote_root = tmp_path / "creditdir" / "chain_a"
    manifest = build_code_currency_manifest(
        chain_id="chain_a",
        head_expected="abc",
        head_now="abc",
        fetch_head="abc",
        pinned_rows=[],
        dry_run=True,
        sync_requested=False,
        local_chain_root=local_root,
        remote_chain_root=remote_root,
        remote_repo_root="/home/gabe/claw-code-hrm-158",
        mismatches=[],
    )
    assert manifest["code_currency_pass"] is True
    assert manifest["local_chain_root"].endswith("chain_a")
    assert manifest["remote_repo_root"] == "/home/gabe/claw-code-hrm-158"


def test_chain_id_validation_allowlist_rejects_invalid_basenames() -> None:
    with pytest.raises(ValueError):
        validate_chain_id("")
    with pytest.raises(ValueError):
        validate_chain_id("   ")
    with pytest.raises(ValueError):
        validate_chain_id("foo/bar")
    with pytest.raises(ValueError):
        validate_chain_id("foo\\bar")
    with pytest.raises(ValueError):
        validate_chain_id(".hidden")
    with pytest.raises(ValueError):
        validate_chain_id("bad id")
    validate_chain_id("chain_fixture")
    validate_chain_id("stage_c_seed2_n10_20260612T125234Z")
