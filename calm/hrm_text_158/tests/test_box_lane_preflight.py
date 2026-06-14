"""CPU tests for box-lane code-currency preflight and science-chain watcher (C6 + V*)."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.box_lane import (
    EXIT_ARTIFACT_RSYNC_MISMATCH,
    EXIT_CODE_CURRENCY_MISMATCH,
    EXIT_OK,
    EXIT_OVERLAP_FAILURE,
    PinnedFile,
    build_code_currency_manifest,
    check_pinned_paths_clean,
    classify_overlap,
    hash_pinned_files,
    process_science_chain_log,
    sha256_file,
    sync_pinned_files,
    validate_chain_id,
    validate_receipt_residency,
    verify_artifact_manifest,
    verify_head_triple,
    verify_pinned_sha_expectations,
)
from scripts.box_lane_chain_watcher import main as watcher_main
from scripts.box_lane_code_currency_preflight import main as preflight_main


def _mock_git_clean_revparse(_root: Path, *args: str) -> str:
    if args and args[0] == "rev-parse":
        return "deadbeef"
    return ""


def test_fetch_head_mismatch_fails() -> None:
    issues = verify_head_triple(
        head_now="abc",
        fetch_head="def",
        head_expected="abc",
    )
    assert "FETCH_HEAD_MISMATCH" in issues


def test_fetch_head_mismatch_skipped_when_not_required() -> None:
    issues = verify_head_triple(
        head_now="abc",
        fetch_head="def",
        head_expected="abc",
        require_fetch_head=False,
    )
    assert issues == []


def test_head_now_mismatch_fails() -> None:
    issues = verify_head_triple(
        head_now="abc",
        fetch_head="abc",
        head_expected="def",
    )
    assert "HEAD_NOW_MISMATCH" in issues


def test_skip_fetch_passes_with_stale_fetch_head_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, _ = _write_pinned_manifest_with_sha(tmp_path, rel)
    repo = tmp_path / "repo"
    out_path = tmp_path / "out.json"
    head = "19cd9f355f8bc6d00d9098179871dac451bf4aba"
    stale_fetch = "ba0ecb05deadbeefdeadbeefdeadbeefdeadbeef"

    def git_revparse(_root: Path, *args: str) -> str:
        if args[0] == "rev-parse" and args[1] == "HEAD":
            return head
        if args[0] == "rev-parse" and args[1] == "FETCH_HEAD":
            return stale_fetch
        return ""

    monkeypatch.setattr("scripts.box_lane_code_currency_preflight.run_git", git_revparse)
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            head,
            "--skip-fetch",
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_OK
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is True
    assert payload["remote_currency_check"] == "skipped_local_only"
    assert payload["head_now"] == head
    assert payload["fetch_head"] == stale_fetch
    assert "FETCH_HEAD_MISMATCH" not in payload["mismatches"]


def test_strict_mode_fails_with_stale_fetch_head_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, _ = _write_pinned_manifest_with_sha(tmp_path, rel)
    repo = tmp_path / "repo"
    out_path = tmp_path / "out.json"
    head = "19cd9f355f8bc6d00d9098179871dac451bf4aba"
    stale_fetch = "ba0ecb05deadbeefdeadbeefdeadbeefdeadbeef"

    def git_revparse(_root: Path, *args: str) -> str:
        if args[0] == "rev-parse" and args[1] == "HEAD":
            return head
        if args[0] == "rev-parse" and args[1] == "FETCH_HEAD":
            return stale_fetch
        return ""

    def fake_fetch(*_args, **_kwargs):
        return subprocess.CompletedProcess([], 0, "", "")

    monkeypatch.setattr("scripts.box_lane_code_currency_preflight.run_git", git_revparse)
    monkeypatch.setattr(subprocess, "run", fake_fetch)
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            head,
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is False
    assert payload["remote_currency_check"] == "enforced"
    assert "FETCH_HEAD_MISMATCH" in payload["mismatches"]


def test_skip_fetch_still_fails_head_now_mismatch_e2e(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, _ = _write_pinned_manifest_with_sha(tmp_path, rel)
    repo = tmp_path / "repo"
    out_path = tmp_path / "out.json"
    head_expected = "19cd9f355f8bc6d00d9098179871dac451bf4aba"
    head_now = "abb55357ce412dc30df5364cb488f4ea94ac5c49"

    def git_revparse(_root: Path, *args: str) -> str:
        if args[0] == "rev-parse" and args[1] == "HEAD":
            return head_now
        if args[0] == "rev-parse" and args[1] == "FETCH_HEAD":
            return "ba0ecb05deadbeefdeadbeefdeadbeefdeadbeef"
        return ""

    monkeypatch.setattr("scripts.box_lane_code_currency_preflight.run_git", git_revparse)
    rc = preflight_main(
        [
            "--repo-root",
            str(repo),
            "--chain-id",
            "chain_fixture",
            "--creditdir",
            str(tmp_path / "creditdir"),
            "--head-expected",
            head_expected,
            "--skip-fetch",
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is False
    assert payload["remote_currency_check"] == "skipped_local_only"
    assert "HEAD_NOW_MISMATCH" in payload["mismatches"]
    assert "FETCH_HEAD_MISMATCH" not in payload["mismatches"]


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


def test_sync_pinned_files_includes_mkpath(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    rows = hash_pinned_files(repo, [PinnedFile("probe", str(rel))])
    seen_cmds: list[list[str]] = []

    def capture_rsync(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    sync_pinned_files(
        repo_root=repo,
        remote_repo="/remote/repo",
        box="box",
        pinned_rows=rows,
        rsync_runner=capture_rsync,
        remote_sha_runner=lambda _box, _remote_rel: str(rows[0]["producer_sha256"]),
    )
    assert seen_cmds
    assert "--mkpath" in seen_cmds[0]


def test_sync_pinned_files_rsync_transport_failure_captured(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    rows = hash_pinned_files(repo, [PinnedFile("probe", str(rel))])

    def fail_rsync(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(3, cmd)

    mismatches, synced_rows = sync_pinned_files(
        repo_root=repo,
        remote_repo="/remote/repo",
        box="box",
        pinned_rows=rows,
        rsync_runner=fail_rsync,
        remote_sha_runner=lambda _box, _remote_rel: "0" * 64,
    )
    assert mismatches == [f"rsync_transport_failure:{rel.as_posix()}"]
    assert synced_rows[0]["rsync_ok"] is False
    assert synced_rows[0]["rsync_exit_code"] == 3
    assert "--mkpath" in synced_rows[0]["rsync_cmd"]


def test_preflight_rsync_transport_failure_writes_receipt_exit_11(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        _mock_git_clean_revparse,
    )
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.probe_rsync_version",
        lambda: "rsync  version 3.2.7",
    )

    def fail_rsync(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(3, cmd)

    monkeypatch.setattr("scripts.box_lane_code_currency_preflight._default_rsync_runner", fail_rsync)
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
            "--sync",
            "--output",
            str(manifest_path),
            "--pinned-manifest",
            str(_write_pinned_manifest(tmp_path, rel)),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is False
    assert payload["sync_requested"] is True
    assert payload["rsync_version"] == "rsync  version 3.2.7"
    assert any(m.startswith("rsync_transport_failure:") for m in payload["mismatches"])
    file_row = payload["files"][0]
    assert file_row["rsync_ok"] is False
    assert file_row["rsync_exit_code"] == 3
    assert "--mkpath" in file_row["rsync_cmd"]


def test_dry_run_avoids_ssh_and_rsync(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True)
    path.write_text("x", encoding="utf-8")
    manifest_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        _mock_git_clean_revparse,
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
        _mock_git_clean_revparse,
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


def _write_preflight_code_currency(
    chain_root: Path,
    *,
    code_currency_pass: bool,
    payload: dict | None = None,
) -> Path:
    path = chain_root / "prelaunch" / "box_code_currency_preflight.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    body = dict(payload or {})
    body.setdefault("code_currency_pin_count", 27)
    body["code_currency_pass"] = code_currency_pass
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _append_overlap_sequence(chain_log: Path) -> None:
    with chain_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n".join(
                [
                    "consumer_audit_start: chain_id=chain_a ts=1.5",
                    "producer_next_capture_start: chain_id=chain_a ts=2.0",
                    "consumer_terminal: chain_id=chain_a status=pass",
                ]
            )
            + "\n"
        )


def test_transport_reads_preflight_true_enables_pipeline_eligible(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main
    from scripts.box_lane_chain_watcher import main as watcher_main

    creditdir = tmp_path / "creditdir"
    chain_id = "chain_a"
    chain_root = creditdir / chain_id
    _write_s2_chain_tree(chain_root)
    _write_preflight_code_currency(chain_root, code_currency_pass=True)
    chain_log = tmp_path / "producer.log"

    rc = transport_main(
        [
            "--chain-id",
            chain_id,
            "--creditdir",
            str(creditdir),
            "--chain-log",
            str(chain_log),
        ]
    )
    assert rc == 0
    assert "code_currency_pass=true" in chain_log.read_text(encoding="utf-8")

    _append_overlap_sequence(chain_log)
    manifest = tmp_path / "overlap.json"
    rc = watcher_main([str(chain_log), "--manifest", str(manifest)])
    assert rc == 0
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["n_overlap"] == 1
    assert payload["entries"][0]["pipeline_eligible"] is True


def test_transport_reads_preflight_false_marks_ineligible(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main

    creditdir = tmp_path / "creditdir"
    chain_id = "chain_a"
    chain_root = creditdir / chain_id
    _write_s2_chain_tree(chain_root)
    _write_preflight_code_currency(chain_root, code_currency_pass=False)
    chain_log = tmp_path / "producer.log"

    rc = transport_main(
        [
            "--chain-id",
            chain_id,
            "--creditdir",
            str(creditdir),
            "--chain-log",
            str(chain_log),
        ]
    )
    assert rc == 0
    assert "code_currency_pass=false" in chain_log.read_text(encoding="utf-8")

    states = process_science_chain_log(chain_log.read_text(encoding="utf-8").splitlines())
    verdict = classify_overlap(states["chain_a"])
    assert "code_currency_not_passed" in verdict.issues
    assert verdict.pipeline_eligible is False


def test_transport_missing_preflight_fails_closed_for_chain_log(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main

    creditdir = tmp_path / "creditdir"
    chain_id = "chain_a"
    chain_root = creditdir / chain_id
    _write_s2_chain_tree(chain_root)
    chain_log = tmp_path / "producer.log"

    rc = transport_main(
        [
            "--chain-id",
            chain_id,
            "--creditdir",
            str(creditdir),
            "--chain-log",
            str(chain_log),
        ]
    )
    assert rc == 0
    assert "code_currency_pass=false" in chain_log.read_text(encoding="utf-8")

    states = process_science_chain_log(chain_log.read_text(encoding="utf-8").splitlines())
    verdict = classify_overlap(states["chain_a"])
    assert "code_currency_not_passed" in verdict.issues
    assert verdict.pipeline_eligible is False


def test_transport_malformed_preflight_fails_closed_for_chain_log(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main

    creditdir = tmp_path / "creditdir"
    chain_id = "chain_a"
    chain_root = creditdir / chain_id
    _write_s2_chain_tree(chain_root)
    malformed = chain_root / "prelaunch" / "box_code_currency_preflight.json"
    malformed.parent.mkdir(parents=True, exist_ok=True)
    malformed.write_text("{not json", encoding="utf-8")
    chain_log = tmp_path / "producer.log"

    rc = transport_main(
        [
            "--chain-id",
            chain_id,
            "--creditdir",
            str(creditdir),
            "--chain-log",
            str(chain_log),
        ]
    )
    assert rc == 0
    assert "code_currency_pass=false" in chain_log.read_text(encoding="utf-8")

    states = process_science_chain_log(chain_log.read_text(encoding="utf-8").splitlines())
    verdict = classify_overlap(states["chain_a"])
    assert "code_currency_not_passed" in verdict.issues
    assert verdict.pipeline_eligible is False


def test_transport_only_chain_log_is_ineligible_without_terminal_pass(tmp_path: Path) -> None:
    log = tmp_path / "producer.log"
    log.write_text(
        "\n".join(
            [
                "capture_complete: chain_id=c1 code_currency_pass=true artifact_sha_verified=true ts=1.0",
                "producer_next_capture_start: chain_id=c1 ts=2.0",
            ],
        ),
        encoding="utf-8",
    )
    states = process_science_chain_log(log.read_text(encoding="utf-8").splitlines())
    verdict = classify_overlap(states["c1"])
    assert verdict.status == "INELIGIBLE"
    assert "consumer_not_started" in verdict.issues
    assert "consumer_terminal_not_pass" in verdict.issues
    assert verdict.pipeline_eligible is False


def test_stray_consumer_audit_start_without_terminal_pass_is_ineligible() -> None:
    lines = [
        "capture_complete: chain_id=c1 code_currency_pass=true artifact_sha_verified=true ts=1.0",
        "consumer_audit_start: chain_id=c1 ts=1.5",
        "producer_next_capture_start: chain_id=c1 ts=2.0",
    ]
    states = process_science_chain_log(lines)
    verdict = classify_overlap(states["c1"])
    assert verdict.status == "INELIGIBLE"
    assert "consumer_terminal_not_pass" in verdict.issues
    assert verdict.pipeline_eligible is False
    assert verdict.verdict_eligible is False


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
        _mock_git_clean_revparse,
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


def _s2_good_receipt() -> dict:
    """Minimal receipt passing pressure_shape preflight (v1 signed shape)."""
    step_reports = {}
    for step in range(3, 11):
        step_reports[str(step)] = {
            "vote_pressure": {
                "mod.a": {
                    "state_key": "mod.a",
                    "vote_positive_count": 1,
                    "vote_negative_count": 1,
                    "pressure_shape_summary": {
                        "schema": "hrm_text_158_pressure_shape_summary/v1",
                        "rank_method": "grouped_bisect_right",
                        "rank_bins": [],
                        "bin_occupancy_count": [50, 50],
                        "bin_mass_fraction": [0.5, 0.5],
                        "candidate_count": 100,
                        "raw_per_proposal_arrays_included": False,
                        "signed_rank_bin_mass": {
                            "schema": "hrm_text_158_signed_rank_bin_mass/v0",
                            "pos_bin_fraction": [0.25, 0.25],
                            "neg_bin_fraction": [0.25, 0.25],
                            "signed_bin_net_fraction": [0.0, 0.0],
                            "total_abs_vote_mass": 1.0,
                            "telemetry_only_net_fraction": True,
                        },
                        "counterfactual_signed_rank_bin_mass": {
                            "a1_order_matched": {
                                "schema": "hrm_text_158_signed_rank_bin_mass/v0",
                                "pos_bin_fraction": [0.25, 0.25],
                                "neg_bin_fraction": [0.25, 0.25],
                                "signed_bin_net_fraction": [0.0, 0.0],
                                "total_abs_vote_mass": 1.0,
                                "telemetry_only_net_fraction": True,
                            },
                            "order_matched_basis": "a1_emitted",
                        },
                    },
                },
            },
            "loss": float(step),
        }
    return {
        "step_reports": step_reports,
        "terminal_status": {"status": "complete"},
    }


def _write_s2_chain_tree(chain_root: Path) -> None:
    receipt = _s2_good_receipt()
    for label in ("S44", "S44_iso43", "S43"):
        for arm in ("on", "off"):
            path = chain_root / label / arm / "receipt.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt), encoding="utf-8")


def test_default_consensus_chain_artifacts_expands_concrete_per_arm_rows() -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import default_consensus_chain_artifacts

    artifacts = default_consensus_chain_artifacts()
    receipt_roles = [a.role for a in artifacts if a.rel_path.endswith("receipt.json")]
    assert receipt_roles == [
        "S44_on_receipt",
        "S44_off_receipt",
        "S44_iso43_on_receipt",
        "S44_iso43_off_receipt",
        "S43_on_receipt",
        "S43_off_receipt",
    ]
    run_log_roles = [a.role for a in artifacts if a.rel_path.endswith("run.log")]
    assert run_log_roles == [
        "S44_on_run_log",
        "S44_off_run_log",
        "S44_iso43_on_run_log",
        "S44_iso43_off_run_log",
        "S43_on_run_log",
        "S43_off_run_log",
    ]
    assert len(artifacts) == 12
    assert all(a.optional for a in artifacts if a.rel_path.endswith("run.log"))
    assert not any(a.optional for a in artifacts if a.rel_path.endswith("receipt.json"))


def test_sync_chain_arm_artifacts_includes_mkpath(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain_a"
    rel = "S44/on/receipt.json"
    path = chain_root / rel
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    from calm.hrm_text_158.native_full_stack.box_lane import (
        ChainArtifact,
        sha256_file,
        sync_chain_arm_artifacts,
    )

    seen_cmds: list[list[str]] = []

    def capture_rsync(cmd: list[str]) -> subprocess.CompletedProcess[str]:
        seen_cmds.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    sync_chain_arm_artifacts(
        local_chain_root=chain_root,
        remote_chain_root="/remote/chain_a",
        box="box",
        artifacts=[ChainArtifact("S44_on_receipt", rel)],
        rsync_runner=capture_rsync,
        remote_sha_runner=lambda _box, _remote_rel: sha256_file(path),
    )
    assert seen_cmds
    assert "--mkpath" in seen_cmds[0]


def test_sync_chain_arm_artifacts_sha_mismatch_fail_closed(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain_a"
    rel = "S44/on/receipt.json"
    path = chain_root / rel
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    from calm.hrm_text_158.native_full_stack.box_lane import (
        ChainArtifact,
        sync_chain_arm_artifacts,
    )

    mismatches, rows = sync_chain_arm_artifacts(
        local_chain_root=chain_root,
        remote_chain_root="/remote/chain_a",
        box="box",
        artifacts=[ChainArtifact("S44_on_receipt", rel)],
        rsync_runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
        remote_sha_runner=lambda _box, _remote_rel: "0" * 64,
    )
    assert any(m.startswith("artifact_sha_mismatch:") for m in mismatches)
    assert rows[0]["rsync_ok"] is False


def test_consumer_audit_passes_pressure_shape_preflight(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain_a"
    _write_s2_chain_tree(chain_root)
    from calm.hrm_text_158.native_full_stack.box_lane import audit_consensus_bounded_delta_consumer

    audit = audit_consensus_bounded_delta_consumer(chain_root)
    assert audit["pass"] is True
    assert audit["consumer_scope"] == "consensus_bounded_delta_receipt_audit"


def test_consumer_audit_fails_pressure_shape_preflight(tmp_path: Path) -> None:
    chain_root = tmp_path / "chain_a"
    _write_s2_chain_tree(chain_root)
    bad = _s2_good_receipt()
    del bad["step_reports"]["3"]["vote_pressure"]["mod.a"]["pressure_shape_summary"]
    (chain_root / "S44" / "on" / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")
    from calm.hrm_text_158.native_full_stack.box_lane import audit_consensus_bounded_delta_consumer

    audit = audit_consensus_bounded_delta_consumer(chain_root)
    assert audit["pass"] is False
    assert any("pressure_shape_preflight_fail" in issue for issue in audit["issues"])


def test_artifact_transport_local_mode_hashes_without_sync(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main

    creditdir = tmp_path / "creditdir"
    chain_root = creditdir / "chain_a"
    _write_s2_chain_tree(chain_root)
    manifest_path = chain_root / "box_artifact_transport.json"
    rc = transport_main(
        [
            "--chain-id",
            "chain_a",
            "--creditdir",
            str(creditdir),
            "--output",
            str(manifest_path),
            "--code-currency-pass",
        ],
    )
    assert rc == 0
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["sync_requested"] is False
    assert payload["artifact_transport_pass"] is True


def test_artifact_transport_emits_capture_complete_only(tmp_path: Path) -> None:
    from scripts.box_lane_artifact_transport import main as transport_main

    creditdir = tmp_path / "creditdir"
    chain_root = creditdir / "chain_a"
    _write_s2_chain_tree(chain_root)
    chain_log = tmp_path / "producer.log"
    rc = transport_main(
        [
            "--chain-id",
            "chain_a",
            "--creditdir",
            str(creditdir),
            "--chain-log",
            str(chain_log),
            "--code-currency-pass",
        ],
    )
    assert rc == 0
    log_text = chain_log.read_text(encoding="utf-8")
    assert "capture_complete:" in log_text
    assert "consumer_audit_start:" not in log_text
    assert "consumer_terminal:" not in log_text


def test_consumer_audit_fails_on_missing_non_optional_transport_row(tmp_path: Path) -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import audit_consensus_bounded_delta_consumer

    chain_root = tmp_path / "chain_a"
    _write_s2_chain_tree(chain_root)
    transport_rows = [
        {
            "role": "S44_on_receipt",
            "rel_path": "S44/on/receipt.json",
            "optional": False,
            "missing": True,
        },
    ]
    audit = audit_consensus_bounded_delta_consumer(chain_root, transport_artifacts=transport_rows)
    assert audit["pass"] is False
    assert any("transport_missing_required_artifact:" in issue for issue in audit["issues"])


def test_missing_non_optional_transport_row_emits_terminal_fail_and_quarantines(
    tmp_path: Path,
) -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import (
        CONSUMER_TERMINAL_RE,
        process_science_chain_log,
    )
    from scripts.box_lane_consensus_consumer_audit import main as consumer_main

    chain_root = tmp_path / "chain_a"
    _write_s2_chain_tree(chain_root)
    transport_manifest = chain_root / "box_artifact_transport.json"
    transport_manifest.write_text(
        json.dumps(
            {
                "artifacts": [
                    {
                        "role": "S44_on_receipt",
                        "rel_path": "S44/on/receipt.json",
                        "optional": False,
                        "missing": True,
                    },
                ],
            },
        ),
        encoding="utf-8",
    )
    chain_log = tmp_path / "producer.log"
    chain_log.write_text(
        "\n".join(
            [
                "capture_complete: chain_id=chain_a code_currency_pass=true artifact_sha_verified=true ts=1.0",
                "producer_next_capture_start: chain_id=chain_b ts=1.5",
            ],
        ),
        encoding="utf-8",
    )
    rc = consumer_main(
        [
            "--chain-root",
            str(chain_root),
            "--chain-id",
            "chain_a",
            "--transport-manifest",
            str(transport_manifest),
            "--chain-log",
            str(chain_log),
        ],
    )
    assert rc == EXIT_ARTIFACT_RSYNC_MISMATCH
    terminal_match = CONSUMER_TERMINAL_RE.search(chain_log.read_text(encoding="utf-8"))
    assert terminal_match is not None
    assert terminal_match.group("status") == "fail"
    states = process_science_chain_log(chain_log.read_text(encoding="utf-8").splitlines())
    assert states["chain_b"].quarantined is True


def test_consumer_audit_script_emits_terminal_fail_and_quarantines_pending_arm(
    tmp_path: Path,
) -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import (
        CONSUMER_TERMINAL_RE,
        process_science_chain_log,
    )
    from scripts.box_lane_consensus_consumer_audit import main as consumer_main

    chain_root = tmp_path / "chain_a"
    _write_s2_chain_tree(chain_root)
    bad = _s2_good_receipt()
    del bad["step_reports"]["3"]["vote_pressure"]["mod.a"]["pressure_shape_summary"]
    (chain_root / "S44" / "on" / "receipt.json").write_text(json.dumps(bad), encoding="utf-8")
    chain_log = tmp_path / "producer.log"
    chain_log.write_text(
        "\n".join(
            [
                "capture_complete: chain_id=chain_a code_currency_pass=true artifact_sha_verified=true ts=1.0",
                "producer_next_capture_start: chain_id=chain_b ts=1.5",
            ],
        ),
        encoding="utf-8",
    )
    rc = consumer_main(
        [
            "--chain-root",
            str(chain_root),
            "--chain-id",
            "chain_a",
            "--chain-log",
            str(chain_log),
        ],
    )
    assert rc == EXIT_ARTIFACT_RSYNC_MISMATCH
    log_text = chain_log.read_text(encoding="utf-8")
    assert "consumer_audit_start: chain_id=chain_a" in log_text
    terminal_match = CONSUMER_TERMINAL_RE.search(log_text)
    assert terminal_match is not None
    assert terminal_match.group("status") == "fail"
    states = process_science_chain_log(log_text.splitlines())
    assert states["chain_b"].quarantined is True


def _write_consensus_k3_chain_tree(chain_root: Path) -> None:
    receipt = _s2_good_receipt()
    for label in ("S44_ord44", "S44_ord43", "S44_ord17"):
        for arm in ("on", "off"):
            path = chain_root / label / arm / "receipt.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(receipt), encoding="utf-8")


def test_consensus_consumer_audit_fails_missing_third_label(tmp_path: Path) -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import audit_consensus_bounded_delta_consumer

    chain_root = tmp_path / "chain_a"
    _write_consensus_k3_chain_tree(chain_root)
    (chain_root / "S44_ord17").rename(chain_root / "S44_ord17_missing")
    audit = audit_consensus_bounded_delta_consumer(
        chain_root,
        primary_label="S44_ord44",
        isolation_label="S44_ord43",
        corroboration_label="S44_ord17",
        consensus_mode=True,
    )
    assert audit["pass"] is False
    assert any("missing_receipt:" in issue for issue in audit["issues"])


def test_consensus_consumer_audit_passes_all_k_labels(tmp_path: Path) -> None:
    from calm.hrm_text_158.native_full_stack.box_lane import audit_consensus_bounded_delta_consumer

    chain_root = tmp_path / "chain_a"
    _write_consensus_k3_chain_tree(chain_root)
    audit = audit_consensus_bounded_delta_consumer(
        chain_root,
        primary_label="S44_ord44",
        isolation_label="S44_ord43",
        corroboration_label="S44_ord17",
        consensus_mode=True,
    )
    assert audit["pass"] is True


def _write_pinned_manifest_with_sha(tmp_path: Path, rel: Path, *, content: str = "x\n") -> tuple[Path, str]:
    repo = tmp_path / "repo"
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    expected_sha = sha256_file(path)
    manifest = tmp_path / "pinned.json"
    manifest.write_text(
        json.dumps(
            {
                "files": [
                    {
                        "role": "probe",
                        "rel_path": str(rel).replace("\\", "/"),
                        "sha256": expected_sha,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest, expected_sha


def test_dirty_pinned_file_fails_pin_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, expected_sha = _write_pinned_manifest_with_sha(tmp_path, rel, content="clean\n")
    repo = tmp_path / "repo"
    (repo / rel).write_text("dirty\n", encoding="utf-8")
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        _mock_git_clean_revparse,
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
            "--skip-fetch",
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is False
    assert any(m.startswith("producer_pin_mismatch:") for m in payload["mismatches"])
    row = payload["files"][0]
    assert row["producer_matches_expected"] is False
    assert "remote_matches_expected" not in row


def test_dirty_pinned_path_git_porcelain_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, _ = _write_pinned_manifest_with_sha(tmp_path, rel)
    repo = tmp_path / "repo"
    out_path = tmp_path / "out.json"

    def dirty_git(_root: Path, *args: str) -> str:
        if args[0] == "rev-parse":
            return "deadbeef"
        if args[0] == "status":
            return " M scripts/probe.py\n"
        return ""

    monkeypatch.setattr("scripts.box_lane_code_currency_preflight.run_git", dirty_git)
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
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_CODE_CURRENCY_MISMATCH
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert any(m.startswith("pinned_path_dirty:") for m in payload["mismatches"])


def test_all_match_passes_with_expected_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rel = Path("scripts/probe.py")
    manifest_path, expected_sha = _write_pinned_manifest_with_sha(tmp_path, rel)
    repo = tmp_path / "repo"
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        _mock_git_clean_revparse,
    )

    def good_remote(_box: str, _remote_rel: str) -> str:
        return expected_sha

    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight._default_remote_sha_runner",
        good_remote,
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
            "--skip-fetch",
            "--sync",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(manifest_path),
        ]
    )
    assert rc == EXIT_OK
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is True
    assert payload["pin_enforcement"] is True
    row = payload["files"][0]
    assert row["producer_matches_expected"] is True
    assert row["remote_matches_expected"] is True
    assert row["match"] is True


def test_legacy_manifest_no_sha_still_passes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    out_path = tmp_path / "out.json"
    monkeypatch.setattr(
        "scripts.box_lane_code_currency_preflight.run_git",
        _mock_git_clean_revparse,
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
            "--skip-fetch",
            "--dry-run",
            "--output",
            str(out_path),
            "--pinned-manifest",
            str(_write_pinned_manifest(tmp_path, rel)),
        ]
    )
    assert rc == EXIT_OK
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["code_currency_pass"] is True
    assert "pin_enforcement" not in payload
    assert payload["files"][0].get("expected_sha256") is None


def test_remote_pin_mismatch_after_rsync(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    rel = Path("scripts/probe.py")
    path = repo / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("x", encoding="utf-8")
    expected = sha256_file(path)
    rows = hash_pinned_files(repo, [PinnedFile("probe", str(rel), expected_sha256=expected)])

    def bad_remote(_box: str, _remote_rel: str) -> str:
        return "0" * 64

    mismatches, synced = sync_pinned_files(
        repo_root=repo,
        remote_repo="/remote/repo",
        box="box",
        pinned_rows=rows,
        rsync_runner=lambda cmd: subprocess.CompletedProcess(cmd, 0, "", ""),
        remote_sha_runner=bad_remote,
    )
    assert any(m.startswith("remote_pin_mismatch:") for m in mismatches)
    assert synced[0]["remote_matches_expected"] is False


def test_verify_pinned_sha_expectations_unit(tmp_path: Path) -> None:
    rel = "scripts/probe.py"
    path = tmp_path / rel
    path.parent.mkdir(parents=True)
    path.write_text("good", encoding="utf-8")
    good_sha = sha256_file(path)
    rows = hash_pinned_files(tmp_path, [PinnedFile("probe", rel, expected_sha256=good_sha)])
    assert verify_pinned_sha_expectations(rows) == []
    path.write_text("bad", encoding="utf-8")
    rows = hash_pinned_files(tmp_path, [PinnedFile("probe", rel, expected_sha256=good_sha)])
    issues = verify_pinned_sha_expectations(rows)
    assert issues == [f"producer_pin_mismatch:{rel}"]


def test_check_pinned_paths_clean_injectable_git_runner() -> None:
    pinned = [PinnedFile("probe", "scripts/probe.py", expected_sha256="abc")]
    issues = check_pinned_paths_clean(
        Path("/tmp"),
        pinned,
        git_runner=lambda _root, *args: " M scripts/probe.py\n",
    )
    assert issues == ["pinned_path_dirty:scripts/probe.py"]
