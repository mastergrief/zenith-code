"""CPU-static hostile PHASE_B_PACKET matrix (12 cases)."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from scripts.p1b_phase_b_packet_mint import (
    RUNTIME_EXECUTABLE_KEYS,
    compute_packet_payload_digest,
    mint_phase_b_packet_o_excl,
)
from scripts.p1b_phaseB_supervisor import (
    EXIT_PACKET_ARGV,
    EXIT_PACKET_BUDGET_PATH,
    EXIT_PACKET_COMMIT,
    EXIT_PACKET_FILE_HASH,
    EXIT_PACKET_FILE_SHA,
    EXIT_PACKET_MISSING,
    EXIT_PACKET_PAYLOAD,
    EXIT_PACKET_SCHEMA,
    EXIT_PACKET_SHA_ARG,
    FROZEN_INNER_SCIENCE_ARGV,
    validate_packet_pre_spawn,
)

ISOL = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")


def _head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ISOL, text=True).strip()


def _runtime_hashes() -> dict[str, str]:
    return {
        rel: hashlib.sha256((ISOL / rel).read_bytes()).hexdigest()
        for rel in RUNTIME_EXECUTABLE_KEYS
    }


def _base_fields(tmp_path: Path) -> dict:
    runtime = _runtime_hashes()
    return {
        "plan_id": "p1b_test",
        "plan_revision": "v15",
        "plan_sha256": "b" * 64,
        "gate_msg_ids": {
            "plus1_implement": "g1",
            "implement_gate1": "g2",
            "implement_gate2": "g3",
            "plus1_commit": "g4",
        },
        "commit_sha": _head(),
        "runtime_executable_sha256s": runtime,
        "watch_wrap_sha256": runtime["bin/watch-wrap"],
        "watch_wrap_command_exact": 'bin/watch-wrap --log <path> --heartbeat 30 --error "Traceback|Error|Killed|OOM|FAILED|assert" --progress "[P1B_PHASE]" --success "TERMINAL_OK" --stop-on "TERMINAL_OK" --replay 20',
        "inner_science_argv": FROZEN_INNER_SCIENCE_ARGV,
        "env": {
            "CUDA_VISIBLE_DEVICES": "0",
            "P1B_LIVE_CONVERSION_RECEIPT_JSON": str(tmp_path / "receipt.json"),
            "PYTHONPATH": ".",
        },
        "paths": {
            "phase_b_log": str(tmp_path / "phaseB.log"),
            "watchdog_event_log": str(tmp_path / "wd.jsonl"),
            "activation_receipt": str(tmp_path / "act.json"),
            "monitor_armed_touch": str(tmp_path / "mon.touch"),
            "p1b_receipt": str(tmp_path / "p1b.json"),
        },
        "phase_budgets": {
            "model_build": 180,
            "forward_backward": 300,
            "vote_apply": 120,
            "checkpoint_roundtrip": 180,
            "receipt_mint": 60,
        },
        "activation_deadlines": {
            "trainer_spawn_to_WATCHDOG_ARMED_sec": 30,
            "WATCHDOG_ARMED_to_Monitor_armed_sec": 60,
        },
        "kill_semantics": {"target": "trainer_process_group_only"},
    }


def _mint(tmp_path: Path, fields: dict | None = None) -> tuple[Path, dict]:
    fields = fields or _base_fields(tmp_path)
    path = tmp_path / "packet.json"
    minted = mint_phase_b_packet_o_excl(path, fields)
    return path, minted


def test_missing_packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            tmp_path / "missing.json",
            "a" * 64,
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_MISSING


def test_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    path = tmp_path / "bad.json"
    raw = b"{not-json"
    path.write_bytes(raw)
    file_sha = hashlib.sha256(raw).hexdigest()
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            file_sha,
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_SCHEMA


def test_schema_incomplete(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    del fields["kill_semantics"]
    # Build bytes with a fake digest so full-file sha can pass.
    fields["packet_payload_digest"] = "c" * 64
    raw = (json.dumps(fields, sort_keys=True) + "\n").encode()
    path = tmp_path / "incomplete.json"
    path.write_bytes(raw)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            hashlib.sha256(raw).hexdigest(),
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_SCHEMA


def test_stale_commit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    fields["commit_sha"] = "0" * 40
    path, minted = _mint(tmp_path, fields)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            minted["packet_file_sha256"],
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_COMMIT


def test_stale_file_hash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    fields["runtime_executable_sha256s"]["scripts/train_hrm_text_158.py"] = "d" * 64
    path, minted = _mint(tmp_path, fields)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            minted["packet_file_sha256"],
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_FILE_HASH


def test_changed_science_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.delenv("P1B_ALLOW_TEST_TRAINER_COMMAND", raising=False)
    fields = _base_fields(tmp_path)
    fields["inner_science_argv"] = FROZEN_INNER_SCIENCE_ARGV + " --evil"
    path, minted = _mint(tmp_path, fields)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            minted["packet_file_sha256"],
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_ARGV


def test_changed_budget_or_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    monkeypatch.delenv("P1B_ALLOW_TEST_BUDGETS", raising=False)
    fields = _base_fields(tmp_path)
    fields["phase_budgets"]["model_build"] = 999
    path, minted = _mint(tmp_path, fields)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            minted["packet_file_sha256"],
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_BUDGET_PATH


def test_second_mint_o_excl_refusal(tmp_path: Path):
    fields = _base_fields(tmp_path)
    path, _ = _mint(tmp_path, fields)
    with pytest.raises(FileExistsError):
        mint_phase_b_packet_o_excl(path, fields)


def test_payload_digest_mismatch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    path, minted = _mint(tmp_path, fields)
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["packet_payload_digest"] = "e" * 64
    # Rewrite preserving approximate size — full-file sha arg must match NEW bytes.
    raw = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            hashlib.sha256(raw).hexdigest(),
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_PAYLOAD


def test_rewritten_consistent_payload_wrong_file_sha(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    path, minted = _mint(tmp_path, fields)
    original_file_sha = minted["packet_file_sha256"]
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["plan_id"] = "tampered"
    obj["packet_payload_digest"] = compute_packet_payload_digest(obj)
    raw = (json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n").encode()
    path.write_bytes(raw)
    # Pass ORIGINAL file sha → 69 before payload checks.
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            original_file_sha,
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_FILE_SHA


def test_whitespace_drift_file_sha_69(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    fields = _base_fields(tmp_path)
    path, minted = _mint(tmp_path, fields)
    obj = json.loads(path.read_text(encoding="utf-8"))
    # Reserialize with indent → byte drift, same logical payload digest.
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            minted["packet_file_sha256"],
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_FILE_SHA


def test_absent_packet_sha256_arg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("P1B_ALLOW_NONCANONICAL_PATHS", "1")
    path, _ = _mint(tmp_path)
    with pytest.raises(SystemExit) as ei:
        validate_packet_pre_spawn(
            path,
            None,
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei.value.code == EXIT_PACKET_SHA_ARG

    with pytest.raises(SystemExit) as ei2:
        validate_packet_pre_spawn(
            path,
            "not-a-sha",
            cwd=ISOL,
            allow_noncanonical_paths=True,
        )
    assert ei2.value.code == EXIT_PACKET_SHA_ARG
