"""S0 CPU tests for minimal_trainer.receipts — no cuda, no model, no torch import."""

from __future__ import annotations

import copy
import hashlib
import subprocess
import sys
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.minimal_trainer.receipts import (
    NINE,
    compare_surfaces,
    load_baseline,
    pins_ok,
    project_named_surfaces,
    rehash_pins,
)

FROZEN_RECEIPT = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/gate1_loop_extract/"
    "phase1_baseline/freeze_v1/phase1_baseline_receipt_v1.json"
)
EXPECTED_RECEIPT_SHA = (
    "98784136632acd4158361f4c2fef62189dcda993b4ae3dc992558f092ef720cf"
)
REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
RECEIPTS_PY = (
    REPO
    / "calm/hrm_text_158/native_full_stack/minimal_trainer/receipts.py"
)

_REFUSE = (
    "projection empty-denominator: duration_seconds not dropped from any "
    "of 9 surfaces; refusing receipt"
)


def _assert_frozen_receipt_sha() -> None:
    data = FROZEN_RECEIPT.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    proc = subprocess.run(
        ["sha256sum", "--", str(FROZEN_RECEIPT)],
        capture_output=True,
        text=True,
        check=True,
    )
    sum_line = proc.stdout.split()[0]
    assert digest == EXPECTED_RECEIPT_SHA, (
        f"frozen receipt moved/changed: hashlib={digest}"
    )
    assert sum_line == EXPECTED_RECEIPT_SHA, (
        f"frozen receipt moved/changed: sha256sum={sum_line}"
    )


def test_known_good_compare_surfaces_silent():
    _assert_frozen_receipt_sha()
    receipt = load_baseline(FROZEN_RECEIPT)
    baseline = receipt["surfaces"]
    candidate = copy.deepcopy(baseline)
    result = compare_surfaces(baseline, candidate)
    assert [row["name"] for row in result["rows"]] == list(NINE)
    by_name = {row["name"]: row for row in result["rows"]}
    for name in NINE:
        assert by_name[name]["match"] is True, name


def test_known_bad_step_reports_sha_fires():
    _assert_frozen_receipt_sha()
    receipt = load_baseline(FROZEN_RECEIPT)
    baseline = receipt["surfaces"]
    candidate = copy.deepcopy(baseline)
    s = candidate["step_reports_sha256"]
    last = int(s[-1], 16)
    candidate["step_reports_sha256"] = s[:-1] + format((last ^ 1) & 0xF, "x")
    result = compare_surfaces(baseline, candidate)
    assert [row["name"] for row in result["rows"]] == list(NINE)
    by_name = {row["name"]: row for row in result["rows"]}
    assert by_name["step_reports"]["match"] is False
    for name in NINE:
        if name == "step_reports":
            continue
        assert by_name[name]["match"] is True, name


def test_known_bad_project_empty_denominator_fires():
    named = {
        "step_reports": {},
        "updater_config": {},
        "states": {},
        "audit_reports": {},
        "stop_reason": "x",
        "steps_completed": 1,
        "b2_full_verdict_state": None,
        "b2b_capture_receipt": None,
        "grad_proxy_ingress_crossing_eligible_count_by_step": [],
    }
    with pytest.raises(ValueError, match="projection empty-denominator") as ei:
        project_named_surfaces(named)
    assert _REFUSE in str(ei.value)


def test_known_good_strip_duration_seconds_silent():
    named = {
        "step_reports": {"duration_seconds": 1.0, "loss": 0.1},
        "updater_config": {"lr": 1e-4},
        "states": {},
        "audit_reports": {"ok": True},
        "stop_reason": "max_steps_completed",
        "steps_completed": 1,
        "b2_full_verdict_state": None,
        "b2b_capture_receipt": None,
        "grad_proxy_ingress_crossing_eligible_count_by_step": [],
    }
    projected, dropped = project_named_surfaces(named)
    assert "duration_seconds" in dropped
    assert "duration_seconds" not in projected["step_reports"]
    assert projected["step_reports"]["loss"] == 0.1


def test_rehash_pins_against_frozen_receipt():
    _assert_frozen_receipt_sha()
    receipt = load_baseline(FROZEN_RECEIPT)
    keys = list(receipt["source_hashes"])
    assert keys, "source_hashes empty"
    result = rehash_pins(receipt, repo_root=REPO)
    assert list(result) == keys
    assert pins_ok(result), {
        k: v for k, v in result.items() if not v.get("matches_receipt")
    }
    assert pins_ok({}) is False
    with pytest.raises(ValueError, match="empty-denominator: source_hashes"):
        rehash_pins({"source_hashes": {}}, repo_root=REPO)
    with pytest.raises(ValueError, match="empty-denominator: baseline_surfaces"):
        compare_surfaces({}, {"states": 1})
    with pytest.raises(ValueError, match="empty-denominator: named surfaces"):
        project_named_surfaces({})


def test_cli_pins_exits_zero():
    _assert_frozen_receipt_sha()
    proc = subprocess.run(
        [
            sys.executable,
            "-B",
            str(RECEIPTS_PY),
            "--pins",
            "--baseline",
            str(FROZEN_RECEIPT),
            "--repo-root",
            str(REPO),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(**{k: v for k, v in __import__("os").environ.items()}),
            "PYTHONPATH": ".",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )
    assert proc.returncode == 0, (proc.stdout, proc.stderr)


def _cli(baseline: Path, repo_root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            "-B",
            str(RECEIPTS_PY),
            "--pins",
            "--baseline",
            str(baseline),
            "--repo-root",
            str(repo_root),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
        check=False,
        env={
            **dict(__import__("os").environ),
            "PYTHONPATH": ".",
            "PYTHONDONTWRITEBYTECODE": "1",
        },
    )


def test_cli_empty_source_hashes_exits_two(tmp_path: Path):
    p = tmp_path / "empty.json"
    p.write_text('{"source_hashes": {}}', encoding="utf-8")
    proc = _cli(p, REPO)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert "empty-denominator" in proc.stdout


def test_cli_wrong_repo_root_exits_two(tmp_path: Path):
    proc = _cli(FROZEN_RECEIPT, tmp_path)
    assert proc.returncode == 2, (proc.stdout, proc.stderr)
    assert proc.returncode != 1


def test_constructed_worktree_divergence_detected(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    pin = repo / "pin.txt"
    pin.write_bytes(b"head\n")
    subprocess.run(["git", "-C", str(repo), "add", "pin.txt"], check=True, capture_output=True)
    subprocess.run(
        ["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t",
         "commit", "-m", "t"],
        check=True,
        capture_output=True,
    )
    head_sha = hashlib.sha256(b"head\n").hexdigest()
    pin.write_bytes(b"dirty\n")
    receipt = {
        "source_hashes": {
            "probe": {"rel": "pin.txt", "hashlib": head_sha, "sha256sum": head_sha}
        }
    }
    result = rehash_pins(receipt, repo_root=repo)
    assert result["probe"]["head_eq_worktree"] is False
    assert pins_ok(result) is False
    # claim: constructed divergence detected. Not a field-verification claim.


def _assert_cli_refusal(proc: subprocess.CompletedProcess[str]) -> None:
    combined = proc.stdout + proc.stderr
    assert proc.returncode == 2, (proc.returncode, combined)
    assert "Traceback" not in combined
    assert "refusing:" in combined or "pin mismatch:" in combined


def test_cli_missing_baseline_refuses(tmp_path: Path):
    _assert_cli_refusal(_cli(tmp_path / "nope.json", REPO))


def test_cli_baseline_is_directory_refuses(tmp_path: Path):
    _assert_cli_refusal(_cli(tmp_path, REPO))


def test_cli_malformed_json_refuses(tmp_path: Path):
    p = tmp_path / "bad.json"
    p.write_text("{", encoding="utf-8")
    _assert_cli_refusal(_cli(p, REPO))


def test_cli_pin_entry_missing_rel_refuses(tmp_path: Path):
    p = tmp_path / "nore.json"
    p.write_text(
        '{"source_hashes": {"probe": {"hashlib": "aa", "sha256sum": "aa"}}}',
        encoding="utf-8",
    )
    _assert_cli_refusal(_cli(p, REPO))
