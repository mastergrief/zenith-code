"""CPU wrapper-extraction smoke for R7 from-clean replay commands (Gate A)."""
from __future__ import annotations

import hashlib
import json
import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures" / "r7_replay"
V3_BAD = FIXTURE_DIR / "replay_v3_bad_eval.json"
V4_GOOD = FIXTURE_DIR / "replay_v4_good_scripts.json"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.box_lane import EXIT_CODE_CURRENCY_MISMATCH
from scripts import hrm_text_158_r7_from_clean_replay_executor as replay_executor
from scripts.hrm_text_158_r7_prelaunch_persistence_witness import run_prelaunch_persistence_witness

R7_SCRIPTS = [
    REPO_ROOT / "scripts" / f"hrm_text_158_r7_{name}.py"
    for name in (
        "prelaunch_argv_validation",
        "flag_witness",
        "prelaunch_persistence_witness",
        "post_gpu_receipt_assert_witness",
        "terminal_receipt_compose",
        "ai_room_terminal_post",
        "resource_lane_acquire",
        "resource_lane_release",
        "from_clean_replay_executor",
    )
]

UNSAFE_EVAL = re.compile(r"\beval\b")
UNSAFE_HEREDOC = re.compile(r"<<")

APPROVED_SCRIPT_PREFIX = "scripts/hrm_text_158_r7_"
APPROVED_SAFE_SCRIPTS = frozenset(
    {
        "scripts/box_lane_code_currency_preflight.py",
        "scripts/hrm_text_158_r7_cap_seam_field_presence_witness.py",
        "scripts/hrm_text_158_full_sub2_runtime_readiness.py",
        "scripts/hrm_text_158_r7_mechanism_classifier_probe.py",
    }
)

MINIMAL_FINALIZE_REPLAY = {
    "test_operator_execution_order": [
        "resource_lane_acquire",
        "pre_step",
        "classifier_command",
        "terminal_receipt_compose_command",
        "ai_room_post_terminal",
        "resource_lane_release",
    ],
    "resource_lane_acquire": "echo acquire",
    "pre_step": "echo pre",
    "classifier_command": "echo classifier",
    "terminal_receipt_compose_command": "echo compose",
    "ai_room_post_terminal": "echo post",
    "resource_lane_release": "echo release",
}


def classify_command_unsafe(command: str) -> str | None:
    if UNSAFE_EVAL.search(command):
        return "eval"
    if UNSAFE_HEREDOC.search(command):
        return "heredoc"
    if re.search(r'python3\s+-c\s+["\']\\n(?:import|from)\s', command):
        return "multiline_c_escaped_newline"
    return None


def is_v4_safe_command(command: str) -> bool:
    if classify_command_unsafe(command):
        return False
    if APPROVED_SCRIPT_PREFIX in command:
        return True
    if any(safe in command for safe in APPROVED_SAFE_SCRIPTS):
        return True
    if command.strip() == "echo gpu_skipped_in_fixture":
        return True
    if "python3 -c" in command:
        return True
    if command.startswith("PYTHONPATH=. python3 scripts/hrm_text_158_r7_from_clean_replay_executor.py"):
        return True
    return False


def load_fixture(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def referenced_scripts_from_replay(replay: dict) -> list[Path]:
    scripts: list[Path] = []
    for key in replay.get("test_operator_execution_order", []):
        cmd = replay.get(key) or replay.get(key + "_command") or ""
        for part in cmd.split():
            if part.endswith(".py") and not part.startswith("-"):
                scripts.append(REPO_ROOT / part)
    return scripts


class TestReplayCommandClassAudit:
    def test_v3_bad_fixture_fails_class_check(self) -> None:
        replay = load_fixture(V3_BAD)
        failures: list[str] = []
        for key in replay["test_operator_execution_order"]:
            cmd = replay[key]
            reason = classify_command_unsafe(cmd)
            if reason:
                failures.append(f"{key}:{reason}")
        assert failures, "v3-bad fixture must contain at least one unsafe command form"
        assert any("argv_validation_command" in f for f in failures)

    def test_v4_good_fixture_passes_class_check(self) -> None:
        replay = load_fixture(V4_GOOD)
        for key in replay["test_operator_execution_order"]:
            cmd = replay.get(key) or replay.get(key + "_command") or ""
            assert classify_command_unsafe(cmd) is None, f"{key} must not use unsafe forms"
            assert is_v4_safe_command(cmd), f"{key} not in approved safe command class"

    def test_py_compile_all_r7_scripts(self) -> None:
        for path in R7_SCRIPTS:
            py_compile.compile(str(path), doraise=True)

    def test_py_compile_referenced_v4_scripts(self) -> None:
        for path in referenced_scripts_from_replay(load_fixture(V4_GOOD)):
            if path.is_file():
                py_compile.compile(str(path), doraise=True)

    def test_ai_room_post_script_has_tool_post_and_witness(self) -> None:
        source = (REPO_ROOT / "scripts/hrm_text_158_r7_ai_room_terminal_post.py").read_text(
            encoding="utf-8"
        )
        assert "tool_post" in source
        assert "ai_room_post_witness.json" in source
        assert "msg_id" in source


class TestReplayExecutorDryRun:
    def test_executor_dry_run_prelaunch_exits_zero(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_from_clean_replay_executor.py"),
                "--replay-json",
                str(V4_GOOD),
                "--run-root",
                str(run_root),
                "--chain-id",
                "r7-smoke",
                "--dry-run-prelaunch",
                "--mock-lane",
                "--skip-ai-room-post",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert (run_root / "prelaunch" / "argv_validation.json").is_file()
        argv_validation = json.loads(
            (run_root / "prelaunch" / "argv_validation.json").read_text(encoding="utf-8")
        )
        assert argv_validation["all_parse_ok"] is True

    def test_dry_run_fixture_substitutes_current_head_at_runtime(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        run_root.mkdir()
        current_head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_from_clean_replay_executor.py"),
                "--replay-json",
                str(V4_GOOD),
                "--run-root",
                str(run_root),
                "--chain-id",
                "r7-head-subst-smoke",
                "--dry-run-prelaunch",
                "--mock-lane",
                "--skip-ai-room-post",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=600,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        preflight = json.loads(
            (run_root / "prelaunch" / "box_code_currency_preflight.json").read_text(
                encoding="utf-8"
            )
        )
        assert preflight["head_sha"] == current_head
        assert preflight["head_now"] == current_head
        assert preflight["code_currency_pass"] is True
        argv_validation = json.loads(
            (run_root / "prelaunch" / "argv_validation.json").read_text(encoding="utf-8")
        )
        assert argv_validation["head_pin"] == current_head

    def test_stale_launch_packet_head_pin_fails_code_currency_fail_closed(
        self, tmp_path: Path
    ) -> None:
        stale_head = "0000000000000000000000000000000000000000"
        out_path = tmp_path / "box_code_currency_preflight.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/box_lane_code_currency_preflight.py"),
                "--skip-fetch",
                "--chain-id",
                "r7-stale-head-pin",
                "--head-expected",
                stale_head,
                "--output",
                str(out_path),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == EXIT_CODE_CURRENCY_MISMATCH
        payload = json.loads(out_path.read_text(encoding="utf-8"))
        assert payload["code_currency_pass"] is False
        assert "HEAD_NOW_MISMATCH" in payload["mismatches"]

    @pytest.mark.skipif(
        not Path(
            "/home/gabe/claw-code-creditdir/transient_fp_credit/"
            "r7_from_clean_cap_defer_seed44_43_20260623T090146Z_52ce6aa6"
        ).is_dir(),
        reason="090146Z run root not present on this host",
    )
    def test_executor_dry_run_on_090146z_argv_validation_no_syntax_error(self) -> None:
        run_root = Path(
            "/home/gabe/claw-code-creditdir/transient_fp_credit/"
            "r7_from_clean_cap_defer_seed44_43_20260623T090146Z_52ce6aa6"
        )
        proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts" / "hrm_text_158_r7_prelaunch_argv_validation.py"),
                str(run_root),
                "284899b9cbf4fe664581bf3bbc01d345373e20cd",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr + proc.stdout
        assert (run_root / "prelaunch" / "argv_validation.json").is_file()


class TestExecutorFinalizeOrdering:
    def test_success_path_finalize_order_exactly_once(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[str] = []

        def fake_run_command(command: str, **kwargs) -> int:
            calls.append(command)
            return 0

        monkeypatch.setattr(replay_executor, "run_command", fake_run_command)
        replay_path = tmp_path / "replay.json"
        replay_path.write_text(json.dumps(MINIMAL_FINALIZE_REPLAY), encoding="utf-8")
        rc = replay_executor.run_replay(
            replay_path,
            tmp_path / "run",
            "chain",
            skip_gpu=True,
            mock_lane=True,
            skip_ai_room_post=True,
        )
        assert rc == 0
        assert calls.count("echo classifier") == 1
        assert calls.count("echo compose") == 1
        assert calls.count("echo post") == 1
        assert calls.count("echo release") == 1
        ordered = [
            "echo acquire",
            "echo pre",
            "echo classifier",
            "echo compose",
            "echo post",
            "echo release",
        ]
        assert [c for c in calls if c in ordered] == ordered

    def test_failure_path_still_finalizes_exactly_once(self, tmp_path: Path, monkeypatch) -> None:
        calls: list[str] = []

        def fake_run_command(command: str, **kwargs) -> int:
            calls.append(command)
            if command == "false":
                return 1
            return 0

        monkeypatch.setattr(replay_executor, "run_command", fake_run_command)
        fail_replay = dict(MINIMAL_FINALIZE_REPLAY)
        fail_replay["pre_step"] = "false"
        replay_path = tmp_path / "replay_fail.json"
        replay_path.write_text(json.dumps(fail_replay), encoding="utf-8")
        rc = replay_executor.run_replay(
            replay_path,
            tmp_path / "run",
            "chain",
            skip_gpu=True,
            mock_lane=True,
            skip_ai_room_post=True,
        )
        assert rc == 1
        assert calls.count("echo classifier") == 1
        assert calls.count("echo compose") == 1
        assert calls.count("echo post") == 1
        assert calls.count("echo release") == 1
        classifier_idx = calls.index("echo classifier")
        compose_idx = calls.index("echo compose")
        post_idx = calls.index("echo post")
        release_idx = calls.index("echo release")
        assert classifier_idx < compose_idx < post_idx < release_idx


class TestHarnessFailClosedTerminalChain:
    def test_real_classifier_compose_post_without_diagnostic_receipt(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        diagnostic = run_root / "diagnostic"
        diagnostic.mkdir(parents=True)
        head = "284899b9cbf4fe664581bf3bbc01d345373e20cd"
        classifier_out = run_root / "r7_mechanism_classifier_receipt.json"

        classifier_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_mechanism_classifier_probe.py"),
                "--run-root",
                str(diagnostic),
                "--head-sha256",
                head,
                "--json-out",
                str(classifier_out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert classifier_proc.returncode == 0, classifier_proc.stderr + classifier_proc.stdout
        assert classifier_out.is_file()

        compose_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_terminal_receipt_compose.py"),
                str(run_root),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert compose_proc.returncode == 0, compose_proc.stderr + compose_proc.stdout
        terminal_path = run_root / "terminal_receipt.json"
        assert terminal_path.is_file()

        post_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_ai_room_terminal_post.py"),
                str(run_root),
                "--skip-post",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert post_proc.returncode == 0, post_proc.stderr + post_proc.stdout
        witness_path = run_root / "post_gpu" / "ai_room_post_witness.json"
        assert witness_path.is_file()

        classifier = json.loads(classifier_out.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        witness = json.loads(witness_path.read_text(encoding="utf-8"))
        assert classifier["branch_selection"]["branch"] == "R7_HARNESS_FAIL"
        assert terminal["primary_branch"] == "R7_HARNESS_FAIL"
        assert terminal["diagnostic_never_launched"] is True
        assert terminal["diagnostic_receipt_path"] is None
        assert int(terminal["steps_observed"]) == 0
        assert witness["post_skipped"] is True
        assert witness["terminal_receipt_sha256"] == hashlib.sha256(
            terminal_path.read_bytes()
        ).hexdigest()


class TestCrashedDiagnosticTerminalChain:
    def test_real_classifier_compose_post_with_cap_defer_branch_no_diagnostic_receipt(
        self, tmp_path: Path
    ) -> None:
        from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
            build_step_chunk,
        )

        run_root = tmp_path / "run"
        diagnostic = run_root / "diagnostic"
        diagnostic.mkdir(parents=True)
        sidecar = diagnostic / "r7_cap_defer_pressure_sidecar.jsonl"
        chunks = []
        for step in range(1, 9):
            candidate = 100
            deferred = 60
            accepted = 40
            summary = {
                "global_rate_cap_enabled": True,
                "global_pre_cap_would_apply_count": candidate,
                "global_rate_cap_accepted_count": accepted,
                "global_rate_cap_deferred_count": deferred,
                "global_rate_cap_cap": 40,
                "global_rate_cap_saturated": True,
                "q_changed_count": accepted,
                "deferred_backlog_size": deferred,
                "deferred_backlog_max_age_steps": 1 if step > 1 else 0,
                "deferred_backlog_max_defer_count": 1 if step > 1 else 0,
                "accepted_from_prior_deferred_count": 0,
                "accepted_fresh_count": accepted,
            }
            pressure = 10 + step * 2
            chunks.append(
                build_step_chunk(
                    step=step,
                    global_summary=summary,
                    pressure_mass=pressure,
                    pressure_mass_delta=None if step == 1 else pressure - chunks[-1]["pressure_mass"],
                )
            )
        sidecar.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")

        head = "2105cb5b72c3446a7018429c3629ed03d5db4721"
        classifier_out = run_root / "r7_mechanism_classifier_receipt.json"
        classifier_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_mechanism_classifier_probe.py"),
                "--run-root",
                str(diagnostic),
                "--head-sha256",
                head,
                "--json-out",
                str(classifier_out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert classifier_proc.returncode == 0, classifier_proc.stderr + classifier_proc.stdout

        compose_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_terminal_receipt_compose.py"),
                str(run_root),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert compose_proc.returncode == 0, compose_proc.stderr + compose_proc.stdout
        terminal_path = run_root / "terminal_receipt.json"
        assert terminal_path.is_file()

        post_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r7_ai_room_terminal_post.py"),
                str(run_root),
                "--skip-post",
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert post_proc.returncode == 0, post_proc.stderr + post_proc.stdout
        witness_path = run_root / "post_gpu" / "ai_room_post_witness.json"
        assert witness_path.is_file()

        classifier = json.loads(classifier_out.read_text(encoding="utf-8"))
        terminal = json.loads(terminal_path.read_text(encoding="utf-8"))
        assert classifier["branch_selection"]["branch"] == "R7_CAP_DEFER_BINDING"
        assert terminal["primary_branch"] == "R7_CAP_DEFER_BINDING"
        assert terminal["run_incomplete"] is True
        assert terminal["diagnostic_crashed"] is True
        assert terminal["diagnostic_receipt_missing"] is True
        assert terminal["diagnostic_receipt_path"] is None
        assert int(terminal["steps_observed"]) >= 8


class TestResourceLaneReleaseAlias:
    def test_release_lane_request_prefers_request_alias_over_canonical(self) -> None:
        from scripts.hrm_text_158_r7_resource_lane_release import release_lane_request

        holding = {
            "request_alias": "gpu:hrm-text-158",
            "acquire_result": {
                "token": "tok-abc",
                "canonical_name": "gpu:TheRig:uuid:GPU-4c376502-9cde-e181-ec2b-9773b307342e",
            },
        }
        lane_name, token = release_lane_request(holding)
        assert lane_name == "gpu:hrm-text-158"
        assert token == "tok-abc"
        assert lane_name != holding["acquire_result"]["canonical_name"]


class TestPrelaunchPersistenceWitness:
    def test_rejects_missing_parent_rehash_match(self, tmp_path: Path) -> None:
        pre = tmp_path / "prelaunch"
        pre.mkdir()
        files = {
            "argv_validation.json": {"all_parse_ok": True},
            "box_code_currency_preflight.json": {"code_currency_pass": True},
            "parent_checkpoint_rehash.json": {"match": False},
            "r7_cap_seam_field_presence_witness.json": {"field_presence_pass": True},
            "r7_multistep_backlog_carry_witness.json": {
                "carry_enabled": True,
                "step_n_plus_1_max_age_steps": 2,
            },
            "sub2_readiness_receipt.json": {
                "ready_for_pre_full_stack_diagnostic": True,
                "ready_for_main_science": False,
                "main_science_launch_blocked": True,
            },
        }
        for name, payload in files.items():
            (pre / name).write_text(json.dumps(payload), encoding="utf-8")
        witness = run_prelaunch_persistence_witness(tmp_path)
        assert witness["prelaunch_persistence_witness_pass"] is False
        assert any("parent_checkpoint_rehash.json:match" in f for f in witness["failures"])

    def test_rejects_multistep_without_backlog_age(self, tmp_path: Path) -> None:
        pre = tmp_path / "prelaunch"
        pre.mkdir()
        files = {
            "argv_validation.json": {"all_parse_ok": True},
            "box_code_currency_preflight.json": {"code_currency_pass": True},
            "parent_checkpoint_rehash.json": {"match": True},
            "r7_cap_seam_field_presence_witness.json": {"field_presence_pass": True},
            "r7_multistep_backlog_carry_witness.json": {
                "carry_enabled": True,
                "step_n_plus_1_max_age_steps": 0,
            },
            "sub2_readiness_receipt.json": {
                "ready_for_pre_full_stack_diagnostic": True,
                "ready_for_main_science": False,
                "main_science_launch_blocked": True,
            },
        }
        for name, payload in files.items():
            (pre / name).write_text(json.dumps(payload), encoding="utf-8")
        witness = run_prelaunch_persistence_witness(tmp_path)
        assert witness["prelaunch_persistence_witness_pass"] is False
        assert any("step_n_plus_1_max_age_steps" in f for f in witness["failures"])
