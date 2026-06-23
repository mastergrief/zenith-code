"""CPU tests for R8 terminal compose/post chain (A-copy class-slice)."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    build_step_chunk,
)
from calm.hrm_text_158.native_full_stack.r8_global_cap_relax_classifier_probe import (
    BRANCH_CAP_WAS_BINDING,
    BRANCH_HARNESS_FAIL,
    R7_BASELINE_RUN_ID,
)
from scripts import hrm_text_158_r7_from_clean_replay_executor as replay_executor


def _write_cap_was_binding_sidecar(sidecar: Path, *, steps: int = 10) -> None:
    chunks = []
    for step in range(1, steps + 1):
        candidate = 100
        deferred = 20
        accepted = 80
        accepted_from_prior = 70
        accepted_fresh = accepted - accepted_from_prior
        summary = {
            "global_rate_cap_enabled": True,
            "global_pre_cap_would_apply_count": candidate,
            "global_rate_cap_accepted_count": accepted,
            "global_rate_cap_deferred_count": deferred,
            "global_rate_cap_cap": 512,
            "global_rate_cap_saturated": False,
            "q_changed_count": accepted,
            "deferred_backlog_size": deferred,
            "deferred_backlog_max_age_steps": 4,
            "deferred_backlog_max_defer_count": 1,
            "accepted_from_prior_deferred_count": accepted_from_prior,
            "accepted_fresh_count": accepted_fresh,
        }
        pressure = 10 if step == 1 else 12
        chunks.append(
            build_step_chunk(
                step=step,
                global_summary=summary,
                pressure_mass=pressure,
                pressure_mass_delta=None if step == 1 else 2,
            )
        )
    sidecar.write_text("\n".join(json.dumps(c) for c in chunks) + "\n", encoding="utf-8")


class TestR8TerminalReceiptCompose:
    def test_harness_fail_without_diagnostic_receipt(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        (run_root / "diagnostic").mkdir(parents=True)
        head = "6b03c38bd02523893938f464b9f0601d9613abe7"
        classifier_out = run_root / "r8_global_cap_relax_classifier_receipt.json"

        classifier_proc = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r8_global_cap_relax_classifier_probe.py"),
                "--run-root",
                str(run_root),
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
                str(REPO_ROOT / "scripts/hrm_text_158_r8_terminal_receipt_compose.py"),
                str(run_root),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert compose_proc.returncode == 0, compose_proc.stderr + compose_proc.stdout
        terminal = json.loads((run_root / "terminal_receipt.json").read_text(encoding="utf-8"))
        classifier = json.loads(classifier_out.read_text(encoding="utf-8"))
        assert classifier["branch_selection"]["branch"] == BRANCH_HARNESS_FAIL
        assert terminal["primary_branch"] == BRANCH_HARNESS_FAIL
        assert terminal["diagnostic_never_launched"] is True
        assert terminal["diagnostic_receipt_path"] is None

    def test_incomplete_diagnostic_emits_run_incomplete(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        diagnostic = run_root / "diagnostic"
        diagnostic.mkdir(parents=True)
        _write_cap_was_binding_sidecar(diagnostic / "r7_cap_defer_pressure_sidecar.jsonl")
        head = "6b03c38bd02523893938f464b9f0601d9613abe7"
        classifier_out = run_root / "r8_global_cap_relax_classifier_receipt.json"

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r8_global_cap_relax_classifier_probe.py"),
                "--run-root",
                str(run_root),
                "--head-sha256",
                head,
                "--json-out",
                str(classifier_out),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=120,
        )
        classifier = json.loads(classifier_out.read_text(encoding="utf-8"))
        assert classifier["branch_selection"]["branch"] == BRANCH_CAP_WAS_BINDING

        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r8_terminal_receipt_compose.py"),
                str(run_root),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=120,
        )
        terminal = json.loads((run_root / "terminal_receipt.json").read_text(encoding="utf-8"))
        assert terminal["primary_branch"] == BRANCH_CAP_WAS_BINDING
        assert terminal["run_incomplete"] is True
        assert terminal["diagnostic_crashed"] is True
        assert terminal["diagnostic_receipt_missing"] is True

    def test_happy_path_forwards_audit_summary_and_baseline(self, tmp_path: Path) -> None:
        run_root = tmp_path / "run"
        diagnostic = run_root / "diagnostic"
        diagnostic.mkdir(parents=True)
        _write_cap_was_binding_sidecar(diagnostic / "r7_cap_defer_pressure_sidecar.jsonl")
        (diagnostic / "receipt.json").write_text(
            json.dumps(
                {
                    "steps_completed": 10,
                    "prior_audit": {
                        "enabled": True,
                        "requested_supports": ["L0b", "math_a0", "L0c1"],
                        "deltas": {},
                        "start_reports": {},
                        "final_reports": {},
                    },
                }
            ),
            encoding="utf-8",
        )
        head = "6b03c38bd02523893938f464b9f0601d9613abe7"
        classifier_out = run_root / "r8_global_cap_relax_classifier_receipt.json"
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r8_global_cap_relax_classifier_probe.py"),
                "--run-root",
                str(run_root),
                "--head-sha256",
                head,
                "--json-out",
                str(classifier_out),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=120,
        )
        subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "scripts/hrm_text_158_r8_terminal_receipt_compose.py"),
                str(run_root),
            ],
            cwd=REPO_ROOT,
            check=True,
            timeout=120,
        )
        terminal = json.loads((run_root / "terminal_receipt.json").read_text(encoding="utf-8"))
        assert terminal["audit_summary"] is not None
        assert terminal["audit_summary"]["enabled"] is True
        assert terminal["baseline_comparison"]["baseline_run_id"] == R7_BASELINE_RUN_ID
        assert terminal["r7_baseline_provenance"]["run_id"] == R7_BASELINE_RUN_ID
        assert terminal["schema_version"] == "hrm_text_158_r8_global_cap_relax_terminal_receipt/v1"


class TestR8AiRoomTerminalPost:
    def test_skip_post_writes_witness_only(self, tmp_path: Path) -> None:
        import scripts.hrm_text_158_r8_ai_room_terminal_post as post_mod

        run_root = tmp_path / "run"
        run_root.mkdir()
        terminal = {
            "primary_branch": "R8_CAP_WAS_BINDING",
            "next_action": "cap_binding_confirmed_relax_helped",
            "steps_observed": 10,
            "run_metrics": {"steps_observed": 10},
            "branch_selection": {"branch": "R8_CAP_WAS_BINDING"},
            "audit_summary": {"enabled": True, "baseline_correct_strict_regression_count": 0},
            "baseline_comparison": {"baseline_run_id": R7_BASELINE_RUN_ID},
            "r7_baseline_provenance": {"run_id": R7_BASELINE_RUN_ID},
            "classifier_receipt_sha256": "abc",
            "explicit_non_claims": ["audit_report_only_not_veto"],
        }
        (run_root / "terminal_receipt.json").write_text(json.dumps(terminal), encoding="utf-8")
        witness = post_mod.post_terminal_to_ai_room(run_root, skip_post=True)
        assert witness["post_skipped"] is True
        assert witness["msg_id"] is None
        payload = json.loads(
            (run_root / "post_gpu" / "ai_room_terminal_post_payload.json").read_text(encoding="utf-8")
        )
        assert payload["audit_summary"]["enabled"] is True
        assert payload["baseline_comparison"]["baseline_run_id"] == R7_BASELINE_RUN_ID

    def test_real_post_forwards_r8_fields_with_command_and_msg_id(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        import scripts.hrm_text_158_r8_ai_room_terminal_post as post_mod

        run_root = tmp_path / "run"
        run_root.mkdir()
        terminal = {
            "primary_branch": "R8_NOT_CAP_BOUND",
            "next_action": "cap_not_primary_binding_mechanism",
            "steps_observed": 10,
            "run_metrics": {"steps_observed": 10, "pressure_growth_ratio": 2.0},
            "branch_selection": {"branch": "R8_NOT_CAP_BOUND"},
            "audit_summary": {
                "enabled": True,
                "baseline_correct_strict_regression_count": 0,
            },
            "baseline_comparison": {"baseline_run_id": R7_BASELINE_RUN_ID},
            "r7_baseline_provenance": {"run_id": R7_BASELINE_RUN_ID},
            "classifier_receipt_sha256": "deadbeef",
            "explicit_non_claims": ["single_variable_cap_only"],
        }
        terminal_path = run_root / "terminal_receipt.json"
        terminal_path.write_text(json.dumps(terminal), encoding="utf-8")

        captured: dict = {}
        messages_mod = types.ModuleType("mcp_server_lib.tools.messages")
        messages_mod.tool_post = lambda handle, payload: (
            captured.update({"handle": handle, "payload": payload})
            or "posted id=r8-test-msg-id"
        )
        monkeypatch.setitem(sys.modules, "mcp_server_lib.tools.messages", messages_mod)
        monkeypatch.setattr(post_mod, "_init_ai_room", lambda: None)

        witness = post_mod.post_terminal_to_ai_room(run_root, skip_post=False)
        payload = captured["payload"]
        body = json.loads(payload["body"])
        assert payload["command"].strip()
        assert "R8 global_cap_relax_512 terminal validation_receipt" in payload["command"]
        assert payload["scope"] == "R8 global_cap_relax_512 from-clean diagnostic terminal"
        assert body["audit_summary"]["enabled"] is True
        assert body["baseline_comparison"]["baseline_run_id"] == R7_BASELINE_RUN_ID
        assert body["r7_baseline_provenance"]["run_id"] == R7_BASELINE_RUN_ID
        assert body["branch_selection"]["branch"] == "R8_NOT_CAP_BOUND"
        assert witness["msg_id"] == "r8-test-msg-id"


class TestExecutorSkipPostForR8:
    def test_skip_ai_room_post_flag_applies_to_r8_post_script(self) -> None:
        r8_argv = [
            "python3",
            "scripts/hrm_text_158_r8_ai_room_terminal_post.py",
            "/tmp/run",
        ]
        out = replay_executor.append_executor_flags(
            r8_argv, mock_lane=False, skip_ai_room_post=True
        )
        assert "--skip-post" in out

    def test_skip_ai_room_post_still_applies_to_r7_post_script(self) -> None:
        r7_argv = [
            "python3",
            "scripts/hrm_text_158_r7_ai_room_terminal_post.py",
            "/tmp/run",
        ]
        out = replay_executor.append_executor_flags(
            r7_argv, mock_lane=False, skip_ai_room_post=True
        )
        assert "--skip-post" in out

    def test_r8_post_in_finalize_order_without_changing_r7_behavior(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        calls: list[str] = []
        r8_post = f"python3 {REPO_ROOT}/scripts/hrm_text_158_r8_ai_room_terminal_post.py"

        def fake_run_command(command: str, **kwargs) -> int:
            env, argv = replay_executor.parse_env_and_argv(command)
            argv = replay_executor.append_executor_flags(
                argv,
                mock_lane=bool(kwargs.get("mock_lane")),
                skip_ai_room_post=bool(kwargs.get("skip_ai_room_post")),
            )
            calls.append(" ".join(argv))
            return 0

        monkeypatch.setattr(replay_executor, "run_command", fake_run_command)
        replay = {
            "test_operator_execution_order": [
                "resource_lane_acquire",
                "classifier_command",
                "terminal_receipt_compose_command",
                "ai_room_post_terminal",
                "resource_lane_release",
            ],
            "resource_lane_acquire": "echo acquire",
            "classifier_command": "echo classifier",
            "terminal_receipt_compose_command": "echo compose",
            "ai_room_post_terminal": f"{r8_post} {{run_root}}",
            "resource_lane_release": "echo release",
        }
        replay_path = tmp_path / "replay_r8.json"
        replay_path.write_text(json.dumps(replay), encoding="utf-8")
        rc = replay_executor.run_replay(
            replay_path,
            tmp_path / "run",
            "r8-chain",
            skip_gpu=True,
            mock_lane=True,
            skip_ai_room_post=True,
        )
        assert rc == 0
        post_cmds = [c for c in calls if "hrm_text_158_r8_ai_room_terminal_post.py" in c]
        assert len(post_cmds) == 1
        assert "--skip-post" in post_cmds[0]
        ordered = [c for c in calls if c.startswith("echo ") or "r8_ai_room" in c]
        assert ordered.index("echo classifier") < ordered.index("echo compose")
        assert "hrm_text_158_r8_ai_room_terminal_post.py" in post_cmds[0]
