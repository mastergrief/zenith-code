#!/usr/bin/env python3
"""Offline CPU replay of post_confirmation_hygiene_assert_command from the launch packet.

Extracts the heredoc body from the artifact JSON and executes it against synthetic
run_roots — the fixture MUST drive the edited artifact command, not a duplicated copy.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REPLAY_JSON = (
    REPO_ROOT
    / "artifacts/consensus_prep/d_recompute_window_feasibility_gpu_launch_packet_v1_replay_commands.json"
)


def _extract_hygiene_script(replay_path: Path) -> str:
    data = json.loads(replay_path.read_text(encoding="utf-8"))
    cmd = data["post_confirmation_hygiene_assert_command"]
    m = re.search(r"<<'PY'\n(.*)\nPY", cmd, re.DOTALL)
    if not m:
        raise RuntimeError("could not extract hygiene heredoc from replay artifact")
    return m.group(1)


def _run_hygiene(run_root: Path, script_body: str) -> tuple[int, dict]:
    run_root.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        [sys.executable, "-", str(run_root)],
        input=script_body,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**dict(__import__("os").environ), "PYTHONPATH": "."},
    )
    receipt_path = run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8")) if receipt_path.is_file() else {}
    return proc.returncode, receipt


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(r) + "\n" for r in rows),
        encoding="utf-8",
    )


def main() -> int:
    script_body = _extract_hygiene_script(REPLAY_JSON)
    failures: list[str] = []

    with tempfile.TemporaryDirectory(prefix="d_hygiene_fixture_") as tmp:
        tmp_path = Path(tmp)

        # (i) valid JSON run.log + receipt + jsonl -> pass + start_count=1
        case1 = tmp_path / "case1_valid"
        scratch1 = case1 / "d_recompute_window_diagnostic"
        scratch1.mkdir(parents=True)
        (scratch1 / "run.log").write_text(
            json.dumps({"phase": "bounded_steps", "event": "start", "step": 1}) + "\n",
            encoding="utf-8",
        )
        log1 = scratch1 / "recompute_window_log.jsonl"
        _write_jsonl(log1, [{"state_key": "k", "step": 1, "lane_indices": [0]}])
        (scratch1 / "receipt.json").write_text(
            json.dumps(
                {
                    "d_recompute_window_instrumentation_enabled": True,
                    "d_recompute_window_log_path": str(log1),
                    "steps_completed": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (case1 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case1 / "prelaunch" / "confirmation_launch_rc.txt").write_text("0\n")
        rc1, r1 = _run_hygiene(case1, script_body)
        if rc1 != 0 or not r1.get("pass"):
            failures.append(f"case1_valid rc={rc1} receipt={r1}")
        elif r1.get("bounded_steps_start_count") != 1:
            failures.append(f"case1_start_count={r1.get('bounded_steps_start_count')}")

        # (ii) DObserverOutOfRangeShadowError @ run.log -> HARNESS_INVALID
        case2 = tmp_path / "case2_out_of_range"
        scratch2 = case2 / "d_recompute_window_diagnostic"
        scratch2.mkdir(parents=True)
        (scratch2 / "run.log").write_text(
            "Traceback ... DObserverOutOfRangeShadowError\n", encoding="utf-8"
        )
        (case2 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case2 / "prelaunch" / "confirmation_launch_rc.txt").write_text("1\n")
        rc2, r2 = _run_hygiene(case2, script_body)
        if r2.get("abnormal_terminal_classifier", {}).get("primary_terminal") != "HARNESS_INVALID":
            failures.append(f"case2_terminal={r2.get('abnormal_terminal_classifier')}")

        # (iii) DObserverShadowUnavailableError @ probe.stdout.log -> HARNESS_INVALID
        case3 = tmp_path / "case3_shadow_unavailable"
        scratch3 = case3 / "d_recompute_window_diagnostic"
        scratch3.mkdir(parents=True)
        (scratch3 / "probe.stdout.log").write_text(
            "DObserverShadowUnavailableError\n", encoding="utf-8"
        )
        (case3 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case3 / "prelaunch" / "confirmation_launch_rc.txt").write_text("1\n")
        rc3, r3 = _run_hygiene(case3, script_body)
        if r3.get("abnormal_terminal_classifier", {}).get("primary_terminal") != "HARNESS_INVALID":
            failures.append(f"case3_terminal={r3.get('abnormal_terminal_classifier')}")

        # (iv) missing receipt -> HARNESS_INVALID / missing_receipt
        case4 = tmp_path / "case4_missing_receipt"
        scratch4 = case4 / "d_recompute_window_diagnostic"
        scratch4.mkdir(parents=True)
        (scratch4 / "run.log").write_text(
            json.dumps({"phase": "bounded_steps", "event": "start"}) + "\n",
            encoding="utf-8",
        )
        (case4 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case4 / "prelaunch" / "confirmation_launch_rc.txt").write_text("1\n")
        rc4, r4 = _run_hygiene(case4, script_body)
        if "missing_receipt" not in r4.get("failures", []):
            failures.append(f"case4_failures={r4.get('failures')}")

        # (v) empty jsonl + receipt -> MISSING_OBSERVABLES
        case5 = tmp_path / "case5_empty_jsonl"
        scratch5 = case5 / "d_recompute_window_diagnostic"
        scratch5.mkdir(parents=True)
        (scratch5 / "run.log").write_text(
            json.dumps({"phase": "bounded_steps", "event": "start", "step": 1}) + "\n",
            encoding="utf-8",
        )
        log5 = scratch5 / "recompute_window_log.jsonl"
        log5.write_text("", encoding="utf-8")
        (scratch5 / "receipt.json").write_text(
            json.dumps(
                {
                    "d_recompute_window_instrumentation_enabled": True,
                    "d_recompute_window_log_path": str(log5),
                    "steps_completed": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (case5 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case5 / "prelaunch" / "confirmation_launch_rc.txt").write_text("0\n")
        rc5, r5 = _run_hygiene(case5, script_body)
        term5 = r5.get("abnormal_terminal_classifier", {}).get("primary_terminal")
        if term5 != "MISSING_OBSERVABLES_OR_INVALID_WINDOW":
            failures.append(f"case5_terminal={term5}")

        # (vi) scalar-contaminated run.log + 1 bounded_steps start -> start_count=1
        case6 = tmp_path / "case6_scalar_contamination"
        scratch6 = case6 / "d_recompute_window_diagnostic"
        scratch6.mkdir(parents=True)
        lines = [
            json.dumps("runtime-resource-failure"),
            json.dumps({"phase": "bounded_steps", "event": "start", "step": 1}),
            json.dumps(42),
        ]
        (scratch6 / "run.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log6 = scratch6 / "recompute_window_log.jsonl"
        _write_jsonl(log6, [{"state_key": "k", "step": 1, "lane_indices": [0]}])
        (scratch6 / "receipt.json").write_text(
            json.dumps(
                {
                    "d_recompute_window_instrumentation_enabled": True,
                    "d_recompute_window_log_path": str(log6),
                    "steps_completed": 1,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (case6 / "prelaunch").mkdir(parents=True, exist_ok=True)
        (case6 / "prelaunch" / "confirmation_launch_rc.txt").write_text("0\n")
        rc6, r6 = _run_hygiene(case6, script_body)
        if rc6 != 0 or r6.get("bounded_steps_start_count") != 1:
            failures.append(f"case6 rc={rc6} start_count={r6.get('bounded_steps_start_count')}")

    if failures:
        print("FAILURES:")
        for f in failures:
            print(f"  - {f}")
        return 1

    print("OK: 6/6 hygiene classifier fixture cases passed (artifact-replay)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
