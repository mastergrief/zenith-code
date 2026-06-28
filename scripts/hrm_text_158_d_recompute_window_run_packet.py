#!/usr/bin/env python3
"""Faithful D recompute-window v2 launch_sequence interpreter.

Thin driver: run_root_free_assert first, scale_smoke launch_allowed gate,
confirmation-onward no-abort-on-RC (confirmation command preserves RC but
exits 0), DRIVER SUMMARY to {run_root}/prelaunch/driver_summary.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SCALE_SMOKE_RECEIPT_REL = "prelaunch/scale_smoke_receipt.json"
CONFIRMATION_STEP = "confirmation_launch_command"
SCALE_SMOKE_RECEIPT_STEP = "scale_smoke_receipt_command"
CONTINUE_AFTER_RC_STEPS = frozenset(
    {
        CONFIRMATION_STEP,
        "post_confirmation_hygiene_assert_command",
        "postrun_command",
    }
)


def resolve_head(cwd: Path) -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return proc.stdout.strip()


def verify_required_ancestor(required_ancestor: str, *, cwd: Path) -> tuple[bool, str, str]:
    """Return (ok, required_ancestor, actual_head).

  Passes when ``required_ancestor`` is reachable from HEAD via
  ``git merge-base --is-ancestor`` (ancestor or equal).
    """
    head = resolve_head(cwd)
    proc = subprocess.run(
        ["git", "merge-base", "--is-ancestor", required_ancestor, "HEAD"],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    return proc.returncode == 0, required_ancestor, head


def substitute_run_root(command: str, run_root: str) -> str:
    return command.replace("{run_root}", run_root)


def resolve_step_command(replay: dict[str, Any], step_key: str) -> str:
    if step_key.startswith("scratch_wipe_commands."):
        subkey = step_key.split(".", 1)[1]
        wipes = replay.get("scratch_wipe_commands") or {}
        cmd = wipes.get(subkey)
        if not cmd:
            raise KeyError(f"missing scratch_wipe_commands.{subkey}")
        return str(cmd)
    cmd = replay.get(step_key)
    if not cmd:
        raise KeyError(f"missing replay key: {step_key}")
    return str(cmd)


def run_shell_command(command: str, *, cwd: Path) -> int:
    proc = subprocess.run(command, cwd=cwd, shell=True)
    return int(proc.returncode)


def load_scale_smoke_launch_allowed(run_root: Path) -> bool | None:
    path = run_root / SCALE_SMOKE_RECEIPT_REL
    if not path.is_file():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    launch_gate = payload.get("launch_gate") or {}
    if "launch_allowed" in launch_gate:
        return bool(launch_gate["launch_allowed"])
    byte_projection = payload.get("byte_projection") or {}
    return bool(byte_projection.get("launch_allowed"))


def write_driver_summary(
    *,
    out_path: Path,
    run_id: str,
    run_root: str,
    packet_path: str,
    replay_path: str,
    head: str,
    step_results: list[dict[str, Any]],
    scale_smoke_launch_allowed: bool | None,
    terminal_rc: int,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "schema": "hrm_text_158_d_recompute_window_run_packet_driver_summary/v1",
        "run_id": run_id,
        "run_root": run_root,
        "packet_path": packet_path,
        "replay_path": replay_path,
        "head": head,
        "scale_smoke_launch_allowed": scale_smoke_launch_allowed,
        "step_results": step_results,
        "terminal_rc": int(terminal_rc),
        "finished_at_unix": int(time.time()),
    }
    out_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_launch_sequence(
    *,
    packet: dict[str, Any],
    replay: dict[str, Any],
    packet_path: Path,
    replay_path: Path,
    cwd: Path,
    require_ancestor: str | None,
) -> int:
    run_root = str(packet["run_root"]).rstrip("/") + "/"
    run_id = str(packet["run_id"])
    if require_ancestor:
        ok, required, head = verify_required_ancestor(require_ancestor, cwd=cwd)
        if not ok:
            print(
                json.dumps(
                    {
                        "pass": False,
                        "error": "required_ancestor_missing",
                        "required_ancestor": required,
                        "actual_head": head,
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 2
    else:
        head = resolve_head(cwd)

    sequence = list(replay.get("launch_sequence") or [])
    if not sequence:
        print(json.dumps({"pass": False, "error": "launch_sequence_missing"}, indent=2), file=sys.stderr)
        return 2

    step_results: list[dict[str, Any]] = []
    scale_smoke_launch_allowed: bool | None = None
    terminal_rc = 0
    saw_confirmation = False

    for step_key in sequence:
        cmd = substitute_run_root(resolve_step_command(replay, step_key), run_root)
        rc = run_shell_command(cmd, cwd=cwd)
        step_results.append({"step": step_key, "rc": rc})

        if step_key == SCALE_SMOKE_RECEIPT_STEP:
            scale_smoke_launch_allowed = load_scale_smoke_launch_allowed(Path(run_root))
            if scale_smoke_launch_allowed is not True:
                print(
                    json.dumps(
                        {
                            "pass": False,
                            "error": "scale_smoke_launch_not_allowed",
                            "launch_allowed": scale_smoke_launch_allowed,
                        },
                        indent=2,
                    ),
                    file=sys.stderr,
                )
                terminal_rc = 3
                break

        if step_key == CONFIRMATION_STEP:
            saw_confirmation = True

        if rc != 0 and step_key not in CONTINUE_AFTER_RC_STEPS and not saw_confirmation:
            print(
                json.dumps(
                    {"pass": False, "error": "step_failed", "step": step_key, "rc": rc},
                    indent=2,
                ),
                file=sys.stderr,
            )
            terminal_rc = rc if terminal_rc == 0 else terminal_rc
            break

        if rc != 0 and step_key not in CONTINUE_AFTER_RC_STEPS and saw_confirmation:
            # Post-confirmation hygiene/postrun may fail; record but continue sequence.
            if terminal_rc == 0:
                terminal_rc = rc

    summary_path = Path(run_root) / "prelaunch" / "driver_summary.json"
    write_driver_summary(
        out_path=summary_path,
        run_id=run_id,
        run_root=run_root,
        packet_path=str(packet_path),
        replay_path=str(replay_path),
        head=head,
        step_results=step_results,
        scale_smoke_launch_allowed=scale_smoke_launch_allowed,
        terminal_rc=terminal_rc,
    )
    print(json.dumps({"pass": terminal_rc == 0, "driver_summary": str(summary_path), "terminal_rc": terminal_rc}, indent=2))
    return int(terminal_rc)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D recompute-window v2 launch_sequence driver")
    parser.add_argument(
        "--packet",
        type=Path,
        required=True,
        help="Launch packet JSON (v2 draft)",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        default=None,
        help="Replay commands JSON (defaults to packet replay_commands_artifact)",
    )
    parser.add_argument(
        "--require-ancestor",
        default=None,
        help=(
            "Required ancestor commit sha; fail closed unless "
            "git merge-base --is-ancestor <sha> HEAD succeeds"
        ),
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Repository root for subprocess cwd",
    )
    args = parser.parse_args(argv)

    packet_path = args.packet if args.packet.is_absolute() else args.repo_root / args.packet
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    replay_rel = args.replay
    if replay_rel is None:
        replay_rel = Path(str(packet.get("replay_commands_artifact") or ""))
    replay_path = replay_rel if replay_rel.is_absolute() else args.repo_root / replay_rel
    replay = json.loads(replay_path.read_text(encoding="utf-8"))

    return run_launch_sequence(
        packet=packet,
        replay=replay,
        packet_path=packet_path,
        replay_path=replay_path,
        cwd=args.repo_root,
        require_ancestor=args.require_ancestor,
    )


if __name__ == "__main__":
    raise SystemExit(main())
