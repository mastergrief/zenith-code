#!/usr/bin/env python3
"""Subprocess replay executor for R7 from-clean GPU launch (no eval)."""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

PRELAUNCH_STOP_KEY = "r7_flag_witness_command"
GPU_KEYS = frozenset({"gpu_diagnostic_command", "post_gpu_receipt_assert_command"})
TERMINAL_ORDER = (
    "classifier_command",
    "terminal_receipt_compose_command",
    "ai_room_post_terminal",
)
RELEASE_KEY = "resource_lane_release"
LOOP_SKIP_KEYS = frozenset(TERMINAL_ORDER) | {RELEASE_KEY}
MOCK_FLAG_SCRIPTS = frozenset(
    {
        "hrm_text_158_r7_resource_lane_acquire.py",
        "hrm_text_158_r7_resource_lane_release.py",
    }
)
SKIP_POST_SCRIPT = "hrm_text_158_r7_ai_room_terminal_post.py"


def substitute_placeholders(command: str, run_root: str, chain_id: str) -> str:
    return command.replace("{run_root}", run_root).replace("{chain_id}", chain_id)


def parse_env_and_argv(command: str) -> tuple[dict[str, str], list[str]]:
    parts = shlex.split(command)
    env = os.environ.copy()
    idx = 0
    while idx < len(parts) and "=" in parts[idx] and not parts[idx].startswith("-"):
        key, value = parts[idx].split("=", 1)
        env[key] = value
        idx += 1
    return env, parts[idx:]


def _script_argv_target(argv: list[str]) -> str | None:
    for part in argv:
        if part.endswith(".py"):
            return part
    return None


def append_executor_flags(argv: list[str], *, mock_lane: bool, skip_ai_room_post: bool) -> list[str]:
    if not argv:
        return argv
    target = _script_argv_target(argv)
    out = list(argv)
    if mock_lane and target and Path(target).name in MOCK_FLAG_SCRIPTS:
        if "--mock" not in out:
            out.append("--mock")
    if skip_ai_room_post and target and Path(target).name == SKIP_POST_SCRIPT:
        if "--skip-post" not in out:
            out.append("--skip-post")
    return out


def run_command(
    command: str,
    *,
    cwd: Path,
    mock_lane: bool,
    skip_ai_room_post: bool,
    use_shell: bool = False,
) -> int:
    env, argv = parse_env_and_argv(command)
    if not use_shell:
        argv = append_executor_flags(argv, mock_lane=mock_lane, skip_ai_room_post=skip_ai_room_post)
    proc = subprocess.run(
        argv if not use_shell else command,
        cwd=cwd,
        env=env,
        shell=use_shell,
    )
    return int(proc.returncode)


def load_replay(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_command(replay: dict, key: str) -> str:
    cmd = replay.get(key) or replay.get(key + "_command") or ""
    if not cmd:
        raise KeyError(f"missing replay key: {key}")
    return cmd


def execution_order(replay: dict) -> list[str]:
    order = replay.get("test_operator_execution_order")
    if not isinstance(order, list) or not order:
        raise ValueError("test_operator_execution_order missing or empty")
    return list(order)


def finalize_terminal_path(
    replay: dict,
    *,
    run_root: str,
    chain_id: str,
    cwd: Path,
    mock_lane: bool,
    skip_ai_room_post: bool,
) -> None:
    for key in TERMINAL_ORDER:
        cmd = substitute_placeholders(resolve_command(replay, key), run_root, chain_id)
        run_command(
            cmd,
            cwd=cwd,
            mock_lane=mock_lane,
            skip_ai_room_post=skip_ai_room_post,
            use_shell=False,
        )


def release_lane_once(
    replay: dict,
    *,
    run_root: str,
    chain_id: str,
    cwd: Path,
    mock_lane: bool,
    skip_ai_room_post: bool,
) -> None:
    release_cmd = substitute_placeholders(
        resolve_command(replay, RELEASE_KEY),
        run_root,
        chain_id,
    )
    run_command(
        release_cmd,
        cwd=cwd,
        mock_lane=mock_lane,
        skip_ai_room_post=skip_ai_room_post,
    )


def finalize_and_release_once(
    replay: dict,
    *,
    run_root: str,
    chain_id: str,
    cwd: Path,
    mock_lane: bool,
    skip_ai_room_post: bool,
    finalized: set[str],
) -> None:
    if finalized:
        return
    finalized.add("done")
    finalize_terminal_path(
        replay,
        run_root=run_root,
        chain_id=chain_id,
        cwd=cwd,
        mock_lane=mock_lane,
        skip_ai_room_post=skip_ai_room_post,
    )
    release_lane_once(
        replay,
        run_root=run_root,
        chain_id=chain_id,
        cwd=cwd,
        mock_lane=mock_lane,
        skip_ai_room_post=skip_ai_room_post,
    )


def run_replay(
    replay_path: Path,
    run_root: Path,
    chain_id: str,
    *,
    dry_run_prelaunch: bool = False,
    skip_gpu: bool = False,
    mock_lane: bool = False,
    skip_ai_room_post: bool = False,
) -> int:
    replay = load_replay(replay_path)
    cwd = REPO_ROOT
    run_root_str = str(run_root)
    step_failure = 0
    finalized: set[str] = set()

    for key in execution_order(replay):
        if key in LOOP_SKIP_KEYS:
            continue
        if dry_run_prelaunch and key == PRELAUNCH_STOP_KEY:
            cmd = substitute_placeholders(resolve_command(replay, key), run_root_str, chain_id)
            return run_command(
                cmd,
                cwd=cwd,
                mock_lane=mock_lane,
                skip_ai_room_post=skip_ai_room_post,
            )
        if skip_gpu and key in GPU_KEYS:
            continue
        cmd = substitute_placeholders(resolve_command(replay, key), run_root_str, chain_id)
        rc = run_command(
            cmd,
            cwd=cwd,
            mock_lane=mock_lane,
            skip_ai_room_post=skip_ai_room_post,
            use_shell=(key == "gpu_diagnostic_command"),
        )
        if rc != 0:
            step_failure = rc
            finalize_and_release_once(
                replay,
                run_root=run_root_str,
                chain_id=chain_id,
                cwd=cwd,
                mock_lane=mock_lane,
                skip_ai_room_post=skip_ai_room_post,
                finalized=finalized,
            )
            return step_failure

    finalize_and_release_once(
        replay,
        run_root=run_root_str,
        chain_id=chain_id,
        cwd=cwd,
        mock_lane=mock_lane,
        skip_ai_room_post=skip_ai_room_post,
        finalized=finalized,
    )
    return step_failure


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 from-clean replay executor.")
    parser.add_argument("--replay-json", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--chain-id", default="r7-from-clean")
    parser.add_argument("--dry-run-prelaunch", action="store_true")
    parser.add_argument("--skip-gpu", action="store_true")
    parser.add_argument("--mock-lane", action="store_true")
    parser.add_argument("--skip-ai-room-post", action="store_true")
    args = parser.parse_args(argv)
    return run_replay(
        args.replay_json,
        args.run_root,
        args.chain_id,
        dry_run_prelaunch=args.dry_run_prelaunch,
        skip_gpu=args.skip_gpu,
        mock_lane=args.mock_lane,
        skip_ai_room_post=args.skip_ai_room_post,
    )


if __name__ == "__main__":
    raise SystemExit(main())
