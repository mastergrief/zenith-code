"""Watch-wrap spawn argv construction (list authority; string is derived)."""
from __future__ import annotations

import shlex
from pathlib import Path
from typing import Mapping, Sequence


DEFAULT_WATCH_WRAP = Path("/mnt/c/Users/gabes/projects/claw-code/bin/watch-wrap")

DEFAULT_ERROR = (
    "Traceback|Error|Killed|OOM|FAILED|assert|R1lLaunchProofAbort|NO-MINT|"
    "S5_TERMINAL_MARKER_UNWRITABLE|PHASE_BUDGET_BREACH|FIRST_PHASE_FAIL"
)
DEFAULT_PROGRESS = (
    "R1-L launch proof|epoch|cuda|receipt_json|RUNNER_|R1L_TERMINAL_PASS|S5_|CHILD_|PHASE_"
)
DEFAULT_SUCCESS = "RUNNER_PASS|R1-L launch proof: receipt_json"


def build_child_timeout_argv(
    *,
    outer_timeout_s: int,
    kill_after_s: int,
    bash_c_body: str,
) -> list[str]:
    if outer_timeout_s <= 0:
        raise ValueError("outer_timeout_s must be positive")
    return [
        "timeout",
        "--signal=TERM",
        f"--kill-after={kill_after_s}",
        str(outer_timeout_s),
        "bash",
        "-c",
        bash_c_body,
    ]


def build_run_phase_bash_c(
    phase_paths: Mapping[str, str],
    budgets: Mapping[str, int],
    phases: Sequence[str],
) -> str:
    """Absolute-path run_phase chain body (no bare relative operands).

    Status capture must not rely on ``cmd; ec=$?`` under ``set -e``: a nonzero
    timeout exits the shell before ``ec=$?``, so named markers never emit while
    the outer rc still looks correct. Capture via ``if …; then …; else ec=$?; fi``.
    """
    for ph in phases:
        path = phase_paths[ph]
        if not path.startswith("/"):
            raise ValueError(f"phase path must be absolute: {ph}={path!r}")
        if " " in path:
            raise ValueError(f"phase path must not contain spaces: {path!r}")
    parts = [
        "trap 'echo CHILD_WRAPPER_SIGNAL status=$? >&2; "
        "pkill -TERM -P $$ 2>/dev/null || true; wait || true; exit 143' TERM INT",
        "set -euo pipefail",
        # if/else captures status without set -e aborting before ec is assigned
        'run_phase(){ local script="$1"; local budget="$2"; local name="$3"; local ec=0; '
        'if timeout --signal=TERM --kill-after=5 "$budget" bash "$script"; then return 0; '
        'else ec=$?; fi; '
        'if [ "$ec" -eq 124 ] || [ "$ec" -eq 137 ]; then '
        "echo PHASE_BUDGET_BREACH phase=$name budget=$budget rc=$ec >&2; exit $ec; fi; "
        "echo FIRST_PHASE_FAIL phase=$name rc=$ec >&2; exit $ec; }",
    ]
    for ph in phases:
        parts.append(f"run_phase {phase_paths[ph]} {int(budgets[ph])} {ph}")
    return "; ".join(parts)


def build_watch_wrap_spawn_argv(
    child_argv: Sequence[str],
    *,
    watch_wrap_path: str | Path = DEFAULT_WATCH_WRAP,
    heartbeat: int = 60,
    error: str = DEFAULT_ERROR,
    progress: str = DEFAULT_PROGRESS,
    success: str = DEFAULT_SUCCESS,
    replay: int = 20,
) -> list[str]:
    ww = str(watch_wrap_path)
    argv = [
        ww,
        "--heartbeat",
        str(heartbeat),
        "--error",
        error,
        "--progress",
        progress,
        "--success",
        success,
        "--replay",
        str(replay),
        "--",
        *list(child_argv),
    ]
    if "--log" in argv:
        raise ValueError("topology (c) forbids --log in spawn argv")
    if "--stop-on" in argv:
        raise ValueError("topology (c) forbids --stop-on in spawn argv")
    # no placeholders
    joined = " ".join(argv)
    if "<" in joined or "…" in joined or "..." in joined:
        # allow only if inside regex patterns — forbid shell placeholders
        if "bash -c '<" in joined or "<fail-stop" in joined:
            raise ValueError("placeholder forbidden in spawn argv")
    return argv


def spawn_suffix_after_double_dash(monitor_argv: Sequence[str]) -> list[str]:
    try:
        i = list(monitor_argv).index("--")
    except ValueError as exc:
        raise ValueError("monitor argv missing --") from exc
    return list(monitor_argv[i + 1 :])


def assert_suffix_equals_child(monitor_argv: Sequence[str], child_argv: Sequence[str]) -> None:
    suf = spawn_suffix_after_double_dash(monitor_argv)
    if suf != list(child_argv):
        raise AssertionError(f"spawn suffix != child argv\n suf={suf!r}\nchild={list(child_argv)!r}")


def render_argv_shell(argv: Sequence[str]) -> str:
    """Runnable string rendering — ONLY via shlex.quote."""
    return " ".join(shlex.quote(str(a)) for a in argv)
