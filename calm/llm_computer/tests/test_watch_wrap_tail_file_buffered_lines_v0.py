"""CPU-static: watch-wrap tail-file select()+readline() buffered-line stall.

Multi-line bursts can leave --stop-on lines in TextIOWrapper while select(fd)
reports idle; Monitor never sees [STOP-TRIGGER].
"""
from __future__ import annotations

import os
import select
import subprocess
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
WATCH_WRAP = Path(os.environ.get("WATCH_WRAP", str(REPO_ROOT / "bin" / "watch-wrap")))
STOP_TOKEN = "STOP_TOKEN_TERMINAL"


def _read_available(proc: subprocess.Popen[str], wait_s: float) -> str:
    assert proc.stdout is not None
    if not select.select([proc.stdout.fileno()], [], [], max(0.0, wait_s))[0]:
        return ""
    return proc.stdout.readline()


def _drain_until(proc, needle: str, timeout_s: float, prefix: str = "") -> str:
    deadline = time.monotonic() + timeout_s
    buf = prefix
    while time.monotonic() < deadline:
        if needle in buf:
            return buf
        line = _read_available(proc, min(0.2, deadline - time.monotonic()))
        if line:
            buf += line
        elif proc.poll() is not None:
            return buf
    return buf


def _start_watcher(log_path: Path, extra: list[str] | None = None):
    cmd = [str(WATCH_WRAP), "--log", str(log_path), "--stop-on", STOP_TOKEN,
           "--heartbeat", "30", "--replay", "0", *(extra or [])]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)


def _prime(proc, log_path: Path) -> str:
    """Event-driven readiness: START then retry probes until tail emits one."""
    buf = _drain_until(proc, "[START tail]", 5.0)
    if "[START tail]" not in buf:
        raise AssertionError(f"no START; out={buf!r}")
    deadline = time.monotonic() + 5.0
    n = 0
    while time.monotonic() < deadline:
        probe = f"READY_PROBE_{os.getpid()}_{n}"
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(f"{probe}\n".encode())
        buf = _drain_until(proc, probe, 0.5, buf)
        if probe in buf:
            return buf
        n += 1
        if proc.poll() is not None:
            raise AssertionError(f"exited during prime rc={proc.returncode}")
    raise AssertionError(f"tail not ready; out={buf!r}")


def _cleanup(proc) -> None:
    if proc.poll() is None:
        proc.kill()
        proc.wait(timeout=2)


def _run(extra=None):
    tmp = tempfile.TemporaryDirectory()
    log_path = Path(tmp.name) / "run.log"
    log_path.write_text("", encoding="utf-8")
    proc = _start_watcher(log_path, extra)
    return tmp, log_path, proc, _prime(proc, log_path)


def test_tail_file_stop_on_fires_for_buffered_multiline_burst() -> None:
    tmp, log_path, proc, started = _run()
    try:
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(f"progress line\n{STOP_TOKEN} terminal\n".encode())
        out = _drain_until(proc, "[STOP-TRIGGER]", 5.0, started)
        try:
            rc = proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired as exc:
            _cleanup(proc)
            raise AssertionError(f"no exit after STOP burst; out={out!r}") from exc
        assert rc == 0 and "[STOP-TRIGGER]" in out and "[HEARTBEAT" not in out
    finally:
        _cleanup(proc)
        tmp.cleanup()


def test_tail_file_partial_line_does_not_stop_until_newline() -> None:
    tmp, log_path, proc, started = _run()
    try:
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(STOP_TOKEN.encode())
        abs_out = _drain_until(proc, "[STOP-TRIGGER]", 0.8, started)
        assert "[STOP-TRIGGER]" not in abs_out and proc.poll() is None
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(b"\n")
        out = _drain_until(proc, "[STOP-TRIGGER]", 5.0, abs_out)
        assert proc.wait(timeout=5.0) == 0 and "[STOP-TRIGGER]" in out
    finally:
        _cleanup(proc)
        tmp.cleanup()


def test_tail_file_multiline_burst_no_drop_no_duplicate_emit() -> None:
    tmp, log_path, proc, started = _run(["--progress", r"^(READY_PROBE_|PROG_)"])
    try:
        lines = [f"PROG_{i}" for i in range(5)] + [STOP_TOKEN]
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(("\n".join(lines) + "\n").encode())
        out = _drain_until(proc, "[STOP-TRIGGER]", 5.0, started)
        assert proc.wait(timeout=5.0) == 0
        for i in range(5):
            assert out.count(f"PROG_{i}") == 1, out
        idxs = [out.index(f"PROG_{i}") for i in range(5)]
        assert idxs == sorted(idxs) and out.count("[STOP-TRIGGER]") == 1
        assert out.index("[STOP-TRIGGER]") > idxs[-1]
    finally:
        _cleanup(proc)
        tmp.cleanup()


def test_tail_file_stop_on_works_when_filters_suppress_line() -> None:
    tmp, log_path, proc, started = _run(
        ["--progress", r"^(READY_PROBE_|ONLY_PROG)"])
    try:
        with open(log_path, "ab", buffering=0) as fh:
            fh.write(f"unrelated noise\n{STOP_TOKEN}\n".encode())
        out = _drain_until(proc, "[STOP-TRIGGER]", 5.0, started)
        assert proc.wait(timeout=5.0) == 0
        assert "[STOP-TRIGGER]" in out and "ONLY_PROG" not in out
    finally:
        _cleanup(proc)
        tmp.cleanup()
