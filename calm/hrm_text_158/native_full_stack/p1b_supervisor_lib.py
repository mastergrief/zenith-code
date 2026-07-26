"""P1b Phase B supervisor facade: packet validation, topology, activation IO.

Importable pure-ish helpers used by the thin CLI orchestrator
``scripts/p1b_phaseB_supervisor.py``. No CLI entrypoint here.
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.p1b_o_excl_copy import write_bytes_o_excl
from scripts.p1b_phase_b_packet_mint import (
    RUNTIME_EXECUTABLE_KEYS,
    compute_packet_payload_digest,
    validate_phase_b_packet_schema,
)

ISOL_WT = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158-p1b-isol-wt")

FROZEN_INNER_SCIENCE_ARGV = (
    "timeout --kill-after=30 900 python3 scripts/train_hrm_text_158.py "
    "--use-ternary-bulk --sub2-authority-live-conversion-proof "
    "--sub2-authority-eligible-scope all-bitlinear --device cuda "
    "--epochs 1 --batch-size 8 --max-len 256 --seed 1"
)

EXPECTED_PHASE_BUDGETS = {
    "model_build": 180,
    "forward_backward": 300,
    "vote_apply": 120,
    "checkpoint_roundtrip": 180,
    "receipt_mint": 60,
}

EXIT_PACKET_MISSING = 61
EXIT_PACKET_SCHEMA = 62
EXIT_PACKET_PAYLOAD = 63
EXIT_PACKET_COMMIT = 64
EXIT_PACKET_FILE_HASH = 65
EXIT_PACKET_ARGV = 66
EXIT_PACKET_PATH_PREEXISTS = 67
EXIT_PACKET_BUDGET_PATH = 68
EXIT_PACKET_FILE_SHA = 69
EXIT_PACKET_SHA_ARG = 70

EXIT_SHARED_PGID = 71
EXIT_WATCHDOG_ARMED_TIMEOUT = 72
EXIT_ACTIVATION_GATE_TIMEOUT = 73
EXIT_LOG_PREEXISTS = 74
EXIT_TEST_SEAM_REFUSED = 75
EXIT_TEST_MODE_CANONICAL_REFUSED = 76
EXIT_OWNERSHIP_FAILURE = 77


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def well_formed_sha256(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return True


def repo_head(cwd: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        text=True,
    ).strip()


def o_excl_touch(path: Path) -> None:
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)


def budget_map(phase_budgets: Mapping[str, Any]) -> dict[str, float]:
    """Normalize packet phase_budgets into name→seconds."""
    out: dict[str, float] = {}
    if "table" in phase_budgets and isinstance(phase_budgets["table"], list):
        for row in phase_budgets["table"]:
            out[str(row["id"])] = float(row["budget_sec"])
        return out
    skip = {
        "authoritative",
        "cite_discipline",
        "silent_phase_seconds_max",
        "total_orchestration_deadline_seconds",
        "superseded_prose_table_note",
    }
    for key, val in phase_budgets.items():
        if key in skip:
            continue
        if isinstance(val, Mapping) and "budget_sec" in val:
            out[str(key)] = float(val["budget_sec"])
        elif isinstance(val, (int, float, str)):
            try:
                out[str(key)] = float(val)
            except ValueError:
                continue
    return out


def validate_packet_pre_spawn(
    packet_path: Path,
    packet_sha256_arg: str | None,
    *,
    cwd: Path,
    allow_noncanonical_paths: bool,
    allow_test_budgets: bool = False,
    allow_test_trainer_command: bool = False,
) -> dict[str, Any]:
    """Run supervisor pre-spawn checks in plan order; SystemExit on refusal."""
    if not packet_sha256_arg or not well_formed_sha256(str(packet_sha256_arg).strip()):
        print("PACKET_SHA256_ARG_ABSENT_OR_MALFORMED", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_SHA_ARG)

    if not packet_path.is_file():
        print("PACKET_MISSING_OR_UNREADABLE", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_MISSING)

    try:
        raw = packet_path.read_bytes()
    except OSError:
        print("PACKET_MISSING_OR_UNREADABLE", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_MISSING)

    file_sha = hashlib.sha256(raw).hexdigest()
    if file_sha != str(packet_sha256_arg).strip().lower():
        print(
            f"PACKET_FILE_SHA256_MISMATCH expected={packet_sha256_arg} got={file_sha}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_PACKET_FILE_SHA)

    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"PACKET_SCHEMA_INCOMPLETE_OR_MALFORMED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_SCHEMA) from exc
    if not isinstance(obj, dict):
        print("PACKET_SCHEMA_INCOMPLETE_OR_MALFORMED: not an object", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_SCHEMA)
    try:
        validate_phase_b_packet_schema(obj)
    except ValueError as exc:
        print(f"PACKET_SCHEMA_INCOMPLETE_OR_MALFORMED: {exc}", file=sys.stderr, flush=True)
        raise SystemExit(EXIT_PACKET_SCHEMA) from exc

    recomputed = compute_packet_payload_digest(obj)
    if recomputed != str(obj.get("packet_payload_digest", "")):
        print(
            f"PACKET_PAYLOAD_DIGEST_MISMATCH expected={obj.get('packet_payload_digest')} got={recomputed}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_PACKET_PAYLOAD)

    head = repo_head(cwd)
    if str(obj["commit_sha"]) != head:
        print(
            f"PACKET_COMMIT_SHA_NE_HEAD packet={obj['commit_sha']} head={head}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_PACKET_COMMIT)

    runtime_hashes = obj["runtime_executable_sha256s"]
    for rel in RUNTIME_EXECUTABLE_KEYS:
        on_disk = cwd / rel
        if not on_disk.is_file():
            print(f"PACKET_RUNTIME_FILE_HASH_MISMATCH missing={rel}", file=sys.stderr, flush=True)
            raise SystemExit(EXIT_PACKET_FILE_HASH)
        got = sha256_file(on_disk)
        want = str(runtime_hashes[rel])
        if got != want:
            print(
                f"PACKET_RUNTIME_FILE_HASH_MISMATCH file={rel} want={want} got={got}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(EXIT_PACKET_FILE_HASH)

    watch_wrap = cwd / "bin/watch-wrap"
    ww_sha = sha256_file(watch_wrap) if watch_wrap.is_file() else ""
    if ww_sha != str(obj["watch_wrap_sha256"]):
        print(
            f"PACKET_RUNTIME_FILE_HASH_MISMATCH watch_wrap want={obj['watch_wrap_sha256']} got={ww_sha}",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(EXIT_PACKET_FILE_HASH)

    if str(obj["inner_science_argv"]) != FROZEN_INNER_SCIENCE_ARGV:
        if not allow_test_trainer_command:
            print("PACKET_SCIENCE_ARGV_MISMATCH", file=sys.stderr, flush=True)
            raise SystemExit(EXIT_PACKET_ARGV)

    budgets = budget_map(obj["phase_budgets"])
    for name, sec in EXPECTED_PHASE_BUDGETS.items():
        if name not in budgets or float(budgets[name]) != float(sec):
            if allow_test_budgets:
                continue
            print(
                f"PACKET_BUDGET_OR_PATH_TAMPER budget={name} want={sec} got={budgets.get(name)}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(EXIT_PACKET_BUDGET_PATH)

    paths = obj["paths"]
    if not allow_noncanonical_paths:
        for key, val in paths.items():
            if not str(val).startswith(str(ISOL_WT)):
                print(
                    f"PACKET_BUDGET_OR_PATH_TAMPER path={key} val={val}",
                    file=sys.stderr,
                    flush=True,
                )
                raise SystemExit(EXIT_PACKET_BUDGET_PATH)

    for key in ("phase_b_log", "activation_receipt", "monitor_armed_touch", "p1b_receipt"):
        p = Path(str(paths[key]))
        if p.exists():
            print(
                f"PACKET_PATH_PREEXISTS_OEXCL_REQUIRED path={p}",
                file=sys.stderr,
                flush=True,
            )
            raise SystemExit(EXIT_PACKET_PATH_PREEXISTS)

    return obj


def build_default_watchdog_cmd(
    packet: Mapping[str, Any],
    *,
    train_pid: int,
    train_pgid: int,
    supervisor_pgid: int,
) -> list[str]:
    budgets = budget_map(packet["phase_budgets"])
    budget_str = ",".join(f"{k}={int(v)}" for k, v in budgets.items())
    paths = packet["paths"]
    return [
        sys.executable,
        "scripts/p1b_phase_watchdog.py",
        "--log",
        str(paths["phase_b_log"]),
        "--target-pid",
        str(train_pid),
        "--target-pgid",
        str(train_pgid),
        "--event-log",
        str(paths["watchdog_event_log"]),
        "--budgets",
        budget_str,
        "--marker-prefix",
        "[P1B_PHASE]",
        "--require-marker-order-before-enforce",
        "--require-monitor-armed-touch",
        str(paths["monitor_armed_touch"]),
        "--supervisor-pgid",
        str(supervisor_pgid),
        "--on-breach",
        "kill-process-group",
    ]


def wait_for_watchdog_armed(
    event_log: Path,
    deadline_sec: float,
    *,
    watchdog_proc: subprocess.Popen,
    log_path: Path | None = None,
) -> str | None:
    deadline = time.monotonic() + deadline_sec
    while time.monotonic() < deadline:
        if event_log.is_file():
            text = event_log.read_text(encoding="utf-8", errors="replace")
            for line in text.splitlines():
                if "WATCHDOG_ARMED" in line:
                    return line
        if log_path is not None and log_path.is_file():
            for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
                if line.startswith("WATCHDOG_ARMED") or "WATCHDOG_ARMED " in line:
                    return line
        if watchdog_proc.poll() is not None and watchdog_proc.returncode not in (None, 0):
            break
        time.sleep(0.05)
    return None


def wait_for_monitor_touch(touch: Path, deadline_sec: float) -> bool:
    deadline = time.monotonic() + deadline_sec
    while time.monotonic() < deadline:
        if touch.exists():
            return True
        time.sleep(0.05)
    return False


def build_activation_receipt(
    *,
    train_pid: int,
    train_pgid: int,
    supervisor_pid: int,
    supervisor_pgid: int,
    watchdog_pid: int,
    watchdog_pgid: int,
    watchdog_command_exact: Sequence[str],
    watchdog_sha256: str,
    watch_wrap_command_exact: str,
    watch_wrap_sha256: str,
    trainer_argv: Sequence[str],
    log_path: Path,
    packet_path: Path,
    packet_file_sha256: str,
    packet_payload_digest: str,
    watchdog_activation_line: str,
    activation_deadlines: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "train_pid": train_pid,
        "train_pgid": train_pgid,
        "supervisor_pid": supervisor_pid,
        "supervisor_pgid": supervisor_pgid,
        "watchdog_pid": watchdog_pid,
        "watchdog_pgid": watchdog_pgid,
        "watchdog_command_exact": list(watchdog_command_exact),
        "watchdog_sha256": watchdog_sha256,
        "watch_wrap_command_exact": str(watch_wrap_command_exact),
        "watch_wrap_sha256": str(watch_wrap_sha256),
        "wall_clock_unix": time.time(),
        "monotonic_ns": time.monotonic_ns(),
        "TRAIN_PGID_ne_SUPERVISOR_PGID": train_pgid != supervisor_pgid,
        "watchdog_pgid_ne_TRAIN_PGID": watchdog_pgid != train_pgid,
        "science_argv_exact": list(trainer_argv),
        "phase_B_log_path": str(log_path),
        "packet_path": str(packet_path),
        "packet_file_sha256": str(packet_file_sha256),
        "packet_payload_digest": str(packet_payload_digest),
        "watchdog_activation_line": watchdog_activation_line,
        "activation_deadlines": dict(activation_deadlines),
    }


def write_activation_receipt_o_excl(path: Path, receipt: Mapping[str, Any]) -> str:
    return write_bytes_o_excl(
        path,
        (json.dumps(dict(receipt), indent=2, sort_keys=True) + "\n").encode("utf-8"),
    )


def terminate_children(
    *,
    train_pgid: int | None,
    trainer: subprocess.Popen | None,
    watchdog: subprocess.Popen | None,
    protected_pgids: set[int] | frozenset[int] | None = None,
    train_wait: float = 10.0,
    watchdog_wait: float = 10.0,
) -> tuple[int | None, int | None]:
    """Kill trainer group + terminate watchdog; reap both. Never killpg protected PGIDs.

    Protected set always includes the caller's PGID. When ``train_pgid`` is protected
    (shared-group anomaly), use PID-only trainer cleanup — never ``killpg`` self.
    """
    protected: set[int] = set(protected_pgids or ())
    try:
        protected.add(os.getpgid(0))
    except OSError:
        pass

    if train_pgid is not None and train_pgid not in protected:
        try:
            os.killpg(train_pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif train_pgid is not None and train_pgid in protected:
        # Shared-PGID anomaly: kill trainer PID only; never killpg(supervisor).
        if trainer is not None and trainer.pid is not None:
            try:
                os.kill(int(trainer.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass

    if trainer is not None and trainer.poll() is None:
        try:
            trainer.kill()
        except ProcessLookupError:
            pass
    if watchdog is not None and watchdog.poll() is None:
        try:
            watchdog.terminate()
        except ProcessLookupError:
            pass
    train_rc = None
    watchdog_rc = None
    if trainer is not None:
        try:
            train_rc = trainer.wait(timeout=train_wait)
        except subprocess.TimeoutExpired:
            try:
                trainer.kill()
            except ProcessLookupError:
                pass
            train_rc = trainer.wait(timeout=5)
    if watchdog is not None:
        try:
            watchdog_rc = watchdog.wait(timeout=watchdog_wait)
        except subprocess.TimeoutExpired:
            try:
                watchdog.kill()
            except ProcessLookupError:
                pass
            watchdog_rc = watchdog.wait(timeout=5)
    return train_rc, watchdog_rc


def parse_shell_argv(command: str) -> list[str]:
    return shlex.split(command)


__all__ = [
    "EXPECTED_PHASE_BUDGETS",
    "EXIT_ACTIVATION_GATE_TIMEOUT",
    "EXIT_LOG_PREEXISTS",
    "EXIT_OWNERSHIP_FAILURE",
    "EXIT_PACKET_ARGV",
    "EXIT_PACKET_BUDGET_PATH",
    "EXIT_PACKET_COMMIT",
    "EXIT_PACKET_FILE_HASH",
    "EXIT_PACKET_FILE_SHA",
    "EXIT_PACKET_MISSING",
    "EXIT_PACKET_PATH_PREEXISTS",
    "EXIT_PACKET_PAYLOAD",
    "EXIT_PACKET_SCHEMA",
    "EXIT_PACKET_SHA_ARG",
    "EXIT_SHARED_PGID",
    "EXIT_TEST_MODE_CANONICAL_REFUSED",
    "EXIT_TEST_SEAM_REFUSED",
    "EXIT_WATCHDOG_ARMED_TIMEOUT",
    "FROZEN_INNER_SCIENCE_ARGV",
    "ISOL_WT",
    "budget_map",
    "build_activation_receipt",
    "build_default_watchdog_cmd",
    "o_excl_touch",
    "parse_shell_argv",
    "repo_head",
    "sha256_file",
    "terminate_children",
    "validate_packet_pre_spawn",
    "wait_for_monitor_touch",
    "wait_for_watchdog_armed",
    "well_formed_sha256",
    "write_activation_receipt_o_excl",
]
