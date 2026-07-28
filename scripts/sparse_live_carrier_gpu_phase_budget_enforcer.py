#!/usr/bin/env python3
"""Per-phase GPU budget enforcer (PLAN_v16 material: findings 7-9 + R3/R4).

Real-time JSONL polling; process-group SIGTERM→grace→SIGKILL on overrun.
Event schema: type + ts_monotonic + node_id + phase (+ duration_s on END).
Terminal class strings EXACT: GPU-SMOKE-FAIL/TIMEOUT, GPU-SMOKE-FAIL/PHASE_TELEMETRY.
Exit: TIMEOUT→124, PHASE_TELEMETRY→2, OK→0.
R3: ONE fail-telemetry path — first error, TERM→KILL once, stop, mint PHASE_TELEMETRY, exit 2.
R4: good_topology self-test freezes formal node acceptance stream.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

PHASE_ORDER = ("forward_backward", "update", "emission", "flush")
ENV_JSONL = "SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"
CLASS_TIMEOUT = "GPU-SMOKE-FAIL/TIMEOUT"
CLASS_TELEMETRY = "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
CLASS_OK = "OK"
CLASS_CHILD = "GPU-SMOKE-FAIL/CHILD_NONZERO"


def _o_excl_write(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_budgets(packet_path: Path | None, cli_budgets: dict[str, float]) -> dict[str, float]:
    budgets = {p: 30.0 for p in PHASE_ORDER}
    budgets.update(cli_budgets)
    if packet_path and packet_path.exists():
        pkt = json.loads(packet_path.read_text())
        pb = (pkt.get("packet_requirements") or {}).get("per_phase_budgets") or pkt.get(
            "per_phase_budgets"
        ) or {}
        for phase in PHASE_ORDER:
            if phase in pb:
                budgets[phase] = float(pb[phase])
    return budgets


def _kill_tree(pgid: int, grace: float = 0.5) -> list[str]:
    actions: list[str] = []
    try:
        os.killpg(pgid, signal.SIGTERM)
        actions.append("TERM")
        time.sleep(grace)
        try:
            os.killpg(pgid, signal.SIGKILL)
            actions.append("KILL")
        except ProcessLookupError:
            pass
    except ProcessLookupError:
        pass
    return actions


def _fail_telemetry(terminal: dict[str, Any], *, error: str, pgid: int, killed: bool) -> bool:
    """R3: ONE fail-telemetry path — record FIRST error only, TERM→KILL once, stop."""
    if killed:
        return True
    terminal["status"] = "GPU-SMOKE-FAIL"
    terminal["terminal_class"] = CLASS_TELEMETRY
    if not terminal.get("error"):
        terminal["error"] = error
    terminal["kill_actions"] = _kill_tree(pgid)
    return True


def _read_new_events(path: Path, offset: int) -> tuple[list[dict[str, Any]], int, list[str]]:
    """Return (events, new_offset, malformed_errors). Malformed lines are NOT silent."""
    if not path.exists():
        return [], offset, []
    data = path.read_bytes()
    if len(data) <= offset:
        return [], offset, []
    chunk = data[offset:].decode("utf-8", errors="replace")
    events: list[dict[str, Any]] = []
    malformed: list[str] = []
    for line in chunk.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except json.JSONDecodeError as exc:
            malformed.append(str(exc))
    return events, len(data), malformed


def _self_test_child(self_test: str) -> list[str] | None:
    """Return synthetic child argv for self-tests, or None if not a self-test."""
    emit_prefix = (
        "import json,os,time,sys\n"
        f"p=os.environ[{ENV_JSONL!r}]\n"
        "nid=os.environ.get('SPARSE_LIVE_CARRIER_EXPECTED_NODE_ID')\n"
    )
    if self_test == "overrun":
        code = (
            "import json,os,time\n"
            f"p=os.environ[{ENV_JSONL!r}]\n"
            "nid=os.environ.get('SPARSE_LIVE_CARRIER_EXPECTED_NODE_ID')\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            "time.sleep(3)\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':3})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "missing_node_id":
        code = (
            "import json,os,time\n"
            f"p=os.environ[{ENV_JSONL!r}]\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','ts_monotonic':time.monotonic()})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "missing_duration":
        code = (
            "import json,os,time\n"
            f"p=os.environ[{ENV_JSONL!r}]\n"
            "nid=os.environ.get('SPARSE_LIVE_CARRIER_EXPECTED_NODE_ID')\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "unknown_type":
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_WEIRD','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "for ph in ('forward_backward','update','emission','flush'):\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_START','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_END','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':0.01})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "wrong_duration_type":
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':'3'})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "negative_duration":
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':-1})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "non_string_node_id":
        code = (
            "import json,os,time\n"
            f"p=os.environ[{ENV_JSONL!r}]\n"
            "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':123,'ts_monotonic':time.monotonic()})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "child_nonzero":
        code = (
            emit_prefix
            + "for ph in ('forward_backward','update','emission','flush'):\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_START','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_END','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':0.01})+'\\n')\n"
            + "sys.exit(1)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "malformed_then_hang":
        code = (
            "import os,time\n"
            f"p=os.environ[{ENV_JSONL!r}]\n"
            "open(p,'a').write('NOT-JSON{{{{\\n')\n"
            "time.sleep(30)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "good_topology":
        # R4: exact formal stream — single START/END per phase, order, valid types, exit 0
        code = (
            emit_prefix
            + "for ph in ('forward_backward','update','emission','flush'):\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_START','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "  open(p,'a').write(json.dumps({'type':'PHASE_END','phase':ph,'node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':0.01})+'\\n')\n"
            + "sys.exit(0)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "invalid_duration_then_hang":
        # R3(a)
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':'bad'})+'\\n')\n"
            + "time.sleep(30)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "unknown_phase_start_then_hang":
        # R3(b): budget-less open-phase counterexample
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'not_a_budget_phase','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "time.sleep(30)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "duplicate_start_then_hang":
        # R3(c)
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':0.01})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "time.sleep(30)\n"
        )
        return [sys.executable, "-c", code]
    if self_test == "missing_coverage":
        code = (
            emit_prefix
            + "open(p,'a').write(json.dumps({'type':'PHASE_START','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic()})+'\\n')\n"
            + "open(p,'a').write(json.dumps({'type':'PHASE_END','phase':'forward_backward','node_id':nid,'ts_monotonic':time.monotonic(),'duration_s':0.01})+'\\n')\n"
        )
        return [sys.executable, "-c", code]
    return None


def run_enforcer(
    *,
    child_argv: list[str],
    budgets: dict[str, float],
    phase_events_jsonl: Path,
    enforcer_receipt: Path,
    expected_node_id: str,
    self_test: str | None = None,
) -> int:
    if enforcer_receipt.exists():
        raise SystemExit(f"enforcer receipt exists (O_EXCL): {enforcer_receipt}")
    if phase_events_jsonl.exists():
        raise SystemExit(f"phase events jsonl exists (O_EXCL): {phase_events_jsonl}")
    phase_events_jsonl.parent.mkdir(parents=True, exist_ok=True)
    enforcer_receipt.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(phase_events_jsonl), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    os.close(fd)

    synthetic = _self_test_child(self_test) if self_test else None
    if synthetic is not None:
        child_argv = synthetic
    if self_test == "overrun":
        budgets = {**budgets, "forward_backward": 0.15}

    env = os.environ.copy()
    env[ENV_JSONL] = str(phase_events_jsonl)
    env["SPARSE_LIVE_CARRIER_EXPECTED_NODE_ID"] = expected_node_id

    terminal: dict[str, Any] = {
        "status": "OK",
        "terminal_class": CLASS_OK,
        "expected_node_id": expected_node_id,
        "observed_node_ids": [],
        "phases_seen": [],
        "raw_events": [],
        "overrun_phase": None,
        "kill_actions": [],
        "child_rc": None,
        "phase_events_jsonl": str(phase_events_jsonl),
        "phase_events_line_count": 0,
        "phase_events_sha256": None,
        "error": None,
    }

    proc = subprocess.Popen(
        child_argv,
        env=env,
        preexec_fn=os.setsid,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert proc.pid
    pgid = os.getpgid(proc.pid)
    offset = 0
    open_phase: str | None = None
    open_phase_start: float | None = None
    seen_starts: list[str] = []
    seen_ends: list[str] = []
    last_ts: float | None = None
    killed = False

    while True:
        events, offset, malformed = _read_new_events(phase_events_jsonl, offset)
        if malformed and not killed:
            killed = _fail_telemetry(
                terminal, error=f"malformed JSONL: {malformed[0]}", pgid=pgid, killed=killed
            )
            break
        stop_events = False
        for ev in events:
            terminal["raw_events"].append(ev)
            etype = ev.get("type")
            if etype not in ("PHASE_START", "PHASE_END"):
                killed = _fail_telemetry(
                    terminal, error=f"unknown or missing type {etype!r}", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            ts = ev.get("ts_monotonic")
            if not isinstance(ts, (int, float)) or isinstance(ts, bool) or not math.isfinite(float(ts)):
                killed = _fail_telemetry(
                    terminal, error="ts_monotonic must be numeric finite", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            nid = ev.get("node_id")
            if not isinstance(nid, str) or not nid:
                killed = _fail_telemetry(
                    terminal, error="node_id must be non-empty str", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            phase = ev.get("phase")
            if not isinstance(phase, str) or not phase:
                killed = _fail_telemetry(
                    terminal, error="phase must be non-empty str", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            if etype == "PHASE_END":
                dur = ev.get("duration_s")
                if not isinstance(dur, (int, float)) or isinstance(dur, bool) or not math.isfinite(float(dur)) or float(dur) < 0:
                    killed = _fail_telemetry(
                        terminal,
                        error="duration_s must be numeric finite >= 0",
                        pgid=pgid,
                        killed=killed,
                    )
                    stop_events = True
                    break
            terminal["observed_node_ids"].append(nid)
            if nid != expected_node_id:
                killed = _fail_telemetry(
                    terminal,
                    error=f"node_id mismatch {nid} != {expected_node_id}",
                    pgid=pgid,
                    killed=killed,
                )
                stop_events = True
                break
            if phase not in PHASE_ORDER:
                killed = _fail_telemetry(
                    terminal, error=f"unknown phase {phase}", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            if last_ts is not None and float(ts) < float(last_ts):
                killed = _fail_telemetry(
                    terminal, error="non-monotonic ts_monotonic", pgid=pgid, killed=killed
                )
                stop_events = True
                break
            last_ts = float(ts)
            if etype == "PHASE_START":
                if open_phase is not None:
                    killed = _fail_telemetry(
                        terminal,
                        error=f"nested/unpaired START while open={open_phase}",
                        pgid=pgid,
                        killed=killed,
                    )
                    stop_events = True
                    break
                if phase in seen_starts:
                    killed = _fail_telemetry(
                        terminal, error=f"duplicate START {phase}", pgid=pgid, killed=killed
                    )
                    stop_events = True
                    break
                open_phase = str(phase)
                open_phase_start = time.monotonic()
                seen_starts.append(str(phase))
                terminal["phases_seen"].append(f"START:{phase}")
            elif etype == "PHASE_END":
                if open_phase != phase:
                    killed = _fail_telemetry(
                        terminal,
                        error=f"END without matching START phase={phase} open={open_phase}",
                        pgid=pgid,
                        killed=killed,
                    )
                    stop_events = True
                    break
                if phase in seen_ends:
                    killed = _fail_telemetry(
                        terminal, error=f"duplicate END {phase}", pgid=pgid, killed=killed
                    )
                    stop_events = True
                    break
                seen_ends.append(str(phase))
                terminal["phases_seen"].append(f"END:{phase}")
                open_phase = None
                open_phase_start = None
        if killed or stop_events:
            break

        if open_phase is not None and open_phase_start is not None and not killed:
            budget = float(budgets.get(open_phase, 0) or 0)
            if budget > 0 and (time.monotonic() - open_phase_start) > budget:
                terminal["status"] = "GPU-SMOKE-FAIL"
                terminal["terminal_class"] = CLASS_TIMEOUT
                terminal["overrun_phase"] = open_phase
                terminal["kill_actions"] = _kill_tree(pgid)
                killed = True
                break

        rc = proc.poll()
        if rc is not None and not events:
            more, offset, more_malformed = _read_new_events(phase_events_jsonl, offset)
            if more_malformed and not killed:
                killed = _fail_telemetry(
                    terminal,
                    error=f"malformed JSONL: {more_malformed[0]}",
                    pgid=pgid,
                    killed=killed,
                )
                break
            if more:
                continue
            break
        time.sleep(0.02)

    if killed:
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            terminal["kill_actions"] = list(terminal.get("kill_actions") or []) + _kill_tree(pgid)
            proc.wait(timeout=2)
    else:
        try:
            proc.wait(timeout=1)
        except subprocess.TimeoutExpired:
            terminal["kill_actions"] = _kill_tree(pgid)
            proc.wait(timeout=2)

    terminal["child_rc"] = proc.returncode
    terminal["observed_node_ids"] = sorted(set(terminal["observed_node_ids"]))
    terminal["phase_events_line_count"] = len(terminal["raw_events"])
    terminal["phase_events_sha256"] = _sha(phase_events_jsonl)

    if self_test == "overrun":
        if terminal["terminal_class"] != CLASS_TIMEOUT:
            for ev in terminal["raw_events"]:
                if (
                    (ev.get("type") == "PHASE_END" or ev.get("kind") == "PHASE_END")
                    and ev.get("phase") == "forward_backward"
                    and float(ev.get("duration_s") or 0) > float(budgets.get("forward_backward", 0) or 0)
                ):
                    terminal["status"] = "GPU-SMOKE-FAIL"
                    terminal["terminal_class"] = CLASS_TIMEOUT
                    terminal["overrun_phase"] = "forward_backward"
                    if not terminal["kill_actions"]:
                        terminal["kill_actions"] = _kill_tree(pgid)
    elif terminal["terminal_class"] == CLASS_OK:
        if seen_starts != list(PHASE_ORDER) or seen_ends != list(PHASE_ORDER):
            terminal["status"] = "GPU-SMOKE-FAIL"
            terminal["terminal_class"] = CLASS_TELEMETRY
            terminal["error"] = f"missing phase coverage starts={seen_starts} ends={seen_ends}"

    if self_test == "missing_coverage" and terminal["terminal_class"] == CLASS_OK:
        terminal["status"] = "GPU-SMOKE-FAIL"
        terminal["terminal_class"] = CLASS_TELEMETRY

    # B4: terminal OK requires child_rc == 0
    if terminal["terminal_class"] == CLASS_OK and terminal.get("child_rc") not in (0,):
        terminal["status"] = "GPU-SMOKE-FAIL"
        terminal["terminal_class"] = CLASS_CHILD
        terminal["error"] = f"child_rc={terminal.get('child_rc')} nonzero with complete phases"

    raw = terminal.pop("raw_events")
    terminal["raw_event_count"] = len(raw)
    if terminal.get("error") is None:
        terminal.pop("error", None)

    _o_excl_write(enforcer_receipt, terminal)
    print(json.dumps(terminal, indent=2))
    tc = terminal["terminal_class"]
    if tc == CLASS_OK:
        return 0
    if tc == CLASS_TIMEOUT:
        return 124
    return 2


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--packet", default=None)
    ap.add_argument("--phase-events-jsonl", required=True)
    ap.add_argument("--enforcer-receipt", required=True)
    ap.add_argument("--expected-node-id", required=True)
    ap.add_argument("--self-test-phase-overrun", action="store_true")
    ap.add_argument("--self-test-missing-coverage", action="store_true")
    ap.add_argument(
        "--self-test",
        choices=(
            "overrun",
            "missing_coverage",
            "missing_node_id",
            "missing_duration",
            "unknown_type",
            "wrong_duration_type",
            "negative_duration",
            "non_string_node_id",
            "child_nonzero",
            "malformed_then_hang",
            "good_topology",
            "invalid_duration_then_hang",
            "unknown_phase_start_then_hang",
            "duplicate_start_then_hang",
        ),
        default=None,
    )
    ap.add_argument("--budget", action="append", default=[])
    ap.add_argument("child", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    self_test = args.self_test
    if args.self_test_phase_overrun:
        self_test = "overrun"
    if args.self_test_missing_coverage:
        self_test = "missing_coverage"
    cli = {}
    for item in args.budget:
        k, v = item.split("=", 1)
        cli[k] = float(v)
    budgets = _load_budgets(Path(args.packet) if args.packet else None, cli)
    child = list(args.child)
    if child and child[0] == "--":
        child = child[1:]
    if not child and not self_test:
        raise SystemExit("child argv required unless self-test")
    return run_enforcer(
        child_argv=child or [sys.executable, "-c", "pass"],
        budgets=budgets,
        phase_events_jsonl=Path(args.phase_events_jsonl),
        enforcer_receipt=Path(args.enforcer_receipt),
        expected_node_id=args.expected_node_id,
        self_test=self_test,
    )


if __name__ == "__main__":
    raise SystemExit(main())
