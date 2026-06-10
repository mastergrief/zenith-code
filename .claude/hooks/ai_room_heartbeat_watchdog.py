#!/usr/bin/env python3
"""ai-room heartbeat watchdog — clock-driven deterministic stall detection + wake.

THE GAP THIS CLOSES: a wedged worker emits no event, so no turn-driven agent
(claude or co_lead) is ever re-invoked to notice it stalled. Channel pushes and
PreToolUse hooks are event-driven — they cannot fire on *silence*. Only an
external clock can. This script is that clock: cron-armed (every 5-10 min), it
reads the room, finds gated work past its heartbeat deadline, proves liveness
read-only, and on no-movement posts a WAKE-BEARING `WATCHDOG_STALL`
(`requires_response_from=claude`) that re-invokes the idle orchestrator — the
external event the wedge swallowed.

NON-DESTRUCTIVE by design: detect, prove, post, wake. NEVER auto-kills (that
risks destroying unsent worker state). Claude decides recycle/re-drive/extend
from the proof packet.

Movement proof is PHASE-AWARE, FRESH, and CORRELATED (co_lead hardening):
  - FRESH: movement counts only since the LAST watchdog check (latest watchdog
    post for this worker), not since the original heartbeat — else a
    moved-once-then-wedged worker would extend forever.
  - PHASE-AWARE: code phases (edit/compile/cpu-proof/dry-run/receipt) prove
    movement by artifact/run-dir freshness ONLY (a code slice needs no live
    trainer); gpu phases (gpu-proof/launch) also accept a run-dir-CORRELATED
    process — never bare process/GPU existence (which an unrelated job fakes).
  - ESCALATION: an EXTEND followed by no fresh movement, or any prior STALL,
    escalates to a recycle-recommending STALL.
  - RETIRE: after RETIRE_THRESHOLD EXACT-worker wake STALLs with no movement, a
    dead heartbeat is RETIRED (one non-wake notice, then CLEAN) — stops a killed
    handle (e.g. `codex_1`) from crying wolf forever. Re-monitors on a newer
    heartbeat. Worker attribution is exact (`\bworker <handle>\b`), so a retired
    `codex_1` stream never counts against live `codex`.
Per-worker: each worker's latest gated post decides its own state — worker B's
terminal never cleans worker A's overdue heartbeat.

Companion to `worker_gate_wake_pairing_gate.py` (which guarantees claude PAIRS
the wake when it gates); this catches a worker that wedges AFTER being woken.

UNANSWERED-GATE TRACKING (second detection stream, 2026-06-10): the heartbeat
stream above only sees workers that POSTED a heartbeat. The ack-idle failure
(worker reads a deadline-bearing execution gate, its turn ends, nothing ever
re-fires it) is invisible to it. This stream closes that gap: any claude post
with `requires_response_from=<worker>` + `response_deadline_secs` whose
deadline lapsed with NO worker reply (threaded reply_to OR body citation of
the gate id) gets an automatic GATE_REWAKE posted directly to the worker
(bounded: GATE_REWAKE_MAX), then a single wake-bearing GATE_ESCALATE to
claude. Non-destructive throughout — wake and report, never kill.

Usage:
  ai_room_heartbeat_watchdog.py [--dry-run] [--channel-log PATH] [--now ISO]
Cron (every 7 min):
  */7 * * * * /usr/bin/env python3 .../ai_room_heartbeat_watchdog.py >> /tmp/ai_room_watchdog.log 2>&1
Exit 0 always (fail-quiet — a watchdog bug must never wedge the room).
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import datetime, timezone

# --- tunables -----------------------------------------------------------------
DEFAULT_CHANNEL_LOG = "/home/gabe/.ai-room/channels/claw-code/messages.jsonl"
TAIL_LINES = 4000
GRACE_SECONDS = 600
DEFAULT_DUE_SECONDS = 1800       # gated IMPLEMENTING without metadata ⇒ due-soon
STALL_RESPONSE_DEADLINE = 1800
RETIRE_THRESHOLD = 3             # exact-worker wake STALLs with no movement before a
                                 # dead heartbeat is retired (ceases alerts; re-monitors
                                 # on a newer heartbeat). Stops dead-handle cry-wolf.
GPU_ACTIVE_MIB = 2000            # reported only; NOT a movement signal by itself
GATE_GRACE_SECONDS = 120         # slack past response_deadline_secs before acting
GATE_REWAKE_MAX = 2              # direct worker re-wakes before escalating to claude
GATE_REWAKE_DEADLINE = 300       # deadline carried on the automatic re-wake
GATE_DEFAULT_DEADLINE = 300      # requires_response_from with no explicit deadline
GATE_MAX_AGE_SECONDS = 7200      # never resurrect gates older than this (recycled
                                 # handles / superseded work make ancient gates moot)
PROCESS_PATTERNS = ("transient_fp_credit_science_train", "train_hrm_text_158",
                    "calm.hrm.train")
NON_WORKER_HANDLES = {"claude", "codex_co_lead", "gabe", "watchdog"}
WATCHDOG_FROM = "watchdog"
HEARTBEAT_MARKERS = ("IMPLEMENTING", "MILESTONE HEARTBEAT", "next_heartbeat_due")
TERMINAL_MARKERS = ("VALIDATION RECEIPT", "VALIDATION_RECEIPT", "CONFOUNDED-NULL",
                    "TERMINAL RECEIPT", "task_complete", "_acquires", "PACKET HOLE",
                    "PUSH RECEIPT")
CODE_PHASES = {"edit", "compile", "cpu-proof", "dry-run", "receipt", ""}
GPU_PHASES = {"gpu-proof", "launch"}

ISO_DUE_RE = re.compile(r"next_heartbeat_due\s*=\s*([0-9T:\-]+Z?)")
PHASE_RE = re.compile(r"\bphase\s*=\s*([a-z\-]+)")
ARTIFACT_RE = re.compile(r"expected_next_artifact\s*=\s*([^\n]+)")
PATHISH_RE = re.compile(r"(/home/[^\s'\"`]+\.(?:json|jsonl|log|pt|py))")
RUNDIR_RE = re.compile(r"(?:--run-dir\s+|run[_-]dir[=:\s]+)(\S+)")
TASKID_RE = re.compile(r"\b(\d{13}-[0-9a-f]{6,8})\b")
POSTED_ID_RE = re.compile(r"\bposted id=([^\s]+)")


def _parse_ts(s):
    if not s:
        return None
    try:
        s = s.strip()
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except Exception:
        return None


def _record_ts(rec):
    ts = rec.get("ts")
    if isinstance(ts, (int, float)):
        return float(ts) / (1000.0 if ts > 1e12 else 1.0)
    if isinstance(ts, str):
        v = _parse_ts(ts)
        if v is not None:
            return v
        try:
            n = float(ts)
            return n / (1000.0 if n > 1e12 else 1.0)
        except Exception:
            return None
    return None


def _read_tail(path, n):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except Exception:
        return []
    out = []
    for line in lines[-n:]:
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            continue
    return out


def _body(rec):
    b = rec.get("body")
    return b if isinstance(b, str) else ""


def _is_worker(handle):
    return bool(handle) and handle not in NON_WORKER_HANDLES


def _build_hb(worker, rec, t_h):
    body = _body(rec)
    due_m = ISO_DUE_RE.search(body)
    due = _parse_ts(due_m.group(1)) if due_m else None
    if due is None and t_h is not None:
        due = t_h + DEFAULT_DUE_SECONDS
    phase_m = PHASE_RE.search(body)
    phase = phase_m.group(1) if phase_m else ""
    art_m = ARTIFACT_RE.search(body)
    artifact = art_m.group(1).strip() if art_m else ""
    path_m = PATHISH_RE.search(artifact) or PATHISH_RE.search(body)
    rd_m = RUNDIR_RE.search(body)
    task_m = TASKID_RE.search(body)
    return {
        "worker": worker, "hb_id": rec.get("id", ""), "hb_ts": t_h, "due": due,
        "phase": phase, "artifact": artifact,
        "artifact_path": path_m.group(1) if path_m else "",
        "run_dir": rd_m.group(1) if rd_m else "",
        "task_id": task_m.group(1) if task_m else "",
        "metadata_present": bool(due_m),
    }


def find_active_heartbeats(records):
    """Per-worker latest gated post; return hb dicts for workers whose latest is
    a heartbeat (terminal ⇒ that worker is done). Worker B's terminal cannot
    clean worker A's overdue heartbeat."""
    latest = {}  # worker -> (ts, rec, is_hb)
    for rec in records:
        frm = rec.get("from", "")
        if not _is_worker(frm):
            continue
        body = _body(rec)
        is_terminal = any(m in body for m in TERMINAL_MARKERS)
        is_hb = any(m in body for m in HEARTBEAT_MARKERS)
        if not (is_terminal or is_hb):
            continue
        t = _record_ts(rec)
        if t is None:
            continue
        prev = latest.get(frm)
        if prev is None or t >= prev[0]:
            latest[frm] = (t, rec, is_hb)  # heartbeat wins ties / supersedes by ts
    out = []
    for frm, (t, rec, is_hb) in latest.items():
        if is_hb:
            out.append(_build_hb(frm, rec, t))
    return out


def _referenced_worker(body):
    """Extract the `worker <handle>` token from a watchdog body, or None. Exact
    worker attribution is AUTHORITATIVE: a body naming a worker counts ONLY for
    that worker — a stale `codex_1` stream never counts against live `codex`,
    even when they share a task id (the substring bug AND the task-fallback
    cross-contamination, both co_lead catches). Task fallback (below) applies
    ONLY to bodies with NO worker token (legacy/history)."""
    m = re.search(r"\bworker\s+([A-Za-z0-9_]+)", body)
    return m.group(1) if m else None


def _references_task(body, task):
    """Exact task attribution fallback (task-ids are unique 13-digit-hex). Used
    ONLY when a watchdog body has no `worker <handle>` token to attribute by."""
    return bool(task) and re.search(rf"\b{re.escape(task)}\b", body) is not None


def watchdog_history(records, hb):
    """Room-derived state for this worker's current heartbeat:
    (last_check_ts, n_extends, n_stalls, n_retired) from watchdog posts after
    hb_ts that EXACTLY reference this worker (or its task). Exact match (not
    substring) so a retired `codex_1` stream never counts against live `codex`."""
    since = hb.get("hb_ts")
    worker, task = hb.get("worker", ""), hb.get("task_id", "")
    last_check, n_ext, n_stall, n_retired = since, 0, 0, 0
    if since is None:
        return since, 0, 0, 0
    for rec in records:
        if rec.get("from") != WATCHDOG_FROM:
            continue
        t = _record_ts(rec)
        if t is None or t <= since:
            continue
        body = _body(rec)
        ref_worker = _referenced_worker(body)
        if ref_worker is not None:
            if ref_worker != worker:
                continue            # body names a DIFFERENT worker — never counts (authoritative)
        elif not _references_task(body, task):
            continue                # no worker token AND task mismatch — skip
        if last_check is None or t > last_check:
            last_check = t
        if "WATCHDOG_RETIRED" in body:
            n_retired += 1
        elif "WATCHDOG_STALL" in body:
            n_stall += 1
        elif "WATCHDOG_HEARTBEAT_EXTEND" in body:
            n_ext += 1
    return last_check, n_ext, n_stall, n_retired


def _max_mtime(paths):
    best = None
    for p in paths:
        if not p:
            continue
        try:
            if os.path.isfile(p):
                m = os.path.getmtime(p)
            elif os.path.isdir(p):
                m = max([os.path.getmtime(os.path.join(p, f))
                         for f in os.listdir(p)] or [os.path.getmtime(p)])
            else:
                continue
            best = m if best is None else max(best, m)
        except Exception:
            continue
    return best


def prove_liveness(hb, last_check_ts):
    """Phase-aware, FRESH, CORRELATED movement proof. Returns evidence + `moved`.

    Test injection (deterministic fixtures): AIWD_TEST_FILE_FRESH ("0"/"1"),
    AIWD_TEST_PROC_CORRELATED ("0"/"1"), AIWD_TEST_GPU_MIB (int). Unset ⇒ real
    read-only probing."""
    phase = hb.get("phase", "")
    ev = {"phase": phase, "file_fresh": False, "proc_correlated": False,
          "gpu_used_mib": None, "fresh_since": last_check_ts}

    f_inj = os.environ.get("AIWD_TEST_FILE_FRESH")
    p_inj = os.environ.get("AIWD_TEST_PROC_CORRELATED")
    g_inj = os.environ.get("AIWD_TEST_GPU_MIB")

    if f_inj in ("0", "1"):
        ev["file_fresh"] = f_inj == "1"
    else:
        mt = _max_mtime([hb.get("artifact_path"), hb.get("run_dir")])
        ev["file_fresh"] = bool(mt and last_check_ts and mt > last_check_ts)

    if p_inj in ("0", "1"):
        ev["proc_correlated"] = p_inj == "1"
    else:
        token = os.path.basename((hb.get("run_dir") or hb.get("artifact_path") or "").rstrip("/"))
        if token:
            try:
                ps = subprocess.run(["ps", "-eo", "args"], capture_output=True,
                                    text=True, timeout=10)
                # correlated = a trainer process whose cmdline names this run
                ev["proc_correlated"] = (token in ps.stdout and
                                         any(p in ps.stdout for p in PROCESS_PATTERNS))
            except Exception:
                pass

    if g_inj is not None:
        try:
            ev["gpu_used_mib"] = int(g_inj)
        except Exception:
            pass
    elif shutil.which("nvidia-smi"):
        try:
            g = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                               "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10)
            first = (g.stdout.strip().splitlines() or [""])[0]
            ev["gpu_used_mib"] = int(first) if first.isdigit() else None
        except Exception:
            pass

    if phase in GPU_PHASES:
        ev["moved"] = bool(ev["file_fresh"] or ev["proc_correlated"])
    else:  # code phases / unknown: artifact/run-dir freshness ONLY
        ev["moved"] = bool(ev["file_fresh"])
    return ev


def decide(hb, ev, n_extends, n_stalls, n_retired):
    if ev["moved"]:
        return {"action": "extend", "kind": "status_update", "wake": False,
                "body": (
                    f"WATCHDOG_HEARTBEAT_EXTEND — task {hb.get('task_id') or '?'} "
                    f"worker {hb['worker']} OVERDUE (hb {hb.get('hb_id')}, phase "
                    f"{ev['phase'] or '?'}) but FRESH movement since last check "
                    f"(file_fresh={ev['file_fresh']} proc_correlated={ev['proc_correlated']} "
                    f"gpu_used_mib={ev['gpu_used_mib']}). Extending due; no stall. "
                    f"Non-destructive watchdog.")}
    if n_stalls >= RETIRE_THRESHOLD:
        # Dead-handle cry-wolf cap: claude was wake-notified RETIRE_THRESHOLD times for
        # this EXACT worker with no movement and chose no recycle/re-drive, so further
        # alerts add no signal. Emit ONE non-wake RETIRED, then CLEAN until a NEWER
        # heartbeat post supersedes this one (find_active_heartbeats takes latest-per-worker).
        if n_retired == 0:
            return {"action": "retire", "kind": "status_update", "wake": False,
                    "body": (
                        f"WATCHDOG_RETIRED — worker {hb['worker']} heartbeat "
                        f"{hb.get('hb_id')} (phase {ev['phase'] or '?'}) retired after "
                        f"{n_stalls} no-movement stalls. Claude was wake-notified {n_stalls}x "
                        f"(file_fresh=false proc_correlated=false) and took no recycle/re-drive, "
                        f"so further alerts add no signal. CEASING alerts for this heartbeat; "
                        f"it re-monitors automatically on a NEWER heartbeat post from this "
                        f"worker. Non-destructive; no state touched.")}
        return {"action": "clean", "kind": "status_update", "wake": False, "body": ""}
    escalate = (n_stalls >= 1) or (n_extends >= 1)
    rec_line = (("RECOMMEND RECYCLE (prior watchdog action then NO fresh movement — "
                 f"extends={n_extends} stalls={n_stalls}).") if escalate else
                "RECOMMEND re-drive (1st miss, no fresh movement) — verify wedge then recycle.")
    return {"action": "stall", "kind": "status_update", "wake": True,
            "body": (
                f"WATCHDOG_STALL — heartbeat_overdue. task {hb.get('task_id') or '?'} "
                f"worker {hb['worker']}; last hb {hb.get('hb_id')} phase "
                f"{ev['phase'] or '?'}; NO FRESH movement since last check "
                f"(file_fresh=false proc_correlated=false gpu_used_mib={ev['gpu_used_mib']}). "
                f"prior_extends={n_extends} prior_stalls={n_stalls}. "
                f"expected_artifact={hb.get('artifact') or '?'} "
                f"(path={hb.get('artifact_path') or 'n/a'}, run_dir={hb.get('run_dir') or 'n/a'}). "
                f"metadata_present={hb['metadata_present']}. {rec_line} "
                f"Non-destructive: claude decides recycle/re-drive/extend from this proof. "
                f"@claude liveness check + decision.")}


def find_unanswered_gates(records, now):
    """Claude-authored deadline-bearing requests to WORKER handles whose deadline
    (+grace) lapsed with no worker response. A response = any later record from
    that worker that threads to the gate (reply_to == gate id) OR cites the gate
    id in its body (covers operators that report without threading).

    Staleness guards: only the LATEST unanswered gate per worker is acted on (a
    newer claude gate to the same worker supersedes older pending ones), and
    gates older than GATE_MAX_AGE_SECONDS are never resurrected (recycled
    handles / closed arcs make ancient gates moot)."""
    candidates = []
    for i, rec in enumerate(records):
        if rec.get("from") != "claude":
            continue
        target = rec.get("requires_response_from", "")
        if not _is_worker(target):
            continue
        t = _record_ts(rec)
        if t is None:
            continue
        try:
            deadline_secs = int(rec.get("response_deadline_secs") or GATE_DEFAULT_DEADLINE)
        except Exception:
            deadline_secs = GATE_DEFAULT_DEADLINE
        gate_id = rec.get("id", "")
        if not gate_id:
            continue
        due = t + deadline_secs + GATE_GRACE_SECONDS
        if now <= due:
            continue
        if now - t > GATE_MAX_AGE_SECONDS:
            continue  # ancient gate — moot, never resurrect
        answered = False
        for later in records[i + 1:]:
            if later.get("from") != target:
                continue
            lt = _record_ts(later)
            if lt is None or lt <= t:
                continue
            if later.get("reply_to") == gate_id or gate_id in _body(later):
                answered = True
                break
        if answered:
            continue
        task_m = TASKID_RE.search(_body(rec))
        candidates.append({"gate_id": gate_id, "worker": target, "gate_ts": t,
                           "deadline_secs": deadline_secs, "due": due,
                           "task_id": task_m.group(1) if task_m else ""})
    # Supersede rule: ANY newer claude deadline-bearing engagement with the same
    # worker (answered or not) supersedes an older pending gate — a newer
    # engagement means claude has moved the worker's contract forward (often
    # after a recycle), and re-driving the old gate would conflict with it.
    latest_engagement = {}
    for rec in records:
        if rec.get("from") != "claude":
            continue
        target = rec.get("requires_response_from", "")
        if not _is_worker(target):
            continue
        t = _record_ts(rec)
        if t is None:
            continue
        if target not in latest_engagement or t > latest_engagement[target]:
            latest_engagement[target] = t
    out = []
    for g in candidates:
        if g["gate_ts"] < latest_engagement.get(g["worker"], g["gate_ts"]):
            continue  # superseded by a newer engagement with this worker
        out.append(g)
    return out


def gate_watchdog_history(records, gate_id):
    """(n_rewakes, n_escalates) — watchdog posts citing this gate id."""
    n_rewake, n_escalate = 0, 0
    for rec in records:
        if rec.get("from") != WATCHDOG_FROM:
            continue
        body = _body(rec)
        if gate_id not in body:
            continue
        # ESCALATE first: escalate bodies contain "after N automatic
        # GATE_REWAKEs", so a REWAKE-first substring match would count
        # escalations as rewakes and re-escalate forever.
        if "GATE_ESCALATE" in body:
            n_escalate += 1
        elif "GATE_REWAKE" in body:
            n_rewake += 1
    return n_rewake, n_escalate


def decide_gate(gate, n_rewakes, n_escalates):
    worker, gate_id = gate["worker"], gate["gate_id"]
    if n_escalates >= 1:
        return {"action": "clean", "kind": "status_update", "wake": False, "body": ""}
    if n_rewakes < GATE_REWAKE_MAX:
        return {"action": "gate_rewake", "kind": "task_dispatch", "wake": True,
                "to": [worker], "requires_from": worker,
                "deadline": GATE_REWAKE_DEADLINE, "worker": worker,
                "body": (
                    f"GATE_REWAKE — worker {worker}: AUTOMATIC re-wake on unanswered "
                    f"deadline-bearing gate {gate_id} (deadline "
                    f"{gate['deadline_secs']}s lapsed; no threaded reply and no body "
                    f"citation of the gate id found; rewake {n_rewakes + 1} of "
                    f"{GATE_REWAKE_MAX}). The gate's authority and instructions are "
                    f"UNCHANGED — re-read gate {gate_id} and act NOW: post the start "
                    f"signal (or classified blocker) FIRST, then execute, then post the "
                    f"terminal validation_receipt threaded to the gate. If the work is "
                    f"already done, post the terminal receipt citing the gate id "
                    f"immediately. Silence past this re-wake escalates to claude. "
                    f"Non-destructive watchdog; no state touched. "
                    f"REPORT_TO: [claude, codex_co_lead] CROSS_THREAD_REQUIRED: yes")}
    return {"action": "gate_escalate", "kind": "status_update", "wake": True,
            "to": ["claude", "codex_co_lead"], "requires_from": "claude",
            "deadline": STALL_RESPONSE_DEADLINE, "worker": worker,
            "body": (
                f"GATE_ESCALATE — worker {worker} unresponsive to gate {gate_id} "
                f"after {n_rewakes} automatic GATE_REWAKEs (deadline "
                f"{gate['deadline_secs']}s + {GATE_REWAKE_MAX} re-wakes, no threaded "
                f"reply, no body citation). RECOMMEND claude verify run-dir/disk state "
                f"(artifacts may exist despite silence — the 2026-06-10 pattern), then "
                f"recycle-and-redispatch or close on disk verification. "
                f"@claude decision required. Non-destructive watchdog.")}


def _channel_from_log_path(path):
    try:
        p = os.path.abspath(os.path.expanduser(path or ""))
    except Exception:
        return ""
    parts = p.split(os.sep)
    if len(parts) >= 3 and parts[-1] == "messages.jsonl" and parts[-3] == "channels":
        return parts[-2]
    return ""


def _resolve_emit_channel(log_path, override):
    return override or _channel_from_log_path(log_path) or os.environ.get("AI_ROOM_CHANNEL", "")


def _trim_for_log(value, limit=500):
    text = (value or "").strip().replace("\n", "\\n")
    if len(text) > limit:
        text = text[:limit - 3] + "..."
    return text or "-"


def _posted_msg_id(stdout):
    m = POSTED_ID_RE.search(stdout or "")
    return m.group(1) if m else ""


def emit(decision, dry_run, channel=""):
    if decision.get("action") == "clean":
        print("CLEAN")
        return
    if dry_run:
        print(f"ACTION={decision['action']} WAKE={decision['wake']} WORKER={decision.get('worker','?')}")
        print(f"POST_BODY={decision['body']}")
        return
    ai_room = shutil.which("ai-room") or os.path.expanduser("~/.local/bin/ai-room")
    cmd = [ai_room]
    if channel:
        cmd += ["--channel", channel]
    cmd += ["post", WATCHDOG_FROM]
    for target in decision.get("to") or ["claude", "codex_co_lead"]:
        cmd += ["--to", target]
    cmd += ["--kind", decision["kind"]]
    if decision["wake"]:
        cmd += ["--requires-response-from", decision.get("requires_from", "claude"),
                "--response-deadline-secs",
                str(decision.get("deadline", STALL_RESPONSE_DEADLINE))]
    cmd += [decision["body"]]
    env = os.environ.copy()
    if channel:
        env["AI_ROOM_CHANNEL"] = channel
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=20, env=env)
        channel_label = channel or "default"
        if proc.returncode == 0:
            msg_id = _posted_msg_id(proc.stdout)
            if msg_id:
                print(f"POSTED action={decision['action']} wake={decision['wake']} "
                      f"channel={channel_label} msg_id={msg_id}")
            else:
                print(f"POSTED action={decision['action']} wake={decision['wake']} "
                      f"channel={channel_label} stdout={_trim_for_log(proc.stdout)}")
            return
        print(f"POST_FAILED action={decision['action']} wake={decision['wake']} "
              f"channel={channel_label} rc={proc.returncode} "
              f"stdout={_trim_for_log(proc.stdout)} stderr={_trim_for_log(proc.stderr)}")
    except Exception as e:
        print(f"POST_FAILED action={decision['action']} wake={decision['wake']} "
              f"channel={channel or 'default'} exception={type(e).__name__}: {_trim_for_log(str(e))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--channel-log", default=None)
    ap.add_argument("--channel", default=None)
    ap.add_argument("--now", default=None)
    args = ap.parse_args()
    try:
        log_path = (args.channel_log or os.environ.get("AI_ROOM_CHANNEL_LOG")
                    or DEFAULT_CHANNEL_LOG)
        emit_channel = _resolve_emit_channel(log_path, args.channel)
        now = _parse_ts(args.now) if args.now else None
        if now is None:
            now = datetime.now(timezone.utc).timestamp()
        records = _read_tail(log_path, TAIL_LINES)
        if not records:
            print("CLEAN")
            return 0
        acted = False
        for hb in find_active_heartbeats(records):
            if hb.get("due") is None or now <= hb["due"] + GRACE_SECONDS:
                continue  # not overdue
            last_check, n_ext, n_stall, n_retired = watchdog_history(records, hb)
            ev = prove_liveness(hb, last_check)
            decision = decide(hb, ev, n_ext, n_stall, n_retired)
            decision["worker"] = hb["worker"]
            emit(decision, args.dry_run, emit_channel)
            acted = True
        for gate in find_unanswered_gates(records, now):
            n_rewake, n_escalate = gate_watchdog_history(records, gate["gate_id"])
            decision = decide_gate(gate, n_rewake, n_escalate)
            if decision.get("action") == "clean":
                continue
            decision.setdefault("worker", gate["worker"])
            emit(decision, args.dry_run, emit_channel)
            acted = True
        if not acted:
            print("CLEAN")
        return 0
    except Exception as e:
        print(f"CLEAN (fail-quiet: {e})")
        return 0


if __name__ == "__main__":
    sys.exit(main())
