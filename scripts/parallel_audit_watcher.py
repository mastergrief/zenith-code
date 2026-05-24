#!/usr/bin/env python3
"""Force-parallel producer/consumer audit watcher (A1).

Codex hard-invariant (msgs 1779639412812 / 1779639482838 / 1779639631768):
producer/consumer train+audit is a METHOD invariant for F.2b+, not an
optional efficiency. Each saved checkpoint is a live compile/error signal
(stop-early / early-bank / recipe-pivot). No silent serial fallback.

Wrapper-level, NOT trainer/model logic. Runs as a SECOND Monitor alongside
the training Monitor. The producer signal is the trainer's tee'd log line
`save_at_step: saved <path>` (checkpoint fully written). On each:
  1. rsync the ckpt to the box (consumer GPU lane),
  2. run the audit bundle on box (--l0c1-audit, --language-supports,
     --anchor-audit, --exhaustive-finite-supports),
  3. record a per-step manifest entry that PROVES OVERLAP — ckpt path,
     producer ts, rsync start/end, audit start/end, artifact paths,
     status (OVERLAP if audit started before `training complete`, else
     SERIAL_FALLBACK), per-mode aggregate strings.
Final: any expected save-step lacking a consumer receipt → MISSED_PARALLELISM.
Each handled step emits a stdout line so the parent (Monitor) gets live
decision-latency events. Exit code 2 if any step is SERIAL_FALLBACK / MISSED
(unless --waive), else 0.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

_SAVE_RE = re.compile(r"save_at_step:\s+saved\s+(\S+\.pt)")
_DONE_RE = re.compile(r"training complete:")
_STEP_RE = re.compile(r"_step0*(\d+)\.pt$")

# (flag, aggregate-line grep pattern) per audit mode.
_AUDIT_MODES = [
    ("l0c1", ["--l0c1-audit"], r"L0C1 AGGREGATE"),
    ("language", ["--language-supports"], r"(L0a |L0b |LANGUAGE AGGREGATE)"),
    ("anchor", ["--anchor-audit", "--anchor-set", "math_fragile_v1"], r"ANCHOR AGGREGATE"),
    ("math_a0", ["--exhaustive-finite-supports"], r"\[probe-exhaustive\] AGGREGATE"),
]
_COMMON_FLAGS = [
    "--use-cached-ternary-infer", "--use-kv-cache-decode",
    "--use-batched-probe-eval", "--probe-batch-size", "32",
]


def _now() -> float:
    return time.time()


def _iso(ts: float | None) -> str | None:
    return None if ts is None else time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


def _run(cmd: list[str], timeout: int) -> tuple[int, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout + p.stderr)
    except subprocess.TimeoutExpired:
        return 124, "TIMEOUT"


def _audit_step(args, ckpt_local: str, step: int) -> dict:
    ckpt_name = Path(ckpt_local).name
    remote_ckpt = f"{args.remote_ckpt_dir}/{ckpt_name}"
    entry: dict = {"step": step, "ckpt": ckpt_local, "remote_ckpt": remote_ckpt}

    rsync_start = _now()
    rc, out = _run(
        ["rsync", "-z", "--no-perms", "--no-owner", "--no-group",
         ckpt_local, f"{args.box}:{remote_ckpt}"],
        timeout=300,
    )
    rsync_end = _now()
    entry["rsync_start"] = _iso(rsync_start)
    entry["rsync_end"] = _iso(rsync_end)
    entry["rsync_ok"] = rc == 0
    if rc != 0:
        entry["status"] = "RSYNC_FAILED"
        entry["rsync_err"] = out[-300:]
        return entry

    audit_start = _now()
    results: dict = {}
    artifacts: list[str] = []
    for name, flags, grep_pat in _AUDIT_MODES:
        json_out = f"/tmp/a1_{name}_step{step}.json"
        remote_cmd = (
            f"cd {args.remote_repo} && PYTHONPATH=. {args.venv_python} "
            f"scripts/probe_hrm_text_158.py --ckpt-path {remote_ckpt} "
            f"{' '.join(flags)} {' '.join(_COMMON_FLAGS)} "
            f"--audit-output-json {json_out}"
        )
        rc, out = _run(["ssh", args.box, remote_cmd], timeout=600)
        agg = [ln.strip() for ln in out.splitlines() if re.search(grep_pat, ln)]
        results[name] = {"rc": rc, "aggregate": agg, "remote_json": json_out}
        artifacts.append(f"{args.box}:{json_out}")
    audit_end = _now()

    entry["audit_start"] = _iso(audit_start)
    entry["audit_end"] = _iso(audit_end)
    entry["artifacts"] = artifacts
    entry["results"] = results
    entry["_audit_start_raw"] = audit_start
    return entry


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--expected-steps", default="500,750,1000")
    ap.add_argument("--box", default="box")
    ap.add_argument("--remote-repo", default="/home/gabe/claw-code-hrm-158")
    ap.add_argument("--remote-ckpt-dir",
                    default="/home/gabe/claw-code-hrm-158/calm/hrm/checkpoints")
    ap.add_argument("--venv-python", default="/home/gabe/hrm158-venv/bin/python")
    ap.add_argument("--run-id", default="f2b")
    ap.add_argument("--waive", action="store_true",
                    help="Downgrade SERIAL_FALLBACK/MISSED to non-fatal (exit 0); "
                         "records waived=true in manifest.")
    ap.add_argument("--drain-secs", type=float, default=5.0,
                    help="Seconds to keep reading after 'training complete'.")
    args = ap.parse_args()

    expected = [int(s) for s in args.expected_steps.split(",") if s.strip()]
    log = Path(args.train_log)
    print(f"[a1-watcher] run={args.run_id} expecting save-steps {expected}; "
          f"tailing {log}", flush=True)

    # Wait for the train log to appear (training Monitor creates it).
    t_wait = _now()
    while not log.exists():
        if _now() - t_wait > 120:
            print("[a1-watcher] ERROR: train log never appeared (120s)", flush=True)
            return 3
        time.sleep(0.5)

    entries: list[dict] = []
    seen_steps: set[int] = set()
    train_complete_ts: float | None = None
    drain_until: float | None = None

    with log.open() as f:
        while True:
            line = f.readline()
            if not line:
                if drain_until is not None and _now() >= drain_until:
                    break
                time.sleep(0.5)
                continue
            if _DONE_RE.search(line) and train_complete_ts is None:
                train_complete_ts = _now()
                drain_until = train_complete_ts + args.drain_secs
                print(f"[a1-watcher] producer: training complete @ "
                      f"{_iso(train_complete_ts)}", flush=True)
                continue
            m = _SAVE_RE.search(line)
            if not m:
                continue
            ckpt = m.group(1)
            sm = _STEP_RE.search(ckpt)
            step = int(sm.group(1)) if sm else -1
            print(f"[a1-watcher] producer: save-step {step} -> {ckpt}; "
                  f"consuming on {args.box}...", flush=True)
            entry = _audit_step(args, ckpt, step)
            astart = entry.pop("_audit_start_raw", None)
            if entry.get("status") == "RSYNC_FAILED":
                pass
            elif train_complete_ts is not None and astart is not None and astart > train_complete_ts:
                entry["status"] = "SERIAL_FALLBACK"
            else:
                entry["status"] = "OVERLAP"
            entries.append(entry)
            seen_steps.add(step)
            l0c1 = next((a for a in entry.get("results", {}).get("l0c1", {}).get("aggregate", [])), "")
            print(f"[a1-watcher] consumer: step {step} status={entry['status']} "
                  f"| {l0c1}", flush=True)

    # Finalize: flag expected-but-missing steps.
    for st in expected:
        if st not in seen_steps:
            entries.append({"step": st, "status": "MISSED_PARALLELISM"})
            print(f"[a1-watcher] FLAG: save-step {st} had no consumer audit "
                  f"-> MISSED_PARALLELISM", flush=True)

    bad = [e for e in entries if e.get("status") in
           ("SERIAL_FALLBACK", "MISSED_PARALLELISM", "RSYNC_FAILED")]
    manifest = {
        "run_id": args.run_id,
        "train_log": str(log),
        "expected_steps": expected,
        "train_complete_ts": _iso(train_complete_ts),
        "waived": bool(args.waive),
        "n_overlap": sum(1 for e in entries if e.get("status") == "OVERLAP"),
        "n_flagged": len(bad),
        "entries": entries,
    }
    Path(args.manifest).write_text(json.dumps(manifest, indent=2))
    print(f"[a1-watcher] manifest -> {args.manifest} "
          f"(overlap={manifest['n_overlap']} flagged={manifest['n_flagged']})",
          flush=True)
    if bad and not args.waive:
        print(f"[a1-watcher] NON-PARALLEL: {[ (e['step'], e['status']) for e in bad ]}",
              flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
