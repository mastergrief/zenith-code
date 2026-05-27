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
  2. run the audit bundle on box (--l0c1-audit, --l0c2-audit,
     --language-supports, --anchor-audit, --exhaustive-finite-supports,
     --l0c-exhaustive-audit),
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
    # F.4-audit: L0c2 bounded-2-digit stair-step acquire-target audit (230 rows,
    # per source_rung:operator composite bucket). Distinct subprocess per mode
    # so --l0c2-audit carries only its own flag (the probe CLI mutex never trips).
    ("l0c2", ["--l0c2-audit"], r"L0C2 AGGREGATE"),
    ("l0c2k1", ["--l0c2k1-audit"], r"L0C2K1 AGGREGATE"),
    ("l0c2k2", ["--l0c2k2-audit"], r"L0C2K2 AGGREGATE"),
    # STEP 1 K2 addition acquisition/diagnostic surfaces. Tokens are
    # trailing-space anchored by " AGGREGATE", so neither cross-matches the
    # legacy L0C2K2 aggregate token.
    ("l0c2k2additionfull", ["--l0c2k2-addition-full-audit"],
     r"L0C2K2ADDITIONFULL AGGREGATE"),
    # K2 addition 2x-density split (k=1..4 subset of the 240). Token is
    # trailing-space anchored by " AGGREGATE" so it never cross-matches the
    # 240 ADDITIONFULL token or the heldout50s token.
    ("l0c2k2addition120", ["--l0c2k2-addition-120-audit"],
     r"L0C2K2ADDITION120 AGGREGATE"),
    # SECOND 2x-density atom (k=5..8). Token L0C2K2ADDITION120K5TO8 is
    # trailing-space anchored so it never cross-matches the k=1..4 token
    # L0C2K2ADDITION120 (which is followed by 'K', not a space).
    ("l0c2k2addition120k5to8", ["--l0c2k2-addition-120-k5to8-audit"],
     r"L0C2K2ADDITION120K5TO8 AGGREGATE"),
    # Result-range extension acquisition target. This is the canonical 50s gate;
    # the legacy heldout-50s mode below is alias-only/non-gating after this rung.
    ("l0c2k2addition50s", ["--l0c2k2-addition-50s-audit"],
     r"L0C2K2ADDITION50S AGGREGATE"),
    # Historical receipt alias for the exact 50s rows. Kept non-gating so old
    # scripts resolve, but no longer a held-out transfer signal after 50s trains.
    ("l0c2k2additionheldout50s", ["--l0c2k2-addition-heldout-50s-audit"],
     r"L0C2K2ADDITIONHELDOUT50S AGGREGATE"),
    ("l0c2k2additionheldout60s", ["--l0c2k2-addition-heldout-60s-audit"],
     r"L0C2K2ADDITIONHELDOUT60S AGGREGATE"),
    ("l0c2k3", ["--l0c2k3-audit"], r"L0C2K3 AGGREGATE"),
    # F.4d-edge: L0c2-K1-edge held-generalization micro-slice acquire surface.
    # Two finite sub-surfaces (train 52 / held 13); the aggregate line is the 65
    # combined, the per-surface train/held/fresh/legacy breakdown is in the mode
    # JSON. Pattern has no overlap with L0C2K1 ("...K1EDGE AGGREGATE" vs
    # "...K1 AGGREGATE"), so the two modes never cross-match.
    ("l0c2k1edge", ["--l0c2k1-edge-audit"], r"L0C2K1EDGE AGGREGATE"),
    # F.4d-identity: suffix-copy precursor surface. Pattern is literal-space
    # anchored against K1/K1EDGE aggregate tokens.
    ("l0c2k1identity", ["--l0c2k1-identity-audit"], r"L0C2K1IDENTITY AGGREGATE"),
    # F.4d-identity-full: full-density 90/90 coverage surface for the
    # emission-primitive rung. Distinct subprocess per mode (only its own flag),
    # and the aggregate token L0C2K1IDENTITYFULL is trailing-space anchored, so
    # it never cross-matches L0C2K1IDENTITY / L0C2K1 / L0C2K1EDGE.
    ("l0c2k1identityfull", ["--l0c2k1-identity-full-audit"], r"L0C2K1IDENTITYFULL AGGREGATE"),
    ("language", ["--language-supports"], r"(L0a |L0b |LANGUAGE AGGREGATE)"),
    ("anchor", ["--anchor-audit", "--anchor-set", "math_fragile_v1"], r"ANCHOR AGGREGATE"),
    ("math_a0", ["--exhaustive-finite-supports"],
     # Surface watch-row status every exhaustive run so accepted exceptions
     # (config.watch_rows) appear in producer/consumer logs without a one-off
     # wrapper tweak (codex msg 1779692376889 fix 2).
     r"(\[probe-exhaustive\] AGGREGATE|\[probe-watch\]|WATCH AGGREGATE)"),
    # F.3b: exhaustive-L0c acquire-target audit (codex msg 1779695455088).
    # Distinct subprocess per mode, so --l0c-exhaustive-audit carries only
    # its own flag (the probe's CLI mutex never trips). Watch rows are the
    # SAME accepted exception mapped to the L0c surface by _l0c_watch_transform.
    ("l0c_exhaustive", ["--l0c-exhaustive-audit"],
     r"(\[probe-l0c-exhaustive\] AGGREGATE|\[probe-watch\]|WATCH AGGREGATE)"),
]

_SUMMARY_ANNOTATIONS = {
    "l0c2k2additionheldout50s": (
        "LEGACY_ALIAS_ONLY_NON_GATING: same rows as L0C2K2ADDITION50S; not transfer"
    ),
    "l0c2k2additionheldout60s": (
        "DIAGNOSTIC_NON_GATING: forward-transfer signal; not gate"
    ),
}


def _summary_aggregate(mode_name: str, aggregate_line: str) -> str:
    """Add human-facing labels that should survive log-only receipts."""
    if not aggregate_line:
        return aggregate_line
    note = _SUMMARY_ANNOTATIONS.get(mode_name)
    if note is None:
        return aggregate_line
    if note in aggregate_line:
        return aggregate_line
    return f"{aggregate_line} [{note}]"


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
            l0c2 = next((a for a in entry.get("results", {}).get("l0c2", {}).get("aggregate", [])), "")
            l0c2_bands = [
                _summary_aggregate(
                    name,
                    next((a for a in entry.get("results", {}).get(name, {}).get("aggregate", [])), ""),
                )
                for name in ("l0c2k1", "l0c2k1edge", "l0c2k1identity", "l0c2k1identityfull", "l0c2k2", "l0c2k2additionfull", "l0c2k2addition120", "l0c2k2addition120k5to8", "l0c2k2addition50s", "l0c2k2additionheldout50s", "l0c2k2additionheldout60s", "l0c2k3")
            ]
            l0cx = next((a for a in entry.get("results", {}).get("l0c_exhaustive", {}).get("aggregate", [])
                         if "[probe-l0c-exhaustive]" in a), "")
            print(f"[a1-watcher] consumer: step {step} status={entry['status']} "
                  f"| {l0c1}" + (f" | {l0c2}" if l0c2 else "")
                  + "".join(f" | {a}" for a in l0c2_bands if a)
                  + (f" | {l0cx}" if l0cx else ""), flush=True)

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
