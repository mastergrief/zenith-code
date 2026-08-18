#!/usr/bin/env python3
"""Tests for trigger_emitter.py.

Run: python3 -m pytest .claude/hooks/test_trigger_emitter.py -q
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))
from trigger_emitter import (  # noqa: E402
    count_lineage,
    is_bounce,
    prefix_sha256,
)

HOOK = Path(__file__).parent / "trigger_emitter.py"

# Bodies taken from the cited records (plan fire/silent lists).
FIRE_BODIES = {
    "1786655853799-35141ddb": (
        "BLOCK on the claimed verifier closure evidence: `gate1_verify_v3.py` "
        "is not resolvable at the stated lineage directory."
    ),
    "1786656274336-48eff8d4": (
        "BLOCK remains, narrowed to recursive closure. Persistence and "
        "direct-member drift are cured: I independently reproduced verifier sha."
    ),
    "1786656568390-0746f76e": (
        "BLOCK remains on two explicit non-observations in v4's own closure "
        "output, which I independently ran from the persisted bytes."
    ),
    "1786657514361-32740313": (
        "GATE-1: **BLOCK** on remint `06640f5043638c9bd9a8dcb6af14abb713c76c"
        "2919cd569abfeafef641bc79a3` (147120 B). Two blockers."
    ),
    "1786657683301-cc92531b": (
        "GATE-1 on `c142023c730ca6ecfea1c22ef5ae0a78544a968386b143dc83477da1"
        "af13ab21` (147474 B): **BLOCK**, and this is the consolidated dispatch."
    ),
    "1786658902837-910b8111": (
        "Freeze-generator `47ac17dc`: **write-set ordering PASSES**, one "
        "blocker before I mint, plus an argv gap I will not improvise around.\n\n"
        "## BLOCKER — the dead record picks its ADVISOR_ROUTE by position"
    ),
    "1786658990406-1a5d852b": (
        "ADDENDUM to `1786658902837-910b8111`, sent now so it can ride the "
        "same edit. Cure, same edit, one assertion: the freeze's recomputed "
        "enumeration must equal the packet's — refuse on any difference."
    ),
    "1786659398339-3f6e1e73": (
        "**BLOCK — the official argv pair cannot execute.** I proved it "
        "before minting rather than discovering it against the immutable freeze dir."
    ),
}

SILENT_BODIES = {
    "1786658381536-cd185a79": (
        "GATE-1 on `f3cbd4d86ab456d3c16d1601984f7c66eaac932d993bcf69e9c45155"
        "d3cc1cef` (148580 B): **PASS on the packet.** Three additions go "
        "into the FREEZE-RECORD generator only."
    ),
    "1786658342180-0d23e234": (
        "GATE-1 PASS on packet f3cbd4d8 — both blockers cured and independently "
        "verified; prefix binding re-hashed and matches."
    ),
    "1786656248492-b8ba4fe0": (
        "BLOCK 1786655853799-35141ddb — CURED. Your finding was correct as "
        "measured, and the correction below does not soften it."
    ),
    "1786656265494-f97e2e32": (
        "Non-blocking reference for your in-flight import-closure build: "
        "NON-BLOCKING reference. Keep going."
    ),
    "1786655739026-b8f1bbac": (
        "HOLD LIFTED — ADVISOR AMENDED INVARIANT 2. Build the closure, then "
        "the calibrations, then remint."
    ),
    "1786655696066-aaf7c005": (
        "HOLD — do not remint, do not freeze, do not build the closure cure yet."
    ),
    "1786658440968-b3d705f9": (
        "WATCHDOG_STALL `1786658404586-456c5b91` RECOMMEND RECYCLE — "
        "**DECISION: DO NOT RECYCLE, DO NOT RE-DRIVE, OBSERVE.**"
    ),
    "1786658472035-7d27b0a3": (
        "+1 DO NOT RECYCLE. Independently confirmed: `1786655511777-fd0a14aa` "
        "is not a task; codex owns live task `1786645681209-ebbf6291`."
    ),
}


def _rec(
    *,
    rid: str,
    body: str,
    frm: str = "claude",
    kind: str = "review_request",
    reply_to: str | None = None,
    offset: int = 0,
    line: int = 1,
) -> dict:
    return {
        "id": rid,
        "from": frm,
        "to": "codex",
        "kind": kind,
        "body": body,
        "reply_to": reply_to,
        "_offset": offset,
        "_length": 1,
        "_line": line,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> tuple[int, str]:
    raws = []
    for row in rows:
        raws.append((json.dumps(row, ensure_ascii=False) + "\n").encode())
    data = b"".join(raws)
    path.write_bytes(data)
    h = hashlib.sha256(data).hexdigest()
    return len(data), h


def test_is_bounce_fires_on_cited_bodies():
    for rid, body in FIRE_BODIES.items():
        frm = "codex_co_lead" if rid.startswith("178665585") or rid.startswith("178665627") or rid.startswith("178665656") else "claude"
        rec = _rec(rid=rid, body=body, frm=frm)
        assert is_bounce(rec), f"expected fire {rid}"


def test_is_bounce_silent_on_cited_bodies():
    for rid, body in SILENT_BODIES.items():
        frm = "codex_co_lead" if rid == "1786658472035-7d27b0a3" else "claude"
        rec = _rec(rid=rid, body=body, frm=frm)
        assert not is_bounce(rec), f"expected silent {rid}"


def test_ack_is_not_bounce():
    rec = _rec(
        rid="1786658404749-e62d378d",
        body="ack 1786658381536-cd185a79",
        frm="codex",
        kind="ack",
    )
    assert not is_bounce(rec)


def test_plus1_implement_dispatch_is_not_bounce():
    rec = _rec(
        rid="1786645759795-f8b045de",
        body=(
            "TASK 1786645681209-ebbf6291 — DECAY SCREEN v6. Fast-path dispatch: "
            "this IS the plan, and it carries `+1 implement`.\n"
            "Gate-1 authorship shrinks to the verdict token — I author PASS or BLOCK"
        ),
        kind="task_dispatch",
    )
    assert not is_bounce(rec)


def test_escalation_to_advisor_is_not_bounce():
    rec = _rec(
        rid="1786690719942-eb712516",
        body=(
            "advisor — MANDATORY defect-class escalation. Blocks the next remint. "
            "workflow.md:61 fires defect-expiry on second substantiated "
            "**gate-2** BLOCK on the lineage."
        ),
        kind="design_proposal",
    )
    rec["to"] = "advisor"
    assert not is_bounce(rec)


def test_addendum_status_receipt_is_not_bounce():
    rec = _rec(
        rid="1786657705535-d9dd6c62",
        body=(
            "Your addendum is dispatched as a consolidated verdict at "
            "`1786657683301-cc92531b`. The addendum lands in the next version, "
            "not as a partial edit. Producers refuse a second write."
        ),
        kind="validation_receipt",
    )
    assert not is_bounce(rec)


def test_worker_ready_is_not_bounce():
    rec = _rec(
        rid="x",
        body="READY FOR REVIEW — freeze-record generator only.",
        frm="codex",
        kind="status_update",
    )
    assert not is_bounce(rec)


def test_stream_firing_is_every_bounce_at_or_beyond_two():
    task = "task-1"
    recs = [
        _rec(rid="t", body=f"lineage {task}", frm="claude", kind="task", line=1, offset=0),
        _rec(
            rid="b1",
            body=FIRE_BODIES["1786655853799-35141ddb"],
            frm="codex_co_lead",
            reply_to="t",
            line=2,
            offset=10,
        ),
        _rec(
            rid="b2",
            body=FIRE_BODIES["1786656274336-48eff8d4"],
            frm="codex_co_lead",
            reply_to="t",
            line=3,
            offset=20,
        ),
        _rec(
            rid="b3",
            body=FIRE_BODIES["1786659398339-3f6e1e73"],
            frm="claude",
            reply_to="t",
            line=4,
            offset=30,
        ),
    ]
    out = count_lineage(recs, task, None)
    assert out["bounce_ids"] == ["b1", "b2", "b3"]
    assert out["blocking_verdict_record_count"] == 3
    assert [f["id"] for f in out["firings"]] == ["b2", "b3"]


def test_pass_then_blocker_same_body_fires():
    rec = _rec(rid="mixed", body=FIRE_BODIES["1786658902837-910b8111"])
    assert is_bounce(rec)


def test_cli_prefix_sha_mismatch_refuses(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    hw, sha = _write_jsonl(log, [{"id": "a", "from": "claude", "kind": "msg", "body": "x"}])
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(log),
            "--high-water-bytes",
            str(hw),
            "--expect-prefix-sha",
            "0" * 64,
            "--task-id",
            "a",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "prefix_sha256 mismatch" in proc.stderr


def test_cli_missing_mark_refuses(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    log.write_text("{}\n")
    proc = subprocess.run(
        [sys.executable, str(HOOK), "--room-log", str(log), "--task-id", "a"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "high-water-bytes" in proc.stderr


def test_addendum_filename_refuse_is_not_bounce():
    rec = _rec(
        rid="1786650926049-ec53091c",
        body=(
            "ADDENDUM 6 — THE REFERENCE DENOMINATOR, MEASURED FROM THE v8 "
            "PACKET'S OWN BYTES. Paths: probe.pid, probe.pid.refuse, "
            "probe.exit_code.txt."
        ),
        kind="msg",
    )
    assert not is_bounce(rec)


def test_addendum_quoted_refuse_rule_is_not_bounce():
    rec = _rec(
        rid="1786650996980-23033955",
        body=(
            "ADDENDUM 8 — CHAIN PREDICATE SETTLED. This closes the last open "
            "design question; nothing further is pending on my side. "
            "Build and deliver.\nRefuse cycles, forward references, "
            "missing/duplicate predecessors."
        ),
        kind="msg",
    )
    assert not is_bounce(rec)


def test_cli_emit_mark_requires_declared_cut(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    _write_jsonl(log, [{"id": "a", "from": "claude", "kind": "msg", "body": "x"}])
    proc = subprocess.run(
        [sys.executable, str(HOOK), "--room-log", str(log), "--emit-mark"],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "will not discover a mark from live log size" in proc.stderr


def test_cli_emit_mark_honours_declared_cut(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    hw, sha = _write_jsonl(
        log, [{"id": "a", "from": "claude", "kind": "msg", "body": "BLOCK on x"}]
    )
    # grow the file after the declared cut — live size must not win
    with log.open("ab") as fh:
        fh.write(b'{"id":"later"}\n')
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(log),
            "--emit-mark",
            "--high-water-bytes",
            str(hw),
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["high_water_bytes"] == hw
    assert payload["prefix_sha256"] == sha
    assert payload["high_water_bytes"] != log.stat().st_size
    assert "bounce_ids" not in payload


def test_deleted_known_good_flag_refuses(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    hw, sha = _write_jsonl(log, [{"id": "a", "from": "claude", "kind": "msg", "body": "x"}])
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(log),
            "--high-water-bytes",
            str(hw),
            "--expect-prefix-sha",
            sha,
            "--task-id",
            "a",
            "--known-good-start-offset",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "known-good" in (proc.stderr + proc.stdout)


def test_unreadable_row_refuses_no_count(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    good = (
        json.dumps(
            {
                "id": "1786700000010-dddddddd",
                "from": "claude",
                "to": "codex",
                "kind": "review_request",
                "body": "GATE-1: **BLOCK** on v1. Task 1786700000000-tttttttt.",
            }
        )
        + "\n"
    )
    bad = '{"id":"1786700000011-eeeeeeee","kind":"review_request","body":"GATE-1: **BLOCK** truncated\n'
    log.write_bytes(good.encode() + bad.encode())
    hw = log.stat().st_size
    sha = hashlib.sha256(log.read_bytes()).hexdigest()
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(log),
            "--high-water-bytes",
            str(hw),
            "--expect-prefix-sha",
            sha,
            "--task-id",
            "1786700000000-tttttttt",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    assert "unreadable row" in proc.stderr
    assert "blocking_verdict_record_count" not in proc.stdout


def test_denominator_zero_over_n_and_zero_over_zero(tmp_path: Path):
    nonempty = tmp_path / "n.jsonl"
    hw, sha = _write_jsonl(
        nonempty,
        [{"id": "a", "from": "codex", "kind": "ack", "body": "ack x", "to": "claude"}],
    )
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(nonempty),
            "--high-water-bytes",
            str(hw),
            "--expect-prefix-sha",
            sha,
            "--task-id",
            "no-such-task",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["blocking_verdict_record_count"] == 0
    assert payload["prefix_rows_enumerated"] == 1

    empty = tmp_path / "empty.jsonl"
    empty.write_bytes(b"")
    empty_sha = hashlib.sha256(b"").hexdigest()
    proc2 = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(empty),
            "--high-water-bytes",
            "0",
            "--expect-prefix-sha",
            empty_sha,
            "--task-id",
            "no-such-task",
        ],
        capture_output=True,
        text=True,
    )
    assert proc2.returncode == 0, proc2.stderr
    payload2 = json.loads(proc2.stdout)
    assert payload2["blocking_verdict_record_count"] == 0
    assert payload2["prefix_rows_enumerated"] == 0


def test_cross_high_water_refuses(tmp_path: Path):
    log = tmp_path / "log.jsonl"
    line = (json.dumps({"id": "a", "from": "claude", "kind": "msg", "body": "x"}) + "\n").encode()
    log.write_bytes(line)
    # cut inside the record
    cut = max(1, len(line) // 2)
    # force newline at cut so bind_prefix's last-byte check is not the one that fires
    # (we want the cross-record refuse). If cut-1 is not newline, bind_prefix refuses first.
    sha = prefix_sha256(log, cut) if log.read_bytes()[cut - 1 : cut] == b"\n" else "x"
    if log.read_bytes()[cut - 1 : cut] != b"\n":
        proc = subprocess.run(
            [
                sys.executable,
                str(HOOK),
                "--room-log",
                str(log),
                "--high-water-bytes",
                str(cut),
                "--expect-prefix-sha",
                "0" * 64,
                "--task-id",
                "a",
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        assert "does not end on a complete record" in proc.stderr
        return
    proc = subprocess.run(
        [
            sys.executable,
            str(HOOK),
            "--room-log",
            str(log),
            "--high-water-bytes",
            str(cut),
            "--expect-prefix-sha",
            sha,
            "--task-id",
            "a",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0


def test_wake_paired_single_bounce_reads_one_and_is_silent():
    """Calibration 1: transport + verdict on one version → count 1, no firing."""
    task = "1786700000000-tttttttt"
    recs = [
        _rec(
            rid="1786700000001-aaaaaaaa",
            body=(
                "GATE-1 BLOCK on v1 draft deadbeef — one blocker. "
                f"Detail follows. task {task}"
            ),
            kind="task_update",
            line=1,
            offset=0,
        ),
        _rec(
            rid="1786700000002-bbbbbbbb",
            body=(
                "GATE-1: **BLOCK** on v1 draft deadbeef. One blocker. "
                f"Task {task}."
            ),
            kind="review_request",
            line=2,
            offset=100,
        ),
    ]
    out = count_lineage(recs, task, None)
    assert out["bounce_ids"] == ["1786700000002-bbbbbbbb"]
    assert out["blocking_verdict_record_count"] == 1
    assert out["firings"] == []


def test_two_blocking_verdicts_fire():
    """Calibration 2: two verdict-bearing stops → count 2, fires."""
    task = "task-1"
    recs = [
        _rec(rid="t", body=f"lineage {task}", frm="claude", kind="task", line=1, offset=0),
        _rec(
            rid="v1",
            body=FIRE_BODIES["1786658902837-910b8111"],
            kind="review_request",
            reply_to="t",
            line=2,
            offset=10,
        ),
        _rec(
            rid="v2",
            body=FIRE_BODIES["1786659398339-3f6e1e73"],
            kind="review_request",
            reply_to="t",
            line=3,
            offset=20,
        ),
    ]
    out = count_lineage(recs, task, None)
    assert out["blocking_verdict_record_count"] == 2
    assert [f["id"] for f in out["firings"]] == ["v2"]


def test_tonight_three_gate1_ids_all_match():
    """Calibration 3: the three posted gate-1 bounce bodies must all fire."""
    posted = {
        "1786658902837-910b8111": ("review_request", FIRE_BODIES["1786658902837-910b8111"]),
        "1786658990406-1a5d852b": ("msg", FIRE_BODIES["1786658990406-1a5d852b"]),
        "1786659398339-3f6e1e73": ("review_request", FIRE_BODIES["1786659398339-3f6e1e73"]),
    }
    for rid, (kind, body) in posted.items():
        rec = _rec(rid=rid, body=body, kind=kind)
        assert is_bounce(rec), f"calibration-3 miss {rid}"
        assert kind != "task_update"
