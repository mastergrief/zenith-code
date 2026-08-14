#!/usr/bin/env python3
"""Raw per-lineage substantiated-bounce counter.

Reads a DECLARED log prefix only (high_water_bytes + prefix_sha256).
Does not classify defect classes. Does not latch. Does not register as a hook.

ADVISOR_ROUTE: 1786690977198-524132ef amended by 1786691638636-e28647d5
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterator

GATE_SIDE = frozenset({"claude", "codex_co_lead"})

# Silent overrides win. Derived from the cited records, not from "BLOCK".
_NON_BLOCKING = re.compile(r"(?i)\bNON-BLOCKING\b")
_HOLD_LIFTED = re.compile(r"(?i)\bHOLD LIFTED\b")
_HOLD_OPEN = re.compile(r"(?i)\A\s*HOLD\b")
_DO_NOT_RECYCLE = re.compile(r"(?i)\bDO NOT RECYCLE\b")
_WATCHDOG = re.compile(r"(?i)\bWATCHDOG_STALL\b")
# Verdict that a cited prior BLOCK is already closed — not "are cured" in passing.
_CURED = re.compile(
    r"(?i)("
    r"BLOCK\s+\S+.{0,40}(?:—|-)\s*CURED"
    r"|instrument BLOCK\s+\S+.{0,40}\bcured\b"
    r"|BLOCK\s+\S+.{0,40}\bcured:"
    r"|BLOCK\s+\S+.{0,40}\bcured\b"
    r")"
)
_PASS_VERDICT = re.compile(
    r"(?i)(GATE-1 PASS\b|\*\*PASS on the packet\.\*\*|\bPASS on the packet\b)"
)

# Present stop on the current artifact (packet / freeze / remint).
# GATE-1 forms observed on the cited fire records; not "I author PASS or BLOCK".
_ARTIFACT_STOP = re.compile(
    r"(?i)("
    r"GATE-1(?:\s+BLOCK\b|:\s*(?:\*\*)?BLOCK\b| on .{0,160}:\s*\*\*BLOCK)"
    r"|\*\*BLOCK\b"
    r"|## BLOCKER\b"
    r"|\bBLOCK remains\b"
    r"|\bone blocker before I mint\b"
    r")"
)
# "BLOCK on <artifact>" only as a leading verdict, not a mid-body rule quote.
_BLOCK_ON_LEAD = re.compile(r"(?i)\A.{0,120}\bBLOCK on\b")
# Gate addendum attaching a stop to a prior record. Numbered "ADDENDUM N —"
# dispatch notes do not match. "refuse on" is the author's required action,
# not a quoted "Refuse …" rule and not a "*.refuse" filename.
_ADDENDUM_REFUSE = re.compile(
    r"(?i)\A\s*ADDENDUM to [`']?\d{13}-[0-9a-f]{8}[`']?[\s\S]{0,1200}\brefuse on\b"
)


def prefix_sha256(path: Path, n: int) -> str:
    h = hashlib.sha256()
    remaining = n
    with path.open("rb") as fh:
        while remaining:
            chunk = fh.read(min(1024 * 1024, remaining))
            if not chunk:
                break
            h.update(chunk)
            remaining -= len(chunk)
    if remaining:
        raise SystemExit(f"refuse: prefix truncated path={path} missing={remaining}")
    return h.hexdigest()


def bind_prefix(path: Path, high_water: int, expect_sha: str) -> None:
    size = path.stat().st_size
    if size < high_water:
        raise SystemExit(
            f"refuse: log size {size} < high_water_bytes {high_water}"
        )
    if high_water < 0:
        raise SystemExit("refuse: high_water_bytes must be >= 0")
    if high_water > 0:
        with path.open("rb") as fh:
            fh.seek(high_water - 1)
            last = fh.read(1)
        if last != b"\n":
            raise SystemExit(
                "refuse: high_water_bytes does not end on a complete record"
            )
    got = prefix_sha256(path, high_water)
    if got != expect_sha:
        raise SystemExit(
            f"refuse: prefix_sha256 mismatch got={got} expect={expect_sha}"
        )


def iter_prefix_records(path: Path, high_water: int) -> Iterator[dict[str, Any]]:
    with path.open("rb") as fh:
        offset = 0
        line = 0
        while offset < high_water:
            raw = fh.readline()
            if not raw:
                raise SystemExit(
                    f"refuse: prefix truncated at offset={offset} high_water={high_water}"
                )
            if offset + len(raw) > high_water:
                raise SystemExit(
                    f"refuse: record crosses high_water offset={offset} length={len(raw)}"
                )
            line += 1
            rec: dict[str, Any]
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"refuse: unreadable row line={line} offset={offset} "
                    f"json={exc.msg}"
                ) from exc
            if not isinstance(parsed, dict):
                raise SystemExit(
                    f"refuse: unreadable row line={line} offset={offset} "
                    "not a json object"
                )
            rec = parsed
            rec["_offset"] = offset
            rec["_length"] = len(raw)
            rec["_line"] = line
            yield rec
            offset += len(raw)
        extra = fh.read(1)
        if extra and offset != high_water:
            raise SystemExit("refuse: prefix walk did not land on high_water")


def _cites_lineage(text: str, task_id: str | None, token: str | None) -> bool:
    if task_id and task_id in text:
        return True
    if token and token in text:
        return True
    return False


def in_lineage(
    rec: dict[str, Any],
    by_id: dict[str, dict[str, Any]],
    task_id: str | None,
    token: str | None,
) -> bool:
    rid = rec.get("id")
    if task_id and rid == task_id:
        return True
    seen: set[str] = set()
    cur: dict[str, Any] | None = rec
    steps = 0
    while cur is not None and steps < 10000:
        steps += 1
        cid = cur.get("id")
        if isinstance(cid, str):
            if cid in seen:
                break
            seen.add(cid)
            if task_id and cid == task_id:
                return True
        body = cur.get("body") or ""
        if isinstance(body, str) and _cites_lineage(body, task_id, token):
            return True
        parent = cur.get("reply_to")
        if not isinstance(parent, str) or not parent:
            break
        cur = by_id.get(parent)
    return False


def is_transport(rec: dict[str, Any]) -> bool:
    """Wake-pairing / board-state carriers. Not a verdict.

    Keyed on kind == task_update. Bounded absence (decay 21-record corpus):
    every task_update there had a distinct verdict carrier. Breaks when a
    gate-1 block is posted as a task_update with no separate verdict record
    — that world reads 0.
    """
    return rec.get("kind") == "task_update"


def is_bounce(rec: dict[str, Any]) -> bool:
    """True iff the record imposes a stop on the present artifact."""
    if rec.get("kind") == "ack":
        return False
    if rec.get("from") not in GATE_SIDE:
        return False
    dest = rec.get("to")
    if dest == "advisor" or (isinstance(dest, list) and dest == ["advisor"]):
        return False
    body = rec.get("body") or ""
    if not isinstance(body, str) or not body:
        return False
    if _NON_BLOCKING.search(body):
        return False
    if _HOLD_LIFTED.search(body):
        return False
    if _HOLD_OPEN.search(body):
        return False
    if _DO_NOT_RECYCLE.search(body):
        return False
    if _WATCHDOG.search(body):
        return False
    if _CURED.search(body):
        return False
    if (
        _ARTIFACT_STOP.search(body)
        or _BLOCK_ON_LEAD.search(body)
        or _ADDENDUM_REFUSE.search(body)
    ):
        return True
    if _PASS_VERDICT.search(body):
        return False
    return False


def count_lineage(
    records: list[dict[str, Any]],
    task_id: str | None,
    token: str | None,
) -> dict[str, Any]:
    by_id = {
        rec["id"]: rec
        for rec in records
        if isinstance(rec.get("id"), str)
    }
    bounce_ids: list[str] = []
    firings: list[dict[str, Any]] = []
    for rec in records:
        if not in_lineage(rec, by_id, task_id, token):
            continue
        if not is_bounce(rec):
            continue
        if is_transport(rec):
            continue
        rid = rec.get("id")
        if not isinstance(rid, str) or not rid:
            raise SystemExit("refuse: bounce record has no citable id")
        bounce_ids.append(rid)
        if len(bounce_ids) >= 2:
            firings.append(
                {
                    "id": rid,
                    "offset": rec["_offset"],
                    "line": rec["_line"],
                }
            )
    return {
        "bounce_ids": bounce_ids,
        "blocking_verdict_record_count": len(bounce_ids),
        "firings": firings,
    }


def emit_mark(path: Path, high_water: int) -> int:
    """Hash a caller-declared cut. Never discover the mark from live size."""
    size = path.stat().st_size
    if size < high_water:
        raise SystemExit(
            f"refuse: log size {size} < high_water_bytes {high_water}"
        )
    if high_water < 0:
        raise SystemExit("refuse: high_water_bytes must be >= 0")
    if high_water > 0:
        with path.open("rb") as fh:
            fh.seek(high_water - 1)
            last = fh.read(1)
        if last != b"\n":
            raise SystemExit(
                "refuse: high_water_bytes does not end on a complete record"
            )
    print(
        json.dumps(
            {
                "path": str(path),
                "high_water_bytes": high_water,
                "prefix_sha256": prefix_sha256(path, high_water),
            }
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--room-log", type=Path, required=True)
    ap.add_argument("--emit-mark", action="store_true")
    ap.add_argument("--high-water-bytes", type=int)
    ap.add_argument("--expect-prefix-sha")
    ap.add_argument("--task-id")
    ap.add_argument("--lineage-token")
    args = ap.parse_args(argv)

    if args.emit_mark:
        if args.high_water_bytes is None:
            raise SystemExit(
                "refuse: --emit-mark requires --high-water-bytes "
                "(will not discover a mark from live log size)"
            )
        return emit_mark(args.room_log, args.high_water_bytes)

    if args.high_water_bytes is None or not args.expect_prefix_sha:
        raise SystemExit(
            "refuse: count mode requires --high-water-bytes and --expect-prefix-sha"
        )
    if not args.task_id and not args.lineage_token:
        raise SystemExit("refuse: count mode requires --task-id and/or --lineage-token")

    bind_prefix(args.room_log, args.high_water_bytes, args.expect_prefix_sha)
    records = list(iter_prefix_records(args.room_log, args.high_water_bytes))
    counted = count_lineage(records, args.task_id, args.lineage_token)
    payload: dict[str, Any] = {
        "task_id": args.task_id,
        "lineage_token": args.lineage_token,
        "prefix": {
            "path": str(args.room_log),
            "high_water_bytes": args.high_water_bytes,
            "prefix_sha256": args.expect_prefix_sha,
        },
        "prefix_rows_enumerated": len(records),
        "bounce_ids": counted["bounce_ids"],
        "blocking_verdict_record_count": counted["blocking_verdict_record_count"],
        "firings": counted["firings"],
    }
    print(json.dumps(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
