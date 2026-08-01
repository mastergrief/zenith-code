#!/usr/bin/env python3
"""Pure room-record authorization seam for commit_precondition_colead_gate.

Step-2 v4/v7 semantics: digest-set, demotion-before-uniqueness, F1–F8 / X1–X3/X5,
A3 pre-freeze worker_ts, A4 kind-agnostic PASS path.

No subprocess, filesystem channel-log IO, VCS CLI, or hook stdin.
Entrypoint imports this module; this module never imports the entrypoint.
"""

from __future__ import annotations

import re
from typing import Any

# --- constants / markers ---------------------------------------------------

# Horizontal whitespace only — never join label to hex across newlines.
_HWS = r"[ \t]*"
# Legacy strict-line form kept as form 1 of the digest-set (fixture parity).
DIFF_DIGEST_RE = re.compile(
    r"(?im)^" + _HWS + r"DIFF_DIGEST" + _HWS + r":" + _HWS + r"`?([0-9a-f]{64})`?" + _HWS + r"$"
)
TASK_ID_RE = re.compile(r"\b(\d{13}-[0-9a-f]{6,8})\b")
HEX64 = r"([0-9a-f]{64})"

# Label-required digest forms (union). NEVER bare hex without DIFF_DIGEST label.
# All forms are line-local (no \s that matches newlines between label and hex).
_DIGEST_FORMS = (
    re.compile(
        r"(?im)^" + _HWS + r"DIFF_DIGEST" + _HWS + r":" + _HWS + r"`?" + HEX64 + r"`?" + _HWS + r"$"
    ),  # 1 strict line
    re.compile(r"(?im)\bDIFF_DIGEST" + _HWS + r":" + _HWS + r"`?" + HEX64 + r"`?"),  # 2 inline colon
    re.compile(r"(?im)\bDIFF_DIGEST" + _HWS + r"`" + HEX64 + r"`"),  # 3 backtick bind
    re.compile(r"(?im)\bFRESH[ \t]+DIFF_DIGEST\b[^\n0-9a-f]{0,80}" + HEX64),  # 4 FRESH
    re.compile(r"(?im)\bDIFF_DIGEST\b[^\n0-9a-f]{0,40}" + HEX64),  # 5 proximity
)

# Authoritative bind forms when |A| > 1 (per non-quoted line only)
_AUTH_STRICT_LINE = re.compile(
    r"(?im)^" + _HWS + r"DIFF_DIGEST" + _HWS + r":" + _HWS + r"`?" + HEX64 + r"`?" + _HWS + r"$"
)
_AUTH_FRESH = re.compile(
    r"(?im)^" + _HWS + r"#{0,3}" + _HWS + r"FRESH[ \t]+DIFF_DIGEST\b[^\n0-9a-f]{0,80}" + HEX64
)
_AUTH_PRIMARY_COLON = re.compile(
    r"(?im)^" + _HWS + r"DIFF_DIGEST" + _HWS + r":" + _HWS + r"`?" + HEX64 + r"`?"
)

_DEMOTION_LINE = re.compile(r"(?i)\b(prior|dead|void|superseded|old)\b")

# F1–F8: anchored line acts only (no leading free-prose prefix)
_F_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("F1", re.compile(r"(?im)^\s*#{0,3}\s*DIFF\s+GATE\s+REQUEST\b")),
    ("F2", re.compile(r"(?im)^\s*#{0,3}\s*(?:claude\s+)?gate-1\b[^\n]{0,40}\bFREEZE\b")),
    ("F3", re.compile(r"(?im)^\s*#{0,3}\s*GATE-1\s+PASS\s*\+\s*[^\n]{0,60}\bFREEZE\b")),
    ("F4", re.compile(r"(?im)^\s*#{0,3}\s*STAGED\s+FREEZE\b")),
    ("F5", re.compile(r"(?im)^\s*#{0,3}\s*FREEZE\s+LOCKED\b")),
    ("F6", re.compile(r"(?im)^\s*#{0,3}\s*frozen\s+handoff\b")),
    ("F7", re.compile(r"(?im)^\s*#{0,3}\s*validation/diff\s+handoff\b")),
    ("F8", re.compile(r"(?im)^\s*#{0,3}\s*FRESH\s+DIFF_DIGEST\b")),
)

# X1 line-start only; X2 nudge; X3 FOLLOW-UP opening; X5 quoted-only F hits
_X1 = re.compile(r"(?im)^\s*CO_LEAD_GATE_OVERRIDE\b")
_X2 = re.compile(
    r"(?i)\b(wake\s+nudge|standing\s+review_request\s+wake-gap\s+workaround)\b"
)
_X3_OPEN = re.compile(r"(?im)^\s*FOLLOW-UP\b")

COLEAD_PASS_MARKERS = (
    re.compile(r"(?im)co_lead\s+gate-2\s+PASS"),
    re.compile(r"(?im)validation/diff\s+(?:review\s*:\s*)?PASS"),
    re.compile(r"(?im)\bgate-2\s+PASS\b"),
)
COLEAD_DEFERRAL_MARKERS = (
    re.compile(r"(?im)\bno\s+(?:co_lead\s+)?approval\b"),
    re.compile(r"(?im)\bno\s+dual-accept\b"),
    re.compile(r"(?im)\bdeferred?\s+until\b"),
    re.compile(r"(?im)\bholding\s+(?:for|until)\b"),
    re.compile(r"(?im)\bvisibility\s+only\b"),
)
COLEAD_BLOCK_MARKERS = (
    re.compile(r"(?im)co_lead\s+gate-2\s+(?:BLOCK|REVISE)"),
    re.compile(r"(?im)\bgate-2\s+(?:BLOCK|REVISE)\b"),
    re.compile(r"(?im)validation/diff\s+.*\b(?:BLOCK|REVISE)\b"),
)


# --- primitives ------------------------------------------------------------


def parse_ts(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        v = float(value)
        return v / (1000.0 if v > 1e12 else 1.0)
    if isinstance(value, str):
        try:
            s = value.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            from datetime import datetime, timezone

            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
        except Exception:
            try:
                v = float(value)
                return v / (1000.0 if v > 1e12 else 1.0)
            except Exception:
                return None
    return None


def body(rec: dict[str, Any]) -> str:
    b = rec.get("body")
    return b if isinstance(b, str) else ""


def _nonquoted_lines(body_text: str) -> list[tuple[int, str]]:
    """Return (index, line) for non-blockquote lines."""
    out: list[tuple[int, str]] = []
    for i, line in enumerate(body_text.splitlines()):
        if line.lstrip().startswith(">"):
            continue
        out.append((i, line))
    return out


# --- A1 digest-set + demotion-before-uniqueness ----------------------------


def extract_digests(body_text: str) -> set[str]:
    """Label-required digests on non-quoted lines only (forms 1–5, union).

    Diagnostic/raw scan for tests; authority uses authoritative_digests.
    Never bare hex. Never joins label to hex across newlines.
    """
    found: set[str] = set()
    for _i, line in _nonquoted_lines(body_text):
        for pat in _DIGEST_FORMS:
            for m in pat.finditer(line):
                found.add(m.group(1).lower())
    return found


def extract_digest(body_text: str) -> str | None:
    """Backward-compat single-digest helper: first strict-line match, else any labeled."""
    for _i, line in _nonquoted_lines(body_text):
        m = DIFF_DIGEST_RE.search(line)
        if m:
            return m.group(1).lower()
    digs = extract_digests(body_text)
    if len(digs) == 1:
        return next(iter(digs))
    return None


def _digest_binding_lines(body_text: str) -> dict[str, list[str]]:
    """Map digest -> list of non-quoted lines that bind it via a label form."""
    by_d: dict[str, list[str]] = {}
    for _i, line in _nonquoted_lines(body_text):
        for pat in _DIGEST_FORMS:
            for m in pat.finditer(line):
                d = m.group(1).lower()
                by_d.setdefault(d, []).append(line)
    return by_d


def authoritative_digests(body_text: str) -> set[str]:
    """Authoritative digests: non-quoted same-line bindings only + demotion-first.

    A raw digest with an empty non-quoted binding-line list does NOT promote
    (quoted-only or cross-line extractions are never authority).
    """
    binding = _digest_binding_lines(body_text)
    if not binding:
        return set()
    A: set[str] = set()
    for d, lines in binding.items():
        # Empty binding list cannot occur here (keys come from binding), but
        # fail-closed if empty: do not promote.
        if not lines:
            continue
        # Demotion-before-uniqueness: survive only if some binding line is not demoted.
        for ln in lines:
            if not _DEMOTION_LINE.search(ln):
                A.add(d)
                break
    return A


def _line_authoritatively_binds(line: str, staged: str) -> bool:
    staged = staged.lower()
    for pat in (_AUTH_STRICT_LINE, _AUTH_FRESH, _AUTH_PRIMARY_COLON):
        m = pat.search(line)
        if m and m.group(1).lower() == staged:
            if _DEMOTION_LINE.search(line):
                return False
            return True
    return False


def record_authoritatively_binds(body_text: str, staged_digest: str) -> bool:
    """Staged digest is authoritatively bound by this record (A1 rules)."""
    staged = staged_digest.lower()
    A = authoritative_digests(body_text)
    if staged not in A:
        return False
    if len(A) == 1:
        return True
    # |A| > 1: need explicit authoritative bind form for staged
    for line in body_text.splitlines():
        if line.lstrip().startswith(">"):
            continue
        if _line_authoritatively_binds(line, staged):
            return True
    return False


# --- A2 freeze-intent ------------------------------------------------------


def _has_anchored_f_marker(body_text: str) -> bool:
    """True if any F1–F8 matches as an anchored non-quoted line act."""
    for _fid, pat in _F_PATTERNS:
        for _i, line in _nonquoted_lines(body_text):
            if pat.search(line):
                return True
    return False


def _exclusion_fires(body_text: str) -> bool:
    """X beats F. X1 line-start; X2 any non-quoted; X3 opening act; X5 quoted-only F."""
    lines = body_text.splitlines()
    nonquoted = [ln for ln in lines if not ln.lstrip().startswith(">")]

    # X1: line-start CO_LEAD_GATE_OVERRIDE
    for ln in lines:
        if _X1.search(ln):
            return True

    # X2: wake-nudge language on a non-quoted line
    for ln in nonquoted:
        if _X2.search(ln):
            return True

    # X3: FOLLOW-UP as body's own opening act (first non-empty non-quoted line)
    for ln in nonquoted:
        if not ln.strip():
            continue
        if _X3_OPEN.search(ln):
            return True
        break

    # X5: only F* hits live exclusively inside blockquotes
    f_in_nonquoted = False
    f_in_quoted = False
    for ln in lines:
        quoted = ln.lstrip().startswith(">")
        for _fid, pat in _F_PATTERNS:
            # For quoted lines, strip leading > for pattern? Patterns use ^\s* so
            # "> DIFF GATE" won't match F1. Check raw content after >
            probe = ln.lstrip()[1:] if quoted else ln
            if pat.search(probe if quoted else ln) or (
                quoted and any(p.search(probe) for _f, p in _F_PATTERNS)
            ):
                if quoted:
                    f_in_quoted = True
                else:
                    f_in_nonquoted = True
    # Simpler X5: has F on quoted-only content and no anchored F on non-quoted
    if not f_in_nonquoted:
        # If the only F-like markers are inside quotes, X5 applies when quoted
        # lines contain freeze vocabulary that would match if unquoted.
        for ln in lines:
            if not ln.lstrip().startswith(">"):
                continue
            probe = re.sub(r"^\s*>\s?", "", ln)
            for _fid, pat in _F_PATTERNS:
                if pat.search(probe):
                    f_in_quoted = True
                    break
        if f_in_quoted and not _has_anchored_f_marker(body_text):
            return True

    return False


def is_claude_freeze(rec: dict[str, Any], staged_digest: str | None = None) -> bool:
    """Freeze iff from=claude, authoritative staged digest, ≥1 F act, no X.

    ``staged_digest`` is required for full v4 semantics. If omitted, returns
    False (callers must pass staged digest for authorization decisions).
    """
    if rec.get("from") != "claude":
        return False
    if not staged_digest:
        return False
    body_text = body(rec)
    if not record_authoritatively_binds(body_text, staged_digest):
        return False
    if _exclusion_fires(body_text):
        return False
    if not _has_anchored_f_marker(body_text):
        return False
    return True


# --- shared helpers --------------------------------------------------------


def is_worker_receipt(rec: dict[str, Any]) -> bool:
    frm = str(rec.get("from", ""))
    if frm in {"claude", "codex_co_lead", "gabe", "watchdog"}:
        return False
    kind = str(rec.get("kind", ""))
    body_text = body(rec)
    if kind == "validation_receipt":
        return True
    return any(
        marker in body_text
        for marker in (
            "VALIDATION RECEIPT",
            "VALIDATION_RECEIPT",
            "IMPLEMENTATION RECEIPT",
            "TERMINAL RECEIPT",
        )
    )


def colead_verdict(body_text: str) -> str:
    for pat in COLEAD_BLOCK_MARKERS:
        if pat.search(body_text):
            return "block"
    for pat in COLEAD_DEFERRAL_MARKERS:
        if pat.search(body_text):
            return "unknown"
    for pat in COLEAD_PASS_MARKERS:
        if pat.search(body_text):
            return "pass"
    return "unknown"


def same_thread(rec: dict[str, Any], anchor_ids: set[str]) -> bool:
    if not anchor_ids:
        return True
    rid = str(rec.get("id", ""))
    reply_to = str(rec.get("reply_to", ""))
    body_text = body(rec)
    if rid in anchor_ids or reply_to in anchor_ids:
        return True
    return any(aid in body_text for aid in anchor_ids)


# --- A3/A4 PASS selection --------------------------------------------------


def find_fresh_colead_pass(
    records: list[dict[str, Any]],
    staged_digest: str,
) -> tuple[bool, str]:
    staged = staged_digest.lower()
    freeze_ts: float | None = None
    freeze_ids: set[str] = set()
    task_ids: set[str] = set()

    for rec in records:
        if not is_claude_freeze(rec, staged):
            continue
        # already verified authoritative bind inside is_claude_freeze
        ts = parse_ts(rec.get("ts"))
        if freeze_ts is None or (ts is not None and ts >= freeze_ts):
            freeze_ts = ts
            freeze_ids = set()
            rid = str(rec.get("id", ""))
            if rid:
                freeze_ids.add(rid)
            reply_to = str(rec.get("reply_to", ""))
            if reply_to:
                freeze_ids.add(reply_to)
            task_ids = set()
            m = TASK_ID_RE.search(body(rec))
            if m:
                task_ids.add(m.group(1))

    if freeze_ts is None:
        return False, "no claude freeze/handoff carrying matching DIFF_DIGEST"

    anchor_ids = set(freeze_ids)
    worker_ts: float | None = None
    for rec in records:
        if not is_worker_receipt(rec):
            continue
        body_text = body(rec)
        if not (
            same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body_text for tid in task_ids))
        ):
            continue
        ts = parse_ts(rec.get("ts"))
        if ts is None:
            continue
        # A3: only pre-freeze worker receipts update worker_ts
        if ts > freeze_ts:
            # may still expand anchors for threading, but do not move worker_ts
            rid = str(rec.get("id", ""))
            if rid:
                anchor_ids.add(rid)
            continue
        if worker_ts is None or ts > worker_ts:
            worker_ts = ts
            rid = str(rec.get("id", ""))
            if rid:
                anchor_ids.add(rid)

    if worker_ts is not None and freeze_ts <= worker_ts:
        return False, "claude freeze must be after scoped worker receipt on-thread"

    best_pass_ts: float | None = None
    for rec in records:
        if rec.get("from") != "codex_co_lead":
            continue
        body_text = body(rec)
        # A4: authoritative membership (not single extract)
        if not record_authoritatively_binds(body_text, staged):
            continue
        verdict = colead_verdict(body_text)
        if verdict != "pass":
            continue
        ts = parse_ts(rec.get("ts"))
        if ts is None or ts <= freeze_ts:
            continue
        if worker_ts is not None and ts <= worker_ts:
            continue
        if not (
            same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body_text for tid in task_ids))
        ):
            continue
        if best_pass_ts is None or ts >= best_pass_ts:
            best_pass_ts = ts

    if best_pass_ts is None:
        return (
            False,
            "no codex_co_lead validation/diff PASS echoing staged DIFF_DIGEST on-thread after freeze",
        )
    return True, "fresh co_lead PASS matches staged DIFF_DIGEST"
