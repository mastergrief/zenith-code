#!/usr/bin/env python3
"""SessionStart hook: persist compact summaries and inject recovery pointers."""

import glob
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone


def resolve_dest_dir(data: dict) -> str:
    override = os.environ.get("CLAUDE_MINUTES_DIR")
    if override:
        return override
    cwd = data.get("cwd") or os.getcwd()
    cur = os.path.abspath(cwd)
    while True:
        candidate = os.path.join(cur, ".claude")
        if os.path.isdir(candidate):
            return os.path.join(candidate, "MEMORY", "minutes")
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.expanduser("~/.claude/MEMORY/minutes")


def safe_filename_part(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return cleaned or "unknown_session"


def text_from_content(content) -> str:
    if isinstance(content, str):
        return content.strip()
    if not isinstance(content, list):
        return ""

    parts: list[str] = []
    for block in content:
        if isinstance(block, str):
            text = block
        elif isinstance(block, dict):
            text = block.get("text") if isinstance(block.get("text"), str) else ""
        else:
            text = ""
        text = text.strip()
        if text:
            parts.append(text)
    return "\n\n".join(parts).strip()


def latest_compact_summary(transcript_path: str) -> dict | None:
    latest: dict | None = None
    try:
        with open(transcript_path, encoding="utf-8") as f:
            for raw in f:
                try:
                    record = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(record, dict):
                    continue
                message = record.get("message")
                if not isinstance(message, dict):
                    message = {}
                is_summary = bool(
                    record.get("isCompactSummary") or message.get("isCompactSummary")
                )
                if not is_summary:
                    continue
                text = text_from_content(message.get("content"))
                if not text:
                    continue
                latest = {
                    "text": text,
                    "uuid": record.get("uuid") or message.get("uuid") or "",
                    "timestamp": record.get("timestamp") or message.get("timestamp") or "",
                    "session_id": record.get("sessionId") or message.get("sessionId") or "",
                }
    except Exception:
        return None
    return latest


def summary_key(summary: dict, session_id: str) -> tuple[str, str]:
    uuid = str(summary.get("uuid") or "").strip()
    if uuid:
        return "uuid", uuid
    digest_src = "\n".join(
        [
            session_id,
            str(summary.get("timestamp") or ""),
            str(summary.get("text") or ""),
        ]
    )
    return "digest", hashlib.sha256(digest_src.encode("utf-8")).hexdigest()[:16]


def resolve_full_minutes_path(minutes_dir: str, session_id: str) -> str | None:
    if not session_id:
        return None
    safe_session = safe_filename_part(session_id)
    exact = os.path.join(minutes_dir, f"{safe_session}.md")
    if os.path.isfile(exact):
        return exact

    short = session_id[:8]
    if not short:
        return None
    matches = [
        path
        for path in glob.glob(os.path.join(minutes_dir, f"*_{short}.md"))
        if os.path.isfile(path)
    ]
    if not matches:
        return None
    return max(matches, key=lambda path: (os.path.getmtime(path), path))


def append_summary(minutes_dir: str, session_id: str, summary: dict) -> str | None:
    safe_session = safe_filename_part(session_id)
    archive_path = os.path.join(minutes_dir, f"{safe_session}_compaction_summaries.md")
    key_kind, key_value = summary_key(summary, session_id)
    marker = f"<!-- compact-summary-{key_kind}:{key_value} -->"

    os.makedirs(minutes_dir, exist_ok=True)
    if os.path.exists(archive_path):
        try:
            with open(archive_path, encoding="utf-8") as f:
                if marker in f.read():
                    return archive_path
        except Exception:
            return None

    timestamp = str(summary.get("timestamp") or "unknown timestamp")
    archived_at = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    uuid = str(summary.get("uuid") or "").strip()
    uuid_line = f"- UUID: `{uuid}`" if uuid else f"- UUID: `(missing; {key_kind} fallback {key_value})`"
    new_file = not os.path.exists(archive_path)
    entry = "\n".join(
        [
            f"## Compaction Summary — {timestamp}",
            marker,
            f"- Session: `{session_id}`",
            uuid_line,
            f"- Archived: `{archived_at}`",
            "",
            str(summary.get("text") or "").strip(),
            "",
        ]
    )
    try:
        with open(archive_path, "a", encoding="utf-8") as f:
            if new_file:
                f.write(f"# Compaction Summaries for `{session_id}`\n\n")
            f.write(entry)
            f.write("\n")
    except Exception:
        return None
    return archive_path


def emit_context(kind: str, archive_path: str | None, full_minutes_path: str | None) -> None:
    if kind == "persisted_summary":
        lines = [
            "[post-compact state preservation]",
            "A compact-summary record was persisted for this session.",
        ]
        if archive_path:
            lines.append(f"Compaction summary archive: {archive_path}")
        if full_minutes_path:
            lines.append(f"Full session minutes: {full_minutes_path}")
        lines.append(
            "For recovery, read the archive first; use full minutes for the complete turn record."
        )
    elif kind == "archive_unavailable":
        lines = [
            "[post-compact full-minutes pointer]",
            "A compact-summary record was found, but it was not archived.",
            f"Full session minutes: {full_minutes_path}",
            "For recovery, read the full minutes file for the complete turn record.",
        ]
    else:
        lines = [
            "[post-compact full-minutes pointer]",
            "No compact-summary record was found in the transcript.",
            f"Full session minutes: {full_minutes_path}",
            "For recovery, read the full minutes file for the complete turn record.",
        ]

    print(
        json.dumps(
            {
                "hookSpecificOutput": {
                    "hookEventName": "SessionStart",
                    "additionalContext": "\n".join(lines),
                }
            },
            ensure_ascii=False,
        )
    )


def load_stdin() -> dict | None:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return None
        data = json.loads(raw)
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def main() -> int:
    data = load_stdin()
    if not data or data.get("source") != "compact":
        return 0

    transcript_path = data.get("transcript_path")
    summary = None
    if isinstance(transcript_path, str) and os.path.isfile(transcript_path):
        summary = latest_compact_summary(transcript_path)

    session_id = str(data.get("session_id") or "").strip()
    if not session_id and summary:
        session_id = str(summary.get("session_id") or "").strip()
    if not session_id and isinstance(transcript_path, str) and transcript_path:
        session_id = os.path.splitext(os.path.basename(transcript_path))[0]
    if not session_id:
        return 0

    minutes_dir = resolve_dest_dir(data)
    full_minutes_path = resolve_full_minutes_path(minutes_dir, session_id)

    if summary:
        archive_path = append_summary(minutes_dir, session_id, summary)
        if archive_path:
            emit_context("persisted_summary", archive_path, full_minutes_path)
        elif full_minutes_path:
            emit_context("archive_unavailable", None, full_minutes_path)
        return 0

    if full_minutes_path:
        emit_context("full_minutes_only", None, full_minutes_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
