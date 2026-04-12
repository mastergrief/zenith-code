"""
CALM diff backend — verified unified diff parsing and analysis.

Models misread diffs constantly — wrong hunk counts, wrong line numbers,
confused about additions vs deletions. This backend parses diffs
deterministically.

Functions: diff_parse, diff_stats, diff_files, diff_hunks, diff_apply, diff_conflicts.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple


def diff_parse(diff_text: str) -> dict:
    """Parse a unified diff into structured data.
    Returns {files: [{path, hunks: [{old_start, old_count, new_start, new_count, lines}]}]}"""
    files = []
    current_file = None
    current_hunk = None

    for line in diff_text.splitlines():
        # New file header.
        if line.startswith("diff --git"):
            m = re.match(r'diff --git a/(.+?) b/(.+)', line)
            if m:
                current_file = {"old_path": m.group(1), "new_path": m.group(2), "hunks": []}
                files.append(current_file)
                current_hunk = None
            continue

        if line.startswith("--- ") and current_file is not None:
            p = line[4:].strip()
            if p.startswith("a/"):
                p = p[2:]
            current_file["old_path"] = p
            continue

        if line.startswith("+++ ") and current_file is not None:
            p = line[4:].strip()
            if p.startswith("b/"):
                p = p[2:]
            current_file["new_path"] = p
            continue

        # Hunk header.
        hunk_match = re.match(r'^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)', line)
        if hunk_match:
            if current_file is None:
                current_file = {"old_path": "unknown", "new_path": "unknown", "hunks": []}
                files.append(current_file)
            current_hunk = {
                "old_start": int(hunk_match.group(1)),
                "old_count": int(hunk_match.group(2) or 1),
                "new_start": int(hunk_match.group(3)),
                "new_count": int(hunk_match.group(4) or 1),
                "header": hunk_match.group(5).strip(),
                "lines": [],
            }
            current_file["hunks"].append(current_hunk)
            continue

        # Diff content lines.
        if current_hunk is not None:
            if line.startswith("+"):
                current_hunk["lines"].append({"type": "add", "content": line[1:]})
            elif line.startswith("-"):
                current_hunk["lines"].append({"type": "del", "content": line[1:]})
            elif line.startswith(" "):
                current_hunk["lines"].append({"type": "ctx", "content": line[1:]})
            elif line.startswith("\\"):
                current_hunk["lines"].append({"type": "meta", "content": line})

    return {"file_count": len(files), "files": files}


def diff_stats(diff_text: str) -> dict:
    """Get statistics from a unified diff.
    Returns {files_changed, additions, deletions, hunks}."""
    parsed = diff_parse(diff_text)
    additions = 0
    deletions = 0
    hunks = 0

    for f in parsed["files"]:
        hunks += len(f["hunks"])
        for h in f["hunks"]:
            for line in h["lines"]:
                if line["type"] == "add":
                    additions += 1
                elif line["type"] == "del":
                    deletions += 1

    return {
        "files_changed": parsed["file_count"],
        "additions": additions,
        "deletions": deletions,
        "net_change": additions - deletions,
        "hunks": hunks,
    }


def diff_files(diff_text: str) -> list:
    """Extract the list of files changed in a diff.
    Returns list of {old_path, new_path, hunks, additions, deletions}."""
    parsed = diff_parse(diff_text)
    result = []
    for f in parsed["files"]:
        adds = sum(1 for h in f["hunks"] for l in h["lines"] if l["type"] == "add")
        dels = sum(1 for h in f["hunks"] for l in h["lines"] if l["type"] == "del")
        result.append({
            "old_path": f["old_path"],
            "new_path": f["new_path"],
            "hunks": len(f["hunks"]),
            "additions": adds,
            "deletions": dels,
        })
    return result


def diff_hunks(diff_text: str, file_path: str = "") -> list:
    """Extract hunks from a diff, optionally filtered by file path.
    Returns list of {old_start, old_count, new_start, new_count, additions, deletions}."""
    parsed = diff_parse(diff_text)
    result = []
    for f in parsed["files"]:
        if file_path and file_path not in f.get("new_path", "") and file_path not in f.get("old_path", ""):
            continue
        for h in f["hunks"]:
            adds = sum(1 for l in h["lines"] if l["type"] == "add")
            dels = sum(1 for l in h["lines"] if l["type"] == "del")
            result.append({
                "file": f["new_path"],
                "old_start": h["old_start"],
                "old_count": h["old_count"],
                "new_start": h["new_start"],
                "new_count": h["new_count"],
                "additions": adds,
                "deletions": dels,
                "header": h.get("header", ""),
            })
    return result


def diff_apply(original: str, diff_text: str) -> str:
    """Apply a unified diff to original text (single file, best-effort).
    Returns the patched text."""
    parsed = diff_parse(diff_text)
    if not parsed["files"]:
        return original

    lines = original.splitlines(keepends=True)
    # Process hunks in reverse order to maintain line numbers.
    file_data = parsed["files"][0]
    for hunk in reversed(file_data["hunks"]):
        start = hunk["old_start"] - 1  # 0-indexed.
        old_count = hunk["old_count"]
        new_lines = []
        for dl in hunk["lines"]:
            if dl["type"] in ("add", "ctx"):
                new_lines.append(dl["content"] + "\n")
            # "del" lines are removed (not included in new_lines).

        # Replace the old range with new lines.
        if start >= 0 and start + old_count <= len(lines):
            lines[start:start + old_count] = new_lines
        else:
            # Best effort — append.
            lines.extend(new_lines)

    return "".join(lines)


def diff_conflicts(diff_text: str) -> list:
    """Detect merge conflict markers in diff content.
    Returns list of {file, line, type} for each conflict marker found."""
    conflicts = []
    parsed = diff_parse(diff_text)
    for f in parsed["files"]:
        for h in f["hunks"]:
            line_num = h["new_start"]
            for dl in h["lines"]:
                content = dl["content"]
                if content.startswith("<<<<<<<"):
                    conflicts.append({"file": f["new_path"], "line": line_num, "type": "start"})
                elif content.startswith("======="):
                    conflicts.append({"file": f["new_path"], "line": line_num, "type": "separator"})
                elif content.startswith(">>>>>>>"):
                    conflicts.append({"file": f["new_path"], "line": line_num, "type": "end"})
                if dl["type"] in ("add", "ctx"):
                    line_num += 1
    return conflicts


DIFF_FUNCTIONS = {
    "diff_parse": diff_parse,
    "diff_stats": diff_stats,
    "diff_files": diff_files,
    "diff_hunks": diff_hunks,
    "diff_apply": diff_apply,
    "diff_conflicts": diff_conflicts,
}
