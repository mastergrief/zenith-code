"""
CALM context/archaeology backend — computed "why does this exist?"

"Why is this code here?" decomposes into data retrieval:
  - When was it added? → git blame
  - What problem did it solve? → commit message
  - What happens without it? → delete + test
  - Who else needs it? → impact analysis

Functions: why_exists, function_history, code_age, change_frequency,
hotspots, contributor_map, related_changes.
"""

from __future__ import annotations

import os
import re
import subprocess
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional


def _git(args: list, cwd: str = None) -> str:
    try:
        proc = subprocess.run(
            ["git"] + args, capture_output=True, text=True,
            timeout=10, cwd=cwd or os.getcwd(),
        )
        return proc.stdout.strip() if proc.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""


def why_exists(path: str, line: int) -> dict:
    """Answer "why does this line exist?" via git archaeology.
    Returns {commit, author, date, message, age_days}.

    The commit message is the closest thing to "why" that's computable.
    """
    blame = _git(["blame", "-L", f"{int(line)},{int(line)}",
                   "--porcelain", str(path)])
    if not blame:
        return {"error": "no git history"}

    result = {"line": int(line), "file": str(path)}
    lines = blame.splitlines()
    if lines:
        result["commit"] = lines[0].split()[0][:8]

    for l in lines:
        if l.startswith("author "):
            result["author"] = l[7:]
        elif l.startswith("author-time "):
            import time
            ts = int(l[12:])
            result["date"] = time.strftime("%Y-%m-%d", time.localtime(ts))
            result["age_days"] = int((time.time() - ts) / 86400)
        elif l.startswith("summary "):
            result["message"] = l[8:]

    return result


def function_history(path: str, function_name: str, n: int = 10) -> list:
    """History of changes to a specific function.
    Uses git log -L to trace function evolution.
    Returns [{commit, author, date, message}].
    """
    out = _git(["log", f"-{int(n)}", "--format=%H|%an|%ai|%s",
                f"-L:^def {function_name}:/{path}"])
    if not out:
        # Fallback: grep for the function name in log.
        out = _git(["log", f"-{int(n)}", "--format=%H|%an|%ai|%s",
                     f"-S", function_name, "--", str(path)])

    results = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            results.append({
                "commit": parts[0][:8],
                "author": parts[1],
                "date": parts[2][:10],
                "message": parts[3],
            })
    return results


def code_age(path: str) -> dict:
    """Age analysis of a file — when was each part last modified?
    Returns {oldest_line, newest_line, avg_age_days, hotspot_lines}.
    """
    blame = _git(["blame", "--porcelain", str(path)])
    if not blame:
        return {"error": "no git history"}

    import time
    timestamps = []
    current_ts = None

    for line in blame.splitlines():
        if line.startswith("author-time "):
            current_ts = int(line[12:])
            timestamps.append(current_ts)

    if not timestamps:
        return {"error": "no timestamps found"}

    now = time.time()
    ages = [(now - ts) / 86400 for ts in timestamps]

    return {
        "file": str(path),
        "lines": len(ages),
        "oldest_days": int(max(ages)),
        "newest_days": int(min(ages)),
        "avg_age_days": int(sum(ages) / len(ages)),
        "recently_changed": sum(1 for a in ages if a < 7),
        "stale": sum(1 for a in ages if a > 180),
    }


def change_frequency(path: str, days: int = 30) -> dict:
    """How often has this file been changed recently?
    Returns {commits, authors, frequency_rating}.
    """
    out = _git(["log", f"--since={int(days)} days ago",
                "--format=%H|%an", "--", str(path)])
    if not out:
        return {"file": str(path), "commits": 0, "authors": [],
                "rating": "stable"}

    commits = out.strip().splitlines()
    authors = list(set(l.split("|")[1] for l in commits if "|" in l))

    n = len(commits)
    rating = (
        "stable" if n <= 2 else
        "active" if n <= 10 else
        "volatile" if n <= 20 else
        "churning"
    )

    return {
        "file": str(path),
        "commits": n,
        "period_days": int(days),
        "authors": authors,
        "rating": rating,
    }


def hotspots(directory: str, n: int = 10) -> list:
    """Find the most frequently changed files — likely complexity hotspots.
    Returns [{file, commits, rating}] sorted by commit count.
    """
    out = _git(["log", "--format=", "--name-only",
                "--diff-filter=M", "-100"], cwd=str(directory))
    if not out:
        return []

    counts = Counter(
        f for f in out.splitlines()
        if f.strip() and f.endswith('.py')
    )

    results = []
    for filepath, count in counts.most_common(int(n)):
        results.append({
            "file": filepath,
            "commits": count,
            "rating": (
                "stable" if count <= 3 else
                "active" if count <= 8 else
                "hotspot"
            ),
        })
    return results


def contributor_map(path: str) -> dict:
    """Who has worked on this file and how much?
    Returns {contributors: [{author, commits, pct}]}.
    """
    out = _git(["log", "--format=%an", "--", str(path)])
    if not out:
        return {"file": str(path), "contributors": []}

    counts = Counter(out.splitlines())
    total = sum(counts.values())

    contributors = [
        {
            "author": author,
            "commits": count,
            "pct": round(count / total * 100, 1),
        }
        for author, count in counts.most_common()
    ]

    return {
        "file": str(path),
        "total_commits": total,
        "contributors": contributors,
        "bus_factor": sum(1 for c in contributors if c["pct"] >= 20),
    }


def related_changes(path: str, n: int = 5) -> list:
    """Files that are frequently changed together with this file.
    Returns [{file, co_changes}] — files that appear in the same commits.

    High co-change = tightly coupled (even if no import relationship).
    """
    # Get commits that touched this file.
    commits = _git(["log", "--format=%H", "-20", "--", str(path)])
    if not commits:
        return []

    # For each commit, get the other files changed.
    co_files = Counter()
    for commit in commits.splitlines()[:20]:
        files = _git(["diff-tree", "--no-commit-id", "--name-only",
                       "-r", commit])
        for f in files.splitlines():
            if f.strip() and f != path and f.endswith('.py'):
                co_files[f] += 1

    return [
        {"file": f, "co_changes": count}
        for f, count in co_files.most_common(int(n))
    ]


CONTEXT_FUNCTIONS = {
    "why_exists": why_exists,
    "function_history": function_history,
    "code_age": code_age,
    "change_frequency": change_frequency,
    "hotspots": hotspots,
    "contributor_map": contributor_map,
    "related_changes": related_changes,
}

CONTEXT_NL_PATTERNS = [
    (r'(?:why does|why is|purpose of)\s+(?:this\s+)?(?:function|file|class|code)', None),
    (r'(?:how old|when was|code age)\s+(?:this\s+)?(?:file|function|code)', None),
    (r'(?:who|which developer|contributor)\s+(?:wrote|owns|maintains)', None),
    (r'(?:hotspots?|most changed|frequently modified)\s+(?:files?|code)', None),
]
