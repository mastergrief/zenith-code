"""
CALM git backend — verified repository operations.

The model writes "there are 5 commits this week" — the engine
runs git and counts.

Functions: log, diff_stat, blame, branch info, status, file history.
"""

from __future__ import annotations

import subprocess
from typing import List, Optional


def _git(args: list, cwd: str = None, timeout: float = 10.0) -> str:
    """Run a git command and return stdout."""
    try:
        proc = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=timeout, cwd=cwd,
        )
        return proc.stdout.strip() if proc.returncode == 0 else f"error: {proc.stderr.strip()}"
    except FileNotFoundError:
        return "error: git not found"
    except subprocess.TimeoutExpired:
        return "error: timeout"


def git_log(n: int = 10, path: str = None) -> list:
    """Recent commits. Returns [{hash, author, date, message}]."""
    args = ["log", f"-{int(n)}", "--format=%H|%an|%ai|%s"]
    if path:
        args += ["--", str(path)]
    out = _git(args)
    if out.startswith("error:"):
        return [{"error": out}]
    results = []
    for line in out.splitlines():
        parts = line.split("|", 3)
        if len(parts) == 4:
            results.append({
                "hash": parts[0][:8],
                "author": parts[1],
                "date": parts[2][:10],
                "message": parts[3],
            })
    return results


def git_diff_stat(ref: str = "HEAD") -> dict:
    """Diff stats against a ref. Returns {files_changed, insertions, deletions, files}."""
    out = _git(["diff", "--stat", ref])
    if out.startswith("error:") or not out:
        return {"files_changed": 0, "insertions": 0, "deletions": 0}
    lines = out.strip().splitlines()
    files = []
    for line in lines[:-1]:
        parts = line.strip().split("|")
        if len(parts) >= 1:
            files.append(parts[0].strip())
    # Parse summary line.
    summary = lines[-1] if lines else ""
    import re
    fc = re.search(r'(\d+) files? changed', summary)
    ins = re.search(r'(\d+) insertions?', summary)
    dels = re.search(r'(\d+) deletions?', summary)
    return {
        "files_changed": int(fc.group(1)) if fc else 0,
        "insertions": int(ins.group(1)) if ins else 0,
        "deletions": int(dels.group(1)) if dels else 0,
        "files": files,
    }


def git_status() -> dict:
    """Working tree status. Returns {modified, added, deleted, untracked}."""
    out = _git(["status", "--porcelain"])
    if out.startswith("error:"):
        return {"error": out}
    modified, added, deleted, untracked = [], [], [], []
    for line in out.splitlines():
        if len(line) < 3:
            continue
        status, path = line[:2], line[3:]
        if 'M' in status:
            modified.append(path)
        elif 'A' in status:
            added.append(path)
        elif 'D' in status:
            deleted.append(path)
        elif '?' in status:
            untracked.append(path)
    return {
        "modified": modified,
        "added": added,
        "deleted": deleted,
        "untracked": untracked,
        "clean": not (modified or added or deleted),
    }


def git_blame(path: str, line: int) -> dict:
    """Blame a specific line. Returns {hash, author, date, content}."""
    out = _git(["blame", "-L", f"{int(line)},{int(line)}", "--porcelain", str(path)])
    if out.startswith("error:"):
        return {"error": out}
    lines = out.splitlines()
    result = {"line": int(line)}
    for l in lines:
        if l.startswith("author "):
            result["author"] = l[7:]
        elif l.startswith("author-time "):
            result["timestamp"] = l[12:]
        elif l.startswith("summary "):
            result["message"] = l[8:]
    if lines:
        result["hash"] = lines[0].split()[0][:8]
    return result


def git_branch() -> dict:
    """Current branch info. Returns {current, branches, remote}."""
    current = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    branches = _git(["branch", "--format=%(refname:short)"])
    remote = _git(["rev-parse", "--abbrev-ref", "@{upstream}"])
    return {
        "current": current,
        "branches": branches.splitlines() if not branches.startswith("error:") else [],
        "remote": remote if not remote.startswith("error:") else None,
    }


def git_file_history(path: str, n: int = 5) -> list:
    """Recent commits that touched a file."""
    return git_log(n=n, path=path)


def git_commit_count(since: str = None) -> int:
    """Count commits, optionally since a date."""
    args = ["rev-list", "--count", "HEAD"]
    if since:
        args += [f"--since={since}"]
    out = _git(args)
    try:
        return int(out)
    except ValueError:
        return 0


GIT_FUNCTIONS = {
    "git_log": git_log,
    "git_diff_stat": git_diff_stat,
    "git_status": git_status,
    "git_blame": git_blame,
    "git_branch": git_branch,
    "git_file_history": git_file_history,
    "git_commit_count": git_commit_count,
}
