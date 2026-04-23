#!/usr/bin/env python3
"""Measure auto-loaded preload size for Claude Code memory tier.

Per Claude Code memory docs, files in .claude/rules/ auto-load at session
start. CLAUDE.md (project root) and .claude/CLAUDE.md also auto-load.
Anything else (.claude/spec/, .claude/MEMORY/, etc.) is query-triggered.

Files in .claude/rules/ with `paths:` YAML frontmatter only inject when
matching files are read — they're path-scoped, not eager.

Token estimate uses chars/4 heuristic — close enough for budgeting; for
exact counts pipe through tiktoken.

Used by /update Phase 5 as a fail-closed gate:
  python3 scripts/measure_preload.py --max-tokens 15000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def chars_to_tokens(n: int) -> int:
    return n // 4


def has_paths_frontmatter(text: str) -> bool:
    """Detect YAML frontmatter with a `paths:` key (Claude Code path-scoped rule)."""
    if not text.startswith("---\n"):
        return False
    end = text.find("\n---\n", 4)
    if end == -1:
        return False
    fm = text[4:end]
    for line in fm.splitlines():
        if line.strip().startswith("paths:"):
            return True
    return False


def measure(repo: Path) -> dict:
    files = []  # (label, path, scope) where scope ∈ {"eager", "path-scoped"}

    root_claude = repo / "CLAUDE.md"
    if root_claude.exists():
        files.append(("CLAUDE.md (root)", root_claude, "eager"))

    claude_md = repo / ".claude" / "CLAUDE.md"
    if claude_md.exists():
        files.append((".claude/CLAUDE.md", claude_md, "eager"))

    rules_dir = repo / ".claude" / "rules"
    if rules_dir.exists():
        for p in sorted(rules_dir.rglob("*.md")):
            rel = p.relative_to(repo)
            scope = "path-scoped" if has_paths_frontmatter(p.read_text(encoding="utf-8")) else "eager"
            files.append((str(rel), p, scope))

    rows = []
    eager_lines = eager_chars = 0
    pscope_lines = pscope_chars = 0
    for label, path, scope in files:
        text = path.read_text(encoding="utf-8")
        lines = text.count("\n") + (1 if text and not text.endswith("\n") else 0)
        chars = len(text)
        rows.append((label, lines, chars, chars_to_tokens(chars), scope))
        if scope == "eager":
            eager_lines += lines
            eager_chars += chars
        else:
            pscope_lines += lines
            pscope_chars += chars

    return {
        "rows": rows,
        "totals": {
            "files": len(rows),
            "eager_files": sum(1 for _, _, _, _, s in rows if s == "eager"),
            "path_scoped_files": sum(1 for _, _, _, _, s in rows if s == "path-scoped"),
            "eager_lines": eager_lines,
            "eager_tokens": chars_to_tokens(eager_chars),
            "path_scoped_lines": pscope_lines,
            "path_scoped_tokens": chars_to_tokens(pscope_chars),
            "total_lines": eager_lines + pscope_lines,
            "total_tokens": chars_to_tokens(eager_chars + pscope_chars),
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true", help="totals only")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="exit non-zero if eager-tier totals exceed this token budget")
    args = ap.parse_args()

    result = measure(REPO)

    if not args.quiet:
        print(f"{'file':<60} {'lines':>6} {'chars':>8} {'~tokens':>8}  scope")
        print("-" * 96)
        for label, lines, chars, tok, scope in result["rows"]:
            print(f"{label:<60} {lines:>6} {chars:>8} {tok:>8}  {scope}")
        print("-" * 96)

    t = result["totals"]
    print(f"EAGER (always-loaded):       {t['eager_files']:>3} files  {t['eager_lines']:>5} lines  ~{t['eager_tokens']:>6} tokens")
    print(f"PATH-SCOPED (on file match): {t['path_scoped_files']:>3} files  {t['path_scoped_lines']:>5} lines  ~{t['path_scoped_tokens']:>6} tokens")
    print(f"TOTAL:                       {t['files']:>3} files  {t['total_lines']:>5} lines  ~{t['total_tokens']:>6} tokens")

    if args.max_tokens is not None and t["eager_tokens"] > args.max_tokens:
        print(f"FAIL: eager {t['eager_tokens']} > {args.max_tokens} max", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
