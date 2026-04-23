#!/usr/bin/env python3
"""Measure auto-loaded preload size for Claude/Codex memory tiers.

Per Claude Code memory docs, files in .claude/rules/ auto-load at session
start. CLAUDE.md (project root) and .claude/CLAUDE.md also auto-load.
Anything else (.claude/spec/, .claude/MEMORY/, etc.) is query-triggered.

For Codex, .codex/AGENTS.md and .codex/rules/*.md are the repo-local eager
instruction surface loaded for Codex sessions.

Files in .claude/rules/ or .codex/rules/ with `paths:` YAML frontmatter
only inject when matching files are read — they're path-scoped, not eager.

Token estimate uses chars/4 heuristic — close enough for budgeting; for
exact counts pipe through tiktoken.

Used by /update Phase 5 as a fail-closed gate:
  python3 scripts/measure_preload.py --surface both --max-tokens 150000
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SURFACES = ("claude", "codex", "both")


def chars_to_tokens(n: int) -> int:
    return n // 4


def has_paths_frontmatter(text: str) -> bool:
    """Detect YAML frontmatter with a `paths:` key (path-scoped rule)."""
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


def add_rule_files(files: list[tuple[str, Path, str]], rules_dir: Path) -> None:
    if not rules_dir.exists():
        return
    for p in sorted(rules_dir.rglob("*.md")):
        rel = p.relative_to(REPO)
        scope = "path-scoped" if has_paths_frontmatter(p.read_text(encoding="utf-8")) else "eager"
        files.append((str(rel), p, scope))


def collect_files(repo: Path, surface: str) -> list[tuple[str, Path, str]]:
    files: list[tuple[str, Path, str]] = []

    if surface in {"claude", "both"}:
        root_claude = repo / "CLAUDE.md"
        if root_claude.exists():
            files.append(("CLAUDE.md (root)", root_claude, "eager"))

        claude_md = repo / ".claude" / "CLAUDE.md"
        if claude_md.exists():
            files.append((".claude/CLAUDE.md", claude_md, "eager"))

        add_rule_files(files, repo / ".claude" / "rules")

    if surface in {"codex", "both"}:
        codex_agents = repo / ".codex" / "AGENTS.md"
        if codex_agents.exists():
            files.append((".codex/AGENTS.md", codex_agents, "eager"))

        add_rule_files(files, repo / ".codex" / "rules")

    return files


def measure(repo: Path, surface: str) -> dict:
    files = collect_files(repo, surface)

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
            "surface": surface,
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
    ap.add_argument("--surface", choices=SURFACES, default="both",
                    help="instruction surface to measure (default: both)")
    ap.add_argument("--max-tokens", type=int, default=None,
                    help="exit non-zero if eager-tier totals exceed this token budget")
    args = ap.parse_args()

    result = measure(REPO, args.surface)

    if not args.quiet:
        print(f"surface: {args.surface}")
        print(f"{'file':<60} {'lines':>6} {'chars':>8} {'~tokens':>8}  scope")
        print("-" * 96)
        for label, lines, chars, tok, scope in result["rows"]:
            print(f"{label:<60} {lines:>6} {chars:>8} {tok:>8}  {scope}")
        print("-" * 96)

    t = result["totals"]
    print(f"SURFACE: {t['surface']}")
    print(f"EAGER (always-loaded):       {t['eager_files']:>3} files  {t['eager_lines']:>5} lines  ~{t['eager_tokens']:>6} tokens")
    print(f"PATH-SCOPED (on file match): {t['path_scoped_files']:>3} files  {t['path_scoped_lines']:>5} lines  ~{t['path_scoped_tokens']:>6} tokens")
    print(f"TOTAL:                       {t['files']:>3} files  {t['total_lines']:>5} lines  ~{t['total_tokens']:>6} tokens")

    if args.max_tokens is not None and t["eager_tokens"] > args.max_tokens:
        print(f"FAIL: eager {t['eager_tokens']} > {args.max_tokens} max", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
