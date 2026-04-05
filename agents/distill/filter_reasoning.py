"""Filter claude_reasoning.jsonl to coding/technical examples only.

Usage:
    python -m agents.distill.filter_reasoning          # filter only
    python -m agents.distill.filter_reasoning --merge   # filter + merge with hand-written data
"""

import argparse
import json
import re
import shutil
from pathlib import Path

from agents.distill.config import DATA_DIR

INPUT_FILE = DATA_DIR / "claude_reasoning.jsonl"
FILTERED_FILE = DATA_DIR / "claude_reasoning_filtered.jsonl"
HANDWRITTEN_FILE = DATA_DIR / "coding_reasoning_claude.jsonl"
BACKUP_FILE = DATA_DIR / "claude_reasoning.jsonl.bak"

# Coding/technical keywords (case-insensitive, checked as substrings)
CODING_KEYWORDS = [
    "function", "api", "debug", "error", "bug", "import", "return", "async",
    "python", "rust", "react", "javascript", "typescript", "sql", "java",
    "class", "method", "variable", "array", "string", "hash", "loop",
    "recursion", "algorithm", "data structure", "compile", "runtime",
    "memory", "promise", "callback", "frontend", "backend", "devops",
    "database", "server", "http", "json", "html", "css", "deploy",
    "docker", "git", "kubernetes", "test", "endpoint", "route",
    "middleware", "auth", "token", "session", "refactor", "optimize",
    "performance", "cache", "index", "query", "migration", "schema",
    "orm", "crud", "rest", "graphql", "monorepo", "webpack", "vite",
    "framework", "library", "package", "module", "component", "hook",
    "config", "cli", "terminal", "bash", "shell", "script", "regex",
    "parse", "serialize", "dependency", "build", "implement",
    "k8s", "cors", "jwt", "oauth", "ssl", "nginx", "webhook",
    "websocket", "socket", "stream", "queue", "logging", "tracing",
]

# Hallucination-prone patterns (factual claims that may be wrong)
HALLUCINATION_PATTERNS = [
    r"bitcoin", r"stock price", r"who won the", r"super bowl",
    r"current president", r"interest rate", r"latest version",
    r"current ceo", r"capital of \w+", r"population of",
    r"how old is", r"when did.*die", r"net worth",
    r"world series", r"price of",
]

# Structured reasoning markers (for keeping good general reasoning)
REASONING_MARKERS = [
    "step", "first", "then", "because", "therefore",
    "tradeoff", "approach", "consider", "analyze", "evaluate",
    "option", "alternative", "pros", "cons", "compare",
]


def is_coding_related(messages: list[dict]) -> bool:
    """Check if any message contains coding keywords or code blocks."""
    for msg in messages:
        content = msg.get("content", "").lower()
        # Code blocks
        if "```" in content:
            return True
        # Keyword match
        for kw in CODING_KEYWORDS:
            if kw in content:
                return True
    return False


def has_hallucination_risk(messages: list[dict]) -> bool:
    """Check if messages contain factual claims likely to be wrong."""
    for msg in messages:
        content = msg.get("content", "").lower()
        for pattern in HALLUCINATION_PATTERNS:
            if re.search(pattern, content):
                return True
    return False


def has_strong_reasoning(messages: list[dict]) -> bool:
    """Check if the assistant response has structured reasoning worth keeping."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if not think_match:
            continue
        think_block = think_match.group(1).lower()
        if len(think_block) < 500:
            continue
        marker_count = sum(1 for m in REASONING_MARKERS if m in think_block)
        if marker_count >= 3:
            return True
    return False


def assistant_length(messages: list[dict]) -> int:
    """Get total length of assistant responses."""
    return sum(
        len(msg.get("content", ""))
        for msg in messages
        if msg.get("role") == "assistant"
    )


def filter_dataset():
    """Filter reasoning dataset to coding/technical examples."""
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    examples = []
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    print(f"Read {len(examples)} examples from {INPUT_FILE}")

    kept_coding = []
    kept_reasoning = []
    rejected = {"short": 0, "hallucination": 0, "non_technical": 0}

    for ex in examples:
        msgs = ex.get("messages", [])

        # Reject short responses
        if assistant_length(msgs) < 200:
            rejected["short"] += 1
            continue

        # Reject hallucination-prone
        if has_hallucination_risk(msgs):
            rejected["hallucination"] += 1
            continue

        # Keep coding-related
        if is_coding_related(msgs):
            kept_coding.append(ex)
            continue

        # Keep strong general reasoning (teaches the thinking pattern)
        if has_strong_reasoning(msgs):
            kept_reasoning.append(ex)
            continue

        rejected["non_technical"] += 1

    # Cap general reasoning examples at 200
    kept_reasoning = kept_reasoning[:200]

    kept = kept_coding + kept_reasoning

    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Kept (coding):    {len(kept_coding)}")
    print(f"  Kept (reasoning): {len(kept_reasoning)}")
    print(f"  Total kept:       {len(kept)}")
    print(f"  Rejected:")
    print(f"    Short (<200):     {rejected['short']}")
    print(f"    Hallucination:    {rejected['hallucination']}")
    print(f"    Non-technical:    {rejected['non_technical']}")
    print(f"\nSaved to {FILTERED_FILE}")


def merge_datasets():
    """Merge filtered + hand-written datasets into final file."""
    if not FILTERED_FILE.exists():
        print(f"Error: {FILTERED_FILE} not found. Run without --merge first.")
        return

    # Backup original
    if INPUT_FILE.exists():
        shutil.copy2(INPUT_FILE, BACKUP_FILE)
        print(f"Backed up original to {BACKUP_FILE}")

    # Read filtered
    filtered = []
    with open(FILTERED_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                filtered.append(line.strip())

    # Read hand-written if it exists
    handwritten = []
    if HANDWRITTEN_FILE.exists():
        with open(HANDWRITTEN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    handwritten.append(line.strip())
        print(f"Read {len(handwritten)} hand-written examples")
    else:
        print(f"Note: {HANDWRITTEN_FILE} not found, merging filtered only")

    # Validate and write
    all_lines = filtered + handwritten
    valid = 0
    with open(INPUT_FILE, "w", encoding="utf-8") as f:
        for line in all_lines:
            try:
                data = json.loads(line)
                msgs = data.get("messages", [])
                if len(msgs) >= 2:
                    f.write(line + "\n")
                    valid += 1
            except json.JSONDecodeError:
                continue

    print(f"\nMerged dataset: {valid} examples written to {INPUT_FILE}")
    print(f"  From filtered:    {len(filtered)}")
    print(f"  From hand-written: {len(handwritten)}")


def main():
    parser = argparse.ArgumentParser(description="Filter reasoning dataset")
    parser.add_argument("--merge", action="store_true", help="Merge filtered + hand-written data")
    args = parser.parse_args()

    if args.merge:
        merge_datasets()
    else:
        filter_dataset()


if __name__ == "__main__":
    main()
