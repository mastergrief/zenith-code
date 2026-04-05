"""Filter claude_reasoning.jsonl to coding/technical examples only.

Usage:
    python -m agents.distill.filter_reasoning          # filter only
    python -m agents.distill.filter_reasoning --merge   # filter + merge with hand-written data
"""

import argparse
import hashlib
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
    r"vitamin\s+c.*mega.*dose", r"einstein.*failed.*math",
    r"immune.*boost",
    # NLP benchmark tasks that sneak through keyword filters
    r"predict the sentiment", r"premise:.*hypothesis:",
    r"pick one category for the following", r"write a text based on",
    r"multi-choice question for", r"tweet:.*predict",
    r"you will be given a (definition|competitive)",
    r"your solution must read input",
    r"what would be the.*rating.*review",
    # Science content
    r"what protein", r"dark matter", r"enzyme.*catalyz",
    r"photon.*wavelength", r"kinetic energy.*potential",
]


# Strong keywords — language names, frameworks, tools. Unlikely to appear
# in non-coding contexts. One of these + 2 general keywords = coding.
STRONG_KEYWORDS = [
    "python", "rust", "react", "javascript", "typescript", "sql", "java ",
    "golang", "ruby", "swift", "kotlin", "scala", "php", "c++",
    "node.js", "django", "fastapi", "flask", "express", "nextjs", "next.js",
    "docker", "kubernetes", "k8s", "nginx", "redis", "postgres", "mongodb",
    "webpack", "vite", "graphql", "rest api", "oauth", "jwt", "cors",
    "git ", "github", "gitlab", "ci/cd", "terraform", "ansible",
    "npm", "pip ", "cargo", "pytest", "jest", "unittest",
    "async/await", "asyncio", "tokio",
]


def is_coding_related(messages: list[dict]) -> bool:
    """Check if messages are coding-related.

    Requires code blocks OR (1 strong keyword + 2 general keywords).
    Single keyword matches produce too many false positives (e.g. "class"
    in a sociology question, "test" in a medical context).
    """
    all_content = ""
    for msg in messages:
        content = msg.get("content", "")
        # Code blocks are a strong signal
        if "```" in content:
            return True
        all_content += " " + content.lower()

    # Check for strong keyword (language/framework/tool name)
    has_strong = any(kw in all_content for kw in STRONG_KEYWORDS)

    # Count general keyword matches
    matches = set()
    for kw in CODING_KEYWORDS:
        if kw in all_content:
            matches.add(kw)

    # Strong keyword + 2 generals, OR 5+ generals without strong
    if has_strong and len(matches) >= 2:
        return True
    if len(matches) >= 5:
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


def has_adequate_think(messages: list[dict], min_length: int = 200) -> bool:
    """Check if the assistant response has a think block of adequate length."""
    for msg in messages:
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content", "")
        think_match = re.search(r"<think>(.*?)</think>", content, re.DOTALL)
        if think_match and len(think_match.group(1)) >= min_length:
            return True
    return False


def assistant_length(messages: list[dict]) -> int:
    """Get total length of assistant responses."""
    return sum(
        len(msg.get("content", ""))
        for msg in messages
        if msg.get("role") == "assistant"
    )


def content_hash(example: dict) -> str:
    """Hash an example by its user message for deduplication."""
    for msg in example.get("messages", []):
        if msg.get("role") == "user":
            return hashlib.md5(msg["content"].encode()).hexdigest()
    return ""


def deduplicate(examples: list[dict]) -> tuple[list[dict], int]:
    """Remove near-duplicate examples by user message prefix."""
    seen = set()
    unique = []
    removed = 0
    for ex in examples:
        user_msg = ""
        for msg in ex.get("messages", []):
            if msg.get("role") == "user":
                user_msg = re.sub(r"\s+", " ", msg["content"][:60].lower().strip())
                break
        if user_msg in seen:
            removed += 1
            continue
        seen.add(user_msg)
        unique.append(ex)
    return unique, removed


def get_handwritten_hashes() -> set[str]:
    """Get content hashes of all hand-written examples."""
    hashes = set()
    if HANDWRITTEN_FILE.exists():
        with open(HANDWRITTEN_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    ex = json.loads(line)
                    hashes.add(content_hash(ex))
    return hashes


def filter_dataset():
    """Filter reasoning dataset to coding/technical examples."""
    if not INPUT_FILE.exists():
        print(f"Error: {INPUT_FILE} not found")
        return

    # Load all examples, separating HF from hand-written
    handwritten_hashes = get_handwritten_hashes()
    hf_examples = []
    handwritten_in_file = 0

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if content_hash(ex) in handwritten_hashes:
                handwritten_in_file += 1
                continue  # skip hand-written, they get added back in merge
            hf_examples.append(ex)

    print(f"Read {len(hf_examples)} HF examples from {INPUT_FILE}")
    if handwritten_in_file:
        print(f"  (skipped {handwritten_in_file} hand-written examples)")

    kept = []
    rejected = {"short": 0, "hallucination": 0, "non_technical": 0, "short_think": 0}

    for ex in hf_examples:
        msgs = ex.get("messages", [])

        # Reject short responses
        if assistant_length(msgs) < 200:
            rejected["short"] += 1
            continue

        # Reject hallucination-prone
        if has_hallucination_risk(msgs):
            rejected["hallucination"] += 1
            continue

        # Only keep coding-related examples
        if is_coding_related(msgs):
            # Code blocks = strong signal, lower think threshold
            has_code = any("```" in m.get("content", "") for m in msgs)
            min_think = 200 if has_code else 400
            if has_adequate_think(msgs, min_length=min_think):
                kept.append(ex)
            else:
                rejected["short_think"] += 1
            continue

        # Not coding-related — reject
        rejected["non_technical"] += 1

    # Deduplicate
    kept, dup_count = deduplicate(kept)

    with open(FILTERED_FILE, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"\nResults:")
    print(f"  Kept (coding + adequate think): {len(kept)}")
    print(f"  Rejected:")
    print(f"    Short (<200):       {rejected['short']}")
    print(f"    Hallucination:      {rejected['hallucination']}")
    print(f"    Non-technical:      {rejected['non_technical']}")
    print(f"    Short think (<200): {rejected['short_think']}")
    print(f"    Duplicates:         {dup_count}")
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
