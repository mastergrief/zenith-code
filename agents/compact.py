"""Transcript compaction — summarize old messages to stay within context limits."""

import os
import re

COMPACT_CONTINUATION_PREAMBLE = (
    "This session is being continued from a previous conversation that ran out "
    "of context. The summary below covers the earlier portion of the conversation.\n\n"
)
COMPACT_RECENT_MESSAGES_NOTE = "Recent messages are preserved verbatim."
COMPACT_DIRECT_RESUME_INSTRUCTION = (
    "Continue the conversation from where it left off without asking the user "
    "any further questions. Resume directly — do not acknowledge the summary, "
    "do not recap what was happening, and do not preface with continuation text."
)

# Auto-compaction thresholds — the number of tokens at which the harness
# starts summarizing old turns. These are NOT the model's trained max — they
# are the *effective* context validated by needle-in-haystack testing. Pushing
# past these values often works but gets unreliable on multi-fact recall.
#
# Values under "llamacpp" refer to specific GGUF base models served via
# llama-server. The key is matched as a substring of the model name string.
# More specific keys (longer substrings) take precedence — see
# detect_context_limit() for the sort order.
#
# Validation source: .claude/MEMORY/evals/2026-04-07_*_needle_256k_*.md
# Methodology: single + multi + distractor NIAH tests across 4K-220K.
MODEL_CONTEXT_LIMITS = {
    # llama.cpp defaults + validated per-model overrides
    "llamacpp": 65536,            # generic fallback for unknown llama.cpp models
    "gemma-4-e4b": 200000,        # NIAH-validated clean through 180K (multi+distractor 7/7);
                                  #   220K drops 1 needle on multi (4/5). 200K sits
                                  #   10% below the failure point — chosen over 180K
                                  #   to use more of the model's available range
    "gemma-4-E4B": 200000,        # case-insensitive match for typical filename
    "qwen3.5-4b": 130000,         # NIAH-validated clean through 130K (multi+distractor);
                                  #   180K drops to 4/5 multi, and Qwen has a 64K/100K
                                  #   distractor dip where it picked a wrong decoy
    "Qwen3.5-4B": 130000,         # case-insensitive match for typical filename

    # Ollama fallbacks (smaller models served via Modelfiles)
    "9b": 2048,
    "8b": 2048,
    "4b": 8192,
    "0.8b": 1024,
    "0.6b": 2048,
}
DEFAULT_CONTEXT_LIMIT = 4096


def detect_context_limit(model_name: str) -> int:
    """Auto-detect context limit from model name string.

    Match order: longer keys first (so "gemma-4-e4b" beats "4b"), then shorter
    fallbacks. Case-sensitive substring match. The CLAW_AUTO_COMPACT_TOKENS
    env var overrides everything.
    """
    env_override = os.environ.get("CLAW_AUTO_COMPACT_TOKENS")
    if env_override:
        return int(env_override)
    # Check longer keys first to avoid "8b" matching before "0.8b"
    # and to avoid "4b" matching before "gemma-4-e4b"
    for key in sorted(MODEL_CONTEXT_LIMITS, key=len, reverse=True):
        if key in model_name:
            return MODEL_CONTEXT_LIMITS[key]
    return DEFAULT_CONTEXT_LIMIT


def estimate_tokens(messages: list[dict]) -> int:
    """Roughly estimate token count. Same heuristic as Rust compact.rs."""
    total = 0
    for msg in messages:
        content = msg.get("content", "")
        if isinstance(content, str):
            total += len(content) // 4 + 1
        # Tool call messages may have nested structure
        tool_calls = msg.get("tool_calls", [])
        for tc in tool_calls:
            fn = tc.get("function", {})
            total += (len(fn.get("name", "")) + len(str(fn.get("arguments", "")))) // 4 + 1
    return total


def should_compact(messages: list[dict], max_tokens: int, preserve_recent: int = 4) -> bool:
    """Check if compaction is needed."""
    # Skip leading compacted summary if present
    start = _compacted_prefix_len(messages)
    compactable = messages[start:]

    if len(compactable) <= preserve_recent:
        return False

    return estimate_tokens(compactable) >= max_tokens


def compact_history(messages: list[dict], preserve_recent: int = 4) -> list[dict]:
    """Compact old messages into a summary, preserving recent tail."""
    start = _compacted_prefix_len(messages)
    existing_summary = _extract_existing_summary(messages) if start > 0 else None

    keep_from = max(start, len(messages) - preserve_recent)
    old_messages = messages[start:keep_from]
    recent = messages[keep_from:]

    if not old_messages:
        return messages

    summary = summarize_messages(old_messages)
    if existing_summary:
        summary = _merge_summaries(existing_summary, summary)

    continuation = COMPACT_CONTINUATION_PREAMBLE + summary
    if recent:
        continuation += f"\n\n{COMPACT_RECENT_MESSAGES_NOTE}"
    continuation += f"\n{COMPACT_DIRECT_RESUME_INSTRUCTION}"

    return [{"role": "system", "content": continuation}] + recent


def summarize_messages(messages: list[dict]) -> str:
    """Produce a structured summary of compacted messages."""
    role_counts = {}
    tool_names = set()
    user_requests = []
    key_files = set()

    for msg in messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

        content = msg.get("content", "")
        if isinstance(content, str):
            # Collect recent user requests
            if role == "user" and content.strip():
                user_requests.append(_truncate(content.strip(), 160))
            # Extract file paths
            for match in re.findall(r'[\w./]+\.\w{1,4}', content):
                if '/' in match:
                    key_files.add(match)

        # Collect tool names
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            if fn.get("name"):
                tool_names.add(fn["name"])

    lines = [
        "<summary>",
        "Conversation summary:",
        f"- Scope: {len(messages)} earlier messages compacted "
        f"(user={role_counts.get('user', 0)}, "
        f"assistant={role_counts.get('assistant', 0)}, "
        f"tool={role_counts.get('tool', 0)}).",
    ]

    if tool_names:
        lines.append(f"- Tools mentioned: {', '.join(sorted(tool_names))}.")

    # Last 3 user requests
    recent = user_requests[-3:]
    if recent:
        lines.append("- Recent user requests:")
        for req in recent:
            lines.append(f"  - {req}")

    if key_files:
        files = sorted(key_files)[:8]
        lines.append(f"- Key files referenced: {', '.join(files)}.")

    lines.append("</summary>")
    return "\n".join(lines)


def compress_summary(text: str, max_chars: int = 1200, max_lines: int = 24) -> str:
    """Compress a summary to fit within size limits."""
    lines = text.split("\n")
    headers = [l for l in lines if l.startswith("#") or l.startswith("- ")]
    other = [l for l in lines if l not in headers]
    seen = set()
    deduped = []
    for l in headers + other:
        key = l.strip().lower()
        if key not in seen:
            seen.add(key)
            deduped.append(l[:160])
    result = "\n".join(deduped[:max_lines])
    return result[:max_chars]


def _compacted_prefix_len(messages: list[dict]) -> int:
    """Return 1 if the first message is a compacted summary, else 0."""
    if not messages:
        return 0
    first = messages[0]
    if first.get("role") == "system":
        content = first.get("content", "")
        if isinstance(content, str) and content.startswith(COMPACT_CONTINUATION_PREAMBLE):
            return 1
    return 0


def _extract_existing_summary(messages: list[dict]) -> str | None:
    """Extract summary text from a prior compaction message."""
    if not messages:
        return None
    first = messages[0]
    content = first.get("content", "")
    if isinstance(content, str) and content.startswith(COMPACT_CONTINUATION_PREAMBLE):
        summary = content[len(COMPACT_CONTINUATION_PREAMBLE):]
        # Strip trailing notes
        for marker in (f"\n\n{COMPACT_RECENT_MESSAGES_NOTE}", f"\n{COMPACT_DIRECT_RESUME_INSTRUCTION}"):
            if marker in summary:
                summary = summary[:summary.index(marker)]
        return summary.strip()
    return None


def _merge_summaries(existing: str, new_summary: str) -> str:
    """Merge previous and new compaction summaries."""
    lines = [
        "<summary>",
        "Conversation summary:",
        "- Previously compacted context:",
    ]
    for line in existing.splitlines():
        stripped = line.strip()
        if stripped and stripped not in ("<summary>", "</summary>", "Conversation summary:"):
            lines.append(f"  {stripped}")

    lines.append("- Newly compacted context:")
    for line in new_summary.splitlines():
        stripped = line.strip()
        if stripped and stripped not in ("<summary>", "</summary>", "Conversation summary:"):
            lines.append(f"  {stripped}")

    lines.append("</summary>")
    return "\n".join(lines)


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + "..."
