#!/usr/bin/env python3
"""
PreToolUse hook on Bash — commit-precondition co_lead gate (routing redesign LANE 1).

Once a Bash command is recognized as `git commit`, fail-CLOSED unless the
ai-room channel log shows a fresh codex_co_lead validation/diff PASS that echoes
the staged DIFF_DIGEST matching `git diff --cached` at commit time.

`git push` is NOT co_lead-gated; only force-push patterns are blocked here
(Claude/Gabe retain push authority after reviewed commit).

Fail-OPEN only: malformed stdin before command recognition, or non-commit/push
commands. Missing/unreadable channel log AFTER commit recognition → BLOCK.

Target-repo resolution principle: if the resolver cannot PROVE the single repo
git will commit to, it BLOCKS. Only three allowlisted command shapes resolve;
everything else is fail-closed with guidance to split into an allowlisted form.

SCOPE BOUNDARY: this PreToolUse static parser is not Turing-complete against
shell runtime expansion. We fail-close on detected dynamics, wrappers, and
multi-segment shapes in the cooperative threat model (CO_LEAD_GATE_OVERRIDE
covers intentional break-glass). Purely-runtime expansion without static
markers is a documented residual limitation, not an unbounded parse chase.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from typing import Any

DEFAULT_CHANNEL_LOG = "/home/gabe/.ai-room/channels/claw-code/messages.jsonl"

GIT_COMMIT_RE = re.compile(r"(?<![\w/.])git\b[^;\n|&]*\bcommit\b", re.IGNORECASE)
GIT_PUSH_RE = re.compile(r"(?<![\w/.])git\b[^;\n|&]*\bpush\b", re.IGNORECASE)
FORCE_PUSH_RE = re.compile(r"(?:--force(?:-with-lease)?|\s-f\b)")
PLUS_REFSPEC_RE = re.compile(r"(?<![\w])\+[\w./:-]+")

DIFF_DIGEST_RE = re.compile(r"(?im)^\s*DIFF_DIGEST\s*:\s*([0-9a-f]{64})\s*$")
COLEAD_GATE_OVERRIDE_RE = re.compile(
    r"(?im)^\s*CO_LEAD_GATE_OVERRIDE\s*:\s*(.+?)\s*$"
)
TASK_ID_RE = re.compile(r"\b(\d{13}-[0-9a-f]{6,8})\b")

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

MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS = 10
HEX64_RE = re.compile(r"[0-9a-f]{64}", re.IGNORECASE)
REPO_PATH_RE = re.compile(r"(?:/[^\s\"']+|(?:\./|\.\./)[^\s\"']+|\S+/[^\s\"']+)")
CD_PREFIX_RE = re.compile(
    r"^\s*cd\s+(\"[^\"]+\"|'[^']+'|\S+)\s+&&",
    re.IGNORECASE,
)
COLEAD_OVERRIDE_PASS_RE = re.compile(
    r"(?i)(?:co_lead.*\bPASS\b|\bgate-2\s+PASS\b.*co_lead|\bPASS\b.*co_lead)"
)
COMMAND_SUBSTITUTION_RE = re.compile(r"\$\(|`")
DYNAMIC_EXPANSION_RE = re.compile(r"[\$\*?\[`]|(?<!\$)\$\{")
INLINE_ENV_RE = re.compile(
    r"(?im)(?:^|[\s;|&])(?:[A-Za-z_][A-Za-z0-9_]*)=(?:[^\s]|$)|(?:^|[\s;|&])env\s+(?:-C\s+\S+\s+)?(?:[A-Za-z_][A-Za-z0-9_]*)="
)
GIT_INVOCATION_RE = re.compile(r"(?<![\w/.])git\b", re.IGNORECASE)
GIT_CI_RE = re.compile(r"(?<![\w/])git\s+ci\b", re.IGNORECASE)
SHELL_BLOCKED_PRECOMMIT_RE = re.compile(
    r"[;|]|(?<![&])&(?!&)|\(|\)|\{|\}|pushd|popd|function\s|if\s|then\b|for\s|while\s|"
    r"bash\s+-c|sh\s+-c|\balias\b"
)
COMMIT_HEREDOC_RE = re.compile(
    r"(<<-?\s*(['\"]?)([A-Za-z0-9_]+)\2\s*$)",
    re.MULTILINE,
)

FAIL_CLOSED_HINT = (
    "Use an allowlisted form: `git commit …` in hook cwd, "
    "`git -C <literal-path> commit …`, or `cd <literal-path> && git commit …` "
    "(optionally with commit-local `-C`). Split complex shell into separate gated commands."
)

GIT_WRAPPER_WORDS = frozenset(
    {
        "command",
        "exec",
        "env",
        "time",
        "builtin",
        "nohup",
        "nice",
        "xargs",
        "sudo",
    }
)

RISKY_DYNAMIC_EXPANSION_RE = re.compile(
    r"\$\(|`|(?<!\$)\$\{|\$IFS\b|\$\{IFS\}"
)
PLAIN_DOLLAR_VAR_RE = re.compile(r"\$[A-Za-z_][A-Za-z0-9_]*")


def fail_open(reason: str) -> int:
    if os.environ.get("COMMIT_PRECONDITION_GATE_DEBUG"):
        print(
            f"[commit_precondition_colead_gate] fail-open: {reason}",
            file=sys.stderr,
        )
    return 0


def _parse_ts(value: Any) -> float | None:
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


def _read_records(path: str) -> list[dict[str, Any]] | None:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            lines = fh.readlines()
    except FileNotFoundError:
        return None
    except OSError:
        return None
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            return None
        if isinstance(rec, dict):
            records.append(rec)
    return records


def _body(rec: dict[str, Any]) -> str:
    body = rec.get("body")
    return body if isinstance(body, str) else ""


def _unquote_shell_token(token: str) -> str:
    token = token.strip()
    if len(token) >= 2 and token[0] == token[-1] and token[0] in "\"'":
        return token[1:-1]
    return token


def _is_git_executable_token(token: str) -> bool:
    """True when token names the git executable (bare, quoted, path-qualified, escaped).

    Excludes `.git` directory path components (e.g. repo/.git, foo.git).
    """
    if token in (r"\git", "\\git"):
        return True
    bare = _unquote_shell_token(token).rstrip("/")
    if not bare:
        return False
    base = os.path.basename(bare)
    if base == ".git":
        return False
    return base.lower() == "git"


def _is_literal_git_command_word(token: str) -> bool:
    """True only for the canonical unquoted bare `git` command word (allowlist)."""
    return token == "git"


_GIT_GLOBAL_FLAGS_WITH_VALUE = frozenset(
    {
        "-C",
        "-c",
        "-F",
        "-m",
        "-e",
        "--git-dir",
        "--work-tree",
        "--namespace",
        "--config",
        "--config-env",
    }
)


def _split_shell_segments(command: str) -> list[str]:
    segments: list[str] = []
    current: list[str] = []
    i = 0
    n = len(command)
    single = False
    double = False
    escape = False
    while i < n:
        ch = command[i]
        if escape:
            current.append(ch)
            escape = False
            i += 1
            continue
        if ch == "\\" and not single:
            current.append(ch)
            escape = True
            i += 1
            continue
        if ch == "'" and not double:
            single = not single
            current.append(ch)
            i += 1
            continue
        if ch == '"' and not single:
            double = not double
            current.append(ch)
            i += 1
            continue
        if not single and not double:
            if command.startswith("&&", i):
                segments.append("".join(current))
                current = []
                i += 2
                continue
            if command.startswith("||", i):
                segments.append("".join(current))
                current = []
                i += 2
                continue
            if ch in ";|":
                segments.append("".join(current))
                current = []
                i += 1
                continue
            if ch == "\n":
                segments.append("".join(current))
                current = []
                i += 1
                continue
            if ch == "&":
                segments.append("".join(current))
                current = []
                i += 1
                continue
        current.append(ch)
        i += 1
    segments.append("".join(current))
    return [segment.strip() for segment in segments if segment.strip()]


def _strip_leading_wrappers(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens):
        word = tokens[index].lower()
        if word not in GIT_WRAPPER_WORDS:
            break
        index += 1
        while index < len(tokens):
            token = tokens[index]
            if word == "env" and "=" in token:
                index += 1
                continue
            if token.startswith("-"):
                index += 1
                if index < len(tokens):
                    nxt = tokens[index]
                    if nxt.startswith("-"):
                        continue
                    if _is_git_executable_token(nxt):
                        continue
                    index += 1
                continue
            break
    return tokens[index:]


def _strip_leading_env_assignments(tokens: list[str]) -> list[str]:
    index = 0
    while index < len(tokens) and "=" in tokens[index] and not tokens[index].startswith("-"):
        index += 1
    return tokens[index:]


def _segment_tokens(segment: str) -> list[str] | None:
    segment = segment.strip()
    if not segment:
        return None
    try:
        return shlex.split(segment, posix=True)
    except ValueError:
        return None


def _tokens_git_commit_bearing(tokens: list[str]) -> bool:
    """True when tokens contain a git executable invocation with a commit subcommand.

    Scans for git (any spelling) at ANY token position — not only as the segment
    command word. Non-canonical leading words (eval, timeout, …) still count as
    git-commit-bearing so the closed allowlist can fail-closed BLOCK them.
    """
    tokens = _strip_leading_env_assignments(tokens)
    for index, token in enumerate(tokens):
        if not _is_git_executable_token(token):
            continue
        pos = index + 1
        while pos < len(tokens):
            t = tokens[pos]
            if t.lower() == "commit":
                return True
            if t.startswith("-"):
                pos += 1
                if pos < len(tokens) and not tokens[pos].startswith("-"):
                    if t in _GIT_GLOBAL_FLAGS_WITH_VALUE or (
                        t.startswith("-C") and len(t) > 2
                    ):
                        pos += 1
                    elif t not in ("-C",) and not t.startswith("--"):
                        pos += 1
                continue
            break
    return False


def _segment_starts_with_wrapper(tokens: list[str]) -> bool:
    return bool(tokens and tokens[0].lower() in GIT_WRAPPER_WORDS)


def _normalized_command(command: str) -> str:
    return _strip_commit_message_heredoc(command)


def _has_plain_dollar_var_git_commit_risk(command: str) -> bool:
    """Plain $VAR dynamics when git+commit evidence exists (command-word positions)."""
    if not _command_mentions_git_and_commit(command):
        return False
    if not PLAIN_DOLLAR_VAR_RE.search(command):
        return False
    normalized = _normalized_command(command)
    for segment in _split_shell_segments(normalized):
        tokens = _segment_tokens(segment)
        if not tokens:
            continue
        if tokens[0] in {"grep", "echo"}:
            continue
        rest = _strip_leading_env_assignments(_strip_leading_wrappers(tokens))
        if not rest:
            continue
        if any(PLAIN_DOLLAR_VAR_RE.search(token) for token in rest):
            return True
    return False


def _has_risky_dynamic_expansion(command: str) -> bool:
    if RISKY_DYNAMIC_EXPANSION_RE.search(command) or COMMAND_SUBSTITUTION_RE.search(command):
        return True
    return _has_plain_dollar_var_git_commit_risk(command)


def _command_mentions_git_and_commit(command: str) -> bool:
    return bool(
        re.search(r"(?<![\w/.])git\b", command, re.IGNORECASE)
        and re.search(r"\bcommit\b", command, re.IGNORECASE)
    )


def _tokens_canonical_git_commit(tokens: list[str]) -> bool:
    """Bare literal `git` + only `-C` globals before `commit`."""
    if tokens != _strip_leading_wrappers(tokens):
        return False
    if not tokens or not _is_literal_git_command_word(tokens[0]):
        return False
    index = 1
    while index < len(tokens):
        token = tokens[index]
        if token.lower() == "commit":
            return True
        if token == "-C":
            if index + 1 >= len(tokens):
                return False
            if not _literal_path_token(tokens[index + 1]):
                return False
            index += 2
            continue
        if token.startswith("-C") and len(token) > 2:
            if not _literal_path_token(token[2:]):
                return False
            index += 1
            continue
        return False
    return False


def _segment_git_commit_bearing(segment: str) -> bool:
    """True when a segment bears git+commit evidence (token-level or raw text).

    grep/echo segments are excluded here — they are handled by pass-through.
    Raw-text evidence catches string-exec hiding (bash -c, env -S, eval "…").
    """
    tokens = _segment_tokens(segment)
    if tokens and tokens[0] in {"grep", "echo"}:
        return False
    if tokens and _tokens_git_commit_bearing(tokens):
        return True
    return _command_mentions_git_and_commit(segment)


def _segment_has_wrapper_git_commit(segment: str) -> bool:
    """True when a segment bears git+commit but does NOT reduce to canonical bare git."""
    if not _segment_git_commit_bearing(segment):
        return False
    return not _segment_is_canonical_git_commit(segment)


def _segment_wrapper_hides_git_commit(segment: str) -> bool:
    return _segment_has_wrapper_git_commit(segment)


def _is_grep_echo_pass_through(command: str) -> bool:
    """Single-segment grep/echo with git+commit only inside quoted args."""
    if _has_risky_dynamic_expansion(command):
        return False
    if INLINE_ENV_RE.search(command):
        return False
    segments = _split_shell_segments(command)
    if len(segments) != 1:
        return False
    tokens = _segment_tokens(segments[0])
    if not tokens or tokens[0] not in {"grep", "echo"}:
        return False
    if not _command_mentions_git_and_commit(segments[0]):
        return False
    return not _segment_git_commit_bearing(segments[0])


def _raw_first_token(segment: str) -> str:
    match = re.match(r"^\s*(\S+)", segment.strip())
    return match.group(1) if match else ""


def _segment_is_canonical_git_commit(segment: str) -> bool:
    if _raw_first_token(segment) != "git":
        return False
    tokens = _segment_tokens(segment)
    return bool(tokens and _tokens_canonical_git_commit(tokens))


def _git_commit_bearing_segments(command: str) -> list[str]:
    normalized = _normalized_command(command)
    return [
        segment
        for segment in _split_shell_segments(normalized)
        if _segment_git_commit_bearing(segment)
    ]


def _commit_command_segments(command: str) -> tuple[str | None, list[str]]:
    """Return (normalized command, segments used for allowlist checks)."""
    normalized = _normalized_command(command)
    cd_match = CD_PREFIX_RE.match(normalized)
    if cd_match:
        if not _literal_path_token(cd_match.group(1)):
            return None, _split_shell_segments(normalized)
        tail = normalized[cd_match.end() :].strip()
        return normalized, _split_shell_segments(tail)
    return normalized, _split_shell_segments(normalized)


def _command_matches_closed_allowlist(command: str) -> bool:
    """EXHAUSTIVELY CLOSED allowlist — only three canonical git-commit forms pass.

  Any segment bearing git+commit evidence must reduce to canonical bare
  `git commit` (token0=git). Leading wrappers (eval, timeout, command, …)
  fail-closed because they do not reduce to token0=git.
    """
    normalized = _normalized_command(command)
    if INLINE_ENV_RE.search(normalized):
        return False
    if _has_risky_dynamic_expansion(normalized):
        return False
    if SHELL_BLOCKED_PRECOMMIT_RE.search(normalized):
        return False

    _, segments = _commit_command_segments(command)
    bearing = [segment for segment in segments if _segment_git_commit_bearing(segment)]
    if len(bearing) != 1:
        return False
    if len(segments) != 1:
        return False
    return _segment_is_canonical_git_commit(bearing[0])


def _tokens_have_commit_subcommand(tokens: list[str]) -> bool:
    return _tokens_git_commit_bearing(tokens)


def _segment_has_git_commit(segment: str) -> bool:
    return _segment_git_commit_bearing(segment)


def _segment_has_git_ci(segment: str) -> bool:
    tokens = _segment_tokens(segment)
    if not tokens:
        return False
    tokens = _strip_leading_wrappers(tokens)
    return (
        len(tokens) >= 2
        and _is_git_executable_token(tokens[0])
        and tokens[1].lower() == "ci"
    )


def _segment_is_allowlisted_git_commit(segment: str) -> bool:
    return _segment_is_canonical_git_commit(segment)


def _allowlisted_commit_invocation_segment(command: str) -> str | None:
    """Return the segment matching the closed allowlist git-commit shape, if any."""
    if not _command_matches_closed_allowlist(command):
        return None
    _, segments = _commit_command_segments(_strip_commit_message_heredoc(command))
    for segment in segments:
        if _segment_is_canonical_git_commit(segment):
            return segment
    return None


def _inside_quotes(prefix: str) -> bool:
    single = False
    double = False
    escape = False
    for ch in prefix:
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == "'" and not double:
            single = not single
        elif ch == '"' and not single:
            double = not double
    return single or double


def _tokens_are_git_commit_invocation(tokens: list[str]) -> bool:
    return _tokens_have_commit_subcommand(tokens)


def _is_actual_git_commit_invocation(command: str) -> bool:
    """True when any shell segment runs git (any form) with a commit subcommand."""
    return bool(_git_commit_bearing_segments(command))


def _wrapper_hides_git_commit(command: str) -> bool:
    if not re.search(r"\bcommit\b", command, re.IGNORECASE):
        return False
    normalized = _normalized_command(command)
    if any(_segment_wrapper_hides_git_commit(segment) for segment in _split_shell_segments(normalized)):
        return True
    if SHELL_BLOCKED_PRECOMMIT_RE.search(command):
        return bool(
            re.search(r"(?<![\w/.])git\b", command, re.IGNORECASE)
            or re.search(r"(?:/|\./|\.\./)[^\s\"']*git\b", command, re.IGNORECASE)
            or re.search(r"""['"]git['"]""", command, re.IGNORECASE)
            or re.search(r"""\\git\b""", command)
        )
    return False


def _is_git_commit_gated_command(command: str) -> bool:
    """Whether this Bash tool call should enter the commit gate (not pass-through).

    Gate-eligible when any segment bears git+commit evidence (git executable
    token + commit subcommand at any position). Pass-through ONLY for
    grep/echo single-segment quoted-arg substrings. Non-canonical bearing
    segments (eval, timeout, wrappers, …) gate → fail-closed BLOCK.
    """
    normalized = _normalized_command(command)
    if _is_grep_echo_pass_through(normalized):
        return False
    if _git_commit_bearing_segments(command):
        return True
    if any(
        _segment_has_git_ci(segment)
        for segment in _split_shell_segments(normalized)
    ):
        return True
    if _has_risky_dynamic_expansion(normalized) and _command_mentions_git_and_commit(
        normalized
    ):
        return True
    return False


def _strip_commit_message_heredoc(command: str) -> str:
    """Allow claude's `git commit -F - <<'EOF' … EOF` message form only."""
    commit_match = GIT_COMMIT_RE.search(command)
    if not commit_match:
        return command
    tail = command[commit_match.end() :]
    heredoc_match = COMMIT_HEREDOC_RE.search(tail)
    if not heredoc_match:
        return command
    opener = heredoc_match.group(1)
    delimiter = heredoc_match.group(3)
    opener_pos = command.find(opener, commit_match.end())
    if opener_pos < 0:
        return command
    rest = command[opener_pos + len(opener) :]
    close_pattern = re.compile(rf"^\s*{re.escape(delimiter)}\s*$", re.MULTILINE)
    close_match = close_pattern.search(rest)
    if not close_match:
        return command
    end = opener_pos + len(opener) + close_match.end()
    stripped = command[:end]
    if "\n" in stripped:
        return re.sub(r"\s+", " ", stripped).strip()
    return stripped


def _git_commit_invocation_prefix(command: str) -> str:
    allowlisted = _allowlisted_commit_invocation_segment(command)
    if allowlisted:
        return allowlisted
    commit_match = GIT_COMMIT_RE.search(command)
    if not commit_match:
        return command
    return command[commit_match.start() : commit_match.end()]


def _pre_commit_prefix(command: str) -> str:
    commit_match = GIT_COMMIT_RE.search(command)
    if not commit_match:
        return ""
    return command[: commit_match.start()]


def _literal_path_token(token: str) -> bool:
    if not token:
        return False
    if DYNAMIC_EXPANSION_RE.search(token):
        return False
    return True


def _shell_shape_blocked(command: str) -> bool:
    return not _command_matches_closed_allowlist(command)


def _tokenize_commit_invocation(command: str) -> list[str] | None:
    invocation = _git_commit_invocation_prefix(command)
    try:
        tokens = shlex.split(invocation, posix=True)
    except ValueError:
        return None
    if not tokens or tokens[0].lower() != "git":
        return None
    return tokens


def _extract_commit_c_paths(tokens: list[str]) -> list[str] | None:
    if not tokens or tokens[0].lower() != "git":
        return None
    index = 1
    c_paths: list[str] = []
    while index < len(tokens):
        token = tokens[index]
        if token.lower() == "commit":
            for path in c_paths:
                if not _literal_path_token(path):
                    return None
            return c_paths
        if token == "-C":
            if index + 1 >= len(tokens):
                return None
            c_paths.append(tokens[index + 1])
            index += 2
            continue
        if token.startswith("-C") and len(token) > 2:
            c_paths.append(token[2:])
            index += 1
            continue
        return None
    return None


def _apply_git_c_path(base_dir: str, c_path: str) -> str:
    expanded = os.path.expanduser(_unquote_shell_token(c_path))
    if os.path.isabs(expanded):
        return os.path.abspath(expanded)
    return os.path.abspath(os.path.join(base_dir, expanded))


def _parse_cd_base(command: str, cwd: str) -> tuple[str, bool]:
    cd_match = CD_PREFIX_RE.match(command)
    if cd_match:
        return os.path.abspath(_unquote_shell_token(cd_match.group(1))), True
    return os.path.abspath(cwd), False


def resolve_target_repo(command: str, cwd: str | None = None) -> tuple[str | None, bool]:
    """Return (repo_dir, explicit_target). repo_dir=None means fail-closed.

    EXHAUSTIVELY CLOSED allowlist — only these pass:
    1. `git commit …` in hook cwd
    2. `git -C <literal> commit …` (cumulative literal -C only)
    3. `cd <literal> && git commit …` (+ optional commit-local literal -C)

    Commit-message heredoc (`git commit -F - <<'EOF' … EOF`) is allowed on top.
    Any other git-commit-bearing shape is gated but NOT resolvable (fail-closed).
    """
    normalized = _strip_commit_message_heredoc(command)
    if not _command_matches_closed_allowlist(normalized):
        return None, True

    allowlisted_segment = _allowlisted_commit_invocation_segment(normalized)
    if allowlisted_segment is None:
        return None, True

    base_dir, explicit = _parse_cd_base(normalized, cwd or os.getcwd())
    try:
        tokens = shlex.split(allowlisted_segment, posix=True)
    except ValueError:
        return None, True
    c_paths = _extract_commit_c_paths(tokens)
    if c_paths is None:
        return None, True

    repo_dir = base_dir
    if c_paths:
        explicit = True
        for c_path in c_paths:
            repo_dir = _apply_git_c_path(repo_dir, c_path)
    elif explicit:
        pass
    return repo_dir, explicit


def _override_reason_valid(reason: str) -> tuple[bool, str]:
    if len(reason) < MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS:
        return (
            False,
            f"too trivial ({len(reason)} chars; need "
            f">={MIN_COLEAD_GATE_OVERRIDE_REASON_CHARS})",
        )
    if not REPO_PATH_RE.search(reason):
        return False, "missing target-repo path"
    if not HEX64_RE.search(reason):
        return False, "missing 64-hex DIFF_DIGEST"
    if not TASK_ID_RE.search(reason):
        return False, "missing co_lead PASS msg id"
    if not COLEAD_OVERRIDE_PASS_RE.search(reason):
        return False, "missing co_lead PASS marker"
    return True, "auditable override"


def _staged_digest(command: str) -> str | None:
    repo_dir, explicit_target = resolve_target_repo(command)
    if repo_dir is None:
        return None
    if explicit_target and not os.path.isdir(repo_dir):
        return None
    try:
        proc = subprocess.run(
            ["git", "-C", repo_dir, "diff", "--cached"],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:
        return None
    if proc.returncode != 0:
        return None
    return hashlib.sha256(proc.stdout.encode("utf-8")).hexdigest()


def _extract_digest(body: str) -> str | None:
    m = DIFF_DIGEST_RE.search(body)
    return m.group(1).lower() if m else None


def _is_worker_receipt(rec: dict[str, Any]) -> bool:
    frm = str(rec.get("from", ""))
    if frm in {"claude", "codex_co_lead", "gabe", "watchdog"}:
        return False
    kind = str(rec.get("kind", ""))
    body = _body(rec)
    if kind == "validation_receipt":
        return True
    return any(
        marker in body
        for marker in (
            "VALIDATION RECEIPT",
            "VALIDATION_RECEIPT",
            "IMPLEMENTATION RECEIPT",
            "TERMINAL RECEIPT",
        )
    )


def _is_claude_freeze(rec: dict[str, Any]) -> bool:
    if rec.get("from") != "claude":
        return False
    body = _body(rec)
    if _extract_digest(body) is None:
        return False
    return any(
        tok in body
        for tok in (
            "gate-1 freeze",
            "FREEZE LOCKED",
            "frozen handoff",
            "validation/diff handoff",
            "review_request",
        )
    ) or str(rec.get("kind", "")) in {"review_request", "msg"}


def _colead_verdict(body: str) -> str:
    for pat in COLEAD_BLOCK_MARKERS:
        if pat.search(body):
            return "block"
    for pat in COLEAD_DEFERRAL_MARKERS:
        if pat.search(body):
            return "unknown"
    for pat in COLEAD_PASS_MARKERS:
        if pat.search(body):
            return "pass"
    return "unknown"


def _has_force_plus_refspec(command: str) -> bool:
    if not GIT_PUSH_RE.search(command):
        return False
    return bool(PLUS_REFSPEC_RE.search(command))


def _same_thread(rec: dict[str, Any], anchor_ids: set[str]) -> bool:
    if not anchor_ids:
        return True
    rid = str(rec.get("id", ""))
    reply_to = str(rec.get("reply_to", ""))
    body = _body(rec)
    if rid in anchor_ids or reply_to in anchor_ids:
        return True
    return any(aid in body for aid in anchor_ids)


def _find_fresh_colead_pass(
    records: list[dict[str, Any]],
    staged_digest: str,
) -> tuple[bool, str]:
    freeze_ts: float | None = None
    freeze_ids: set[str] = set()
    task_ids: set[str] = set()

    for rec in records:
        if not _is_claude_freeze(rec):
            continue
        digest = _extract_digest(_body(rec))
        if digest != staged_digest:
            continue
        ts = _parse_ts(rec.get("ts"))
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
            m = TASK_ID_RE.search(_body(rec))
            if m:
                task_ids.add(m.group(1))

    if freeze_ts is None:
        return False, "no claude freeze/handoff carrying matching DIFF_DIGEST"

    anchor_ids = set(freeze_ids)
    worker_ts: float | None = None
    for rec in records:
        if not _is_worker_receipt(rec):
            continue
        body = _body(rec)
        if not (
            _same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body for tid in task_ids))
        ):
            continue
        ts = _parse_ts(rec.get("ts"))
        if ts is not None and (worker_ts is None or ts > worker_ts):
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
        body = _body(rec)
        digest = _extract_digest(body)
        if digest != staged_digest:
            continue
        verdict = _colead_verdict(body)
        if verdict != "pass":
            continue
        ts = _parse_ts(rec.get("ts"))
        if ts is None or ts <= freeze_ts:
            continue
        if worker_ts is not None and ts <= worker_ts:
            continue
        if not (
            _same_thread(rec, anchor_ids)
            or (task_ids and any(tid in body for tid in task_ids))
        ):
            continue
        if best_pass_ts is None or ts >= best_pass_ts:
            best_pass_ts = ts

    if best_pass_ts is None:
        return False, "no codex_co_lead validation/diff PASS echoing staged DIFF_DIGEST on-thread after freeze"
    return True, "fresh co_lead PASS matches staged DIFF_DIGEST"


def _bash_command(payload: dict[str, Any]) -> str:
    tool_input = payload.get("tool_input") or {}
    if isinstance(tool_input, dict):
        cmd = tool_input.get("command")
        if isinstance(cmd, str):
            return cmd
    cmd = payload.get("command")
    return cmd if isinstance(cmd, str) else ""


def main() -> int:
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            return fail_open("empty stdin")
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        return fail_open(f"json decode failed: {exc}")
    except Exception as exc:  # noqa: BLE001
        return fail_open(f"stdin read failed: {exc}")

    command = _bash_command(payload)
    if not command.strip():
        return fail_open("empty command")

    is_commit = _is_git_commit_gated_command(command)
    is_push = bool(GIT_PUSH_RE.search(command))
    if not is_commit and not is_push:
        return 0

    if any(_segment_has_git_ci(segment) for segment in _split_shell_segments(command)) and not _is_actual_git_commit_invocation(command):
        msg = (
            "BLOCKED [commit_precondition_colead_gate] git alias/wrapper "
            "(`git ci`) is not allowlisted — use literal `git commit`."
        )
        print(msg, file=sys.stderr)
        return 2

    if is_push and (
        FORCE_PUSH_RE.search(command) or _has_force_plus_refspec(command)
    ):
        msg = (
            "BLOCKED [commit_precondition_colead_gate] force push is forbidden "
            "via this hook (use non-force fast-forward after reviewed commit)."
        )
        print(msg, file=sys.stderr)
        return 2

    if COLEAD_GATE_OVERRIDE_RE.search(command):
        if not is_commit:
            msg = (
                "BLOCKED [commit_precondition_colead_gate] CO_LEAD_GATE_OVERRIDE "
                "is valid only for git commit commands."
            )
            print(msg, file=sys.stderr)
            return 2
        m = COLEAD_GATE_OVERRIDE_RE.search(command)
        reason = (m.group(1).strip() if m else "")
        ok, detail = _override_reason_valid(reason)
        if not ok:
            msg = (
                "BLOCKED [commit_precondition_colead_gate] CO_LEAD_GATE_OVERRIDE "
                f"reason rejected ({detail}) — require target-repo path + 64-hex "
                "DIFF_DIGEST + co_lead PASS msg id."
            )
            print(msg, file=sys.stderr)
            return 2
        return 0

    if is_push and not is_commit:
        return 0

    log_path = (
        os.environ.get("AI_ROOM_CHANNEL_LOG")
        or os.environ.get("AI_ROOM_CHANNEL_LOG_PATH")
        or DEFAULT_CHANNEL_LOG
    )
    records = _read_records(log_path)
    if records is None:
        msg = (
            "BLOCKED [commit_precondition_colead_gate] git commit recognized but "
            f"channel log missing/unreadable at {log_path!r} — no proven co_lead "
            "validation/diff gate."
        )
        print(msg, file=sys.stderr)
        return 2

    staged_digest = _staged_digest(command)
    if staged_digest is None:
        repo_dir, explicit_target = resolve_target_repo(command)
        if repo_dir is None:
            detail = (
                "commit command not in allowlisted target-repo shape "
                f"({FAIL_CLOSED_HINT})"
            )
        elif explicit_target:
            detail = f"target repo {repo_dir!r} invalid or git diff --cached failed"
        else:
            detail = "git diff --cached failed"
        msg = (
            "BLOCKED [commit_precondition_colead_gate] git commit recognized but "
            f"could not compute staged diff digest ({detail})."
        )
        print(msg, file=sys.stderr)
        return 2

    ok, reason = _find_fresh_colead_pass(records, staged_digest)
    if ok:
        return 0

    msg_lines = [
        "BLOCKED [commit_precondition_colead_gate] git commit without fresh co_lead validation/diff gate:",
        f"  staged DIFF_DIGEST: {staged_digest}",
        f"  reason: {reason}",
        "",
        "Required chain: worker receipt → claude gate-1 freeze/handoff with DIFF_DIGEST",
        "→ codex_co_lead gate-2 PASS echoing the same DIFF_DIGEST on-thread.",
        "git push is not co_lead-gated; only reviewed commits pass this hook.",
    ]
    print("\n".join(msg_lines), file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
