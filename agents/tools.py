"""Tools that agents can invoke: bash, read, write, edit, grep, glob, web, todos, etc."""

import json
import os
import subprocess
import glob as globmod
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from html.parser import HTMLParser

from agents.permissions import check_permission, PermissionMode


# Per-subagent_type tool allowlists for the Agent tool. Mirrors the Rust port's
# `allowed_tools_for_subagent` table (rust/crates/tools/src/lib.rs:3187) adapted
# to Zenith's tool surface. The second element of each tuple is the permission
# mode the sub-agent runs in.
#
# Notes on what's intentionally absent:
# - `Agent` — sub-agents cannot spawn their own sub-agents (no recursion).
# - `AskUserQuestion` — sub-agents are designed to be autonomous; the parent
#   asks user questions on their behalf if needed.
# - `TodoWrite` — multi-step planning is a parent-level activity; sub-agents
#   are focused on a single delegated task.
# - `Skill` / `ToolSearch` / `StructuredOutput` (Rust port has these on Explore) —
#   they don't translate to Zenith's scale (no skill registry, only ~12 tools
#   total so search is pointless, no JSON-schema response_format wrapper).
ALLOWED_TOOLS_BY_SUBAGENT: dict[str, tuple[set[str], PermissionMode]] = {
    "explore":         ({"read_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch"}, PermissionMode.READ_ONLY),
    "plan":            ({"read_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch"}, PermissionMode.READ_ONLY),
    "verification":    ({"bash", "read_file", "grep", "list_files", "list_directory", "Sleep"}, PermissionMode.WORKSPACE_WRITE),
    "general-purpose": ({"bash", "read_file", "write_file", "edit_file", "grep", "list_files", "list_directory", "WebFetch", "WebSearch", "Sleep", "MultiEdit"}, PermissionMode.WORKSPACE_WRITE),
}


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Execute a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "The shell command to execute",
                    }
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file and return its contents with line numbers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    },
                    "offset": {
                        "type": "integer",
                        "description": "Start line (0-based). Default: 0",
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Max lines to return. Default: 500",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates the file if it doesn't exist.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to write",
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": (
                "Search for a regex pattern in files. Backed by ripgrep when available (fast, "
                "respects .gitignore), with a Python fallback. Returns matching lines as "
                "`path:line:content`. Use `type` for filetype filtering ('py', 'rs', 'ts', etc.), "
                "`context` for surrounding lines, `word_match` for whole-word search."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Regex pattern to search for",
                    },
                    "path": {
                        "type": "string",
                        "description": "Directory or file to search in (default: current directory)",
                    },
                    "glob_pattern": {
                        "type": "string",
                        "description": "File glob pattern to include (e.g. '*.py', '**/*.rs')",
                    },
                    "type": {
                        "type": "string",
                        "description": "ripgrep filetype shortcut (e.g. 'py', 'rs', 'ts', 'md'). Faster than glob for known types.",
                    },
                    "context": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of context lines to show before AND after each match (default 0)",
                    },
                    "word_match": {
                        "type": "boolean",
                        "description": "Match whole words only (rg -w)",
                    },
                    "case_insensitive": {
                        "type": "boolean",
                        "description": "Case-insensitive search (default: false)",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {
                        "type": "string",
                        "description": "Glob pattern (e.g. '**/*.py', 'src/*.rs')",
                    }
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_directory",
            "description": (
                "List the contents of a single directory (one level deep). "
                "Returns directories first then files, both alphabetized, with a "
                "trailing '/' on directory names. Use this instead of constructing "
                "globs when you just want to see what's in a folder. For recursive "
                "or pattern-based search, use list_files or grep."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current working directory)",
                    },
                    "show_hidden": {
                        "type": "boolean",
                        "description": "Include dotfiles/dotdirs (default: false)",
                    },
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "edit_file",
            "description": "Replace a specific string in a file. Use for surgical edits instead of rewriting the whole file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "old_string": {
                        "type": "string",
                        "description": "The exact string to find and replace",
                    },
                    "new_string": {
                        "type": "string",
                        "description": "The replacement string",
                    },
                    "replace_all": {
                        "type": "boolean",
                        "description": "Replace all occurrences (default: false, requires unique match)",
                    },
                },
                "required": ["path", "old_string", "new_string"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Agent",
            "description": (
                "Launch an isolated sub-agent to handle a delegated task. The sub-agent "
                "runs with a fresh conversation history and a filtered tool allowlist "
                "based on subagent_type. Use 'explore' or 'plan' for read-only investigation, "
                "'verification' for read+bash (tests/lint), or 'general-purpose' (default) "
                "for full file editing. Sub-agents cannot spawn their own sub-agents. "
                "Returns the sub-agent's final text as a string. "
                "Only `prompt` is required — `description` is auto-derived from the prompt if omitted."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The full task instructions for the sub-agent",
                    },
                    "description": {
                        "type": "string",
                        "description": "Optional short label for the task (auto-derived from prompt if omitted)",
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": ["explore", "plan", "verification", "general-purpose"],
                        "description": "Subagent type — controls tool allowlist and permission mode (default: general-purpose)",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentCreate",
            "description": (
                "Spawn a PERSISTENT sub-agent that you can re-engage on later turns. "
                "Unlike the one-shot `Agent` tool (which discards the sub-agent after one "
                "response), AgentCreate registers the sub-agent under an ID you can address "
                "with AgentMessage. Use this for cross-validation (spawn two reviewers, "
                "have them critique each other) or iterative dialogue (refine a sub-agent's "
                "output with follow-up questions). Returns the agent_id and the first response."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The initial task or instructions for the sub-agent",
                    },
                    "name": {
                        "type": "string",
                        "description": "Optional human-readable ID (e.g. 'reviewer_a'). Lowercase alphanum + _-, max 40 chars. Auto-generated if omitted.",
                    },
                    "subagent_type": {
                        "type": "string",
                        "enum": ["explore", "plan", "verification", "general-purpose"],
                        "description": "Subagent type — controls tool allowlist and permission mode (default: general-purpose)",
                    },
                    "role": {
                        "type": "string",
                        "description": "Optional one-line role description, used in the sub-agent's system prompt (e.g. 'security reviewer skeptical of injection claims')",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentMessage",
            "description": (
                "Send another turn to a persistent sub-agent created with AgentCreate. "
                "The `content` is appended to the target agent's history as a user turn, "
                "the agent runs one chat() loop, and you receive its next response. Use "
                "this to ask follow-up questions, deliver feedback from another agent, or "
                "redirect the agent's investigation. Errors if no agent has the given ID."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {
                        "type": "string",
                        "description": "The agent_id returned by a prior AgentCreate call",
                    },
                    "content": {
                        "type": "string",
                        "description": "The message to send to the sub-agent (treated as a user turn)",
                    },
                },
                "required": ["agent_id", "content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentGet",
            "description": (
                "Inspect a persistent sub-agent's current state without prompting it. "
                "Returns the agent's role, model, history length, todo list, last assistant "
                "message, and timestamps. Useful for checking on a sub-agent before deciding "
                "whether to re-engage it."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentList",
            "description": (
                "List all currently registered persistent sub-agents. Returns each agent's "
                "ID, role, history length, and age. Empty list if no agents are registered."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AgentTerminate",
            "description": (
                "Remove a persistent sub-agent from the registry. After this, AgentMessage "
                "/ AgentGet on the same ID will error. Use to free up the agent when its "
                "task is complete and no further dialogue is needed."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "agent_id": {"type": "string"},
                },
                "required": ["agent_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "Sleep",
            "description": "Wait for a specified duration without holding a shell process. Capped at 60000ms (60 seconds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "duration_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Number of milliseconds to wait (max 60000)",
                    },
                },
                "required": ["duration_ms"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WebFetch",
            "description": (
                "Fetch a URL over HTTP(S), strip HTML to readable text, and return the content. "
                "Capped at 8000 characters output. Only http/https schemes allowed. "
                "The `prompt` field is captured in the response header for your own context — "
                "the tool returns the page text, then YOU answer the prompt on the next turn."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "Full http:// or https:// URL to fetch",
                    },
                    "prompt": {
                        "type": "string",
                        "description": "What you want to learn from the page (used as context, not answered by the tool)",
                    },
                },
                "required": ["url", "prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "WebSearch",
            "description": (
                "Search the web for current information via DuckDuckGo HTML and return cited results. "
                "Note: uses DuckDuckGo HTML scraping; may break if their page structure changes. "
                "Returns up to 10 results with title, URL, and snippet."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (must be at least 2 characters)",
                    },
                    "allowed_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domain suffixes to restrict results to (e.g. ['github.com', 'docs.python.org'])",
                    },
                    "blocked_domains": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of domain suffixes to exclude from results",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "AskUserQuestion",
            "description": (
                "Ask the user a question and wait for their response. "
                "Use sparingly — only for genuinely ambiguous decisions where you can't proceed without input. "
                "Provide `options` for multiple-choice questions; otherwise the user types a freeform answer."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to ask (one clear sentence)",
                    },
                    "options": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of choices the user can pick from (numbered 1..N) or type freeform",
                    },
                },
                "required": ["question"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoWrite",
            "description": (
                "Update the structured task list for the current session. REPLACES the entire list each call. "
                "Use to plan multi-step work and track progress. At most ONE todo may be `in_progress` at a time. "
                "Each todo needs `content` (imperative form, e.g. 'Add tests'), `activeForm` (gerund, e.g. 'Adding tests'), "
                "and `status` ∈ {pending, in_progress, completed}. Returns a rendered markdown view of the new state."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "todos": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "content": {
                                    "type": "string",
                                    "description": "Imperative form, e.g. 'Add tests for the parser'",
                                },
                                "activeForm": {
                                    "type": "string",
                                    "description": "Gerund form, e.g. 'Adding tests for the parser'",
                                },
                                "status": {
                                    "type": "string",
                                    "enum": ["pending", "in_progress", "completed"],
                                },
                            },
                            "required": ["content", "activeForm", "status"],
                        },
                    },
                },
                "required": ["todos"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "TodoRead",
            "description": (
                "Read the current structured task list for the session (set by TodoWrite). "
                "Returns the same markdown view TodoWrite produces. Useful at the start of a "
                "long task or after compaction to recover plan state."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "MultiEdit",
            "description": (
                "Apply multiple find-and-replace edits to a single file ATOMICALLY. Edits are "
                "applied sequentially in the order given (each edit sees the result of the prior). "
                "If ANY edit fails (old_string not found, ambiguous match without replace_all), "
                "NONE are written — the file is unchanged. Use for refactors that touch multiple "
                "spots in the same file in one tool round."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to edit",
                    },
                    "edits": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "old_string": {"type": "string"},
                                "new_string": {"type": "string"},
                                "replace_all": {
                                    "type": "boolean",
                                    "description": "Replace all occurrences (default false; requires unique match)",
                                },
                            },
                            "required": ["old_string", "new_string"],
                        },
                    },
                },
                "required": ["path", "edits"],
            },
        },
    },
]


def execute_tool(
    name: str,
    args: dict,
    confirm_fn=None,
    mode=None,
    *,
    parent_model: str | None = None,
    parent_backend: str | None = None,
    ask_user_fn=None,
    agent=None,
) -> str:
    """Execute a tool with permission checking.

    The Agent tool needs `parent_model` / `parent_backend` so the spawned
    sub-agent inherits the parent's loaded model and backend (so /swap and
    /backend choices propagate). AskUserQuestion needs `ask_user_fn` (a
    callback wired by the harness onto the Agent instance) to prompt the
    user via input(). TodoWrite needs `agent` so it can mutate
    `agent.todos` (the in-memory per-conversation task list). All other
    tools ignore these kwargs.
    """
    mode = mode or PermissionMode.WORKSPACE_WRITE
    allowed, reason = check_permission(name, args, mode)
    if not allowed:
        if "Needs confirmation" in reason and confirm_fn:
            if not confirm_fn(name, args, reason):
                return f"Denied by user: {reason}"
        else:
            return reason

    try:
        if name == "bash":
            return _run_bash(args["command"])
        elif name == "read_file":
            return _read_file(args)
        elif name == "write_file":
            return _write_file(args["path"], args["content"])
        elif name == "edit_file":
            return _edit_file(
                args["path"],
                args["old_string"],
                args["new_string"],
                args.get("replace_all", False),
            )
        elif name == "grep":
            return _grep(
                args["pattern"],
                args.get("path", "."),
                args.get("glob_pattern"),
                file_type=args.get("type"),
                context=args.get("context", 0),
                word_match=bool(args.get("word_match", False)),
                case_insensitive=bool(args.get("case_insensitive", False)),
            )
        elif name == "list_files":
            return _list_files(args["pattern"])
        elif name == "list_directory":
            return _list_directory(
                args.get("path", "."),
                show_hidden=bool(args.get("show_hidden", False)),
            )
        elif name == "Agent":
            return _execute_agent_tool(
                args,
                parent_model=parent_model,
                parent_backend=parent_backend,
                parent_confirm_fn=confirm_fn,
            )
        elif name == "AgentCreate":
            return _run_agent_create(
                args,
                parent_model=parent_model,
                parent_backend=parent_backend,
                parent_confirm_fn=confirm_fn,
            )
        elif name == "AgentMessage":
            return _run_agent_message(args)
        elif name == "AgentGet":
            return _run_agent_get(args)
        elif name == "AgentList":
            return _run_agent_list(args)
        elif name == "AgentTerminate":
            return _run_agent_terminate(args)
        elif name == "Sleep":
            return _run_sleep(args)
        elif name == "WebFetch":
            return _run_web_fetch(args)
        elif name == "WebSearch":
            return _run_web_search(args)
        elif name == "AskUserQuestion":
            return _run_ask_user_question(args, ask_user_fn)
        elif name == "TodoWrite":
            return _run_todo_write(args, agent)
        elif name == "TodoRead":
            return _run_todo_read(agent)
        elif name == "MultiEdit":
            return _run_multi_edit(args["path"], args["edits"])
        else:
            return f"Error: unknown tool '{name}'"
    except Exception as e:
        return f"Error: {e}"


def _run_bash(command: str) -> str:
    """Execute a shell command with timeout."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
        )
        output = result.stdout
        if result.stderr:
            output += f"\nSTDERR: {result.stderr}"
        if result.returncode != 0:
            output += f"\n(exit code {result.returncode})"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out (30s)"


def _read_file(args: dict) -> str:
    """Read file contents with optional windowing and binary detection."""
    path = os.path.expanduser(args["path"])
    offset = args.get("offset", 0)
    limit = args.get("limit", 500)

    # Binary detection: check first 8KB for NUL bytes
    with open(path, "rb") as f:
        sample = f.read(8192)
        if b"\x00" in sample:
            return f"Error: binary file detected ({path})"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    total = len(lines)
    offset = min(offset, total)
    selected = lines[offset:offset + limit]
    if not selected:
        return f"({total} lines total, no lines in range)"
    numbered = [f"{i + offset + 1:4d} | {line.rstrip()}" for i, line in enumerate(selected)]
    end = min(offset + limit, total)
    header = f"({total} lines total, showing {offset+1}-{end})"
    return header + "\n" + "\n".join(numbered)


def _write_file(path: str, content: str) -> str:
    """Write content to file."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


_GREP_OUTPUT_LINE_CAP = 200
_GREP_OUTPUT_CHAR_CAP = 16_000
_RG_BIN_CACHE: str | None | bool = False  # False = unprobed, None = not found, str = path


def _find_rg() -> str | None:
    """Locate ripgrep on PATH. Cached after first call."""
    global _RG_BIN_CACHE
    if _RG_BIN_CACHE is not False:
        return _RG_BIN_CACHE  # type: ignore[return-value]
    from shutil import which
    _RG_BIN_CACHE = which("rg")
    return _RG_BIN_CACHE


def _grep(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    *,
    file_type: str | None = None,
    context: int = 0,
    word_match: bool = False,
    case_insensitive: bool = False,
) -> str:
    """Search for a regex pattern. Uses ripgrep when available, Python regex fallback otherwise."""
    rg = _find_rg()
    if rg is not None:
        return _grep_with_rg(
            rg, pattern, path, file_glob,
            file_type=file_type,
            context=context,
            word_match=word_match,
            case_insensitive=case_insensitive,
        )
    return _grep_with_python(pattern, path, file_glob, case_insensitive=case_insensitive)


def _grep_with_rg(
    rg_bin: str,
    pattern: str,
    path: str,
    file_glob: str | None,
    *,
    file_type: str | None,
    context: int,
    word_match: bool,
    case_insensitive: bool,
) -> str:
    """Run ripgrep as a subprocess and format its output."""
    expanded_path = os.path.expanduser(path)
    cmd: list[str] = [rg_bin, "--line-number", "--no-heading", "--color", "never"]
    if case_insensitive:
        cmd.append("--ignore-case")
    if word_match:
        cmd.append("--word-regexp")
    if context and context > 0:
        cmd.extend(["--context", str(int(context))])
    if file_type:
        cmd.extend(["--type", str(file_type)])
    if file_glob:
        cmd.extend(["--glob", str(file_glob)])
    cmd.extend(["--", pattern, expanded_path])

    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=15, cwd=os.getcwd()
        )
    except subprocess.TimeoutExpired:
        return "Error: ripgrep timed out (15s)"
    except FileNotFoundError:
        return _grep_with_python(pattern, path, file_glob, case_insensitive=case_insensitive)

    # rg exit codes: 0 = matches, 1 = no matches, 2 = error
    if proc.returncode == 1:
        return "No matches found."
    if proc.returncode != 0:
        err = (proc.stderr or "").strip().splitlines()
        first_err = err[0] if err else f"rg exit {proc.returncode}"
        return f"Error: ripgrep: {first_err}"

    output = proc.stdout
    lines = output.splitlines()
    truncated_lines = len(lines) > _GREP_OUTPUT_LINE_CAP
    if truncated_lines:
        lines = lines[:_GREP_OUTPUT_LINE_CAP]
    result = "\n".join(lines)
    truncated_chars = len(result) > _GREP_OUTPUT_CHAR_CAP
    if truncated_chars:
        result = result[:_GREP_OUTPUT_CHAR_CAP] + "\n... (output truncated)"
    elif truncated_lines:
        result = result + f"\n... (truncated at {_GREP_OUTPUT_LINE_CAP} lines)"
    return result if result else "No matches found."


def _grep_with_python(
    pattern: str,
    path: str = ".",
    file_glob: str | None = None,
    *,
    case_insensitive: bool = False,
) -> str:
    """Pure-Python fallback for environments without ripgrep."""
    path = os.path.expanduser(path)
    results: list[str] = []
    flags = re.IGNORECASE if case_insensitive else 0
    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return f"Error: invalid regex {pattern!r}: {e}"

    if os.path.isfile(path):
        files = [path]
    else:
        glob_pat = file_glob or "**/*"
        files = globmod.glob(os.path.join(path, glob_pat), recursive=True)
        files = [f for f in files if os.path.isfile(f)]

    for filepath in files[:100]:
        try:
            with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                for i, line in enumerate(f, 1):
                    if regex.search(line):
                        results.append(f"{filepath}:{i}:{line.rstrip()}")
                        if len(results) >= _GREP_OUTPUT_LINE_CAP:
                            results.append(f"... (truncated at {_GREP_OUTPUT_LINE_CAP} matches)")
                            return "\n".join(results)
        except (OSError, UnicodeDecodeError):
            continue

    return "\n".join(results) if results else "No matches found."


def _edit_file(path: str, old_string: str, new_string: str, replace_all: bool = False) -> str:
    """Replace a string in a file."""
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if old_string == new_string:
        return "Error: old_string and new_string must differ"

    if old_string not in content:
        return f"Error: old_string not found in {path}"

    count = content.count(old_string)
    if not replace_all and count > 1:
        return f"Error: old_string appears {count} times — use replace_all=true or provide a more specific string"

    if replace_all:
        updated = content.replace(old_string, new_string)
    else:
        updated = content.replace(old_string, new_string, 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    n = count if replace_all else 1

    # Context preview around edit location
    edited_lines = updated.split("\n")
    edit_start = updated.index(new_string)
    edit_line = updated[:edit_start].count("\n")
    ctx_start = max(0, edit_line - 3)
    ctx_end = min(len(edited_lines), edit_line + new_string.count("\n") + 4)
    context = [f"{i+1:4d} | {edited_lines[i]}" for i in range(ctx_start, ctx_end)]
    return f"Edited {path}:{edit_line+1}: replaced {n} occurrence(s)\n" + "\n".join(context)


def _list_files(pattern: str) -> str:
    """List files matching glob pattern."""
    files = globmod.glob(pattern, recursive=True)
    files = sorted(files)[:100]
    return "\n".join(files) if files else "No files found."


_LIST_DIR_MAX_ENTRIES = 500


def _list_directory(path: str = ".", *, show_hidden: bool = False) -> str:
    """List the immediate contents of a directory.

    Returns a header line with counts, then directories (with trailing /),
    then files, both alphabetized. Hidden entries (dotfiles) are excluded
    by default. Capped at 500 entries.
    """
    expanded = os.path.expanduser(path or ".")
    if not os.path.exists(expanded):
        return f"Error: path does not exist: {path}"
    if not os.path.isdir(expanded):
        return f"Error: not a directory: {path}"

    try:
        entries = list(os.scandir(expanded))
    except PermissionError:
        return f"Error: permission denied: {path}"
    except OSError as e:
        return f"Error: could not list {path}: {e}"

    dirs: list[str] = []
    files: list[str] = []
    hidden_skipped = 0
    for entry in entries:
        if entry.name.startswith(".") and not show_hidden:
            hidden_skipped += 1
            continue
        try:
            if entry.is_dir(follow_symlinks=False):
                dirs.append(entry.name)
            elif entry.is_file(follow_symlinks=False):
                files.append(entry.name)
            else:
                # Symlinks, sockets, fifos — show with trailing marker
                files.append(f"{entry.name}@")
        except OSError:
            # Broken symlink or race — still show the name
            files.append(entry.name)

    dirs.sort(key=str.lower)
    files.sort(key=str.lower)

    total = len(dirs) + len(files)
    truncated = False
    combined: list[str] = []
    for d in dirs:
        combined.append(f"{d}/")
        if len(combined) >= _LIST_DIR_MAX_ENTRIES:
            truncated = True
            break
    if not truncated:
        for f in files:
            combined.append(f)
            if len(combined) >= _LIST_DIR_MAX_ENTRIES:
                truncated = True
                break

    # Header
    rel = os.path.relpath(expanded) if not os.path.isabs(path) else expanded
    header_parts = [f"{rel}/"]
    counts = []
    if dirs:
        counts.append(f"{len(dirs)} dir{'s' if len(dirs) != 1 else ''}")
    if files:
        counts.append(f"{len(files)} file{'s' if len(files) != 1 else ''}")
    if counts:
        header_parts.append(f"({', '.join(counts)})")
    if hidden_skipped:
        header_parts.append(f"[{hidden_skipped} hidden, use show_hidden=true to include]")
    header = " ".join(header_parts) + ":"

    if not combined:
        return header + "\n  (empty)"

    body = "\n".join(f"  {line}" for line in combined)
    if truncated:
        body += f"\n  ... (truncated at {_LIST_DIR_MAX_ENTRIES} entries; total {total})"
    return f"{header}\n{body}"


def _execute_agent_tool(
    args: dict,
    *,
    parent_model: str | None,
    parent_backend: str | None,
    parent_confirm_fn=None,
) -> str:
    """Spawn a synchronous, isolated sub-agent and return its final text.

    The sub-agent gets a fresh history, a filtered tool allowlist (per
    `subagent_type`), and a per-type permission mode. It inherits the parent's
    loaded model + backend so /swap propagates. Confirmation prompts still flow
    back to the user via `parent_confirm_fn`. Sub-agents cannot themselves spawn
    sub-agents — `Agent` is absent from every entry in `ALLOWED_TOOLS_BY_SUBAGENT`.
    """
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return "Error: prompt must not be empty"
    # description is optional — derive it from the prompt if missing/empty.
    # Stock Gemma 4 E4B occasionally drops optional-looking required fields,
    # so being lenient here saves a wasted tool round.
    description = (args.get("description") or "").strip() or prompt[:60]

    raw_type = (args.get("subagent_type") or "general-purpose").strip().lower()
    subagent_type = raw_type if raw_type in ALLOWED_TOOLS_BY_SUBAGENT else "general-purpose"
    allowed, perm_mode = ALLOWED_TOOLS_BY_SUBAGENT[subagent_type]

    # Lazy import to avoid the circular agents.tools <-> agents.agent dependency
    from agents.agent import Agent

    sub_name = re.sub(r"[^a-z0-9-]+", "-", description.lower())[:40].strip("-") or "subagent"
    sub_role = f"isolated {subagent_type} sub-agent"
    sub_system = (
        f"You are a background sub-agent of type `{subagent_type}`. "
        f"Work only on the delegated task. You have access to: {', '.join(sorted(allowed))}. "
        "Do not ask the user questions; finish with a concise result."
    )

    init_kwargs: dict = {
        "name": sub_name,
        "role": sub_role,
        "system_prompt": sub_system,
        "tools": True,
        # Sub-agents need more headroom than the 10-round default — they often
        # enumerate then process per-item (e.g. list_files + read_file × N).
        # 32 matches the Rust port's DEFAULT_AGENT_MAX_ITERATIONS.
        "max_tool_rounds": 32,
    }
    if parent_model is not None:
        init_kwargs["model"] = parent_model
    if parent_backend is not None:
        init_kwargs["backend"] = parent_backend

    sub = Agent(**init_kwargs)
    sub.allowed_tool_names = allowed
    sub.permission_mode = perm_mode
    if parent_confirm_fn is not None:
        sub.confirm_fn = parent_confirm_fn

    try:
        return sub.chat(prompt)
    except Exception as e:
        return f"Error: sub-agent failed: {e}"


# ── Sleep ───────────────────────────────────────────────────────────────

SLEEP_MAX_MS = 60_000


def _run_sleep(args: dict) -> str:
    """Sleep for `duration_ms` milliseconds, capped at 60 seconds."""
    raw = args.get("duration_ms")
    if raw is None:
        return "Error: duration_ms is required"
    try:
        ms = int(raw)
    except (TypeError, ValueError):
        return f"Error: duration_ms must be an integer (got {raw!r})"
    if ms < 0:
        return f"Error: duration_ms must be >= 0 (got {ms})"
    if ms > SLEEP_MAX_MS:
        return f"Error: duration_ms must be <= {SLEEP_MAX_MS} (got {ms})"
    time.sleep(ms / 1000)
    return f"Slept {ms}ms"


# ── WebFetch ────────────────────────────────────────────────────────────

WEB_USER_AGENT = "Mozilla/5.0 (Zenith Code; +https://github.com/mastergrief/zenith-code)"
WEB_TIMEOUT_SEC = 10
WEB_MAX_BODY_BYTES = 1_000_000  # 1 MB
WEB_OUTPUT_CHAR_CAP = 8000

ALLOWED_FETCH_CONTENT_TYPES = (
    "text/html",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xml",
    "text/xml",
)


class _TextExtractor(HTMLParser):
    """Strip HTML tags and collect visible text. Drops <script>/<style>/<noscript>."""

    _SKIP_TAGS = {"script", "style", "noscript"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._chunks: list[str] = []
        self._skip_depth = 0

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth == 0:
            text = data.strip()
            if text:
                self._chunks.append(text)

    def text(self) -> str:
        return "\n".join(self._chunks)


def _strip_html(body: str) -> str:
    parser = _TextExtractor()
    try:
        parser.feed(body)
        parser.close()
    except Exception:
        # Malformed HTML — fall back to a regex strip
        return re.sub(r"<[^>]+>", "", body)
    return parser.text()


def _run_web_fetch(args: dict) -> str:
    """Fetch a URL, strip HTML to text, return capped content with prompt header."""
    url = (args.get("url") or "").strip()
    prompt = (args.get("prompt") or "").strip()
    if not url:
        return "Error: url is required"
    if not prompt:
        return "Error: prompt is required"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"Error: only http and https URLs allowed (got scheme {parsed.scheme!r})"
    if not parsed.netloc:
        return f"Error: URL has no host: {url!r}"

    req = urllib.request.Request(url, headers={"User-Agent": WEB_USER_AGENT})
    try:
        with urllib.request.urlopen(req, timeout=WEB_TIMEOUT_SEC) as resp:
            ctype = resp.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype and not any(ctype == t for t in ALLOWED_FETCH_CONTENT_TYPES):
                return f"Error: content-type {ctype!r} not allowed (allowed: {', '.join(ALLOWED_FETCH_CONTENT_TYPES)})"
            raw = resp.read(WEB_MAX_BODY_BYTES + 1)
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} from {url}: {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: could not fetch {url}: {e.reason}"
    except (TimeoutError, OSError) as e:
        return f"Error: network failure fetching {url}: {e}"

    truncated_bytes = len(raw) > WEB_MAX_BODY_BYTES
    body_bytes = raw[:WEB_MAX_BODY_BYTES]
    try:
        body = body_bytes.decode("utf-8", errors="replace")
    except Exception:
        body = body_bytes.decode("latin-1", errors="replace")

    if "html" in ctype:
        text = _strip_html(body)
    else:
        text = body

    truncated_chars = len(text) > WEB_OUTPUT_CHAR_CAP
    if truncated_chars:
        suffix_drop = len(text) - WEB_OUTPUT_CHAR_CAP
        text = text[:WEB_OUTPUT_CHAR_CAP] + f"\n... [{suffix_drop} chars truncated]"
    elif truncated_bytes:
        text = text + "\n... [body exceeded 1 MB, partial content]"

    return f"URL: {url}\nPrompt: {prompt}\n---\n{text}"


# ── WebSearch ───────────────────────────────────────────────────────────

DDG_HTML_URL = "https://html.duckduckgo.com/html/"
WEB_SEARCH_MAX_RESULTS = 10

# DDG result structure (as of 2026-04-08): each result is wrapped in
#   <a class="result__a" href="...">title</a> ... <a class="result__snippet">snippet</a>
_DDG_RESULT_RE = re.compile(
    r'<a[^>]*class="[^"]*result__a[^"]*"[^>]*href="([^"]+)"[^>]*>(.*?)</a>'
    r'.*?<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(.*?)</a>',
    re.DOTALL,
)
_TAG_RE = re.compile(r"<[^>]+>")


def _ddg_clean(html: str) -> str:
    """Strip tags and unescape from a DDG HTML fragment."""
    text = _TAG_RE.sub("", html)
    text = text.replace("&amp;", "&").replace("&lt;", "<").replace("&gt;", ">")
    text = text.replace("&quot;", '"').replace("&#x27;", "'").replace("&nbsp;", " ")
    return text.strip()


def _ddg_resolve_url(href: str) -> str:
    """DDG sometimes wraps result URLs as /l/?uddg=<encoded>. Unwrap when present."""
    if href.startswith("//duckduckgo.com/l/") or href.startswith("/l/"):
        try:
            qs = urllib.parse.urlparse(href).query
            params = urllib.parse.parse_qs(qs)
            if "uddg" in params:
                return urllib.parse.unquote(params["uddg"][0])
        except Exception:
            return href
    return href


def _run_web_search(args: dict) -> str:
    """Search DuckDuckGo HTML and return numbered markdown results."""
    query = (args.get("query") or "").strip()
    if len(query) < 2:
        return "Error: query must be at least 2 characters"

    allowed_domains = args.get("allowed_domains") or []
    blocked_domains = args.get("blocked_domains") or []

    data = urllib.parse.urlencode({"q": query}).encode("utf-8")
    req = urllib.request.Request(
        DDG_HTML_URL,
        data=data,
        headers={
            "User-Agent": WEB_USER_AGENT,
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=WEB_TIMEOUT_SEC) as resp:
            html = resp.read(WEB_MAX_BODY_BYTES).decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return f"Error: HTTP {e.code} from DuckDuckGo: {e.reason}"
    except urllib.error.URLError as e:
        return f"Error: could not reach DuckDuckGo: {e.reason}"
    except (TimeoutError, OSError) as e:
        return f"Error: network failure: {e}"

    matches = _DDG_RESULT_RE.findall(html)
    if not matches:
        return "Error: no results parsed from DuckDuckGo response (their HTML structure may have changed)"

    results: list[tuple[str, str, str]] = []
    for href, title_html, snippet_html in matches:
        url = _ddg_resolve_url(href)
        title = _ddg_clean(title_html)
        snippet = _ddg_clean(snippet_html)
        if not (url and title):
            continue

        host = urllib.parse.urlparse(url).netloc.lower()
        if allowed_domains and not any(host.endswith(d.lower()) for d in allowed_domains):
            continue
        if blocked_domains and any(host.endswith(d.lower()) for d in blocked_domains):
            continue

        results.append((title, url, snippet))
        if len(results) >= WEB_SEARCH_MAX_RESULTS:
            break

    if not results:
        return "No results matched after filtering."

    lines = [f"Found {len(results)} result(s) for {query!r}:"]
    for i, (title, url, snippet) in enumerate(results, 1):
        lines.append(f"{i}. **{title}** — {url}")
        if snippet:
            lines.append(f"   {snippet}")
    return "\n".join(lines)


# ── AskUserQuestion ─────────────────────────────────────────────────────


def _run_ask_user_question(args: dict, ask_user_fn) -> str:
    """Prompt the user via the harness-supplied callback and return their answer."""
    question = (args.get("question") or "").strip()
    if not question:
        return "Error: question must not be empty"
    if ask_user_fn is None:
        return "Error: AskUserQuestion not available in non-interactive context"

    options = args.get("options") or None
    if options is not None and not isinstance(options, list):
        return "Error: options must be a list of strings"
    if options is not None:
        options = [str(o) for o in options]
    try:
        answer = ask_user_fn(question, options)
    except Exception as e:
        return f"Error: ask_user_fn failed: {e}"
    return str(answer) if answer is not None else "(no answer)"


# ── TodoWrite ───────────────────────────────────────────────────────────

_VALID_TODO_STATUS = {"pending", "in_progress", "completed"}
_STATUS_MARKER = {"completed": "[x]", "in_progress": "[~]", "pending": "[ ]"}


def _run_todo_write(args: dict, agent) -> str:
    """Replace `agent.todos` with a validated list and return a markdown render."""
    if agent is None:
        return "Error: TodoWrite requires an agent context"
    raw = args.get("todos")
    if not isinstance(raw, list):
        return "Error: todos must be a list"

    cleaned: list[dict] = []
    in_progress_count = 0
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            return f"Error: todo {i} must be an object, got {type(item).__name__}"
        content = item.get("content")
        active = item.get("activeForm")
        status = item.get("status")
        if not content or not isinstance(content, str):
            return f"Error: todo {i} missing/invalid 'content'"
        if not active or not isinstance(active, str):
            return f"Error: todo {i} missing/invalid 'activeForm'"
        if status not in _VALID_TODO_STATUS:
            return f"Error: todo {i} status must be one of {sorted(_VALID_TODO_STATUS)} (got {status!r})"
        if status == "in_progress":
            in_progress_count += 1
        cleaned.append({"content": content, "activeForm": active, "status": status})

    if in_progress_count > 1:
        return f"Error: at most one todo may be in_progress (got {in_progress_count})"

    # Replace the agent's todo list. Persisted only in memory for this slice
    # — session.py does NOT save agent.todos. Adding session persistence is a
    # tiny follow-up but explicitly out of scope here.
    agent.todos = cleaned

    return _render_todos(cleaned)


def _render_todos(todos: list[dict]) -> str:
    """Render a todo list as markdown — shared by TodoWrite and TodoRead."""
    if not todos:
        return "## Todos\n(no todos set yet — call TodoWrite to create some)"
    pending = sum(1 for t in todos if t["status"] == "pending")
    inprog = sum(1 for t in todos if t["status"] == "in_progress")
    done = sum(1 for t in todos if t["status"] == "completed")
    header = f"## Todos ({pending} pending, {inprog} in progress, {done} completed)"
    lines = [header]
    for t in todos:
        marker = _STATUS_MARKER[t["status"]]
        label = t["activeForm"] if t["status"] == "in_progress" else t["content"]
        lines.append(f"- {marker} {label}")
    return "\n".join(lines)


def _run_todo_read(agent) -> str:
    """Return the current todo list (or a sentinel if none / no agent context)."""
    if agent is None:
        return "Error: TodoRead requires an agent context"
    return _render_todos(getattr(agent, "todos", []) or [])


# ── MultiEdit ───────────────────────────────────────────────────────────


def _run_multi_edit(path: str, edits: list) -> str:
    """Apply N find-and-replace edits to a single file atomically.

    Edits are applied sequentially in the order given (each edit operates on
    the result of the previous one). If ANY edit fails, NONE are written —
    the file on disk is left unchanged.
    """
    if not isinstance(edits, list) or not edits:
        return "Error: edits must be a non-empty list"

    expanded = os.path.expanduser(path)
    try:
        with open(expanded, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except FileNotFoundError:
        return f"Error: file not found: {path}"
    except OSError as e:
        return f"Error: could not read {path}: {e}"

    original = content
    for i, edit in enumerate(edits):
        if not isinstance(edit, dict):
            return f"Error: edit {i} must be an object"
        old = edit.get("old_string")
        new = edit.get("new_string")
        replace_all = bool(edit.get("replace_all", False))
        if old is None or not isinstance(old, str):
            return f"Error: edit {i} missing/invalid 'old_string'"
        if new is None or not isinstance(new, str):
            return f"Error: edit {i} missing/invalid 'new_string'"
        if old == new:
            return f"Error: edit {i}: old_string and new_string must differ"
        if old not in content:
            return f"Error: edit {i}: old_string not found in current file state"
        count = content.count(old)
        if not replace_all and count > 1:
            return (
                f"Error: edit {i}: old_string appears {count} times — "
                "use replace_all=true or provide a more specific string"
            )
        if replace_all:
            content = content.replace(old, new)
        else:
            content = content.replace(old, new, 1)

    if content == original:
        return f"No changes (all {len(edits)} edits were no-ops)"

    try:
        with open(expanded, "w", encoding="utf-8") as f:
            f.write(content)
    except OSError as e:
        return f"Error: could not write {path}: {e}"

    # Compute summary stats
    line_diff = content.count("\n") - original.count("\n")
    char_diff = len(content) - len(original)
    sign = "+" if char_diff >= 0 else ""
    return (
        f"Applied {len(edits)} edit(s) to {path} "
        f"({sign}{char_diff} chars, {sign}{line_diff} lines)"
    )


# ── Persistent agent registry tools ─────────────────────────────────────
#
# These tools share the same `Agent` class as the one-shot `Agent` tool,
# but the spawned sub-agents are stored in a process-global registry so
# they outlive a single tool call. Use AgentCreate to spawn, AgentMessage
# to re-engage, AgentGet/List to inspect, AgentTerminate to release.


def _build_subagent(
    *,
    prompt: str,
    subagent_type_raw: str | None,
    role_override: str | None,
    name_hint: str | None,
    parent_model: str | None,
    parent_backend: str | None,
    parent_confirm_fn,
):
    """Construct a sub-agent Agent instance with the standard sub-agent wiring.

    Returns the Agent instance (not yet registered, not yet chatted).
    Shared by the one-shot Agent tool and the AgentCreate registry tool.
    """
    raw_type = (subagent_type_raw or "general-purpose").strip().lower()
    subagent_type = raw_type if raw_type in ALLOWED_TOOLS_BY_SUBAGENT else "general-purpose"
    allowed, perm_mode = ALLOWED_TOOLS_BY_SUBAGENT[subagent_type]

    from agents.agent import Agent  # lazy import — circular avoidance

    fallback_name = name_hint or prompt
    sub_name = re.sub(r"[^a-z0-9-]+", "-", fallback_name.lower())[:40].strip("-") or "subagent"
    sub_role = (role_override or "").strip() or f"isolated {subagent_type} sub-agent"
    sub_system = (
        f"You are a background sub-agent of type `{subagent_type}`. "
        f"Role: {sub_role}. "
        f"You have access to: {', '.join(sorted(allowed))}. "
        "Work only on the delegated task. Finish with a concise result."
    )

    init_kwargs: dict = {
        "name": sub_name,
        "role": sub_role,
        "system_prompt": sub_system,
        "tools": True,
        "max_tool_rounds": 32,
    }
    if parent_model is not None:
        init_kwargs["model"] = parent_model
    if parent_backend is not None:
        init_kwargs["backend"] = parent_backend

    sub = Agent(**init_kwargs)
    sub.allowed_tool_names = allowed
    sub.permission_mode = perm_mode
    if parent_confirm_fn is not None:
        sub.confirm_fn = parent_confirm_fn
    return sub


def _run_agent_create(
    args: dict,
    *,
    parent_model: str | None,
    parent_backend: str | None,
    parent_confirm_fn=None,
) -> str:
    """Spawn a persistent sub-agent, register it, run first turn, return ID + response."""
    prompt = (args.get("prompt") or "").strip()
    if not prompt:
        return "Error: prompt must not be empty"

    name_hint = (args.get("name") or "").strip() or None

    sub = _build_subagent(
        prompt=prompt,
        subagent_type_raw=args.get("subagent_type"),
        role_override=args.get("role"),
        name_hint=name_hint,
        parent_model=parent_model,
        parent_backend=parent_backend,
        parent_confirm_fn=parent_confirm_fn,
    )

    from agents.agent_registry import get_registry
    registry = get_registry()

    try:
        agent_id = registry.register(sub, agent_id=name_hint)
    except ValueError as e:
        return f"Error: {e}"

    try:
        first_response = sub.chat(prompt)
    except Exception as e:
        # Roll back registration on failure so the model doesn't see a
        # half-spawned ghost agent.
        registry.terminate(agent_id)
        return f"Error: sub-agent failed during initial turn: {e}"

    registry.touch(agent_id)
    return f"agent_id: {agent_id}\n---\n{first_response}"


def _run_agent_message(args: dict) -> str:
    """Send another turn to a registered sub-agent."""
    aid = (args.get("agent_id") or "").strip()
    content = (args.get("content") or "").strip()
    if not aid:
        return "Error: agent_id is required"
    if not content:
        return "Error: content must not be empty"

    from agents.agent_registry import get_registry
    registry = get_registry()
    ra = registry.get(aid)
    if ra is None:
        return f"Error: no agent registered with id {aid!r}"

    try:
        response = ra.agent.chat(content)
    except Exception as e:
        return f"Error: agent {aid!r} failed: {e}"

    registry.touch(aid)
    return response


def _run_agent_get(args: dict) -> str:
    """Return a state snapshot of a registered sub-agent as a markdown block."""
    aid = (args.get("agent_id") or "").strip()
    if not aid:
        return "Error: agent_id is required"

    from agents.agent_registry import get_registry
    registry = get_registry()
    ra = registry.get(aid)
    if ra is None:
        return f"Error: no agent registered with id {aid!r}"

    last_assistant = ""
    for msg in reversed(ra.agent.history):
        if msg.get("role") == "assistant":
            last_assistant = (msg.get("content") or "")[:500]
            break

    todos = getattr(ra.agent, "todos", []) or []
    age = round(time.time() - ra.created_at, 1)
    lines = [
        f"## Agent {aid}",
        f"- role: {ra.agent.role}",
        f"- model: {ra.agent.model}",
        f"- history_len: {len(ra.agent.history)}",
        f"- todos: {len(todos)}",
        f"- age: {age}s",
    ]
    if last_assistant:
        lines.append("- last_assistant_message:")
        lines.append(f"  {last_assistant}")
    else:
        lines.append("- last_assistant_message: (none yet)")
    return "\n".join(lines)


def _run_agent_list(_args: dict) -> str:
    """List all registered sub-agents as a markdown table."""
    from agents.agent_registry import get_registry
    registry = get_registry()
    entries = registry.list_agents()
    if not entries:
        return "No agents registered. Use AgentCreate to spawn one."

    lines = [f"## Registered agents ({len(entries)})"]
    for e in entries:
        lines.append(
            f"- **{e['id']}** — {e['role']} "
            f"(history={e['history_len']}, todos={e['todos_len']}, age={e['age_seconds']}s)"
        )
    return "\n".join(lines)


def _run_agent_terminate(args: dict) -> str:
    """Remove a registered sub-agent."""
    aid = (args.get("agent_id") or "").strip()
    if not aid:
        return "Error: agent_id is required"

    from agents.agent_registry import get_registry
    registry = get_registry()
    if registry.terminate(aid):
        return f"Terminated agent {aid!r}"
    return f"Error: no agent registered with id {aid!r}"
