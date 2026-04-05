"""Tools that agents can invoke: bash, read, write, edit, grep, glob."""

import json
import os
import subprocess
import glob as globmod
import re

from agents.permissions import DEFAULT_PERMISSIONS


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
            "description": "Read a file and return its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Path to the file to read",
                    }
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
            "description": "Search for a pattern in files. Returns matching lines with file paths and line numbers.",
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
                        "description": "File glob pattern to filter (e.g. '*.py')",
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
]


def execute_tool(name: str, args: dict) -> str:
    """Execute a tool by name with given arguments."""
    # Permission check
    blocked = DEFAULT_PERMISSIONS.blocks(name, args)
    if blocked:
        return f"Blocked: {blocked}"

    try:
        if name == "bash":
            return _run_bash(args["command"])
        elif name == "read_file":
            return _read_file(args["path"])
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
            )
        elif name == "list_files":
            return _list_files(args["pattern"])
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


def _read_file(path: str) -> str:
    """Read file contents."""
    path = os.path.expanduser(path)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()
    if len(lines) > 500:
        content = "".join(lines[:500])
        content += f"\n... ({len(lines) - 500} more lines truncated)"
        return content
    return "".join(lines)


def _write_file(path: str, content: str) -> str:
    """Write content to file."""
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Wrote {len(content)} bytes to {path}"


def _grep(pattern: str, path: str = ".", file_glob: str | None = None) -> str:
    """Search for pattern in files."""
    path = os.path.expanduser(path)
    results = []
    regex = re.compile(pattern, re.IGNORECASE)

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
                        results.append(f"{filepath}:{i}: {line.rstrip()}")
                        if len(results) >= 50:
                            results.append("... (results truncated at 50 matches)")
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
    return f"Edited {path}: replaced {n} occurrence(s)"


def _list_files(pattern: str) -> str:
    """List files matching glob pattern."""
    files = globmod.glob(pattern, recursive=True)
    files = sorted(files)[:100]
    return "\n".join(files) if files else "No files found."
