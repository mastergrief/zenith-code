"""Tool permission enforcement — classification-based validation."""

import os
import re
from enum import Enum


class PermissionMode(Enum):
    READ_ONLY = "readonly"        # Only read tools allowed
    WORKSPACE_WRITE = "workspace"  # Write within project, no system paths
    FULL_ACCESS = "full"           # Everything allowed (still blocks catastrophic)


class BashRisk(Enum):
    SAFE = "safe"                # read-only commands (ls, cat, grep, git status)
    WRITE = "write"              # creates/modifies files (touch, cp, mv, mkdir, npm install)
    DESTRUCTIVE = "destructive"  # data loss risk (rm, git reset --hard, docker rm)
    BLOCKED = "blocked"          # always blocked (rm -rf /, fork bombs, shred)


# Commands that are always safe (read-only)
SAFE_COMMANDS = {
    "ls", "cat", "head", "tail", "less", "more", "wc", "file", "stat",
    "find", "grep", "rg", "ag", "which", "whereis", "type", "echo",
    "pwd", "env", "printenv", "date", "whoami", "hostname", "uname",
    "df", "du", "free", "top", "ps", "id", "groups",
}

# Git subcommands: read-only vs write
GIT_SAFE = {"status", "log", "diff", "show", "branch", "tag", "stash list", "remote", "blame"}
GIT_WRITE = {"add", "commit", "merge", "rebase", "checkout", "switch", "stash push", "stash pop"}
GIT_DESTRUCTIVE = {"push", "reset", "clean", "checkout --", "restore", "push --force", "stash drop"}

# Always blocked patterns
BLOCKED_PATTERNS = [
    r"rm\s+-rf\s+/",
    r":\(\)\{.*:\|:.*\}",  # fork bomb
    r"mkfs\.", r"dd\s+if=", r"shred",
    r">\s*/dev/sd", r"chmod\s+-R\s+777\s+/",
]

# Write redirect detection
WRITE_REDIRECT_PATTERN = r"(?<![12])>{1,2}\s*[^&]"

# System paths (never write to these)
SYSTEM_PATHS = ["/etc/", "/usr/", "/var/", "/boot/", "/sys/", "/proc/", "~/.ssh/"]


def classify_bash(command: str) -> BashRisk:
    """Classify a bash command by risk level."""
    # Check blocked patterns first
    for pattern in BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return BashRisk.BLOCKED

    # Check write redirects
    if re.search(WRITE_REDIRECT_PATTERN, command):
        return BashRisk.WRITE

    # Extract first word (the command)
    first_word = command.strip().split()[0] if command.strip() else ""

    # Git subcommand classification
    if first_word == "git":
        parts = command.strip().split()
        subcmd = parts[1] if len(parts) > 1 else ""
        # Check two-word subcommands first (e.g. "checkout --", "push --force")
        two_word = " ".join(parts[1:3]) if len(parts) > 2 else ""
        if two_word in GIT_DESTRUCTIVE or subcmd in GIT_DESTRUCTIVE:
            return BashRisk.DESTRUCTIVE
        if two_word in GIT_WRITE or subcmd in GIT_WRITE:
            return BashRisk.WRITE
        if two_word in GIT_SAFE or subcmd in GIT_SAFE:
            return BashRisk.SAFE
        return BashRisk.WRITE  # unknown git = assume write

    # Safe commands
    if first_word in SAFE_COMMANDS:
        return BashRisk.SAFE

    # Path traversal check
    if "../" in command or command.startswith("~/."):
        for sys_path in SYSTEM_PATHS:
            if sys_path in command:
                return BashRisk.DESTRUCTIVE

    # Default: classify as write (ask user)
    return BashRisk.WRITE


def check_permission(tool_name: str, tool_args: dict, mode: PermissionMode) -> tuple[bool, str]:
    """Returns (allowed, reason). If not allowed, reason explains why."""
    if mode == PermissionMode.FULL_ACCESS:
        # Still block catastrophic
        if tool_name == "bash":
            risk = classify_bash(tool_args.get("command", ""))
            if risk == BashRisk.BLOCKED:
                return False, "Blocked: catastrophic command"
        # Always block system paths, even in full access
        if tool_name in ("write_file", "edit_file"):
            path = os.path.expanduser(tool_args.get("path", ""))
            for sys_path in SYSTEM_PATHS:
                expanded_sys = os.path.expanduser(sys_path)
                if path.startswith(expanded_sys):
                    return False, f"Blocked: system path {sys_path}"
        return True, ""

    if mode == PermissionMode.READ_ONLY:
        if tool_name in ("write_file", "edit_file"):
            return False, "Blocked: read-only mode"
        if tool_name == "bash":
            risk = classify_bash(tool_args.get("command", ""))
            if risk != BashRisk.SAFE:
                return False, f"Blocked: read-only mode (command classified as {risk.value})"
        return True, ""

    # WORKSPACE_WRITE (default)
    if tool_name == "bash":
        risk = classify_bash(tool_args.get("command", ""))
        if risk == BashRisk.BLOCKED:
            return False, "Blocked: catastrophic command"
        if risk == BashRisk.DESTRUCTIVE:
            return False, "Needs confirmation: destructive command"
        if risk == BashRisk.WRITE:
            return False, "Needs confirmation: write command"
    if tool_name in ("write_file", "edit_file"):
        path = os.path.expanduser(tool_args.get("path", ""))
        for sys_path in SYSTEM_PATHS:
            expanded_sys = os.path.expanduser(sys_path)
            if path.startswith(expanded_sys):
                return False, f"Blocked: system path {sys_path}"
        return False, "Needs confirmation: file write"
    return True, ""
