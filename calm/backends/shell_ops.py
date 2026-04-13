"""
CALM shell backend — verified command/signal/exit code knowledge.

Models say "run rm -rf /" without flagging danger. This backend
validates commands, explains exit codes, and identifies risky operations.

Functions: exit_code, signal_name, is_dangerous, command_exists,
parse_command, env_var.
"""

from __future__ import annotations

import os
import shutil
import signal as _signal


def exit_code_meaning(code: int) -> dict:
    """Explain what a shell exit code means."""
    code = int(code)
    meanings = {
        0: "success",
        1: "general error",
        2: "misuse of shell builtin",
        126: "command found but not executable",
        127: "command not found",
        128: "invalid exit argument",
        130: "terminated by Ctrl+C (SIGINT)",
        137: "killed (SIGKILL / OOM)",
        139: "segmentation fault (SIGSEGV)",
        141: "broken pipe (SIGPIPE)",
        143: "terminated (SIGTERM)",
    }
    if code > 128 and code < 256:
        sig_num = code - 128
        sig_name = signal_name(sig_num)
        return {"code": code, "meaning": f"killed by signal {sig_num} ({sig_name})",
                "category": "signal"}
    return {
        "code": code,
        "meaning": meanings.get(code, "application-specific"),
        "category": "success" if code == 0 else "error",
    }


def signal_name(num: int) -> str:
    """Get signal name from number. signal_name(9) → 'SIGKILL'."""
    num = int(num)
    try:
        return _signal.Signals(num).name
    except (ValueError, AttributeError):
        return f"UNKNOWN({num})"


def signal_number(name: str) -> int:
    """Get signal number from name. signal_number('SIGKILL') → 9."""
    name = name.upper()
    if not name.startswith("SIG"):
        name = "SIG" + name
    try:
        return _signal.Signals[name].value
    except (KeyError, AttributeError):
        return -1


def is_dangerous_command(cmd: str) -> dict:
    """Check if a shell command is potentially dangerous.
    Returns {dangerous, risk_level, reasons}."""
    cmd_lower = cmd.lower().strip()
    reasons = []

    _DANGEROUS = [
        (r"rm\s+(-rf?|--recursive)\s+/", "recursive delete from root"),
        (r"rm\s+(-rf?|--recursive)\s+\*", "recursive delete with wildcard"),
        (r"mkfs\.", "filesystem format"),
        (r"dd\s+if=.+of=/dev/", "raw disk write"),
        (r":\(\)\{.*\}", "fork bomb"),
        (r"chmod\s+-R\s+777", "world-writable permissions"),
        (r"chmod\s+777", "world-writable permissions"),
        (r">\s*/dev/sd", "overwrite disk device"),
        (r"curl.*\|\s*sh", "pipe remote script to shell"),
        (r"wget.*\|\s*sh", "pipe remote script to shell"),
        (r"eval\s+\$", "eval with variable expansion"),
        (r"sudo\s+rm\s+-rf", "sudo recursive delete"),
    ]

    _RISKY = [
        (r"rm\s+-r", "recursive delete"),
        (r"git\s+reset\s+--hard", "discard all changes"),
        (r"git\s+push\s+--force", "force push (may overwrite history)"),
        (r"git\s+clean\s+-f", "delete untracked files"),
        (r"DROP\s+TABLE", "SQL table drop"),
        (r"DROP\s+DATABASE", "SQL database drop"),
        (r"TRUNCATE", "SQL truncate"),
        (r"kill\s+-9", "force kill process"),
        (r"pkill", "kill processes by name"),
        (r"shutdown", "system shutdown"),
        (r"reboot", "system reboot"),
    ]

    import re
    for pattern, reason in _DANGEROUS:
        if re.search(pattern, cmd_lower):
            reasons.append(f"DANGEROUS: {reason}")

    for pattern, reason in _RISKY:
        if re.search(pattern, cmd_lower):
            reasons.append(f"RISKY: {reason}")

    if reasons:
        has_dangerous = any("DANGEROUS" in r for r in reasons)
        return {
            "dangerous": has_dangerous,
            "risk_level": "critical" if has_dangerous else "high",
            "reasons": reasons,
            "command": cmd[:100],
        }

    return {"dangerous": False, "risk_level": "safe", "reasons": [], "command": cmd[:100]}


def command_exists(cmd: str) -> bool:
    """Check if a command is available in PATH."""
    return shutil.which(cmd) is not None


def env_var(name: str) -> str:
    """Get an environment variable value. Returns '' if not set."""
    return os.environ.get(name, "")


def parse_shebang(path: str) -> str:
    """Extract the shebang line from a script file."""
    try:
        with open(path, 'r') as f:
            first_line = f.readline().strip()
        return first_line if first_line.startswith("#!") else ""
    except (FileNotFoundError, UnicodeDecodeError):
        return ""


SHELL_FUNCTIONS = {
    "exit_code_meaning": exit_code_meaning,
    "signal_name": signal_name,
    "signal_number": signal_number,
    "is_dangerous_command": is_dangerous_command,
    "command_exists": command_exists,
    "env_var": env_var,
    "parse_shebang": parse_shebang,
}

SHELL_NL_PATTERNS = [
    (r'(?:what does|explain)\s+exit\s+(?:code|status)\s+(\d+)', 'exit_code_meaning({0})'),
    (r'(?:is)\s+["`]?(.+?)["`]?\s+(?:a\s+)?(?:dangerous|destructive)\s+(?:command|shell)', 'is_dangerous_command("{0}")'),
]
