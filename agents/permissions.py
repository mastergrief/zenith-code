"""Tool permission enforcement — deny-list for dangerous operations."""

from dataclasses import dataclass, field


# Destructive bash patterns from rust bash_validation.rs
_DEFAULT_BASH_DENY = (
    ("rm -rf /", "Recursive forced deletion at root"),
    ("rm -rf ~", "Recursive forced deletion of home directory"),
    ("rm -rf *", "Recursive forced deletion of all files in cwd"),
    ("rm -rf .", "Recursive forced deletion of current directory"),
    ("> /dev/sd", "Writing to raw disk device"),
    (":(){ :|:& };:", "Fork bomb"),
    ("mkfs", "Filesystem creation destroys existing data"),
    ("dd if=", "Direct disk write"),
    ("chmod -R 777", "Recursively setting world-writable permissions"),
    ("chmod -R 000", "Recursively removing all permissions"),
)

_DEFAULT_WRITE_DENY_PREFIXES = (
    "/etc/",
    "/usr/",
    "/sys/",
    "/proc/",
    "/dev/",
    "/boot/",
    "/sbin/",
)

_ALWAYS_DESTRUCTIVE = frozenset({"shred", "wipefs"})


@dataclass(frozen=True)
class ToolPermissions:
    """Deny-list for tool calls. Blocks known-dangerous operations."""

    deny_names: frozenset[str] = field(default_factory=frozenset)
    bash_deny_patterns: tuple[tuple[str, str], ...] = _DEFAULT_BASH_DENY
    write_deny_prefixes: tuple[str, ...] = _DEFAULT_WRITE_DENY_PREFIXES

    def blocks(self, tool_name: str, tool_args: dict) -> str | None:
        """Return reason string if blocked, None if allowed."""
        if tool_name in self.deny_names:
            return f"Tool '{tool_name}' is denied"

        if tool_name == "bash":
            return self._check_bash(tool_args.get("command", ""))

        if tool_name == "write_file":
            return self._check_write_path(tool_args.get("path", ""))

        if tool_name == "edit_file":
            return self._check_write_path(tool_args.get("path", ""))

        return None

    def _check_bash(self, command: str) -> str | None:
        # Check destructive patterns
        for pattern, reason in self.bash_deny_patterns:
            if pattern in command:
                return f"Destructive command blocked: {reason}"

        # Check always-destructive commands
        first = command.strip().split()[0] if command.strip() else ""
        if first in _ALWAYS_DESTRUCTIVE:
            return f"Command '{first}' is inherently destructive"

        return None

    def _check_write_path(self, path: str) -> str | None:
        import os

        expanded = os.path.expanduser(path)
        for prefix in self.write_deny_prefixes:
            if expanded.startswith(prefix):
                return f"Writing to '{prefix}' is not allowed"

        # Block ~/.ssh/ writes
        ssh_dir = os.path.expanduser("~/.ssh")
        if expanded.startswith(ssh_dir):
            return "Writing to ~/.ssh/ is not allowed"

        return None


DEFAULT_PERMISSIONS = ToolPermissions()
