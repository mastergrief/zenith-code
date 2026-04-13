"""
CALM Git reference knowledge backend — commands, concepts, common workflows.

Models mix up rebase/merge, confuse reset modes, hallucinate flags.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_COMMANDS = {
    "init": {"description": "Create empty repo or reinitialize", "creates_files": ".git/", "safe": True},
    "clone": {"description": "Clone a repository into new directory", "network": True, "safe": True},
    "add": {"description": "Stage changes for next commit", "modifies": "index", "safe": True},
    "commit": {"description": "Record staged changes to repository", "modifies": "HEAD", "safe": True},
    "status": {"description": "Show working tree status", "modifies": "nothing", "safe": True},
    "log": {"description": "Show commit history", "modifies": "nothing", "safe": True},
    "diff": {"description": "Show changes between commits, working tree, etc.", "modifies": "nothing", "safe": True},
    "branch": {"description": "List, create, or delete branches", "modifies": "refs", "safe": True},
    "checkout": {"description": "Switch branches or restore working tree files", "modifies": "working tree + HEAD", "destructive": "can discard uncommitted changes with -- paths"},
    "switch": {"description": "Switch branches (safer than checkout)", "modifies": "working tree + HEAD", "safe": True},
    "merge": {"description": "Join two or more development histories", "modifies": "working tree + HEAD", "creates": "merge commit (unless fast-forward)"},
    "rebase": {"description": "Reapply commits on top of another base", "modifies": "commit history", "destructive": "rewrites history — never rebase shared/pushed commits"},
    "pull": {"description": "Fetch + merge (or rebase with --rebase)", "network": True, "modifies": "working tree + HEAD"},
    "push": {"description": "Upload local commits to remote", "network": True, "modifies": "remote refs"},
    "fetch": {"description": "Download objects and refs from remote", "network": True, "modifies": "remote-tracking branches only", "safe": True},
    "stash": {"description": "Temporarily shelve uncommitted changes", "modifies": "working tree + stash stack", "safe": True},
    "reset": {"description": "Reset HEAD to a specific state", "modifies": "HEAD + maybe index + maybe working tree", "modes": {"--soft": "moves HEAD only", "--mixed": "moves HEAD + resets index (default)", "--hard": "moves HEAD + resets index + working tree (DESTRUCTIVE)"}},
    "revert": {"description": "Create new commit that undoes a previous commit", "modifies": "working tree + HEAD", "safe": True},
    "cherry-pick": {"description": "Apply a specific commit to current branch", "modifies": "working tree + HEAD", "safe": True},
    "tag": {"description": "Create, list, delete, or verify tags", "modifies": "refs", "safe": True},
    "remote": {"description": "Manage set of tracked repositories", "modifies": "config", "safe": True},
    "bisect": {"description": "Binary search for commit that introduced a bug", "modifies": "HEAD (temporarily)", "safe": True},
    "blame": {"description": "Show who last modified each line of a file", "modifies": "nothing", "safe": True},
    "reflog": {"description": "Show reference logs (safety net for lost commits)", "modifies": "nothing", "safe": True},
    "clean": {"description": "Remove untracked files", "destructive": "permanently deletes files not in git"},
    "worktree": {"description": "Manage multiple working trees from one repo", "modifies": "creates new directory", "safe": True},
}

_RESET_MODES = {
    "soft": {"HEAD": "moves", "index": "unchanged", "working_tree": "unchanged", "use_case": "undo commit but keep changes staged"},
    "mixed": {"HEAD": "moves", "index": "reset", "working_tree": "unchanged", "use_case": "undo commit + unstage (default)"},
    "hard": {"HEAD": "moves", "index": "reset", "working_tree": "reset", "use_case": "discard everything — DESTRUCTIVE"},
}

_MERGE_VS_REBASE = {
    "merge": {
        "history": "preserves — creates merge commit",
        "shared_branches": "safe",
        "conflicts": "resolve once",
        "result": "non-linear history with merge bubbles",
        "golden_rule": "use for shared/public branches",
    },
    "rebase": {
        "history": "rewrites — replays commits linearly",
        "shared_branches": "DANGEROUS — rewrites published commits",
        "conflicts": "may need to resolve per-commit",
        "result": "clean linear history",
        "golden_rule": "only rebase local/unpushed commits",
    },
}

_COMMON_ALIASES = {
    "co": "checkout",
    "ci": "commit",
    "br": "branch",
    "st": "status",
    "df": "diff",
    "lg": "log --oneline --graph --decorate",
    "unstage": "reset HEAD --",
    "last": "log -1 HEAD",
    "amend": "commit --amend",
}


def git_command(name: str) -> dict:
    """Get details about a git command."""
    key = str(name).lower().strip().lstrip('-')
    entry = _COMMANDS.get(key)
    if not entry:
        return {"error": f"Unknown command: {name}", "valid": sorted(_COMMANDS.keys())}
    return {"command": f"git {key}", **entry}


def reset_mode(mode: str) -> dict:
    """Explain a git reset mode (soft, mixed, hard)."""
    key = str(mode).lower().strip().lstrip('-')
    entry = _RESET_MODES.get(key)
    if not entry:
        return {"error": f"Unknown mode: {mode}", "valid": list(_RESET_MODES.keys())}
    return {"mode": f"--{key}", **entry}


def merge_vs_rebase() -> dict:
    """Compare git merge vs rebase."""
    return _MERGE_VS_REBASE


def is_destructive(command: str) -> bool:
    """Whether a git command can cause data loss."""
    key = str(command).lower().strip()
    entry = _COMMANDS.get(key)
    if not entry:
        return False
    return "destructive" in entry or entry.get("safe") is not True


def list_safe_commands() -> list[str]:
    """List all non-destructive git commands."""
    return sorted(k for k, v in _COMMANDS.items() if v.get("safe"))


def common_alias(alias: str) -> str:
    """Expand a common git alias."""
    entry = _COMMON_ALIASES.get(str(alias).lower().strip())
    return f"git {entry}" if entry else f"Unknown alias: {alias}"


GIT_REF_FUNCTIONS = {
    "git_command": git_command,
    "reset_mode": reset_mode,
    "merge_vs_rebase": merge_vs_rebase,
    "is_destructive": is_destructive,
    "list_safe_commands": list_safe_commands,
    "common_alias": common_alias,
}

GIT_REF_NL_PATTERNS = [
    (r'(?:what does|explain|what is)\s+git\s+(\w+)', 'git_command("{0}")'),
    (r'(?:difference between|vs)\s+(?:git\s+)?merge\s+(?:and|vs)\s+rebase', 'merge_vs_rebase()'),
    (r'(?:what does|explain)\s+(?:git\s+)?reset\s+--(soft|mixed|hard)', 'reset_mode("{0}")'),
    (r'(?:is)\s+git\s+(\w+)\s+(?:destructive|dangerous|safe)', 'is_destructive("{0}")'),
]
