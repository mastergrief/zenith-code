"""
CALM Error Code knowledge backend — exit codes, errno, signals.

Models invent error numbers. POSIX + common application codes.
"""

from __future__ import annotations

_DATA_VERSION = "2025-01"

_EXIT_CODES = {
    0: "Success",
    1: "General error",
    2: "Misuse of shell command",
    126: "Command invoked cannot execute (permission)",
    127: "Command not found",
    128: "Invalid exit argument",
    130: "Script terminated by Ctrl+C (128+SIGINT)",
    137: "Process killed (128+SIGKILL, often OOM)",
    139: "Segmentation fault (128+SIGSEGV)",
    141: "Broken pipe (128+SIGPIPE)",
    143: "Terminated (128+SIGTERM)",
    255: "Exit status out of range",
}

_ERRNO = {
    1: ("EPERM", "Operation not permitted"),
    2: ("ENOENT", "No such file or directory"),
    3: ("ESRCH", "No such process"),
    4: ("EINTR", "Interrupted system call"),
    5: ("EIO", "I/O error"),
    9: ("EBADF", "Bad file descriptor"),
    11: ("EAGAIN", "Resource temporarily unavailable (try again)"),
    12: ("ENOMEM", "Out of memory"),
    13: ("EACCES", "Permission denied"),
    14: ("EFAULT", "Bad address"),
    17: ("EEXIST", "File exists"),
    20: ("ENOTDIR", "Not a directory"),
    21: ("EISDIR", "Is a directory"),
    22: ("EINVAL", "Invalid argument"),
    23: ("ENFILE", "Too many open files in system"),
    24: ("EMFILE", "Too many open files"),
    28: ("ENOSPC", "No space left on device"),
    32: ("EPIPE", "Broken pipe"),
    36: ("ENAMETOOLONG", "File name too long"),
    110: ("ETIMEDOUT", "Connection timed out"),
    111: ("ECONNREFUSED", "Connection refused"),
    113: ("EHOSTUNREACH", "No route to host"),
}

_SIGNALS = {
    1: ("SIGHUP", "Hangup (terminal closed)"),
    2: ("SIGINT", "Interrupt (Ctrl+C)"),
    3: ("SIGQUIT", "Quit (core dump)"),
    6: ("SIGABRT", "Abort"),
    9: ("SIGKILL", "Kill (cannot be caught)"),
    11: ("SIGSEGV", "Segmentation fault"),
    13: ("SIGPIPE", "Broken pipe"),
    14: ("SIGALRM", "Alarm clock"),
    15: ("SIGTERM", "Termination (graceful)"),
    17: ("SIGCHLD", "Child process status changed"),
    19: ("SIGSTOP", "Stop (cannot be caught)"),
    20: ("SIGTSTP", "Terminal stop (Ctrl+Z)"),
}


def exit_code(code: int) -> str:
    """Explain a process exit code."""
    code = int(code)
    if code in _EXIT_CODES:
        return f"exit {code}: {_EXIT_CODES[code]}"
    if 129 <= code <= 192:
        sig = code - 128
        if sig in _SIGNALS:
            name, desc = _SIGNALS[sig]
            return f"exit {code}: killed by {name} ({desc})"
        return f"exit {code}: killed by signal {sig}"
    return f"exit {code}: non-standard"


def errno_info(num: int) -> str:
    """Explain a POSIX errno value."""
    num = int(num)
    if num in _ERRNO:
        name, desc = _ERRNO[num]
        return f"errno {num} ({name}): {desc}"
    return f"errno {num}: unknown"


def signal_info(num: int) -> str:
    """Explain a Unix signal number."""
    num = int(num)
    if num in _SIGNALS:
        name, desc = _SIGNALS[num]
        return f"signal {num} ({name}): {desc}"
    return f"signal {num}: unknown"


def errno_by_name(name: str) -> str:
    """Look up errno by name (e.g., 'ENOENT')."""
    name = name.strip().upper()
    for num, (ename, desc) in _ERRNO.items():
        if ename == name:
            return f"{ename} (errno {num}): {desc}"
    return f"unknown errno: {name}"


ERROR_CODE_FUNCTIONS = {
    "exit_code": exit_code,
    "errno_info": errno_info,
    "signal_info": signal_info,
    "errno_by_name": errno_by_name,
}

ERROR_CODE_NL_PATTERNS = [
    (r'(?:what is|explain)\s+exit\s+(?:code\s+)?(\d+)', 'exit_code({0})'),
    (r'(?:what is|explain)\s+errno\s+(\d+)', 'errno_info({0})'),
    (r'(?:what is|explain)\s+signal\s+(\d+)', 'signal_info({0})'),
    (r'(?:what is|explain)\s+(ENOENT|EACCES|EPERM|EINVAL|ENOMEM|ENOSPC|EPIPE|ECONNREFUSED|ETIMEDOUT)', 'errno_by_name("{0}")'),
    (r'(?:what does|why)\s+(?:exit\s+)?(?:code\s+)?137\s+mean', 'exit_code(137)'),
]
