"""
CALM Linux/Unix backend — file permissions, signals, process states, common commands.

Models botch chmod octal, confuse SIGTERM/SIGKILL, miscalculate umask.
"""

from __future__ import annotations


def chmod_to_symbolic(octal: int) -> str:
    """Convert octal permissions (e.g. 755) to symbolic (rwxr-xr-x)."""
    # Input is the literal digits (755), not Python octal (0o755=493).
    # Parse each digit directly.
    s = str(int(octal)).zfill(3)
    if len(s) == 4:
        s = s[1:]  # strip special bits prefix for basic conversion
    result = []
    for ch in s:
        bits = int(ch)
        result.append(('r' if bits & 4 else '-') +
                      ('w' if bits & 2 else '-') +
                      ('x' if bits & 1 else '-'))
    return ''.join(result)


def chmod_to_octal(symbolic: str) -> int:
    """Convert symbolic permissions (e.g. rwxr-xr-x) to octal (755)."""
    s = str(symbolic).strip()
    if len(s) == 10:  # strip leading type char (drwxr-xr-x)
        s = s[1:]
    if len(s) != 9:
        return -1
    # Parse each 3-char group into an octal digit, return as decimal representation
    digits = []
    for i in range(0, 9, 3):
        group = s[i:i + 3]
        val = (4 if group[0] == 'r' else 0) + (2 if group[1] == 'w' else 0) + (1 if group[2] in ('x', 's', 't') else 0)
        digits.append(str(val))
    return int(''.join(digits))


def umask_result(umask: int, base: int = 666) -> int:
    """Calculate resulting file permissions: base & ~umask (octal digit math)."""
    # Both inputs are literal octal digits (e.g. 022, 666)
    u = str(int(umask)).zfill(3)
    b = str(int(base)).zfill(3)
    result = []
    for ud, bd in zip(u, b):
        result.append(str(int(bd) & ~int(ud) & 7))
    return int(''.join(result))


def umask_for_dirs(umask: int) -> int:
    """Calculate directory permissions from umask (base 777)."""
    return umask_result(umask, 777)


def signal_info(sig: str) -> dict:
    """Get info about a Unix signal."""
    signals = {
        "SIGHUP": {"number": 1, "default": "terminate", "description": "Hangup — terminal closed or daemon reload", "catchable": True},
        "SIGINT": {"number": 2, "default": "terminate", "description": "Interrupt from keyboard (Ctrl+C)", "catchable": True},
        "SIGQUIT": {"number": 3, "default": "terminate + core dump", "description": "Quit from keyboard (Ctrl+\\)", "catchable": True},
        "SIGILL": {"number": 4, "default": "terminate + core dump", "description": "Illegal instruction", "catchable": True},
        "SIGTRAP": {"number": 5, "default": "terminate + core dump", "description": "Trace/breakpoint trap", "catchable": True},
        "SIGABRT": {"number": 6, "default": "terminate + core dump", "description": "Abort signal from abort(3)", "catchable": True},
        "SIGBUS": {"number": 7, "default": "terminate + core dump", "description": "Bus error (bad memory access)", "catchable": True},
        "SIGFPE": {"number": 8, "default": "terminate + core dump", "description": "Floating-point exception", "catchable": True},
        "SIGKILL": {"number": 9, "default": "terminate", "description": "Kill signal — cannot be caught or ignored", "catchable": False},
        "SIGUSR1": {"number": 10, "default": "terminate", "description": "User-defined signal 1", "catchable": True},
        "SIGSEGV": {"number": 11, "default": "terminate + core dump", "description": "Segmentation fault (invalid memory ref)", "catchable": True},
        "SIGUSR2": {"number": 12, "default": "terminate", "description": "User-defined signal 2", "catchable": True},
        "SIGPIPE": {"number": 13, "default": "terminate", "description": "Broken pipe — write to pipe with no reader", "catchable": True},
        "SIGALRM": {"number": 14, "default": "terminate", "description": "Timer signal from alarm(2)", "catchable": True},
        "SIGTERM": {"number": 15, "default": "terminate", "description": "Termination signal — polite kill request", "catchable": True},
        "SIGCHLD": {"number": 17, "default": "ignore", "description": "Child process stopped or terminated", "catchable": True},
        "SIGCONT": {"number": 18, "default": "continue", "description": "Continue if stopped", "catchable": True},
        "SIGSTOP": {"number": 19, "default": "stop", "description": "Stop process — cannot be caught or ignored", "catchable": False},
        "SIGTSTP": {"number": 20, "default": "stop", "description": "Stop from terminal (Ctrl+Z)", "catchable": True},
        "SIGTTIN": {"number": 21, "default": "stop", "description": "Background process attempts read from terminal", "catchable": True},
        "SIGTTOU": {"number": 22, "default": "stop", "description": "Background process attempts write to terminal", "catchable": True},
        "SIGURG": {"number": 23, "default": "ignore", "description": "Urgent condition on socket", "catchable": True},
        "SIGXCPU": {"number": 24, "default": "terminate + core dump", "description": "CPU time limit exceeded", "catchable": True},
        "SIGXFSZ": {"number": 25, "default": "terminate + core dump", "description": "File size limit exceeded", "catchable": True},
        "SIGWINCH": {"number": 28, "default": "ignore", "description": "Window resize signal", "catchable": True},
    }
    key = str(sig).upper().strip()
    if not key.startswith("SIG"):
        # Try as number
        try:
            num = int(key)
            for k, v in signals.items():
                if v["number"] == num:
                    return {"signal": k, **v}
            return {"error": f"Unknown signal number: {num}"}
        except ValueError:
            key = "SIG" + key
    entry = signals.get(key)
    if not entry:
        return {"error": f"Unknown signal: {sig}"}
    return {"signal": key, **entry}


def process_state(code: str) -> str:
    """Explain a Linux process state code (from ps STAT column)."""
    states = {
        "D": "Uninterruptible sleep (usually I/O)",
        "I": "Idle kernel thread",
        "R": "Running or runnable (on run queue)",
        "S": "Interruptible sleep (waiting for event)",
        "T": "Stopped by job control signal (Ctrl+Z)",
        "t": "Stopped by debugger during tracing",
        "W": "Paging (not valid since Linux 2.6)",
        "X": "Dead (should never be seen)",
        "Z": "Zombie — terminated but not reaped by parent",
        "<": "High-priority (not nice to other users)",
        "N": "Low-priority (nice to other users)",
        "L": "Has pages locked into memory",
        "s": "Session leader",
        "l": "Multi-threaded (using CLONE_THREAD)",
        "+": "In the foreground process group",
    }
    parts = []
    for c in str(code):
        desc = states.get(c)
        if desc:
            parts.append(f"{c}: {desc}")
    return "; ".join(parts) if parts else f"Unknown state: {code}"


def file_type_char(char: str) -> str:
    """Explain the first character of ls -l output."""
    types = {
        "-": "regular file",
        "d": "directory",
        "l": "symbolic link",
        "c": "character device",
        "b": "block device",
        "p": "named pipe (FIFO)",
        "s": "socket",
    }
    return types.get(str(char), f"Unknown type: {char}")


def exit_code_meaning(code: int) -> str:
    """Standard meaning of common exit codes."""
    codes = {
        0: "Success",
        1: "General error",
        2: "Misuse of shell builtins (bash)",
        126: "Command invoked cannot execute (permission problem)",
        127: "Command not found",
        128: "Invalid argument to exit",
        130: "Script terminated by Ctrl+C (128 + SIGINT=2)",
        137: "Process killed by SIGKILL (128 + 9)",
        139: "Segmentation fault (128 + SIGSEGV=11)",
        143: "Terminated by SIGTERM (128 + 15)",
        255: "Exit status out of range",
    }
    c = int(code)
    if c in codes:
        return codes[c]
    if 129 <= c <= 192:
        sig = c - 128
        return f"Killed by signal {sig}"
    return f"Non-standard exit code: {c}"


def special_permissions(bits: str) -> str:
    """Explain setuid/setgid/sticky bit."""
    perms = {
        "setuid": "4000 — file executes as file owner, not caller. On dirs: no effect (Linux).",
        "setgid": "2000 — file executes as group owner. On dirs: new files inherit directory's group.",
        "sticky": "1000 — on dirs: only file owner can delete files (e.g. /tmp). On files: historically keep in swap.",
        "4000": "setuid — file executes as file owner",
        "2000": "setgid — file executes as group owner; on dirs new files inherit group",
        "1000": "sticky bit — on dirs only owner can delete files",
    }
    key = str(bits).lower().strip()
    return perms.get(key, f"Unknown: {bits}. Valid: setuid (4000), setgid (2000), sticky (1000)")


LINUX_FUNCTIONS = {
    "chmod_to_symbolic": chmod_to_symbolic,
    "chmod_to_octal": chmod_to_octal,
    "umask_result": umask_result,
    "umask_for_dirs": umask_for_dirs,
    "signal_info": signal_info,
    "process_state": process_state,
    "file_type_char": file_type_char,
    "exit_code_meaning": exit_code_meaning,
    "special_permissions": special_permissions,
}

LINUX_NL_PATTERNS = [
    (r'chmod\s+(\d{3,4})\s+(?:to|in)\s+(?:symbolic|rwx)', 'chmod_to_symbolic({0})'),
    (r'(?:what is|convert)\s+([rwx-]{9,10})\s+(?:to|in)\s+(?:octal|numeric)', 'chmod_to_octal("{0}")'),
    (r'umask\s+(\d{3,4}).*?(?:result|permission|file)', 'umask_result({0})'),
    (r'(?:what is|explain)\s+(?:signal\s+)?(?:SIG)?(\w+)\s+signal', 'signal_info("{0}")'),
    (r'(?:what does|what is)\s+exit\s+code\s+(\d+)', 'exit_code_meaning({0})'),
    (r'(?:what is|explain)\s+(?:the\s+)?(setuid|setgid|sticky)', 'special_permissions("{0}")'),
]
