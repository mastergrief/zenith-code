"""
CALM security backend — deterministic vulnerability detection.

AST + regex analysis for common security issues. The LLM identifies
suspicious patterns; this module verifies them with real analysis.

Usage in <calm> blocks:
    security.audit("auth.py")           # full scan, all checks
    security.sql_injection("db.py")     # SQL injection vectors
    security.xss("template.py")        # XSS via unsanitized output
    security.secrets("config.py")       # hardcoded credentials
    security.unsafe_exec("handler.py")  # eval/exec/subprocess
    security.path_traversal("api.py")   # path injection
    security.crypto("auth.py")          # weak crypto (MD5, SHA1)
    security.permissions("app.py")      # overly permissive operations
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Dict, List

from calm.stack_vm import (
    Backend,
    CalmRuntimeError,
    Dispatcher,
    Instruction,
    VMState,
    _pop_n,
)


def _read_file(path_str: str) -> tuple:
    """Read file, return (content, lines) or raise."""
    p = Path(path_str)
    if not p.exists():
        raise CalmRuntimeError(f"security: file not found: {p}")
    content = p.read_text(encoding="utf-8", errors="replace")
    return content, content.splitlines()


# ---------------------------------------------------------------------------
# SQL Injection
# ---------------------------------------------------------------------------

_SQL_PATTERNS = [
    # String formatting in SQL
    (r'(?:execute|cursor\.execute|query|raw)\s*\(\s*["\'].*%[sd]', "format string in SQL"),
    (r'(?:execute|cursor\.execute|query|raw)\s*\(\s*f["\']', "f-string in SQL"),
    (r'(?:execute|cursor\.execute|query|raw)\s*\(\s*["\'].*\+', "string concat in SQL"),
    (r'(?:execute|cursor\.execute|query|raw)\s*\(\s*["\'].*\.format\(', ".format() in SQL"),
    # Django/SQLAlchemy raw queries
    (r'\.raw\s*\(.*%', "raw query with format"),
    (r'\.extra\s*\(.*where.*%', "extra() with format"),
    (r'text\s*\(\s*f["\']', "SQLAlchemy text() with f-string"),
]


def _check_sql_injection(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        for pattern, desc in _SQL_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append({"line": i, "severity": "HIGH", "type": "sql_injection",
                                "detail": desc, "code": line.strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# XSS
# ---------------------------------------------------------------------------

_XSS_PATTERNS = [
    (r'\.innerHTML\s*=', "innerHTML assignment"),
    (r'document\.write\s*\(', "document.write()"),
    (r'dangerouslySetInnerHTML', "React dangerouslySetInnerHTML"),
    (r'Markup\s*\(.*\+', "Jinja2 Markup with concatenation"),
    (r'\|\s*safe\b', "Django/Jinja2 |safe filter"),
    (r'render_template_string\s*\(', "Flask render_template_string"),
    (r'HttpResponse\s*\(.*\+', "Django HttpResponse with concatenation"),
]


def _check_xss(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        for pattern, desc in _XSS_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line": i, "severity": "HIGH", "type": "xss",
                                "detail": desc, "code": line.strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# Hardcoded Secrets
# ---------------------------------------------------------------------------

_SECRET_PATTERNS = [
    (r'(?:password|passwd|pwd)\s*=\s*["\'][^"\']+["\']', "hardcoded password"),
    (r'(?:api_key|apikey|api_secret)\s*=\s*["\'][^"\']+["\']', "hardcoded API key"),
    (r'(?:secret_key|SECRET_KEY)\s*=\s*["\'][^"\']{8,}["\']', "hardcoded secret key"),
    (r'(?:token|TOKEN)\s*=\s*["\'][A-Za-z0-9_\-]{20,}["\']', "hardcoded token"),
    (r'(?:aws_access_key|AWS_ACCESS)\s*=\s*["\']AK', "AWS access key"),
    (r'(?:private_key|PRIVATE_KEY)\s*=\s*["\']-----BEGIN', "hardcoded private key"),
    (r'(?:mongodb|mysql|postgres)://[^:]+:[^@]+@', "database connection string with credentials"),
]


def _check_secrets(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        # Skip comments and docstrings
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        for pattern, desc in _SECRET_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append({"line": i, "severity": "CRITICAL", "type": "hardcoded_secret",
                                "detail": desc, "code": line.strip()[:60] + "..."})
    return findings


# ---------------------------------------------------------------------------
# Unsafe Execution
# ---------------------------------------------------------------------------

def _check_unsafe_exec(content: str, lines: list) -> List[dict]:
    findings = []
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return findings

    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            name = ""
            if isinstance(node.func, ast.Name):
                name = node.func.id
            elif isinstance(node.func, ast.Attribute):
                name = node.func.attr

            if name in ("eval", "exec"):
                findings.append({"line": node.lineno, "severity": "HIGH",
                                "type": "unsafe_exec", "detail": f"{name}() call",
                                "code": lines[node.lineno - 1].strip()[:100]})
            elif name in ("system", "popen", "call", "run", "Popen"):
                # Check if it's os.system, subprocess.run, etc.
                if isinstance(node.func, ast.Attribute):
                    findings.append({"line": node.lineno, "severity": "MEDIUM",
                                    "type": "command_injection",
                                    "detail": f"subprocess/os.{name}() — verify input sanitization",
                                    "code": lines[node.lineno - 1].strip()[:100]})
            elif name == "pickle" or name in ("loads", "load"):
                if isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
                    if node.func.value.id == "pickle":
                        findings.append({"line": node.lineno, "severity": "HIGH",
                                        "type": "insecure_deserialization",
                                        "detail": "pickle deserialization — arbitrary code execution",
                                        "code": lines[node.lineno - 1].strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# Path Traversal
# ---------------------------------------------------------------------------

_PATH_PATTERNS = [
    (r'open\s*\(.*\+', "open() with string concatenation"),
    (r'open\s*\(.*f["\']', "open() with f-string"),
    (r'(?:send_file|send_from_directory)\s*\(.*\+', "file send with concatenation"),
    (r'os\.path\.join\s*\(.*request', "os.path.join with user input"),
]


def _check_path_traversal(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        for pattern, desc in _PATH_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line": i, "severity": "MEDIUM", "type": "path_traversal",
                                "detail": desc, "code": line.strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# Weak Crypto
# ---------------------------------------------------------------------------

_CRYPTO_PATTERNS = [
    (r'\bmd5\b', "MD5 — broken for passwords, use bcrypt/argon2"),
    (r'\bsha1\b', "SHA1 — weak, use SHA-256+"),
    (r'\bDES\b', "DES — broken, use AES"),
    (r'\bRC4\b', "RC4 — broken"),
    (r'random\.(random|randint|choice|uniform)', "random module — not cryptographically secure"),
    (r'hashlib\.md5\s*\(', "hashlib.md5 — not for passwords"),
    (r'hashlib\.sha1\s*\(', "hashlib.sha1 — use sha256+"),
]


def _check_crypto(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        stripped = line.strip()
        if stripped.startswith("#") or stripped.startswith("//"):
            continue
        for pattern, desc in _CRYPTO_PATTERNS:
            if re.search(pattern, line, re.IGNORECASE):
                findings.append({"line": i, "severity": "MEDIUM", "type": "weak_crypto",
                                "detail": desc, "code": line.strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# Permissions
# ---------------------------------------------------------------------------

_PERM_PATTERNS = [
    (r'chmod\s*\(\s*["\']?\s*0?777', "chmod 777 — world writable"),
    (r'chmod\s*\(\s*["\']?\s*0?666', "chmod 666 — world writable"),
    (r'DEBUG\s*=\s*True', "DEBUG mode enabled"),
    (r'CORS_ALLOW_ALL|allow_all_origins|AllowAny', "overly permissive CORS/auth"),
    (r'verify\s*=\s*False', "SSL verification disabled"),
    (r'ALLOWED_HOSTS\s*=\s*\[\s*["\']\*["\']', "Django ALLOWED_HOSTS = ['*']"),
]


def _check_permissions(content: str, lines: list) -> List[dict]:
    findings = []
    for i, line in enumerate(lines, 1):
        for pattern, desc in _PERM_PATTERNS:
            if re.search(pattern, line):
                findings.append({"line": i, "severity": "MEDIUM", "type": "permissions",
                                "detail": desc, "code": line.strip()[:100]})
    return findings


# ---------------------------------------------------------------------------
# Full Audit
# ---------------------------------------------------------------------------

def _audit(content: str, lines: list) -> dict:
    """Run all checks and return a structured report."""
    all_findings = []
    all_findings.extend(_check_sql_injection(content, lines))
    all_findings.extend(_check_xss(content, lines))
    all_findings.extend(_check_secrets(content, lines))
    all_findings.extend(_check_unsafe_exec(content, lines))
    all_findings.extend(_check_path_traversal(content, lines))
    all_findings.extend(_check_crypto(content, lines))
    all_findings.extend(_check_permissions(content, lines))

    by_severity = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for f in all_findings:
        by_severity[f["severity"]] = by_severity.get(f["severity"], 0) + 1

    return {
        "file": None,  # set by caller
        "total_findings": len(all_findings),
        "by_severity": {k: v for k, v in by_severity.items() if v > 0},
        "findings": all_findings[:20],  # cap at 20
    }


# ---------------------------------------------------------------------------
# Stack VM integration
# ---------------------------------------------------------------------------

def _make_check(check_fn, check_name):
    def _fn(state: VMState, instr: Instruction) -> None:
        (path_str,) = _pop_n(state, 1, f"security.{check_name}")
        content, lines = _read_file(str(path_str))
        state.stack.append(check_fn(content, lines))
    return _fn


def _b_audit(state: VMState, instr: Instruction) -> None:
    (path_str,) = _pop_n(state, 1, "security.audit")
    content, lines = _read_file(str(path_str))
    result = _audit(content, lines)
    result["file"] = str(path_str)
    state.stack.append(result)


SECURITY_WORDS: Dict[str, Backend] = {
    "security.audit": _b_audit,
    "security.sql_injection": _make_check(_check_sql_injection, "sql_injection"),
    "security.xss": _make_check(_check_xss, "xss"),
    "security.secrets": _make_check(_check_secrets, "secrets"),
    "security.unsafe_exec": _make_check(_check_unsafe_exec, "unsafe_exec"),
    "security.path_traversal": _make_check(_check_path_traversal, "path_traversal"),
    "security.crypto": _make_check(_check_crypto, "crypto"),
    "security.permissions": _make_check(_check_permissions, "permissions"),
}


def register(dispatcher: Dispatcher) -> None:
    for name, fn in SECURITY_WORDS.items():
        dispatcher.register_backend(name, fn)
