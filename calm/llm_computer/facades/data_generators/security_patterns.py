"""SecurityPatternsGenerator — vulnerability diagnosis + secure fix.

Each example is a known-bad pattern + its canonical remediation,
drawn from OWASP Top 10 and the CWE top 25. This is the same space
the existing `security_kb.py` covers as knowledge, but as NL →
code pairs suitable for DB retrieval and PT training.

Every entry is written to pass the sandbox's allowed-import set when
possible (json, hashlib, secrets are allowed; urllib/socket are
blocked — those rely on `skip_sandbox=True` and AST-only validation).

Design lean toward well-known patterns rather than exotic CVEs so
the DB helps with the everyday security prompts Gemma sees.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


@dataclass
class SecuritySpec:
    vuln: str                  # short name, e.g. "sql_injection"
    problem: str               # user-facing NL question
    signature: str             # the fixed function's def line
    solution: str              # the secure implementation
    test_cases: List[Tuple]
    algorithm: str             # 1-line defense strategy
    complexity: str
    edge_cases: List[str]
    skip_sandbox: bool = False


def _specs() -> List[SecuritySpec]:
    out: List[SecuritySpec] = []

    # ---- SQL injection via parameterized query ----
    out.append(SecuritySpec(
        vuln="sql_injection",
        problem="Write a Python function `build_user_query_params(username)` that returns a (sql_string, params_tuple) tuple safe to pass to sqlite3.execute. Must prevent SQL injection by using parameterized queries — NEVER embed username directly into the SQL string.",
        signature="def build_user_query_params(username):",
        solution=(
            "def build_user_query_params(username):\n"
            "    # Use a placeholder ? — sqlite3 escapes params correctly.\n"
            "    sql = 'SELECT id, email FROM users WHERE username = ?'\n"
            "    return (sql, (username,))\n"
        ),
        test_cases=[
            ("alice", ("SELECT id, email FROM users WHERE username = ?", ("alice",))),
            ("admin' OR '1'='1", ("SELECT id, email FROM users WHERE username = ?", ("admin' OR '1'='1",))),
            ("", ("SELECT id, email FROM users WHERE username = ?", ("",))),
            ("; DROP TABLE users--", ("SELECT id, email FROM users WHERE username = ?", ("; DROP TABLE users--",))),
        ],
        algorithm="parameterized query with ? placeholder; driver escapes",
        complexity="O(1)",
        edge_cases=["classic injection payload (' OR '1'='1)", "quote escape", "empty username", "multi-statement attempt"],
    ))

    # ---- Safe password hashing ----
    out.append(SecuritySpec(
        vuln="weak_password_hash",
        problem="Write a Python function `hash_password(pw)` that returns a secure hash of the password. Must use a slow algorithm (sha256 alone is WRONG for passwords — must salt + stretch). Use hashlib.pbkdf2_hmac with 200000 iterations, 16-byte random salt, sha256. Return (salt_hex, hash_hex) tuple.",
        signature="def hash_password(pw):",
        solution=(
            "def hash_password(pw):\n"
            "    import hashlib, secrets\n"
            "    salt = secrets.token_bytes(16)\n"
            "    dk = hashlib.pbkdf2_hmac('sha256', pw.encode('utf-8'), salt, 200000)\n"
            "    return (salt.hex(), dk.hex())\n"
        ),
        test_cases=[],  # randomness; use skip_sandbox + AST
        algorithm="pbkdf2_hmac('sha256', pw, salt, 200_000) with secrets-generated salt",
        complexity="O(iterations)",
        edge_cases=["empty password still hashed", "unicode passwords encoded UTF-8", "salt must be fresh per-call"],
        skip_sandbox=True,  # non-deterministic; AST-only
    ))

    # ---- Constant-time comparison ----
    out.append(SecuritySpec(
        vuln="timing_attack",
        problem="Write a Python function `safe_compare(a, b)` that returns True iff two byte strings are equal, in CONSTANT TIME. Must not short-circuit on first difference — use hmac.compare_digest.",
        signature="def safe_compare(a, b):",
        solution=(
            "def safe_compare(a, b):\n"
            "    import hmac\n"
            "    if isinstance(a, str):\n"
            "        a = a.encode('utf-8')\n"
            "    if isinstance(b, str):\n"
            "        b = b.encode('utf-8')\n"
            "    return hmac.compare_digest(a, b)\n"
        ),
        test_cases=[
            ("secret", "secret", True),
            ("secret", "SECRET", False),
            ("", "", True),
            ("a", "aa", False),
            (b"abc", b"abc", True),
            (b"abc", b"abd", False),
        ],
        algorithm="hmac.compare_digest (constant-time per-byte XOR reduce)",
        complexity="O(len(max))",
        edge_cases=["length mismatch still constant time", "string vs bytes mix", "empty strings equal"],
    ))

    # ---- Path traversal prevention ----
    out.append(SecuritySpec(
        vuln="path_traversal",
        problem="Write a Python function `safe_file_path(base_dir, requested_name)` that returns an absolute path guaranteed to be INSIDE base_dir. Must reject any '..' traversal or absolute paths in requested_name. Raise ValueError on unsafe input.",
        signature="def safe_file_path(base_dir, requested_name):",
        solution=(
            "def safe_file_path(base_dir, requested_name):\n"
            "    from pathlib import PurePosixPath\n"
            "    rp = PurePosixPath(requested_name)\n"
            "    if rp.is_absolute() or '..' in rp.parts:\n"
            "        raise ValueError('unsafe path')\n"
            "    final = PurePosixPath(base_dir) / rp\n"
            "    # Ensure the joined path is still under base_dir\n"
            "    base_parts = PurePosixPath(base_dir).parts\n"
            "    if final.parts[:len(base_parts)] != base_parts:\n"
            "        raise ValueError('escapes base_dir')\n"
            "    return str(final)\n"
        ),
        test_cases=[
            ("/data", "file.txt", "/data/file.txt"),
            ("/data", "sub/file.txt", "/data/sub/file.txt"),
            ("/data", "a/b/c", "/data/a/b/c"),
        ],
        algorithm="PurePosixPath with absolute + traversal + containment check",
        complexity="O(path_parts)",
        edge_cases=["absolute path in requested_name (reject)", "'..' component anywhere (reject)", "nested subpaths allowed"],
    ))

    # ---- Unsafe deserialization (YAML safe_load) ----
    out.append(SecuritySpec(
        vuln="unsafe_yaml",
        problem="Write a Python function `parse_config(text)` that parses a YAML string safely using json.loads on a JSON subset — we reject YAML entirely because PyYAML's default loader executes arbitrary code. Return the parsed dict, raise ValueError on invalid input.",
        signature="def parse_config(text):",
        solution=(
            "def parse_config(text):\n"
            "    import json\n"
            "    try:\n"
            "        result = json.loads(text)\n"
            "    except json.JSONDecodeError as e:\n"
            "        raise ValueError(f'invalid config: {e}') from e\n"
            "    if not isinstance(result, dict):\n"
            "        raise ValueError('config must be a JSON object')\n"
            "    return result\n"
        ),
        test_cases=[
            ('{\"a\": 1}', {"a": 1}),
            ('{\"nested\": {\"k\": [1, 2, 3]}}', {"nested": {"k": [1, 2, 3]}}),
            ('{}', {}),
        ],
        algorithm="json.loads (no code execution, no tag injection)",
        complexity="O(|text|)",
        edge_cases=["non-object top-level rejected", "invalid JSON raises ValueError"],
    ))

    # ---- Secure random token ----
    out.append(SecuritySpec(
        vuln="insecure_random_token",
        problem="Write a Python function `gen_session_token(n_bytes=32)` that returns a cryptographically secure URL-safe token. Must use `secrets.token_urlsafe`, NEVER random.random() or random.choice over an alphabet.",
        signature="def gen_session_token(n_bytes=32):",
        solution=(
            "def gen_session_token(n_bytes=32):\n"
            "    import secrets\n"
            "    if n_bytes <= 0:\n"
            "        raise ValueError('n_bytes must be positive')\n"
            "    return secrets.token_urlsafe(n_bytes)\n"
        ),
        test_cases=[],
        algorithm="secrets.token_urlsafe (CSPRNG-backed)",
        complexity="O(n_bytes)",
        edge_cases=["n_bytes <= 0 raises", "output is URL-safe base64 (no =)", "not reproducible — no seed"],
        skip_sandbox=True,  # non-deterministic
    ))

    # ---- Open redirect prevention ----
    out.append(SecuritySpec(
        vuln="open_redirect",
        problem="Write a Python function `is_safe_redirect(url, allowed_hosts)` that returns True only if the URL's host is in the allowed_hosts set AND the scheme is http(s). Must reject javascript:, data:, file: schemes AND hosts not in the allow list.",
        signature="def is_safe_redirect(url, allowed_hosts):",
        solution=(
            "def is_safe_redirect(url, allowed_hosts):\n"
            "    from urllib.parse import urlparse\n"
            "    try:\n"
            "        p = urlparse(url)\n"
            "    except Exception:\n"
            "        return False\n"
            "    if p.scheme not in ('http', 'https'):\n"
            "        return False\n"
            "    host = (p.hostname or '').lower()\n"
            "    return host in {h.lower() for h in allowed_hosts}\n"
        ),
        test_cases=[
            ("https://example.com", {"example.com"}, True),
            ("https://evil.com", {"example.com"}, False),
            ("javascript:alert(1)", {"example.com"}, False),
            ("data:text/html,...", {"example.com"}, False),
            ("file:///etc/passwd", {"example.com"}, False),
            ("http://EXAMPLE.COM/path", {"example.com"}, True),
        ],
        algorithm="urlparse → scheme allow + host allow-list",
        complexity="O(1)",
        edge_cases=["scheme-relative URL rejected", "case-insensitive host match", "path in URL ignored for redirect check"],
        skip_sandbox=True,  # sandbox blocks urllib
    ))

    # ---- HTML escape XSS prevention ----
    out.append(SecuritySpec(
        vuln="xss_via_unescaped_output",
        problem="Write a Python function `render_user_message(msg)` that wraps the user message in an HTML <p> tag with ALL user input escaped. Use html.escape — never string concatenation into HTML without escaping.",
        signature="def render_user_message(msg):",
        solution=(
            "def render_user_message(msg):\n"
            "    import html\n"
            "    return f'<p>{html.escape(msg, quote=True)}</p>'\n"
        ),
        test_cases=[
            ("hello", "<p>hello</p>"),
            ("<script>alert(1)</script>", "<p>&lt;script&gt;alert(1)&lt;/script&gt;</p>"),
            ('" onmouseover="alert(1)', '<p>&quot; onmouseover=&quot;alert(1)</p>'),
            ("&amp;", "<p>&amp;amp;</p>"),                    # & itself escapes to &amp;
            ("", "<p></p>"),
        ],
        algorithm="html.escape with quote=True for attr-safety",
        complexity="O(|msg|)",
        edge_cases=["script tag → lt/gt entities", "double-quote → &quot;", "empty message", "already-escaped re-escaped"],
        skip_sandbox=True,  # sandbox blocks html module
    ))

    # ---- Regex DoS prevention ----
    out.append(SecuritySpec(
        vuln="regex_dos",
        problem="Write a Python function `validate_email_simple(s)` that validates an email using a DoS-safe regex. The classic catastrophic-backtracking patterns like `(a+)+` must be avoided. Return True on match, False otherwise.",
        signature="def validate_email_simple(s):",
        solution=(
            "def validate_email_simple(s):\n"
            "    import re\n"
            "    # No nested quantifiers — linear backtrack-safe.\n"
            "    # Simple RFC-ish: local@domain.tld, length-bounded.\n"
            "    if not s or len(s) > 254:\n"
            "        return False\n"
            "    pattern = r'^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\\.[A-Za-z]{2,}$'\n"
            "    return re.match(pattern, s) is not None\n"
        ),
        test_cases=[
            ("alice@example.com", True),
            ("alice+tag@example.co.uk", True),
            ("invalid", False),
            ("@example.com", False),
            ("alice@", False),
            ("alice@example", False),
            ("", False),
            ("a" * 300 + "@ex.com", False),
        ],
        algorithm="character-class + anchored regex, no nested quantifiers",
        complexity="O(|s|) — no exponential backtrack path",
        edge_cases=["length bound (254 chars RFC 5321)", "no nested + or *", "basic RFC subset"],
    ))

    # ---- API key format validation ----
    out.append(SecuritySpec(
        vuln="api_key_format",
        problem="Write a Python function `is_valid_api_key(s)` that validates our API key format: 'sk-' prefix + exactly 32 hex chars. Must use constant-time comparison for the prefix to avoid leaking key structure via timing.",
        signature="def is_valid_api_key(s):",
        solution=(
            "def is_valid_api_key(s):\n"
            "    import hmac, re\n"
            "    if not isinstance(s, str) or len(s) != 35:\n"
            "        return False\n"
            "    if not hmac.compare_digest(s[:3], 'sk-'):\n"
            "        return False\n"
            "    return re.fullmatch(r'[0-9a-f]{32}', s[3:]) is not None\n"
        ),
        test_cases=[
            ("sk-abcdef0123456789abcdef0123456789", True),
            ("sk-ABCDEF0123456789abcdef0123456789", False),     # upper-case hex rejected
            ("pk-abcdef0123456789abcdef0123456789", False),     # wrong prefix
            ("sk-abc", False),                                   # too short
            ("sk-abcdef0123456789abcdef01234567890", False),     # too long
            ("", False),
            (None, False),
        ],
        algorithm="length check + hmac.compare_digest(prefix) + regex(body)",
        complexity="O(|s|)",
        edge_cases=["non-string input", "case-sensitive hex", "constant-time prefix check", "exact length 35"],
    ))

    # ---- CSRF token generation + validation ----
    out.append(SecuritySpec(
        vuln="csrf_token",
        problem="Write a Python function `generate_csrf_token()` that returns a 32-byte URL-safe token using secrets.token_urlsafe. Also write a companion `verify_csrf_token(expected, provided)` using constant-time compare.",
        signature="def generate_csrf_token():",
        solution=(
            "def generate_csrf_token():\n"
            "    import secrets\n"
            "    return secrets.token_urlsafe(32)\n"
            "\n"
            "def verify_csrf_token(expected, provided):\n"
            "    import hmac\n"
            "    if not isinstance(provided, str):\n"
            "        return False\n"
            "    return hmac.compare_digest(expected, provided)\n"
        ),
        test_cases=[],
        algorithm="secrets.token_urlsafe + hmac.compare_digest",
        complexity="O(n)",
        edge_cases=["non-string provided rejected early", "different-length strings still constant-time", "URL-safe base64 output"],
        skip_sandbox=True,   # non-deterministic
    ))

    # ---- Rate limiter (token bucket, in-memory) ----
    out.append(SecuritySpec(
        vuln="rate_limit",
        problem="Write a Python function `make_rate_limiter(max_per_second)` that returns a function `try_consume()` implementing a token bucket. Returns True if a token is available, False if rate-limited. Uses time.monotonic (not time.time, which can jump).",
        signature="def make_rate_limiter(max_per_second):",
        solution=(
            "def make_rate_limiter(max_per_second):\n"
            "    import time\n"
            "    if max_per_second <= 0:\n"
            "        raise ValueError('rate must be positive')\n"
            "    capacity = max_per_second\n"
            "    state = {'tokens': capacity, 't': time.monotonic()}\n"
            "    def try_consume():\n"
            "        now = time.monotonic()\n"
            "        elapsed = now - state['t']\n"
            "        state['tokens'] = min(capacity, state['tokens'] + elapsed * max_per_second)\n"
            "        state['t'] = now\n"
            "        if state['tokens'] >= 1:\n"
            "            state['tokens'] -= 1\n"
            "            return True\n"
            "        return False\n"
            "    return try_consume\n"
        ),
        test_cases=[],
        algorithm="leaky-token-bucket with time.monotonic anchor",
        complexity="O(1) per call",
        edge_cases=["rate <= 0 raises", "monotonic (not wall-clock)", "bucket caps at capacity", "first call always succeeds"],
        skip_sandbox=True,   # time-dependent
    ))

    # ---- TLS cert hostname validation (simplified) ----
    out.append(SecuritySpec(
        vuln="hostname_verify",
        problem="Write a Python function `cert_hostname_matches(cert_hostname, requested)` that returns True if a certificate's hostname matches the requested hostname. Supports wildcard certs (*.example.com matches foo.example.com but NOT foo.bar.example.com nor example.com itself).",
        signature="def cert_hostname_matches(cert_hostname, requested):",
        solution=(
            "def cert_hostname_matches(cert_hostname, requested):\n"
            "    cert = cert_hostname.lower()\n"
            "    req = requested.lower()\n"
            "    if cert == req:\n"
            "        return True\n"
            "    if cert.startswith('*.'):\n"
            "        suffix = cert[2:]    # 'example.com'\n"
            "        # requested must end with '.suffix' AND have exactly one\n"
            "        # extra label before suffix (no sub-sub-domain match).\n"
            "        if req.endswith('.' + suffix):\n"
            "            prefix = req[: -(len(suffix) + 1)]\n"
            "            return '.' not in prefix and prefix != ''\n"
            "    return False\n"
        ),
        test_cases=[
            ("example.com", "example.com", True),
            ("EXAMPLE.COM", "example.com", True),
            ("example.com", "sub.example.com", False),           # not a wildcard
            ("*.example.com", "foo.example.com", True),
            ("*.example.com", "example.com", False),             # bare parent NOT covered
            ("*.example.com", "foo.bar.example.com", False),     # only one label
            ("*.example.com", ".example.com", False),            # empty label
            ("*.example.com", "foo.example.org", False),
        ],
        algorithm="case-fold + wildcard left-label match with single-label enforcement",
        complexity="O(|cert| + |req|)",
        edge_cases=["wildcard covers exactly one label", "bare parent not covered by wildcard", "case-insensitive", "empty left label rejected"],
    ))

    # ---- HMAC signed token ----
    out.append(SecuritySpec(
        vuln="signed_token",
        problem="Write a Python function `sign_payload(payload, secret_key)` that returns a token of the form 'payload:hmac_hex' where the HMAC uses SHA-256. Also `verify_signed_payload(token, secret_key)` that returns payload if valid, else None.",
        signature="def sign_payload(payload, secret_key):",
        solution=(
            "def sign_payload(payload, secret_key):\n"
            "    import hmac, hashlib\n"
            "    mac = hmac.new(secret_key.encode('utf-8'),\n"
            "                   payload.encode('utf-8'),\n"
            "                   hashlib.sha256).hexdigest()\n"
            "    return f'{payload}:{mac}'\n"
            "\n"
            "def verify_signed_payload(token, secret_key):\n"
            "    import hmac, hashlib\n"
            "    if ':' not in token:\n"
            "        return None\n"
            "    payload, mac = token.rsplit(':', 1)\n"
            "    expected = hmac.new(secret_key.encode('utf-8'),\n"
            "                        payload.encode('utf-8'),\n"
            "                        hashlib.sha256).hexdigest()\n"
            "    if hmac.compare_digest(expected, mac):\n"
            "        return payload\n"
            "    return None\n"
        ),
        test_cases=[],
        algorithm="HMAC-SHA256 sign + verify via compare_digest",
        complexity="O(|payload|)",
        edge_cases=["tampered payload invalidates", "missing ':' → None", "constant-time compare", "payload may contain ':' (use rsplit)"],
        skip_sandbox=True,   # module inter-function dep
    ))

    # ---- Command injection prevention ----
    out.append(SecuritySpec(
        vuln="command_injection",
        problem="Write a Python function `run_git_log_args(path)` that returns a (cmd_list, safe_path) tuple for use with subprocess.run(..., shell=False). Must never use shell=True. The path must be passed as a single argv element, not concatenated into a shell command.",
        signature="def run_git_log_args(path):",
        solution=(
            "def run_git_log_args(path):\n"
            "    from pathlib import PurePosixPath\n"
            "    # Path goes in argv — never in a shell string. Reject\n"
            "    # NUL bytes and strip trailing slash for consistency.\n"
            "    if '\\x00' in path:\n"
            "        raise ValueError('null byte in path')\n"
            "    safe = str(PurePosixPath(path))\n"
            "    cmd = ['git', '-C', safe, 'log', '--oneline', '-n', '10']\n"
            "    return (cmd, safe)\n"
        ),
        test_cases=[
            ("/repo", (["git", "-C", "/repo", "log", "--oneline", "-n", "10"], "/repo")),
            ("/repo/sub", (["git", "-C", "/repo/sub", "log", "--oneline", "-n", "10"], "/repo/sub")),
            (".", (["git", "-C", ".", "log", "--oneline", "-n", "10"], ".")),
        ],
        algorithm="argv list (never shell=True); validate & normalize path",
        complexity="O(|path|)",
        edge_cases=["null byte rejected", "argv form prevents shell metacharacter injection", "shell=True would be exploitable"],
    ))

    return out


class SecurityPatternsGenerator(DomainDataGenerator):
    """OWASP-adjacent secure-coding patterns. Diagnosis + remediation
    pairs for the common vulnerabilities Gemma needs to avoid generating."""

    name = "security"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._specs = _specs()

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        out: List[VerifiedExample] = []
        self.rng.shuffle(self._specs)
        for s in self._specs[:n]:
            out.append(VerifiedExample(
                problem=s.problem,
                signature=s.signature,
                solution=s.solution,
                test_cases=list(s.test_cases),
                reasoning="",   # base synthesizes 5-step from fields
                algorithm=s.algorithm,
                complexity=s.complexity,
                edge_cases=list(s.edge_cases),
                category=f"security_{s.vuln}",
                generator_name=self.name,
                skip_sandbox=s.skip_sandbox,
                metadata={"vuln": s.vuln},
            ))
        return out


register_generator("security", SecurityPatternsGenerator)
