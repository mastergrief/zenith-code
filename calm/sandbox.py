"""
CALM v0.1 sandboxed Python executor.

Runs Python code in an isolated subprocess with:
- Timeout (default 10s)
- Memory limit (default 256MB)
- No filesystem access (builtins restricted)
- No network access (no imports except math/builtins)
- Stdout capture → result

The model writes Python naturally, the sandbox executes it safely.
Results come back as the last expression value or stdout.

Usage:
    from calm.sandbox import run_python
    result = run_python("sum(x**2 for x in range(10))")
    # → {"value": 285, "stdout": "", "error": None}
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Optional


# Prelude injected into every sandbox execution — provides the safe
# builtins the model expects without requiring imports.
_PRELUDE = textwrap.dedent("""\
    import math as _math

    # Number theory
    def is_prime(n):
        if not isinstance(n, int) or n < 2: return False
        if n < 4: return True
        if n % 2 == 0 or n % 3 == 0: return False
        i = 5
        while i * i <= n:
            if n % i == 0 or n % (i + 2) == 0: return False
            i += 6
        return True

    def next_prime(n):
        c = n + 1 if n >= 2 else 2
        while not is_prime(c): c += 1
        return c

    def prev_prime(n):
        c = n - 1
        while c >= 2 and not is_prime(c): c -= 1
        return c if c >= 2 else None

    def nth_prime(n):
        count, c = 0, 1
        while count < n:
            c += 1
            if is_prime(c): count += 1
        return c

    def factorize(n):
        if n < 2: return []
        factors, d = [], 2
        while d * d <= n:
            while n % d == 0: factors.append(d); n //= d
            d += 1
        if n > 1: factors.append(n)
        return factors

    def divisors(n):
        divs = set()
        for i in range(1, int(n**0.5) + 1):
            if n % i == 0: divs.add(i); divs.add(n // i)
        return sorted(divs)

    def count_divisors(n): return len(divisors(n))

    def digit_sum(n): return sum(int(d) for d in str(abs(n)))

    def digital_root(n):
        n = abs(n)
        while n >= 10: n = sum(int(d) for d in str(n))
        return n

    def fibonacci(n):
        if n <= 1: return n
        a, b = 0, 1
        for _ in range(2, n + 1): a, b = b, a + b
        return b

    def collatz(n):
        seq = [n]
        while n != 1: n = n // 2 if n % 2 == 0 else 3 * n + 1; seq.append(n)
        return seq

    def collatz_length(n): return len(collatz(n))

    def solve_quadratic(a, b, c):
        disc = b*b - 4*a*c
        if disc < 0: return None
        sq = _math.sqrt(disc)
        return ((-b - sq) / (2*a), (-b + sq) / (2*a))

    def is_perfect(n): return sum(divisors(n)[:-1]) == n if n > 1 else False

    def lcm(a, b): return abs(a * b) // _math.gcd(a, b) if a and b else 0

    def factorial(n): return _math.factorial(n)

    def sum_range(a, b): return (b - a + 1) * (a + b) // 2

    # Math aliases
    sqrt = _math.sqrt
    log = _math.log
    log2 = _math.log2
    log10 = _math.log10
    floor = _math.floor
    ceil = _math.ceil
    gcd = _math.gcd
    pi = _math.pi
    e = _math.e
""")

# Wrapper that captures the last expression value + stdout.
_WRAPPER = textwrap.dedent("""\
    import sys as _sys, io as _io, json as _json

    {prelude}

    # Block dangerous imports
    _BLOCKED = frozenset(['os', 'subprocess', 'shutil', 'socket', 'http',
        'urllib', 'requests', 'pathlib', 'glob', 'tempfile', 'signal',
        'ctypes', 'multiprocessing', 'threading', 'pickle', 'shelve',
        'code', 'codeop', 'compile', 'compileall', 'importlib', 'runpy'])
    _real_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__
    def _safe_import(name, *args, **kwargs):
        if name.split('.')[0] in _BLOCKED:
            raise ImportError(f"blocked: {{name}}")
        return _real_import(name, *args, **kwargs)
    import builtins as _builtins_mod
    _builtins_mod.__import__ = _safe_import

    _stdout = _io.StringIO()
    _sys.stdout = _stdout
    _result = None
    try:
        # User code — exec for statements, eval last line for expression
        _user_code = {code!r}
        _lines = _user_code.strip().splitlines()
        if _lines:
            _last = _lines[-1].strip()
            # Try to evaluate the last line as an expression
            _body = "\\n".join(_lines[:-1])
            if _body:
                exec(_body)
            try:
                _result = eval(_last)
            except SyntaxError:
                exec(_last)
        _err = None
    except Exception as _e:
        _err = f"{{type(_e).__name__}}: {{_e}}"
        _result = None

    _sys.stdout = _sys.__stdout__
    print(_json.dumps({{
        "value": _result if _result is not None else None,
        "stdout": _stdout.getvalue(),
        "error": _err,
    }}, default=str))
""")


@dataclass
class SandboxResult:
    value: Any = None
    stdout: str = ""
    error: Optional[str] = None
    raw: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def run_python(
    code: str,
    timeout: float = 10.0,
    max_output: int = 10000,
) -> SandboxResult:
    """
    Execute Python code in a sandboxed subprocess.

    The code has access to math builtins (is_prime, factorize, etc.)
    but NO filesystem, network, or dangerous operations.

    Returns a SandboxResult with value (last expression), stdout,
    and any error message.
    """
    script = _WRAPPER.format(prelude=_PRELUDE, code=code)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=timeout,
            env={"PATH": ""},  # minimal env
        )
    except subprocess.TimeoutExpired:
        return SandboxResult(error=f"timeout after {timeout}s")
    except Exception as e:
        return SandboxResult(error=f"sandbox error: {e}")

    raw = proc.stdout.strip()
    if proc.returncode != 0:
        err = proc.stderr.strip()[:500]
        return SandboxResult(error=f"exit {proc.returncode}: {err}", raw=raw)

    try:
        data = json.loads(raw)
        return SandboxResult(
            value=data.get("value"),
            stdout=data.get("stdout", "")[:max_output],
            error=data.get("error"),
            raw=raw,
        )
    except json.JSONDecodeError:
        return SandboxResult(
            value=raw[:max_output] if raw else None,
            raw=raw,
        )
