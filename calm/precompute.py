"""
Auto-CALM Layer 2 — precomputation and system prompt.

Extracts computable expressions from prompts, evaluates them before the
model responds, and injects verified facts into the system prompt.
Handles pronoun resolution ("is it prime?" → links to prior value).

Usage:
    from calm.precompute import precompute, build_system_prompt
    facts = precompute("What is fibonacci(30)? Is it prime?")
    # → {'fibonacci(30)': 832040, 'is_prime(832040)': False}
"""

from __future__ import annotations

import re
from typing import Dict

from calm.expression import safe_eval, ExpressionError


def build_system_prompt() -> str:
    """Build system prompt listing available verified backends.
    Auto-discovers all registered functions — no hardcoded lists."""
    from calm.expression import _FUNCTIONS

    # Auto-discover categories from backend module names.
    # Each backend exports *_FUNCTIONS from *_ops.py — the module
    # name IS the category.
    categories = {}

    # Discover by scanning which backend module each function came from.
    _MODULE_TO_CAT = {}
    _backend_modules = [
        ("calm.backends.math_ops", "math"),
        ("calm.backends.string_ops", "strings"),
        ("calm.backends.wasm_ops", "wasm"),
        ("calm.backends.code_ops", "code"),
        ("calm.backends.security_ops", "security"),
        ("calm.backends.date_ops", "dates"),
        ("calm.backends.convert_ops", "units"),
        ("calm.backends.data_ops", "stats"),
        ("calm.backends.algo_ops", "algorithms"),
        ("calm.backends.quality_ops", "quality"),
        ("calm.backends.readability_ops", "readability"),
        ("calm.backends.regex_ops", "regex"),
        ("calm.backends.json_ops", "json"),
        ("calm.backends.encoding_ops", "encoding"),
        ("calm.backends.git_ops", "git"),
        ("calm.backends.network_ops", "network"),
    ]
    for mod_name, cat in _backend_modules:
        try:
            mod = __import__(mod_name, fromlist=["x"])
            # Find the *_FUNCTIONS dict in the module.
            for attr in dir(mod):
                if attr.endswith("_FUNCTIONS") and isinstance(getattr(mod, attr), dict):
                    for func_name in getattr(mod, attr):
                        _MODULE_TO_CAT[func_name] = cat
        except ImportError:
            pass

    # Built-in expression.py functions.
    _BUILTINS = {'len', 'sorted', 'reversed', 'sum', 'any', 'all',
                 'zip', 'range', 'map_expr', 'filter_expr',
                 'find_int', 'count_if', 'sqrt', 'pow', 'abs',
                 'floor', 'ceil', 'log', 'log2', 'log10', 'pi', 'e',
                 'min', 'max', 'round'}

    for name in sorted(_FUNCTIONS.keys()):
        if '.' in name:
            cat = name.split('.')[0]
        elif name in _MODULE_TO_CAT:
            cat = _MODULE_TO_CAT[name]
        elif name in _BUILTINS:
            cat = 'math'
        else:
            cat = 'math'  # default for expression.py builtins
        categories.setdefault(cat, []).append(name)

    # Skip noisy categories.
    skip = {'wasm'}
    lines = [
        f"  {cat}: {', '.join(funcs)}"
        for cat, funcs in sorted(categories.items())
        if cat not in skip
    ]

    return (
        "You are a helpful assistant with verified compute backends.\n"
        "Every computation you state will be checked by a CPU engine.\n"
        "Write naturally — use function names when precise answers matter.\n"
        "The engine verifies your claims and corrects any errors.\n\n"
        "Available verified functions:\n" + "\n".join(lines)
    )


# NL → expression patterns for "What is X?" style prompts.
_NL_PATTERNS = [
    (r'the (\d+)(?:st|nd|rd|th) [Ff]ibonacci number', r'fibonacci(\1)'),
    (r'fibonacci\((\d+)\)', r'fibonacci(\1)'),
    (r'factorial\((\d+)\)', r'factorial(\1)'),
    (r'the (\d+)(?:st|nd|rd|th) prime', r'nth_prime(\1)'),
    (r'(?:the )?length of the [Cc]ollatz .+ (\d+)', r'collatz_length(\1)'),
    (r'(?:how long|length).*[Cc]ollatz.*from (\d+)', r'collatz_length(\1)'),
    (r'the [Cc]ollatz sequence (?:starting |)from (\d+)', r'collatz_length(\1)'),
    (r'the digit(?:al)? (?:sum|root) of (\d+)', None),
    (r'the smallest prime (?:greater than|>) (\d+)', r'next_prime(\1)'),
    (r'the (?:prime )?factors of (\d+)', r'factorize(\1)'),
    (r'the GCD of (\d+) and (\d+)', r'gcd(\1, \2)'),
    (r'the LCM of (\d+) and (\d+)', r'lcm(\1, \2)'),
]

# Extended NL patterns for dates, conversions, stats.
_EXTRA_NL_PATTERNS = [
    (r'convert\s+([\d.]+)\s+(\w+)\s+to\s+(\w+)', 'convert'),
    (r'([\d.]+)\s+(?:celsius|C)\s+(?:to|in)\s+(?:fahrenheit|F)', 'c_to_f'),
    (r'([\d.]+)\s+(?:fahrenheit|F)\s+(?:to|in)\s+(?:celsius|C)', 'f_to_c'),
    (r'[Ii]s\s+(\d{4})\s+a?\s*leap\s+year', 'leap'),
    (r'days?\s+between\s+([\d/-]+)\s+and\s+([\d/-]+)', 'days_between'),
    (r'(?:what|which)\s+day.*?(\d{4}-\d{2}-\d{2})', 'day_of_week'),
    (r'(\d+)\s+choose\s+(\d+)', 'choose'),
    (r'[Cc]\((\d+)\s*,\s*(\d+)\)', 'choose'),
    (r'(?:mean|average)\s+of\s+\[([\d,.\s]+)\]', 'mean'),
    (r'median\s+of\s+\[([\d,.\s]+)\]', 'median'),
]


def _normalize_expr(expr: str) -> str:
    """Normalize raw expression for safe_eval."""
    expr = expr.strip()
    expr = expr.replace("\\times", "*").replace("\\cdot", "*").replace("\\div", "/")
    expr = expr.replace("×", "*").replace("÷", "/")
    expr = expr.replace("^", "**").replace(",", "")
    return re.sub(r'\s+', ' ', expr)


def precompute(prompt: str) -> Dict[str, object]:
    """Extract and evaluate computations from a prompt."""
    results = {}

    # "What is X?" / "Compute X?" / "Calculate X?"
    for pat in [r'[Ww]hat is (.+?)[\?\.]', r'[Cc]ompute (.+?)[\?\.]',
                r'[Cc]alculate (.+?)[\?\.]', r'[Ff]ind (.+?)[\?\.]']:
        m = re.search(pat, prompt)
        if not m:
            continue
        raw = m.group(1).strip()
        expr = None
        for nl_pat, nl_repl in _NL_PATTERNS:
            nl_m = re.search(nl_pat, raw, re.IGNORECASE)
            if nl_m and nl_repl:
                expr = re.sub(nl_pat, nl_repl, raw, flags=re.IGNORECASE)
                break
        if not expr:
            expr = _normalize_expr(raw)
        try:
            results[expr] = safe_eval(expr)
        except ExpressionError:
            pass

    # NL patterns outside "What is" form.
    for nl_pat, nl_repl in _NL_PATTERNS:
        if nl_repl:
            for nl_m in re.finditer(nl_pat, prompt, re.IGNORECASE):
                expr = re.sub(nl_pat, nl_repl, nl_m.group(0), flags=re.IGNORECASE)
                if expr not in results:
                    try:
                        results[expr] = safe_eval(expr)
                    except ExpressionError:
                        pass

    # Extended patterns (dates, conversions, stats).
    for pat, kind in _EXTRA_NL_PATTERNS:
        for m in re.finditer(pat, prompt, re.IGNORECASE):
            expr = None
            try:
                if kind == 'convert':
                    expr = f'convert({m.group(1)}, "{m.group(2)}", "{m.group(3)}")'
                elif kind == 'c_to_f':
                    expr = f'celsius_to_fahrenheit({m.group(1)})'
                elif kind == 'f_to_c':
                    expr = f'fahrenheit_to_celsius({m.group(1)})'
                elif kind == 'leap':
                    expr = f'is_leap_year({m.group(1)})'
                elif kind == 'days_between':
                    expr = f'days_between("{m.group(1)}", "{m.group(2)}")'
                elif kind == 'day_of_week':
                    expr = f'day_of_week("{m.group(1)}")'
                elif kind == 'choose':
                    expr = f'nCr({m.group(1)}, {m.group(2)})'
                elif kind == 'mean':
                    expr = f'mean([{m.group(1)}])'
                elif kind == 'median':
                    expr = f'median([{m.group(1)}])'
            except (IndexError, AttributeError):
                continue
            if expr and expr not in results:
                try:
                    results[expr] = safe_eval(expr)
                except ExpressionError:
                    pass

    # Pronoun resolution: "Is it prime?" → link to first precomputed value.
    if results:
        primary = list(results.values())[0]
        if isinstance(primary, (int, float)):
            pv = int(primary) if isinstance(primary, float) and primary == int(primary) else primary
            _pronoun_checks = [
                (r'[Ii]s\s+(?:it|the\s+result|this)\s+(?:a\s+)?prime', f"is_prime({pv})"),
                (r'(?:how many|number of)\s+divisors', f"count_divisors({pv})"),
                (r'[Ii]s\s+(?:it|the\s+result)\s+divisible\s+by\s+(\d+)', None),
                (r'digit\s+sum', f"digit_sum({pv})"),
                (r'(?:factor|factori[sz]e)', f"factorize({pv})"),
            ]
            for check_pat, check_expr in _pronoun_checks:
                m = re.search(check_pat, prompt, re.IGNORECASE)
                if not m:
                    continue
                if check_expr is None:
                    # Divisibility — needs the divisor from the match.
                    check_expr = f"{pv} % {m.group(1)} == 0"
                if check_expr not in results:
                    try:
                        results[check_expr] = safe_eval(check_expr)
                    except ExpressionError:
                        pass

    # Boolean precomputes: "Is X prime?", "Is X perfect?", etc.
    bool_pats = [
        (r'[Ii]s\s+(\d[\d,]*)\s+(?:a\s+)?prime', lambda n: f"is_prime({n})"),
        (r'[Ii]s\s+(\d[\d,]*)\s+(?:a\s+)?perfect', lambda n: f"is_perfect({n})"),
        (r'[Ii]s\s+(\d[\d,]*)\s+divisible\s+by\s+(\d+)', None),
    ]
    for pat_info in bool_pats:
        pat = pat_info[0]
        for m in re.finditer(pat, prompt):
            if pat_info[1] is None:
                n, d = m.group(1).replace(",", ""), m.group(2)
                expr = f"{n} % {d} == 0"
            else:
                n = m.group(1).replace(",", "")
                expr = pat_info[1](n)
            if expr not in results:
                try:
                    results[expr] = safe_eval(expr)
                except ExpressionError:
                    pass

    # Auto-discovery: scan the prompt for any registered function calls.
    # If the prompt contains "sha256("hello")" or "url_parse(...)" and
    # that function exists in the registry, evaluate it.
    # This means new backends are automatically precomputable — zero config.
    from calm.expression import _FUNCTIONS
    for func_name in _FUNCTIONS:
        if '.' in func_name:
            continue  # Skip dotted names (code.read, security.audit)
        # Look for function_name("arg") or function_name(arg) in prompt.
        pat = re.escape(func_name) + r'\s*\(([^)]*)\)'
        for m in re.finditer(pat, prompt):
            expr = f"{func_name}({m.group(1)})"
            if expr not in results:
                try:
                    results[expr] = safe_eval(expr)
                except ExpressionError:
                    pass

    # Also detect NL references to backend functions.
    # "base64 encoding of X" → base64_encode("X")
    # "SHA-256 hash of X" → sha256("X")
    # "parse this URL: X" → url_parse("X")
    _NL_FUNC_MAP = [
        (r'base64\s+(?:encoding|encode)\s+(?:of\s+)?["\']([^"\']+)["\']', 'base64_encode'),
        (r'(?:SHA-?256|sha256)\s+(?:hash|digest)\s+(?:of\s+)?["\']([^"\']+)["\']', 'sha256'),
        (r'(?:MD5|md5)\s+(?:hash|digest)\s+(?:of\s+)?["\']([^"\']+)["\']', 'md5'),
        (r'(?:parse|breakdown)\s+(?:this\s+)?(?:URL|url)[:\s]+(\S+)', 'url_parse'),
        (r'(?:HTTP|http)\s+(?:status|code)\s+(?:is\s+)?(\d{3})', 'http_status'),
        (r'(?:what|which)\s+(?:HTTP|http)\s+(?:status|code)\s+is\s+(\d{3})', 'http_status'),
        (r'(?:valid|validate)\s+(?:this\s+)?(?:IP|ip)[:\s]+([\d.]+)', 'is_valid_ip'),
        (r'(?:valid|validate)\s+(?:this\s+)?email[:\s]+(\S+)', 'is_valid_email'),
        (r'[Ii]s\s+(\S+@\S+\.\S+)\s+(?:a\s+)?valid\s+email', 'is_valid_email'),
        (r'(?:regex|pattern)\s+["\']([^"\']+)["\']\s+match(?:es)?\s+["\']([^"\']+)["\']', None),
    ]
    for pat, func in _NL_FUNC_MAP:
        for m in re.finditer(pat, prompt, re.IGNORECASE):
            try:
                if func is None and 'regex' in pat:
                    # Special: regex test with two args.
                    expr = f'regex_test("{m.group(1)}", "{m.group(2)}")'
                elif func in ('http_status',):
                    expr = f'{func}({m.group(1)})'
                else:
                    arg = m.group(1)
                    expr = f'{func}("{arg}")'
                if expr not in results:
                    results[expr] = safe_eval(expr)
            except (ExpressionError, IndexError):
                pass

    return results
