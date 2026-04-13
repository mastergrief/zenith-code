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

    # Auto-discover categories from the backend registry.
    from calm.backends import FUNCTION_CATEGORIES
    categories = {}

    # Built-in expression.py functions.
    _BUILTINS = {'len', 'sorted', 'reversed', 'sum', 'any', 'all',
                 'zip', 'range', 'map_expr', 'filter_expr',
                 'find_int', 'count_if', 'sqrt', 'pow', 'abs',
                 'floor', 'ceil', 'log', 'log2', 'log10', 'pi', 'e',
                 'min', 'max', 'round'}

    for name in sorted(_FUNCTIONS.keys()):
        if '.' in name:
            cat = name.split('.')[0]
        elif name in FUNCTION_CATEGORIES:
            cat = FUNCTION_CATEGORIES[name]
        elif name in _BUILTINS:
            cat = 'math'
        else:
            cat = 'math'  # default for expression.py builtins
        categories.setdefault(cat, []).append(name)

    # Compact listing: category + count, not every function name.
    # Full function names are too noisy for 4B models (~968 tokens).
    # The precompute auto-discovery handles function resolution anyway.
    skip = {'wasm'}
    lines = []
    for cat, funcs in sorted(categories.items()):
        if cat in skip:
            continue
        # Show top 5 functions + count.
        shown = funcs[:5]
        remaining = len(funcs) - len(shown)
        suffix = f" +{remaining} more" if remaining > 0 else ""
        lines.append(f"  {cat} ({len(funcs)}): {', '.join(shown)}{suffix}")

    return (
        "You are a helpful assistant. Answer questions in natural language.\n"
        "State computed results directly in your text. Do NOT use tool calls.\n"
        "A compute engine verifies your claims and corrects any errors.\n\n"
        "You have access to verified functions in these domains:\n" + "\n".join(lines)
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
    # GCD/LCM — support function calls as args: "GCD of fibonacci(10) and fibonacci(15)"
    (r'the GCD of (\w+\(\d+\)|\d+) and (\w+\(\d+\)|\d+)', r'gcd(\1, \2)'),
    (r'the LCM of (\w+\(\d+\)|\d+) and (\w+\(\d+\)|\d+)', r'lcm(\1, \2)'),
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

    # Search/filter precomputes: "find all X where/whose Y"
    # These map to list comprehensions with the registered functions.
    search_pats = [
        # "all primes under N whose digit sum is also prime"
        (r'(?:find |list )?all\s+primes?\s+(?:under|below|less than)\s+(\d+)\s+whose\s+digit\s+sum\s+is\s+(?:also\s+)?prime',
         lambda m: f'[p for p in range(2, {m.group(1)}) if is_prime(p) and is_prime(digit_sum(p))]'),
        # "all primes between N and M"
        (r'(?:find |list )?all\s+primes?\s+between\s+(\d+)\s+and\s+(\d+)',
         lambda m: f'[p for p in range({m.group(1)}, {int(m.group(2))+1}) if is_prime(p)]'),
        # "all divisors of N"
        (r'(?:find |list )?all\s+divisors?\s+of\s+(\d+)',
         lambda m: f'divisors({m.group(1)})'),
        # "all fibonacci numbers under N"
        (r'(?:find |list )?all\s+fibonacci\s+numbers?\s+(?:under|below)\s+(\d+)',
         lambda m: f'[fibonacci(i) for i in range(1, 50) if fibonacci(i) < {m.group(1)}]'),
    ]
    for pat, builder in search_pats:
        m_search = re.search(pat, prompt, re.IGNORECASE)
        if m_search:
            expr = builder(m_search)
            if expr not in results:
                try:
                    results[expr] = safe_eval(expr)
                except ExpressionError:
                    pass

    # Backend-declared NL patterns: auto-collected from *_NL_PATTERNS in backends.
    # Each pattern is (compiled_regex, template) where template uses {0}, {1}, etc.
    # This is the scalable path — backends declare their own NL triggers.
    from calm.backends import NL_PATTERNS
    for compiled_pat, template in NL_PATTERNS:
        if template is None:
            continue  # signal-only pattern, no evaluable expression
        for m in compiled_pat.finditer(prompt):
            try:
                expr = template.format(*m.groups())
                if expr not in results:
                    results[expr] = safe_eval(expr)
            except (ExpressionError, IndexError, KeyError):
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
        # Color ops
        (r'(?:WCAG|contrast|combination).*?([#][0-9a-fA-F]{3,8})\s+(?:and|on|vs\.?|over|against)\s+([#][0-9a-fA-F]{3,8})', None),
        (r'(?:convert|change)\s+([#\w]+)\s+(?:to|into)\s+(?:RGB|rgb)', None),
        (r'(?:complementary|complement)\s+(?:color\s+)?(?:of|for)\s+([#\w]+)', None),
        (r'(?:lighten|darken)\s+([#\w]+)\s+(?:by\s+)?(\d+)%?', None),
        # UUID ops
        (r'[Ii]s\s+["\']?([0-9a-fA-F-]{36})["\']?\s+(?:a\s+)?valid\s+UUID', 'uuid_validate'),
        # MIME types
        (r'(?:MIME|mime)\s+type\s+(?:for|of)\s+\.?(\w+)', 'mime_type'),
        (r'(?:what|which)\s+(?:MIME|mime)\s+type\s+(?:does|is|for)\s+\.?(\w+)', 'mime_type'),
        # Base conversion
        # "convert binary number X to hex/decimal" — binary-aware (group 1 = number, group 2 = target base)
        (r'(?:convert\s+)?(?:the\s+)?binary\s+(?:number\s+)?([01]+)\s+(?:to|in(?:to)?)\s+(hex|hexadecimal|decimal|base\s*10|octal)', None),
        # Generic decimal → base
        (r'(?:convert\s+)?(\d+)\s+(?:to|in)\s+binary', None),
        (r'(?:convert\s+)?(\d+)\s+(?:to|in)\s+(?:hex|hexadecimal)', None),
        (r'(?:convert\s+)?(\d+)\s+(?:to|in)\s+octal', None),
        (r'(?:what is\s+)?([01]+)\s+in\s+(?:decimal|base\s*10)', None),
        # Byte sizes (3-group pattern first so it matches before 1-group)
        (r'(?:how (?:many|much)|what is|convert)\s+(\d+)\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s+(?:to|in)\s+(bytes|B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)', None),
        (r'(\d+)\s*(KB|MB|GB|TB|KiB|MiB|GiB|TiB)\s+(?:to|in)\s+(bytes|B|KB|MB|GB|TB|KiB|MiB|GiB|TiB)', None),
        # Duration
        (r'(?:how many|convert)\s+(\d+)\s*(hours?|minutes?|days?|weeks?)\s+(?:to|in|into)\s+(seconds?|minutes?|hours?|days?)', None),
        (r'(?:how many)\s+seconds?\s+in\s+(\d+)\s*(hours?|minutes?|days?|weeks?)', None),
        (r'(?:parse|what is)\s+["\']?(\d+[hms]\s*(?:\d+[hms]\s*)*)["\']?\s+in\s+seconds', None),
        # Checksum / Luhn
        (r'(?:is\s+)?(\d{13,19})\s+(?:a\s+)?valid\s+(?:credit\s+card|Luhn|card\s+number)', None),
        (r'(?:validate|check|verify)\s+(?:ISBN|isbn)[- ]?(?:13)?[:\s]+([0-9X-]+)', None),
        # Timezone
        (r'(?:convert|what is)\s+(\d{1,2}:\d{2})\s+([A-Za-z/_]+)\s+(?:to|in)\s+([A-Za-z/_]+)', None),
        (r'(?:UTC|utc)\s+offset\s+(?:for|of|in)\s+([A-Za-z/_]+)', 'tz_offset'),
        # Port lookups
        (r'(?:what\s+)?(?:service|runs?)\s+(?:on|at)\s+port\s+(\d+)', 'port_info'),
        (r'(?:port|default port)\s+(?:for|of)\s+(\w+)', 'service_port'),
        # Geometry
        (r'area\s+of\s+(?:a\s+)?circle\s+(?:with\s+)?radius\s+([\d.]+)', None),
        (r'volume\s+of\s+(?:a\s+)?sphere\s+(?:with\s+)?radius\s+([\d.]+)', None),
        (r'area\s+of\s+(?:a\s+)?trapezoid\s+.*?(?:sides?\s+)?([\d.]+)\s+and\s+([\d.]+)\s+.*?height\s+([\d.]+)', None),
        (r'(?:interior|internal)\s+angle\s+of\s+(?:a\s+)?regular\s+(\w+)', None),
        # Roman numerals
        (r'(\d+)\s+in\s+[Rr]oman\s+numerals?', None),
        (r'([MDCLXVI]+)\s+in\s+(?:decimal|arabic|base\s*10)', None),
        # Probability
        (r'(\d+)\s+choose\s+(\d+)', None),
        # ASCII
        (r'(?:ASCII|ascii)\s+code\s+(?:for|of)\s+(?:the\s+)?(?:letter\s+)?["\']?(\w)["\']?', 'ascii_code'),
        # Country knowledge
        (r'(?:capital|capitol)\s+(?:of|city of)\s+([A-Z][\w\s]+)', 'capital'),
        (r'(?:what|which)\s+(?:is\s+)?(?:the\s+)?(?:capital|capitol)\s+(?:of|city of)\s+([A-Z][\w\s]+)', 'capital'),
        (r'(?:currency|money)\s+(?:of|in|used in)\s+([A-Z][\w\s]+)', 'country_currency'),
        (r'(?:calling|phone|dial)\s+code\s+(?:of|for)\s+([A-Z][\w\s]+)', 'country_calling_code'),
        (r'(?:ISO|iso)\s+code\s+(?:of|for)\s+([A-Z][\w\s]+)', 'country_iso2'),
        # Element knowledge
        (r'(?:atomic\s+)?(?:weight|mass)\s+of\s+(\w+)', 'atomic_weight'),
        (r'(?:atomic\s+)?number\s+of\s+(\w+)', 'atomic_number'),
        (r'(?:electron\s+)?config(?:uration)?\s+(?:of|for)\s+(\w+)', 'electron_config'),
        (r'(?:symbol|chemical symbol)\s+(?:of|for)\s+(\w+)', 'element_symbol'),
        # Physical constants
        (r'(?:speed of light|planck.s? constant|boltzmann.s? constant|avogadro.s? number|gravitational constant|elementary charge|gas constant|fine.structure constant|bohr radius|standard gravity|standard atmosphere)', None),
        # Algorithm complexity
        (r'(?:time\s+)?complexity\s+of\s+(quicksort|mergesort|merge sort|heapsort|heap sort|timsort|insertion sort|bubble sort|selection sort|radix sort|counting sort|bucket sort)', 'sort_complexity'),
        (r'worst\s+case\s+(?:of|for)\s+(\w[\w\s]*)', 'worst_case'),
        (r'[Ii]s\s+(quicksort|mergesort|merge sort|heapsort|heap sort|timsort|insertion sort|bubble sort|selection sort)\s+stable', 'is_stable_sort'),
    ]
    for pat, func in _NL_FUNC_MAP:
        for m in re.finditer(pat, prompt, re.IGNORECASE):
            try:
                if func is None and 'regex' in pat:
                    expr = f'regex_test("{m.group(1)}", "{m.group(2)}")'
                elif func is None and 'contrast' in pat.lower():
                    expr = f'color_contrast("{m.group(1)}", "{m.group(2)}")'
                elif func is None and 'convert' in pat and 'RGB' in pat:
                    expr = f'color_hex_to_rgb("{m.group(1)}")'
                elif func is None and 'complement' in pat:
                    expr = f'color_complementary("{m.group(1)}")'
                elif func is None and ('lighten' in pat or 'darken' in pat):
                    word = m.group(0).split()[0].lower()
                    fn = 'color_lighten' if 'lighten' in word else 'color_darken'
                    expr = f'{fn}("{m.group(1)}", {m.group(2)})'
                elif func is None and '[01]' in pat and 'binary' in pat:
                    # "binary number X to hex/decimal/octal" — 2 groups
                    target = m.group(2).lower().strip()
                    if 'hex' in target:
                        expr = f'base_convert("{m.group(1)}", 2, 16)'
                    elif 'decimal' in target or 'base' in target:
                        expr = f'from_binary("{m.group(1)}")'
                    elif 'octal' in target:
                        expr = f'base_convert("{m.group(1)}", 2, 8)'
                    else:
                        continue
                elif func is None and 'binary' in pat and '[01]' not in pat:
                    expr = f'to_binary({m.group(1)})'
                elif func is None and 'hex' in pat and 'binary' not in pat:
                    expr = f'to_hex({m.group(1)})'
                elif func is None and 'octal' in pat and 'binary' not in pat:
                    expr = f'to_octal({m.group(1)})'
                elif func is None and 'base.10' in pat.replace(' ', '.').replace('\\', '.'):
                    expr = f'from_binary("{m.group(1)}")'
                elif func is None and ('MB' in pat or 'GB' in pat or 'KB' in pat or 'KiB' in pat or 'MiB' in pat or 'GiB' in pat or 'TiB' in pat):
                    # Byte size conversions
                    groups = m.groups()
                    if len(groups) == 3:
                        expr = f'bytes_convert({groups[0]}, "{groups[1]}", "{groups[2]}")'
                    elif len(groups) == 1 and 'MB' in pat and 'bytes' in pat.lower():
                        expr = f'bytes_parse("{m.group(1)} MB")'
                    elif len(groups) == 2:
                        expr = f'bytes_convert({groups[0]}, "{groups[1]}", "bytes")'
                    else:
                        expr = f'bytes_parse("{m.group(0).strip()}")'
                elif func is None and ('hours' in pat or 'minutes' in pat or 'days' in pat or 'weeks' in pat) and ('to' in pat or 'in' in pat):
                    groups = m.groups()
                    if len(groups) == 3:
                        expr = f'duration_convert({groups[0]}, "{groups[1]}", "{groups[2]}")'
                    elif len(groups) == 2 and 'seconds' in pat:
                        expr = f'seconds_in({groups[0]}, "{groups[1]}")'
                    else:
                        expr = f'duration_parse("{m.group(1)}")'
                elif func is None and '[hms]' in pat:
                    expr = f'duration_parse("{m.group(1)}")'
                elif func is None and 'Luhn' in pat:
                    expr = f'luhn_validate("{m.group(1)}")'
                elif func is None and 'ISBN' in pat:
                    isbn = m.group(1).replace("-", "").replace(" ", "")
                    fn = 'isbn13_validate' if len(isbn) == 13 else 'isbn10_validate'
                    expr = f'{fn}("{isbn}")'
                elif func is None and (':\\d' in pat and 'A-Za-z' in pat):
                    # Timezone conversion: HH:MM from_tz to to_tz
                    expr = f'tz_convert("{m.group(1)}", "{m.group(2)}", "{m.group(3)}")'
                elif func is None and 'speed of light' in pat:
                    matched = m.group(0).lower().strip()
                    expr = f'physical_constant("{matched}")'
                elif func is None and 'circle' in pat and 'radius' in pat:
                    expr = f'circle_area({m.group(1)})'
                elif func is None and 'sphere' in pat and 'radius' in pat:
                    expr = f'sphere_volume({m.group(1)})'
                elif func is None and 'trapezoid' in pat:
                    expr = f'trapezoid_area({m.group(1)}, {m.group(2)}, {m.group(3)})'
                elif func is None and 'interior' in pat and 'regular' in pat:
                    # Map polygon name to sides
                    shape = m.group(1).lower()
                    sides_map = {"triangle": 3, "square": 4, "pentagon": 5,
                                 "hexagon": 6, "heptagon": 7, "octagon": 8,
                                 "nonagon": 9, "decagon": 10}
                    n = sides_map.get(shape)
                    if n:
                        expr = f'polygon_interior_angle({n})'
                    else:
                        continue
                elif func is None and 'oman' in pat and '\\d+' in pat:
                    expr = f'to_roman({m.group(1)})'
                elif func is None and '[MDCLXVI]' in pat:
                    expr = f'from_roman("{m.group(1)}")'
                elif func is None and 'choose' in pat and 'convert' not in pat:
                    expr = f'combinations({m.group(1)}, {m.group(2)})'
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
