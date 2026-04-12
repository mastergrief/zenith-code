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
    """Build system prompt listing available verified backends."""
    from calm.expression import _FUNCTIONS

    _CATEGORIES = {
        'math': {'sqrt', 'pow', 'abs', 'floor', 'ceil', 'log', 'log2',
                 'log10', 'pi', 'e', 'min', 'max', 'round', 'factorial',
                 'gcd', 'lcm', 'is_prime', 'next_prime', 'prev_prime',
                 'nth_prime', 'factorize', 'divisors', 'count_divisors',
                 'is_perfect', 'digit_sum', 'digital_root', 'fibonacci',
                 'collatz', 'collatz_length', 'solve_quadratic',
                 'sum_range', 'product_range'},
        'dates': {'days_between', 'day_of_week', 'is_leap_year',
                  'days_in_month', 'add_days', 'date_diff'},
        'units': {'convert', 'celsius_to_fahrenheit', 'fahrenheit_to_celsius',
                  'celsius_to_kelvin', 'kelvin_to_celsius'},
        'stats': {'mean', 'median', 'mode', 'variance', 'stdev',
                  'percentile', 'correlation', 'linear_regression',
                  'normalize', 'zscore', 'histogram'},
        'algorithms': {'sort_list', 'unique', 'binary_search', 'nCr', 'nPr',
                       'list_combinations', 'list_permutations', 'shortest_path',
                       'is_connected', 'topological_sort', 'cumsum',
                       'running_max', 'longest_increasing_subsequence'},
        'quality': {'cyclomatic_complexity', 'max_nesting_depth',
                    'function_length', 'naming_check', 'dead_code',
                    'code_quality', 'code_quality_file'},
        'readability': {'flesch_kincaid', 'jargon_density', 'vocabulary_complexity',
                        'text_structure', 'readability_report'},
    }

    categories = {}
    for name in sorted(_FUNCTIONS.keys()):
        if '.' in name:
            cat = name.split('.')[0]
        else:
            cat = next((c for c, funcs in _CATEGORIES.items() if name in funcs), None)
            if not cat:
                continue
        categories.setdefault(cat, []).append(name)

    lines = [f"  {cat}: {', '.join(funcs)}" for cat, funcs in sorted(categories.items())]

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

    return results
