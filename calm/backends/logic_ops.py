"""
CALM Logic/boolean backend — truth tables, propositional logic, set operations.

Models botch De Morgan's, confuse XOR/XNOR, mess up truth tables.
"""

from __future__ import annotations

from itertools import product as iter_product


def truth_table(expression: str, variables: list) -> list[dict]:
    """Generate truth table for a boolean expression.
    Supports: and, or, not, xor, implies (->), iff (<->).
    Variables are single letters."""
    vars_list = [str(v) for v in variables]
    n = len(vars_list)
    results = []
    for combo in iter_product([False, True], repeat=n):
        env = dict(zip(vars_list, combo))
        expr = str(expression)
        # Replace operators
        expr = expr.replace('<->', ' == ')
        expr = expr.replace('->', ' <= ')  # p->q is not(p) or q, equivalent to p<=q for bools
        expr = expr.replace('implies', ' <= ')
        expr = expr.replace('iff', ' == ')
        expr = expr.replace('xor', ' != ')
        expr = expr.replace('AND', ' and ')
        expr = expr.replace('OR', ' or ')
        expr = expr.replace('NOT', ' not ')
        for v in vars_list:
            expr = expr.replace(v, str(env[v]))
        try:
            result = bool(eval(expr))  # noqa: S307 — controlled expression
        except Exception:
            result = None
        row = dict(env)
        row['result'] = result
        results.append(row)
    return results


def de_morgan(expression: str) -> str:
    """Apply De Morgan's law to simplify NOT(A AND B) or NOT(A OR B)."""
    e = str(expression).strip()
    if e.startswith("not(") and e.endswith(")"):
        inner = e[4:-1].strip()
        if " and " in inner:
            parts = inner.split(" and ", 1)
            return f"(not {parts[0].strip()}) or (not {parts[1].strip()})"
        elif " or " in inner:
            parts = inner.split(" or ", 1)
            return f"(not {parts[0].strip()}) and (not {parts[1].strip()})"
    return e


def is_tautology(expression: str, variables: list) -> bool:
    """Whether a boolean expression is always true (tautology)."""
    table = truth_table(expression, variables)
    return all(row['result'] is True for row in table)


def is_contradiction(expression: str, variables: list) -> bool:
    """Whether a boolean expression is always false (contradiction)."""
    table = truth_table(expression, variables)
    return all(row['result'] is False for row in table)


def is_satisfiable(expression: str, variables: list) -> bool:
    """Whether a boolean expression can be true for some assignment."""
    table = truth_table(expression, variables)
    return any(row['result'] is True for row in table)


# --- Set operations ---

def set_union(a: list, b: list) -> list:
    """Union of two sets."""
    return sorted(set(a) | set(b), key=str)


def set_intersection(a: list, b: list) -> list:
    """Intersection of two sets."""
    return sorted(set(a) & set(b), key=str)


def set_difference(a: list, b: list) -> list:
    """Set difference A - B."""
    return sorted(set(a) - set(b), key=str)


def set_symmetric_difference(a: list, b: list) -> list:
    """Symmetric difference A Δ B."""
    return sorted(set(a) ^ set(b), key=str)


def is_subset(a: list, b: list) -> bool:
    """Whether A ⊆ B."""
    return set(a) <= set(b)


def is_superset(a: list, b: list) -> bool:
    """Whether A ⊇ B."""
    return set(a) >= set(b)


def power_set(s: list) -> list[list]:
    """Power set of a set (all subsets). Warning: 2^n elements."""
    items = list(s)
    n = len(items)
    if n > 20:
        return [["error: set too large (max 20 elements)"]]
    result = []
    for i in range(2 ** n):
        subset = [items[j] for j in range(n) if i & (1 << j)]
        result.append(subset)
    return result


def set_cardinality(s: list) -> int:
    """Number of unique elements in a set."""
    return len(set(s))


LOGIC_FUNCTIONS = {
    "truth_table": truth_table,
    "de_morgan": de_morgan,
    "is_tautology": is_tautology,
    "is_contradiction": is_contradiction,
    "is_satisfiable": is_satisfiable,
    "set_union": set_union,
    "set_intersection": set_intersection,
    "set_difference": set_difference,
    "set_symmetric_difference": set_symmetric_difference,
    "is_subset": is_subset,
    "is_superset": is_superset,
    "power_set": power_set,
    "set_cardinality": set_cardinality,
}

LOGIC_NL_PATTERNS = [
    (r'truth\s+table\s+(?:for|of)', None),
    (r'(?:is)\s+.+\s+(?:a\s+)?tautology', None),
    (r'(?:is)\s+.+\s+(?:a\s+)?contradiction', None),
    (r'de\s+morgan', None),
    (r'(?:union|intersection|difference)\s+(?:of\s+)?\{', None),
    (r'power\s+set\s+(?:of\s+)?\{', None),
    (r'(?:is)\s+\{.*\}\s+(?:a\s+)?subset\s+(?:of)\s+\{', None),
]
