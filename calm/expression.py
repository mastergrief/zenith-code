"""
CALM v0.1 expression evaluator — safe math expression evaluation.

Allows the model to write full expressions inside <calm> blocks:
  <calm>
  (17 * 23) + (42 * 19) - 100
  </calm>

Or function calls:
  <calm>
  gcd(391, 782)
  is_prime(391)
  sqrt(1764) + 1
  </calm>

Each line is evaluated independently. Results accumulate on an
output list. No stack management, no push/pop — the harness
handles everything.

Security: uses AST parsing, NOT eval(). Only allows numeric
literals, arithmetic operators, and whitelisted function calls.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any, Callable, Dict, List, Optional, Tuple


# Placeholder — populated after function definitions below.
_FUNCTIONS: Dict[str, Callable] = {}

# Safe binary operators.
_BINOPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.BitXor: operator.pow,  # Allow ^ as power (common in math)
}

_UNARYOPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_CMPOPS = {
    ast.Eq: operator.eq,
    ast.NotEq: operator.ne,
    ast.Lt: operator.lt,
    ast.LtE: operator.le,
    ast.Gt: operator.gt,
    ast.GtE: operator.ge,
}


def _is_prime(n: int) -> bool:
    """Primality test — deterministic trial division."""
    if not isinstance(n, int) or n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def _factorize(n: int) -> List[int]:
    """Return prime factors of n."""
    if not isinstance(n, int) or n < 2:
        return []
    factors = []
    d = 2
    tmp = n
    while d * d <= tmp:
        while tmp % d == 0:
            factors.append(d)
            tmp //= d
        d += 1
    if tmp > 1:
        factors.append(tmp)
    return factors


# ---------------------------------------------------------------------------
# Higher-level reasoning functions — one dispatch, CPU does the search/solve
# ---------------------------------------------------------------------------

def _next_prime(n: int) -> int:
    """Find the smallest prime strictly greater than n."""
    if not isinstance(n, int):
        raise ExpressionError("next_prime: need int")
    candidate = n + 1 if n >= 2 else 2
    while not _is_prime(candidate):
        candidate += 1
        if candidate > 10**12:
            raise ExpressionError("next_prime: search exceeded limit")
    return candidate


def _prev_prime(n: int) -> int:
    """Find the largest prime strictly less than n."""
    if not isinstance(n, int) or n <= 2:
        raise ExpressionError("prev_prime: no prime less than input")
    candidate = n - 1
    while candidate >= 2 and not _is_prime(candidate):
        candidate -= 1
    if candidate < 2:
        raise ExpressionError("prev_prime: no prime found")
    return candidate


def _nth_prime(n: int) -> int:
    """Return the n-th prime (1-indexed: nth_prime(1)=2, nth_prime(4)=7)."""
    if not isinstance(n, int) or n < 1:
        raise ExpressionError("nth_prime: need positive int")
    count = 0
    candidate = 1
    while count < n:
        candidate += 1
        if _is_prime(candidate):
            count += 1
    return candidate


def _divisors(n: int) -> List[int]:
    """Return all divisors of n in ascending order."""
    if not isinstance(n, int) or n < 1:
        raise ExpressionError("divisors: need positive int")
    divs = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            divs.append(i)
            if i != n // i:
                divs.append(n // i)
    return sorted(divs)


def _count_divisors(n: int) -> int:
    """Count the number of divisors of n."""
    return len(_divisors(n))


def _solve_quadratic(a, b, c) -> Tuple:
    """Solve ax² + bx + c = 0. Returns (x1, x2) or ('no real roots',)."""
    disc = b * b - 4 * a * c
    if disc < 0:
        return ("no real roots",)
    if disc == 0:
        x = -b / (2 * a)
        return (x,)
    sq = math.sqrt(disc)
    x1 = (-b + sq) / (2 * a)
    x2 = (-b - sq) / (2 * a)
    return (x1, x2) if x1 <= x2 else (x2, x1)


def _fibonacci(n: int) -> int:
    """Return the n-th Fibonacci number (fib(0)=0, fib(1)=1, fib(10)=55)."""
    if not isinstance(n, int) or n < 0:
        raise ExpressionError("fibonacci: need non-negative int")
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b


def _collatz(n: int) -> List[int]:
    """Return the Collatz sequence starting from n until reaching 1."""
    if not isinstance(n, int) or n < 1:
        raise ExpressionError("collatz: need positive int")
    seq = [n]
    while n != 1:
        n = n // 2 if n % 2 == 0 else 3 * n + 1
        seq.append(n)
        if len(seq) > 10000:
            raise ExpressionError("collatz: sequence too long")
    return seq


def _collatz_length(n: int) -> int:
    """Return the length of the Collatz sequence from n to 1."""
    return len(_collatz(n))


def _sum_range(a: int, b: int) -> int:
    """Sum of integers from a to b inclusive."""
    if a > b:
        return 0
    return (b - a + 1) * (a + b) // 2


def _product_range(a: int, b: int) -> int:
    """Product of integers from a to b inclusive (generalized factorial)."""
    result = 1
    for i in range(a, b + 1):
        result *= i
    return result


def _factorial(n: int) -> int:
    """n! (factorial)."""
    if not isinstance(n, int) or n < 0:
        raise ExpressionError("factorial: need non-negative int")
    return _product_range(1, n) if n > 0 else 1


def _lcm(a: int, b: int) -> int:
    """Least common multiple."""
    if a == 0 or b == 0:
        return 0
    return abs(a * b) // math.gcd(a, b)


def _is_perfect(n: int) -> bool:
    """Is n a perfect number? (sum of proper divisors = n)."""
    if not isinstance(n, int) or n < 2:
        return False
    return sum(_divisors(n)[:-1]) == n


def _digital_root(n: int) -> int:
    """Repeated digit sum until single digit."""
    if not isinstance(n, int):
        raise ExpressionError("digital_root: need int")
    n = abs(n)
    while n >= 10:
        n = sum(int(d) for d in str(n))
    return n


def _digit_sum(n: int) -> int:
    """Sum of digits."""
    return sum(int(d) for d in str(abs(n)))


# ---------------------------------------------------------------------------
# Logic / search / data functions
# ---------------------------------------------------------------------------

def _find_int(lo: int, hi: int, *predicates) -> Optional[int]:
    """
    Find the first integer in [lo, hi] satisfying all predicates.
    Each predicate is a string expression where 'x' is the candidate.
    Example: find_int(1, 100, "is_prime(x)", "digit_sum(x) == 7")
    """
    for x in range(int(lo), int(hi) + 1):
        local_fns = dict(_FUNCTIONS)
        local_fns["x"] = x
        try:
            if all(safe_eval(str(p), local_fns) for p in predicates):
                return x
        except ExpressionError:
            continue
    return None


def _count_if(lo: int, hi: int, *predicates) -> int:
    """Count integers in [lo, hi] satisfying all predicates."""
    count = 0
    for x in range(int(lo), int(hi) + 1):
        local_fns = dict(_FUNCTIONS)
        local_fns["x"] = x
        try:
            if all(safe_eval(str(p), local_fns) for p in predicates):
                count += 1
        except ExpressionError:
            continue
    return count


def _map_expr(expr_str, items) -> list:
    """
    Map an expression over a list. 'x' is the current element.
    Example: map_expr("x * 2", [1, 2, 3]) → [2, 4, 6]
    """
    result = []
    for item in items:
        local_fns = dict(_FUNCTIONS)
        local_fns["x"] = item
        result.append(safe_eval(str(expr_str), local_fns))
    return result


def _filter_expr(expr_str, items) -> list:
    """
    Filter a list by an expression. 'x' is the current element.
    Example: filter_expr("x > 3", [1, 2, 3, 4, 5]) → [4, 5]
    """
    result = []
    for item in items:
        local_fns = dict(_FUNCTIONS)
        local_fns["x"] = item
        if safe_eval(str(expr_str), local_fns):
            result.append(item)
    return result


# ---------------------------------------------------------------------------
# Code operation wrappers for expression evaluator
# These adapt the stack-based code_ops to plain function calls.
# ---------------------------------------------------------------------------

def _code_call(fn_name, *args):
    """Call a code_ops function by simulating stack operations."""
    from calm.backends.code_ops import CODE_WORDS
    from calm.stack_vm import VMState, Instruction
    fn = CODE_WORDS.get(fn_name)
    if not fn:
        raise ExpressionError(f"unknown code function: {fn_name}")
    state = VMState()
    for a in args:
        state.stack.append(a)
    fn(state, Instruction(word=fn_name))
    return state.stack[-1] if state.stack else None

def _sec_call(fn_name, *args):
    """Call a security_ops function by simulating stack operations."""
    from calm.backends.security_ops import SECURITY_WORDS
    from calm.stack_vm import VMState, Instruction
    fn = SECURITY_WORDS.get(fn_name)
    if not fn:
        raise ExpressionError(f"unknown security function: {fn_name}")
    state = VMState()
    for a in args:
        state.stack.append(a)
    fn(state, Instruction(word=fn_name))
    return state.stack[-1] if state.stack else None

def _sec_audit(path): return _sec_call("security.audit", path)
def _sec_sql_injection(path): return _sec_call("security.sql_injection", path)
def _sec_xss(path): return _sec_call("security.xss", path)
def _sec_secrets(path): return _sec_call("security.secrets", path)
def _sec_unsafe_exec(path): return _sec_call("security.unsafe_exec", path)
def _sec_path_traversal(path): return _sec_call("security.path_traversal", path)
def _sec_crypto(path): return _sec_call("security.crypto", path)
def _sec_permissions(path): return _sec_call("security.permissions", path)

def _code_read(path): return _code_call("code.read", path)
def _code_write(path, content): return _code_call("code.write", path, content)
def _code_syntax_check(path): return _code_call("code.syntax_check", path)
def _code_run(path): return _code_call("code.run", path)
def _code_test(path): return _code_call("code.test", path)
def _code_lint(path): return _code_call("code.lint", path)
def _code_search(pattern, path): return _code_call("code.search", pattern, path)
def _code_find(pattern, path): return _code_call("code.find", pattern, path)
def _code_diff(path): return _code_call("code.diff", path)
def _code_edit(path, line, content): return _code_call("code.edit", path, line, content)
def _code_insert(path, line, content): return _code_call("code.insert", path, line, content)
def _code_delete(path, line): return _code_call("code.delete", path, line)
def _code_count_lines(path): return _code_call("code.count_lines", path)
def _code_functions(path): return _code_call("code.functions", path)
def _code_classes(path): return _code_call("code.classes", path)
def _code_imports(path): return _code_call("code.imports", path)


# Now that all functions are defined, populate the whitelist.
_FUNCTIONS.update({
    # Basic math
    "sqrt": math.sqrt,
    "pow": pow,
    "abs": abs,
    "floor": math.floor,
    "ceil": math.ceil,
    "log": math.log,
    "log2": math.log2,
    "log10": math.log10,
    "pi": lambda: math.pi,
    "e": lambda: math.e,
    "min": min,
    "max": max,
    "round": round,
    "factorial": _factorial,
    # Number theory
    "gcd": math.gcd,
    "lcm": _lcm,
    "is_prime": _is_prime,
    "next_prime": _next_prime,
    "prev_prime": _prev_prime,
    "nth_prime": _nth_prime,
    "factorize": _factorize,
    "divisors": _divisors,
    "count_divisors": _count_divisors,
    "is_perfect": _is_perfect,
    "digit_sum": _digit_sum,
    "digital_root": _digital_root,
    # Sequences
    "fibonacci": _fibonacci,
    "collatz": _collatz,
    "collatz_length": _collatz_length,
    # Algebra
    "solve_quadratic": _solve_quadratic,
    # Ranges
    "sum_range": _sum_range,
    "product_range": _product_range,
    # Security operations
    "security.audit": _sec_audit,
    "security.sql_injection": _sec_sql_injection,
    "security.xss": _sec_xss,
    "security.secrets": _sec_secrets,
    "security.unsafe_exec": _sec_unsafe_exec,
    "security.path_traversal": _sec_path_traversal,
    "security.crypto": _sec_crypto,
    "security.permissions": _sec_permissions,
    # Code operations (wrappers for calm/backends/code_ops.py)
    "code.read": _code_read,
    "code.write": _code_write,
    "code.syntax_check": _code_syntax_check,
    "code.run": _code_run,
    "code.test": _code_test,
    "code.lint": _code_lint,
    "code.search": _code_search,
    "code.find": _code_find,
    "code.diff": _code_diff,
    "code.edit": _code_edit,
    "code.insert": _code_insert,
    "code.delete": _code_delete,
    "code.count_lines": _code_count_lines,
    "code.functions": _code_functions,
    "code.classes": _code_classes,
    "code.imports": _code_imports,
    # Logic / search
    "find_int": _find_int,
    "count_if": _count_if,
    # Data
    "len": len,
    "sorted": sorted,
    "reversed": lambda x: list(reversed(x)),
    "sum": sum,
    "any": any,
    "all": all,
    "zip": lambda *a: list(zip(*a)),
    "range": lambda *a: list(range(*a)),
    "map_expr": _map_expr,
    "filter_expr": _filter_expr,
})

# Register new modular backends.
try:
    from calm.backends.date_ops import DATE_FUNCTIONS
    _FUNCTIONS.update(DATE_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.convert_ops import CONVERT_FUNCTIONS
    _FUNCTIONS.update(CONVERT_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.data_ops import DATA_FUNCTIONS
    _FUNCTIONS.update(DATA_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.algo_ops import ALGO_FUNCTIONS
    _FUNCTIONS.update(ALGO_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.quality_ops import QUALITY_FUNCTIONS
    _FUNCTIONS.update(QUALITY_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.readability_ops import READABILITY_FUNCTIONS
    _FUNCTIONS.update(READABILITY_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.regex_ops import REGEX_FUNCTIONS
    _FUNCTIONS.update(REGEX_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.json_ops import JSON_FUNCTIONS
    _FUNCTIONS.update(JSON_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.encoding_ops import ENCODING_FUNCTIONS
    _FUNCTIONS.update(ENCODING_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.git_ops import GIT_FUNCTIONS
    _FUNCTIONS.update(GIT_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.network_ops import NETWORK_FUNCTIONS
    _FUNCTIONS.update(NETWORK_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.creative_ops import CREATIVE_FUNCTIONS
    _FUNCTIONS.update(CREATIVE_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.impact_ops import IMPACT_FUNCTIONS
    _FUNCTIONS.update(IMPACT_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.python_ops import PYTHON_FUNCTIONS
    _FUNCTIONS.update(PYTHON_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.math_extended_ops import MATH_EXTENDED_FUNCTIONS
    _FUNCTIONS.update(MATH_EXTENDED_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.perf_ops import PERF_FUNCTIONS
    _FUNCTIONS.update(PERF_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.deps_ops import DEPS_FUNCTIONS
    _FUNCTIONS.update(DEPS_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.refactor_ops import REFACTOR_FUNCTIONS
    _FUNCTIONS.update(REFACTOR_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.type_ops import TYPE_FUNCTIONS
    _FUNCTIONS.update(TYPE_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.test_ops import TEST_FUNCTIONS
    _FUNCTIONS.update(TEST_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.doc_ops import DOC_FUNCTIONS
    _FUNCTIONS.update(DOC_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.shell_ops import SHELL_FUNCTIONS
    _FUNCTIONS.update(SHELL_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.semver_ops import SEMVER_FUNCTIONS
    _FUNCTIONS.update(SEMVER_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.config_ops import CONFIG_FUNCTIONS
    _FUNCTIONS.update(CONFIG_FUNCTIONS)
except ImportError:
    pass

try:
    from calm.backends.context_ops import CONTEXT_FUNCTIONS
    _FUNCTIONS.update(CONTEXT_FUNCTIONS)
except ImportError:
    pass


class ExpressionError(Exception):
    """Raised when an expression can't be safely evaluated."""


def safe_eval(expr: str, functions: Optional[Dict[str, Callable]] = None) -> Any:
    """
    Safely evaluate a math expression. Uses AST parsing — never calls
    Python's eval(). Only allows:
    - Numeric literals (int, float)
    - Arithmetic operators (+, -, *, /, //, %, **)
    - Comparison operators (==, !=, <, <=, >, >=)
    - Whitelisted function calls
    - Boolean literals (True, False)
    - Tuple/list literals for multi-return
    """
    fns = dict(_FUNCTIONS)
    if functions:
        fns.update(functions)

    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ExpressionError(f"syntax error: {e}")

    return _eval_node(tree.body, fns)


def _eval_node(node: ast.AST, fns: dict) -> Any:
    """Recursively evaluate an AST node."""

    # Numeric/string/bool literals
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float, bool, str)):
            return node.value
        raise ExpressionError(f"unsupported literal type: {type(node.value)}")

    # Unary ops: -x, +x
    if isinstance(node, ast.UnaryOp):
        op = _UNARYOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported unary op: {type(node.op).__name__}")
        return op(_eval_node(node.operand, fns))

    # Binary ops: x + y, x * y, etc.
    if isinstance(node, ast.BinOp):
        op = _BINOPS.get(type(node.op))
        if op is None:
            raise ExpressionError(f"unsupported binary op: {type(node.op).__name__}")
        left = _eval_node(node.left, fns)
        right = _eval_node(node.right, fns)
        try:
            return op(left, right)
        except ZeroDivisionError:
            raise ExpressionError("division by zero")

    # Comparisons: x == y, x < y
    if isinstance(node, ast.Compare):
        left = _eval_node(node.left, fns)
        for op, comparator in zip(node.ops, node.comparators):
            cmp_fn = _CMPOPS.get(type(op))
            if cmp_fn is None:
                raise ExpressionError(f"unsupported comparison: {type(op).__name__}")
            right = _eval_node(comparator, fns)
            if not cmp_fn(left, right):
                return False
            left = right
        return True

    # Function calls: sqrt(16), gcd(12, 8)
    if isinstance(node, ast.Call):
        # Support both simple names (sqrt) and dotted names (code.read)
        if isinstance(node.func, ast.Name):
            name = node.func.id
        elif isinstance(node.func, ast.Attribute) and isinstance(node.func.value, ast.Name):
            name = f"{node.func.value.id}.{node.func.attr}"
        else:
            raise ExpressionError(f"only simple or dotted function names allowed")
        if name not in fns:
            raise ExpressionError(f"unknown function: {name}")
        args = [_eval_node(arg, fns) for arg in node.args]
        try:
            return fns[name](*args)
        except Exception as e:
            raise ExpressionError(f"{name}() error: {e}")

    # Name references (for constants like pi, e)
    if isinstance(node, ast.Name):
        if node.id in fns:
            result = fns[node.id]
            return result() if callable(result) else result
        raise ExpressionError(f"unknown name: {node.id}")

    # Boolean ops: and, or
    if isinstance(node, ast.BoolOp):
        if isinstance(node.op, ast.And):
            result = True
            for val in node.values:
                result = _eval_node(val, fns)
                if not result:
                    return result
            return result
        if isinstance(node.op, ast.Or):
            for val in node.values:
                result = _eval_node(val, fns)
                if result:
                    return result
            return result

    # Ternary: x if cond else y
    if isinstance(node, ast.IfExp):
        if _eval_node(node.test, fns):
            return _eval_node(node.body, fns)
        return _eval_node(node.orelse, fns)

    # Tuple: (a, b) — used for multi-return
    if isinstance(node, ast.Tuple):
        return tuple(_eval_node(elt, fns) for elt in node.elts)

    # List comprehension: [expr for x in iterable if cond]
    if isinstance(node, ast.ListComp):
        return _eval_comprehension(node, fns, list)

    # Set/Generator comprehension
    if isinstance(node, ast.SetComp):
        return _eval_comprehension(node, fns, set)

    if isinstance(node, ast.GeneratorExp):
        return _eval_comprehension(node, fns, list)

    # List literal: [1, 2, 3]
    if isinstance(node, ast.List):
        return [_eval_node(elt, fns) for elt in node.elts]

    # Dict literal: {"a": 1, "b": 2}
    if isinstance(node, ast.Dict):
        keys = [_eval_node(k, fns) for k in node.keys]
        values = [_eval_node(v, fns) for v in node.values]
        return dict(zip(keys, values))

    # Subscript: x[0], x[1:3]
    if isinstance(node, ast.Subscript):
        value = _eval_node(node.value, fns)
        if isinstance(node.slice, ast.Constant):
            return value[node.slice.value]
        if isinstance(node.slice, ast.Slice):
            lo = _eval_node(node.slice.lower, fns) if node.slice.lower else None
            hi = _eval_node(node.slice.upper, fns) if node.slice.upper else None
            step = _eval_node(node.slice.step, fns) if node.slice.step else None
            return value[lo:hi:step]
        idx = _eval_node(node.slice, fns)
        return value[idx]

    raise ExpressionError(f"unsupported expression: {ast.dump(node)}")


def _eval_comprehension(node, fns, container_type):
    """Evaluate a list/set comprehension safely."""
    results = []
    generators = node.generators

    def _eval_gen(gen_idx, local_fns):
        if gen_idx >= len(generators):
            results.append(_eval_node(node.elt, local_fns))
            return
        gen = generators[gen_idx]
        if not isinstance(gen.target, ast.Name):
            raise ExpressionError("only simple variable names in comprehensions")
        var_name = gen.target.id
        iterable = _eval_node(gen.iter, local_fns)
        for item in iterable:
            inner_fns = dict(local_fns)
            inner_fns[var_name] = item
            # Check all conditions
            if all(_eval_node(cond, inner_fns) for cond in gen.ifs):
                _eval_gen(gen_idx + 1, inner_fns)
            if len(results) > 10000:
                raise ExpressionError("comprehension too large")

    _eval_gen(0, dict(fns))
    return container_type(results)


def eval_calm_block(block: str, functions: Optional[Dict] = None) -> List[dict]:
    """
    Evaluate a CALM expression block. Each non-empty, non-comment line
    is evaluated as an expression. Returns a list of results.
    """
    results = []
    for i, line in enumerate(block.splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("\\") or line.startswith("//") or line.startswith("#"):
            continue
        # Strip claim suffix if present (-> [...] or -> <pending>)
        import re
        line = re.sub(r'\s*->.*$', '', line)
        if not line:
            continue
        try:
            value = safe_eval(line, functions)
            results.append({"line": i, "expr": line, "value": value, "error": None})
        except ExpressionError as e:
            results.append({"line": i, "expr": line, "value": None, "error": str(e)})
    return results
