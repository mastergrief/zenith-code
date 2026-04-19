"""R53 Phase 1b — Complex multi-step coding eval.

6 hand-curated problems across 3 categories. Each requires multi-step
reasoning (diagnose + fix, parse + compute + serialize, plan + implement
+ test). Behavioral tests, format-agnostic code extraction.

Three conditions per problem:
  A. STOCK      — chat-templated prompt, no augmentation
  B. HINTED     — same + CodeVerifierFacade hints (retrieved examples)
  C. SANITY     — same + random (wrong-domain) retrieved examples
                  controls for prompt-length effect; if B ≈ C, gains are
                  from length alone, not retrieval content.

Scoring: each problem has N behavioral test cases. Pass rate per path.
Aggregates: total PASS/FAIL across conditions.

Run:
  bin/gemma-run scripts/r53_eval_complex.py
"""

from __future__ import annotations

import random
import re
from typing import List, NamedTuple, Optional


# Shared token-budget helper: adaptive-per-prompt with 16K ceiling.
# Mirrors r53_21_import_inject.py so all R53 evals stay consistent.
# Gemma 4 E4B trains at 131K ctx; 16K eval ceiling is safe headroom.
MAX_TOKENS_CEILING = 16384


def _adaptive_budget(prompt: str) -> int:
    """Per-prompt output-token budget via AdaptiveBudget, clamped.
    Returns the budget int (tier kept internal)."""
    try:
        from calm.adaptive import AdaptiveBudget
        est = AdaptiveBudget().estimate(prompt)
        return min(est.budget, MAX_TOKENS_CEILING)
    except Exception:
        return MAX_TOKENS_CEILING


def _reload_facades():
    import sys
    for m in list(sys.modules.keys()):
        if (m.startswith("calm.llm_computer.facades.")
                or m == "calm.llm_computer.facades"):
            del sys.modules[m]


# -------------------------------------------------------------------
# Corpus
# -------------------------------------------------------------------

class ComplexProblem(NamedTuple):
    name: str
    category: str
    prompt: str
    starter: str       # starter code (empty string if none)
    required: List[str]  # required function/class names
    test_code: str     # uses `print("PASS")` / `print("FAIL ...")` per test


CORPUS: List[ComplexProblem] = [

    # ----- Multi-bug-fix (2) -----

    ComplexProblem(
        name="linked_list_bugs",
        category="multi_bug",
        prompt=(
            "The following Python LinkedList implementation has three bugs that make "
            "one or more methods incorrect. Fix ALL bugs. The class must support "
            "`append(v)`, `remove(v)` (removes first occurrence, no-op if absent), "
            "and `to_list()` (returns values in order). Do not change the class name "
            "or method signatures.\n\n"
            "```python\n"
            "class LinkedList:\n"
            "    def __init__(self):\n"
            "        self.head = None\n"
            "        self.tail = None\n\n"
            "    def append(self, v):\n"
            "        node = {'v': v, 'next': None}\n"
            "        if self.head is None:\n"
            "            self.head = node\n"
            "        else:\n"
            "            self.tail['next'] = node\n"
            "        # BUG 1: tail never updated after append\n\n"
            "    def remove(self, v):\n"
            "        cur = self.head\n"
            "        while cur is not None:\n"
            "            if cur['v'] == v:\n"
            "                cur = cur['next']\n"
            "                return\n"
            "            # BUG 2: advance missing when not matched\n\n"
            "    def to_list(self):\n"
            "        out = []\n"
            "        cur = self.head\n"
            "        while cur:\n"
            "            out.append(cur['v'])\n"
            "            # BUG 3: infinite loop — cur never advances\n"
            "        return out\n"
            "```\n\n"
            "Return the full fixed class."
        ),
        starter="",
        required=["LinkedList"],
        test_code="""
# Test: basic append + to_list
ll = LinkedList()
ll.append(1); ll.append(2); ll.append(3)
print("PASS" if ll.to_list() == [1, 2, 3] else f"FAIL append+to_list got {ll.to_list()}")

# Test: empty to_list
print("PASS" if LinkedList().to_list() == [] else "FAIL empty to_list")

# Test: remove head
ll2 = LinkedList()
for v in [1, 2, 3]: ll2.append(v)
ll2.remove(1)
print("PASS" if ll2.to_list() == [2, 3] else f"FAIL remove head got {ll2.to_list()}")

# Test: remove middle
ll3 = LinkedList()
for v in [1, 2, 3]: ll3.append(v)
ll3.remove(2)
print("PASS" if ll3.to_list() == [1, 3] else f"FAIL remove middle got {ll3.to_list()}")

# Test: remove absent (no-op)
ll4 = LinkedList()
for v in [1, 2]: ll4.append(v)
ll4.remove(99)
print("PASS" if ll4.to_list() == [1, 2] else f"FAIL remove absent got {ll4.to_list()}")
""",
    ),

    ComplexProblem(
        name="date_validation_chain",
        category="multi_bug",
        prompt=(
            "The following `valid_date(y, m, d)` function is supposed to return True "
            "if (y, m, d) form a valid calendar date. It has THREE bugs: one in "
            "month validation, one in day-of-month validation, one in leap year "
            "logic. Fix all three.\n\n"
            "```python\n"
            "def valid_date(y, m, d):\n"
            "    # BUG 1: wrong month upper bound\n"
            "    if not (1 <= m < 12):\n"
            "        return False\n"
            "    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]\n"
            "    # BUG 2: leap year — doesn't handle centuries correctly\n"
            "    if m == 2 and y % 4 == 0:\n"
            "        days_in_month[1] = 29\n"
            "    # BUG 3: off-by-one in day range\n"
            "    if not (1 < d <= days_in_month[m - 1]):\n"
            "        return False\n"
            "    return True\n"
            "```\n\n"
            "Return the full corrected function."
        ),
        starter="",
        required=["valid_date"],
        test_code="""
cases = [
    (2024, 2, 29, True),   # leap year
    (2023, 2, 29, False),  # non-leap
    (2000, 2, 29, True),   # div-by-400 leap
    (1900, 2, 29, False),  # century non-leap
    (2024, 1, 1, True),    # smallest day
    (2024, 1, 31, True),   # month end
    (2024, 1, 32, False),  # out of range
    (2024, 12, 31, True),  # last month + day
    (2024, 13, 1, False),  # bad month high
    (2024, 0, 1, False),   # bad month low
    (2024, 4, 31, False),  # April has 30
    (2024, 4, 30, True),
]
for y, m, d, expected in cases:
    got = valid_date(y, m, d)
    print("PASS" if got == expected else f"FAIL ({y},{m},{d}) got={got} expected={expected}")
""",
    ),

    # ----- Library-composition (2) -----

    ComplexProblem(
        name="log_level_counts",
        category="lib_compose",
        prompt=(
            "Write a Python function `log_level_counts(text)` that takes a multi-line "
            "string of log lines. Each line looks like:\n"
            "    2024-01-15 10:23:45 LEVEL: message here\n"
            "where LEVEL is one of DEBUG, INFO, WARNING, ERROR, CRITICAL.\n\n"
            "Use `re` to extract levels and `collections.Counter` to count them.\n"
            "Return a dict mapping level -> count. Lines that don't match the "
            "format should be silently skipped. Empty input returns {}."
        ),
        starter="",
        required=["log_level_counts"],
        test_code="""
t1 = '''2024-01-15 10:23:45 INFO: app started
2024-01-15 10:23:46 ERROR: connection failed
2024-01-15 10:23:47 INFO: retrying
garbage line with no level
2024-01-15 10:23:48 ERROR: retry failed
2024-01-15 10:23:49 DEBUG: stack trace...'''
r = log_level_counts(t1)
print("PASS" if r.get('INFO') == 2 else f"FAIL INFO got {r.get('INFO')}")
print("PASS" if r.get('ERROR') == 2 else f"FAIL ERROR got {r.get('ERROR')}")
print("PASS" if r.get('DEBUG') == 1 else f"FAIL DEBUG got {r.get('DEBUG')}")
# Unknown levels not counted
print("PASS" if 'garbage' not in r else "FAIL garbage leaked")

# Empty input
print("PASS" if log_level_counts('') == {} else "FAIL empty")

# All-malformed
print("PASS" if log_level_counts('foo\\nbar\\nbaz') == {} else "FAIL all-malformed")
""",
    ),

    ComplexProblem(
        name="csv_column_stats",
        category="lib_compose",
        prompt=(
            "Write a Python function `csv_column_stats(text)` that takes a CSV "
            "string (first line = header) and returns a dict mapping each numeric "
            "column name to a sub-dict of {'mean', 'stdev', 'min', 'max'} over the "
            "column's values. Non-numeric columns must be SKIPPED (not an error). "
            "Uses `csv` for parsing and `statistics` for computation.\n\n"
            "Requirements:\n"
            "- Parse with csv.reader (first row is header)\n"
            "- Try to float() each cell; if any cell in a column fails, skip that column\n"
            "- For numeric columns: mean, stdev (sample stdev, statistics.stdev; if <2 rows, stdev=0.0), min, max\n"
            "- Empty input returns {}"
        ),
        starter="",
        required=["csv_column_stats"],
        test_code="""
t = '''name,age,score
Alice,30,95.5
Bob,25,82.0
Carol,35,88.5'''
r = csv_column_stats(t)
print("PASS" if 'name' not in r else "FAIL name should be skipped (non-numeric)")
print("PASS" if 'age' in r else "FAIL age missing")
print("PASS" if abs(r['age']['mean'] - 30.0) < 1e-9 else f"FAIL age mean got {r['age']['mean']}")
print("PASS" if r['age']['min'] == 25.0 else f"FAIL age min got {r['age']['min']}")
print("PASS" if r['age']['max'] == 35.0 else f"FAIL age max got {r['age']['max']}")
print("PASS" if abs(r['score']['mean'] - 88.666666) < 1e-3 else f"FAIL score mean got {r['score']['mean']}")

# Empty
print("PASS" if csv_column_stats('') == {} else "FAIL empty")

# Single data row: stdev = 0.0
r2 = csv_column_stats('a,b\\n1,2')
print("PASS" if r2['a']['stdev'] == 0.0 else f"FAIL single-row stdev got {r2['a']['stdev']}")
""",
    ),

    # ----- Plan-then-code (2) -----

    ComplexProblem(
        name="token_bucket_rate_limiter",
        category="plan_code",
        prompt=(
            "Implement a `TokenBucket` class implementing token-bucket rate limiting:\n\n"
            "- `__init__(self, rate, capacity)`: rate = tokens added per second, capacity = max tokens.\n"
            "  Bucket starts full.\n"
            "- `allow(self)`: if at least 1 token is available, consume 1 and return True; else False.\n"
            "- `tokens(self)`: current token count (float).\n\n"
            "Must use `time.monotonic` (NOT time.time) for elapsed time computation. "
            "Refill rate is continuous — tokens grow proportional to elapsed time, capped at capacity.\n\n"
            "Required invariants:\n"
            "- Bucket starts at capacity (full)\n"
            "- allow() returns True up to capacity times immediately\n"
            "- After waiting elapsed seconds, at most (capacity) new tokens appear (cap enforced)"
        ),
        starter="",
        required=["TokenBucket"],
        test_code="""
import time

# Start full
tb = TokenBucket(rate=10, capacity=5)
print("PASS" if abs(tb.tokens() - 5.0) < 1e-6 else f"FAIL initial tokens {tb.tokens()}")

# Consume up to capacity
tb2 = TokenBucket(rate=0.1, capacity=3)   # very slow refill — no significant refill during test
successes = sum(tb2.allow() for _ in range(3))
print("PASS" if successes == 3 else f"FAIL first 3 calls got {successes}")
print("PASS" if not tb2.allow() else "FAIL 4th call should fail immediately")

# Refill over time (sleep a bit)
tb3 = TokenBucket(rate=100, capacity=5)   # very fast refill
for _ in range(5): tb3.allow()            # drain
time.sleep(0.05)                          # 50ms × 100/s = 5 tokens
# Now should allow several more
refilled = sum(tb3.allow() for _ in range(3))
print("PASS" if refilled >= 2 else f"FAIL after refill allowed only {refilled} of 3")

# Capacity cap
tb4 = TokenBucket(rate=1000, capacity=2)  # huge refill rate but small cap
time.sleep(0.2)                            # would refill 200 tokens if uncapped
# Must be capped at capacity
print("PASS" if tb4.tokens() <= 2.0001 else f"FAIL cap violated: {tb4.tokens()}")
""",
    ),

    ComplexProblem(
        name="lru_cache_class",
        category="plan_code",
        prompt=(
            "Implement an `LRUCache` class with O(1) operations. Use a dict + doubly "
            "linked list OR `collections.OrderedDict`. API:\n\n"
            "- `__init__(self, capacity)`: max entries.\n"
            "- `get(self, key)`: returns the value (marking key as most recent), "
            "  or None if absent.\n"
            "- `put(self, key, value)`: inserts or updates. On capacity overflow, "
            "  evicts the LEAST recently used entry.\n"
            "- `__len__(self)`: current size.\n\n"
            "Invariants:\n"
            "- get on missing key returns None without raising\n"
            "- put on existing key updates value AND marks as most recent\n"
            "- When full and inserting new key, least-recently-used key is evicted\n"
            "- get() counts as \"used\" (refreshes recency)"
        ),
        starter="",
        required=["LRUCache"],
        test_code="""
c = LRUCache(3)
c.put('a', 1); c.put('b', 2); c.put('c', 3)
print("PASS" if len(c) == 3 else f"FAIL len got {len(c)}")
print("PASS" if c.get('a') == 1 else f"FAIL get a got {c.get('a')}")
print("PASS" if c.get('missing') is None else "FAIL missing should return None")

# Now a is most recent. Insert d → should evict b (LRU)
c.put('d', 4)
print("PASS" if c.get('b') is None else "FAIL b should have been evicted")
print("PASS" if c.get('a') == 1 else "FAIL a should still be here")
print("PASS" if c.get('d') == 4 else "FAIL d should be here")

# Update existing key
c2 = LRUCache(2)
c2.put('x', 1); c2.put('y', 2)
c2.put('x', 99)    # update + refresh
c2.put('z', 3)     # evicts y (x is more recent)
print("PASS" if c2.get('y') is None else "FAIL y should be evicted after update-refresh")
print("PASS" if c2.get('x') == 99 else f"FAIL x got {c2.get('x')}")

# Length after fills
c3 = LRUCache(2)
c3.put('a', 1); c3.put('b', 2); c3.put('c', 3)
print("PASS" if len(c3) == 2 else f"FAIL len should stay at capacity got {len(c3)}")
""",
    ),
]


# -------------------------------------------------------------------
# Format-agnostic code extractor
# -------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)
_DEF_OR_CLASS = re.compile(r"^(def\s+\w+|class\s+\w+)\b", re.MULTILINE)


def extract_code(text: str, required: List[str]) -> str:
    """Return extractable Python code from Gemma's output.

    Strategy (in order, first that yields AST-valid Python wins):
      1. ```python-fenced block(s) — concatenate all that contain required name
      2. Slice from first def/class to EOF, strip anything after a blank outdent
      3. Entire raw output, trusting Gemma to have emitted only code
    """
    from calm.backends.ast_ops import ast_parse

    candidates: List[str] = []

    # 1. Fenced blocks — prefer those containing required names
    fence_matches = _FENCE_RE.findall(text)
    for m in fence_matches:
        if any(r in m for r in required):
            candidates.append(m)
    # Any fence blocks as fallback
    candidates.extend(fence_matches)

    # 2. Slice from first def/class
    m = _DEF_OR_CLASS.search(text)
    if m:
        tail = text[m.start():]
        # cut at first triple-backtick (could close a fence Gemma opened)
        cut = tail.find("```")
        if cut >= 0:
            tail = tail[:cut]
        # cut at turn markers
        for mark in ("<turn|>", "<end_of_turn>", "<|turn>", "<start_of_turn>"):
            k = tail.find(mark)
            if k >= 0:
                tail = tail[:k]
        candidates.append(tail)

    # 3. Whole raw text as last resort
    candidates.append(text)

    # Return first AST-valid candidate that contains ALL required names
    for c in candidates:
        parsed = ast_parse(c)
        if not parsed.get("valid"):
            continue
        names = {f["name"] for f in parsed.get("functions", [])}
        names |= {cl["name"] for cl in parsed.get("classes", [])}
        if all(r in names for r in required):
            return c

    # Fall back: longest AST-valid candidate regardless of names
    for c in sorted(candidates, key=len, reverse=True):
        parsed = ast_parse(c)
        if parsed.get("valid"):
            return c

    return ""


# -------------------------------------------------------------------
# Prompting (chat template)
# -------------------------------------------------------------------

BASE_SYSTEM = "You are a careful, correct Python coding assistant."

STOCK_PROMPT = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "{prompt}\n\n"
    "Write ONLY the Python code — use ```python fencing. "
    "No explanation, no prose.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

HINTED_PROMPT = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "{hints}\n\n"
    "{prompt}\n\n"
    "Write ONLY the Python code — use ```python fencing. "
    "No explanation, no prose.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


def _trim_markers(text: str) -> str:
    for mark in ("<turn|>", "<end_of_turn>", "<|turn>", "<start_of_turn>"):
        i = text.find(mark)
        if i >= 0:
            text = text[:i]
    return text


def gen_stock(m, tok, p: ComplexProblem,
              max_tokens: Optional[int] = None,
              use_tq4_kv: bool = False) -> str:
    prompt = STOCK_PROMPT.format(system=BASE_SYSTEM, prompt=p.prompt)
    budget = max_tokens if max_tokens is not None else _adaptive_budget(p.prompt)
    out = m.generate(prompt, tok, max_tokens=budget, device="cuda",
                     stop_on_eos=True, use_tq4_kv=use_tq4_kv)
    return _trim_markers(out["text"])


def _build_hints(db, rng: random.Random, p: ComplexProblem,
                 sanity_random: bool) -> str:
    """Return a formatted hints block. If sanity_random=True, pick
    random unrelated examples (length-matched control)."""
    from calm.llm_computer.facades.code_verifier import CodeVerifierFacade
    facade = CodeVerifierFacade(db=db, top_k=2)
    hints = facade.compute_hints(p.prompt)
    if sanity_random:
        # Overwrite retrieved with random examples
        n = len(db.examples)
        if n > 0:
            random_indices = rng.sample(range(n), min(2, n))
            from calm.llm_computer.facades.code_example_db import RetrievalHit
            hints.retrieved_examples = [
                RetrievalHit(example=db.examples[i], score=0.0)
                for i in random_indices
            ]
    block = hints.to_system_prefix(max_example_chars=240)
    # Hard cap
    if len(block) > 2400:
        block = block[:2400] + "\n..."
    return block


def gen_hinted(m, tok, p: ComplexProblem, db, rng: random.Random,
               sanity_random: bool = False,
               max_tokens: Optional[int] = None,
               use_tq4_kv: bool = False) -> str:
    hints = _build_hints(db, rng, p, sanity_random)
    prompt = HINTED_PROMPT.format(
        system=BASE_SYSTEM, hints=hints, prompt=p.prompt)
    budget = max_tokens if max_tokens is not None else _adaptive_budget(p.prompt)
    out = m.generate(prompt, tok, max_tokens=budget, device="cuda",
                     stop_on_eos=True, use_tq4_kv=use_tq4_kv)
    return _trim_markers(out["text"])


# -------------------------------------------------------------------
# Scoring
# -------------------------------------------------------------------

def score(raw_output: str, p: ComplexProblem) -> tuple[int, int, str]:
    """Returns (passed, total, diagnostic). Runs extracted code + tests
    in sandbox. diagnostic is first FAIL line or error summary."""
    from calm.sandbox import run_python

    code = extract_code(raw_output, list(p.required))
    if not code:
        return 0, 0, "no extractable code"

    script = code + "\n\n" + p.test_code + "\npass\n"
    r = run_python(script, timeout=8.0)
    if r.error:
        return 0, 0, f"err: {str(r.error)[:80]}"
    out = r.stdout or ""
    passed = out.count("PASS")
    failed = out.count("FAIL")
    total = passed + failed
    diag = ""
    if failed:
        fail_lines = [l for l in out.splitlines() if l.startswith("FAIL")]
        if fail_lines:
            diag = fail_lines[0][:80]
    return passed, total, diag


# -------------------------------------------------------------------
# Eval runner
# -------------------------------------------------------------------

def run_eval(m, tok, max_tokens: int = 8192, seed: int = 0) -> None:
    from calm.llm_computer.facades.code_example_db import CodeExampleDB
    db = CodeExampleDB.load_default()
    db.load_indices("/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db")
    rng = random.Random(seed)

    print(f"[r53.2b] facade DB: {len(db)} "
          f"(tfidf={db.has_tfidf()}, dense={db.has_dense()})", flush=True)
    print(f"[r53.2b] corpus: {len(CORPUS)} complex problems", flush=True)
    print(f"[r53.2b] conditions: STOCK, HINTED, SANITY-RANDOM", flush=True)
    print()

    rows = []
    totals = {"stock": [0, 0], "hinted": [0, 0], "sanity": [0, 0]}

    for i, p in enumerate(CORPUS):
        print(f"[{i + 1}/{len(CORPUS)}] {p.name} ({p.category})", flush=True)

        # STOCK
        raw_s = gen_stock(m, tok, p, max_tokens)
        s_pass, s_tot, s_diag = score(raw_s, p)
        totals["stock"][0] += s_pass
        totals["stock"][1] += s_tot

        # HINTED (real retrieval)
        raw_h = gen_hinted(m, tok, p, db, rng, sanity_random=False,
                           max_tokens=max_tokens)
        h_pass, h_tot, h_diag = score(raw_h, p)
        totals["hinted"][0] += h_pass
        totals["hinted"][1] += h_tot

        # SANITY-RANDOM (same prompt length, wrong content)
        raw_r = gen_hinted(m, tok, p, db, rng, sanity_random=True,
                           max_tokens=max_tokens)
        r_pass, r_tot, r_diag = score(raw_r, p)
        totals["sanity"][0] += r_pass
        totals["sanity"][1] += r_tot

        print(f"  stock   {s_pass}/{s_tot}  {s_diag[:60]}", flush=True)
        print(f"  hinted  {h_pass}/{h_tot}  {h_diag[:60]}", flush=True)
        print(f"  sanity  {r_pass}/{r_tot}  {r_diag[:60]}", flush=True)
        rows.append((p.name, p.category,
                     (s_pass, s_tot), (h_pass, h_tot), (r_pass, r_tot)))

    print()
    print("=" * 80, flush=True)
    print(f"  {'name':<28} {'cat':<12} {'stock':>8} {'hinted':>8} {'sanity':>8}",
          flush=True)
    print("-" * 80, flush=True)
    for r in rows:
        name, cat, s, h, rr = r
        print(f"  {name:<28} {cat:<12} "
              f"{s[0]:>3}/{s[1]:<3} "
              f"{h[0]:>3}/{h[1]:<3} "
              f"{rr[0]:>3}/{rr[1]:<3}", flush=True)
    print("-" * 80, flush=True)
    sp, st = totals["stock"]
    hp, ht = totals["hinted"]
    rp, rt = totals["sanity"]
    print(f"  TOTAL: stock {sp}/{st}  hinted {hp}/{ht}  sanity {rp}/{rt}",
          flush=True)
    print()
    delta_real = (hp / ht if ht else 0) - (sp / st if st else 0)
    delta_sanity = (rp / rt if rt else 0) - (sp / st if st else 0)
    print(f"  Δ hinted-vs-stock : {delta_real * 100:+.1f}pp", flush=True)
    print(f"  Δ sanity-vs-stock : {delta_sanity * 100:+.1f}pp  "
          f"(control for prompt length)", flush=True)
    real_gain = delta_real - delta_sanity
    print(f"  retrieval-attributable gain: {real_gain * 100:+.1f}pp", flush=True)


# Daemon entrypoint
if "m" in globals() and "tok" in globals():
    _reload_facades()
    run_eval(m, tok)                                     # noqa: F821
elif __name__ == "__main__":
    print("daemon globals `m`, `tok` not found — run via bin/gemma-run",
          flush=True)
