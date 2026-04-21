"""R53 Phase 1 eval — facade hints vs stock Gemma on code problems.

Runs 12 coding problems through:
  (A) stock Gemma — plain prompt
  (B) Gemma + CodeVerifierFacade hints prepended to the prompt

Scores each completion via CodeVerifierFacade.verify() using hand-
written test cases. Reports per-prompt win/loss/tie and aggregate.

This is the gate that decides whether to invest in PT training +
L24/L30 install: if the facade's verified-context injection already
moves the needle on stock Gemma, the substrate install is worth
building; if not, the hypothesis needs rework before sinking days
into training.

Run via daemon (assumes `m`, `tok` pre-loaded):

  bin/gemma-run scripts/r53_eval_phase1.py
"""

from __future__ import annotations

import re
from typing import List, NamedTuple, Optional


# NOTE: @dataclass in a daemon-exec'd script crashes dataclasses' type
# resolution because the script's module isn't registered in
# sys.modules. Use NamedTuple instead — same constructor, works fine
# under exec.

# When run via daemon, calm.llm_computer.facades.* may already be
# imported from a prior session. Force reload so we pick up any
# recent edits to retrieval.py / code_example_db.py.
def _reload_facades():
    import sys
    for m in list(sys.modules.keys()):
        if (m.startswith("calm.llm_computer.facades.")
                or m == "calm.llm_computer.facades"):
            del sys.modules[m]


# ----- problem corpus -----

class Problem(NamedTuple):
    name: str
    prompt: str             # NL problem statement
    signature: str          # the function-def line Gemma should complete
    test_code: str          # uses `print("PASS"/"FAIL")` — count to score


CORPUS: List[Problem] = [
    Problem(
        name="is_prime",
        prompt="Write a Python function `is_prime(n)` that returns True if n is a prime number, False otherwise. Handle n <= 1 correctly.",
        signature="def is_prime(n):",
        test_code="""
for n, expected in [(2, True), (3, True), (4, False), (1, False), (0, False), (-3, False), (17, True), (15, False), (97, True), (100, False)]:
    got = is_prime(n)
    print("PASS" if got == expected else f"FAIL n={n} got={got} expected={expected}")
""",
    ),
    Problem(
        name="gcd",
        prompt="Write a Python function `gcd(a, b)` that returns the greatest common divisor of two non-negative integers. gcd(0, 0) should return 0.",
        signature="def gcd(a, b):",
        test_code="""
for a, b, expected in [(12, 18, 6), (100, 75, 25), (17, 5, 1), (0, 7, 7), (7, 0, 7), (0, 0, 0), (48, 18, 6)]:
    got = gcd(a, b)
    print("PASS" if got == expected else f"FAIL gcd({a},{b}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="fibonacci",
        prompt="Write an iterative Python function `fib(n)` that returns the n-th Fibonacci number. fib(0) = 0, fib(1) = 1.",
        signature="def fib(n):",
        test_code="""
for n, expected in [(0, 0), (1, 1), (2, 1), (3, 2), (10, 55), (15, 610), (20, 6765)]:
    got = fib(n)
    print("PASS" if got == expected else f"FAIL fib({n}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="balanced_parens",
        prompt="Write a Python function `balanced(s)` that returns True if the string contains balanced parentheses '(', ')', '[', ']', '{', '}'. Other characters should be ignored.",
        signature="def balanced(s):",
        test_code="""
for s, expected in [("()", True), ("()[]", True), ("(]", False), ("({[]})", True), ("(((", False), ("", True), ("a(b)c", True), ("(]()", False)]:
    got = balanced(s)
    print("PASS" if got == expected else f"FAIL balanced({s!r}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="binary_search",
        prompt="Write a Python function `bsearch(arr, target)` that returns the index of target in a sorted list, or -1 if not found. Use binary search.",
        signature="def bsearch(arr, target):",
        test_code="""
arr = [1, 3, 5, 7, 9, 11, 13, 15]
for t, expected in [(7, 3), (1, 0), (15, 7), (4, -1), (100, -1), (-5, -1)]:
    got = bsearch(arr, t)
    print("PASS" if got == expected else f"FAIL bsearch({t}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="roman_to_int",
        prompt="Write a Python function `roman_to_int(s)` that converts a Roman numeral string to an integer. Handle subtractive notation (IV=4, IX=9, XL=40, XC=90, CD=400, CM=900).",
        signature="def roman_to_int(s):",
        test_code="""
for s, expected in [("III", 3), ("IV", 4), ("IX", 9), ("LVIII", 58), ("MCMXCIV", 1994), ("XL", 40), ("CD", 400), ("MMMCMXCIX", 3999)]:
    got = roman_to_int(s)
    print("PASS" if got == expected else f"FAIL roman({s}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="flatten",
        prompt="Write a Python function `flatten(x)` that flattens arbitrarily nested lists into a single flat list. Non-list elements should be kept as-is.",
        signature="def flatten(x):",
        test_code="""
for x, expected in [([1, [2, 3]], [1, 2, 3]), ([[1, 2], [3, [4, [5]]]], [1, 2, 3, 4, 5]), ([], []), ([1, 2, 3], [1, 2, 3]), ([[], []], [])]:
    got = flatten(x)
    print("PASS" if got == expected else f"FAIL flatten({x}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="levenshtein",
        prompt="Write a Python function `levenshtein(a, b)` that returns the Levenshtein edit distance between two strings.",
        signature="def levenshtein(a, b):",
        test_code="""
for a, b, expected in [("kitten", "sitting", 3), ("", "abc", 3), ("abc", "", 3), ("same", "same", 0), ("ab", "ba", 2), ("cat", "cats", 1)]:
    got = levenshtein(a, b)
    print("PASS" if got == expected else f"FAIL lev({a!r},{b!r}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="caesar",
        prompt="Write a Python function `caesar(text, shift)` that applies a Caesar cipher to letters. Preserves case, leaves non-letters unchanged. Shift can be negative or > 26.",
        signature="def caesar(text, shift):",
        test_code="""
for text, shift, expected in [("abc", 1, "bcd"), ("xyz", 3, "abc"), ("Hello, World!", 13, "Uryyb, Jbeyq!"), ("abc", 0, "abc"), ("abc", -1, "zab"), ("abc", 27, "bcd")]:
    got = caesar(text, shift)
    print("PASS" if got == expected else f"FAIL caesar({text!r},{shift}) got={got!r} expected={expected!r}")
""",
    ),
    Problem(
        name="run_length",
        prompt="Write a Python function `rle_encode(s)` that runs a run-length encoding of a string. Returns a string like 'aaabb' -> 'a3b2'. Single chars append '1'.",
        signature="def rle_encode(s):",
        test_code="""
for s, expected in [("aaabb", "a3b2"), ("a", "a1"), ("", ""), ("abcd", "a1b1c1d1"), ("aabbaa", "a2b2a2")]:
    got = rle_encode(s)
    print("PASS" if got == expected else f"FAIL rle({s!r}) got={got!r} expected={expected!r}")
""",
    ),
    Problem(
        name="two_sum",
        prompt="Write a Python function `two_sum(nums, target)` that returns the indices (i, j) of two numbers in nums that add to target. Return None if no solution.",
        signature="def two_sum(nums, target):",
        test_code="""
for nums, target, expected in [([2, 7, 11, 15], 9, (0, 1)), ([3, 2, 4], 6, (1, 2)), ([3, 3], 6, (0, 1)), ([1, 2, 3], 100, None)]:
    got = two_sum(nums, target)
    ok = (got == expected) or (expected is not None and got is not None and set(got) == set(expected))
    print("PASS" if ok else f"FAIL two_sum({nums},{target}) got={got} expected={expected}")
""",
    ),
    Problem(
        name="safe_url",
        prompt="Write a Python function `is_safe_url(url)` that returns True only if the URL's hostname is NOT a private/loopback/internal address. Block: 127.0.0.0/8, 10.0.0.0/8, 192.168.0.0/16, 172.16.0.0/12, 169.254.0.0/16, 0.0.0.0, and hostnames like 'localhost', 'metadata.google.internal'. Return False for non-http(s) schemes.",
        signature="def is_safe_url(url):",
        test_code="""
for url, expected in [
    ("https://example.com", True),
    ("http://127.0.0.1/admin", False),
    ("http://localhost/", False),
    ("http://10.0.0.5/", False),
    ("http://192.168.1.1/", False),
    ("http://169.254.169.254/", False),
    ("https://metadata.google.internal/", False),
    ("file:///etc/passwd", False),
    ("javascript:alert(1)", False),
    ("https://8.8.8.8/", True),
]:
    got = is_safe_url(url)
    print("PASS" if got == expected else f"FAIL is_safe_url({url!r}) got={got} expected={expected}")
""",
    ),
]


# ----- generation + extraction -----

BASE_SYSTEM = "You are a careful, correct Python coding assistant."

# Gemma 4 E4B was chat-tuned with <start_of_turn>role ... <end_of_turn>
# markers. Raw plain-text prompts cause the model to emit degraded
# output (random <turn|> tokens, partial restarts). Use the chat
# template for both stock + hinted conditions.
STOCK_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "Problem: {prompt}\n\n"
    "Write ONLY the function `{signature_no_colon}` and nothing else.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)

HINTED_PROMPT_TEMPLATE = (
    "<start_of_turn>user\n"
    "{system}\n\n"
    "{hints}\n\n"
    "Problem: {prompt}\n\n"
    "Write ONLY the function `{signature_no_colon}` and nothing else.\n"
    "<end_of_turn>\n"
    "<start_of_turn>model\n"
)


_CODE_FENCE_RE = re.compile(r"```(?:python)?\n(.*?)```", re.DOTALL)
_DEF_RE = re.compile(r"(def\s+\w+\s*\([^)]*\)\s*:)", re.MULTILINE)


def extract_function(text: str, signature: str) -> Optional[str]:
    """Pull a parseable function out of Gemma's generation.

    Strategy:
      1. If a ```python code block contains the expected def, take it.
      2. Otherwise, slice from the first `def ` to the first line that
         does not start with whitespace, `def `, or `#`.
      3. Prepend `signature` if missing — Gemma often continues from
         our prefix without re-emitting the def.
    """
    # 1. Code fence
    for m in _CODE_FENCE_RE.finditer(text):
        body = m.group(1)
        if "def " in body:
            return body

    # 2. Slice from first def
    # Signature may appear or Gemma may start with indented body.
    if "def " in text:
        start = text.find("def ")
        tail = text[start:]
    else:
        # No def in generation — Gemma continued from `signature\n`.
        # Take the raw continuation and prepend signature.
        tail = signature + "\n" + text

    # Cut at the first line that looks like outside-function text
    # (starts with no-indent alphanumeric AND is not def/class/import).
    lines = tail.splitlines()
    end = len(lines)
    for i, ln in enumerate(lines):
        if i == 0:
            continue
        if not ln.strip():
            continue
        stripped = ln.lstrip()
        # Outside-function signals: inline commentary / next problem
        if ln[0].isalpha() and not (
            stripped.startswith("def ")
            or stripped.startswith("class ")
            or stripped.startswith("import ")
            or stripped.startswith("from ")
            or stripped.startswith("#")
        ):
            end = i
            break
    return "\n".join(lines[:end]).rstrip() + "\n"


def _sig_no_colon(signature: str) -> str:
    return signature.rstrip().rstrip(":").strip()


def gen_stock(m, tok, p: Problem, max_tokens: int = 16384) -> str:
    prompt = STOCK_PROMPT_TEMPLATE.format(
        system=BASE_SYSTEM, prompt=p.prompt,
        signature_no_colon=_sig_no_colon(p.signature))
    out = m.generate(prompt, tok, max_tokens=max_tokens,
                     device="cuda", stop_on_eos=True)
    return _trim_turn_markers(out["text"])


def gen_hinted(m, tok, p: Problem, facade,
               max_tokens: int = 16384) -> str:
    # Daemon's Gemma has max_len=1024 positional embeddings. Stock
    # prompt (chat-wrapped) ~120 tokens + max_tokens 200 = ~320 used.
    # Leaves ~700 tokens headroom for hint block.
    facade.top_k = 2
    hints = facade.compute_hints(p.prompt)
    hint_block = hints.to_system_prefix(max_example_chars=200)
    if len(hint_block) > 2000:
        hint_block = hint_block[:2000] + "\n..."
    prompt = HINTED_PROMPT_TEMPLATE.format(
        system=BASE_SYSTEM, hints=hint_block, prompt=p.prompt,
        signature_no_colon=_sig_no_colon(p.signature))
    out = m.generate(prompt, tok, max_tokens=max_tokens,
                     device="cuda", stop_on_eos=True)
    return _trim_turn_markers(out["text"])


def _trim_turn_markers(text: str) -> str:
    """Chop output at any <turn|> / <end_of_turn> marker — Gemma emits
    these as in-vocab tokens when the chat template isn't fully
    recognized by the tokenizer's EOS logic."""
    for marker in ("<turn|>", "<end_of_turn>", "<|turn>",
                    "<start_of_turn>"):
        i = text.find(marker)
        if i >= 0:
            text = text[:i]
    return text


# ----- eval loop -----

def run_eval(m, tok, max_tokens: int = 16384) -> None:
    # Load facade + attach prebuilt hybrid indices
    from calm.llm_computer.facades.code_example_db import CodeExampleDB
    from calm.llm_computer.facades.code_verifier import CodeVerifierFacade
    db = CodeExampleDB.load_default()
    db.load_indices("/mnt/c/Users/gabes/projects/claw-code/.cache/r53_code_db")
    facade = CodeVerifierFacade(db=db)
    print(f"[r53.2] facade DB: {len(facade.db)} examples "
          f"(tfidf={facade.db.has_tfidf()}, dense={facade.db.has_dense()})",
          flush=True)
    print(f"[r53.2] corpus: {len(CORPUS)} problems", flush=True)
    print()

    rows = []
    stock_pass = 0
    hinted_pass = 0
    gains = []
    regressions = []

    for i, p in enumerate(CORPUS):
        print(f"[{i + 1}/{len(CORPUS)}] {p.name}", flush=True)

        # Stock
        raw_stock = gen_stock(m, tok, p, max_tokens)
        code_stock = extract_function(raw_stock, p.signature) or ""
        v_stock = facade.verify(
            code=code_stock if "def " in code_stock else (
                p.signature + "\n" + code_stock),
            test_code=p.test_code)
        s_ok = v_stock.ok and (
            v_stock.tests_total and v_stock.tests_passed == v_stock.tests_total)

        # Hinted
        raw_hinted = gen_hinted(m, tok, p, facade, max_tokens)
        code_hinted = extract_function(raw_hinted, p.signature) or ""
        v_hinted = facade.verify(
            code=code_hinted if "def " in code_hinted else (
                p.signature + "\n" + code_hinted),
            test_code=p.test_code)
        h_ok = v_hinted.ok and (
            v_hinted.tests_total and v_hinted.tests_passed == v_hinted.tests_total)

        stock_pass += int(bool(s_ok))
        hinted_pass += int(bool(h_ok))

        verdict = "tie" if s_ok == h_ok else ("GAIN" if (h_ok and not s_ok) else "REGR")
        if verdict == "GAIN":
            gains.append(p.name)
        elif verdict == "REGR":
            regressions.append(p.name)

        s_score = f"{v_stock.tests_passed or 0}/{v_stock.tests_total or 0}"
        h_score = f"{v_hinted.tests_passed or 0}/{v_hinted.tests_total or 0}"
        rows.append((p.name, s_score, h_score, verdict,
                     v_stock.sandbox_error or v_stock.errors[:1],
                     v_hinted.sandbox_error or v_hinted.errors[:1]))
        print(f"  stock={s_score}  hinted={h_score}  {verdict}", flush=True)

    print()
    print("=" * 76, flush=True)
    print(f"  {'name':<16} {'stock':>7} {'hinted':>8} {'verdict':>8}",
          flush=True)
    print("-" * 76, flush=True)
    for r in rows:
        print(f"  {r[0]:<16} {r[1]:>7} {r[2]:>8} {r[3]:>8}", flush=True)
    print("-" * 76, flush=True)
    print(f"  stock pass : {stock_pass}/{len(CORPUS)}", flush=True)
    print(f"  hinted pass: {hinted_pass}/{len(CORPUS)}", flush=True)
    print(f"  GAINS      : {gains}", flush=True)
    print(f"  REGRESSIONS: {regressions}", flush=True)
    print()
    delta = hinted_pass - stock_pass
    print(f"  NET Δ      : {delta:+d} "
          f"({'FACADE WINS' if delta > 0 else 'FACADE TIES' if delta == 0 else 'FACADE LOSES'})",
          flush=True)


# Daemon entrypoint
if "m" in globals() and "tok" in globals():
    _reload_facades()
    run_eval(m, tok)  # type: ignore[name-defined]
elif __name__ == "__main__":
    print("daemon globals `m`, `tok` not found — run via bin/gemma-run",
          flush=True)
