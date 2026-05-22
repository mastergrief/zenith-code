"""Phase 3 curriculum synthetic generators.

Per codex msg 1779458774209 + 1779459000384 (2-axis curriculum locked):

Axis 1 — math complexity under stable language wrapper:
  R0: digit copy / format        `what is N?` -> `N`
  R1: single-digit ±             `what is A plus B?` -> `C`
  R2: carry multi-digit ±        `what is 47 plus 28?` -> `75`
  R3: multiplication only        `what is A times B?` -> `C`  (NO division)
  R4: multi-step compound        `what is A plus B times C?` -> `D`

Axis 2 — language variability grows, math stable from R0-R4:
  R5: paraphrases over R0-R4 primitives
  R6: templated word problems
  R7: GSM8k full corpus (delegated to existing load_gsm8k_splits)

All generators are deterministic per (rung, seed, split). split="train" and
split="held_out" use NON-OVERLAPPING operand ranges to prevent leakage.
Cross-rung train/held_out invariant enforced by splits.assert_no_train_holdout_overlap.

Win/falsifier per codex msg 1779457170889 + 1779458774209:
- monotonic rung accuracy
- low forgetting (>90% retention on prior rungs)
- canonical 17×23 tracks multiplication-rung mastery
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, Literal


RUNG_NAMES = ("R0", "R1", "R2", "R3", "R4", "R5", "R6", "R7")


def _stable_seed(*parts) -> int:
    """Process-stable seed derivation.

    Python's builtin `hash()` is salted per-process under default
    `PYTHONHASHSEED=random`, so `hash(("R1_partition", 42))` returns
    different values across restarts — that would silently produce
    different train/held_out partitions per run, invalidating
    retention probes and ckpt comparability.

    `hashlib.sha256(repr(parts).encode("utf-8"))` is deterministic
    across processes and Python versions for the same input tuple.
    """
    blob = repr(parts).encode("utf-8")
    digest = hashlib.sha256(blob).digest()[:4]
    return int.from_bytes(digest, "little")


# Per-rung sampling spec.
#
# CROSS-RUNG INVARIANT (per codex msg 1779458774209): NO row in any rung's
# held_out may appear in ANY rung's train. Operand ranges chosen to make
# this hold by construction:
#
#   - R0 template "what is N?" is UNIQUE to R0 (no other rung uses it).
#     Train [0,99] vs held_out [100,999] gives no internal overlap.
#
#   - R1 + R2 share the "what is A plus/minus B?" template.
#     SOLUTION: R1 confined to single-digit [0,9]²; R2 confined to
#     [10,99]² for train AND [100,999]² for held_out. Cross-rung-train
#     can't see R1's held_out because R1 held_out lives in a strict
#     subset of [0,9]² withheld via deterministic partition (NOT
#     [10,99] which would alias with R2 train).
#
#   - R3 template "what is A times B?" is UNIQUE to R3 (no other rung
#     uses "times" with this shape). 17×23 canonical FORCED in held_out.
#
#   - R4 template "what is A op1 B op2 C?" is UNIQUE (3-operand structure).
#
#   - R5 uses arithmetic-operator chars (+, -, *) instead of words; R6
#     uses word-problem templates. Both have unique surface forms.
#
# Implementation: R0 + R1 + R3 enumerate full operand spaces + deterministic-
# shuffle-partition. R2 / R4 use disjoint operand ranges (no overlap by
# construction).
#
# R0 design correction (codex msg 1779464341737-43a42cae after R0 launch
# msg 1779464300667 reported G1 fail):
#   - Previous R0 design: train [0,99] vs held_out [100,999]. Tests OOD
#     length generalization, not in-distribution digit-copy memorization
#     (model learned 1-2 digit copy but couldn't extend to 3-digit -> G1=0).
#   - New R0 design: STRATIFIED in-distribution partition over [0,99].
#     Train + held_out both contain 1-digit AND 2-digit examples. Bucket
#     [0,9] partitioned 80/20 separately from [10,99] so the digit-length
#     mix is preserved on both sides. Tests in-distribution memorization
#     of digit copy (the actual R0 primitive).
#   - R0 max N <= 99 in both splits. OOD length generalization is NOT
#     an R0 gate.

_RUNG_SPEC: dict[str, dict[str, dict]] = {
    "R0": {
        # R0 stratified partition: bucket [0,9] (one-digit) + bucket [10,99]
        # (two-digit) split 80/20 each, then unioned per split. Held_out
        # max N <= 99 — in-distribution memorization gate.
        "train":     {"N_range": (0, 99), "partition": "enumerate_stratified"},
        "held_out":  {"N_range": (0, 99), "partition": "enumerate_stratified"},
    },
    # R1 uses enumerate-partition (see _enumerate_partition_r1) — operand
    # range constant [0,9]² for both train + held_out, deterministically
    # split. This entry is informational only.
    "R1": {
        "train":     {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
        "held_out":  {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
    },
    "R2": {
        "train":     {"A_range": (10, 99), "B_range": (10, 99)},
        "held_out":  {"A_range": (100, 999), "B_range": (100, 999)},
    },
    # R3 uses enumerate-partition over [0,9]² × {times}. 17×23 specifically
    # force-injected into held_out (out of [0,9]² range).
    "R3": {
        "train":     {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
        "held_out":  {"A_range": (0, 9), "B_range": (0, 9), "partition": "enumerate"},
    },
    "R4": {
        "train":     {"A_range": (0, 9), "B_range": (0, 9), "C_range": (0, 9)},
        "held_out":  {"A_range": (10, 30), "B_range": (10, 30), "C_range": (10, 30)},
    },
    # R5/R6 use template banks rather than operand ranges; R7 = GSM8k (out of scope)
}


def _enumerate_partition_r0(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R0's [0,99] operand space.

    Per codex msg 1779464341737-43a42cae: split bucket [0,9] (one-digit)
    AND bucket [10,99] (two-digit) separately with bucket-distinct
    `_stable_seed("R0_partition", seed, bucket)` so a flat shuffle can't
    accidentally segregate all 1-digit Ns onto one side.

    Result: train ~= 8 one-digit + 72 two-digit; held_out ~= 2 one-digit
    + 18 two-digit. Both splits contain both digit-length classes -> tests
    in-distribution digit-copy memorization, not OOD length generalization.

    Cross-rung-train doesn't see R0's held_out because R0's template
    `what is N?` is UNIQUE to R0 (no other rung uses single-operand
    completion).
    """
    train_set: set = set()
    held_out_set: set = set()
    # Bucket 1: one-digit [0,9]  (10 Ns -> 8 train / 2 held_out at 0.8)
    # Bucket 2: two-digit [10,99] (90 Ns -> 72 train / 18 held_out at 0.8)
    for bucket_label, lo, hi in (("one_digit", 0, 9), ("two_digit", 10, 99)):
        bucket_ns = list(range(lo, hi + 1))
        rng = random.Random(_stable_seed("R0_partition", seed, bucket_label))
        rng.shuffle(bucket_ns)
        split = int(len(bucket_ns) * train_frac)
        train_set.update(bucket_ns[:split])
        held_out_set.update(bucket_ns[split:])
    return train_set, held_out_set


def _enumerate_partition_r1(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Deterministic 80/20 partition of R1's operand space.
    Returns (train_set, held_out_set) of (A, B, op) tuples.
    Cross-rung-train can't see R1 held_out because R2 train is [10,99]² (disjoint)
    and R3+ use different templates."""
    pairs = [(A, B, op) for A in range(10) for B in range(10) for op in ("plus", "minus")]
    rng = random.Random(_stable_seed("R1_partition", seed))
    rng.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    return set(pairs[:split]), set(pairs[split:])


def _enumerate_partition_r3(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Deterministic 80/20 partition of R3's operand space [0,9]² × {times}.
    17×23 canonical force-injected into held_out (out-of-range A=17 B=23)."""
    pairs = [(A, B) for A in range(10) for B in range(10)]
    rng = random.Random(_stable_seed("R3_partition", seed))
    rng.shuffle(pairs)
    split = int(len(pairs) * train_frac)
    train = set(pairs[:split])
    held_out = set(pairs[split:])
    # Force 17×23 into held_out (canonical mastery gate)
    held_out.add((17, 23))
    return train, held_out


# Phrasings for R5 paraphrase rung
_R5_PHRASINGS_TRAIN = [
    "what is {expr}?",
    "compute {expr}",
    "{expr} equals what?",
]
_R5_PHRASINGS_HELDOUT = [
    "what's {expr}",
    "calculate {expr}",
    "the value of {expr} is what?",
]

# Templates for R6 word problems
_R6_TEMPLATES_TRAIN = [
    ("Alice has {A} apples. Bob gives her {B} more. How many apples does Alice have?",  # +
     lambda A, B: A + B),
    ("There are {A} cookies in a jar. {B} are eaten. How many cookies are left?",        # -
     lambda A, B: A - B),
    ("A box contains {A} items. There are {B} such boxes. How many items in total?",     # *
     lambda A, B: A * B),
]
_R6_TEMPLATES_HELDOUT = [
    ("Carlos collected {A} stamps. He buys {B} more at a fair. How many stamps now?",    # +
     lambda A, B: A + B),
    ("A library has {A} books. {B} are checked out. How many remain?",                   # -
     lambda A, B: A - B),
    ("Each shelf holds {A} books. There are {B} shelves. How many books in total?",      # *
     lambda A, B: A * B),
]


def _make_rng(rung: str, seed: int, split: str) -> random.Random:
    """Stable per-(rung, seed, split) RNG. Different splits get distinct
    seeds derived from the input seed so train and held_out never share
    state."""
    salt = {"train": 0, "held_out": 17}[split]
    return random.Random(_stable_seed(rung, seed, salt))


def _gen_r0(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R0: digit copy `what is N? -> N`.

    Stratified in-distribution partition over [0,99] (codex msg
    1779464341737-43a42cae). Train pool ~80 Ns (8 one-digit + 72 two-digit);
    held_out pool ~20 Ns (2 one-digit + 18 two-digit). Both contain both
    digit-length classes. NO operand >= 100 in either split."""
    train_pool, held_out_pool = _enumerate_partition_r0(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order before rng.choice
    out = []
    while len(out) < n:
        N = rng.choice(pool_list)
        out.append({"question": f"what is {N}?", "expected": N, "rung": "R0"})
    return out


def _gen_r1(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1: single-digit ± on [0,9]². Uses deterministic 80/20 partition
    of operand space to prevent cross-rung overlap with R2 train ([10,99]²).
    Train pool = 160 unique (A, B, op) tuples; held_out pool = 40."""
    train_pool, held_out_pool = _enumerate_partition_r1(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order before rng.choice
    out = []
    while len(out) < n:
        A, B, op = rng.choice(pool_list)
        if op == "plus":
            expected = A + B
            q = f"what is {A} plus {B}?"
        else:
            expected = A - B
            q = f"what is {A} minus {B}?"
        out.append({"question": q, "expected": expected, "rung": "R1"})
    return out


def _gen_r2(rng: random.Random, spec: dict, n: int) -> list[dict]:
    # Same template shape as R1 but multi-digit (carry needed)
    a_lo, a_hi = spec["A_range"]
    b_lo, b_hi = spec["B_range"]
    out = []
    while len(out) < n:
        A = rng.randint(a_lo, a_hi)
        B = rng.randint(b_lo, b_hi)
        op = rng.choice(["plus", "minus"])
        if op == "plus":
            expected = A + B
            q = f"what is {A} plus {B}?"
        else:
            expected = A - B
            q = f"what is {A} minus {B}?"
        out.append({"question": q, "expected": expected, "rung": "R2"})
    return out


def _gen_r3(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R3: multiplication-only on [0,9]² with 17×23 force-injected into
    held_out. NO division per codex spec (formatting/remainder ambiguity).
    Train pool = 80 unique (A, B) pairs; held_out pool = 20 + (17, 23)."""
    train_pool, held_out_pool = _enumerate_partition_r3(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order
    out = []
    while len(out) < n:
        A, B = rng.choice(pool_list)
        expected = A * B
        q = f"what is {A} times {B}?"
        out.append({"question": q, "expected": expected, "rung": "R3"})
    return out


def _gen_r4(rng: random.Random, spec: dict, n: int) -> list[dict]:
    # Multi-step compound: (A op B) op C with no division
    a_lo, a_hi = spec["A_range"]
    b_lo, b_hi = spec["B_range"]
    c_lo, c_hi = spec["C_range"]
    out = []
    while len(out) < n:
        A = rng.randint(a_lo, a_hi)
        B = rng.randint(b_lo, b_hi)
        C = rng.randint(c_lo, c_hi)
        # Mix +, -, ×; honor operator precedence (× before +/-)
        op1 = rng.choice(["plus", "minus", "times"])
        op2 = rng.choice(["plus", "minus", "times"])
        # Evaluate respecting precedence
        # Convert to Python and eval safely (small ints, no float)
        py_op = {"plus": "+", "minus": "-", "times": "*"}
        expr = f"{A} {py_op[op1]} {B} {py_op[op2]} {C}"
        # Restrict to int-safe arithmetic
        expected = eval(expr, {"__builtins__": {}})
        q = f"what is {A} {op1} {B} {op2} {C}?"
        out.append({"question": q, "expected": expected, "rung": "R4"})
    return out


def _gen_r5(rng: random.Random, n: int, split: str) -> list[dict]:
    """Paraphrase rung — same math primitives as R0-R4 but varied phrasing.
    train: 3 phrasings; held_out: 3 different phrasings."""
    phrasings = _R5_PHRASINGS_TRAIN if split == "train" else _R5_PHRASINGS_HELDOUT
    out = []
    while len(out) < n:
        # Choose primitive
        primitive_rung = rng.choice(["R1", "R3"])  # single-digit ± or ×
        A = rng.randint(0, 9)
        B = rng.randint(0, 9)
        if primitive_rung == "R1":
            op = rng.choice(["+", "-"])
            expr = f"{A} {op} {B}"
            expected = A + B if op == "+" else A - B
        else:
            expr = f"{A} * {B}"
            expected = A * B
        template = rng.choice(phrasings)
        q = template.format(expr=expr)
        out.append({"question": q, "expected": expected, "rung": "R5"})
    return out


def _gen_r6(rng: random.Random, n: int, split: str) -> list[dict]:
    """Templated word problem rung. Train and held_out use disjoint
    template banks (with same math operations) so the model must transfer
    primitives across novel surface form."""
    templates = _R6_TEMPLATES_TRAIN if split == "train" else _R6_TEMPLATES_HELDOUT
    out = []
    while len(out) < n:
        template, fn = rng.choice(templates)
        A = rng.randint(1, 20)
        B = rng.randint(1, 20)
        # For subtraction templates, ensure A >= B (so result non-negative
        # for the word-problem framing)
        if "checked out" in template or "eaten" in template:
            A = max(A, B)
        expected = fn(A, B)
        q = template.format(A=A, B=B)
        out.append({"question": q, "expected": expected, "rung": "R6"})
    return out


def make_rung_examples(
    rung: str,
    n: int,
    *,
    seed: int,
    split: Literal["train", "held_out"],
) -> list[dict]:
    """Deterministic synthetic examples for a single rung.

    Returns list of dicts: {"question": str, "expected": int, "rung": str}.

    R7 (GSM8k) is NOT generated here — it's served from
    scripts.train_hrm_text_158.load_gsm8k_splits directly.
    """
    if rung == "R7":
        raise ValueError(
            "R7 is GSM8k; load from load_gsm8k_splits() rather than make_rung_examples"
        )
    if rung not in RUNG_NAMES:
        raise ValueError(f"unknown rung {rung!r}; valid: {RUNG_NAMES}")
    if split not in ("train", "held_out"):
        raise ValueError(f"split must be 'train' or 'held_out'; got {split!r}")
    if n <= 0:
        raise ValueError(f"n must be positive; got {n}")

    rng = _make_rng(rung, seed, split)
    if rung == "R0":
        return _gen_r0(rng, _RUNG_SPEC["R0"][split], n, seed=seed, split=split)
    if rung == "R1":
        return _gen_r1(rng, _RUNG_SPEC["R1"][split], n, seed=seed, split=split)
    if rung == "R2":
        return _gen_r2(rng, _RUNG_SPEC["R2"][split], n)
    if rung == "R3":
        return _gen_r3(rng, _RUNG_SPEC["R3"][split], n, seed=seed, split=split)
    if rung == "R4":
        return _gen_r4(rng, _RUNG_SPEC["R4"][split], n)
    if rung == "R5":
        return _gen_r5(rng, n, split)
    if rung == "R6":
        return _gen_r6(rng, n, split)
    raise NotImplementedError(f"generator for {rung!r} not implemented")
