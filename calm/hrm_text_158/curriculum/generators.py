"""Phase 3 curriculum synthetic generators.

Per codex msg 1779458774209 + 1779459000384 (2-axis curriculum locked):

Axis 1 — math complexity under stable language wrapper:
  R0: digit copy / format        `what is N?` -> `N`
  R1: identity-operator bridge   `what is A plus 0?` -> `A`   (also `0 plus A`, `A minus 0`)
                                 [redefined per codex msg 1779466025267 after R1 v1
                                  failed acquisition; bridges R0 copy -> two-operand parsing
                                  + operator binding without full arithmetic]
  R1b: minimal arithmetic (±1)   `what is A plus 1?` -> `A+1` (also `1 plus A`, `A minus 1`)
                                 [codex msg 1779467425298 after R1 identity-bridge pass at
                                  c6e94578; min-incremental jump from identity to actual
                                  arithmetic, output constrained to [0,99] to avoid new
                                  digit-length class]
  R2: carry multi-digit ±        `what is 47 plus 28?` -> `75`
  R3: multiplication only        `what is A times B?` -> `C`  (NO division)
  R4: multi-step compound        `what is A plus B times C?` -> `D`

Axis 2 — language variability grows, math stable from R0-R4:
  R5: paraphrases over R0-R4 primitives
  R6: templated word problems
  R7: GSM8k full corpus (delegated to existing load_gsm8k_splits)

All generators are deterministic per (rung, seed, split). split="train" and
split="held_out" produce NON-OVERLAPPING ROWS via either disjoint operand
ranges (R2/R4) or deterministic row partitioning over a shared support
(R0/R1/R1b/R3 enumerate / stratified partitions). Cross-rung train/held_out
invariant enforced by splits.assert_no_train_holdout_overlap.

Win/falsifier per codex msg 1779457170889 + 1779458774209:
- monotonic rung accuracy
- low forgetting (>90% retention on prior rungs)
- canonical 17×23 tracks multiplication-rung mastery
"""
from __future__ import annotations

import hashlib
import random
from typing import Iterable, Literal


RUNG_NAMES = ("R0", "R1", "R1b", "R2", "R3", "R4", "R5", "R6", "R7")


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
#     R0 uses STRATIFIED in-distribution partition over [0,99]
#     (codex msg 1779464341737 after v1 OOD-shift fix): bucket [0,9] and
#     bucket [10,99] each partitioned 80/20 separately so train + held_out
#     both contain 1- and 2-digit Ns. No operand >= 100 in either split.
#
#   - R1 (identity-bridge per codex msg 1779466025267): three templates
#     `A plus 0` / `0 plus A` / `A minus 0`, all output A. A in [0,99]
#     stratified by (template, digit-bucket) 80/20. R2 cannot collide:
#     R2 train requires B in [10,99] so no R2 row has B=0; R1 always
#     has B=0 or A=0 in the question text. Disjoint by row content.
#
#   - R1b (minimal ±1 bridge per codex msg 1779467425298): three templates
#     `A plus 1` / `1 plus A` / `A minus 1` with per-template A ranges
#     chosen to drop the cross-rung row collisions:
#         A_plus_1   A in [1,98]  -- drop A=0 (else "what is 0 plus 1?" -> 1
#                                   duplicates R1 0_plus_A A=1)
#                                  -- cap A=98 (keeps output <= 99, no new
#                                     digit-length class)
#         1_plus_A   A in [2,98]  -- drop A=0 (else "what is 1 plus 0?" -> 1
#                                   duplicates R1 A_plus_0 A=1)
#                                  -- drop A=1 (else "what is 1 plus 1?" -> 2
#                                   duplicates intra-R1b A_plus_1 A=1)
#         A_minus_1  A in [1,99]  -- drop A=0 (output would be -1, schema
#                                   mismatch)
#     Output stays in [0,99]. R2 cannot collide: R2 train requires
#     B in [10,99] so no R2 row has B=1. Pool 234 train + 60 held_out.
#
#   - R3 template "what is A times B?" is UNIQUE to R3 (no other rung
#     uses "times" with this shape). 17×23 canonical FORCED in held_out.
#
#   - R4 template "what is A op1 B op2 C?" is UNIQUE (3-operand structure).
#
#   - R5 uses arithmetic-operator chars (+, -, *) instead of words; R6
#     uses word-problem templates. Both have unique surface forms.
#
# Implementation: R0 + R1 + R1b + R3 enumerate full operand spaces +
# deterministic shuffle-partition (stratified by template / digit-bucket).
# R2 / R4 use disjoint operand ranges (no overlap by construction).
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
    # R1 (codex msg 1779466025267 redefinition): identity-bridge.
    # 3 templates (A_plus_0 / 0_plus_A / A_minus_0) × [0,99] A values,
    # stratified by (template, digit-bucket) 80/20.
    # Train 239 (template, A) pairs; held_out 60. Both splits contain
    # all 3 templates AND both 1-digit + 2-digit A. "0_plus_A" drops A=0
    # (collision with "A_plus_0" + A=0, both emit "what is 0 plus 0?").
    "R1": {
        "train":     {"A_range": (0, 99), "partition": "enumerate_stratified_identity"},
        "held_out":  {"A_range": (0, 99), "partition": "enumerate_stratified_identity"},
    },
    # R1b (codex msg 1779467425298): minimal arithmetic bridge from R1
    # identity to single-digit ±1. Templates A_plus_1 / 1_plus_A /
    # A_minus_1 with A-range constrained per-template to prevent
    # cross-template AND cross-rung row collisions:
    #   A_plus_1:  A in [1,98]  (drop A=0 -> avoids R1 0_plus_A/A=1 collision)
    #   1_plus_A:  A in [2,98]  (drop A=0 -> avoids R1 A_plus_0/A=1 collision;
    #                             drop A=1 -> avoids intra-R1b A_plus_1/A=1 collision)
    #   A_minus_1: A in [1,99]  (drop A=0 to avoid negative output)
    # Output stays in [0,99] -> no new digit-length class.
    "R1b": {
        "train":     {"A_range": (1, 99), "partition": "enumerate_stratified_pm1"},
        "held_out":  {"A_range": (1, 99), "partition": "enumerate_stratified_pm1"},
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


R1_IDENTITY_TEMPLATES = ("A_plus_0", "0_plus_A", "A_minus_0")


def _enumerate_partition_r1(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1's identity-bridge
    operand space (codex msg 1779466025267 redefinition).

    R1 v1 (single-digit ± on [0,9]^2) failed acquisition at 0.230: the
    rung combined two-operand parsing + operator binding + arithmetic
    + unseen-pair generalization into a single jump from R0. R1
    redefined as IDENTITY BRIDGE: 3 templates (A_plus_0 / 0_plus_A /
    A_minus_0), all evaluating to A. Teaches two-operand parsing +
    operator binding without yet requiring actual arithmetic
    computation; preserves R0's digit-copy primitive on the output.

    Stratification (per codex Step 1 spec): each
    (template, digit-bucket) pair partitioned 80/20 separately with a
    bucket-distinct `_stable_seed("R1_identity_partition", seed,
    template, bucket)`. Result:
      one-digit bucket A in [0,9]: 8 train + 2 held_out PER template
      two-digit bucket A in [10,99]: 72 train + 18 held_out PER template
      total: 80 + 79 + 80 = 239 train pairs
             (one fewer than 3 * (8 + 72) = 240 because "0_plus_A" drops
             A=0 to prevent row collision with "A_plus_0" + A=0; see
             row-collision-fix comment in this function's body)
             20 + 20 + 20 = 60 held_out pairs

    Both splits contain ALL 3 templates AND BOTH digit-length buckets.

    Cross-rung-train invariant: R1 identity rows have B=0 or A=0 in the
    question text; R2 train (`A in [10,99] B in [10,99]`) cannot emit
    B=0 rows; R3 uses "times" not "plus"/"minus"; R5/R6 use distinct
    surface forms. assert_no_train_holdout_overlap verifies by full row
    equality.
    """
    train_set: set = set()
    held_out_set: set = set()
    for template in R1_IDENTITY_TEMPLATES:
        for bucket_label, lo, hi in (("one_digit", 0, 9), ("two_digit", 10, 99)):
            bucket_as = list(range(lo, hi + 1))
            # ROW-COLLISION FIX: ('0_plus_A', 0) emits the same row as
            # ('A_plus_0', 0): both yield "what is 0 plus 0?" -> 0.
            # Drop A=0 from "0_plus_A" so the row exists in exactly one
            # partition cell. Multi-seed sweep confirmed 50% of random
            # seeds otherwise cross-side-collide on this row. ("A_minus_0"
            # with A=0 emits "what is 0 minus 0?" which is unique;
            # safe to keep.)
            if template == "0_plus_A" and bucket_label == "one_digit":
                bucket_as = [a for a in bucket_as if a != 0]
            rng = random.Random(
                _stable_seed("R1_identity_partition", seed, template, bucket_label)
            )
            rng.shuffle(bucket_as)
            split = int(len(bucket_as) * train_frac)
            train_set.update((template, a) for a in bucket_as[:split])
            held_out_set.update((template, a) for a in bucket_as[split:])
    return train_set, held_out_set


R1B_TEMPLATES = ("A_plus_1", "1_plus_A", "A_minus_1")


def _enumerate_partition_r1b(seed: int, train_frac: float = 0.8) -> tuple[set, set]:
    """Stratified deterministic 80/20 partition of R1b's ±1 operand space
    (codex msg 1779467425298 after R1 identity-bridge pass).

    R1b is the minimal-arithmetic bridge from R1 identity to actual
    single-digit ±1. Three templates with per-template A constraints
    chosen to prevent cross-template AND cross-rung row collisions:

      A_plus_1:  A in [1, 98]    -- "what is A plus 1?"  -> A+1
                                 -- drop A=0: would emit "what is 0 plus 1?" -> 1,
                                    duplicating R1 0_plus_A with A=1
                                 -- cap A=98: keeps output <= 99 (no 3-digit class)
      1_plus_A:  A in [2, 98]    -- "what is 1 plus A?"  -> A+1
                                 -- drop A=0: would emit "what is 1 plus 0?" -> 1,
                                    duplicating R1 A_plus_0 with A=1
                                 -- drop A=1: would emit "what is 1 plus 1?" -> 2,
                                    duplicating intra-R1b A_plus_1 with A=1
                                 -- cap A=98: keeps output <= 99
      A_minus_1: A in [1, 99]    -- "what is A minus 1?" -> A-1
                                 -- drop A=0: would emit "what is 0 minus 1?" -> -1
                                    (negative output, schema mismatch)

    Pool sizes per template:
      A_plus_1:  [1,9]   ( 9 vals) -> 7 train + 2 held;
                 [10,98] (89 vals) -> 71 train + 18 held; total 78 train + 20 held
      1_plus_A:  [2,9]   ( 8 vals) -> 6 train + 2 held;
                 [10,98] (89 vals) -> 71 train + 18 held; total 77 train + 20 held
      A_minus_1: [1,9]   ( 9 vals) -> 7 train + 2 held;
                 [10,99] (90 vals) -> 72 train + 18 held; total 79 train + 20 held
      TOTAL: 234 train + 60 held_out pairs

    All splits contain ALL 3 templates AND BOTH digit-length buckets.
    """
    train_set: set = set()
    held_out_set: set = set()

    template_specs = {
        "A_plus_1":  {"one_digit": list(range(1, 10)),  "two_digit": list(range(10, 99))},
        "1_plus_A":  {"one_digit": list(range(2, 10)),  "two_digit": list(range(10, 99))},
        "A_minus_1": {"one_digit": list(range(1, 10)),  "two_digit": list(range(10, 100))},
    }

    for template in R1B_TEMPLATES:
        for bucket_label in ("one_digit", "two_digit"):
            bucket_as = list(template_specs[template][bucket_label])
            rng = random.Random(
                _stable_seed("R1b_partition", seed, template, bucket_label)
            )
            rng.shuffle(bucket_as)
            split = int(len(bucket_as) * train_frac)
            train_set.update((template, a) for a in bucket_as[:split])
            held_out_set.update((template, a) for a in bucket_as[split:])
    return train_set, held_out_set


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
    """R1 identity-bridge (codex msg 1779466025267 redefinition).

    3 templates × A in [0,99] stratified by (template, digit-bucket):
      `what is A plus 0?` -> A     (template A_plus_0)
      `what is 0 plus A?` -> A     (template 0_plus_A)
      `what is A minus 0?` -> A    (template A_minus_0)

    Output = A in every case (preserves R0 digit-copy primitive on the
    answer side). Trains two-operand parsing + operator binding without
    yet requiring arithmetic computation.

    Train pool = 239 (template, A) pairs; held_out pool = 60. The
    "0_plus_A" template drops A=0 to prevent row collision with
    "A_plus_0" + A=0 (both emit "what is 0 plus 0?"). Both splits
    contain ALL 3 templates AND BOTH digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)  # deterministic order before rng.choice
    out = []
    while len(out) < n:
        template, A = rng.choice(pool_list)
        if template == "A_plus_0":
            q = f"what is {A} plus 0?"
        elif template == "0_plus_A":
            q = f"what is 0 plus {A}?"
        elif template == "A_minus_0":
            q = f"what is {A} minus 0?"
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown R1 identity template: {template!r}")
        out.append({"question": q, "expected": A, "rung": "R1"})
    return out


def _gen_r1b(rng: random.Random, spec: dict, n: int, seed: int, split: str) -> list[dict]:
    """R1b ±1 (codex msg 1779467425298 after R1 identity pass at c6e94578).

    3 templates × stratified-partitioned A range:
      `what is A plus 1?`  -> A+1  (A in [1,98])
      `what is 1 plus A?`  -> A+1  (A in [2,98])
      `what is A minus 1?` -> A-1  (A in [1,99])

    Output stays in [0,99]; no new digit-length class. Train pool = 234
    (template, A) pairs; held_out pool = 60. Both splits contain all 3
    templates + both digit-length buckets."""
    train_pool, held_out_pool = _enumerate_partition_r1b(seed)
    pool = train_pool if split == "train" else held_out_pool
    pool_list = sorted(pool)
    out = []
    while len(out) < n:
        template, A = rng.choice(pool_list)
        if template == "A_plus_1":
            q = f"what is {A} plus 1?"
            expected = A + 1
        elif template == "1_plus_A":
            q = f"what is 1 plus {A}?"
            expected = A + 1
        elif template == "A_minus_1":
            q = f"what is {A} minus 1?"
            expected = A - 1
        else:  # pragma: no cover - exhaustive
            raise ValueError(f"unknown R1b template: {template!r}")
        out.append({"question": q, "expected": expected, "rung": "R1b"})
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
    if rung == "R1b":
        return _gen_r1b(rng, _RUNG_SPEC["R1b"][split], n, seed=seed, split=split)
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
