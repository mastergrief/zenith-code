"""HRM-Text-1.58 Phase 3 Step 0 curriculum infrastructure tests.

Per task #51, board task 1779460303130-742c8cbd, codex msg 1779460698439
(Phase 3 Step 0 +1 with A1 byte-level UTF-8 + 7 guardrails).

Covers:
- BroadTokenizer determinism + vocab spec + roundtrip
- Synthetic generators (R0-R1, R1b, R2-R6) determinism + held-out non-overlap
- Cross-rung invariant (held_out ∩ all_train = ∅)
- Retention probe schema + delta computation + G2 gate
- --load-from ckpt compat validation (vocab/normalizer/ternary/arch mismatch)
- Code-syntax smoke string coverage
- Length histogram BroadTokenizer vs GSM8k char tokenizer (sequence-length gate)

NO model training. NO probe ckpt build. Pure data/probe infra.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

import pytest

sys.path.insert(0, ".")

from calm.hrm_text_158.curriculum import (
    BROAD_NORMALIZER_VERSION,
    BroadTokenizer,
    RUNG_NAMES,
    RungProbeResult,
    assert_no_train_holdout_overlap,
    build_rung_splits,
    compute_retention_deltas,
    make_rung_examples,
    validate_load_from_ckpt_compat,
)
from calm.hrm_text_158.curriculum.retention import check_retention_gate


# ----------------------------------------------------------------------------- #
# Code-syntax smoke strings (codex msg 1779460698439: must cover indent,
# underscores, braces, brackets, quotes, backticks, operators, comments, imports, newlines)
# ----------------------------------------------------------------------------- #

CODE_SMOKE_STRINGS = [
    "def f(x): return x + 1",
    "for i in range(10):",
    "    print('hello world')",  # indent
    "if not x: pass  # comment",  # hash comment
    "x = [1, 2, 3]; y = {'a': 1}",  # brackets + braces + quotes
    "result = obj.method(arg1, arg2)",
    "import numpy as np",
    "class Foo(Bar):",
    "    def __init__(self):",
    "        self._x = 0",  # underscore + nested indent
    "lambda x: x ** 2",  # operators
    "@decorator\ndef wrapped(): ...",  # decorator + newline
    "with open('f.txt') as f: data = f.read()",
    "regex = r'\\d+\\.\\d+'",  # backslash + raw string
    "f-string: f'{name!r}: {value}'",  # f-string punct
    "x is not None and y >= 5",
    "// not-python comment style",
    "/* multi-line comment */",
    "fn rust(x: i32) -> i32 { x + 1 }",
    "let result = await fetch('/api');",
]


# ============================================================================ #
# BroadTokenizer: determinism, vocab spec, encode/decode roundtrip
# ============================================================================ #

def test_broad_tokenizer_vocab_deterministic() -> None:
    """Two BroadTokenizer instances must produce identical vocab lists.
    NEVER built from corpus."""
    tok_a = BroadTokenizer()
    tok_b = BroadTokenizer()
    assert tok_a.vocab_as_list() == tok_b.vocab_as_list()


def test_broad_tokenizer_vocab_size_260() -> None:
    """4 specials + 256 byte values = 260 total."""
    tok = BroadTokenizer()
    assert tok.vocab_size == 260
    assert len(tok.vocab_as_list()) == 260


def test_broad_tokenizer_special_ids() -> None:
    """Specials at ids 0-3 in fixed order."""
    tok = BroadTokenizer()
    assert tok.pad_id == 0
    assert tok.bos_id == 1
    assert tok.eos_id == 2
    assert tok.sep_id == 3


def test_broad_tokenizer_byte_ids_start_at_4() -> None:
    """Byte values 0x00-0xff map to ids 4-259 (offset by 4)."""
    tok = BroadTokenizer()
    vocab = tok.vocab_as_list()
    assert vocab[4] == "<byte:00>"
    assert vocab[5] == "<byte:01>"
    assert vocab[259] == "<byte:ff>"


def test_broad_tokenizer_normalizer_version() -> None:
    """normalizer_version = byte_utf8_v1, identity normalization."""
    tok = BroadTokenizer()
    assert tok.normalizer_version == BROAD_NORMALIZER_VERSION
    assert BROAD_NORMALIZER_VERSION == "byte_utf8_v1"


def test_broad_tokenizer_ascii_encode() -> None:
    """ASCII text: each char encodes to (byte + 4)."""
    tok = BroadTokenizer()
    ids = tok.encode("abc")
    # 'a' = 0x61 = 97 → id = 101
    # 'b' = 0x62 = 98 → id = 102
    # 'c' = 0x63 = 99 → id = 103
    assert ids == [101, 102, 103]


def test_broad_tokenizer_ascii_roundtrip_smoke_strings() -> None:
    """All CODE_SMOKE_STRINGS encode then decode back exactly."""
    tok = BroadTokenizer()
    for s in CODE_SMOKE_STRINGS:
        ids = tok.encode(s)
        decoded = tok.decode(ids, stop_at_eos=False)
        assert decoded == s, f"roundtrip failed: {s!r} -> {decoded!r}"


def test_broad_tokenizer_non_ascii_encode_multi_byte() -> None:
    """Non-ASCII chars use multiple bytes; e.g. 'é' = 2 bytes."""
    tok = BroadTokenizer()
    ids = tok.encode("é")
    # 'é' UTF-8 = b'\xc3\xa9' = 2 bytes
    assert len(ids) == 2
    # Roundtrip exact
    assert tok.decode(ids, stop_at_eos=False) == "é"


def test_broad_tokenizer_encode_example_shape() -> None:
    """encode_example returns (ids, sep_pos) with shape:
        ids = [bos, q_chars..., sep, t_chars..., eos]
        sep_pos = 1 + len(q_chars)"""
    tok = BroadTokenizer()
    ids, sep_pos = tok.encode_example("what is 1?", 1)
    # Expected layout:
    # [bos=1, w=119+4, h=104+4, a=97+4, t=116+4, ' '=32+4, i=105+4, s=115+4, ' '=32+4, 1=49+4, ?=63+4, sep=3, 1=49+4, eos=2]
    assert ids[0] == 1  # bos
    assert ids[-1] == 2  # eos
    assert sep_pos == 1 + len("what is 1?")
    assert ids[sep_pos] == 3  # sep at expected position


def test_broad_tokenizer_assert_corpus_covered_never_raises() -> None:
    """Byte-level is OOV-free; assert_corpus_covered is a no-op for any content."""
    tok = BroadTokenizer()
    # Even crazy Unicode + control chars must not raise
    rows = [
        {"question": "what is 1?", "expected": 1},
        {"question": "héllo 中文 🎉", "expected": 0},
        {"question": "\x00\x01\x02 control bytes", "expected": 0},
    ]
    tok.assert_corpus_covered(rows, label="test")


def test_broad_tokenizer_assert_corpus_covered_rejects_non_dict() -> None:
    """Defensive: rows must be dicts; raises TypeError on bad shape."""
    tok = BroadTokenizer()
    with pytest.raises(TypeError):
        tok.assert_corpus_covered([("a", 1)], label="test")  # tuple, not dict


def test_broad_tokenizer_decode_skips_pad() -> None:
    """<pad> tokens render as empty during decode."""
    tok = BroadTokenizer()
    ids = [tok.bos_id, tok.pad_id, tok.pad_id] + tok.encode("hi") + [tok.eos_id]
    decoded = tok.decode(ids, stop_at_eos=True)
    assert "<bos>" in decoded
    assert "hi" in decoded
    assert "<pad>" not in decoded


# ============================================================================ #
# Generators: determinism + held-out non-overlap
# ============================================================================ #

@pytest.mark.parametrize("rung", ["R0", "R1", "R1b1", "R1b2a", "R1b2", "R1b3", "R1b4", "R1b", "R2a", "R2", "R3", "R4", "R5", "R6"])
def test_generator_deterministic_per_seed(rung) -> None:
    """Same (rung, seed, split) -> same examples list."""
    examples_a = make_rung_examples(rung, n=20, seed=42, split="train")
    examples_b = make_rung_examples(rung, n=20, seed=42, split="train")
    assert examples_a == examples_b


@pytest.mark.parametrize("rung", ["R0", "R1", "R1b1", "R1b2a", "R1b2", "R1b3", "R1b4", "R1b", "R2a", "R2", "R3", "R4", "R5", "R6"])
def test_generator_train_holdout_distinct(rung) -> None:
    """Train and held_out splits produce different examples for same seed
    (different RNG salt per split)."""
    train = make_rung_examples(rung, n=20, seed=42, split="train")
    held_out = make_rung_examples(rung, n=20, seed=42, split="held_out")
    # At least one example should differ
    assert train != held_out


# ============================================================================ #
# R0 stratified in-distribution partition (codex msg 1779464341737-43a42cae)
# ============================================================================ #

def _r0_n(ex: dict) -> int:
    """Extract the operand N from an R0 example's question `what is N?`."""
    return int(ex["question"].split()[2].rstrip("?"))


def test_generator_r0_train_holdout_exact_row_disjoint() -> None:
    """R0 train + held_out must NEVER share an example row.

    Stratified partition splits [0,9] and [10,99] separately; train and
    held_out pools are disjoint by construction. n=2000 sampling exhausts
    both pools repeatedly; verify zero overlap on the sampled rows."""
    train = make_rung_examples("R0", n=2000, seed=42, split="train")
    held = make_rung_examples("R0", n=2000, seed=42, split="held_out")
    train_keys = {(ex["question"], ex["expected"]) for ex in train}
    held_keys = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_keys & held_keys
    assert not overlap, f"R0 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r0_held_out_max_n_le_99() -> None:
    """R0 held_out operand N must be in [0,99] (in-distribution).

    Previous design had held_out [100,999] (OOD length shift); fixed
    design (codex msg 1779464341737) keeps held_out in [0,99]."""
    held = make_rung_examples("R0", n=500, seed=42, split="held_out")
    for ex in held:
        n = _r0_n(ex)
        assert 0 <= n <= 99, (
            f"R0 held_out has N={n} outside [0,99] — OOD length shift not allowed"
        )


def test_generator_r0_train_max_n_le_99() -> None:
    """R0 train operand N must also be in [0,99]."""
    train = make_rung_examples("R0", n=500, seed=42, split="train")
    for ex in train:
        n = _r0_n(ex)
        assert 0 <= n <= 99, f"R0 train has N={n} outside [0,99]"


def test_generator_r0_train_contains_both_digit_lengths() -> None:
    """Stratified partition ensures R0 train has BOTH 1-digit and 2-digit Ns.

    A flat shuffle of 100 Ns could accidentally put all 1-digit Ns on one
    side; codex msg 1779464341737 requires bucket-stratified partition."""
    train = make_rung_examples("R0", n=500, seed=42, split="train")
    has_one_digit = any(_r0_n(ex) < 10 for ex in train)
    has_two_digit = any(_r0_n(ex) >= 10 for ex in train)
    assert has_one_digit, "R0 train missing 1-digit examples (stratification bug)"
    assert has_two_digit, "R0 train missing 2-digit examples (stratification bug)"


def test_generator_r0_held_out_contains_both_digit_lengths() -> None:
    """Stratified partition ensures R0 held_out has BOTH 1-digit and 2-digit Ns."""
    held = make_rung_examples("R0", n=500, seed=42, split="held_out")
    has_one_digit = any(_r0_n(ex) < 10 for ex in held)
    has_two_digit = any(_r0_n(ex) >= 10 for ex in held)
    assert has_one_digit, "R0 held_out missing 1-digit examples (stratification bug)"
    assert has_two_digit, "R0 held_out missing 2-digit examples (stratification bug)"


def test_generator_r0_partition_stable_across_pythonhashseed() -> None:
    """R0 stratified partition output identical across PYTHONHASHSEED values
    (uses sha256-stable _stable_seed, not builtin hash)."""
    import json
    import os
    import subprocess
    import sys

    code = (
        "import json\n"
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r0\n"
        "train, held = _enumerate_partition_r0(42)\n"
        "print(json.dumps({'train': sorted(train), 'held': sorted(held)}))\n"
    )

    def _run(pyhs: str) -> str:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["PYTHONHASHSEED"] = pyhs
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, cwd=".", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    a = _run("0")
    b = _run("999")
    c = _run("random")
    assert a == b == c, f"R0 partition diverged: PYHS=0:{a[:80]} PYHS=999:{b[:80]}"


def test_generator_r0_partition_pool_sizes() -> None:
    """Stratified partition sizes: 8 one-digit train + 72 two-digit train,
    2 one-digit held_out + 18 two-digit held_out (codex msg 1779464341737
    recommendation)."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r0
    train, held = _enumerate_partition_r0(42)
    train_one = {n for n in train if n < 10}
    train_two = {n for n in train if n >= 10}
    held_one = {n for n in held if n < 10}
    held_two = {n for n in held if n >= 10}
    assert len(train_one) == 8, f"train one-digit count: {len(train_one)} expected 8"
    assert len(train_two) == 72, f"train two-digit count: {len(train_two)} expected 72"
    assert len(held_one) == 2, f"held one-digit count: {len(held_one)} expected 2"
    assert len(held_two) == 18, f"held two-digit count: {len(held_two)} expected 18"
    # Disjoint by construction
    assert (train_one | train_two) & (held_one | held_two) == set()


# ============================================================================ #
# R1 identity-bridge stratified partition (codex msg 1779466025267 redefinition)
# ============================================================================ #

_R1_IDENTITY_QS = {
    # (template_key, A) -> expected question string
    "A_plus_0":  lambda A: f"what is {A} plus 0?",
    "0_plus_A":  lambda A: f"what is 0 plus {A}?",
    "A_minus_0": lambda A: f"what is {A} minus 0?",
}


def _r1_identity_decode(ex: dict) -> tuple[str, int]:
    """Identify (template_key, A) from an R1 identity-bridge example."""
    q = ex["question"]
    if " plus 0?" in q and not q.startswith("what is 0"):
        # "what is A plus 0?"
        A = int(q.split()[2])
        return ("A_plus_0", A)
    if q.startswith("what is 0 plus "):
        # "what is 0 plus A?"
        A = int(q.split()[4].rstrip("?"))
        return ("0_plus_A", A)
    if " minus 0?" in q:
        # "what is A minus 0?"
        A = int(q.split()[2])
        return ("A_minus_0", A)
    raise ValueError(f"unrecognized R1 identity question shape: {q!r}")


def test_generator_r1_identity_train_holdout_exact_row_disjoint() -> None:
    """R1 train + held_out must NEVER share an example row (codex Step 1
    assertion: exact-row disjoint)."""
    train = make_rung_examples("R1", n=2000, seed=42, split="train")
    held = make_rung_examples("R1", n=2000, seed=42, split="held_out")
    train_keys = {(ex["question"], ex["expected"]) for ex in train}
    held_keys = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_keys & held_keys
    assert not overlap, f"R1 identity train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1_identity_all_templates_in_train_and_holdout() -> None:
    """R1 stratification by template ensures BOTH splits contain ALL 3
    identity templates (A_plus_0 / 0_plus_A / A_minus_0)."""
    train = make_rung_examples("R1", n=500, seed=42, split="train")
    held = make_rung_examples("R1", n=500, seed=42, split="held_out")
    train_templates = {_r1_identity_decode(ex)[0] for ex in train}
    held_templates = {_r1_identity_decode(ex)[0] for ex in held}
    expected = {"A_plus_0", "0_plus_A", "A_minus_0"}
    assert train_templates == expected, f"R1 train missing template(s): {expected - train_templates}"
    assert held_templates == expected, f"R1 held_out missing template(s): {expected - held_templates}"


def test_generator_r1_identity_both_digit_lengths_in_both_splits() -> None:
    """R1 stratification by digit-bucket ensures BOTH splits contain
    1-digit AND 2-digit A values."""
    train = make_rung_examples("R1", n=500, seed=42, split="train")
    held = make_rung_examples("R1", n=500, seed=42, split="held_out")
    train_one = any(_r1_identity_decode(ex)[1] < 10 for ex in train)
    train_two = any(_r1_identity_decode(ex)[1] >= 10 for ex in train)
    held_one = any(_r1_identity_decode(ex)[1] < 10 for ex in held)
    held_two = any(_r1_identity_decode(ex)[1] >= 10 for ex in held)
    assert train_one, "R1 train missing 1-digit A (stratification bug)"
    assert train_two, "R1 train missing 2-digit A (stratification bug)"
    assert held_one, "R1 held_out missing 1-digit A (stratification bug)"
    assert held_two, "R1 held_out missing 2-digit A (stratification bug)"


def test_generator_r1_identity_max_a_le_99() -> None:
    """R1 A must always be in [0,99] (no OOD length shift, mirror of R0 fix)."""
    train = make_rung_examples("R1", n=500, seed=42, split="train")
    held = make_rung_examples("R1", n=500, seed=42, split="held_out")
    for ex in train + held:
        _, A = _r1_identity_decode(ex)
        assert 0 <= A <= 99, f"R1 has A={A} outside [0,99]"


def test_generator_r1_identity_expected_always_equals_A() -> None:
    """For all 3 identity templates, expected == A (output preserves R0
    digit-copy primitive)."""
    rows = make_rung_examples("R1", n=300, seed=42, split="train") + \
           make_rung_examples("R1", n=300, seed=42, split="held_out")
    for ex in rows:
        _, A = _r1_identity_decode(ex)
        assert ex["expected"] == A, (
            f"R1 identity broken: question={ex['question']!r} expected={ex['expected']} A={A}"
        )


def test_generator_r1_identity_pool_sizes() -> None:
    """Stratified pool sizes: "0_plus_A" drops A=0 to avoid row collision
    with "A_plus_0" (both produce "what is 0 plus 0?" -> 0).

    Pool composition:
      A_plus_0:  [0,9]  -> 8+2,  [10,99] -> 72+18  (80 train + 20 held_out)
      0_plus_A:  [1,9]  -> 7+2,  [10,99] -> 72+18  (79 train + 20 held_out)
      A_minus_0: [0,9]  -> 8+2,  [10,99] -> 72+18  (80 train + 20 held_out)
      total:                    239 train + 60 held_out
    Note: int(9 * 0.8) = 7, hence 7+2 for the 9-element bucket."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1
    train, held = _enumerate_partition_r1(42)
    assert len(train) == 239, f"R1 train pool size: {len(train)} expected 239"
    assert len(held) == 60, f"R1 held_out pool size: {len(held)} expected 60"
    # All 3 templates present in each
    train_templates = {t for t, _ in train}
    held_templates = {t for t, _ in held}
    assert train_templates == {"A_plus_0", "0_plus_A", "A_minus_0"}
    assert held_templates == {"A_plus_0", "0_plus_A", "A_minus_0"}
    # Disjoint by construction
    assert train & held == set()
    # A=0 in "0_plus_A" template MUST be absent (collision-fix)
    assert ("0_plus_A", 0) not in train
    assert ("0_plus_A", 0) not in held


def test_generator_r1_identity_no_row_collision_across_seeds() -> None:
    """Multi-seed sweep regression for the A=0 cross-template collision:
    `("A_plus_0", 0)` and `("0_plus_A", 0)` both emit "what is 0 plus 0?".
    With the row-collision fix dropping A=0 from "0_plus_A", train/held_out
    must remain row-disjoint across multiple seeds."""
    for seed in (0, 1, 7, 42, 999, 12345):
        train = make_rung_examples("R1", n=2000, seed=seed, split="train")
        held = make_rung_examples("R1", n=2000, seed=seed, split="held_out")
        train_keys = {(ex["question"], ex["expected"]) for ex in train}
        held_keys = {(ex["question"], ex["expected"]) for ex in held}
        overlap = train_keys & held_keys
        assert not overlap, (
            f"seed={seed}: R1 identity row collision detected: "
            f"{sorted(overlap)[:5]}"
        )


# ============================================================================ #
# R1b1 single-template +1 stratified partition
# (codex msg 1779469364293 + 1779469638068 falsifier-protocol split
#  after R1b v2 failed at 0.845 with 2x training steps, 0d152dd)
# ============================================================================ #

def _r1b1_a(ex: dict) -> int:
    """Extract A from an R1b1 example: question is `what is A plus 1?`."""
    q = ex["question"]
    assert q.endswith(" plus 1?"), f"R1b1 question must end ' plus 1?': {q!r}"
    # "what is A plus 1?" -> tokens[2] == A
    return int(q.split()[2])


def test_generator_r1b1_in_rung_names_index_2() -> None:
    """R1b1 must sit at RUNG_NAMES index 2 (between R1 and R1b2) so the
    trainer's prior_rungs derivation (RUNG_NAMES[:cur_idx]) auto-resolves
    to (R0, R1) for R1b1 launches and the failed full R1b is excluded
    from the active chain. Codex msg 1779469638068 + 1779471212090."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[2] == "R1b1", f"R1b1 must be at index 2; got {RUNG_NAMES}"
    # R0/R1 sit before R1b1 so prior_rungs[:2] for R1b1 = (R0, R1)
    assert RUNG_NAMES[:2] == ("R0", "R1")


def test_generator_r1b1_train_holdout_exact_row_disjoint() -> None:
    """R1b1 train + held_out must be exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b1", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R1b1 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b1_single_template_only() -> None:
    """R1b1 emits ONLY `what is A plus 1?`; never `A minus 1`, never any
    R1b template (those are diagnosis-only after the falsifier split).

    Note: A=1 legitimately emits `what is 1 plus 1?` (still the
    A_plus_1 template, just with A=1) — single-template invariant is
    verified by suffix structure + absence of 'minus', not by
    prefix-distinguishing from the symmetric 1_plus_A form (impossible
    at A=1)."""
    rows = make_rung_examples("R1b1", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R1b1 prefix violated: {q!r}"
        assert q.endswith(" plus 1?"), f"R1b1 must end ' plus 1?'; got {q!r}"
        # Single-template invariant: must contain exactly one " plus "
        # and zero " minus " occurrences
        assert q.count(" plus ") == 1, f"R1b1 must contain ' plus ' exactly once: {q!r}"
        assert " minus " not in q, f"R1b1 must not emit 'minus' template: {q!r}"
        # Token shape: 'what is <A> plus 1?' -> 5 tokens
        toks = q.split()
        assert len(toks) == 5, f"R1b1 question must have 5 tokens; got {len(toks)}: {q!r}"
        assert toks[0] == "what" and toks[1] == "is" and toks[3] == "plus" and toks[4] == "1?", (
            f"R1b1 template shape violated: {q!r}"
        )


def test_generator_r1b1_no_a_zero() -> None:
    """R1b1 must NEVER emit A=0 -- "what is 0 plus 1?" -> 1 duplicates
    R1's 0_plus_A row with A=1 (cross-rung collision)."""
    rows = make_rung_examples("R1b1", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b1_a(ex)
        assert A != 0, f"R1b1 must not emit A=0; got {ex['question']!r}"


def test_generator_r1b1_no_a_99() -> None:
    """R1b1 must NEVER emit A=99 -- output would be 100, introducing a
    new digit-length class outside R1b1's [0,99] design."""
    rows = make_rung_examples("R1b1", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b1_a(ex)
        assert A != 99, f"R1b1 must not emit A=99; got {ex['question']!r}"
        assert 1 <= A <= 98, f"R1b1 A out of [1,98]; got A={A} q={ex['question']!r}"


def test_generator_r1b1_expected_matches_arithmetic() -> None:
    """For every R1b1 row, expected must equal A+1."""
    rows = make_rung_examples("R1b1", n=500, seed=42, split="train") + \
           make_rung_examples("R1b1", n=500, seed=42, split="held_out")
    for ex in rows:
        A = _r1b1_a(ex)
        assert ex["expected"] == A + 1, (
            f"R1b1 expected mismatch: A={A} expected={ex['expected']}"
        )


def test_generator_r1b1_output_in_2_to_99() -> None:
    """R1b1 output ∈ [2, 99] (A in [1,98] -> A+1 in [2,99])."""
    rows = make_rung_examples("R1b1", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b1", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 2 <= ex["expected"] <= 99, (
            f"R1b1 output out of [2,99]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b1_both_digit_lengths_in_both_splits() -> None:
    """Both splits must contain one-digit AND two-digit A (stratification gate)."""
    train = make_rung_examples("R1b1", n=500, seed=42, split="train")
    held = make_rung_examples("R1b1", n=500, seed=42, split="held_out")
    assert any(_r1b1_a(ex) < 10 for ex in train), "R1b1 train missing 1-digit A"
    assert any(_r1b1_a(ex) >= 10 for ex in train), "R1b1 train missing 2-digit A"
    assert any(_r1b1_a(ex) < 10 for ex in held), "R1b1 held missing 1-digit A"
    assert any(_r1b1_a(ex) >= 10 for ex in held), "R1b1 held missing 2-digit A"


def test_generator_r1b1_pool_sizes() -> None:
    """R1b1 pool sizes per codex msg 1779469638068 correction:
      one_digit [1,9]:   9 vals  -> 7 train + 2 held
      two_digit [10,98]: 89 vals -> 71 train + 18 held
      TOTAL:             98 vals -> 78 train + 20 held
    """
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b1
    train_pool, held_pool = _enumerate_partition_r1b1(seed=42)
    assert len(train_pool) == 78, f"R1b1 train pool must be 78; got {len(train_pool)}"
    assert len(held_pool) == 20, f"R1b1 held_out pool must be 20; got {len(held_pool)}"
    # Pool integers cover [1,98] with NO A=0 and NO A=99
    full = train_pool | held_pool
    assert full == set(range(1, 99)), (
        f"R1b1 pool must equal {{1..98}}; got {sorted(full)[:5]}...{sorted(full)[-5:]}"
    )
    assert 0 not in full, "R1b1 pool must not contain A=0"
    assert 99 not in full, "R1b1 pool must not contain A=99"
    # Disjoint
    assert not (train_pool & held_pool), "R1b1 pools must be disjoint"


def test_generator_r1b1_no_collision_with_r1() -> None:
    """R1b1 rows must NEVER appear in R1's train OR held_out (cross-rung
    invariant for the active chain {R0, R1, R1b1}).

    R1 emits `A plus 0` / `0 plus A` / `A minus 0`; R1b1 emits only
    `A plus 1`. By template-suffix structure these are disjoint. Verify
    empirically at high sample count to catch any latent collision."""
    r1_train = make_rung_examples("R1", n=2000, seed=42, split="train")
    r1_held = make_rung_examples("R1", n=2000, seed=42, split="held_out")
    r1b1_train = make_rung_examples("R1b1", n=2000, seed=42, split="train")
    r1b1_held = make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    r1_rows = {(ex["question"], ex["expected"]) for ex in r1_train + r1_held}
    r1b1_rows = {(ex["question"], ex["expected"]) for ex in r1b1_train + r1b1_held}
    overlap = r1_rows & r1b1_rows
    assert not overlap, f"R1 vs R1b1 row collision: {sorted(overlap)[:5]}"


def test_generator_r1b1_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r1b1 partition must be IDENTICAL across
    PYTHONHASHSEED restarts. Uses _stable_seed not builtin hash()."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b1; "
        "train, held = _enumerate_partition_r1b1(seed=42); "
        "print(','.join(str(x) for x in sorted(train)) + '|' + ','.join(str(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "0"},
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "random"},
    ).decode().strip()
    assert out1 == out2 == out3, (
        f"R1b1 partition diverges across PYTHONHASHSEED: "
        f"PYTHONHASHSEED=0 -> {out1[:60]}...; "
        f"PYTHONHASHSEED=12345 -> {out2[:60]}...; "
        f"PYTHONHASHSEED=random -> {out3[:60]}..."
    )


# ============================================================================ #
# R1b2a low-A subtraction stratified partition
# (codex msg 1779472124507 + 1779472300306 falsifier-protocol split
#  after R1b2 FAILED at 6fd2fec: G1 R1b2=0.860, G2 R1b1 retention -0.050 decay)
# ============================================================================ #

def _r1b2a_a(ex: dict) -> int:
    """Extract A from an R1b2a example: question is `what is A minus 1?`."""
    q = ex["question"]
    assert q.endswith(" minus 1?"), f"R1b2a question must end ' minus 1?': {q!r}"
    return int(q.split()[2])


def test_generator_r1b2a_in_rung_names_index_3() -> None:
    """R1b2a must sit at RUNG_NAMES index 3 (after R1b1, before failed
    R1b2/R1b) so the trainer's prior_rungs derivation auto-resolves to
    (R0, R1, R1b1). Codex msg 1779472300306."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[3] == "R1b2a", f"R1b2a must be at index 3; got {RUNG_NAMES}"
    assert RUNG_NAMES[2] == "R1b1", f"R1b1 must be at index 2; got {RUNG_NAMES}"
    assert RUNG_NAMES[4] == "R1b2", f"R1b2 must be at index 4 (diagnosis-only); got {RUNG_NAMES}"
    # R0/R1/R1b1 sit before R1b2a so prior_rungs[:3] for R1b2a = (R0, R1, R1b1)
    assert RUNG_NAMES[:3] == ("R0", "R1", "R1b1")


def test_generator_r1b2a_train_holdout_exact_row_disjoint() -> None:
    """R1b2a train + held_out must be exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b2a", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R1b2a train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b2a_single_template_only() -> None:
    """R1b2a emits ONLY `what is A minus 1?`; never `A plus 1`, never any
    R1b template."""
    rows = make_rung_examples("R1b2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R1b2a prefix violated: {q!r}"
        assert q.endswith(" minus 1?"), f"R1b2a must end ' minus 1?'; got {q!r}"
        assert q.count(" minus ") == 1, f"R1b2a must contain ' minus ' exactly once: {q!r}"
        assert " plus " not in q, f"R1b2a must not emit 'plus' template: {q!r}"
        toks = q.split()
        assert len(toks) == 5, f"R1b2a question must have 5 tokens; got {len(toks)}: {q!r}"
        assert toks[0] == "what" and toks[1] == "is" and toks[3] == "minus" and toks[4] == "1?", (
            f"R1b2a template shape violated: {q!r}"
        )


def test_generator_r1b2a_no_a_zero() -> None:
    """R1b2a must NEVER emit A=0 -- output would be -1 (negative; schema
    mismatch)."""
    rows = make_rung_examples("R1b2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b2a_a(ex)
        assert A != 0, f"R1b2a must not emit A=0; got {ex['question']!r}"
        assert 1 <= A <= 19, f"R1b2a A out of [1,19]; got A={A} q={ex['question']!r}"


def test_generator_r1b2a_no_a_ge_20() -> None:
    """R1b2a must NEVER emit A >= 20 -- low-A only (no two-digit borrow
    pattern). Codex msg 1779472300306: 'isolates minimal -1 operator on
    low-A operands'."""
    rows = make_rung_examples("R1b2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b2a_a(ex)
        assert A < 20, f"R1b2a must not emit A>=20; got A={A} q={ex['question']!r}"


def test_generator_r1b2a_expected_matches_arithmetic() -> None:
    """For every R1b2a row, expected must equal A-1."""
    rows = make_rung_examples("R1b2a", n=500, seed=42, split="train") + \
           make_rung_examples("R1b2a", n=500, seed=42, split="held_out")
    for ex in rows:
        A = _r1b2a_a(ex)
        assert ex["expected"] == A - 1, (
            f"R1b2a expected mismatch: A={A} expected={ex['expected']}"
        )


def test_generator_r1b2a_output_in_0_to_18() -> None:
    """R1b2a output ∈ [0, 18] (A in [1,19] -> A-1 in [0,18])."""
    rows = make_rung_examples("R1b2a", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b2a", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 0 <= ex["expected"] <= 18, (
            f"R1b2a output out of [0,18]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b2a_both_buckets_in_both_splits() -> None:
    """Both splits must contain one-digit AND teen A (stratification gate)."""
    train = make_rung_examples("R1b2a", n=500, seed=42, split="train")
    held = make_rung_examples("R1b2a", n=500, seed=42, split="held_out")
    assert any(_r1b2a_a(ex) < 10 for ex in train), "R1b2a train missing one_digit A"
    assert any(_r1b2a_a(ex) >= 10 for ex in train), "R1b2a train missing teen A"
    assert any(_r1b2a_a(ex) < 10 for ex in held), "R1b2a held missing one_digit A"
    assert any(_r1b2a_a(ex) >= 10 for ex in held), "R1b2a held missing teen A"


def test_generator_r1b2a_pool_sizes() -> None:
    """R1b2a pool sizes per codex msg 1779472300306:
      one_digit [1,9]:  9 vals  -> 7 train + 2 held
      teen      [10,19]: 10 vals -> 8 train + 2 held
      TOTAL:             19 vals -> 15 train + 4 held
    """
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b2a
    train_pool, held_pool = _enumerate_partition_r1b2a(seed=42)
    assert len(train_pool) == 15, f"R1b2a train pool must be 15; got {len(train_pool)}"
    assert len(held_pool) == 4, f"R1b2a held_out pool must be 4; got {len(held_pool)}"
    full = train_pool | held_pool
    assert full == set(range(1, 20)), (
        f"R1b2a pool must equal {{1..19}}; got {sorted(full)}"
    )
    assert 0 not in full, "R1b2a pool must not contain A=0"
    assert 20 not in full, "R1b2a pool must not contain A=20"
    assert not (train_pool & held_pool), "R1b2a pools must be disjoint"


def test_generator_r1b2a_held_out_unique_ge_4() -> None:
    """Codex implementation guardrail msg 1779472239175: held-out must
    have >= 4 UNIQUE rows so G1 isn't oversampled over too small a pool.

    With A in [1,19] bucket-stratified, 2 unique held-out per bucket
    (one_digit + teen) yields 4 unique held-out rows. This test asserts
    the unique-row guarantee directly so any regression on the bucket
    split would fail the build."""
    held = make_rung_examples("R1b2a", n=200, seed=42, split="held_out")
    unique = {(ex["question"], ex["expected"]) for ex in held}
    assert len(unique) >= 4, (
        f"R1b2a held_out unique count must be >= 4 (codex guardrail "
        f"1779472239175); got {len(unique)}: {sorted(unique)}"
    )


def test_generator_r1b2a_no_collision_with_r1() -> None:
    """R1b2a rows must NEVER appear in R1's train OR held_out.
    R1 emits `A minus 0` (B=0); R1b2a emits `A minus 1` (B=1)."""
    r1_train = make_rung_examples("R1", n=2000, seed=42, split="train")
    r1_held = make_rung_examples("R1", n=2000, seed=42, split="held_out")
    r1b2a_train = make_rung_examples("R1b2a", n=2000, seed=42, split="train")
    r1b2a_held = make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    r1_rows = {(ex["question"], ex["expected"]) for ex in r1_train + r1_held}
    r1b2a_rows = {(ex["question"], ex["expected"]) for ex in r1b2a_train + r1b2a_held}
    overlap = r1_rows & r1b2a_rows
    assert not overlap, f"R1 vs R1b2a row collision: {sorted(overlap)[:5]}"


def test_generator_r1b2a_no_collision_with_r1b1() -> None:
    """R1b2a rows must NEVER appear in R1b1's train OR held_out.
    R1b1 emits `A plus 1`; R1b2a emits `A minus 1` (operator distinct)."""
    r1b1_train = make_rung_examples("R1b1", n=2000, seed=42, split="train")
    r1b1_held = make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    r1b2a_train = make_rung_examples("R1b2a", n=2000, seed=42, split="train")
    r1b2a_held = make_rung_examples("R1b2a", n=2000, seed=42, split="held_out")
    r1b1_rows = {(ex["question"], ex["expected"]) for ex in r1b1_train + r1b1_held}
    r1b2a_rows = {(ex["question"], ex["expected"]) for ex in r1b2a_train + r1b2a_held}
    overlap = r1b1_rows & r1b2a_rows
    assert not overlap, f"R1b1 vs R1b2a row collision: {sorted(overlap)[:5]}"


def test_generator_r1b2a_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r1b2a partition must be IDENTICAL across
    PYTHONHASHSEED restarts."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b2a; "
        "train, held = _enumerate_partition_r1b2a(seed=42); "
        "print(','.join(str(x) for x in sorted(train)) + '|' + ','.join(str(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "0"},
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "random"},
    ).decode().strip()
    assert out1 == out2 == out3, (
        f"R1b2a partition diverges across PYTHONHASHSEED: "
        f"PYTHONHASHSEED=0 -> {out1[:60]}...; "
        f"PYTHONHASHSEED=12345 -> {out2[:60]}...; "
        f"PYTHONHASHSEED=random -> {out3[:60]}..."
    )


def test_cross_rung_r1b2a_and_r1b2_together_collide() -> None:
    """Negative test: opting BOTH R1b2a and R1b2 into build_rung_splits
    is expected to fail assert_no_train_holdout_overlap because R1b2a's
    A_minus_1 over [1,19] is a STRICT SUBSET of R1b2's [1,99]. Same
    policy as R1b1+R1b: R1b2 is now diagnosis-only post-FAIL (6fd2fec)
    and stays excluded from active default. Codex msg 1779472300306."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b1", "R1b2a", "R1b2"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


# ============================================================================ #
# R1b2 single-template -1 stratified partition
# (codex msg 1779471073874 + 1779471212090 after R1b1 PASS at 66b9747;
#  diagnosis-only after FAIL at 6fd2fec)
# ============================================================================ #

def _r1b2_a(ex: dict) -> int:
    """Extract A from an R1b2 example: question is `what is A minus 1?`."""
    q = ex["question"]
    assert q.endswith(" minus 1?"), f"R1b2 question must end ' minus 1?': {q!r}"
    return int(q.split()[2])


def test_generator_r1b2_in_rung_names_after_r1b2a() -> None:
    """R1b2 PASSED at c2686cc, stays canonical. RUNG_NAMES position:
    R1b2a@3, R1b2@4, R1b3@5, R1b4@6, R1b@7. R1b4 inserted between R1b3
    and R1b per codex msg 1779482125661 after R1b3 v2 schedule PASS at
    175d327; R1b4 is constant K=3 successor."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[4] == "R1b2", f"R1b2 must be at index 4 (post-R1b2a); got {RUNG_NAMES}"
    assert RUNG_NAMES[3] == "R1b2a", f"R1b2a must be at index 3; got {RUNG_NAMES}"
    assert RUNG_NAMES[5] == "R1b3", f"R1b3 must be at index 5 (post-R1b2); got {RUNG_NAMES}"
    assert RUNG_NAMES[6] == "R1b4", f"R1b4 must be at index 6 (post-R1b3); got {RUNG_NAMES}"
    assert RUNG_NAMES[7] == "R1b", f"R1b must be at index 7 (post-R1b4); got {RUNG_NAMES}"


def test_generator_r1b2_train_holdout_exact_row_disjoint() -> None:
    """R1b2 train + held_out must be exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b2", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b2", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R1b2 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b2_single_template_only() -> None:
    """R1b2 emits ONLY `what is A minus 1?`; never `A plus 1`, never any
    R1b template (those are diagnosis-only after the falsifier split)."""
    rows = make_rung_examples("R1b2", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b2", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R1b2 prefix violated: {q!r}"
        assert q.endswith(" minus 1?"), f"R1b2 must end ' minus 1?'; got {q!r}"
        # Single-template invariant: exactly one " minus ", zero " plus "
        assert q.count(" minus ") == 1, f"R1b2 must contain ' minus ' exactly once: {q!r}"
        assert " plus " not in q, f"R1b2 must not emit 'plus' template: {q!r}"
        # Token shape: 'what is <A> minus 1?' -> 5 tokens
        toks = q.split()
        assert len(toks) == 5, f"R1b2 question must have 5 tokens; got {len(toks)}: {q!r}"
        assert toks[0] == "what" and toks[1] == "is" and toks[3] == "minus" and toks[4] == "1?", (
            f"R1b2 template shape violated: {q!r}"
        )


def test_generator_r1b2_no_a_zero() -> None:
    """R1b2 must NEVER emit A=0 -- output would be -1 (negative; schema
    mismatches non-negative integer answers)."""
    rows = make_rung_examples("R1b2", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b2", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b2_a(ex)
        assert A != 0, f"R1b2 must not emit A=0; got {ex['question']!r}"
        assert 1 <= A <= 99, f"R1b2 A out of [1,99]; got A={A} q={ex['question']!r}"


def test_generator_r1b2_expected_matches_arithmetic() -> None:
    """For every R1b2 row, expected must equal A-1."""
    rows = make_rung_examples("R1b2", n=500, seed=42, split="train") + \
           make_rung_examples("R1b2", n=500, seed=42, split="held_out")
    for ex in rows:
        A = _r1b2_a(ex)
        assert ex["expected"] == A - 1, (
            f"R1b2 expected mismatch: A={A} expected={ex['expected']}"
        )


def test_generator_r1b2_output_in_0_to_98() -> None:
    """R1b2 output ∈ [0, 98] (A in [1,99] -> A-1 in [0,98])."""
    rows = make_rung_examples("R1b2", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b2", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 0 <= ex["expected"] <= 98, (
            f"R1b2 output out of [0,98]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b2_both_digit_lengths_in_both_splits() -> None:
    """Both splits must contain one-digit AND two-digit A (stratification gate)."""
    train = make_rung_examples("R1b2", n=500, seed=42, split="train")
    held = make_rung_examples("R1b2", n=500, seed=42, split="held_out")
    assert any(_r1b2_a(ex) < 10 for ex in train), "R1b2 train missing 1-digit A"
    assert any(_r1b2_a(ex) >= 10 for ex in train), "R1b2 train missing 2-digit A"
    assert any(_r1b2_a(ex) < 10 for ex in held), "R1b2 held missing 1-digit A"
    assert any(_r1b2_a(ex) >= 10 for ex in held), "R1b2 held missing 2-digit A"


def test_generator_r1b2_pool_sizes() -> None:
    """R1b2 pool sizes per codex msg 1779471212090 correction:
      one_digit [1,9]:   9 vals  -> 7 train + 2 held
      two_digit [10,99]: 90 vals -> 72 train + 18 held
      TOTAL:             99 vals -> 79 train + 20 held
    """
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b2
    train_pool, held_pool = _enumerate_partition_r1b2(seed=42)
    assert len(train_pool) == 79, f"R1b2 train pool must be 79; got {len(train_pool)}"
    assert len(held_pool) == 20, f"R1b2 held_out pool must be 20; got {len(held_pool)}"
    # Pool integers cover [1,99] with NO A=0
    full = train_pool | held_pool
    assert full == set(range(1, 100)), (
        f"R1b2 pool must equal {{1..99}}; got {sorted(full)[:5]}...{sorted(full)[-5:]}"
    )
    assert 0 not in full, "R1b2 pool must not contain A=0"
    # Disjoint
    assert not (train_pool & held_pool), "R1b2 pools must be disjoint"


def test_generator_r1b2_no_collision_with_r1() -> None:
    """R1b2 rows must NEVER appear in R1's train OR held_out (cross-rung
    invariant for the active chain).

    R1 emits `A plus 0` / `0 plus A` / `A minus 0`; R1b2 emits only
    `A minus 1`. R1's minus template has B=0; R1b2's has B=1.
    Disjoint by B-value in question text."""
    r1_train = make_rung_examples("R1", n=2000, seed=42, split="train")
    r1_held = make_rung_examples("R1", n=2000, seed=42, split="held_out")
    r1b2_train = make_rung_examples("R1b2", n=2000, seed=42, split="train")
    r1b2_held = make_rung_examples("R1b2", n=2000, seed=42, split="held_out")
    r1_rows = {(ex["question"], ex["expected"]) for ex in r1_train + r1_held}
    r1b2_rows = {(ex["question"], ex["expected"]) for ex in r1b2_train + r1b2_held}
    overlap = r1_rows & r1b2_rows
    assert not overlap, f"R1 vs R1b2 row collision: {sorted(overlap)[:5]}"


def test_generator_r1b2_no_collision_with_r1b1() -> None:
    """R1b2 rows must NEVER appear in R1b1's train OR held_out.

    R1b1 emits `A plus 1`; R1b2 emits `A minus 1`. Disjoint by
    operator word."""
    r1b1_train = make_rung_examples("R1b1", n=2000, seed=42, split="train")
    r1b1_held = make_rung_examples("R1b1", n=2000, seed=42, split="held_out")
    r1b2_train = make_rung_examples("R1b2", n=2000, seed=42, split="train")
    r1b2_held = make_rung_examples("R1b2", n=2000, seed=42, split="held_out")
    r1b1_rows = {(ex["question"], ex["expected"]) for ex in r1b1_train + r1b1_held}
    r1b2_rows = {(ex["question"], ex["expected"]) for ex in r1b2_train + r1b2_held}
    overlap = r1b1_rows & r1b2_rows
    assert not overlap, f"R1b1 vs R1b2 row collision: {sorted(overlap)[:5]}"


def test_generator_r1b2_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r1b2 partition must be IDENTICAL across
    PYTHONHASHSEED restarts. Uses _stable_seed not builtin hash()."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b2; "
        "train, held = _enumerate_partition_r1b2(seed=42); "
        "print(','.join(str(x) for x in sorted(train)) + '|' + ','.join(str(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "0"},
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "12345"},
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet],
        env={**os.environ, "PYTHONHASHSEED": "random"},
    ).decode().strip()
    assert out1 == out2 == out3, (
        f"R1b2 partition diverges across PYTHONHASHSEED: "
        f"PYTHONHASHSEED=0 -> {out1[:60]}...; "
        f"PYTHONHASHSEED=12345 -> {out2[:60]}...; "
        f"PYTHONHASHSEED=random -> {out3[:60]}..."
    )


# ============================================================================ #
# R1b ±1 stratified partition (codex msg 1779467425298 design;
#  diagnosis-only after R1b v2 failure -- see splits.py default tuple)
# ============================================================================ #

def _r1b_decode(ex: dict) -> tuple[str, int]:
    """Identify (template_key, A) from an R1b ±1 example."""
    q = ex["question"]
    if q.startswith("what is 1 plus "):
        # "what is 1 plus A?"
        A = int(q.split()[4].rstrip("?"))
        return ("1_plus_A", A)
    if " plus 1?" in q:
        # "what is A plus 1?"
        A = int(q.split()[2])
        return ("A_plus_1", A)
    if " minus 1?" in q:
        # "what is A minus 1?"
        A = int(q.split()[2])
        return ("A_minus_1", A)
    raise ValueError(f"unrecognized R1b question shape: {q!r}")


def test_generator_r1b_train_holdout_exact_row_disjoint() -> None:
    """R1b train + held_out must be exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b", n=2000, seed=42, split="held_out")
    train_keys = {(ex["question"], ex["expected"]) for ex in train}
    held_keys = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_keys & held_keys
    assert not overlap, f"R1b train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b_all_templates_in_both_splits() -> None:
    """Stratification by template: both splits contain all 3 templates."""
    train = make_rung_examples("R1b", n=500, seed=42, split="train")
    held = make_rung_examples("R1b", n=500, seed=42, split="held_out")
    train_templates = {_r1b_decode(ex)[0] for ex in train}
    held_templates = {_r1b_decode(ex)[0] for ex in held}
    expected = {"A_plus_1", "1_plus_A", "A_minus_1"}
    assert train_templates == expected
    assert held_templates == expected


def test_generator_r1b_both_digit_lengths_in_both_splits() -> None:
    """Both splits contain 1-digit AND 2-digit A."""
    train = make_rung_examples("R1b", n=500, seed=42, split="train")
    held = make_rung_examples("R1b", n=500, seed=42, split="held_out")
    assert any(_r1b_decode(ex)[1] < 10 for ex in train), "R1b train missing 1-digit A"
    assert any(_r1b_decode(ex)[1] >= 10 for ex in train), "R1b train missing 2-digit A"
    assert any(_r1b_decode(ex)[1] < 10 for ex in held), "R1b held missing 1-digit A"
    assert any(_r1b_decode(ex)[1] >= 10 for ex in held), "R1b held missing 2-digit A"


def test_generator_r1b_output_in_0_to_99() -> None:
    """R1b output must stay in [0,99] -- no new digit-length class (codex
    msg 1779467425298 output-range constraint). 'A_plus_1' caps at A=98
    (output=99); 'A_minus_1' starts at A=1 (output=0)."""
    rows = make_rung_examples("R1b", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 0 <= ex["expected"] <= 99, (
            f"R1b output out of [0,99]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b_expected_matches_arithmetic() -> None:
    """For each template, output schema matches the corresponding ±1 op:
    A_plus_1 -> expected=A+1; 1_plus_A -> expected=A+1; A_minus_1 -> expected=A-1."""
    rows = make_rung_examples("R1b", n=500, seed=42, split="train") + \
           make_rung_examples("R1b", n=500, seed=42, split="held_out")
    for ex in rows:
        template, A = _r1b_decode(ex)
        if template == "A_plus_1":
            assert ex["expected"] == A + 1, f"{ex['question']!r}: expected {A+1}, got {ex['expected']}"
        elif template == "1_plus_A":
            assert ex["expected"] == A + 1
        elif template == "A_minus_1":
            assert ex["expected"] == A - 1


def test_generator_r1b_no_a_zero_in_plus_templates() -> None:
    """Collision-fix regression: A=0 MUST be absent from both 'A_plus_1'
    and '1_plus_A' templates. Otherwise these would emit rows that
    collide with R1 identity rows ('what is 0 plus 1?' or 'what is 1 plus 0?')."""
    rows = make_rung_examples("R1b", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b", n=2000, seed=42, split="held_out")
    for ex in rows:
        template, A = _r1b_decode(ex)
        if template in ("A_plus_1", "1_plus_A"):
            assert A != 0, (
                f"{template} has A=0: {ex['question']!r}; would collide with R1"
            )


def test_generator_r1b_no_a_one_in_1_plus_a() -> None:
    """Intra-R1b collision-fix regression: ('1_plus_A', A=1) MUST be
    absent from the partition. (A_plus_1, A=1) emits 'what is 1 plus 1?' -> 2;
    if (1_plus_A, A=1) were present it'd emit the SAME row.

    Checked at the partition-pool level (not via row-decoder) because
    'what is 1 plus 1?' is row-ambiguous: it matches both templates'
    string patterns, so the row-decoder cannot distinguish which
    template emitted it. The partition is the authoritative source."""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b
    train, held = _enumerate_partition_r1b(42)
    assert ("1_plus_A", 1) not in train, (
        "partition has ('1_plus_A', 1); would duplicate ('A_plus_1', 1)"
    )
    assert ("1_plus_A", 1) not in held, (
        "partition has ('1_plus_A', 1); would duplicate ('A_plus_1', 1)"
    )
    # Sanity: ('A_plus_1', 1) IS present somewhere (canonical owner of "1+1")
    assert ("A_plus_1", 1) in train or ("A_plus_1", 1) in held


def test_generator_r1b_pool_sizes() -> None:
    """Exact pool sizes per codex msg 1779467425298 spec:
       A_plus_1:  [1,9] -> 7+2 / [10,98] -> 71+18 = 78 train + 20 held
       1_plus_A:  [2,9] -> 6+2 / [10,98] -> 71+18 = 77 train + 20 held
       A_minus_1: [1,9] -> 7+2 / [10,99] -> 72+18 = 79 train + 20 held
       TOTAL: 234 train + 60 held"""
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b
    train, held = _enumerate_partition_r1b(42)
    assert len(train) == 234, f"R1b train pool size: {len(train)} expected 234"
    assert len(held) == 60, f"R1b held_out pool size: {len(held)} expected 60"
    train_templates = {t for t, _ in train}
    held_templates = {t for t, _ in held}
    assert train_templates == {"A_plus_1", "1_plus_A", "A_minus_1"}
    assert held_templates == {"A_plus_1", "1_plus_A", "A_minus_1"}
    assert train & held == set()


def test_generator_r1b_cross_rung_collision_rows_excluded() -> None:
    """Explicit regression: the two specific cross-rung collision rows
    that codex msg 1779467425298 flagged MUST NEVER appear in R1b:
      - 'what is 0 plus 1?' -> 1   (would duplicate R1 0_plus_A A=1)
      - 'what is 1 plus 0?' -> 1   (would duplicate R1 A_plus_0 A=1)
    These rows belong to R1 identity. R1b ranges drop A=0 from both
    plus-templates."""
    rows = make_rung_examples("R1b", n=4000, seed=42, split="train") + \
           make_rung_examples("R1b", n=4000, seed=42, split="held_out")
    keys = {(ex["question"], ex["expected"]) for ex in rows}
    assert ("what is 0 plus 1?", 1) not in keys, "R1b leaked R1-collision row"
    assert ("what is 1 plus 0?", 1) not in keys, "R1b leaked R1-collision row"
    assert ("what is 1 plus 1?", 2) in keys, "R1b should still have its A_plus_1 A=1 row"


def test_generator_r1b_partition_stable_across_pythonhashseed() -> None:
    """R1b partition stable across PYTHONHASHSEED (sha256-stable seed)."""
    import os
    import subprocess
    import sys

    code = (
        "import json\n"
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b\n"
        "train, held = _enumerate_partition_r1b(42)\n"
        "print(json.dumps({'train': sorted(train), 'held': sorted(held)}))\n"
    )

    def _run(pyhs: str) -> str:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        env["PYTHONHASHSEED"] = pyhs
        proc = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True,
            env=env, cwd=".", timeout=60,
        )
        assert proc.returncode == 0, proc.stderr
        return proc.stdout.strip()

    a = _run("0"); b = _run("999"); c = _run("random")
    assert a == b == c, "R1b partition diverged across PYTHONHASHSEED"


# ============================================================================ #
# R1b3 constant K=2 addition (codex msg 1779479973262-6d7445d2 after R2a
#  v1 failed 0.045 at 558fcc1; variable-B reframed as the structural blocker)
# ============================================================================ #

def _r1b3_a(ex: dict) -> int:
    """Extract A from an R1b3 example: question is `what is A plus 2?`."""
    q = ex["question"]
    assert q.endswith(" plus 2?"), f"R1b3 question must end ' plus 2?': {q!r}"
    return int(q.split()[2])


def test_generator_r1b3_in_rung_names_index_5() -> None:
    """R1b3 at RUNG_NAMES index 5 (after R1b2 at 4, before R1b4 at 6).
    Trainer's prior_rungs derivation auto-resolves to (R0, R1, R1b1, R1b2).
    Codex msg 1779479973262-6d7445d2; R1b4 inserted at 6 per
    1779482125661-b2c0ca2a."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[5] == "R1b3", f"R1b3 must be at index 5; got {RUNG_NAMES}"
    assert RUNG_NAMES[4] == "R1b2", f"R1b2 must be at index 4 (pre-R1b3); got {RUNG_NAMES}"
    assert RUNG_NAMES[6] == "R1b4", f"R1b4 must be at index 6 (post-R1b3); got {RUNG_NAMES}"


def test_generator_r1b3_train_holdout_exact_row_disjoint() -> None:
    """R1b3 train + held_out exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b3", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b3", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R1b3 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b3_single_template_only() -> None:
    """R1b3 emits ONLY `what is A plus 2?`."""
    rows = make_rung_examples("R1b3", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b3", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R1b3 prefix violated: {q!r}"
        assert q.endswith(" plus 2?"), f"R1b3 must end ' plus 2?': {q!r}"
        assert q.count(" plus ") == 1, f"R1b3 must contain ' plus ' exactly once: {q!r}"
        assert " minus " not in q, f"R1b3 must not emit minus: {q!r}"
        toks = q.split()
        assert len(toks) == 5 and toks[3] == "plus" and toks[4] == "2?", (
            f"R1b3 template shape violated: {q!r}"
        )


def test_generator_r1b3_no_a_zero() -> None:
    """R1b3 never emits A=0 (symmetric with R1b1's drop)."""
    rows = make_rung_examples("R1b3", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b3", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b3_a(ex)
        assert A != 0, f"R1b3 must not emit A=0: {ex['question']!r}"
        assert 1 <= A <= 97, f"R1b3 A out of [1,97]: A={A}"


def test_generator_r1b3_no_a_ge_98() -> None:
    """R1b3 never emits A>=98 (output A+2 would be 100+, 3-digit class)."""
    rows = make_rung_examples("R1b3", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b3", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b3_a(ex)
        assert A < 98, f"R1b3 must not emit A>=98: A={A} q={ex['question']!r}"


def test_generator_r1b3_expected_matches_arithmetic() -> None:
    """For every R1b3 row, expected == A + 2."""
    rows = make_rung_examples("R1b3", n=500, seed=42, split="train") + \
           make_rung_examples("R1b3", n=500, seed=42, split="held_out")
    for ex in rows:
        A = _r1b3_a(ex)
        assert ex["expected"] == A + 2, f"R1b3 expected mismatch: A={A} expected={ex['expected']}"


def test_generator_r1b3_output_in_3_to_99() -> None:
    """R1b3 output in [3, 99] (A in [1,97] -> A+2 in [3,99])."""
    rows = make_rung_examples("R1b3", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b3", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 3 <= ex["expected"] <= 99, (
            f"R1b3 output out of [3,99]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b3_both_buckets_in_both_splits() -> None:
    """Both one_digit + two_digit buckets in both splits."""
    train = make_rung_examples("R1b3", n=500, seed=42, split="train")
    held = make_rung_examples("R1b3", n=500, seed=42, split="held_out")
    assert any(_r1b3_a(ex) < 10 for ex in train), "R1b3 train missing one_digit"
    assert any(_r1b3_a(ex) >= 10 for ex in train), "R1b3 train missing two_digit"
    assert any(_r1b3_a(ex) < 10 for ex in held), "R1b3 held missing one_digit"
    assert any(_r1b3_a(ex) >= 10 for ex in held), "R1b3 held missing two_digit"


def test_generator_r1b3_pool_sizes() -> None:
    """R1b3 pool sizes per codex msg 1779479973262:
      one_digit [1,9]:    9 vals -> 7 train + 2 held
      two_digit [10,97]: 88 vals -> 70 train + 18 held
      TOTAL:             97 vals -> 77 train + 20 held_out
    """
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b3
    train_pool, held_pool = _enumerate_partition_r1b3(seed=42)
    assert len(train_pool) == 77, f"R1b3 train pool must be 77; got {len(train_pool)}"
    assert len(held_pool) == 20, f"R1b3 held_out pool must be 20; got {len(held_pool)}"
    full = train_pool | held_pool
    assert full == set(range(1, 98)), f"R1b3 pool must equal {{1..97}}"
    assert 0 not in full and 98 not in full
    assert not (train_pool & held_pool), "R1b3 pools must be disjoint"


def test_generator_r1b3_held_out_unique_count() -> None:
    """R1b3 held_out unique = 20 (codex direction: unique audit support)."""
    held = make_rung_examples("R1b3", n=200, seed=42, split="held_out")
    unique = {(ex["question"], ex["expected"]) for ex in held}
    assert len(unique) == 20, f"R1b3 held_out unique must be 20; got {len(unique)}"


def test_generator_r1b3_no_collision_with_priors() -> None:
    """R1b3 (B=2) disjoint from R0/R1 (no/B=0), R1b1/R1b2 (B=1)."""
    r1b3_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R1b3", n=2000, seed=42, split=split):
            r1b3_rows.add((ex["question"], ex["expected"]))
    for prior in ("R0", "R1", "R1b1", "R1b2"):
        prior_rows = set()
        for split in ("train", "held_out"):
            for ex in make_rung_examples(prior, n=2000, seed=42, split=split):
                prior_rows.add((ex["question"], ex["expected"]))
        overlap = prior_rows & r1b3_rows
        assert not overlap, f"{prior} vs R1b3 collision: {sorted(overlap)[:5]}"


def test_generator_r1b3_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r1b3 stable across PYTHONHASHSEED restarts."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b3; "
        "train, held = _enumerate_partition_r1b3(seed=42); "
        "print(','.join(str(x) for x in sorted(train)) + '|' + ','.join(str(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "0"}).decode().strip()
    out2 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "333"}).decode().strip()
    out3 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "random"}).decode().strip()
    assert out1 == out2 == out3, "R1b3 partition diverged across PYTHONHASHSEED"


def test_cross_rung_r1b3_and_r2a_together_collide() -> None:
    """Negative test: R1b3 A in [10,19] B=2 overlaps R2a A in [10,19]
    B in [2,9] when B=2 selected. R2a stays diagnosis-only post-558fcc1."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b1", "R1b2", "R1b3", "R2a"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


# ============================================================================ #
# R1b4 constant K=3 addition (codex msg 1779482125661-b2c0ca2a after R1b3
#  v2 schedule PASS at 175d327; continues locked constant-K jigsaw pattern)
# ============================================================================ #

def _r1b4_a(ex: dict) -> int:
    """Extract A from an R1b4 example: question is `what is A plus 3?`."""
    q = ex["question"]
    assert q.endswith(" plus 3?"), f"R1b4 question must end ' plus 3?': {q!r}"
    return int(q.split()[2])


def test_generator_r1b4_in_rung_names_index_6() -> None:
    """R1b4 at RUNG_NAMES index 6 (after R1b3 at 5). Trainer's
    prior_rungs derivation auto-resolves to (R0, R1, R1b1, R1b2, R1b3).
    Codex msg 1779482125661-b2c0ca2a."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[6] == "R1b4", f"R1b4 must be at index 6; got {RUNG_NAMES}"
    assert RUNG_NAMES[5] == "R1b3", f"R1b3 must be at index 5; got {RUNG_NAMES}"


def test_generator_r1b4_train_holdout_exact_row_disjoint() -> None:
    """R1b4 train + held_out exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R1b4", n=2000, seed=42, split="train")
    held = make_rung_examples("R1b4", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R1b4 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r1b4_single_template_only() -> None:
    """R1b4 emits ONLY `what is A plus 3?`."""
    rows = make_rung_examples("R1b4", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b4", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R1b4 prefix violated: {q!r}"
        assert q.endswith(" plus 3?"), f"R1b4 must end ' plus 3?': {q!r}"
        assert q.count(" plus ") == 1, f"R1b4 must contain ' plus ' exactly once: {q!r}"
        assert " minus " not in q, f"R1b4 must not emit minus: {q!r}"
        toks = q.split()
        assert len(toks) == 5 and toks[3] == "plus" and toks[4] == "3?", (
            f"R1b4 template shape violated: {q!r}"
        )


def test_generator_r1b4_no_a_zero() -> None:
    """R1b4 never emits A=0 (symmetric with R1b1/R1b3 drops)."""
    rows = make_rung_examples("R1b4", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b4", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b4_a(ex)
        assert A != 0, f"R1b4 must not emit A=0: {ex['question']!r}"
        assert 1 <= A <= 96, f"R1b4 A out of [1,96]: A={A}"


def test_generator_r1b4_no_a_ge_97() -> None:
    """R1b4 never emits A>=97 (output A+3 would be 100+, 3-digit class)."""
    rows = make_rung_examples("R1b4", n=2000, seed=42, split="train") + \
           make_rung_examples("R1b4", n=2000, seed=42, split="held_out")
    for ex in rows:
        A = _r1b4_a(ex)
        assert A < 97, f"R1b4 must not emit A>=97: A={A} q={ex['question']!r}"


def test_generator_r1b4_expected_matches_arithmetic() -> None:
    """For every R1b4 row, expected == A + 3."""
    rows = make_rung_examples("R1b4", n=500, seed=42, split="train") + \
           make_rung_examples("R1b4", n=500, seed=42, split="held_out")
    for ex in rows:
        A = _r1b4_a(ex)
        assert ex["expected"] == A + 3, f"R1b4 expected mismatch: A={A} expected={ex['expected']}"


def test_generator_r1b4_output_in_4_to_99() -> None:
    """R1b4 output in [4, 99] (A in [1,96] -> A+3 in [4,99])."""
    rows = make_rung_examples("R1b4", n=1000, seed=42, split="train") + \
           make_rung_examples("R1b4", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 4 <= ex["expected"] <= 99, (
            f"R1b4 output out of [4,99]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r1b4_both_buckets_in_both_splits() -> None:
    """Both one_digit + two_digit buckets in both splits."""
    train = make_rung_examples("R1b4", n=500, seed=42, split="train")
    held = make_rung_examples("R1b4", n=500, seed=42, split="held_out")
    assert any(_r1b4_a(ex) < 10 for ex in train), "R1b4 train missing one_digit"
    assert any(_r1b4_a(ex) >= 10 for ex in train), "R1b4 train missing two_digit"
    assert any(_r1b4_a(ex) < 10 for ex in held), "R1b4 held missing one_digit"
    assert any(_r1b4_a(ex) >= 10 for ex in held), "R1b4 held missing two_digit"


def test_generator_r1b4_pool_sizes() -> None:
    """R1b4 pool sizes per codex msg 1779482125661:
      one_digit [1,9]:    9 vals -> 7 train + 2 held
      two_digit [10,96]: 87 vals -> 69 train + 18 held
      TOTAL:             96 vals -> 76 train + 20 held_out
    """
    from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b4
    train_pool, held_pool = _enumerate_partition_r1b4(seed=42)
    assert len(train_pool) == 76, f"R1b4 train pool must be 76; got {len(train_pool)}"
    assert len(held_pool) == 20, f"R1b4 held_out pool must be 20; got {len(held_pool)}"
    full = train_pool | held_pool
    assert full == set(range(1, 97)), f"R1b4 pool must equal {{1..96}}"
    assert 0 not in full and 97 not in full
    assert not (train_pool & held_pool), "R1b4 pools must be disjoint"


def test_generator_r1b4_held_out_unique_count() -> None:
    """R1b4 held_out unique = 20."""
    held = make_rung_examples("R1b4", n=200, seed=42, split="held_out")
    unique = {(ex["question"], ex["expected"]) for ex in held}
    assert len(unique) == 20, f"R1b4 held_out unique must be 20; got {len(unique)}"


def test_generator_r1b4_no_collision_with_active_priors() -> None:
    """R1b4 (B=3) disjoint from R0/R1 (no/B=0), R1b1 (B=1), R1b2 (B=1 minus),
    R1b3 (B=2)."""
    r1b4_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R1b4", n=2000, seed=42, split=split):
            r1b4_rows.add((ex["question"], ex["expected"]))
    for prior in ("R0", "R1", "R1b1", "R1b2", "R1b3"):
        prior_rows = set()
        for split in ("train", "held_out"):
            for ex in make_rung_examples(prior, n=2000, seed=42, split=split):
                prior_rows.add((ex["question"], ex["expected"]))
        overlap = prior_rows & r1b4_rows
        assert not overlap, f"{prior} vs R1b4 collision: {sorted(overlap)[:5]}"


def test_generator_r1b4_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r1b4 stable across PYTHONHASHSEED restarts."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1b4; "
        "train, held = _enumerate_partition_r1b4(seed=42); "
        "print(','.join(str(x) for x in sorted(train)) + '|' + ','.join(str(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "0"}).decode().strip()
    out2 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "777"}).decode().strip()
    out3 = subprocess.check_output([sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "random"}).decode().strip()
    assert out1 == out2 == out3, "R1b4 partition diverged across PYTHONHASHSEED"


def test_cross_rung_r1b4_and_r2a_together_collide() -> None:
    """Negative test: R1b4 A in [10,19] B=3 overlaps R2a A in [10,19]
    B in [2,9] for B=3 subset. R2a stays diagnosis-only."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4", "R2a"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


# ============================================================================ #
# R2a teens addition-only (codex msg 1779478819906-0e30503e after full R2
#  failed v1+v2 n_train=8000 at c2f4f8d; jigsaw-curriculum operator split;
#  DIAGNOSIS-ONLY after v1 failed 0.045 at 558fcc1)
# ============================================================================ #

def _r2a_decode(ex: dict) -> tuple[int, int]:
    """Identify (A, B) from an R2a example: question is `what is A plus B?`."""
    q = ex["question"]
    toks = q.split()
    assert len(toks) == 5 and toks[0] == "what" and toks[1] == "is" and toks[3] == "plus", q
    A = int(toks[2])
    B = int(toks[4].rstrip("?"))
    return (A, B)


def test_generator_r2a_in_rung_names_after_r1b() -> None:
    """R2a was demoted to diagnosis-only after v1 failed 0.045 at 558fcc1.
    With R1b4 inserted at index 6 post-R1b3 PASS at 175d327, positions:
    R1b3@5, R1b4@6, R1b@7, R2a@8, R2@9."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    assert RUNG_NAMES[8] == "R2a", f"R2a must be at index 8; got {RUNG_NAMES}"
    assert RUNG_NAMES[9] == "R2", f"R2 must be at index 9 (diagnosis-only); got {RUNG_NAMES}"


def test_generator_r2a_train_holdout_exact_row_disjoint() -> None:
    """R2a train + held_out exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R2a", n=2000, seed=42, split="train")
    held = make_rung_examples("R2a", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R2a train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r2a_addition_only() -> None:
    """R2a emits ONLY `what is A plus B?`; never minus."""
    rows = make_rung_examples("R2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        q = ex["question"]
        assert q.startswith("what is "), f"R2a prefix violated: {q!r}"
        assert q.endswith("?"), f"R2a suffix violated: {q!r}"
        assert " plus " in q, f"R2a must contain ' plus ': {q!r}"
        assert " minus " not in q, f"R2a must not emit 'minus': {q!r}"
        toks = q.split()
        assert len(toks) == 5 and toks[3] == "plus", f"R2a shape violated: {q!r}"


def test_generator_r2a_a_in_teens() -> None:
    """R2a A in [10, 19]."""
    rows = make_rung_examples("R2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        A, _ = _r2a_decode(ex)
        assert 10 <= A <= 19, f"R2a A out of [10,19]: A={A} q={ex['question']!r}"


def test_generator_r2a_b_in_2_to_9() -> None:
    """R2a B in [2, 9] (never 0/1; cross-rung B-disjoint with R1/R1b1/R1b2)."""
    rows = make_rung_examples("R2a", n=2000, seed=42, split="train") + \
           make_rung_examples("R2a", n=2000, seed=42, split="held_out")
    for ex in rows:
        _, B = _r2a_decode(ex)
        assert 2 <= B <= 9, f"R2a B out of [2,9]: B={B} q={ex['question']!r}"


def test_generator_r2a_both_phenomena_in_both_splits() -> None:
    """Both plus_no_carry and plus_carry phenomena in both splits
    (codex direction msg 1779478819906)."""
    from calm.hrm_text_158.curriculum.generators import _r2a_phenomenon
    train = make_rung_examples("R2a", n=500, seed=42, split="train")
    held = make_rung_examples("R2a", n=500, seed=42, split="held_out")
    train_phenoms = {_r2a_phenomenon(*_r2a_decode(ex)) for ex in train}
    held_phenoms = {_r2a_phenomenon(*_r2a_decode(ex)) for ex in held}
    expected = {"plus_no_carry", "plus_carry"}
    assert train_phenoms == expected, f"R2a train missing phenoms: {expected - train_phenoms}"
    assert held_phenoms == expected, f"R2a held missing phenoms: {expected - held_phenoms}"


def test_generator_r2a_expected_matches_arithmetic() -> None:
    """For every R2a row, expected == A + B."""
    rows = make_rung_examples("R2a", n=500, seed=42, split="train") + \
           make_rung_examples("R2a", n=500, seed=42, split="held_out")
    for ex in rows:
        A, B = _r2a_decode(ex)
        assert ex["expected"] == A + B, f"R2a expected mismatch: {ex}"


def test_generator_r2a_output_in_12_to_28() -> None:
    """R2a output in [12, 28] (min 10+2=12; max 19+9=28). No 3-digit."""
    rows = make_rung_examples("R2a", n=1000, seed=42, split="train") + \
           make_rung_examples("R2a", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 12 <= ex["expected"] <= 28, (
            f"R2a output out of [12,28]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r2a_pool_sizes() -> None:
    """R2a pool sizes per codex msg 1779478819906 with 75/25 split:
      plus_no_carry: 36 -> 27 train + 9 held  (36 * 0.75 = 27.0)
      plus_carry:    44 -> 33 train + 11 held (44 * 0.75 = 33.0)
      TOTAL:         80 -> 60 train + 20 held
    Restores R1b1-style ~50x multiplicity at 6000 rows."""
    from calm.hrm_text_158.curriculum.generators import (
        _enumerate_partition_r2a,
        _r2a_phenomenon,
    )
    train_pool, held_pool = _enumerate_partition_r2a(seed=42)
    assert len(train_pool) == 60, f"R2a train pool must be 60; got {len(train_pool)}"
    assert len(held_pool) == 20, f"R2a held_out pool must be 20; got {len(held_pool)}"
    assert not (train_pool & held_pool), "R2a pools must be disjoint"
    full = train_pool | held_pool
    assert len(full) == 80, f"R2a union pool must be 80; got {len(full)}"
    # Per-phenomenon
    from collections import Counter
    train_phenom = Counter(_r2a_phenomenon(A, B) for (A, B) in train_pool)
    held_phenom = Counter(_r2a_phenomenon(A, B) for (A, B) in held_pool)
    assert train_phenom == {"plus_no_carry": 27, "plus_carry": 33}, train_phenom
    assert held_phenom == {"plus_no_carry": 9, "plus_carry": 11}, held_phenom


def test_generator_r2a_held_out_unique_count() -> None:
    """R2a held_out unique = 20. Codex direction: 75/25 split chosen to
    preserve 20-row unique audit support."""
    held = make_rung_examples("R2a", n=200, seed=42, split="held_out")
    unique = {(ex["question"], ex["expected"]) for ex in held}
    assert len(unique) == 20, f"R2a held_out unique must be 20; got {len(unique)}"


def test_generator_r2a_no_collision_with_priors() -> None:
    """R2a rows must NEVER appear in R0/R1/R1b1/R1b2 train OR held_out.
    All priors have B in {0, 1}; R2a has B in [2,9]."""
    r2a_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R2a", n=2000, seed=42, split=split):
            r2a_rows.add((ex["question"], ex["expected"]))
    for prior in ("R0", "R1", "R1b1", "R1b2"):
        prior_rows = set()
        for split in ("train", "held_out"):
            for ex in make_rung_examples(prior, n=2000, seed=42, split=split):
                prior_rows.add((ex["question"], ex["expected"]))
        overlap = prior_rows & r2a_rows
        assert not overlap, f"{prior} vs R2a collision: {sorted(overlap)[:5]}"


def test_generator_r2a_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r2a stable across PYTHONHASHSEED restarts."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r2a; "
        "train, held = _enumerate_partition_r2a(seed=42); "
        "print(','.join(repr(x) for x in sorted(train)) + '|' + ','.join(repr(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "0"}
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "555"}
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "random"}
    ).decode().strip()
    assert out1 == out2 == out3, "R2a partition diverged across PYTHONHASHSEED"


def test_cross_rung_r2a_and_r2_together_collide() -> None:
    """Negative test: opting BOTH R2a and R2 into build_rung_splits
    expected to fail assert_no_train_holdout_overlap because R2a's
    A_plus_B rows are a STRICT SUBSET of R2's A_plus_B rows (R2 also
    has A_minus_B). Codex msg 1779478819906: R2 stays diagnosis-only."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b1", "R1b2", "R2a", "R2"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


# ============================================================================ #
# R2 teens variable-B ± stratified partition
# (codex msg 1779476750248-2dca0aa7 after R1b2 v2 replay50 PASS at c2686cc;
#  DIAGNOSIS-ONLY after v1+v2 fail at c2f4f8d; R2a is the operator-split successor)
# ============================================================================ #

def _r2_decode(ex: dict) -> tuple[str, int, int]:
    """Identify (template, A, B) from an R2 example."""
    q = ex["question"]
    toks = q.split()  # ['what', 'is', 'A', 'op', 'B?']
    assert len(toks) == 5 and toks[0] == "what" and toks[1] == "is", q
    A = int(toks[2])
    op = toks[3]
    B = int(toks[4].rstrip("?"))
    if op == "plus":
        return ("A_plus_B", A, B)
    if op == "minus":
        return ("A_minus_B", A, B)
    raise ValueError(f"unrecognized R2 question shape: {q!r}")


def test_generator_r2_train_holdout_exact_row_disjoint() -> None:
    """R2 train + held_out exact-row disjoint at n=2000 sampling."""
    train = make_rung_examples("R2", n=2000, seed=42, split="train")
    held = make_rung_examples("R2", n=2000, seed=42, split="held_out")
    train_rows = {(ex["question"], ex["expected"]) for ex in train}
    held_rows = {(ex["question"], ex["expected"]) for ex in held}
    overlap = train_rows & held_rows
    assert not overlap, f"R2 train/held_out share rows: {sorted(overlap)[:5]}"


def test_generator_r2_both_templates_in_both_splits() -> None:
    """Both A_plus_B and A_minus_B appear in both splits."""
    train = make_rung_examples("R2", n=500, seed=42, split="train")
    held = make_rung_examples("R2", n=500, seed=42, split="held_out")
    train_templates = {_r2_decode(ex)[0] for ex in train}
    held_templates = {_r2_decode(ex)[0] for ex in held}
    assert train_templates == {"A_plus_B", "A_minus_B"}, train_templates
    assert held_templates == {"A_plus_B", "A_minus_B"}, held_templates


def test_generator_r2_all_phenomena_in_both_splits() -> None:
    """All 4 phenomena (plus_no_carry, plus_carry, minus_no_borrow,
    minus_borrow) appear in BOTH train and held_out splits (codex
    direction msg 1779476750248: stratify so carry+borrow guaranteed)."""
    from calm.hrm_text_158.curriculum.generators import _r2_phenomenon
    train = make_rung_examples("R2", n=1000, seed=42, split="train")
    held = make_rung_examples("R2", n=1000, seed=42, split="held_out")
    train_phenoms = {_r2_phenomenon(*_r2_decode(ex)) for ex in train}
    held_phenoms = {_r2_phenomenon(*_r2_decode(ex)) for ex in held}
    expected = {"plus_no_carry", "plus_carry", "minus_no_borrow", "minus_borrow"}
    assert train_phenoms == expected, f"train missing phenomena: {expected - train_phenoms}"
    assert held_phenoms == expected, f"held missing phenomena: {expected - held_phenoms}"


def test_generator_r2_b_never_0_or_1() -> None:
    """R2 B in [2,9]. NEVER 0 (collides R1 templates) NEVER 1 (collides
    R1b1/R1b2)."""
    rows = make_rung_examples("R2", n=2000, seed=42, split="train") + \
           make_rung_examples("R2", n=2000, seed=42, split="held_out")
    for ex in rows:
        _, _, B = _r2_decode(ex)
        assert 2 <= B <= 9, f"R2 B out of [2,9]: B={B} q={ex['question']!r}"


def test_generator_r2_a_in_teens() -> None:
    """R2 A in [10,19] (teens only — smallest multi-digit bridge)."""
    rows = make_rung_examples("R2", n=2000, seed=42, split="train") + \
           make_rung_examples("R2", n=2000, seed=42, split="held_out")
    for ex in rows:
        _, A, _ = _r2_decode(ex)
        assert 10 <= A <= 19, f"R2 A out of [10,19]: A={A} q={ex['question']!r}"


def test_generator_r2_expected_matches_arithmetic() -> None:
    """For every R2 row, expected == A op B per template."""
    rows = make_rung_examples("R2", n=500, seed=42, split="train") + \
           make_rung_examples("R2", n=500, seed=42, split="held_out")
    for ex in rows:
        template, A, B = _r2_decode(ex)
        if template == "A_plus_B":
            assert ex["expected"] == A + B, f"R2 plus mismatch: {ex}"
        else:
            assert ex["expected"] == A - B, f"R2 minus mismatch: {ex}"


def test_generator_r2_output_in_1_to_28() -> None:
    """R2 output in [1, 28]. No 3-digit class.

    plus max: 19+9=28; plus min: 10+2=12.
    minus max: 19-2=17; minus min: 10-9=1.
    Combined output range = [1, 28]."""
    rows = make_rung_examples("R2", n=1000, seed=42, split="train") + \
           make_rung_examples("R2", n=1000, seed=42, split="held_out")
    for ex in rows:
        assert 1 <= ex["expected"] <= 28, (
            f"R2 output out of [1,28]: {ex['question']!r} -> {ex['expected']}"
        )


def test_generator_r2_pool_sizes() -> None:
    """R2 pool sizes per codex msg 1779476750248:
      plus_no_carry: 36 -> 28 train + 8 held
      plus_carry:    44 -> 35 train + 9 held
      minus_no_borrow: 36 -> 28 train + 8 held
      minus_borrow:  44 -> 35 train + 9 held
      TOTAL:         160 -> 126 train + 34 held
    """
    from calm.hrm_text_158.curriculum.generators import (
        _enumerate_partition_r2,
        _r2_phenomenon,
    )
    train_pool, held_pool = _enumerate_partition_r2(seed=42)
    assert len(train_pool) == 126, f"R2 train pool must be 126; got {len(train_pool)}"
    assert len(held_pool) == 34, f"R2 held_out pool must be 34; got {len(held_pool)}"
    # Disjoint
    assert not (train_pool & held_pool), "R2 pools must be disjoint"
    # Total covers all 160 (template, A, B) tuples
    full = train_pool | held_pool
    assert len(full) == 160, f"R2 union pool must be 160; got {len(full)}"
    # Per-phenomenon counts
    from collections import Counter
    train_phenom_counts = Counter(
        _r2_phenomenon(template, A, B) for (template, A, B) in train_pool
    )
    held_phenom_counts = Counter(
        _r2_phenomenon(template, A, B) for (template, A, B) in held_pool
    )
    # 80/20 floor split: 36 -> 28+8; 44 -> 35+9
    assert train_phenom_counts == {
        "plus_no_carry": 28, "plus_carry": 35,
        "minus_no_borrow": 28, "minus_borrow": 35,
    }, train_phenom_counts
    assert held_phenom_counts == {
        "plus_no_carry": 8, "plus_carry": 9,
        "minus_no_borrow": 8, "minus_borrow": 9,
    }, held_phenom_counts


def test_generator_r2_held_out_unique_count() -> None:
    """R2 held_out unique = 34. Per codex held-out support rule for
    sampled-with-replacement probes."""
    held = make_rung_examples("R2", n=200, seed=42, split="held_out")
    unique = {(ex["question"], ex["expected"]) for ex in held}
    assert len(unique) == 34, (
        f"R2 held_out unique count must be 34 (codex msg 1779476750248); "
        f"got {len(unique)}"
    )


def test_generator_r2_no_collision_with_r1() -> None:
    """R2 rows must NEVER appear in R1's train OR held_out (R1 B=0, R2 B>=2)."""
    r1_train = make_rung_examples("R1", n=2000, seed=42, split="train")
    r1_held = make_rung_examples("R1", n=2000, seed=42, split="held_out")
    r2_train = make_rung_examples("R2", n=2000, seed=42, split="train")
    r2_held = make_rung_examples("R2", n=2000, seed=42, split="held_out")
    r1_rows = {(ex["question"], ex["expected"]) for ex in r1_train + r1_held}
    r2_rows = {(ex["question"], ex["expected"]) for ex in r2_train + r2_held}
    overlap = r1_rows & r2_rows
    assert not overlap, f"R1 vs R2 collision: {sorted(overlap)[:5]}"


def test_generator_r2_no_collision_with_r1b1_or_r1b2() -> None:
    """R2 rows must NEVER appear in R1b1 (B=1, plus) or R1b2 (B=1, minus).
    R2 has B in [2,9] -> disjoint by B-value."""
    r1b1_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R1b1", n=2000, seed=42, split=split):
            r1b1_rows.add((ex["question"], ex["expected"]))
    r1b2_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R1b2", n=2000, seed=42, split=split):
            r1b2_rows.add((ex["question"], ex["expected"]))
    r2_rows = set()
    for split in ("train", "held_out"):
        for ex in make_rung_examples("R2", n=2000, seed=42, split=split):
            r2_rows.add((ex["question"], ex["expected"]))
    assert not (r1b1_rows & r2_rows), f"R1b1 vs R2 collision: {sorted(r1b1_rows & r2_rows)[:5]}"
    assert not (r1b2_rows & r2_rows), f"R1b2 vs R2 collision: {sorted(r1b2_rows & r2_rows)[:5]}"


def test_generator_r2_partition_stable_across_pythonhashseed() -> None:
    """_enumerate_partition_r2 stable across PYTHONHASHSEED restarts."""
    import os
    import subprocess
    import sys

    snippet = (
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r2; "
        "train, held = _enumerate_partition_r2(seed=42); "
        "print(','.join(repr(x) for x in sorted(train)) + '|' + ','.join(repr(x) for x in sorted(held)))"
    )
    out1 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "0"}
    ).decode().strip()
    out2 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "999"}
    ).decode().strip()
    out3 = subprocess.check_output(
        [sys.executable, "-c", snippet], env={**os.environ, "PYTHONHASHSEED": "random"}
    ).decode().strip()
    assert out1 == out2 == out3, "R2 partition diverged across PYTHONHASHSEED"


def test_generator_r3_held_out_includes_canonical_17x23() -> None:
    """R3 held-out MUST contain the exact canonical probe row
    `("what is 17 times 23?", 391)` — _enumerate_partition_r3 force-injects
    (17, 23) into the held_out pool, and n=500 sampling with rng.choice
    over the ~21-pair held_out pool guarantees it surfaces."""
    held_out = make_rung_examples("R3", n=500, seed=42, split="held_out")
    canonical = {"question": "what is 17 times 23?", "expected": 391, "rung": "R3"}
    assert canonical in held_out, (
        f"R3 held_out missing canonical 17×23=391 probe. "
        f"First 3 held_out rows: {held_out[:3]}"
    )


def test_generator_r3_canonical_never_in_train() -> None:
    """The canonical 17×23 row must NEVER appear in R3 train (force-injected
    into held_out only; not in the [0,9]² train pool)."""
    train = make_rung_examples("R3", n=2000, seed=42, split="train")
    canonical_q = "what is 17 times 23?"
    assert all(ex["question"] != canonical_q for ex in train), (
        "Canonical 17×23 leaked into R3 train"
    )


def test_generator_r7_raises() -> None:
    """R7 is GSM8k; not handled by synthetic generator."""
    with pytest.raises(ValueError, match="R7 is GSM8k"):
        make_rung_examples("R7", n=10, seed=42, split="train")


def test_generator_unknown_rung_raises() -> None:
    with pytest.raises(ValueError, match="unknown rung"):
        make_rung_examples("R99", n=10, seed=42, split="train")


def test_generator_invalid_split_raises() -> None:
    with pytest.raises(ValueError, match="split must be"):
        make_rung_examples("R0", n=10, seed=42, split="validation")


def test_generator_arithmetic_correctness() -> None:
    """Verify R3 (multiplication) actually computes A×B correctly."""
    examples = make_rung_examples("R3", n=20, seed=42, split="train")
    for ex in examples:
        # "what is A times B?" → expected = A*B
        parts = ex["question"].split()
        A = int(parts[2])
        B = int(parts[4].rstrip("?"))
        assert ex["expected"] == A * B, (
            f"R3 arithmetic wrong: {ex['question']} -> expected={ex['expected']} but A*B={A*B}"
        )


# ============================================================================ #
# Cross-rung invariant: held_out ∩ all_train = ∅
# ============================================================================ #

def test_cross_rung_no_train_holdout_overlap() -> None:
    """Active-chain cross-rung invariant per codex msg 1779479973262-6d7445d2
    structural fix after R2a v1 failed 0.045 at 558fcc1 (variable-B reframed
    as the blocker).

    `build_rung_splits` default tuple is now `("R0", "R1", "R1b1",
    "R1b2", "R1b3", "R3", "R4", "R5", "R6")` -- R1b3 is the constant
    K=2 addition successor extending the locked constant-B pattern.
    Diagnosis-only and OUT of default: R1b2a, R1b, R2, R2a. The
    invariant asserts no row in any rung's held_out appears in any
    rung's train set across the active chain only."""
    splits = build_rung_splits(n_train=200, n_held_out=50, seed=42)
    assert_no_train_holdout_overlap(splits)
    # Active chain: R0, R1, R1b1, R1b2, R1b3, R1b4, R3-R6 (R1b2a/R1b/R2/R2a diagnosis-only; R7 = GSM8k)
    assert set(splits.keys()) == {"R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4", "R3", "R4", "R5", "R6"}


def test_cross_rung_r1b1_and_r1b_together_collide() -> None:
    """Negative test: opting BOTH R1b1 and R1b into build_rung_splits is
    expected to fail assert_no_train_holdout_overlap because R1b1's
    A_plus_1 rows over A in [1,98] overlap R1b's A_plus_1 rows over
    A in [1,98] (same template, same range, different shuffle seeds).
    Codex msg 1779469638068 explicitly notes this is expected and is
    why R1b is diagnosis-only post-R1b1."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b1", "R1b"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


def test_cross_rung_r1b2_and_r1b_together_collide() -> None:
    """Negative test: opting BOTH R1b2 and R1b into build_rung_splits is
    expected to fail assert_no_train_holdout_overlap because R1b2's
    A_minus_1 rows over A in [1,99] overlap R1b's A_minus_1 rows over
    A in [1,99] (same template, same range, different shuffle seeds).
    Codex msg 1779471212090: R1b stays excluded from active chain;
    diagnosis-only. Same policy as R1b1+R1b."""
    splits = build_rung_splits(
        n_train=2000,
        n_held_out=400,
        seed=42,
        rungs=("R0", "R1", "R1b2", "R1b"),
    )
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


def test_cross_rung_invariant_detects_violation() -> None:
    """Manually construct a violation: copy a held_out row into another rung's train.
    assert_no_train_holdout_overlap must raise."""
    splits = build_rung_splits(n_train=20, n_held_out=10, seed=42)
    # Inject violation: take an R1 held_out row and add it to R3 train
    # (R2 dropped from default active chain at c2f4f8d -> R2a; use R3 here)
    violation_row = splits["R1"]["held_out"][0].copy()
    splits["R3"]["train"].append(violation_row)
    with pytest.raises(AssertionError, match="overlap detected"):
        assert_no_train_holdout_overlap(splits)


# ============================================================================ #
# Retention probe schema + delta computation
# ============================================================================ #

def test_retention_delta_basic() -> None:
    """Retention delta = current accuracy - prior accuracy per shared rung."""
    prior = RungProbeResult(
        rung="R1", ckpt_path="r1.pt", step=100, n_params=29_000_000,
        rung_accuracy={"R0": 0.95, "R1": 0.80},
    )
    current = RungProbeResult(
        rung="R2", ckpt_path="r2.pt", step=200, n_params=29_000_000,
        rung_accuracy={"R0": 0.90, "R1": 0.78, "R2": 0.60},  # R2 newly added
    )
    deltas = compute_retention_deltas(current, prior)
    assert deltas["R0"] == pytest.approx(-0.05)  # 0.90 - 0.95
    assert deltas["R1"] == pytest.approx(-0.02)  # 0.78 - 0.80
    assert "R2" not in deltas  # R2 didn't exist in prior, not a retention check


def test_retention_delta_none_prior() -> None:
    """R0 has no prior; deltas = {}."""
    current = RungProbeResult(
        rung="R0", ckpt_path="r0.pt", step=100, n_params=29_000_000,
        rung_accuracy={"R0": 0.95},
    )
    deltas = compute_retention_deltas(current, prior=None)
    assert deltas == {}


def test_retention_gate_passes_within_threshold() -> None:
    """G2 gate: no prior rung drops > 10% absolute (default threshold)."""
    deltas = {"R0": -0.05, "R1": -0.03}  # both within 10%
    passed, violators = check_retention_gate(deltas, threshold=-0.10)
    assert passed
    assert violators == []


def test_retention_gate_fails_on_catastrophic_drop() -> None:
    """G2 violation: R0 drops 15% (below -0.10 threshold)."""
    deltas = {"R0": -0.15, "R1": -0.02}
    passed, violators = check_retention_gate(deltas, threshold=-0.10)
    assert not passed
    assert violators == ["R0"]


# ============================================================================ #
# --load-from compat validation
# ============================================================================ #

@dataclass
class _MockCfg:
    """Minimal config object matching HierarchicalReasoningModelConfig field names."""
    use_ternary_bulk: bool = True
    hidden_size: int = 512
    n_layers: int = 8
    num_heads: int = 4
    H_cycles: int = 2
    L_cycles: int = 3
    half_layers: bool = True
    expansion: float = 4
    max_seq_len: int = 384
    attn_type: str = "prefixlm"
    init_type: str = "lecun_normal"
    norm_type: str = "pre"


def _good_ckpt_config():
    """A loaded ckpt config that matches the current cfg + tokenizer."""
    tok = BroadTokenizer()
    return {
        "gsm8k_char_vocab": tok.vocab_as_list(),
        "gsm8k_normalizer_version": tok.normalizer_version,
        "use_ternary_bulk": True,
        "hidden_size": 512,
        "n_layers": 8,
        "num_heads": 4,
        "H_cycles": 2,
        "L_cycles": 3,
        "half_layers": True,
        "expansion": 4,
        "max_seq_len": 384,
        "attn_type": "prefixlm",
        "init_type": "lecun_normal",
        "norm_type": "pre",
    }


def test_load_from_compat_passes_on_match() -> None:
    """Matching loaded config + current = no error."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    validate_load_from_ckpt_compat(
        loaded_ckpt_config=_good_ckpt_config(),
        current_cfg=cfg,
        current_vocab_list=tok.vocab_as_list(),
        current_normalizer_version=tok.normalizer_version,
    )


def test_load_from_compat_vocab_mismatch_fails() -> None:
    """Vocab list differs (e.g., Phase 2 GSM8k ckpt has vocab=98 chars) -> hard fail."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    bad = _good_ckpt_config()
    bad["gsm8k_char_vocab"] = ["<pad>", "<bos>", "<eos>", "<sep>", "a", "b", "c"]  # 7 entries
    with pytest.raises(ValueError, match="vocab.*differs|vocab.*mismatch"):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


def test_load_from_compat_normalizer_mismatch_fails() -> None:
    """normalizer_version drift -> hard fail."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    bad = _good_ckpt_config()
    bad["gsm8k_normalizer_version"] = "v2"  # different from byte_utf8_v1
    with pytest.raises(ValueError, match="normalizer.*mismatch"):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


def test_load_from_compat_ternary_flag_mismatch_fails() -> None:
    """use_ternary_bulk drift -> hard fail. Curriculum can't switch FP <-> ternary."""
    tok = BroadTokenizer()
    cfg = _MockCfg(use_ternary_bulk=False)
    bad = _good_ckpt_config()
    # bad has use_ternary_bulk=True, cfg has False
    with pytest.raises(ValueError, match="ternary"):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


@pytest.mark.parametrize("field,bad_value", [
    ("hidden_size", 256),
    ("n_layers", 4),
    ("num_heads", 2),
    ("H_cycles", 3),
    ("L_cycles", 4),
    ("max_seq_len", 256),
])
def test_load_from_compat_arch_mismatch_fails(field, bad_value) -> None:
    """Any arch field drift -> hard fail with that field surfaced."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    bad = _good_ckpt_config()
    bad[field] = bad_value
    with pytest.raises(ValueError, match=field):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


def test_load_from_compat_missing_vocab_field_fails() -> None:
    """Loaded ckpt missing gsm8k_char_vocab -> hard fail."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    bad = _good_ckpt_config()
    del bad["gsm8k_char_vocab"]
    with pytest.raises(ValueError, match="missing.*gsm8k_char_vocab"):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


def test_load_from_compat_phase2_gsm8k_ckpt_rejected() -> None:
    """Realistic regression: a Phase 2 ckpt with the GSM8k char vocab
    (98 entries) loaded against a Phase 3 broad-vocab config (260) must
    fail with a clear vocab-mismatch message."""
    tok = BroadTokenizer()
    cfg = _MockCfg()
    # Simulate a Phase 2 ckpt with 98-char GSM8k vocab
    phase2_vocab = ["<pad>", "<bos>", "<eos>", "<sep>"] + [chr(i) for i in range(33, 127)]
    assert len(phase2_vocab) == 4 + 94  # 98 entries
    bad = _good_ckpt_config()
    bad["gsm8k_char_vocab"] = phase2_vocab
    bad["gsm8k_normalizer_version"] = "v2"  # Phase 2 normalizer
    with pytest.raises(ValueError, match="vocab.*differs|vocab.*mismatch"):
        validate_load_from_ckpt_compat(
            loaded_ckpt_config=bad,
            current_cfg=cfg,
            current_vocab_list=tok.vocab_as_list(),
            current_normalizer_version=tok.normalizer_version,
        )


# ============================================================================ #
# Length histogram: BroadTokenizer vs GSM8k char tokenizer
# ============================================================================ #

def test_length_histogram_broad_vs_char_ascii() -> None:
    """Quick ASCII parity sanity: short ASCII strings must encode to identical
    length under BroadTokenizer and the GSM8k char tokenizer.

    The real corpus-level measurement (codex gate) is in
    test_length_histogram_real_gsm8k at max_len=384."""
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    corpus = [
        {"question": "what is 17 plus 23?", "expected": 40},
        {"question": "what is 5 times 6?", "expected": 30},
    ]
    gsm8k_tok = Gsm8kTokenizer.from_corpus(corpus)
    broad_tok = BroadTokenizer()

    for s in ["what is 17 plus 23?", "what is 5 times 6?"]:
        gsm_len = len(gsm8k_tok.encode(s))
        broad_len = len(broad_tok.encode(s))
        assert gsm_len == broad_len == len(s), (
            f"ASCII length differs: {s!r} -> gsm={gsm_len}, broad={broad_len}, len={len(s)}"
        )


# ============================================================================ #
# Cross-process / PYTHONHASHSEED regression: stable seed derivation
# ============================================================================ #

def _subproc_capture(code: str, env_overrides: dict | None = None) -> str:
    """Run `code` in a fresh Python subprocess with optional env overrides.
    Returns stdout (stripped). Used to prove seed derivation is stable
    across processes / PYTHONHASHSEED values."""
    import os
    import subprocess
    import sys

    env = os.environ.copy()
    # Drop any cached compiled modules path tricks; let import resolve fresh
    env.pop("PYTHONDONTWRITEBYTECODE", None)
    if env_overrides:
        env.update(env_overrides)
    # Ensure repo on sys.path
    env["PYTHONPATH"] = "."
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        cwd=".",
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subproc failed (rc={proc.returncode}):\nSTDOUT:\n{proc.stdout}\n"
            f"STDERR:\n{proc.stderr}"
        )
    return proc.stdout.strip()


def test_r1_partition_stable_across_pythonhashseed() -> None:
    """R1 enumerate-partition output must be identical across processes
    with different PYTHONHASHSEED values. Regression for the original
    builtin-`hash()` bug (codex msg 1779461471151)."""
    code = (
        "import json\n"
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r1\n"
        "train, held = _enumerate_partition_r1(42)\n"
        "print(json.dumps({'train': sorted(map(list, train)), "
        "'held': sorted(map(list, held))}))\n"
    )
    out_a = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "0"})
    out_b = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "1"})
    out_c = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "random"})
    assert out_a == out_b == out_c, (
        "R1 partition diverged across PYTHONHASHSEED — builtin hash() salt "
        f"leaking into seed derivation. PYHS=0:\n{out_a[:200]}\n"
        f"PYHS=1:\n{out_b[:200]}"
    )


def test_r3_partition_stable_across_pythonhashseed() -> None:
    """R3 enumerate-partition (with 17×23 force-inject) must be identical
    across processes with different PYTHONHASHSEED values."""
    code = (
        "import json\n"
        "from calm.hrm_text_158.curriculum.generators import _enumerate_partition_r3\n"
        "train, held = _enumerate_partition_r3(42)\n"
        "print(json.dumps({'train': sorted(map(list, train)), "
        "'held': sorted(map(list, held))}))\n"
    )
    out_a = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "0"})
    out_b = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "999"})
    assert out_a == out_b, "R3 partition diverged across PYTHONHASHSEED"


def test_make_rng_stable_across_pythonhashseed() -> None:
    """`_make_rng(rung, seed, split)` outputs must be reproducible across
    processes — covers the third use of builtin hash() in generators.py."""
    code = (
        "from calm.hrm_text_158.curriculum.generators import _make_rng\n"
        "rng = _make_rng('R0', 42, 'train')\n"
        "print(','.join(str(rng.randint(0, 10**9)) for _ in range(8)))\n"
    )
    out_a = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "0"})
    out_b = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "12345"})
    out_c = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "random"})
    assert out_a == out_b == out_c, (
        f"_make_rng diverged across PYTHONHASHSEED: "
        f"PYHS=0:{out_a} PYHS=12345:{out_b} PYHS=random:{out_c}"
    )


def test_make_rung_examples_stable_across_pythonhashseed() -> None:
    """End-to-end: same `(rung, seed, split)` must produce identical
    examples across separate Python processes."""
    code = (
        "import json\n"
        "from calm.hrm_text_158.curriculum.generators import make_rung_examples\n"
        "rows = make_rung_examples('R1', n=20, seed=42, split='held_out')\n"
        "print(json.dumps(rows))\n"
    )
    out_a = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "0"})
    out_b = _subproc_capture(code, env_overrides={"PYTHONHASHSEED": "7"})
    assert out_a == out_b, "make_rung_examples diverged across PYTHONHASHSEED"


# ============================================================================ #
# Real GSM8k length histogram — broad-byte vs legacy char at max_len=384
# ============================================================================ #

def _load_gsm8k_or_skip() -> tuple[list[dict], list[dict]]:
    """Load real GSM8k train + val; skip the test if HF cache unavailable
    (e.g., CI without HF token / network). Returns (train, val)."""
    try:
        from scripts.train_hrm_text_158 import load_gsm8k_splits
    except ImportError:
        pytest.skip("scripts.train_hrm_text_158 not importable")
    try:
        train, val, _test = load_gsm8k_splits(val_frac=0.10)
    except Exception as exc:
        pytest.skip(f"GSM8k load_dataset failed (no HF cache/network?): {exc}")
    if not train or not val:
        pytest.skip(f"GSM8k empty train/val: train={len(train)} val={len(val)}")
    return train, val


def test_length_histogram_real_gsm8k_train_val() -> None:
    """Codex gate (msg 1779461471151): real GSM8k train+val length histogram
    under BroadTokenizer vs Gsm8kTokenizer at max_len=384.

    Measures actual cached corpus, not toy ASCII strings. Asserts:
    - dropped (encoded-len > max_len) counts within ±5% between tokenizers
    - too_long pre-EOS counts within ±5%

    If the deltas are zero (ASCII-only GSM8k), the gate passes trivially.
    If non-ASCII (curly quotes, em-dashes) shows up, multi-byte UTF-8
    pushes BroadTokenizer length above the char tokenizer at the affected
    rows — gate surfaces a data-side issue."""
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    train, val = _load_gsm8k_or_skip()
    MAX_LEN = 384

    # Build Gsm8kTokenizer from train+val (the canonical Phase 2 corpus)
    gsm8k_tok = Gsm8kTokenizer.from_corpus(train + val)
    broad_tok = BroadTokenizer()

    def _hist(rows: list[dict], tok) -> dict:
        n = len(rows)
        n_dropped = 0
        total_len = 0
        max_seen = 0
        for r in rows:
            ids, _sep = tok.encode_example(r["question"], r["expected"])
            L = len(ids)
            total_len += L
            max_seen = max(max_seen, L)
            if L > MAX_LEN:
                n_dropped += 1
        return {
            "n": n,
            "n_dropped": n_dropped,
            "drop_frac": n_dropped / n if n else 0.0,
            "avg_len": total_len / n if n else 0.0,
            "max_len_seen": max_seen,
        }

    rows = train + val  # full canonical corpus
    h_broad = _hist(rows, broad_tok)
    h_char = _hist(rows, gsm8k_tok)

    # Receipt — surfaced on test failure
    receipt = (
        f"\nGSM8k length histogram (max_len={MAX_LEN}):\n"
        f"  rows                = {h_char['n']}\n"
        f"  char  dropped       = {h_char['n_dropped']} ({h_char['drop_frac']:.3%})\n"
        f"  broad dropped       = {h_broad['n_dropped']} ({h_broad['drop_frac']:.3%})\n"
        f"  char  avg_len       = {h_char['avg_len']:.2f}\n"
        f"  broad avg_len       = {h_broad['avg_len']:.2f}\n"
        f"  char  max_len_seen  = {h_char['max_len_seen']}\n"
        f"  broad max_len_seen  = {h_broad['max_len_seen']}\n"
    )

    # ±5% absolute frac (not relative — handles drop_frac==0 base case
    # without div-by-zero asymmetry)
    drop_frac_delta = abs(h_broad["drop_frac"] - h_char["drop_frac"])
    assert drop_frac_delta <= 0.05, (
        f"drop_frac diverges >5% absolute between broad and char: "
        f"|{h_broad['drop_frac']:.4f} - {h_char['drop_frac']:.4f}| = "
        f"{drop_frac_delta:.4f}{receipt}"
    )

    # Average length must be close (ASCII parity expectation)
    if h_char["avg_len"] > 0:
        avg_rel = abs(h_broad["avg_len"] - h_char["avg_len"]) / h_char["avg_len"]
        assert avg_rel <= 0.05, (
            f"avg encoded length diverges >5% relative: char={h_char['avg_len']:.2f} "
            f"broad={h_broad['avg_len']:.2f} (rel={avg_rel:.4f}){receipt}"
        )

    # Persist receipt to stdout for commit-receipt capture
    print(receipt)
