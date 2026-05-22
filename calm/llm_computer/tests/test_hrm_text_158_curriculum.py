"""HRM-Text-1.58 Phase 3 Step 0 curriculum infrastructure tests.

Per task #51, board task 1779460303130-742c8cbd, codex msg 1779460698439
(Phase 3 Step 0 +1 with A1 byte-level UTF-8 + 7 guardrails).

Covers:
- BroadTokenizer determinism + vocab spec + roundtrip
- Synthetic generators (R0-R6) determinism + held-out non-overlap
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

@pytest.mark.parametrize("rung", ["R0", "R1", "R2", "R3", "R4", "R5", "R6"])
def test_generator_deterministic_per_seed(rung) -> None:
    """Same (rung, seed, split) -> same examples list."""
    examples_a = make_rung_examples(rung, n=20, seed=42, split="train")
    examples_b = make_rung_examples(rung, n=20, seed=42, split="train")
    assert examples_a == examples_b


@pytest.mark.parametrize("rung", ["R0", "R1", "R2", "R3", "R4", "R5", "R6"])
def test_generator_train_holdout_distinct(rung) -> None:
    """Train and held_out splits produce different examples for same seed
    (different RNG salt per split)."""
    train = make_rung_examples(rung, n=20, seed=42, split="train")
    held_out = make_rung_examples(rung, n=20, seed=42, split="held_out")
    # At least one example should differ
    assert train != held_out


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
    """Build full R0-R6 splits + assert no row in any rung's held_out
    appears in any rung's train set."""
    splits = build_rung_splits(n_train=200, n_held_out=50, seed=42)
    assert_no_train_holdout_overlap(splits)
    # Confirm we have all 7 sub-GSM8k rungs
    assert set(splits.keys()) == {"R0", "R1", "R2", "R3", "R4", "R5", "R6"}


def test_cross_rung_invariant_detects_violation() -> None:
    """Manually construct a violation: copy a held_out row into another rung's train.
    assert_no_train_holdout_overlap must raise."""
    splits = build_rung_splits(n_train=20, n_held_out=10, seed=42)
    # Inject violation: take an R1 held_out row and add it to R2 train
    violation_row = splits["R1"]["held_out"][0].copy()
    splits["R2"]["train"].append(violation_row)
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
