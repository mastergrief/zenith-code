"""Tests for `code_dt_data.py` — extraction + paraphrase augmentation."""
import pytest

from calm.hrm.code_dt_data import (
    CODE_VOCAB_SIZE,
    CodeProblem,
    _clean_prob,
    _extract_skeleton,
    _paraphrase_augment,
    code_detokenize,
    code_tokenize,
)


def test_vocab_contains_colon():
    """Required for function headers."""
    from calm.hrm.code_dt_data import _CODE_CHAR_TO_ID
    assert ":" in _CODE_CHAR_TO_ID


def test_vocab_size():
    assert CODE_VOCAB_SIZE == 81   # 4 specials + 77 chars


def test_tokenize_detokenize_roundtrip():
    text = "def FN(arr, n):"
    ids = code_tokenize(text)
    out = code_detokenize(ids)
    assert out == text


def test_clean_prob_accepts_normal_problem():
    prob = "Write a function to find the longest chain which can be formed"
    cleaned = _clean_prob(prob)
    assert cleaned == prob


def test_clean_prob_rejects_too_short():
    assert _clean_prob("short") is None


def test_clean_prob_collapses_whitespace():
    prob = "Write   a\n\nfunction  to do  stuff here now."
    cleaned = _clean_prob(prob)
    assert "  " not in cleaned


def test_clean_prob_drops_unicode():
    prob = "Write a function to convert emoji 🔥 to ASCII equivalents."
    cleaned = _clean_prob(prob)
    assert "🔥" not in cleaned


def test_extract_skeleton_uses_placeholder():
    sol = "```python\ndef max_chain_length(arr, n):\n    return 0\n```"
    result = _extract_skeleton(sol)
    assert result is not None
    fn_name, skeleton = result
    assert fn_name == "max_chain_length"
    assert skeleton == "def FN(arr, n):"
    assert "max_chain_length" not in skeleton


def test_extract_skeleton_prefers_top_level():
    """MBPP pattern: helper class method, target fn last."""
    sol = """
class Helper:
    def __init__(self, a):
        self.a = a

def target_fn(x, y):
    return x + y
"""
    result = _extract_skeleton(sol)
    assert result is not None
    fn_name, skeleton = result
    assert fn_name == "target_fn"
    assert skeleton == "def FN(x, y):"


def test_extract_skeleton_custom_placeholder():
    sol = "def foo(x):\n    pass"
    result = _extract_skeleton(sol, placeholder="F")
    assert result is not None
    _, skeleton = result
    assert skeleton == "def F(x):"


def test_paraphrase_augment_expands():
    pairs = [
        CodeProblem(
            question="Write a function to compute x.",
            expression="def FN(x):",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=4, seed=0)
    # Original + up to 3 paraphrases
    assert len(augmented) >= 2
    assert len(augmented) <= 4
    # All variants have the same skeleton
    assert all(p.expression == "def FN(x):" for p in augmented)
    # Distinct question strings
    qs = {p.question for p in augmented}
    assert len(qs) >= 2


def test_paraphrase_preserves_original():
    """The original prompt always survives augmentation."""
    pairs = [
        CodeProblem(
            question="Write a function to square x.",
            expression="def FN(x):",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=5, seed=0)
    questions = [p.question for p in augmented]
    assert "Write a function to square x." in questions


def test_paraphrase_no_match_returns_single():
    """If no template prefix matches, only the original is emitted."""
    pairs = [
        CodeProblem(
            question="Utterly arbitrary prefix that no template matches here now.",
            expression="def FN():",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=5, seed=0)
    # No template matched → only the original passes through
    assert len(augmented) == 1
    assert augmented[0].question == pairs[0].question


def test_paraphrase_factor_1_noop():
    pairs = [
        CodeProblem(
            question="Write a function to do something small.",
            expression="def FN():",
        ),
    ]
    augmented = _paraphrase_augment(pairs, factor=1, seed=0)
    # factor=1 means "only the original" — factor-1 = 0 extra
    assert len(augmented) == 1


# --- Balanced sampler weights (Round 3a) ---

def _zipf_pairs(distribution):
    """Helper: build CodeProblem list with per-class counts from dict."""
    pairs = []
    for skel, n in distribution.items():
        for _ in range(n):
            pairs.append(CodeProblem(question=f"prompt for {skel}",
                                      expression=skel))
    return pairs


def test_balanced_weights_uniform_is_noop():
    from calm.hrm.code_dt_data import build_balanced_sampler_weights
    pairs = _zipf_pairs({"def FN(n):": 100, "def FN(s):": 2})
    w = build_balanced_sampler_weights(pairs, strategy="uniform")
    assert w == [1.0] * len(pairs)


def test_balanced_weights_inverse_flattens():
    """inverse weighting: rare class gets 50× more weight than common
    (100:2 raw → 1/2 vs 1/100 per-item weight)."""
    from calm.hrm.code_dt_data import build_balanced_sampler_weights
    pairs = _zipf_pairs({"def FN(n):": 100, "def FN(s):": 2})
    w = build_balanced_sampler_weights(pairs, strategy="inverse")
    # Common class pair weight
    common_w = next(w_i for p, w_i in zip(pairs, w)
                    if p.expression == "def FN(n):")
    rare_w = next(w_i for p, w_i in zip(pairs, w)
                  if p.expression == "def FN(s):")
    assert rare_w / common_w == pytest.approx(50.0)


def test_balanced_weights_sqrt_inverse_moderates():
    """sqrt_inverse: rare class gets sqrt(50) ≈ 7.07× more weight.
    Less aggressive than pure inverse — preserves some frequency signal.
    """
    import math
    from calm.hrm.code_dt_data import build_balanced_sampler_weights
    pairs = _zipf_pairs({"def FN(n):": 100, "def FN(s):": 2})
    w = build_balanced_sampler_weights(pairs, strategy="sqrt_inverse")
    common_w = next(w_i for p, w_i in zip(pairs, w)
                    if p.expression == "def FN(n):")
    rare_w = next(w_i for p, w_i in zip(pairs, w)
                  if p.expression == "def FN(s):")
    assert rare_w / common_w == pytest.approx(math.sqrt(50.0))


def test_balanced_weights_capped_bounds_lift():
    """capped: once count ≤ cap, further reduction doesn't lift more."""
    from calm.hrm.code_dt_data import build_balanced_sampler_weights
    pairs = _zipf_pairs({
        "def FN(n):": 100,
        "def FN(s):": 2,
        "def FN(x):": 1,
    })
    w = build_balanced_sampler_weights(pairs, strategy="capped", cap=5)
    # Classes with count ≥ 5 clamp to 1/5; classes below clamp to 1/count
    weights_by_skel = {}
    for p, w_i in zip(pairs, w):
        weights_by_skel[p.expression] = w_i
    # FN(s) (count=2) should clamp to 1/min(2,5) = 1/2
    assert weights_by_skel["def FN(s):"] == pytest.approx(1/2)
    # FN(x) (count=1) should clamp to 1/min(1,5) = 1/1
    assert weights_by_skel["def FN(x):"] == pytest.approx(1.0)
    # FN(n) (count=100) clamps to 1/min(100,5) = 1/5
    assert weights_by_skel["def FN(n):"] == pytest.approx(1/5)


def test_balanced_weights_sample_entropy_increases():
    """Raw measurement: sampling with sqrt_inverse weights produces
    a more uniform skeleton distribution than sampling uniformly.
    This is the Round 3a raw-path metric."""
    import math
    from collections import Counter
    import torch
    from torch.utils.data import WeightedRandomSampler
    from calm.hrm.code_dt_data import build_balanced_sampler_weights

    # 5:1 skew over 3 classes
    pairs = _zipf_pairs({
        "def FN(n):": 100,
        "def FN(s):": 20,
        "def FN(x):": 4,
    })
    w = build_balanced_sampler_weights(pairs, strategy="sqrt_inverse")

    g = torch.Generator().manual_seed(42)
    sampler = WeightedRandomSampler(
        weights=w, num_samples=len(pairs), replacement=True, generator=g,
    )
    drawn = Counter(pairs[i].expression for i in sampler)

    # Baseline Zipf entropy
    base_counter = Counter(p.expression for p in pairs)
    base_total = sum(base_counter.values())
    base_ent = -sum((c/base_total) * math.log(c/base_total)
                    for c in base_counter.values())

    # Weighted-sample entropy
    drawn_total = sum(drawn.values())
    drawn_ent = -sum((c/drawn_total) * math.log(c/drawn_total)
                     for c in drawn.values())

    # Max entropy for 3 classes = log(3) ≈ 1.099
    max_ent = math.log(3)

    # Weighted should be closer to uniform than raw
    assert drawn_ent > base_ent, (
        f"weighted entropy ({drawn_ent:.3f}) should exceed "
        f"raw entropy ({base_ent:.3f})"
    )
    # sqrt_inverse on 100:20:4 skew has analytical expectation ~0.83
    # of max entropy (weighted-mass ratios 10:4.47:2 → p=0.61:0.27:0.12
    # → ent/max ≈ 0.83). Allow finite-sample slack to 0.70.
    assert drawn_ent / max_ent > 0.70


def test_balanced_weights_unknown_strategy_raises():
    from calm.hrm.code_dt_data import build_balanced_sampler_weights
    pairs = _zipf_pairs({"def FN(n):": 5})
    with pytest.raises(ValueError):
        build_balanced_sampler_weights(pairs, strategy="bogus")


# --- Round 5: copy-gate bias init ---

def test_copy_gate_bias_default_preserved():
    """Ensure default behavior (v4 baseline) is unchanged."""
    from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
    model = build_copy_augmented_delta(vocab_size=16, max_len=32)
    assert model.copy_gate.bias.item() == pytest.approx(-2.0)


def test_copy_gate_bias_configurable():
    """Non-default init applies correctly."""
    from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
    for init_val in [-2.0, 0.0, 1.0, 2.0]:
        model = build_copy_augmented_delta(
            vocab_size=16, max_len=32,
            copy_gate_bias_init=init_val,
        )
        assert model.copy_gate.bias.item() == pytest.approx(init_val), (
            f"init_val={init_val} → bias={model.copy_gate.bias.item()}"
        )


# --- Round 8: data expansion (extract-all-defs) ---

def test_extract_all_skeletons_single_def():
    from calm.hrm.code_dt_data import _extract_all_skeletons
    sol = "def target(x):\n    return x"
    out = _extract_all_skeletons(sol)
    assert out == [("target", "def FN(x):")]


def test_extract_all_skeletons_multiple_defs():
    """Multi-def solution: both defs emitted as separate skeletons."""
    from calm.hrm.code_dt_data import _extract_all_skeletons
    sol = """
def helper(a):
    return a + 1

def target(x, y):
    return helper(x) + y
"""
    out = _extract_all_skeletons(sol)
    skels = {s for _, s in out}
    assert "def FN(a):" in skels
    assert "def FN(x, y):" in skels


def test_extract_all_skeletons_dedups_within_solution():
    """Same skeleton twice in one solution = emit once."""
    from calm.hrm.code_dt_data import _extract_all_skeletons
    sol = """
def f1(n):
    return n

def f2(n):
    return n + 1
"""
    out = _extract_all_skeletons(sol)
    skels = [s for _, s in out]
    # Both defs have same args → dedup to one skeleton
    assert skels.count("def FN(n):") == 1


def test_extract_all_skeletons_skips_indented():
    """Class-method defs are indented, not top-level."""
    from calm.hrm.code_dt_data import _extract_all_skeletons
    sol = """
class Helper:
    def method(self):
        pass

def target(x):
    return x
"""
    out = _extract_all_skeletons(sol)
    skels = {s for _, s in out}
    assert "def FN(x):" in skels
    # method is indented → not included
    assert "def FN(self):" not in skels


# --- Round 7: output-family split ---

def test_arg_count_zero():
    from calm.hrm.code_dt_data import arg_count
    assert arg_count("def FN():") == 0


def test_arg_count_single():
    from calm.hrm.code_dt_data import arg_count
    assert arg_count("def FN(n):") == 1
    assert arg_count("def FN(self):") == 1


def test_arg_count_multi():
    from calm.hrm.code_dt_data import arg_count
    assert arg_count("def FN(a, b):") == 2
    assert arg_count("def FN(a,b):") == 2
    assert arg_count("def FN(a, b, c):") == 3


def test_arg_count_malformed():
    from calm.hrm.code_dt_data import arg_count
    assert arg_count("not a skeleton") == -1


def test_family_bucket():
    from calm.hrm.code_dt_data import family_bucket
    assert family_bucket("def FN():") == "zero"
    assert family_bucket("def FN(n):") == "one"
    assert family_bucket("def FN(a, b):") == "two"
    assert family_bucket("def FN(a, b, c):") == "three_plus"
    assert family_bucket("def FN(a, b, c, d):") == "three_plus"
    assert family_bucket("bogus") == "unknown"


def test_split_pairs_by_family():
    from calm.hrm.code_dt_data import split_pairs_by_family
    pairs = [
        CodeProblem(question="q0", expression="def FN():"),
        CodeProblem(question="q1", expression="def FN(n):"),
        CodeProblem(question="q2", expression="def FN(s):"),
        CodeProblem(question="q3", expression="def FN(a, b):"),
        CodeProblem(question="q4", expression="def FN(a, b, c):"),
    ]
    buckets = split_pairs_by_family(pairs)
    assert len(buckets["zero"]) == 1
    assert len(buckets["one"]) == 2
    assert len(buckets["two"]) == 1
    assert len(buckets["three_plus"]) == 1


# --- Round 6: skeleton normalization + drop rare ---

def test_normalize_skeleton_collapses_spacing():
    from calm.hrm.code_dt_data import normalize_skeleton
    # All three should collapse to the same canonical form
    assert normalize_skeleton("def FN(a, b):") == "def FN(a, b):"
    assert normalize_skeleton("def FN(a,b):") == "def FN(a, b):"
    assert normalize_skeleton("def FN( a , b ):") == "def FN(a, b):"


def test_normalize_skeleton_preserves_single_arg():
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("def FN(n):") == "def FN(n):"
    assert normalize_skeleton("def FN( n ):") == "def FN(n):"


def test_normalize_skeleton_preserves_empty():
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("def FN():") == "def FN():"


def test_normalize_skeleton_leaves_malformed():
    """If input isn't a valid skeleton, return as-is."""
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("not a skeleton") == "not a skeleton"


def test_normalize_strips_type_annotation_default():
    from calm.hrm.code_dt_data import normalize_skeleton
    # Default strip_annotations=True
    assert normalize_skeleton("def FN(n: int):") == "def FN(n):"
    assert normalize_skeleton("def FN(s: str):") == "def FN(s):"
    assert normalize_skeleton("def FN(l: list):") == "def FN(l):"


def test_normalize_strips_default_values():
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("def FN(n: int = 10):") == "def FN(n):"
    assert normalize_skeleton("def FN(capacity=100):") == "def FN(capacity):"


def test_normalize_strips_multi_arg_annotations():
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("def FN(l: list, t: int):") == "def FN(l, t):"
    assert normalize_skeleton(
        "def FN(text: str, visible: int = 6):"
    ) == "def FN(text, visible):"


def test_normalize_opt_out():
    """strip_annotations=False preserves annotations (for v4 compat)."""
    from calm.hrm.code_dt_data import normalize_skeleton
    assert normalize_skeleton("def FN(n: int):",
                                 strip_annotations=False) == "def FN(n: int):"


def test_filter_rare_classes_drops_below_threshold():
    from calm.hrm.code_dt_data import filter_rare_classes
    pairs = (
        [CodeProblem(question=f"q{i}", expression="def FN(n):") for i in range(10)] +
        [CodeProblem(question="q_rare1", expression="def FN(z):")] +
        [CodeProblem(question="q_rare2", expression="def FN(y):")] +
        [CodeProblem(question="q_s1", expression="def FN(s):")] * 2
    )
    filtered = filter_rare_classes(pairs, min_count=3)
    # FN(n) has 10 → stays; FN(z) has 1 → drops; FN(y) has 1 → drops;
    # FN(s) has 2 → drops
    classes_kept = {p.expression for p in filtered}
    assert classes_kept == {"def FN(n):"}
    assert len(filtered) == 10


def test_filter_rare_classes_preserves_common():
    from calm.hrm.code_dt_data import filter_rare_classes
    pairs = [CodeProblem(question=f"q{i}", expression="def FN(n):")
             for i in range(5)]
    filtered = filter_rare_classes(pairs, min_count=3)
    assert len(filtered) == 5


# --- Round 19: dedupe ambiguous prompts ---

def test_dedupe_drops_conceptual_prompts():
    """Prompts with 3+ distinct skeletons are dropped entirely."""
    from calm.hrm.code_dt_data import dedupe_ambiguous_prompts
    pairs = (
        [CodeProblem("conceptual", "def FN(a):"),
         CodeProblem("conceptual", "def FN(b):"),
         CodeProblem("conceptual", "def FN(c):")] +
        [CodeProblem("clear q", "def FN(n):")]
    )
    out = dedupe_ambiguous_prompts(pairs, drop_if_skels_geq=3)
    # conceptual dropped; clear q kept
    assert [p.question for p in out] == ["clear q"]


def test_dedupe_majority_vote_on_two_skels():
    """With 2 skeletons, keep the globally-more-common one."""
    from calm.hrm.code_dt_data import dedupe_ambiguous_prompts
    pairs = (
        # Global: FN(n) appears 5 times, FN(s) appears 1
        [CodeProblem(f"common_q{i}", "def FN(n):") for i in range(5)] +
        [CodeProblem("amb", "def FN(s):")] +
        [CodeProblem("amb", "def FN(n):")]  # amb maps to both; FN(n) wins
    )
    out = dedupe_ambiguous_prompts(pairs, drop_if_skels_geq=3)
    # Only one "amb" pair survives (the FN(n) one)
    amb_pairs = [p for p in out if p.question == "amb"]
    assert len(amb_pairs) == 1
    assert amb_pairs[0].expression == "def FN(n):"


def test_dedupe_unambiguous_passes_through():
    """Single-skeleton prompts are unchanged."""
    from calm.hrm.code_dt_data import dedupe_ambiguous_prompts
    pairs = [
        CodeProblem("q1", "def FN(n):"),
        CodeProblem("q2", "def FN(s):"),
    ]
    out = dedupe_ambiguous_prompts(pairs)
    assert len(out) == 2


def test_copy_gate_bias_sigmoid_makes_sense():
    """At init=+1.0, sigmoid(p_copy_pre_data) ~ 0.73 — favors copy path."""
    import torch
    from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
    model = build_copy_augmented_delta(
        vocab_size=16, max_len=32, copy_gate_bias_init=1.0,
    )
    # With zero linear weights (default init is small), bias dominates
    # initial gate value. sigmoid(1.0) ≈ 0.731.
    sigmoid_bias = torch.sigmoid(model.copy_gate.bias).item()
    assert sigmoid_bias > 0.7
    # And default stays below 0.2
    m2 = build_copy_augmented_delta(vocab_size=16, max_len=32)
    assert torch.sigmoid(m2.copy_gate.bias).item() < 0.2
