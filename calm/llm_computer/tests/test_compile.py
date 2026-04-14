"""Compiler correctness tests.

For each hand-wired primitive in `programs/`, build the IR version via
`compile_program` and verify it produces the same forward-pass output on
every canonical input.

Bit-match vs behavioral-match: `add_one`, `threshold`, and
`increment_counter` should bit-match hand-wired weights (IR→weights is
1:1 deterministic). `copy_past` does NOT bit-match — the hand-wired
version packs 2 channels per upper head; the IR uses 1 channel per head.
Both produce identical forward passes, so we test behavioral equivalence
across all 4 and only assert bit equality where deterministic.
"""

from __future__ import annotations

import torch

from calm.llm_computer.programs.add_one import build_add_one
from calm.llm_computer.programs.add_one_ir import build_add_one_ir
from calm.llm_computer.programs.adder_tiny import build_adder_tiny
from calm.llm_computer.programs.copy_past import build_copy_past
from calm.llm_computer.programs.copy_past_ir import build_copy_past_ir
from calm.llm_computer.programs.increment_counter import build_increment_counter
from calm.llm_computer.programs.increment_counter_ir import build_increment_counter_ir
from calm.llm_computer.programs.read_by_key import build_read_by_key
from calm.llm_computer.programs.retrieve_by_index import build_retrieve_by_index
from calm.llm_computer.programs.retrieve_threshold import build_retrieve_threshold
from calm.llm_computer.programs.threshold import build_threshold
from calm.llm_computer.programs.threshold_ir import build_threshold_ir


def _argmax_decode(model, x: torch.Tensor) -> list[int]:
    with torch.no_grad():
        return model(x)[0].argmax(dim=-1).tolist()


def test_add_one_behavioral_match():
    hand = build_add_one(vocab_size=8)
    ir = build_add_one_ir(vocab_size=8)
    assert hand.param_count() == ir.param_count() == 1280
    for k in range(8):
        x = torch.tensor([[k]], dtype=torch.long)
        assert _argmax_decode(hand, x) == _argmax_decode(ir, x), f"add_one mismatch at k={k}"


def test_add_one_bit_match():
    """add_one is fully deterministic IR→weights — bit-match expected."""
    hand = build_add_one(vocab_size=8)
    ir = build_add_one_ir(vocab_size=8)
    for name_hand, p_hand in hand.named_parameters():
        p_ir = dict(ir.named_parameters())[name_hand]
        assert torch.equal(p_hand, p_ir), f"add_one bit-match failed at {name_hand}"


def test_threshold_behavioral_match():
    hand = build_threshold(vocab_size=8, threshold_value=4)
    ir = build_threshold_ir(vocab_size=8, threshold_value=4)
    assert hand.param_count() == ir.param_count() == 216
    for k in range(8):
        x = torch.tensor([[k]], dtype=torch.long)
        assert _argmax_decode(hand, x) == _argmax_decode(ir, x), f"threshold mismatch at k={k}"


def test_threshold_bit_match():
    """FFN neuron allocation is deterministic — bit-match expected."""
    hand = build_threshold(vocab_size=8, threshold_value=4)
    ir = build_threshold_ir(vocab_size=8, threshold_value=4)
    for name_hand, p_hand in hand.named_parameters():
        p_ir = dict(ir.named_parameters())[name_hand]
        assert torch.equal(p_hand, p_ir), f"threshold bit-match failed at {name_hand}"


def test_increment_counter_behavioral_match():
    hand = build_increment_counter(vocab_size=8)
    ir = build_increment_counter_ir(vocab_size=8)
    assert hand.param_count() == ir.param_count() == 2176
    # Length 1..8, arbitrary input tokens.
    for length in (1, 3, 5, 8):
        x = torch.zeros(1, length, dtype=torch.long)
        assert _argmax_decode(hand, x) == _argmax_decode(ir, x), \
            f"increment_counter mismatch at length={length}"


def test_increment_counter_bit_match():
    hand = build_increment_counter(vocab_size=8)
    ir = build_increment_counter_ir(vocab_size=8)
    for name_hand, p_hand in hand.named_parameters():
        p_ir = dict(ir.named_parameters())[name_hand]
        assert torch.equal(p_hand, p_ir), f"increment_counter bit-match failed at {name_hand}"


def test_copy_past_behavioral_match():
    hand = build_copy_past(vocab_size=8)
    ir = build_copy_past_ir(vocab_size=8)
    assert hand.param_count() == ir.param_count() == 2560
    for inp in ([3, 7, 2, 5, 1], [0, 1, 2, 3, 4, 5, 6, 7], [7, 0], [5]):
        x = torch.tensor([inp], dtype=torch.long)
        assert _argmax_decode(hand, x) == _argmax_decode(ir, x), \
            f"copy_past mismatch at input={inp}"


def test_retrieve_by_index_parabolic_keys():
    """Parabolic-key LookUpExact: at the last position, retrieve the
    token whose value was stored at input position `query_idx`.
    Validates RESEARCH/02 §5's key construction."""
    V = 4
    N = 4  # 4 values stored at positions 0..3, query at position 4
    model = build_retrieve_by_index(vocab_size=V, max_len=N + 1)
    # All (values-permutation, query_idx) combinations.
    import itertools
    for values in itertools.product(range(V), repeat=N):
        for q in range(N):
            inp = list(values) + [q]
            x = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, N].argmax().item())
            assert got == values[q], \
                f"retrieve_by_index: values={values} q={q} got={got} exp={values[q]}"


def test_read_by_key_semantic_lookup():
    """Semantic-keyed LookUpExact: retrieve position where a named key
    was stored. Exercises ReGLU-based key-squaring and coefficient-
    scaled K projection (pos_key0_coef=2.0 on the scalar key channel)."""
    import itertools
    V = 4
    model = build_read_by_key(vocab_size=V, max_len=V + 1)
    for perm in itertools.permutations(range(V)):
        for query_key in range(V):
            inp = list(perm) + [query_key]
            expected = perm.index(query_key)
            x = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, V].argmax().item())
            assert got == expected, \
                f"read_by_key: keys={perm} q={query_key} got={got} exp={expected}"


def test_retrieve_threshold_same_layer_composition():
    """LookUpExact (attn) → ReGLU (FFN) in one layer. Validates that the
    transformer's attn-before-FFN ordering means FFN reads attn output
    in the same layer, so multi-primitive programs don't need multiple
    layers for pure data-dependency chains."""
    import itertools
    V, T, N = 4, 2, 4
    model = build_retrieve_threshold(vocab_size=V, threshold=T, max_len=N + 1)
    for values in itertools.product(range(V), repeat=N):
        for q in range(N):
            inp = list(values) + [q]
            x = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, N].argmax().item())
            expected = 1 if values[q] >= T else 0
            assert got == expected, \
                f"retrieve_threshold: values={values} q={q} got={got} exp={expected}"


def test_adder_tiny_compositional():
    """1-digit adder composing LookUp + ReGLU. a, b in [0, 3]."""
    model = build_adder_tiny(vocab_size=8, max_len=4)
    assert model.param_count() == 1020
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, 1].argmax().item())
            assert got == a + b, f"adder_tiny: {a}+{b} expected {a+b} got {got}"


def test_copy_past_weight_diff_is_documented():
    """Sanity-check that the 'different head packing' claim is true —
    if a future change makes them bit-match, this test will fail and we
    can promote copy_past to the bit-match tier."""
    hand = build_copy_past(vocab_size=8)
    ir = build_copy_past_ir(vocab_size=8)
    diffs = []
    for name_hand, p_hand in hand.named_parameters():
        p_ir = dict(ir.named_parameters())[name_hand]
        if not torch.equal(p_hand, p_ir):
            diffs.append(name_hand)
    assert "W_qkv.0.weight" in diffs, \
        "copy_past head packing changed — remove this test and promote to bit-match"


if __name__ == "__main__":
    test_add_one_behavioral_match()
    print("[ok] add_one behavioral match")
    test_add_one_bit_match()
    print("[ok] add_one bit match")
    test_threshold_behavioral_match()
    print("[ok] threshold behavioral match")
    test_threshold_bit_match()
    print("[ok] threshold bit match")
    test_increment_counter_behavioral_match()
    print("[ok] increment_counter behavioral match")
    test_increment_counter_bit_match()
    print("[ok] increment_counter bit match")
    test_copy_past_behavioral_match()
    print("[ok] copy_past behavioral match")
    test_copy_past_weight_diff_is_documented()
    print("[ok] copy_past weight diff is documented")
    test_adder_tiny_compositional()
    print("[ok] adder_tiny compositional (1-digit adder, 16 cases)")
    test_retrieve_by_index_parabolic_keys()
    print("[ok] retrieve_by_index parabolic keys (V=4, 256 combinations)")
    print("\noverall: PASS (4 primitives + adder + LookUpExact all compile from IR)")
