"""NL → math data pipeline tests.

Every NL problem the generator produces should:
  1. Have an expression that `parse_expression` + `interpret` evaluate
     to the same value as `safe_eval` (i.e., the pipeline downstream of
     HRM can recover the answer).
  2. Tokenize cleanly — every char in question/expression is in the
     HRM char vocab (no unknown-char bailout).
  3. Fit inside default encoder/decoder lengths (48/24).
"""

from __future__ import annotations

from calm.expression import safe_eval
from calm.hrm.data import _CHAR_TO_ID
from calm.hrm.nl_data import NLMathDataGenerator, NLMathSeq2SeqDataset
from calm.llm_computer.interpret import interpret
from calm.llm_computer.parse import parse_expression


def test_nl_expressions_roundtrip():
    gen = NLMathDataGenerator(seed=99)
    problems = gen.generate(200)
    for p in problems:
        graph = parse_expression(p.expression)
        got = interpret(graph)
        expected = safe_eval(p.expression)
        assert got == expected, f"{p.expression!r}: parse→{got} eval→{expected}"


def test_nl_chars_in_vocab():
    gen = NLMathDataGenerator(seed=1)
    problems = gen.generate(500)
    for p in problems:
        for c in p.question:
            assert c in _CHAR_TO_ID, f"NL char {c!r} missing from vocab in {p.question!r}"
        for c in p.expression:
            assert c in _CHAR_TO_ID, f"expr char {c!r} missing from vocab in {p.expression!r}"


def test_nl_lengths_fit():
    gen = NLMathDataGenerator(seed=2)
    problems = gen.generate(500)
    for p in problems:
        # +2 for <bos> / <eos>, +1 for `=` terminator on decoder side
        assert len(p.question) + 2 <= 48, f"question too long: {p.question!r}"
        assert len(p.expression) + 2 + 1 <= 24, f"expression too long: {p.expression!r}"


def test_nl_dataset_shapes():
    gen = NLMathDataGenerator(seed=3)
    problems = gen.generate(10)
    ds = NLMathSeq2SeqDataset(problems, max_enc_len=48, max_dec_len=24)
    item = ds[0]
    assert item["encoder_ids"].shape == (48,)
    assert item["decoder_input_ids"].shape == (24,)
    assert item["decoder_target_ids"].shape == (24,)
    assert item["loss_mask"].shape == (24,)


if __name__ == "__main__":
    test_nl_expressions_roundtrip()
    print("[ok] NL expressions round-trip through parse+interpret")
    test_nl_chars_in_vocab()
    print("[ok] NL chars all in HRM vocab")
    test_nl_lengths_fit()
    print("[ok] NL lengths fit in max_enc=48 / max_dec=24")
    test_nl_dataset_shapes()
    print("[ok] NL dataset tensor shapes")
