"""Unit tests for CardRouter — pure Python, no Gemma involvement."""

import torch

from calm.hrm.data import _ID_TO_CHAR, _CHAR_TO_ID, VOCAB_SIZE
from calm.llm_computer.card_router import CardRouter, Route


def _pt_string_as_logprobs(text: str, vocab: int = VOCAB_SIZE
                            ) -> torch.Tensor:
    """Encode a string as one-hot log-probs across the PT vocab. Returns
    a (1, S, V) tensor usable as a fake residual channel range."""
    ids = [_CHAR_TO_ID.get(c, 0) for c in text]
    S = len(ids)
    out = torch.full((1, S, vocab), -100.0)
    for i, tok in enumerate(ids):
        out[0, i, tok] = 0.0  # argmax → tok
    return out


def _fake_residual(pt_out: torch.Tensor, ch_lo: int, d_model: int = 4096
                    ) -> torch.Tensor:
    """Embed pt_out at channels [ch_lo:ch_lo+V] in a zero residual."""
    B, S, V = pt_out.shape
    h = torch.zeros(B, S, d_model)
    h[..., ch_lo:ch_lo + V] = pt_out
    return h


def test_decode_pt_output():
    router = CardRouter(id_to_char=_ID_TO_CHAR)
    pt = _pt_string_as_logprobs("12+7=")
    h = _fake_residual(pt, ch_lo=2400)
    text = router.decode_pt_output(h, 2400, 2400 + VOCAB_SIZE)
    assert text == "12+7=", f"got {text!r}"


def test_decode_strips_specials():
    router = CardRouter(id_to_char=_ID_TO_CHAR)
    # Include <bos> and <eos> surrounding the expression
    ids = ([_CHAR_TO_ID["<bos>"]]
           + [_CHAR_TO_ID[c] for c in "5+3"]
           + [_CHAR_TO_ID["<eos>"]])
    S = len(ids)
    pt = torch.full((1, S, VOCAB_SIZE), -100.0)
    for i, t in enumerate(ids):
        pt[0, i, t] = 0.0
    h = _fake_residual(pt, ch_lo=2400)
    text = router.decode_pt_output(h, 2400, 2400 + VOCAB_SIZE)
    assert text == "5+3", f"got {text!r}"


def test_parse_operands_basic():
    parsed = CardRouter._parse_operands("12+7", "+")
    assert parsed == [12, 7]


def test_parse_operands_with_equals_tail():
    parsed = CardRouter._parse_operands("5+3=", "+")
    assert parsed == [5, 3]


def test_parse_operands_with_whitespace():
    parsed = CardRouter._parse_operands(" 100 + 25 ", "+")
    assert parsed == [100, 25]


def test_parse_operands_missing_operator():
    assert CardRouter._parse_operands("12+7", "*") is None


def test_parse_operands_malformed_returns_none():
    assert CardRouter._parse_operands("abc+def", "+") is None


def test_route_forward_dispatches():
    calls = {"n": 0, "last_ops": None}

    def translate(ops):
        calls["n"] += 1
        calls["last_ops"] = ops
        return torch.tensor([[ops[0], ops[1]]])

    router = CardRouter(id_to_char=_ID_TO_CHAR)
    router.register(Route(
        source_ch=(2400, 2400 + VOCAB_SIZE),
        operator="+",
        target_card_slot=None,  # routing doesn't actually call the card
        translator=translate,
    ))

    pt = _pt_string_as_logprobs("14+28=")
    h = _fake_residual(pt, ch_lo=2400)
    result = router.route_forward(h, 0)

    assert calls["n"] == 1
    assert calls["last_ops"] == [14, 28]
    assert result.tolist() == [[14, 28]]


def test_fallback_on_parse_failure():
    router = CardRouter(id_to_char=_ID_TO_CHAR)
    router.register(Route(
        source_ch=(2400, 2400 + VOCAB_SIZE),
        operator="+",
        target_card_slot=None,
        translator=lambda ops: torch.tensor([[ops[0], ops[1]]]),
        fallback_operands=[9, 9],
    ))
    # No '+' in the PT output
    pt = _pt_string_as_logprobs("gcd")
    h = _fake_residual(pt, ch_lo=2400)
    result = router.route_forward(h, 0)
    assert result.tolist() == [[9, 9]]


def test_multiple_routes_dispatch_independently():
    plus_calls = {"n": 0}
    mul_calls = {"n": 0}

    def plus_trans(ops):
        plus_calls["n"] += 1
        return torch.tensor([[ops[0], ops[1]]])

    def mul_trans(ops):
        mul_calls["n"] += 1
        return torch.tensor([[ops[0] + 100, ops[1] + 100]])

    router = CardRouter(id_to_char=_ID_TO_CHAR)
    router.register(Route(
        source_ch=(2400, 2400 + VOCAB_SIZE),
        operator="+", target_card_slot=None, translator=plus_trans))
    router.register(Route(
        source_ch=(2400, 2400 + VOCAB_SIZE),
        operator="*", target_card_slot=None, translator=mul_trans))

    pt = _pt_string_as_logprobs("4*5=")
    h = _fake_residual(pt, ch_lo=2400)
    # Route 0 is '+' but PT says '*' — operator match fails, fallback
    r0 = router.route_forward(h, 0)
    # Route 1 is '*' — dispatches
    r1 = router.route_forward(h, 1)

    # Plus translator runs once (on fallback operands), mul translator once.
    assert plus_calls["n"] == 1
    assert mul_calls["n"] == 1
    assert r0.tolist() == [[0, 0]]  # default fallback_operands
    assert r1.tolist() == [[104, 105]]


def test_make_card_input_fn_closure():
    router = CardRouter(id_to_char=_ID_TO_CHAR)
    idx = router.register(Route(
        source_ch=(2400, 2400 + VOCAB_SIZE),
        operator="+", target_card_slot=None,
        translator=lambda ops: torch.tensor([ops])))
    fn = router.make_card_input_fn(idx)

    pt = _pt_string_as_logprobs("7+8")
    h = _fake_residual(pt, ch_lo=2400)
    assert fn(h).tolist() == [[7, 8]]
