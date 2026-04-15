"""Tests for the v3 card abstraction + CHRLM-General build.

Verifies that the orchestrator routes compiled-program queries through
the exact tier and falls through to trained / external cards only when
nothing exact applies. Does NOT load the trained checkpoint (that would
require GPU/CPU compute on every test run); uses stub cards where needed.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from calm.llm_computer.cards import (
    CardOrchestrator,
    CompiledProgramCard,
    RouteResult,
)
from calm.llm_computer.builds.chrlm_general import (
    _eval_binary_arith,
    _eval_func_one_arg,
    _eval_func_two_args,
    build,
    compiled_cards,
)


# ----- Compiled-program card routing -----

def test_binary_arith_card_routes():
    cards = compiled_cards()
    orch = CardOrchestrator(cards=cards)
    r = orch.route("what is 17 + 23")
    assert r.answer == "40", f"expected 40, got {r.answer!r}"
    assert r.card == "binary_arith"


def test_gcd_card_routes():
    cards = compiled_cards()
    orch = CardOrchestrator(cards=cards)
    r = orch.route("compute gcd(12, 18)")
    assert r.answer == "6", f"expected 6, got {r.answer!r}"
    assert r.card == "gcd"


def test_factorial_card_routes():
    cards = compiled_cards()
    orch = CardOrchestrator(cards=cards)
    r = orch.route("factorial(6)")
    assert r.answer == "720", f"expected 720, got {r.answer!r}"
    assert r.card == "factorial"


def test_is_prime_card_routes():
    cards = compiled_cards()
    orch = CardOrchestrator(cards=cards)
    r = orch.route("is_prime(97)")
    assert r.answer == "True", f"expected True, got {r.answer!r}"


def test_no_match_returns_no_answer():
    """With only compiled cards, an open-ended query should return
    (answer=None, card=None)."""
    orch = CardOrchestrator(cards=compiled_cards())
    r = orch.route("explain what a closure is")
    assert r.answer is None
    assert r.card is None


# ----- Priority ordering -----

def test_priority_ordering():
    """Lower priority should be tried first."""
    @dataclass
    class _Always:
        name: str
        priority: int

        def applies_to(self, query, context): return True
        def invoke(self, query, context): return self.name

    orch = CardOrchestrator(cards=[
        _Always(name="third", priority=300),
        _Always(name="first", priority=100),
        _Always(name="second", priority=200),
    ])
    r = orch.route("anything")
    assert r.card == "first"
    assert r.tried == ["first"]  # short-circuited after first hit


def test_fall_through_on_none():
    """Card returning None falls through to next."""
    @dataclass
    class _Refuses:
        name: str = "refuser"
        priority: int = 100

        def applies_to(self, query, context): return True
        def invoke(self, query, context): return None

    @dataclass
    class _Answers:
        name: str = "answerer"
        priority: int = 200

        def applies_to(self, query, context): return True
        def invoke(self, query, context): return "answer"

    orch = CardOrchestrator(cards=[_Refuses(), _Answers()])
    r = orch.route("anything")
    assert r.card == "answerer"
    assert r.answer == "answer"
    assert r.tried == ["refuser", "answerer"]


# ----- Evaluator unit tests -----

def test_eval_binary_arith_handles_ops():
    m = re.search(r"(\b\d+\b)\s*([\+\-\*/%])\s*(\b\d+\b)", "compute 12 - 5 quickly")
    assert _eval_binary_arith(m) == "7"
    m = re.search(r"(\b\d+\b)\s*([\+\-\*/%])\s*(\b\d+\b)", "8 * 7 = ?")
    assert _eval_binary_arith(m) == "56"


def test_eval_func_one_arg_formats_bool():
    m = re.search(r"\b(is_prime)\(\s*(\d+)\s*\)", "is_prime(7)")
    assert _eval_func_one_arg(m) == "True"
    m = re.search(r"\b(is_prime)\(\s*(\d+)\s*\)", "is_prime(8)")
    assert _eval_func_one_arg(m) == "False"


def test_eval_func_two_args_gcd():
    m = re.search(r"\b(gcd)\(\s*(\d+)\s*,\s*(\d+)\s*\)", "gcd(42, 28)")
    assert _eval_func_two_args(m) == "14"


# ----- Build factory -----

def test_build_with_no_trained_no_gemma():
    """Disabling trained brain + Gemma yields only the exact tier."""
    chrlm = build(include_trained_brain=False, include_gemma_fallback=False)
    for card in chrlm.cards:
        assert card.priority < 300, (
            f"card {card.name} has priority {card.priority} — exact tier "
            f"should be <300"
        )
    # Should still answer arithmetic
    r = chrlm.route("what is 17 * 23")
    assert r.answer == "391"


def test_build_includes_gemma_card_when_requested():
    """Gemma card is present even if endpoint is offline (applies_to
    handles the health check at query time)."""
    chrlm = build(include_trained_brain=False, include_gemma_fallback=True)
    names = {c.name for c in chrlm.cards}
    assert "gemma_fallback" in names


def test_route_result_dataclass():
    r = RouteResult(answer="42", card="test", tried=["a", "b", "test"])
    assert r.answer == "42"
    assert r.card == "test"
    assert r.tried == ["a", "b", "test"]


if __name__ == "__main__":
    test_binary_arith_card_routes()
    print("[ok] binary_arith routes")
    test_gcd_card_routes()
    print("[ok] gcd routes")
    test_factorial_card_routes()
    print("[ok] factorial routes")
    test_is_prime_card_routes()
    print("[ok] is_prime routes")
    test_no_match_returns_no_answer()
    print("[ok] no match → no answer")
    test_priority_ordering()
    print("[ok] priority ordering")
    test_fall_through_on_none()
    print("[ok] None return falls through")
    test_eval_binary_arith_handles_ops()
    print("[ok] binary arith evaluator")
    test_eval_func_one_arg_formats_bool()
    print("[ok] is_prime evaluator formats bool")
    test_eval_func_two_args_gcd()
    print("[ok] gcd evaluator")
    test_build_with_no_trained_no_gemma()
    print("[ok] build(no trained, no gemma)")
    test_build_includes_gemma_card_when_requested()
    print("[ok] build includes gemma fallback card")
    test_route_result_dataclass()
    print("[ok] RouteResult dataclass")
