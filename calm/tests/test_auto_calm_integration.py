"""End-to-end feedback-loop integration test.

Mocks the LLM generation inside AutoCalmEngine so we can exercise the
complete loop without running Gemma:

  prompt → precompute + learned patterns → system prompt → (mock LLM) →
           verify → correct → auto_learn.learn_from_correction → next prompt
           of similar shape gets the precompute before generation

This is the strongest proof the loop CLOSES IN PRODUCTION short of
actually running against Gemma.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import pytest

from calm.auto_calm import AutoCalmEngine
from calm.auto_learn import AutoLearner


class _MockLLM:
    """Programmable mock — returns pre-queued (content, thinking, timings)
    on each `_generate` call. First queue element consumed first."""
    def __init__(self):
        self.queue: List[Tuple[str, str, dict]] = []
        self.calls: List[list] = []
        self.seen_system_prompts: List[str] = []

    def enqueue(self, content: str, thinking: str = "", tps: float = 10.0):
        self.queue.append((content, thinking, {"predicted_per_second": tps}))

    def __call__(self, messages):
        # Record system prompt for later assertions about precompute injection.
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        self.seen_system_prompts.append(sys_msg)
        self.calls.append(messages)
        if not self.queue:
            raise AssertionError("mock LLM queue empty — test didn't enqueue enough responses")
        return self.queue.pop(0)


@pytest.fixture
def engine_with_mock(tmp_path, monkeypatch):
    """AutoCalmEngine with _generate replaced by a mock, and AutoLearner
    pointed at a temp DB to isolate from the real learned_patterns.jsonl."""
    # Point AutoLearner's default DB at the tmp path so the engine picks
    # up our isolated instance.
    monkeypatch.setattr(
        "calm.auto_learn.DEFAULT_DB",
        tmp_path / "learned.jsonl",
    )
    # Also patch `precompute` to return empty so we isolate the learner's
    # contribution from the built-in NL-pattern precompute.
    monkeypatch.setattr("calm.auto_calm.precompute", lambda p: {})

    engine = AutoCalmEngine()
    mock = _MockLLM()
    engine._generate = mock
    return engine, mock


def test_loop_closes_in_auto_calm_engine(engine_with_mock):
    """Round 1: LLM says wrong thing → verifier corrects → learner records.
    Round 2: similar prompt → precompute injects before generation → LLM's
             system prompt contains the verified fact."""
    engine, mock = engine_with_mock

    # Round 1: prompt "what is 17 * 23?" — LLM says wrong answer.
    mock.enqueue("17 * 23 = 400. That's the product.", thinking="")
    # The retry after correction. Mock says correct answer this time.
    mock.enqueue("17 * 23 = 391. Corrected.", thinking="")

    r1 = engine.run("what is 17 * 23?")
    assert r1.claims_corrected >= 1
    # Verify the learner picked up the pattern.
    assert engine.learner.stats()["total"] >= 1
    assert "N * O" in engine.learner._patterns

    # Round 2: different multiplication prompt.
    # Mock returns correct answer (because it's seeing the precompute fact).
    mock.enqueue("347 * 289 = 100283. That's it.", thinking="")
    r2 = engine.run("what is 347 * 289?")

    # Assertion 1: Round 2's system prompt contains the precompute fact.
    round2_system = mock.seen_system_prompts[-1]
    assert "Verified facts" in round2_system, \
        f"expected precompute injection in system prompt, got:\n{round2_system[-500:]}"
    assert "347 * 289" in round2_system
    assert "100283" in round2_system

    # Assertion 2: Round 2 needed no correction (precompute worked).
    assert r2.claims_corrected == 0


def test_loop_shape_gate_prevents_noise(engine_with_mock):
    """Prove the shape-gate fix actually reduces pollution end-to-end —
    a multiplication pattern doesn't inject factorial precomputes on
    a multiplication prompt."""
    engine, mock = engine_with_mock

    # Pre-seed both a multiplication and a factorial pattern from prior runs.
    engine.learner.learn_from_correction(_MakeClaim("17 * 23"))
    engine.learner.learn_from_correction(_MakeClaim("factorial(5)"))
    assert "N * O" in engine.learner._patterns
    assert "factorial(N)" in engine.learner._patterns

    # Queue ONE response (no retry expected — shape-gated patterns mean
    # the precompute is focused).
    mock.enqueue("347 * 289 = 100283.", thinking="")
    engine.run("what is 347 * 289?")

    system = mock.seen_system_prompts[-1]
    # Extract just the injected-facts section (the base system prompt
    # mentions "factorial" in its capability listing; we only care about
    # what auto_learn's shape-gated matcher injected).
    if "Verified facts" in system:
        facts_section = system.split("Verified facts")[-1]
    else:
        facts_section = ""

    # Assertion: multiplication fact injected.
    assert "347 * 289" in facts_section, \
        f"expected '347 * 289' in facts, got:\n{facts_section}"
    # Assertion: factorial NOT injected into the facts section.
    assert "factorial" not in facts_section.lower(), \
        f"factorial leaked into facts despite shape gate:\n{facts_section}"


def test_verified_claim_does_not_learn(engine_with_mock):
    """If the LLM says the correct answer the first time, no learning
    should happen (no correction to learn from)."""
    engine, mock = engine_with_mock

    mock.enqueue("17 * 23 = 391.", thinking="")
    r = engine.run("what is 17 * 23?")

    assert r.claims_corrected == 0
    assert engine.learner.stats()["total"] == 0


# Helper ---------------------------------------------------------------


class _FakeClaim:
    """Minimal Claim duck for learn_from_correction."""
    def __init__(self, expression, correct=False, actual_value=0):
        self.expression = expression
        self.correct = correct
        self.actual_value = actual_value


def _MakeClaim(expr: str) -> _FakeClaim:
    return _FakeClaim(expr)
