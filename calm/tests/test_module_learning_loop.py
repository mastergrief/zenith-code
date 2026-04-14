"""Closed-loop tests for the cognitive-module learner.

Parallel to test_auto_learn_loop.py but for ModuleLearner:
  module reports issue → record → future prompts with same context
  get the prevention injected as a system-prompt addition.

Same Vector 1 goal: prove the loop closes and document its behavior.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List

import pytest

from calm.module_learning import IssueTrend, ModuleLearner


@dataclass
class _FakeModuleResult:
    module_name: str
    issues_found: int
    summary: str


@dataclass
class _FakeReport:
    results: List[_FakeModuleResult]


@pytest.fixture
def tmp_db(tmp_path) -> Path:
    return tmp_path / "module_learning.json"


def test_record_creates_trend(tmp_db):
    ml = ModuleLearner(db_path=tmp_db)
    ml.record("scope", "overgeneralization", "comparison")
    key = "scope:overgeneralization:comparison"
    assert key in ml._trends
    assert ml._trends[key].frequency == 1


def test_record_increments_frequency(tmp_db):
    ml = ModuleLearner(db_path=tmp_db)
    for _ in range(5):
        ml.record("scope", "overgeneralization", "comparison")
    key = "scope:overgeneralization:comparison"
    assert ml._trends[key].frequency == 5


def test_prevention_generated_on_first_record(tmp_db):
    """Prevention text should be populated from the preventions map."""
    ml = ModuleLearner(db_path=tmp_db)
    ml.record("scope", "overgeneralization", "comparison")
    key = "scope:overgeneralization:comparison"
    assert ml._trends[key].prevention != ""
    # Should mention "absolute" — that's the scope/overgeneralization advice.
    assert "absolute" in ml._trends[key].prevention.lower()


def test_suggest_requires_3_occurrences(tmp_db):
    """Under 3 records → no suggestions (below the recurring threshold)."""
    ml = ModuleLearner(db_path=tmp_db)
    ml.record("scope", "overgeneralization", "comparison")
    ml.record("scope", "overgeneralization", "comparison")
    additions = ml.suggest_prompt_additions("compare Python and Rust")
    assert additions == []


def test_loop_closes_after_3_occurrences(tmp_db):
    """THE LOOP TEST: 3 similar issues → prevention fires on next
    matching prompt. This is the cognitive-module analog of
    auto_learn's close-the-loop test."""
    ml = ModuleLearner(db_path=tmp_db)
    for _ in range(3):
        ml.record("scope", "overgeneralization", "comparison")

    additions = ml.suggest_prompt_additions("compare Python and Rust")
    assert len(additions) == 1
    assert "absolute" in additions[0].lower()


def test_context_detection_classifies_prompts(tmp_db):
    ml = ModuleLearner(db_path=tmp_db)
    assert ml._detect_context("compare X and Y") == "comparison"
    assert ml._detect_context("debug this error") == "debugging"
    assert ml._detect_context("explain recursion") == "explanation"
    assert ml._detect_context("design a system") == "design"
    assert ml._detect_context("deploy to prod") == "operations"
    assert ml._detect_context("write me a story") == "general"


def test_suggestion_context_matches(tmp_db):
    """Preventions are context-scoped: comparison-context patterns don't
    fire on debugging prompts."""
    ml = ModuleLearner(db_path=tmp_db)
    for _ in range(3):
        ml.record("scope", "overgeneralization", "comparison")

    # Comparison prompt → fires.
    assert ml.suggest_prompt_additions("compare A and B") != []
    # Debugging prompt → does NOT fire (different context).
    assert ml.suggest_prompt_additions("debug this crash") == []


def test_general_context_fires_everywhere(tmp_db):
    """Patterns recorded as context='general' fire for any prompt."""
    ml = ModuleLearner(db_path=tmp_db)
    for _ in range(3):
        ml.record("density", "filler", "general")

    # Any prompt should get the general prevention.
    assert ml.suggest_prompt_additions("anything at all") != []
    assert ml.suggest_prompt_additions("debug this") != []


def test_persistence_roundtrip(tmp_db):
    """State survives a save/load cycle."""
    l1 = ModuleLearner(db_path=tmp_db)
    for _ in range(3):
        l1.record("scope", "overgeneralization", "comparison")
    del l1

    l2 = ModuleLearner(db_path=tmp_db)
    key = "scope:overgeneralization:comparison"
    assert key in l2._trends
    assert l2._trends[key].frequency == 3


def test_record_from_report(tmp_db):
    """CognitiveReport-shaped input records per-module per-issue."""
    ml = ModuleLearner(db_path=tmp_db)
    report = _FakeReport(results=[
        _FakeModuleResult("scope", 1, "overgeneralization in 2 places"),
        _FakeModuleResult("precision", 1, "vague: 3 terms"),
        _FakeModuleResult("relevance", 0, "well-scoped (95%)"),  # 0 issues → skip
    ])
    ml.record_from_report(report, "compare things")
    # Two records should have landed — 'scope' and 'precision'.
    n_trends = sum(1 for t in ml._trends.values() if t.frequency > 0)
    assert n_trends == 2


def test_recurring_issues_property(tmp_db):
    """recurring_issues returns only patterns with frequency >= 3."""
    ml = ModuleLearner(db_path=tmp_db)
    for _ in range(5):
        ml.record("scope", "overgeneralization", "comparison")
    for _ in range(2):
        ml.record("precision", "vague", "general")

    recurring = ml.recurring_issues
    assert len(recurring) == 1
    assert recurring[0].module == "scope"


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        db = Path(tmp) / "m.json"
        test_loop_closes_after_3_occurrences(db)
        print("[ok] module_learning loop closes after 3 occurrences")
