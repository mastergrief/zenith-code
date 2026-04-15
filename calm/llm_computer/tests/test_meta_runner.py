"""Tests for meta-runner phase-ordering heuristics."""

from __future__ import annotations

from calm.llm_computer.meta_runner import MetaRunner, PhaseOutcome, TransitionStat
from calm.llm_computer.phase_runner import PhaseResult


def _result(name, passed, gate, regressions=None):
    return PhaseResult(
        phase_id=0, name=name, passed=passed, gate_score=gate,
        min_threshold=0.75, train_wallclock_s=0.0,
        regression_scores=regressions or {}, final_loss=0.0,
    )


def test_observe_records_transition():
    mr = MetaRunner()
    mr.observe(
        prior_passed=["adder"],
        result=_result("echo_a", passed=True, gate=1.0,
                      regressions={"adder": 1.0}),
    )
    stats = mr.transitions_seen()
    assert len(stats) == 1
    s = stats[0]
    assert s.from_phase == "adder"
    assert s.to_phase == "echo_a"
    assert s.n_attempts == 1
    assert s.n_succeeded == 1
    assert s.avg_retention == 1.0


def test_unknown_transition_scores_neutral():
    mr = MetaRunner()
    assert mr.score_transition("never_seen", "other") == 0.5


def test_successful_transition_scores_high():
    mr = MetaRunner()
    mr.observe(["a"], _result("b", True, 0.9, {"a": 1.0}))
    mr.observe(["a"], _result("b", True, 0.9, {"a": 1.0}))
    mr.observe(["a"], _result("b", True, 0.9, {"a": 1.0}))
    score = mr.score_transition("a", "b")
    assert score == 1.0  # 100% success + 100% retention


def test_failed_transition_scores_low():
    mr = MetaRunner()
    mr.observe(["a"], _result("bad", False, 0.3, {"a": 0.2}))  # regression
    mr.observe(["a"], _result("bad", False, 0.3, {"a": 0.2}))
    score = mr.score_transition("a", "bad")
    assert score < 0.5  # low success + low retention


def test_mixed_transition_scores_middle():
    mr = MetaRunner()
    mr.observe(["a"], _result("b", True, 0.9, {"a": 1.0}))
    mr.observe(["a"], _result("b", False, 0.3, {"a": 0.5}))
    score = mr.score_transition("a", "b")
    # success 50%, retention avg 75% → 0.5 * 0.5 + 0.5 * 0.75 = 0.625
    assert 0.55 < score < 0.70


def test_suggest_next_ranks_candidates():
    mr = MetaRunner()
    # 'good' has perfect record after 'a'
    mr.observe(["a"], _result("good", True, 0.95, {"a": 1.0}))
    # 'bad' always regresses 'a'
    mr.observe(["a"], _result("bad", False, 0.3, {"a": 0.1}))
    # 'unknown' never tried
    suggestions = mr.suggest_next(
        candidates=["good", "bad", "unknown"],
        currently_passed=["a"],
    )
    # Good should rank first, bad last
    names = [n for n, _ in suggestions]
    assert names[0] == "good"
    assert names[-1] == "bad"


def test_suggest_next_excludes_already_passed():
    mr = MetaRunner()
    suggestions = mr.suggest_next(
        candidates=["a", "b", "c"],
        currently_passed=["a"],
    )
    assert "a" not in [n for n, _ in suggestions]


def test_predicted_success_averages_across_prior():
    mr = MetaRunner()
    mr.observe(["a"], _result("c", True, 1.0, {"a": 1.0}))
    mr.observe(["b"], _result("c", True, 1.0, {"b": 1.0}))
    # Both "a → c" and "b → c" are perfect; predicted = 1.0
    assert mr.predicted_success("c", ["a", "b"]) == 1.0


def test_outcome_history_preserves_order():
    mr = MetaRunner()
    mr.observe([], _result("phase0", True, 1.0))
    mr.observe(["phase0"], _result("phase1", True, 0.9))
    mr.observe(["phase0", "phase1"], _result("phase2", False, 0.3,
                                              {"phase0": 1.0, "phase1": 0.5}))
    h = mr.outcome_history()
    assert [o.phase_name for o in h] == ["phase0", "phase1", "phase2"]
    assert not h[-1].passed


def test_running_average_of_retention():
    mr = MetaRunner()
    # First attempt: retention 1.0
    mr.observe(["a"], _result("b", True, 1.0, {"a": 1.0}))
    stat = mr.transitions_seen()[0]
    assert stat.avg_retention == 1.0
    # Second attempt: retention 0.5 → running avg 0.75
    mr.observe(["a"], _result("b", True, 0.8, {"a": 0.5}))
    stat = mr.transitions_seen()[0]
    assert stat.avg_retention == 0.75


if __name__ == "__main__":
    test_observe_records_transition()
    print("[ok] observe records transition")
    test_unknown_transition_scores_neutral()
    print("[ok] unknown = neutral score 0.5")
    test_successful_transition_scores_high()
    print("[ok] perfect history scores 1.0")
    test_failed_transition_scores_low()
    print("[ok] regression history scores low")
    test_mixed_transition_scores_middle()
    print("[ok] mixed history scores middle")
    test_suggest_next_ranks_candidates()
    print("[ok] suggest_next ranks by score")
    test_suggest_next_excludes_already_passed()
    print("[ok] suggest_next excludes already-passed")
    test_predicted_success_averages_across_prior()
    print("[ok] predicted_success averages prior")
    test_outcome_history_preserves_order()
    print("[ok] outcome history preserves order")
    test_running_average_of_retention()
    print("[ok] retention averages across attempts")
