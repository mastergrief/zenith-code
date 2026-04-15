"""Meta-runner — learns phase orderings from observed outcomes.

PhaseRunner executes a fixed ladder. MetaRunner observes outcomes and
suggests orderings: which phases have tried together, what regressed,
which pairings compose well vs. poorly.

MVP heuristic (NOT learned yet, just bookkeeping + suggestions):
  - Track every `(phase_name, phase_name)` transition's outcome
  - Score a transition high if phase N didn't regress phase N-1
  - Score low if phase N regressed something
  - Suggest next phase as the one with highest predicted success
    against all currently-passing phases

Full learning (RL policy over phase DAGs) is future work. MVP exposes
the INTERFACE so that smarter orderers can slot in.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable

from calm.llm_computer.phase_runner import PhaseResult


@dataclass
class TransitionStat:
    """Observed outcome of running phase `to_phase` AFTER `from_phase`."""
    from_phase: str
    to_phase: str
    n_attempts: int = 0
    n_succeeded: int = 0
    # Average regression on from_phase when to_phase ran after it
    # (1.0 = no regression, 0.0 = full forgetting)
    avg_retention: float = 1.0


@dataclass
class PhaseOutcome:
    """Summary of a single phase run for the meta-runner's log."""
    phase_name: str
    passed: bool
    gate_score: float
    regression_scores: dict[str, float] = field(default_factory=dict)


class MetaRunner:
    """Observes PhaseResults; maintains a transition-score table.

    Interface:
      - observe(prior_names, current_result) → updates stats
      - score_transition(from_name, to_name) → float in [0, 1]
      - suggest_next(candidates, already_passed) → ordered list of best
        next candidates ranked by predicted transition quality
    """

    def __init__(self):
        self._transitions: dict[tuple[str, str], TransitionStat] = {}
        self._outcomes: list[PhaseOutcome] = []

    def observe(
        self,
        prior_passed: list[str],
        result: PhaseResult,
    ) -> None:
        """Record outcome of `result.name` run after prior_passed.

        Updates transition stats for each (prior, result.name) pair.
        """
        outcome = PhaseOutcome(
            phase_name=result.name,
            passed=result.passed,
            gate_score=result.gate_score,
            regression_scores=dict(result.regression_scores),
        )
        self._outcomes.append(outcome)
        for prior in prior_passed:
            key = (prior, result.name)
            stat = self._transitions.setdefault(
                key, TransitionStat(from_phase=prior, to_phase=result.name),
            )
            stat.n_attempts += 1
            if result.passed:
                stat.n_succeeded += 1
            # Retention: how much of prior's ability survived
            retention = result.regression_scores.get(prior, 1.0)
            # Running average
            stat.avg_retention = (
                (stat.avg_retention * (stat.n_attempts - 1) + retention)
                / stat.n_attempts
            )

    def score_transition(self, from_name: str, to_name: str) -> float:
        """Score in [0, 1]: higher = historically safer transition.

        Unknown transitions return 0.5 (no prior information — neutral).
        Known transitions combine success rate and retention.
        """
        key = (from_name, to_name)
        stat = self._transitions.get(key)
        if stat is None or stat.n_attempts == 0:
            return 0.5
        success_rate = stat.n_succeeded / stat.n_attempts
        # Weight success 50%, retention 50%
        return 0.5 * success_rate + 0.5 * stat.avg_retention

    def predicted_success(
        self, candidate: str, currently_passed: Iterable[str],
    ) -> float:
        """Average transition score from every passed phase to candidate."""
        passed_list = list(currently_passed)
        if not passed_list:
            return 0.5  # no prior info
        scores = [self.score_transition(p, candidate) for p in passed_list]
        return sum(scores) / len(scores)

    def suggest_next(
        self,
        candidates: Iterable[str],
        currently_passed: Iterable[str],
    ) -> list[tuple[str, float]]:
        """Rank candidates by predicted success. Returns list of
        (name, score) sorted descending."""
        passed_list = list(currently_passed)
        scored = [
            (c, self.predicted_success(c, passed_list))
            for c in candidates
            if c not in passed_list
        ]
        return sorted(scored, key=lambda x: x[1], reverse=True)

    def transitions_seen(self) -> list[TransitionStat]:
        """All observed transitions, sorted by name for stability."""
        return sorted(
            self._transitions.values(),
            key=lambda s: (s.from_phase, s.to_phase),
        )

    def outcome_history(self) -> list[PhaseOutcome]:
        return list(self._outcomes)
