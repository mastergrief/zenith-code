"""
Auto-CALM Conversation State — persistent cognitive state across turns.

Wraps the cognitive modules that benefit from cross-turn memory:
consistency (track claims), calibration (learn accuracy), goals
(track user intent), and provenance (accumulate trust).

Single object that lives for the duration of a conversation.

Usage:
    from calm.conversation import ConversationState
    cs = ConversationState()
    # Turn 1
    cs.add_turn("What is the capital of France?", "The capital of France is Paris.")
    # Turn 2
    cs.add_turn("What about Germany?", "The capital of Germany is Berlin.")
    # Turn 3 — contradiction!
    cs.add_turn("Actually, is the capital of France Lyon?", "The capital of France is Lyon.")
    print(cs.contradictions)  # caught!
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict, Optional

from calm.consistency import ConsistencyTracker, Contradiction
from calm.calibration import ConfidenceCalibrator
from calm.goal_tracking import GoalTracker, Goal
from calm.provenance import ProvenanceTracker


@dataclass
class TurnRecord:
    """Record of a single conversation turn."""
    turn: int
    prompt: str
    response: str
    quality_score: float = 0.0
    claims_verified: int = 0
    claims_corrected: int = 0
    issues_found: int = 0


class ConversationState:
    """Persistent cognitive state across conversation turns."""

    def __init__(self):
        self._consistency = ConsistencyTracker()
        self._calibration = ConfidenceCalibrator(db_path=None)
        self._goals = GoalTracker()
        self._provenance = ProvenanceTracker()
        self._turns: List[TurnRecord] = []
        self._turn_count = 0

    def add_turn(self, prompt: str, response: str,
                 quality_score: float = 0.0,
                 claims_verified: int = 0,
                 claims_corrected: int = 0,
                 issues_found: int = 0) -> Dict:
        """Record a conversation turn and update all trackers.
        Returns dict of cross-turn insights."""
        self._turn_count += 1
        insights = {}

        # Track the turn
        self._turns.append(TurnRecord(
            turn=self._turn_count,
            prompt=prompt,
            response=response,
            quality_score=quality_score,
            claims_verified=claims_verified,
            claims_corrected=claims_corrected,
            issues_found=issues_found,
        ))

        # Consistency: check for contradictions
        contradictions = self._consistency.add_claims(response)
        if contradictions:
            insights["contradictions"] = [str(c) for c in contradictions]

        # Goals: track user intent
        new_goals = self._goals.add_user_message(prompt)
        self._goals.add_assistant_response(response)
        drift = self._goals.drift_check()
        if new_goals:
            insights["new_goals"] = [g.description for g in new_goals]
        if "DRIFT" in drift:
            insights["drift"] = drift

        # Calibration: update from verification results
        if claims_verified > 0 or claims_corrected > 0:
            domain = self._calibration.detect_domain(prompt)
            self._calibration._stats[domain].correct += claims_verified
            self._calibration._stats[domain].incorrect += claims_corrected

        # Provenance: track what was verified
        self._provenance.tag_from_autocalm(
            precomputed={},
            claims_verified=claims_verified,
            claims_corrected=claims_corrected,
            response=response,
        )

        # Quality trend
        if len(self._turns) >= 3:
            recent = [t.quality_score for t in self._turns[-3:] if t.quality_score > 0]
            if len(recent) >= 2:
                trend = recent[-1] - recent[0]
                if trend < -0.1:
                    insights["quality_trend"] = f"declining ({recent[0]:.0%} → {recent[-1]:.0%})"

        return insights

    @property
    def contradictions(self) -> List[Contradiction]:
        return self._consistency.all_contradictions

    @property
    def active_goals(self) -> List[Goal]:
        return self._goals.active_goals

    @property
    def turn_count(self) -> int:
        return self._turn_count

    def confidence_for(self, text: str) -> float:
        """Get calibrated confidence for a domain."""
        return self._calibration.assess(text).confidence

    def goal_drift(self) -> str:
        return self._goals.drift_check()

    def summary(self) -> str:
        """Full conversation state summary."""
        parts = [f"{self._turn_count} turns"]

        # Consistency
        contras = len(self._consistency.all_contradictions)
        if contras:
            parts.append(f"{contras} contradictions!")

        # Goals
        active = len(self._goals.active_goals)
        if active:
            parts.append(f"{active} active goals")

        # Quality trend
        scores = [t.quality_score for t in self._turns if t.quality_score > 0]
        if scores:
            parts.append(f"avg quality {sum(scores)/len(scores):.0%}")

        # Provenance
        report = self._provenance.report()
        if report.tags:
            parts.append(f"trust {report.overall_trust:.0%}")

        return ", ".join(parts)
