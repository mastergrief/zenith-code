"""
Auto-CALM Goal Tracking — track what the user wants across turns.

Detects the user's goals from their messages, tracks progress toward
each goal, and detects goal drift (when the conversation moves away
from what the user originally asked).

Usage:
    from calm.goal_tracking import GoalTracker
    gt = GoalTracker()
    gt.add_user_message("Help me optimize this SQL query that's timing out")
    gt.add_assistant_response("Let me look at the query plan...")
    print(gt.active_goals)
    print(gt.drift_check())
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Goal:
    """A tracked user goal."""
    description: str
    category: str          # "fix", "build", "understand", "optimize", "decide", "explore"
    turn_introduced: int
    status: str = "active"  # "active", "achieved", "abandoned", "blocked"
    progress_notes: List[str] = field(default_factory=list)
    relevance_score: float = 1.0  # decays if not referenced


@dataclass
class GoalTrackingResult:
    """Current state of goal tracking."""
    active_goals: List[Goal] = field(default_factory=list)
    achieved_goals: List[Goal] = field(default_factory=list)
    drift_detected: bool = False
    drift_description: str = ""

    def summary(self) -> str:
        lines = [f"Goals: {len(self.active_goals)} active, {len(self.achieved_goals)} achieved"]
        for g in self.active_goals:
            lines.append(f"  [{g.status}] {g.description} (relevance: {g.relevance_score:.0%})")
        if self.drift_detected:
            lines.append(f"  DRIFT: {self.drift_description}")
        return "\n".join(lines)


# Patterns for extracting goals from user messages
_GOAL_PATTERNS = [
    (re.compile(r'(?:help me|I need to|I want to|can you)\s+(.{10,80}?)(?:\.|$|\?)', re.IGNORECASE), None),
    (re.compile(r'(?:fix|debug|solve|resolve)\s+(.{5,60})', re.IGNORECASE), "fix"),
    (re.compile(r'(?:build|create|implement|add|develop)\s+(.{5,60})', re.IGNORECASE), "build"),
    (re.compile(r'(?:explain|understand|how does|what is|why does)\s+(.{5,60})', re.IGNORECASE), "understand"),
    (re.compile(r'(?:optimize|speed up|improve|make faster|reduce)\s+(.{5,60})', re.IGNORECASE), "optimize"),
    (re.compile(r'(?:should I|which is better|compare|decide|choose)\s+(.{5,60})', re.IGNORECASE), "decide"),
    (re.compile(r'(?:explore|investigate|look into|research|find out)\s+(.{5,60})', re.IGNORECASE), "explore"),
]

# Achievement signals in assistant responses
_ACHIEVEMENT_PATTERNS = [
    re.compile(r'\b(?:fixed|resolved|solved|implemented|completed|done|working now)\b', re.IGNORECASE),
    re.compile(r'\b(?:here.s the (?:solution|fix|implementation)|that should (?:fix|solve))\b', re.IGNORECASE),
]


class GoalTracker:
    """Tracks user goals across conversation turns."""

    def __init__(self):
        self._goals: List[Goal] = []
        self._turn: int = 0
        self._last_relevant_turn: int = 0

    def add_user_message(self, message: str) -> List[Goal]:
        """Extract and track goals from a user message."""
        self._turn += 1
        new_goals = []

        for pat, category in _GOAL_PATTERNS:
            for m in pat.finditer(message):
                desc = m.group(1).strip().rstrip('.,;:?!')
                if len(desc) < 5:
                    continue

                # Detect category if not specified
                if category is None:
                    category = self._detect_category(desc)

                # Check if this is a new goal or revisiting an existing one
                existing = self._find_similar_goal(desc)
                if existing:
                    existing.relevance_score = 1.0
                    self._last_relevant_turn = self._turn
                    continue

                goal = Goal(
                    description=desc,
                    category=category or "explore",
                    turn_introduced=self._turn,
                )
                self._goals.append(goal)
                new_goals.append(goal)
                self._last_relevant_turn = self._turn

        # Decay relevance of unreferenced goals
        for g in self._goals:
            if g.status == "active" and g.turn_introduced < self._turn:
                g.relevance_score *= 0.9

        return new_goals

    def add_assistant_response(self, response: str):
        """Check if any goals were achieved in the response."""
        for pat in _ACHIEVEMENT_PATTERNS:
            if pat.search(response):
                # Find the most relevant active goal
                active = [g for g in self._goals if g.status == "active"]
                if active:
                    # Most recent active goal is most likely the one achieved
                    best = max(active, key=lambda g: g.relevance_score)
                    best.progress_notes.append(f"Turn {self._turn}: achievement signal detected")

    def _detect_category(self, desc: str) -> str:
        """Detect goal category from description."""
        desc_lower = desc.lower()
        if any(w in desc_lower for w in ["fix", "bug", "error", "broken", "crash"]):
            return "fix"
        if any(w in desc_lower for w in ["build", "create", "add", "implement"]):
            return "build"
        if any(w in desc_lower for w in ["understand", "explain", "how", "why"]):
            return "understand"
        if any(w in desc_lower for w in ["optimize", "faster", "slow", "improve"]):
            return "optimize"
        if any(w in desc_lower for w in ["should", "better", "compare", "choose"]):
            return "decide"
        return "explore"

    def _find_similar_goal(self, desc: str) -> Optional[Goal]:
        """Find an existing goal similar to the description."""
        desc_words = set(desc.lower().split())
        for goal in self._goals:
            if goal.status != "active":
                continue
            goal_words = set(goal.description.lower().split())
            # If >50% word overlap, consider it the same goal
            overlap = len(desc_words & goal_words)
            total = len(desc_words | goal_words)
            if total > 0 and overlap / total > 0.4:
                return goal
        return None

    def drift_check(self) -> str:
        """Check if the conversation has drifted from the original goals."""
        if not self._goals:
            return "No goals tracked yet"

        active = [g for g in self._goals if g.status == "active"]
        if not active:
            return "All goals achieved"

        turns_since_relevant = self._turn - self._last_relevant_turn
        if turns_since_relevant >= 3:
            original = active[0].description if active else "unknown"
            return (f"DRIFT: {turns_since_relevant} turns since goals were last referenced. "
                    f"Original goal: '{original}'")

        low_relevance = [g for g in active if g.relevance_score < 0.5]
        if low_relevance:
            return (f"WARNING: {len(low_relevance)} goals losing relevance: "
                    f"{', '.join(g.description[:30] for g in low_relevance)}")

        return "On track"

    @property
    def active_goals(self) -> List[Goal]:
        return [g for g in self._goals if g.status == "active"]

    @property
    def state(self) -> GoalTrackingResult:
        return GoalTrackingResult(
            active_goals=[g for g in self._goals if g.status == "active"],
            achieved_goals=[g for g in self._goals if g.status == "achieved"],
            drift_detected=self._turn - self._last_relevant_turn >= 3,
            drift_description=self.drift_check(),
        )
