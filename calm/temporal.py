"""
Auto-CALM Temporal Reasoning — verify ordering, detect impossible sequences.

Models mess up "before/after", can't sequence events reliably, and
produce impossible timelines. This module tracks temporal claims and
validates ordering consistency.

Usage:
    from calm.temporal import TemporalReasoner
    tr = TemporalReasoner()
    tr.add("A happens before B")
    tr.add("B happens before C")
    issues = tr.add("C happens before A")  # cycle!
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import List, Set, Dict, Optional, Tuple


@dataclass
class TemporalClaim:
    """A temporal ordering claim."""
    event_a: str
    event_b: str
    relation: str    # "before", "after", "during", "simultaneous"
    original: str


@dataclass
class TemporalIssue:
    """A detected temporal problem."""
    issue_type: str   # "cycle", "contradiction", "impossible"
    description: str
    events: List[str] = field(default_factory=list)


_TEMPORAL_PATTERNS = [
    (re.compile(r'(\w[\w\s]{2,30}?)\s+(?:happens?|occurs?|comes?)\s+before\s+(\w[\w\s]{2,30})', re.IGNORECASE), "before"),
    (re.compile(r'(\w[\w\s]{2,30}?)\s+(?:happens?|occurs?|comes?)\s+after\s+(\w[\w\s]{2,30})', re.IGNORECASE), "after"),
    (re.compile(r'(?:first|before)\s+(\w[\w\s]{2,30}?)\s*,\s*(?:then|after)\s+(\w[\w\s]{2,30})', re.IGNORECASE), "before"),
    (re.compile(r'(\w[\w\s]{2,30}?)\s+(?:precedes?|leads?\s+to)\s+(\w[\w\s]{2,30})', re.IGNORECASE), "before"),
    (re.compile(r'(\w[\w\s]{2,30}?)\s+(?:follows?|results?\s+from)\s+(\w[\w\s]{2,30})', re.IGNORECASE), "after"),
    (re.compile(r'(\w[\w\s]{2,30}?)\s+(?:during|while|simultaneously)\s+(\w[\w\s]{2,30})', re.IGNORECASE), "during"),
    # Step numbering: "Step 1: X ... Step 2: Y" implies X before Y
    (re.compile(r'(?:step|phase|stage)\s+(\d+).*?(?:step|phase|stage)\s+(\d+)', re.IGNORECASE | re.DOTALL), "step_order"),
]


class TemporalReasoner:
    """Tracks and verifies temporal ordering claims."""

    def __init__(self):
        # Directed graph: a → b means "a happens before b"
        self._before: Dict[str, Set[str]] = defaultdict(set)
        self._claims: List[TemporalClaim] = []

    def _normalize(self, event: str) -> str:
        return event.strip().lower()

    def add(self, text: str) -> List[TemporalIssue]:
        """Extract temporal claims from text and check for issues."""
        issues = []

        for pat, relation in _TEMPORAL_PATTERNS:
            for m in pat.finditer(text):
                if relation == "step_order":
                    a = f"step {m.group(1)}"
                    b = f"step {m.group(2)}"
                    if int(m.group(1)) < int(m.group(2)):
                        relation = "before"
                    else:
                        relation = "after"
                else:
                    a = self._normalize(m.group(1))
                    b = self._normalize(m.group(2))

                claim = TemporalClaim(
                    event_a=a, event_b=b,
                    relation=relation, original=m.group(0),
                )
                self._claims.append(claim)

                # Record ordering
                if relation == "before":
                    first, second = a, b
                elif relation == "after":
                    first, second = b, a
                elif relation == "during" or relation == "simultaneous":
                    continue  # Don't add to ordering graph
                else:
                    continue

                # Check for cycle before adding
                if self._would_create_cycle(first, second):
                    cycle_path = self._find_path(second, first)
                    issues.append(TemporalIssue(
                        issue_type="cycle",
                        description=(f"Temporal cycle: '{first}' before '{second}' "
                                    f"contradicts existing ordering: "
                                    f"{' → '.join(cycle_path + [first])}"),
                        events=[first, second] + cycle_path,
                    ))
                else:
                    self._before[first].add(second)

        return issues

    def _would_create_cycle(self, first: str, second: str) -> bool:
        """Check if adding first→second would create a cycle."""
        # If there's already a path from second to first, adding first→second creates a cycle
        return self._has_path(second, first)

    def _has_path(self, start: str, end: str) -> bool:
        """BFS: is there a path from start to end?"""
        visited = set()
        queue = deque([start])
        while queue:
            node = queue.popleft()
            if node == end:
                return True
            if node in visited:
                continue
            visited.add(node)
            for next_node in self._before.get(node, set()):
                queue.append(next_node)
        return False

    def _find_path(self, start: str, end: str) -> List[str]:
        """Find the path from start to end (for error reporting)."""
        visited = set()
        queue = deque([(start, [start])])
        while queue:
            node, path = queue.popleft()
            if node == end:
                return path
            if node in visited:
                continue
            visited.add(node)
            for next_node in self._before.get(node, set()):
                queue.append((next_node, path + [next_node]))
        return []

    def get_order(self) -> List[str]:
        """Topological sort of all known events."""
        # Kahn's algorithm
        in_degree = defaultdict(int)
        all_nodes = set()
        for node, successors in self._before.items():
            all_nodes.add(node)
            for s in successors:
                in_degree[s] += 1
                all_nodes.add(s)

        queue = deque([n for n in all_nodes if in_degree[n] == 0])
        result = []
        while queue:
            node = queue.popleft()
            result.append(node)
            for successor in self._before.get(node, set()):
                in_degree[successor] -= 1
                if in_degree[successor] == 0:
                    queue.append(successor)

        return result

    def verify_sequence(self, events: List[str]) -> List[TemporalIssue]:
        """Verify that a proposed sequence is consistent with known ordering."""
        issues = []
        for i in range(len(events)):
            for j in range(i + 1, len(events)):
                a = self._normalize(events[i])
                b = self._normalize(events[j])
                # Check if b is known to come before a
                if self._has_path(b, a):
                    issues.append(TemporalIssue(
                        issue_type="contradiction",
                        description=f"'{events[j]}' should come before '{events[i]}' based on known ordering",
                        events=[events[i], events[j]],
                    ))
        return issues

    def summary(self) -> str:
        order = self.get_order()
        if order:
            return f"Temporal ordering ({len(order)} events): {' → '.join(order)}"
        return "No temporal claims tracked"
