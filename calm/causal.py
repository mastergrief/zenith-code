"""
Auto-CALM Causal Reasoning — dependency tracing and impact prediction.

Given "if I change X, what breaks?", builds a dependency graph and
traces forward to predict effects. Also detects when the model
confuses correlation with causation.

Two capabilities:
  1. Dependency graph: track what depends on what
  2. Impact trace: given a change, predict downstream effects

Usage:
    from calm.causal import CausalEngine
    ce = CausalEngine()
    ce.add_dependency("auth_middleware", "user_session")
    ce.add_dependency("user_session", "database")
    impact = ce.trace_impact("database")
    # → ["user_session", "auth_middleware"]
"""

from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple


@dataclass
class CausalLink:
    """A causal relationship: cause → effect."""
    cause: str
    effect: str
    relationship: str = "depends_on"  # depends_on, causes, enables, blocks
    confidence: float = 1.0
    source: str = ""  # where this was established


@dataclass
class ImpactResult:
    """Result of tracing the impact of a change."""
    changed: str
    directly_affected: List[str] = field(default_factory=list)
    transitively_affected: List[str] = field(default_factory=list)
    total_affected: int = 0
    max_depth: int = 0
    critical_path: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.total_affected == 0:
            return f"changing '{self.changed}' has no tracked downstream effects"
        return (f"changing '{self.changed}' affects {self.total_affected} components "
                f"(depth {self.max_depth}): {' → '.join(self.critical_path)}")


# Patterns for extracting causal relationships from text
_CAUSAL_PATTERNS = [
    # "X depends on Y"
    (re.compile(r'(\w[\w\s]*?)\s+depends?\s+on\s+(\w[\w\s]*)', re.IGNORECASE), "depends_on"),
    # "X requires Y"
    (re.compile(r'(\w[\w\s]*?)\s+requires?\s+(\w[\w\s]*)', re.IGNORECASE), "depends_on"),
    # "X uses Y"
    (re.compile(r'(\w[\w\s]*?)\s+uses?\s+(\w[\w\s]*)', re.IGNORECASE), "depends_on"),
    # "X calls Y"
    (re.compile(r'(\w[\w\s]*?)\s+calls?\s+(\w[\w\s]*)', re.IGNORECASE), "depends_on"),
    # "X causes Y"
    (re.compile(r'(\w[\w\s]*?)\s+causes?\s+(\w[\w\s]*)', re.IGNORECASE), "causes"),
    # "X leads to Y"
    (re.compile(r'(\w[\w\s]*?)\s+leads?\s+to\s+(\w[\w\s]*)', re.IGNORECASE), "causes"),
    # "if X then Y" / "when X, Y"
    (re.compile(r'(?:if|when)\s+(\w[\w\s]*?)\s*,\s*(?:then\s+)?(\w[\w\s]*)', re.IGNORECASE), "causes"),
    # "Y because X"
    (re.compile(r'(\w[\w\s]*?)\s+because\s+(\w[\w\s]*)', re.IGNORECASE), "caused_by"),
    # "X blocks Y"
    (re.compile(r'(\w[\w\s]*?)\s+blocks?\s+(\w[\w\s]*)', re.IGNORECASE), "blocks"),
    # "X enables Y"
    (re.compile(r'(\w[\w\s]*?)\s+enables?\s+(\w[\w\s]*)', re.IGNORECASE), "enables"),
]

# Correlation-vs-causation red flags
_CORRELATION_PATTERNS = [
    re.compile(r'(\w+)\s+(?:correlates?|is correlated)\s+with\s+(\w+)', re.IGNORECASE),
    re.compile(r'(?:as|when)\s+(\w+)\s+(?:increases?|decreases?)\s*,\s*(\w+)\s+(?:also\s+)?(?:increases?|decreases?)', re.IGNORECASE),
]


class CausalEngine:
    """Dependency graph with forward/backward impact tracing."""

    def __init__(self):
        # Adjacency lists: forward (cause → effects) and backward (effect → causes)
        self._forward: Dict[str, Set[str]] = defaultdict(set)
        self._backward: Dict[str, Set[str]] = defaultdict(set)
        self._links: List[CausalLink] = []
        self._nodes: Set[str] = set()

    def _normalize(self, name: str) -> str:
        s = name.strip().lower()
        s = re.sub(r'^(?:the|a|an)\s+', '', s)
        return s

    def add_dependency(self, dependent: str, dependency: str,
                       relationship: str = "depends_on"):
        """Record that `dependent` depends on `dependency`."""
        dep = self._normalize(dependent)
        src = self._normalize(dependency)
        self._forward[src].add(dep)
        self._backward[dep].add(src)
        self._nodes.add(dep)
        self._nodes.add(src)
        self._links.append(CausalLink(cause=src, effect=dep, relationship=relationship))

    def add_from_text(self, text: str) -> int:
        """Extract causal relationships from text. Returns count added."""
        count = 0
        for pat, rel in _CAUSAL_PATTERNS:
            for m in pat.finditer(text):
                a = m.group(1).strip()
                b = m.group(2).strip()
                if len(a) < 2 or len(b) < 2:
                    continue
                if rel == "caused_by":
                    self.add_dependency(a, b, "depends_on")
                elif rel in ("causes", "enables"):
                    self.add_dependency(b, a, rel)
                else:
                    self.add_dependency(a, b, rel)
                count += 1
        return count

    def trace_impact(self, changed: str) -> ImpactResult:
        """Trace forward: if `changed` breaks, what else breaks?"""
        changed = self._normalize(changed)
        result = ImpactResult(changed=changed)

        if changed not in self._nodes:
            return result

        # BFS forward through the dependency graph
        visited = set()
        queue = deque([(changed, 0)])
        visited.add(changed)
        depths = {}

        while queue:
            node, depth = queue.popleft()
            for dependent in self._forward.get(node, set()):
                if dependent not in visited:
                    visited.add(dependent)
                    depths[dependent] = depth + 1
                    if depth == 0:
                        result.directly_affected.append(dependent)
                    else:
                        result.transitively_affected.append(dependent)
                    queue.append((dependent, depth + 1))

        result.total_affected = len(result.directly_affected) + len(result.transitively_affected)
        result.max_depth = max(depths.values()) if depths else 0

        # Find the critical path (longest dependency chain)
        if depths:
            path = [changed]
            current = changed
            while True:
                children = [n for n in self._forward.get(current, set()) if n in depths]
                if not children:
                    break
                # Follow the deepest child
                next_node = max(children, key=lambda n: depths[n])
                path.append(next_node)
                current = next_node
            result.critical_path = path

        return result

    def trace_root_cause(self, symptom: str) -> List[str]:
        """Trace backward: what could have caused this symptom?"""
        symptom = self._normalize(symptom)
        if symptom not in self._nodes:
            return []

        # BFS backward
        visited = set()
        queue = deque([symptom])
        visited.add(symptom)
        root_causes = []

        while queue:
            node = queue.popleft()
            parents = self._backward.get(node, set())
            if not parents:
                if node != symptom:
                    root_causes.append(node)
            for parent in parents:
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)

        return root_causes

    def detect_cycles(self) -> List[List[str]]:
        """Detect circular dependencies."""
        cycles = []
        visited = set()
        rec_stack = set()

        def _dfs(node, path):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in self._forward.get(node, set()):
                if neighbor not in visited:
                    _dfs(neighbor, path)
                elif neighbor in rec_stack:
                    # Found a cycle
                    cycle_start = path.index(neighbor)
                    cycles.append(path[cycle_start:] + [neighbor])

            path.pop()
            rec_stack.discard(node)

        for node in self._nodes:
            if node not in visited:
                _dfs(node, [])

        return cycles

    def detect_correlation_claims(self, text: str) -> List[str]:
        """Flag claims that look like correlation dressed as causation."""
        warnings = []
        for pat in _CORRELATION_PATTERNS:
            for m in pat.finditer(text):
                warnings.append(
                    f"Correlation ≠ causation: '{m.group(0)}' — "
                    f"does {m.group(1)} actually CAUSE {m.group(2)}, or just co-occur?"
                )
        return warnings

    @property
    def node_count(self) -> int:
        return len(self._nodes)

    @property
    def edge_count(self) -> int:
        return sum(len(deps) for deps in self._forward.values())

    def summary(self) -> str:
        return f"{self.node_count} components, {self.edge_count} dependencies"
