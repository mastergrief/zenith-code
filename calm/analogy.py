"""
Auto-CALM Analogy Verification — structural mapping validation.

When the model says "X is like Y", verify that the relevant properties
actually map. Bad analogies are one of the most persuasive failure modes
— they sound insightful but break down on inspection.

Two checks:
  1. Structural: do the relevant properties of X exist in Y?
  2. Scope: does the analogy hold for the specific aspect being discussed?

Usage:
    from calm.analogy import AnalogyVerifier
    av = AnalogyVerifier()
    result = av.check("A database index is like a book's table of contents")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict, Set, Optional


@dataclass
class AnalogyMapping:
    """A mapping between source and target domains."""
    source: str          # "book's table of contents"
    target: str          # "database index"
    shared_properties: List[str] = field(default_factory=list)
    source_only: List[str] = field(default_factory=list)   # properties unique to source
    target_only: List[str] = field(default_factory=list)   # properties unique to target
    strength: float = 0.0  # 0-1, proportion of shared properties


@dataclass
class AnalogyResult:
    """Result of analogy verification."""
    original: str
    source: str
    target: str
    mapping: Optional[AnalogyMapping] = None
    valid: bool = True
    warnings: List[str] = field(default_factory=list)
    strength: str = "unknown"  # "strong", "moderate", "weak", "misleading"

    def summary(self) -> str:
        if self.mapping:
            shared = len(self.mapping.shared_properties)
            total = shared + len(self.mapping.source_only) + len(self.mapping.target_only)
            return (f"'{self.target}' is like '{self.source}': {self.strength} "
                    f"({shared}/{total} properties shared)")
        return f"Analogy: {self.strength}"


# Known domain properties for common analogy targets
_DOMAIN_PROPERTIES = {
    "stack": {"lifo", "push", "pop", "last-in-first-out", "top", "ordered", "sequential"},
    "queue": {"fifo", "enqueue", "dequeue", "first-in-first-out", "front", "back", "ordered"},
    "tree": {"root", "branches", "leaves", "hierarchical", "parent", "child", "depth"},
    "graph": {"nodes", "edges", "connected", "paths", "cycles", "directed", "weighted"},
    "hash table": {"key", "value", "lookup", "constant-time", "collision", "hash", "bucket"},
    "array": {"index", "sequential", "contiguous", "random-access", "fixed-size"},
    "linked list": {"nodes", "pointers", "sequential", "dynamic", "insertion", "traversal"},
    "database": {"tables", "rows", "columns", "queries", "index", "transactions", "acid"},
    "cache": {"fast", "temporary", "eviction", "hit", "miss", "invalidation", "stale"},
    "pipe": {"flow", "input", "output", "sequential", "unidirectional", "buffer"},
    "factory": {"creation", "abstraction", "interface", "product", "decoupled"},
    "observer": {"subscribe", "notify", "event", "listener", "decoupled", "publish"},
    "singleton": {"one-instance", "global", "shared", "state", "access-point"},
    "book": {"pages", "chapters", "index", "table-of-contents", "sequential", "searchable"},
    "recipe": {"steps", "ingredients", "sequential", "instructions", "ordered", "repeatable"},
    "map": {"territory", "abstraction", "representation", "scale", "simplified", "navigation"},
    "toolbox": {"tools", "collection", "purpose-specific", "reusable", "organized"},
    "assembly line": {"sequential", "stages", "specialization", "throughput", "pipeline"},
    "immune system": {"detection", "response", "memory", "adaptive", "defense", "pattern"},
    "nervous system": {"signals", "distributed", "fast", "reactive", "connected", "parallel"},
}

# Analogy extraction patterns
_ANALOGY_PATTERNS = [
    re.compile(r'(\w[\w\s-]*?)\s+is\s+(?:like|similar to|analogous to|comparable to)\s+(?:a\s+|an\s+)?(\w[\w\s-]*?)(?:\.|,|$)', re.IGNORECASE),
    re.compile(r'(?:think of|imagine)\s+(\w[\w\s-]*?)\s+as\s+(?:a\s+|an\s+)?(\w[\w\s-]*?)(?:\.|,|$)', re.IGNORECASE),
    re.compile(r'(\w[\w\s-]*?)\s+(?:works|functions|operates)\s+(?:just\s+)?like\s+(?:a\s+|an\s+)?(\w[\w\s-]*?)(?:\.|,|$)', re.IGNORECASE),
    re.compile(r'(?:just as|in the same way)\s+(?:a\s+|an\s+)?(\w[\w\s-]*?)\s+.{5,40},\s*(\w[\w\s-]*?)\s', re.IGNORECASE),
]


class AnalogyVerifier:
    """Verifies structural mappings in analogies."""

    def __init__(self, domain_properties: Optional[Dict[str, Set[str]]] = None):
        self._domains = domain_properties or _DOMAIN_PROPERTIES

    def extract_analogies(self, text: str) -> List[tuple]:
        """Extract (target, source) analogy pairs from text."""
        pairs = []
        for pat in _ANALOGY_PATTERNS:
            for m in pat.finditer(text):
                target = m.group(1).strip().lower()
                source = m.group(2).strip().lower()
                if len(target) > 2 and len(source) > 2:
                    pairs.append((target, source))
        return pairs

    def check(self, text: str) -> List[AnalogyResult]:
        """Check all analogies in text."""
        results = []
        pairs = self.extract_analogies(text)

        for target, source in pairs:
            result = self._check_pair(target, source, text)
            results.append(result)

        return results

    def _check_pair(self, target: str, source: str, original: str) -> AnalogyResult:
        """Check a single analogy pair."""
        result = AnalogyResult(original=original, source=source, target=target)

        # Look up known properties
        source_props = self._find_properties(source)
        target_props = self._find_properties(target)

        if source_props and target_props:
            shared = source_props & target_props
            mapping = AnalogyMapping(
                source=source,
                target=target,
                shared_properties=sorted(shared),
                source_only=sorted(source_props - target_props),
                target_only=sorted(target_props - source_props),
            )
            total = len(shared) + len(mapping.source_only) + len(mapping.target_only)
            mapping.strength = len(shared) / total if total > 0 else 0
            result.mapping = mapping

            if mapping.strength > 0.5:
                result.strength = "strong"
            elif mapping.strength > 0.25:
                result.strength = "moderate"
            elif mapping.strength > 0:
                result.strength = "weak"
                result.warnings.append(
                    f"Low overlap ({len(shared)}/{total} properties) — "
                    f"analogy may be misleading"
                )
            else:
                result.strength = "misleading"
                result.valid = False
                result.warnings.append("No shared properties found — analogy breaks down")
        else:
            result.strength = "unknown"
            if not source_props:
                result.warnings.append(f"No known properties for '{source}'")
            if not target_props:
                result.warnings.append(f"No known properties for '{target}'")

        return result

    def _find_properties(self, term: str) -> Optional[Set[str]]:
        """Find known properties for a term."""
        term_lower = term.lower()

        # Direct match
        if term_lower in self._domains:
            return self._domains[term_lower]

        # Substring match
        for domain, props in self._domains.items():
            if domain in term_lower or term_lower in domain:
                return props

        return None

    def suggest_better_analogy(self, target: str) -> List[str]:
        """Suggest domains with high property overlap for a target."""
        target_props = self._find_properties(target)
        if not target_props:
            return []

        scored = []
        for domain, props in self._domains.items():
            if domain.lower() == target.lower():
                continue
            shared = len(target_props & props)
            total = len(target_props | props)
            if shared > 0:
                scored.append((domain, shared / total))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [f"{domain} ({score:.0%} overlap)" for domain, score in scored[:5]]
