"""
Auto-CALM Scope Tracking — detect over/under-generalization.

Models use "always" when they mean "usually" and "sometimes" when
they mean "always." This module detects scope claims and verifies
them against known facts where possible.

Types of scope issues:
  1. Overgeneralization: "X always causes Y" (does it really?)
  2. Undergeneralization: "X sometimes works" (when it always does)
  3. Scope creep: starts specific, drifts to general without justification
  4. Missing qualifiers: "X is better than Y" (for what? measured how?)

Usage:
    from calm.scope import ScopeTracker
    st = ScopeTracker()
    issues = st.check("Python is always slower than C")
    # → [ScopeIssue: "always" — overgeneralization]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class ScopeIssue:
    """A detected scope problem."""
    text: str            # the problematic text
    issue_type: str      # "overgeneralization", "undergeneralization", "missing_qualifier", "scope_creep"
    trigger: str         # what word/phrase triggered it
    suggestion: str = "" # how to fix it
    severity: str = "warning"  # "warning" or "error"

    def __str__(self):
        return f"[{self.issue_type}] '{self.trigger}': {self.suggestion}"


# Absolute quantifiers that signal overgeneralization
_ABSOLUTE_QUANTIFIERS = [
    (re.compile(r'\b(always|never|every|all|none|no one|everyone|everything|nothing|impossible|guaranteed|certainly|definitely|undeniably)\b', re.IGNORECASE),
     "overgeneralization",
     lambda m: f"'{m.group(1)}' is absolute — consider 'usually', 'rarely', 'most', or specify conditions"),
    (re.compile(r'\b(best|worst|fastest|slowest|largest|smallest|most important|least important)\b', re.IGNORECASE),
     "missing_qualifier",
     lambda m: f"'{m.group(1)}' needs a qualifier — best FOR WHAT? measured HOW?"),
]

# Weak quantifiers that might be undergeneralization
_WEAK_QUANTIFIERS = [
    (re.compile(r'\b(sometimes|maybe|possibly|might|perhaps|could|occasionally)\b', re.IGNORECASE),
     "undergeneralization",
     lambda m: f"'{m.group(1)}' is vague — can you be more specific about when/how often?"),
]

# Unqualified comparisons
_COMPARISON_PATTERNS = [
    (re.compile(r'(\w+)\s+is\s+(better|worse|faster|slower|easier|harder|simpler|more complex)\s+than\s+(\w+)', re.IGNORECASE),
     "missing_qualifier",
     lambda m: f"'{m.group(1)} is {m.group(2)} than {m.group(3)}' — by what metric? in what context?"),
    (re.compile(r'you should (?:always |never |)(?:use|choose|prefer|avoid)\s+(\w+)', re.IGNORECASE),
     "overgeneralization",
     lambda m: f"blanket recommendation for '{m.group(1)}' — when specifically? what are the exceptions?"),
]

# Scope creep patterns (specific → general without justification)
_SCOPE_CREEP = [
    (re.compile(r'in (?:this|my|our) (?:case|project|codebase|system)\b.{20,200}?\b(?:in general|generally|always|universally|across all|every)\b', re.IGNORECASE | re.DOTALL),
     "scope_creep",
     lambda m: "Started specific ('in this case') but drifted to general ('always/generally') without justification"),
]


class ScopeTracker:
    """Detects scope issues in model reasoning."""

    def check(self, text: str) -> List[ScopeIssue]:
        """Check text for scope issues."""
        issues = []

        # Check absolute quantifiers
        for pat, issue_type, suggestion_fn in _ABSOLUTE_QUANTIFIERS:
            for m in pat.finditer(text):
                # Skip if it's in a quote or code block
                start = max(0, m.start() - 5)
                prefix = text[start:m.start()]
                if '`' in prefix or '"' in prefix or "'" in prefix:
                    continue
                issues.append(ScopeIssue(
                    text=self._get_context(text, m),
                    issue_type=issue_type,
                    trigger=m.group(0),
                    suggestion=suggestion_fn(m),
                ))

        # Check weak quantifiers
        for pat, issue_type, suggestion_fn in _WEAK_QUANTIFIERS:
            for m in pat.finditer(text):
                issues.append(ScopeIssue(
                    text=self._get_context(text, m),
                    issue_type=issue_type,
                    trigger=m.group(0),
                    suggestion=suggestion_fn(m),
                ))

        # Check unqualified comparisons
        for pat, issue_type, suggestion_fn in _COMPARISON_PATTERNS:
            for m in pat.finditer(text):
                issues.append(ScopeIssue(
                    text=self._get_context(text, m),
                    issue_type=issue_type,
                    trigger=m.group(0),
                    suggestion=suggestion_fn(m),
                ))

        # Check scope creep
        for pat, issue_type, suggestion_fn in _SCOPE_CREEP:
            for m in pat.finditer(text):
                issues.append(ScopeIssue(
                    text=self._get_context(text, m),
                    issue_type=issue_type,
                    trigger=m.group(0)[:80],
                    suggestion=suggestion_fn(m),
                ))

        # Deduplicate by trigger
        seen = set()
        unique = []
        for issue in issues:
            key = (issue.issue_type, issue.trigger.lower())
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        return unique

    def _get_context(self, text: str, match: re.Match, window: int = 40) -> str:
        """Get surrounding context for a match."""
        start = max(0, match.start() - window)
        end = min(len(text), match.end() + window)
        return text[start:end].strip()

    def score(self, text: str) -> Tuple[float, str]:
        """Score scope precision: 1.0 = perfectly qualified, 0.0 = all absolute claims."""
        issues = self.check(text)
        if not issues:
            return 1.0, "well-scoped"

        # Weight by severity
        total_weight = 0
        for issue in issues:
            if issue.issue_type == "overgeneralization":
                total_weight += 2
            elif issue.issue_type == "missing_qualifier":
                total_weight += 1.5
            elif issue.issue_type == "scope_creep":
                total_weight += 3
            else:
                total_weight += 0.5

        # Normalize: more text = more chances for issues, so adjust by length
        words = len(text.split())
        density = total_weight / max(words / 50, 1)  # issues per 50 words
        score = max(0, 1 - density * 0.2)

        if score > 0.8:
            label = "well-scoped"
        elif score > 0.6:
            label = "mostly scoped"
        elif score > 0.4:
            label = "needs qualification"
        else:
            label = "overgeneralized"

        return round(score, 2), label

    def summary(self, issues: List[ScopeIssue]) -> str:
        """Summarize scope issues."""
        if not issues:
            return "No scope issues detected"
        types = {}
        for i in issues:
            types[i.issue_type] = types.get(i.issue_type, 0) + 1
        parts = [f"{count} {itype}" for itype, count in types.items()]
        return f"{len(issues)} scope issues: {', '.join(parts)}"
