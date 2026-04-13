"""
Auto-CALM Perspective Taking — detect which viewpoints are represented.

Flags when important perspectives are missing from an analysis.
"You considered performance but not user experience" or "This only
looks at the developer perspective, not operations."

Usage:
    from calm.perspective import PerspectiveChecker
    pc = PerspectiveChecker()
    result = pc.check("We should use microservices for better scalability")
    print(result.covered)     # ["engineering"]
    print(result.missing)     # ["operations", "user", "business"]
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class PerspectiveResult:
    """Result of perspective analysis."""
    covered: List[str] = field(default_factory=list)
    missing: List[str] = field(default_factory=list)
    dominant: str = ""
    balance_score: float = 0.0  # 0-1, how well-balanced

    def summary(self) -> str:
        if not self.covered:
            return "No clear perspective detected"
        parts = [f"Covered: {', '.join(self.covered)}"]
        if self.missing:
            parts.append(f"Missing: {', '.join(self.missing)}")
        parts.append(f"Balance: {self.balance_score:.0%}")
        return ". ".join(parts)


# Perspective definitions: name → signal keywords
_PERSPECTIVES = {
    "user": {
        "keywords": re.compile(r'\b(?:user|customer|client|audience|visitor|player|reader|experience|UX|UI|usability|accessibility|intuitive|friendly|onboarding)\b', re.IGNORECASE),
        "description": "End-user experience and needs",
    },
    "engineering": {
        "keywords": re.compile(r'\b(?:code|implementation|algorithm|data structure|architecture|pattern|refactor|technical|developer|DX|API|library|framework|language|runtime)\b', re.IGNORECASE),
        "description": "Technical implementation concerns",
    },
    "operations": {
        "keywords": re.compile(r'\b(?:deploy|infrastructure|monitor|logging|alert|uptime|SLA|incident|on.call|scale|load|capacity|CDN|CI.?CD|pipeline|container|cloud)\b', re.IGNORECASE),
        "description": "Deployment, monitoring, reliability",
    },
    "security": {
        "keywords": re.compile(r'\b(?:security|vulnerab|threat|attack|auth|encrypt|permission|access control|compliance|audit|pentest|CVE|OWASP)\b', re.IGNORECASE),
        "description": "Security and compliance",
    },
    "business": {
        "keywords": re.compile(r'\b(?:revenue|cost|profit|market|competitor|stakeholder|ROI|budget|timeline|deadline|strategy|roadmap|priority|feature|product|growth|retention)\b', re.IGNORECASE),
        "description": "Business value and strategy",
    },
    "data": {
        "keywords": re.compile(r'\b(?:data|analytics|metrics|KPI|dashboard|tracking|measurement|A.B test|experiment|insight|report|visualization)\b', re.IGNORECASE),
        "description": "Data, measurement, analytics",
    },
    "team": {
        "keywords": re.compile(r'\b(?:team|hiring|onboard|skill|training|knowledge|documentation|review|collaboration|process|workflow|standup|sprint|retro)\b', re.IGNORECASE),
        "description": "Team dynamics and processes",
    },
    "legal": {
        "keywords": re.compile(r'\b(?:legal|regulation|compliance|GDPR|CCPA|license|copyright|patent|terms|privacy|policy|consent|liability)\b', re.IGNORECASE),
        "description": "Legal and regulatory requirements",
    },
}

# Context → which perspectives are typically relevant
_CONTEXT_EXPECTED_PERSPECTIVES = {
    "technical_decision": ["engineering", "operations", "security", "user"],
    "product_decision": ["user", "business", "engineering", "data"],
    "architecture": ["engineering", "operations", "security", "user"],
    "incident": ["operations", "engineering", "user", "business"],
    "planning": ["business", "engineering", "user", "team"],
    "default": ["user", "engineering", "business"],
}


class PerspectiveChecker:
    """Checks which perspectives are represented in reasoning."""

    def check(self, text: str, context: str = "default") -> PerspectiveResult:
        """Check which perspectives are covered and missing."""
        result = PerspectiveResult()

        # Score each perspective
        scores = {}
        for name, info in _PERSPECTIVES.items():
            matches = len(info["keywords"].findall(text))
            scores[name] = matches

        # Covered = any matches
        result.covered = [name for name, score in scores.items() if score > 0]

        # Dominant = highest scoring
        if scores:
            result.dominant = max(scores, key=scores.get)

        # Missing = expected for this context but not covered
        expected = _CONTEXT_EXPECTED_PERSPECTIVES.get(context,
                    _CONTEXT_EXPECTED_PERSPECTIVES["default"])
        result.missing = [p for p in expected if p not in result.covered]

        # Balance score
        if expected:
            result.balance_score = len([p for p in expected if p in result.covered]) / len(expected)

        return result

    def detect_context(self, text: str) -> str:
        """Auto-detect what kind of decision/analysis this is."""
        text_lower = text.lower()
        if any(w in text_lower for w in ["incident", "outage", "down", "alert", "page"]):
            return "incident"
        if any(w in text_lower for w in ["architect", "design", "system", "service"]):
            return "architecture"
        if any(w in text_lower for w in ["plan", "roadmap", "quarter", "sprint", "milestone"]):
            return "planning"
        if any(w in text_lower for w in ["feature", "product", "user story", "requirement"]):
            return "product_decision"
        if any(w in text_lower for w in ["choose", "compare", "vs", "which", "should"]):
            return "technical_decision"
        return "default"

    def full_check(self, text: str) -> PerspectiveResult:
        """Auto-detect context and check perspectives."""
        context = self.detect_context(text)
        return self.check(text, context)

    def suggest_questions(self, missing: List[str]) -> List[str]:
        """Suggest questions to address missing perspectives."""
        questions = {
            "user": "How does this affect the end-user experience?",
            "engineering": "What are the technical implications and tradeoffs?",
            "operations": "How will this be deployed, monitored, and maintained?",
            "security": "What are the security implications? Any new attack surfaces?",
            "business": "What is the business impact? Cost? Timeline?",
            "data": "How will we measure success? What metrics matter?",
            "team": "Does the team have the skills? What training is needed?",
            "legal": "Are there legal, compliance, or licensing concerns?",
        }
        return [questions[p] for p in missing if p in questions]
