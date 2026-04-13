"""
Auto-CALM Explanation Quality — is the explanation actually explanatory?

Detects when an "explanation" just restates the question ("what" not "why"),
is circular, or relies on jargon without defining it. Forces explanations
to actually explain.

Usage:
    from calm.explanation import ExplanationChecker
    ec = ExplanationChecker()
    result = ec.check(
        question="Why does Python use a GIL?",
        answer="Python uses a GIL because it has a Global Interpreter Lock."
    )
    print(result.is_circular)  # True
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class ExplanationResult:
    """Assessment of explanation quality."""
    is_circular: bool = False
    is_tautological: bool = False
    answers_why: bool = False
    jargon_undefined: List[str] = field(default_factory=list)
    depth_level: str = "surface"  # "surface", "mechanism", "root_cause"
    quality_score: float = 0.5
    issues: List[str] = field(default_factory=list)

    def summary(self) -> str:
        parts = [f"depth={self.depth_level}, score={self.quality_score:.0%}"]
        if self.is_circular:
            parts.append("CIRCULAR")
        if self.is_tautological:
            parts.append("TAUTOLOGICAL")
        if self.jargon_undefined:
            parts.append(f"undefined jargon: {', '.join(self.jargon_undefined[:3])}")
        if self.issues:
            parts.append(f"{len(self.issues)} issues")
        return ", ".join(parts)


# Common technical jargon that should be defined when used in explanations
_JARGON = {
    "mutex", "semaphore", "deadlock", "race condition", "atomic",
    "idempotent", "deterministic", "polymorphism", "encapsulation",
    "abstraction", "serialization", "marshalling", "memoization",
    "closure", "currying", "monad", "functor", "thunk",
    "sharding", "partitioning", "replication", "consensus",
    "eventual consistency", "CAP theorem", "ACID", "BASE",
    "backpressure", "circuit breaker", "bulkhead",
    "dependency injection", "inversion of control",
    "garbage collection", "reference counting",
    "JIT", "AOT", "bytecode", "AST",
    "TCP", "UDP", "TLS", "DNS", "DHCP",
    "REST", "GraphQL", "gRPC", "WebSocket",
    "OAuth", "JWT", "CORS", "CSRF", "XSS",
    "FIFO", "LIFO", "LRU", "LFU",
}

# Depth indicators
_MECHANISM_SIGNALS = re.compile(
    r'\b(?:because|the reason|this happens|works by|mechanism|under the hood|internally|step by step)\b',
    re.IGNORECASE,
)
_ROOT_CAUSE_SIGNALS = re.compile(
    r'\b(?:root cause|fundamental|underlying|the real reason|ultimately|core issue|at its heart|designed this way)\b',
    re.IGNORECASE,
)
_SURFACE_SIGNALS = re.compile(
    r'\b(?:basically|simply|just|essentially|in short|put simply)\b',
    re.IGNORECASE,
)


class ExplanationChecker:
    """Checks whether explanations are actually explanatory."""

    def check(self, question: str, answer: str) -> ExplanationResult:
        """Check explanation quality."""
        result = ExplanationResult()

        # 1. Circularity check
        result.is_circular = self._check_circular(question, answer)
        if result.is_circular:
            result.issues.append("Explanation is circular — restates the question")
            result.quality_score -= 0.3

        # 2. Tautology check
        result.is_tautological = self._check_tautological(answer)
        if result.is_tautological:
            result.issues.append("Explanation is tautological — X because X")
            result.quality_score -= 0.2

        # 3. Does it answer "why"?
        if re.match(r'\b(?:why|how)\b', question, re.IGNORECASE):
            result.answers_why = bool(_MECHANISM_SIGNALS.search(answer) or
                                      _ROOT_CAUSE_SIGNALS.search(answer))
            if not result.answers_why:
                result.issues.append("Question asks 'why/how' but answer doesn't explain mechanism")
                result.quality_score -= 0.2

        # 4. Undefined jargon
        result.jargon_undefined = self._find_undefined_jargon(answer)
        if result.jargon_undefined:
            result.issues.append(f"Uses jargon without defining: {', '.join(result.jargon_undefined[:3])}")
            result.quality_score -= 0.1 * min(len(result.jargon_undefined), 3)

        # 5. Depth level
        if _ROOT_CAUSE_SIGNALS.search(answer):
            result.depth_level = "root_cause"
            result.quality_score += 0.2
        elif _MECHANISM_SIGNALS.search(answer):
            result.depth_level = "mechanism"
            result.quality_score += 0.1
        else:
            result.depth_level = "surface"

        result.quality_score = max(0, min(1, result.quality_score))
        return result

    def _check_circular(self, question: str, answer: str) -> bool:
        """Check if the answer is just restating the question."""
        # Extract key content words from question
        q_words = set(re.findall(r'[a-z]+', question.lower()))
        q_words -= {"what", "why", "how", "does", "is", "are", "the", "a", "an",
                     "of", "in", "to", "for", "it", "do", "can", "this", "that"}

        if not q_words:
            return False

        # Check first sentence of answer
        first_sentence = re.split(r'[.!?]', answer)[0].lower()
        a_words = set(re.findall(r'[a-z]+', first_sentence))

        # If >70% of question words appear in first answer sentence,
        # and answer doesn't add many new words, it's likely circular
        overlap = len(q_words & a_words)
        new_words = a_words - q_words - {"what", "why", "how", "does", "is", "are",
                                          "the", "a", "an", "because", "since",
                                          "it", "has", "uses", "with", "by"}
        if q_words and overlap / len(q_words) > 0.6 and len(new_words) < 3:
            return True

        return False

    def _check_tautological(self, answer: str) -> bool:
        """Check for tautological patterns: "X because X"."""
        # "X because of X" or "X is X"
        sentences = re.split(r'[.!?]', answer)
        for sent in sentences:
            m = re.search(r'(.{5,30}?)\s+because\s+(?:of\s+)?(?:the\s+)?(.{5,30})', sent, re.IGNORECASE)
            if m:
                a = set(re.findall(r'[a-z]+', m.group(1).lower()))
                b = set(re.findall(r'[a-z]+', m.group(2).lower()))
                if a and b:
                    overlap = len(a & b)
                    if overlap / max(len(a), len(b)) > 0.6:
                        return True
        return False

    def _find_undefined_jargon(self, text: str) -> List[str]:
        """Find technical jargon used but not defined/explained."""
        text_lower = text.lower()
        used = []
        for term in _JARGON:
            if term.lower() in text_lower:
                # Check if it's followed by a definition (", which", "—", "i.e.")
                idx = text_lower.index(term.lower())
                after = text_lower[idx + len(term):idx + len(term) + 30]
                if not re.search(r'(?:which|meaning|i\.e\.|that is|defined as|\(|—)', after):
                    used.append(term)
        return used
