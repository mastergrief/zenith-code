"""
Auto-CALM Logical Inference — verify argument structure, detect fallacies.

Extends chain_verify from math proofs to logical proofs. Validates
syllogisms, detects non-sequiturs, flags common fallacies.

This catches the structural errors in reasoning that math verification
misses: "All X are Y. Z is X. Therefore Z is Y" is valid, but
"All X are Y. Z is Y. Therefore Z is X" is affirming the consequent.

Usage:
    from calm.logic import LogicVerifier
    lv = LogicVerifier()
    result = lv.check_argument(premises, conclusion)
    print(result.valid)
    print(result.fallacies)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Optional, Tuple


@dataclass
class LogicalForm:
    """A logical proposition in normalized form."""
    quantifier: str = ""   # "all", "some", "no", ""
    subject: str = ""
    predicate: str = ""
    negated: bool = False
    original: str = ""


@dataclass
class FallacyDetection:
    """A detected logical fallacy."""
    name: str              # formal name
    description: str       # human-readable explanation
    severity: str = "error"  # "error" (invalid) or "warning" (weak but not invalid)
    location: str = ""     # which part of the argument


@dataclass
class ArgumentResult:
    """Result of checking a logical argument."""
    premises: List[LogicalForm] = field(default_factory=list)
    conclusion: Optional[LogicalForm] = None
    valid: Optional[bool] = None      # True, False, or None (can't determine)
    fallacies: List[FallacyDetection] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def is_sound(self) -> bool:
        return self.valid is True and len(self.fallacies) == 0

    def summary(self) -> str:
        if self.valid is None:
            status = "undetermined"
        elif self.valid:
            status = "valid"
        else:
            status = "INVALID"
        parts = [f"Argument: {status}"]
        if self.fallacies:
            parts.append(f", {len(self.fallacies)} fallacies: " +
                        ", ".join(f.name for f in self.fallacies))
        if self.warnings:
            parts.append(f", {len(self.warnings)} warnings")
        return "".join(parts)


# Patterns for extracting logical propositions
_PROPOSITION_PATTERNS = [
    # "All X are Y"
    (re.compile(r'[Aa]ll\s+([\w][\w\s-]*?)\s+are\s+([\w][\w\s-]*?)(?:\.|,|$)'), "all", False),
    # "No X are Y"
    (re.compile(r'[Nn]o\s+([\w][\w\s-]*?)\s+are\s+([\w][\w\s-]*?)(?:\.|,|$)'), "no", False),
    # "Some X are Y"
    (re.compile(r'[Ss]ome\s+([\w][\w\s-]*?)\s+are\s+([\w][\w\s-]*?)(?:\.|,|$)'), "some", False),
    # "X is Y" / "X is a Y"
    (re.compile(r'([\w][\w\s-]*?)\s+is\s+(?:a\s+|an\s+)?([\w][\w\s-]*?)(?:\.|,|$)'), "", False),
    # "X is not Y"
    (re.compile(r'([\w][\w\s-]*?)\s+is\s+not\s+(?:a\s+|an\s+)?([\w][\w\s-]*?)(?:\.|,|$)'), "", True),
    # "All X are not Y" / "No X is Y"
    (re.compile(r'[Aa]ll\s+([\w][\w\s-]*?)\s+are\s+not\s+([\w][\w\s-]*?)(?:\.|,|$)'), "all", True),
]

# Common informal fallacy patterns in text
_FALLACY_PATTERNS = [
    # Ad hominem
    (re.compile(r'\b(?:stupid|idiot|fool|incompetent|ignorant)\b.*\b(?:therefore|so|thus|hence)\b', re.IGNORECASE | re.DOTALL),
     "Ad Hominem", "Attacking the person instead of the argument"),
    # Appeal to authority
    (re.compile(r'\b(?:expert|authority|professor|doctor|scientist)\s+(?:says?|said|believes?|thinks?)\b.*\b(?:therefore|so|must be)\b', re.IGNORECASE | re.DOTALL),
     "Appeal to Authority", "Using authority status as proof (authority can be wrong)"),
    # Appeal to popularity
    (re.compile(r'\b(?:everyone|most people|majority|popular|widely)\b.*\b(?:therefore|so|must be|proves?)\b', re.IGNORECASE | re.DOTALL),
     "Appeal to Popularity", "Popularity doesn't determine truth"),
    # False dichotomy
    (re.compile(r'\b(?:either|only two|must be one|no other)\b.*\b(?:or)\b', re.IGNORECASE),
     "False Dichotomy", "Presenting only two options when more exist"),
    # Slippery slope
    (re.compile(r'\b(?:lead to|inevitably|eventually|end up|next thing)\b.*\b(?:lead to|result in|cause|inevitably)\b', re.IGNORECASE | re.DOTALL),
     "Slippery Slope", "Assuming one event inevitably leads to an extreme outcome"),
    # Circular reasoning
    (re.compile(r'\b(?:because|since)\s+.{5,50}\b(?:because|since)\b', re.IGNORECASE),
     "Circular Reasoning", "Using the conclusion as a premise"),
    # Straw man
    (re.compile(r'\b(?:what you.re really saying|so you think|that means you believe)\b', re.IGNORECASE),
     "Straw Man", "Misrepresenting the opposing position"),
    # Hasty generalization
    (re.compile(r'\b(?:always|never|every time|without exception)\b.*\b(?:because|since)\s+(?:one|a single|this one|that)\b', re.IGNORECASE | re.DOTALL),
     "Hasty Generalization", "Drawing a broad conclusion from limited evidence"),
]


class LogicVerifier:
    """Verifies logical argument structure and detects fallacies."""

    def _strip_article(self, s: str) -> str:
        """Remove leading articles."""
        return re.sub(r'^(?:a|an|the)\s+', '', s.strip().lower())

    def extract_propositions(self, text: str) -> List[LogicalForm]:
        """Extract logical propositions from text."""
        props = []
        for pat, quantifier, negated in _PROPOSITION_PATTERNS:
            for m in pat.finditer(text):
                prop = LogicalForm(
                    quantifier=quantifier,
                    subject=self._strip_article(m.group(1)),
                    predicate=self._strip_article(m.group(2)),
                    negated=negated,
                    original=m.group(0).strip(),
                )
                # Skip trivially short or generic
                if len(prop.subject) > 1 and len(prop.predicate) > 1:
                    props.append(prop)
        return props

    def _terms_match(self, a: str, b: str) -> bool:
        """Check if two terms match (handles singular/plural)."""
        if a == b:
            return True
        # Simple singular/plural: add/remove trailing 's'
        if a + "s" == b or a == b + "s":
            return True
        # -es plural
        if a + "es" == b or a == b + "es":
            return True
        # Prefix match (4+ chars) as fallback
        if len(a) >= 4 and len(b) >= 4 and a[:4] == b[:4]:
            return True
        return False

    def check_syllogism(self, premises: List[LogicalForm],
                         conclusion: LogicalForm) -> ArgumentResult:
        """Check a categorical syllogism for validity."""
        result = ArgumentResult(premises=premises, conclusion=conclusion)

        if len(premises) < 2:
            result.valid = None
            result.warnings.append("Need at least 2 premises for a syllogism")
            return result

        # Classic syllogism: All A are B. C is A. Therefore C is B.
        # Check for affirming the consequent: All A are B. C is B. Therefore C is A.
        p1, p2 = premises[0], premises[1]
        c = conclusion

        # Valid: All A are B. X is A. → X is B.
        if (p1.quantifier == "all" and not p1.negated and
            self._terms_match(p2.subject, c.subject)):
            if self._terms_match(p2.predicate, p1.subject) and self._terms_match(c.predicate, p1.predicate):
                result.valid = True
                return result

        # Invalid: All A are B. X is B. → X is A. (affirming the consequent)
        if (p1.quantifier == "all" and not p1.negated and
            self._terms_match(p2.subject, c.subject)):
            if self._terms_match(p2.predicate, p1.predicate) and self._terms_match(c.predicate, p1.subject):
                result.valid = False
                result.fallacies.append(FallacyDetection(
                    name="Affirming the Consequent",
                    description=(f"'All {p1.subject} are {p1.predicate}' and "
                                f"'{p2.subject} is {p2.predicate}' does NOT mean "
                                f"'{c.subject} is {p1.subject}'"),
                    severity="error",
                ))
                return result

        # Valid: No A are B. X is A. → X is not B.
        if (p1.quantifier == "no" and not p1.negated and
            self._terms_match(p2.subject, c.subject) and c.negated):
            if self._terms_match(p2.predicate, p1.subject) and self._terms_match(c.predicate, p1.predicate):
                result.valid = True
                return result

        # Modus ponens: If P then Q. P. Therefore Q.
        # Modus tollens: If P then Q. Not Q. Therefore not P.
        # These need different extraction — handled via pattern matching

        result.valid = None
        result.warnings.append("Could not determine validity (non-standard form)")
        return result

    def detect_fallacies(self, text: str) -> List[FallacyDetection]:
        """Detect informal logical fallacies in text."""
        fallacies = []
        for pat, name, description in _FALLACY_PATTERNS:
            if pat.search(text):
                fallacies.append(FallacyDetection(
                    name=name,
                    description=description,
                    severity="warning",
                ))
        return fallacies

    def check_argument(self, text: str) -> ArgumentResult:
        """Full argument check: extract, verify structure, detect fallacies."""
        # Split into premises and conclusion
        # Look for "therefore", "thus", "so", "hence" as conclusion markers
        conclusion_markers = re.compile(
            r'\b(?:therefore|thus|so|hence|it follows that|we can conclude|this means)\b',
            re.IGNORECASE
        )

        parts = conclusion_markers.split(text, maxsplit=1)
        if len(parts) == 2:
            premise_text = parts[0]
            conclusion_text = parts[1]
        else:
            # No explicit conclusion marker — last sentence is conclusion
            sentences = re.split(r'[.!]\s+', text)
            if len(sentences) >= 2:
                premise_text = ". ".join(sentences[:-1])
                conclusion_text = sentences[-1]
            else:
                premise_text = text
                conclusion_text = ""

        premises = self.extract_propositions(premise_text)
        conclusions = self.extract_propositions(conclusion_text)
        conclusion = conclusions[0] if conclusions else None

        # Check syllogistic validity
        if premises and conclusion:
            result = self.check_syllogism(premises, conclusion)
        else:
            result = ArgumentResult(premises=premises, conclusion=conclusion)
            result.valid = None

        # Also check for informal fallacies
        fallacies = self.detect_fallacies(text)
        result.fallacies.extend(fallacies)

        return result
