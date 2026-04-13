"""
Auto-CALM Information Density — detect filler vs substance.

Models pad responses with filler, repetition, and pleasantries. This
module measures information density and flags low-information sections.

Metrics:
  - Unique information rate: new info per sentence
  - Filler ratio: pleasantries, transitions, padding
  - Repetition detection: same idea restated
  - Signal-to-noise ratio: substance vs filler

Usage:
    from calm.density import DensityAnalyzer
    da = DensityAnalyzer()
    result = da.analyze("Sure! Great question! I'd be happy to help!")
    print(result.density_score)  # low
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class DensityResult:
    """Information density analysis result."""
    density_score: float = 0.0       # 0-1 (1 = pure information)
    filler_ratio: float = 0.0        # 0-1 (proportion of filler)
    repetition_ratio: float = 0.0    # 0-1 (proportion of repeated info)
    unique_info_count: int = 0       # distinct informational sentences
    total_sentences: int = 0
    filler_examples: List[str] = field(default_factory=list)
    repeated_examples: List[str] = field(default_factory=list)
    label: str = "unknown"           # "dense", "adequate", "padded", "filler"

    def summary(self) -> str:
        return (f"{self.label} ({self.density_score:.0%} density, "
                f"{self.filler_ratio:.0%} filler, {self.repetition_ratio:.0%} repetition)")


# Filler patterns — add no information
_FILLER_PATTERNS = [
    re.compile(r'^\s*(?:Sure|Of course|Absolutely|Great question|Good question|Certainly|Happy to help|I\'d be happy to|Let me help|I can help)[!.]*\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:That\'s a great|That\'s an excellent|That\'s a good|What a great) (?:question|point|observation)[!.]*\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:Here\'s|Here is) (?:the|my|a) (?:answer|response|explanation|breakdown)[.:]\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:Let me explain|Let me break|I\'ll explain|Allow me to)\b', re.IGNORECASE),
    re.compile(r'^\s*(?:In conclusion|To summarize|In summary|To sum up|Overall)[,:]?\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:I hope this helps|Hope that helps|Let me know if|Feel free to ask|Don\'t hesitate)[!.]*\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:As mentioned|As I said|As noted|As stated) (?:earlier|above|before|previously)', re.IGNORECASE),
]

# Transition padding — necessary but low-info
_TRANSITION_PATTERNS = [
    re.compile(r'^\s*(?:Now|Next|Moving on|Furthermore|Additionally|Moreover|Also)\s*[,:]?\s*$', re.IGNORECASE),
    re.compile(r'^\s*(?:With that said|That being said|Having said that|On that note)\s*[,:]?\s*$', re.IGNORECASE),
]

# Weasel words that dilute information
_WEASEL_WORDS = re.compile(
    r'\b(?:basically|essentially|actually|literally|very|really|quite|'
    r'somewhat|fairly|rather|pretty much|kind of|sort of|in a way|'
    r'to be honest|to be fair|at the end of the day|it goes without saying|'
    r'needless to say|it should be noted|it is worth mentioning)\b',
    re.IGNORECASE,
)


class DensityAnalyzer:
    """Measures information density of text."""

    def analyze(self, text: str) -> DensityResult:
        """Analyze information density."""
        result = DensityResult()

        # Split into sentences
        sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if s.strip()]
        result.total_sentences = len(sentences)

        if not sentences:
            result.label = "empty"
            return result

        filler_count = 0
        info_sentences = []
        info_stems = []

        for sent in sentences:
            # Check for filler
            is_filler = False
            for pat in _FILLER_PATTERNS + _TRANSITION_PATTERNS:
                if pat.match(sent):
                    is_filler = True
                    result.filler_examples.append(sent[:60])
                    break

            if is_filler:
                filler_count += 1
            else:
                info_sentences.append(sent)
                # Stem for repetition detection
                stems = self._stem_set(sent)
                info_stems.append(stems)

        result.filler_ratio = filler_count / len(sentences) if sentences else 0

        # Repetition detection: pairwise Jaccard similarity
        repeated = 0
        for i in range(len(info_stems)):
            for j in range(i + 1, len(info_stems)):
                sim = self._jaccard(info_stems[i], info_stems[j])
                if sim > 0.5:
                    repeated += 1
                    if len(result.repeated_examples) < 3:
                        result.repeated_examples.append(
                            f"'{info_sentences[i][:30]}...' ≈ '{info_sentences[j][:30]}...'"
                        )

        max_pairs = len(info_stems) * (len(info_stems) - 1) / 2
        result.repetition_ratio = repeated / max_pairs if max_pairs > 0 else 0

        # Weasel word density
        weasel_count = len(_WEASEL_WORDS.findall(text))
        word_count = len(text.split())
        weasel_density = weasel_count / word_count if word_count > 0 else 0

        # Unique information count
        result.unique_info_count = len(info_sentences) - repeated

        # Overall density score
        result.density_score = max(0, min(1,
            (1 - result.filler_ratio) * 0.4 +
            (1 - result.repetition_ratio) * 0.3 +
            (1 - weasel_density * 10) * 0.3
        ))

        # Label
        if result.density_score > 0.8:
            result.label = "dense"
        elif result.density_score > 0.6:
            result.label = "adequate"
        elif result.density_score > 0.3:
            result.label = "padded"
        else:
            result.label = "filler"

        return result

    def _stem_set(self, text: str) -> Set[str]:
        """Extract stemmed significant words."""
        words = re.findall(r'[a-z]+', text.lower())
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "this", "that", "it", "and", "or", "but", "not", "you", "we",
                "they", "can", "could", "should", "may", "might"}
        return {w[:5] for w in words if w not in stop and len(w) > 2}

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)

    def score_response(self, response: str) -> str:
        """Quick scoring: return a one-line assessment."""
        result = self.analyze(response)
        return result.summary()
