"""
Auto-CALM Semantic Compression — same answer, fewer words, no info loss.

The inverse of density detection. Instead of just measuring filler,
actively compress by removing redundancy while preserving information.

Usage:
    from calm.compression import SemanticCompressor
    sc = SemanticCompressor()
    result = sc.compress("Sure! Great question! The answer is 42. I hope this helps!")
    # → "The answer is 42."
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class CompressionResult:
    """Result of semantic compression."""
    original: str = ""
    compressed: str = ""
    original_words: int = 0
    compressed_words: int = 0
    compression_ratio: float = 0.0   # 0-1 (how much was removed)
    removed_sections: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (f"{self.original_words} → {self.compressed_words} words "
                f"({self.compression_ratio:.0%} removed)")


# Filler patterns to remove entirely
_REMOVE_PATTERNS = [
    re.compile(r'^\s*(?:Sure|Of course|Absolutely|Great question|Good question|Certainly)[!.]*\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:That\'s a great|That\'s an excellent|That\'s a good|What a great) (?:question|point|observation)[!.]*\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:I\'d be happy to|Happy to help|Let me help|I can help)[!.]*\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:I hope this helps|Hope that helps|Let me know if|Feel free to ask|Don\'t hesitate)[!.]*\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:Here\'s|Here is) (?:the|my|a) (?:answer|response|explanation|breakdown)[.:]\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:Let me explain|Let me break|I\'ll explain|Allow me to explain)[.!]*\s*$', re.IGNORECASE | re.MULTILINE),
    re.compile(r'^\s*(?:In conclusion|To summarize|In summary|To sum up|Overall)[,:]\s*$', re.IGNORECASE | re.MULTILINE),
]

# Weasel phrases to trim (keep the rest of the sentence)
_TRIM_PATTERNS = [
    (re.compile(r'\b(?:basically|essentially|actually|literally|really|quite|rather|pretty much)\s+', re.IGNORECASE), ''),
    (re.compile(r'\b(?:it\'s worth noting that|it should be noted that|it is important to note that)\s+', re.IGNORECASE), ''),
    (re.compile(r'\b(?:as mentioned earlier|as I said before|as noted above|as stated previously),?\s+', re.IGNORECASE), ''),
    (re.compile(r'\b(?:needless to say|it goes without saying|of course),?\s+', re.IGNORECASE), ''),
    (re.compile(r'\b(?:in other words|that is to say|what I mean is),?\s+', re.IGNORECASE), ''),
]

# Redundant sentence patterns (same idea restated)
_REDUNDANCY_THRESHOLD = 0.5  # Jaccard similarity for "same idea"


class SemanticCompressor:
    """Compresses text by removing filler while preserving information."""

    def compress(self, text: str) -> CompressionResult:
        """Compress text by removing filler, weasel words, and redundancy."""
        result = CompressionResult(original=text)
        result.original_words = len(text.split())

        compressed = text

        # Phase 1: Remove filler sentences
        # Split into sentences first so each can be matched independently
        sentences = re.split(r'(?<=[.!?])\s+', compressed)
        kept = []
        for sent in sentences:
            is_filler = False
            for pat in _REMOVE_PATTERNS:
                if pat.match(sent.strip()):
                    is_filler = True
                    result.removed_sections.append(sent.strip())
                    break
            if not is_filler:
                kept.append(sent)
        compressed = " ".join(kept)

        # Phase 2: Trim weasel phrases
        for pat, replacement in _TRIM_PATTERNS:
            compressed = pat.sub(replacement, compressed)

        # Phase 3: Remove redundant sentences
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', compressed) if s.strip()]
        unique = []
        seen_stems = []
        for sent in sentences:
            stems = self._stem_set(sent)
            is_redundant = False
            for prev_stems in seen_stems:
                if self._jaccard(stems, prev_stems) > _REDUNDANCY_THRESHOLD:
                    is_redundant = True
                    result.removed_sections.append(f"(redundant) {sent[:40]}...")
                    break
            if not is_redundant:
                unique.append(sent)
                seen_stems.append(stems)

        compressed = " ".join(unique)

        # Phase 4: Clean up whitespace
        compressed = re.sub(r'\n{3,}', '\n\n', compressed)
        compressed = re.sub(r'  +', ' ', compressed)
        compressed = compressed.strip()

        result.compressed = compressed
        result.compressed_words = len(compressed.split()) if compressed else 0
        if result.original_words > 0:
            result.compression_ratio = 1 - result.compressed_words / result.original_words

        return result

    def _stem_set(self, text):
        words = re.findall(r'[a-z]+', text.lower())
        stop = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                "have", "has", "had", "do", "does", "did", "will", "would",
                "to", "of", "in", "for", "on", "with", "at", "by", "from",
                "this", "that", "it", "and", "or", "but", "not", "you", "we"}
        return {w[:4] for w in words if w not in stop and len(w) > 2}

    def _jaccard(self, a, b):
        if not a or not b:
            return 0.0
        return len(a & b) / len(a | b)
