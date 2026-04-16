"""Creative writing compute backend.

Deterministic metrics for poetry form verification, structural analysis,
and quantitative style assessment. Gemma generates the text; these
functions verify and measure it.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import List

_VOWELS = set("aeiouy")


def syllable_count(word: str) -> int:
    """Count syllables via vowel-cluster heuristic (~90% accuracy)."""
    word = word.lower().strip().rstrip(".,!?;:'\"")
    if not word:
        return 0
    if len(word) <= 2:
        return 1
    count = 0
    prev_vowel = False
    for c in word:
        is_v = c in _VOWELS
        if is_v and not prev_vowel:
            count += 1
        prev_vowel = is_v
    if word.endswith("e") and count > 1:
        if not word.endswith(("le", "ce", "ge", "se", "ve", "ze")):
            count -= 1
    if word.endswith("ed") and len(word) > 3 and count > 1:
        if word[-3] not in "td":
            count -= 1
    return max(count, 1)


def line_syllables(line: str) -> int:
    """Total syllables in a line."""
    return sum(syllable_count(w) for w in line.split())


def rhymes(word1: str, word2: str) -> bool:
    """Check if two words rhyme (matching vowel endings)."""
    def ending(w):
        w = w.lower().strip().rstrip(".,!?;:'\"")
        for i in range(len(w) - 1, -1, -1):
            if w[i] in _VOWELS:
                return w[i:]
        return w[-2:] if len(w) >= 2 else w
    e1, e2 = ending(word1), ending(word2)
    return e1 == e2 and word1.lower().strip() != word2.lower().strip()


def stress_pattern(line: str) -> str:
    """Approximate stress: u=unstressed, S=stressed."""
    pattern = []
    for word in line.split():
        sc = syllable_count(word)
        if sc == 1:
            pattern.append("S" if len(word) > 3 else "u")
        else:
            for i in range(sc):
                pattern.append("S" if i % 2 == 1 else "u")
    return "".join(pattern)


def meter_type(line: str) -> str:
    """Classify meter: iambic, trochaic, anapestic, dactylic, or free."""
    p = stress_pattern(line)
    if not p:
        return "empty"
    pairs = [p[i:i+2] for i in range(0, len(p) - 1, 2)]
    if all(x == "uS" for x in pairs):
        return "iambic"
    if all(x == "Su" for x in pairs):
        return "trochaic"
    triples = [p[i:i+3] for i in range(0, len(p) - 2, 3)]
    if all(x == "uuS" for x in triples):
        return "anapestic"
    if all(x == "Suu" for x in triples):
        return "dactylic"
    return "free"


def haiku_valid(line1: str, line2: str, line3: str) -> bool:
    """Validate haiku: 5-7-5 syllable pattern."""
    return (line_syllables(line1) == 5 and
            line_syllables(line2) == 7 and
            line_syllables(line3) == 5)


def word_count(text: str) -> int:
    """Count words."""
    return len(text.split())


def sentence_count(text: str) -> int:
    """Count sentences."""
    return max(1, len(re.findall(r"[.!?]+(?:\s|$)", text)))


def avg_sentence_len(text: str) -> float:
    """Average words per sentence."""
    return round(word_count(text) / max(sentence_count(text), 1), 1)


def readability_score(text: str) -> float:
    """Flesch-Kincaid reading ease (0-100, higher = easier)."""
    words = text.split()
    wc = len(words)
    if wc == 0:
        return 0.0
    sc = sentence_count(text)
    syls = sum(syllable_count(w) for w in words)
    fk = 206.835 - 1.015 * (wc / max(sc, 1)) - 84.6 * (syls / wc)
    return round(max(0, min(100, fk)), 1)


def alliteration_density(text: str) -> float:
    """Fraction of consecutive word pairs sharing first letter."""
    words = [w.lower().strip(".,!?;:'\"") for w in text.split() if w]
    if len(words) < 2:
        return 0.0
    pairs = sum(1 for i in range(len(words) - 1)
                if words[i] and words[i+1] and words[i][0] == words[i+1][0])
    return round(pairs / (len(words) - 1), 3)


def passive_voice_pct(text: str) -> float:
    """Approximate % of sentences with passive voice."""
    sents = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    if not sents:
        return 0.0
    pat = re.compile(r"\b(was|were|been|being|is|are|am)\s+\w+ed\b", re.I)
    return round(sum(1 for s in sents if pat.search(s)) / len(sents) * 100, 1)


def unique_word_ratio(text: str) -> float:
    """Vocabulary richness: unique words / total words."""
    words = [w.lower().strip(".,!?;:'\"") for w in text.split() if w]
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 3)


def word_frequency(text: str, top_n: int = 10) -> list:
    """Most frequent non-stop words."""
    stops = {"the", "a", "an", "is", "are", "was", "were", "in", "on", "at",
             "to", "for", "of", "and", "or", "but", "it", "its", "i", "he",
             "she", "they", "we", "you", "my", "his", "her", "this", "that"}
    words = [w.lower().strip(".,!?;:'\"") for w in text.split()
             if w.lower().strip(".,!?;:'\"") not in stops]
    return Counter(words).most_common(top_n)


WRITING_FUNCTIONS = {
    "syllable_count": syllable_count,
    "line_syllables": line_syllables,
    "rhymes": rhymes,
    "stress_pattern": stress_pattern,
    "meter_type": meter_type,
    "haiku_valid": haiku_valid,
    "word_count": word_count,
    "sentence_count": sentence_count,
    "avg_sentence_len": avg_sentence_len,
    "readability_score": readability_score,
    "alliteration_density": alliteration_density,
    "passive_voice_pct": passive_voice_pct,
    "unique_word_ratio": unique_word_ratio,
    "word_frequency": word_frequency,
}

WRITING_NL_PATTERNS = [
    (r"(?:how many|count)\s+syllables?\s+in\s+", "syllable_count"),
    (r"does?\s+\w+\s+rhyme\s+with\s+\w+", "rhymes"),
    (r"(?:what|detect)\s+(?:is the )?meter\s+", "meter_type"),
    (r"(?:is this|check|validate)\s+(?:a )?\s*haiku", "haiku_valid"),
    (r"(?:word|sentence)\s+count", "word_count"),
    (r"readability\s+(?:score|level|grade)", "readability_score"),
    (r"alliteration", "alliteration_density"),
    (r"passive\s+voice", "passive_voice_pct"),
    (r"(?:vocabulary|lexical)\s+(?:richness|diversity)", "unique_word_ratio"),
    (r"stress\s+pattern", "stress_pattern"),
]
