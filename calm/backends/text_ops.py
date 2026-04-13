"""
CALM Text analysis backend — word count, readability metrics, text statistics.

Models estimate word counts, hallucinate reading times. Pure computation.
"""

from __future__ import annotations

import re
import math


def word_count(text: str) -> int:
    """Count words in text."""
    return len(str(text).split())


def char_count(text: str, include_spaces: bool = True) -> int:
    """Count characters in text."""
    t = str(text)
    return len(t) if include_spaces else len(t.replace(' ', ''))


def sentence_count(text: str) -> int:
    """Count sentences (split on . ! ?)."""
    return len(re.split(r'[.!?]+', str(text).strip())) - 1 or 1


def paragraph_count(text: str) -> int:
    """Count paragraphs (split on blank lines)."""
    paras = [p for p in str(text).split('\n\n') if p.strip()]
    return max(len(paras), 1)


def reading_time(text: str, wpm: int = 200) -> float:
    """Estimated reading time in minutes (default 200 WPM)."""
    words = word_count(text)
    return round(words / max(int(wpm), 1), 1)


def speaking_time(text: str, wpm: int = 130) -> float:
    """Estimated speaking time in minutes (default 130 WPM)."""
    words = word_count(text)
    return round(words / max(int(wpm), 1), 1)


def unique_words(text: str) -> int:
    """Count unique words (case-insensitive)."""
    words = re.findall(r'\b\w+\b', str(text).lower())
    return len(set(words))


def lexical_diversity(text: str) -> float:
    """Type-token ratio: unique words / total words."""
    words = re.findall(r'\b\w+\b', str(text).lower())
    if not words:
        return 0.0
    return round(len(set(words)) / len(words), 4)


def avg_word_length(text: str) -> float:
    """Average word length in characters."""
    words = re.findall(r'\b\w+\b', str(text))
    if not words:
        return 0.0
    return round(sum(len(w) for w in words) / len(words), 2)


def avg_sentence_length(text: str) -> float:
    """Average sentence length in words."""
    sentences = re.split(r'[.!?]+', str(text).strip())
    sentences = [s for s in sentences if s.strip()]
    if not sentences:
        return 0.0
    return round(sum(len(s.split()) for s in sentences) / len(sentences), 1)


def syllable_count(word: str) -> int:
    """Estimate syllable count for English word."""
    w = str(word).lower().strip()
    if not w:
        return 0
    if len(w) <= 3:
        return 1
    w = re.sub(r'(?:es|ed|e)$', '', w) or w
    vowels = re.findall(r'[aeiouy]+', w)
    return max(len(vowels), 1)


def flesch_reading_ease(text: str) -> float:
    """Flesch reading ease score (0-100, higher = easier)."""
    words = re.findall(r'\b\w+\b', str(text))
    if not words:
        return 0.0
    sentences = max(sentence_count(text), 1)
    syllables = sum(syllable_count(w) for w in words)
    total_words = len(words)
    score = 206.835 - 1.015 * (total_words / sentences) - 84.6 * (syllables / total_words)
    return round(max(0, min(100, score)), 1)


def flesch_grade_level(text: str) -> float:
    """Flesch-Kincaid grade level (US school grade)."""
    words = re.findall(r'\b\w+\b', str(text))
    if not words:
        return 0.0
    sentences = max(sentence_count(text), 1)
    syllables = sum(syllable_count(w) for w in words)
    total_words = len(words)
    grade = 0.39 * (total_words / sentences) + 11.8 * (syllables / total_words) - 15.59
    return round(max(0, grade), 1)


def line_count(text: str) -> int:
    """Count lines in text."""
    return len(str(text).splitlines()) or 1


def most_common_words(text: str, n: int = 10) -> list[tuple]:
    """N most common words (case-insensitive)."""
    from collections import Counter
    words = re.findall(r'\b\w+\b', str(text).lower())
    # Filter stopwords
    stops = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
             'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
             'would', 'could', 'should', 'may', 'might', 'can', 'shall',
             'and', 'or', 'but', 'if', 'in', 'on', 'at', 'to', 'for',
             'of', 'with', 'by', 'from', 'it', 'its', 'this', 'that',
             'not', 'no', 'as', 'i', 'we', 'you', 'he', 'she', 'they'}
    filtered = [w for w in words if w not in stops and len(w) > 1]
    return Counter(filtered).most_common(int(n))


TEXT_FUNCTIONS = {
    "word_count": word_count,
    "char_count": char_count,
    "sentence_count": sentence_count,
    "paragraph_count": paragraph_count,
    "reading_time": reading_time,
    "speaking_time": speaking_time,
    "unique_words": unique_words,
    "lexical_diversity": lexical_diversity,
    "avg_word_length": avg_word_length,
    "avg_sentence_length": avg_sentence_length,
    "syllable_count": syllable_count,
    "flesch_reading_ease": flesch_reading_ease,
    "flesch_grade_level": flesch_grade_level,
    "line_count": line_count,
    "most_common_words": most_common_words,
}

TEXT_NL_PATTERNS = [
    (r'(?:how many|count|number of)\s+words?\s+in', None),
    (r'(?:how many|count|number of)\s+(?:characters?|chars?)\s+in', None),
    (r'(?:how many|count|number of)\s+sentences?\s+in', None),
    (r'(?:reading|read)\s+time\s+(?:of|for)', None),
    (r'(?:Flesch|readability|reading ease|grade level)\s+(?:of|for|score)', None),
    (r'(?:most common|frequent)\s+words?\s+in', None),
]
