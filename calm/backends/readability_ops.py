"""
CALM readability backend — turns "is this explanation clear?" into numbers.

Computes readability metrics on text: Flesch-Kincaid, vocabulary complexity,
sentence structure, jargon density. Makes subjective judgments verifiable.
"""

from __future__ import annotations

import re
from typing import List


def _syllable_count(word: str) -> int:
    """Estimate syllable count for a word."""
    word = word.lower().strip()
    if not word:
        return 0
    if len(word) <= 3:
        return 1
    # Remove trailing e.
    word = re.sub(r'e$', '', word)
    # Count vowel groups.
    vowels = re.findall(r'[aeiouy]+', word)
    count = max(1, len(vowels))
    return count


def _sentences(text: str) -> List[str]:
    """Split text into sentences."""
    return [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]


def _words(text: str) -> List[str]:
    """Extract words from text."""
    return [w for w in re.findall(r"[a-zA-Z']+", text) if len(w) > 0]


def flesch_kincaid(text: str) -> dict:
    """Flesch-Kincaid readability scores.
    Returns {reading_ease, grade_level, rating}."""
    words = _words(text)
    sentences = _sentences(text)
    if not words or not sentences:
        return {"reading_ease": 0, "grade_level": 0, "rating": "no text"}

    total_syllables = sum(_syllable_count(w) for w in words)
    avg_sentence_len = len(words) / len(sentences)
    avg_syllables = total_syllables / len(words)

    # Flesch Reading Ease.
    ease = 206.835 - 1.015 * avg_sentence_len - 84.6 * avg_syllables
    ease = round(max(0, min(100, ease)), 1)

    # Flesch-Kincaid Grade Level.
    grade = 0.39 * avg_sentence_len + 11.8 * avg_syllables - 15.59
    grade = round(max(0, grade), 1)

    rating = (
        "very easy (5th grade)" if ease >= 80 else
        "easy (6th-7th grade)" if ease >= 65 else
        "moderate (8th-9th grade)" if ease >= 50 else
        "difficult (college)" if ease >= 30 else
        "very difficult (graduate)"
    )

    return {
        "reading_ease": ease,
        "grade_level": grade,
        "rating": rating,
        "words": len(words),
        "sentences": len(sentences),
        "avg_sentence_length": round(avg_sentence_len, 1),
        "avg_syllables_per_word": round(avg_syllables, 2),
    }


# Common jargon/buzzword lists by domain.
_TECH_JARGON = {
    "leverage", "utilize", "synergy", "paradigm", "scalable", "robust",
    "granular", "holistic", "agile", "iterate", "pivoting", "disruptive",
    "blockchain", "ecosystem", "microservices", "serverless", "devops",
    "refactor", "abstraction", "polymorphism", "encapsulation", "middleware",
    "monolithic", "orthogonal", "idempotent", "deterministic", "heuristic",
}


def jargon_density(text: str, domain_words: list = None) -> dict:
    """Compute jargon/complexity density.
    Returns {density, jargon_count, total_words, jargon_found}."""
    words = _words(text)
    jargon_set = _TECH_JARGON
    if domain_words:
        jargon_set = jargon_set | set(w.lower() for w in domain_words)

    found = [w for w in words if w.lower() in jargon_set]
    density = len(found) / len(words) * 100 if words else 0

    rating = (
        "plain" if density < 2 else
        "moderate jargon" if density < 5 else
        "jargon-heavy" if density < 10 else
        "extremely dense"
    )

    return {
        "density": round(density, 1),
        "rating": rating,
        "jargon_count": len(found),
        "total_words": len(words),
        "jargon_found": list(set(w.lower() for w in found)),
    }


def vocabulary_complexity(text: str) -> dict:
    """Analyze vocabulary: long words, rare words, repetition.
    Returns {long_word_pct, unique_word_ratio, avg_word_length, rating}."""
    words = _words(text)
    if not words:
        return {"rating": "no text"}

    long_words = [w for w in words if len(w) > 8]
    unique = set(w.lower() for w in words)

    long_pct = len(long_words) / len(words) * 100
    unique_ratio = len(unique) / len(words)
    avg_len = sum(len(w) for w in words) / len(words)

    rating = (
        "simple" if avg_len < 5 and long_pct < 10 else
        "moderate" if avg_len < 6 and long_pct < 20 else
        "complex" if avg_len < 7 else
        "very complex"
    )

    return {
        "long_word_pct": round(long_pct, 1),
        "unique_word_ratio": round(unique_ratio, 2),
        "avg_word_length": round(avg_len, 1),
        "total_words": len(words),
        "rating": rating,
    }


def text_structure(text: str) -> dict:
    """Analyze text structure: paragraph count, list items, code blocks, headers.
    Returns {paragraphs, lists, code_blocks, headers, rating}."""
    paragraphs = len([p for p in text.split('\n\n') if p.strip()])
    lists = len(re.findall(r'^\s*[-*•]\s', text, re.MULTILINE))
    lists += len(re.findall(r'^\s*\d+[\.\)]\s', text, re.MULTILINE))
    code_blocks = len(re.findall(r'```', text)) // 2
    headers = len(re.findall(r'^#{1,6}\s', text, re.MULTILINE))

    has_structure = paragraphs > 1 or lists > 0 or code_blocks > 0 or headers > 0
    rating = (
        "well-structured" if (headers > 0 and (lists > 0 or code_blocks > 0)) else
        "has structure" if has_structure else
        "wall of text"
    )

    return {
        "paragraphs": paragraphs,
        "list_items": lists,
        "code_blocks": code_blocks,
        "headers": headers,
        "rating": rating,
    }


def readability_report(text: str) -> dict:
    """Full readability analysis — combines all metrics into one report."""
    fk = flesch_kincaid(text)
    jd = jargon_density(text)
    vc = vocabulary_complexity(text)
    ts = text_structure(text)

    # Overall score 0-100.
    score = 100
    if fk.get("reading_ease", 100) < 50: score -= 15
    elif fk.get("reading_ease", 100) < 30: score -= 30
    if jd.get("density", 0) > 5: score -= 10
    elif jd.get("density", 0) > 10: score -= 20
    if vc.get("rating") == "very complex": score -= 15
    if ts.get("rating") == "wall of text": score -= 20
    score = max(0, min(100, score))

    return {
        "score": score,
        "flesch_kincaid": fk,
        "jargon": jd,
        "vocabulary": vc,
        "structure": ts,
    }


READABILITY_FUNCTIONS = {
    "flesch_kincaid": flesch_kincaid,
    "jargon_density": jargon_density,
    "vocabulary_complexity": vocabulary_complexity,
    "text_structure": text_structure,
    "readability_report": readability_report,
}
