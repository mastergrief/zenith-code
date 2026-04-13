"""
Auto-CALM Confidence Calibration — knowing what you know.

Tracks prediction accuracy over time by domain. When the system
answers a question, it can report calibrated confidence based on
historical accuracy in that domain.

The self-learning loop already generates correction data. Calibration
is the statistical layer on top — "we're 95% accurate on arithmetic
but 40% on organic chemistry."

Usage:
    from calm.calibration import ConfidenceCalibrator
    cc = ConfidenceCalibrator()
    cc.record("math", correct=True)
    cc.record("math", correct=True)
    cc.record("math", correct=False)
    print(cc.confidence("math"))  # 0.667
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class DomainStats:
    """Accuracy statistics for a domain."""
    correct: int = 0
    incorrect: int = 0
    unverifiable: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.incorrect

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0.0

    @property
    def confidence(self) -> float:
        """Bayesian-adjusted confidence: (correct + 1) / (total + 2).
        Uses Laplace smoothing so new domains start at 0.5, not 0 or 1."""
        return (self.correct + 1) / (self.total + 2)


@dataclass
class ConfidenceReport:
    """Confidence assessment for a specific query."""
    domain: str
    confidence: float       # 0-1, calibrated
    sample_size: int        # how many observations this is based on
    accuracy: float         # raw accuracy (no smoothing)
    tier: str               # "high", "medium", "low", "unknown"
    explanation: str = ""

    def __str__(self):
        return f"[{self.tier}] {self.domain}: {self.confidence:.0%} confidence ({self.sample_size} observations)"


# Domain detection from question/content
_DOMAIN_PATTERNS = {
    "arithmetic": re.compile(r'\b(?:\d+\s*[\*×\+\-\/]\s*\d+|calculate|compute|sum|product)\b', re.IGNORECASE),
    "number_theory": re.compile(r'\b(?:prime|factor|divisor|gcd|lcm|fibonacci|collatz)\b', re.IGNORECASE),
    "geometry": re.compile(r'\b(?:area|volume|circle|sphere|triangle|radius|angle|perimeter|hypotenuse)\b', re.IGNORECASE),
    "probability": re.compile(r'\b(?:probability|chance|odds|dice|coin|binomial|combinations|permutations)\b', re.IGNORECASE),
    "dates": re.compile(r'\b(?:date|day|month|year|leap|calendar|timezone)\b', re.IGNORECASE),
    "encoding": re.compile(r'\b(?:base64|hex|sha256|md5|hash|encode|decode|url.encode)\b', re.IGNORECASE),
    "countries": re.compile(r'\b(?:capital|country|currency|population|ISO.code|calling.code)\b', re.IGNORECASE),
    "chemistry": re.compile(r'\b(?:element|atomic|periodic|molecule|chemical|compound|electron)\b', re.IGNORECASE),
    "physics": re.compile(r'\b(?:speed.of.light|planck|newton|gravity|force|energy|momentum)\b', re.IGNORECASE),
    "algorithms": re.compile(r'\b(?:quicksort|mergesort|binary.search|big.o|complexity|hash.table|bst|graph)\b', re.IGNORECASE),
    "networking": re.compile(r'\b(?:port|tcp|udp|http|dns|ip.address|subnet|cidr)\b', re.IGNORECASE),
    "security": re.compile(r'\b(?:xss|sql.injection|csrf|owasp|vulnerability|exploit|injection)\b', re.IGNORECASE),
    "licensing": re.compile(r'\b(?:license|gpl|mit|apache|copyleft|open.source|spdx)\b', re.IGNORECASE),
    "base_conversion": re.compile(r'\b(?:binary|hexadecimal|octal|base.\d+|roman.numeral)\b', re.IGNORECASE),
    "units": re.compile(r'\b(?:convert|celsius|fahrenheit|miles|kilometers|bytes|megabytes|gigabytes)\b', re.IGNORECASE),
}

DEFAULT_DB_PATH = Path("calm/.calibration.json")


class ConfidenceCalibrator:
    """Tracks accuracy by domain and produces calibrated confidence scores."""

    def __init__(self, db_path: Optional[Path] = None):
        self._stats: Dict[str, DomainStats] = defaultdict(DomainStats)
        self._db_path = db_path or DEFAULT_DB_PATH
        self._load()

    def _load(self):
        """Load calibration data from disk."""
        if not self._db_path.exists():
            return
        try:
            data = json.loads(self._db_path.read_text())
            for domain, stats in data.items():
                self._stats[domain] = DomainStats(
                    correct=stats.get("correct", 0),
                    incorrect=stats.get("incorrect", 0),
                    unverifiable=stats.get("unverifiable", 0),
                )
        except (json.JSONDecodeError, KeyError):
            pass

    def _save(self):
        """Persist calibration data to disk."""
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            domain: {
                "correct": s.correct,
                "incorrect": s.incorrect,
                "unverifiable": s.unverifiable,
            }
            for domain, s in self._stats.items()
        }
        self._db_path.write_text(json.dumps(data, indent=2))

    def detect_domain(self, text: str) -> str:
        """Detect the most likely domain for a text."""
        scores = {}
        for domain, pat in _DOMAIN_PATTERNS.items():
            matches = len(pat.findall(text))
            if matches > 0:
                scores[domain] = matches
        if scores:
            return max(scores, key=scores.get)
        return "general"

    def record(self, domain: str, correct: bool):
        """Record a verification result for a domain."""
        if correct:
            self._stats[domain].correct += 1
        else:
            self._stats[domain].incorrect += 1
        self._save()

    def record_unverifiable(self, domain: str):
        """Record that a claim in this domain couldn't be verified."""
        self._stats[domain].unverifiable += 1

    def confidence(self, domain: str) -> float:
        """Calibrated confidence for a domain (Laplace-smoothed)."""
        return self._stats[domain].confidence

    def assess(self, text: str) -> ConfidenceReport:
        """Full confidence assessment for a query."""
        domain = self.detect_domain(text)
        stats = self._stats[domain]
        conf = stats.confidence

        if stats.total < 5:
            tier = "unknown"
            explanation = f"insufficient data ({stats.total} observations)"
        elif conf >= 0.9:
            tier = "high"
            explanation = f"{stats.correct}/{stats.total} correct in this domain"
        elif conf >= 0.7:
            tier = "medium"
            explanation = f"{stats.correct}/{stats.total} correct — verify important claims"
        else:
            tier = "low"
            explanation = f"{stats.correct}/{stats.total} correct — high error rate, verify everything"

        return ConfidenceReport(
            domain=domain,
            confidence=conf,
            sample_size=stats.total,
            accuracy=stats.accuracy,
            tier=tier,
            explanation=explanation,
        )

    def record_from_verify_report(self, prompt: str, claims_ok: int, claims_wrong: int):
        """Record results from an Auto-CALM verify pass."""
        domain = self.detect_domain(prompt)
        self._stats[domain].correct += claims_ok
        self._stats[domain].incorrect += claims_wrong
        self._save()

    def all_domains(self) -> Dict[str, DomainStats]:
        """All tracked domain stats."""
        return dict(self._stats)

    def summary(self) -> str:
        """Summary of calibration state."""
        if not self._stats:
            return "no calibration data yet"
        lines = []
        for domain in sorted(self._stats, key=lambda d: self._stats[d].total, reverse=True):
            s = self._stats[domain]
            if s.total > 0:
                lines.append(f"  {domain}: {s.accuracy:.0%} ({s.correct}/{s.total})")
        return f"Calibration across {len(self._stats)} domains:\n" + "\n".join(lines)
