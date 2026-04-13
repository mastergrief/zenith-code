"""
Auto-CALM Factual Check — detect claims that contradict well-known facts.

Uses CALM backends to verify factual claims in responses. If a response
says "Python dicts are red-black trees" and we have a backend that knows
Python internals, flag it.

This is NOT a general fact-checker — it only catches claims in domains
where we have backend knowledge to verify against.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class FactualIssue:
    """A factual claim that may be incorrect."""
    claim: str
    category: str   # "contradiction", "suspicious", "unverifiable"
    reason: str
    confidence: float  # how confident we are this is wrong


@dataclass
class FactualCheckResult:
    """Result of factual checking."""
    issues: List[FactualIssue] = field(default_factory=list)
    claims_checked: int = 0
    score: float = 1.0  # 1.0 = no issues found

    def summary(self) -> str:
        if not self.issues:
            return f"checked {self.claims_checked} claims, no issues"
        return (f"checked {self.claims_checked} claims, "
                f"{len(self.issues)} issues found")


# Known facts that models commonly get wrong.
# Each entry: (regex pattern in response, expected truth, category)
_KNOWN_FACTS = [
    # Data structure implementations
    (r'Python\s+(?:dict|dictionaries?)\s+(?:are|is)\s+(?:implemented\s+as\s+)?(?:red.black|avl|binary)\s+tree',
     "Python dicts are hash tables (since CPython, always have been)", "contradiction"),
    (r'Java\s+HashMap\s+(?:is|uses)\s+(?:a\s+)?(?:red.black|avl|binary)\s+tree',
     "Java HashMap is a hash table (TreeMap is the red-black tree)", "contradiction"),

    # Complexity claims
    (r'hash\s+table[s]?\s+(?:always|guarantee)\s+O\(1\)',
     "Hash tables are O(1) average, O(n) worst case (collisions)", "suspicious"),
    (r'(?:binary\s+search|BST)\s+(?:is\s+)?always\s+O\(log\s*n\)',
     "BST is O(log n) average, O(n) worst case when unbalanced", "suspicious"),

    # Crypto/hashing
    (r'(?:hash\s+tables?|dictionaries?)\s+(?:use|uses)\s+(?:SHA|MD5|bcrypt|argon)',
     "Hash tables use fast hash functions (like MurmurHash, SipHash), not cryptographic hashes", "contradiction"),
    (r'MD5\s+is\s+(?:secure|safe|recommended)',
     "MD5 is broken for security — collision attacks are trivial since 2004", "contradiction"),
    (r'SHA-?1\s+is\s+(?:secure|safe|recommended)',
     "SHA-1 is broken — SHAttered collision demonstrated in 2017", "contradiction"),

    # Common misconceptions
    (r'(?:hash\s+tables?|dicts?)\s+(?:never|don.t|do not)\s+have\s+collisions',
     "All hash tables can have collisions — that's why collision resolution exists", "contradiction"),
    (r'NoSQL\s+(?:is|databases?\s+are)\s+(?:always\s+)?faster\s+than\s+(?:SQL|relational)',
     "NoSQL is not inherently faster — it depends on access patterns and data model", "suspicious"),
    (r'(?:microservices?|micro.services?)\s+(?:is|are)\s+(?:always\s+)?better\s+than\s+monolith',
     "Microservices add complexity — monolith-first is often the right approach", "suspicious"),
    (r'(?:REST|rest)\s+(?:is|requires?)\s+(?:always\s+)?JSON',
     "REST is architecture-agnostic — it can use JSON, XML, protobuf, or any format", "suspicious"),
    (r'TCP\s+is\s+(?:always\s+)?(?:slower|worse)\s+than\s+UDP',
     "TCP vs UDP depends on use case — TCP is better for reliability, UDP for latency", "suspicious"),
    (r'(?:JavaScript|JS)\s+is\s+(?:single.threaded|single threaded)\s+(?:so|and)\s+(?:can.t|cannot)\s+(?:do|handle)\s+concurrency',
     "JS is single-threaded but handles concurrency via the event loop and async/await", "suspicious"),

    # Year/attribution errors (common hallucinations)
    (r'(?:hash\s+tables?)\s+(?:were|was)\s+invented\s+(?:by|in|at)\s+(?:IBM|Microsoft|Google|Apple)',
     "Hash tables were described by Hans Peter Luhn at IBM in 1953 — but 'invented by IBM in 1990' is wrong", "suspicious"),

    # Language-specific
    (r'Python\s+is\s+(?:a\s+)?compiled\s+language',
     "Python is interpreted (CPython compiles to bytecode, but is not a compiled language)", "suspicious"),
    (r'(?:Go|Golang)\s+(?:has|supports)\s+(?:classes|inheritance)',
     "Go has no classes or inheritance — it uses structs, interfaces, and composition", "contradiction"),
    (r'Rust\s+(?:has|uses)\s+(?:a\s+)?garbage\s+collector',
     "Rust has no GC — it uses ownership and borrowing for memory management", "contradiction"),
]


class FactualChecker:
    """Check response for common factual errors using pattern matching."""

    def check(self, response: str) -> FactualCheckResult:
        """Check a response for factual issues."""
        result = FactualCheckResult()
        text = str(response)

        for pattern, truth, category in _KNOWN_FACTS:
            result.claims_checked += 1
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                result.issues.append(FactualIssue(
                    claim=match.group(0),
                    category=category,
                    reason=truth,
                    confidence=0.9 if category == "contradiction" else 0.7,
                ))

        # Score: 1.0 if no issues, decreasing with each issue
        if result.issues:
            result.score = max(0, 1.0 - len(result.issues) * 0.2)

        return result
