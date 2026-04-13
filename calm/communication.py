"""
Auto-CALM Communication Adaptation — match response style to audience.

Detect user expertise level and communication preferences from their
messages, and adjust response style accordingly. A senior engineer
gets different output than a student.

Usage:
    from calm.communication import CommunicationAdapter
    ca = CommunicationAdapter()
    profile = ca.analyze_user("Can you explain what a closure is?")
    style = ca.recommend_style(profile)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Dict


@dataclass
class UserProfile:
    """Detected user communication preferences."""
    expertise: str = "unknown"    # "beginner", "intermediate", "advanced", "expert"
    style: str = "unknown"       # "terse", "detailed", "conversational"
    domain_signals: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def summary(self) -> str:
        return f"expertise={self.expertise}, style={self.style}, confidence={self.confidence:.0%}"


@dataclass
class StyleRecommendation:
    """Recommended response style."""
    verbosity: str = "medium"      # "terse", "medium", "detailed"
    use_jargon: bool = True
    include_examples: bool = True
    include_analogies: bool = False
    include_caveats: bool = True
    code_comments: str = "minimal" # "none", "minimal", "verbose"
    explanation_depth: str = "mechanism"  # "surface", "mechanism", "root_cause"

    def summary(self) -> str:
        return (f"verbosity={self.verbosity}, jargon={'yes' if self.use_jargon else 'no'}, "
                f"examples={'yes' if self.include_examples else 'no'}, depth={self.explanation_depth}")


# Expertise signals
_EXPERT_SIGNALS = [
    re.compile(r'\b(?:impl|async|mutex|semaphore|vtable|ABI|ISA|SIMD|FFI|monomorphization)\b'),
    re.compile(r'\b(?:eigenvalue|eigenvector|Jacobian|gradient descent|backprop|convolution)\b'),
    re.compile(r'\b(?:CAP theorem|consensus|Raft|Paxos|linearizability|eventual consistency)\b'),
    re.compile(r'\b(?:perf|strace|gdb|valgrind|flamegraph|syscall|eBPF)\b'),
    re.compile(r'\b(?:RFC \d+|IEEE \d+|POSIX|SUS)\b'),
]

_ADVANCED_SIGNALS = [
    re.compile(r'\b(?:architecture|microservice|distributed|sharding|replication|partitioning)\b', re.IGNORECASE),
    re.compile(r'\b(?:dependency injection|design pattern|SOLID|DDD|hexagonal|clean architecture)\b', re.IGNORECASE),
    re.compile(r'\b(?:profil|benchmark|optimization|cache miss|branch prediction|pipeline)\b', re.IGNORECASE),
    re.compile(r'\b(?:k8s|terraform|helm|istio|envoy|prometheus|grafana)\b', re.IGNORECASE),
    re.compile(r'\b(?:amortized|asymptotic|big.o|time complexity|space complexity|load factor|hash.?table|B.tree|red.black)\b', re.IGNORECASE),
    re.compile(r'\b(?:concurrency|mutex|deadlock|race condition|lock.free|wait.free|CAS|compare.and.swap)\b', re.IGNORECASE),
]

_INTERMEDIATE_SIGNALS = [
    re.compile(r'\b(?:class|interface|abstract|polymorphism|inheritance|encapsulation)\b', re.IGNORECASE),
    re.compile(r'\b(?:REST|API|endpoint|middleware|ORM|migration|schema)\b', re.IGNORECASE),
    re.compile(r'\b(?:git|branch|merge|rebase|pull request|CI/CD|deploy)\b', re.IGNORECASE),
    re.compile(r'\b(?:docker|container|nginx|redis|postgres|mongodb)\b', re.IGNORECASE),
]

_BEGINNER_SIGNALS = [
    re.compile(r'\b(?:what is|explain|how does|can you tell me|I.m new|learning|beginner|tutorial)\b', re.IGNORECASE),
    re.compile(r'\b(?:what.s the difference|why do we|when should I|is it possible)\b', re.IGNORECASE),
    re.compile(r'\b(?:for loop|variable|function|if statement|array|string|print|hello world)\b', re.IGNORECASE),
]

# Style signals
_TERSE_SIGNALS = [
    re.compile(r'^.{5,30}[?.]$'),  # Very short message
    re.compile(r'\b(?:tldr|just|quickly|short|brief)\b', re.IGNORECASE),
    re.compile(r'^(?:how|what|why|when|where)\s+', re.IGNORECASE),  # Direct question
]

_DETAILED_SIGNALS = [
    re.compile(r'\b(?:explain|walk me through|step by step|in detail|thorough|comprehensive)\b', re.IGNORECASE),
    re.compile(r'\b(?:I want to understand|help me learn|teach me|can you elaborate)\b', re.IGNORECASE),
]


class CommunicationAdapter:
    """Adapts communication style to the user."""

    def analyze_user(self, *messages: str) -> UserProfile:
        """Analyze user messages to build a profile."""
        profile = UserProfile()
        text = " ".join(messages)

        # Detect expertise
        expert_score = sum(len(pat.findall(text)) for pat in _EXPERT_SIGNALS)
        advanced_score = sum(len(pat.findall(text)) for pat in _ADVANCED_SIGNALS)
        intermediate_score = sum(len(pat.findall(text)) for pat in _INTERMEDIATE_SIGNALS)
        beginner_score = sum(len(pat.findall(text)) for pat in _BEGINNER_SIGNALS)

        # Expert/advanced signals override beginner phrasing — an expert
        # asking "what is X" where X is expert jargon is still an expert.
        if expert_score > 0:
            beginner_score = 0
        if advanced_score > 0 and beginner_score <= 1:
            beginner_score = 0

        scores = {
            "expert": expert_score * 3,
            "advanced": advanced_score * 2,
            "intermediate": intermediate_score,
            "beginner": beginner_score,
        }
        total = sum(scores.values())
        if total > 0:
            profile.expertise = max(scores, key=scores.get)
            profile.confidence = scores[profile.expertise] / total
        else:
            profile.expertise = "intermediate"  # safe default
            profile.confidence = 0.3

        # Detect style preference
        terse_score = sum(1 for pat in _TERSE_SIGNALS if pat.search(text))
        detailed_score = sum(1 for pat in _DETAILED_SIGNALS if pat.search(text))

        if terse_score > detailed_score:
            profile.style = "terse"
        elif detailed_score > terse_score:
            profile.style = "detailed"
        else:
            profile.style = "conversational"

        return profile

    def recommend_style(self, profile: UserProfile) -> StyleRecommendation:
        """Recommend a response style based on user profile."""
        rec = StyleRecommendation()

        if profile.expertise == "expert":
            rec.verbosity = "terse"
            rec.use_jargon = True
            rec.include_examples = False
            rec.include_analogies = False
            rec.code_comments = "none"
            rec.explanation_depth = "root_cause"
        elif profile.expertise == "advanced":
            rec.verbosity = "medium"
            rec.use_jargon = True
            rec.include_examples = True
            rec.include_analogies = False
            rec.code_comments = "minimal"
            rec.explanation_depth = "mechanism"
        elif profile.expertise == "intermediate":
            rec.verbosity = "medium"
            rec.use_jargon = True
            rec.include_examples = True
            rec.include_analogies = True
            rec.code_comments = "minimal"
            rec.explanation_depth = "mechanism"
        else:  # beginner
            rec.verbosity = "detailed"
            rec.use_jargon = False
            rec.include_examples = True
            rec.include_analogies = True
            rec.code_comments = "verbose"
            rec.explanation_depth = "surface"

        # Override with style preference
        if profile.style == "terse":
            rec.verbosity = "terse"
            rec.include_analogies = False
        elif profile.style == "detailed":
            rec.verbosity = "detailed"

        return rec
