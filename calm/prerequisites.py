"""
Auto-CALM Prerequisite Detection — "before you can understand X, you need Y."

Detects when an explanation assumes knowledge the user might not have.
Uses the communication adapter's expertise profile to determine what
needs prerequisite explanation.

Usage:
    from calm.prerequisites import PrerequisiteDetector
    pd = PrerequisiteDetector()
    result = pd.check("Use dependency injection to decouple the service layer",
                       user_expertise="beginner")
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set, Dict


@dataclass
class Prerequisite:
    """A concept that should be explained first."""
    concept: str
    definition: str
    complexity: str    # "basic", "intermediate", "advanced"
    needed_for: str    # what it's a prerequisite for


@dataclass
class PrerequisiteResult:
    """Result of prerequisite analysis."""
    prerequisites: List[Prerequisite] = field(default_factory=list)
    concepts_used: List[str] = field(default_factory=list)
    user_level: str = "intermediate"
    gap_count: int = 0

    def summary(self) -> str:
        if not self.prerequisites:
            return "no prerequisite gaps detected"
        return (f"{self.gap_count} prerequisite gaps for {self.user_level} user: "
                f"{', '.join(p.concept for p in self.prerequisites[:5])}")


# Concept → (definition, complexity, prerequisites)
_CONCEPT_DB = {
    # Basic concepts
    "variable": ("A named storage location for data", "basic", []),
    "function": ("A reusable block of code that performs a task", "basic", ["variable"]),
    "loop": ("A construct that repeats code", "basic", ["variable"]),
    "array": ("An ordered collection of elements", "basic", ["variable"]),
    "string": ("A sequence of characters", "basic", ["variable"]),
    "boolean": ("A true/false value", "basic", ["variable"]),
    "conditional": ("Code that runs only when a condition is met", "basic", ["boolean"]),
    # Intermediate concepts
    "class": ("A blueprint for creating objects", "intermediate", ["function", "variable"]),
    "object": ("An instance of a class", "intermediate", ["class"]),
    "interface": ("A contract that defines methods without implementation", "intermediate", ["class"]),
    "inheritance": ("A class deriving properties from a parent class", "intermediate", ["class"]),
    "polymorphism": ("Objects of different types responding to the same message", "intermediate", ["inheritance", "interface"]),
    "encapsulation": ("Hiding internal state behind methods", "intermediate", ["class"]),
    "recursion": ("A function that calls itself", "intermediate", ["function"]),
    "callback": ("A function passed as an argument to another function", "intermediate", ["function"]),
    "closure": ("A function that captures variables from its enclosing scope", "intermediate", ["function", "scope"]),
    "api": ("Application Programming Interface — a contract for communication", "intermediate", ["function"]),
    "database": ("Organized storage for structured data", "intermediate", []),
    "sql": ("Structured Query Language for database queries", "intermediate", ["database"]),
    "http": ("Protocol for web communication (request/response)", "intermediate", []),
    "rest": ("Architectural style for web APIs using HTTP methods", "intermediate", ["http", "api"]),
    "json": ("JavaScript Object Notation — data interchange format", "intermediate", []),
    "git": ("Version control system for tracking code changes", "intermediate", []),
    # Advanced concepts
    "dependency injection": ("Providing dependencies to a class from outside", "advanced", ["class", "interface", "encapsulation"]),
    "inversion of control": ("Framework calls your code, not the other way around", "advanced", ["dependency injection"]),
    "middleware": ("Code that runs between request and response", "advanced", ["http", "function"]),
    "orm": ("Object-Relational Mapping — database rows as objects", "advanced", ["class", "sql", "database"]),
    "microservice": ("Small, independent service that does one thing", "advanced", ["api", "http"]),
    "event driven": ("Architecture where components communicate via events", "advanced", ["callback"]),
    "concurrency": ("Multiple tasks making progress simultaneously", "advanced", ["function"]),
    "mutex": ("Mutual exclusion lock for thread safety", "advanced", ["concurrency"]),
    "deadlock": ("Two threads waiting for each other forever", "advanced", ["mutex", "concurrency"]),
    "race condition": ("Bug from unsynchronized concurrent access", "advanced", ["concurrency"]),
    "design pattern": ("Reusable solution to a common software problem", "advanced", ["class", "interface"]),
    "solid": ("Five principles for maintainable OOP code", "advanced", ["class", "interface", "dependency injection"]),
    "ci/cd": ("Continuous Integration / Continuous Deployment", "advanced", ["git"]),
    "docker": ("Container platform for packaging applications", "advanced", []),
    "kubernetes": ("Container orchestration platform", "advanced", ["docker", "microservice"]),
}

_LEVEL_THRESHOLDS = {
    "beginner": {"basic"},
    "intermediate": {"basic", "intermediate"},
    "advanced": {"basic", "intermediate", "advanced"},
    "expert": {"basic", "intermediate", "advanced"},  # experts know everything
}


class PrerequisiteDetector:
    """Detects when explanations assume knowledge the user might not have."""

    def check(self, text: str, user_expertise: str = "intermediate") -> PrerequisiteResult:
        """Check text for concepts that exceed the user's level."""
        result = PrerequisiteResult(user_level=user_expertise)
        known_levels = _LEVEL_THRESHOLDS.get(user_expertise, {"basic", "intermediate"})

        # Find all concepts mentioned in the text
        text_lower = text.lower()
        for concept, (definition, complexity, prereqs) in _CONCEPT_DB.items():
            if concept in text_lower:
                result.concepts_used.append(concept)

                # Is this concept above the user's level?
                if complexity not in known_levels:
                    result.prerequisites.append(Prerequisite(
                        concept=concept,
                        definition=definition,
                        complexity=complexity,
                        needed_for=concept,
                    ))
                    result.gap_count += 1

                # Are any prerequisites of this concept above user's level?
                for prereq in prereqs:
                    if prereq in _CONCEPT_DB:
                        prereq_complexity = _CONCEPT_DB[prereq][1]
                        if prereq_complexity not in known_levels:
                            result.prerequisites.append(Prerequisite(
                                concept=prereq,
                                definition=_CONCEPT_DB[prereq][0],
                                complexity=prereq_complexity,
                                needed_for=concept,
                            ))
                            result.gap_count += 1

        # Deduplicate
        seen = set()
        unique = []
        for p in result.prerequisites:
            if p.concept not in seen:
                seen.add(p.concept)
                unique.append(p)
        result.prerequisites = unique
        result.gap_count = len(unique)

        return result
