"""
Auto-CALM Disambiguation — detect ambiguous questions before answering.

Models confidently answer the wrong interpretation. This module detects
when a question has multiple valid interpretations and produces clarifying
questions instead of guessing.

Types of ambiguity:
  1. Lexical: "bank" (financial vs river), "table" (furniture vs database)
  2. Scope: "fix the bug" (which bug? in which file?)
  3. Reference: "it", "that", "the function" (which one?)
  4. Criteria: "best" (by what measure?)

Usage:
    from calm.disambiguation import Disambiguator
    d = Disambiguator()
    result = d.check("How do I fix the performance issue?")
    print(result.ambiguities)
    print(result.clarifying_questions)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Set


@dataclass
class Ambiguity:
    """A detected ambiguity in a question."""
    text: str              # the ambiguous part
    ambiguity_type: str    # "lexical", "scope", "reference", "criteria", "implicit"
    interpretations: List[str] = field(default_factory=list)
    clarifying_question: str = ""


@dataclass
class DisambiguationResult:
    """Result of disambiguation analysis."""
    ambiguities: List[Ambiguity] = field(default_factory=list)
    is_ambiguous: bool = False
    confidence: float = 1.0   # 0-1, how confident we are in the interpretation
    clarifying_questions: List[str] = field(default_factory=list)

    def summary(self) -> str:
        if not self.is_ambiguous:
            return "unambiguous"
        return (f"{len(self.ambiguities)} ambiguities detected "
                f"(confidence: {self.confidence:.0%})")


# Lexical ambiguity: words with multiple technical meanings
_LEXICAL_AMBIGUITY = {
    "table": (["database table", "HTML table", "data table / spreadsheet", "lookup table"],
              "Which kind of table? (database, HTML, data, lookup)"),
    "index": (["database index", "array index", "search index", "file index"],
              "Which kind of index? (database, array, search engine)"),
    "key": (["API key", "encryption key", "dictionary/map key", "primary key", "SSH key"],
            "Which kind of key? (API, encryption, dictionary, database, SSH)"),
    "branch": (["git branch", "code branch (if/else)", "organizational branch"],
               "Git branch or code branch (if/else)?"),
    "port": (["network port", "I/O port", "porting (migration)"],
             "Network port, hardware port, or porting/migration?"),
    "node": (["Node.js", "DOM node", "tree/graph node", "server node"],
             "Node.js, DOM element, data structure node, or server?"),
    "pool": (["connection pool", "thread pool", "memory pool", "resource pool"],
             "Which kind of pool? (connection, thread, memory)"),
    "token": (["auth token", "API token", "lexer token", "NLP token"],
              "Auth/API token or parser/NLP token?"),
    "link": (["hyperlink", "symbolic link", "linked list link", "linker (compilation)"],
             "Hyperlink, symlink, linked list, or compiler linker?"),
    "driver": (["device driver", "database driver", "test driver"],
               "Device driver, database driver, or test driver?"),
    "shell": (["Unix shell", "shell script", "outer shell/wrapper"],
              "Command-line shell or code wrapper?"),
    "log": (["log file / logging", "logarithm", "git log", "audit log"],
            "Application log, math logarithm, or git log?"),
    "model": (["ML model", "data model", "domain model", "MVC model"],
              "ML model, database schema, domain object, or MVC pattern?"),
    "pipe": (["Unix pipe", "data pipeline", "named pipe (FIFO)"],
             "Unix pipe (|), data pipeline, or named pipe?"),
    "service": (["microservice", "system service/daemon", "cloud service", "SaaS"],
                "Microservice, system daemon, or cloud/SaaS?"),
}

# Scope ambiguity patterns
_SCOPE_PATTERNS = [
    (re.compile(r'\b(?:fix|solve|resolve|handle)\s+(?:the|this|that)\s+(?:bug|issue|error|problem)\b', re.IGNORECASE),
     "scope", "Which specific bug/issue? What are the symptoms?"),
    (re.compile(r'\b(?:update|change|modify|edit)\s+(?:the|this|that)\s+(?:code|file|function|config)\b', re.IGNORECASE),
     "scope", "Which specific file/function/config? What change is needed?"),
    (re.compile(r'\b(?:improve|optimize|speed up|fix)\s+(?:the\s+)?performance\b', re.IGNORECASE),
     "scope", "Performance of what specifically? Which metric (latency, throughput, memory)?"),
    (re.compile(r'\b(?:add|implement|create|build)\s+(?:a\s+)?(?:feature|functionality)\b', re.IGNORECASE),
     "scope", "What feature specifically? What are the requirements?"),
    (re.compile(r'\b(?:deploy|release|ship|launch)\s+(?:the|this|it)\b', re.IGNORECASE),
     "scope", "Deploy to which environment? Using what process?"),
]

# Reference ambiguity: pronouns and vague references
_REFERENCE_PATTERNS = [
    (re.compile(r'\b(?:fix|change|update|delete|move)\s+it\b', re.IGNORECASE),
     "reference", "What does 'it' refer to?"),
    (re.compile(r'\b(?:that|this)\s+(?:doesn.t|isn.t|won.t|can.t)\s+work\b', re.IGNORECASE),
     "reference", "What specifically doesn't work? What's the expected vs actual behavior?"),
    (re.compile(r'\bthe (?:function|method|class|variable|component)\b(?!\s+\w+\()', re.IGNORECASE),
     "reference", "Which function/method/class specifically?"),
    (re.compile(r'\b(?:make|do)\s+(?:it|this|that)\s+(?:work|better|faster|right)\b', re.IGNORECASE),
     "reference", "What does 'it' refer to? What does 'better/faster/right' mean concretely?"),
]

# Criteria ambiguity: evaluative terms without criteria
_CRITERIA_PATTERNS = [
    (re.compile(r'\b(?:what.s|which is)\s+(?:the\s+)?best\b', re.IGNORECASE),
     "criteria", "Best by what criteria? (performance, ease of use, cost, community size)"),
    (re.compile(r'\bshould (?:I|we)\s+(?:use|choose|pick|go with)\b', re.IGNORECASE),
     "criteria", "What are your priorities? (performance, simplicity, team familiarity, cost)"),
    (re.compile(r'\b(?:good|bad|better|worse)\s+(?:way|approach|practice|idea)\b', re.IGNORECASE),
     "criteria", "Good/bad by what standard? What are the constraints?"),
    (re.compile(r'\b(?:right|correct|proper|appropriate)\s+way\b', re.IGNORECASE),
     "criteria", "Right for what context? What are the requirements?"),
]


class Disambiguator:
    """Detects ambiguous questions and suggests clarifications."""

    def check(self, text: str) -> DisambiguationResult:
        """Check text for ambiguities."""
        result = DisambiguationResult()
        text_lower = text.lower()

        # Lexical ambiguity
        for word, (interpretations, question) in _LEXICAL_AMBIGUITY.items():
            if re.search(r'\b' + re.escape(word) + r'\b', text_lower):
                # Check if context already disambiguates
                context_clues = self._check_context(word, text_lower, interpretations)
                if not context_clues:
                    result.ambiguities.append(Ambiguity(
                        text=word,
                        ambiguity_type="lexical",
                        interpretations=interpretations,
                        clarifying_question=question,
                    ))

        # Scope ambiguity
        for pat, atype, question in _SCOPE_PATTERNS:
            if pat.search(text):
                result.ambiguities.append(Ambiguity(
                    text=pat.pattern[:40],
                    ambiguity_type=atype,
                    clarifying_question=question,
                ))

        # Reference ambiguity
        for pat, atype, question in _REFERENCE_PATTERNS:
            if pat.search(text):
                result.ambiguities.append(Ambiguity(
                    text=pat.pattern[:40],
                    ambiguity_type=atype,
                    clarifying_question=question,
                ))

        # Criteria ambiguity
        for pat, atype, question in _CRITERIA_PATTERNS:
            if pat.search(text):
                result.ambiguities.append(Ambiguity(
                    text=pat.pattern[:40],
                    ambiguity_type=atype,
                    clarifying_question=question,
                ))

        # Deduplicate clarifying questions
        seen = set()
        for a in result.ambiguities:
            if a.clarifying_question not in seen:
                seen.add(a.clarifying_question)
                result.clarifying_questions.append(a.clarifying_question)

        result.is_ambiguous = len(result.ambiguities) > 0
        result.confidence = max(0.1, 1.0 - len(result.ambiguities) * 0.15)

        return result

    def _check_context(self, word: str, text: str, interpretations: List[str]) -> str:
        """Check if surrounding context disambiguates a word."""
        # Simple heuristic: if a specific interpretation keyword appears near the word,
        # consider it disambiguated
        context_clues = {
            "table": {"database": "database table", "html": "HTML table", "csv": "data table",
                       "sql": "database table", "create table": "database table",
                       "<table": "HTML table", "spreadsheet": "data table"},
            "key": {"api": "API key", "ssh": "SSH key", "primary": "primary key",
                     "encrypt": "encryption key", "dict": "dictionary key",
                     "map": "dictionary key", "foreign": "foreign key"},
            "index": {"database": "database index", "array": "array index",
                       "search": "search index", "elasticsearch": "search index",
                       "[": "array index", "create index": "database index"},
            "model": {"train": "ML model", "neural": "ML model", "schema": "data model",
                       "domain": "domain model", "mvc": "MVC model", "weights": "ML model"},
        }
        clues = context_clues.get(word, {})
        for clue, interpretation in clues.items():
            if clue in text:
                return interpretation
        return ""
