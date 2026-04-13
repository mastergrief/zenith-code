"""
CALM creativity backend — computed creative exploration.

Creativity isn't magic — it's search in a large possibility space.
This backend systematically explores combinations, analogies,
and constraint relaxations that the model might not try on its own.

The model provides domain knowledge. The engine explores it.

Functions: brainstorm, combine, analogize, constrain, diverge, score.
"""

from __future__ import annotations

import hashlib
import random
import re
from itertools import combinations, product
from typing import List, Optional


# ---------------------------------------------------------------------------
# Brainstorming — systematic candidate generation
# ---------------------------------------------------------------------------

def brainstorm_names(prefix: str, purpose: str, n: int = 10) -> list:
    """Generate function/variable name candidates from semantic components.
    Combines action verbs with purpose nouns systematically.

    Example: brainstorm_names("", "user authentication", 10)
    → ["verify_user", "authenticate_user", "check_auth", ...]
    """
    actions = [
        "get", "set", "create", "update", "delete", "find", "check",
        "validate", "compute", "build", "parse", "format", "convert",
        "load", "save", "fetch", "send", "process", "handle", "run",
        "verify", "ensure", "resolve", "detect", "extract", "generate",
        "apply", "filter", "merge", "split", "transform", "normalize",
    ]

    # Extract key words from purpose.
    words = [w.lower() for w in re.findall(r'[a-zA-Z]+', purpose) if len(w) > 2]

    candidates = []
    for action in actions:
        for word in words:
            name = f"{prefix}{action}_{word}" if prefix else f"{action}_{word}"
            candidates.append(name)

    # Also try compound nouns.
    if len(words) >= 2:
        for w1, w2 in combinations(words, 2):
            for action in actions[:10]:
                candidates.append(f"{action}_{w1}_{w2}")

    # Deduplicate and score by brevity + clarity.
    seen = set()
    unique = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique.append(c)

    # Sort by a heuristic: shorter names with common verbs first.
    unique.sort(key=lambda c: (len(c), c))
    return unique[:int(n)]


def brainstorm_approaches(problem: str, domain: str = "code") -> list:
    """Generate solution approaches by systematic pattern exploration.

    Returns a list of {approach, pattern, rationale} dicts.
    """
    # Known solution patterns by domain.
    _PATTERNS = {
        "code": [
            ("divide and conquer", "Split the problem into sub-problems, solve each, combine results"),
            ("cache/memoize", "Store computed results to avoid redundant work"),
            ("iterator/generator", "Process items one at a time instead of all at once"),
            ("builder pattern", "Construct complex objects step by step"),
            ("strategy pattern", "Define a family of algorithms, make them interchangeable"),
            ("observer pattern", "Notify dependents automatically when state changes"),
            ("pipeline", "Chain transformations, each step's output is next step's input"),
            ("state machine", "Model the problem as states and transitions"),
            ("recursive descent", "Process nested structures by recursive decomposition"),
            ("two-pointer", "Use two indices to scan a sequence efficiently"),
            ("sliding window", "Maintain a window over sequential data"),
            ("hash map lookup", "Trade space for time with O(1) lookups"),
            ("binary search", "Divide sorted space in half each step"),
            ("topological sort", "Order items respecting dependencies"),
            ("union-find", "Track connected components efficiently"),
        ],
        "architecture": [
            ("monolith first", "Start simple, split when you have evidence of where to split"),
            ("microservices", "Independent services communicating via APIs"),
            ("event sourcing", "Store events instead of state, derive state from events"),
            ("CQRS", "Separate read and write models for different optimization"),
            ("hexagonal/ports-adapters", "Core logic isolated from external dependencies"),
            ("serverless", "Functions triggered by events, no server management"),
            ("actor model", "Isolated actors communicating via messages"),
            ("pub/sub", "Decouple producers and consumers via message topics"),
        ],
        "data": [
            ("normalize", "Remove redundancy, ensure consistency"),
            ("denormalize", "Duplicate data for read performance"),
            ("index", "Add lookup structures for common queries"),
            ("partition", "Split data across shards for scale"),
            ("batch process", "Process in bulk during off-peak"),
            ("stream process", "Process in real-time as data arrives"),
            ("materialized view", "Pre-compute aggregations"),
        ],
    }

    patterns = _PATTERNS.get(domain, _PATTERNS["code"])

    # Score each pattern's relevance to the problem.
    problem_words = set(w.lower() for w in re.findall(r'[a-zA-Z]+', problem))

    results = []
    for name, rationale in patterns:
        pattern_words = set(w.lower() for w in re.findall(r'[a-zA-Z]+', name + " " + rationale))
        relevance = len(problem_words & pattern_words)
        results.append({
            "approach": name,
            "rationale": rationale,
            "relevance": relevance,
        })

    results.sort(key=lambda r: -r["relevance"])
    return results


# ---------------------------------------------------------------------------
# Combination — systematic recombination of ideas
# ---------------------------------------------------------------------------

def combine_concepts(concepts: list) -> list:
    """Systematically combine pairs of concepts.
    Returns [{combo, components}] for all unique pairs.

    Example: combine_concepts(["cache", "observer", "pipeline"])
    → [{"combo": "cache + observer", ...}, ...]
    """
    results = []
    for a, b in combinations(concepts, 2):
        results.append({
            "combo": f"{a} + {b}",
            "components": [a, b],
            "question": f"What if we applied {a} principles to {b}?",
        })
    return results


def combine_patterns(pattern_a: str, pattern_b: str) -> dict:
    """Describe what happens when two patterns are combined."""
    return {
        "combination": f"{pattern_a} × {pattern_b}",
        "question": f"What if {pattern_a} was implemented using {pattern_b}?",
        "inverse": f"What if {pattern_b} was implemented using {pattern_a}?",
    }


# ---------------------------------------------------------------------------
# Constraint exploration — what becomes possible?
# ---------------------------------------------------------------------------

def relax_constraints(constraints: list) -> list:
    """Systematically relax each constraint and ask what becomes possible.

    Example: relax_constraints(["must be real-time", "8GB VRAM", "Python only"])
    → [{removed: "must be real-time", question: "What if latency didn't matter?"}]
    """
    results = []
    for i, c in enumerate(constraints):
        others = [constraints[j] for j in range(len(constraints)) if j != i]
        results.append({
            "removed": c,
            "remaining": others,
            "question": f"What becomes possible if we drop '{c}'?",
        })
    return results


def invert_assumptions(assumptions: list) -> list:
    """Invert each assumption to explore opposite approaches.

    Example: invert_assumptions(["data fits in memory", "users are authenticated"])
    → [{original: "data fits in memory", inverted: "data does NOT fit in memory"}]
    """
    return [
        {
            "original": a,
            "inverted": f"What if {a.lower()} is FALSE?",
            "question": f"How would the design change if {a.lower()} were not true?",
        }
        for a in assumptions
    ]


# ---------------------------------------------------------------------------
# Divergent thinking — inject randomness
# ---------------------------------------------------------------------------

def random_approach(options: list, seed: int = None) -> str:
    """Pick a random approach from options.
    Useful when the model is stuck in a local optimum."""
    if seed is not None:
        random.seed(int(seed))
    return random.choice(options) if options else ""


def shuffle_priority(items: list, seed: int = None) -> list:
    """Randomly reorder items to break priority bias.
    Sometimes the best idea isn't the first one that comes to mind."""
    result = list(items)
    if seed is not None:
        random.seed(int(seed))
    random.shuffle(result)
    return result


# ---------------------------------------------------------------------------
# Novelty scoring — how different is this from what exists?
# ---------------------------------------------------------------------------

def score_novelty(idea: str, existing: list) -> dict:
    """Score how novel an idea is compared to existing ones.
    Uses character-level similarity (simple but effective).
    Returns {score: 0-100, most_similar, similarity}."""
    if not existing:
        return {"score": 100, "most_similar": None, "similarity": 0}

    idea_set = set(idea.lower().split())

    best_sim = 0
    best_match = ""
    for ex in existing:
        ex_set = set(ex.lower().split())
        if not idea_set or not ex_set:
            continue
        overlap = len(idea_set & ex_set)
        union = len(idea_set | ex_set)
        sim = overlap / union if union > 0 else 0
        if sim > best_sim:
            best_sim = sim
            best_match = ex

    novelty = int((1 - best_sim) * 100)
    return {
        "score": novelty,
        "most_similar": best_match,
        "similarity": round(best_sim, 2),
    }


def _deterministic_seed(text: str) -> int:
    """Generate a deterministic seed from text for reproducible randomness."""
    return int(hashlib.md5(text.encode()).hexdigest()[:8], 16)


CREATIVE_FUNCTIONS = {
    "brainstorm_names": brainstorm_names,
    "brainstorm_approaches": brainstorm_approaches,
    "combine_concepts": combine_concepts,
    "combine_patterns": combine_patterns,
    "relax_constraints": relax_constraints,
    "invert_assumptions": invert_assumptions,
    "random_approach": random_approach,
    "shuffle_priority": shuffle_priority,
    "score_novelty": score_novelty,
}

CREATIVE_NL_PATTERNS = [
    (r'(?:brainstorm|suggest|generate)\s+(?:names?|ideas?)\s+for', None),
    (r'(?:combine|merge|mix)\s+(?:concepts?|ideas?|patterns?)', None),
    (r'(?:invert|flip|reverse)\s+(?:the\s+)?assumptions?', None),
]
