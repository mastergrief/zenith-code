"""Multi-step reasoning knowledge backend.

Factual lookups for valid inference patterns, logical fallacies,
and common reasoning rules. CALM's factual_check layer uses these
to catch invalid reasoning in model output.
"""

from __future__ import annotations

_DATA_VERSION = "2026-04-16"

# Valid syllogism forms (Aristotelian)
_VALID_SYLLOGISMS = {
    "barbara": {"form": "All M are P. All S are M. ∴ All S are P.", "figure": 1, "mood": "AAA"},
    "celarent": {"form": "No M are P. All S are M. ∴ No S are P.", "figure": 1, "mood": "EAE"},
    "darii": {"form": "All M are P. Some S are M. ∴ Some S are P.", "figure": 1, "mood": "AII"},
    "ferio": {"form": "No M are P. Some S are M. ∴ Some S are not P.", "figure": 1, "mood": "EIO"},
    "camestres": {"form": "All P are M. No S are M. ∴ No S are P.", "figure": 2, "mood": "AEE"},
    "festino": {"form": "No P are M. Some S are M. ∴ Some S are not P.", "figure": 2, "mood": "EIO"},
    "baroco": {"form": "All P are M. Some S are not M. ∴ Some S are not P.", "figure": 2, "mood": "AOO"},
    "datisi": {"form": "All M are P. Some M are S. ∴ Some S are P.", "figure": 3, "mood": "AII"},
    "disamis": {"form": "Some M are P. All M are S. ∴ Some S are P.", "figure": 3, "mood": "IAI"},
    "ferison": {"form": "No M are P. Some M are S. ∴ Some S are not P.", "figure": 3, "mood": "EIO"},
}

# Common logical fallacies
_FALLACIES = {
    "affirming_consequent": {
        "pattern": "If P then Q. Q is true. Therefore P.",
        "why_wrong": "Q could be caused by other things, not just P.",
        "example": "If it rains, the ground is wet. The ground is wet. Therefore it rained. (Sprinkler could cause it.)",
    },
    "denying_antecedent": {
        "pattern": "If P then Q. P is false. Therefore Q is false.",
        "why_wrong": "Q could be true for other reasons.",
        "example": "If I study, I pass. I didn't study. Therefore I didn't pass. (Could pass anyway.)",
    },
    "undistributed_middle": {
        "pattern": "All A are B. All C are B. Therefore all A are C.",
        "why_wrong": "B contains both A and C but they may not overlap.",
        "example": "All dogs are animals. All cats are animals. Therefore all dogs are cats.",
    },
    "hasty_generalization": {
        "pattern": "X is true for sample S. Therefore X is true for population P.",
        "why_wrong": "Sample may not be representative.",
        "example": "I met two rude people from city X. Everyone from city X is rude.",
    },
    "false_dilemma": {
        "pattern": "Either A or B. Not A. Therefore B.",
        "why_wrong": "Assumes only two options exist when more may exist.",
        "example": "You're either with us or against us.",
    },
    "circular_reasoning": {
        "pattern": "P is true because Q is true. Q is true because P is true.",
        "why_wrong": "The conclusion is used as a premise.",
        "example": "The Bible is true because it's the word of God. We know it's the word of God because the Bible says so.",
    },
}

# Transitive relations (if aRb and bRc then aRc)
_TRANSITIVE_RELATIONS = ["greater_than", "less_than", "equal_to", "ancestor_of",
                          "subset_of", "implies", "older_than", "taller_than",
                          "heavier_than", "faster_than"]

# Non-transitive relations (aRb and bRc does NOT imply aRc)
_NON_TRANSITIVE_RELATIONS = ["friend_of", "enemy_of", "sibling_of", "parent_of",
                              "loves", "beats"]  # rock-paper-scissors


def valid_syllogism_forms() -> dict:
    """Return all valid Aristotelian syllogism forms."""
    return _VALID_SYLLOGISMS


def get_syllogism(name: str) -> dict | None:
    """Look up a specific syllogism form by name."""
    return _VALID_SYLLOGISMS.get(name.lower())


def logical_fallacies() -> dict:
    """Return common logical fallacies with patterns and examples."""
    return _FALLACIES


def get_fallacy(name: str) -> dict | None:
    """Look up a specific fallacy by name."""
    return _FALLACIES.get(name.lower())


def is_transitive(relation: str) -> bool:
    """Check if a relation is transitive."""
    return relation.lower() in _TRANSITIVE_RELATIONS


def transitive_relations() -> list:
    """List all known transitive relations."""
    return list(_TRANSITIVE_RELATIONS)


def non_transitive_relations() -> list:
    """List all known non-transitive relations."""
    return list(_NON_TRANSITIVE_RELATIONS)


REASONING_KB_FUNCTIONS = {
    "valid_syllogism_forms": valid_syllogism_forms,
    "get_syllogism": get_syllogism,
    "logical_fallacies": logical_fallacies,
    "get_fallacy": get_fallacy,
    "is_transitive": is_transitive,
    "transitive_relations": transitive_relations,
    "non_transitive_relations": non_transitive_relations,
}

REASONING_KB_NL_PATTERNS = [
    (r"(?:valid|aristotelian)\s+syllogism", "valid_syllogism_forms"),
    (r"(?:what is|explain)\s+(?:a |the )?\w+\s+(?:syllogism|fallacy)", "get_syllogism"),
    (r"(?:logical|common)\s+fallac", "logical_fallacies"),
    (r"is\s+\w+\s+(?:a )?transitive\s+relation", "is_transitive"),
    (r"(?:transitive|non-transitive)\s+relations?", "transitive_relations"),
]
