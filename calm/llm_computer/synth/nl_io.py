"""NL → IO parser.

Extract (a, b, out) triples from free-form English. Supported forms:

  Single-argument:
    "8 becomes 4, 6 becomes 3, 10 becomes 5"
    "8 -> 4, 6 -> 3, 10 -> 5"
    "8 = 4, 6 = 3, 10 = 5"
    "if I give you 8 you return 4, 6 returns 3, 10 returns 5"

  Two-argument:
    "3 and 5 give 8, 2 and 7 give 9, 4 and 6 give 10"
    "(3, 5) -> 8, (2, 7) -> 9, (4, 6) -> 10"
    "3+5=8, 2+7=9, 4+6=10"

  IO-style (pre-existing, kept for compatibility):
    "a=3 b=5: 8 | a=2 b=7: 9 | a=4 b=6: 10"

  Query:
    "what about 7?"         (single-arg query)
    "what about 3 and 4?"   (two-arg query)
    "? a=3 b=4"              (IO-style query)
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple

from calm.llm_computer.synth.data import SynthSample


# IO-style pairs: a=3 b=5 : 8
_IO_PAIR = re.compile(r"a\s*=\s*(-?\d+)\s+b\s*=\s*(-?\d+)\s*:\s*(-?\d+)")
_IO_QUERY = re.compile(r"\?\s*a\s*=\s*(-?\d+)\s+b\s*=\s*(-?\d+)")

# Two-argument NL: (3, 5) -> 8  OR  3 and 5 give 8  OR  3+5=8
_TWO_ARG_PAREN = re.compile(
    r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\s*(?:->|=>|→|gives?|=|is)\s*(-?\d+)"
)
_TWO_ARG_AND = re.compile(
    r"(-?\d+)\s+(?:and|,)\s+(-?\d+)\s+(?:gives?|make|makes|=|is|equals?|→|->|=>)\s+(-?\d+)"
)
_TWO_ARG_PLUS = re.compile(r"(-?\d+)\s*[+\-*]\s*(-?\d+)\s*=\s*(-?\d+)")

# Single-argument NL: 8 becomes 4  OR  8 -> 4  OR  8 returns 4
_ONE_ARG = re.compile(
    r"(-?\d+)\s*(?:->|=>|→|becomes?|returns?|gives?|maps? to|=)\s*(-?\d+)"
)

# Query forms
_NL_QUERY_SINGLE = re.compile(r"what\s+(?:about|if|is|for)\s+(-?\d+)\??\s*$", re.IGNORECASE)
_NL_QUERY_DOUBLE = re.compile(
    r"what\s+(?:about|if|is|for)\s+(-?\d+)\s+(?:and|,)\s+(-?\d+)\??\s*$",
    re.IGNORECASE,
)


def parse_nl_io(text: str) -> Optional[SynthSample]:
    """Extract at least 3 examples + 1 query from free-form text."""
    # Try IO-style first (most structured).
    io_pairs = _IO_PAIR.findall(text)
    io_query = _IO_QUERY.search(text)
    if len(io_pairs) >= 3 and io_query is not None:
        examples = [(int(a), int(b), int(o)) for a, b, o in io_pairs[:3]]
        qa, qb = int(io_query.group(1)), int(io_query.group(2))
        return SynthSample(template="<user>", examples=examples,
                           query_a=qa, query_b=qb, query_out=0)

    # Two-argument NL patterns.
    pairs: List[Tuple[int, int, int]] = []
    for rx in (_TWO_ARG_PAREN, _TWO_ARG_AND, _TWO_ARG_PLUS):
        for m in rx.finditer(text):
            pairs.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))

    if len(pairs) >= 3:
        q = _NL_QUERY_DOUBLE.search(text)
        if q is not None:
            qa, qb = int(q.group(1)), int(q.group(2))
            return SynthSample(template="<user>", examples=pairs[:3],
                               query_a=qa, query_b=qb, query_out=0)

    # Single-argument NL patterns. Convert to (a, 0, out) triples.
    one_pairs = _ONE_ARG.findall(text)
    one_pairs = [(int(a), 0, int(o)) for a, o in one_pairs]
    if len(one_pairs) >= 3:
        q = _NL_QUERY_SINGLE.search(text)
        if q is not None:
            return SynthSample(template="<user>", examples=one_pairs[:3],
                               query_a=int(q.group(1)), query_b=0, query_out=0)

    return None


def _canon(expr: str) -> str:
    """Canonicalize an expression for library keying: strip spaces, normalize."""
    return expr.replace(" ", "")
