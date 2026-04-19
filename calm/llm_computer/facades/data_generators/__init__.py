"""Data generator framework for R53 DB + PT training.

Each generator subclass produces verified (problem, solution, tests)
examples in a domain-specific way. All outputs flow to three sinks:

  1. Messages JSONL  → for CodeExampleDB ingest (retrieval corpus)
  2. PT training     → char-level seq for CopyAugmentedTransformer (R53.5)
  3. Knowledge backend → optional compile to *_kb.py for CALM precompute

Generators register via `register_generator()`. The unified runner
`scripts/r53_run_data_generators.py` invokes each and writes to the
right locations.

See `base.py` for the abstract contract + VerifiedExample schema.
Each `*.py` file in this package should `register_generator(name, cls)`
at import time so the CLI can enumerate available generators.
"""

from __future__ import annotations

from typing import Dict, Type

from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)


_REGISTRY: Dict[str, Type[DomainDataGenerator]] = {}


def register_generator(name: str, cls: Type[DomainDataGenerator]) -> None:
    """Register a generator class under a short name. Idempotent —
    re-registration just overwrites."""
    _REGISTRY[name] = cls


def get_generator(name: str) -> Type[DomainDataGenerator]:
    if name not in _REGISTRY:
        raise KeyError(f"unknown generator {name!r}; "
                       f"have: {sorted(_REGISTRY.keys())}")
    return _REGISTRY[name]


def list_generators() -> list[str]:
    return sorted(_REGISTRY.keys())


# Eagerly import concrete generators so they register themselves.
# Keep these imports at the bottom to avoid circular refs.
from calm.llm_computer.facades.data_generators import (       # noqa: E402, F401
    algorithm_problems,
    stdlib_usage,
    bug_fix_pairs,
    security_patterns,
    parameterized_math,
    regex_patterns,
    data_structures,
    datetime_utils,
    functional_patterns,
)


__all__ = [
    "DomainDataGenerator",
    "VerifiedExample",
    "register_generator",
    "get_generator",
    "list_generators",
]
