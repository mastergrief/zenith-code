"""AlgorithmProblemsGenerator — refactor of `generate_multi_step_code_data.py`.

Reuses the existing 40-template catalog from the standalone script
without duplicating code. Each template returns a `CodeProblem` in
the script's schema; this adapter re-wraps it as a `VerifiedExample`
suitable for the unified generator framework.

Downstream, the base `DomainDataGenerator` handles dedup, AST +
sandbox verify, and emits to all three sinks (messages JSONL, PT
training, optional kb entry).
"""

from __future__ import annotations

import random
from typing import List

from calm.llm_computer.facades.data_generators import register_generator
from calm.llm_computer.facades.data_generators.base import (
    DomainDataGenerator,
    VerifiedExample,
)

# The script at scripts/generate_multi_step_code_data.py owns the
# template catalog. Pull it in as-is rather than duplicating 40+
# template fns here.
import importlib.util
from pathlib import Path

_SCRIPT_PATH = (Path(__file__).resolve().parents[4]
                / "scripts" / "generate_multi_step_code_data.py")


def _load_templates():
    """Dynamic import of the existing template module. Isolated so we
    don't pay the import cost at package load time.

    Must register the module in `sys.modules` BEFORE exec_module so
    that `@dataclass` can introspect its module context — otherwise
    `dataclasses._is_type` crashes with AttributeError on cls.__module__
    not found in sys.modules.
    """
    import sys as _sys
    mod_name = "_gen_mscd"
    if mod_name in _sys.modules:
        return _sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, _SCRIPT_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
    _sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)                         # type: ignore[union-attr]
    return mod


class AlgorithmProblemsGenerator(DomainDataGenerator):
    """Algorithmic / stdlib-free coding problems with sandbox-verified
    solutions. Covers ~40 canonical problems across number theory,
    collections, strings, algorithms, and lightweight security/parsing."""

    name = "algorithms"

    def __init__(self, rng=None):
        super().__init__(rng)
        self._mod = _load_templates()
        self._templates = self._mod.TEMPLATES

    def generate_raw(self, n: int) -> List[VerifiedExample]:
        out: List[VerifiedExample] = []
        attempts = 0
        max_attempts = max(n * 4, len(self._templates) * 2)
        while len(out) < n and attempts < max_attempts:
            tpl = self._templates[attempts % len(self._templates)]
            try:
                cp = tpl(self.rng)
            except Exception:
                attempts += 1
                continue
            attempts += 1
            if cp is None:
                continue
            out.append(VerifiedExample(
                problem=cp.problem,
                signature=cp.signature,
                solution=cp.solution,
                test_cases=list(cp.test_cases),
                reasoning="",                     # base will synthesize 5-step
                algorithm=cp.algorithm,
                complexity=cp.complexity,
                edge_cases=list(cp.edge_cases),
                category=cp.category,
                generator_name=self.name,
                skip_sandbox=cp.skip_sandbox,
            ))
        return out


register_generator("algorithms", AlgorithmProblemsGenerator)
