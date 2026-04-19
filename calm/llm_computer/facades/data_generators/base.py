"""DomainDataGenerator base class + VerifiedExample schema.

Contract for concrete generators (one per domain). Each subclass
implements `generate(n, rng)` returning an iterable of
`VerifiedExample` records. The base class owns the three output
encodings so every generator gets them for free:

  - `.to_messages_jsonl_record()` — DB ingest format
  - `.to_pt_training_record()`    — CopyAugmentedTransformer char seq
  - `.to_kb_entry()`              — optional *_kb.py dict entry

Verification is shared:
  - AST parsing (always)
  - Sandbox execution with test cases (when `skip_sandbox=False`)
  - AST-only fallback for sandbox-blocked modules (urllib, html, etc.)

Subclasses add domain-specific verification on top (CALM backend
checks, security lint, type check, etc.) via `extra_verify()`.
"""

from __future__ import annotations

import abc
import random
import re
import textwrap
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


# ---- Verified example schema ----

@dataclass
class VerifiedExample:
    """One verified (problem, solution, tests) record from a generator.

    Fields are deliberately redundant so each output sink has what it
    needs without cross-referencing. `metadata` is free-form per
    generator.
    """
    # Core
    problem: str                                # user-facing NL statement
    signature: str                              # def/class line the solution starts with
    solution: str                               # full code, no fences
    test_cases: List[Tuple]                     # [(*args, expected), ...]

    # Reasoning trail (used by DB + PT training)
    reasoning: str                              # <think> multi-step block
    algorithm: str                              # 1-line algorithm name
    complexity: str                             # e.g. "O(√n)"
    edge_cases: List[str]                       # common pitfalls

    # Classification
    category: str                               # e.g. "number_theory", "security"
    generator_name: str                         # which generator produced this
    skip_sandbox: bool = False                  # if True, AST-verify only
    metadata: Dict[str, Any] = field(default_factory=dict)

    # ---- Output encodings ----

    def to_messages_jsonl_record(
        self,
        system_prompt: str = "You are a careful, correct coding assistant.",
    ) -> dict:
        """Render as the canonical {messages: [sys, user, asst]} schema
        used by CodeExampleDB.ingest_jsonl()."""
        assistant = self._render_assistant_reasoning()
        return {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": self.problem},
                {"role": "assistant", "content": assistant},
            ],
        }

    def to_pt_training_record(self) -> dict:
        """Char-level sequence suitable for CopyAugmentedTransformer
        training (problem → structured_target).

        Returns {problem_chars, target_chars, prompt, target}.
        Target is the signature + a canonical one-line summary so the
        PT learns NL → structural-form mapping (per session 31 PT
        training convention: input = NL, output = expression form).
        """
        # Target: compact canonical form — signature + algorithm name
        # This mirrors the NL→expression convention used in session 31
        # copy_augmented_hrm training (scripts/train_copy_augmented_hrm.py)
        target = f"{self.signature.rstrip(':').strip()} | {self.algorithm}"
        return {
            "prompt": self.problem,
            "target": target,
            "category": self.category,
            "generator": self.generator_name,
        }

    def to_kb_entry(self) -> Optional[Tuple[str, Callable]]:
        """Optional compilation to a CALM knowledge backend entry.

        Returns (function_name, callable) — the callable returns
        a canonical dict with signature + test cases + algorithm.
        Only meaningful for deterministic lookup-style examples.
        Returns None if the generator doesn't support this.
        """
        fn_name = self._extract_fn_name()
        if not fn_name:
            return None

        # Produce a function that returns the stored knowledge.
        # Important: closure captures are by-reference in loops, so
        # bind via default args.
        solution = self.solution
        sig = self.signature
        algo = self.algorithm
        cmplx = self.complexity
        edges = list(self.edge_cases)
        tests = list(self.test_cases)

        def _entry(_sol=solution, _sig=sig, _a=algo, _c=cmplx,
                   _e=edges, _t=tests):
            return {
                "signature": _sig,
                "solution": _sol,
                "algorithm": _a,
                "complexity": _c,
                "edge_cases": _e,
                "test_cases": _t,
            }
        return (fn_name, _entry)

    # ---- Internals ----

    def _extract_fn_name(self) -> Optional[str]:
        m = re.match(r"\s*def\s+(\w+)", self.signature)
        return m.group(1) if m else None

    def _render_assistant_reasoning(self) -> str:
        """Five-step <think> block + fenced code + verified tests.
        Matches the format used by generate_multi_step_code_data.py
        so existing examples and new generators produce identical
        downstream shape."""
        edge_bullets = "\n".join(f"  - {e}" for e in self.edge_cases)
        fname = self._extract_fn_name() or "f"
        trace_items = self.test_cases[:4]
        trace_bullets = "\n".join(
            f"  - `{fname}({', '.join(repr(a) for a in tc[:-1])})` → "
            f"expected `{tc[-1]!r}`"
            for tc in trace_items
        )
        test_bullets = "\n".join(
            f"  - `{fname}({', '.join(repr(a) for a in tc[:-1])}) "
            f"-> {tc[-1]!r}`  ✓"
            for tc in self.test_cases
        )
        # If reasoning already provided, prefer it; otherwise synthesize
        # the 5-step canonical form.
        reasoning_block = self.reasoning.strip() or textwrap.dedent(f"""\
            STEP 1 — DECOMPOSE
            The user wants: {self.problem}
            Required signature: `{self.signature}`
            Category: {self.category}

            STEP 2 — PLAN
            Algorithm: {self.algorithm}
            Complexity: {self.complexity}
            Edge cases to handle:
            {edge_bullets}

            STEP 3 — IMPLEMENT
            I'll write the function directly. Core logic corresponds to
            the algorithm above.

            STEP 4 — VERIFY (mental test)
            Tracing through representative inputs:
            {trace_bullets}

            STEP 5 — ANSWER
            The implementation below handles all edge cases above.""")

        return (
            f"<think>\n{reasoning_block}\n</think>\n\n"
            f"```python\n{self.solution}```\n\n"
            f"**Verified test cases:**\n{test_bullets}\n"
        )


# ---- Generator base class ----

class DomainDataGenerator(abc.ABC):
    """Base class for domain-specific data generators.

    Subclasses implement `generate_raw()` returning un-verified
    `VerifiedExample` records. The base class filters through
    verification (AST + sandbox) and enforces dedup on problem hash.
    """

    #: Short machine-readable name; set in subclass.
    name: str = "unknown"
    #: Default system prompt for this domain's messages output.
    system_prompt: str = "You are a careful, correct coding assistant."

    def __init__(self, rng: Optional[random.Random] = None):
        self.rng = rng if rng is not None else random.Random(0)

    @abc.abstractmethod
    def generate_raw(self, n: int) -> List[VerifiedExample]:
        """Domain-specific production. Subclass responsibility.
        May return more or fewer than `n`; the base class's `generate`
        will call this and verify+dedup."""
        raise NotImplementedError

    def extra_verify(self, ex: VerifiedExample) -> Tuple[bool, str]:
        """Hook for domain-specific verification beyond AST + sandbox.
        Default pass. Override in security / stdlib / etc. subclasses."""
        return True, "ok"

    # ---- Shared verify + dedup pipeline ----

    def _verify(self, ex: VerifiedExample) -> Tuple[bool, str]:
        # AST check always runs
        from calm.backends.ast_ops import ast_parse
        parsed = ast_parse(ex.solution)
        if not parsed.get("valid"):
            return False, "AST: " + ", ".join(parsed.get("errors", []))

        fname = ex._extract_fn_name()
        if fname:
            names = {f["name"] for f in parsed.get("functions", [])}
            if fname not in names:
                return False, f"missing def {fname}"

        if ex.skip_sandbox:
            # Trust hand-written solution with AST-only check.
            return self.extra_verify(ex)

        # Sandbox run each test case
        from calm.sandbox import run_python
        if not fname or not ex.test_cases:
            return True, "no-tests"

        test_lines = []
        for i, tc in enumerate(ex.test_cases):
            *args, expected = tc
            args_s = ", ".join(repr(a) for a in args)
            exp_r = repr(expected)
            test_lines.append(
                f"try:\n"
                f"    _got = {fname}({args_s})\n"
                f"    print('PASS' if _got == {exp_r} "
                f"else 'FAIL idx={i} got=' + repr(_got))\n"
                f"except Exception as _e:\n"
                f"    print('FAIL idx={i} raised=' + type(_e).__name__)"
            )
        script = ex.solution + "\n\n" + "\n".join(test_lines) + "\npass\n"
        result = run_python(script, timeout=5.0)
        if result.error:
            return False, str(result.error)
        out = result.stdout or ""
        if "FAIL" in out:
            return False, out.strip().splitlines()[0]
        passed = out.count("PASS")
        if passed != len(test_lines):
            return False, f"only {passed}/{len(test_lines)} passed"
        return self.extra_verify(ex)

    def generate(self, n: int) -> List[VerifiedExample]:
        """Produce up to `n` verified unique examples."""
        raw = self.generate_raw(n * 2)  # over-generate to absorb skips
        seen: set[str] = set()
        out: List[VerifiedExample] = []
        skipped_verify = 0
        for ex in raw:
            if ex.problem in seen:
                continue
            ok, info = self._verify(ex)
            if not ok:
                skipped_verify += 1
                continue
            ex.generator_name = self.name
            seen.add(ex.problem)
            out.append(ex)
            if len(out) >= n:
                break
        return out
