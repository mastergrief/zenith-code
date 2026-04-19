"""CodeVerifierFacade — R53's verified-code-reasoning oracle.

Built per `.claude/rules/augmentation_thesis.md` §"Tier-2 stacking":
additive wrapper around existing CALM compute + DB, no replacement of
Gemma weights. Lives alongside `MultiStepReasoningFacade` (R46.2) as
the domain analogue for code problems.

Two phases of use:

  Phase 1 — prompt-level augmentation (no install needed):
      f = CodeVerifierFacade()
      hints = f.compute_hints("write a function that parses CSV")
      augmented_prompt = f.inject_hints(prompt, hints)
      # pass augmented_prompt to Gemma

  Phase 2 — substrate install (requires trained code PT + KnowledgeStore):
      f.install(gemma, tokenizer, pt_ckpt=..., knowledge_store=...)
      result = f.generate(prompt)   # biased decode

Phase 1 is the first measurable gate: does CALM-backed retrieval +
verification on top of stock Gemma beat stock Gemma alone? Only then
do we invest in PT training + L24/L30 install.

Composition (all via shipped backends — nothing new invented):

  NL prompt ─► classify_intent ─► problem_kind
                │
                ├─► DB retrieve top-k similar (problem, solution) pairs
                │
                ├─► AST/syntax scan for any code in the prompt
                │
                ├─► Security heuristics (SSRF / SQLi / XSS markers)
                │
                └─► Arithmetic precompute (reuse CALM precompute layer)

  Output: CodeHints dataclass
          │
          └─► inject_hints(): formats into a "Verified context:" block
                             prepended to the system prompt

Post-generation:

  CodeVerifierFacade.verify(code) ─► VerificationResult
      │
      ├─► ast_parse: syntax valid?
      ├─► ast_imports: what does it import?
      ├─► sandbox run_python: does it execute without error?
      │       (with user-supplied test cases if present)
      └─► returns {ok, errors, stdout, passes}
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from calm.llm_computer.facades.code_example_db import (
    CodeExampleDB, RetrievalHit,
)


# ---------- Problem classification ----------

# Heuristic tags derived from prompt text. Order matters — the first
# matching pattern wins to keep classification single-label for now.
# Regex phase-one; upgrade to the code PT's classifier head once it
# ships (R53.5).
_INTENT_PATTERNS: List[Tuple[str, re.Pattern]] = [
    ("debug_error",   re.compile(r"\b(fix|debug|error|crash|failing|throws|exception|traceback)\b", re.I)),
    ("security",      re.compile(r"\b(ssrf|xss|sql injection|csrf|auth|jwt|password|secret|vulnerab|exploit|token|sanitize)\b", re.I)),
    ("concurrency",   re.compile(r"\b(race|mutex|lock|concurrent|goroutine|async|thread|atomic|channel|deadlock)\b", re.I)),
    ("performance",   re.compile(r"\b(slow|fast|performance|optimi[sz]e|bottleneck|memory|oom|leak|profile)\b", re.I)),
    ("architecture",  re.compile(r"\b(architect|design|pattern|scale|microservice|monolith|database|schema|queue|trade[- ]?off)\b", re.I)),
    ("refactor",      re.compile(r"\b(refactor|clean|simplify|rewrite|restructure|extract|rename)\b", re.I)),
    ("test",          re.compile(r"\b(test|pytest|unit test|mock|fixture|assert|coverage)\b", re.I)),
    ("write_function",re.compile(r"\b(write|create|implement|build|code)\b.*\b(function|class|method|script|program)\b", re.I)),
    ("explain",       re.compile(r"\b(explain|what (is|does)|how (does|do)|why)\b", re.I)),
]

# Module names commonly hinted by problem text. If Gemma is going to
# write code, it usually needs one of these. Injecting them as hints
# reduces the chance of hallucinated libs or wrong module paths.
_LIBRARY_HINTS: List[Tuple[str, re.Pattern]] = [
    ("re",        re.compile(r"\b(regex|pattern match|expression)\b", re.I)),
    ("json",      re.compile(r"\bjson\b", re.I)),
    ("csv",       re.compile(r"\bcsv\b", re.I)),
    ("asyncio",   re.compile(r"\basync(io)?\b|\bawait\b|\bcoroutine\b", re.I)),
    ("pathlib",   re.compile(r"\b(path|file system|directory)\b", re.I)),
    ("dataclass", re.compile(r"\bdataclass\b|\bdata class\b", re.I)),
    ("typing",    re.compile(r"\btype hint(s)?\b|\btyping\b|\bOptional\b|\bList\b|\bDict\b"),),
    ("pytest",    re.compile(r"\bpytest\b|\bunit test\b|\btest case\b", re.I)),
    ("sqlalchemy",re.compile(r"\bsqlalchemy\b|\borm\b", re.I)),
    ("requests",  re.compile(r"\bhttp request\b|\brequests library\b", re.I)),
]

# Security-adjacent red flags. Any match flips a flag in hints so the
# caller can prepend a "security-critical" cue to the prompt.
_SECURITY_RED_FLAGS: List[Tuple[str, re.Pattern]] = [
    ("ssrf",            re.compile(r"\bssrf\b|\bfetch.*url\b|\bserver[- ]side request\b", re.I)),
    ("sql_injection",   re.compile(r"\bsql injection\b|\bsqli\b|\braw sql\b|\bstring concat.*query\b", re.I)),
    ("xss",             re.compile(r"\bxss\b|\bcross[- ]site script", re.I)),
    ("path_traversal",  re.compile(r"\bpath traversal\b|\b\.\./\b|\bdirectory traversal\b", re.I)),
    ("auth_bypass",     re.compile(r"\bauth(entication)? bypass\b|\bidor\b|\binsecure direct\b", re.I)),
    ("csrf",            re.compile(r"\bcsrf\b|\bcross[- ]site request", re.I)),
    ("csp",             re.compile(r"\bcsp\b|\bcontent security policy\b", re.I)),
    ("secret_leak",     re.compile(r"\bsecret(s)? in\b|\bhardcoded (secret|password|key|token)\b", re.I)),
]

# Code-block extraction — pulls any ``` delimited code from the prompt
# so we can AST-scan / sandbox-run it as part of hints (e.g. "this code
# throws ValueError, fix it" — the code is IN the prompt).
_CODE_BLOCK_RE = re.compile(r"```(?:\w+)?\n(.*?)```", re.DOTALL)


# ---------- Hints + verification ----------

@dataclass
class CodeHints:
    """Everything the facade derived from a prompt. Dumped into a
    "Verified context:" block by `inject_hints` so Gemma sees it
    before the original user query."""
    prompt: str
    intent: str                                      # one of the _INTENT_PATTERNS keys, or "general"
    suggested_libraries: List[str] = field(default_factory=list)
    security_flags: List[str] = field(default_factory=list)
    retrieved_examples: List[RetrievalHit] = field(default_factory=list)
    embedded_code: List[str] = field(default_factory=list)
    embedded_code_valid: List[bool] = field(default_factory=list)
    arithmetic_precompute: List[Tuple[str, Any]] = field(default_factory=list)

    def to_system_prefix(self, max_example_chars: int = 400) -> str:
        """Render hints into a short system-prompt prefix."""
        lines: List[str] = ["Verified context (from local compute + example DB):"]
        lines.append(f"- problem_kind: {self.intent}")
        if self.suggested_libraries:
            lines.append(
                "- likely imports: " + ", ".join(self.suggested_libraries))
        if self.security_flags:
            lines.append(
                "- security concerns: " + ", ".join(self.security_flags)
                + " (apply defense-in-depth; don't rely on a single check)"
            )
        if self.arithmetic_precompute:
            for expr, val in self.arithmetic_precompute:
                lines.append(f"- verified: {expr} = {val}")
        if self.embedded_code:
            for i, (src, ok) in enumerate(
                    zip(self.embedded_code, self.embedded_code_valid)):
                status = "valid" if ok else "INVALID (syntax error)"
                lines.append(
                    f"- code block #{i + 1}: {status}, "
                    f"{len(src.splitlines())} lines, "
                    f"{len(src)} chars")
        if self.retrieved_examples:
            lines.append("- related past solutions (for pattern reuse):")
            for i, hit in enumerate(self.retrieved_examples):
                preview = hit.example.solution_preview[:max_example_chars]
                lines.append(
                    f"  [{i + 1}] ({hit.score:.2f}) problem: "
                    f"{hit.example.problem[:120]}")
                lines.append(f"      solution: {preview}")
        return "\n".join(lines)


@dataclass
class VerificationResult:
    """Post-generation verification of a code response."""
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    parsed_funcs: List[str] = field(default_factory=list)
    parsed_imports: List[str] = field(default_factory=list)
    sandbox_stdout: Optional[str] = None
    sandbox_error: Optional[str] = None
    tests_passed: Optional[int] = None
    tests_total: Optional[int] = None


# ---------- The facade ----------

class CodeVerifierFacade:
    """Tier-2 augmentation for code-reasoning prompts.

    Wraps CALM ast/sandbox/security backends + `CodeExampleDB` without
    modifying Gemma. Phase 1 usage is prompt-level only (inject hints
    into system prompt); Phase 2 (install at L24/L30 via CardSlot) is
    layered on top in R53.6 without reopening the facade's interface.
    """

    DEFAULT_TOP_K = 3
    DEFAULT_MIN_SCORE = 0.08

    def __init__(
        self,
        db: Optional[CodeExampleDB] = None,
        top_k: int = DEFAULT_TOP_K,
        min_score: float = DEFAULT_MIN_SCORE,
    ):
        self.db = db if db is not None else CodeExampleDB.load_default()
        self.top_k = top_k
        self.min_score = min_score
        self._gemma = None
        self._tokenizer = None

    # ----- Phase 1: prompt-level augmentation -----

    def classify_intent(self, prompt: str) -> str:
        for name, pat in _INTENT_PATTERNS:
            if pat.search(prompt):
                return name
        return "general"

    def suggest_libraries(self, prompt: str) -> List[str]:
        hits: List[str] = []
        seen: set[str] = set()
        for name, pat in _LIBRARY_HINTS:
            if pat.search(prompt) and name not in seen:
                hits.append(name)
                seen.add(name)
        return hits

    def detect_security_flags(self, prompt: str) -> List[str]:
        return [n for n, pat in _SECURITY_RED_FLAGS if pat.search(prompt)]

    def extract_code_blocks(self, prompt: str) -> List[str]:
        return [m.group(1) for m in _CODE_BLOCK_RE.finditer(prompt)]

    def validate_code_block(self, source: str) -> bool:
        """Quick AST syntax check via the shipped CALM ast backend."""
        # Local import keeps module import-time light.
        from calm.backends.ast_ops import ast_parse
        return bool(ast_parse(source).get("valid"))

    def arithmetic_precompute(self, prompt: str) -> List[Tuple[str, Any]]:
        """Reuse CALM's Layer 2 precompute for any arithmetic in the
        prompt. Returns list of (expression, verified_value) pairs.

        Empty list on any error — this is best-effort enrichment, not
        the gate."""
        try:
            from calm.precompute import precompute as _precompute
            hits = _precompute(prompt) or {}
            return [(str(k), v) for k, v in hits.items()]
        except Exception:
            return []

    def compute_hints(self, prompt: str) -> CodeHints:
        """Full Phase-1 pipeline. Deterministic, fast (<100 ms)."""
        embedded = self.extract_code_blocks(prompt)
        embedded_valid = [self.validate_code_block(c) for c in embedded]
        retrieved = self.db.retrieve(
            prompt, k=self.top_k, min_score=self.min_score)
        return CodeHints(
            prompt=prompt,
            intent=self.classify_intent(prompt),
            suggested_libraries=self.suggest_libraries(prompt),
            security_flags=self.detect_security_flags(prompt),
            retrieved_examples=retrieved,
            embedded_code=embedded,
            embedded_code_valid=embedded_valid,
            arithmetic_precompute=self.arithmetic_precompute(prompt),
        )

    def inject_hints(
        self,
        prompt: str,
        hints: Optional[CodeHints] = None,
        base_system: str = "You are a careful, correct coding assistant.",
    ) -> str:
        """Return a system prompt that prepends the verified context.
        The caller combines this with the user turn as usual."""
        if hints is None:
            hints = self.compute_hints(prompt)
        prefix = hints.to_system_prefix()
        return f"{base_system}\n\n{prefix}"

    # ----- Post-generation verification -----

    def verify(
        self,
        code: str,
        test_code: Optional[str] = None,
        timeout: float = 5.0,
    ) -> VerificationResult:
        """Run syntax + sandbox checks on a generated code string.
        Optionally executes `test_code` (which should assert + print)
        under the same sandbox.

        Uses shipped CALM backends — `ast_parse`, `run_python`.
        """
        from calm.backends.ast_ops import ast_parse

        parsed = ast_parse(code)
        if not parsed.get("valid"):
            return VerificationResult(
                ok=False,
                errors=list(parsed.get("errors", ["invalid AST"])),
            )

        funcs = [f["name"] for f in parsed.get("functions", [])]
        imports = [i["module"] for i in parsed.get("imports", [])]

        sandbox_stdout: Optional[str] = None
        sandbox_error: Optional[str] = None
        tests_passed: Optional[int] = None
        tests_total: Optional[int] = None

        try:
            from calm.sandbox import run_python
            # The sandbox's wrapper tries to eval the final line as an
            # expression, which breaks on multi-line def/class bodies.
            # Guard with a trailing `pass` so the last-line eval is a
            # harmless no-op, and only use the body-only run to catch
            # import / top-level errors (not for stdout).
            body_result = run_python(code + "\npass\n", timeout=timeout)
            if body_result.error:
                sandbox_error = str(body_result.error)
            sandbox_stdout = body_result.stdout

            if test_code and sandbox_error is None:
                combined = code + "\n\n" + test_code + "\npass\n"
                tests_result = run_python(combined, timeout=timeout)
                tests_stdout = tests_result.stdout or ""
                passed = tests_stdout.count("PASS")
                failed = tests_stdout.count("FAIL")
                tests_passed = passed
                tests_total = passed + failed
                if tests_result.error:
                    sandbox_error = str(tests_result.error)
                sandbox_stdout = (sandbox_stdout or "") + tests_stdout
        except ImportError:
            sandbox_error = "sandbox unavailable"
        except Exception as e:
            sandbox_error = f"sandbox exception: {type(e).__name__}: {e}"

        ok = (
            sandbox_error is None
            and (tests_total is None or tests_passed == tests_total)
        )
        warnings: List[str] = []
        if sandbox_error:
            warnings.append(f"sandbox: {sandbox_error}")
        return VerificationResult(
            ok=ok,
            errors=[],
            warnings=warnings,
            parsed_funcs=funcs,
            parsed_imports=imports,
            sandbox_stdout=sandbox_stdout,
            sandbox_error=sandbox_error,
            tests_passed=tests_passed,
            tests_total=tests_total,
        )

    # ----- Phase 2 stubs (R53.6) -----

    def install(self, gemma, tokenizer, pt_ckpt=None, knowledge_store=None):
        """Placeholder — wired up in R53.6 once code PT + KnowledgeStore
        exist. Left as a stub here so the call site in eval scripts is
        stable across rounds."""
        self._gemma = gemma
        self._tokenizer = tokenizer
        # R53.6: CardSlot(layer_idx=24, ...) for PT
        # R53.6: CardSlot(layer_idx=30, ...) for knowledge_store recall card
        # R53.6: VerificationHook biasing logits toward verified tokens

    def detach(self):
        self._gemma = None
        self._tokenizer = None
