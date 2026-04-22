"""LLM-written Level-2 — Gemma synthesizes FacadeSpec parameters.

Level-2 MetaFacade (`recursion.py`) generates specs from (fn_name, arity)
using hard-coded canonical patterns. Level-2b lets an LLM suggest the
parse-pattern list + fn_name + arity from a free-text domain request,
and the existing CALM-oracle + ast.parse + validate_facade gates reject
anything that fails.

Why safe against RLAIF-style drift: the LLM proposes; the CALM oracle
*disposes*. Only specs whose safe_eval computation matches held-out
test cases ship to disk. No code path trusts the LLM's output without
independent verification.

Pipeline:
    user_request: "I want a facade for Carmichael lambda"
      ↓
    build_synth_prompt(user_request) → prompt string for Gemma
      ↓
    gemma.generate(prompt) → raw text with JSON-shaped suggestion
      ↓
    parse_llm_suggestion(text) → dict with fn_name, arity, nl_patterns, ...
                              (returns None if extraction fails)
      ↓
    validate_suggestion(suggestion) → list[str] of issues; empty = pass
      ↓
    synthesize_spec(suggestion) → FacadeSpec via MetaFacade.from_oracle
      ↓
    validate_facade(spec, oracle_cases) — CALM gate
      ↓
    generate_facade + install + test

Gate chain: 3 independent checks between LLM output and shipped facade:
    1. JSON extraction (malformed JSON → None)
    2. Field validation (fn_name in safe_eval? arity in {1,2}? patterns compile?)
    3. CALM oracle validation (safe_eval returns correct answers?)

Failure at any gate = LLM output rejected, no facade generated. Zero
drift risk: bad suggestions get dropped silently, good ones get shipped.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional


PROMPT_TEMPLATE = """\
You are a substrate-facade synthesizer. Given a user request, output a
JSON object describing a safe_eval-backed compute facade.

Schema (all fields required):
  - fn_name: string — a function in safe_eval's registry (e.g. "factorial",
    "combinations", "gcd", "is_prime").
  - arity: integer 1 or 2 — how many numeric arguments the function takes.
  - nl_patterns: list of regex strings — NL phrases that should trigger
    this facade. Each regex must contain exactly `arity` integer capture
    groups, e.g. r"factorial of (-?\\d+)".
  - operand_type: "int" or "str".
  - output_type: "int" or "bool".
  - max_operand: integer guard (reject operands beyond this).

User request: {user_request}

Output ONLY the JSON object, no commentary:
"""


@dataclass
class LlmSuggestion:
    fn_name: str
    arity: int
    nl_patterns: list[str]
    operand_type: str = "int"
    output_type: str = "int"
    max_operand: int = 1000
    raw_json: str = ""


def build_synth_prompt(user_request: str) -> str:
    """Format a Gemma-ready prompt for oracle-facade synthesis."""
    return PROMPT_TEMPLATE.format(user_request=user_request)


def parse_llm_suggestion(text: str) -> Optional[LlmSuggestion]:
    """Extract the first valid JSON object from LLM output and map to
    LlmSuggestion. Returns None if extraction or schema validation fails.

    Handles common LLM output noise:
      - Leading text before `{...}`
      - Fenced code blocks ```json ... ```
      - Trailing commentary after `}`
    """
    text = text.strip()

    # Strip common fences / wrappers
    for fence in ("```json", "```python", "```"):
        if text.startswith(fence):
            text = text[len(fence):].strip()
    if text.endswith("```"):
        text = text[:-3].strip()

    # Find the first { ... } balanced substring
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    end = -1
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end == -1:
        return None
    raw = text[start:end]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict):
        return None

    required = ["fn_name", "arity", "nl_patterns"]
    for k in required:
        if k not in data:
            return None

    try:
        return LlmSuggestion(
            fn_name=str(data["fn_name"]),
            arity=int(data["arity"]),
            nl_patterns=[str(p) for p in data["nl_patterns"]],
            operand_type=str(data.get("operand_type", "int")),
            output_type=str(data.get("output_type", "int")),
            max_operand=int(data.get("max_operand", 1000)),
            raw_json=raw,
        )
    except (KeyError, ValueError, TypeError):
        return None


def validate_suggestion(sug: LlmSuggestion) -> list[str]:
    """Independent-validation gate. Returns list of issues; empty list
    means the suggestion passes all pre-oracle checks."""
    issues: list[str] = []

    # Gate 1: fn_name must be in safe_eval's registry
    try:
        from calm.expression import _FUNCTIONS
    except ImportError:
        issues.append("cannot import safe_eval registry")
        return issues
    if sug.fn_name not in _FUNCTIONS:
        issues.append(f"fn_name {sug.fn_name!r} not in safe_eval")

    # Gate 2: arity in {1, 2}
    if sug.arity not in (1, 2):
        issues.append(f"arity must be 1 or 2, got {sug.arity}")

    # Gate 3: operand_type in {"int", "str"}
    if sug.operand_type not in ("int", "str"):
        issues.append(f"operand_type must be 'int' or 'str', got "
                      f"{sug.operand_type!r}")

    # Gate 4: output_type in {"int", "bool"}
    if sug.output_type not in ("int", "bool"):
        issues.append(f"output_type must be 'int' or 'bool', got "
                      f"{sug.output_type!r}")

    # Gate 5: every nl_pattern compiles as regex AND has arity capture groups
    for i, pat in enumerate(sug.nl_patterns):
        try:
            compiled = re.compile(pat, re.IGNORECASE)
        except re.error as e:
            issues.append(f"nl_patterns[{i}] invalid regex: {e}")
            continue
        if compiled.groups != sug.arity:
            issues.append(f"nl_patterns[{i}] has {compiled.groups} groups, "
                          f"expected {sug.arity}")

    return issues


def synthesize_spec(sug: LlmSuggestion, module_name: Optional[str] = None):
    """Convert validated LlmSuggestion into a FacadeSpec via MetaFacade.
    The LLM's nl_patterns are passed as extra_patterns; canonical patterns
    from MetaFacade are also included (belt-and-suspenders on coverage).
    """
    from calm.llm_computer.recursion import MetaFacade
    return MetaFacade.from_oracle(
        fn_name=sug.fn_name,
        arity=sug.arity,
        module_name=module_name or f"{sug.fn_name}_llm",
        max_operand=sug.max_operand,
        operand_type=sug.operand_type,
        output_type=sug.output_type,
        extra_patterns=sug.nl_patterns,
    )


def synthesize_and_validate(
    llm_output: str,
    oracle_cases: list,
    module_name: Optional[str] = None,
) -> tuple[bool, dict]:
    """Full pipeline: parse LLM output → validate → synth spec → CALM gate.

    Returns (success, details) where details is a dict documenting each
    gate pass/fail. Success only when every gate passes. The FacadeSpec
    and the number of oracle passes are included in details on success.

    Does NOT write any file or install anything — caller decides next step.
    """
    details: dict = {"stages": []}

    # Stage 1: JSON extraction
    sug = parse_llm_suggestion(llm_output)
    if sug is None:
        details["stages"].append(("parse", False, "JSON extraction failed"))
        return False, details
    details["stages"].append(("parse", True, ""))
    details["suggestion"] = {
        "fn_name": sug.fn_name, "arity": sug.arity,
        "nl_patterns_count": len(sug.nl_patterns),
        "operand_type": sug.operand_type,
        "output_type": sug.output_type,
    }

    # Stage 2: field validation
    issues = validate_suggestion(sug)
    if issues:
        details["stages"].append(("validate", False, issues))
        return False, details
    details["stages"].append(("validate", True, ""))

    # Stage 3: spec synthesis
    try:
        spec = synthesize_spec(sug, module_name=module_name)
    except Exception as e:
        details["stages"].append(("synth", False, repr(e)))
        return False, details
    details["stages"].append(("synth", True, ""))
    details["spec_module"] = spec.module_name

    # Stage 4: CALM oracle validate
    from calm.llm_computer.recursion import validate_facade
    passed, total = validate_facade(spec, oracle_cases)
    details["oracle_validation"] = f"{passed}/{total}"
    if passed != total:
        details["stages"].append(("oracle", False, f"{passed}/{total}"))
        return False, details
    details["stages"].append(("oracle", True, f"{passed}/{total}"))
    details["spec"] = spec
    return True, details
