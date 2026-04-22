"""Oracle signature inference — map NL prompt → (fn_name, arity, output_type).

Closes the autonomous loop:
    CALM verifier catches Gemma failure (wrong answer) on some prompt
      ↓
    infer_oracle_signature(prompt) → (fn_name, arity, output_type, operand_type)
      ↓
    MetaFacade.from_oracle(**signature) → FacadeSpec
      ↓
    validate_facade(spec, oracle_cases) → CALM gate
      ↓
    generate_facade(spec) → write .py file
      ↓
    import_facade_class(spec).install(...) → live substrate capability

After this runs once for a domain, subsequent prompts in that domain
are answered exactly (not by Gemma's prior) with zero retraining.

Design:
  - Curated catalog of (nl_keyword, fn_name, arity, output_type) triples.
  - Scan prompt for keyword + count integer/date literals adjacent.
  - Return first match; None if no catalog entry fires.

Catalog entries are deliberately conservative — only fn_names already
in safe_eval's registry (`calm.expression._FUNCTIONS`). Expansion
via user-supplied entries: `register_signature(keyword, fn_name, arity)`.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class OracleSignature:
    """Result of inferring the safe_eval target from a NL prompt."""
    fn_name: str
    arity: int
    operand_type: str = "int"       # "int" | "str"
    output_type: str = "int"        # "int" | "bool"
    # NL patterns / aliases that made the inference (for logging).
    matched_alias: Optional[str] = None


# Curated catalog of (NL alias regex → oracle signature).
# Tuple format: (alias_regex, fn_name, arity, operand_type, output_type).
# Alias regexes are case-insensitive keyword matches. Arity constrains
# the number of numeric literals the prompt must contain (gate below).
_SIGNATURES: list[tuple[str, str, int, str, str]] = [
    # 1-arg integer-in → integer-out
    (r"\bfactorial\b|\bn!\b|\b\d+\s*!",    "factorial",    1, "int", "int"),
    (r"\bfibonacci\b|\bfib\b",              "fibonacci",    1, "int", "int"),
    (r"\bnext\s+prime\b|\bsmallest\s+prime\s+(?:greater|larger|bigger)",
                                            "next_prime",   1, "int", "int"),
    (r"\bcollatz",                          "collatz_length", 1, "int", "int"),
    (r"\bdigit\s+sum\b|\bsum\s+of\s+(?:the\s+)?digits", "digit_sum", 1, "int", "int"),
    (r"\btotient\b|\beuler'?s?\s+totient",  "totient",      1, "int", "int"),

    # 2-arg integer-in → integer-out
    (r"\b\d+\s+choose\s+\d+|\bcombinations?\b|\bbinomial\s+coefficient",
                                            "combinations", 2, "int", "int"),
    (r"\b\d+\s+permute\s+\d+|\bpermutations?\s+of",
                                            "permutations", 2, "int", "int"),
    (r"\bgcd\s+of\b|\bgreatest\s+common\s+divisor",
                                            "gcd",          2, "int", "int"),
    (r"\blcm\s+of\b|\bleast\s+common\s+multiple",
                                            "lcm",          2, "int", "int"),
    (r"\bto\s+the\s+power\b|\braised\s+to\b|\d+\s*\^\s*\d+|\d+\s*\*\*\s*\d+",
                                            "pow",          2, "int", "int"),

    # 1-arg integer-in → bool-out
    (r"\bis\s+\d+\s+(?:a\s+)?prime\b|\bis_prime\b",
                                            "is_prime",     1, "int", "bool"),
    (r"\bis\s+\d+\s+(?:a\s+)?perfect\s+(?:number|square)?\b|\bis_perfect\b",
                                            "is_perfect",   1, "int", "bool"),
    (r"\bis\s+\d+\s+(?:a\s+)?leap\s+year\b|\bis_leap_year\b",
                                            "is_leap_year", 1, "int", "bool"),

    # 2-arg str-in (ISO date) → integer-out
    (r"\bdays\s+(?:between|from)\b", "days_between", 2, "str", "int"),
]


_INT_LITERAL_RE = re.compile(r"-?\b\d+\b")
_DATE_LITERAL_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def infer_oracle_signature(prompt: str) -> Optional[OracleSignature]:
    """Return the first catalog entry matching `prompt`, or None.

    Matching gate: alias regex matches AND operand-count in prompt matches
    arity. Avoids false positives on stray numbers (e.g. "prime" near a
    phone number).
    """
    low = prompt  # re.IGNORECASE on each search, not pre-lowered (date regex)
    int_count = len(_INT_LITERAL_RE.findall(prompt))
    date_count = len(_DATE_LITERAL_RE.findall(prompt))

    for alias, fn, arity, op_type, out_type in _SIGNATURES:
        m = re.search(alias, low, re.IGNORECASE)
        if not m:
            continue
        # operand-count gate
        if op_type == "str":
            if date_count < arity:
                continue
        else:
            if int_count < arity:
                continue
        return OracleSignature(
            fn_name=fn, arity=arity,
            operand_type=op_type, output_type=out_type,
            matched_alias=m.group(0),
        )
    return None


def register_signature(
    alias_regex: str,
    fn_name: str,
    arity: int,
    operand_type: str = "int",
    output_type: str = "int",
) -> None:
    """Append a user-supplied signature entry. Entries added here are
    checked after the built-in catalog. `fn_name` must be valid in
    safe_eval's registry — we do not verify here."""
    _SIGNATURES.append((alias_regex, fn_name, arity, operand_type, output_type))


def propose_facade_spec(prompt: str, domain_hint: str | None = None):
    """Full inference → FacadeSpec pipeline. Runs
    infer_oracle_signature, then synthesizes a FacadeSpec via MetaFacade.

    Returns None if inference fails (no catalog entry fires).
    """
    sig = infer_oracle_signature(prompt)
    if sig is None:
        return None
    from calm.llm_computer.recursion import MetaFacade
    # Use domain hint as module name if provided; else defaults
    kwargs = dict(
        fn_name=sig.fn_name,
        arity=sig.arity,
        operand_type=sig.operand_type,
        output_type=sig.output_type,
    )
    if domain_hint:
        kwargs["module_name"] = f"{domain_hint}_inferred"
        kwargs["domain_name"] = domain_hint.capitalize()
    return MetaFacade.from_oracle(**kwargs)
