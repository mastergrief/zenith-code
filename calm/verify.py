"""
Auto-CALM Layer 1 — claim extraction and verification.

Extracts computational claims from model text, verifies them on CPU,
and corrects wrong ones. Handles arithmetic, boolean, function calls,
GCD/LCM, with LaTeX/markdown stripping and conditional context filtering.

Usage:
    from calm.verify import AutoCalm
    ac = AutoCalm()
    corrected, report = ac.verify_and_correct(text)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from calm.expression import safe_eval, ExpressionError


@dataclass
class Claim:
    """A computational claim extracted from model text."""
    original: str
    expression: str
    claimed_value: str
    actual_value: object = None
    correct: bool = False
    span: Tuple[int, int] = (0, 0)


@dataclass
class VerifyReport:
    """Result of verifying all claims in a text."""
    claims: List[Claim] = field(default_factory=list)
    corrections: int = 0
    verified: int = 0
    unverifiable: int = 0


class AutoCalm:
    """Transparent compute verification for model output."""

    _CLAIM_PATTERNS = [
        # "17 × 23 = 391" — arithmetic with operator required in LHS
        re.compile(
            r'(\d[\d,\s]*(?:[\*×÷\+\-\/%\^\(\)]|\\times|\\cdot|\\div)'
            r'[\d,\s\*×÷\+\-\/%\^\(\)\.]*\d)'
            r'\s*[=≈]\s*'
            r'([\-]?\d[\d,]*)'
            r'(?!\s*(?:[\*×÷\+\-\/%\^]|\\times|\\cdot|\\div))',
        ),
        # "result/answer/mean is 391"
        re.compile(
            r'((?:result|answer|total|sum|product|difference|quotient|value'
            r'|mean|median|average|mode|variance|stdev|correlation)'
            r'\s+(?:is|equals|gives|=)\s+)([\-]?\d[\d,\.]*)',
        ),
        # "factorial(10) = 3628800"
        re.compile(
            r'([a-z_]\w*\([^)]+\))\s*[=≈]\s*'
            r'([\-]?\d[\d,]*)'
            r'(?!\s*[\*×÷\+\-\/%\^\\])',
        ),
        # "GCD of 391 and 782 is 391"
        re.compile(
            r'(?:GCD|gcd|LCM|lcm)\s+of\s+(\d+)\s+and\s+(\d+)\s+'
            r'(?:is|equals|=)\s+([\-]?\d[\d,\.]*)',
        ),
    ]

    _BOOL_PATTERNS = [
        re.compile(r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?(?:a\s+)?prime(?:\s+number)?', re.IGNORECASE),
        re.compile(r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?(?:a\s+)?perfect\b(?:\s+number)?', re.IGNORECASE),
        re.compile(r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?divisible\s+by\s+(\d+)', re.IGNORECASE),
    ]

    # Base conversion claims: "10110011 in hexadecimal is b3", "255 in binary is 11111111"
    _BASE_CLAIM_PATTERNS = [
        # "X in hexadecimal/hex is Y" (input could be binary or decimal)
        re.compile(r'(?:binary\s+(?:number\s+)?)?([01]+)\s+(?:in|to)\s+(?:hexadecimal|hex)\s+(?:is|=|equals)\s+\**([0-9a-fA-F]+)\**', re.IGNORECASE),
        # "X in binary is Y"
        re.compile(r'(\d+)\s+(?:in|to)\s+binary\s+(?:is|=|equals)\s+\**([01]+)\**', re.IGNORECASE),
        # "X in octal is Y"
        re.compile(r'(\d+)\s+(?:in|to)\s+octal\s+(?:is|=|equals)\s+\**([0-7]+)\**', re.IGNORECASE),
        # "X in decimal/base 10 is Y" (input is binary/hex)
        re.compile(r'(?:binary\s+(?:number\s+)?)?([01]+)\s+(?:in|to)\s+(?:decimal|base\s*10)\s+(?:is|=|equals)\s+\**(\d+)\**', re.IGNORECASE),
        # "hexadecimal/hex representation of X is Y"
        re.compile(r'(?:hexadecimal|hex)\s+(?:representation|conversion|value|equivalent)\s+(?:of\s+)?(?:(?:the\s+)?binary\s+(?:number\s+)?)?([01]+)\s+is\s+\**([0-9a-fA-F]+)\**', re.IGNORECASE),
    ]

    _CONDITIONAL_RE = re.compile(r'\b(?:if|whether|check|determine|test|verify)\b', re.IGNORECASE)

    def extract_claims(self, text: str) -> List[Claim]:
        claims: List[Claim] = []
        seen_spans = set()

        for m in self._CLAIM_PATTERNS[0].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            expr = self._normalize_expr(m.group(1))
            claims.append(Claim(
                original=m.group(0), expression=expr,
                claimed_value=m.group(2).replace(",", ""), span=span,
            ))
            seen_spans.add(span)

        for m in self._CLAIM_PATTERNS[2].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            expr = self._normalize_expr(m.group(1))
            claims.append(Claim(
                original=m.group(0), expression=expr,
                claimed_value=m.group(2).replace(",", ""), span=span,
            ))
            seen_spans.add(span)

        for m in self._CLAIM_PATTERNS[3].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            func = "gcd" if "gcd" in m.group(0).lower() else "lcm"
            expr = f"{func}({m.group(1)}, {m.group(2)})"
            claims.append(Claim(
                original=m.group(0), expression=expr,
                claimed_value=m.group(3).replace(",", ""), span=span,
            ))
            seen_spans.add(span)

        # Base conversion claims
        for i, pat in enumerate(self._BASE_CLAIM_PATTERNS):
            for m in pat.finditer(text):
                span = m.span()
                if self._overlaps(span, seen_spans):
                    continue
                input_val = m.group(1)
                claimed = m.group(2).lower()
                # Determine conversion based on pattern index
                if i == 0 or i == 4:  # binary → hex
                    expr = f'base_convert("{input_val}", 2, 16)'
                elif i == 1:  # decimal → binary
                    expr = f'to_binary({input_val})'
                elif i == 2:  # decimal → octal
                    expr = f'to_octal({input_val})'
                elif i == 3:  # binary → decimal
                    expr = f'from_binary("{input_val}")'
                else:
                    continue
                claims.append(Claim(
                    original=m.group(0), expression=expr,
                    claimed_value=claimed, span=span,
                ))
                seen_spans.add(span)

        return claims

    def _is_conditional_match(self, m, text: str) -> bool:
        start = max(0, m.start() - 50)
        prefix = text[start:m.start()]
        return bool(self._CONDITIONAL_RE.search(prefix))

    def extract_bool_claims(self, text: str) -> List[Claim]:
        claims: List[Claim] = []

        for m in self._BOOL_PATTERNS[0].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            claims.append(Claim(
                original=m.group(0), expression=f"is_prime({n})",
                claimed_value="False" if negated else "True", span=m.span(),
            ))

        for m in self._BOOL_PATTERNS[1].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            claims.append(Claim(
                original=m.group(0), expression=f"is_perfect({n})",
                claimed_value="False" if negated else "True", span=m.span(),
            ))

        for m in self._BOOL_PATTERNS[2].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            divisor = m.group(3)
            claims.append(Claim(
                original=m.group(0), expression=f"{n} % {divisor} == 0",
                claimed_value="False" if negated else "True", span=m.span(),
            ))

        return claims

    def verify_claims(self, claims: List[Claim], text: str = "") -> List[Claim]:
        for claim in claims:
            try:
                result = safe_eval(claim.expression)
                claim.actual_value = result
                if self._is_integer_division_context(claim, text) and "/" in claim.expression:
                    result = safe_eval(claim.expression.replace("/", "//"))
                    claim.actual_value = result
                actual_str = self._normalize_value(result)
                claimed_str = self._normalize_value(self._parse_value(claim.claimed_value))
                claim.correct = (actual_str == claimed_str)
            except ExpressionError:
                claim.actual_value = None
                claim.correct = True
        return claims

    def _is_integer_division_context(self, claim: Claim, text: str) -> bool:
        if not text or "/" not in claim.expression:
            return False
        _, end = claim.span
        after = text[end:end + 30]
        return bool(re.search(r'remainder|R\s*\d|\bmod\b', after, re.IGNORECASE))

    @staticmethod
    def _strip_formatting(text: str) -> str:
        s = text
        s = re.sub(r'\\(?:mathbf|textbf|mathrm|text)\{([^}]*)\}', r'\1', s)
        s = s.replace('$', '')
        s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
        return s

    def verify_and_correct(self, text: str) -> Tuple[str, VerifyReport]:
        report = VerifyReport()
        clean = self._strip_formatting(text)
        numeric_claims = self.extract_claims(clean)
        bool_claims = self.extract_bool_claims(clean)
        all_claims = numeric_claims + bool_claims

        if not all_claims:
            return text, report

        self.verify_claims(all_claims, clean)
        report.claims = all_claims
        corrected = clean
        corrections = []

        for claim in sorted(all_claims, key=lambda c: c.span[0], reverse=True):
            if claim.actual_value is None:
                report.unverifiable += 1
            elif claim.correct:
                report.verified += 1
            else:
                report.corrections += 1
                new_text = self._build_correction(claim)
                if new_text:
                    corrections.append((claim, new_text))

        if not corrections:
            return text, report

        for claim, new_text in corrections:
            start, end = claim.span
            corrected = corrected[:start] + new_text + corrected[end:]

        return corrected, report

    def _build_correction(self, claim: Claim) -> Optional[str]:
        if claim.actual_value is None:
            return None
        if isinstance(claim.actual_value, bool):
            if claim.actual_value is True:
                return re.sub(r'\bis\s+not\s+', 'is ', claim.original)
            else:
                return re.sub(r'\bis\s+(?!not\b)', 'is not ', claim.original)
        actual_str = self._format_value(claim.actual_value)
        if "=" in claim.original or "≈" in claim.original:
            parts = re.split(r'[=≈]', claim.original, maxsplit=1)
            if len(parts) == 2:
                return f"{parts[0].rstrip()} = {actual_str}"
        if re.search(r'(?:is|equals)\s+\S+$', claim.original):
            return re.sub(r'((?:is|equals)\s+)\S+$', rf'\g<1>{actual_str}', claim.original)
        return None

    def _normalize_expr(self, expr: str) -> str:
        expr = expr.strip()
        expr = expr.replace("\\times", "*").replace("\\cdot", "*")
        expr = expr.replace("\\div", "/")
        expr = expr.replace("×", "*").replace("÷", "/")
        expr = expr.replace("^", "**").replace(",", "")
        return re.sub(r'\s+', ' ', expr)

    def _normalize_value(self, val) -> str:
        if isinstance(val, bool): return str(val)
        if isinstance(val, float):
            return str(int(val)) if val == int(val) else f"{val:.10g}"
        return str(val)

    def _parse_value(self, s: str) -> object:
        s = s.strip().replace(",", "")
        if s in ("True", "true"): return True
        if s in ("False", "false"): return False
        try: return int(s)
        except ValueError:
            try: return float(s)
            except ValueError: return s

    def _format_value(self, val) -> str:
        if isinstance(val, bool): return str(val)
        if isinstance(val, float) and val == int(val): return str(int(val))
        if isinstance(val, int) and abs(val) >= 1000:
            s = str(abs(val))
            formatted = ""
            for i, c in enumerate(reversed(s)):
                if i > 0 and i % 3 == 0: formatted = "," + formatted
                formatted = c + formatted
            return ("-" + formatted) if val < 0 else formatted
        return str(val)

    @staticmethod
    def _overlaps(span, seen_spans) -> bool:
        s, e = span
        return any(s < se and e > ss for ss, se in seen_spans)
