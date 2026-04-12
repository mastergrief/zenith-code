"""
Auto-CALM — transparent compute verification for model output.

The model writes naturally. The engine intercepts, verifies, and
corrects every computational claim without the model knowing CALM
exists. Three layers:

Layer 1: Claim Verification — extract "X = Y" from output, verify, correct
Layer 2: Computation Extraction — detect "let me compute X", evaluate, inject
Layer 3: Intent-to-Edit — detect code change descriptions, generate, apply, test

Usage:
    from calm.auto_calm import AutoCalm
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
    original: str       # the full matched text, e.g. "17 × 23 = 401"
    expression: str     # the LHS expression, e.g. "17 * 23"
    claimed_value: str  # what the model said, e.g. "401"
    actual_value: object = None   # what we computed
    correct: bool = False
    span: Tuple[int, int] = (0, 0)  # position in original text


@dataclass
class VerifyReport:
    """Result of verifying all claims in a text."""
    claims: List[Claim] = field(default_factory=list)
    corrections: int = 0
    verified: int = 0
    unverifiable: int = 0


class AutoCalm:
    """Transparent compute verification for model output."""

    # Patterns that extract "expression = value" claims.
    # Each pattern must have group(1)=expression, group(2)=value.
    _CLAIM_PATTERNS = [
        # "17 × 23 = 391" or "17 * 23 = 391" or "17\times 23 = 391"
        # LHS must contain at least one operator (not just digits).
        # RHS must be a plain number NOT followed by more operators.
        # Supports LaTeX: \times, \div, \cdot. Allows commas in numbers.
        re.compile(
            r'(\d[\d,\s]*(?:[\*×÷\+\-\/%\^\(\)]|\\times|\\cdot|\\div)'
            r'[\d,\s\*×÷\+\-\/%\^\(\)\.]*\d)'
            r'\s*[=≈]\s*'
            r'([\-]?\d[\d,]*)'
            r'(?!\s*(?:[\*×÷\+\-\/%\^]|\\times|\\cdot|\\div))',
        ),
        # "result is 391" / "answer is 391" / "product is 5481"
        # Captures the keyword and value for cross-reference.
        re.compile(
            r'((?:result|answer|total|sum|product|difference|quotient|value)'
            r'\s+(?:is|equals|gives|=)\s+)([\-]?\d[\d,]*)',
        ),
        # "factorial(10) = 3628800" — function call = value
        re.compile(
            r'([a-z_]\w*\([^)]+\))\s*[=≈]\s*'
            r'([\-]?\d[\d,]*)'
            r'(?!\s*[\*×÷\+\-\/%\^\\])',
        ),
        # "GCD of 391 and 782 is 391" / "LCM of 12 and 8 is 24"
        re.compile(
            r'(?:GCD|gcd|LCM|lcm)\s+of\s+(\d+)\s+and\s+(\d+)\s+'
            r'(?:is|equals|=)\s+([\-]?\d[\d,\.]*)',
        ),
    ]

    # Patterns for boolean claims: "X is prime", "X is not prime"
    _BOOL_PATTERNS = [
        # "391 is prime" / "(391) is not prime"
        re.compile(
            r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?(?:a\s+)?prime(?:\s+number)?',
            re.IGNORECASE,
        ),
        # "391 is a perfect number" / "(28) is a perfect number"
        # \b after "perfect" prevents matching "perfectly"
        re.compile(
            r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?(?:a\s+)?perfect\b(?:\s+number)?',
            re.IGNORECASE,
        ),
        # "391 is divisible by 3" / "(1089) is not divisible by 3"
        re.compile(
            r'\(?(\d[\d,]*)\)?\s+is\s+(not\s+)?divisible\s+by\s+(\d+)',
            re.IGNORECASE,
        ),
    ]

    # Words in the ~50 chars before a bool match that signal a question.
    _CONDITIONAL_RE = re.compile(
        r'\b(?:if|whether|check|determine|test|verify)\b',
        re.IGNORECASE,
    )

    # NL expression patterns: "X times Y", "X plus Y", etc.
    _NL_MATH = [
        (re.compile(r'(\d+)\s*(?:times|×|multiplied by)\s*(\d+)'), r'\1 * \2'),
        (re.compile(r'(\d+)\s*(?:plus|\+|added to)\s*(\d+)'), r'\1 + \2'),
        (re.compile(r'(\d+)\s*(?:minus|\-|subtracted (?:from|by))\s*(\d+)'), r'\1 - \2'),
        (re.compile(r'(\d+)\s*(?:divided by|÷|over)\s*(\d+)'), r'\1 / \2'),
        (re.compile(r'(\d+)\s*(?:to the power of|\*\*|raised to)\s*(\d+)'), r'\1 ** \2'),
    ]

    def extract_claims(self, text: str) -> List[Claim]:
        """Extract all computational claims from text."""
        claims: List[Claim] = []
        seen_spans = set()

        # Pattern 0: expression = value (arithmetic)
        for m in self._CLAIM_PATTERNS[0].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            expr = self._normalize_expr(m.group(1))
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=m.group(2).replace(",", ""),
                span=span,
            ))
            seen_spans.add(span)

        # Pattern 1: "product is X" — standalone value claims.
        # These are verified via prompt_expression if available.
        # (cross-reference deferred to Layer 2)

        # Pattern 2: function(args) = value
        for m in self._CLAIM_PATTERNS[2].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            expr = self._normalize_expr(m.group(1))
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=m.group(2).replace(",", ""),
                span=span,
            ))
            seen_spans.add(span)

        # Pattern 4: GCD/LCM of X and Y is Z
        for m in self._CLAIM_PATTERNS[3].finditer(text):
            span = m.span()
            if self._overlaps(span, seen_spans):
                continue
            func = "gcd" if "gcd" in m.group(0).lower() else "lcm"
            expr = f"{func}({m.group(1)}, {m.group(2)})"
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=m.group(3).replace(",", ""),
                span=span,
            ))
            seen_spans.add(span)

        return claims

    def _is_conditional_match(self, m, text: str) -> bool:
        """Check if a regex match is in a conditional context within ~50 chars."""
        start = max(0, m.start() - 50)
        prefix = text[start:m.start()]
        return bool(self._CONDITIONAL_RE.search(prefix))

    def extract_bool_claims(self, text: str) -> List[Claim]:
        """Extract boolean claims like 'X is prime'.
        Skips conditional contexts ('if X is prime', 'whether X is prime')."""
        claims: List[Claim] = []

        # "X is [not] prime"
        for m in self._BOOL_PATTERNS[0].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            expr = f"is_prime({n})"
            claimed = "False" if negated else "True"
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=claimed,
                span=m.span(),
            ))

        # "X is [not] perfect"
        for m in self._BOOL_PATTERNS[1].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            expr = f"is_perfect({n})"
            claimed = "False" if negated else "True"
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=claimed,
                span=m.span(),
            ))

        # "X is [not] divisible by Y"
        for m in self._BOOL_PATTERNS[2].finditer(text):
            if self._is_conditional_match(m, text):
                continue
            n = m.group(1).replace(",", "")
            negated = bool(m.group(2))
            divisor = m.group(3)
            expr = f"{n} % {divisor} == 0"
            claimed = "False" if negated else "True"
            claims.append(Claim(
                original=m.group(0),
                expression=expr,
                claimed_value=claimed,
                span=m.span(),
            ))

        return claims

    def verify_claims(self, claims: List[Claim], text: str = "") -> List[Claim]:
        """Verify each claim by computing the expression independently."""
        for claim in claims:
            try:
                result = safe_eval(claim.expression)
                claim.actual_value = result

                # If the text after this claim mentions "remainder",
                # this is integer division — compare with //.
                if self._is_integer_division_context(claim, text):
                    if "/" in claim.expression:
                        int_expr = claim.expression.replace("/", "//")
                        result = safe_eval(int_expr)
                        claim.actual_value = result

                # Compare: normalize both to comparable form.
                actual_str = self._normalize_value(result)
                claimed_str = self._normalize_value(
                    self._parse_value(claim.claimed_value)
                )
                claim.correct = (actual_str == claimed_str)
            except ExpressionError:
                # Can't verify — leave as unverifiable.
                claim.actual_value = None
                claim.correct = True  # benefit of the doubt
        return claims

    def _is_integer_division_context(self, claim: Claim, text: str) -> bool:
        """Check if a division claim is in a 'remainder' context."""
        if not text or "/" not in claim.expression:
            return False
        _, end = claim.span
        after = text[end:end + 30]
        return bool(re.search(r'remainder|R\s*\d|\bmod\b', after, re.IGNORECASE))

    @staticmethod
    def _strip_formatting(text: str) -> str:
        """Strip LaTeX and markdown formatting for claim extraction.
        Preserves structure but removes \\mathbf{}, \\text{}, $, **, etc."""
        s = text
        # Strip LaTeX commands that wrap values: \mathbf{X} → X
        s = re.sub(r'\\(?:mathbf|textbf|mathrm|text)\{([^}]*)\}', r'\1', s)
        # Strip inline math delimiters
        s = s.replace('$', '')
        # Strip markdown bold: **text** → text
        # Don't strip single * — conflicts with multiplication operator.
        s = re.sub(r'\*\*([^*]+)\*\*', r'\1', s)
        return s

    def verify_and_correct(self, text: str) -> Tuple[str, VerifyReport]:
        """
        Full pipeline: extract claims, verify, correct wrong ones.
        Returns (corrected_text, report).
        """
        report = VerifyReport()

        # Strip LaTeX formatting for extraction, but apply
        # corrections to the original text.
        clean = self._strip_formatting(text)

        # Extract numeric and boolean claims from cleaned text.
        numeric_claims = self.extract_claims(clean)
        bool_claims = self.extract_bool_claims(clean)
        all_claims = numeric_claims + bool_claims

        if not all_claims:
            return text, report

        # Verify all claims (pass cleaned text for context).
        self.verify_claims(all_claims, clean)
        report.claims = all_claims

        # Count results. Apply corrections to the CLEANED text,
        # then map back to original if needed.
        corrected = clean
        corrections = []

        for claim in sorted(all_claims, key=lambda c: c.span[0], reverse=True):
            if claim.actual_value is None:
                report.unverifiable += 1
                continue

            if claim.correct:
                report.verified += 1
            else:
                report.corrections += 1
                new_text = self._build_correction(claim)
                if new_text:
                    corrections.append((claim, new_text))

        if not corrections:
            return text, report

        # Apply corrections to the cleaned text.
        for claim, new_text in corrections:
            start, end = claim.span
            corrected = corrected[:start] + new_text + corrected[end:]

        return corrected, report

    def _build_correction(self, claim: Claim) -> Optional[str]:
        """Build corrected text for a wrong claim."""
        if claim.actual_value is None:
            return None

        # Boolean claims: flip the is/is not assertion.
        if isinstance(claim.actual_value, bool):
            if claim.actual_value is True:
                return re.sub(r'\bis\s+not\s+', 'is ', claim.original)
            else:
                return re.sub(r'\bis\s+(?!not\b)', 'is not ', claim.original)

        actual_str = self._format_value(claim.actual_value)

        # For "X = Y" claims, replace Y with actual.
        if "=" in claim.original or "≈" in claim.original:
            parts = re.split(r'[=≈]', claim.original, maxsplit=1)
            if len(parts) == 2:
                return f"{parts[0].rstrip()} = {actual_str}"

        # For "GCD of X and Y is Z", replace Z.
        if re.search(r'(?:is|equals)\s+\S+$', claim.original):
            return re.sub(
                r'((?:is|equals)\s+)\S+$',
                rf'\g<1>{actual_str}',
                claim.original,
            )

        return None

    # Regex to find expressions in preceding text for cross-reference.
    _EXPR_BACKREF_RE = re.compile(
        r'(\d[\d,]*)\s*(?:[\*×]|\\times)\s*(\d[\d,]*)',
    )

    def _find_nearby_expression(self, prefix: str) -> Optional[str]:
        """Find the most recent arithmetic expression in preceding text."""
        # Search backward through the last ~300 chars for an expression.
        search_text = prefix[-300:]
        matches = list(self._EXPR_BACKREF_RE.finditer(search_text))
        if not matches:
            return None
        # Take the last (most recent) match.
        m = matches[-1]
        a = m.group(1).replace(",", "")
        b = m.group(2).replace(",", "")
        return f"{a} * {b}"

    def _normalize_expr(self, expr: str) -> str:
        """Normalize a raw expression for safe_eval."""
        expr = expr.strip()
        # LaTeX operators → Python
        expr = expr.replace("\\times", "*").replace("\\cdot", "*")
        expr = expr.replace("\\div", "/")
        # Unicode operators → Python
        expr = expr.replace("×", "*").replace("÷", "/")
        expr = expr.replace("^", "**")
        expr = expr.replace(",", "")
        # Collapse whitespace.
        expr = re.sub(r'\s+', ' ', expr)
        return expr

    def _normalize_value(self, val) -> str:
        """Normalize a value to a canonical string for comparison."""
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, float):
            if val == int(val):
                return str(int(val))
            return f"{val:.10g}"
        return str(val)

    def _parse_value(self, s: str) -> object:
        """Parse a claimed value string to a Python value."""
        s = s.strip().replace(",", "")
        if s in ("True", "true"):
            return True
        if s in ("False", "false"):
            return False
        try:
            return int(s)
        except ValueError:
            try:
                return float(s)
            except ValueError:
                return s

    def _format_value(self, val) -> str:
        """Format a computed value for display in corrected text."""
        if isinstance(val, bool):
            return str(val)
        if isinstance(val, float) and val == int(val):
            return str(int(val))
        if isinstance(val, int) and abs(val) >= 1000:
            # Add commas for readability.
            s = str(abs(val))
            formatted = ""
            for i, c in enumerate(reversed(s)):
                if i > 0 and i % 3 == 0:
                    formatted = "," + formatted
                formatted = c + formatted
            return ("-" + formatted) if val < 0 else formatted
        return str(val)

    @staticmethod
    def _overlaps(span, seen_spans) -> bool:
        """Check if a span overlaps with any seen span."""
        s, e = span
        for ss, se in seen_spans:
            if s < se and e > ss:
                return True
        return False


# ---------------------------------------------------------------------------
# Auto-CALM Engine — transparent verification without <calm> blocks
# ---------------------------------------------------------------------------

AUTO_SYSTEM_PROMPT = """\
You are a helpful assistant. Answer questions clearly and accurately.
Show your work when doing calculations. Be precise with numbers."""


@dataclass
class AutoCalmResult:
    """Result from the Auto-CALM engine."""
    response: str = ""              # final (possibly corrected) response
    original_response: str = ""     # model's raw output before corrections
    claims_found: int = 0
    claims_corrected: int = 0
    claims_verified: int = 0
    thinking_chars: int = 0
    tok_per_sec: float = 0.0
    corrections: List[Claim] = field(default_factory=list)


class AutoCalmEngine:
    """
    Auto-CALM engine — transparent compute verification.

    No <calm> blocks, no special syntax. The model writes naturally,
    and the engine verifies every computational claim after the fact.
    Wrong claims are corrected in the output.
    """

    def __init__(
        self,
        server: str = "http://localhost:8080",
        system_prompt: str = AUTO_SYSTEM_PROMPT,
        max_tokens: int = 16384,
        thinking_budget: int = 32768,
        precompute: bool = True,
    ):
        self.server = server
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.precompute = precompute
        self.verifier = AutoCalm()

    def run(self, prompt: str, verbose: bool = False) -> AutoCalmResult:
        """Run a prompt, verify claims, retry if wrong (max 1 retry)."""
        import json
        import urllib.request

        result = AutoCalmResult()

        # Layer 2: pre-compute expressions from the prompt.
        precomputed = self._precompute(prompt) if self.precompute else {}
        system = self.system_prompt
        if precomputed:
            facts = "; ".join(f"{k} = {v}" for k, v in precomputed.items())
            system += f"\n\nVerified facts: {facts}"
            if verbose:
                print(f"[precompute] {facts}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Generate initial response.
        content, thinking, timings = self._generate(messages)
        result.original_response = content
        result.thinking_chars = len(thinking)
        result.tok_per_sec = timings.get("predicted_per_second", 0)

        if verbose and thinking:
            preview = thinking[:200].replace('\n', ' ')
            print(f"[think] {len(thinking)} chars: {preview}...")

        # Layer 1: verify inline claims in the response.
        corrected, report = self.verifier.verify_and_correct(content)

        # Layer 2: verify the answer against the prompt expression.
        prompt_check = self._verify_prompt_answer(prompt, corrected)
        if prompt_check:
            report.claims.append(prompt_check)
            if not prompt_check.correct:
                report.corrections += 1
            else:
                report.verified += 1

        # If prompt-level check failed, retry with correction.
        if prompt_check and not prompt_check.correct:
            actual_str = self.verifier._format_value(prompt_check.actual_value)
            if verbose:
                print(f"[auto-calm] answer wrong: {prompt_check.expression}"
                      f" = {actual_str}, not {prompt_check.claimed_value}")
                print(f"[auto-calm] retrying with correction...")

            messages.append({"role": "assistant", "content": content})
            messages.append({"role": "user", "content": (
                f"Your computation has an error. "
                f"The correct value of {prompt_check.expression} is "
                f"{actual_str}. Please give a corrected answer."
            )})

            content2, thinking2, timings2 = self._generate(messages)
            result.thinking_chars += len(thinking2)
            if timings2.get("predicted_per_second", 0):
                result.tok_per_sec = timings2["predicted_per_second"]

            # Re-verify the retry.
            corrected2, report2 = self.verifier.verify_and_correct(content2)
            prompt_check2 = self._verify_prompt_answer(prompt, corrected2)

            if prompt_check2 and prompt_check2.correct:
                corrected = corrected2
                report = report2
                report.claims.append(prompt_check2)
                report.verified += 1
                if verbose:
                    print(f"[auto-calm] retry succeeded")
            else:
                # Retry also failed — use original with correction note.
                corrected += (
                    f"\n\n[Auto-CALM correction: {prompt_check.expression}"
                    f" = {actual_str}, not {prompt_check.claimed_value}]"
                )
                if verbose:
                    print(f"[auto-calm] retry failed, appending note")

        result.response = corrected
        result.claims_found = len(report.claims)
        result.claims_corrected = report.corrections
        result.claims_verified = report.verified
        result.corrections = [c for c in report.claims if not c.correct]

        if verbose:
            print(f"[auto-calm] {result.claims_found} claims: "
                  f"{result.claims_verified} OK, "
                  f"{result.claims_corrected} corrected, "
                  f"{report.unverifiable} unverifiable")
            for c in result.corrections:
                print(f"  FIX: {c.expression} = {c.claimed_value} → {c.actual_value}")

        return result

    def _generate(self, messages):
        """Send a chat completion request. Returns (content, thinking, timings)."""
        import json
        import urllib.request

        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": False,
        }
        if self.thinking_budget > 0:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.thinking_budget

        req = urllib.request.Request(
            f"{self.server}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read())

        choice = data["choices"][0]
        content = choice["message"].get("content", "")
        thinking = choice["message"].get("reasoning_content", "")
        timings = data.get("timings", {})
        return content, thinking, timings

    def _precompute(self, prompt: str) -> dict:
        """
        Extract computable expressions from the prompt and evaluate them.
        Returns {expression: value} for injection into system prompt.
        """
        results = {}

        # Patterns to extract from the prompt.
        patterns = [
            r'[Ww]hat is (.+?)[\?\.]',
            r'[Cc]ompute (.+?)[\?\.]',
            r'[Cc]alculate (.+?)[\?\.]',
            r'[Ff]ind (.+?)[\?\.]',
        ]

        # NL function patterns.
        nl_patterns = [
            (r'the (\d+)(?:st|nd|rd|th) [Ff]ibonacci number', r'fibonacci(\1)'),
            (r'fibonacci\((\d+)\)', r'fibonacci(\1)'),
            (r'factorial\((\d+)\)', r'factorial(\1)'),
            (r'the (\d+)(?:st|nd|rd|th) prime', r'nth_prime(\1)'),
            (r'(?:the )?length of the [Cc]ollatz .+ (\d+)', r'collatz_length(\1)'),
            (r'(?:how long|length).*[Cc]ollatz.*from (\d+)', r'collatz_length(\1)'),
            (r'the [Cc]ollatz sequence (?:starting |)from (\d+)', r'collatz_length(\1)'),
            (r'the digit(?:al)? (?:sum|root) of (\d+)', None),
            (r'the smallest prime (?:greater than|>) (\d+)', r'next_prime(\1)'),
            (r'the (?:prime )?factors of (\d+)', r'factorize(\1)'),
            (r'the GCD of (\d+) and (\d+)', r'gcd(\1, \2)'),
            (r'the LCM of (\d+) and (\d+)', r'lcm(\1, \2)'),
        ]

        for pat in patterns:
            m = re.search(pat, prompt)
            if not m:
                continue
            raw = m.group(1).strip()

            # Try NL patterns first.
            expr = None
            for nl_pat, nl_repl in nl_patterns:
                nl_m = re.search(nl_pat, raw, re.IGNORECASE)
                if nl_m and nl_repl:
                    expr = re.sub(nl_pat, nl_repl, raw, flags=re.IGNORECASE)
                    break

            if not expr:
                expr = self.verifier._normalize_expr(raw)

            try:
                val = safe_eval(expr)
                results[expr] = val
            except ExpressionError:
                pass

        # Also look for computations not in "What is" form.
        for nl_pat, nl_repl in nl_patterns:
            if nl_repl:
                for nl_m in re.finditer(nl_pat, prompt, re.IGNORECASE):
                    expr = re.sub(nl_pat, nl_repl, nl_m.group(0), flags=re.IGNORECASE)
                    if expr not in results:
                        try:
                            val = safe_eval(expr)
                            results[expr] = val
                        except ExpressionError:
                            pass

        return results

    def _verify_prompt_answer(self, prompt: str, response: str) -> Optional[Claim]:
        """
        Extract a computation from the prompt, compute it, and check
        if the model's response contains the correct answer.
        """
        # Extract computable expression from the prompt.
        expr_patterns = [
            (r'[Ww]hat is (.+?)[\?\.]', None),
            (r'[Cc]ompute (.+?)[\?\.]', None),
            (r'[Cc]alculate (.+?)[\?\.]', None),
        ]

        expr = None
        for pat, _ in expr_patterns:
            m = re.search(pat, prompt)
            if m:
                raw = m.group(1).strip()
                expr = self.verifier._normalize_expr(raw)
                break

        if not expr:
            return None

        # Compute the expected answer.
        try:
            expected = safe_eval(expr)
        except ExpressionError:
            return None

        if expected is None:
            return None

        # Check if the response contains the expected answer.
        expected_strs = {str(expected)}
        if isinstance(expected, float) and expected == int(expected):
            expected_strs.add(str(int(expected)))
        if isinstance(expected, int):
            # Comma-formatted version.
            s = str(abs(expected))
            if len(s) > 3:
                formatted = ""
                for i, c in enumerate(reversed(s)):
                    if i > 0 and i % 3 == 0:
                        formatted = "," + formatted
                    formatted = c + formatted
                if expected < 0:
                    formatted = "-" + formatted
                expected_strs.add(formatted)

        response_clean = response.replace(",", "")
        found = any(
            es in response or es.replace(",", "") in response_clean
            for es in expected_strs
        )

        # Find what the model claimed as the answer.
        # Look for numbers near "product/result/answer is" or the
        # largest number that could be the answer.
        answer_m = re.search(
            r'(?:product|result|answer)\s+(?:is|=)\s+[\*]*(\d[\d,]*)',
            response, re.IGNORECASE,
        )
        if answer_m:
            claimed = answer_m.group(1).replace(",", "")
        else:
            numbers = re.findall(r'\b(\d[\d,]*\d)\b', response)
            claimed = numbers[-1].replace(",", "") if numbers else "?"

        return Claim(
            original=f"[prompt: {expr}]",
            expression=expr,
            claimed_value=claimed,
            actual_value=expected,
            correct=found,
            span=(0, 0),
        )


def run_auto(prompt: str, verbose: bool = True, **kwargs) -> AutoCalmResult:
    """CLI convenience function."""
    engine = AutoCalmEngine(**kwargs)
    result = engine.run(prompt, verbose=verbose)

    print(f"\n{'='*60}")
    if result.claims_corrected:
        print(f"CORRECTED Response:\n{result.response}")
        print(f"\nOriginal (wrong):\n{result.original_response}")
    else:
        print(f"Response:\n{result.response}")
    print(f"\nClaims:       {result.claims_found} found, "
          f"{result.claims_verified} OK, {result.claims_corrected} fixed")
    print(f"Thinking:     {result.thinking_chars} chars")
    print(f"Speed:        {result.tok_per_sec:.1f} tok/s")
    return result


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "What is 17 * 23? Is the result prime? "
        "What is its GCD with 782?"
    )
    run_auto(prompt)
