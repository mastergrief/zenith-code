"""
Streaming Auto-CALM — verify claims as tokens arrive.

Instead of waiting for the full response, scans the SSE token stream
for computational claims in real-time. When a wrong claim is detected
mid-generation, it's flagged immediately.

The stream accumulates tokens and runs claim verification on every
sentence boundary (period, newline, closing paren). This catches
errors at token ~50 instead of waiting for token ~500.

Usage:
    from calm.stream_auto import StreamingAutoCalmEngine
    engine = StreamingAutoCalmEngine()
    result = engine.run("What is 347 * 289? Is it prime?", verbose=True)
"""

from __future__ import annotations

import json
import re
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from calm.verify import AutoCalm, Claim, VerifyReport
from calm.precompute import build_system_prompt, precompute
from calm.expression import safe_eval, ExpressionError


@dataclass
class StreamClaim:
    """A claim detected during streaming."""
    claim: Claim
    token_offset: int      # how many content tokens in when detected
    time_offset: float     # seconds from stream start when detected


@dataclass
class StreamAutoResult:
    """Result from streaming Auto-CALM."""
    response: str = ""
    original_response: str = ""
    claims_found: int = 0
    claims_corrected: int = 0
    claims_verified: int = 0
    thinking_chars: int = 0
    tok_per_sec: float = 0.0
    stream_claims: List[StreamClaim] = field(default_factory=list)
    first_error_token: int = -1   # token offset of first wrong claim
    first_error_time: float = 0.0 # seconds to first wrong claim
    total_tokens: int = 0


class StreamingAutoCalmEngine:
    """
    Streaming Auto-CALM — real-time claim verification.

    Scans the SSE token stream for claims at sentence boundaries.
    Detects errors as they're generated, not after the full response.
    """

    def __init__(
        self,
        server: str = "http://localhost:8080",
        max_tokens: int = 16384,
        thinking_budget: int = 32768,
        precompute_enabled: bool = True,
        on_claim: Optional[Callable] = None,
    ):
        self.server = server
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget
        self.precompute_enabled = precompute_enabled
        self.on_claim = on_claim  # callback(StreamClaim) for real-time events
        self.verifier = AutoCalm()

    def run(self, prompt: str, verbose: bool = False) -> StreamAutoResult:
        """Stream a response, verify claims in real-time."""
        result = StreamAutoResult()

        # Precompute.
        precomputed = precompute(prompt) if self.precompute_enabled else {}
        from calm.auto_learn import AutoLearner
        learner = AutoLearner()
        learned = learner.suggest_precomputes(prompt)
        if learned:
            precomputed.update(learned)

        system = build_system_prompt()
        if precomputed:
            facts = "; ".join(f"{k} = {v}" for k, v in precomputed.items())
            system += f"\n\nVerified facts: {facts}"
            if verbose:
                print(f"[precompute] {facts}")

        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ]

        # Stream and verify.
        t0 = time.time()
        thinking, content, token_count = self._stream_and_verify(
            messages, result, t0, verbose,
        )

        result.original_response = content
        result.response = content
        result.thinking_chars = len(thinking)
        result.total_tokens = token_count

        # Final full-text verification pass (catches anything streaming missed).
        corrected, report = self.verifier.verify_and_correct(content)
        if report.corrections > 0:
            result.response = corrected
        result.claims_found = len(report.claims) + len(result.stream_claims)
        result.claims_corrected = report.corrections
        result.claims_verified = report.verified

        elapsed = time.time() - t0
        if elapsed > 0 and token_count > 0:
            result.tok_per_sec = token_count / elapsed

        if verbose:
            print(f"\n[stream] {token_count} tokens in {elapsed:.1f}s "
                  f"({result.tok_per_sec:.1f} tok/s)")
            print(f"[stream] {len(result.stream_claims)} claims detected during stream")
            if result.first_error_token >= 0:
                print(f"[stream] first error at token {result.first_error_token} "
                      f"({result.first_error_time:.2f}s)")
            print(f"[final] {result.claims_found} claims: "
                  f"{result.claims_verified} OK, {result.claims_corrected} corrected")

        # Learn from corrections.
        if result.claims_corrected > 0:
            for c in report.claims:
                learner.learn_from_correction(c)

        return result

    def _stream_and_verify(self, messages, result, t0, verbose):
        """Stream SSE tokens, verify claims at sentence boundaries."""
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "stream": True,
        }
        if self.thinking_budget > 0:
            payload["enable_thinking"] = True
            payload["thinking_budget"] = self.thinking_budget

        req = urllib.request.Request(
            f"{self.server}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        thinking = ""
        content = ""
        token_count = 0
        last_checked_len = 0

        # Sentence boundary markers for incremental verification.
        _BOUNDARIES = re.compile(r'[.\n\)\]!?]')

        with urllib.request.urlopen(req, timeout=300) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                if line == "data: [DONE]":
                    break

                try:
                    data = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                delta = data.get("choices", [{}])[0].get("delta", {})
                think_tok = delta.get("reasoning_content", "")
                content_tok = delta.get("content", "")

                if think_tok:
                    thinking += think_tok
                if content_tok:
                    content += content_tok
                    token_count += 1

                    # Check at sentence boundaries.
                    new_text = content[last_checked_len:]
                    if _BOUNDARIES.search(new_text) and len(content) - last_checked_len > 10:
                        claims = self._check_incremental(content, last_checked_len)
                        last_checked_len = len(content)

                        for claim in claims:
                            sc = StreamClaim(
                                claim=claim,
                                token_offset=token_count,
                                time_offset=time.time() - t0,
                            )
                            result.stream_claims.append(sc)

                            if not claim.correct and result.first_error_token < 0:
                                result.first_error_token = token_count
                                result.first_error_time = sc.time_offset

                            if verbose:
                                status = "OK" if claim.correct else "WRONG"
                                print(f"  [{status} @tok {token_count}] "
                                      f"{claim.expression} = {claim.claimed_value}"
                                      + (f" (actual: {claim.actual_value})"
                                         if not claim.correct else ""))

                            if self.on_claim:
                                self.on_claim(sc)

        return thinking, content, token_count

    def _check_incremental(self, content: str, from_pos: int) -> List[Claim]:
        """Check the newly accumulated text for claims."""
        # Strip formatting on the new chunk + some context.
        context_start = max(0, from_pos - 50)
        chunk = self.verifier._strip_formatting(content[context_start:])

        # Extract and verify claims in this chunk.
        numeric = self.verifier.extract_claims(chunk)
        boolean = self.verifier.extract_bool_claims(chunk)
        all_claims = numeric + boolean

        if all_claims:
            self.verifier.verify_claims(all_claims, chunk)

        return all_claims


def run_stream_auto(prompt: str, verbose: bool = True, **kwargs) -> StreamAutoResult:
    """CLI convenience."""
    engine = StreamingAutoCalmEngine(**kwargs)
    result = engine.run(prompt, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"Response:\n{result.response[:800]}")
    print(f"\nStream claims: {len(result.stream_claims)}")
    if result.first_error_token >= 0:
        print(f"First error:   token {result.first_error_token} "
              f"({result.first_error_time:.2f}s)")
    print(f"Final:         {result.claims_found} claims, "
          f"{result.claims_corrected} corrected")
    print(f"Speed:         {result.tok_per_sec:.1f} tok/s")
    return result


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "What is 347 * 289? Is the result prime?"
    )
    run_stream_auto(prompt)
