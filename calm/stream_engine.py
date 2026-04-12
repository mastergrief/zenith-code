"""
CALM v0.2 streaming engine — SSE-based interception with thinking on every turn.

Unlike v0.1 (engine.py) which uses stop=["</calm>"], this engine
streams tokens via SSE and detects </calm> in the stream. This means:
- Thinking works on EVERY turn (not just turn 0)
- No assistant prefill needed
- Mid-stream interception: detect <calm>...</calm> as tokens arrive

The flow:
  1. Send chat completion with stream=True, enable_thinking=True
  2. Accumulate tokens, watching for <calm>...</calm> boundaries
  3. When </calm> detected: process the block, build injection
  4. Append assistant turn (with block + injection) + user turn
  5. Start a new streaming request (with thinking on the new turn)
  6. Repeat until no more <calm> blocks

Usage:
    from calm.stream_engine import StreamEngine
    engine = StreamEngine()
    result = engine.run("Find bugs in auth.py and fix them")
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass, field
from typing import List, Optional

from calm.interceptor import Event, EventType, Interceptor
from calm.training import TrainingCollector
from calm.verifier import make_verified_dispatcher


SERVER = "http://localhost:8080"


@dataclass
class StreamResult:
    response: str = ""
    thinking_chars: int = 0
    calm_blocks: int = 0
    vm_outputs: list = field(default_factory=list)
    training_log: list = field(default_factory=list)
    iterations: int = 0
    tok_per_sec: float = 0.0


class StreamEngine:
    """CALM engine with SSE streaming — thinking on every turn."""

    # Import system prompt from v0.1 engine
    from calm.engine import SYSTEM_PROMPT

    def __init__(
        self,
        server: str = SERVER,
        system_prompt: str = None,
        max_iterations: int = 30,
        max_tokens_per_turn: int = 16384,
        thinking_budget: int = 32768,
    ):
        self.server = server
        self.system_prompt = system_prompt or self.SYSTEM_PROMPT
        self.max_iterations = max_iterations
        self.max_tokens_per_turn = max_tokens_per_turn
        self.thinking_budget = thinking_budget
        self.dispatcher = make_verified_dispatcher()

    def run(self, prompt: str, verbose: bool = False) -> StreamResult:
        result = StreamResult()
        assembled = ""
        interceptor = Interceptor(
            dispatcher=self.dispatcher, strict=False, persist_state=False,
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt},
        ]

        consecutive_errors = 0

        for i in range(self.max_iterations):
            result.iterations = i + 1

            # Stream with thinking enabled on EVERY turn.
            thinking, content = self._stream_until_calm_end(messages, verbose, i)

            result.thinking_chars += len(thinking)
            if verbose and thinking:
                preview = thinking[:150].replace('\n', ' ')
                print(f"[iter {i+1}] think: {len(thinking)} chars: {preview}...")

            # Check if we captured a complete CALM block.
            has_calm = (
                ("<calm>" in content or "<|tool_call>call:calm" in content or "<|tool_call>call:code" in content or "<|tool_call>call:security" in content)
                and ("</calm>" in content or "<channel|>" in content or self._ends_at_calm_close)
            )

            if has_calm:
                import re

                # Extract raw block content for sandbox fallback.
                calm_match = re.search(
                    r'(?:<calm>|<\|tool_call>call:\w+\n?)(.*?)(?:</calm>|<channel\|>|$)',
                    content, re.DOTALL,
                )
                raw_block = calm_match.group(1).strip() if calm_match else ""

                # Try interceptor first (handles simple expressions).
                events = interceptor.feed(content)
                block_errors = [e for e in events if e.type == EventType.ERROR]

                # If interceptor had errors, try the WHOLE block as
                # one Python chunk via sandbox. This handles multi-line
                # code like triple-quoted strings, for loops, etc.
                if block_errors and raw_block:
                    from calm.sandbox import run_python
                    clean = "\n".join(
                        ln for ln in raw_block.splitlines()
                        if not ln.strip().startswith(("[engine:", ))
                    )
                    clean = re.sub(r'\s*->.*$', '', clean, flags=re.MULTILINE)
                    if clean.strip():
                        sr = run_python(clean, timeout=15.0)
                        if sr.ok:
                            val = sr.value if sr.value is not None else sr.stdout.strip()
                            if val:
                                interceptor.state.stack = [val]
                            block_errors = []
                            events = [
                                Event(type=EventType.CALM_START),
                                Event(type=EventType.EXECUTED,
                                      instruction="[python block]",
                                      actual_stack=list(interceptor.state.stack),
                                      text=f"python={str(val)[:200]}"),
                                Event(type=EventType.CALM_END),
                            ]

                result.calm_blocks += 1
                result.vm_outputs = list(interceptor.state.output)

                injection = self._format_injection(
                    list(interceptor.state.stack),
                    list(interceptor.state.output),
                    block_errors,
                )

                if block_errors:
                    consecutive_errors += 1
                else:
                    consecutive_errors = 0

                if verbose:
                    print(f"[iter {i+1}] CALM block {result.calm_blocks}"
                          + (f" ({consecutive_errors} errs)" if block_errors else ""))
                    print(f"  inject: {injection[:200]}")

                if consecutive_errors >= 3:
                    if verbose:
                        print(f"[iter {i+1}] bailing — 3 consecutive errors")
                    assembled += content
                    break

                assembled += content + "\n" + injection + "\n"

                # Multi-turn: close assistant, add injection as user.
                messages.append({"role": "assistant", "content": content + "\n" + injection})
                if block_errors:
                    messages.append({
                        "role": "user",
                        "content": (
                            f"Engine result: {injection}\n"
                            f"Some errors occurred. Write simpler expressions. "
                            f"Continue answering."
                        ),
                    })
                else:
                    messages.append({
                        "role": "user",
                        "content": f"Engine result: {injection} Continue answering.",
                    })

            else:
                # No CALM block — natural finish.
                assembled += content
                interceptor.feed(content)
                if verbose:
                    print(f"[iter {i+1}] done ({len(content)} chars, "
                          f"{len(thinking)} chars thinking)")
                break

        result.response = assembled
        result.training_log = interceptor.training_log

        if result.training_log:
            collector = TrainingCollector()
            collector.save(result, prompt=prompt)

        return result

    def _stream_until_calm_end(self, messages, verbose, iteration):
        """
        Stream SSE tokens. Accumulate thinking + content.
        Stop streaming when </calm> is detected in content, OR when
        the stream ends naturally.
        """
        payload = {
            "messages": messages,
            "temperature": 0.0,
            "max_tokens": self.max_tokens_per_turn,
            "stream": True,
            "enable_thinking": True,
            "thinking_budget": self.thinking_budget,
        }

        req = urllib.request.Request(
            f"{self.server}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )

        thinking = ""
        content = ""
        self._ends_at_calm_close = False

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

                    # Check for CALM block end in accumulated content.
                    if "</calm>" in content or "<channel|>" in content:
                        self._ends_at_calm_close = True
                        break

        return thinking, content

    def _format_injection(self, stack, output, errors):
        """Same format as v0.1 engine, with readable output."""
        from calm.engine import CalmEngine
        return CalmEngine._format_injection(
            CalmEngine, stack, output, errors, []
        )


def run_stream(prompt: str, verbose: bool = True, **kwargs) -> StreamResult:
    """Convenience function."""
    engine = StreamEngine(**kwargs)
    result = engine.run(prompt, verbose=verbose)

    print(f"\n{'='*60}")
    print(f"Response:\n{result.response[-800:]}")
    print(f"\nCALM blocks:  {result.calm_blocks}")
    print(f"Thinking:     {result.thinking_chars} chars total")
    print(f"Iterations:   {result.iterations}")
    print(f"Training log: {len(result.training_log)} entries")
    return result


if __name__ == "__main__":
    import sys
    prompt = sys.argv[1] if len(sys.argv) > 1 else (
        "Review /tmp/subtle_vulns.py for security issues. "
        "What are the top 3 most dangerous findings? "
        "Suggest a fix for each one."
    )
    run_stream(prompt)
