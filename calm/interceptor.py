"""
CALM v0.1 stream interceptor — Option B (LLM owns stack state).

Parses CALM instructions from a streamed token string, executes each
against the reference stack_vm, and validates the LLM's stack-state
claims. On mismatch, produces an <error> tag for injection back into
the LLM's thinking stream.

The interceptor is stateful: feed it tokens as they arrive via
feed(), and it accumulates partial lines. When a complete instruction
line is detected, it executes and validates immediately.

Lifecycle:
    interceptor = Interceptor()
    for token in llm_stream:
        events = interceptor.feed(token)
        for event in events:
            if event.type == "error":
                inject(event.text)  # feed back to LLM
            elif event.type == "result":
                ...  # instruction executed OK
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from calm.stack_vm import (
    CalmRuntimeError,
    Dispatcher,
    VMState,
    default_dispatcher,
    parse_program,
)
from calm.nl_parser import normalize_calm_line
from calm.expression import safe_eval, ExpressionError
from calm.sandbox import run_python

# Deferred import to avoid circular dependency at module level.
_VerifiedDispatcher = None

def _is_verified(dispatcher):
    global _VerifiedDispatcher
    if _VerifiedDispatcher is None:
        try:
            from calm.verifier import VerifiedDispatcher
            _VerifiedDispatcher = VerifiedDispatcher
        except ImportError:
            _VerifiedDispatcher = type(None)  # never matches
    return isinstance(dispatcher, _VerifiedDispatcher)


class EventType(Enum):
    CALM_START = "calm_start"      # <calm> detected
    CALM_END = "calm_end"          # </calm> detected
    EXECUTED = "executed"          # instruction ran OK
    VALIDATED = "validated"        # stack claim matches
    MISMATCH = "mismatch"         # stack claim differs from VM
    RESOLVED = "resolved"         # <pending> resolved with actual value
    VERIFIED = "verified"         # triple redundancy: all lanes agree
    DIVERGENCE = "divergence"     # triple redundancy: lanes disagree
    ERROR = "error"               # runtime error from VM
    COMMENT = "comment"           # comment line (no-op)


@dataclass
class Event:
    type: EventType
    instruction: str = ""
    claimed_stack: Optional[List] = None
    actual_stack: Optional[List] = None
    text: str = ""                # human-readable message or error tag


# Regex for Option B: instruction followed by ` -> [...]` or ` -> <pending>`
_CLAIM_RE = re.compile(r'^(.*?)\s+->\s+\[([^\]]*)\]$')
_PENDING_RE = re.compile(r'^(.*?)\s+->\s+<pending>$')


def _parse_stack_claim(claim_str: str) -> List:
    """Parse a stack claim string like '1, 5, "hello"' into Python values."""
    if not claim_str.strip():
        return []
    values = []
    for item in _split_claim_items(claim_str):
        item = item.strip()
        if not item:
            continue
        if item.startswith('"') and item.endswith('"'):
            values.append(item[1:-1])
        elif item == "true":
            values.append(True)
        elif item == "false":
            values.append(False)
        else:
            try:
                values.append(int(item))
            except ValueError:
                try:
                    values.append(float(item))
                except ValueError:
                    values.append(item)
    return values


def _split_claim_items(s: str) -> List[str]:
    """Split claim items on commas, respecting quoted strings."""
    items = []
    current = []
    in_str = False
    for c in s:
        if c == '"':
            in_str = not in_str
            current.append(c)
        elif c == ',' and not in_str:
            items.append(''.join(current))
            current = []
        else:
            current.append(c)
    if current:
        items.append(''.join(current))
    return items


class Interceptor:
    """
    Stateful CALM stream interceptor.

    Feed tokens via feed(). Returns a list of Events per call.
    The interceptor tracks whether we're inside a <calm>...</calm>
    block and only processes lines when inside one.

    Modes:
      strict=True  (default): mismatches generate ERROR events that
                   the harness should inject back into the stream.
      strict=False: mismatches are logged as MISMATCH (training signal)
                   but execution continues with the VM's result.
                   The model never gets blocked by wrong predictions.
    """

    def __init__(
        self,
        dispatcher: Optional[Dispatcher] = None,
        strict: bool = False,
        persist_state: bool = False,
    ):
        self.dispatcher = dispatcher or default_dispatcher()
        self.strict = strict
        self.persist_state = persist_state
        self.state = VMState()
        self._inside_calm = False
        self._buffer = ""  # partial line accumulator
        # Training signal log: (instruction, claimed, actual, correct?)
        self.training_log: List[dict] = []
        # Variable namespace persists across CALM blocks.
        # Model writes "result = expr", next block can use "result".
        self.variables: dict = {}

    # CALM block start patterns: (marker, chars to skip past it)
    _START_MARKERS = [
        ("<calm>", 6),
        ("<|tool_call>call:calm\n", 21),
        ("<|tool_call>call:calm", 20),
    ]
    # CALM block end patterns: (marker, chars to skip past it)
    _END_MARKERS = [
        ("</calm>", 7),
        ("<channel|>", 10),  # Gemma's tool-call end marker
    ]

    def reset(self) -> None:
        """Reset VM state and buffer for a new CALM block."""
        self.state = VMState()
        self._inside_calm = False
        self._buffer = ""

    def _find_calm_start(self):
        """Find the earliest CALM block start in the buffer. Returns (index, skip) or (-1, 0)."""
        best = -1
        best_skip = 0
        for marker, skip in self._START_MARKERS:
            idx = self._buffer.find(marker)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
                best_skip = skip
        return best, best_skip

    def _find_calm_end(self):
        """Find the earliest CALM block end in the buffer. Returns (index, skip) or (-1, 0)."""
        best = -1
        best_skip = 0
        for marker, skip in self._END_MARKERS:
            idx = self._buffer.find(marker)
            if idx != -1 and (best == -1 or idx < best):
                best = idx
                best_skip = skip
        return best, best_skip

    def feed(self, text: str) -> List[Event]:
        """
        Feed a chunk of text (one or more tokens) and return events.

        The text may contain partial lines — the interceptor buffers
        until a newline or </calm> is seen.
        """
        events: List[Event] = []
        self._buffer += text

        while True:
            if not self._inside_calm:
                # Look for CALM block start markers.
                # Support: <calm>, <|tool_call>call:calm
                idx, skip = self._find_calm_start()
                if idx == -1:
                    if len(self._buffer) > 20:
                        self._buffer = self._buffer[-20:]
                    break
                self._buffer = self._buffer[idx + skip:]
                self._inside_calm = True
                if not self.persist_state:
                    self.state = VMState()
                events.append(Event(type=EventType.CALM_START))
                continue

            # Inside <calm> — look for end marker or newlines.
            end_idx, end_skip = self._find_calm_end()
            nl_idx = self._buffer.find("\n")

            if end_idx != -1 and (nl_idx == -1 or end_idx < nl_idx):
                remaining = self._buffer[:end_idx].strip()
                if remaining:
                    # Skip model-fabricated engine results.
                    if not remaining.startswith("[engine:"):
                        events.extend(self._process_line(remaining))
                self._buffer = self._buffer[end_idx + end_skip:]
                self._inside_calm = False
                events.append(Event(type=EventType.CALM_END))
                continue

            if nl_idx == -1:
                # No complete line yet — wait for more tokens.
                break

            # Complete line available.
            line = self._buffer[:nl_idx]
            self._buffer = self._buffer[nl_idx + 1:]
            line = line.strip()
            if line and not line.startswith("[engine:"):
                events.extend(self._process_line(line))

        return events

    def _process_line(self, line: str) -> List[Event]:
        """Process a single complete CALM instruction line."""
        events: List[Event] = []

        # Skip full-line comments (\ or // or # prefix).
        if line.startswith("\\") or line.startswith("//") or line.startswith("#"):
            events.append(Event(
                type=EventType.COMMENT,
                instruction=line,
            ))
            return events

        # Handle Python-style patterns the model naturally writes:
        # - "result = expr" → evaluate expr AND store in variables
        # - "print(expr)" → evaluate expr (treat as emit)
        import re as _re
        var_name = None
        assign_match = _re.match(r'^([a-zA-Z_]\w*)\s*=\s*(.+)$', line)
        if assign_match and '==' not in line:
            var_name = assign_match.group(1)
            line = assign_match.group(2)
        if line.startswith("print(") and line.endswith(")"):
            line = line[6:-1]

        # Strip inline comments: anything after // or \ that's outside
        # the Option B claim. We strip BEFORE claim parsing so that
        # `mul -> [391] // comment` doesn't pollute the claim.
        for marker in ("//", "\\"):
            idx = line.find(marker)
            if idx > 0:
                line = line[:idx].rstrip()

        # Check for Option B stack claim or <pending> placeholder.
        pending_match = _PENDING_RE.match(line)
        claim_match = _CLAIM_RE.match(line) if not pending_match else None
        is_pending = pending_match is not None

        if pending_match:
            instruction_text = pending_match.group(1)
        elif claim_match:
            instruction_text = claim_match.group(1)
        else:
            instruction_text = line

        claimed_stack = None
        if claim_match:
            claimed_stack = _parse_stack_claim(claim_match.group(2))

        # Three-tier parsing strategy:
        # 1. NL normalization ("multiply 17 by 23" → stack code)
        # 2. Standard stack_vm parser ("push 17\npush 23\nmul")
        # 3. Expression evaluator ("17 * 23 + 42 * 19 - 100")
        # Each tier falls through to the next on failure.

        normalized = normalize_calm_line(instruction_text)
        parse_text = normalized if normalized else instruction_text

        # Try standard parse (tiers 1+2).
        parse_ok = True
        try:
            instrs = parse_program(parse_text)
        except Exception:
            parse_ok = False
            instrs = []

        # Try standard execution. On failure, fall through to expression eval.
        stack_executed = False
        if parse_ok and instrs:
            saved_stack = list(self.state.stack)
            stack_error = None
            temp_events = []
            for instr in instrs:
                try:
                    self.dispatcher.execute(self.state, instr)
                except CalmRuntimeError as e:
                    stack_error = e
                    break

                temp_events.append(Event(
                    type=EventType.EXECUTED,
                    instruction=str(instr),
                    actual_stack=list(self.state.stack),
                ))

                # Triple redundancy check.
                if _is_verified(self.dispatcher):
                    v = self.dispatcher.last_verification
                    if v is not None:
                        if v.unanimous:
                            lanes = ", ".join(r.name for r in v.all_results)
                            temp_events.append(Event(
                                type=EventType.VERIFIED,
                                instruction=str(instr),
                                actual_stack=list(self.state.stack),
                                text=f"[{len(v.all_results)} lanes agree: {lanes}]",
                            ))
                        else:
                            details = "; ".join(
                                f"{r.name}={r.stack}" for r in v.all_results
                            )
                            temp_events.append(Event(
                                type=EventType.DIVERGENCE,
                                instruction=str(instr),
                                actual_stack=list(self.state.stack),
                                text=(
                                    f"<error>DIVERGENCE on {instr.word}: "
                                    f"{details}</error>"
                                ),
                            ))

            if stack_error is None:
                events.extend(temp_events)
                stack_executed = True
            elif "unknown word" in str(stack_error):
                # Unknown word — restore stack and try expression eval.
                self.state.stack = saved_stack
            else:
                # Real runtime error (underflow, type mismatch, etc.)
                # Report it directly.
                events.append(Event(
                    type=EventType.ERROR,
                    instruction=instruction_text,
                    actual_stack=list(self.state.stack),
                    text=f"<error>{stack_error}</error>",
                ))
                return events

        if not stack_executed:
            # Tier 3: expression evaluator.
            try:
                result = safe_eval(instruction_text, functions=self.variables)
                self.state.stack.append(result)
                events.append(Event(
                    type=EventType.EXECUTED,
                    instruction=instruction_text,
                    actual_stack=list(self.state.stack),
                    text=f"expr={result}",
                ))
            except ExpressionError:
                # Tier 4: sandboxed Python execution.
                # Prepend variable bindings for the sandbox.
                var_prelude = "\n".join(
                    f"{k} = {repr(v)}" for k, v in self.variables.items()
                )
                sandbox_code = (var_prelude + "\n" + instruction_text) if var_prelude else instruction_text
                sr = run_python(sandbox_code, timeout=5.0)
                if sr.ok and sr.value is not None:
                    self.state.stack.append(sr.value)
                    events.append(Event(
                        type=EventType.EXECUTED,
                        instruction=instruction_text,
                        actual_stack=list(self.state.stack),
                        text=f"python={sr.value}",
                    ))
                elif sr.ok and sr.stdout.strip():
                    # Code printed but returned None — use stdout.
                    events.append(Event(
                        type=EventType.EXECUTED,
                        instruction=instruction_text,
                        actual_stack=list(self.state.stack),
                        text=f"python:stdout={sr.stdout.strip()[:200]}",
                    ))
                else:
                    err = sr.error or "unknown instruction"
                    events.append(Event(
                        type=EventType.ERROR,
                        instruction=instruction_text,
                        text=f"<error>{err}</error>",
                    ))
                return events

        actual = list(self.state.stack)

        # Handle <pending>: model deferred prediction to the VM.
        if is_pending:
            events.append(Event(
                type=EventType.RESOLVED,
                instruction=instruction_text,
                actual_stack=actual,
                text=f"-> {actual}",
            ))
            self._log_training(instruction_text, None, actual, True)
            return events

        # Validate stack claim (Option B).
        if claimed_stack is not None:
            correct = (actual == claimed_stack)
            self._log_training(instruction_text, claimed_stack, actual, correct)

            if correct:
                events.append(Event(
                    type=EventType.VALIDATED,
                    instruction=instruction_text,
                    claimed_stack=claimed_stack,
                    actual_stack=actual,
                ))
            else:
                events.append(Event(
                    type=EventType.MISMATCH,
                    instruction=instruction_text,
                    claimed_stack=claimed_stack,
                    actual_stack=actual,
                    text=(
                        f"<error>stack mismatch: you said {claimed_stack}, "
                        f"VM says {actual}</error>"
                    ),
                ))

        # Store variable binding if this was an assignment.
        if var_name and self.state.stack:
            self.variables[var_name] = self.state.stack[-1]

        return events

    def _log_training(
        self, instruction: str, claimed, actual, correct: bool
    ) -> None:
        """Log a training signal entry."""
        self.training_log.append({
            "instruction": instruction,
            "claimed": claimed,
            "actual": actual,
            "correct": correct,
        })
