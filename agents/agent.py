"""Base agent that communicates with a local LLM (Ollama or llama.cpp) with tool support."""

import json
import os
import time
import urllib.request
from typing import Any

from agents.tools import TOOL_DEFINITIONS, execute_tool
from agents.compact import should_compact, compact_history, detect_context_limit

OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5:4b"

EFFORT_LEVELS = {
    "low": {
        "max_tokens": 1024,
        "prompt_prefix": "Be concise and direct. Skip lengthy reasoning.",
    },
    "medium": {
        "max_tokens": 2048,
        "prompt_prefix": "",
    },
    "max": {
        "max_tokens": 8192,
        "prompt_prefix": "Think deeply and carefully. Explore multiple approaches, consider edge cases, verify your reasoning step by step before answering.",
    },
}


def detect_backend() -> str:
    """Auto-detect which backend is available. Prefers llama.cpp."""
    try:
        req = urllib.request.Request("http://localhost:8080/health")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.loads(resp.read())
            if data.get("status") == "ok":
                return "llamacpp"
    except Exception:
        pass
    try:
        req = urllib.request.Request("http://localhost:11434/api/tags")
        with urllib.request.urlopen(req, timeout=2) as resp:
            if resp.status == 200:
                return "ollama"
    except Exception:
        pass
    return "ollama"


def _find_halved_duplicate(text: str, min_half: int = 40) -> str | None:
    """Detect exact `A + A` duplication with any (or no) separator between copies.

    Walks split points near the midpoint with whitespace tolerance. Returns the
    deduplicated first half if found, else None. Catches the case where the model
    re-emits a full response back-to-back without a paragraph boundary.
    """
    stripped = text.strip()
    n = len(stripped)
    if n < min_half * 2:
        return None
    mid = n // 2
    # Allow the split to drift to tolerate small separators (spaces, newlines)
    # between the copies. Bounded so we don't match degenerate short prefixes.
    drift = max(8, min(n // 4, 256))
    for offset in range(0, drift + 1):
        for pos in (mid + offset, mid - offset) if offset else (mid,):
            if pos < min_half or pos > n - min_half:
                continue
            first = stripped[:pos].rstrip()
            second = stripped[pos:].lstrip()
            if first and first == second:
                return first
    return None


def _is_repeating(text: str, window: int = 100) -> bool:
    """Detect if the end of text is repeating an earlier section.

    Two signals:
    1. Exact-half duplication: the text is an `A + A` pattern with any separator.
       Catches the case where the backend emits the full response in 1-2 large
       chunks, before the tail-window check has a chance to fire incrementally.
    2. Tail repetition: the last `window` chars appear earlier in the text.
       Catches incremental streaming loops token-by-token.
    """
    if len(text) < window * 2:
        return False
    # Signal 1: full-response duplication (handles large-chunk deltas)
    if _find_halved_duplicate(text, min_half=window) is not None:
        return True
    # Signal 2: tail repetition (handles token-by-token streaming loops)
    tail = text[-window:]
    earlier = text[:-window]
    return tail in earlier


def _dedup_blocks(text: str, min_block: int = 20) -> str:
    """Remove repeated text blocks from model output.

    Order of operations:
    1. Exact `A + A` halving check (any separator, including none) — catches
       the common case of the model re-emitting the full response back-to-back.
    2. Paragraph-boundary midpoint check — legacy path for `A + \\n\\n + A`.
    3. Paragraph-level dedup — removes duplicate paragraphs/list items.
    """
    if len(text) < min_block * 2:
        return text
    # 1. Exact full-response duplication regardless of separator.
    halved = _find_halved_duplicate(text, min_half=min_block * 2)
    if halved is not None:
        return halved
    # 2. Legacy: find a \n\n boundary near the midpoint and compare halves.
    mid = len(text) // 2
    for offset in range(0, min(200, mid)):
        for pos in (mid + offset, mid - offset):
            if pos < len(text) - 1 and text[pos:pos+2] == "\n\n":
                first, second = text[:pos].strip(), text[pos:].strip()
                if first and second and first == second:
                    return first
                break
    # 3. Paragraph-level dedup.
    blocks = text.split("\n\n")
    seen = []
    for block in blocks:
        stripped = block.strip()
        if stripped and stripped not in seen:
            seen.append(stripped)
    return "\n\n".join(seen)


def _find_claude_md() -> str | None:
    """Walk up from cwd looking for CLAUDE.md."""
    d = os.getcwd()
    for _ in range(5):
        for name in ("CLAUDE.md", ".claude/CLAUDE.md"):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                with open(p) as f:
                    return f.read()
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


class Agent:
    """A single agent backed by a local LLM via Ollama or llama.cpp with tool calling."""

    def __init__(
        self,
        name: str,
        role: str,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        tools: bool = True,
        max_tool_rounds: int = 10,
        max_context_tokens: int | None = None,
        backend: str | None = None,
        enable_thinking: bool = True,
    ):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt or f"You are {name}, a {role}."
        self.history: list[dict[str, Any]] = []
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.backend = backend or detect_backend()
        self.enable_thinking = enable_thinking
        self.effort = "medium"
        self.max_context_tokens = max_context_tokens or detect_context_limit(
            model if self.backend == "ollama" else "llamacpp"
        )
        self._last_usage = {"input_tokens": 0, "output_tokens": 0}

    # ── Shared helpers ──────────────────────────────────────────────

    def _request_with_retry(self, url, payload, max_retries=3):
        """Make HTTP request with exponential backoff."""
        data = json.dumps(payload).encode()
        delays = [1, 2, 4]
        last_error = None
        for attempt in range(max_retries):
            try:
                req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
                return urllib.request.urlopen(req, timeout=120)
            except (urllib.error.URLError, ConnectionError, OSError) as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(delays[attempt])
        backend = "llama-server" if "8080" in url else "Ollama"
        raise ConnectionError(f"{backend} unreachable after {max_retries} attempts: {last_error}")

    def build_system_prompt(self) -> str:
        """Build a ~500 token system prompt with project context."""
        effort_prefix = EFFORT_LEVELS.get(self.effort, {}).get("prompt_prefix", "")
        parts = [effort_prefix, self.system_prompt] if effort_prefix else [self.system_prompt]
        parts.append(f"\nWorking directory: {os.getcwd()}")
        parts.append(f"Date: {time.strftime('%Y-%m-%d')}")

        claude_md = _find_claude_md()
        if claude_md:
            parts.append(f"\nProject context (from CLAUDE.md):\n{claude_md[:1000]}")

        if self.tools:
            tool_names = [t["function"]["name"] for t in TOOL_DEFINITIONS]
            parts.append(f"\nAvailable tools: {', '.join(tool_names)}")

        prompt = "\n".join(parts)
        return prompt[:2000]

    def _estimate_tokens(self, messages):
        """Estimate token count from messages (chars / 4)."""
        total = sum(len(m.get("content", "")) for m in messages)
        return total // 4

    # ── Ollama backend ──────────────────────────────────────────────

    def _call_ollama(self, messages: list[dict]) -> dict:
        """Make a non-streaming request to the Ollama API."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": False,
        }
        if self.tools:
            payload["tools"] = TOOL_DEFINITIONS

        with self._request_with_retry(OLLAMA_URL, payload) as resp:
            return json.loads(resp.read())

    def _call_ollama_stream(self, messages: list[dict], on_event=None) -> dict:
        """Make a streaming request. Yields tokens via on_event, returns final message."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
        }
        if self.tools:
            payload["tools"] = TOOL_DEFINITIONS

        final_message = None
        with self._request_with_retry(OLLAMA_URL, payload) as resp:
            for line in resp:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                msg = chunk.get("message", {})

                # Stream text tokens
                token = msg.get("content", "")
                if token and on_event:
                    on_event("token", {"text": token})

                if chunk.get("done"):
                    final_message = msg
                    break

        return final_message or {"role": "assistant", "content": ""}

    # ── llama.cpp backend (OpenAI-compatible API) ───────────────────

    def _call_llamacpp(self, messages: list[dict]) -> dict:
        """Make a non-streaming request to the llama.cpp API."""
        max_tok = EFFORT_LEVELS.get(self.effort, {}).get("max_tokens", 2048)
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": 0.7,
            "frequency_penalty": 0.8,
            "presence_penalty": 0.3,
        }
        if self.enable_thinking:
            payload["enable_thinking"] = True
        if self.tools:
            payload["tools"] = TOOL_DEFINITIONS

        with self._request_with_retry(LLAMACPP_URL, payload) as resp:
            result = json.loads(resp.read())

        choice = result["choices"][0]["message"]
        content = choice.get("content", "")
        reasoning = choice.get("reasoning_content", "")

        # Build Ollama-compatible message dict
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        # Pass through tool_calls if present
        if choice.get("tool_calls"):
            msg["tool_calls"] = choice["tool_calls"]
        return {"message": msg}

    def _call_llamacpp_stream(self, messages: list[dict], on_event=None) -> dict:
        """Make a streaming request to llama.cpp via SSE."""
        max_tok = EFFORT_LEVELS.get(self.effort, {}).get("max_tokens", 2048)
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": max_tok,
            "stream": True,
            "temperature": 0.7,
            "frequency_penalty": 0.8,
            "presence_penalty": 0.3,
        }
        if self.enable_thinking:
            payload["enable_thinking"] = True
        if self.tools:
            payload["tools"] = TOOL_DEFINITIONS

        full_content = ""
        full_reasoning = ""
        in_reasoning = False
        tool_calls: list[dict] = []
        tc_index = -1

        with self._request_with_retry(LLAMACPP_URL, payload) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break

                chunk = json.loads(payload_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Tool call deltas
                tc_deltas = delta.get("tool_calls")
                if tc_deltas:
                    if in_reasoning and on_event:
                        on_event("thinking_end", {})
                        in_reasoning = False
                    for tcd in tc_deltas:
                        idx = tcd.get("index", 0)
                        if idx > tc_index:
                            tc_index = idx
                            tool_calls.append({
                                "id": tcd.get("id", f"call_{idx}"),
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            })
                        fn = tcd.get("function", {})
                        if fn.get("name"):
                            tool_calls[idx]["function"]["name"] = fn["name"]
                        if fn.get("arguments"):
                            tool_calls[idx]["function"]["arguments"] += fn["arguments"]
                    continue

                # Reasoning content (thinking)
                reasoning_token = delta.get("reasoning_content", "")
                if reasoning_token:
                    if not in_reasoning and on_event:
                        on_event("thinking_start", {})
                    in_reasoning = True
                    full_reasoning += reasoning_token
                    if on_event:
                        on_event("thinking_token", {"text": reasoning_token})
                    continue

                # Regular content
                token = delta.get("content", "")
                if token:
                    if in_reasoning and on_event:
                        on_event("thinking_end", {})
                        in_reasoning = False
                    full_content += token
                    if on_event:
                        on_event("token", {"text": token})
                    # Streaming repetition detection: if content is long enough,
                    # check if recent output repeats an earlier section
                    if len(full_content) > 200 and _is_repeating(full_content):
                        if on_event:
                            on_event("token", {"text": "\n\n[truncated: repetition detected]"})
                        break

        if in_reasoning and on_event:
            on_event("thinking_end", {})

        msg: dict[str, Any] = {"role": "assistant", "content": full_content}
        if full_reasoning:
            msg["reasoning_content"] = full_reasoning
        if tool_calls:
            # Parse arguments from JSON strings
            for tc in tool_calls:
                args_str = tc["function"]["arguments"]
                try:
                    tc["function"]["arguments"] = json.loads(args_str) if args_str else {}
                except json.JSONDecodeError:
                    tc["function"]["arguments"] = {}
            msg["tool_calls"] = tool_calls
        return msg

    def chat(self, message: str, think: bool = True, on_event=None, stream: bool = True) -> str:
        """Send a message and get a response, executing tools as needed.

        Args:
            message: The user message.
            think: Whether to enable chain-of-thought.
            on_event: Optional callback(event_type, data) for streaming updates.
                      event_type: 'token', 'thinking_token', 'thinking_start',
                                  'thinking_end', 'tool_call', 'tool_result',
                                  'response', 'compact'
            stream: Whether to stream tokens (only works with on_event).
        """
        content = message if think else f"/no_think {message}"
        self.history.append({"role": "user", "content": content})

        # Auto-compact if history exceeds context budget
        if should_compact(self.history, self.max_context_tokens):
            old_len = len(self.history)
            self.history = compact_history(self.history)
            if on_event:
                on_event("compact", {"removed": old_len - len(self.history), "remaining": len(self.history)})

        messages = [{"role": "system", "content": self.build_system_prompt()}] + self.history
        use_stream = stream and on_event is not None
        is_llamacpp = self.backend == "llamacpp"

        for _ in range(self.max_tool_rounds):
            if is_llamacpp:
                if use_stream:
                    msg = self._call_llamacpp_stream(messages, on_event=on_event)
                else:
                    result = self._call_llamacpp(messages)
                    msg = result["message"]
            else:
                if use_stream:
                    msg = self._call_ollama_stream(messages, on_event=on_event)
                else:
                    result = self._call_ollama(messages)
                    msg = result["message"]

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Agent wants to use tools — switch to non-streaming for tool rounds
                use_stream = False
                self.history.append(msg)
                messages.append(msg)

                for tc in tool_calls:
                    fn = tc["function"]
                    tool_name = fn["name"]
                    tool_args = fn["arguments"]

                    if on_event:
                        on_event("tool_call", {"name": tool_name, "args": tool_args})

                    confirm_fn = getattr(self, 'confirm_fn', None)
                    perm_mode = getattr(self, 'permission_mode', None)
                    tool_output = execute_tool(tool_name, tool_args, confirm_fn=confirm_fn, mode=perm_mode)

                    if on_event:
                        on_event("tool_result", {"name": tool_name, "output": tool_output})

                    tool_msg = {"role": "tool", "content": tool_output}
                    # llama.cpp (OpenAI API) requires tool_call_id
                    tc_id = tc.get("id")
                    if tc_id:
                        tool_msg["tool_call_id"] = tc_id
                    self.history.append(tool_msg)
                    messages.append(tool_msg)

                # Re-enable streaming for the next response after tools
                use_stream = stream and on_event is not None
            else:
                # Final text response
                reply = _dedup_blocks(msg.get("content", ""))
                reasoning = msg.get("reasoning_content", "")
                self.history.append({"role": "assistant", "content": reply})

                self._last_usage = {
                    "input_tokens": self._estimate_tokens(messages),
                    "output_tokens": self._estimate_tokens([{"content": reply}]),
                }

                if on_event:
                    on_event("response", {"content": reply, "reasoning": reasoning})
                    on_event("usage", self._last_usage)

                return reply

        return "(max tool rounds reached)"

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self.model!r})"
