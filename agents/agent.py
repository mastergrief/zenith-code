"""Base agent that communicates with a local LLM (Ollama or llama.cpp) with tool support."""

import json
import urllib.request
from typing import Any

from agents.tools import TOOL_DEFINITIONS, execute_tool
from agents.compact import should_compact, compact_history, detect_context_limit

OLLAMA_URL = "http://localhost:11434/api/chat"
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_MODEL = "qwen3.5:4b"


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
        self.max_context_tokens = max_context_tokens or detect_context_limit(
            model if self.backend == "ollama" else "llamacpp"
        )

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

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
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

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            OLLAMA_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        final_message = None
        with urllib.request.urlopen(req, timeout=120) as resp:
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
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": 2048,
        }
        if self.enable_thinking:
            payload["enable_thinking"] = True

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            LLAMACPP_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())

        choice = result["choices"][0]["message"]
        content = choice.get("content", "")
        reasoning = choice.get("reasoning_content", "")

        # Build Ollama-compatible message dict
        msg: dict[str, Any] = {"role": "assistant", "content": content}
        if reasoning:
            msg["reasoning_content"] = reasoning
        return {"message": msg}

    def _call_llamacpp_stream(self, messages: list[dict], on_event=None) -> dict:
        """Make a streaming request to llama.cpp via SSE."""
        payload: dict[str, Any] = {
            "messages": messages,
            "max_tokens": 2048,
            "stream": True,
        }
        if self.enable_thinking:
            payload["enable_thinking"] = True

        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            LLAMACPP_URL,
            data=data,
            headers={"Content-Type": "application/json"},
        )

        full_content = ""
        full_reasoning = ""
        in_reasoning = False

        with urllib.request.urlopen(req, timeout=120) as resp:
            for raw_line in resp:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line or not line.startswith("data: "):
                    continue
                payload_str = line[6:]
                if payload_str == "[DONE]":
                    break

                chunk = json.loads(payload_str)
                delta = chunk.get("choices", [{}])[0].get("delta", {})

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

        if in_reasoning and on_event:
            on_event("thinking_end", {})

        msg: dict[str, Any] = {"role": "assistant", "content": full_content}
        if full_reasoning:
            msg["reasoning_content"] = full_reasoning
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

        messages = [{"role": "system", "content": self.system_prompt}] + self.history
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

                    tool_output = execute_tool(tool_name, tool_args)

                    if on_event:
                        on_event("tool_result", {"name": tool_name, "output": tool_output})

                    tool_msg = {"role": "tool", "content": tool_output}
                    self.history.append(tool_msg)
                    messages.append(tool_msg)

                # Re-enable streaming for the next response after tools
                use_stream = stream and on_event is not None
            else:
                # Final text response
                reply = msg.get("content", "")
                reasoning = msg.get("reasoning_content", "")
                self.history.append({"role": "assistant", "content": reply})

                if on_event:
                    on_event("response", {"content": reply, "reasoning": reasoning})

                return reply

        return "(max tool rounds reached)"

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self.model!r})"
