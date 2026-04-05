"""Base agent that communicates with a local Ollama model with tool support."""

import json
import urllib.request
from typing import Any

from agents.tools import TOOL_DEFINITIONS, execute_tool
from agents.compact import should_compact, compact_history, detect_context_limit

OLLAMA_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen3.5:4b"


class Agent:
    """A single agent backed by a local LLM via Ollama with tool calling."""

    def __init__(
        self,
        name: str,
        role: str,
        model: str = DEFAULT_MODEL,
        system_prompt: str | None = None,
        tools: bool = True,
        max_tool_rounds: int = 10,
        max_context_tokens: int | None = None,
    ):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt or f"You are {name}, a {role}."
        self.history: list[dict[str, Any]] = []
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds
        self.max_context_tokens = max_context_tokens or detect_context_limit(model)

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

    def chat(self, message: str, think: bool = True, on_event=None, stream: bool = True) -> str:
        """Send a message and get a response, executing tools as needed.

        Args:
            message: The user message.
            think: Whether to enable chain-of-thought.
            on_event: Optional callback(event_type, data) for streaming updates.
                      event_type: 'token', 'tool_call', 'tool_result', 'response', 'compact'
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

        for _ in range(self.max_tool_rounds):
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
                self.history.append({"role": "assistant", "content": reply})

                if on_event:
                    on_event("response", {"content": reply})

                return reply

        return "(max tool rounds reached)"

    def reset(self):
        """Clear conversation history."""
        self.history.clear()

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, role={self.role!r}, model={self.model!r})"
