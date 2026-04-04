"""Base agent that communicates with a local Ollama model with tool support."""

import json
import urllib.request
from typing import Any

from agents.tools import TOOL_DEFINITIONS, execute_tool

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
    ):
        self.name = name
        self.role = role
        self.model = model
        self.system_prompt = system_prompt or f"You are {name}, a {role}."
        self.history: list[dict[str, Any]] = []
        self.tools = tools
        self.max_tool_rounds = max_tool_rounds

    def _call_ollama(self, messages: list[dict], stream: bool = False) -> dict:
        """Make a request to the Ollama API."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": stream,
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

    def chat(self, message: str, think: bool = True, on_event=None) -> str:
        """Send a message and get a response, executing tools as needed.

        Args:
            message: The user message.
            think: Whether to enable chain-of-thought.
            on_event: Optional callback(event_type, data) for streaming updates.
                      event_type: 'thinking', 'tool_call', 'tool_result', 'response'
        """
        content = message if think else f"/no_think {message}"
        self.history.append({"role": "user", "content": content})

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        for _ in range(self.max_tool_rounds):
            result = self._call_ollama(messages)
            msg = result["message"]

            tool_calls = msg.get("tool_calls")
            if tool_calls:
                # Agent wants to use tools
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
