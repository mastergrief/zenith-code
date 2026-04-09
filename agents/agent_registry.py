"""Module-global registry of persistent sub-agents addressable by ID.

This is the foundation for the AgentCreate / AgentMessage / AgentGet /
AgentList / AgentTerminate tool family. Sub-agents registered here outlive
a single tool call, so a parent agent can spawn one, then re-engage it on
later turns by ID — enabling cross-validation, iterative dialogue, and
multi-agent orchestration patterns.

The existing one-shot `Agent` tool (`agents.tools._execute_agent_tool`) does
NOT touch the registry — it spawns an ephemeral sub-agent, runs it once,
returns the text, and discards. The new tools use the registry exclusively.

Threading: a single Lock guards the registry dict. We're sync today
(`--parallel 1` at llama-server), but the lock makes it safe for any
future thread-based async dispatch.
"""

from __future__ import annotations

import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agents.agent import Agent


_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")


@dataclass
class RegisteredAgent:
    """Holds an Agent instance plus registry metadata."""
    agent: "Agent"
    name: str
    created_at: float = field(default_factory=time.time)
    last_active_at: float = field(default_factory=time.time)


class AgentRegistry:
    """Process-global mapping from agent_id → RegisteredAgent.

    The same instance is shared across all tool calls in a session via
    the module-level `_REGISTRY` singleton (use `get_registry()`).
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._agents: dict[str, RegisteredAgent] = {}

    def register(self, agent: "Agent", agent_id: str | None = None) -> str:
        """Add an Agent to the registry under `agent_id` (or auto-generated).

        If `agent_id` is provided, it MUST be unique and match `_NAME_RE`
        (lowercase alphanum + `_-`, 1-40 chars). Returns the actual ID used.
        Raises ValueError on collision or invalid name.
        """
        with self._lock:
            if agent_id is None:
                aid = self._mint_id_locked()
            else:
                if not _NAME_RE.match(agent_id):
                    raise ValueError(
                        f"agent_id {agent_id!r} must match {_NAME_RE.pattern}"
                    )
                if agent_id in self._agents:
                    raise ValueError(f"agent_id {agent_id!r} already exists")
                aid = agent_id
            self._agents[aid] = RegisteredAgent(agent=agent, name=aid)
            return aid

    def get(self, agent_id: str) -> RegisteredAgent | None:
        with self._lock:
            return self._agents.get(agent_id)

    def terminate(self, agent_id: str) -> bool:
        """Remove an agent from the registry. Returns True if removed."""
        with self._lock:
            return self._agents.pop(agent_id, None) is not None

    def list_agents(self) -> list[dict]:
        """Snapshot of all live agents — id, role, history length, age."""
        with self._lock:
            now = time.time()
            return [
                {
                    "id": aid,
                    "role": ra.agent.role,
                    "model": ra.agent.model,
                    "history_len": len(ra.agent.history),
                    "todos_len": len(getattr(ra.agent, "todos", []) or []),
                    "created_at": ra.created_at,
                    "last_active_at": ra.last_active_at,
                    "age_seconds": round(now - ra.created_at, 1),
                }
                for aid, ra in self._agents.items()
            ]

    def touch(self, agent_id: str) -> None:
        """Update last_active_at after the agent processes a message."""
        with self._lock:
            ra = self._agents.get(agent_id)
            if ra is not None:
                ra.last_active_at = time.time()

    def clear(self) -> None:
        """Remove all agents. Used by tests and `/reset`."""
        with self._lock:
            self._agents.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._agents)

    def _mint_id_locked(self) -> str:
        """Generate a short unique ID (must be called with lock held)."""
        # 8-char hex prefix, e.g. "agt_a1b2c3d4". Collisions exceedingly rare
        # but we retry to be safe.
        for _ in range(10):
            candidate = f"agt_{uuid.uuid4().hex[:8]}"
            if candidate not in self._agents:
                return candidate
        raise RuntimeError("could not mint a unique agent_id after 10 tries")


_REGISTRY: AgentRegistry | None = None


def get_registry() -> AgentRegistry:
    """Return the process-global AgentRegistry singleton."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = AgentRegistry()
    return _REGISTRY


def reset_registry() -> None:
    """Reset the singleton — used by tests and harness `/reset`."""
    global _REGISTRY
    _REGISTRY = AgentRegistry()
