"""SpecialistCoordinator: routes tasks to domain-specific fine-tuned models.

Two operating modes, auto-selected at construction:

1. **Hot-swap mode** — when llama-server is running and specialist GGUFs
   are discovered on disk. All agents share the same :8080 endpoint; the
   coordinator swaps the loaded model before each delegated call. This is
   the intended deployment on 8 GB VRAM where only one specialist fits.

2. **Ollama multi-model mode** (fallback) — when specialist Ollama models
   are pulled. Each agent has its own model name; Ollama keeps them hot.

If neither is available, every worker falls back to the base model.
"""

from __future__ import annotations

import json
import urllib.request
from pathlib import Path
from typing import Optional

from agents.agent import Agent
from agents.coordinator import Coordinator
from agents.distill.config import DOMAINS, OLLAMA_URL
from agents.model_swap import (
    LlamaServerManager,
    ModelSwapError,
    default_base_model,
    discover_specialist_models,
)


def detect_specialists() -> dict[str, str]:
    """Query Ollama for available specialist models. Returns {domain: model_name}."""
    try:
        req = urllib.request.Request(f"{OLLAMA_URL}/api/tags")
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read())

        available = {}
        ollama_models = {m["name"] for m in data.get("models", [])}

        for domain, config in DOMAINS.items():
            ollama_name = config["ollama_name"]
            # Check both "specialist-py" and "specialist-py:latest"
            if ollama_name in ollama_models or f"{ollama_name}:latest" in ollama_models:
                available[domain] = ollama_name

        return available
    except Exception:
        return {}


class SpecialistCoordinator(Coordinator):
    """Coordinator that routes tasks to distilled specialist models.

    Auto-selects between hot-swap (llama.cpp) and multi-model (Ollama) modes
    based on what's available. In hot-swap mode, ``self._hot_swap`` is True
    and ``self._llama_mgr`` is set; swaps happen inside ``run()``.
    """

    def __init__(self, fallback_model: str = "qwen3.5:4b"):
        self.fallback_model = fallback_model
        self._hot_swap = False
        self._llama_mgr: Optional[LlamaServerManager] = None
        self._specialist_paths: dict[str, Path] = {}
        self._base_path: Optional[Path] = None
        self._swap_log: list[tuple[str, float]] = []  # (model_name, elapsed)

        # ── Path 1: hot-swap eligibility ──
        llama_mgr = LlamaServerManager()
        if llama_mgr.is_running():
            specialists_on_disk = discover_specialist_models()
            if specialists_on_disk:
                self._enter_hot_swap_mode(llama_mgr, specialists_on_disk)
                return

        # ── Path 2: Ollama multi-model ──
        available = detect_specialists()
        agents = []
        for domain, config in DOMAINS.items():
            if domain == "orchestrator":
                continue
            model = available.get(domain, fallback_model)
            agent = Agent(
                name=domain,
                role=config["description"],
                model=model,
                system_prompt=config["system_prompt"],
            )
            agents.append(agent)

        super().__init__(agents, model=fallback_model)

        if "orchestrator" in available:
            self.leader.model = available["orchestrator"]

        self._available = available

    # ── Hot-swap setup ─────────────────────────────────────────────

    def _enter_hot_swap_mode(
        self,
        llama_mgr: LlamaServerManager,
        specialists_on_disk: dict[str, Path],
    ) -> None:
        """Initialize hot-swap state and build workers pointing at :8080."""
        self._hot_swap = True
        self._llama_mgr = llama_mgr
        self._specialist_paths = specialists_on_disk
        self._base_path = llama_mgr.current_model() or default_base_model()

        agents = []
        for domain, config in DOMAINS.items():
            if domain == "orchestrator":
                continue
            # All workers hit the same llamacpp endpoint; the model that
            # responds is whichever GGUF is currently loaded. The ``model``
            # field is a placeholder for display/logging.
            agent = Agent(
                name=domain,
                role=config["description"],
                model=f"hot-swap:{domain}",
                backend="llamacpp",
                system_prompt=config["system_prompt"],
            )
            agents.append(agent)

        super().__init__(agents, model=self.fallback_model)
        # Leader also talks to llama.cpp at :8080 with the base model.
        self.leader.backend = "llamacpp"
        self.leader.model = "hot-swap:base"

        self._available = {
            domain: f"hot-swap:{specialists_on_disk[domain].name}"
            for domain in specialists_on_disk
        }

    # ── Run override ───────────────────────────────────────────────

    def run(self, task: str, max_rounds: int = 5) -> str:
        """Route task through leader, swap model on each delegation."""
        if not self._hot_swap:
            return super().run(task, max_rounds=max_rounds)

        return self._run_hot_swap(task, max_rounds)

    def _run_hot_swap(self, task: str, max_rounds: int) -> str:
        """Hot-swap variant of Coordinator.run().

        Mirrors the base class logic but swaps the llama-server model
        before each delegate call and swaps back to base for the leader.
        """
        assert self._llama_mgr is not None and self._base_path is not None

        # Ensure base model is loaded for the leader's initial turn.
        self._swap_to(self._base_path)

        response = self.leader.chat(task, think=False)

        for _ in range(max_rounds):
            parsed = self._try_parse(response)

            if parsed and "final" in parsed:
                return parsed["final"]

            if parsed and "delegate" in parsed:
                agent_name = parsed["delegate"]
                sub_task = parsed["task"]

                if agent_name not in self.agents:
                    response = self.leader.chat(
                        f"Agent '{agent_name}' not found. "
                        f"Available: {list(self.agents.keys())}",
                        think=False,
                    )
                    continue

                # Swap to specialist if we have one, else stay on base.
                target = self._specialist_paths.get(agent_name, self._base_path)
                try:
                    self._swap_to(target)
                except ModelSwapError as e:
                    response = self.leader.chat(
                        f"Failed to load specialist for '{agent_name}': {e}. "
                        f"Try a different agent or handle the task yourself.",
                        think=False,
                    )
                    self._swap_to(self._base_path)
                    continue

                agent = self.agents[agent_name]
                result = agent.chat(sub_task)

                # Swap back for the leader's next turn.
                self._swap_to(self._base_path)
                response = self.leader.chat(
                    f"Agent '{agent_name}' responded:\n{result}",
                    think=False,
                )
            else:
                return response

        return response

    def _swap_to(self, model_path: Path) -> None:
        """Swap llama-server to the given model, recording elapsed time."""
        assert self._llama_mgr is not None
        elapsed = self._llama_mgr.swap(model_path)
        if elapsed > 0:
            self._swap_log.append((model_path.name, elapsed))

    # ── Status ─────────────────────────────────────────────────────

    @property
    def specialist_status(self) -> dict[str, str]:
        """Return status of each domain: model name or 'fallback'."""
        status = {}
        for domain, config in DOMAINS.items():
            if self._hot_swap:
                if domain in self._specialist_paths:
                    status[domain] = f"hot-swap: {self._specialist_paths[domain].name}"
                else:
                    status[domain] = f"hot-swap fallback (base)"
            else:
                if domain in self._available:
                    status[domain] = config["ollama_name"]
                else:
                    status[domain] = f"fallback ({self.fallback_model})"
        return status

    @property
    def mode(self) -> str:
        """Return 'hot-swap', 'ollama', or 'fallback' depending on active path."""
        if self._hot_swap:
            return "hot-swap"
        if self._available:
            return "ollama"
        return "fallback"

    @property
    def swap_log(self) -> list[tuple[str, float]]:
        """Recent swap events as (model_name, elapsed_seconds) pairs."""
        return list(self._swap_log)
