"""SpecialistCoordinator: routes tasks to domain-specific fine-tuned models."""

import json
import urllib.request

from agents.agent import Agent
from agents.coordinator import Coordinator
from agents.distill.config import DOMAINS, OLLAMA_URL


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
    """Coordinator that uses distilled specialist models for each agent.

    Each agent uses its own fine-tuned 0.6B model. Falls back to the
    default model if a specialist isn't available.
    """

    def __init__(self, fallback_model: str = "qwen3.5:4b"):
        self.fallback_model = fallback_model
        available = detect_specialists()

        # Create agents — use specialist model if available, else fallback
        agents = []
        for domain, config in DOMAINS.items():
            if domain == "orchestrator":
                continue  # Orchestrator is the leader, not a worker

            model = available.get(domain, fallback_model)
            agent = Agent(
                name=domain,
                role=config["description"],
                model=model,
                system_prompt=config["system_prompt"],
            )
            agents.append(agent)

        # Initialize parent Coordinator
        super().__init__(agents, model=fallback_model)

        # Override leader model if orchestrator specialist is available
        if "orchestrator" in available:
            self.leader.model = available["orchestrator"]

        self._available = available

    @property
    def specialist_status(self) -> dict[str, str]:
        """Return status of each specialist: model name or 'fallback'."""
        status = {}
        for domain, config in DOMAINS.items():
            if domain in self._available:
                status[domain] = config["ollama_name"]
            else:
                status[domain] = f"fallback ({self.fallback_model})"
        return status
