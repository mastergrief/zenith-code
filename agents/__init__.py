"""Multi-agent system powered by local Qwen 3.5 models via Ollama."""

from agents.agent import Agent
from agents.coordinator import Coordinator
from agents.swarm import Swarm
from agents.specialist_coordinator import SpecialistCoordinator
from agents.history import HistoryLog
from agents.permissions import ToolPermissions
from agents.compact import compact_history, should_compact

__all__ = [
    "Agent",
    "Coordinator",
    "Swarm",
    "SpecialistCoordinator",
    "HistoryLog",
    "ToolPermissions",
    "compact_history",
    "should_compact",
]
