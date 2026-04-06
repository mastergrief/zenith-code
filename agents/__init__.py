"""Multi-agent system powered by local Qwen 3.5 models via Ollama."""

from agents.agent import Agent
from agents.coordinator import Coordinator
from agents.swarm import Swarm
from agents.specialist_coordinator import SpecialistCoordinator
from agents.history import HistoryLog
from agents.permissions import BashRisk, PermissionMode, classify_bash, check_permission
from agents.compact import compact_history, should_compact

__all__ = [
    "Agent",
    "Coordinator",
    "Swarm",
    "SpecialistCoordinator",
    "HistoryLog",
    "BashRisk",
    "PermissionMode",
    "classify_bash",
    "check_permission",
    "compact_history",
    "should_compact",
]
