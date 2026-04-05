"""Audit log for agent events — tool calls, responses, commands."""

import time
from dataclasses import dataclass, field


@dataclass(frozen=True)
class HistoryEvent:
    timestamp: float
    event_type: str
    detail: str


@dataclass
class HistoryLog:
    """Append-only event log for a harness session."""

    events: list[HistoryEvent] = field(default_factory=list)

    def add(self, event_type: str, detail: str) -> None:
        self.events.append(HistoryEvent(
            timestamp=time.time(),
            event_type=event_type,
            detail=detail,
        ))

    def as_markdown(self) -> str:
        if not self.events:
            return "No events recorded."
        lines = ["# Session History", ""]
        for event in self.events:
            ts = time.strftime("%H:%M:%S", time.localtime(event.timestamp))
            lines.append(f"- [{ts}] {event.event_type}: {event.detail}")
        return "\n".join(lines)
