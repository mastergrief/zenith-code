"""Session persistence — save and load agent conversation history."""

import json
from datetime import datetime, timezone
from pathlib import Path

SESSION_DIR = Path(".zenith_sessions")


def save_session(agent) -> Path:
    """Save an agent's conversation history to disk."""
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    session_id = f"{agent.name}_{ts}"
    path = SESSION_DIR / f"{session_id}.json"
    data = {
        "session_id": session_id,
        "name": agent.name,
        "model": agent.model,
        "role": agent.role,
        "history": agent.history,
        "saved_at": datetime.now(timezone.utc).isoformat(),
    }
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
    return path


def load_session(session_id: str) -> dict:
    """Load a session by ID (filename stem)."""
    path = SESSION_DIR / f"{session_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def list_sessions() -> list[dict]:
    """List available sessions, sorted by time (newest first)."""
    if not SESSION_DIR.exists():
        return []
    sessions = []
    for path in sorted(SESSION_DIR.glob("*.json"), reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions.append({
                "id": path.stem,
                "name": data.get("name", "?"),
                "model": data.get("model", "?"),
                "saved_at": data.get("saved_at", "?"),
                "messages": len(data.get("history", [])),
            })
        except (json.JSONDecodeError, OSError):
            continue
    return sessions
