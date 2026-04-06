"""Configuration loading — .clawrc / claw.json / env vars."""

import json
import os
from pathlib import Path

DEFAULTS = {
    "model": "qwen3.5:4b",
    "backend": None,
    "context_size": 65536,
    "auto_compact_threshold": None,
    "permission_mode": "workspace",
}


def load_config() -> dict:
    """Load config from file and env vars. Env vars override file values."""
    config = dict(DEFAULTS)
    for name in (".clawrc", "claw.json"):
        p = Path(name)
        if p.exists():
            with open(p) as f:
                config.update(json.load(f))
            break
    for key in DEFAULTS:
        env_key = f"CLAW_{key.upper()}"
        val = os.environ.get(env_key)
        if val:
            config[key] = int(val) if val.isdigit() else val
    return config
