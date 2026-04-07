"""Configuration loading — .clawrc / claw.json / env vars."""

import json
import os
from pathlib import Path

DEFAULTS = {
    "model": "qwen3.5:4b",
    "backend": None,
    # 262144 matches bin/claw's CLAW_CTX default — both Gemma 4 E4B and
    # Qwen 3.5 4B are trained at 256K and fit 8 GB VRAM with Q4 KV cache.
    # Harness computes the actual compaction threshold from this ctx_size
    # capped by agents.compact.MODEL_CONTEXT_LIMITS per loaded GGUF.
    "ctx_size": 262144,
    "auto_compact_tokens": None,
    "permission_mode": "workspace",
    "effort": "medium",
}

# Explicit env var names — aligned with existing conventions in bin/claw
# (CLAW_CTX) and compact.py (CLAW_AUTO_COMPACT_TOKENS). Don't auto-build
# from key names; the historical conventions don't match an auto scheme.
ENV_VARS = {
    "model": "CLAW_MODEL",
    "backend": "CLAW_BACKEND",
    "ctx_size": "CLAW_CTX",
    "auto_compact_tokens": "CLAW_AUTO_COMPACT_TOKENS",
    "permission_mode": "CLAW_PERMISSION_MODE",
    "effort": "CLAW_EFFORT",
}

INT_KEYS = {"ctx_size", "auto_compact_tokens"}


def load_config() -> dict:
    """Load config. Precedence (high → low): env vars → file → defaults."""
    config = dict(DEFAULTS)
    for name in (".clawrc", "claw.json"):
        p = Path(name)
        if p.exists():
            try:
                with open(p) as f:
                    file_data = json.load(f)
                # Only accept known keys to avoid silent typos
                for key in DEFAULTS:
                    if key in file_data:
                        config[key] = file_data[key]
            except (json.JSONDecodeError, OSError):
                pass
            break
    for key, env_key in ENV_VARS.items():
        val = os.environ.get(env_key)
        if val:
            if key in INT_KEYS:
                try:
                    config[key] = int(val)
                except ValueError:
                    pass
            else:
                config[key] = val
    return config
