"""Configuration loading — .zenithrc / zenith.json / env vars."""

import json
import os
from pathlib import Path

DEFAULTS = {
    "model": "~/models/gemma-4-E4B-it-tq4-aligned.gguf",
    "backend": "llamacpp",
    # 524288 (512K) — fits 8 GB VRAM with tq4+tq4 stack (7.6 GB at 512K).
    # For Q5_K_M + f16 KV (6.9 GB at 256K), override: ZENITH_CTX=262144.
    # Harness computes the actual compaction threshold from this ctx_size
    # capped by agents.compact.MODEL_CONTEXT_LIMITS per loaded GGUF.
    "ctx_size": 524288,
    "auto_compact_tokens": None,
    "permission_mode": "workspace",
    "effort": "medium",
}

# Explicit env var names — aligned with existing conventions in bin/zenith
# (ZENITH_CTX) and compact.py (ZENITH_AUTO_COMPACT_TOKENS). Don't auto-build
# from key names; the historical conventions don't match an auto scheme.
ENV_VARS = {
    "model": "ZENITH_MODEL",
    "backend": "ZENITH_BACKEND",
    "ctx_size": "ZENITH_CTX",
    "auto_compact_tokens": "ZENITH_AUTO_COMPACT_TOKENS",
    "permission_mode": "ZENITH_PERMISSION_MODE",
    "effort": "ZENITH_EFFORT",
}

INT_KEYS = {"ctx_size", "auto_compact_tokens"}


def load_config() -> dict:
    """Load config. Precedence (high → low): env vars → file → defaults."""
    config = dict(DEFAULTS)
    for name in (".zenithrc", "zenith.json"):
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
