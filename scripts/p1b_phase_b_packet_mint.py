#!/usr/bin/env python3
"""PHASE_B_PACKET schema + O_EXCL mint helper (post-+1 commit; not Phase A production)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from scripts.p1b_o_excl_copy import write_bytes_o_excl

RUNTIME_EXECUTABLE_KEYS: tuple[str, ...] = (
    "scripts/p1b_phaseB_supervisor.py",
    "calm/hrm_text_158/native_full_stack/p1b_supervisor_lib.py",
    "scripts/p1b_phase_watchdog.py",
    "scripts/p1b_o_excl_copy.py",
    "scripts/train_hrm_text_158.py",
    "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py",
    "scripts/hrm_text_158_full_sub2_runtime_readiness.py",
    "calm/hrm_text_158/native_full_stack/full_sub2_runtime_readiness.py",
    "bin/watch-wrap",
)

PATH_KEYS: tuple[str, ...] = (
    "phase_b_log",
    "watchdog_event_log",
    "activation_receipt",
    "monitor_armed_touch",
    "p1b_receipt",
)

GATE_MSG_ID_KEYS: tuple[str, ...] = (
    "plus1_implement",
    "implement_gate1",
    "implement_gate2",
    "plus1_commit",
)

ENV_KEYS: tuple[str, ...] = (
    "CUDA_VISIBLE_DEVICES",
    "P1B_LIVE_CONVERSION_RECEIPT_JSON",
    "PYTHONPATH",
)

SCHEMA_REQUIRED_TOP: tuple[str, ...] = (
    "plan_id",
    "plan_revision",
    "plan_sha256",
    "gate_msg_ids",
    "commit_sha",
    "runtime_executable_sha256s",
    "watch_wrap_sha256",
    "watch_wrap_command_exact",
    "inner_science_argv",
    "env",
    "paths",
    "phase_budgets",
    "activation_deadlines",
    "kill_semantics",
    "packet_payload_digest",
)


def _canonical_json_bytes(obj: Any) -> bytes:
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode(
        "utf-8"
    )


def compute_packet_payload_digest(obj: Mapping[str, Any]) -> str:
    """sha256 of canonical JSON with packet_payload_digest omitted."""
    payload = dict(obj)
    payload.pop("packet_payload_digest", None)
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


def validate_phase_b_packet_schema(obj: Mapping[str, Any]) -> None:
    """Raise ValueError if packet schema is incomplete or mistyped."""
    missing = [k for k in SCHEMA_REQUIRED_TOP if k not in obj]
    if missing:
        raise ValueError(f"PHASE_B_PACKET schema incomplete; missing={missing}")

    gates = obj["gate_msg_ids"]
    if not isinstance(gates, Mapping):
        raise ValueError("gate_msg_ids must be an object")
    for key in GATE_MSG_ID_KEYS:
        if key not in gates or not str(gates[key]).strip():
            raise ValueError(f"gate_msg_ids.{key} required")
    # Explicitly reject legacy dual_accept names as substitutes.
    if "dual_accept_gate1" in gates or "dual_accept_gate2" in gates:
        raise ValueError("gate_msg_ids must use implement_gate1/implement_gate2 (not dual_accept_*)")

    hashes = obj["runtime_executable_sha256s"]
    if not isinstance(hashes, Mapping):
        raise ValueError("runtime_executable_sha256s must be an object")
    for key in RUNTIME_EXECUTABLE_KEYS:
        if key not in hashes or not str(hashes[key]).strip():
            raise ValueError(f"runtime_executable_sha256s.{key} required")

    env = obj["env"]
    if not isinstance(env, Mapping):
        raise ValueError("env must be an object")
    for key in ENV_KEYS:
        if key not in env:
            raise ValueError(f"env.{key} required")

    paths = obj["paths"]
    if not isinstance(paths, Mapping):
        raise ValueError("paths must be an object")
    for key in PATH_KEYS:
        if key not in paths or not str(paths[key]).strip():
            raise ValueError(f"paths.{key} required")

    for key in (
        "plan_id",
        "plan_revision",
        "plan_sha256",
        "commit_sha",
        "watch_wrap_sha256",
        "watch_wrap_command_exact",
        "inner_science_argv",
        "packet_payload_digest",
    ):
        if not str(obj[key]).strip():
            raise ValueError(f"{key} must be non-empty")

    if not isinstance(obj["phase_budgets"], Mapping):
        raise ValueError("phase_budgets must be an object")
    if not isinstance(obj["activation_deadlines"], Mapping):
        raise ValueError("activation_deadlines must be an object")
    if not isinstance(obj["kill_semantics"], Mapping):
        raise ValueError("kill_semantics must be an object")


def mint_phase_b_packet_o_excl(out_path: str | Path, fields: Mapping[str, Any]) -> dict[str, str]:
    """Validate, embed packet_payload_digest, O_EXCL-write packet; return digests.

    ``packet_file_sha256`` is EXTERNAL (returned, not necessarily inside file bytes).
    ``packet_payload_digest`` is written inside the file.
    """
    payload = dict(fields)
    # Compute digest with field omitted, then embed.
    digest = compute_packet_payload_digest(payload)
    payload["packet_payload_digest"] = digest
    validate_phase_b_packet_schema(payload)
    # Recompute after validate to ensure embedded value matches canonical form.
    digest = compute_packet_payload_digest(payload)
    payload["packet_payload_digest"] = digest
    encoded = _canonical_json_bytes(payload)
    file_sha = write_bytes_o_excl(out_path, encoded)
    return {
        "path": str(out_path),
        "packet_file_sha256": file_sha,
        "packet_payload_digest": digest,
    }


__all__ = [
    "GATE_MSG_ID_KEYS",
    "PATH_KEYS",
    "RUNTIME_EXECUTABLE_KEYS",
    "SCHEMA_REQUIRED_TOP",
    "compute_packet_payload_digest",
    "mint_phase_b_packet_o_excl",
    "validate_phase_b_packet_schema",
]
