"""Runtime scratch / O_EXCL IO helpers for LANDS-AB (IMPLEMENT_v16).

Never write raw observations under repo artifacts/acc_entropy.
"""
from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path
from typing import Any, Mapping


def o_excl_write_json(path: Path, payload: Mapping[str, Any]) -> str:
    """O_EXCL write — no pre-delete; unique runtime-scratch path only."""
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)
    return hashlib.sha256(data).hexdigest()


def o_excl_write_text(path: Path, text: str) -> str:
    """O_EXCL write for text/JSON dumps — no pre-delete; fail if path exists."""
    p = Path(path)
    if "artifacts" in p.parts and "acc_entropy" in p.parts and p.name.startswith("lands_ab_raw_obs_"):
        raise ValueError("raw_obs_must_not_write_to_repo_artifacts")
    data = text if text.endswith("\n") else (text + "\n")
    raw = data.encode("utf-8")
    p.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(p), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, raw)
    finally:
        os.close(fd)
    return hashlib.sha256(raw).hexdigest()


def runtime_scratch_raw_path(
    *,
    scratch_dir: Path,
    gating_row: str,
    run_nonce: str,
) -> Path:
    """Unique runtime-scratch path — never under artifacts/."""
    scratch = Path(scratch_dir)
    if "artifacts" in scratch.parts and "acc_entropy" in scratch.parts:
        raise ValueError("raw_obs_must_not_write_to_repo_artifacts")
    scratch.mkdir(parents=True, exist_ok=True)
    return scratch / f"lands_ab_raw_obs_{gating_row}_{run_nonce}.json"


def resolve_run_scratch_dir(*, create: bool = True) -> Path:
    """Prefer LANDS_AB_RUN_ROOT (formal nonce run root); else unique under LANDS_AB_RUNTIME_SCRATCH."""
    env_root = os.environ.get("LANDS_AB_RUN_ROOT")
    if env_root:
        p = Path(env_root)
        if "artifacts" in p.parts and "acc_entropy" in p.parts:
            raise ValueError("run_root_must_not_be_repo_artifacts")
        if create:
            p.mkdir(parents=True, exist_ok=True)
        return p
    base = Path(os.environ.get("LANDS_AB_RUNTIME_SCRATCH", "/tmp/lands_ab_runtime_scratch"))
    p = base / uuid.uuid4().hex
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def harvest_exactly_one_raw_obs(*, run_root: Path, gating_row: str) -> Path:
    """Exactly one lands_ab_raw_obs_<row>_*.json under run_root; STOP on zero or multiple."""
    root = Path(run_root)
    if "artifacts" in root.parts and "acc_entropy" in root.parts:
        raise ValueError("run_root_must_not_be_repo_artifacts")
    matches = sorted(root.glob(f"lands_ab_raw_obs_{gating_row}_*.json"))
    if len(matches) == 0:
        raise ValueError(f"raw_obs_harvest_zero:{gating_row}")
    if len(matches) > 1:
        raise ValueError(f"raw_obs_harvest_multiple:{gating_row}:{len(matches)}")
    return matches[0]
