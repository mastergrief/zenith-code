"""Pin collect + rollup for A′ slice1 fidelity instruments. No terminal print.

Head/dirty expectations are packet-provided (never hardcoded self-staling pins).
PARENT_SHA / PROBE_PIN remain frozen science constants.
"""
from __future__ import annotations

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from scripts.a_prime_slice1_fidelity_core import sha256_file

DEFAULT_REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
PARENT_REL = (
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_"
    "lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
PROBE_PIN = "d39c3ead23f56edc36ec16409f38fed2129179723482b401ebac2a5aa6757701"


def dirty_cmd(repo: Path) -> list[str]:
    return [
        "git",
        "-C",
        str(repo),
        "status",
        "--porcelain=v1",
        "--",
        ".",
        ":(exclude)artifacts/a_prime",
        ":(exclude)scripts/a_prime_slice1_retained_credit_fidelity_reducer_v0.py",
        ":(exclude)scripts/a_prime_slice1_retained_credit_fidelity_wrapper_v0.py",
        ":(exclude)scripts/a_prime_slice1_fidelity_core.py",
        ":(exclude)scripts/a_prime_slice1_fidelity_pins.py",
        ":(exclude)scripts/a_prime_slice1_fidelity_manifest.py",
        ":(exclude)tests/test_a_prime_slice1_fidelity_core.py",
        ":(exclude)tests/test_a_prime_slice1_fidelity_wrapper_exit.py",
    ]


def collect_git_head(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def collect_dirty(repo: Path) -> tuple[str, int]:
    dirty = subprocess.check_output(dirty_cmd(repo))
    return hashlib.sha256(dirty).hexdigest(), len(dirty.splitlines())


def compute_rollup(repo: Path) -> tuple[str, int]:
    """Read-only rollup: import probe+reducer+wrapper; hash calm/+scripts/ modules."""
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    spec = importlib.util.spec_from_file_location(
        "hrm_text_158_bounded_delta_acquisition_probe",
        repo / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py",
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    import scripts.a_prime_slice1_retained_credit_fidelity_reducer_v0  # noqa: F401
    import scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0  # noqa: F401

    repo = repo.resolve()
    paths_set: set[str] = set()
    for m in sys.modules.values():
        f = getattr(m, "__file__", None)
        if not f:
            continue
        try:
            rp = Path(f).resolve()
            rel = str(rp.relative_to(repo))
        except (ValueError, OSError):
            continue
        if not rp.is_file():
            continue
        if rel.startswith("calm/") or rel.startswith("scripts/"):
            paths_set.add(rel)
    paths = sorted(paths_set)
    h = hashlib.sha256()
    for rel in paths:
        d = hashlib.sha256((repo / rel).read_bytes()).hexdigest()
        h.update(d.encode())
        h.update(b"\n")
    return h.hexdigest(), len(paths)


def derive_preflight_status(*, synthetic: bool, pin_errors: list[str]) -> str:
    """status derived, never constant.
    REAL only if not synthetic AND pin_errors empty.
    SYNTHETIC if synthetic run (regardless of pin_errors — instrument pins still listed).
    PIN_FAIL if non-synthetic with pin_errors.
    """
    if synthetic:
        return "SYNTHETIC"
    if pin_errors:
        return "PIN_FAIL"
    return "REAL"


def run_preflight_checks(
    repo: Path,
    *,
    expect_head: str,
    expect_dirty_sha: str,
    expect_dirty_n: int,
    expect_probe_sha: str,
    expect_reducer_sha: str,
    expect_wrapper_sha: str,
    expect_rollup_sha: str,
    expect_rollup_n: int,
    synthetic: bool = False,
) -> tuple[dict[str, Any], list[str]]:
    """Return (payload, pin_errors). Does not write or print terminal markers.

    Head/dirty expectations are required packet inputs (activation contract).
    synthetic=True: still checks instrument pins; status becomes SYNTHETIC;
    head/dirty/parent env pins still checked when expects provided (science parity
    optional for dry: tests may pass current live head/dirty as expects).
    """
    pin_errors: list[str] = []
    head = collect_git_head(repo)
    dirty_sha, dirty_n = collect_dirty(repo)
    parent = repo / PARENT_REL
    parent_sha = sha256_file(parent) if parent.is_file() else None
    probe = repo / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
    reducer = repo / "scripts/a_prime_slice1_retained_credit_fidelity_reducer_v0.py"
    wrapper = repo / "scripts/a_prime_slice1_retained_credit_fidelity_wrapper_v0.py"
    probe_sha = sha256_file(probe) if probe.is_file() else None
    reducer_sha = sha256_file(reducer) if reducer.is_file() else None
    wrapper_sha = sha256_file(wrapper) if wrapper.is_file() else None

    # head/dirty always compared to packet-provided expects (no hardcoded pins)
    if head != expect_head:
        pin_errors.append(f"head {head}!={expect_head}")
    if dirty_sha != expect_dirty_sha or dirty_n != int(expect_dirty_n):
        pin_errors.append(
            f"dirty {dirty_n}/{dirty_sha}!={expect_dirty_n}/{expect_dirty_sha}"
        )
    if parent_sha != PARENT_SHA:
        pin_errors.append(f"parent {parent_sha}!={PARENT_SHA}")
    if probe_sha != expect_probe_sha:
        pin_errors.append(f"probe {probe_sha}!={expect_probe_sha}")
    if reducer_sha != expect_reducer_sha:
        pin_errors.append(f"reducer {reducer_sha}!={expect_reducer_sha}")
    if wrapper_sha != expect_wrapper_sha:
        pin_errors.append(f"wrapper {wrapper_sha}!={expect_wrapper_sha}")

    try:
        rollup, n_files = compute_rollup(repo)
    except Exception as e:
        pin_errors.append(f"rollup_error:{e}")
        rollup, n_files = None, None
    if rollup != expect_rollup_sha or n_files != int(expect_rollup_n):
        pin_errors.append(
            f"rollup {n_files}/{rollup}!={expect_rollup_n}/{expect_rollup_sha}"
        )

    status = derive_preflight_status(synthetic=synthetic, pin_errors=pin_errors)
    payload = {
        "schema": "a_prime_slice1_launch_preflight/v3",
        "status": status,
        "synthetic": synthetic,
        "head": head,
        "head_expect": expect_head,
        "head_match": head == expect_head,
        "dirty_lines": dirty_n,
        "dirty_sha256": dirty_sha,
        "dirty_expect_sha": expect_dirty_sha,
        "dirty_expect_n": int(expect_dirty_n),
        "dirty_match": dirty_sha == expect_dirty_sha and dirty_n == int(expect_dirty_n),
        "parent_path": PARENT_REL,
        "parent_sha256": parent_sha,
        "parent_match": parent_sha == PARENT_SHA,
        "probe_sha256": probe_sha,
        "reducer_sha256": reducer_sha,
        "wrapper_sha256": wrapper_sha,
        "probe_pin_match": probe_sha == expect_probe_sha,
        "reducer_pin_match": reducer_sha == expect_reducer_sha,
        "wrapper_pin_match": wrapper_sha == expect_wrapper_sha,
        "rollup_sha256": rollup,
        "rollup_n_files": n_files,
        "rollup_match": rollup == expect_rollup_sha and n_files == int(expect_rollup_n),
        "pin_errors": pin_errors,
    }
    return payload, pin_errors
