#!/usr/bin/env python3
"""A′ slice1 fidelity wrapper — thin outer supervisor.

SOLE owner of run order: statuses → reduce → finalize (PACKET_TERMINAL).
Authoritative branch = terminal_receipt.json.branch only.
Finalize orchestration lives in a_prime_slice1_fidelity_manifest.
"""
from __future__ import annotations

import argparse
import json
import os
import secrets
import subprocess
import sys
import time
from pathlib import Path

from scripts.a_prime_slice1_fidelity_manifest import (
    finalize,
    make_fixture_receipt,
)

__all__ = ["finalize", "main", "make_fixture_receipt"]

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
REDUCER = REPO / "scripts/a_prime_slice1_retained_credit_fidelity_reducer_v0.py"
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"
PARENT = (
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_"
    "lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"


def write_status(run_root: Path, name: str, payload: dict) -> None:
    d = run_root / "command_status"
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{name}.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
    )


def run_cmd(
    run_root: Path, name: str, argv: list[str], env: dict, timeout: float | None = None
) -> int:
    t0 = time.time()
    try:
        proc = subprocess.run(argv, cwd=str(REPO), env=env, timeout=timeout)
        rc = int(proc.returncode)
        write_status(
            run_root,
            name,
            {
                "name": name,
                "argv": argv,
                "rc": rc,
                "wall_seconds": time.time() - t0,
                "timeout": False,
                "oom": False,
            },
        )
        return rc
    except subprocess.TimeoutExpired:
        write_status(
            run_root,
            name,
            {
                "name": name,
                "argv": argv,
                "rc": 124,
                "wall_seconds": time.time() - t0,
                "timeout": True,
                "oom": False,
            },
        )
        return 124


def probe_argv(steps: int, scratch: Path, nondense: bool) -> list[str]:
    argv = [
        sys.executable,
        "-u",
        "-B",
        str(PROBE),
        "--enable-bounded-delta-probe",
        "--allow-gpu-launch",
        "--device",
        "cuda:0",
        "--parent",
        PARENT,
        "--parent-sha256",
        PARENT_SHA,
        "--curriculum-seed",
        "17",
        "--max-steps-hard",
        "50",
        "--steps",
        str(steps),
        "--scratch-root",
        str(scratch),
        "--prior-audit-supports",
        "L0b,math_a0",
        "--emit-progress",
        "--phase-heartbeat-seconds",
        "30",
        "--phase-timeout-seconds",
        "120",
        "--total-timeout-seconds",
        "3600",
        "--max-silent-phase-seconds",
        "600",
    ]
    if nondense:
        argv += [
            "--persistent-accumulator-event-coded-live",
            "--event-coded-sparse-vote-authority",
        ]
    return argv


def require_env(name: str) -> str:
    v = os.environ.get(name, "").strip()
    if not v:
        raise SystemExit(f"missing required env {name} (set by frozen packet shell line)")
    return v


def _write_synthetic_arm_statuses(run_root: Path, branch: str) -> None:
    for arm in [
        "dense_screen",
        "nondense_screen",
        "dense_verdict",
        "nondense_verdict",
    ]:
        rc = 1 if (branch == "LIVENESS_FAIL" and arm == "dense_screen") else 0
        write_status(
            run_root,
            arm,
            {
                "name": arm,
                "argv": ["synthetic"],
                "rc": rc,
                "timeout": False,
                "oom": False,
                "wall_seconds": 0.01,
                "synthetic": True,
            },
        )


def _write_synthetic_receipts(run_root: Path, branch: str) -> tuple[Path, Path]:
    dscratch = run_root / "arm_dense_verdict" / "c2p1_impl_cpu"
    nscratch = run_root / "arm_nondense_verdict" / "c2p1_impl_cpu"
    if branch == "FIDELITY_COLLAPSE":
        make_fixture_receipt(
            dscratch / "receipt.json",
            {"L0b": "230/230", "math_a0": "1255/1255"},
        )
        make_fixture_receipt(
            nscratch / "receipt.json",
            {"L0b": "100/230", "math_a0": "800/1255"},
        )
    elif branch == "INSTRUMENT_GAP":
        pass
    else:
        make_fixture_receipt(
            dscratch / "receipt.json",
            {"L0b": "230/230", "math_a0": "1255/1255"},
        )
        make_fixture_receipt(
            nscratch / "receipt.json",
            {"L0b": "220/230", "math_a0": "1200/1255"},
        )
    return dscratch, nscratch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=None)
    ap.add_argument("--dry-preflight-only", action="store_true")
    ap.add_argument(
        "--dry-synthetic-final",
        choices=[
            "PAIRED_ACHIEVED_FIDELITY_AT_N",
            "FIDELITY_COLLAPSE",
            "LIVENESS_FAIL",
            "INSTRUMENT_GAP",
        ],
        default="",
    )
    ap.add_argument("--inject-candidate-branch", default="")
    ap.add_argument("--inject-postpub-fail", action="store_true")
    args = ap.parse_args()

    expect_probe = require_env("A_PRIME_EXPECT_PROBE_SHA")
    expect_reducer = require_env("A_PRIME_EXPECT_REDUCER_SHA")
    expect_wrapper = require_env("A_PRIME_EXPECT_WRAPPER_SHA")
    expect_rollup = require_env("A_PRIME_EXPECT_ROLLUP_SHA")
    expect_rollup_n = require_env("A_PRIME_EXPECT_ROLLUP_N")
    expect_head = require_env("A_PRIME_EXPECT_HEAD")
    expect_dirty_sha = require_env("A_PRIME_EXPECT_DIRTY_SHA")
    expect_dirty_n = require_env("A_PRIME_EXPECT_DIRTY_N")

    nonce = secrets.token_hex(4)
    if args.run_root is None:
        run_root = Path(f"/tmp/a_prime_slice1_retained_credit_fidelity_v0_run_{nonce}")
    else:
        run_root = args.run_root

    run_root_abs = str(run_root.resolve()) if run_root.exists() else str(run_root)
    print(f"PHASE_MARKER RUN_ROOT={run_root_abs}", flush=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"

    preflight_argv = [
        sys.executable,
        "-u",
        "-B",
        str(REDUCER),
        "--mode",
        "preflight",
        "--run-root",
        str(run_root),
        "--repo",
        str(REPO),
        "--nonce",
        nonce,
        "--expect-head",
        expect_head,
        "--expect-dirty-sha",
        expect_dirty_sha,
        "--expect-dirty-n",
        str(expect_dirty_n),
        "--expect-probe-sha",
        expect_probe,
        "--expect-reducer-sha",
        expect_reducer,
        "--expect-wrapper-sha",
        expect_wrapper,
        "--expect-rollup-sha",
        expect_rollup,
        "--expect-rollup-n",
        str(expect_rollup_n),
    ]
    if args.dry_synthetic_final:
        preflight_argv.append("--synthetic")

    fin_kw = dict(
        synthetic=bool(args.dry_synthetic_final),
        inject_candidate_branch=args.inject_candidate_branch or None,
        inject_postpub_fail=bool(args.inject_postpub_fail),
    )

    rc = run_cmd(run_root, "preflight", preflight_argv, env)
    if rc != 0:
        return finalize(run_root, reduce_rc=rc, **fin_kw)
    if args.dry_preflight_only:
        return 0

    if args.dry_synthetic_final:
        _write_synthetic_arm_statuses(run_root, args.dry_synthetic_final)
        dscratch, nscratch = _write_synthetic_receipts(
            run_root, args.dry_synthetic_final
        )
        reduce_rc = run_cmd(
            run_root,
            "reduce",
            [
                sys.executable,
                "-u",
                "-B",
                str(REDUCER),
                "--mode",
                "reduce",
                "--run-root",
                str(run_root),
                "--dense-scratch-root",
                str(dscratch),
                "--nondense-scratch-root",
                str(nscratch),
                "--synthetic",
            ],
            env,
        )
        return finalize(run_root, reduce_rc=reduce_rc, **fin_kw)

    arms = [
        ("dense_screen", 20, run_root / "arm_dense_screen" / "c2p1_impl_cpu", False),
        ("nondense_screen", 20, run_root / "arm_nondense_screen" / "c2p1_impl_cpu", True),
        ("dense_verdict", 50, run_root / "arm_dense_verdict" / "c2p1_impl_cpu", False),
        ("nondense_verdict", 50, run_root / "arm_nondense_verdict" / "c2p1_impl_cpu", True),
    ]
    for name, steps, scratch, nd in arms:
        scratch.mkdir(parents=True, exist_ok=True)
        rc = run_cmd(run_root, name, probe_argv(steps, scratch, nd), env)
        if rc != 0:
            break
    reduce_rc = run_cmd(
        run_root,
        "reduce",
        [
            sys.executable,
            "-u",
            "-B",
            str(REDUCER),
            "--mode",
            "reduce",
            "--run-root",
            str(run_root),
            "--dense-scratch-root",
            str(run_root / "arm_dense_verdict" / "c2p1_impl_cpu"),
            "--nondense-scratch-root",
            str(run_root / "arm_nondense_verdict" / "c2p1_impl_cpu"),
        ],
        env,
    )
    return finalize(run_root, reduce_rc=reduce_rc, **fin_kw)


if __name__ == "__main__":
    sys.exit(main())
