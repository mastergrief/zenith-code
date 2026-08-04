#!/usr/bin/env python3
"""A′ slice1 retained-credit fidelity reducer — thin CLI modes only.

Authoritative branch ONLY in terminal_receipt.json.branch.
PHASE_MARKER only — NEVER PACKET_TERMINAL.
NEVER writes terminal_manifest.json (wrapper owns finalization).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice1_fidelity_core import (
    DEFAULT_PINNED_SUPPORTS,
    FINAL_BRANCHES,
    any_liveness_fail,
    classify_branch,
    extract_non_q_bpw,
    extract_prior_rates,
    sha256_file,
)
from scripts.a_prime_slice1_fidelity_manifest import load_json, read_command_statuses
from scripts.a_prime_slice1_fidelity_pins import PROBE_PIN, run_preflight_checks

assert FINAL_BRANCHES


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n")


def print_phase(marker: str) -> None:
    print(f"PHASE_MARKER {marker}", flush=True)


def write_terminal_receipt(
    run_root: Path,
    *,
    branch: str,
    extra: Mapping[str, Any] | None = None,
    synthetic: bool = False,
    run_root_abs: str | None = None,
) -> None:
    assert branch in FINAL_BRANCHES or branch == "INSTRUMENT_GAP"
    payload = {
        "schema": "a_prime_slice1_terminal_receipt/v3",
        "branch": branch,
        "run_root": run_root_abs or str(run_root.resolve()),
        "override_used": False,
        "budget_unresolved": True,
        "diagnostic_class": "pre_full_stack_diagnostic",
        "synthetic": synthetic,
    }
    if extra:
        payload.update(dict(extra))
    write_json(run_root / "terminal_receipt.json", payload)


def mode_preflight(args: argparse.Namespace) -> int:
    repo = Path(args.repo)
    run_root = Path(args.run_root)
    synthetic = bool(args.synthetic)
    run_root_abs = str(run_root.resolve()) if run_root.exists() else str(run_root)

    if run_root.exists():
        existing = [p for p in run_root.rglob("*") if p.is_file()]
        if existing:
            run_root.mkdir(parents=True, exist_ok=True)
            write_terminal_receipt(
                run_root,
                branch="INSTRUMENT_GAP",
                extra={
                    "reason": "stale_run_root_not_empty",
                    "details": {
                        "files": [str(p.relative_to(run_root)) for p in existing[:50]]
                    },
                },
                synthetic=synthetic,
                run_root_abs=str(run_root.resolve()),
            )
            return 2
    run_root.mkdir(parents=True, exist_ok=True)
    run_root_abs = str(run_root.resolve())
    for name in (
        "arm_dense_screen",
        "arm_nondense_screen",
        "arm_dense_verdict",
        "arm_nondense_verdict",
    ):
        (run_root / name / "c2p1_impl_cpu").mkdir(parents=True, exist_ok=True)

    payload, pin_errors = run_preflight_checks(
        repo,
        expect_head=args.expect_head,
        expect_dirty_sha=args.expect_dirty_sha,
        expect_dirty_n=int(args.expect_dirty_n),
        expect_probe_sha=args.expect_probe_sha,
        expect_reducer_sha=args.expect_reducer_sha,
        expect_wrapper_sha=args.expect_wrapper_sha,
        expect_rollup_sha=args.expect_rollup_sha,
        expect_rollup_n=int(args.expect_rollup_n),
        synthetic=synthetic,
    )
    payload["run_root"] = run_root_abs
    payload["nonce"] = args.nonce
    payload["scratch_roots"] = {
        "dense_screen": str(run_root / "arm_dense_screen" / "c2p1_impl_cpu"),
        "nondense_screen": str(run_root / "arm_nondense_screen" / "c2p1_impl_cpu"),
        "dense_verdict": str(run_root / "arm_dense_verdict" / "c2p1_impl_cpu"),
        "nondense_verdict": str(run_root / "arm_nondense_verdict" / "c2p1_impl_cpu"),
    }
    write_json(run_root / "launch_preflight.json", payload)
    status = payload["status"]
    ok = status in ("REAL", "SYNTHETIC") and not pin_errors
    print(
        json.dumps(
            {
                "mode": "preflight",
                "ok": ok,
                "status": status,
                "head_match": payload["head_match"],
                "dirty_match": payload["dirty_match"],
                "parent_match": payload["parent_match"],
                "rollup_match": payload["rollup_match"],
                "pin_errors": pin_errors,
            },
            sort_keys=True,
        ),
        flush=True,
    )
    if not ok:
        write_terminal_receipt(
            run_root,
            branch="INSTRUMENT_GAP",
            extra={"reason": "preflight_failure", "details": {"pin_errors": pin_errors, "status": status}},
            synthetic=synthetic,
            run_root_abs=run_root_abs,
        )
        return 2
    print_phase("PREFLIGHT_OK")
    return 0


def mode_schema(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    write_json(
        run_root / "artifact_schema.json",
        {
            "schema": "a_prime_slice1_artifact_schema/v3",
            "required_outputs": [
                "launch_preflight.json",
                "command_status/*.json",
                "arm_DENSE_INT16_BASELINE/metrics.json",
                "arm_NONDENSE_CANDIDATE/metrics.json",
                "paired_probe_report.json",
                "non_q_bpw_receipt.json",
                "branch_verdict.json",
                "terminal_receipt.json",
                "terminal_manifest.json",
            ],
        },
    )
    print(json.dumps({"mode": "schema", "exit": 0}, sort_keys=True), flush=True)
    print_phase("SCHEMA_OK")
    return 0


def _preflight_admissible(preflight: Mapping[str, Any], *, synthetic: bool) -> tuple[bool, str]:
    """Consumer checks failure fields, not just the status label.

    Synthetic and real paths share the same strictness: pin_errors must be empty
    and all 7 match flags true. Only the required status label differs
    (SYNTHETIC vs REAL).
    """
    status = preflight.get("status")
    pin_errors = list(preflight.get("pin_errors") or [])
    pf_synth = bool(preflight.get("synthetic"))
    if pf_synth != synthetic:
        return False, f"synthetic_parity_mismatch:preflight={pf_synth}:args={synthetic}"
    expected_status = "SYNTHETIC" if synthetic else "REAL"
    if status != expected_status:
        return False, f"expected_{expected_status}_got_{status}"
    if pin_errors:
        return False, f"pin_errors_nonempty:{pin_errors}"
    for key in (
        "head_match",
        "dirty_match",
        "parent_match",
        "probe_pin_match",
        "reducer_pin_match",
        "wrapper_pin_match",
        "rollup_match",
    ):
        if preflight.get(key) is not True:
            return False, f"match_fail:{key}"
    return True, "ok"


def mode_reduce(args: argparse.Namespace) -> int:
    run_root = Path(args.run_root)
    synthetic = bool(args.synthetic)
    run_root_abs = str(run_root.resolve())
    preflight_path = run_root / "launch_preflight.json"

    if not preflight_path.is_file():
        write_terminal_receipt(
            run_root,
            branch="INSTRUMENT_GAP",
            extra={"reason": "missing_launch_preflight"},
            synthetic=synthetic,
            run_root_abs=run_root_abs,
        )
        return 2
    preflight = load_json(preflight_path)
    ok, reason = _preflight_admissible(preflight, synthetic=synthetic)
    if not ok:
        write_terminal_receipt(
            run_root,
            branch="INSTRUMENT_GAP",
            extra={"reason": f"launch_preflight_inadmissible:{reason}", "details": preflight},
            synthetic=synthetic,
            run_root_abs=run_root_abs,
        )
        return 2

    statuses = read_command_statuses(run_root)
    live_fail = any_liveness_fail(statuses)
    if live_fail is not None:
        write_terminal_receipt(
            run_root,
            branch="LIVENESS_FAIL",
            extra={
                "failing_command_status": {
                    "path": live_fail.get("_status_path"),
                    "sha256": live_fail.get("_status_sha256"),
                    "rc": live_fail.get("rc"),
                    "name": live_fail.get("name"),
                }
            },
            synthetic=synthetic,
            run_root_abs=run_root_abs,
        )
        return 3

    dense_receipt_path = Path(args.dense_scratch_root) / "receipt.json"
    nondense_receipt_path = Path(args.nondense_scratch_root) / "receipt.json"
    if not dense_receipt_path.is_file() or not nondense_receipt_path.is_file():
        write_terminal_receipt(
            run_root,
            branch="INSTRUMENT_GAP",
            extra={
                "reason": "missing_verdict_receipts_with_clean_status",
                "details": {
                    "dense": str(dense_receipt_path),
                    "nondense": str(nondense_receipt_path),
                },
            },
            synthetic=synthetic,
            run_root_abs=run_root_abs,
        )
        return 2

    dense_receipt = load_json(dense_receipt_path)
    nondense_receipt = load_json(nondense_receipt_path)
    dense_prior = extract_prior_rates(
        dense_receipt, pinned_supports=DEFAULT_PINNED_SUPPORTS
    )
    nondense_prior = extract_prior_rates(
        nondense_receipt, pinned_supports=DEFAULT_PINNED_SUPPORTS
    )
    nonq_cand = extract_non_q_bpw(nondense_receipt)
    nonq_dense = extract_non_q_bpw(dense_receipt)

    write_json(
        run_root / "arm_DENSE_INT16_BASELINE" / "metrics.json",
        {
            "arm": "DENSE_INT16_BASELINE",
            "receipt_sha256": sha256_file(dense_receipt_path),
            "prior": dense_prior,
        },
    )
    write_json(
        run_root / "arm_NONDENSE_CANDIDATE" / "metrics.json",
        {
            "arm": "NONDENSE_CANDIDATE",
            "receipt_sha256": sha256_file(nondense_receipt_path),
            "prior": nondense_prior,
        },
    )

    branch, delta = classify_branch(
        dense_prior=dense_prior,
        nondense_prior=nondense_prior,
        delta_collapse=args.delta_collapse,
    )
    assert branch in FINAL_BRANCHES

    paired = {
        "dense_retained_rate": dense_prior.get("aggregate_exact_rate"),
        "candidate_retained_rate": nondense_prior.get("aggregate_exact_rate"),
        "delta_dense_minus_candidate": delta,
        "delta_collapse_input": args.delta_collapse,
        "row_weighted_aggregate": True,
        "budget_unresolved": True,
        "comparison": "paired_achieved_fidelity_at_N_fidelity_only",
    }
    write_json(run_root / "paired_probe_report.json", paired)
    write_json(
        run_root / "non_q_bpw_receipt.json",
        {
            "dense": nonq_dense,
            "candidate": nonq_cand,
            "budget_bpw": args.non_q_budget_bpw,
            "note": "budget UNRESOLVED for this diagnostic",
        },
    )
    write_json(
        run_root / "branch_verdict.json",
        {
            "branch": branch,
            "paired": paired,
            "non_q": {"dense": nonq_dense, "candidate": nonq_cand},
            "dense_pin_errors": dense_prior.get("pin_errors"),
            "nondense_pin_errors": nondense_prior.get("pin_errors"),
            "override_used": False,
            "diagnostic_class": "pre_full_stack_diagnostic",
            "synthetic": synthetic,
        },
    )
    write_terminal_receipt(
        run_root,
        branch=branch,
        extra={
            "paired_achieved_fidelity": paired,
            "non_q_bpw_candidate_or_instrument_gap": nonq_cand.get("status"),
            "geometry_storage_closed_v4_live": True,
            "note": "fidelity-only diagnostic; not A′ storage feasibility",
        },
        synthetic=synthetic,
        run_root_abs=run_root_abs,
    )
    print(json.dumps({"branch": branch, "delta": delta}, sort_keys=True), flush=True)
    return 0 if branch != "INSTRUMENT_GAP" else 2


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=["preflight", "schema", "reduce"], required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument(
        "--repo",
        type=Path,
        default=Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"),
    )
    ap.add_argument("--dense-scratch-root", type=Path, default=None)
    ap.add_argument("--nondense-scratch-root", type=Path, default=None)
    ap.add_argument("--delta-collapse", type=float, default=0.10)
    ap.add_argument("--non-q-budget-bpw", type=float, default=0.899682)
    ap.add_argument("--nonce", default="")
    ap.add_argument("--synthetic", action="store_true")
    ap.add_argument("--expect-head", default="")
    ap.add_argument("--expect-dirty-sha", default="")
    # default=None so 0 is a legitimate dirty denominator (presence semantics)
    ap.add_argument("--expect-dirty-n", type=int, default=None)
    ap.add_argument("--expect-probe-sha", default=PROBE_PIN)
    ap.add_argument("--expect-reducer-sha", default="")
    ap.add_argument("--expect-wrapper-sha", default="")
    ap.add_argument("--expect-rollup-sha", default="")
    # default=None so 0 is distinguishable from "flag omitted"
    ap.add_argument("--expect-rollup-n", type=int, default=None)
    args = ap.parse_args(argv)
    # NO self-derive of expect_reducer_sha / expect_wrapper_sha — authority
    # inputs are never derived from the artifact they authorize.
    if args.mode == "preflight":
        missing = [
            n
            for n, v in [
                ("--expect-head", args.expect_head),
                ("--expect-dirty-sha", args.expect_dirty_sha),
                ("--expect-reducer-sha", args.expect_reducer_sha),
                ("--expect-wrapper-sha", args.expect_wrapper_sha),
                ("--expect-rollup-sha", args.expect_rollup_sha),
            ]
            if not v
        ]
        if args.expect_dirty_n is None:
            missing.append("--expect-dirty-n")
        if args.expect_rollup_n is None:
            missing.append("--expect-rollup-n")
        if missing:
            print(f"preflight requires {missing}", file=sys.stderr)
            return 2
        return mode_preflight(args)
    if args.mode == "schema":
        return mode_schema(args)
    if args.dense_scratch_root is None or args.nondense_scratch_root is None:
        print("reduce requires dense/nondense scratch roots", file=sys.stderr)
        return 2
    return mode_reduce(args)


if __name__ == "__main__":
    sys.exit(main())
