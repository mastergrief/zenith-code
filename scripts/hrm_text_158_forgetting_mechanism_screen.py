"""Forgetting-mechanism screen — thin CLI facade (PLAN_v9 / r6b).

Bound by PLAN_v9 sha 07a02aff… + authority 1784812148229 + +1 implement 1784812700643
+ defect-cycle completion + r6b/r6c runner thinning.

Train/probe/receipt: screen_run_loop shim -> model_runtime / execution_loop / receipt_output.
Phase-1 aggregate orchestration stays here (CLI/IO + state-machine wiring only;
validators live in phase_receipt_contracts).

Developer checks: py_compile, CPU-static tests, --schema-only smoke,
vote_lifetime nonregression, --aggregate-phase1 dry orchestration.
Formal 1-step GPU --correctness-smoke and Phase-0/1 science runs are
claude/test-operator (not plan-dev).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from typing import Any

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.family_classifier import (  # noqa: E402
    ARM0,
    ARM1,
    ARM2,
    ARM3,
    FAMILY_F4,
)
from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (  # noqa: E402
    ArmReceiptContractError,
    build_phase1_terminal_receipt,
    decide_phase0_aggregate_transition,
    validate_phase0_receipt_for_aggregate,
    validate_shared_held_fixed_arm_receipts,
)
from calm.hrm_text_158.native_full_stack.screen_run_loop import (  # noqa: E402
    AUTHORITY_DISPATCH,
    COMMIT_SURFACE_FILES,
    EXPECTED_PARENT_SHA256,
    PLAN_SHA256,
    TOPK_PER_STEP,
    run_arm_screen,
    run_schema_only,
)

__all__ = [
    "_run_aggregate_phase1",
    "run_arm_screen",
    "run_schema_only",
    "PLAN_SHA256",
    "EXPECTED_PARENT_SHA256",
    "AUTHORITY_DISPATCH",
    "COMMIT_SURFACE_FILES",
]


def _load_arm_receipts(arm_receipts_arg: str | None) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    """Load exactly 4 arm receipts. Raises SystemExit on bad input."""
    if not arm_receipts_arg:
        raise SystemExit(
            "--arm-receipts required on cleared Phase-0 path "
            "(exactly 4 comma-separated arm0,arm1,arm2,arm3 receipts)"
        )
    paths = [p.strip() for p in str(arm_receipts_arg).split(",") if p.strip()]
    if len(paths) != 4:
        raise SystemExit(
            "--arm-receipts requires exactly 4 comma-separated arm receipts "
            "(arm0,arm1,arm2,arm3)"
        )
    loaded = []
    source_shas: dict[str, str] = {}
    for p in paths:
        with open(p, encoding="utf-8") as f:
            raw = f.read()
        r = json.loads(raw)
        loaded.append(r)
        source_shas[str(r.get("arm", p))] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    by_arm = {str(r.get("arm")): r for r in loaded}
    for need in (ARM0, ARM1, ARM2, ARM3):
        if need not in by_arm:
            raise SystemExit(f"missing arm receipt for {need} among {list(by_arm)}")
    return paths, by_arm, source_shas


def _run_aggregate_phase1(args: argparse.Namespace) -> int:
    """Orchestration: Phase-0 state machine first, then optional Phase-1 arms.

    Formal path requires a validated Phase-0 receipt (fail-closed). Synthetic
    `--phase0-censor-cleared` is dry/test-only and marks the receipt
    non-authoritative; it cannot substitute for Phase-0 proof on the formal path.

    Arm receipts are required ONLY on the cleared Phase-0 → Phase-1 path.
    """
    # --- Phase-0 proof FIRST (fail-closed) ---
    synthetic = False
    phase0_proof_meta: dict[str, Any] = {}
    p0_obj = None
    p0_pred_obj = None
    if args.phase0_receipt:
        with open(args.phase0_receipt, encoding="utf-8") as f:
            p0_raw = f.read()
        p0_obj = json.loads(p0_raw)
        phase0_proof_meta["phase0_receipt_sha256"] = hashlib.sha256(
            p0_raw.encode("utf-8")
        ).hexdigest()
        phase0_proof_meta["phase0_receipt_path"] = str(args.phase0_receipt)
    elif args.phase0_censor_cleared is not None:
        synthetic = True
        phase0_proof_meta["synthetic_phase0_override"] = True
        phase0_proof_meta["synthetic_value"] = int(args.phase0_censor_cleared)

    if getattr(args, "phase0_predecessor_receipt", None):
        with open(args.phase0_predecessor_receipt, encoding="utf-8") as f:
            pred_raw = f.read()
        p0_pred_obj = json.loads(pred_raw)
        phase0_proof_meta["phase0_predecessor_receipt_sha256"] = hashlib.sha256(
            pred_raw.encode("utf-8")
        ).hexdigest()
        phase0_proof_meta["phase0_predecessor_receipt_path"] = str(
            args.phase0_predecessor_receipt
        )

    p0_val = validate_phase0_receipt_for_aggregate(
        p0_obj,
        phase0_predecessor_receipt=p0_pred_obj,
    )
    phase0_proof_meta.update(p0_val)

    arms_arg = getattr(args, "arm_receipts", None)
    arms_supplied = bool(arms_arg and str(arms_arg).strip())

    if p0_obj is None and not synthetic:
        if arms_supplied:
            raise SystemExit(
                "arm receipts supplied without Phase-0 proof "
                "(contract violation — evaluate Phase-0 first)"
            )
        receipt = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={},
            arm_receipts={},
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            phase0_proof=phase0_proof_meta,
            authoritative=False,
            synthetic_phase0_override=False,
            force_null_reason="phase0_proof_missing",
            null_family=FAMILY_F4,
            arms_classified=False,
        )
        return _write_phase1(args, receipt, [])

    if synthetic and p0_obj is None:
        # Dry/test-only synthetic path still needs arms to exercise classifier.
        paths, by_arm, source_shas = _load_arm_receipts(arms_arg)
        phase0_cleared = bool(args.phase0_censor_cleared)
        try:
            shared = validate_shared_held_fixed_arm_receipts(
                by_arm,
                expected_plan_sha256=PLAN_SHA256,
                expected_parent_sha256=EXPECTED_PARENT_SHA256,
                expected_authority_dispatch=AUTHORITY_DISPATCH,
            )
        except ArmReceiptContractError as e:
            raise SystemExit(f"arm receipt contract fail-closed: {e}") from e
        receipt = build_phase1_terminal_receipt(
            phase0_censor_cleared=phase0_cleared,
            control_receipt=by_arm[ARM0],
            arm_receipts={
                ARM1: by_arm[ARM1],
                ARM2: by_arm[ARM2],
                ARM3: by_arm[ARM3],
            },
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            phase0_proof=phase0_proof_meta,
            source_receipt_sha256s=source_shas,
            shared_contract=shared,
            authoritative=False,
            synthetic_phase0_override=True,
            force_null_reason=(None if phase0_cleared else "phase0_censor_uncleared"),
        )
        return _write_phase1(args, receipt, paths)

    # Formal Phase-0 present — state machine.
    decision = decide_phase0_aggregate_transition(p0_val)

    if decision["action"] == "malformed":
        if arms_supplied:
            raise SystemExit(
                "arm receipts supplied with malformed Phase-0 proof "
                "(contract violation)"
            )
        receipt = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={},
            arm_receipts={},
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            phase0_proof=phase0_proof_meta,
            authoritative=False,
            synthetic_phase0_override=False,
            force_null_reason=str(decision["stop_reason"]),
            null_family=FAMILY_F4,
            transition=decision.get("transition"),
            arms_classified=False,
        )
        return _write_phase1(args, receipt, [])

    if decision["action"] == "fallback_required":
        # Uncleared 150 → transition; do NOT classify arms (even if supplied).
        if arms_supplied:
            phase0_proof_meta["arms_supplied_without_cleared_gate"] = True
            phase0_proof_meta["arms_rejected"] = True
        receipt = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={},
            arm_receipts={},
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            phase0_proof=phase0_proof_meta,
            authoritative=False,
            synthetic_phase0_override=False,
            force_null_reason="phase0_censor_uncleared_fallback_required",
            null_family=None,
            transition="fallback_required",
            arms_classified=False,
        )
        return _write_phase1(args, receipt, [])

    if decision["action"] == "design_null_censor_unreducible":
        # Failed 600 fallback → authoritative design-null; no Phase-1 arms.
        if arms_supplied:
            raise SystemExit(
                "arm receipts supplied on uncleared Phase-0b path "
                "(design_null_censor_unreducible forbids Phase-1 arms)"
            )
        receipt = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={},
            arm_receipts={},
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            phase0_proof=phase0_proof_meta,
            authoritative=True,
            synthetic_phase0_override=False,
            force_null_reason="design_null_censor_unreducible",
            null_family=FAMILY_F4,
            transition="design_null_censor_unreducible",
            arms_classified=False,
        )
        return _write_phase1(args, receipt, [])

    # Cleared Phase-0/0b → require arms + classify.
    assert decision["action"] == "enter_phase1"
    paths, by_arm, source_shas = _load_arm_receipts(arms_arg)
    try:
        shared = validate_shared_held_fixed_arm_receipts(
            by_arm,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=EXPECTED_PARENT_SHA256,
            expected_authority_dispatch=AUTHORITY_DISPATCH,
            expected_steps=int(p0_val["steps"]),
            phase0_receipt=p0_obj,
        )
    except ArmReceiptContractError as e:
        raise SystemExit(f"arm receipt contract fail-closed: {e}") from e

    receipt = build_phase1_terminal_receipt(
        phase0_censor_cleared=True,
        control_receipt=by_arm[ARM0],
        arm_receipts={
            ARM1: by_arm[ARM1],
            ARM2: by_arm[ARM2],
            ARM3: by_arm[ARM3],
        },
        plan_sha256=PLAN_SHA256,
        authority_dispatch=AUTHORITY_DISPATCH,
        phase0_proof=phase0_proof_meta,
        source_receipt_sha256s=source_shas,
        shared_contract=shared,
        authoritative=True,
        synthetic_phase0_override=False,
        arms_classified=True,
    )
    return _write_phase1(args, receipt, paths)


def _write_phase1(args: argparse.Namespace, receipt: dict, paths: list[str]) -> int:
    receipt["source_arm_receipts"] = paths
    out = args.output_json or (
        "artifacts/acc_entropy/forgetting_mechanism_phase1_receipt.json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(receipt, f, indent=2)
    print(
        f"[forget-mech] phase1 aggregate -> {out} family={receipt['family']} "
        f"reason={receipt['stop_reason']} authoritative={receipt.get('authoritative')}",
        flush=True,
    )
    return 0



def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--topk", type=int, default=TOPK_PER_STEP)
    ap.add_argument("--arm", default=ARM0)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument("--correctness-smoke", action="store_true")
    ap.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip step0/final exact-match probes (auto under --correctness-smoke).",
    )
    ap.add_argument(
        "--aggregate-phase1",
        action="store_true",
        help="Aggregate 4 arm receipts into forgetting_mechanism_phase1_receipt.json",
    )
    ap.add_argument(
        "--arm-receipts",
        default=None,
        help=(
            "Comma-separated arm0,arm1,arm2,arm3 receipt paths. Required only on "
            "cleared Phase-0 → Phase-1 path; forbidden/ignored on uncleared transitions."
        ),
    )
    ap.add_argument("--phase0-receipt", default=None)
    ap.add_argument(
        "--phase0-predecessor-receipt",
        default=None,
        help=(
            "Required when Phase-0 receipt steps==600 (fallback-once): hash-bound "
            "failed 150-step predecessor (lcf>=0.50, full provenance+geometry)."
        ),
    )
    ap.add_argument(
        "--phase0-censor-cleared",
        type=int,
        choices=[0, 1],
        default=None,
        help=(
            "DRY/TEST-ONLY synthetic Phase-0 override. Formal aggregation requires "
            "--phase0-receipt; synthetic marks receipt non-authoritative."
        ),
    )
    args = ap.parse_args()

    if args.aggregate_phase1:
        return _run_aggregate_phase1(args)

    if args.schema_only:
        return run_schema_only(args)

    if not args.ckpt_path:
        raise SystemExit("--ckpt-path is required unless --schema-only/--aggregate-phase1")

    if args.correctness_smoke:
        args.steps = 1
        args.skip_probes = True

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    return run_arm_screen(args)


if __name__ == "__main__":
    raise SystemExit(main())
