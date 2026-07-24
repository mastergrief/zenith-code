"""Forgetting-mechanism screen — thin CLI facade (PLAN_v10r4).

Bound by frozen PLAN_v10r4 msg 1784890890052 + +1 1784891014883
+ sixth-path amendment 1784891334017 + defect-cycle 1784892185413.

Train/probe/receipt: screen_run_loop shim.
Aggregate: pin formal-150 control + exactly 3 mechanism receipts (F1/F2/F3).

Legacy flags --phase0-receipt / --phase0-predecessor-receipt /
    --phase0-censor-cleared HARD-REFUSE under v10.
--control-baseline-json + --control-baseline-sha256 MANDATORY on aggregate.
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
)
from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (  # noqa: E402
    FORMAL150_CONTROL_SHA256,
    V10ArmReceiptContractError,
    build_v10_terminal_receipt,
    pin_and_load_formal_control_baseline,
    validate_three_mechanism_arm_receipts_v10,
)
from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (  # noqa: E402
    sanitize_receipt_for_strict_json,
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

_LEGACY_PHASE0_FLAGS = (
    "phase0_receipt",
    "phase0_predecessor_receipt",
    "phase0_censor_cleared",
)


def _refuse_legacy_phase0_flags(args: argparse.Namespace) -> None:
    present = [
        name
        for name in _LEGACY_PHASE0_FLAGS
        if getattr(args, name, None) not in (None, "")
    ]
    if present:
        raise SystemExit(
            "HARD-REFUSE under PLAN_v10: legacy Phase-0 flags not accepted: "
            + ", ".join(f"--{n.replace('_', '-')}" for n in present)
        )


def _require_control_baseline(args: argparse.Namespace) -> None:
    if not getattr(args, "control_baseline_json", None):
        raise SystemExit(
            "FAIL-CLOSED: --control-baseline-json is mandatory on "
            "--aggregate-phase1 / v10 classification paths"
        )
    if not getattr(args, "control_baseline_sha256", None):
        raise SystemExit(
            "FAIL-CLOSED: --control-baseline-sha256 is mandatory on "
            "--aggregate-phase1 / v10 classification paths"
        )


def _load_mechanism_receipts(
    arm_receipts_arg: str | None,
) -> tuple[list[str], dict[str, Any], dict[str, str]]:
    if not arm_receipts_arg:
        raise SystemExit(
            "--arm-receipts required on v10 classify path "
            "(exactly 3 comma-separated arm1,arm2,arm3 mechanism receipts)"
        )
    paths = [p.strip() for p in str(arm_receipts_arg).split(",") if p.strip()]
    if len(paths) != 3:
        raise SystemExit(
            "--arm-receipts requires exactly 3 comma-separated mechanism receipts "
            "(arm1,arm2,arm3); formal-150 artifact is the sole control"
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
    for need in (ARM1, ARM2, ARM3):
        if need not in by_arm:
            raise SystemExit(f"missing arm receipt for {need} among {list(by_arm)}")
    return paths, by_arm, source_shas


def _run_aggregate_phase1(args: argparse.Namespace) -> int:
    """v10 aggregate: pin formal control, then classify three mechanism arms."""
    _refuse_legacy_phase0_flags(args)
    _require_control_baseline(args)

    control_bind = pin_and_load_formal_control_baseline(
        args.control_baseline_json,
        supplied_sha256=str(args.control_baseline_sha256),
    )
    if not control_bind.get("ok"):
        receipt = build_v10_terminal_receipt(
            control_bind=control_bind,
            arm_receipts={},
            plan_sha256=PLAN_SHA256,
            authority_dispatch=AUTHORITY_DISPATCH,
            force_null_reason=str(control_bind.get("reason") or "control_baseline_not_ok"),
        )
        return _write_phase1(args, receipt, [])

    paths, by_arm, source_shas = _load_mechanism_receipts(getattr(args, "arm_receipts", None))
    try:
        shared = validate_three_mechanism_arm_receipts_v10(
            by_arm,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=EXPECTED_PARENT_SHA256,
            expected_authority_dispatch=AUTHORITY_DISPATCH,
        )
    except V10ArmReceiptContractError as e:
        raise SystemExit(f"arm receipt contract fail-closed: {e}") from e

    receipt = build_v10_terminal_receipt(
        control_bind=control_bind,
        arm_receipts={
            ARM1: by_arm[ARM1],
            ARM2: by_arm[ARM2],
            ARM3: by_arm[ARM3],
        },
        plan_sha256=PLAN_SHA256,
        authority_dispatch=AUTHORITY_DISPATCH,
    )
    receipt["source_receipt_sha256s"] = source_shas
    receipt["shared_contract"] = shared
    receipt["control_arm0_receipt_present"] = False
    return _write_phase1(args, receipt, paths)


def _write_phase1(args: argparse.Namespace, receipt: dict, paths: list[str]) -> int:
    receipt["source_arm_receipts"] = paths
    out = args.output_json or (
        "artifacts/acc_entropy/forgetting_mechanism_phase1_receipt.json"
    )
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    payload = sanitize_receipt_for_strict_json(receipt)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, allow_nan=False)
    print(
        f"[forget-mech] v10 aggregate -> {out} family={receipt['family']} "
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
    ap.add_argument(
        "--schema-only",
        action="store_true",
        help=(
            "Shape/schema smoke only — does NOT perform control-bind or "
            "v10 classification; cannot bypass the control validator."
        ),
    )
    ap.add_argument("--correctness-smoke", action="store_true")
    ap.add_argument(
        "--skip-probes",
        action="store_true",
        help="Skip step0/final exact-match probes (auto under --correctness-smoke).",
    )
    ap.add_argument(
        "--telemetry",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Attach DeviceLifecycleStore R1 observer (default ON). "
            "--no-telemetry is timing-only / observer-cost pairing; OFF receipts "
            "are marked telemetry=false and lack demand/deferred_survival so they "
            "cannot pass v10 G0/three-arm as formal arm receipts."
        ),
    )
    ap.add_argument(
        "--aggregate-phase1",
        action="store_true",
        help="Aggregate 3 mechanism receipts under pinned formal-150 control (PLAN_v10).",
    )
    ap.add_argument(
        "--arm-receipts",
        default=None,
        help="Comma-separated arm1,arm2,arm3 receipt paths (required on classify).",
    )
    ap.add_argument(
        "--control-baseline-json",
        default=None,
        help="Path to formal-150 control baseline JSON (MANDATORY on aggregate).",
    )
    ap.add_argument(
        "--control-baseline-sha256",
        default=None,
        help=(
            "Must equal FORMAL150_CONTROL_SHA256 (pinned; not operator-selectable). "
            f"Formal: {FORMAL150_CONTROL_SHA256}"
        ),
    )
    # Legacy Phase-0 flags retained ONLY so we can HARD-REFUSE if supplied.
    ap.add_argument("--phase0-receipt", default=None, help=argparse.SUPPRESS)
    ap.add_argument("--phase0-predecessor-receipt", default=None, help=argparse.SUPPRESS)
    ap.add_argument(
        "--phase0-censor-cleared",
        type=int,
        choices=[0, 1],
        default=None,
        help=argparse.SUPPRESS,
    )
    args = ap.parse_args()

    if args.aggregate_phase1:
        return _run_aggregate_phase1(args)

    # Refuse legacy flags on ALL entrypoints (not only aggregate).
    _refuse_legacy_phase0_flags(args)

    if args.schema_only:
        # Explicit non-authority: schema/shape only; no control-bind claim.
        return run_schema_only(args)

    if not args.ckpt_path:
        raise SystemExit(
            "--ckpt-path is required unless --schema-only/--aggregate-phase1"
        )

    if args.correctness_smoke:
        args.steps = 1
        args.skip_probes = True

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but unavailable")

    return run_arm_screen(args)


if __name__ == "__main__":
    raise SystemExit(main())
