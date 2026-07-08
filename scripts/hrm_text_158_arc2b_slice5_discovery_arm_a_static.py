#!/usr/bin/env python3
"""Arc #2b Slice-5 discovery Arm A static CPU harness (dense-width arithmetic floor).

Frozen v6 plan (co_lead gate-2 PASS 1783512484577, +1 implement 1783526612437).
Arm A = STATIC CPU arithmetic: W8=8 bpw, W7=7 bpw, both >> 0.4 budget.
Fixed-width cannot reach sub-2 ceiling (ternary_hybrid_stack.md L50).
0 GPU. NOT on the live decay-gap curve. NOT a branch-verdict arm — establishes
fixed-width floor only.

This harness computes the arithmetic bpw floor for W8 (±127 clip) and W7 (±63
clip) dense accumulators and confirms neither can reach sub-2 (0.4 bpw strict).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.arc2b_slice5_discovery_branch import (
    ARM_A_W7_BPW,
    ARM_A_W8_BPW,
    CLASSIFIER,
    DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    EVIDENCE_ARM_A_STATIC,
    RECEIPT_SCHEMA,
)

ACTIVE_TASK_ID = "1783272482268-052281aa"


def compute_w8_bpw() -> float:
    """W8 dense accumulator: 8 bits per weight (int8 container, ±127 clip).
    Physical bpw = 8.0. Cannot reach sub-2 (0.4 bpw)."""
    return float(ARM_A_W8_BPW)


def compute_w7_bpw() -> float:
    """W7 dense accumulator: 7 bits per weight (int8 container with ±63 clip,
    but physical storage is still 8 bits; logical bpw = 7).
    Cannot reach sub-2 (0.4 bpw)."""
    return float(ARM_A_W7_BPW)


def compute_w8_gap_bpw(
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
) -> float:
    """budget_gap_bpw for W8 = 8.0 - 0.4 = 7.6 (way over budget)."""
    return compute_w8_bpw() - float(effective_acc_budget_bpw)


def compute_w7_gap_bpw(
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
) -> float:
    """budget_gap_bpw for W7 = 7.0 - 0.4 = 6.6 (way over budget)."""
    return compute_w7_bpw() - float(effective_acc_budget_bpw)


def can_reach_sub2(bpw: float) -> bool:
    """Fixed-width bpw < 0.4 strict? W8=8, W7=7 => both False."""
    return float(bpw) < DEFAULT_EFFECTIVE_ACC_BUDGET_BPW


def build_arm_a_receipt(
    *,
    effective_acc_budget_bpw: float = DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
) -> dict[str, Any]:
    """Build the Arm A static CPU receipt (arithmetic floor, 0 GPU)."""
    w8_bpw = compute_w8_bpw()
    w7_bpw = compute_w7_bpw()
    w8_gap = compute_w8_gap_bpw(effective_acc_budget_bpw)
    w7_gap = compute_w7_gap_bpw(effective_acc_budget_bpw)
    return {
        "schema": RECEIPT_SCHEMA,
        "task_id": ACTIVE_TASK_ID,
        "classifier": CLASSIFIER,
        "evidence_source": EVIDENCE_ARM_A_STATIC,
        "arm_a_bpw_w8": w8_bpw,
        "arm_a_bpw_w7": w7_bpw,
        "arm_a_gap_w8": w8_gap,
        "arm_a_gap_w7": w7_gap,
        "arm_a_w8_can_reach_sub2": can_reach_sub2(w8_bpw),
        "arm_a_w7_can_reach_sub2": can_reach_sub2(w7_bpw),
        "arm_a_finding": (
            "fixed_width_dense_accumulator_cannot_reach_sub2_ceiling: "
            f"W8={w8_bpw} bpw, W7={w7_bpw} bpw, both >> {effective_acc_budget_bpw} budget"
        ),
        "effective_acc_budget_bpw": float(effective_acc_budget_bpw),
        "ready_for_main_science": False,
        "counts_as_sub2": False,
        "pre_full_stack_diagnostic": True,
        "autonomy_rung": "arm_a_static_cpu",
        "generated_at_unix": int(time.time()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output receipt path (default: stdout)",
    )
    ap.add_argument(
        "--effective-acc-budget-bpw",
        type=float,
        default=DEFAULT_EFFECTIVE_ACC_BUDGET_BPW,
    )
    args = ap.parse_args()

    receipt = build_arm_a_receipt(
        effective_acc_budget_bpw=args.effective_acc_budget_bpw,
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
