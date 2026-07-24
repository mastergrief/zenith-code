#!/usr/bin/env python3
"""CPU-first ARM1 decay suppression discriminator smoke (PLAN_v10.1r8).

Validates exhaustive 765-pair transfer table + compact pre/post schema helpers
+ observer cost/scale smoke. GPU 25-step diagnostic is SEPARATELY GATED.
This script MUST NOT compare any mass to the formal-150 150-step denominator.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from calm.hrm_text_158.native_full_stack.forgetting_screen_pre_post_telemetry import (
    PrePostTransformAccumulator,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_1_contract import (
    PRE_POST_SCHEMA,
    TRANSFER_CARDINALITY,
    TRANSFER_LAW,
    build_exhaustive_transfer_table,
    classify_discriminator_branch,
    pre_post_evidence_schema_valid,
    transfer_pair,
    trunc_toward_zero_mul_31_32,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
    assert_hotpath_sync_allowlist,
    assert_pre_post_telemetry_single_d2h,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema-only", action="store_true")
    ap.add_argument(
        "--out",
        default="artifacts/acc_entropy/arm1_decay_suppression_discriminator_cpu_smoke.json",
    )
    args = ap.parse_args()

    rows = build_exhaustive_transfer_table()
    assert len(rows) == TRANSFER_CARDINALITY
    by = {(r["acc"], r["move"]): r for r in rows}
    assert by[(127, 1)]["pre"] == 127
    assert by[(127, 1)]["out"] == trunc_toward_zero_mul_31_32(127)
    assert by[(-127, -1)]["pre"] == -127
    for acc, move in ((0, 1), (1, 0), (31, 0), (32, 0), (-5, 1)):
        pre, out = transfer_pair(acc, move)
        assert pre == max(-127, min(127, acc + move))
        assert out == int(math.trunc(pre * 31 / 32))

    synthetic = {
        "schema": PRE_POST_SCHEMA,
        "law": TRANSFER_LAW,
        "move_abs_bins": {"1": 50},
        "move_nonzero_count": 50,
        "post_projection": {"nonzero": 40, "abs_max": 1, "abs_p50": 1.0, "abs_p90": 1.0},
        "post_decay": {"nonzero": 0, "abs_max": 0, "abs_p50": 0.0, "abs_p90": 0.0},
        "pre_nonzero_to_post_zero_count": 40,
        "pre_nonzero_to_post_zero_frac": 1.0,
        "post_decay_candidate_count": 0,
        "law_mismatch_count": 0,
        "steps_accumulated": 25,
    }
    ok, reason = pre_post_evidence_schema_valid(synthetic)
    assert ok, reason
    branch = classify_discriminator_branch(synthetic)
    assert branch == "strong_S1"

    hotpath = assert_hotpath_sync_allowlist()
    d2h = assert_pre_post_telemetry_single_d2h()

    acc = PrePostTransformAccumulator(device="cpu")
    pre = {"w": torch.randint(-2, 3, (1024,), dtype=torch.int16)}
    post = {"w": torch.zeros(1024, dtype=torch.int16)}
    moves = {"w": torch.randint(-1, 2, (1024,), dtype=torch.int8)}
    t0 = time.perf_counter()
    for _ in range(32):
        acc.accumulate_step(
            moves=moves, acc_pre_decay=pre, acc_post_decay=post, n_cand_after_decay=0
        )
    cost_out = acc.finalize()
    dt = time.perf_counter() - t0
    assert cost_out["steps_accumulated"] == 32
    assert "0" not in cost_out["move_abs_bins"]

    out = {
        "schema": "arm1_decay_suppression_discriminator_cpu_smoke_v1",
        "transfer_cardinality": len(rows),
        "transfer_law": TRANSFER_LAW,
        "boundary_rows_ok": True,
        "synthetic_pre_post_schema_ok": True,
        "discriminator_branch_synthetic": branch,
        "hotpath_allowlist_ok": True,
        "d2h_single_finalize": d2h,
        "cost_smoke": {
            "steps": 32,
            "n": 1024,
            "dt_s": dt,
            "move_nonzero_count": cost_out["move_nonzero_count"],
        },
        "credited_mass_vs_formal150": "NOT_APPLICABLE_25step_or_cpu_smoke",
        "gpu_25_step": "SEPARATELY_GATED_packet_slice",
        "schema_only": bool(args.schema_only),
        "hotpath_files": sorted(hotpath.keys()),
    }
    path = Path(args.out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"ok": True, "out": str(path), **out}, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
