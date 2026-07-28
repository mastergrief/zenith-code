#!/usr/bin/env python3
"""Dry-exec harness for sparse live carrier production landing (PLAN_v16)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CANONICAL_AUDIT = (
    "artifacts/acc_entropy/"
    "optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json"
)
CANONICAL_SHA = "783f279986ebaa9bd7d170b5996146a319e9c8f1980939ec8ee49ac4b5d5db2f"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _o_excl_write(path: Path, payload: dict[str, Any]) -> None:
    data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    try:
        os.write(fd, data)
    finally:
        os.close(fd)


def _validate_pin_entry_keys(meta: dict[str, Any], *, required: set[str], optional: set[str]) -> None:
    keys = set(meta)
    missing = required - keys
    unknown = keys - (required | optional)
    if missing or unknown:
        raise SystemExit(f"pin entry key schema invalid missing={missing} unknown={unknown}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--mode", required=True, choices=("fused_only", "oracle_on_schema_only"))
    ap.add_argument("--snapshot", required=True, help="FINAL_SNAPSHOT_v1.json path")
    ap.add_argument("--dry-exec-out", required=True)
    args = ap.parse_args()

    plan_path = Path(args.plan)
    if not plan_path.exists():
        raise SystemExit(f"plan missing: {plan_path}")
    plan = json.loads(plan_path.read_text())
    if plan.get("plan_revision") != "v16" or not str(plan_path).endswith("PLAN_v16.json"):
        raise SystemExit(
            f"operative_plan_binding reject: plan_revision={plan.get('plan_revision')!r} path={plan_path}"
        )

    snap_path = Path(args.snapshot)
    if not snap_path.exists():
        raise SystemExit(f"snapshot missing (S3 requires FINAL_SNAPSHOT before dry-exec): {snap_path}")
    snapshot = json.loads(snap_path.read_text())
    if snapshot.get("schema_version") != "sparse_live_carrier_final_implementation_snapshot_v1":
        raise SystemExit("snapshot schema_version mismatch")
    if str(snap_path) in (snapshot.get("entries") or {}):
        raise SystemExit("snapshot self-referential hash forbidden")
    entries = snapshot.get("entries") or {}
    if not isinstance(entries, dict) or not entries:
        raise SystemExit("snapshot entries empty")
    for rel, meta in entries.items():
        if not isinstance(meta, dict):
            raise SystemExit(f"snapshot entry not object: {rel}")
        _validate_pin_entry_keys(meta, required={"expected_sha256", "why"}, optional=set())
        live = _sha(Path(rel))
        if live != meta["expected_sha256"]:
            raise SystemExit(
                f"snapshot drift RED {rel}: live={live} expected={meta['expected_sha256']}"
            )

    before = _sha(Path(CANONICAL_AUDIT))
    if before != CANONICAL_SHA:
        raise SystemExit(f"canonical audit sha mismatch before: {before}")

    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        normalize_sparse_vote_authority_mode,
        SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
        OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
        build_optimizer_credit_state_fail_closed_receipt,
    )
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
    )

    if args.mode == "fused_only":
        resolved = normalize_sparse_vote_authority_mode()
    else:
        resolved = normalize_sparse_vote_authority_mode("oracle_on")

    residual = build_optimizer_credit_state_fail_closed_receipt()
    geometry = plan.get("frozen_production_call_geometry", {})
    receipt = {
        "dry_exec": True,
        "sparse_vote_authority_mode": resolved,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "candidate_oracle_control_enabled": False,
        "votes_by_key_applied": None,
        "candidate_mode": ACCUMULATOR_SUBSTITUTE_LOCAL_VOTE_UPDATE_EXECUTABLE,
        "transient_over2_tensors": ["weighted_grad"],
        "frozen_geometry": geometry.get("apply_bounded_delta_vote_step_kwargs_exact"),
        "plan_path": str(plan_path),
        "plan_sha256": _sha(plan_path),
        "plan_revision": plan.get("plan_revision"),
        "snapshot_path": str(snap_path),
        "snapshot_sha256": _sha(snap_path),
        "canonical_audit_sha256_before": before,
        "readiness_residual_weighted_grad": "weighted_grad" in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
        "readiness_blocked_reason_prefix": str(residual.blocked_reason)[:120],
        "mode_selection_seam_default": "fused_only",
        "phase_budget_enforcement_design": plan.get("phase_budget_enforcement", {}).get("design_choice"),
        "pin_source": "final_implementation_snapshot",
        "slice_readiness_claim": False,
    }
    for bad in ("credit", "projected_moves", "dense_rank_votes", "dense_votes_by_key"):
        if bad in receipt:
            raise SystemExit(f"forbidden field present: {bad}")

    out = Path(args.dry_exec_out)
    if out.exists():
        raise SystemExit(f"dry-exec out exists (O_EXCL refuse overwrite): {out}")
    out.parent.mkdir(parents=True, exist_ok=True)
    _o_excl_write(out, receipt)

    after = _sha(Path(CANONICAL_AUDIT))
    if after != CANONICAL_SHA:
        raise SystemExit(f"canonical audit sha mismatch after: {after}")

    print(
        json.dumps(
            {
                "ok": True,
                "dry_exec_out": str(out),
                "mode": resolved,
                "canonical": after,
                "snapshot_sha256": receipt["snapshot_sha256"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
