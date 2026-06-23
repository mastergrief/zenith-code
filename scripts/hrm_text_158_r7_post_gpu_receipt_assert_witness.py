#!/usr/bin/env python3
"""R7 post-GPU diagnostic receipt assert witness."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_post_gpu_receipt_assert(run_root: Path) -> dict:
    receipt_path = run_root / "diagnostic" / "receipt.json"
    if not receipt_path.is_file():
        raise SystemExit("missing diagnostic receipt.json")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    if receipt.get("r7_cap_defer_pressure_instrumentation_enabled") is not True:
        failures.append("r7_instrumentation_not_enabled")
    if receipt.get("r7_deferred_backlog_carry_enabled") is not True:
        failures.append("r7_carry_not_enabled")
    sidecar = receipt.get("r7_cap_defer_pressure_sidecar_path")
    if not sidecar or not Path(sidecar).is_file():
        failures.append("missing_r7_sidecar")
    if int(receipt.get("steps_completed", 0)) < 10:
        failures.append("steps_incomplete")
    witness = {
        "schema": "hrm_text_158_r7_post_gpu_receipt_witness/v0",
        "post_gpu_receipt_assert_pass": not failures,
        "failures": failures,
        "steps_completed": receipt.get("steps_completed"),
    }
    out = run_root / "post_gpu" / "post_gpu_receipt_assert.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 post-GPU receipt assert witness.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    witness = run_post_gpu_receipt_assert(args.run_root)
    print(json.dumps(witness))
    return 0 if witness["post_gpu_receipt_assert_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
