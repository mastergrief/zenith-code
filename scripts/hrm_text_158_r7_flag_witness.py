#!/usr/bin/env python3
"""R7 flag witness — verifies required diagnostic tokens in argv echo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser
from scripts.hrm_text_158_s3bb_prelaunch_argv_validation import probe_tokens_from_argv_list


def run_flag_witness(run_root: Path) -> dict:
    argv_echo = json.loads((run_root / "prelaunch" / "argv_echo.json").read_text(encoding="utf-8"))
    entry = argv_echo["arms"]["diagnostic"]
    tokens = probe_tokens_from_argv_list(list(entry["argv"]))
    parser = build_arg_parser()
    ns = parser.parse_args(tokens)
    failures: list[str] = []
    required_tokens = [
        "--r7-deferred-backlog-carry",
        "--r7-cap-defer-pressure-instrumentation",
        "--global-cap-contract",
        "c1_banked_faithful_long_run_global_cap",
        "--persistent-accumulator-w6-byte-packed",
        "--persistent-q-ternary-byte-packed",
        "--curriculum-seed",
        "44",
        "--support-order-seed",
        "43",
        "--eligible-scope",
        "all-bitlinear",
    ]
    for tok in required_tokens:
        if tok not in tokens:
            failures.append(f"missing_token:{tok}")
    if not bool(ns.r7_deferred_backlog_carry):
        failures.append("r7_deferred_backlog_carry_false")
    if not bool(ns.r7_cap_defer_pressure_instrumentation):
        failures.append("r7_instrumentation_false")
    if str(ns.global_cap_contract) != "c1_banked_faithful_long_run_global_cap":
        failures.append("global_cap_contract_mismatch")
    witness = {
        "schema": "hrm_text_158_r7_flag_witness/v0",
        "r7_flag_witness_pass": not failures,
        "failures": failures,
        "tokens": tokens,
    }
    out = run_root / "prelaunch" / "r7_flag_witness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 flag witness.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    witness = run_flag_witness(args.run_root)
    print(json.dumps(witness))
    return 0 if witness["r7_flag_witness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
