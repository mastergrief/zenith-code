#!/usr/bin/env python3
"""R8 flag witness — global_cap_relax_512 + prior-audit supports in argv echo."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    B1_PRIOR_AUDIT_SUPPORTS,
    build_arg_parser,
)
from scripts.hrm_text_158_s3bb_prelaunch_argv_validation import probe_tokens_from_argv_list

REQUIRED_PRIOR_AUDIT = "L0b,math_a0,L0c1"


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
        GLOBAL_CAP_RELAX_512_CONTRACT_NAME,
        "--prior-audit-supports",
        REQUIRED_PRIOR_AUDIT,
        "--two-tier-carry-w6-enabled",
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
    if C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME in tokens:
        failures.append("forbidden_c1_banked_contract_token")
    if not bool(ns.r7_deferred_backlog_carry):
        failures.append("r7_deferred_backlog_carry_false")
    if not bool(ns.r7_cap_defer_pressure_instrumentation):
        failures.append("r7_instrumentation_false")
    if str(ns.global_cap_contract) != GLOBAL_CAP_RELAX_512_CONTRACT_NAME:
        failures.append("global_cap_contract_mismatch")
    if str(ns.global_cap_contract) == C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME:
        failures.append("c1_banked_contract_selected")
    prior_audit = str(getattr(ns, "prior_audit_supports", "") or "")
    if prior_audit.replace(" ", "") != REQUIRED_PRIOR_AUDIT:
        failures.append("prior_audit_supports_mismatch")
    for support in B1_PRIOR_AUDIT_SUPPORTS:
        if support not in prior_audit:
            failures.append(f"missing_prior_audit_support:{support}")
    witness = {
        "schema": "hrm_text_158_r8_flag_witness/v0",
        "r8_flag_witness_pass": not failures,
        "failures": failures,
        "tokens": tokens,
        "global_cap_contract": str(ns.global_cap_contract),
        "prior_audit_supports": prior_audit,
    }
    out = run_root / "prelaunch" / "r8_flag_witness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R8 flag witness.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    witness = run_flag_witness(args.run_root)
    print(json.dumps(witness))
    return 0 if witness["r8_flag_witness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
