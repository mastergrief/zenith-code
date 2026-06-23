#!/usr/bin/env python3
"""R7 prelaunch persistence witness — checks prior witness JSON outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def run_prelaunch_persistence_witness(run_root: Path) -> dict:
    failures: list[str] = []
    checks: dict = {}

    def check_file(fname: str, field_checks: dict) -> None:
        p = run_root / "prelaunch" / fname
        if not p.is_file():
            failures.append(f"missing:{fname}")
            checks[fname] = {"exists": False}
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        checks[fname] = {"exists": True}
        for field, expected in field_checks.items():
            actual = data.get(field)
            if actual != expected:
                failures.append(f"{fname}:{field}_expected_{expected}_got_{actual!r}")
            checks[fname][field] = actual

    def check_multistep_backlog_witness() -> None:
        fname = "r7_multistep_backlog_carry_witness.json"
        p = run_root / "prelaunch" / fname
        if not p.is_file():
            failures.append(f"missing:{fname}")
            checks[fname] = {"exists": False}
            return
        data = json.loads(p.read_text(encoding="utf-8"))
        checks[fname] = {"exists": True, "carry_enabled": data.get("carry_enabled")}
        if data.get("carry_enabled") is not True:
            failures.append(f"{fname}:carry_enabled_expected_True_got_{data.get('carry_enabled')!r}")
        max_age = int(data.get("step_n_plus_1_max_age_steps", 0))
        checks[fname]["step_n_plus_1_max_age_steps"] = max_age
        if max_age < 1:
            failures.append(f"{fname}:step_n_plus_1_max_age_steps_expected_>=1_got_{max_age}")

    check_file("argv_validation.json", {"all_parse_ok": True})
    check_file("box_code_currency_preflight.json", {"code_currency_pass": True})
    check_file("parent_checkpoint_rehash.json", {"match": True})
    check_file("r7_cap_seam_field_presence_witness.json", {"field_presence_pass": True})
    check_multistep_backlog_witness()
    check_file(
        "sub2_readiness_receipt.json",
        {
            "ready_for_pre_full_stack_diagnostic": True,
            "ready_for_main_science": False,
            "main_science_launch_blocked": True,
        },
    )
    witness = {
        "schema": "hrm_text_158_prelaunch_persistence_witness/v1.r7",
        "prelaunch_persistence_witness_pass": not failures,
        "failures": failures,
        "checks": checks,
    }
    out = run_root / "prelaunch" / "prelaunch_persistence_witness.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 prelaunch persistence witness.")
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args(argv)
    witness = run_prelaunch_persistence_witness(args.run_root)
    print(json.dumps(witness))
    return 0 if witness["prelaunch_persistence_witness_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
