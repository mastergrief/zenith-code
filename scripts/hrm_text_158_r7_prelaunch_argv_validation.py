#!/usr/bin/env python3
"""R7 single-arm prelaunch argv validation (extracted from replay heredoc)."""
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


def run_argv_validation(run_root: Path, head_pin: str) -> dict:
    argv_echo = json.loads((run_root / "prelaunch" / "argv_echo.json").read_text(encoding="utf-8"))
    parser = build_arg_parser()
    failures: list[str] = []
    arms_out: list[dict] = []
    for arm, cfg in sorted((argv_echo.get("arms") or {}).items()):
        tokens = probe_tokens_from_argv_list(list(cfg["argv"]))
        try:
            ns = parser.parse_args(tokens)
            arms_out.append(
                {
                    "arm": arm,
                    "parse_ok": True,
                    "phase": str(ns.phase),
                    "eligible_scope": str(ns.eligible_scope),
                    "global_cap_contract": str(ns.global_cap_contract),
                }
            )
        except Exception as exc:
            failures.append(f"{arm}_{type(exc).__name__}:{exc}")
    receipt = {
        "schema": "hrm_text_158_r7_prelaunch_argv_validation/v0",
        "argv_echo_path": str(run_root / "prelaunch" / "argv_echo.json"),
        "head_pin": head_pin,
        "all_parse_ok": (not failures and len(arms_out) == 1),
        "failures": failures,
        "arms": arms_out,
        "note": (
            "R7 single-arm diagnostic; S3bb validator requires >=2 arms so this uses "
            "R7-specific single-arm argparse validation"
        ),
    }
    out = run_root / "prelaunch" / "argv_validation.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R7 prelaunch argv validation witness.")
    parser.add_argument("run_root", type=Path)
    parser.add_argument("head_pin")
    args = parser.parse_args(argv)
    receipt = run_argv_validation(args.run_root, args.head_pin)
    print(json.dumps(receipt))
    return 0 if receipt["all_parse_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
