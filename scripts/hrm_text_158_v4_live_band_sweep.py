#!/usr/bin/env python3
"""Thin CLI for V4-LIVE CPU band sweep."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import DEFAULT_VERDICT_NUMEL
from calm.hrm_text_158.native_full_stack.v4_live_band_sweep import (
    build_sweep_table_payload,
    run_band_sweep,
    write_sweep_table_json,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run V4-LIVE CPU band sweep.")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--numel", type=int, default=DEFAULT_VERDICT_NUMEL)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    result = run_band_sweep(numel=int(args.numel))
    out_path = write_sweep_table_json(run_root=args.run_root, result=result)
    payload = build_sweep_table_payload(result)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"sweep_table_path": str(out_path), "cpu_verdict": payload["cpu_verdict"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
