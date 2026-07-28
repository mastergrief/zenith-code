#!/usr/bin/env python3
"""Test collection census for sparse live carrier production landing."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

CPU_PATH = Path("calm/llm_computer/tests/test_hrm_text_158_sparse_live_carrier_production_landing_cpu.py")
GPU_PATH = Path("calm/llm_computer/tests/test_hrm_text_158_sparse_live_carrier_production_landing_gpu_live.py")

REQUIRED_GPU_NODES = [
    "test_gpu_live_b1_default_fused_only_emits_phase_events",
    "test_gpu_live_b2_roundtrip_default_fused_only",
    "test_gpu_live_b3_landing_wrapper_default_fused_only",
    "test_gpu_live_oracle_on_events_equal_parity",
]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _test_names(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    return sorted(
        n.name
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name.startswith("test_")
    )


def build_census() -> dict[str, Any]:
    cpu = _test_names(CPU_PATH)
    gpu = _test_names(GPU_PATH)
    return {
        "schema_version": "sparse_live_carrier_test_collection_census_v2",
        "cpu_module": str(CPU_PATH),
        "gpu_module": str(GPU_PATH),
        "cpu_nodes": cpu,
        "gpu_nodes": gpu,
        "required_gpu_nodes": list(REQUIRED_GPU_NODES),
        "cpu_sha256": _sha(CPU_PATH),
        "gpu_sha256": _sha(GPU_PATH),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mint", action="store_true")
    ap.add_argument("--assert-cpu", action="store_true")
    ap.add_argument("--assert-gpu", action="store_true")
    ap.add_argument("--census", required=True)
    args = ap.parse_args()
    census_path = Path(args.census)
    if args.mint:
        if census_path.exists():
            raise SystemExit(f"census exists O_EXCL: {census_path}")
        payload = build_census()
        missing = [n for n in REQUIRED_GPU_NODES if n not in payload["gpu_nodes"]]
        if missing:
            raise SystemExit(f"gpu nodes missing required: {missing}")
        data = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
        fd = os.open(str(census_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
        print(json.dumps({"ok": True, "minted": str(census_path), "sha256": _sha(census_path)}))
        return 0

    if not census_path.exists():
        raise SystemExit(f"census missing: {census_path}")
    doc = json.loads(census_path.read_text())
    live = build_census()
    if args.assert_cpu:
        if doc.get("cpu_nodes") != live["cpu_nodes"] or doc.get("cpu_sha256") != live["cpu_sha256"]:
            raise SystemExit("cpu census mismatch")
        print(json.dumps({"ok": True, "assert": "cpu", "n": len(live["cpu_nodes"])}))
        return 0
    if args.assert_gpu:
        if doc.get("gpu_nodes") != live["gpu_nodes"] or doc.get("gpu_sha256") != live["gpu_sha256"]:
            raise SystemExit("gpu census mismatch")
        for n in REQUIRED_GPU_NODES:
            if n not in doc.get("gpu_nodes", []):
                raise SystemExit(f"required gpu node missing: {n}")
        print(json.dumps({"ok": True, "assert": "gpu", "n": len(live["gpu_nodes"])}))
        return 0
    raise SystemExit("specify --mint or --assert-cpu/--assert-gpu")


if __name__ == "__main__":
    raise SystemExit(main())
