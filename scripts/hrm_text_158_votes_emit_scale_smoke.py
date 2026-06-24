#!/usr/bin/env python3
"""CPU scale-smoke for votes-emit sidecar overhead."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_c2p1_probe


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        raise ValueError("percentile requires at least one sample")
    ordered = sorted(float(value) for value in values)
    index = int(round((float(pct) / 100.0) * (len(ordered) - 1)))
    return float(ordered[max(0, min(index, len(ordered) - 1))])


def build_votes_emit_scale_smoke_receipt(
    *,
    steps: int,
    baseline_seconds: float,
    emit_enabled_seconds: float,
    manifest_path: Path,
    per_step_dir: Path,
    emit_timings_ms: list[float],
) -> dict[str, Any]:
    if len(emit_timings_ms) == 0:
        raise ValueError(
            "votes-emit scale-smoke requires at least one emit timing sample; "
            "got emit_sample_count=0"
        )
    step_files = sorted(per_step_dir.glob("*.json")) if per_step_dir.is_dir() else []
    bytes_per_step = (
        statistics.mean([path.stat().st_size for path in step_files])
        if step_files
        else 0.0
    )
    overhead_fraction = (
        (emit_enabled_seconds - baseline_seconds) / baseline_seconds
        if baseline_seconds > 0
        else 0.0
    )
    receipt: dict[str, Any] = {
        "schema_version": "hrm_text_158_votes_emit_scale_smoke/v0",
        "steps": int(steps),
        "baseline_seconds": float(baseline_seconds),
        "emit_enabled_seconds": float(emit_enabled_seconds),
        "overhead_fraction": float(overhead_fraction),
        "overhead_pass": bool(overhead_fraction < 0.05),
        "bytes_per_step_mean": float(bytes_per_step),
        "per_step_file_count": int(len(step_files)),
        "manifest_path": str(manifest_path),
        "manifest_exists": bool(manifest_path.is_file()),
        "emit_sample_count": int(len(emit_timings_ms)),
        "emit_p50_ms": _percentile(emit_timings_ms, 50.0),
        "emit_p95_ms": _percentile(emit_timings_ms, 95.0),
    }
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        receipt["manifest_sha256"] = manifest.get("manifest_sha256")
        receipt["step_count"] = manifest.get("step_count")
        receipt["manifest_emit_sample_count"] = manifest.get("emit_sample_count")
    return receipt


def run_votes_emit_scale_smoke(
    *,
    scratch_root: Path,
    parent: Path,
    steps: int = 10,
) -> dict[str, Any]:
    scratch_root = Path(scratch_root)
    scratch_root.mkdir(parents=True, exist_ok=True)
    baseline_root = scratch_root / "baseline"
    emit_root = scratch_root / "emit"
    baseline_root.mkdir(parents=True, exist_ok=True)
    emit_root.mkdir(parents=True, exist_ok=True)

    baseline_start = time.perf_counter()
    run_c2p1_probe(
        parent=Path(parent),
        parent_sha256=None,
        scratch_root=baseline_root,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=int(steps),
        batch_size=8,
        enabled=True,
        votes_emit_enabled=False,
    )
    baseline_seconds = time.perf_counter() - baseline_start

    emit_start = time.perf_counter()
    run_c2p1_probe(
        parent=Path(parent),
        parent_sha256=None,
        scratch_root=emit_root,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=int(steps),
        batch_size=8,
        enabled=True,
        votes_emit_enabled=True,
        votes_emit_root=emit_root,
    )
    emit_seconds = time.perf_counter() - emit_start

    manifest_path = emit_root / "votes_emit" / "v1" / "manifest.json"
    per_step_dir = emit_root / "votes_emit" / "v1" / "per_step"
    if not manifest_path.is_file():
        raise ValueError(f"votes-emit manifest missing at {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    emit_timings_ms = [float(value) for value in manifest.get("emit_timings_ms", [])]
    return build_votes_emit_scale_smoke_receipt(
        steps=int(steps),
        baseline_seconds=float(baseline_seconds),
        emit_enabled_seconds=float(emit_seconds),
        manifest_path=manifest_path,
        per_step_dir=per_step_dir,
        emit_timings_ms=emit_timings_ms,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Votes-emit CPU scale-smoke harness")
    parser.add_argument("--scratch-root", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args(argv)

    receipt = run_votes_emit_scale_smoke(
        scratch_root=Path(args.scratch_root),
        parent=Path(args.parent),
        steps=int(args.steps),
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0 if bool(receipt["overhead_pass"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
