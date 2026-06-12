#!/usr/bin/env python3
"""Thin orchestrator for paired selector-value identity/outcome receipt analysis."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.selector_value_analysis import (
    load_paired_receipts,
    run_full_analysis,
    run_identity_analysis,
    run_outcome_analysis,
    write_analysis_memo,
    write_run_manifest,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only paired selector-value analysis over on/off probe receipts. "
            "Writes new artifacts under RUN_ROOT/analysis/ only."
        )
    )
    ap.add_argument(
        "run_root",
        type=Path,
        help="Literal run root containing on/receipt.json and off/receipt.json",
    )
    ap.add_argument(
        "--mode",
        choices=("identity", "outcome", "full"),
        default="full",
        help="Analysis mode (default: full).",
    )
    ap.add_argument(
        "--repo-head",
        default=None,
        help="Optional repo HEAD sha for manifest provenance.",
    )
    return ap


def _resolve_repo_head(explicit: str | None) -> str | None:
    if explicit:
        return explicit
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            or None
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    run_root = args.run_root.resolve()
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    on_receipt = run_root / "on" / "receipt.json"
    off_receipt = run_root / "off" / "receipt.json"
    if not on_receipt.exists() or not off_receipt.exists():
        raise FileNotFoundError(
            f"Expected paired receipts at {on_receipt} and {off_receipt}"
        )

    on, off = load_paired_receipts(run_root)
    output_paths: list[Path] = []
    identity_summary: dict | None = None
    outcome_summary: dict | None = None

    if args.mode in ("identity", "full"):
        identity_summary = run_identity_analysis(on, off, include_overlap_band=True)
        identity_path = analysis_dir / "stage_c_identity_summary.json"
        identity_path.write_text(json.dumps(identity_summary, indent=2), encoding="utf-8")
        output_paths.append(identity_path)

    if args.mode in ("outcome", "full"):
        outcome_summary = run_outcome_analysis(on, off)
        outcome_path = analysis_dir / "stage_c_outcome_summary.json"
        outcome_path.write_text(json.dumps(outcome_summary, indent=2), encoding="utf-8")
        output_paths.append(outcome_path)

    memo_path = analysis_dir / "stage_c_outcome_memo.md"
    write_analysis_memo(
        memo_path,
        identity=identity_summary,
        outcome=outcome_summary,
    )
    output_paths.append(memo_path)

    manifest_path = analysis_dir / "run_manifest.json"
    write_run_manifest(
        manifest_path,
        run_root=run_root,
        mode=str(args.mode),
        argv=list(sys.argv if argv is None else [sys.argv[0], *argv]),
        repo_head=_resolve_repo_head(args.repo_head),
        script_path=Path(__file__).resolve(),
        on_receipt=on_receipt,
        off_receipt=off_receipt,
        output_paths=output_paths + [manifest_path],
    )
    output_paths.append(manifest_path)

    print(
        json.dumps(
            {
                "mode": args.mode,
                "identity_verdict": (identity_summary or {}).get("verdict"),
                "outcome_verdict": (outcome_summary or {}).get("verdict"),
                "output_paths": [str(path) for path in output_paths],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
