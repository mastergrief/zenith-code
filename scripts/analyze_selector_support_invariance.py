#!/usr/bin/env python3
"""Post-run selector_support_invariance_v0 analysis (compact artifacts only)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.selector_support_invariance_analysis import (
    run_selector_support_consensus_analysis,
    run_selector_support_invariance_analysis,
    sha256_file,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Read-only cross-seed selector support invariance analysis. "
            "Expects RUN_ROOT/{S44,S44_iso43,S43}/{on,off}/receipt.json."
        )
    )
    ap.add_argument(
        "run_root",
        type=Path,
        help="Literal run root with per-arm subdirectories",
    )
    ap.add_argument(
        "--primary-label",
        default="S44",
        help="Primary support arm label (default: S44).",
    )
    ap.add_argument(
        "--isolation-label",
        default="S44_iso43",
        help="Support-order isolation arm (default: S44_iso43).",
    )
    ap.add_argument(
        "--corroboration-label",
        default="S43",
        help="Corroboration replicate label (default: S43).",
    )
    ap.add_argument(
        "--consensus",
        action="store_true",
        help="Run K=3 consensus analysis (writes selector_support_consensus_summary.json).",
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
    if args.consensus:
        summary = run_selector_support_consensus_analysis(
            run_root,
            primary_label=args.primary_label,
            isolation_label=args.isolation_label,
            corroboration_label=args.corroboration_label,
        )
        analysis_dir = run_root / "analysis"
        analysis_dir.mkdir(parents=True, exist_ok=True)
        summary_path = analysis_dir / "selector_support_consensus_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        manifest = {
            "schema": "hrm_text_158_selector_support_consensus_manifest/v1",
            "run_root": str(run_root),
            "summary_path": str(summary_path),
            "summary_sha256": sha256_file(summary_path),
            "repo_head": _resolve_repo_head(args.repo_head),
            "primary_label": args.primary_label,
            "isolation_label": args.isolation_label,
            "corroboration_label": args.corroboration_label,
            "K": 3,
            "branch_precedence": summary.get("branch_precedence_receipt", {}).get("branch"),
            "invalid_data_routed": summary.get("invalid_data_routed"),
            "intersection_core_fraction": (
                summary.get("consensus_identity", {}).get("intersection_core_fraction")
            ),
        }
        manifest_path = analysis_dir / "selector_support_consensus_manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(manifest, indent=2))
        return 0
    summary = run_selector_support_invariance_analysis(
        run_root,
        primary_label=args.primary_label,
        isolation_label=args.isolation_label,
        corroboration_label=args.corroboration_label,
    )
    analysis_dir = run_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    summary_path = analysis_dir / "selector_support_invariance_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema": "hrm_text_158_selector_support_invariance_manifest/v0",
        "run_root": str(run_root),
        "summary_path": str(summary_path),
        "summary_sha256": sha256_file(summary_path),
        "repo_head": _resolve_repo_head(args.repo_head),
        "branch_precedence": summary.get("branch_precedence_receipt", {}).get("branch"),
    }
    manifest_path = analysis_dir / "selector_support_invariance_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
