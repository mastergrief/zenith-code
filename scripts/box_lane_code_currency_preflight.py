#!/usr/bin/env python3
"""Fail-closed box code-currency preflight for science-chain packets (§A)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from calm.hrm_text_158.native_full_stack.box_lane import (
    ANALYZER_PINNED_FILES,
    EXIT_CODE_CURRENCY_MISMATCH,
    EXIT_OK,
    PinnedFile,
    build_code_currency_manifest,
    chain_roots,
    check_pinned_paths_clean,
    hash_pinned_files,
    load_pinned_manifest,
    probe_rsync_version,
    run_git,
    sync_pinned_files,
    verify_head_triple,
    verify_pinned_sha_expectations,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Box-lane code-currency preflight (§A).")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--box", default="box")
    ap.add_argument("--remote-repo", default="/home/gabe/claw-code-hrm-158")
    ap.add_argument("--chain-id", required=True)
    ap.add_argument("--creditdir", default="/home/gabe/claw-code-creditdir/transient_fp_credit")
    ap.add_argument("--head-expected", required=True)
    ap.add_argument("--pinned-manifest", type=Path, default=None)
    ap.add_argument("--include-analyzer-surfaces", action="store_true")
    ap.add_argument(
        "--include-phase3-obmalloc-surfaces",
        action="store_true",
        help="Include Phase-3 obmalloc attribution script in pinned manifest.",
    )
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Explicitly enable ssh/rsync to box. Default is hash-only (no network).",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Manifest output path (default: <local_chain_root>/box_code_currency_preflight.json)",
    )
    ap.add_argument("--skip-fetch", action="store_true", help="Local-only mode: skip git fetch and remote FETCH_HEAD currency check.")
    return ap


def _default_rsync_runner(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=True, capture_output=True, text=True)


def _default_remote_sha_runner(box: str, remote_rel: str) -> str:
    proc = subprocess.run(
        ["ssh", box, "sha256sum", remote_rel],
        check=True,
        capture_output=True,
        text=True,
    )
    return proc.stdout.split()[0]


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    repo_root = args.repo_root.resolve()
    local_chain_root, remote_chain_root = chain_roots(args.chain_id, creditdir=args.creditdir)
    local_chain_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output or (local_chain_root / "box_code_currency_preflight.json")

    if not args.skip_fetch:
        subprocess.run(
            ["git", "fetch", "origin", "refs/heads/feature/hrm-text-1.58"],
            cwd=repo_root,
            check=True,
        )
    head_now = run_git(repo_root, "rev-parse", "HEAD")
    fetch_head = run_git(repo_root, "rev-parse", "FETCH_HEAD")

    mismatches = verify_head_triple(
        head_now=head_now,
        fetch_head=fetch_head,
        head_expected=args.head_expected,
        require_fetch_head=not args.skip_fetch,
    )
    remote_currency_check = "skipped_local_only" if args.skip_fetch else "enforced"

    pinned = load_pinned_manifest(args.pinned_manifest)
    if args.include_analyzer_surfaces:
        pinned.extend(PinnedFile(role, rel) for role, rel in ANALYZER_PINNED_FILES)
    if args.include_phase3_obmalloc_surfaces:
        from calm.hrm_text_158.native_full_stack.box_lane import (
            PHASE3_OBMALLOC_SURFACE_PINNED_FILES,
        )

        pinned.extend(
            PinnedFile(role, rel) for role, rel in PHASE3_OBMALLOC_SURFACE_PINNED_FILES
        )
    pinned_rows = hash_pinned_files(repo_root, pinned)
    mismatches.extend(verify_pinned_sha_expectations(pinned_rows))
    mismatches.extend(check_pinned_paths_clean(repo_root, pinned, git_runner=run_git))

    sync_requested = bool(args.sync and not args.dry_run)
    rsync_version = probe_rsync_version() if sync_requested else None
    if sync_requested and not mismatches:
        sync_mismatches, pinned_rows = sync_pinned_files(
            repo_root=repo_root,
            remote_repo=args.remote_repo,
            box=args.box,
            pinned_rows=pinned_rows,
            rsync_runner=_default_rsync_runner,
            remote_sha_runner=_default_remote_sha_runner,
        )
        mismatches.extend(sync_mismatches)

    manifest = build_code_currency_manifest(
        chain_id=args.chain_id,
        head_expected=args.head_expected,
        head_now=head_now,
        fetch_head=fetch_head,
        pinned_rows=pinned_rows,
        dry_run=args.dry_run,
        sync_requested=sync_requested,
        local_chain_root=local_chain_root,
        remote_chain_root=remote_chain_root,
        remote_repo_root=args.remote_repo,
        mismatches=mismatches,
        rsync_version=rsync_version,
        remote_currency_check=remote_currency_check,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output_path), "code_currency_pass": manifest["code_currency_pass"]}))
    if mismatches:
        return EXIT_CODE_CURRENCY_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
