#!/usr/bin/env python3
"""Per-arm artifact transport for box-lane science chains (§C tooling)."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from calm.hrm_text_158.native_full_stack.box_lane import (
    EXIT_ARTIFACT_RSYNC_MISMATCH,
    EXIT_OK,
    build_artifact_transport_manifest,
    chain_roots,
    default_consensus_chain_artifacts,
    format_capture_complete_line,
    sync_chain_arm_artifacts,
)


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Box-lane per-arm artifact transport (§C).")
    ap.add_argument("--chain-id", required=True)
    ap.add_argument("--creditdir", default="/home/gabe/claw-code-creditdir/transient_fp_credit")
    ap.add_argument("--box", default="box")
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Transport manifest path (default: <local_chain_root>/box_artifact_transport.json)",
    )
    ap.add_argument(
        "--chain-log",
        type=Path,
        default=None,
        help="Append watcher log lines (capture_complete only).",
    )
    ap.add_argument("--code-currency-pass", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument(
        "--sync",
        action="store_true",
        help="Explicitly enable rsync/ssh to box. Default hashes local files only.",
    )
    ap.add_argument("--primary-label", default="S44")
    ap.add_argument("--isolation-label", default="S44_iso43")
    ap.add_argument("--corroboration-label", default="S43")
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


def read_code_currency_pass_from_preflight(chain_root: Path) -> bool:
    """Fail-closed: only True when preflight JSON explicitly has code_currency_pass==true."""
    path = chain_root / "prelaunch" / "box_code_currency_preflight.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return False
    if not isinstance(payload, dict):
        return False
    return payload.get("code_currency_pass") is True


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    local_chain_root, remote_chain_root = chain_roots(args.chain_id, creditdir=args.creditdir)
    output_path = args.output or (local_chain_root / "box_artifact_transport.json")
    artifacts = default_consensus_chain_artifacts(
        labels=(args.primary_label, args.isolation_label, args.corroboration_label),
    )
    sync_requested = bool(args.sync)
    mismatches: list[str] = []
    rows: list[dict] = []
    if sync_requested:
        mismatches, rows = sync_chain_arm_artifacts(
            local_chain_root=local_chain_root,
            remote_chain_root=str(remote_chain_root),
            box=args.box,
            artifacts=artifacts,
            rsync_runner=_default_rsync_runner,
            remote_sha_runner=_default_remote_sha_runner,
        )
    else:
        from calm.hrm_text_158.native_full_stack.box_lane import sha256_file

        for artifact in artifacts:
            rel = artifact.rel_path
            local_path = local_chain_root / rel
            if not local_path.exists():
                if artifact.optional:
                    rows.append(
                        {
                            "role": artifact.role,
                            "rel_path": rel,
                            "optional": True,
                            "missing": True,
                            "skipped": True,
                        },
                    )
                    continue
                mismatches.append(f"missing:{rel}")
                rows.append({"role": artifact.role, "rel_path": rel, "missing": True})
                continue
            producer_sha = sha256_file(local_path)
            rows.append(
                {
                    "role": artifact.role,
                    "rel_path": rel,
                    "producer_sha256": producer_sha,
                    "consumer_sha256": producer_sha,
                    "rsync_ok": True,
                },
            )

    manifest = build_artifact_transport_manifest(
        chain_id=args.chain_id,
        local_chain_root=local_chain_root,
        remote_chain_root=str(remote_chain_root),
        artifacts=rows,
        mismatches=mismatches,
        sync_requested=sync_requested,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    now = time.time()
    if args.chain_log is not None:
        code_currency_pass = read_code_currency_pass_from_preflight(local_chain_root)
        line = format_capture_complete_line(
            chain_id=args.chain_id,
            code_currency_pass=code_currency_pass,
            artifact_sha_verified=not mismatches,
            ts=now,
            seed=args.seed,
        )
        args.chain_log.parent.mkdir(parents=True, exist_ok=True)
        with args.chain_log.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")

    print(json.dumps({"output": str(output_path), "artifact_transport_pass": manifest["artifact_transport_pass"]}))
    if mismatches:
        return EXIT_ARTIFACT_RSYNC_MISMATCH
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
