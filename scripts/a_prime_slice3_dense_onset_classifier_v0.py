"""A′ slice-3 dense onset/shape classifier harness — CLI + dual-key finalization.

Pure classification lives in scripts.a_prime_slice3_onset_reducer_v0.
Plan authority: A_prime_slice3_dense_collapse_onset_PLAN_v5.json
(sha 7ba78320eab8f0d5cc92b5af86a6d158ee2e473d5185547d95aeba178d43d567).

No torch/GPU. Dual-key finalization follows scripts/a_prime_slice1_fidelity_manifest.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.a_prime_slice3_onset_reducer_v0 import (
    CLASS_PRIORITY,
    ELIGIBLE_MODULE_DEFAULT,
    HORIZONS,
    classify_suite,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice3_onset_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice3_onset_terminal_manifest/v0"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_bytes(text.encode("utf-8"))


def build_terminal_receipt(
    classification: Mapping[str, Any],
    *,
    run_root: Path,
    source_shas: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": classification["branch"],
        "run_root": str(run_root.resolve()),
        "source_shas": dict(source_shas),
        "classification": dict(classification),
        "terminal_authority": "manifest+marker",
        "synthetic": bool(classification.get("details", {}).get("synthetic", False)),
    }


def build_terminal_manifest(
    run_root: Path,
    *,
    branch: str,
    outputs: Mapping[str, str],
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "branch": branch,
        "terminal_authority": "manifest+marker",
        "run_root": str(run_root.resolve()),
        "outputs": dict(outputs),
        "expected_status_set": [],
        "failing_status": None,
        "synthetic": False,
    }


def write_manifest_candidate(run_root: Path, payload: Mapping[str, Any]) -> Path:
    tmp = run_root / f"{TERMINAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    return tmp


def publish_manifest_atomic(temp_path: Path, final_path: Path) -> None:
    os.replace(str(temp_path), str(final_path))


def verify_published_manifest(
    final_path: Path,
    *,
    receipt_branch: str,
    expected_hashes: Mapping[str, str],
    candidate_sha256: str | None = None,
) -> tuple[bool, str]:
    if not final_path.is_file():
        return False, "final_manifest_absent"
    if candidate_sha256 is not None:
        final_sha = sha256_file(final_path)
        if final_sha != candidate_sha256:
            return False, f"candidate_byte_mismatch:{final_sha}!={candidate_sha256}"
    try:
        payload = load_json(final_path)
    except Exception as e:
        return False, f"final_manifest_unparseable:{e}"
    if payload.get("branch") != receipt_branch:
        return False, f"branch {payload.get('branch')!r}!={receipt_branch!r}"
    if payload.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority_missing_or_wrong"
    outs = payload.get("outputs") or {}
    for rel, exp in expected_hashes.items():
        if outs.get(rel) != exp:
            return False, f"hash_mismatch:{rel}"
    run_root = final_path.parent
    for rel, exp in outs.items():
        p = run_root / rel
        if not p.is_file():
            return False, f"missing_output:{rel}"
        if sha256_file(p) != exp:
            return False, f"stale_output_hash:{rel}"
    return True, "ok"


def finalize_dual_key(
    run_root: Path,
    classification: Mapping[str, Any],
    *,
    source_shas: Mapping[str, str],
    inject_candidate_branch: str | None = None,
    inject_postpub_fail: bool = False,
) -> int:
    """Candidate validate → atomic publish → re-read verify → PACKET_TERMINAL.

    Returns 0 on success; 2 state-P; 4 state-Q INCOMPLETE_FINALIZATION.
    """
    run_root = Path(run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    receipt = build_terminal_receipt(
        classification, run_root=run_root, source_shas=source_shas
    )
    write_json(run_root / TERMINAL_RECEIPT_NAME, receipt)
    branch = receipt["branch"]
    if branch not in CLASS_PRIORITY:
        print(
            f"INCOMPLETE_FINALIZATION branch_not_in_priority:{branch}",
            file=sys.stderr,
            flush=True,
        )
        return 2

    outputs = {
        TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME),
    }
    for rel, h in source_shas.items():
        p = run_root / rel
        if p.is_file():
            outputs[rel] = sha256_file(p)

    cand_branch = inject_candidate_branch if inject_candidate_branch is not None else branch
    payload = build_terminal_manifest(run_root, branch=cand_branch, outputs=outputs)
    if payload.get("branch") != branch and inject_candidate_branch is None:
        print("INCOMPLETE_FINALIZATION candidate_branch_ne_receipt", file=sys.stderr, flush=True)
        return 2
    if inject_candidate_branch is not None and payload.get("branch") != branch:
        print("INCOMPLETE_FINALIZATION candidate_branch_ne_receipt", file=sys.stderr, flush=True)
        return 2

    if receipt.get("branch") != branch:
        print("INCOMPLETE_FINALIZATION receipt_branch_mismatch", file=sys.stderr, flush=True)
        return 2

    tmp = write_manifest_candidate(run_root, payload)
    candidate_sha = sha256_file(tmp)
    try:
        publish_manifest_atomic(tmp, run_root / TERMINAL_MANIFEST_NAME)
    except OSError as e:
        print(f"INCOMPLETE_FINALIZATION publish_failed:{e}", file=sys.stderr, flush=True)
        return 2

    if inject_postpub_fail:
        print(
            f"INCOMPLETE_FINALIZATION postpub_inject branch={branch}",
            file=sys.stderr,
            flush=True,
        )
        return 4

    vok, vreason = verify_published_manifest(
        run_root / TERMINAL_MANIFEST_NAME,
        receipt_branch=branch,
        expected_hashes=outputs,
        candidate_sha256=candidate_sha,
    )
    if not vok:
        print(f"INCOMPLETE_FINALIZATION {vreason}", file=sys.stderr, flush=True)
        return 4

    print(f"PACKET_TERMINAL {branch}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="A′ slice-3 dense onset/shape classifier")
    ap.add_argument(
        "--horizon-receipt",
        action="append",
        default=[],
        help="N=/path/to/receipt.json (repeat for N in 1,5,10,20,35,50)",
    )
    ap.add_argument("--run-root", type=Path, required=True)
    # --skip-prefix deliberately NOT exposed: prefix gate is mandatory on the CLI path.
    ap.add_argument("--eligible-module", default=ELIGIBLE_MODULE_DEFAULT)
    args = ap.parse_args(argv)
    if not args.horizon_receipt:
        print("missing --horizon-receipt", file=sys.stderr)
        return 2
    receipts: dict[int, dict[str, Any]] = {}
    source_shas: dict[str, str] = {}
    for item in args.horizon_receipt:
        if "=" not in item:
            print(f"malformed --horizon-receipt (expected N=path): {item!r}", file=sys.stderr)
            return 2
        n_s, path_s = item.split("=", 1)
        try:
            n = int(n_s)
        except ValueError:
            print(f"malformed horizon N: {n_s!r}", file=sys.stderr)
            return 2
        # C4: unknown key fail-closed
        if n not in HORIZONS:
            print(
                f"unknown horizon N={n}; required exact set {list(HORIZONS)}",
                file=sys.stderr,
            )
            return 2
        # C4: duplicate N fail-closed (no last-write-wins)
        if n in receipts:
            print(f"duplicate horizon N={n}", file=sys.stderr)
            return 2
        path = Path(path_s)
        if not path.is_file():
            print(f"missing receipt file: {path}", file=sys.stderr)
            return 2
        try:
            receipts[n] = load_json(path)
        except (OSError, json.JSONDecodeError) as e:
            print(f"malformed receipt JSON {path}: {e}", file=sys.stderr)
            return 2
        source_shas[f"input/N{n}"] = sha256_file(path)
    # C4: require exact set equality before classification
    if set(receipts.keys()) != set(HORIZONS):
        print(
            f"horizon key set {sorted(receipts.keys())} != required {list(HORIZONS)}",
            file=sys.stderr,
        )
        return 2
    classification = classify_suite(
        receipts,
        skip_prefix=False,
        eligible_module=str(args.eligible_module),
    )
    return finalize_dual_key(
        Path(args.run_root), classification, source_shas=source_shas
    )


if __name__ == "__main__":
    raise SystemExit(main())
