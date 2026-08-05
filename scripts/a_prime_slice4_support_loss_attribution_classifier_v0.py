"""A′ slice-4 Rung-0 support-loss attribution classifier — CLI + dual-key finalization.

Pure classification: scripts.a_prime_slice4_support_loss_attribution_reducer_v0
Plan: A_prime_slice4_cause_localization_PLAN_v2.json
  sha bdefb0180d26dc2a65926edd43c4ed8280ce608ad7b3d88e6d264887f3f3e295

Cycle-2: frozen terminal sha always enforced on CLI; pure candidate validator
before publish; exclusive run-root mint (fail if exists/nonempty).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice4_support_loss_attribution_reducer_v0 import (
    CLIFF_VALUES,
    ENDPOINT_VALUES,
    HORIZONS,
    START_SURVIVOR_DENOMINATORS,
    bind_and_extract,
    classify_from_counts,
    sha256_hex,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice4_support_loss_attribution_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice4_support_loss_attribution_terminal_manifest/v0"
# Required top-level + embedded claim_boundary for attribution branches (cycle-6).
REQUIRED_ATTRIBUTION_CLAIM_BOUNDARY: dict[str, bool] = {
    "attribution_only": True,
    "pre_cause": True,
    "pre_carrier": True,
    "absolute_share_not_branch_input": True,
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_hex(text.encode("utf-8"))


def build_terminal_receipt(
    classification: Mapping[str, Any],
    *,
    run_root: Path,
    source_shas: Mapping[str, str],
) -> dict[str, Any]:
    # Prefer reducer-emitted boundary; fall back to required constant for attribution.
    boundary = classification.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        boundary = dict(REQUIRED_ATTRIBUTION_CLAIM_BOUNDARY)
    else:
        boundary = dict(boundary)
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": classification["branch"],
        "endpoint_profile": classification.get("endpoint_profile"),
        "cliff_profile": classification.get("cliff_profile"),
        # run_root is path label only (non-authoritative for science; not in core)
        "run_root": str(run_root.resolve()),
        "source_shas": dict(source_shas),
        "classification": dict(classification),
        "terminal_authority": "manifest+marker",
        "synthetic": False,
        "claim_boundary": boundary,
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
        "synthetic": False,
    }


def classification_core(cls: Mapping[str, Any]) -> dict[str, Any]:
    """Exact core projection over a classification dict (cycle-5).

    Includes: branch, axes, survivor denoms, support rows, all six counts,
    endpoint dict, both complete cliff dicts (every field), claim_boundary.
    No field lists for partial compare — callers use dict equality.
    """
    counts_raw = cls.get("counts") or {}
    counts: dict[str, dict[str, int]] = {}
    for n in HORIZONS:
        key = str(n)
        if key not in counts_raw:
            raise ValueError(f"counts_missing_N{n}")
        entry = counts_raw[key]
        counts[key] = {
            "L0b": int(entry["L0b"]),
            "math_a0": int(entry["math_a0"]),
        }
    cliffs = cls.get("cliffs")
    if not isinstance(cliffs, list) or len(cliffs) != 2:
        n = len(cliffs) if isinstance(cliffs, list) else type(cliffs).__name__
        raise ValueError(f"cliffs_len_ne_2:{n}")
    # deep-copy cliffs as plain dicts (every emitted field)
    cliffs_core = [dict(c) for c in cliffs]
    endpoint = cls.get("endpoint")
    if not isinstance(endpoint, Mapping):
        raise ValueError("endpoint_missing")
    boundary = cls.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        raise ValueError("claim_boundary_missing")
    return {
        "branch": cls.get("branch"),
        "endpoint_profile": cls.get("endpoint_profile"),
        "cliff_profile": cls.get("cliff_profile"),
        "survivor_denominators": dict(cls.get("survivor_denominators") or {}),
        "support_rows_expected": dict(cls.get("support_rows_expected") or {}),
        "counts": counts,
        "endpoint": dict(endpoint),
        "cliffs": cliffs_core,
        "claim_boundary": dict(boundary),
    }


def validate_candidate_receipt(
    receipt: Mapping[str, Any],
    *,
    source_shas: Mapping[str, str],
) -> tuple[bool, str]:
    """Pure pre-publish validator. Fail-closed via exact core-projection equality."""
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return False, f"schema={receipt.get('schema')!r}"
    if receipt.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority"
    if receipt.get("synthetic") is not False:
        return False, "synthetic"
    branch = receipt.get("branch")
    if not isinstance(branch, str) or not branch:
        return False, "branch_missing"
    cls = receipt.get("classification") or {}
    if branch == "INSTRUMENT_OR_BIND_FAIL":
        if receipt.get("endpoint_profile") is not None or receipt.get("cliff_profile") is not None:
            return False, "instrument_has_axes"
        if not cls.get("instrument_fail"):
            return False, "instrument_flag"
        return True, "ok"

    ep = receipt.get("endpoint_profile")
    cp = receipt.get("cliff_profile")
    if ep not in ENDPOINT_VALUES:
        return False, f"endpoint_profile={ep!r}"
    if cp not in CLIFF_VALUES:
        return False, f"cliff_profile={cp!r}"
    if branch != f"{ep}__{cp}":
        return False, f"composite_mismatch:{branch}!={ep}__{cp}"

    # top-level must match embedded classification axes before recompute
    if cls.get("branch") != branch:
        return False, f"embedded_branch_mismatch:{cls.get('branch')!r}!={branch!r}"
    if cls.get("endpoint_profile") != ep:
        return False, f"embedded_endpoint_mismatch:{cls.get('endpoint_profile')!r}"
    if cls.get("cliff_profile") != cp:
        return False, f"embedded_cliff_mismatch:{cls.get('cliff_profile')!r}"

    src = receipt.get("source_shas") or {}
    if not isinstance(src, Mapping):
        return False, "source_shas_not_mapping"
    for n in HORIZONS:
        key = f"input/N{n}"
        if key not in src:
            return False, f"source_shas_missing:{key}"
        if source_shas.get(key) is not None and src.get(key) != source_shas.get(key):
            return False, f"source_shas_ne_bind:{key}"

    # build candidate core (requires all six counts + 2 cliffs + boundary)
    try:
        cand_core = classification_core(cls)
    except Exception as e:
        return False, f"candidate_core:{e}"

    # recompute from candidate counts (integer table only)
    try:
        count_table = {
            int(k): {"L0b": int(v["L0b"]), "math_a0": int(v["math_a0"])}
            for k, v in cand_core["counts"].items()
        }
    except Exception as e:
        return False, f"counts_unparseable:{e}"
    recomputed = classify_from_counts(
        count_table, survivor_denoms=START_SURVIVOR_DENOMINATORS
    )
    try:
        re_core = classification_core(recomputed)
    except Exception as e:
        return False, f"recomputed_core:{e}"

    if cand_core != re_core:
        return False, "core_projection_mismatch"

    # Cycle-6: WHOLE published receipt denominator — top-level claim_boundary
    # must exist and equal core(candidate) == core(recomputed) == required constant.
    if "claim_boundary" not in receipt:
        return False, "top_claim_boundary_missing"
    top_boundary = receipt.get("claim_boundary")
    if not isinstance(top_boundary, Mapping):
        return False, "top_claim_boundary_not_mapping"
    top_boundary_d = dict(top_boundary)
    if top_boundary_d != cand_core["claim_boundary"]:
        return False, "top_claim_boundary_ne_candidate_core"
    if top_boundary_d != re_core["claim_boundary"]:
        return False, "top_claim_boundary_ne_recomputed_core"
    if top_boundary_d != REQUIRED_ATTRIBUTION_CLAIM_BOUNDARY:
        return False, "top_claim_boundary_required_values"

    # Full-receipt cross-level checklist (explicit; no third-round hole):
    # - schema: checked above (exact RECEIPT_SCHEMA)
    # - terminal_authority: checked above (== manifest+marker)
    # - synthetic: checked above (=== false)
    # - branch / endpoint_profile / cliff_profile: top-level vs embedded vs recompute
    # - claim_boundary: top-level vs core(candidate) vs core(recomputed) vs required
    # - source_shas: complete + bind equality checked above
    # - classification core: exact dict equality vs recompute
    # - run_root: NON-AUTHORITATIVE path label only (not validated against science core)
    return True, "ok"


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
        payload = json.loads(final_path.read_text(encoding="utf-8"))
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


def mint_exclusive_run_root(run_root: Path) -> tuple[bool, str]:
    """Create run_root solely via mkdir(exist_ok=False). Fail on ANY pre-existing path."""
    run_root = Path(run_root)
    if run_root.exists():
        return False, f"run_root_exists:{run_root}"
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False, f"run_root_race:{run_root}"
    except OSError as e:
        return False, f"run_root_mkdir:{e}"
    return True, "ok"


def finalize_dual_key(
    run_root: Path,
    classification: Mapping[str, Any],
    *,
    source_shas: Mapping[str, str],
    inject_postpub_fail: bool = False,
    inject_receipt_mutator: Any = None,
) -> int:
    """Exclusive mint → candidate validate → atomic publish → verify → PACKET_TERMINAL.

    Exclusive run-root mint is unconditional (no caller bypass).
    inject_receipt_mutator: test-only fail-closed path (can force INCOMPLETE only).
    """
    run_root = Path(run_root)
    ok, reason = mint_exclusive_run_root(run_root)
    if not ok:
        print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    receipt = build_terminal_receipt(
        classification, run_root=run_root, source_shas=source_shas
    )
    if inject_receipt_mutator is not None:
        inject_receipt_mutator(receipt)
    vok, vreason = validate_candidate_receipt(receipt, source_shas=source_shas)
    if not vok:
        print(f"INCOMPLETE_FINALIZATION candidate_invalid:{vreason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    write_json(run_root / TERMINAL_RECEIPT_NAME, receipt)
    branch = str(receipt["branch"])

    outputs = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    payload = build_terminal_manifest(run_root, branch=branch, outputs=outputs)
    if payload.get("branch") != branch:
        print("INCOMPLETE_FINALIZATION candidate_branch_ne_receipt", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    tmp = run_root / f"{TERMINAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    tmp.write_text(text, encoding="utf-8")
    candidate_sha = sha256_file(tmp)
    try:
        os.replace(str(tmp), str(run_root / TERMINAL_MANIFEST_NAME))
    except OSError as e:
        print(f"INCOMPLETE_FINALIZATION publish_failed:{e}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    if inject_postpub_fail:
        print(f"INCOMPLETE_FINALIZATION postpub_inject branch={branch}", flush=True)
        print("WRAPPER_RC 4", flush=True)
        return 4

    vok, vreason = verify_published_manifest(
        run_root / TERMINAL_MANIFEST_NAME,
        receipt_branch=branch,
        expected_hashes=outputs,
        candidate_sha256=candidate_sha,
    )
    if not vok:
        print(f"INCOMPLETE_FINALIZATION {vreason}", flush=True)
        print("WRAPPER_RC 4", flush=True)
        return 4

    print(f"PACKET_TERMINAL {branch}", flush=True)
    print("WRAPPER_RC 0", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="A′ slice-4 Rung-0 support-loss surface attribution classifier"
    )
    ap.add_argument(
        "--terminal-receipt",
        type=Path,
        required=True,
        help="slice-3 terminal_receipt.json (frozen authority)",
    )
    ap.add_argument(
        "--horizon-receipt",
        action="append",
        default=[],
        help="N=/path/to/receipt.json (repeat for N in 1,5,10,20,35,50)",
    )
    ap.add_argument("--run-root", type=Path, required=True)
    # frozen terminal sha always enforced; no CLI bypass flag (cycle-2 cure 1)
    args = ap.parse_args(argv)

    if not args.horizon_receipt:
        print("INCOMPLETE_FINALIZATION missing_horizon_receipt", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    tpath = Path(args.terminal_receipt)
    if not tpath.is_file():
        print(f"INCOMPLETE_FINALIZATION missing_terminal:{tpath}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    traw = tpath.read_bytes()
    try:
        terminal = json.loads(traw.decode("utf-8"))
    except Exception as e:
        print(f"INCOMPLETE_FINALIZATION terminal_parse:{e}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    tsha = sha256_hex(traw)

    rbytes: dict[int, bytes] = {}
    seen: set[int] = set()
    for item in args.horizon_receipt:
        if "=" not in item:
            print(f"INCOMPLETE_FINALIZATION malformed_horizon_receipt:{item!r}", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        n_s, path_s = item.split("=", 1)
        try:
            n = int(n_s)
        except ValueError:
            print(f"INCOMPLETE_FINALIZATION malformed_horizon_n:{n_s!r}", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        if n not in HORIZONS:
            print(f"INCOMPLETE_FINALIZATION unknown_horizon:{n}", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        if n in seen:
            print(f"INCOMPLETE_FINALIZATION duplicate_horizon:{n}", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        seen.add(n)
        path = Path(path_s)
        if not path.is_file():
            print(f"INCOMPLETE_FINALIZATION missing_receipt:{path}", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        rbytes[n] = path.read_bytes()

    if set(rbytes.keys()) != set(HORIZONS):
        print(
            f"INCOMPLETE_FINALIZATION horizon_set:{sorted(rbytes.keys())}",
            flush=True,
        )
        print("WRAPPER_RC 2", flush=True)
        return 2

    # frozen terminal sha ALWAYS enforced on CLI path
    classification = bind_and_extract(
        terminal=terminal,
        terminal_sha256=tsha,
        receipt_bytes_by_n=rbytes,
        require_frozen_terminal_sha=True,
    )
    source_shas = dict(classification.get("source_shas") or {})
    return finalize_dual_key(
        Path(args.run_root), classification, source_shas=source_shas
    )


if __name__ == "__main__":
    raise SystemExit(main())
