"""A' slice-4 Rung-2 protection-package classifier — CLI + dual-key (cycle-5).

Dependency: classifier -> reducer -> schema; schema imports nothing local.
Mandatory authority envelope for attribution AND instrument terminals.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice4_protection_package_reducer_v0 import (
    EFFECT_VALUES,
    FROZEN_OUT_TERMINAL_SHA256,
    HORIZONS,
    REQUIRED_CLAIM_BOUNDARY,
    REQUIRED_OUT_AUTHORITY,
    SUPPORT_VALUES,
    bind_and_classify_package,
    check_package_binding,
    classification_core,
    classify_from_counts,
    is_exact_bool,
    is_exact_list,
    is_exact_str,
    sha256_hex,
)
from scripts.a_prime_slice4_protection_package_schema_v0 import (
    validate_claim_boundary_envelope,
    validate_out_authority_envelope,
    validate_source_shas_envelope,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice4_protection_package_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice4_protection_package_terminal_manifest/v0"


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
    boundary = classification.get("claim_boundary")
    if not isinstance(boundary, Mapping):
        boundary = dict(REQUIRED_CLAIM_BOUNDARY)
    else:
        boundary = dict(boundary)
    cls = dict(classification)
    cls["source_shas"] = dict(source_shas)
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": classification["branch"],
        "package_effect_profile": classification.get("package_effect_profile"),
        "support_response_profile": classification.get("support_response_profile"),
        "successor": classification.get("successor"),
        "run_root": str(run_root.resolve()),
        "source_shas": dict(source_shas),
        "classification": cls,
        "terminal_authority": "manifest+marker",
        "synthetic": False,
        "claim_boundary": boundary,
    }


def build_terminal_manifest(
    run_root: Path, *, branch: str, outputs: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "branch": branch,
        "terminal_authority": "manifest+marker",
        "run_root": str(run_root.resolve()),
        "outputs": dict(outputs),
        "synthetic": False,
    }


def validate_candidate_receipt(
    receipt: Mapping[str, Any],
    *,
    source_shas: Mapping[str, str],
    canonical_snapshot: Mapping[str, Any] | None = None,
) -> tuple[bool, str]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return False, f"schema={receipt.get('schema')!r}"
    if receipt.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority"
    if receipt.get("synthetic") is not False:
        return False, "synthetic"
    branch = receipt.get("branch")
    if not is_exact_str(branch) or not branch:
        return False, "branch_missing"
    cls = receipt.get("classification")
    if not isinstance(cls, Mapping):
        return False, "classification_missing"

    # MANDATORY authority — absence fails for BOTH instrument and attribution
    top_src = receipt.get("source_shas")
    ok, reason = validate_source_shas_envelope(
        top_src, bind_shas=source_shas, require_out=True
    )
    if not ok:
        return False, reason
    emb_src = cls.get("source_shas")
    if not isinstance(emb_src, Mapping):
        return False, "embedded_source_shas_missing"
    if dict(emb_src) != dict(top_src):
        return False, "embedded_source_shas_ne_top"
    ok, reason = validate_out_authority_envelope(cls.get("out_authority"))
    if not ok:
        return False, reason
    ok, reason = validate_claim_boundary_envelope(receipt, cls)
    if not ok:
        return False, reason

    top_succ = receipt.get("successor")
    emb_succ = cls.get("successor")
    if top_succ != emb_succ:
        return False, f"successor_top_ne_embedded:{top_succ!r}!={emb_succ!r}"

    if branch == "INSTRUMENT_OR_BIND_FAIL":
        # Snapshot is MANDATORY for instrument — absence fails (class cure).
        if canonical_snapshot is None:
            return False, "instrument_snapshot_missing"
        if not isinstance(canonical_snapshot, Mapping):
            return False, "instrument_snapshot_not_mapping"
        if receipt.get("package_effect_profile") is not None:
            return False, "instrument_has_effect"
        if receipt.get("support_response_profile") is not None:
            return False, "instrument_has_support"
        if not is_exact_bool(cls.get("instrument_fail")) or cls.get("instrument_fail") is not True:
            return False, "instrument_flag"
        if cls.get("branch") != "INSTRUMENT_OR_BIND_FAIL":
            return False, "instrument_embedded_branch"
        if top_succ != "instrument repair only":
            return False, f"instrument_successor={top_succ!r}"
        # WHOLE-PAYLOAD exact equality vs pre-build snapshot (types included).
        # Closes reasons/package_binding/counts and every future diagnostic field as a class.
        def _deep_eq(a, b):
            if isinstance(a, Mapping) and isinstance(b, Mapping):
                if set(a.keys()) != set(b.keys()):
                    return False
                return all(_deep_eq(a[k], b[k]) for k in a)
            if isinstance(a, list) and isinstance(b, list):
                return len(a) == len(b) and all(_deep_eq(x, y) for x, y in zip(a, b))
            return type(a) is type(b) and a == b
        if not _deep_eq(cls, canonical_snapshot):
            return False, "instrument_snapshot_ne_embedded"
        return True, "ok"

    eff = receipt.get("package_effect_profile")
    sup = receipt.get("support_response_profile")
    if eff not in EFFECT_VALUES:
        return False, f"package_effect_profile={eff!r}"
    if sup not in SUPPORT_VALUES:
        return False, f"support_response_profile={sup!r}"
    if branch != f"{eff}__{sup}":
        return False, f"composite_mismatch:{branch}!={eff}__{sup}"
    if cls.get("branch") != branch:
        return False, f"embedded_branch_mismatch:{cls.get('branch')!r}"
    if cls.get("package_effect_profile") != eff:
        return False, "embedded_effect_mismatch"
    if cls.get("support_response_profile") != sup:
        return False, "embedded_support_mismatch"

    counts_raw = cls.get("counts")
    if not isinstance(counts_raw, Mapping):
        return False, "counts_missing"
    try:
        recomputed = classify_from_counts(counts_raw)
    except Exception as e:
        return False, f"recompute_fail:{e}"
    binding = cls.get("package_binding")
    if not isinstance(binding, Mapping):
        return False, "package_binding_missing"
    bind_reasons = check_package_binding(binding)
    if bind_reasons:
        return False, f"package_binding_invalid:{';'.join(bind_reasons)}"
    recomputed["package_binding"] = dict(binding)
    recomputed["out_authority"] = dict(cls["out_authority"])
    recomputed["source_shas"] = dict(top_src)
    recomputed["instrument_fail"] = False
    recomputed["reasons"] = []

    try:
        cand_core = classification_core(cls)
        re_core = classification_core(recomputed)
    except Exception as e:
        return False, f"core:{e}"
    if cand_core != re_core:
        return False, "core_projection_mismatch"
    if top_succ != recomputed.get("successor"):
        return False, f"successor_ne_recomputed:{top_succ!r}"
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
        if sha256_file(final_path) != candidate_sha256:
            return False, "candidate_byte_mismatch"
    try:
        payload = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"final_manifest_unparseable:{e}"
    if payload.get("branch") != receipt_branch:
        return False, "branch_mismatch"
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
    run_root = Path(run_root)
    ok, reason = mint_exclusive_run_root(run_root)
    if not ok:
        print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    # Immutable snapshot of bind_and_classify_package output BEFORE receipt build/mutator.
    # Instrument validation requires whole-payload equality against this snapshot.
    # Ensure source_shas embedded matches what build_terminal_receipt will set.
    snap_src = dict(classification)
    snap_src["source_shas"] = dict(source_shas)
    canonical_snapshot = copy.deepcopy(snap_src)

    receipt = build_terminal_receipt(
        classification, run_root=run_root, source_shas=source_shas
    )
    if inject_receipt_mutator is not None:
        inject_receipt_mutator(receipt)
    vok, vreason = validate_candidate_receipt(
        receipt, source_shas=source_shas, canonical_snapshot=canonical_snapshot
    )
    if not vok:
        print(f"INCOMPLETE_FINALIZATION candidate_invalid:{vreason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    write_json(run_root / TERMINAL_RECEIPT_NAME, receipt)
    branch = str(receipt["branch"])
    outputs = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    payload = build_terminal_manifest(run_root, branch=branch, outputs=outputs)
    tmp = run_root / f"{TERMINAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
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
        description="A' slice-4 Rung-2 composite protection-package sensitivity classifier"
    )
    ap.add_argument("--package-receipt", action="append", default=[])
    ap.add_argument("--out-terminal-receipt", type=Path, required=True)
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args(argv)

    if not args.package_receipt:
        print("INCOMPLETE_FINALIZATION missing_package_receipt", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    rbytes: dict[int, bytes] = {}
    seen: set[int] = set()
    for item in args.package_receipt:
        if "=" not in item:
            print("INCOMPLETE_FINALIZATION malformed_package_receipt", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        n_s, path_s = item.split("=", 1)
        try:
            n = int(n_s)
        except ValueError:
            print("INCOMPLETE_FINALIZATION malformed_horizon_n", flush=True)
            print("WRAPPER_RC 2", flush=True)
            return 2
        if n not in HORIZONS or n in seen:
            print("INCOMPLETE_FINALIZATION bad_or_duplicate_horizon", flush=True)
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
        print("INCOMPLETE_FINALIZATION horizon_set", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    tpath = Path(args.out_terminal_receipt)
    if not tpath.is_file():
        print("INCOMPLETE_FINALIZATION missing_out_terminal", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    try:
        traw = tpath.read_bytes()
        out_sha = sha256_hex(traw)
        out_terminal = json.loads(traw.decode("utf-8"))
        if not isinstance(out_terminal, dict):
            raise ValueError("out_terminal_not_object")
    except Exception as e:
        print(f"INCOMPLETE_FINALIZATION out_terminal_unreadable:{e}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    classification = bind_and_classify_package(
        package_receipt_bytes_by_n=rbytes,
        out_terminal=out_terminal,
        out_terminal_sha256=out_sha,
        require_frozen_out_terminal_sha=True,
    )
    source_shas = dict(classification.get("source_shas") or {})
    return finalize_dual_key(
        Path(args.run_root), classification, source_shas=source_shas
    )


if __name__ == "__main__":
    raise SystemExit(main())
