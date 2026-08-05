"""Rung-3 residual classification classifier — CLI + dual-key (STEP-2).

classifier → reducer → schema. Exclusive run-root; six same-byte receipts.
PLAN v6 argv_template: --run-root + --package-receipt + --out-receipt.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice4_residual_classification_reducer_v0 import (
    NON_AUTHORITATIVE_KEYS,
    classification_core,
    classify_from_projections,
)
from scripts.a_prime_slice4_residual_classification_schema_v0 import (
    ARMS,
    HORIZONS,
    REQUIRED_CLAIM_BOUNDARY,
    SUPPORTS,
    is_exact_dict,
    is_exact_int,
    is_exact_list,
    is_exact_str,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice4_residual_classification_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice4_residual_classification_terminal_manifest/v0"
# Top-level receipt fields that must equal classification[field] (class-level bind).
DECLARED_TOP_EMBEDDED_FIELDS: tuple[str, ...] = (
    "identity_profile",
    "survivor_overlap_profile",
    "rescue_persistence_profile",
    "residual_bucket_profile",
    "successor",
    "composite_terminal",
)


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_hex(text.encode("utf-8"))


def extract_horizon_view(receipt: Mapping[str, Any], support: str) -> dict[str, Any]:
    """Build reducer horizon view from prior_audit.final_reports.<support>."""
    if not is_exact_dict(receipt):
        raise ValueError("receipt_not_dict")
    pa = receipt.get("prior_audit")
    if not is_exact_dict(pa):
        raise ValueError("prior_audit_missing")
    fr = pa.get("final_reports")
    if not is_exact_dict(fr) or support not in fr or not is_exact_dict(fr[support]):
        raise ValueError(f"final_reports_missing:{support}")
    rep = fr[support]
    batches = rep.get("batch_reports")
    if not is_exact_list(batches):
        raise ValueError(f"batch_reports_missing:{support}")
    row_ids: list[str] = []
    sample_hashes: list[str] = []
    source_buckets: list[str] = []
    for b in batches:
        if not is_exact_dict(b):
            raise ValueError("batch_not_dict")
        md = b.get("metadata")
        if not is_exact_dict(md):
            raise ValueError("metadata_missing")
        rids = md.get("row_ids")
        shs = md.get("sample_hashes")
        sbs = md.get("source_buckets")
        if not (is_exact_list(rids) and is_exact_list(shs) and is_exact_list(sbs)):
            raise ValueError("meta_lists_bad")
        if not (
            all(is_exact_str(x) for x in rids)
            and all(is_exact_str(x) for x in shs)
            and all(is_exact_str(x) for x in sbs)
        ):
            raise ValueError("meta_lists_types")
        row_ids.extend(rids)
        sample_hashes.extend(shs)
        source_buckets.extend(sbs)
    fails = rep.get("strict_failure_row_ids")
    if not is_exact_list(fails) or not all(is_exact_str(x) for x in fails):
        raise ValueError("strict_failure_row_ids_bad")
    audited = rep.get("support_rows_audited")
    if not is_exact_int(audited):
        raise ValueError("support_rows_audited_bad")
    return {
        "row_ids": row_ids,
        "sample_hashes": sample_hashes,
        "source_buckets": source_buckets,
        "strict_failure_row_ids": list(fails),
        "support_rows_audited": audited,
    }


def build_projections(
    package_objs: Mapping[int, Mapping[str, Any]],
    out_objs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    projections: dict[str, Any] = {}
    for support in SUPPORTS:
        projections[support] = {"package": {}, "out": {}}
        for h in HORIZONS:
            projections[support]["package"][h] = extract_horizon_view(
                package_objs[h], support
            )
            projections[support]["out"][h] = extract_horizon_view(out_objs[h], support)
    return projections


def build_terminal_receipt(
    classification: Mapping[str, Any],
    *,
    run_root: Path,
    source_shas: Mapping[str, str],
) -> dict[str, Any]:
    boundary = classification.get("claim_boundary")
    if not is_exact_dict(boundary):
        boundary = dict(REQUIRED_CLAIM_BOUNDARY)
    else:
        boundary = dict(boundary)
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        if boundary.get(k) is not v:
            raise ValueError(f"claim_boundary_mismatch:{k}")
    cls = dict(classification)
    cls["source_shas"] = dict(source_shas)
    branch = classification.get("composite_terminal")
    if not is_exact_str(branch) or not branch:
        raise ValueError("composite_terminal_missing")
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": branch,
        "composite_terminal": branch,
        "identity_profile": classification.get("identity_profile"),
        "survivor_overlap_profile": classification.get("survivor_overlap_profile"),
        "rescue_persistence_profile": classification.get("rescue_persistence_profile"),
        "residual_bucket_profile": classification.get("residual_bucket_profile"),
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
    canonical_snapshot: Mapping[str, Any],
    expected_run_root: Path,
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
    if receipt.get("composite_terminal") != branch:
        return False, "composite_terminal_ne_branch"
    expected_root = str(Path(expected_run_root).resolve())
    if receipt.get("run_root") != expected_root:
        return False, f"run_root_mismatch:{receipt.get('run_root')!r}!={expected_root!r}"
    cls = receipt.get("classification")
    if not is_exact_dict(cls):
        return False, "classification_missing"
    top_src = receipt.get("source_shas")
    if not is_exact_dict(top_src) or dict(top_src) != dict(source_shas):
        return False, "source_shas_mismatch"
    emb_src = cls.get("source_shas")
    if not is_exact_dict(emb_src) or dict(emb_src) != dict(top_src):
        return False, "embedded_source_shas"
    boundary = receipt.get("claim_boundary")
    if not is_exact_dict(boundary):
        return False, "claim_boundary_missing"
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        if boundary.get(k) is not v:
            return False, f"claim_boundary:{k}"
    emb_b = cls.get("claim_boundary")
    if not is_exact_dict(emb_b) or dict(emb_b) != dict(boundary):
        return False, "embedded_claim_boundary"
    # Declared top-level ↔ embedded classification binds (class cure).
    for field in DECLARED_TOP_EMBEDDED_FIELDS:
        if receipt.get(field) != cls.get(field):
            return False, f"top_ne_embedded:{field}"
    if cls.get("composite_terminal") != branch:
        return False, "embedded_composite_terminal"
    # Snapshot equality: classification_core of embedded == core of pre-build snapshot
    try:
        cand = classification_core(cls)
        snap = classification_core(canonical_snapshot)
    except Exception as e:
        return False, f"core:{e}"
    if cand != snap:
        return False, "core_snapshot_ne_embedded"
    # mechanical: every snapshot key (minus NON_AUTHORITATIVE) present
    for k in snap:
        if k not in NON_AUTHORITATIVE_KEYS and k not in cand:
            return False, f"core_missing:{k}"
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
    if candidate_sha256 is not None and sha256_file(final_path) != candidate_sha256:
        return False, "candidate_byte_mismatch"
    try:
        payload = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as e:
        return False, f"final_manifest_unparseable:{e}"
    if not is_exact_dict(payload):
        return False, f"manifest_payload_not_exact_dict:{type(payload).__name__}"
    if payload.get("schema") != MANIFEST_SCHEMA:
        return False, f"manifest_schema={payload.get('schema')!r}"
    if payload.get("synthetic") is not False:
        return False, "manifest_synthetic"
    expected_root = str(final_path.parent.resolve())
    if payload.get("run_root") != expected_root:
        return False, f"manifest_run_root_mismatch:{payload.get('run_root')!r}!={expected_root!r}"
    if payload.get("branch") != receipt_branch:
        return False, "branch_mismatch"
    if payload.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority_missing_or_wrong"
    outs = payload.get("outputs")
    if not is_exact_dict(outs):
        return False, f"manifest_outputs_not_exact_dict:{type(outs).__name__}"
    # Exact artifact-set equality (no subset, no extra) before path traversal.
    if dict(outs) != dict(expected_hashes):
        return False, "manifest_outputs_set_ne_expected"
    run_root = final_path.parent
    for rel, exp in outs.items():
        if not is_exact_str(rel) or not is_exact_str(exp):
            return False, f"manifest_outputs_entry_types:{rel!r}"
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
    inject_receipt_mutator: Any = None,
) -> int:
    run_root = Path(run_root)
    ok, reason = mint_exclusive_run_root(run_root)
    if not ok:
        print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    snap_src = dict(classification)
    snap_src["source_shas"] = dict(source_shas)
    canonical_snapshot = copy.deepcopy(snap_src)
    try:
        receipt = build_terminal_receipt(
            classification, run_root=run_root, source_shas=source_shas
        )
    except Exception as e:
        print(f"INCOMPLETE_FINALIZATION receipt_build:{e}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    if inject_receipt_mutator is not None:
        inject_receipt_mutator(receipt)
    vok, vreason = validate_candidate_receipt(
        receipt,
        source_shas=source_shas,
        canonical_snapshot=canonical_snapshot,
        expected_run_root=run_root,
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


def _parse_horizon_paths(
    items: list[str], *, label: str
) -> tuple[dict[int, Path] | None, str]:
    out: dict[int, Path] = {}
    for item in items:
        if "=" not in item:
            return None, f"malformed_{label}"
        n_s, path_s = item.split("=", 1)
        try:
            n = int(n_s)
        except ValueError:
            return None, f"malformed_horizon_n_{label}"
        if n not in HORIZONS or n in out:
            return None, f"bad_or_duplicate_horizon_{label}"
        path = Path(path_s)
        if not path.is_file():
            return None, f"missing_receipt:{path}"
        out[n] = path
    if set(out.keys()) != set(HORIZONS):
        return None, f"horizon_set_{label}"
    return out, "ok"


def load_receipts_same_byte(
    paths: Mapping[int, Path],
) -> tuple[dict[int, dict[str, Any]], dict[int, str]] | tuple[None, str]:
    objs: dict[int, dict[str, Any]] = {}
    shas: dict[int, str] = {}
    for n, path in paths.items():
        raw = path.read_bytes()
        digest = sha256_hex(raw)
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return None, f"unparseable:{path}:{e}"
        if not is_exact_dict(obj):
            return None, f"not_object:{path}"
        objs[n] = obj
        shas[n] = digest
    return objs, shas


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="A' slice-4 Rung-3 residual classification dual-key classifier"
    )
    ap.add_argument("--package-receipt", action="append", default=[])
    ap.add_argument("--out-receipt", action="append", default=[])
    ap.add_argument("--run-root", type=Path, required=True)
    args = ap.parse_args(argv)

    pkg_paths, reason = _parse_horizon_paths(args.package_receipt, label="package")
    if pkg_paths is None:
        print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    out_paths, reason = _parse_horizon_paths(args.out_receipt, label="out")
    if out_paths is None:
        print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2

    loaded_pkg = load_receipts_same_byte(pkg_paths)
    if loaded_pkg[0] is None:
        print(f"INCOMPLETE_FINALIZATION {loaded_pkg[1]}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    pkg_objs, pkg_shas = loaded_pkg  # type: ignore[misc]
    loaded_out = load_receipts_same_byte(out_paths)
    if loaded_out[0] is None:
        print(f"INCOMPLETE_FINALIZATION {loaded_out[1]}", flush=True)
        print("WRAPPER_RC 2", flush=True)
        return 2
    out_objs, out_shas = loaded_out  # type: ignore[misc]

    source_shas = {
        **{f"package/N{n}": pkg_shas[n] for n in HORIZONS},
        **{f"out/N{n}": out_shas[n] for n in HORIZONS},
    }
    try:
        projections = build_projections(pkg_objs, out_objs)
        classification = classify_from_projections(projections)
    except Exception as e:
        # instrument path: wrap as INSTRUMENT_OR_BIND_FAIL
        classification = {
            "identity_profile": "INSTRUMENT_OR_BIND_FAIL",
            "identity_reasons": [f"extract_or_classify:{e}"],
            "identity_raw": {},
            "survivor_overlap_profile": None,
            "rescue_persistence_profile": None,
            "residual_bucket_profile": None,
            "counter_loss_table": None,
            "composite_terminal": "INSTRUMENT_OR_BIND_FAIL",
            "successor": "instrument repair only; no science successor",
            "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
            "instrument_fail": True,
        }
    return finalize_dual_key(
        Path(args.run_root), classification, source_shas=source_shas
    )


if __name__ == "__main__":
    raise SystemExit(main())
