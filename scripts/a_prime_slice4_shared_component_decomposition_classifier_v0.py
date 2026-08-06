"""Rung-5 dual-key classifier thin orchestrator. PLAN v4 a2e7420aeaee715ed181b46f4f1de4d0b93deb47a29da6e3bded0fd431e48421
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping

from scripts.a_prime_slice4_residual_classification_classifier_v0 import (
    extract_horizon_view,
)
from scripts.a_prime_slice4_shared_component_decomposition_reducer_v0 import (
    NON_AUTHORITATIVE_KEYS,
    component_core,
    component_from_projections,
)
from scripts.a_prime_slice4_shared_component_decomposition_schema_v0 import (
    HORIZONS,
    REQUIRED_CLAIM_BOUNDARY,
    SUPPORTS,
    is_exact_dict,
    is_exact_str,
)
from scripts.a_prime_slice4_shared_component_runtime_source_contract_v0 import (
    ALGORITHM,
    MANIFEST_SCHEMA_ID,
    ORDERED_RUNTIME_PATHS,
    check_frozen_input_pins,
    compare_expected_observed_sha,
    validate_runtime_source,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice4_shared_component_decomposition_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice4_shared_component_decomposition_terminal_manifest/v0"
DECLARED_TOP_EMBEDDED_FIELDS: tuple[str, ...] = (
    "identity_profile",
    "authority_profile",
    "C1_profile",
    "C2_profile",
    "A1_secondary",
    "B_report_only",
    "successor",
    "composite_terminal",
)
_HEX64 = re.compile(r"^[0-9a-f]{64}$")
_WORKTREE = Path(__file__).resolve().parents[1]

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def sha256_file(path: Path) -> str:
    return sha256_hex(path.read_bytes())

def write_json(path: Path, payload: Mapping[str, Any]) -> str:
    text = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8")
    return sha256_hex(text.encode("utf-8"))

def markerless(reason: str, rc: int = 2) -> int:
    print(f"INCOMPLETE_FINALIZATION {reason}", flush=True)
    print(f"WRAPPER_RC {rc}", flush=True)
    return rc

def build_projections(
    package_objs: Mapping[int, Mapping[str, Any]],
    out_objs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for support in SUPPORTS:
        out[support] = {
            "package": {h: extract_horizon_view(package_objs[h], support) for h in HORIZONS},
            "out": {h: extract_horizon_view(out_objs[h], support) for h in HORIZONS},
        }
    return out

def build_terminal_receipt(
    classification: Mapping[str, Any],
    *,
    run_root: Path,
    source_shas: Mapping[str, Any],
) -> dict[str, Any]:
    boundary = classification.get("claim_boundary")
    boundary = dict(boundary) if is_exact_dict(boundary) else dict(REQUIRED_CLAIM_BOUNDARY)
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        if boundary.get(k) is not v:
            raise ValueError(f"claim_boundary_mismatch:{k}")
    cls = dict(classification)
    cls["source_shas"] = copy.deepcopy(dict(source_shas))
    branch = classification.get("composite_terminal")
    if not is_exact_str(branch) or not branch:
        raise ValueError("composite_terminal_missing")
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": branch,
        "composite_terminal": branch,
        "identity_profile": classification.get("identity_profile"),
        "authority_profile": classification.get("authority_profile"),
        "C1_profile": classification.get("C1_profile"),
        "C2_profile": classification.get("C2_profile"),
        "A1_secondary": classification.get("A1_secondary"),
        "B_report_only": classification.get("B_report_only"),
        "successor": classification.get("successor"),
        "run_root": str(run_root.resolve()),
        "source_shas": copy.deepcopy(dict(source_shas)),
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
    source_shas: Mapping[str, Any],
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
    if receipt.get("run_root") != str(Path(expected_run_root).resolve()):
        return False, f"run_root_mismatch:{receipt.get('run_root')!r}"
    cls = receipt.get("classification")
    if not is_exact_dict(cls):
        return False, "classification_missing"
    top_src = receipt.get("source_shas")
    if not is_exact_dict(top_src) or top_src != source_shas:
        return False, "source_shas_mismatch"
    emb_src = cls.get("source_shas")
    if not is_exact_dict(emb_src) or emb_src != top_src:
        return False, "embedded_source_shas"
    exp = top_src.get("runtime_source_manifest_sha256_expected")
    obs = top_src.get("runtime_source_manifest_sha256_observed")
    eq = top_src.get("runtime_source_manifest_sha256_equal")
    if not (is_exact_str(exp) and is_exact_str(obs) and type(eq) is bool):
        return False, "runtime_source_manifest_sha_fields"
    if eq is not True or exp != obs:
        return False, "runtime_source_manifest_sha_not_equal"
    rs = top_src.get("runtime_source")
    if not is_exact_dict(rs) or rs.get("algorithm") != ALGORITHM:
        return False, "runtime_source_section"
    if rs.get("manifest_schema_id") != MANIFEST_SCHEMA_ID:
        return False, "runtime_source_schema_id"
    boundary = receipt.get("claim_boundary")
    if not is_exact_dict(boundary):
        return False, "claim_boundary_missing"
    for k, v in REQUIRED_CLAIM_BOUNDARY.items():
        if boundary.get(k) is not v:
            return False, f"claim_boundary:{k}"
    emb_b = cls.get("claim_boundary")
    if not is_exact_dict(emb_b) or dict(emb_b) != dict(boundary):
        return False, "embedded_claim_boundary"
    for field in DECLARED_TOP_EMBEDDED_FIELDS:
        if receipt.get(field) != cls.get(field):
            return False, f"top_ne_embedded:{field}"
    if cls.get("composite_terminal") != branch:
        return False, "embedded_composite_terminal"
    try:
        cand = component_core(cls)
        snap = component_core(canonical_snapshot)
    except Exception as e:
        return False, f"core:{e}"
    if cand != snap:
        return False, "core_snapshot_ne_embedded"
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
    if payload.get("run_root") != str(final_path.parent.resolve()):
        return False, "manifest_run_root_mismatch"
    if payload.get("branch") != receipt_branch:
        return False, "branch_mismatch"
    if payload.get("terminal_authority") != "manifest+marker":
        return False, "terminal_authority_missing_or_wrong"
    outs = payload.get("outputs")
    if not is_exact_dict(outs):
        return False, f"manifest_outputs_not_exact_dict:{type(outs).__name__}"
    if dict(outs) != dict(expected_hashes):  # whole-map
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
    except (FileExistsError, OSError) as e:
        return False, f"run_root_mkdir:{e}"
    return True, "ok"

def finalize_dual_key(
    run_root: Path,
    classification: Mapping[str, Any],
    *,
    source_shas: Mapping[str, Any],
    inject_receipt_mutator: Any = None,
) -> int:
    run_root = Path(run_root)
    ok, reason = mint_exclusive_run_root(run_root)
    if not ok:
        return markerless(reason, 2)
    snap_src = dict(classification)
    snap_src["source_shas"] = copy.deepcopy(dict(source_shas))
    canonical_snapshot = copy.deepcopy(snap_src)
    try:
        receipt = build_terminal_receipt(
            classification, run_root=run_root, source_shas=source_shas
        )
    except Exception as e:
        return markerless(f"receipt_build:{e}", 2)
    if inject_receipt_mutator is not None:
        inject_receipt_mutator(receipt)
    vok, vreason = validate_candidate_receipt(
        receipt,
        source_shas=source_shas,
        canonical_snapshot=canonical_snapshot,
        expected_run_root=run_root,
    )
    if not vok:
        return markerless(f"candidate_invalid:{vreason}", 2)
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
        return markerless(f"publish_failed:{e}", 2)
    vok, vreason = verify_published_manifest(
        run_root / TERMINAL_MANIFEST_NAME,
        receipt_branch=branch,
        expected_hashes=outputs,
        candidate_sha256=candidate_sha,
    )
    if not vok:
        return markerless(vreason, 4)
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
        if type(n) is not int or n not in HORIZONS or n in out:
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
        try:
            obj = json.loads(raw.decode("utf-8"))
        except Exception as e:
            return None, f"unparseable:{path}:{e}"
        if not is_exact_dict(obj):
            return None, f"not_object:{path}"
        objs[n], shas[n] = obj, sha256_hex(raw)
    return objs, shas

def _load_terminal(path: Path, *, label: str) -> tuple[dict[str, Any] | None, str, str]:
    if not path.is_file():
        return None, "", f"missing_{label}:{path}"
    raw = path.read_bytes()
    sha = sha256_hex(raw)
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return None, sha, f"{label}_unparseable:{e}"
    if not is_exact_dict(obj):
        return None, sha, f"{label}_not_object"
    return obj, sha, "ok"

def _preflight_runtime_source(
    manifest_path: Path, expected_sha: str
) -> tuple[dict[str, Any] | None, str]:
    if type(expected_sha) is not str or not _HEX64.match(expected_sha):
        return None, "expected_sha_malformed"
    if not manifest_path.is_file():
        return None, f"manifest_missing:{manifest_path}"
    raw = manifest_path.read_bytes()
    observed = sha256_hex(raw)
    ok, reason = compare_expected_observed_sha(expected_sha, observed)
    if not ok:
        return None, reason
    try:
        obj = json.loads(raw.decode("utf-8"))
    except Exception as e:
        return None, f"manifest_unparseable:{e}"
    clf_path = Path(__file__).resolve()

    def reader(rel: str) -> bytes:
        if rel.endswith("shared_component_decomposition_classifier_v0.py"):
            return clf_path.read_bytes()
        return (_WORKTREE / rel).read_bytes()

    vok, vreason, observed_map, digest = validate_runtime_source(
        manifest_obj=obj,
        expected_manifest_sha256=expected_sha,
        observed_manifest_sha256=observed,
        worktree_root_or_readers=reader,
    )
    if not vok or observed_map is None or digest is None:
        return None, vreason
    return {
        "expected": expected_sha,
        "observed": observed,
        "equal": True,
        "per_file": observed_map,
        "runtime_source_digest": digest,
        "algorithm": ALGORITHM,
        "manifest_schema_id": MANIFEST_SCHEMA_ID,
    }, "ok"

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Rung-5 shared-component dual-key classifier")
    ap.add_argument("--package-receipt", action="append", default=[])
    ap.add_argument("--out-receipt", action="append", default=[])
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--rung3-terminal-receipt", type=Path, required=True)
    ap.add_argument("--rung4-terminal-receipt", type=Path, required=True)
    ap.add_argument("--runtime-source-manifest", type=Path, required=True)
    ap.add_argument("--runtime-source-manifest-sha256", type=str, required=True)
    args = ap.parse_args(argv)

    rs, reason = _preflight_runtime_source(
        Path(args.runtime_source_manifest), args.runtime_source_manifest_sha256
    )
    if rs is None:
        return markerless(reason, 2)

    pkg_paths, reason = _parse_horizon_paths(args.package_receipt, label="package")
    if pkg_paths is None:
        return markerless(reason, 2)
    out_paths, reason = _parse_horizon_paths(args.out_receipt, label="out")
    if out_paths is None:
        return markerless(reason, 2)

    loaded_pkg = load_receipts_same_byte(pkg_paths)
    if loaded_pkg[0] is None:
        return markerless(str(loaded_pkg[1]), 2)
    pkg_objs, pkg_shas = loaded_pkg  # type: ignore[misc]
    loaded_out = load_receipts_same_byte(out_paths)
    if loaded_out[0] is None:
        return markerless(str(loaded_out[1]), 2)
    out_objs, out_shas = loaded_out  # type: ignore[misc]

    r3_path, r4_path = Path(args.rung3_terminal_receipt), Path(args.rung4_terminal_receipt)
    r3_obj, r3_sha, r3_reason = _load_terminal(r3_path, label="rung3")
    if r3_obj is None:
        return markerless(r3_reason, 2)
    r4_obj, r4_sha, r4_reason = _load_terminal(r4_path, label="rung4")
    if r4_obj is None:
        return markerless(r4_reason, 2)

    pok, preason = check_frozen_input_pins(
        package_paths=pkg_paths,
        out_paths=out_paths,
        rung3_path=r3_path,
        rung4_path=r4_path,
        package_shas=pkg_shas,
        out_shas=out_shas,
        rung3_sha=r3_sha,
        rung4_sha=r4_sha,
    )
    if not pok:
        return markerless(preason, 2)

    source_shas: dict[str, Any] = {
        **{f"package/N{n}": pkg_shas[n] for n in HORIZONS},
        **{f"out/N{n}": out_shas[n] for n in HORIZONS},
        "rung3_terminal": r3_sha,
        "rung4_terminal": r4_sha,
        "runtime_source_manifest_sha256_expected": rs["expected"],
        "runtime_source_manifest_sha256_observed": rs["observed"],
        "runtime_source_manifest_sha256_equal": True,
        "runtime_source": {
            "per_file": dict(rs["per_file"]),
            "runtime_source_digest": rs["runtime_source_digest"],
            "algorithm": rs["algorithm"],
            "manifest_schema_id": rs["manifest_schema_id"],
        },
    }
    try:
        projections = build_projections(pkg_objs, out_objs)
        classification = component_from_projections(
            projections, rung3_terminal=r3_obj, rung4_terminal=r4_obj
        )
    except Exception as e:
        msg = f"extract_or_classify:{e}"
        classification = {
            "identity_profile": "INSTRUMENT_OR_BIND_FAIL",
            "authority_profile": None,
            "identity_reasons": [msg],
            "terminal_reasons": [msg],
            "terminal_kind": "INSTRUMENT_OR_BIND_FAIL",
            "identity_raw": {},
            "C1_profile": None, "C2_profile": None,
            "A1_secondary": None, "B_report_only": None,
            "composite_terminal": "INSTRUMENT_OR_BIND_FAIL",
            "successor": "instrument repair only; no science successor",
            "claim_boundary": dict(REQUIRED_CLAIM_BOUNDARY),
            "instrument_fail": True,
        }
    return finalize_dual_key(Path(args.run_root), classification, source_shas=source_shas)

if __name__ == "__main__":
    raise SystemExit(main())
