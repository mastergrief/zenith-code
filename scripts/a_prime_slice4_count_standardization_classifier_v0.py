"""Rung-6 dual-key classifier thin orchestrator (STEP-2).

Loads one Rung-5 terminal receipt, extracts C1 counts, calls pure reducer,
exclusive dual-key mint. PLAN v6 ee9628cdcc45515dd8007de065960cae344b43f5ccaa600b3d8bafaa3066b900
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

from scripts.a_prime_slice4_count_standardization_reducer_v0 import (
    standardize_from_c1_raw,
)
from scripts.a_prime_slice4_count_standardization_runtime_source_contract_v0 import (
    FROZEN_RUNG5_TERMINAL_PIN,
    ORDERED_RUNTIME_PATHS,
    PLAN_REVISION_BINDING,
    check_rung5_terminal_pin,
    compare_expected_observed_sha,
    validate_runtime_source,
)
from scripts.a_prime_slice4_count_standardization_schema_v0 import (
    CLAIM_BOUNDARY_REQUIRED,
    COMPONENTS,
    SUPPORTS,
)

TERMINAL_RECEIPT_NAME = "terminal_receipt.json"
TERMINAL_MANIFEST_NAME = "terminal_manifest.json"
RECEIPT_SCHEMA = "a_prime_slice4_count_standardization_terminal_receipt/v0"
MANIFEST_SCHEMA = "a_prime_slice4_count_standardization_terminal_manifest/v0"
DECLARED_TOP_EMBEDDED = (
    "primary",
    "composite_terminal",
    "cells",
    "cell_labels",
    "kitagawa",
    "integer_margins",
    "successor",
    "claim_boundary",
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


_PUBLISHED_LABELS_OK = frozenset(
    {"E_MIXED", "E_TRANSIENT", "E_PERSISTENT", "MIXED", "TRANSIENT", "PERSISTENT"}
)


def extract_inputs_from_rung5_terminal(
    receipt: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, int]], dict[str, str]]:
    """Pull C1_profile.raw, C2 aggregates, published D2 labels.

    Fail-closed: every support MUST carry an explicit published aggregate D2
    label (C2.aggregate_d2_labels[s] or C2.raw[s].aggregate.label). No default
    fabrication (gate-1 bounce 1786009209108 — missing label must not become E_MIXED).
    """
    c1 = receipt.get("C1_profile") or receipt.get("classification", {}).get(
        "C1_profile"
    )
    c2 = receipt.get("C2_profile") or receipt.get("classification", {}).get(
        "C2_profile"
    )
    if not isinstance(c1, dict) or "raw" not in c1:
        raise ValueError("missing_C1_profile_raw")
    if not isinstance(c2, dict):
        raise ValueError("missing_C2_profile")
    raw = c1["raw"]
    aggregates: dict[str, dict[str, int]] = {}
    published: dict[str, str] = {}
    labels = c2.get("aggregate_d2_labels")
    if labels is None:
        labels = {}
    if not isinstance(labels, dict):
        raise ValueError("aggregate_d2_labels_not_dict")
    c2raw = c2.get("raw") or {}
    if not isinstance(c2raw, dict):
        c2raw = {}
    for s in SUPPORTS:
        # Explicit label only — never fabricate a default.
        lab = labels.get(s)
        if lab is None or lab == "":
            lab = (c2raw.get(s) or {}).get("aggregate", {}).get("label")
        if lab is None or lab == "":
            raise ValueError(f"missing_published_d2_label:{s}")
        if not isinstance(lab, str) or lab not in _PUBLISHED_LABELS_OK:
            raise ValueError(f"invalid_published_d2_label:{s}:{lab!r}")
        published[s] = lab if lab.startswith("E_") else f"E_{lab}"

        agg = (c2raw.get(s) or {}).get("aggregate") or {}
        if agg:
            aggregates[s] = {
                "N50": int(agg.get("|E50|_row_ids", agg.get("N50", 0))),
                "present_N20": int(
                    agg.get(
                        "present_at_package_N20_row_id_intersection",
                        agg.get("present_N20", 0),
                    )
                ),
                "absent_N20": int(
                    agg.get(
                        "absent_from_package_N20_row_id_difference",
                        agg.get("absent_N20", 0),
                    )
                ),
            }
        else:
            n = p = a = 0
            for c in COMPONENTS:
                cell = raw[s][c]
                if "N50" in cell:
                    n += int(cell["N50"])
                    p += int(cell["present_N20"])
                    a += int(cell["absent_N20"])
                else:
                    n += int(cell["|B50|_row_ids"])
                    p += int(cell["present_at_package_N20_row_id_intersection"])
                    a += int(cell["absent_from_package_N20_row_id_difference"])
            aggregates[s] = {"N50": n, "present_N20": p, "absent_N20": a}
    return dict(raw), aggregates, published


def _jsonable(obj: Any) -> Any:
    """Drop Fraction (and other non-JSON) objects from embedded classification."""
    from fractions import Fraction

    if isinstance(obj, Fraction):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def build_terminal_receipt(
    classification: Mapping[str, Any],
    *,
    run_root: Path,
    source_shas: Mapping[str, Any],
) -> dict[str, Any]:
    # No fabrication: missing/non-dict claim_boundary → ValueError (not default).
    boundary = classification.get("claim_boundary")
    if not isinstance(boundary, dict):
        raise ValueError("claim_boundary_missing")
    boundary = dict(boundary)
    for k, v in CLAIM_BOUNDARY_REQUIRED.items():
        if boundary.get(k) is not v:
            raise ValueError(f"claim_boundary_mismatch:{k}")
    # No primary fallback: composite_terminal required as non-empty str.
    branch = classification.get("composite_terminal")
    if not isinstance(branch, str) or not branch:
        raise ValueError("composite_terminal_missing")
    primary = classification.get("primary")
    if primary is None:
        raise ValueError("primary_missing")
    cls = _jsonable(dict(classification))
    if isinstance(cls, dict):
        # drop Fraction-only helper map; string tables remain
        kita_c = cls.get("kitagawa")
        if isinstance(kita_c, dict):
            kita_c.pop("tables_by_base_fraction", None)
        cls["source_shas"] = copy.deepcopy(dict(source_shas))
    kita = cls.get("kitagawa") if isinstance(cls, dict) else None
    return {
        "schema": RECEIPT_SCHEMA,
        "branch": branch,
        "primary": primary,
        "composite_terminal": branch,
        "cells": classification.get("cells"),
        "cell_labels": classification.get("cell_labels"),
        "kitagawa": kita,
        "integer_margins": classification.get("integer_margins"),
        "successor": classification.get("successor"),
        "claim_boundary": boundary,
        "run_root": str(run_root.resolve()),
        "source_shas": copy.deepcopy(dict(source_shas)),
        "classification": cls,
        "plan_revision_binding": PLAN_REVISION_BINDING,
        "terminal_authority": "manifest+marker",
        "synthetic": False,
    }


def build_terminal_manifest(
    run_root: Path, *, branch: str, outputs: Mapping[str, str]
) -> dict[str, Any]:
    return {
        "schema": MANIFEST_SCHEMA,
        "branch": branch,
        "run_root": str(run_root.resolve()),
        "outputs": dict(outputs),
        "terminal_authority": "manifest+marker",
        "synthetic": False,
        "plan_revision_binding": PLAN_REVISION_BINDING,
    }


def validate_candidate_receipt(
    receipt: Mapping[str, Any],
    *,
    classification: Mapping[str, Any],
    expected_run_root: Path,
) -> tuple[bool, str]:
    if receipt.get("schema") != RECEIPT_SCHEMA:
        return False, "receipt_schema"
    if receipt.get("run_root") != str(Path(expected_run_root).resolve()):
        return False, "run_root_mismatch"
    if receipt.get("synthetic") is not False:
        return False, "synthetic_not_false"
    if receipt.get("plan_revision_binding") != PLAN_REVISION_BINDING:
        return False, "plan_revision_binding"
    emb_root = receipt.get("classification")
    if not isinstance(emb_root, dict):
        return False, "classification_missing"
    # BIND_FAIL/TIE reduced field set (declared): primary, composite_terminal,
    # claim_boundary, successor — still require both-sides presence+equality.
    failish = classification.get("terminal_kind") in (
        "STANDARDIZATION_BIND_FAIL",
        "BOUNDARY_TIE",
    ) or str(classification.get("primary", "")).endswith("BIND_FAIL")
    reduced = frozenset(
        {"primary", "composite_terminal", "claim_boundary", "successor"}
    )
    required = reduced if failish else frozenset(DECLARED_TOP_EMBEDDED)
    for k in required:
        if k not in receipt:
            return False, f"top_missing:{k}"
        if k not in emb_root:
            return False, f"embedded_missing:{k}"
        if receipt.get(k) != emb_root.get(k):
            return False, f"top_embedded_mismatch:{k}"
    # Success-path authority: exact boundary + plan binding + source_shas equality.
    if receipt.get("claim_boundary") != CLAIM_BOUNDARY_REQUIRED:
        return False, "claim_boundary_not_required"
    if emb_root.get("plan_revision_binding") != PLAN_REVISION_BINDING:
        return False, "embedded_plan_revision_binding"
    top_ss = receipt.get("source_shas")
    emb_ss = emb_root.get("source_shas")
    if top_ss is None or emb_ss is None:
        return False, "source_shas_missing"
    if top_ss != emb_ss:
        return False, "source_shas_mismatch"
    return True, "ok"


def mint_exclusive_run_root(run_root: Path) -> tuple[bool, str]:
    run_root = Path(run_root)
    if run_root.exists():
        return False, f"run_root_exists:{run_root}"
    try:
        run_root.mkdir(parents=True, exist_ok=False)
    except OSError as e:
        return False, f"run_root_mkdir:{e}"
    return True, "ok"


def finalize_dual_key(
    run_root: Path,
    classification: Mapping[str, Any],
    *,
    source_shas: Mapping[str, Any],
) -> int:
    run_root = Path(run_root)
    ok, reason = mint_exclusive_run_root(run_root)
    if not ok:
        return markerless(reason)
    try:
        receipt = build_terminal_receipt(
            classification, run_root=run_root, source_shas=source_shas
        )
    except ValueError as e:
        return markerless(f"receipt_build:{e}")
    vok, vreason = validate_candidate_receipt(
        receipt, classification=classification, expected_run_root=run_root
    )
    if not vok:
        return markerless(vreason)
    write_json(run_root / TERMINAL_RECEIPT_NAME, receipt)
    branch = str(receipt["branch"])
    outputs = {TERMINAL_RECEIPT_NAME: sha256_file(run_root / TERMINAL_RECEIPT_NAME)}
    payload = build_terminal_manifest(run_root, branch=branch, outputs=outputs)
    tmp = run_root / f"{TERMINAL_MANIFEST_NAME}.tmp.{os.getpid()}"
    write_json(tmp, payload)
    try:
        os.replace(str(tmp), str(run_root / TERMINAL_MANIFEST_NAME))
    except OSError as e:
        return markerless(f"manifest_replace:{e}")
    print(f"PACKET_TERMINAL {branch}", flush=True)
    print("WRAPPER_RC 0", flush=True)
    return 0


def _preflight_runtime_source(
    *,
    manifest_path: Path,
    expected_sha: str,
) -> tuple[bool, str, dict[str, Any], dict[str, str], str]:
    if not _HEX64.match(expected_sha or ""):
        return False, "expected_manifest_sha_malformed", {}, {}, ""
    try:
        raw = manifest_path.read_bytes()
    except OSError as e:
        return False, f"manifest_read:{e}", {}, {}, ""
    observed = sha256_hex(raw)
    ok, reason = compare_expected_observed_sha(expected_sha, observed)
    if not ok:
        return False, reason, {}, {}, ""
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as e:
        return False, f"manifest_json:{e}", {}, {}, ""

    def reader(rel: str) -> bytes:
        return (_WORKTREE / rel).read_bytes()

    vok, vreason, observed_map, digest = validate_runtime_source(
        manifest_obj=obj,
        expected_manifest_sha256=expected_sha,
        observed_manifest_sha256=observed,
        read_bytes=reader,
    )
    if not vok or observed_map is None or digest is None:
        return False, vreason, {}, {}, ""
    return True, "ok", obj, observed_map, digest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Rung-6 count-standardization dual-key classifier"
    )
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--rung5-terminal-receipt", type=Path, required=True)
    ap.add_argument("--runtime-source-manifest", type=Path, required=True)
    ap.add_argument("--runtime-source-manifest-sha256", type=str, required=True)
    args = ap.parse_args(argv)

    ok, reason, _mobj, per_file, digest = _preflight_runtime_source(
        manifest_path=Path(args.runtime_source_manifest),
        expected_sha=str(args.runtime_source_manifest_sha256),
    )
    if not ok:
        return markerless(reason)

    rpath = Path(args.rung5_terminal_receipt)
    try:
        rbytes = rpath.read_bytes()
    except OSError as e:
        return markerless(f"rung5_read:{e}")
    rsha = sha256_hex(rbytes)
    # Accept iff (literal argv path OR resolve() path) matches pin AND sha matches.
    pok_lit, preason_lit = check_rung5_terminal_pin(
        path=str(args.rung5_terminal_receipt), sha256=rsha
    )
    pok_res, preason_res = check_rung5_terminal_pin(
        path=str(rpath.resolve()), sha256=rsha
    )
    if not (pok_lit or pok_res):
        return markerless(preason_lit if not pok_lit else preason_res)
    try:
        receipt = json.loads(rbytes.decode("utf-8"))
    except (UnicodeError, json.JSONDecodeError) as e:
        return markerless(f"rung5_json:{e}")

    try:
        c1_raw, aggregates, published = extract_inputs_from_rung5_terminal(receipt)
        classification = standardize_from_c1_raw(c1_raw, aggregates, published)
    except Exception as e:  # controlled surface: markerless
        return markerless(f"standardize:{e}")

    source_shas = {
        "rung5_terminal": {
            "path": FROZEN_RUNG5_TERMINAL_PIN["path"],
            "sha256": rsha,
        },
        "runtime_source": {
            "per_file": per_file,
            "ordered_concat_v0": digest,
            "manifest_sha256": str(args.runtime_source_manifest_sha256),
        },
        "plan_revision_binding": PLAN_REVISION_BINDING,
    }
    return finalize_dual_key(
        Path(args.run_root), classification, source_shas=source_shas
    )


if __name__ == "__main__":
    raise SystemExit(main())
