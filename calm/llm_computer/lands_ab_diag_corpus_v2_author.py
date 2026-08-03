"""Deterministic v2 fixture producer for LANDS-AB dry-exec diag corpus (plan v58 §7.3).

Pure build_v2_payloads + O_EXCL mint (§7.1c helper). Entry:
  python -m calm.llm_computer.lands_ab_diag_corpus_v2_author
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import sys
from pathlib import Path

CANON_ROOT_DEFAULT = "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"
HEAD_B_MANIFEST = (
    "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
    "LANDS_AB_science_source_manifest_HEAD_B.json"
)
FIXTURE_B = "FAITHFUL_BASE_PACKET_v8_dry_fixture (_packet_for_dry over HEAD_B manifest)"
V2_ROWS_SCHEMA = "LANDS_AB_dry_exec_diag_corpus_rows/v2"
V2_BASELINE_SCHEMA = "LANDS_AB_dry_exec_diag_corpus_baseline/v2"
V2_BASELINE_NAME = "BASELINE_TOOL_bc0d7f56.json"
V2_BASELINE_HEAD = "bc0d7f56"
TOOL_REL = "scripts/lands_ab_packet_dry_exec.py"
HARNESS_REL = "calm/llm_computer/tests/test_hrm_text_158_lands_ab_science_source_manifest.py"
V1_FIXTURE_DIR = "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v1"
V2_FIXTURE_DIR = "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v2"
PIN_V1_PARENT_ROWS_SHA256 = "d61530c0dfeadc67d610ac6cc9b043f3c1d0bb2a490e356594f0047fe1923b89"
PIN_V1_PARENT_BASELINE_SHA256 = "a18a99f98f36051cdd739e7b43091184d8149a0f4ae1ce305583615f1b81460b"
SOURCE_ARTIFACT_SHA = "b011fb91dc61593ef27081c50fdbb75962ff7c237eb818d23daa97df5bdaaa8c"
SOURCE_ARTIFACT = "A0_DIAGNOSTIC_CLASS_CORPUS_v4"
PRODUCER = "calm.llm_computer.lands_ab_diag_corpus_v2_author"
CANONICAL = dict(indent=2, sort_keys=True)
TOOL_SHA_PIN = "bc0d7f56e67cfbea58d7775f017d04ef01f2253f1c15f1b3867d67128fc52d02"
HARNESS_SHA_PIN = "b37736e4059f034039a3ea57155a040a547160a0969c510466d4951e3054f2a8"

V1_METHOD = (
    "direct field projection from the frozen artifact's rows[]; "
    "NO reconstruction from predecessor increments; EXCEPT the enumerated "
    "normalization(s) recorded in top-level `normalizations`"
)
assert len(V1_METHOD) == 183

DERIVED_M06 = (
    "CLI manifest path 'artifacts/acc_entropy/optimizer_credit_state_sparse_vote_"
    "authority_LANDS_AB_science_source_manifest_HEAD_B.json' != "
    "packet.science_source_manifest_path 'artifacts/acc_entropy/other.json'"
)


def canonical_json(obj) -> str:
    return json.dumps(obj, **CANONICAL) + "\n"


def oexcl_write_bytes(path: Path, data: bytes, mode: int = 0o444) -> str:
    """Create-only authority write. Refuse if path exists. fsync + chmod 0444. Return sha256.
    C1: NEVER mkdir parents. Auto-mkdir would create RECEIPT_DIR and make plain-mkdir hit EEXIST."""
    path = Path(path)
    if path.exists():
        raise SystemExit(f"EXISTING_TARGET_REFUSE path={path}")
    if not path.parent.is_dir():
        raise SystemExit(f"OEXCL_PARENT_MISSING path={path} parent={path.parent}")
    fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        view = memoryview(data)
        w = 0
        while w < len(data):
            n = os.write(fd, view[w:])
            if n <= 0:
                raise OSError("short")
            w += n
        os.fsync(fd)
    finally:
        os.close(fd)
    os.chmod(path, mode)
    return hashlib.sha256(data).hexdigest()


def _cpath_is_canon_rooted_string(s, canon_root=CANON_ROOT_DEFAULT):
    if not isinstance(s, str) or not s:
        return False
    cr = canon_root.rstrip("/")
    return s == cr or s.startswith(cr + "/")


def _cpath_relative_remainder(s, canon_root=CANON_ROOT_DEFAULT):
    if not _cpath_is_canon_rooted_string(s, canon_root):
        return None
    cr = canon_root.rstrip("/")
    if s == cr:
        return "."
    return s[len(cr) + 1 :]


def _rewrite_cpath_value(val, canon_root=CANON_ROOT_DEFAULT):
    if isinstance(val, str) and _cpath_is_canon_rooted_string(val, canon_root):
        rel = _cpath_relative_remainder(val, canon_root)
        return {"$typed_repo_rel": rel}
    if isinstance(val, list):
        return [_rewrite_cpath_value(v, canon_root) for v in val]
    if isinstance(val, dict):
        return {k: _rewrite_cpath_value(v, canon_root) for k, v in val.items()}
    return val


def _rewrite_mutation_spec(spec, canon_root=CANON_ROOT_DEFAULT):
    if not isinstance(spec, dict):
        return spec
    out = copy.deepcopy(spec)
    ops = out.get("ops")
    if not isinstance(ops, list):
        return out
    for op in ops:
        if isinstance(op, dict) and "value" in op:
            op["value"] = _rewrite_cpath_value(op["value"], canon_root)
    return out


def _a4_allowed(rows: dict):
    al = rows.get("ast_allowlist")
    if isinstance(al, dict):
        per = al.get("per_step")
        if isinstance(per, dict):
            a4 = per.get("A4")
            if isinstance(a4, dict) and "allowed_over_150" in a4:
                return list(a4["allowed_over_150"])
    return ["_validate_i_series_consistency"]


def build_v2_payloads(inputs: dict) -> dict:
    """Pure — no disk writes. Returns rows/generation/migration dicts."""
    v1 = inputs["v1_rows"]
    tool_sha = inputs["tool_sha256"]
    harness_sha = inputs["harness_sha256"]
    prep = inputs["prep_package_sha256"]
    parent_rows = inputs.get("parent_rows_sha256", PIN_V1_PARENT_ROWS_SHA256)
    parent_base = inputs.get("parent_baseline_sha256", PIN_V1_PARENT_BASELINE_SHA256)
    canon_root = inputs.get("canon_root", CANON_ROOT_DEFAULT)

    gf = v1.get("generated_from") or {}
    if gf.get("method") != V1_METHOD:
        raise RuntimeError("V1_METHOD_MISMATCH")
    if gf.get("artifact") != SOURCE_ARTIFACT:
        raise RuntimeError("V1_ARTIFACT_MISMATCH")
    if gf.get("sha256") != SOURCE_ARTIFACT_SHA:
        raise RuntimeError("V1_SOURCE_SHA_MISMATCH")

    rows = copy.deepcopy(v1)
    rows["generation"] = "v2"
    rows["schema"] = V2_ROWS_SCHEMA
    rows["bound_to"] = {
        "base_manifest": HEAD_B_MANIFEST,
        "fixture": FIXTURE_B,
        "harness_sha256": harness_sha,
        "tool": TOOL_REL,
        "tool_sha256": tool_sha,
    }
    rows["generated_from"] = {
        "artifact": SOURCE_ARTIFACT,
        "generator": PRODUCER,
        "method": V1_METHOD,
        "sha256": SOURCE_ARTIFACT_SHA,
    }
    for r in rows["rows"]:
        r["fixture_id"] = FIXTURE_B
        if r.get("row_id") == "M_main_06":
            r["expected_msg_key"] = DERIVED_M06
        if r.get("row_id") in ("M_main_27", "M_main_56", "M_main_61"):
            r["mutation_spec"] = _rewrite_mutation_spec(r.get("mutation_spec"), canon_root)

    census_v1 = v1.get("census") or {}
    residual_n = len(rows.get("residual_sites") or [])
    rows_n = len(rows.get("rows") or [])
    census = {
        "identity": census_v1.get("identity"),
        "residual": residual_n,
        "rows": rows_n,
        "total": residual_n + rows_n,
    }
    if census != census_v1:
        raise RuntimeError(f"CENSUS_MISMATCH got={census} want={census_v1}")
    rows["census"] = census

    rows_sha = hashlib.sha256(canonical_json(rows).encode()).hexdigest()
    a4 = _a4_allowed(v1)
    migration = {
        "schema": "LANDS_AB_dry_exec_diag_corpus_migration/v1_to_v2",
        "parent_generation": "v1",
        "child_generation": "v2",
        "parent_rows_sha256": parent_rows,
        "parent_baseline_sha256": parent_base,
        "child_rows_sha256": rows_sha,
        "migration_kind": "generation_rebind_source_currency",
        "prep_package_sha256": prep,
        "producer": PRODUCER,
    }
    mig_sha = hashlib.sha256(canonical_json(migration).encode()).hexdigest()
    generation = {
        "generation": "v2",
        "schema_rows": V2_ROWS_SCHEMA,
        "schema_baseline": V2_BASELINE_SCHEMA,
        "baseline_name": V2_BASELINE_NAME,
        "baseline_head": V2_BASELINE_HEAD,
        "tool_sha256_at_authoring": tool_sha,
        "migration_carrier_sha256": mig_sha,
        "parent_generation": "v1",
        "parent_rows_sha256": parent_rows,
        "parent_baseline_sha256": parent_base,
        "prep_package_sha256": prep,
        "ast_allowlist_A4_allowed_over_150": a4,
        "v2_rows_sha256": rows_sha,
    }
    return {"rows": rows, "generation": generation, "migration": migration}


def mint_v2_fixtures(repo: Path, *, prep_package_sha256: str, oexcl_write=None) -> dict:
    write = oexcl_write or oexcl_write_bytes
    repo = Path(repo)
    v1_rows = json.loads((repo / V1_FIXTURE_DIR / "ROWS.json").read_text())
    tool_sha = hashlib.sha256((repo / TOOL_REL).read_bytes()).hexdigest()
    harness_sha = hashlib.sha256((repo / HARNESS_REL).read_bytes()).hexdigest()
    if tool_sha != TOOL_SHA_PIN:
        raise SystemExit(f"TOOL_SHA_MISMATCH: {tool_sha}")
    if harness_sha != HARNESS_SHA_PIN:
        raise SystemExit(f"HARNESS_SHA_MISMATCH: {harness_sha}")
    parent_rows = hashlib.sha256((repo / V1_FIXTURE_DIR / "ROWS.json").read_bytes()).hexdigest()
    parent_base = hashlib.sha256(
        (repo / V1_FIXTURE_DIR / "BASELINE_TOOL_6d2978d3.json").read_bytes()
    ).hexdigest()
    if parent_rows != PIN_V1_PARENT_ROWS_SHA256:
        raise SystemExit(f"PARENT_ROWS_SHA_MISMATCH: {parent_rows}")
    if parent_base != PIN_V1_PARENT_BASELINE_SHA256:
        raise SystemExit(f"PARENT_BASE_SHA_MISMATCH: {parent_base}")
    inputs = {
        "v1_rows": v1_rows,
        "tool_sha256": tool_sha,
        "harness_sha256": harness_sha,
        "prep_package_sha256": prep_package_sha256,
        "parent_rows_sha256": parent_rows,
        "parent_baseline_sha256": parent_base,
    }
    a = build_v2_payloads(inputs)
    b = build_v2_payloads(inputs)
    for k in ("rows", "generation", "migration"):
        if canonical_json(a[k]) != canonical_json(b[k]):
            raise SystemExit(f"NON_DETERMINISTIC: {k}")
    flipped = dict(inputs)
    flipped["tool_sha256"] = ("0" if inputs["tool_sha256"][0] != "0" else "1") + inputs["tool_sha256"][1:]
    ctrl = build_v2_payloads(flipped)
    if ctrl["rows"]["bound_to"]["tool_sha256"] == a["rows"]["bound_to"]["tool_sha256"]:
        raise SystemExit("NON_VACUOUS_CONTROL_FAILED")
    root = repo / V2_FIXTURE_DIR
    # parent must pre-exist (no auto-mkdir inside oexcl); create dir as non-authority mkdir only
    if not root.is_dir():
        root.mkdir(parents=False)
    paths = {
        "rows": root / "ROWS.json",
        "migration": root / "MIGRATION_v1_to_v2.json",
        "generation": root / "GENERATION.json",
    }
    shas = {
        "rows": write(paths["rows"], canonical_json(a["rows"]).encode()),
        "migration": write(paths["migration"], canonical_json(a["migration"]).encode()),
        "generation": write(paths["generation"], canonical_json(a["generation"]).encode()),
    }
    try:
        write(paths["rows"], canonical_json(a["rows"]).encode())
        raise SystemExit("EXISTING_TARGET_REFUSE_NOT_ENFORCED")
    except SystemExit as e:
        if "EXISTING_TARGET_REFUSE" not in str(e):
            raise
    modes = {k: oct(paths[k].stat().st_mode & 0o777) for k in paths}
    return {"paths": {k: str(v) for k, v in paths.items()}, "shas": shas, "modes": modes, "payloads": a}


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    prep = None
    repo = None
    i = 0
    while i < len(argv):
        if argv[i] == "--prep-package-sha256":
            prep = argv[i + 1]; i += 2; continue
        if argv[i] in ("--repo", "--repo-root"):
            repo = Path(argv[i + 1]); i += 2; continue
        if not argv[i].startswith("-") and repo is None:
            repo = Path(argv[i]); i += 1; continue
        i += 1
    if repo is None:
        repo = Path.cwd()
    if prep is None:
        plan = Path(os.environ.get(
            "SLICE_A_PLAN_MD",
            "/home/gabe/plan-dev-scratch/repin/fixture_gen_v2_slice_a_plan_v58/PLAN.md",
        ))
        prep = hashlib.sha256(plan.read_bytes()).hexdigest()
    out = mint_v2_fixtures(repo, prep_package_sha256=prep)
    print(json.dumps({"ok": True, "paths": out["paths"], "shas": out["shas"], "modes": out["modes"]},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
