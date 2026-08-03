"""Source-identity and artifact-contract facade for the LANDS-AB diagnostic-class corpus.

Extracted from scripts/lands_ab_dry_exec_diag_corpus.py under Phase E of
A0_SUCCESSOR_PLAN_v2 (sha f381736d…): behavior-preserving, no semantic change.

Seam contract (architecture_discipline.md §Required Seams — "artifact, manifest, hash
and resume contracts" + the single named import facade): this module owns every path
constant, every sha computation, the baseline/corpus reads, the source-pin preflight,
and the ONE dynamic import of the packet fixture builder.

Dependency direction: harness -> sources -> generation_authority / stdlib.
This module imports neither the harness nor the reducers.

Generation v1 authoring-slice: ACTIVE_GENERATION selector + generation-aware paths.
Generation-authority validation lives in lands_ab_generation_authority (plan v8);
this module provides thin injection wrappers preserving external call signatures.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from calm.llm_computer import lands_ab_generation_authority as GA

TOOL_REL = "scripts/lands_ab_packet_dry_exec.py"
HARNESS_REL = "calm/llm_computer/tests/test_hrm_text_158_lands_ab_science_source_manifest.py"

# Active generation (activation commit sets "v1"; isolated authoring may pass generation= explicitly)
ACTIVE_GENERATION = "v1"
# Sealed v58 PLAN.md sha — prep_package pin authority (not receipt-sourced)
SEALED_V58_PLAN_SHA256 = (
    "228b8e817379e8286747cea7663012be1b29463b0a867ddc3400373dae9e9c1b"
)

GENERATIONS = {
    "v0": {
        "fixture_dir": "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v0",
        "baseline_name": "BASELINE_HEAD_9f471b3.json",
        "baseline_schema": "LANDS_AB_dry_exec_diag_corpus_baseline/v0",
        "baseline_head": "9f471b3",
        "rows_schema": "LANDS_AB_dry_exec_diag_corpus/v0",
        "has_generation_json": False,
    },
    "v1": {
        "fixture_dir": "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v1",
        "baseline_name": "BASELINE_TOOL_6d2978d3.json",  # overridden by GENERATION.json when present
        "baseline_schema": "LANDS_AB_dry_exec_diag_corpus_baseline/v1",
        "baseline_head": "6d2978d3",
        "rows_schema": "LANDS_AB_dry_exec_diag_corpus_rows/v1",
        "has_generation_json": True,
    },
    "v2": {
        "fixture_dir": "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v2",
        "baseline_name": "BASELINE_TOOL_bc0d7f56.json",
        "baseline_schema": "LANDS_AB_dry_exec_diag_corpus_baseline/v2",
        "baseline_head": "bc0d7f56",
        "rows_schema": "LANDS_AB_dry_exec_diag_corpus_rows/v2",
        "has_generation_json": True,
    },
}

# Backward-compat module-level names (default to ACTIVE_GENERATION resolved statically for imports)
FIXTURE_DIR = GENERATIONS[ACTIVE_GENERATION]["fixture_dir"]
BASELINE_NAME = GENERATIONS[ACTIVE_GENERATION]["baseline_name"]
BASELINE_SCHEMA = GENERATIONS[ACTIVE_GENERATION]["baseline_schema"]
BASELINE_HEAD = GENERATIONS[ACTIVE_GENERATION]["baseline_head"]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gen(generation: str | None = None) -> str:
    """Resolve generation id fail-closed. Unknown → RuntimeError(unknown_generation:…)."""
    g = ACTIVE_GENERATION if generation is None else generation
    if g not in GENERATIONS:
        raise RuntimeError(f"unknown_generation: {g}")
    return g


# --- plan v8 injection wrappers (≤10 physical body lines each; zero validation logic) ---

def _validate_v2_generation_receipt(repo, receipt, *, check_live_tool=False):
    """v2 exact-set + value pins + required carriers + migration shape (plan §7.6/§7.6a)."""
    meta = GENERATIONS["v2"]
    fixture_dir = meta["fixture_dir"]
    fails = []
    if not fixture_dir.endswith("_v2"):
        fails.append({"code": "generation_path_mismatch", "fixture_dir": fixture_dir})
        return fails
    if not isinstance(receipt, dict):
        return list(GA.validate_generation_exact_set(receipt))

    rows_p = repo / fixture_dir / "ROWS.json"
    mig_path = repo / fixture_dir / "MIGRATION_v1_to_v2.json"
    rows_sha = None
    mig_sha = None
    migration = None
    if not rows_p.is_file():
        fails.append({
            "code": "generation_pin_mismatch", "field": "v2_rows_sha256",
            "reason": "rows_absent", "path": str(rows_p),
        })
    else:
        rows_sha = sha256_file(rows_p)
    if not mig_path.is_file():
        fails.append({
            "code": "generation_pin_mismatch", "field": "migration_carrier_sha256",
            "reason": "migration_absent", "path": str(mig_path),
        })
    else:
        mig_sha = sha256_file(mig_path)
        try:
            migration = json.loads(mig_path.read_text())
        except (json.JSONDecodeError, OSError) as exc:
            fails.append({
                "code": "migration_receipt_mismatch", "reason": "unparseable", "error": str(exc),
            })
            migration = None

    tool_sha = None
    tool_path = repo / TOOL_REL
    if tool_path.is_file():
        tool_sha = sha256_file(tool_path)
    elif check_live_tool:
        fails.append({"code": "generation_tool_mismatch", "reason": "tool_absent", "path": TOOL_REL})

    exact_kw = {"prep_package_sha256": SEALED_V58_PLAN_SHA256}
    if rows_sha is not None:
        exact_kw["v2_rows_sha256"] = rows_sha
    if mig_sha is not None:
        exact_kw["migration_carrier_sha256"] = mig_sha
    if tool_sha is not None:
        exact_kw["tool_sha256_at_authoring"] = tool_sha
    fails.extend(GA.validate_generation_exact_set(receipt, **exact_kw))

    if migration is not None and rows_sha is not None:
        fails.extend(GA.validate_migration_receipt_shape(
            migration,
            prep_package_sha256=SEALED_V58_PLAN_SHA256,
            child_rows_sha256=rows_sha,
            parent_rows_sha256=GA.PIN_V1_PARENT_ROWS_SHA256,
            parent_baseline_sha256=GA.PIN_V1_PARENT_BASELINE_SHA256,
        ))
    if check_live_tool and tool_sha is not None:
        want_tool = receipt.get("tool_sha256_at_authoring")
        if tool_sha != want_tool:
            fails.append({
                "code": "generation_tool_mismatch",
                "tool_sha256": tool_sha, "generation_tool_sha256": want_tool,
            })
    return fails


def validate_generation_receipt(repo, generation, receipt, *, check_live_tool=False):
    g = _gen(generation)
    if g == "v2":
        return _validate_v2_generation_receipt(repo, receipt, check_live_tool=check_live_tool)
    meta, v0 = GENERATIONS[g], GENERATIONS["v0"]
    return GA.validate_generation_receipt(
        repo, g, receipt, contract=GA.V1_GENERATION_CONTRACT,
        required_field_types=GA._V1_REQUIRED_FIELD_TYPES, fixture_dir=meta["fixture_dir"],
        v0_fixture_dir=v0["fixture_dir"], v0_baseline_name=v0["baseline_name"],
        tool_rel=TOOL_REL, sha256_file=sha256_file, check_live_tool=check_live_tool)


def preflight_generation(repo, generation=None, tool_sha=None, check_live_tool=False):
    g = ACTIVE_GENERATION if generation is None else generation
    if g == "v2":
        fails = []
        if g not in GENERATIONS:
            return [{"code": "unknown_generation", "generation": g}]
        meta = GENERATIONS[g]
        fixture_dir = meta["fixture_dir"]
        if not fixture_dir.endswith("_v2"):
            fails.append({"code": "generation_path_mismatch", "fixture_dir": fixture_dir})
            return fails
        try:
            receipt = load_generation_receipt(repo, g)
        except RuntimeError as e:
            msg = str(e)
            if "absent" in msg:
                fails.append({"code": "generation_receipt_absent", "error": msg})
            else:
                fails.append({"code": "generation_receipt_mismatch", "error": msg})
            return fails
        fails += validate_generation_receipt(repo, g, receipt, check_live_tool=check_live_tool)
        return fails
    return GA.preflight_generation(
        repo, g, generations=GENERATIONS, active_generation=ACTIVE_GENERATION,
        contract=GA.V1_GENERATION_CONTRACT, required_field_types=GA._V1_REQUIRED_FIELD_TYPES,
        tool_rel=TOOL_REL, sha256_file=sha256_file, tool_sha=tool_sha,
        check_live_tool=check_live_tool)


def classify_corpus_load_error(exc):
    return GA.classify_corpus_load_error(exc)


def load_generation_receipt(repo, generation=None):
    g = _gen(generation)
    return GA.load_generation_receipt(repo, g, fixture_dir=GENERATIONS[g]["fixture_dir"])


def generation_meta(repo: Path, generation: str | None = None) -> dict:
    """Resolve generation metadata. Adopts receipt fields ONLY after validation."""
    g = _gen(generation)
    meta = dict(GENERATIONS[g])
    if not meta.get("has_generation_json"):
        return meta
    try:
        receipt = load_generation_receipt(repo, g)
    except RuntimeError:
        return meta
    if validate_generation_receipt(repo, g, receipt):
        return meta
    meta["baseline_name"] = receipt["baseline_name"]
    meta["baseline_head"] = receipt["baseline_head"]
    meta["baseline_schema"] = receipt["schema_baseline"]
    meta["rows_schema"] = receipt["schema_rows"]
    meta["generation_receipt"] = receipt
    return meta


def tool_source(repo: Path) -> str:
    """Tool bytes as text, for the AST reducers (which never touch the filesystem)."""
    return (repo / TOOL_REL).read_text()


def rows_path(repo: Path, generation: str | None = None) -> Path:
    g = _gen(generation)
    return repo / GENERATIONS[g]["fixture_dir"] / "ROWS.json"


def canonical_baseline_path(repo: Path, generation: str | None = None) -> Path:
    meta = generation_meta(repo, generation)
    return repo / meta["fixture_dir"] / meta["baseline_name"]


def load_corpus(repo: Path, generation: str | None = None) -> dict:
    """Load ROWS (+ require GENERATION for gens that carry one). Fail-closed RuntimeError."""
    g = _gen(generation)
    if GENERATIONS[g].get("has_generation_json"):
        load_generation_receipt(repo, g)
    path = rows_path(repo, g)
    if not path.is_file():
        raise RuntimeError(f"rows_absent: {path}")
    try:
        text = path.read_text()
    except OSError as exc:
        raise RuntimeError(f"rows_absent: {path}: {exc}") from None
    try:
        rows = json.loads(text)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"rows_unparseable: {exc}") from None
    if not isinstance(rows, dict):
        raise RuntimeError(f"rows_unparseable: not_an_object: {type(rows).__name__}")
    return rows


def load_baseline(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def import_fixture_builder(repo: Path, expected_sha: str):
    """Single named import facade for the packet fixture builder.

    Reason this dynamic import exists: the faithful base packet is defined by the
    tracked test harness, and duplicating it here would create a second source of
    truth for the fixture.

    Contract, all fail-closed BEFORE any module code executes:
      path     -- HARNESS_REL must exist under repo
      hash     -- bytes must equal the sha pinned in ROWS.json bound_to.harness_sha256
      surface  -- module must expose _packet_for_dry, _sha, DRY
    """
    src = repo / HARNESS_REL
    if not src.is_file():
        raise RuntimeError(f"fixture builder absent: {src}")
    actual = sha256_file(src)
    if actual != expected_sha:
        raise RuntimeError(
            f"fixture builder hash drift: expected {expected_sha}, got {actual} ({src}). "
            "The imported source defines every generated packet; refusing to import."
        )
    spec = importlib.util.spec_from_file_location("lands_ab_science_manifest_tests", src)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(repo))
    spec.loader.exec_module(mod)
    for attr in ("_packet_for_dry", "_sha", "DRY"):
        if not hasattr(mod, attr):
            raise RuntimeError(f"fixture builder missing {attr} ({src})")
    return mod


def tool_identity(repo: Path, corpus: dict, baseline: dict | None) -> dict:
    """The three tool-sha values, kept explicitly distinct."""
    return {
        "candidate_tool_sha256": sha256_file(repo / TOOL_REL),
        "rows_bound_to_tool_sha256": (corpus.get("bound_to") or {}).get("tool_sha256"),
        "baseline_tool_sha256": (baseline or {}).get("tool_sha256"),
    }


def preflight_manifest_currency(repo: Path, corpus: dict) -> list:
    """Candidate currency: every manifest entry must match its on-disk bytes."""
    rel = (corpus.get("bound_to") or {}).get("base_manifest")
    if not rel:
        return [{"code": "manifest_binding_absent"}]
    path = repo / rel
    if not path.is_file():
        return [{"code": "manifest_absent", "path": rel}]
    try:
        manifest = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        return [{"code": "manifest_unparseable", "path": rel, "error": str(exc)}]
    stale, missing = [], []
    for entry in manifest.get("entries") or []:
        entry_path = repo / entry.get("path", "")
        if not entry_path.is_file():
            missing.append(entry.get("path"))
        elif sha256_file(entry_path) != entry.get("sha256"):
            stale.append(entry.get("path"))
    if stale or missing:
        return [{"code": "manifest_stale", "manifest": rel,
                 "stale_entries": sorted(stale)[:10], "stale_count": len(stale),
                 "missing_entries": sorted(missing)[:10], "missing_count": len(missing),
                 "cure": "regenerate the science-source manifest against the candidate"}]
    return []


def preflight_identity(repo: Path, corpus: dict, baseline: dict | None, step: str) -> list:
    """Step-aware tool identity. See tool_identity() for the provenance/currency split."""
    fails = []
    ident = tool_identity(repo, corpus, baseline)
    candidate = ident["candidate_tool_sha256"]
    rows_prov = ident["rows_bound_to_tool_sha256"]
    base_prov = ident["baseline_tool_sha256"]
    if baseline is not None and rows_prov != base_prov:
        fails.append({"code": "provenance_internal_mismatch",
                      "rows_bound_to_tool_sha256": rows_prov,
                      "baseline_tool_sha256": base_prov,
                      "why": "ROWS.bound_to and BASELINE record the same authoring event; "
                             "a mismatch means a frozen reference was rewritten"})
    if step == "A0":
        if candidate != rows_prov:
            fails.append({"code": "candidate_sha_not_baseline_at_A0", "step": step,
                          "candidate_tool_sha256": candidate,
                          "provenance_tool_sha256": rows_prov})
    return fails


def preflight_bindings(repo: Path, corpus: dict, baseline: dict | None,
                       require_baseline: bool = True,
                       generation: str | None = None) -> list:
    """Baseline binding + fixture-builder pin. Reads bytes; no import, no subprocess."""
    fails = []
    g = _gen(generation)
    meta = generation_meta(repo, g) if GENERATIONS[g].get("has_generation_json") or g in GENERATIONS else GENERATIONS[g]

    def bad(code, **kw):
        fails.append(dict(code=code, **kw))

    bound = corpus.get("bound_to") or {}
    harness_pin = bound.get("harness_sha256")
    if not harness_pin:
        bad("harness_pin_absent")
    else:
        src = repo / HARNESS_REL
        got = sha256_file(src) if src.is_file() else None
        if got != harness_pin:
            bad("harness_sha_drift", expected=harness_pin, got=got, path=HARNESS_REL)

    if baseline is None:
        if require_baseline:
            bad("baseline_absent", path=f"{meta['fixture_dir']}/{meta['baseline_name']}")
        return fails
    expected_schema = meta["baseline_schema"]
    expected_head = meta["baseline_head"]
    if baseline.get("schema") != expected_schema:
        bad("baseline_schema", expected=expected_schema, got=baseline.get("schema"))
    if baseline.get("head") != expected_head:
        bad("baseline_head", expected=expected_head, got=baseline.get("head"))
    rows_sha = sha256_file(rows_path(repo, g))
    if baseline.get("rows_sha256") != rows_sha:
        bad("baseline_rows_sha", expected=rows_sha, got=baseline.get("rows_sha256"))
    if baseline.get("harness_sha256") != bound.get("harness_sha256"):
        bad("baseline_source_pin", field="harness_sha256",
            baseline=baseline.get("harness_sha256"), corpus=bound.get("harness_sha256"))
    if baseline.get("generated_from_rows") != corpus.get("generated_from"):
        bad("baseline_generated_from")
    bmap = baseline.get("map")
    if not isinstance(bmap, dict):
        bad("baseline_map_missing")
    else:
        want = {r.get("row_id") for r in corpus.get("rows") or []}
        got = set(bmap)
        if got != want:
            bad("baseline_rowid_set", missing=sorted(want - got)[:10],
                unexpected=sorted(got - want)[:10])
    return fails
