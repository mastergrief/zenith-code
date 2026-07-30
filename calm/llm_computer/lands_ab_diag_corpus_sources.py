"""Source-identity and artifact-contract facade for the LANDS-AB diagnostic-class corpus.

Extracted from scripts/lands_ab_dry_exec_diag_corpus.py under Phase E of
A0_SUCCESSOR_PLAN_v2 (sha f381736d…): behavior-preserving, no semantic change.

Seam contract (architecture_discipline.md §Required Seams — "artifact, manifest, hash
and resume contracts" + the single named import facade): this module owns every path
constant, every sha computation, the baseline/corpus reads, the source-pin preflight,
and the ONE dynamic import of the packet fixture builder.

Dependency direction: harness -> sources -> stdlib. This module imports neither the
harness nor the reducers.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

FIXTURE_DIR = "calm/llm_computer/tests/fixtures/lands_ab_dry_exec_diag_corpus_v0"
BASELINE_NAME = "BASELINE_HEAD_9f471b3.json"
BASELINE_SCHEMA = "LANDS_AB_dry_exec_diag_corpus_baseline/v0"
BASELINE_HEAD = "9f471b3"
TOOL_REL = "scripts/lands_ab_packet_dry_exec.py"
HARNESS_REL = "calm/llm_computer/tests/test_hrm_text_158_lands_ab_science_source_manifest.py"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tool_source(repo: Path) -> str:
    """Tool bytes as text, for the AST reducers (which never touch the filesystem)."""
    return (repo / TOOL_REL).read_text()


def rows_path(repo: Path) -> Path:
    return repo / FIXTURE_DIR / "ROWS.json"


def canonical_baseline_path(repo: Path) -> Path:
    return repo / FIXTURE_DIR / BASELINE_NAME


def load_corpus(repo: Path) -> dict:
    return json.loads(rows_path(repo).read_text())


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
    """The three tool-sha values, kept explicitly distinct.

    PROVENANCE -- "is this the tool the baseline was captured from?" True only at A0.
      rows_bound_to : ROWS.bound_to.tool_sha256   (provenance of authoring)
      baseline_tool : BASELINE.tool_sha256        (provenance of capture)
    CURRENCY -- "do we know exactly which tool bytes we are testing?" Every step.
      candidate     : sha256 of the tool as it exists right now

    The two provenance values record the SAME authoring event and must always agree.
    Only `candidate` is permitted to move, and only at A1+.
    """
    return {
        "candidate_tool_sha256": sha256_file(repo / TOOL_REL),
        "rows_bound_to_tool_sha256": (corpus.get("bound_to") or {}).get("tool_sha256"),
        "baseline_tool_sha256": (baseline or {}).get("tool_sha256"),
    }


def preflight_manifest_currency(repo: Path, corpus: dict) -> list:
    """Candidate currency: every manifest entry must match its on-disk bytes.

    This is the SOLE candidate authority at A1+. The manifest pins the tool, so a
    refactor without a manifest regen lands here -- pre-subprocess -- instead of
    corrupting all 149 rows with the validator's own source-pin diagnostic.
    """
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

    # Provenance internal consistency -- ALL steps. These two are records of the same
    # authoring event; a disagreement means one was REWRITTEN, which is exactly the
    # reference-refresh bypass (operator "repairs" an A1 failure by editing bound_to).
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
    # A1+: NO equality against either provenance value. Divergence there is CORRECT
    # AND EXPECTED, not drift; candidate authority is manifest currency alone.
    return fails


def preflight_bindings(repo: Path, corpus: dict, baseline: dict | None,
                       require_baseline: bool = True) -> list:
    """Baseline binding + fixture-builder pin. Reads bytes; no import, no subprocess.

    Tool identity is NOT checked here -- see preflight_identity(), which is step-aware.

    require_baseline=False is used ONLY by the mint path, where no baseline can exist
    yet by construction; every source pin is still enforced.
    """
    fails = []

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
            bad("baseline_absent", path=f"{FIXTURE_DIR}/{BASELINE_NAME}")
        return fails
    if baseline.get("schema") != BASELINE_SCHEMA:
        bad("baseline_schema", expected=BASELINE_SCHEMA, got=baseline.get("schema"))
    if baseline.get("head") != BASELINE_HEAD:
        bad("baseline_head", expected=BASELINE_HEAD, got=baseline.get("head"))
    rows_sha = sha256_file(rows_path(repo))
    if baseline.get("rows_sha256") != rows_sha:
        bad("baseline_rows_sha", expected=rows_sha, got=baseline.get("rows_sha256"))
    # tool_sha256 is deliberately NOT compared here: baseline-vs-corpus tool provenance
    # is owned by preflight_identity's provenance_internal_consistency check, so the
    # receipt names that relationship once with its own code instead of twice.
    if baseline.get("harness_sha256") != bound.get("harness_sha256"):
        bad("baseline_source_pin", field="harness_sha256",
            baseline=baseline.get("harness_sha256"), corpus=bound.get("harness_sha256"))
    if baseline.get("generated_from_rows") != corpus.get("generated_from"):
        bad("baseline_generated_from")
    bmap = baseline.get("map")
    if not isinstance(bmap, dict):
        bad("baseline_map_missing")
    else:
        # .get: a row missing row_id is a schema failure reported by preflight_schema;
        # this check must still classify rather than raise, so the caller receives a
        # complete failure list instead of a traceback.
        want = {r.get("row_id") for r in corpus.get("rows") or []}
        got = set(bmap)
        if got != want:
            bad("baseline_rowid_set", missing=sorted(want - got)[:10],
                unexpected=sorted(got - want)[:10])
    return fails
