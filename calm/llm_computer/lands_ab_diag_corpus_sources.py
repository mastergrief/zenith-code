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


def preflight_bindings(repo: Path, corpus: dict, baseline: dict | None,
                       require_baseline: bool = True) -> list:
    """Source-set pinning + baseline binding. Reads bytes; no import, no subprocess.

    require_baseline=False is used ONLY by the mint path, where no baseline can exist
    yet by construction; every source pin is still enforced.
    """
    fails = []

    def bad(code, **kw):
        fails.append(dict(code=code, **kw))

    bound = corpus.get("bound_to") or {}
    tool_sha = sha256_file(repo / TOOL_REL)
    if tool_sha != bound.get("tool_sha256"):
        bad("tool_sha_drift", expected=bound.get("tool_sha256"), got=tool_sha)
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
    for field in ("tool_sha256", "harness_sha256"):
        if baseline.get(field) != bound.get(field):
            bad("baseline_source_pin", field=field, baseline=baseline.get(field),
                corpus=bound.get(field))
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
