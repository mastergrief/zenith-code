"""Generation-authority facade for LANDS-AB diagnostic corpus (plan v8 extraction).

Owns V1 contract pins, pure receipt-shape validation, disk/hash IO validation,
classified GENERATION.json load, preflight_generation, and classify_corpus_load_error.

Dependency: never imports lands_ab_diag_corpus_sources, harness, battery, or launch glue.
All external state is passed as parameters (argument injection).
"""
from __future__ import annotations

import json
from pathlib import Path


V1_GENERATION_CONTRACT = {
    "schema_rows": "LANDS_AB_dry_exec_diag_corpus_rows/v1",
    "schema_baseline": "LANDS_AB_dry_exec_diag_corpus_baseline/v1",
    "baseline_name": "BASELINE_TOOL_6d2978d3.json",
    "baseline_head": "6d2978d3",
    "tool_sha256_at_authoring": (
        "6d2978d3a7cf526ee8e2815b5688ef586e9608a640a3eda7398577a0b77977c9"
    ),
    "parent_generation": "v0",
    "parent_rows_sha256": (
        "0100d08707f4322f56d024ffa230eb80d5123a84dce1b231c72415befa79e315"
    ),
    "parent_baseline_sha256": (
        "5cfe9db70105e6c740c96f55d46ac997f917baf9de283e95e9c33ce0c3b04c9c"
    ),
    "prep_package_sha256": (
        "9d6eeb0407b55ea2d03435e45f1fde03a69250d642d5bd8ff5cca2dcae7820ee"
    ),
    "ast_allowlist_A4_allowed_over_150": ["_validate_i_series_consistency"],
}

_V1_REQUIRED_FIELD_TYPES = {
    "generation": str,
    "schema_rows": str,
    "schema_baseline": str,
    "baseline_name": str,
    "baseline_head": str,
    "tool_sha256_at_authoring": str,
    "migration_carrier_sha256": str,
    "parent_generation": str,
    "parent_rows_sha256": str,
    "parent_baseline_sha256": str,
    "prep_package_sha256": str,
    "ast_allowlist_A4_allowed_over_150": list,
    "v1_rows_sha256": str,
}


def _safe_baseline_basename(name: str) -> bool:
    """True iff name is a single path segment with no traversal."""
    if not isinstance(name, str) or not name or name in (".", ".."):
        return False
    if "/" in name or "\\" in name or "\x00" in name:
        return False
    if name.startswith("~") or (len(name) >= 2 and name[1] == ":"):
        return False
    p = Path(name)
    return p.name == name and ".." not in p.parts


def _rows_a4_allowed_over_150(rows: dict) -> tuple[object | None, list]:
    """Extract ROWS ast_allowlist.per_step.A4.allowed_over_150 fail-closed."""
    fails: list = []
    if not isinstance(rows, dict):
        fails.append({"code": "generation_receipt_mismatch",
                      "reason": "rows_allowlist_malformed", "node": "rows",
                      "got_type": type(rows).__name__})
        return None, fails
    al = rows.get("ast_allowlist")
    if al is None:
        return None, fails
    if not isinstance(al, dict):
        fails.append({"code": "generation_receipt_mismatch",
                      "reason": "rows_allowlist_malformed", "node": "ast_allowlist",
                      "got_type": type(al).__name__})
        return None, fails
    per = al.get("per_step")
    if per is None:
        return None, fails
    if not isinstance(per, dict):
        fails.append({"code": "generation_receipt_mismatch",
                      "reason": "rows_allowlist_malformed", "node": "per_step",
                      "got_type": type(per).__name__})
        return None, fails
    a4 = per.get("A4")
    if a4 is None:
        return None, fails
    if not isinstance(a4, dict):
        fails.append({"code": "generation_receipt_mismatch",
                      "reason": "rows_allowlist_malformed", "node": "A4",
                      "got_type": type(a4).__name__})
        return None, fails
    return a4.get("allowed_over_150"), fails


def validate_generation_receipt_shape(
    generation: str,
    receipt,
    contract: dict,
    required_field_types: dict,
) -> list:
    """Pure: object/types/required/frozen pin equality/A4 allowlist pin. No FS."""
    fails: list = []

    def bad(code: str, **kw):
        fails.append(dict(code=code, **kw))

    if not isinstance(receipt, dict):
        bad("generation_receipt_mismatch", reason="not_an_object",
            got_type=type(receipt).__name__)
        return fails

    if generation != "v1":
        if receipt.get("generation") != generation:
            bad("generation_receipt_mismatch",
                file=receipt.get("generation"), expected=generation)
        return fails

    for field, typ in required_field_types.items():
        if field not in receipt:
            bad("generation_receipt_mismatch", reason="required_field_absent", field=field)
            continue
        if not isinstance(receipt[field], typ):
            bad("generation_receipt_mismatch", reason="required_field_bad_type",
                field=field, expected=typ.__name__,
                got_type=type(receipt[field]).__name__)

    if fails:
        return fails

    if receipt.get("generation") != generation:
        bad("generation_receipt_mismatch",
            file=receipt.get("generation"), expected=generation)
        return fails

    pin_fields = (
        "schema_rows", "schema_baseline", "baseline_name", "baseline_head",
        "tool_sha256_at_authoring", "parent_generation",
        "parent_rows_sha256", "parent_baseline_sha256", "prep_package_sha256",
    )
    for field in pin_fields:
        if receipt.get(field) != contract[field]:
            bad("generation_pin_mismatch", field=field,
                expected=contract[field], got=receipt.get(field))

    allow = receipt.get("ast_allowlist_A4_allowed_over_150")
    if allow != contract["ast_allowlist_A4_allowed_over_150"]:
        bad("generation_pin_mismatch", field="ast_allowlist_A4_allowed_over_150",
            expected=contract["ast_allowlist_A4_allowed_over_150"], got=allow)
    return fails


def validate_generation_receipt_disk(
    repo: Path,
    generation: str,
    receipt: dict,
    *,
    fixture_dir: str,
    v0_fixture_dir: str,
    v0_baseline_name: str,
    tool_rel: str,
    sha256_file,
    check_live_tool: bool = False,
) -> list:
    """IO: baseline basename containment; migration/parent/v1 ROWS hashes; ROWS bind; live tool."""
    fails: list = []

    def bad(code: str, **kw):
        fails.append(dict(code=code, **kw))

    baseline_name = receipt.get("baseline_name")
    fixture_root = (repo / fixture_dir).resolve()
    if not _safe_baseline_basename(baseline_name):
        bad("generation_path_escape", baseline_name=baseline_name,
            reason="unsafe_basename")
    else:
        candidate = (fixture_root / baseline_name).resolve()
        try:
            candidate.relative_to(fixture_root)
            contained = True
        except ValueError:
            contained = False
        if not contained:
            bad("generation_path_escape", baseline_name=baseline_name,
                resolved=str(candidate), fixture_dir=fixture_dir,
                reason="not_contained_under_fixture_dir")

    mig_path = repo / fixture_dir / "MIGRATION_v0_to_v1.json"
    if not mig_path.is_file():
        bad("generation_pin_mismatch", field="migration_carrier_sha256",
            reason="migration_absent", path=str(mig_path))
    else:
        got_mig = sha256_file(mig_path)
        if got_mig != receipt.get("migration_carrier_sha256"):
            bad("generation_pin_mismatch", field="migration_carrier_sha256",
                expected=receipt.get("migration_carrier_sha256"), got=got_mig)

    v0_rows = repo / v0_fixture_dir / "ROWS.json"
    v0_base = repo / v0_fixture_dir / v0_baseline_name
    if not v0_rows.is_file():
        bad("generation_pin_mismatch", field="parent_rows_sha256",
            reason="parent_rows_absent")
    else:
        got = sha256_file(v0_rows)
        if got != receipt.get("parent_rows_sha256"):
            bad("generation_pin_mismatch", field="parent_rows_sha256",
                expected=receipt.get("parent_rows_sha256"), got=got)
    if not v0_base.is_file():
        bad("generation_pin_mismatch", field="parent_baseline_sha256",
            reason="parent_baseline_absent")
    else:
        got = sha256_file(v0_base)
        if got != receipt.get("parent_baseline_sha256"):
            bad("generation_pin_mismatch", field="parent_baseline_sha256",
                expected=receipt.get("parent_baseline_sha256"), got=got)

    rows_p = repo / fixture_dir / "ROWS.json"
    if not rows_p.is_file():
        bad("generation_pin_mismatch", field="v1_rows_sha256", reason="rows_absent")
        return fails

    rows_sha = sha256_file(rows_p)
    if rows_sha != receipt.get("v1_rows_sha256"):
        bad("generation_pin_mismatch", field="v1_rows_sha256",
            expected=receipt.get("v1_rows_sha256"), got=rows_sha)

    try:
        rows = json.loads(rows_p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        bad("generation_receipt_mismatch", reason="rows_unparseable", error=str(exc))
        return fails

    if not isinstance(rows, dict):
        bad("generation_receipt_mismatch", reason="rows_not_an_object")
        return fails

    if rows.get("schema") != receipt.get("schema_rows"):
        bad("generation_pin_mismatch", field="rows.schema",
            expected=receipt.get("schema_rows"), got=rows.get("schema"))
    if rows.get("generation") != generation:
        bad("generation_pin_mismatch", field="rows.generation",
            expected=generation, got=rows.get("generation"))
    bound_raw = rows.get("bound_to")
    if bound_raw is not None and not isinstance(bound_raw, dict):
        bad("generation_receipt_mismatch", reason="rows_bound_to_malformed",
            got_type=type(bound_raw).__name__)
        bound = {}
    else:
        bound = bound_raw if isinstance(bound_raw, dict) else {}
    if bound.get("tool_sha256") != receipt.get("tool_sha256_at_authoring"):
        bad("generation_pin_mismatch", field="rows.bound_to.tool_sha256",
            expected=receipt.get("tool_sha256_at_authoring"),
            got=bound.get("tool_sha256"))

    allow_rows, al_fails = _rows_a4_allowed_over_150(rows)
    fails.extend(al_fails)
    if not al_fails and allow_rows != receipt.get("ast_allowlist_A4_allowed_over_150"):
        bad("generation_pin_mismatch", field="rows.ast_allowlist.A4",
            expected=receipt.get("ast_allowlist_A4_allowed_over_150"),
            got=allow_rows)

    if check_live_tool:
        tool_path = repo / tool_rel
        if not tool_path.is_file():
            bad("generation_tool_mismatch", reason="tool_absent", path=tool_rel)
        else:
            got_tool = sha256_file(tool_path)
            want_tool = receipt.get("tool_sha256_at_authoring")
            if got_tool != want_tool:
                bad("generation_tool_mismatch",
                    tool_sha256=got_tool, generation_tool_sha256=want_tool)
    return fails


def validate_generation_receipt(
    repo: Path,
    generation: str,
    receipt,
    *,
    contract: dict,
    required_field_types: dict,
    fixture_dir: str,
    v0_fixture_dir: str,
    v0_baseline_name: str,
    tool_rel: str,
    sha256_file,
    check_live_tool: bool = False,
) -> list:
    """Glue ≤40L: shape then disk; merged fails. No import of sources."""
    fails = validate_generation_receipt_shape(
        generation, receipt, contract, required_field_types)
    # Match prior early returns: non-object, non-v1, required-field fails, gen-key mismatch
    if not isinstance(receipt, dict) or generation != "v1":
        return fails
    if any(f.get("reason") in (
            "not_an_object", "required_field_absent", "required_field_bad_type")
           for f in fails):
        return fails
    if any(f.get("code") == "generation_receipt_mismatch"
           and "file" in f and "expected" in f and "reason" not in f for f in fails):
        return fails
    fails = list(fails) + validate_generation_receipt_disk(
        repo, generation, receipt,
        fixture_dir=fixture_dir,
        v0_fixture_dir=v0_fixture_dir,
        v0_baseline_name=v0_baseline_name,
        tool_rel=tool_rel,
        sha256_file=sha256_file,
        check_live_tool=check_live_tool,
    )
    return fails


def load_generation_receipt(
    repo: Path,
    generation: str,
    *,
    fixture_dir: str,
) -> dict:
    """IO load GENERATION.json; classified RuntimeError prefixes only."""
    path = repo / fixture_dir / "GENERATION.json"
    if not path.is_file():
        raise RuntimeError(f"generation_receipt_absent: {path}")
    try:
        receipt = json.loads(path.read_text())
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"generation_receipt_mismatch: unparseable: {exc}") from None
    if not isinstance(receipt, dict):
        raise RuntimeError(
            f"generation_receipt_mismatch: not_an_object: {type(receipt).__name__}"
        )
    if receipt.get("generation") != generation:
        raise RuntimeError(
            f"generation_receipt_mismatch: file={receipt.get('generation')} expected={generation}"
        )
    return receipt


def preflight_generation(
    repo: Path,
    generation: str,
    *,
    generations: dict,
    active_generation: str,
    contract: dict,
    required_field_types: dict,
    tool_rel: str,
    sha256_file,
    tool_sha: str | None = None,
    check_live_tool: bool = False,
) -> list:
    """Fail-closed generation preflight; unknown id classified; routes through validate_*."""
    fails = []
    g = generation
    if g not in generations:
        return [{"code": "unknown_generation", "generation": g}]
    meta = generations[g]
    fixture_dir = meta["fixture_dir"]
    if g == "v1" and not fixture_dir.endswith("_v1"):
        fails.append({"code": "generation_path_mismatch", "fixture_dir": fixture_dir})
    if g == "v1" and "_v0" in fixture_dir:
        fails.append({"code": "generation_path_mismatch", "fixture_dir": fixture_dir})
    if not meta.get("has_generation_json"):
        return fails
    try:
        receipt = load_generation_receipt(repo, g, fixture_dir=fixture_dir)
    except RuntimeError as e:
        msg = str(e)
        if "absent" in msg:
            fails.append({"code": "generation_receipt_absent", "error": msg})
        else:
            fails.append({"code": "generation_receipt_mismatch", "error": msg})
        return fails
    v0 = generations["v0"]
    fails += validate_generation_receipt(
        repo, g, receipt,
        contract=contract,
        required_field_types=required_field_types,
        fixture_dir=fixture_dir,
        v0_fixture_dir=v0["fixture_dir"],
        v0_baseline_name=v0["baseline_name"],
        tool_rel=tool_rel,
        sha256_file=sha256_file,
        check_live_tool=check_live_tool,
    )
    return fails


def classify_corpus_load_error(exc: BaseException) -> dict | None:
    """Pure map RuntimeError prefixes → stable preflight fail dict."""
    if not isinstance(exc, RuntimeError):
        return None
    msg = str(exc)
    prefixes = (
        ("unknown_generation", "unknown_generation"),
        ("generation_receipt_absent", "generation_receipt_absent"),
        ("generation_receipt_mismatch", "generation_receipt_mismatch"),
        ("rows_absent", "rows_absent"),
        ("rows_unparseable", "rows_unparseable"),
    )
    for prefix, code in prefixes:
        if msg == prefix or msg.startswith(prefix + ":") or msg.startswith(prefix + " "):
            return {"code": code, "error": msg}
    return None

# §7.6a exact-set consumers — thin re-export only (C5; body lives in exact_set_consumers)
from calm.llm_computer.lands_ab_diag_exact_set_consumers import (  # noqa: E402
    GENERATION_V2_LITERALS,
    GENERATION_V2_REQUIRED_FIELD_TYPES,
    GENERATION_V2_REQUIRED_KEYS,
    MIGRATION_V1_TO_V2_LITERALS,
    MIGRATION_V1_TO_V2_REQUIRED_KEYS,
    PIN_V1_PARENT_BASELINE_SHA256,
    PIN_V1_PARENT_ROWS_SHA256,
    run_slice_a_exact_set_and_parent_hash_negatives,
    validate_generation_exact_set,
    validate_migration_receipt_shape,
)
