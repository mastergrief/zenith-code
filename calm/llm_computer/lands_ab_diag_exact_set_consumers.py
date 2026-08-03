"""Independent pure exact-set / migration validators for LANDS-AB diag corpus v2 (plan v58 §7.6a).

No FS. Reject missing AND extras. Frozen value pins from plan §7.6 (not from receipt under test).
"""
from __future__ import annotations

GENERATION_V2_REQUIRED_KEYS = frozenset({
    "generation", "schema_rows", "schema_baseline", "baseline_name", "baseline_head",
    "tool_sha256_at_authoring", "migration_carrier_sha256", "parent_generation",
    "parent_rows_sha256", "parent_baseline_sha256", "prep_package_sha256",
    "ast_allowlist_A4_allowed_over_150", "v2_rows_sha256",
})
MIGRATION_V1_TO_V2_REQUIRED_KEYS = frozenset({
    "schema", "parent_generation", "child_generation",
    "parent_rows_sha256", "parent_baseline_sha256", "child_rows_sha256",
    "migration_kind", "prep_package_sha256", "producer",
})
MIGRATION_V1_TO_V2_LITERALS = {
    "schema": "LANDS_AB_dry_exec_diag_corpus_migration/v1_to_v2",
    "parent_generation": "v1",
    "child_generation": "v2",
    "migration_kind": "generation_rebind_source_currency",
    "producer": "calm.llm_computer.lands_ab_diag_corpus_v2_author",
}
# Frozen GENERATION value pins from plan §7.6 (descriptor/pin authority — never receipt-sourced)
GENERATION_V2_LITERALS = {
    "generation": "v2",
    "schema_rows": "LANDS_AB_dry_exec_diag_corpus_rows/v2",
    "schema_baseline": "LANDS_AB_dry_exec_diag_corpus_baseline/v2",
    "baseline_name": "BASELINE_TOOL_bc0d7f56.json",
    "baseline_head": "bc0d7f56",
    "parent_generation": "v1",
}
GENERATION_V2_REQUIRED_FIELD_TYPES = {
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
    "v2_rows_sha256": str,
}
PIN_V1_PARENT_ROWS_SHA256 = "d61530c0dfeadc67d610ac6cc9b043f3c1d0bb2a490e356594f0047fe1923b89"
PIN_V1_PARENT_BASELINE_SHA256 = "a18a99f98f36051cdd739e7b43091184d8149a0f4ae1ce305583615f1b81460b"
PIN_A4_ALLOWED_OVER_150 = ["_validate_i_series_consistency"]
SHA_FIELDS = (
    "tool_sha256_at_authoring", "migration_carrier_sha256", "parent_rows_sha256",
    "parent_baseline_sha256", "prep_package_sha256", "v2_rows_sha256",
)


def _is_sha256_hex(v) -> bool:
    return isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v)


def validate_generation_exact_set(
    receipt,
    required_keys=GENERATION_V2_REQUIRED_KEYS,
    *,
    literals=None,
    required_field_types=None,
    parent_rows_sha256=PIN_V1_PARENT_ROWS_SHA256,
    parent_baseline_sha256=PIN_V1_PARENT_BASELINE_SHA256,
    a4_allowed=None,
    prep_package_sha256=None,
    tool_sha256_at_authoring=None,
    migration_carrier_sha256=None,
    v2_rows_sha256=None,
):
    """Pure. Exact key set + types + frozen value pins + optional live binds. No FS."""
    fails = []
    if not isinstance(receipt, dict):
        return [{"code": "generation_receipt_mismatch", "reason": "not_an_object"}]
    keys = set(receipt)
    for k in sorted(required_keys - keys):
        fails.append({"code": "generation_receipt_mismatch", "reason": "required_field_absent", "field": k})
    for k in sorted(keys - required_keys):
        fails.append({"code": "generation_receipt_mismatch", "reason": "extra_field", "field": k})
    types = required_field_types if required_field_types is not None else GENERATION_V2_REQUIRED_FIELD_TYPES
    type_bad = set()
    for field, typ in types.items():
        if field not in receipt:
            continue
        if not isinstance(receipt[field], typ):
            type_bad.add(field)
            fails.append({
                "code": "generation_receipt_mismatch", "reason": "required_field_bad_type",
                "field": field, "expected": typ.__name__, "got_type": type(receipt[field]).__name__,
            })
    lit = dict(GENERATION_V2_LITERALS if literals is None else literals)
    for k, want in lit.items():
        if k in type_bad:
            continue  # type failure is the class; do not double-count as pin mismatch
        if k in receipt and receipt.get(k) != want:
            fails.append({
                "code": "generation_pin_mismatch", "reason": "literal_mismatch",
                "field": k, "expected": want, "got": receipt.get(k),
            })
    # frozen parent pins (plan §7.6a)
    if receipt.get("parent_rows_sha256") != parent_rows_sha256:
        fails.append({
            "code": "generation_pin_mismatch", "reason": "parent_rows_sha_mismatch",
            "field": "parent_rows_sha256", "expected": parent_rows_sha256,
            "got": receipt.get("parent_rows_sha256"),
        })
    if receipt.get("parent_baseline_sha256") != parent_baseline_sha256:
        fails.append({
            "code": "generation_pin_mismatch", "reason": "parent_baseline_sha_mismatch",
            "field": "parent_baseline_sha256", "expected": parent_baseline_sha256,
            "got": receipt.get("parent_baseline_sha256"),
        })
    want_a4 = list(PIN_A4_ALLOWED_OVER_150 if a4_allowed is None else a4_allowed)
    if receipt.get("ast_allowlist_A4_allowed_over_150") != want_a4:
        fails.append({
            "code": "generation_pin_mismatch", "reason": "literal_mismatch",
            "field": "ast_allowlist_A4_allowed_over_150",
            "expected": want_a4, "got": receipt.get("ast_allowlist_A4_allowed_over_150"),
        })
    for field in SHA_FIELDS:
        if field in receipt and not _is_sha256_hex(receipt.get(field)):
            fails.append({
                "code": "generation_receipt_mismatch", "reason": "sha_field_bad_type",
                "field": field, "got_type": type(receipt.get(field)).__name__,
            })
    # optional live binds (caller supplies disk/tool/prep authority — never from receipt alone)
    binds = {
        "prep_package_sha256": prep_package_sha256,
        "tool_sha256_at_authoring": tool_sha256_at_authoring,
        "migration_carrier_sha256": migration_carrier_sha256,
        "v2_rows_sha256": v2_rows_sha256,
    }
    for field, want in binds.items():
        if want is None:
            continue
        if receipt.get(field) != want:
            fails.append({
                "code": "generation_pin_mismatch", "reason": f"{field}_mismatch",
                "field": field, "expected": want, "got": receipt.get(field),
            })
    return fails


def validate_migration_receipt_shape(
    migration,
    *,
    prep_package_sha256,
    child_rows_sha256,
    parent_rows_sha256=PIN_V1_PARENT_ROWS_SHA256,
    parent_baseline_sha256=PIN_V1_PARENT_BASELINE_SHA256,
    required_keys=MIGRATION_V1_TO_V2_REQUIRED_KEYS,
    literals=MIGRATION_V1_TO_V2_LITERALS,
):
    """Pure. Exact key set, literals, prep/child/PARENT sha binds + SHA shape. No FS."""
    fails = []
    if not isinstance(migration, dict):
        return [{"code": "migration_receipt_mismatch", "reason": "not_an_object"}]
    keys = set(migration)
    for k in sorted(required_keys - keys):
        fails.append({"code": "migration_receipt_mismatch", "reason": "required_field_absent", "field": k})
    for k in sorted(keys - required_keys):
        fails.append({"code": "migration_receipt_mismatch", "reason": "extra_field", "field": k})
    for k, want in literals.items():
        if migration.get(k) != want:
            fails.append({
                "code": "migration_receipt_mismatch", "reason": "literal_mismatch",
                "field": k, "expected": want, "got": migration.get(k),
            })
    for field in ("parent_rows_sha256", "parent_baseline_sha256", "child_rows_sha256", "prep_package_sha256"):
        if field in migration and not _is_sha256_hex(migration.get(field)):
            fails.append({
                "code": "migration_receipt_mismatch", "reason": "sha_field_bad_type",
                "field": field, "got_type": type(migration.get(field)).__name__,
            })
    if migration.get("prep_package_sha256") != prep_package_sha256:
        fails.append({
            "code": "migration_receipt_mismatch", "reason": "prep_package_sha_mismatch",
            "expected": prep_package_sha256, "got": migration.get("prep_package_sha256"),
        })
    if migration.get("child_rows_sha256") != child_rows_sha256:
        fails.append({
            "code": "migration_receipt_mismatch", "reason": "child_rows_sha_mismatch",
            "expected": child_rows_sha256, "got": migration.get("child_rows_sha256"),
        })
    if migration.get("parent_rows_sha256") != parent_rows_sha256:
        fails.append({
            "code": "migration_receipt_mismatch", "reason": "parent_rows_sha_mismatch",
            "field": "parent_rows_sha256",
            "expected": parent_rows_sha256, "got": migration.get("parent_rows_sha256"),
        })
    if migration.get("parent_baseline_sha256") != parent_baseline_sha256:
        fails.append({
            "code": "migration_receipt_mismatch", "reason": "parent_baseline_sha_mismatch",
            "field": "parent_baseline_sha256",
            "expected": parent_baseline_sha256, "got": migration.get("parent_baseline_sha256"),
        })
    return fails


def _base_generation_receipt(
    *,
    prep_package_sha256: str,
    tool_sha256_at_authoring: str,
    migration_carrier_sha256: str,
    v2_rows_sha256: str,
):
    return {
        "generation": "v2",
        "schema_rows": GENERATION_V2_LITERALS["schema_rows"],
        "schema_baseline": GENERATION_V2_LITERALS["schema_baseline"],
        "baseline_name": GENERATION_V2_LITERALS["baseline_name"],
        "baseline_head": GENERATION_V2_LITERALS["baseline_head"],
        "tool_sha256_at_authoring": tool_sha256_at_authoring,
        "migration_carrier_sha256": migration_carrier_sha256,
        "parent_generation": "v1",
        "parent_rows_sha256": PIN_V1_PARENT_ROWS_SHA256,
        "parent_baseline_sha256": PIN_V1_PARENT_BASELINE_SHA256,
        "prep_package_sha256": prep_package_sha256,
        "ast_allowlist_A4_allowed_over_150": list(PIN_A4_ALLOWED_OVER_150),
        "v2_rows_sha256": v2_rows_sha256,
    }


def run_slice_a_exact_set_and_parent_hash_negatives(
    *,
    prep_package_sha256: str,
    child_rows_sha256: str,
    tool_sha256_at_authoring: str | None = None,
    migration_carrier_sha256: str | None = None,
    good_generation: dict | None = None,
):
    """Executable pure-call suite for §7.6a negatives (key/type/literal/parent/live binds)."""
    def _reasons(fails):
        return {f.get("reason") for f in (fails or []) if f.get("reason")}

    def _codes(fails):
        return sorted({f.get("code") for f in (fails or []) if f.get("code")})

    def _record(name, fails, fam, expected_reason, prereg_name=None):
        reasons = _reasons(fails)
        rename = {
            "gen_extra": "v2_gen_extra_field", "gen_missing": "v2_gen_missing_field",
            "mig_extra": "v2_mig_extra_field", "mig_missing": "v2_mig_missing_field",
            "mig_literal": "v2_mig_literal_mismatch",
            "mig_prep_mismatch": "v2_mig_prep_mismatch",
            "mig_child_mismatch": "v2_mig_child_mismatch",
        }
        preg = prereg_name or rename.get(name, "v2_" + name)
        return {
            "name": name, "prereg_name": preg, "family": fam,
            "expected_reason": expected_reason,
            "observed_reasons": sorted(reasons), "observed_codes": _codes(fails),
            "fail_n": len(fails or []), "ok": expected_reason in reasons,
        }

    tool = tool_sha256_at_authoring or ("e" * 64)
    mig = migration_carrier_sha256 or ("f" * 64)
    child = child_rows_sha256
    if isinstance(good_generation, dict) and set(good_generation) >= GENERATION_V2_REQUIRED_KEYS:
        base_gen = {k: good_generation[k] for k in GENERATION_V2_REQUIRED_KEYS}
    else:
        base_gen = _base_generation_receipt(
            prep_package_sha256=prep_package_sha256,
            tool_sha256_at_authoring=tool,
            migration_carrier_sha256=mig,
            v2_rows_sha256=child,
        )
    bind_kw = dict(
        prep_package_sha256=prep_package_sha256,
        tool_sha256_at_authoring=tool,
        migration_carrier_sha256=mig,
        v2_rows_sha256=child,
    )
    _base_gen_fails = validate_generation_exact_set(base_gen, **bind_kw)
    if _base_gen_fails:
        raise RuntimeError(f"BASE_GEN_INVALID_BEFORE_MUTATION fails={_base_gen_fails}")

    base_mig = {k: MIGRATION_V1_TO_V2_LITERALS.get(k, "a" * 64) for k in MIGRATION_V1_TO_V2_REQUIRED_KEYS}
    base_mig["prep_package_sha256"] = prep_package_sha256
    base_mig["child_rows_sha256"] = child
    base_mig["parent_rows_sha256"] = PIN_V1_PARENT_ROWS_SHA256
    base_mig["parent_baseline_sha256"] = PIN_V1_PARENT_BASELINE_SHA256
    _base_mig_fails = validate_migration_receipt_shape(
        base_mig, prep_package_sha256=prep_package_sha256, child_rows_sha256=child
    )
    if _base_mig_fails:
        raise RuntimeError(f"BASE_MIG_INVALID_BEFORE_MUTATION fails={_base_mig_fails}")

    cases = []
    extra = dict(base_gen); extra["EXTRA_KEY"] = True
    cases.append(_record("gen_extra", validate_generation_exact_set(extra, **bind_kw), "exact_set", "extra_field"))
    missing = dict(base_gen); missing.pop("generation", None)
    cases.append(_record("gen_missing", validate_generation_exact_set(missing, **bind_kw), "exact_set", "required_field_absent"))
    # type-correct wrong values (load-bearing semantic class)
    for name, field, val, reason in [
        ("gen_generation_v9", "generation", "v9", "literal_mismatch"),
        ("gen_parent_generation_v0", "parent_generation", "v0", "literal_mismatch"),
        ("gen_baseline_name_wrong", "baseline_name", "BASELINE_TAMPERED.json", "literal_mismatch"),
        ("gen_baseline_head_wrong", "baseline_head", "deadbeef", "literal_mismatch"),
        ("gen_schema_rows_wrong", "schema_rows", "TAMPERED", "literal_mismatch"),
        ("gen_schema_baseline_wrong", "schema_baseline", "TAMPERED", "literal_mismatch"),
        ("gen_parent_rows_ff", "parent_rows_sha256", "f" * 64, "parent_rows_sha_mismatch"),
        ("gen_bad_type_generation", "generation", 9, "required_field_bad_type"),
    ]:
        bad = dict(base_gen); bad[field] = val
        cases.append(_record(name, validate_generation_exact_set(bad, **bind_kw), "exact_set", reason))

    mig_extra = dict(base_mig); mig_extra["EXTRA"] = 1
    cases.append(_record("mig_extra", validate_migration_receipt_shape(mig_extra, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "exact_set", "extra_field"))
    mig_missing = dict(base_mig); mig_missing.pop("schema", None)
    cases.append(_record("mig_missing", validate_migration_receipt_shape(mig_missing, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "exact_set", "required_field_absent"))
    mig_lit = dict(base_mig); mig_lit["schema"] = "WRONG"
    cases.append(_record("mig_literal", validate_migration_receipt_shape(mig_lit, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "exact_set", "literal_mismatch"))
    mig_prep = dict(base_mig); mig_prep["prep_package_sha256"] = "a" * 64
    cases.append(_record("mig_prep_mismatch", validate_migration_receipt_shape(mig_prep, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "exact_set", "prep_package_sha_mismatch"))
    mig_child = dict(base_mig); mig_child["child_rows_sha256"] = "b" * 64
    cases.append(_record("mig_child_mismatch", validate_migration_receipt_shape(mig_child, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "exact_set", "child_rows_sha_mismatch"))

    for name, field, val, reason in [
        ("mig_parent_rows_ff", "parent_rows_sha256", "f" * 64, "parent_rows_sha_mismatch"),
        ("mig_parent_base_ff", "parent_baseline_sha256", "f" * 64, "parent_baseline_sha_mismatch"),
        ("mig_parent_rows_type", "parent_rows_sha256", 123, "sha_field_bad_type"),
        ("mig_parent_base_mal", "parent_baseline_sha256", "WRONG", "sha_field_bad_type"),
        ("mig_parent_rows_zero", "parent_rows_sha256", "0" * 64, "parent_rows_sha_mismatch"),
    ]:
        bad = dict(base_mig); bad[field] = val
        cases.append(_record(name, validate_migration_receipt_shape(bad, prep_package_sha256=prep_package_sha256, child_rows_sha256=child), "parent_hash", reason))

    exact_cases = [c for c in cases if c["family"] == "exact_set"]
    parent_cases = [c for c in cases if c["family"] == "parent_hash"]
    if len(exact_cases) < 3:
        raise RuntimeError(f"EXACT_SET_FAMILY_TOO_SMALL n={len(exact_cases)}")
    if len(parent_cases) < 3:
        raise RuntimeError(f"PARENT_HASH_FAMILY_TOO_SMALL n={len(parent_cases)}")
    return {
        "n_cases": len(cases),
        "n_fail": sum(1 for c in cases if c["ok"]),
        "exact_set_negatives_observed_fail": all(c["ok"] for c in exact_cases),
        "parent_hash_negatives_observed_fail": all(c["ok"] for c in parent_cases),
        "exact_set_n": len(exact_cases),
        "parent_hash_n": len(parent_cases),
        "cases": cases,
    }
