"""PURE primitive-metric → surface-cell reducer (IMPLEMENT_v5).

No IO, no torch, no wall clock. Derives S* from per-key primitives only.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import APPLICABILITY_MAP

# Strict metric key sets per row class
APPLY_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "post_q_sha256_by_key",
        "post_logical_acc_sha256_by_key",
        "events_equal_by_key",
        "sparse_event_count",
        "q_changed_count_sparse",
        "q_changed_count_dense",
        "s6_geometry",
        "d1_densify_from_sparse_used",
        "builder_receipt_pass",
        "production_sparse_matches_twin",
    }
)
# optional diagnostics allowed but not required
APPLY_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "events_equal",
        "q_match",
        "logical_acc_match",
        "q_changed_match",
        "prestate_digests",
        "physical_carrier_equal_diagnostic",
        "site_tag",
        "production_site",
        "builder_receipt_pass",
        "production_sparse_matches_twin",
        "weighted_grad_sha256",
        "q_levels_sha256",
        "recarry_receipt",
        "fixture_recipe_name",
        "parity_fixture_descriptor_sha256",
        "compositional_reduction_holds",
        "reason",
        "production_post_q_sha256_by_key",
        "production_post_logical_acc_sha256_by_key",
        "production_applied_row_identities_sha256_by_key",
        "production_binding",
        "production_reapply_crosscheck",
        "logical_acc_absent_reason",
        "cell_author_error",
        "receipt_extract",
        "twin_post_authoritative_state_payload_sha256",
        "pre_update_authoritative_state_payload_sha256",
        "post_update_authoritative_state_payload_sha256",
        "post_update_payload_sha256",
        "p1b_pass_receipt",
        "weighted_grad_capture_sha256_by_key",
        "named_applied_row_identities_sha256_by_key",
        "mismatches",
        "transition_fields",
        "per_key_field_equal",
        "phase_stream_class",
        "phase_stream_anomaly",
        "phase_events_synthesized",
        "transition_fields_equal",
    }
)
ORACLE_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "events_equal_by_key",
        "events_equal_fused_vs_dense_derived",
        "independent_two_branch_recompute_ok",
        "dense_derived_provenance",
        "d1_densify_from_sparse_used",
        "sparse_vote_authority_mode",
        "votes_by_key_applied",
        "builder_receipt_pass",
        "oracle_mode_on_named_site",
    }
)
ORACLE_OPTIONAL_KEYS: frozenset[str] = frozenset(
    {
        "site_tag",
        "production_site",
        "resolved_mode",
        "builder_receipt_pass",
        "reason",
        "named_builder_returned_receipt",
        "named_builder_oracle_only_keys",
        "named_events_equal_map_present",
        "path_oracle_fallback_used",
        "builder_exception",
        "oracle_mode_on_named_site",
        "phase_stream_class",
        "phase_stream_anomaly",
        "phase_events_synthesized",
        "injective_post_acc_binding_ro_available",
    }
)
CPU_STATIC_METRIC_KEYS: frozenset[str] = frozenset(
    {
        "events_equal_by_key",
        "compositional_reduction_holds",
        "post_q_sha256_by_key",
        "post_logical_acc_sha256_by_key",
        "sparse_event_count",
        "q_changed_count_sparse",
        "q_changed_count_dense",
        "s6_geometry",
        "d1_densify_from_sparse_used",
        "fixture_recipe_name",
        "parity_fixture_descriptor_sha256",
        "recarry_receipt",
    }
)
CPU_STATIC_OPTIONAL: frozenset[str] = frozenset(
    {
        "events_equal",
        "q_match",
        "logical_acc_match",
        "q_changed_match",
        "prestate_digests",
        "physical_carrier_equal_diagnostic",
        "weighted_grad_sha256",
        "q_levels_sha256",
        "production_site",
    }
)


def _require_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"metric_not_mapping:{name}")
    return value


def _require_int(name: str, value: Any) -> int:
    if type(value) is not int:
        # reject bool (bool is int subclass)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"metric_not_int:{name}")
    return int(value)


def _sha_pair_map(name: str, value: Any) -> dict[str, dict[str, str]]:
    m = _require_mapping(name, value)
    out: dict[str, dict[str, str]] = {}
    for k, v in m.items():
        vm = _require_mapping(f"{name}[{k}]", v)
        if set(vm.keys()) != {"sparse", "dense"}:
            raise ValueError(f"sha_pair_keys:{name}[{k}]")
        for side in ("sparse", "dense"):
            if type(vm[side]) is not str or len(vm[side]) != 64:
                raise ValueError(f"sha_pair_bad:{name}[{k}].{side}")
        out[str(k)] = {"sparse": str(vm["sparse"]), "dense": str(vm["dense"])}
    return out


def _bool_map(name: str, value: Any) -> dict[str, bool]:
    m = _require_mapping(name, value)
    out: dict[str, bool] = {}
    for k, v in m.items():
        if type(v) is not bool:
            raise ValueError(f"bool_map_non_bool:{name}[{k}]")
        out[str(k)] = v
    return out


def validate_metrics_schema(*, gating_row: str, metrics: Mapping[str, Any]) -> None:
    if not isinstance(metrics, Mapping):
        raise ValueError("metrics_not_mapping")
    keys = set(metrics.keys())
    if gating_row == "G_CPU_STATIC_AB":
        required, optional = CPU_STATIC_METRIC_KEYS, CPU_STATIC_OPTIONAL
    elif gating_row.startswith("G_CUDA_ORACLE_"):
        required, optional = ORACLE_METRIC_KEYS, ORACLE_OPTIONAL_KEYS
    elif gating_row.startswith("G_CUDA_"):
        required, optional = APPLY_METRIC_KEYS, APPLY_OPTIONAL_KEYS
    else:
        raise ValueError(f"unknown_gating_row:{gating_row}")
    missing = required - keys
    extra = keys - required - optional
    if missing:
        raise ValueError(f"metrics_missing_keys:{gating_row}:{sorted(missing)}")
    if extra:
        raise ValueError(f"metrics_extra_keys:{gating_row}:{sorted(extra)}")


def _per_key_sha_equal(pairs: Mapping[str, Mapping[str, str]], universe: Sequence[str]) -> bool:
    if set(pairs.keys()) != set(universe):
        return False
    return all(pairs[k]["sparse"] == pairs[k]["dense"] for k in universe)


def recompute_surface_cells_from_primitives(
    *,
    gating_row: str,
    metrics: Mapping[str, Any],
    key_universe: Sequence[str],
    fixture_contract_raw_fail: bool,
) -> dict[str, bool]:
    """Derive every applicable surface from per-key primitives (never summary booleans)."""
    validate_metrics_schema(gating_row=gating_row, metrics=metrics)
    universe = [str(k) for k in key_universe]
    if len(universe) != len(set(universe)):
        raise ValueError("key_universe_duplicate")
    out: dict[str, bool] = {}
    surfaces = APPLICABILITY_MAP[gating_row]

    if "s1" in surfaces:
        eebk = _bool_map("events_equal_by_key", metrics["events_equal_by_key"])
        if set(eebk.keys()) != set(universe):
            raise ValueError("s1_events_equal_key_set_mismatch")
        # optional aggregate consistency
        if "events_equal" in metrics and type(metrics["events_equal"]) is bool:
            if metrics["events_equal"] is not all(eebk.values()):
                raise ValueError("aggregate_vs_primitive_mismatch:events_equal")
        out["s1"] = bool(universe) and all(eebk[k] for k in universe)

    if "s2" in surfaces:
        if type(metrics["compositional_reduction_holds"]) is not bool:
            raise ValueError("s2_not_bool")
        out["s2"] = bool(metrics["compositional_reduction_holds"] is True)

    if "s3" in surfaces:
        q_pairs = _sha_pair_map("post_q_sha256_by_key", metrics["post_q_sha256_by_key"])
        a_pairs = _sha_pair_map(
            "post_logical_acc_sha256_by_key", metrics["post_logical_acc_sha256_by_key"]
        )
        if set(q_pairs.keys()) != set(universe) or set(a_pairs.keys()) != set(universe):
            raise ValueError("s3_key_set_mismatch")
        q_eq = _per_key_sha_equal(q_pairs, universe)
        a_eq = _per_key_sha_equal(a_pairs, universe)
        qcs = _require_int("q_changed_count_sparse", metrics["q_changed_count_sparse"])
        qcd = _require_int("q_changed_count_dense", metrics["q_changed_count_dense"])
        q_changed_match = qcs == qcd
        # aggregate consistency if present
        for agg, prim in (
            ("q_match", q_eq),
            ("logical_acc_match", a_eq),
            ("q_changed_match", q_changed_match),
        ):
            if agg in metrics and type(metrics[agg]) is bool and metrics[agg] is not prim:
                raise ValueError(f"aggregate_vs_primitive_mismatch:{agg}")
        # production binding required only on CUDA apply rows
        if gating_row.startswith("G_CUDA_") and not gating_row.startswith("G_CUDA_ORACLE_"):
            if type(metrics.get("builder_receipt_pass")) is not bool:
                raise ValueError("builder_receipt_pass_not_bool")
            if type(metrics.get("production_sparse_matches_twin")) is not bool:
                raise ValueError("production_sparse_matches_twin_not_bool")
            binding_ok = (
                metrics["builder_receipt_pass"] is True
                and metrics["production_sparse_matches_twin"] is True
            )
        else:
            binding_ok = True
        out["s3"] = bool(
            bool(universe)
            and q_eq
            and a_eq
            and q_changed_match
            and binding_ok
            and not fixture_contract_raw_fail
        )

    if "s4" in surfaces:
        sec = _require_int("sparse_event_count", metrics["sparse_event_count"])
        qcs = _require_int("q_changed_count_sparse", metrics["q_changed_count_sparse"])
        out["s4"] = bool(sec > 0 and qcs > 0)

    if "s5" in surfaces:
        eebk = _bool_map("events_equal_by_key", metrics["events_equal_by_key"])
        if set(eebk.keys()) != set(universe):
            raise ValueError("s5_events_equal_key_set_mismatch")
        if type(metrics["events_equal_fused_vs_dense_derived"]) is not bool:
            raise ValueError("s5_fused_vs_dense_not_bool")
        if type(metrics["independent_two_branch_recompute_ok"]) is not bool:
            raise ValueError("s5_independent_not_bool")
        if metrics.get("dense_derived_provenance") != "two_branch_parallel_dense_vote_derivation":
            raise ValueError("s5_bad_provenance")
        if metrics.get("d1_densify_from_sparse_used") is not False:
            raise ValueError("s5_d1_used")
        prim = bool(universe) and all(eebk[k] for k in universe)
        if metrics["events_equal_fused_vs_dense_derived"] is not prim:
            raise ValueError("aggregate_vs_primitive_mismatch:events_equal_fused_vs_dense_derived")
        if type(metrics.get("builder_receipt_pass")) is not bool:
            raise ValueError("oracle_builder_receipt_pass_not_bool")
        if metrics.get("oracle_mode_on_named_site") is not True:
            raise ValueError("oracle_mode_not_on_named_site")
        if metrics.get("sparse_vote_authority_mode") != "oracle_on":
            raise ValueError("oracle_mode_not_oracle_on")
        out["s5"] = bool(
            prim
            and metrics["independent_two_branch_recompute_ok"] is True
            and metrics["events_equal_fused_vs_dense_derived"] is True
            and metrics["builder_receipt_pass"] is True
            and metrics["oracle_mode_on_named_site"] is True
        )

    if "s6" in surfaces:
        geom = _require_mapping("s6_geometry", metrics["s6_geometry"])
        out["s6"] = bool(
            geom.get("votes_by_key_applied") is None
            and geom.get("sparse_vote_authority_only") is True
            and list(geom.get("transient_over2_tensors") or []) == ["weighted_grad"]
            and geom.get("oracle_only_absent_on_fused") is True
            and not fixture_contract_raw_fail
        )

    if set(out.keys()) != set(surfaces):
        raise ValueError(f"surface_set_incomplete:{sorted(out)}")
    return out


def validate_required_key_universe(
    *,
    required_key_set: Sequence[str],
    row_key_universes: Mapping[str, Sequence[str]],
    per_key_maps: Sequence[Mapping[str, Any]] | None = None,
) -> list[str]:
    """Require nonempty required set; exact equality vs every row universe and per-key maps."""
    if required_key_set is None:
        raise ValueError("required_key_set_required")
    keys = [str(k) for k in required_key_set]
    if not keys:
        raise ValueError("required_key_set_empty")
    if len(keys) != len(set(keys)):
        raise ValueError("required_key_set_duplicate")
    req = set(keys)
    for row, univ in row_key_universes.items():
        u = set(str(x) for x in univ)
        if u != req:
            raise ValueError(
                f"key_universe_mismatch row={row} required={sorted(req)} row={sorted(u)}"
            )
    for i, m in enumerate(per_key_maps or []):
        if set(str(x) for x in m.keys()) != req:
            raise ValueError(f"per_key_map_key_set_mismatch:{i}")
    return keys
