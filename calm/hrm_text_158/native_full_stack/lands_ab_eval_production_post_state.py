"""RO production post-state + named-vs-reapply transition-proof equality (IMPLEMENT_v12).

No TSA edits. Re-executes resolve+apply (same path as fused local-update builder),
decodes logical int16 via TSA/BDA helpers, and fail-closed full-field compares the
named receipt proof against re-apply proof_by_key for every post-state-determining field.
"""
from __future__ import annotations

from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    clone_prior_states,
    require_canonical_rank_spec,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    _default_local_vote_update_spec,
    resolve_sparse_vote_authority_path,
)
from calm.hrm_text_158.native_full_stack.named_receipt_binding import (
    build_sparse_event_binding_by_key,
    logical_shape_by_key_from_q_levels,
    require_lowercase_sha256_hex,
)

# Post-state-determining fields present on both named + re-apply proofs (RO).
# PLAN_v6 Phase A: injective post-acc binding is available via NRB recompute (S1) +
# family S2 (B1 decode / B2-B3 payload VALUE). Flag True only because real compare path exists.
INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE: bool = True
INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES: tuple[str, ...] = ()
INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON: str = (
    "injective_compare_failed_or_maps_absent"
)

TRANSITION_PROOF_FIELDS: tuple[str, ...] = (
    "candidate_q_sha256_after",
    "q_changed_identities_sha256",
    "applied_row_identities_sha256",
    "ordered_applied_row_identities_sha256",
    "applied_directions_sha256",
    "applied_thresholds_sha256",
    "residual_after_threshold_sha256",
    "bounded_accumulator_summary_after",
    "q_changed_count",
    "applied_row_count",
    "event_vote_count",
    "candidate_count",
)


def _candidate_mode() -> str:
    return "accumulator_substitute.local_vote_update_executable"


def logical_int16_from_tensor_state(state: Any) -> torch.Tensor:
    if hasattr(state, "decoded_accumulators"):
        return state.decoded_accumulators().detach().cpu().contiguous()
    return decode_bounded_accumulator_to_i16(state.bounded_accumulator).detach().cpu().contiguous()


def production_fused_apply_post_states(
    *,
    prior_states: Mapping[str, Any],
    weighted_grad_by_key: Mapping[str, torch.Tensor],
    sparse_vote_authority_mode: str = SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
) -> dict[str, Any]:
    states = clone_prior_states(prior_states)
    keys = sorted(states.keys())
    empty = {
        "post_states": {},
        "sparse_events_by_key": {},
        "sparse_event_count": 0,
        "q_changed_count": 0,
        "resolved_mode": sparse_vote_authority_mode,
        "candidate_local_update_pass": False,
        "proof_by_key": {},
        "fixture_contract_raw_fail": True,
        "reason": "empty_eligible",
        "path_oracle_only": {},
    }
    if not keys:
        return empty
    if set(weighted_grad_by_key) != set(keys):
        out = dict(empty)
        out["reason"] = "key_universe_incomplete"
        return out
    rank_spec = require_canonical_rank_spec()
    update_spec = _default_local_vote_update_spec()
    vote_specs = {k: update_spec for k in keys}
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key={k: weighted_grad_by_key[k] for k in keys},
        q_levels_by_key={k: states[k].q_levels for k in keys},
        rank_spec=rank_spec,
        sparse_vote_authority_mode=sparse_vote_authority_mode,
    )
    sparse_events_by_key = dict(path["sparse_events_by_key"])
    step_result = apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_mode=_candidate_mode(),
        candidate_sparse_vote_events_by_key=sparse_events_by_key,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    proof_by_key = dict(
        step_result.global_summary.get("candidate_local_update_proof_by_key") or {}
    )
    return {
        "post_states": dict(step_result.tensor_states),
        "sparse_events_by_key": sparse_events_by_key,
        "sparse_event_count": int(sum(e.event_count() for e in sparse_events_by_key.values())),
        "q_changed_count": int(step_result.global_summary.get("q_changed_count", 0)),
        "resolved_mode": str(path.get("resolved_mode") or sparse_vote_authority_mode),
        "candidate_local_update_pass": bool(
            step_result.global_summary.get("candidate_local_update_pass")
        ),
        "proof_by_key": proof_by_key,
        "fixture_contract_raw_fail": False,
        "reason": "",
        "path_oracle_only": dict(path.get("oracle_only") or {}),
    }


def production_post_q_and_logical_acc_sha256_by_key(
    post_states: Mapping[str, Any],
) -> dict[str, Any]:
    post_q: dict[str, str] = {}
    post_acc: dict[str, str] = {}
    for k in sorted(post_states.keys()):
        st = post_states[k]
        post_q[str(k)] = tensor_sha256(st.q_levels.detach().cpu().contiguous())
        post_acc[str(k)] = tensor_sha256(logical_int16_from_tensor_state(st))
    return {
        "production_post_q_sha256_by_key": post_q,
        "production_post_logical_acc_sha256_by_key": post_acc,
        "logical_acc_ro_observable": bool(post_q)
        and set(post_q.keys()) == set(post_acc.keys())
        and all(len(v) == 64 for v in post_acc.values()),
    }


def applied_identities_from_proof_by_key(
    proof_by_key: Mapping[str, Any],
) -> dict[str, str]:
    out: dict[str, str] = {}
    for k, proof in sorted(proof_by_key.items()):
        if not isinstance(proof, Mapping):
            continue
        aid = proof.get("applied_row_identities_sha256") or proof.get(
            "ordered_applied_row_identities_sha256"
        )
        if isinstance(aid, str) and len(aid) == 64:
            out[str(k)] = aid
    return out




def _norm_shape_map(shape_by_key: Mapping[str, Any] | None) -> dict[str, tuple[int, ...]]:
    out: dict[str, tuple[int, ...]] = {}
    if not shape_by_key:
        return out
    for k, v in shape_by_key.items():
        if isinstance(v, (list, tuple)):
            out[str(k)] = tuple(int(d) for d in v)
        else:
            raise ValueError(f"shape for {k!r} must be list/tuple, got {type(v)}")
    return out


def _norm_sha_map(m: Mapping[str, Any] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    if not m:
        return out
    for k, v in m.items():
        if isinstance(v, str) and len(v) == 64 and all(c in "0123456789abcdef" for c in v):
            out[str(k)] = v
    return out


def recompute_s1_and_compare(
    *,
    sparse_events_by_key: Mapping[str, Any],
    prior_states: Mapping[str, Any],
    named_binding_by_key: Mapping[str, str] | None,
    named_shape_by_key: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """S1 injective authority: NRB recompute from independent events+shapes only.

    Named shape is equality-only (never recompute input). PLAN_v6 G1 + D5:
    on compare paths require exact non-empty FOUR-WAY key equality:
    independent production keys == reapply-event keys == named binding keys
    == named shape keys. Absent named-shape map FAILS (not soft-ok).
    """
    q_levels = {str(k): prior_states[k].q_levels for k in prior_states}
    independent_shapes = logical_shape_by_key_from_q_levels(q_levels)
    events = {str(k): sparse_events_by_key[k] for k in sparse_events_by_key}
    indep_keys = set(independent_shapes)
    event_keys = set(events)
    named_bind = _norm_sha_map(named_binding_by_key)
    named_shapes = _norm_shape_map(named_shape_by_key)
    named_bind_keys = set(named_bind)
    named_shape_keys = set(named_shapes)
    # D5: four-way exact non-empty key equality (no intersection fallback)
    keys_nonempty = bool(indep_keys) and bool(event_keys) and bool(named_bind_keys) and bool(named_shape_keys)
    keys_equal = (
        keys_nonempty
        and indep_keys == event_keys == named_bind_keys == named_shape_keys
    )
    if keys_equal:
        # recompute over the exact shared key set using independent shapes only
        keys_sorted = sorted(indep_keys)
        recomputed = build_sparse_event_binding_by_key(
            {k: events[k] for k in keys_sorted},
            logical_shape_by_key={k: independent_shapes[k] for k in keys_sorted},
        )
        s1_binding_ok = named_bind == recomputed
        shape_ok = all(named_shapes[k] == independent_shapes[k] for k in keys_sorted)
    else:
        # still compute recomputed for diagnostics on the intersection only
        common = sorted(event_keys & indep_keys)
        if common:
            recomputed = build_sparse_event_binding_by_key(
                {k: events[k] for k in common},
                logical_shape_by_key={k: independent_shapes[k] for k in common},
            )
        else:
            recomputed = {}
        s1_binding_ok = False
        shape_ok = False
    return {
        "recomputed_binding_by_key": recomputed,
        "independent_shapes_by_key": {k: list(v) for k, v in independent_shapes.items()},
        "named_binding_by_key": named_bind,
        "named_shapes_by_key": {k: list(v) for k, v in named_shapes.items()},
        "s1_ok": bool(s1_binding_ok and shape_ok and keys_equal),
        "s1_binding_ok": bool(s1_binding_ok and keys_equal),
        "shape_ok": bool(shape_ok and keys_equal),
        "keys_equal": bool(keys_equal),
        "key_sets": {
            "independent": sorted(indep_keys),
            "events": sorted(event_keys),
            "named_binding": sorted(named_bind_keys),
            "named_shape": sorted(named_shape_keys),
        },
        "path_map_equality_diagnostic_only": True,
    }


def evaluate_family_s2(
    *,
    family: str | None,
    named_s2_decode_by_key: Mapping[str, str] | None,
    production_logical_acc_by_key: Mapping[str, str] | None,
    named_post_payload_sha256: str | None,
    twin_or_canonical_post_payload_sha256: str | None,
) -> dict[str, Any]:
    """Per-family S2 authority from existing fields (PLAN_v6)."""
    fam = (family or "").upper()
    if fam in ("B1", "G_CUDA_B1_APPLY", "G_CUDA_ORACLE_B1"):
        named = _norm_sha_map(named_s2_decode_by_key)
        prod = _norm_sha_map(production_logical_acc_by_key)
        ok = bool(named) and bool(prod) and named == prod and set(named) == set(prod)
        return {"family": "B1", "s2_ok": ok, "named_decode": named, "production_logical_acc": prod}
    if fam in ("B2", "G_CUDA_B2_APPLY", "G_CUDA_ORACLE_B2", "B3", "G_CUDA_B3_APPLY", "G_CUDA_ORACLE_B3"):
        n = named_post_payload_sha256 if isinstance(named_post_payload_sha256, str) else ""
        t = twin_or_canonical_post_payload_sha256 if isinstance(twin_or_canonical_post_payload_sha256, str) else ""
        ok = (
            len(n) == 64
            and all(c in "0123456789abcdef" for c in n)
            and len(t) == 64
            and all(c in "0123456789abcdef" for c in t)
            and n == t
        )
        return {
            "family": "B2" if fam.startswith("B2") or "B2" in fam else "B3",
            "s2_ok": ok,
            "named_post_payload_sha256": n,
            "twin_or_canonical_post_payload_sha256": t,
        }
    return {"family": family or "", "s2_ok": False, "reason": "unknown_family"}


def _family_bucket(family: str | None) -> str:
    fam = (family or "").upper()
    if fam in ("B1", "G_CUDA_B1_APPLY", "G_CUDA_ORACLE_B1") or fam.startswith("B1"):
        return "B1"
    if fam in ("B2", "G_CUDA_B2_APPLY", "G_CUDA_ORACLE_B2") or "B2" in fam:
        return "B2"
    if fam in ("B3", "G_CUDA_B3_APPLY", "G_CUDA_ORACLE_B3") or "B3" in fam:
        return "B3"
    return fam or ""


def crosscheck_production_q_vs_receipt_proof(
    *,
    production_post_q_sha256_by_key: Mapping[str, str],
    receipt_proof_by_key: Mapping[str, Any],
    builder_receipt_pass: bool = False,
    reapply_proof_by_key: Mapping[str, Any] | None = None,
    s1_compare: Mapping[str, Any] | None = None,
    s2_compare: Mapping[str, Any] | None = None,
    family: str | None = None,
) -> dict[str, Any]:
    """Family-specific production crosscheck + injective S1/S2 (PLAN_v6 + D1).

    B1 (builder_pass): per-key TRANSITION_PROOF_FIELDS equality + S1 + S2.
    B2/B3 (builder_pass): S1 recompute + payload-VALUE S2 only — do NOT require
    B1-style per-key transition proof (PLAN_v6; no fabricated fields).
    Fail-closed if flag True but maps/compare absent (H_VACUOUS).
    """
    prod_keys = {str(k) for k in production_post_q_sha256_by_key}
    bucket = _family_bucket(family)
    flag = bool(INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE)
    s1 = dict(s1_compare or {})
    s2 = dict(s2_compare or {})

    if builder_receipt_pass is not True:
        return {
            "crosscheck_ok": True,
            "reason": "builder_not_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "family": bucket,
        }

    s1_ok = bool(s1.get("s1_ok")) if s1 else False
    s2_ok = bool(s2.get("s2_ok")) if s2 else False
    maps_present = bool(s1) or bool(s2)
    injective_ok = bool(flag and s1_ok and s2_ok)

    # --- B2/B3 family path: S1+S2 authority only (D1) ---
    if bucket in ("B2", "B3"):
        if flag and not maps_present:
            return {
                "crosscheck_ok": False,
                "reason": "injective_flag_true_without_maps_or_compare",
                "per_key_q_equal": {},
                "per_key_field_equal": {},
                "mismatches": [],
                "transition_fields": list(TRANSITION_PROOF_FIELDS),
                "transition_fields_equal": None,
                "injective_post_acc_binding_ro_available": True,
                "injective_post_acc_binding_missing_surfaces": ["s1_compare", "s2_compare"],
                "s1_compare": s1,
                "s2_compare": s2,
                "family": bucket,
            }
        if not injective_ok:
            reason = (
                "s1_recompute_mismatch"
                if s1 and not s1_ok
                else (
                    "s2_family_mismatch"
                    if s2 and not s2_ok
                    else INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON
                )
            )
            return {
                "crosscheck_ok": False,
                "reason": reason,
                "per_key_q_equal": {},
                "per_key_field_equal": {},
                "mismatches": [],
                "transition_fields": list(TRANSITION_PROOF_FIELDS),
                "transition_fields_equal": None,
                "injective_post_acc_binding_ro_available": flag,
                "injective_post_acc_binding_missing_surfaces": list(
                    INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
                ),
                "s1_compare": s1,
                "s2_compare": s2,
                "family": bucket,
            }
        return {
            "crosscheck_ok": True,
            "reason": "family_s1_s2_injective_post_acc_equal",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "mismatches": [],
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "transition_fields_equal": None,
            "injective_post_acc_binding_ro_available": flag,
            "injective_post_acc_binding_missing_surfaces": list(
                INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
            ),
            "s1_compare": s1,
            "s2_compare": s2,
            "family": bucket,
        }

    # --- B1 / default: transition proof + S1 + S2 ---
    if not receipt_proof_by_key:
        return {
            "crosscheck_ok": False,
            "reason": "no_receipt_proof_to_crosscheck_while_builder_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "family": bucket or "B1",
            "s1_compare": s1,
            "s2_compare": s2,
        }
    reapply = dict(reapply_proof_by_key or {})
    if not reapply:
        return {
            "crosscheck_ok": False,
            "reason": "no_reapply_proof_while_builder_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "family": bucket or "B1",
            "s1_compare": s1,
            "s2_compare": s2,
        }
    if set(str(k) for k in receipt_proof_by_key) != prod_keys:
        return {
            "crosscheck_ok": False,
            "reason": "receipt_proof_key_set_mismatch",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "family": bucket or "B1",
            "s1_compare": s1,
            "s2_compare": s2,
        }
    if set(str(k) for k in reapply) != prod_keys:
        return {
            "crosscheck_ok": False,
            "reason": "reapply_proof_key_set_mismatch",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "family": bucket or "B1",
            "s1_compare": s1,
            "s2_compare": s2,
        }

    per_q: dict[str, bool] = {}
    per_field: dict[str, dict[str, bool]] = {}
    mismatches: list[str] = []
    for k in sorted(prod_keys):
        named = receipt_proof_by_key.get(k) or receipt_proof_by_key.get(str(k))
        rep = reapply.get(k) or reapply.get(str(k))
        if not isinstance(named, Mapping) or not isinstance(rep, Mapping):
            mismatches.append(f"{k}:proof_not_mapping")
            per_field[k] = {}
            per_q[k] = False
            continue
        field_eq: dict[str, bool] = {}
        for f in TRANSITION_PROOF_FIELDS:
            if f not in named or f not in rep:
                field_eq[f] = False
                mismatches.append(f"{k}:{f}:missing")
                continue
            nv, rv = named[f], rep[f]
            field_eq[f] = nv == rv
            if not field_eq[f]:
                mismatches.append(f"{k}:{f}:mismatch")
        per_field[k] = field_eq
        cq_n = named.get("candidate_q_sha256_after")
        cq_p = production_post_q_sha256_by_key.get(k)
        per_q[k] = (
            isinstance(cq_n, str)
            and len(cq_n) == 64
            and cq_n == cq_p
            and field_eq.get("candidate_q_sha256_after") is True
        )

    fields_ok = (
        bool(prod_keys)
        and not mismatches
        and all(all(v.values()) for v in per_field.values())
        and all(per_q.values())
    )
    # H_VACUOUS after transition fields when maps absent (B1)
    if flag and not maps_present:
        return {
            "crosscheck_ok": False,
            "reason": "injective_flag_true_without_maps_or_compare",
            "per_key_q_equal": per_q,
            "per_key_field_equal": per_field,
            "mismatches": mismatches[:32],
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "transition_fields_equal": bool(fields_ok),
            "injective_post_acc_binding_ro_available": True,
            "injective_post_acc_binding_missing_surfaces": ["s1_compare", "s2_compare"],
            "s1_compare": s1,
            "s2_compare": s2,
            "family": bucket or "B1",
        }
    if fields_ok and not injective_ok:
        reason = (
            "s1_recompute_mismatch"
            if s1 and not s1_ok
            else (
                "s2_family_mismatch"
                if s2 and not s2_ok
                else INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON
            )
        )
        return {
            "crosscheck_ok": False,
            "reason": reason,
            "per_key_q_equal": per_q,
            "per_key_field_equal": per_field,
            "mismatches": mismatches[:32],
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "transition_fields_equal": True,
            "injective_post_acc_binding_ro_available": flag,
            "injective_post_acc_binding_missing_surfaces": list(
                INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
            ),
            "s1_compare": s1,
            "s2_compare": s2,
            "family": bucket or "B1",
        }
    ok = bool(fields_ok and injective_ok)
    return {
        "crosscheck_ok": bool(ok),
        "reason": (
            "transition_proof_and_injective_post_acc_equal"
            if ok
            else ("transition_proof_mismatch" if not fields_ok else INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON)
        ),
        "per_key_q_equal": per_q,
        "per_key_field_equal": per_field,
        "mismatches": mismatches[:32],
        "transition_fields": list(TRANSITION_PROOF_FIELDS),
        "transition_fields_equal": bool(fields_ok),
        "injective_post_acc_binding_ro_available": flag,
        "injective_post_acc_binding_missing_surfaces": list(
            INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
        ),
        "s1_compare": s1,
        "s2_compare": s2,
        "family": bucket or "B1",
    }
