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

# Post-state-determining fields present on both named + re-apply proofs (RO).
# Named-receipt RO investigation (IMPLEMENT_v14):
# (a) full sparse event carrier hash — NOT on named receipt (only event counts/mode flags)
# (b) full bounded-acc payload / decoded logical-int16 — NOT on named receipt
#     (bounded_accumulator_summary_after has raw_arrays_included=False; no decode sha on fused)
# Until TSA/BDL expose one of these RO, injective post-acc binding is unavailable.
INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE: bool = False
INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES: tuple[str, ...] = (
    "named_receipt.full_sparse_event_carrier_or_hash_per_key",
    "named_receipt.full_bounded_accumulator_payload_or_decoded_logical_int16_sha_per_key",
)
INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON: str = (
    "neither_full_sparse_event_carrier_hash_nor_full_bounded_acc_payload_hash_on_named_receipt"
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


def crosscheck_production_q_vs_receipt_proof(
    *,
    production_post_q_sha256_by_key: Mapping[str, str],
    receipt_proof_by_key: Mapping[str, Any],
    builder_receipt_pass: bool = False,
    reapply_proof_by_key: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Full named-vs-reapply transition-proof equality (IMPLEMENT_v12).

    When builder_receipt_pass is True: require reapply_proof_by_key covering the
    full production key universe, and EXACT equality on every TRANSITION_PROOF_FIELDS
    entry per key. Fail-closed on missing/partial fields. q-only match is insufficient.
    """
    prod_keys = {str(k) for k in production_post_q_sha256_by_key}
    if builder_receipt_pass is not True:
        # no-pass: soft-ok; bind gate fails the row separately
        return {
            "crosscheck_ok": True,
            "reason": "builder_not_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
        }
    if not receipt_proof_by_key:
        return {
            "crosscheck_ok": False,
            "reason": "no_receipt_proof_to_crosscheck_while_builder_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
        }
    reapply = dict(reapply_proof_by_key or {})
    if not reapply:
        return {
            "crosscheck_ok": False,
            "reason": "no_reapply_proof_while_builder_pass",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
        }
    if set(str(k) for k in receipt_proof_by_key) != prod_keys:
        return {
            "crosscheck_ok": False,
            "reason": "receipt_proof_key_set_mismatch",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
        }
    if set(str(k) for k in reapply) != prod_keys:
        return {
            "crosscheck_ok": False,
            "reason": "reapply_proof_key_set_mismatch",
            "per_key_q_equal": {},
            "per_key_field_equal": {},
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
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
            # dict summaries compared by equality
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
        if not all(field_eq.values()):
            # already recorded mismatches
            pass

    fields_ok = (
        bool(prod_keys)
        and not mismatches
        and all(all(v.values()) for v in per_field.values())
        and all(per_q.values())
    )
    # 12-field equality is NECESSARY but NOT SUFFICIENT for post-acc identity.
    # Injective sufficiency layer: full event-carrier hash or full acc payload hash
    # on the named receipt. Currently not RO-available → fail-closed when pass=True.
    injective_ok = bool(INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE)
    if fields_ok and not injective_ok:
        return {
            "crosscheck_ok": False,
            "reason": INJECTIVE_POST_ACC_BINDING_UNAVAILABLE_REASON,
            "per_key_q_equal": per_q,
            "per_key_field_equal": per_field,
            "mismatches": mismatches[:32],
            "transition_fields": list(TRANSITION_PROOF_FIELDS),
            "transition_fields_equal": True,
            "injective_post_acc_binding_ro_available": False,
            "injective_post_acc_binding_missing_surfaces": list(
                INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
            ),
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
        "injective_post_acc_binding_ro_available": bool(INJECTIVE_POST_ACC_BINDING_RO_AVAILABLE),
        "injective_post_acc_binding_missing_surfaces": list(
            INJECTIVE_POST_ACC_BINDING_MISSING_SURFACES
        ),
    }
