"""LANDS-AB twin fork + sparse-vs-dense reference apply (IMPLEMENT_v3 seam a).

No CLI. Torch/BDL/TSA read-only. No artifact IO.
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaAccumulatorState,
    BoundedDeltaTensorState,
    apply_bounded_delta_vote_step,
    credit_from_weighted_grad,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    make_candidate_authority_tensor_state,
    project_s1_gradient_to_moves,
    rank_bucketed_int16_votes,
    sparse_rank_bucketed_int16_vote_events_from_weighted_grad,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    RANK_SPEC_DIGEST_EXPECTED,
)
from calm.hrm_text_158.native_full_stack.recarry_measurement_evidence import (
    RANK_SPEC_DIGEST_EXPECTED as RECARRY_RANK_DIGEST,
    _rank_spec_digest,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.lands_ab_eval_authoritative_payload import (
    authoritative_sidecar_payload_sha256,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    _default_local_vote_update_spec,
    _sparse_vote_events,
    select_trainer_eligible_bitlinears,
)


def rank_spec_content_digest(spec: Any | None = None) -> str:
    spec = spec if spec is not None else default_dry_run_rank_vote_spec()
    return _rank_spec_digest(spec)


def require_canonical_rank_spec(spec: Any | None = None) -> Any:
    if spec is None:
        spec = default_dry_run_rank_vote_spec()
    digest = rank_spec_content_digest(spec)
    if digest != RANK_SPEC_DIGEST_EXPECTED:
        raise ValueError(
            f"rank_spec_drift: expected={RANK_SPEC_DIGEST_EXPECTED} actual={digest}"
        )
    if digest != RECARRY_RANK_DIGEST:
        raise ValueError("rank_spec_recarry_pin_mismatch")
    live = spec.to_live_dict()
    if live.get("mode") != "ported_s1_rank_bucketed_integer_votes":
        raise ValueError("rank_spec_alternate_mode_rejected")
    if live.get("rank_method") != "grouped_bisect_right":
        raise ValueError("rank_spec_alternate_rank_method_rejected")
    return spec


def two_branch_dense_votes(
    weighted_grad: torch.Tensor,
    q_levels: torch.Tensor,
    rank_spec: Any | None = None,
) -> dict[str, torch.Tensor]:
    rank_spec = require_canonical_rank_spec(rank_spec)
    credit = credit_from_weighted_grad(weighted_grad)
    moves = project_s1_gradient_to_moves(weighted_grad, q_levels)
    votes = rank_bucketed_int16_votes(credit, moves, rank_spec)
    return {"credit": credit, "moves": moves, "votes": votes}


def tensor_sha256(t: torch.Tensor) -> str:
    """Match production TSA/BDL tensor_sha256 (dtype+shape+bytes)."""
    cpu = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    flat = cpu.view(-1)
    if flat.numel() == 0:
        return h.hexdigest()
    element_size = flat.element_size()
    chunk_elems = max(1, (1024 * 1024) // element_size)
    for start in range(0, int(flat.numel()), chunk_elems):
        chunk = flat[start : start + chunk_elems]
        h.update(chunk.numpy().tobytes())
    return h.hexdigest()


def scale_sha256(scale: torch.Tensor) -> str:
    s = scale.detach().cpu().to(torch.float32).reshape(()).contiguous()
    return tensor_sha256(s)


def key_universe_sha256(keys: Sequence[str]) -> str:
    payload = json.dumps(sorted(str(k) for k in keys), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def logical_int16_accumulator(state: BoundedDeltaTensorState) -> torch.Tensor:
    if state.exact_accumulator_shadow is not None:
        return state.exact_accumulator_shadow.detach().cpu().to(torch.int16).contiguous()
    return state.decoded_accumulators().detach().cpu().to(torch.int16).contiguous()


def clone_bounded_accumulator(
    acc: BoundedDeltaAccumulatorState,
) -> BoundedDeltaAccumulatorState:
    return BoundedDeltaAccumulatorState(
        logical_shape=tuple(int(x) for x in acc.logical_shape),
        cold_default_value=int(acc.cold_default_value),
        hot_exact_indices=tuple(int(i) for i in acc.hot_exact_indices),
        hot_exact_values=tuple(int(v) for v in acc.hot_exact_values),
        cold_exception_indices=tuple(int(i) for i in acc.cold_exception_indices),
        cold_exception_values=tuple(int(v) for v in acc.cold_exception_values),
        candidate_name=str(acc.candidate_name),
        raw_arrays_included=bool(acc.raw_arrays_included),
    )


def build_twin_states_from_prior(
    prior_states: Mapping[str, BoundedDeltaTensorState],
) -> dict[str, dict[str, BoundedDeltaTensorState]]:
    sparse: dict[str, BoundedDeltaTensorState] = {}
    dense: dict[str, BoundedDeltaTensorState] = {}
    for key, prior in sorted(prior_states.items()):
        q = prior.q_levels.detach().cpu().to(torch.int8).contiguous().clone()
        scale = prior.frozen_scale.detach().cpu().to(torch.float32).reshape(()).clone()
        logical = logical_int16_accumulator(prior).clone()
        sparse[key] = make_candidate_authority_tensor_state(
            prior,
            q_levels=q.clone(),
            bounded_accumulator=clone_bounded_accumulator(prior.bounded_accumulator),
        )
        if sparse[key].exact_accumulator_shadow is not None:
            raise ValueError("sparse twin must have exact_accumulator_shadow=None")
        if sparse[key].bounded_accumulator is prior.bounded_accumulator:
            raise ValueError("sparse twin reused prior.bounded_accumulator object")
        dense[key] = make_bounded_tensor_state(
            prior.state_key,
            q_levels=q.clone(),
            frozen_scale=scale.clone(),
            accumulators=logical.clone(),
        )
        if dense[key].exact_accumulator_shadow is None:
            raise ValueError("dense twin missing exact_accumulator_shadow")
        if not torch.equal(
            sparse[key].frozen_scale.detach().cpu().to(torch.float32).reshape(()),
            dense[key].frozen_scale.detach().cpu().to(torch.float32).reshape(()),
        ):
            raise ValueError(f"twin scale value mismatch on {key}")
    return {"sparse": sparse, "dense": dense}


def assert_twins_independent(
    sparse: Mapping[str, BoundedDeltaTensorState],
    dense: Mapping[str, BoundedDeltaTensorState],
) -> None:
    if set(sparse) != set(dense):
        raise ValueError("twin key mismatch")
    for k in sparse:
        s, d = sparse[k], dense[k]
        if s.q_levels.data_ptr() == d.q_levels.data_ptr():
            raise ValueError(f"alias q_levels on {k}")
        if s.frozen_scale.data_ptr() == d.frozen_scale.data_ptr():
            raise ValueError(f"alias scale on {k}")
        if d.exact_accumulator_shadow is None:
            raise ValueError(f"dense twin missing shadow on {k}")
        if s.exact_accumulator_shadow is not None:
            raise ValueError(f"sparse twin has unexpected shadow on {k}")
        if s.bounded_accumulator is d.bounded_accumulator:
            raise ValueError(f"alias bounded_accumulator object on {k}")
        if not torch.equal(s.q_levels, d.q_levels):
            raise ValueError(f"pre-q mismatch on {k}")
        if not torch.equal(
            s.frozen_scale.detach().cpu().to(torch.float32).reshape(()),
            d.frozen_scale.detach().cpu().to(torch.float32).reshape(()),
        ):
            raise ValueError(f"pre-scale value mismatch on {k}")
        if not torch.equal(logical_int16_accumulator(s), logical_int16_accumulator(d)):
            raise ValueError(f"pre-logical-acc mismatch on {k}")


def prestate_digests(
    sparse: Mapping[str, BoundedDeltaTensorState],
    dense: Mapping[str, BoundedDeltaTensorState],
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k in sorted(sparse):
        s, d = sparse[k], dense[k]
        out[k] = {
            "sparse_q_sha256": tensor_sha256(s.q_levels),
            "dense_q_sha256": tensor_sha256(d.q_levels),
            "sparse_scale_sha256": scale_sha256(s.frozen_scale),
            "dense_scale_sha256": scale_sha256(d.frozen_scale),
            "sparse_logical_acc_sha256": tensor_sha256(logical_int16_accumulator(s)),
            "dense_logical_acc_sha256": tensor_sha256(logical_int16_accumulator(d)),
            "q_equal": bool(torch.equal(s.q_levels, d.q_levels)),
            "scale_equal": bool(
                torch.equal(
                    s.frozen_scale.detach().cpu().to(torch.float32).reshape(()),
                    d.frozen_scale.detach().cpu().to(torch.float32).reshape(()),
                )
            ),
            "logical_acc_equal": bool(
                torch.equal(logical_int16_accumulator(s), logical_int16_accumulator(d))
            ),
        }
    return out


def required_keys_for_model(model: torch.nn.Module) -> frozenset[str]:
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    return frozenset(eligible.keys())


def assert_key_universe_complete(required: frozenset[str], observed: set[str]) -> None:
    if observed != set(required):
        raise ValueError(
            f"key_universe incomplete: missing={sorted(required - observed)} "
            f"extra={sorted(observed - required)}"
        )


def events_maps_equal(a: SparseVoteEvents, b: SparseVoteEvents) -> bool:
    if a.event_count() != b.event_count():
        return False
    if a.event_count() == 0:
        return True
    return bool(torch.equal(a.indices, b.indices) and torch.equal(a.values, b.values))


def clone_prior_states(
    states: Mapping[str, BoundedDeltaTensorState],
) -> dict[str, BoundedDeltaTensorState]:
    out: dict[str, BoundedDeltaTensorState] = {}
    for k, st in sorted(states.items()):
        q = st.q_levels.detach().cpu().to(torch.int8).contiguous().clone()
        out[k] = make_candidate_authority_tensor_state(
            st,
            q_levels=q,
            bounded_accumulator=clone_bounded_accumulator(st.bounded_accumulator),
        )
    return out



def run_twin_apply_compare(
    *,
    prior_states: Mapping[str, BoundedDeltaTensorState],
    weighted_grad_by_key: Mapping[str, torch.Tensor],
    rank_spec: Any | None = None,
) -> dict[str, Any]:
    """Sparse fused authority apply vs true dense-votes apply on twin prestates."""
    rank_spec = require_canonical_rank_spec(rank_spec)
    keys = frozenset(prior_states)
    if set(weighted_grad_by_key) != set(keys):
        raise ValueError(
            f"key_universe incomplete: missing={sorted(keys - set(weighted_grad_by_key))} "
            f"extra={sorted(set(weighted_grad_by_key) - keys)}"
        )
    empty = {
        "fixture_contract_raw_fail": True,
        "reason": "empty_eligible",
        "events_equal": False,
        "s1_pass": False,
        "s3_pass": False,
        "s4_pass": False,
        "s6_pass": False,
        "sparse_event_count": 0,
        "q_changed_count_sparse": 0,
        "q_changed_count_dense": 0,
        "keys": [],
        "prestate_digests": {},
        "physical_carrier_equal_diagnostic": None,
        "d1_densify_from_sparse_used": False,
        "q_match": False,
        "logical_acc_match": False,
        "q_changed_match": False,
        "s6_geometry": {
            "votes_by_key_applied": None,
            "sparse_vote_authority_only": True,
            "transient_over2_tensors": ["weighted_grad"],
            "oracle_only_absent_on_fused": True,
        },
        "events_equal_by_key": {},
        "post_q_sha256_by_key": {},
        "post_logical_acc_sha256_by_key": {},
        "twin_post_authoritative_state_payload_sha256": "",
        "twin_pre_authoritative_state_payload_sha256": "",
        "rank_spec_digest": RANK_SPEC_DIGEST_EXPECTED,
    }
    if not keys:
        return empty

    twins = build_twin_states_from_prior(prior_states)
    fixture_contract_raw_fail = False
    try:
        assert_twins_independent(twins["sparse"], twins["dense"])
    except ValueError:
        fixture_contract_raw_fail = True

    digests = prestate_digests(twins["sparse"], twins["dense"])
    if any(
        not (digests[k]["q_equal"] and digests[k]["scale_equal"] and digests[k]["logical_acc_equal"])
        for k in digests
    ):
        fixture_contract_raw_fail = True

    update_spec = _default_local_vote_update_spec()
    vote_specs = {k: update_spec for k in keys}

    sparse_events: dict[str, SparseVoteEvents] = {}
    dense_votes: dict[str, torch.Tensor] = {}
    dense_derived_events: dict[str, SparseVoteEvents] = {}
    for k in sorted(keys):
        q = prior_states[k].q_levels
        wg = weighted_grad_by_key[k]
        sparse_events[k] = sparse_rank_bucketed_int16_vote_events_from_weighted_grad(
            wg, q, rank_spec
        )
        dens = two_branch_dense_votes(wg, q, rank_spec)
        dense_votes[k] = dens["votes"]
        dense_derived_events[k] = _sparse_vote_events(dens["votes"])

    events_equal = all(
        events_maps_equal(sparse_events[k], dense_derived_events[k]) for k in keys
    )

    sparse_result = apply_bounded_delta_vote_step(
        {k: twins["sparse"][k] for k in keys},
        None,
        vote_specs,
        candidate_mode="accumulator_substitute.local_vote_update_executable",
        candidate_sparse_vote_events_by_key=sparse_events,
        candidate_oracle_control_enabled=False,
        sparse_vote_authority_only=True,
    )
    dense_result = apply_bounded_delta_vote_step(
        {k: twins["dense"][k] for k in keys},
        dense_votes,
        vote_specs,
        sparse_vote_authority_only=False,
        candidate_oracle_control_enabled=False,
    )

    q_match = all(
        torch.equal(
            sparse_result.tensor_states[k].q_levels,
            dense_result.tensor_states[k].q_levels,
        )
        for k in keys
    )
    logical_acc_match = all(
        torch.equal(
            logical_int16_accumulator(sparse_result.tensor_states[k]),
            logical_int16_accumulator(dense_result.tensor_states[k]),
        )
        for k in keys
    )
    qcs = int(sparse_result.global_summary.get("q_changed_count", 0))
    qcd = int(dense_result.global_summary.get("q_changed_count", 0))
    q_changed_match = qcs == qcd
    sparse_event_count = int(sum(e.event_count() for e in sparse_events.values()))

    s6_geometry = {
        "votes_by_key_applied": None,
        "sparse_vote_authority_only": True,
        "transient_over2_tensors": ["weighted_grad"],
        "oracle_only_absent_on_fused": True,
    }
    s6_pass = (
        s6_geometry["votes_by_key_applied"] is None
        and s6_geometry["sparse_vote_authority_only"] is True
        and s6_geometry["transient_over2_tensors"] == ["weighted_grad"]
        and not fixture_contract_raw_fail
    )
    s3_pass = bool(q_match and logical_acc_match and q_changed_match and not fixture_contract_raw_fail)
    s4_pass = bool(sparse_event_count > 0 and qcs > 0)
    s1_pass = bool(events_equal)

    physical_diag: dict[str, Any] = {}
    for k in keys:
        sp_acc = sparse_result.tensor_states[k].bounded_accumulator
        dn_sh = dense_result.tensor_states[k].exact_accumulator_shadow
        physical_diag[k] = {
            "reported": True,
            "dense_shadow_sha256": tensor_sha256(dn_sh) if dn_sh is not None else None,
            "sparse_bounded_hot_count": len(sp_acc.hot_exact_indices),
            "gating": False,
        }

    return {
        "fixture_contract_raw_fail": bool(fixture_contract_raw_fail),
        "events_equal": bool(events_equal),
        "s1_pass": s1_pass,
        "q_match": bool(q_match),
        "logical_acc_match": bool(logical_acc_match),
        "q_changed_match": bool(q_changed_match),
        "s3_pass": s3_pass,
        "s4_pass": s4_pass,
        "s6_pass": s6_pass,
        "sparse_event_count": sparse_event_count,
        "q_changed_count_sparse": qcs,
        "q_changed_count_dense": qcd,
        "keys": sorted(keys),
        "prestate_digests": digests,
        "s6_geometry": s6_geometry,
        "physical_carrier_equal_diagnostic": physical_diag,
        "d1_densify_from_sparse_used": False,
        "rank_spec_digest": rank_spec_content_digest(rank_spec),
        "events_equal_by_key": {
            k: events_maps_equal(sparse_events[k], dense_derived_events[k]) for k in sorted(keys)
        },
        "post_q_sha256_by_key": {
            k: {
                "sparse": tensor_sha256(sparse_result.tensor_states[k].q_levels),
                "dense": tensor_sha256(dense_result.tensor_states[k].q_levels),
            }
            for k in sorted(keys)
        },
        "post_logical_acc_sha256_by_key": {
            k: {
                "sparse": tensor_sha256(logical_int16_accumulator(sparse_result.tensor_states[k])),
                "dense": tensor_sha256(logical_int16_accumulator(dense_result.tensor_states[k])),
            }
            for k in sorted(keys)
        },
        # Canonical sidecar shas (production serializer, twin states substituted).
        # step=1 matches roundtrip/landing post-update convention (prior step=0 → post step=1).
        "twin_pre_authoritative_state_payload_sha256": authoritative_sidecar_payload_sha256(
            {k: twins["sparse"][k] for k in keys},
            step=0,
        ),
        "twin_post_authoritative_state_payload_sha256": authoritative_sidecar_payload_sha256(
            sparse_result.tensor_states,
            step=1,
        ),
    }
