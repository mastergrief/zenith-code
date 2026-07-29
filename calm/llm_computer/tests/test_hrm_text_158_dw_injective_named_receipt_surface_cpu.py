"""CPU tests for DW_INJECTIVE named-receipt binding (PLAN_v7 + gate-2 C1–C4).

Hostile count authoritative: EIGHT core hostiles +
D3/D4 tamper proofs (delete-all evidence, oracle bool/tag).
"""
from __future__ import annotations

import json

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    bounded_accumulator_decoded_sha256,
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import tensor_sha256
from calm.hrm_text_158.native_full_stack.named_receipt_binding import (
    build_named_receipt_path_bindings,
    emit_candidate_bounded_decode_sha256_after,
    oracle_only_serializable_projection,
    require_lowercase_sha256_hex,
    sparse_event_map_binding_sha256,
    validate_named_receipt_evidence_maps,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    SparseVoteExecutionWitness,
    _build_vote_projection_proof,
    assemble_b2_post_resume_update_proof,
    assemble_b3_named_receipt_subproof_fields,
    default_dry_run_rank_vote_spec,
    resolve_sparse_vote_authority_path,
    validate_sparse_vote_authority_landing_receipt,
    SparseVoteAuthorityLandingReceipt,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _events(pairs: dict[int, int]) -> SparseVoteEvents:
    return SparseVoteEvents.from_dict(pairs)


def _rank_spec():
    return default_dry_run_rank_vote_spec()


def _update_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
    )


def test_H_CAP_COLLISION_CARRIER_SEPARABLE():
    shape = (8,)
    a = {"0": 127, "1": 126, "2": 1}
    b = {"0": 127, "1": 126, "2": 2}
    ha = sparse_event_map_binding_sha256(
        {int(k): int(v) for k, v in a.items()}, logical_shape=shape
    )
    hb = sparse_event_map_binding_sha256(
        {int(k): int(v) for k, v in b.items()}, logical_shape=shape
    )
    assert ha != hb
    assert len(ha) == 64 and len(hb) == 64
    require_lowercase_sha256_hex(ha, field="ha")
    require_lowercase_sha256_hex(hb, field="hb")


def test_H_DUPLICATE_INDEX_FAIL_CLOSED():
    events = SparseVoteEvents(
        indices=torch.tensor([0, 0], dtype=torch.int64),
        values=torch.tensor([1, 2], dtype=torch.int16),
    )
    with pytest.raises(ValueError, match="duplicate"):
        sparse_event_map_binding_sha256(events, logical_shape=(4,))


def test_geometry_required_and_in_preimage():
    e = {0: 1}
    h1 = sparse_event_map_binding_sha256(e, logical_shape=(4,))
    h2 = sparse_event_map_binding_sha256(e, logical_shape=(8,))
    assert h1 != h2
    with pytest.raises(ValueError):
        sparse_event_map_binding_sha256(e, logical_shape=())  # type: ignore[arg-type]


def test_H_DECODE_SHA_AUTHORITY_streaming_equals_dense():
    cases = [
        BoundedDeltaAccumulatorState(
            logical_shape=(4,),
            cold_default_value=0,
            hot_exact_indices=(),
            hot_exact_values=(),
        ),
        BoundedDeltaAccumulatorState(
            logical_shape=(4,),
            cold_default_value=3,
            hot_exact_indices=(),
            hot_exact_values=(),
        ),
        BoundedDeltaAccumulatorState(
            logical_shape=(4,),
            cold_default_value=0,
            hot_exact_indices=(),
            hot_exact_values=(),
            cold_exception_indices=(1,),
            cold_exception_values=(5,),
        ),
        BoundedDeltaAccumulatorState(
            logical_shape=(4,),
            cold_default_value=0,
            hot_exact_indices=(2,),
            hot_exact_values=(9,),
            cold_exception_indices=(0,),
            cold_exception_values=(1,),
        ),
    ]
    for st in cases:
        stream = emit_candidate_bounded_decode_sha256_after(st)
        dense = tensor_sha256(decode_bounded_accumulator_to_i16(st))
        assert stream == dense
        assert stream == bounded_accumulator_decoded_sha256(st)
        require_lowercase_sha256_hex(stream, field="stream")


def test_path_bindings_key_set_and_interval():
    events = {"lin": _events({0: 1, 2: -3})}
    shapes = {"lin": (4,)}
    out = build_named_receipt_path_bindings(
        sparse_events_by_key=events,
        logical_shape_by_key=shapes,
        oracle_only=None,
        resolved_mode="fused_only",
    )
    assert set(out["sparse_event_map_binding_sha256_by_key"]) == {"lin"}
    assert out["oracle_only_serializable"] is None
    assert out["s1_binding_interval_seconds"] >= 0.0
    require_lowercase_sha256_hex(
        out["sparse_event_map_binding_sha256_by_key"]["lin"], field="bind"
    )


def test_resolve_returns_path_maps_fused():
    wg = {"k": torch.randn(4, dtype=torch.float32)}
    q = {"k": torch.zeros(4, dtype=torch.int8)}
    rank_spec = _rank_spec()
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=wg,
        q_levels_by_key=q,
        rank_spec=rank_spec,
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    assert "sparse_event_map_binding_sha256_by_key" in path
    assert set(path["sparse_event_map_binding_sha256_by_key"]) == {"k"}
    assert path["oracle_only_serializable"] is None
    assert path["s1_binding_interval_seconds"] >= 0.0
    proof = _build_vote_projection_proof(
        rank_spec=rank_spec,
        update_spec=_update_spec(),
        resolved_mode=str(path["resolved_mode"]),
        total_sparse_events=sum(e.event_count() for e in path["sparse_events_by_key"].values()),
        oracle_only=path.get("oracle_only_serializable"),
        sparse_event_map_binding_sha256_by_key=path["sparse_event_map_binding_sha256_by_key"],
        sparse_event_count_by_key=path["sparse_event_count_by_key"],
        sparse_event_logical_shape_by_key=path["sparse_event_logical_shape_by_key"],
        s1_binding_interval_seconds=path["s1_binding_interval_seconds"],
    )
    assert proof["sparse_event_map_binding_sha256_by_key"] == path[
        "sparse_event_map_binding_sha256_by_key"
    ]
    assert "s1_binding_interval_seconds_diagnostic" in proof
    assert "oracle_only" not in proof


def test_H_MISSING_CARRIER_KEY_FAIL():
    with pytest.raises(ValueError, match="requires path-returned"):
        _build_vote_projection_proof(
            rank_spec=_rank_spec(),
            update_spec=_update_spec(),
            resolved_mode="fused_only",
            total_sparse_events=0,
            oracle_only=None,
            sparse_event_map_binding_sha256_by_key=None,
        )


def test_H_B3_TAG_ONLY_ORACLE_FORBIDDEN_and_relay():
    w = SparseVoteExecutionWitness()
    w.note_path_resolved_mode(SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON)
    assert w.named_receipt_bindings_observed is False
    bindings = build_named_receipt_path_bindings(
        sparse_events_by_key={"k": _events({0: 1})},
        logical_shape_by_key={"k": (2,)},
        oracle_only={
            "events_equal_by_key": {"k": True},
            "events_equal_fused_vs_dense_derived": True,
            "dense_reference_tagged": "oracle_only",
        },
        resolved_mode="oracle_on",
    )
    w.note_named_receipt_bindings(bindings)
    assert w.named_receipt_bindings_observed is True
    assert w.oracle_only_serializable is not None
    assert "events_equal_by_key" in w.oracle_only_serializable


def test_H_B3_MISSING_WITNESS_RELAY_semantics():
    """Invoke production B3 helper; missing witness note hard-errors."""
    w = SparseVoteExecutionWitness()
    w.note_path_resolved_mode(SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY)
    assert w.named_receipt_bindings_observed is False
    with pytest.raises(ValueError, match="missing named_receipt_bindings"):
        assemble_b3_named_receipt_subproof_fields(
            w, resolved_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY
        )


def test_oracle_on_resolve_serializable_projection():
    wg = {"k": torch.randn(4, dtype=torch.float32)}
    q = {"k": torch.zeros(4, dtype=torch.int8)}
    rank_spec = _rank_spec()
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=wg,
        q_levels_by_key=q,
        rank_spec=rank_spec,
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    )
    assert path["oracle_only_serializable"] is not None
    assert "events_equal_by_key" in path["oracle_only_serializable"]
    assert set(path["oracle_only_serializable"]["events_equal_by_key"]) == set(
        path["sparse_event_map_binding_sha256_by_key"]
    )
    # JSON round-trip of serializable projection (no SparseVoteEvents/tensor leakage)
    blob = json.dumps(path["oracle_only_serializable"], sort_keys=True)
    restored = json.loads(blob)
    assert restored["events_equal_by_key"] == path["oracle_only_serializable"][
        "events_equal_by_key"
    ]


def test_H_B2_ORACLE_SERIALIZABLE_json_roundtrip():
    """Production B2 assembler under oracle_on: JSON-safe, no SparseVoteEvents leakage."""
    wg = {"k": torch.randn(4, dtype=torch.float32)}
    q = {"k": torch.zeros(4, dtype=torch.int8)}
    rank_spec = _rank_spec()
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=wg,
        q_levels_by_key=q,
        rank_spec=rank_spec,
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    )
    proof = assemble_b2_post_resume_update_proof(
        path=path,
        loss_finite=True,
        total_sparse_events=sum(e.event_count() for e in path["sparse_events_by_key"].values()),
        step_result_global_summary={
            "q_changed_count": 0,
            "candidate_local_update_pass": True,
            "candidate_dense_decode_used": False,
            "candidate_dense_vote_authority_used": False,
        },
        post_resume_mutated=False,
    )
    assert "oracle_only" in proof
    assert "dense_derived_sparse_events_by_key" not in proof["oracle_only"]
    assert type(proof["oracle_only"]["events_equal_fused_vs_dense_derived"]) is bool
    for v in proof["oracle_only"]["events_equal_by_key"].values():
        assert type(v) is bool
    assert proof["oracle_only"]["dense_reference_tagged"] == "oracle_only"
    blob = json.dumps(proof, sort_keys=True)
    assert "SparseVoteEvents" not in blob
    # no raw tensor/event objects in oracle_only (transient_over2 name may contain "tensor")
    assert "dense_derived_sparse_events_by_key" not in blob
    restored = json.loads(blob)
    assert restored["oracle_only"]["events_equal_by_key"]
    # fused-default ABSENCE on fused path
    path_f = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=wg,
        q_levels_by_key=q,
        rank_spec=rank_spec,
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    )
    proof_f = assemble_b2_post_resume_update_proof(
        path=path_f,
        loss_finite=True,
        total_sparse_events=0,
        step_result_global_summary={
            "q_changed_count": 0,
            "candidate_local_update_pass": True,
            "candidate_dense_decode_used": False,
            "candidate_dense_vote_authority_used": False,
        },
        post_resume_mutated=False,
    )
    assert "oracle_only" not in proof_f


def test_H_SECOND_WITNESS_NOTE_FORBIDDEN():
    w = SparseVoteExecutionWitness()
    bindings = build_named_receipt_path_bindings(
        sparse_events_by_key={"k": _events({0: 1})},
        logical_shape_by_key={"k": (2,)},
        oracle_only=None,
        resolved_mode="fused_only",
    )
    w.note_named_receipt_bindings(bindings)
    with pytest.raises(ValueError, match="second note forbidden"):
        w.note_named_receipt_bindings(bindings)


def test_require_lowercase_sha256_hex_hostile():
    good = "a" * 64
    assert require_lowercase_sha256_hex(good, field="g") == good
    with pytest.raises(ValueError, match="64-hex"):
        require_lowercase_sha256_hex("A" * 64, field="upper")  # uppercase rejected
    with pytest.raises(ValueError, match="64-hex"):
        require_lowercase_sha256_hex("z" * 64, field="nonhex")
    with pytest.raises(ValueError, match="64-hex"):
        require_lowercase_sha256_hex("a" * 63, field="short")


def test_oracle_projection_fail_closed_missing_fused_flag():
    with pytest.raises(ValueError, match="events_equal_fused_vs_dense_derived"):
        oracle_only_serializable_projection(
            {"events_equal_by_key": {"k": True}}  # missing fused flag
        )


def test_validate_named_receipt_evidence_maps_fused_absence():
    evidence = {
        "sparse_event_map_binding_sha256_by_key": {"k": "a" * 64},
        "sparse_event_count_by_key": {"k": 1},
        "sparse_event_logical_shape_by_key": {"k": [2]},
        "s1_binding_interval_seconds_diagnostic": 0.0,
        "oracle_only": {"events_equal_by_key": {"k": True}},
    }
    with pytest.raises(ValueError, match="ABSENCE"):
        validate_named_receipt_evidence_maps(
            evidence, resolved_mode="fused_only", require_oracle_only_key=True
        )


def test_validate_named_receipt_evidence_maps_oracle_on_ok():
    evidence = {
        "sparse_event_map_binding_sha256_by_key": {"k": "a" * 64},
        "sparse_event_count_by_key": {"k": 1},
        "sparse_event_logical_shape_by_key": {"k": [2]},
        "s1_binding_interval_seconds_diagnostic": 0.001,
        "oracle_only": {
            "events_equal_by_key": {"k": True},
            "events_equal_fused_vs_dense_derived": True,
            "dense_reference_tagged": "oracle_only",
        },
    }
    validate_named_receipt_evidence_maps(
        evidence, resolved_mode="oracle_on", require_oracle_only_key=True
    )


def test_b1_rejects_non_hex_binding():
    with pytest.raises(ValueError, match="64-hex"):
        _build_vote_projection_proof(
            rank_spec=_rank_spec(),
            update_spec=_update_spec(),
            resolved_mode="fused_only",
            total_sparse_events=0,
            oracle_only=None,
            sparse_event_map_binding_sha256_by_key={"k": "Z" * 64},
            sparse_event_count_by_key={"k": 0},
            sparse_event_logical_shape_by_key={"k": [2]},
            s1_binding_interval_seconds=0.0,
        )


def test_D3_delete_all_evidence_fail_closed_default():
    """Default landing validation rejects missing named-evidence maps."""
    evidence = {
        "sparse_vote_authority_mode": "fused_only",
        # no binding/count/shape maps
    }
    with pytest.raises(ValueError, match="named-receipt evidence maps required"):
        validate_named_receipt_evidence_maps(
            evidence, resolved_mode="fused_only", require_oracle_only_key=True
        )
    # legacy opt-in still works
    validate_named_receipt_evidence_maps(
        evidence,
        resolved_mode="fused_only",
        allow_legacy_without_named_evidence=True,
    )


def test_D4_oracle_bool_and_tag_tamper():
    base = {
        "sparse_event_map_binding_sha256_by_key": {"k": "a" * 64},
        "sparse_event_count_by_key": {"k": 1},
        "sparse_event_logical_shape_by_key": {"k": [2]},
        "s1_binding_interval_seconds_diagnostic": 0.0,
        "oracle_only": {
            "events_equal_by_key": {"k": True},
            "events_equal_fused_vs_dense_derived": True,
            "dense_reference_tagged": "oracle_only",
        },
    }
    validate_named_receipt_evidence_maps(base, resolved_mode="oracle_on")
    # int for bool
    bad = dict(base)
    bad["oracle_only"] = dict(base["oracle_only"])
    bad["oracle_only"]["events_equal_by_key"] = {"k": 1}
    with pytest.raises(ValueError, match="must be bool"):
        validate_named_receipt_evidence_maps(bad, resolved_mode="oracle_on")
    # string for fused flag
    bad2 = dict(base)
    bad2["oracle_only"] = dict(base["oracle_only"])
    bad2["oracle_only"]["events_equal_fused_vs_dense_derived"] = "true"
    with pytest.raises(ValueError, match="must be bool"):
        validate_named_receipt_evidence_maps(bad2, resolved_mode="oracle_on")
    # mutated tag
    bad3 = dict(base)
    bad3["oracle_only"] = dict(base["oracle_only"])
    bad3["oracle_only"]["dense_reference_tagged"] = "not_oracle"
    with pytest.raises(ValueError, match="dense_reference_tagged"):
        validate_named_receipt_evidence_maps(bad3, resolved_mode="oracle_on")


def test_D2_b3_helper_ok_with_note():
    w = SparseVoteExecutionWitness()
    bindings = build_named_receipt_path_bindings(
        sparse_events_by_key={"k": _events({0: 1})},
        logical_shape_by_key={"k": (2,)},
        oracle_only=None,
        resolved_mode="fused_only",
    )
    w.note_named_receipt_bindings(bindings)
    fields = assemble_b3_named_receipt_subproof_fields(
        w, resolved_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY
    )
    assert "sparse_event_map_binding_sha256_by_key" in fields
    assert "oracle_only" not in fields
