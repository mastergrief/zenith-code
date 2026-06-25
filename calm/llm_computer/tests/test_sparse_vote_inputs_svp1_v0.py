"""Exactness + fail-closed tests for SVP1 sparse_vote_inputs sidecars."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_live_carrier import (
    EventCodedAccLiveState,
    observed_surfaces_dict,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.sparse_vote_inputs_svp1 import (
    SPARSE_VOTE_PAIRS_ENCODING,
    build_sparse_vote_inputs_stub,
    decode_sparse_vote_inputs_svp1,
    encode_sparse_vote_inputs_svp1,
    inline_sparse_vote_inputs_by_state_key,
    inline_sparse_votes_from_record,
    resolve_sidecar_path,
    sparse_votes_from_emit_record,
    verify_sparse_vote_inputs_stub,
    write_sidecar_atomically,
)
from calm.hrm_text_158.native_full_stack.v4_live_phase_a_parity_witness import (
    run_phase_a_parity_witness,
)
from calm.hrm_text_158.native_full_stack.votes_emit_collector import (
    VOTES_EMIT_SECTION6_CONTRACT_FIELDS,
    VotesEmitCollector,
    maybe_emit_votes_step_record,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec


def _vote_spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=1,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4096,
        decay_numerator=1,
        decay_denominator=2,
    )


def _assert_decode_matches_inline(votes_by_key: dict[str, torch.Tensor]) -> None:
    inline = inline_sparse_votes_from_record(
        inline_sparse_vote_inputs_by_state_key(votes_by_key)
    )
    sidecar_bytes, per_state, total = encode_sparse_vote_inputs_svp1(votes_by_key)
    decoded, decoded_per_state, decoded_total = decode_sparse_vote_inputs_svp1(sidecar_bytes)
    assert decoded == inline
    assert decoded_per_state == {str(k): int(v) for k, v in per_state.items()}
    assert decoded_total == total


def test_svp1_round_trip_empty_state() -> None:
    votes = {"proj": torch.zeros(8, dtype=torch.int16)}
    _assert_decode_matches_inline(votes)


def test_svp1_round_trip_negative_and_max_int16() -> None:
    votes = {
        "proj": torch.tensor(
            [0, -32768, 0, 32767, 0, -12, 0, 5],
            dtype=torch.int16,
        )
    }
    _assert_decode_matches_inline(votes)


def test_svp1_round_trip_sparse_and_dense() -> None:
    for numel in (64, 1024):
        sparse = torch.zeros(numel, dtype=torch.int16)
        sparse[0] = 7
        sparse[17] = -9
        _assert_decode_matches_inline({"a": sparse})
        dense = torch.randint(-100, 101, (numel,), dtype=torch.int16)
        dense[dense == 0] = 1
        _assert_decode_matches_inline({"b": dense})


def test_svp1_encode_rejects_non_int16_vote_overflow() -> None:
    votes = {"proj": torch.tensor([70000], dtype=torch.int32)}
    with pytest.raises(ValueError, match="int16"):
        encode_sparse_vote_inputs_svp1(votes)


def test_svp1_decode_fail_closed_duplicate_state_key() -> None:
    votes = {"a": torch.tensor([1, 0], dtype=torch.int16)}
    sidecar_bytes, _, _ = encode_sparse_vote_inputs_svp1(votes)
    tampered = sidecar_bytes + sidecar_bytes[8:]
    with pytest.raises(ValueError):
        decode_sparse_vote_inputs_svp1(tampered)


def test_svp1_verify_fail_closed_sha_mismatch(tmp_path: Path) -> None:
    votes = {"proj": torch.tensor([0, 3, 0, -2], dtype=torch.int16)}
    sidecar_bytes, per_state, total = encode_sparse_vote_inputs_svp1(votes)
    emit_root = tmp_path / "votes_emit" / "v1"
    per_step = emit_root / "per_step"
    per_step.mkdir(parents=True)
    sidecar_path = per_step / "00000_sparse_votes.svp1"
    write_sidecar_atomically(sidecar_path, sidecar_bytes)
    stub = build_sparse_vote_inputs_stub(
        step_name="00000",
        per_state=per_state,
        total=total,
        sidecar_sha256="0" * 64,
    )
    with pytest.raises(ValueError, match="sha256 mismatch"):
        verify_sparse_vote_inputs_stub(
            stub,
            votes_emit_root=emit_root,
            emit_path=per_step / "00000.json",
        )


def test_svp1_verify_fail_closed_missing_sidecar(tmp_path: Path) -> None:
    votes = {"proj": torch.tensor([0, 3], dtype=torch.int16)}
    sidecar_bytes, per_state, total = encode_sparse_vote_inputs_svp1(votes)
    emit_root = tmp_path / "votes_emit" / "v1"
    stub = build_sparse_vote_inputs_stub(
        step_name="00000",
        per_state=per_state,
        total=total,
        sidecar_sha256=hashlib.sha256(sidecar_bytes).hexdigest(),
    )
    with pytest.raises(ValueError, match="missing SVP1 sidecar"):
        verify_sparse_vote_inputs_stub(
            stub,
            votes_emit_root=emit_root,
            emit_path=emit_root / "per_step" / "00000.json",
        )


def test_svp1_backward_compat_inline_record() -> None:
    record = {
        "sparse_vote_inputs_by_state_key": {
            "proj": {"1": 5, "3": -3},
        }
    }
    assert sparse_votes_from_emit_record(record) == {  # type: ignore[call-arg]
        "proj": {1: 5, 3: -3},
    }


def test_svp1_resolve_rejects_escape(tmp_path: Path) -> None:
    emit_root = tmp_path / "votes_emit" / "v1"
    emit_root.mkdir(parents=True)
    with pytest.raises(ValueError, match="\\.\\."):
        resolve_sidecar_path(emit_root, "../escape.svp1")


def test_witness_e2e_on_svp1_emit_bundle(tmp_path: Path) -> None:
    numel = 32
    q = torch.zeros(numel, dtype=torch.int8)
    state = make_event_coded_live_tensor_state("toy.proj", q, 1.0, demotion_band=1)
    votes = torch.zeros(numel, dtype=torch.int16)
    votes[0] = 8
    votes[3] = -4
    phase_root = tmp_path / "phase_a"
    maybe_emit_votes_step_record(
        root=phase_root,
        enabled=True,
        optimizer_step_index=0,
        tensor_states={"toy.proj": state},
        votes_by_key={"toy.proj": votes},
        vote_specs_by_key={"toy.proj": _vote_spec()},
        max_abs_per_tensor=4096,
        collector=VotesEmitCollector(phase_root),
        local_selection_ordering_mode=LOCAL_SELECTION_ORDER_CURRENT_MARGIN_INDEX,
    )
    carrier = EventCodedAccLiveState(logical_numel=numel, demotion_band=1)
    carrier.apply_step(0, votes={0: 8, 3: -4})
    gpu_surfaces = observed_surfaces_dict(carrier.step_records[-1])
    emit_payload = json.loads(
        (phase_root / "votes_emit" / "v1" / "per_step" / "00000.json").read_text(
            encoding="utf-8"
        )
    )
    assert emit_payload["sparse_vote_inputs_by_state_key"]["encoding"] == SPARSE_VOTE_PAIRS_ENCODING
    for field in VOTES_EMIT_SECTION6_CONTRACT_FIELDS:
        assert field in emit_payload
    receipt = {
        "step_reports": {
            "0": {
                "step_result": {
                    "tensor_stats": {
                        "toy.proj": {
                            "logical_numel": numel,
                            "dense_accumulator_materialized_numel": 0,
                            "v4_live_observed_surfaces": gpu_surfaces,
                            "event_coded_live_carrier_content_sha256_after": carrier_content_sha256(
                                carrier
                            ),
                            "c8_persistent_authority_scope": "persistent carrier; transient dense compute only; no dense persistent authority",
                            "transient_dense_compute_numel": numel,
                        }
                    }
                }
            }
        },
        "r4v_persistent_ledger": {"ledger_pass": True},
    }
    phase_root.mkdir(parents=True, exist_ok=True)
    (phase_root / "receipt.json").write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    verdict = run_phase_a_parity_witness(phase_root, demotion_band=1)
    assert verdict["phase_a_parity_pass"] is True
