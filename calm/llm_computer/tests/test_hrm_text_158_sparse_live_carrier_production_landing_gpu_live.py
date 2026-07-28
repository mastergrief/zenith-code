"""GPU-live production-builder proofs (PLAN_v16 finding C).

FAIL-CLOSED without CUDA. Each node calls the EXACT production builder named
in the plan. Optional _PHASE_EMITTER instruments internal phase boundaries.
Non-vacuity: fused event_count > 0 AND mutation (q_changed) > 0.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack import trainer_sub2_authority as tsa
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    build_sparse_vote_authority_landing_receipt,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    resolve_sparse_vote_authority_path,
    select_trainer_eligible_bitlinears,
    derive_trainer_sub2_authority_states,
    trainer_authoritative_forward_context,
    save_trainer_sub2_live_checkpoint_envelope,
    is_p1_live_sub2_checkpoint,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    default_dry_run_rank_vote_spec,
)

B3_NODE = "test_gpu_live_b3_landing_wrapper_default_fused_only"
ENV_JSONL = "SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"

def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for gpu_live nodes (fail-closed; no CPU fallback)")
    return torch.device("cuda")

def _jsonl_emitter(node_id: str):
    path = os.environ.get(ENV_JSONL)
    if not path:
        raise RuntimeError(f"{ENV_JSONL} required in gpu_live (fail-closed)")
    open_starts: dict[str, float] = {}

    def emit(kind: str, phase: str) -> None:
        ev = {
            "type": kind,
            "phase": phase,
            "node_id": node_id,
            "ts_monotonic": time.monotonic(),
        }
        if kind == "PHASE_START":
            open_starts[phase] = time.monotonic()
        elif kind == "PHASE_END":
            t0 = open_starts.pop(phase, time.monotonic())
            ev["duration_s"] = time.monotonic() - t0
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(ev) + "\n")

    return emit

class _TinyTernary(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x):
        return self.tail(self.proj(x))

def _model(device):
    m = _TinyTernary().to(device)
    with torch.no_grad():
        m.proj.weight.zero_()
        m.tail.weight.fill_(0.25)
        m.tail.bias.zero_()
    return m

def _batch(device):
    return {
        "x": (torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0).to(device),
        "target": torch.ones(2, 4, device=device),
    }

def _loss(model, batch):
    return torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])

def _output(model, batch):
    return model(batch["x"])

def test_gpu_live_b1_default_fused_only_emits_phase_events():
    device = _require_cuda()
    node = "test_gpu_live_b1_default_fused_only_emits_phase_events"
    model = _model(device)
    batch = _batch(device)
    emit = _jsonl_emitter(node)
    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = emit
    try:
        receipt = build_trainer_sub2_authority_local_update_receipt(
            model=model,
            batch=batch,
            forward_loss_fn=_loss,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
        # R1: builder owns emission; test owns only flush (post-return asserts unmetered)
        assert receipt.total_sparse_vote_event_count > 0
        assert int(receipt.candidate_step_summary.get("q_changed_count", 0)) > 0
        assert receipt.vote_projection_proof["sparse_vote_authority_mode"] == "fused_only"
        assert list(receipt.transient_over2_tensors) == ["weighted_grad"]
        emit("PHASE_START", "flush")
        torch.cuda.synchronize()
        emit("PHASE_END", "flush")
    finally:
        tsa._PHASE_EMITTER = prev
    print(f"device_type={device.type} device_name={torch.cuda.get_device_name(0)}")

def test_gpu_live_b2_roundtrip_default_fused_only():
    device = _require_cuda()
    node = "test_gpu_live_b2_roundtrip_default_fused_only"
    emit = _jsonl_emitter(node)
    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = emit  # builder supplies forward_backward/update phases
    try:
        def fresh():
            return _model(device)

        receipt = build_trainer_sub2_authority_roundtrip_receipt(
            model=_model(device),
            fresh_model_fn=fresh,
            batch=_batch(device),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
        # R1: builder owns emission; test owns only flush
        proof = receipt.post_resume_update_proof
        assert proof["sparse_vote_authority_mode"] == "fused_only"
        assert proof["sparse_vote_authority_only"] is True
        assert proof["dense_vote_authority_skipped"] is True
        assert proof["votes_by_key_applied"] is None
        assert list(proof["transient_over2_tensors"]) == ["weighted_grad"]
        assert "oracle_only" not in proof
        assert int(proof.get("total_sparse_vote_event_count", 0)) > 0
        assert int(proof.get("q_changed_count", 0)) > 0
        emit("PHASE_START", "flush")
        torch.cuda.synchronize()
        emit("PHASE_END", "flush")
    finally:
        tsa._PHASE_EMITTER = prev
    print(f"device_type={device.type} device_name={torch.cuda.get_device_name(0)}")

def test_gpu_live_b3_landing_wrapper_default_fused_only():
    """C2: production landing only — fail-closed, no B1 substitution."""
    device = _require_cuda()
    node = B3_NODE
    emit = _jsonl_emitter(node)
    prev = tsa._PHASE_EMITTER
    tsa._PHASE_EMITTER = emit
    try:
        def fresh():
            return _model(device)

        model = _model(device)
        batch = _batch(device)
        # C1 (addendum): envelope = FULL canonical producer blob
        # (build_trainer_sub2_authority_checkpoint_blob → schema_version at TSA :1702)
        # + checkpoint_format on top. Use production save path so fields cannot be
        # hand-selected/dropped again (prior defect: only trainer_sub2_authority/
        # model_state/format reconstructed).
        p1 = save_trainer_sub2_live_checkpoint_envelope(
            model,
            use_ternary_bulk=True,
            eligible_scope="all-bitlinear",
            step=0,
            config={"proof": "gpu_live_b3"},
            source_pin="gpu_live_b3",
            epoch=0,
        )
        if not is_p1_live_sub2_checkpoint(p1):
            pytest.fail(
                f"B3 P1 envelope preflight failed after production save path: "
                f"keys={list(p1)} schema_version={p1.get('schema_version')!r} "
                f"checkpoint_format={p1.get('checkpoint_format')!r}"
            )
        if p1.get("schema_version") != tsa.TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION:
            pytest.fail(
                f"B3 production envelope missing/ mismatched schema_version: "
                f"{p1.get('schema_version')!r}"
            )

        landing = build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="gpu_live_b3",
            p1_checkpoint=p1,
            p1_envelope_bytes=b"{}",
            fresh_model_fn=fresh,
            batch=batch,
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            parity_max_abs_diff_by_site={
                "cache_builder": 0.0,
                "main_kl": 0.0,
                "retained_fallback": 0.0,
            },
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
        )
        # R1: landing wrapper owns single emission pair (R2); test owns only flush
        assert landing.slice_readiness_claim is False
        assert landing.core_execution_identity["forward_backward_count"] == 1
        assert landing.core_execution_identity["update_count"] == 1
        assert landing.sparse_vote_authority_subproof["fused_sparse_event_count_total"] > 0
        assert int(landing.p1b_live_conversion_receipt.q_changed_count) > 0
        emit("PHASE_START", "flush")
        torch.cuda.synchronize()
        emit("PHASE_END", "flush")
    finally:
        tsa._PHASE_EMITTER = prev
    print(f"device_type={device.type} device_name={torch.cuda.get_device_name(0)} node={B3_NODE}")

def test_gpu_live_b3_substitution_impossible_on_corrupt_envelope():
    """C2 hostile: corrupt envelope → node FAILURE; no B1 substitution path."""
    device = _require_cuda()
    # corrupt envelope intentionally
    with pytest.raises((ValueError, TypeError, KeyError, AssertionError)):
        build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="hostile",
            p1_checkpoint={"not": "p1"},
            p1_envelope_bytes=b"{}",
            fresh_model_fn=lambda: _model(device),
            batch=_batch(device),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            parity_max_abs_diff_by_site={
                "cache_builder": 0.0,
                "main_kl": 0.0,
                "retained_fallback": 0.0,
            },
            use_ternary_bulk=True,
            device=device,
        )

def test_gpu_live_oracle_on_events_equal_parity():
    device = _require_cuda()
    node = "test_gpu_live_oracle_on_events_equal_parity"
    emit = _jsonl_emitter(node)
    model = _model(device)
    batch = _batch(device)
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    emit("PHASE_START", "forward_backward")
    weighted = {}
    model.train(True)
    model.zero_grad(set_to_none=True)
    with trainer_authoritative_forward_context(
        eligible, states, device=device, requires_grad=True
    ) as handle:
        _loss(model, batch).backward()
        for key in sorted(states):
            weighted[key] = handle.weighted_grad(key)
    emit("PHASE_END", "forward_backward")
    emit("PHASE_START", "update")
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key=weighted,
        q_levels_by_key={k: s.q_levels for k, s in states.items()},
        rank_spec=default_dry_run_rank_vote_spec(),
        sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    )
    emit("PHASE_END", "update")
    emit("PHASE_START", "emission")
    assert path["resolved_mode"] == "oracle_on"
    assert path["oracle_only"]["events_equal_fused_vs_dense_derived"] is True
    assert sum(e.event_count() for e in path["sparse_events_by_key"].values()) > 0
    emit("PHASE_END", "emission")
    emit("PHASE_START", "flush")
    torch.cuda.synchronize()
    emit("PHASE_END", "flush")
    print(f"device_type={device.type} device_name={torch.cuda.get_device_name(0)}")
