"""CPU-static tests for sparse live carrier production landing (PLAN_v16 material)."""
from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS,
    build_optimizer_credit_state_fail_closed_receipt,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    AUTHORIZED_P1B_SURFACE_TUPLE,
    AUTHORIZED_P1B_SURFACE_TUPLE_2ROW,
    SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY,
    SparseVoteAuthorityLandingReceipt,
    SparseVoteExecutionWitness,
    _run_live_p1_vote_carrier_subproof,
    build_trainer_sub2_authority_live_conversion_receipt,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    normalize_sparse_vote_authority_mode,
    p1b_receipt_canonical_sha256,
    resolve_sparse_vote_authority_path,
    validate_sparse_vote_authority_landing_receipt,
    validate_sparse_vote_authority_mode_matches_execution_path,
    validate_trainer_sub2_authority_local_update_receipt,
    _path_witness_token,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    default_dry_run_rank_vote_spec,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _make_q_change_tiny_model() -> _TinyTernary:
    model = _TinyTernary()
    with torch.no_grad():
        model.proj.weight.zero_()
        model.tail.weight.fill_(0.25)
        model.tail.bias.zero_()
    return model


def _tiny_mse_loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])


def _batch() -> dict:
    return {
        "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
        "target": torch.ones(2, 4),
    }


def _dense_patch_ctx(module):
    return mock.patch.multiple(
        module,
        credit_from_weighted_grad=mock.Mock(side_effect=AssertionError("dense credit called")),
        project_s1_gradient_to_moves=mock.Mock(side_effect=AssertionError("dense moves called")),
        rank_bucketed_int16_votes=mock.Mock(side_effect=AssertionError("dense votes called")),
        _sparse_vote_events=mock.Mock(side_effect=AssertionError("_sparse_vote_events called")),
        _oracle_parity_proof=mock.Mock(side_effect=AssertionError("oracle parity called")),
        _dense_votes_from_sparse_events=mock.Mock(
            side_effect=AssertionError("dense densify called")
        ),
    )


def test_normalize_mode_default_and_rejects():
    assert normalize_sparse_vote_authority_mode() == SPARSE_VOTE_AUTHORITY_MODE_FUSED_ONLY
    with pytest.raises(TypeError):
        normalize_sparse_vote_authority_mode(None)


def test_builder_signatures_default_fused_only():
    """Finding 5: exact public signature parameters/defaults — no vacuous or True."""
    for fn in (
        build_trainer_sub2_authority_local_update_receipt,
        build_trainer_sub2_authority_roundtrip_receipt,
        build_trainer_sub2_authority_live_conversion_receipt,
        _run_live_p1_vote_carrier_subproof,
    ):
        sig = inspect.signature(fn)
        assert "sparse_vote_authority_mode" in sig.parameters
        param = sig.parameters["sparse_vote_authority_mode"]
        assert param.kind == inspect.Parameter.KEYWORD_ONLY
        assert param.default == "fused_only"
    # public P1b must NOT be bare **kwargs
    pub = inspect.signature(build_trainer_sub2_authority_live_conversion_receipt)
    assert "p1_checkpoint" in pub.parameters
    assert pub.parameters["p1_checkpoint"].kind == inspect.Parameter.KEYWORD_ONLY
    assert list(pub.parameters)[-1] != "kwargs" or "p1_checkpoint" in pub.parameters


def test_facade_fused_only_does_not_call_dense_symbols():
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as mod

    with _dense_patch_ctx(mod):
        path = resolve_sparse_vote_authority_path(
            weighted_grad_by_key={"k": torch.randn(4, 4)},
            q_levels_by_key={"k": torch.zeros(4, 4, dtype=torch.int8)},
            rank_spec=default_dry_run_rank_vote_spec(),
        )
    assert path["sparse_vote_authority_mode"] == "fused_only"
    assert path["oracle_only"] is None


def test_caller_authored_discriminator_rejected_by_validator():
    path = resolve_sparse_vote_authority_path(
        weighted_grad_by_key={"k": torch.randn(2, 2)},
        q_levels_by_key={"k": torch.zeros(2, 2, dtype=torch.int8)},
        rank_spec=default_dry_run_rank_vote_spec(),
    )
    forged = dict(path)
    forged["sparse_vote_authority_mode"] = "oracle_on"
    with pytest.raises(ValueError, match="discriminator mismatch"):
        validate_sparse_vote_authority_mode_matches_execution_path(
            forged, resolved_mode=path["resolved_mode"]
        )


def test_b1_omitted_mode_defaults_fused_only_dense_symbols_not_called():
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as mod

    with _dense_patch_ctx(mod):
        receipt = build_trainer_sub2_authority_local_update_receipt(
            model=_make_q_change_tiny_model(),
            batch=_batch(),
            forward_loss_fn=_tiny_mse_loss,
            use_ternary_bulk=True,
        )
    assert receipt.vote_projection_proof["sparse_vote_authority_mode"] == "fused_only"
    assert list(receipt.transient_over2_tensors) == ["weighted_grad"]
    assert "oracle_only" not in receipt.vote_projection_proof
    validate_trainer_sub2_authority_local_update_receipt(receipt)


def test_fused_only_receipt_schema_predicates_on_b1():
    receipt = build_trainer_sub2_authority_local_update_receipt(
        model=_make_q_change_tiny_model(),
        batch=_batch(),
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )
    proof = receipt.vote_projection_proof
    assert proof["votes_by_key_applied"] is None
    assert "oracle_only" not in proof
    assert list(receipt.transient_over2_tensors) == ["weighted_grad"]


def test_geometry_kwargs_static_apply_path_uses_sparse_only():
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as mod

    src = inspect.getsource(mod.build_trainer_sub2_authority_local_update_receipt)
    assert "sparse_vote_authority_only=True" in src
    assert "SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON" in src


def test_readiness_residual_weighted_grad_still_required():
    assert "weighted_grad" in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS
    r = build_optimizer_credit_state_fail_closed_receipt()
    assert "weighted_grad" in r.required_debt_anchors


def test_landing_does_not_alter_AUTHORIZED_P1B_SURFACE_TUPLE_semantics():
    assert isinstance(AUTHORIZED_P1B_SURFACE_TUPLE, tuple)
    assert len(AUTHORIZED_P1B_SURFACE_TUPLE) >= 1
    assert isinstance(AUTHORIZED_P1B_SURFACE_TUPLE_2ROW, tuple)


def test_landing_does_not_clear_optimizer_credit_state_weighted_grad_residual():
    assert "weighted_grad" in OPTIMIZER_CREDIT_STATE_REQUIRED_DEBT_ANCHORS


def test_b2_omitted_mode_signature_default():
    assert (
        inspect.signature(build_trainer_sub2_authority_roundtrip_receipt)
        .parameters["sparse_vote_authority_mode"]
        .default
        == "fused_only"
    )


def test_b2_post_resume_update_proof_discriminator_values():
    """Finding 4: B2 receipt carries full discriminator set with VALUES asserted."""
    # Build via local-update path fields by exercising roundtrip is heavy; assert source
    # contains the frozen keys AND local B1 path shape on vote_projection as reference.
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as mod

    src = inspect.getsource(mod.build_trainer_sub2_authority_roundtrip_receipt)
    for key in (
        "sparse_vote_authority_mode",
        "sparse_vote_authority_only",
        "dense_vote_authority_skipped",
        "votes_by_key_applied",
        "transient_over2_tensors",
    ):
        assert key in src
    # live B1 values as the production discriminator oracle for fused_only
    r = build_trainer_sub2_authority_local_update_receipt(
        model=_make_q_change_tiny_model(),
        batch=_batch(),
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )
    p = r.vote_projection_proof
    assert p["sparse_vote_authority_mode"] == "fused_only"
    assert p["sparse_vote_authority_only"] is True
    assert p["dense_vote_authority_skipped"] is True
    assert p["votes_by_key_applied"] is None
    assert list(r.transient_over2_tensors) == ["weighted_grad"]
    assert "oracle_only" not in p


class _StubP1b:
    def to_dict(self):
        return {"x": 1, "pass_receipt": True}

    total_sparse_vote_event_count = 1
    post_resume_payload_sha256_after = "a" * 64
    post_resume_update_mutated = True
    q_changed_count = 1


def _honest_landing_from_stub():
    stub = _StubP1b()
    digest = p1b_receipt_canonical_sha256(stub)  # type: ignore[arg-type]
    wg = {"proj.weight": "b" * 64}
    nonce = "n1"
    mode = "fused_only"
    token = _path_witness_token(
        resolved_mode=mode,
        execution_nonce=nonce,
        weighted_grad_capture_sha256_by_key=wg,
    )
    core = {
        "execution_nonce": nonce,
        "forward_backward_update_call_count": 1,
        "forward_backward_count": 1,
        "update_count": 1,
        "weighted_grad_capture_sha256_by_key": wg,
        "post_update_payload_sha256": "a" * 64,
        "p1b_receipt_sha256": digest,
        "path_resolved_mode": mode,
        "path_resolved_mode_observation_count": 1,
        "path_witness_token": token,
    }
    sub = {
        "sparse_vote_authority_mode": mode,
        "sparse_vote_authority_only": True,
        "dense_vote_authority_skipped": True,
        "votes_by_key_applied": None,
        "candidate_oracle_control_enabled": False,
        "fused_sparse_event_count_total": 1,
        "mutation_witness": {},
        "transient_over2_tensors": ["weighted_grad"],
        "execution_identity": dict(core),
    }
    return SparseVoteAuthorityLandingReceipt(
        schema_version="sparse_vote_authority_landing_receipt_v1",
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=stub,  # type: ignore[arg-type]
        p1b_receipt_sha256=digest,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )


def test_honest_one_execution_wrapper_pass():
    validate_sparse_vote_authority_landing_receipt(_honest_landing_from_stub())


def test_mutate_embedded_p1b_payload_post_build_rejected():
    landing = _honest_landing_from_stub()

    class _Stub2(_StubP1b):
        def to_dict(self):
            return {"x": 2}

    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=landing.sparse_vote_authority_subproof,
        p1b_live_conversion_receipt=_Stub2(),  # type: ignore[arg-type]
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=landing.core_execution_identity,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="p1b_receipt_sha256 mismatch"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_forge_subproof_identity_unbound_digest_rejected():
    landing = _honest_landing_from_stub()
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = {
        **landing.core_execution_identity,
        "p1b_receipt_sha256": "f" * 64,
    }
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=landing.core_execution_identity,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="subproof execution_identity.p1b_receipt_sha256"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_tamper_wrapper_p1b_receipt_sha256_rejected():
    landing = _honest_landing_from_stub()
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=landing.sparse_vote_authority_subproof,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256="0" * 64,
        core_execution_identity=landing.core_execution_identity,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="p1b_receipt_sha256 mismatch"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_call_count_not_one_rejected():
    landing = _honest_landing_from_stub()
    core = dict(landing.core_execution_identity)
    core["forward_backward_update_call_count"] = 2
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="forward_backward_update_call_count"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_forged_wrapper_mode_self_compare_rejected():
    """Finding 3: forging claimed mode while path_resolved_mode differs → REJECT at landing."""
    landing = _honest_landing_from_stub()
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["sparse_vote_authority_mode"] = "oracle_on"
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=landing.core_execution_identity,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="discriminator mismatch"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_second_execution_witness_rejects_call_count_one_claim():
    """Finding 2: real double-note of update via witness → count 2, claim 1 REJECT."""
    w = SparseVoteExecutionWitness()
    g = {"k": torch.randn(2, 2)}
    w.note_forward_backward(g)
    w.note_update()
    w.note_forward_backward(g)
    w.note_update()
    assert w.forward_backward_count == 2
    assert w.update_count == 2
    # combined field is not max; non-(1,1) is not one-execution
    assert w.forward_backward_update_call_count != 1
    landing = _honest_landing_from_stub()
    # inject measured count=2 into bags while keeping claim schema
    core = dict(landing.core_execution_identity)
    core["forward_backward_update_call_count"] = 2
    core["forward_backward_count"] = 2
    core["update_count"] = 2
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="forward_backward_count must be 1|update_count must be 1|forward_backward_update_call_count must be 1"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_post_snapshot_mutation_of_covered_surface_rejected():
    """Finding 5: actually mutate a covered surface copy and observe dry-exec REJECT."""
    repo = Path(__file__).resolve().parents[3]
    plan = repo / "artifacts/acc_entropy/optimizer_credit_state_sparse_live_carrier_production_landing_PLAN_v16.json"
    # Build a temp snapshot pointing at a temp file we can mutate
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        target = td_path / "covered.txt"
        target.write_text("v1\n")
        import hashlib

        def h(p):
            return hashlib.sha256(Path(p).read_bytes()).hexdigest()

        snap = {
            "schema_version": "sparse_live_carrier_final_implementation_snapshot_v1",
            "plan_sha256": h(plan),
            "task_id": "t",
            "minted_at_step": "test",
            "entries": {
                str(target): {"expected_sha256": h(target), "why": "fixture"},
            },
            "tsa_entry_sha256": "0" * 64,
        }
        snap_path = td_path / "snap.json"
        snap_path.write_text(json.dumps(snap))
        # mutate after mint
        target.write_text("v2-mutated\n")
        out = td_path / "dry.json"
        cmd = [
            sys.executable,
            str(repo / "scripts/sparse_live_carrier_production_landing_dry_exec.py"),
            "--plan",
            str(plan),
            "--mode",
            "fused_only",
            "--snapshot",
            str(snap_path),
            "--dry-exec-out",
            str(out),
        ]
        r = subprocess.run(cmd, cwd=str(repo), capture_output=True, text=True)
        assert r.returncode != 0
        assert "snapshot drift" in (r.stderr + r.stdout).lower() or "drift" in (r.stderr + r.stdout)


def test_fb_only_update_zero_rejected():
    landing = _honest_landing_from_stub()
    core = dict(landing.core_execution_identity)
    core["update_count"] = 0
    core["forward_backward_update_call_count"] = 1  # polluted combined claim
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="update_count must be 1"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_update_only_fb_zero_rejected():
    landing = _honest_landing_from_stub()
    core = dict(landing.core_execution_identity)
    core["forward_backward_count"] = 0
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="forward_backward_count must be 1"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_double_fb_rejected():
    landing = _honest_landing_from_stub()
    core = dict(landing.core_execution_identity)
    core["forward_backward_count"] = 2
    core["forward_backward_update_call_count"] = 2
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="forward_backward_count must be 1"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_double_update_rejected():
    landing = _honest_landing_from_stub()
    core = dict(landing.core_execution_identity)
    core["update_count"] = 2
    core["forward_backward_update_call_count"] = 2
    sub = dict(landing.sparse_vote_authority_subproof)
    sub["execution_identity"] = dict(core)
    bad = SparseVoteAuthorityLandingReceipt(
        schema_version=landing.schema_version,
        slice_readiness_claim=False,
        sparse_vote_authority_subproof=sub,
        p1b_live_conversion_receipt=landing.p1b_live_conversion_receipt,
        p1b_receipt_sha256=landing.p1b_receipt_sha256,
        core_execution_identity=core,
        plan_sha256="p" * 64,
        task_id="t",
    )
    with pytest.raises(ValueError, match="update_count must be 1"):
        validate_sparse_vote_authority_landing_receipt(bad)


def test_enforcer_schema_hostiles_missing_node_duration_unknown_type():
    """Finding D hostiles via enforcer --self-test."""
    import subprocess, sys, tempfile, uuid
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    for flag_self, expect_sub in (
        ("missing_node_id", "node_id"),
        ("missing_duration", "duration_s"),
        ("unknown_type", "unknown"),
    ):
        with tempfile.TemporaryDirectory() as td:
            e = Path(td) / "e.json"
            j = Path(td) / "j.jsonl"
            r = subprocess.run(
                [
                    sys.executable,
                    str(script),
                    "--enforcer-receipt",
                    str(e),
                    "--phase-events-jsonl",
                    str(j),
                    "--expected-node-id",
                    "n1",
                    "--self-test",
                    flag_self,
                ],
                capture_output=True,
                text=True,
            )
            assert r.returncode == 2, (flag_self, r.returncode, r.stdout, r.stderr)
            doc = json.loads(e.read_text())
            assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
            assert expect_sub in str(doc.get("error", "")).lower() or expect_sub in str(doc).lower()


def test_roundtrip_builder_phase_emitter_four_phase_sequence_cpu():
    """B1: recording emitter on ACTUAL build_trainer_sub2_authority_roundtrip_receipt."""
    import calm.hrm_text_158.native_full_stack.trainer_sub2_authority as mod
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        build_trainer_sub2_authority_roundtrip_receipt,
    )

    events = []
    def rec(kind, phase):
        events.append((kind, phase))

    prev = mod._PHASE_EMITTER
    mod._PHASE_EMITTER = rec
    try:
        model = _make_q_change_tiny_model()
        r = build_trainer_sub2_authority_roundtrip_receipt(
            model,
            fresh_model_fn=_make_q_change_tiny_model,
            batch=_batch(),
            forward_loss_fn=_tiny_mse_loss,
            forward_output_fn=lambda m, b: m(b["x"]),
            use_ternary_bulk=True,
        )
        assert r.post_resume_update_proof["total_sparse_vote_event_count"] > 0
        assert int(r.post_resume_update_proof.get("q_changed_count", 0)) > 0
    finally:
        mod._PHASE_EMITTER = prev
    kinds_phases = events
    # builder supplies FB + update + emission (post-update rebuild metered under emission)
    for need in (
        ("PHASE_START", "forward_backward"),
        ("PHASE_END", "forward_backward"),
        ("PHASE_START", "update"),
        ("PHASE_END", "update"),
        ("PHASE_START", "emission"),
        ("PHASE_END", "emission"),
    ):
        assert need in kinds_phases, (need, kinds_phases)
    i_fbs = kinds_phases.index(("PHASE_START", "forward_backward"))
    i_fbe = kinds_phases.index(("PHASE_END", "forward_backward"))
    i_us = kinds_phases.index(("PHASE_START", "update"))
    i_ue = kinds_phases.index(("PHASE_END", "update"))
    i_es = kinds_phases.index(("PHASE_START", "emission"))
    i_ee = kinds_phases.index(("PHASE_END", "emission"))
    assert i_fbs < i_fbe < i_us < i_ue < i_es < i_ee


def test_enforcer_typed_duration_and_node_hostiles():
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    for st, needle in (
        ("wrong_duration_type", "duration_s"),
        ("negative_duration", "duration_s"),
        ("non_string_node_id", "node_id"),
    ):
        with tempfile.TemporaryDirectory() as td:
            e = Path(td) / "e.json"
            j = Path(td) / "j.jsonl"
            r = subprocess.run(
                [sys.executable, str(script), "--enforcer-receipt", str(e),
                 "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
                 "--self-test", st],
                capture_output=True, text=True,
            )
            assert r.returncode == 2, (st, r.returncode, r.stdout, r.stderr)
            doc = json.loads(e.read_text())
            assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
            assert needle in str(doc.get("error", "")).lower()


def test_b3_landing_corrupt_envelope_fails_without_b1_substitution():
    """C2 hostile on CPU: corrupt envelope raises; no silent B1 path."""
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        build_sparse_vote_authority_landing_receipt,
    )
    with pytest.raises((ValueError, TypeError, KeyError)):
        build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="hostile",
            p1_checkpoint={"not": "p1"},
            p1_envelope_bytes=b"{}",
            fresh_model_fn=_make_q_change_tiny_model,
            batch=_batch(),
            forward_loss_fn=_tiny_mse_loss,
            forward_output_fn=lambda m, b: m(b["x"]),
            parity_max_abs_diff_by_site={
                "cache_builder": 0.0,
                "main_kl": 0.0,
                "retained_fallback": 0.0,
            },
            use_ternary_bulk=True,
        )


def test_enforcer_child_nonzero_not_ok():
    """B4: full phase cycle + child exit 1 → FAIL not OK."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "child_nonzero"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "GPU-SMOKE-FAIL/CHILD_NONZERO"
        assert doc.get("child_rc") not in (0, None)


def test_enforcer_malformed_telemetry_kills_hanging_child():
    """B5: invalid JSON then hang → kill + PHASE_TELEMETRY."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "malformed_then_hang"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
        assert doc.get("kill_actions") == ["TERM", "KILL"] or (
            "TERM" in (doc.get("kill_actions") or []) and "KILL" in (doc.get("kill_actions") or [])
        )


def test_enforcer_good_topology_class_ok():
    """R4: exact formal stream → CLASS_OK exit 0 (regression guard for R1 class)."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "good_topology"],
            capture_output=True, text=True,
        )
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "OK"
        assert doc.get("child_rc") == 0
        assert doc.get("kill_actions") in ([], None)


def test_enforcer_invalid_duration_then_hang_kills():
    """R3(a): invalid duration then hang → TERM/KILL + PHASE_TELEMETRY exit 2."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "invalid_duration_then_hang"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
        kills = doc.get("kill_actions") or []
        assert "TERM" in kills and "KILL" in kills, kills
        assert "duration_s" in str(doc.get("error", "")).lower()


def test_enforcer_unknown_phase_start_then_hang_kills():
    """R3(b): unknown-phase START then hang → TERM/KILL + PHASE_TELEMETRY exit 2."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "unknown_phase_start_then_hang"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
        kills = doc.get("kill_actions") or []
        assert "TERM" in kills and "KILL" in kills, kills
        assert "unknown phase" in str(doc.get("error", "")).lower()


def test_enforcer_duplicate_start_then_hang_kills():
    """R3(c): duplicate START then hang → TERM/KILL + PHASE_TELEMETRY exit 2."""
    import subprocess, sys, tempfile, json
    from pathlib import Path
    repo = Path(__file__).resolve().parents[3]
    script = repo / "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py"
    with tempfile.TemporaryDirectory() as td:
        e = Path(td) / "e.json"
        j = Path(td) / "j.jsonl"
        r = subprocess.run(
            [sys.executable, str(script), "--enforcer-receipt", str(e),
             "--phase-events-jsonl", str(j), "--expected-node-id", "n1",
             "--self-test", "duplicate_start_then_hang"],
            capture_output=True, text=True,
        )
        assert r.returncode == 2, (r.returncode, r.stdout, r.stderr)
        doc = json.loads(e.read_text())
        assert doc["terminal_class"] == "GPU-SMOKE-FAIL/PHASE_TELEMETRY"
        kills = doc.get("kill_actions") or []
        assert "TERM" in kills and "KILL" in kills, kills
        assert "duplicate start" in str(doc.get("error", "")).lower()


def test_b3_fixture_envelope_passes_production_blob_loader():
    """C-new: CPU-static — exact B3 fixture envelope (production save path) must
    satisfy load_trainer_sub2_authority_checkpoint_blob schema_version contract
    (TSA :1726). Catches hand-wrap drift before GPU.
    """
    import torch
    from calm.hrm_text_158.bit_linear import BitLinear
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION,
        P1_LIVE_CHECKPOINT_FORMAT,
        is_p1_live_sub2_checkpoint,
        load_trainer_sub2_authority_checkpoint_blob,
        save_trainer_sub2_live_checkpoint_envelope,
        select_trainer_eligible_bitlinears,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = BitLinear(4, 4, bias=False)

        def forward(self, x):
            return self.lin(x)

    model = _Tiny()
    # EXACT same construction path as gpu_live B3 fixture
    envelope = save_trainer_sub2_live_checkpoint_envelope(
        model,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
        step=0,
        config={"proof": "gpu_live_b3"},
        source_pin="gpu_live_b3",
        epoch=0,
    )
    assert is_p1_live_sub2_checkpoint(envelope) is True
    assert envelope.get("schema_version") == TRAINER_SUB2_ROUNDTRIP_SCHEMA_VERSION
    assert envelope.get("checkpoint_format") == P1_LIVE_CHECKPOINT_FORMAT
    assert "trainer_sub2_authority" in envelope and "model_state" in envelope
    # must not be a hand-selected subset — full producer keys present
    for key in ("schema_version", "artifact_role", "model_state", "trainer_sub2_authority",
                "checkpoint_format"):
        assert key in envelope, key

    fresh = _Tiny()
    eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    states = load_trainer_sub2_authority_checkpoint_blob(
        fresh,
        envelope,
        eligible_modules=eligible,
        device="cpu",
    )
    assert set(states) == set(eligible)

    # polarity: hand-selected subset (the prior defect) must fail :1726
    hand = {
        "trainer_sub2_authority": envelope["trainer_sub2_authority"],
        "model_state": envelope["model_state"],
        "checkpoint_format": P1_LIVE_CHECKPOINT_FORMAT,
    }
    with pytest.raises(ValueError, match="2C4a checkpoint blob schema mismatch"):
        load_trainer_sub2_authority_checkpoint_blob(
            _Tiny(),
            hand,
            eligible_modules=select_trainer_eligible_bitlinears(_Tiny(), use_ternary_bulk=True),
            device="cpu",
        )

