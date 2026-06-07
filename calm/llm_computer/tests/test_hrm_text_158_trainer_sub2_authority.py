"""Tests for the 2C1 trainer sub-2 authority construction proof."""
from __future__ import annotations

import copy
from dataclasses import replace
import inspect
from pathlib import Path

import pytest
import torch

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    TRAINER_SUB2_AUTHORITY_NON_CLAIMS,
    TRAINER_SUB2_ACTIVE_CONTROL_PARAMETER_NAMES,
    TRAINER_SUB2_LOCAL_UPDATE_NON_CLAIMS,
    TRAINER_SUB2_ROUNDTRIP_NON_CLAIMS,
    _roundtrip_payload_sha256,
    build_trainer_sub2_authority_checkpoint_blob,
    build_trainer_sub2_authority_construction_receipt,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
    trainer_local_update_builder_active_control_parameters,
    validate_trainer_sub2_authority_construction_receipt,
    validate_trainer_sub2_authority_local_update_receipt,
    validate_trainer_sub2_authority_roundtrip_receipt,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _make_tiny_gsm8k_rows(n: int = 8) -> list[dict]:
    return [
        {
            "question": f"what is {i + 1} {'plus' if i % 2 == 0 else 'times'} {i + 1}?",
            "expected": (2 * (i + 1)) if i % 2 == 0 else ((i + 1) ** 2),
        }
        for i in range(n)
    ]


def _splits_loader_factory(rows):
    def _loader(val_frac: float = 0.10):
        return rows, rows[:2], rows[:2]

    return _loader


def _make_q_change_tiny_model() -> _TinyTernary:
    model = _TinyTernary()
    with torch.no_grad():
        model.proj.weight.zero_()
        model.tail.weight.fill_(0.25)
        model.tail.bias.zero_()
    return model


def _tiny_mse_loss(model: torch.nn.Module, batch: dict) -> torch.Tensor:
    return torch.nn.functional.mse_loss(model(batch["x"]), batch["target"])


def _make_roundtrip_blob():
    model = _make_q_change_tiny_model()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
    )
    return model, eligible, blob


def _common_train_kwargs(tmp_path: Path, **overrides) -> dict:
    base = dict(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        hidden_size=32,
        n_layers=2,
        num_heads=2,
        expansion=4,
        H_cycles=2,
        L_cycles=2,
        half_layers=True,
        bp_warmup_ratio=0.0,
        bp_min_steps=2,
        bp_max_steps=2,
        max_len=32,
        seed=158,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        log_every=1,
        device="cpu",
    )
    base.update(overrides)
    return base


def test_receipt_constructs_payload_counts_sub2_and_excludes_optimizer_masters():
    model = _TinyTernary()

    receipt = build_trainer_sub2_authority_construction_receipt(
        model,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
    )

    validate_trainer_sub2_authority_construction_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.trainer_entrypoint_can_construct_sub2_authority is True
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.learner_update_called is False
    assert receipt.optimizer_step_called is False
    assert receipt.checkpoint_written is False
    assert receipt.gpu_launched is False
    assert receipt.eligible_module_count == 1
    assert receipt.optimizer_exclusion_proof["eligible_params_in_optimizer"] == 0
    assert receipt.optimizer_exclusion_proof["eligible_optimizer_state_entries"] == 0
    assert receipt.checkpoint_payload_validated is True
    assert receipt.checkpoint_payload_summary["checkpoint_written"] is False
    assert receipt.checkpoint_payload_summary["dry_run"] is True
    assert receipt.persistent_authority_bits_per_weight < 2.0
    assert receipt.dense_int16_persistent_authority_bits_counted == 0
    assert receipt.fp_master_persistent_authority_bits_counted == 0
    assert receipt.non_claims == TRAINER_SUB2_AUTHORITY_NON_CLAIMS


def test_receipt_fails_closed_without_ternary_bulk_or_bitlinear():
    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        select_trainer_eligible_bitlinears(
            _TinyTernary(),
            use_ternary_bulk=False,
        )
    with pytest.raises(RuntimeError, match="no eligible BitLinear"):
        select_trainer_eligible_bitlinears(
            torch.nn.Linear(4, 4),
            use_ternary_bulk=True,
        )


def test_forbidden_broad_claims_fail_validation():
    receipt = build_trainer_sub2_authority_construction_receipt(
        _TinyTernary(),
        use_ternary_bulk=True,
    )

    with pytest.raises(ValueError, match="trainer-used candidate"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, trainer_entrypoint_uses_candidate=True)
        )
    with pytest.raises(ValueError, match="live runtime"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, live_runtime_authority_converted=True)
        )
    with pytest.raises(ValueError, match="readiness row"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, readiness_row_flip_authorized=True)
        )


def test_forward_context_wrapper_uses_q_state_without_update_claim():
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = native_full_stack.derive_trainer_sub2_authority_states(eligible)
    x = torch.randn(2, 16)

    with trainer_authoritative_forward_context(
        eligible,
        states,
        requires_grad=False,
    ) as handle:
        out = model.proj(x)
        expected = torch.nn.functional.linear(
            x,
            states["proj"].materialized_weight(requires_grad=False),
            None,
        )

    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
    assert handle.capture_enabled is False


def test_local_update_receipt_runs_trainer_style_qacc_update_and_exact_parity():
    model = _make_q_change_tiny_model()
    batch = {
        "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
        "target": torch.ones(2, 4),
    }

    receipt = build_trainer_sub2_authority_local_update_receipt(
        model,
        batch=batch,
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
    )

    validate_trainer_sub2_authority_local_update_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.default_off_trainer_local_qacc_update_proof_exercised is True
    assert receipt.default_off_trainer_active_controls_inactive_proven is True
    assert receipt.global_cap_spec_passed is False
    assert receipt.global_rate_cap_enabled is False
    assert receipt.deferred_backlog_input_present is False
    assert receipt.deferred_backlog_output_entry_count == 0
    assert receipt.replay_ce_veto_maps_present is False
    assert receipt.pc_aux_maps_present is False
    assert receipt.pc_aux_mode_effective == "not_enabled"
    assert receipt.front_c_identity_observer_present is False
    assert receipt.candidate_mode_rejects_active_controls is True
    assert receipt.trainer_builder_has_no_active_control_parameters is True
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.learner_update_called is True
    assert receipt.optimizer_step_called is False
    assert receipt.checkpoint_written is False
    assert receipt.total_sparse_vote_event_count > 0
    assert receipt.q_changed_count > 0
    assert receipt.authority_state_shadow_free_after is True
    assert receipt.eligible_fp_masters_byte_identical is True
    assert receipt.checkpoint_payload_written is False
    assert receipt.checkpoint_payload_contains_oracle is False
    assert "weighted_grad" in receipt.transient_over2_tensors
    assert "dense_oracle_qacc_reference_result" in receipt.transient_over2_tensors
    assert receipt.non_claims == TRAINER_SUB2_LOCAL_UPDATE_NON_CLAIMS
    assert receipt.candidate_step_summary["candidate_local_update_pass"] is True
    assert (
        receipt.candidate_step_summary[
            "default_off_trainer_active_controls_inactive_proven"
        ]
        is True
    )
    assert receipt.candidate_step_summary["global_rate_cap_enabled"] is False
    assert receipt.candidate_step_summary["deferred_backlog_output_entry_count"] == 0
    assert receipt.candidate_step_summary["candidate_dense_decode_used"] is False
    assert receipt.candidate_step_summary["candidate_dense_vote_authority_used"] is False
    proof = receipt.forward_backward_capture_proof["weighted_grad_capture_by_key"]["proj"]
    assert proof["weighted_grad_nonzero_count"] > 0
    assert proof["sparse_vote_event_count"] > 0
    parity = receipt.exact_local_parity_proof_by_key["proj"]
    assert parity["parity_pass"] is True
    assert parity["candidate_q_sha256_after"] == parity["oracle_q_sha256_after"]
    assert (
        parity["candidate_bounded_decode_sha256_after"]
        == parity["oracle_acc_sha256_after"]
    )
    assert (
        parity["candidate_applied_row_identities_sha256"]
        == parity["oracle_applied_row_identities_sha256"]
    )
    assert (
        parity["candidate_ordered_applied_row_identities_sha256"]
        == parity["oracle_ordered_applied_row_identities_sha256"]
    )
    assert (
        parity["candidate_applied_directions_sha256"]
        == parity["oracle_applied_directions_sha256"]
    )
    assert (
        parity["candidate_applied_thresholds_sha256"]
        == parity["oracle_applied_thresholds_sha256"]
    )
    assert (
        parity["candidate_residual_after_threshold_sha256"]
        == parity["oracle_residual_after_threshold_sha256"]
    )


def test_local_update_receipt_forbidden_broad_claims_fail_validation():
    receipt = build_trainer_sub2_authority_local_update_receipt(
        _make_q_change_tiny_model(),
        batch={
            "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
            "target": torch.ones(2, 4),
        },
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )

    with pytest.raises(ValueError, match="broad trainer_entrypoint"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, trainer_entrypoint_uses_candidate=True)
        )
    with pytest.raises(ValueError, match="exact shadow"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, authority_state_shadow_free_after=False)
        )
    with pytest.raises(ValueError, match="FP masters changed"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, eligible_fp_masters_byte_identical=False)
        )
    with pytest.raises(ValueError, match="active controls inactive proof"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, default_off_trainer_active_controls_inactive_proven=False)
        )
    with pytest.raises(ValueError, match="global cap spec"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, global_cap_spec_passed=True)
        )
    with pytest.raises(ValueError, match="global cap"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, global_rate_cap_enabled=True)
        )
    with pytest.raises(ValueError, match="deferred backlog input"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, deferred_backlog_input_present=True)
        )
    with pytest.raises(ValueError, match="deferred backlog"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, deferred_backlog_output_entry_count=1)
        )
    with pytest.raises(ValueError, match="replay CE veto"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, replay_ce_veto_maps_present=True)
        )
    with pytest.raises(ValueError, match="PC auxiliary maps"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, pc_aux_maps_present=True)
        )
    with pytest.raises(ValueError, match="PC auxiliary mode"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, pc_aux_mode_effective="veto")
        )
    with pytest.raises(ValueError, match="front-C"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, front_c_identity_observer_present=True)
        )
    with pytest.raises(ValueError, match="active-control rejection"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, candidate_mode_rejects_active_controls=False)
        )
    with pytest.raises(ValueError, match="no trainer active-control parameters"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, trainer_builder_has_no_active_control_parameters=False)
        )
    bad_summary = dict(receipt.candidate_step_summary)
    bad_summary["global_rate_cap_enabled"] = True
    with pytest.raises(ValueError, match="candidate summary.*global cap"):
        validate_trainer_sub2_authority_local_update_receipt(
            replace(receipt, candidate_step_summary=bad_summary)
        )


def test_local_update_builder_has_no_active_control_parameters():
    signature = inspect.signature(build_trainer_sub2_authority_local_update_receipt)
    forbidden = sorted(
        name
        for name in signature.parameters
        if name in TRAINER_SUB2_ACTIVE_CONTROL_PARAMETER_NAMES
    )

    assert forbidden == []
    assert trainer_local_update_builder_active_control_parameters() == ()


def test_roundtrip_receipt_excludes_fp_masters_and_falsifies_poisoned_forward():
    model = _make_q_change_tiny_model()
    batch = {
        "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
        "target": torch.ones(2, 4),
    }

    receipt = build_trainer_sub2_authority_roundtrip_receipt(
        model,
        fresh_model_fn=_make_q_change_tiny_model,
        batch=batch,
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
    )

    validate_trainer_sub2_authority_roundtrip_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.persistent_authority_state_roundtrip_pass is True
    assert receipt.trainer_state_mutation_uses_sub2_authority is True
    assert receipt.resumed_forward_uses_sidecar_authority is True
    assert receipt.poisoned_fp_master_bypass_falsified is True
    assert receipt.eligible_fp_masters_authoritative is False
    assert receipt.eligible_fp_master_keys_excluded_from_authoritative_model_state is True
    assert receipt.raw_state_dict_eligible_weight_fallback_rejected is True
    assert receipt.normal_bitlinear_weight_forward_not_claimed is True
    assert receipt.dense_int16_persistent_accumulator_saved is False
    assert receipt.dense_int16_persistent_accumulator_loaded is False
    assert receipt.q_scale_sidecar_bounded_hash_roundtrip_pass is True
    assert receipt.post_resume_update_mutated_resumed_sub2_authority is True
    assert receipt.update_law_quality_claim is False
    assert receipt.learning_claim is False
    assert receipt.optimizer_credit_state_resolved is False
    assert receipt.credit_ranking_uninformative_update_law_pivot_deferred is True
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.broad_runtime_authority_converted is False
    assert receipt.full_sub2_runtime_readiness_claim is False
    assert receipt.checkpoint_payload_summary["eligible_weight_keys_excluded"] is True
    assert receipt.checkpoint_load_proof["strict_noneligible_model_state_load"] is True
    assert receipt.checkpoint_load_proof["missing_keys_exactly_eligible_weights"] is True
    assert receipt.checkpoint_load_proof["post_update_payload_hash_roundtrip_pass"] is True
    assert receipt.poison_forward_proof["normal_no_context_forward_changed_after_poison"] is True
    assert receipt.poison_forward_proof["resumed_context_forward_matches_sidecar_expected"] is True
    assert receipt.post_resume_update_proof["candidate_local_update_pass"] is True
    assert receipt.post_resume_update_proof["q_changed_count"] > 0
    assert receipt.non_claims == TRAINER_SUB2_ROUNDTRIP_NON_CLAIMS


def test_roundtrip_checkpoint_loader_rejects_eligible_raw_weight_fallback():
    model, _eligible, blob = _make_roundtrip_blob()
    bad_blob = copy.deepcopy(blob)
    bad_blob["model_state"]["proj.weight"] = model.proj.weight.detach().cpu().clone()
    fresh = _make_q_change_tiny_model()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)

    with pytest.raises(ValueError, match="raw state_dict eligible-weight fallback rejected"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad_blob,
            eligible_modules=fresh_eligible,
        )


def test_roundtrip_checkpoint_loader_rejects_corrupted_sidecar_hash():
    _model, _eligible, blob = _make_roundtrip_blob()
    bad_blob = copy.deepcopy(blob)
    bad_blob["trainer_sub2_authority"]["authoritative_state_payload_sha256"] = "0" * 64
    fresh = _make_q_change_tiny_model()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)

    with pytest.raises(ValueError, match="authoritative payload hash mismatch"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad_blob,
            eligible_modules=fresh_eligible,
        )


def test_roundtrip_checkpoint_loader_rejects_dense_int16_sidecar_flags():
    _model, _eligible, blob = _make_roundtrip_blob()
    bad_top = copy.deepcopy(blob)
    bad_top["trainer_sub2_authority"]["dense_int16_persistent_accumulator_saved"] = True
    fresh_top = _make_q_change_tiny_model()
    fresh_top_eligible = select_trainer_eligible_bitlinears(fresh_top, use_ternary_bulk=True)

    with pytest.raises(ValueError, match="dense int16 persistent accumulators"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh_top,
            bad_top,
            eligible_modules=fresh_top_eligible,
        )

    bad_bounded = copy.deepcopy(blob)
    sidecar = bad_bounded["trainer_sub2_authority"]
    sidecar["tensor_payloads"]["proj"]["bounded_accumulator"][
        "dense_int16_accumulator_persisted"
    ] = True
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh_bounded = _make_q_change_tiny_model()
    fresh_bounded_eligible = select_trainer_eligible_bitlinears(
        fresh_bounded,
        use_ternary_bulk=True,
    )

    with pytest.raises(ValueError, match="dense int16 accumulators"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh_bounded,
            bad_bounded,
            eligible_modules=fresh_bounded_eligible,
        )


def test_roundtrip_checkpoint_loader_rejects_noneligible_state_drift():
    _model, _eligible, blob = _make_roundtrip_blob()
    bad_blob = copy.deepcopy(blob)
    bad_state = dict(blob["model_state"])
    bad_state.pop("tail.weight")
    bad_blob["model_state"] = bad_state
    fresh = _make_q_change_tiny_model()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)

    with pytest.raises(ValueError, match="strict non-eligible model_state"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad_blob,
            eligible_modules=fresh_eligible,
        )


def test_roundtrip_receipt_forbidden_claims_fail_validation():
    receipt = build_trainer_sub2_authority_roundtrip_receipt(
        _make_q_change_tiny_model(),
        fresh_model_fn=_make_q_change_tiny_model,
        batch={
            "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
            "target": torch.ones(2, 4),
        },
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )

    with pytest.raises(ValueError, match="readiness row flip"):
        validate_trainer_sub2_authority_roundtrip_receipt(
            replace(receipt, readiness_row_flip_authorized=True)
        )
    with pytest.raises(ValueError, match="learning claim"):
        validate_trainer_sub2_authority_roundtrip_receipt(
            replace(receipt, learning_claim=True)
        )
    with pytest.raises(ValueError, match="eligible FP masters authoritative"):
        validate_trainer_sub2_authority_roundtrip_receipt(
            replace(receipt, eligible_fp_masters_authoritative=True)
        )
    with pytest.raises(ValueError, match="dense int16 persistent accumulator saved"):
        validate_trainer_sub2_authority_roundtrip_receipt(
            replace(receipt, dense_int16_persistent_accumulator_saved=True)
        )
    with pytest.raises(ValueError, match="poisoned FP-master bypass falsified"):
        validate_trainer_sub2_authority_roundtrip_receipt(
            replace(receipt, poisoned_fp_master_bypass_falsified=False)
        )


def test_trainer_default_off_smoke_still_writes_checkpoint(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert ckpt_root.exists()
    assert ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c1_proof_exits_before_checkpoint_write(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
        sub2_authority_construction_proof=True,
        sub2_authority_eligible_scope="first-bitlinear",
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert not ckpt_root.exists()
    assert not ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c2_local_update_proof_exits_before_checkpoint_write(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
        sub2_authority_local_update_proof=True,
        sub2_authority_eligible_scope="first-bitlinear",
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert not ckpt_root.exists()
    assert not ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c4a_roundtrip_proof_exits_before_checkpoint_write(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
        sub2_authority_roundtrip_proof=True,
        sub2_authority_eligible_scope="first-bitlinear",
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert not ckpt_root.exists()
    assert not ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c1_proof_requires_ternary_bulk(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        sub2_authority_construction_proof=True,
        use_ternary_bulk=False,
    )

    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_trainer_enabled_2c2_proof_requires_ternary_bulk(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        sub2_authority_local_update_proof=True,
        use_ternary_bulk=False,
    )

    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_trainer_enabled_2c4a_proof_requires_ternary_bulk(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        sub2_authority_roundtrip_proof=True,
        use_ternary_bulk=False,
    )

    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_exports_from_native_full_stack_facade():
    receipt = native_full_stack.build_trainer_sub2_authority_construction_receipt(
        _TinyTernary(),
        use_ternary_bulk=True,
    )
    local_receipt = native_full_stack.build_trainer_sub2_authority_local_update_receipt(
        _make_q_change_tiny_model(),
        batch={
            "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
            "target": torch.ones(2, 4),
        },
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )
    roundtrip_receipt = native_full_stack.build_trainer_sub2_authority_roundtrip_receipt(
        _make_q_change_tiny_model(),
        fresh_model_fn=_make_q_change_tiny_model,
        batch={
            "x": torch.arange(32, dtype=torch.float32).view(2, 16) / 16.0,
            "target": torch.ones(2, 4),
        },
        forward_loss_fn=_tiny_mse_loss,
        use_ternary_bulk=True,
    )

    assert receipt.pass_receipt is True
    assert local_receipt.pass_receipt is True
    assert roundtrip_receipt.pass_receipt is True
    assert "build_trainer_sub2_authority_construction_receipt" in native_full_stack.__all__
    assert "build_trainer_sub2_authority_local_update_receipt" in native_full_stack.__all__
    assert "validate_trainer_sub2_authority_local_update_receipt" in native_full_stack.__all__
    assert "build_trainer_sub2_authority_roundtrip_receipt" in native_full_stack.__all__
    assert "validate_trainer_sub2_authority_roundtrip_receipt" in native_full_stack.__all__
