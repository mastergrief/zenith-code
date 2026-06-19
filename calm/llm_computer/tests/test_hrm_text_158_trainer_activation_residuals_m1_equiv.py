"""R2-A-M1 CPU lossless equivalence tests (V0–V21)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.config import HierarchicalReasoningModelConfig
from calm.hrm_text_158.hrm import HierarchicalReasoningModel
from calm.hrm_text_158.native_full_stack import (
    ACTIVATION_RESIDUAL_TARGET_FAMILIES,
    apply_live_activation_residuals_surface_overrides,
)
from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
    MECHANISM_ID,
    PROOF_KIND_CPU_PRODUCTION_LOSSLESS_EQUIVALENCE,
    Tier1LosslessSeamSavedTensorHookRematCodec,
    UpstreamRecomputeKeyBinding,
    audit_recompute_closure,
    build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5,
    build_trainer_activation_residuals_lossless_equivalence_receipt,
    gate_b_passes_from_audit,
    resolve_capture_binding,
    sanitize_seq_info_for_recompute,
)
from calm.hrm_text_158.transformer import Transformer, TransformerBlock
from calm.hrm_text_158.config import TransformerConfig
from calm.llm_computer.tests.test_hrm_text_158_activation_relief import _tiny_config
from calm.llm_computer.tests.test_hrm_text_158_trainer_activation_residuals_seam_proof import (
    _common_train_kwargs,
    _make_tiny_gsm8k_rows,
    _splits_loader_factory,
)
from scripts.train_hrm_text_158 import train


def _tiny_transformer(*, n_layers: int = 2) -> Transformer:
    cfg = TransformerConfig(
        max_seq_len=32,
        n_layers=n_layers,
        hidden_size=32,
        num_heads=2,
        expansion=4,
        attn_type="prefixlm",
        init_type="lecun_normal",
        norm_type="pre",
        norm_eps=1e-5,
        pos_emb_type="rope",
        rope_theta=10000.0,
    )
    return Transformer(cfg)


def _tiny_transformer_block() -> TransformerBlock:
    return _tiny_transformer(n_layers=1).layers[0]


def _run_hrm_with_codec(
    *,
    bp_steps: int = 5,
    H_cycles: int = 2,
    L_cycles: int = 2,
) -> Tier1LosslessSeamSavedTensorHookRematCodec:
    torch.manual_seed(158)
    cfg = _tiny_config()
    cfg = HierarchicalReasoningModelConfig(
        **{**cfg.__dict__, "H_cycles": H_cycles, "L_cycles": L_cycles}
    )
    hrm = HierarchicalReasoningModel(cfg)
    hrm.train()
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(model=hrm)
    x = torch.randn(2, 16, 32, requires_grad=True)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)
    with codec.saved_tensor_hook_scope():
        _, hidden = hrm(
            None,
            x,
            bp_steps=bp_steps,
            sep_positions=sep,
            position_ids=pos,
            activation_codec_seam=codec,
        )
        hidden.square().sum().backward()
    return codec


def test_v0_mechanism_store_has_no_full_fp_payloads():
    codec = _run_hrm_with_codec()
    telemetry = codec.telemetry()
    assert telemetry["registered_seam_tensor_in_closure_count"] == 0
    assert telemetry["cross_family_full_seam_output_tensor_capture_count"] == 0


def test_v1_all_four_families_fire_pack_path():
    codec = _run_hrm_with_codec()
    families = {event["family"] for event in codec.seam_events}
    assert families == set(ACTIVATION_RESIDUAL_TARGET_FAMILIES)
    assert int(codec.telemetry()["seam_handle_pack_count"]) > 0


def test_v2_unpack_recompute_count_positive():
    codec = _run_hrm_with_codec()
    assert codec.recipe_store.unpack_recompute_count > 0


def test_v3_bit_identical_loss_and_grads_via_trainer(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_lossless_equivalence_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_v4_zero_full_tensor_packs():
    codec = _run_hrm_with_codec()
    assert int(codec.telemetry()["m1_seam_full_tensor_save_count_at_pack"]) == 0


def test_v4b_full_tensor_fallback_detection_and_telemetry():
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(
        model=torch.nn.Linear(4, 4),
    )
    registered = torch.randn(2, 4, requires_grad=True)
    codec.handle_registry.register_seam_tensor(registered, 1)
    alias = registered.view(2, 4)
    assert codec.handle_registry.lookup_registered_seam_tensor(alias) is None
    assert codec.handle_registry.is_registered_seam_tensor_or_alias(alias)

    def pack_hook(t: torch.Tensor) -> object:
        key = codec.handle_registry.lookup_registered_seam_tensor(t)
        if key is not None:
            codec.recipe_store.record_pack(
                recompute_key=key,
                tensor_id=id(t),
                kind="handle",
            )
            return ("seam_remat_handle", key)
        if codec.handle_registry.is_registered_seam_tensor_or_alias(t):
            codec.recipe_store.record_full_tensor_pack(tensor_id=id(t))
        return t

    packed = pack_hook(alias)
    assert packed is alias
    assert int(codec.telemetry()["registered_seam_tensor_full_pack_count"]) == 1


def _build_degenerate_lossless_equiv_receipt(**overrides):
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        zL_init_observation_from_hrm_module,
    )
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        TrainerActivationResidualsLosslessEquivalenceReceipt,
    )
    from calm.hrm_text_158.hrm import HierarchicalReasoningModel
    from calm.llm_computer.tests.test_hrm_text_158_activation_relief import _tiny_config

    codec = _run_hrm_with_codec()
    hrm = HierarchicalReasoningModel(_tiny_config())
    base = build_trainer_activation_residuals_lossless_equivalence_receipt(
        source_commit_sha="abc123",
        proof_command_argv=("test",),
        seam_events=codec.seam_events,
        zL_init_observation=zL_init_observation_from_hrm_module(hrm),
        telemetry=codec.telemetry(),
        main_path_proven=True,
        main_autograd_path_differs_from_baseline=True,
    )
    payload = {
        "schema_version": base.schema_version,
        "target_name": base.target_name,
        "proof_kind": base.proof_kind,
        "mechanism_id": base.mechanism_id,
        "source_commit_sha": base.source_commit_sha,
        "proof_command_argv": base.proof_command_argv,
        "main_path_proven": base.main_path_proven,
        "main_autograd_path_differs_from_baseline": base.main_autograd_path_differs_from_baseline,
        "universal_capture_to_key_rule_applied": base.universal_capture_to_key_rule_applied,
        "hook_scope_wraps_forward_saves": base.hook_scope_wraps_forward_saves,
        "registered_seam_tensor_pack_count": base.registered_seam_tensor_pack_count,
        "seam_handle_pack_count": base.seam_handle_pack_count,
        "registered_seam_tensor_full_pack_count": base.registered_seam_tensor_full_pack_count,
        "m1_seam_remat_unpack_recompute_count_total": base.m1_seam_remat_unpack_recompute_count_total,
        "registered_seam_tensor_in_closure_count": base.registered_seam_tensor_in_closure_count,
        "cross_family_full_seam_output_tensor_capture_count": base.cross_family_full_seam_output_tensor_capture_count,
        "recompute_registration_side_effect_count": base.recompute_registration_side_effect_count,
        "fail_closed_receipt": base.fail_closed_receipt,
        "live_readiness_row_flip_authorized": base.live_readiness_row_flip_authorized,
        "readiness_row_flip_authorized_surface_names": base.readiness_row_flip_authorized_surface_names,
        "optimizer_step_called": base.optimizer_step_called,
        "non_claims": base.non_claims,
    }
    payload.update(overrides)
    return TrainerActivationResidualsLosslessEquivalenceReceipt(**payload)


def test_validator_rejects_main_autograd_path_not_differing():
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )

    receipt = _build_degenerate_lossless_equiv_receipt(
        main_autograd_path_differs_from_baseline=False,
    )
    with pytest.raises(ValueError, match="main autograd path must differ"):
        validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)


def test_validator_rejects_zero_handle_packs():
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )

    receipt = _build_degenerate_lossless_equiv_receipt(
        registered_seam_tensor_pack_count=0,
        seam_handle_pack_count=0,
        m1_seam_remat_unpack_recompute_count_total=0,
    )
    with pytest.raises(ValueError, match="pack count must be positive"):
        validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)


def test_validator_rejects_hook_scope_not_wrapping_forward():
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )

    receipt = _build_degenerate_lossless_equiv_receipt(
        hook_scope_wraps_forward_saves=False,
    )
    with pytest.raises(ValueError, match="hook scope must wrap forward"):
        validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)


def test_validator_rejects_gate_a_without_unpack_recompute():
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        ActivationResidualLiveTensorFamilyObservation,
        ActivationResidualsFailClosedReceipt,
        ZLInitPersistentNonClaim,
    )
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )

    receipt = _build_degenerate_lossless_equiv_receipt()
    forged_fc = ActivationResidualsFailClosedReceipt(
        schema_version=receipt.fail_closed_receipt.schema_version,
        target_name=receipt.fail_closed_receipt.target_name,
        target_families=receipt.fail_closed_receipt.target_families,
        activations_residuals_sub2_claim=False,
        real_sub2_or_remat_or_offload_mechanism_present=True,
        no_hidden_bf16_authority_proven=True,
        gpu_memory_receipt_present=False,
        lossy_or_compression_claim=False,
        ready_to_flip=False,
        blocked_reason=receipt.fail_closed_receipt.blocked_reason,
        observed_families=receipt.fail_closed_receipt.observed_families,
        zL_init_non_claim=receipt.fail_closed_receipt.zL_init_non_claim,
        smallest_missing_proof=receipt.fail_closed_receipt.smallest_missing_proof,
        non_claims=receipt.fail_closed_receipt.non_claims,
    )
    receipt = _build_degenerate_lossless_equiv_receipt(
        fail_closed_receipt=forged_fc,
        m1_seam_remat_unpack_recompute_count_total=0,
        registered_seam_tensor_pack_count=0,
        seam_handle_pack_count=0,
    )
    with pytest.raises(ValueError, match="pack count must be positive"):
        validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)


def test_audit_inspects_container_tensor_in_sanitized_kwargs():
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(
        model=torch.nn.Linear(4, 4),
    )
    registry = codec.handle_registry
    tensor = torch.randn(2, 4)
    registry.register_seam_tensor(tensor, 9)
    sanitized = {"nested": {"hidden": tensor}}

    def bad_fn() -> torch.Tensor:
        return sanitized["nested"]["hidden"]

    bad_fn.__closure_capture__ = {"sanitized_kwargs": sanitized}  # type: ignore[attr-defined]
    audit = audit_recompute_closure(bad_fn, handle_registry=registry)
    assert int(audit["registered_seam_tensor_in_closure_count"]) > 0
    assert not gate_b_passes_from_audit(audit)


def test_v7_receipt_gates_a_and_b_true_gpu_false(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    receipt_path = tmp_path / "m1_equiv.json"
    monkeypatch.setenv("R2A_M1_LOSSLESS_EQUIV_RECEIPT_JSON", str(receipt_path))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_lossless_equivalence_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["proof_kind"] == PROOF_KIND_CPU_PRODUCTION_LOSSLESS_EQUIVALENCE
    assert payload["mechanism_id"] == MECHANISM_ID
    fc = payload["fail_closed_receipt"]
    assert fc["real_sub2_or_remat_or_offload_mechanism_present"] is True
    assert fc["no_hidden_bf16_authority_proven"] is True
    assert fc["gpu_memory_receipt_present"] is False
    assert fc["ready_to_flip"] is False
    assert payload["live_readiness_row_flip_authorized"] is False


def test_v8_applier_rejects_cpu_lossless_equiv_receipt(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    receipt_path = tmp_path / "m1_equiv.json"
    monkeypatch.setenv("R2A_M1_LOSSLESS_EQUIV_RECEIPT_JSON", str(receipt_path))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_lossless_equivalence_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    from calm.hrm_text_158.native_full_stack.activation_residuals_m1_remat import (
        TrainerActivationResidualsLosslessEquivalenceReceipt,
        validate_trainer_activation_residuals_lossless_equivalence_receipt,
    )
    from calm.hrm_text_158.native_full_stack.activation_relief import (
        ActivationResidualLiveTensorFamilyObservation,
        ActivationResidualsFailClosedReceipt,
        ZLInitPersistentNonClaim,
    )

    fc = payload["fail_closed_receipt"]
    fail_closed = ActivationResidualsFailClosedReceipt(
        schema_version=fc["schema_version"],
        target_name=fc["target_name"],
        target_families=tuple(fc["target_families"]),
        activations_residuals_sub2_claim=fc["activations_residuals_sub2_claim"],
        real_sub2_or_remat_or_offload_mechanism_present=fc[
            "real_sub2_or_remat_or_offload_mechanism_present"
        ],
        no_hidden_bf16_authority_proven=fc["no_hidden_bf16_authority_proven"],
        gpu_memory_receipt_present=fc["gpu_memory_receipt_present"],
        lossy_or_compression_claim=fc["lossy_or_compression_claim"],
        ready_to_flip=fc["ready_to_flip"],
        blocked_reason=fc["blocked_reason"],
        observed_families=tuple(
            ActivationResidualLiveTensorFamilyObservation(
                family=obs["family"],
                observed_count=int(obs["observed_count"]),
                shapes=tuple(tuple(shape) for shape in obs["shapes"]),
                dtypes=tuple(obs["dtypes"]),
                devices=tuple(obs["devices"]),
                requires_grad_values=tuple(obs["requires_grad_values"]),
                mechanism=str(obs.get("mechanism", "observer_returns_original_tensor")),
            )
            for obs in fc["observed_families"]
        ),
        zL_init_non_claim=ZLInitPersistentNonClaim(
            name="zL_init",
            classification=fc["zL_init_non_claim"]["classification"],
            registry_anchor=fc["zL_init_non_claim"]["registry_anchor"],
            source_anchor=fc["zL_init_non_claim"]["source_anchor"],
            dtype=fc["zL_init_non_claim"]["dtype"],
            shape=tuple(fc["zL_init_non_claim"]["shape"]),
            persistent=fc["zL_init_non_claim"]["persistent"],
            non_claim=fc["zL_init_non_claim"]["non_claim"],
        ),
        smallest_missing_proof=fc["smallest_missing_proof"],
        non_claims=tuple(fc["non_claims"]),
    )
    receipt = TrainerActivationResidualsLosslessEquivalenceReceipt(
        schema_version=payload["schema_version"],
        target_name=payload["target_name"],
        proof_kind=payload["proof_kind"],
        mechanism_id=payload["mechanism_id"],
        source_commit_sha=payload["source_commit_sha"],
        proof_command_argv=tuple(payload["proof_command_argv"]),
        main_path_proven=payload["main_path_proven"],
        main_autograd_path_differs_from_baseline=payload[
            "main_autograd_path_differs_from_baseline"
        ],
        universal_capture_to_key_rule_applied=payload[
            "universal_capture_to_key_rule_applied"
        ],
        hook_scope_wraps_forward_saves=payload["hook_scope_wraps_forward_saves"],
        registered_seam_tensor_pack_count=payload["registered_seam_tensor_pack_count"],
        seam_handle_pack_count=payload["seam_handle_pack_count"],
        registered_seam_tensor_full_pack_count=payload[
            "registered_seam_tensor_full_pack_count"
        ],
        m1_seam_remat_unpack_recompute_count_total=payload[
            "m1_seam_remat_unpack_recompute_count_total"
        ],
        registered_seam_tensor_in_closure_count=payload[
            "registered_seam_tensor_in_closure_count"
        ],
        cross_family_full_seam_output_tensor_capture_count=payload[
            "cross_family_full_seam_output_tensor_capture_count"
        ],
        recompute_registration_side_effect_count=payload[
            "recompute_registration_side_effect_count"
        ],
        fail_closed_receipt=fail_closed,
        live_readiness_row_flip_authorized=payload["live_readiness_row_flip_authorized"],
        readiness_row_flip_authorized_surface_names=tuple(
            payload["readiness_row_flip_authorized_surface_names"]
        ),
        optimizer_step_called=payload["optimizer_step_called"],
        non_claims=tuple(payload["non_claims"]),
    )
    validate_trainer_activation_residuals_lossless_equivalence_receipt(receipt)
    with pytest.raises(ValueError, match="cannot flip live scaffold"):
        apply_live_activation_residuals_surface_overrides(receipt)


def test_v9_flag_mutually_exclusive_with_r2a_seam(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_lossless_equivalence_proof=True,
        activation_residuals_fail_closed_proof=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_v13_sanitize_strips_activation_codec_seam():
    seq = {"activation_codec_seam": object(), "bp_steps": 3}
    sanitized = sanitize_seq_info_for_recompute(seq)
    assert "activation_codec_seam" not in sanitized
    assert sanitized["bp_steps"] == 3


def test_v17_negative_hook_ordering_yields_zero_handle_packs():
    torch.manual_seed(21)
    hrm = HierarchicalReasoningModel(_tiny_config())
    hrm.train()
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(model=hrm)
    x = torch.randn(2, 16, 32, requires_grad=True)
    sep = torch.tensor([5, 7], dtype=torch.long)
    pos = torch.arange(16, dtype=torch.long).unsqueeze(0).expand(2, -1)
    _, hidden = hrm(
        None,
        x,
        bp_steps=5,
        sep_positions=sep,
        position_ids=pos,
        activation_codec_seam=codec,
    )

    def pack_hook(tensor: torch.Tensor) -> object:
        key = codec.handle_registry.lookup_registered_seam_tensor(tensor)
        if key is not None:
            return ("seam_remat_handle", key)
        return tensor

    def unpack_hook(saved: object) -> torch.Tensor:
        if isinstance(saved, tuple) and saved[0] == "seam_remat_handle":
            return codec.recipe_store.recompute(int(saved[1]))
        return saved  # type: ignore[return-value]

    with torch.autograd.graph.saved_tensors_hooks(pack_hook, unpack_hook):
        hidden.square().sum().backward()
    assert len(codec.recipe_store.pack_events) == 0


def test_v18_positive_hook_during_forward_yields_handle_packs():
    codec = _run_hrm_with_codec()
    assert int(codec.telemetry()["seam_handle_pack_count"]) > 0


def test_v19_recurrence_second_plus_l_iter_z_l_in_is_upstream_key():
    codec = _run_hrm_with_codec(H_cycles=2, L_cycles=2)
    z_l_events = [
        event
        for event in codec.seam_events
        if event["family"] == "recurrent.z_L_update"
    ]
    assert len(z_l_events) >= 2, "expected multiple L iterations for recurrence topology"
    later_events = z_l_events[1:]
    assert all(
        event["binding_classes"][0] == "upstream_recompute_key"
        for event in later_events
    ), "2nd+ L iteration z_L_in must bind upstream key from prior z_L_update"


def test_v20_multi_layer_transformer_forward_x_prev_from_prior_post_mlp():
    torch.manual_seed(220)
    transformer = _tiny_transformer(n_layers=2)
    transformer.train()
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(
        model=transformer,
    )
    x = torch.randn(2, 8, 32, requires_grad=True)
    sep = torch.tensor([3, 5], dtype=torch.long)
    pos = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    with codec.saved_tensor_hook_scope():
        out = transformer(
            x,
            position_ids=pos,
            sep_positions=sep,
            activation_codec_seam=codec,
        )
        out.square().sum().backward()
    post_attn_events = [
        event for event in codec.seam_events if event["family"] == "residual.post_attn"
    ]
    assert len(post_attn_events) >= 2, "expected post_attn events from both layers"
    layer1_post_attn = post_attn_events[1]
    assert layer1_post_attn["binding_classes"][0] == "upstream_recompute_key", (
        "layer-1 x_prev must bind upstream key from layer-0 post_mlp seam output"
    )


def test_v21_universal_negative_registered_tensor_in_closure_fails_gate_b():
    registry = Tier1LosslessSeamSavedTensorHookRematCodec(
        model=torch.nn.Linear(4, 4),
    ).handle_registry
    tensor = torch.randn(2, 4)
    registry.register_seam_tensor(tensor, 7)

    def bad_fn() -> torch.Tensor:
        return tensor

    audit = audit_recompute_closure(bad_fn, handle_registry=registry)
    assert int(audit["registered_seam_tensor_in_closure_count"]) > 0
    assert not gate_b_passes_from_audit(audit)


def test_universal_resolver_prefers_upstream_key_over_tensor():
    codec = build_tier1_lossless_seam_saved_tensor_hook_remat_codec_v5(
        model=torch.nn.Linear(4, 4),
    )
    tensor = torch.randn(2, 4)
    codec.handle_registry.register_seam_tensor(tensor, 3)
    binding = resolve_capture_binding(
        tensor,
        handle_registry=codec.handle_registry,
        model=codec.model,
        aux_role_by_id={},
    )
    assert isinstance(binding, UpstreamRecomputeKeyBinding)
    assert binding.recompute_key == 3
