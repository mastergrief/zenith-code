"""C2.1 bounded-delta acquisition harness CPU/static tests."""
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.curriculum import BroadTokenizer
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    make_bounded_tensor_state,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    DEFAULT_PARENT_SHA256,
    FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
    HISTORICAL_IDENTITY_CONTROL,
    RUN_C2_GPU_LAUNCH_ENV,
    build_identity_full_batch,
    build_model_from_checkpoint,
    compute_forward_level_init_fidelity,
    cuda_memory_receipt,
    cuda_memory_stats_device_arg,
    derive_bounded_tensor_state_from_weight,
    derive_tensor_states_and_check_init_fidelity,
    file_sha256,
    guard_gpu_launch,
    identity_full_support_control_proof,
    native_ternary_effective_weight,
    reset_cuda_memory_stats,
    run_c2p1_probe,
    select_eligible_bitlinears,
)
from scripts.train_hrm_text_158 import _build_ckpt_config, SOURCE_PIN


TINY_ARCH = dict(
    max_len=64,
    hidden_size=64,
    n_layers=2,
    num_heads=2,
    expansion=4,
    H_cycles=1,
    L_cycles=1,
    half_layers=True,
    bp_warmup_ratio=0.2,
    bp_min_steps=1,
    bp_max_steps=2,
)


def _tiny_parent_blob(*, batch_size: int = 2) -> dict:
    tok = BroadTokenizer()
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=TINY_ARCH["max_len"],
        n_layers=TINY_ARCH["n_layers"],
        hidden_size=TINY_ARCH["hidden_size"],
        num_heads=TINY_ARCH["num_heads"],
        expansion=TINY_ARCH["expansion"],
        H_cycles=TINY_ARCH["H_cycles"],
        L_cycles=TINY_ARCH["L_cycles"],
        half_layers=TINY_ARCH["half_layers"],
        bp_warmup_ratio=TINY_ARCH["bp_warmup_ratio"],
        bp_min_steps=TINY_ARCH["bp_min_steps"],
        bp_max_steps=TINY_ARCH["bp_max_steps"],
        use_ternary_bulk=True,
    )
    model = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=tok.vocab_size))
    return {
        "model_state": model.state_dict(),
        "config": _build_ckpt_config(
            model,
            tok,
            cfg,
            TINY_ARCH["max_len"],
            batch_size=batch_size,
            curriculum_rung="L0c1",
            curriculum_seed=17,
            replay_ratio=0.0,
            prior_rungs=[],
        ),
        "step": 50,
        "epoch": 1,
        "source_pin": SOURCE_PIN,
    }


def _tiny_forward_fixture(*, batch_size: int = 2, eligible_scope: str = "first-bitlinear"):
    device = torch.device("cpu")
    ckpt = _tiny_parent_blob(batch_size=batch_size)
    model, tok, _cfg = build_model_from_checkpoint(ckpt, device)
    batch, _batch_proof = build_identity_full_batch(
        tok=tok,
        max_len=TINY_ARCH["max_len"],
        batch_size=batch_size,
        curriculum_seed=17,
        device=device,
    )
    eligible = select_eligible_bitlinears(model, eligible_scope=eligible_scope)
    states, report = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)
    assert report["all_pass"] is True
    return model, batch, eligible, states


def test_gpu_guard_requires_explicit_launch_env(monkeypatch):
    monkeypatch.delenv(RUN_C2_GPU_LAUNCH_ENV, raising=False)

    with pytest.raises(RuntimeError, match="persisted \\+1 LAUNCH"):
        guard_gpu_launch(torch.device("cuda:0"), allow_gpu_launch=False)

    with pytest.raises(RuntimeError, match=RUN_C2_GPU_LAUNCH_ENV):
        guard_gpu_launch(torch.device("cuda:0"), allow_gpu_launch=True)


def test_cuda_memory_stats_device_arg_normalizes_without_real_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 4)

    assert cuda_memory_stats_device_arg(torch.device("cuda:2")) == 2
    assert cuda_memory_stats_device_arg(torch.device("cuda")) == 4


def test_cuda_memory_stats_reset_and_receipt_use_normalized_device(monkeypatch):
    calls = []

    def record(name, value):
        calls.append((name, value))

    monkeypatch.setattr(torch.cuda, "set_device", lambda device: record("set_device", device))
    monkeypatch.setattr(
        torch.cuda,
        "reset_peak_memory_stats",
        lambda device: record("reset_peak_memory_stats", device),
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_allocated",
        lambda device: record("max_memory_allocated", device) or 11,
    )
    monkeypatch.setattr(
        torch.cuda,
        "max_memory_reserved",
        lambda device: record("max_memory_reserved", device) or 22,
    )
    monkeypatch.setattr(
        torch.cuda,
        "memory_allocated",
        lambda device: record("memory_allocated", device) or 3,
    )

    assert reset_cuda_memory_stats(torch.device("cuda:0")) == 0
    receipt = cuda_memory_receipt(torch.device("cuda:0"))

    assert receipt["cuda_memory_stats_device"] == 0
    assert receipt["cuda_peak_allocated_bytes"] == 11
    assert receipt["cuda_peak_reserved_bytes"] == 22
    assert receipt["cuda_final_allocated_bytes"] == 3
    assert calls == [
        ("set_device", 0),
        ("reset_peak_memory_stats", 0),
        ("set_device", 0),
        ("max_memory_allocated", 0),
        ("max_memory_reserved", 0),
        ("memory_allocated", 0),
    ]


def test_identity_full_control_is_historical_positive_not_same_harness_control():
    proof = identity_full_support_control_proof(17)

    assert proof["historical_control"] == HISTORICAL_IDENTITY_CONTROL
    assert proof["historical_control"]["parent_sha256"] == DEFAULT_PARENT_SHA256
    assert proof["train_rows_qe_match_support"] is True
    assert proof["seed_independent_support"] is True
    assert proof["same_harness_paired_int16_control"] is False
    assert proof["inline_control_required"] is False


def test_weight_level_init_fidelity_detects_corrupted_q_state():
    module = BitLinear(3, 2, bias=False)
    with torch.no_grad():
        module.weight.fill_(0.5)
    eligible = {"proj": module}

    states, report = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)

    assert report["all_pass"] is True
    assert report["modules"]["proj"]["max_abs_diff"] == 0.0

    corrupted = {
        "proj": make_bounded_tensor_state(
            "proj",
            torch.zeros_like(states["proj"].q_levels),
            states["proj"].frozen_scale,
        )
    }
    bounded_effective = corrupted["proj"].materialized_weight(device="cpu", requires_grad=False)
    native_effective = native_ternary_effective_weight(module)
    assert torch.max(torch.abs(bounded_effective)).item() == 0.0
    assert torch.max(torch.abs(bounded_effective - native_effective)).item() == 0.5


def test_forward_level_init_fidelity_passes_on_tiny_real_model():
    model, batch, eligible, states = _tiny_forward_fixture()

    report = compute_forward_level_init_fidelity(
        model,
        batch,
        states,
        eligible,
        device=torch.device("cpu"),
        threshold=0.0,
        eligible_scope="first-bitlinear",
        total_steps=1,
    )

    assert report["status"] == "computed"
    assert report["pass"] is True
    assert report["threshold_requested"] == 0.0
    assert report["threshold"] == FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL
    assert "Native BitLinear training forward" in report["threshold_reason"]
    assert report["eligible_module_count"] == 1
    assert report["eligible_modules"] == sorted(eligible)
    assert report["logits_max_abs_diff"] <= report["threshold"]
    assert report["loss_abs_diff"] <= report["threshold"]
    assert report["module_output_max_abs_diff"] <= report["threshold"]
    assert report["max_abs_diff"] <= report["threshold"]
    module_fidelity = report["module_output_fidelity"]
    assert module_fidelity["eligible_scope"] == "first-bitlinear"
    assert module_fidelity["all_pass"] is True
    assert module_fidelity["module_count"] == len(eligible)
    assert module_fidelity["eligible_modules"] == sorted(eligible)
    module_report = module_fidelity["modules"][next(iter(eligible))]
    assert module_report["invocation_count"] > 0
    assert module_report["native_invocation_count"] == module_report["bounded_invocation_count"]
    assert module_report["max_abs_diff"] <= report["threshold"]
    assert module_report["allclose"] is True
    assert module_report["pass"] is True


def test_forward_level_init_fidelity_emits_module_output_telemetry_for_all_bitlinears():
    model, batch, eligible, states = _tiny_forward_fixture(eligible_scope="all-bitlinear")

    report = compute_forward_level_init_fidelity(
        model,
        batch,
        states,
        eligible,
        device=torch.device("cpu"),
        threshold=0.0,
        eligible_scope="all-bitlinear",
        total_steps=1,
    )

    module_fidelity = report["module_output_fidelity"]
    assert report["pass"] is True
    assert report["module_outputs_allclose"] is True
    assert module_fidelity["schema"] == "hrm_text_158_c2p1_module_output_init_fidelity/v0"
    assert module_fidelity["eligible_scope"] == "all-bitlinear"
    assert module_fidelity["all_pass"] is True
    assert module_fidelity["module_count"] == len(eligible)
    assert module_fidelity["eligible_modules"] == sorted(eligible)
    assert set(module_fidelity["modules"]) == set(eligible)
    for module_report in module_fidelity["modules"].values():
        assert module_report["native_invocation_count"] == module_report["bounded_invocation_count"]
        assert module_report["invocation_count"] == module_report["aligned_invocation_count"]
        assert module_report["invocation_count"] > 0
        assert module_report["first_output_shape"] is not None
        assert module_report["shape_mismatch_count"] == 0
        assert module_report["max_abs_diff"] <= report["threshold"]
        assert module_report["allclose"] is True
        assert module_report["pass"] is True


def test_forward_level_init_fidelity_hard_fails_on_corrupted_q_state():
    model, batch, eligible, states = _tiny_forward_fixture()
    state_key = next(iter(states))
    corrupted_q = states[state_key].q_levels.clone().mul(-1)
    if torch.equal(corrupted_q, states[state_key].q_levels):
        corrupted_q.fill_(1)
    corrupted_states = dict(states)
    corrupted_states[state_key] = make_bounded_tensor_state(
        state_key,
        corrupted_q,
        states[state_key].frozen_scale,
    )

    with pytest.raises(RuntimeError, match="forward-level init-fidelity allclose failed"):
        compute_forward_level_init_fidelity(
            model,
            batch,
            corrupted_states,
            eligible,
            device=torch.device("cpu"),
            threshold=0.0,
            eligible_scope="first-bitlinear",
            total_steps=1,
        )


def test_forward_level_init_fidelity_hard_fails_on_corrupted_all_bitlinear_q_state():
    model, batch, eligible, states = _tiny_forward_fixture(eligible_scope="all-bitlinear")
    state_key = sorted(states)[0]
    corrupted_q = states[state_key].q_levels.clone().mul(-1)
    if torch.equal(corrupted_q, states[state_key].q_levels):
        corrupted_q.fill_(1)
    corrupted_states = dict(states)
    corrupted_states[state_key] = make_bounded_tensor_state(
        state_key,
        corrupted_q,
        states[state_key].frozen_scale,
    )

    with pytest.raises(RuntimeError, match="forward-level init-fidelity allclose failed"):
        compute_forward_level_init_fidelity(
            model,
            batch,
            corrupted_states,
            eligible,
            device=torch.device("cpu"),
            threshold=0.0,
            eligible_scope="all-bitlinear",
            total_steps=1,
        )


def test_tiny_real_model_cpu_step_receipt_is_scratch_only(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=2,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
    )

    assert receipt["gpu_launched"] is False
    assert receipt["checkpoint_written"] is False
    assert receipt["banked_pt_mutated"] is False
    assert receipt["creditdir_mutated"] is False
    assert receipt["parent_hash_before"] == parent_sha
    assert receipt["parent_hash_after"] == parent_sha
    assert receipt["parent_hash_unchanged"] is True
    assert receipt["eligible_module_count"] == 1
    assert receipt["weight_level_init_fidelity"]["all_pass"] is True
    assert receipt["forward_level_init_fidelity"]["status"] == "computed"
    assert receipt["forward_level_init_fidelity"]["pass"] is True
    assert receipt["forward_level_init_fidelity"]["threshold_requested"] == 0.0
    assert receipt["forward_level_init_fidelity"]["threshold"] == FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL
    assert receipt["forward_level_init_fidelity"]["logits_max_abs_diff"] <= FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL
    assert receipt["forward_level_init_fidelity"]["loss_abs_diff"] <= FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL
    assert receipt["forward_level_init_fidelity"]["module_outputs_allclose"] is True
    assert receipt["forward_level_init_fidelity"]["module_output_fidelity"]["all_pass"] is True
    assert receipt["forward_backward_update_executed"] is True
    assert receipt["step_reports"]["1"]["loss_finite"] is True
    assert receipt["step_reports"]["1"]["weighted_grad_finite"] is True
    assert receipt["step_reports"]["1"]["optimizer_identity_proof"]["pass"] is True
    assert receipt["checkpoint_payload"]["checkpoint_written"] is False
    assert Path(receipt["receipt_path"]).exists()


def test_default_derivation_import_is_exercised_for_script_surface():
    module = BitLinear(2, 1, bias=False)
    state = derive_bounded_tensor_state_from_weight("proj", module.weight)

    assert state.state_key == "proj"
    assert state.q_levels.dtype == torch.int8
