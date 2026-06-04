"""C2.1 bounded-delta acquisition harness CPU/static tests."""
from __future__ import annotations

import json
import math
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
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    authoritative_forward_context,
    make_bounded_tensor_state,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
    C2P2_TIMING_SCHEMA_VERSION,
    C2PhaseTimeout,
    DEFAULT_PARENT_SHA256,
    FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
    HISTORICAL_IDENTITY_CONTROL,
    PhaseProgress,
    RUN_C2_GPU_LAUNCH_ENV,
    aggregate_identity_full_audit_batch_reports,
    build_identity_full_batch,
    build_identity_full_support_batches,
    build_model_from_checkpoint,
    compare_module_output_fidelity,
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
    score_strict_exact_and_parsed_from_logits,
    select_eligible_bitlinears,
    _capture_eligible_module_outputs,
    enforce_phase_bound,
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


def _assert_finite_non_negative(value: float | int) -> None:
    assert math.isfinite(float(value))
    assert float(value) >= 0.0


def _metric_scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().item()
    return value


def _metrics_without_logits(metrics: dict) -> dict:
    out = {}
    for key, value in metrics.items():
        if key == "logits":
            continue
        if isinstance(value, tuple):
            out[key] = tuple(_metric_scalar(item) for item in value)
        else:
            out[key] = _metric_scalar(value)
    return out


def _forward_init_value_subset(
    *,
    native_loss: torch.Tensor,
    native_metrics: dict,
    native_module_outputs: dict,
    bounded_loss: torch.Tensor,
    bounded_metrics: dict,
    bounded_module_outputs: dict,
    eligible_scope: str,
) -> dict:
    native_logits = native_metrics["logits"].detach().to(torch.float32).cpu()
    bounded_logits = bounded_metrics["logits"].detach().to(torch.float32).cpu()
    logits_diff = (bounded_logits - native_logits).abs()
    loss_abs_diff = float(
        (
            bounded_loss.detach().to(torch.float32).cpu()
            - native_loss.detach().to(torch.float32).cpu()
        ).abs().item()
    )
    module_fidelity = compare_module_output_fidelity(
        native_module_outputs,
        bounded_module_outputs,
        threshold=FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
        eligible_scope=eligible_scope,
    )
    module_output_max_abs_diff = max(
        (
            float(item["max_abs_diff"])
            for item in module_fidelity["modules"].values()
        ),
        default=0.0,
    )
    logits_max_abs_diff = float(logits_diff.max().item()) if logits_diff.numel() else 0.0
    return {
        "logits_max_abs_diff": logits_max_abs_diff,
        "loss_abs_diff": loss_abs_diff,
        "module_output_max_abs_diff": module_output_max_abs_diff,
        "max_abs_diff": max(logits_max_abs_diff, loss_abs_diff, module_output_max_abs_diff),
        "logits_allclose": bool(
            torch.allclose(
                bounded_logits,
                native_logits,
                atol=FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
                rtol=0.0,
            )
        ),
        "loss_allclose": bool(
            torch.allclose(
                bounded_loss.detach().to(torch.float32).cpu(),
                native_loss.detach().to(torch.float32).cpu(),
                atol=FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
                rtol=0.0,
            )
        ),
        "module_outputs_allclose": bool(module_fidelity["all_pass"]),
    }


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


def test_support_cycler_builds_distinct_full_support_batches():
    tok = BroadTokenizer()

    batches, proof = build_identity_full_support_batches(
        tok=tok,
        max_len=TINY_ARCH["max_len"],
        batch_size=45,
        curriculum_seed=17,
        device=torch.device("cpu"),
    )

    assert proof["covers_full_support"] is True
    assert proof["usable_rows"] == 90
    assert proof["batch_count"] == 2
    assert proof["distinct_batch_count"] == 2
    assert proof["has_at_least_two_distinct_batches"] is True
    assert len({batch["metadata"]["batch_content_hash16"] for batch in batches}) == 2
    assert batches[0]["metadata"]["row_ids"] != batches[1]["metadata"]["row_ids"]


def test_audit_score_counts_known_k_and_parsed_independently():
    tok = BroadTokenizer()
    batches, _proof = build_identity_full_support_batches(
        tok=tok,
        max_len=TINY_ARCH["max_len"],
        batch_size=4,
        curriculum_seed=17,
        device=torch.device("cpu"),
    )
    labels = batches[0]["batch"]["labels"].detach().cpu()
    pred_ids = torch.full_like(labels, tok.pad_id)
    masks = labels != IGNORE_LABEL_ID

    # Rows 0-1 are strict-correct. Row 2 has correct numeric digits but a
    # wrong EOS token, proving parsed accuracy is not a strict-exact alias.
    for row_index in (0, 1, 2):
        pred_ids[row_index][masks[row_index]] = labels[row_index][masks[row_index]]
    eos_pos = int(torch.nonzero(masks[2], as_tuple=False)[-1].item())
    pred_ids[2, eos_pos] = tok.pad_id

    logits = torch.full(
        (labels.shape[0], labels.shape[1], tok.vocab_size),
        -1000.0,
        dtype=torch.float32,
    )
    for row_index in range(labels.shape[0]):
        for pos_index in range(labels.shape[1]):
            logits[row_index, pos_index, int(pred_ids[row_index, pos_index])] = 1000.0

    score = score_strict_exact_and_parsed_from_logits(
        logits,
        labels,
        tok=tok,
        include_row_results=True,
    )

    assert score["strict_exact_count"] == 2
    assert score["strict_exact_total"] == 4
    assert score["parsed_exact_count"] == 3
    assert score["parsed_exact_total"] == 4
    assert score["strict_exact_and_parsed_independent"] is True
    assert score["row_results"][2]["strict_exact"] is False
    assert score["row_results"][2]["parsed_exact"] is True


def test_audit_aggregation_counts_hand_built_k_of_n_fixture():
    report = aggregate_identity_full_audit_batch_reports(
        step=250,
        bp_steps=1,
        batch_reports=[
            {
                "metadata": {"batch_content_hash16": "batch-a"},
                "loss": 0.25,
                "metric_strict": {
                    "count": 2,
                    "total": 4,
                    "strict_exact": "2/4",
                },
                "strict_recomputed": {
                    "count": 2,
                    "total": 4,
                    "strict_exact": "2/4",
                },
                "parsed": {
                    "count": 3,
                    "total": 4,
                    "parsed_exact": "3/4",
                },
                "failure_examples": [],
            },
            {
                "metadata": {"batch_content_hash16": "batch-b"},
                "loss": 0.75,
                "metric_strict": {
                    "count": 1,
                    "total": 3,
                    "strict_exact": "1/3",
                },
                "strict_recomputed": {
                    "count": 1,
                    "total": 3,
                    "strict_exact": "1/3",
                },
                "parsed": {
                    "count": 2,
                    "total": 3,
                    "parsed_exact": "2/3",
                },
                "failure_examples": [],
            },
        ],
    )

    assert report["step"] == 250
    assert report["strict_exact_count"] == 3
    assert report["strict_exact_total"] == 7
    assert report["strict_exact"] == "3/7"
    assert report["parsed_exact_count"] == 5
    assert report["parsed_exact_total"] == 7
    assert report["parsed_exact"] == "5/7"
    assert report["strict_exact_recompute_matches_metric"] is True
    assert report["audit_mismatch"] is False
    assert report["audited_distinct_batch_count"] == 2
    assert report["loss_mean"] == 0.5


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


def test_no_grad_authoritative_forward_matches_native_outputs_metrics_and_captures_nothing():
    model, batch, eligible, states = _tiny_forward_fixture()
    extras = model.compute_train_extra_args(0, 1)

    model.train()
    model.zero_grad(set_to_none=True)
    with torch.no_grad():
        native_loss, native_metrics, native_module_outputs = _capture_eligible_module_outputs(
            model,
            batch,
            eligible,
            extras,
        )
        with authoritative_forward_context(
            eligible,
            states,
            device=torch.device("cpu"),
            requires_grad=True,
        ) as capture_on_handle:
            capture_on_loss, capture_on_metrics, capture_on_module_outputs = _capture_eligible_module_outputs(
                model,
                batch,
                eligible,
                extras,
            )
        with authoritative_forward_context(
            eligible,
            states,
            device=torch.device("cpu"),
            requires_grad=False,
        ) as capture_off_handle:
            capture_off_loss, capture_off_metrics, capture_off_module_outputs = _capture_eligible_module_outputs(
                model,
                batch,
                eligible,
                extras,
            )

    torch.testing.assert_close(
        capture_off_metrics["logits"].detach().cpu(),
        capture_on_metrics["logits"].detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    torch.testing.assert_close(
        capture_off_loss.detach().cpu(),
        capture_on_loss.detach().cpu(),
        atol=0.0,
        rtol=0.0,
    )
    assert _metrics_without_logits(capture_off_metrics) == _metrics_without_logits(capture_on_metrics)
    for state_key in eligible:
        capture_on_outputs = capture_on_module_outputs[state_key]
        capture_off_outputs = capture_off_module_outputs[state_key]
        assert len(capture_on_outputs) == len(capture_off_outputs)
        assert len(capture_on_outputs) > 0
        for capture_on_output, capture_off_output in zip(capture_on_outputs, capture_off_outputs):
            torch.testing.assert_close(capture_off_output, capture_on_output, atol=0.0, rtol=0.0)

    capture_on_fidelity_values = _forward_init_value_subset(
        native_loss=native_loss,
        native_metrics=native_metrics,
        native_module_outputs=native_module_outputs,
        bounded_loss=capture_on_loss,
        bounded_metrics=capture_on_metrics,
        bounded_module_outputs=capture_on_module_outputs,
        eligible_scope="first-bitlinear",
    )
    capture_off_fidelity_values = _forward_init_value_subset(
        native_loss=native_loss,
        native_metrics=native_metrics,
        native_module_outputs=native_module_outputs,
        bounded_loss=capture_off_loss,
        bounded_metrics=capture_off_metrics,
        bounded_module_outputs=capture_off_module_outputs,
        eligible_scope="first-bitlinear",
    )
    assert capture_off_fidelity_values == capture_on_fidelity_values
    public_report = compute_forward_level_init_fidelity(
        model,
        batch,
        states,
        eligible,
        device=torch.device("cpu"),
        threshold=0.0,
        eligible_scope="first-bitlinear",
        total_steps=1,
    )
    assert {
        key: public_report[key]
        for key in capture_off_fidelity_values
    } == capture_off_fidelity_values
    assert capture_on_handle.capture_enabled is True
    for capture in capture_on_handle.captures.values():
        assert len(capture["inputs"]) > 0
        assert capture["grad_outputs"] == []
    with pytest.raises(RuntimeError, match="captured inputs and grad_outputs"):
        capture_on_handle.weighted_grad(next(iter(eligible)))
    assert capture_off_handle.capture_enabled is False
    for capture in capture_off_handle.captures.values():
        assert capture["inputs"] == []
        assert capture["grad_outputs"] == []
    with pytest.raises(RuntimeError, match="capture is disabled"):
        capture_off_handle.weighted_grad(next(iter(eligible)))


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


def test_phase_progress_emits_schema_events_and_timeout_payload(capsys):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    progress = PhaseProgress(
        enabled=True,
        device=torch.device("cpu"),
        phase_timeout_seconds=10.0,
        total_timeout_seconds=10.0,
        clock=clock,
    )

    with progress.phase("synthetic", step=7):
        clock.now = 0.25

    emitted = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert [item["event"] for item in emitted] == ["start", "end"]
    assert {item["schema"] for item in emitted} == {C2P2_PHASE_TELEMETRY_SCHEMA_VERSION}
    assert {item["phase"] for item in emitted} == {"synthetic"}
    assert emitted[1]["duration_seconds"] == 0.25
    assert progress.to_dict()["event_count"] == 2

    with pytest.raises(C2PhaseTimeout) as excinfo:
        enforce_phase_bound(
            phase="synthetic-over-bound",
            duration_seconds=2.0,
            timeout_seconds=1.0,
            bound_kind="phase",
        )
    assert excinfo.value.payload["schema"] == C2P2_PHASE_TELEMETRY_SCHEMA_VERSION
    assert excinfo.value.payload["phase"] == "synthetic-over-bound"
    assert excinfo.value.payload["event"] == "phase_timeout"
    assert excinfo.value.payload["bound_kind"] == "phase"


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
    step_result = receipt["step_reports"]["1"]["step_result"]
    assert step_result["tensor_state_summaries_included"] is False
    assert "tensor_state_summaries" not in step_result
    assert receipt["checkpoint_payload"]["checkpoint_written"] is False
    for tensor_summary in receipt["checkpoint_payload"]["tensor_summaries"].values():
        assert tensor_summary["bounded_decode_parity_checked"] is True
        assert tensor_summary["exact_shadow_matches_bounded_decode"] is True
    assert Path(receipt["receipt_path"]).exists()


def test_tiny_cpu_audit_receipt_proves_distinct_support_batches_and_step0_baseline(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=45), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=2,
        batch_size=45,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        audit_interval=1,
        max_steps_hard=5,
        emit_progress=True,
        phase_timeout_seconds=600.0,
        total_timeout_seconds=600.0,
        enabled=True,
    )

    step_hashes = [
        receipt["step_reports"]["1"]["support_batch"]["batch_content_hash16"],
        receipt["step_reports"]["2"]["support_batch"]["batch_content_hash16"],
    ]
    trajectory = receipt["acquisition_trajectory"]
    timing_summary = receipt["timing_summary"]
    phase_telemetry = receipt["phase_telemetry"]

    assert receipt["audit_interval"] == 1
    assert receipt["steps_completed"] == 2
    assert receipt["device_guard"]["pass"] is True
    assert phase_telemetry["schema"] == C2P2_PHASE_TELEMETRY_SCHEMA_VERSION
    assert phase_telemetry["enabled"] is True
    assert phase_telemetry["event_count"] > 0
    phase_names = {event["phase"] for event in phase_telemetry["events"]}
    assert {
        "load",
        "support_build",
        "state_init",
        "forward_fidelity",
        "audit0",
        "step",
        "step_update",
        "checkpoint_payload",
        "receipt_write",
    }.issubset(phase_names)
    assert receipt["support_cycler"]["covers_full_support"] is True
    assert receipt["support_cycler"]["has_at_least_two_distinct_batches"] is True
    assert len(set(step_hashes)) == 2
    assert trajectory["enabled"] is True
    assert trajectory["support_cycler_distinctness"]["trained_at_least_two_distinct_batches"] is True
    assert trajectory["support_cycler_distinctness"]["audited_at_least_two_distinct_batches"] is True
    assert trajectory["audit_steps"] == [0, 1, 2]
    assert trajectory["baseline_strict_exact_at_step0"]["strict_exact_total"] == 90
    assert trajectory["final_audit"]["strict_exact_total"] == 90
    assert trajectory["acquisition_verdict"] in {"acquired", "no_acquisition_verdict"}
    if trajectory["acquisition_verdict"] == "no_acquisition_verdict":
        assert trajectory["null_attribution_class"] is not None
    assert receipt["audit_reports"]["0"]["strict_exact_total"] == 90
    assert receipt["audit_reports"]["0"]["strict_exact_recompute_matches_metric"] is True
    assert receipt["audit_reports"]["2"]["audited_distinct_batch_count"] == 2
    assert timing_summary["schema"] == C2P2_TIMING_SCHEMA_VERSION
    assert timing_summary["step_report_count"] == 2
    assert timing_summary["step_timing_count"] == 2
    assert timing_summary["audit_report_count"] == 3
    assert timing_summary["audit_timing_count"] == 3
    assert set(timing_summary["step_duration_seconds_by_step"]) == {"1", "2"}
    assert set(timing_summary["audit_duration_seconds_by_step"]) == {"0", "1", "2"}
    assert timing_summary["audit_overhead_seconds_by_step"] == timing_summary["audit_duration_seconds_by_step"]
    assert len(timing_summary["step_duration_seconds"]) == 2
    assert len(timing_summary["audit_duration_seconds"]) == 3
    _assert_finite_non_negative(receipt["step_reports"]["1"]["duration_seconds"])
    _assert_finite_non_negative(receipt["audit_reports"]["0"]["duration_seconds"])
    _assert_finite_non_negative(timing_summary["total_run_duration_seconds"])
    _assert_finite_non_negative(timing_summary["median_step_duration_seconds"])
    _assert_finite_non_negative(timing_summary["total_step_duration_seconds"])
    _assert_finite_non_negative(timing_summary["median_audit_duration_seconds"])
    _assert_finite_non_negative(timing_summary["total_audit_duration_seconds"])
    for duration in timing_summary["step_duration_seconds"]:
        _assert_finite_non_negative(duration)
    for duration in timing_summary["audit_duration_seconds"]:
        _assert_finite_non_negative(duration)
    assert trajectory["timing_summary"] == timing_summary
    assert receipt["checkpoint_written"] is False
    assert receipt["parent_hash_after"] == parent_sha


def test_default_derivation_import_is_exercised_for_script_surface():
    module = BitLinear(2, 1, bias=False)
    state = derive_bounded_tensor_state_from_weight("proj", module.weight)

    assert state.state_key == "proj"
    assert state.q_levels.dtype == torch.int8
