"""C2.1 bounded-delta acquisition harness CPU/static tests."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import torch

import scripts.hrm_text_158_bounded_delta_acquisition_probe as probe_module
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
    candidate_weighted_grad_proxies_from_captures,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.optimizer_update_law_science import (
    ARM_A0_RANK_BUCKET_CURRENT,
    ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    ARM_B_RANK_FREE_SIGN_PRESSURE,
    BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
    BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
    BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
    BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
    BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE,
    BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
    BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
    BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
    FIXED_RANK_BUCKET_NON_TARGET_AUX,
    ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES,
    ORACLE_SCREEN_BRANCHES,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
    PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
    PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
    TIE_POLICY_CURRENT_MARGIN_INDEX,
    TIE_POLICY_DETERMINISTIC_HASH_MATCHED,
    WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
    oracle_screen_budget_max_seconds,
)
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT,
    ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE,
    ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
    ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR,
    _fraction_gte_observed,
    _fraction_lte_observed,
    _pivot_is_poor_rank_position,
    _pivot_poor_rank_position_threshold,
    _pivot_tie_band_is_ambiguous,
    _sampled_rank_fraction,
    _sampled_rank_position,
    _within_tie_band_family_metrics,
    run_activation_credit_measurement_oracle_screen,
    run_activation_credit_scale_smoke_oracle_screen,
    run_credit_ranking_pivot_measurement_oracle_screen,
    run_candidate_set_viability_oracle_screen,
    run_within_tie_band_discriminator_oracle_screen,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    B1_PRIOR_AUDIT_PINS,
    B1_PRIOR_AUDIT_SCHEMA_VERSION,
    B1_PRIOR_AUDIT_SUPPORTS,
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    C2P2_DEFAULT_GPU_SILENT_PHASE_TIMEOUT_SECONDS,
    C2P2_FAULTHANDLER_SCHEMA_VERSION,
    B2_FULL_VERDICT_SCHEMA_VERSION,
    B2_RETAINED_SUPPORT_SCHEMA_VERSION,
    C2P2_PHASE_TELEMETRY_SCHEMA_VERSION,
    C2P2_TIMING_SCHEMA_VERSION,
    C2PhaseTimeout,
    DEFAULT_PARENT_SHA256,
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    FORWARD_LEVEL_INIT_FIDELITY_STE_ATOL,
    GLOBAL_CAP_CONTRACT_OFF,
    HISTORICAL_IDENTITY_CONTROL,
    ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
    PhaseProgress,
    RUN_C2_ACQUISITION_PROBE_ENV,
    RUN_C2_GPU_LAUNCH_ENV,
    aggregate_identity_full_audit_batch_reports,
    b2_full_coverage_gate_met,
    b2_full_required_snapshot_names,
    build_arg_parser,
    build_b2_full_prior_snapshot,
    build_identity_full_batch,
    build_identity_full_support_batches,
    build_b2_retained_support_sets,
    build_model_from_checkpoint,
    build_prior_audit_support_batches,
    build_prior_audit_support_rows,
    compare_module_output_fidelity,
    compute_forward_level_init_fidelity,
    cuda_memory_receipt,
    cuda_memory_stats_device_arg,
    derive_bounded_tensor_state_from_weight,
    derive_tensor_states_and_check_init_fidelity,
    file_sha256,
    finalize_b2_full_verdict_state,
    guard_gpu_launch,
    identity_full_support_control_proof,
    native_ternary_effective_weight,
    new_b2_full_coverage_tracker,
    new_b2_full_verdict_state,
    parse_b2_retained_supports,
    parse_prior_audit_supports,
    record_b2_full_prior_snapshot,
    register_probe_faulthandler,
    reset_cuda_memory_stats,
    resolve_max_silent_phase_seconds,
    run_c2p1_probe,
    score_strict_exact_and_parsed_from_logits,
    select_eligible_bitlinears,
    snapshot_b2_full_coverage_tracker,
    update_strict_exact_stop_state,
    update_b2_full_coverage_tracker,
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


def _b2_prior_report(
    step: int,
    *,
    strict_count: int = 10,
    parsed_count: int = 10,
    total: int = 10,
    strict_failures: tuple[str, ...] = (),
    parsed_failures: tuple[str, ...] = (),
    source: str = "fixture_source",
) -> dict:
    strict_failure_ids = [str(row_id) for row_id in strict_failures]
    parsed_failure_ids = [str(row_id) for row_id in parsed_failures]
    return {
        "step": int(step),
        "strict_exact": f"{int(strict_count)}/{int(total)}",
        "strict_exact_count": int(strict_count),
        "strict_exact_total": int(total),
        "parsed_exact": f"{int(parsed_count)}/{int(total)}",
        "parsed_exact_count": int(parsed_count),
        "parsed_exact_total": int(total),
        "duration_seconds": 0.0,
        "strict_failure_row_ids": strict_failure_ids,
        "parsed_failure_row_ids": parsed_failure_ids,
        "strict_failure_sources_by_row_id": {
            row_id: source
            for row_id in strict_failure_ids
        },
        "parsed_failure_sources_by_row_id": {
            row_id: source
            for row_id in parsed_failure_ids
        },
    }


def _step_hash_subset(receipt: dict) -> dict:
    hash_keys = (
        "q_sha256_before",
        "q_sha256_after",
        "exact_accumulator_shadow_sha256_after",
        "votes_sha256",
    )
    return {
        step: {
            "q_changed_count": int(report["q_changed_count"]),
            "tensor_stats": {
                key: {
                    hash_key: stats[hash_key]
                    for hash_key in hash_keys
                }
                for key, stats in report["step_result"]["tensor_stats"].items()
            },
        }
        for step, report in receipt["step_reports"].items()
    }


def _target_audit_metric_subset(receipt: dict) -> dict:
    return {
        step: {
            "strict_exact": report["strict_exact"],
            "parsed_exact": report["parsed_exact"],
            "strict_exact_count": int(report["strict_exact_count"]),
            "parsed_exact_count": int(report["parsed_exact_count"]),
        }
        for step, report in receipt["audit_reports"].items()
    }


def _trajectory_without_timing(receipt: dict) -> dict:
    trajectory = json.loads(json.dumps(receipt["acquisition_trajectory"], sort_keys=True))
    trajectory.pop("timing_summary", None)
    return trajectory


def _state_parity_subset(receipt: dict) -> dict:
    checkpoint = receipt["checkpoint_payload"]
    return {
        "steps_completed": int(receipt["steps_completed"]),
        "stop_reason": receipt["stop_reason"],
        "step_report_keys": sorted(receipt["step_reports"]),
        "step_hashes": _step_hash_subset(receipt),
        "target_audit_reports": _target_audit_metric_subset(receipt),
        "acquisition_trajectory": _trajectory_without_timing(receipt),
        "authoritative_state_sha256": checkpoint["authoritative_state_sha256"],
        "tensor_summaries": checkpoint["tensor_summaries"],
    }


def test_science_arm_vote_builder_keeps_a0_default_and_adds_order_matched_b():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.tensor([[-1, 0, 0, 1]], dtype=torch.int8),
        0.5,
        torch.zeros((1, 4), dtype=torch.int16),
    )
    weighted_grads = {"toy.proj": torch.tensor([[-1.0, -2.0, 3.0, 4.0]])}
    rank_spec = probe_module.default_dry_run_rank_vote_spec()
    vote_spec = probe_module.default_vote_update_spec(max_abs_per_tensor=16)

    a0_votes, a0_pressure, a0_finite = probe_module._weighted_grads_to_science_arm_votes(
        weighted_grads,
        {"toy.proj": state},
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=ARM_A0_RANK_BUCKET_CURRENT,
    )
    a1_votes, a1_pressure, a1_finite = probe_module._weighted_grads_to_science_arm_votes(
        weighted_grads,
        {"toy.proj": state},
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=ARM_A1_RANK_BUCKET_ORDER_MATCHED,
    )
    b_votes, b_pressure, b_finite = probe_module._weighted_grads_to_science_arm_votes(
        weighted_grads,
        {"toy.proj": state},
        rank_spec=rank_spec,
        vote_spec=vote_spec,
        science_arm=ARM_B_RANK_FREE_SIGN_PRESSURE,
    )

    assert a0_finite is True
    assert a1_finite is True
    assert b_finite is True
    assert a0_votes["toy.proj"].tolist() == a1_votes["toy.proj"].tolist()
    assert a0_pressure["toy.proj"]["tie_policy_id"] == TIE_POLICY_CURRENT_MARGIN_INDEX
    assert a1_pressure["toy.proj"]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    assert b_pressure["toy.proj"]["tie_policy_id"] == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    assert b_votes["toy.proj"].tolist() == [[1, 1, -1, -1]]
    assert b_pressure["toy.proj"]["vote_abs_min"] == 1
    assert b_pressure["toy.proj"]["vote_abs_max"] == 1
    assert probe_module.FIXED_RANK_BUCKET_NON_TARGET_AUX == FIXED_RANK_BUCKET_NON_TARGET_AUX


def test_science_arm_a1_local_ordering_is_operational_when_global_cap_off():
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.zeros((1, 4), dtype=torch.int8),
        0.5,
        torch.zeros((1, 4), dtype=torch.int16),
    )
    weighted_grads = {"toy.proj": torch.tensor([[-1.0, -2.0, -3.0, -4.0]])}
    rank_spec = probe_module.default_dry_run_rank_vote_spec()
    vote_spec = probe_module.default_vote_update_spec(max_abs_per_tensor=1)
    states = {"toy.proj": state}
    specs = {"toy.proj": vote_spec}

    def run_arm(arm: str):
        votes, _pressure, _finite = probe_module._weighted_grads_to_science_arm_votes(
            weighted_grads,
            states,
            rank_spec=rank_spec,
            vote_spec=vote_spec,
            science_arm=arm,
        )
        return probe_module.apply_bounded_delta_vote_step(
            states,
            votes,
            specs,
            local_selection_ordering_mode=probe_module._science_local_selection_ordering_mode(arm),
            local_selection_ordering_seed=probe_module.SCIENCE_LOCAL_SELECTION_ORDERING_SEED,
            local_selection_ordering_step=1,
        )

    a0_result = run_arm(ARM_A0_RANK_BUCKET_CURRENT)
    a1_result = run_arm(ARM_A1_RANK_BUCKET_ORDER_MATCHED)
    b_result = run_arm(ARM_B_RANK_FREE_SIGN_PRESSURE)

    assert a0_result.global_summary["global_rate_cap_enabled"] is False
    assert a1_result.global_summary["global_rate_cap_enabled"] is False
    assert b_result.global_summary["global_rate_cap_enabled"] is False
    assert a0_result.tensor_states["toy.proj"].q_levels.tolist() == [[0, 1, 0, 0]]
    assert a1_result.tensor_states["toy.proj"].q_levels.tolist() == [[0, 0, 1, 0]]
    assert b_result.tensor_states["toy.proj"].q_levels.tolist() == [[0, 0, 1, 0]]
    assert (
        a1_result.tensor_stats["toy.proj"]["local_selection_ordering_mode"]
        == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    )
    assert (
        b_result.tensor_stats["toy.proj"]["local_selection_ordering_mode"]
        == TIE_POLICY_DETERMINISTIC_HASH_MATCHED
    )
    assert (
        a0_result.tensor_stats["toy.proj"]["local_selection_ordering_mode"]
        == TIE_POLICY_CURRENT_MARGIN_INDEX
    )


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


def test_prior_audit_flag_does_not_bypass_default_off(monkeypatch, tmp_path: Path):
    monkeypatch.delenv(RUN_C2_ACQUISITION_PROBE_ENV, raising=False)

    with pytest.raises(RuntimeError, match="default-off"):
        run_c2p1_probe(
            parent=tmp_path / "missing_parent.pt",
            parent_sha256=None,
            scratch_root=tmp_path / "scratch",
            prior_audit_supports="L0c1",
        )


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


def test_prior_audit_support_builders_pin_counts_hashes_and_l0c1_metadata():
    tok = BroadTokenizer()

    for support in B1_PRIOR_AUDIT_SUPPORTS:
        rows, proof = build_prior_audit_support_rows(support, curriculum_seed=17)
        pins = B1_PRIOR_AUDIT_PINS[support]

        assert len(rows) == pins["expected_count"]
        assert proof["row_count"] == pins["expected_count"]
        assert proof["support_hash16"] == pins["expected_hash16"]
        assert proof["builder_path"] == pins["builder_path"]
        assert proof["pinned_count_hash_pass"] is True
        assert proof["direct_kl"] is False
        assert proof["replay_pc"] == "OUT"
        assert proof["target_parent_kl"] is False

    l0c1 = build_prior_audit_support_batches(
        support="L0c1",
        tok=tok,
        max_len=TINY_ARCH["max_len"],
        batch_size=64,
        curriculum_seed=17,
        device=torch.device("cpu"),
    )

    assert l0c1["proof"]["expected_count"] == 121
    assert l0c1["proof"]["support_hash16"] == "7bc8cd771daab878"
    assert l0c1["proof"]["support_role"] == "close_wrapper_report_only"
    assert l0c1["proof"]["close_wrapper_report_only"] == {
        "direct_kl": False,
        "replay_pc": "OUT",
        "target_parent_kl": False,
    }
    assert l0c1["proof"]["batch_count"] == 2
    assert sum(batch["metadata"]["row_count"] for batch in l0c1["batches"]) == 121


def test_prior_audit_support_parser_rejects_unknowns_and_duplicates():
    assert parse_prior_audit_supports(None) == ()
    assert parse_prior_audit_supports("") == ()
    assert parse_prior_audit_supports("L0b, math_a0,L0c1") == (
        "L0b",
        "math_a0",
        "L0c1",
    )

    with pytest.raises(ValueError, match="unknown prior audit support"):
        parse_prior_audit_supports("L0b,not-a-support")
    with pytest.raises(ValueError, match="duplicate prior audit support"):
        parse_prior_audit_supports("L0b,L0b")


def test_b2_retained_support_parser_and_builders_keep_l0c1_report_only():
    tok = BroadTokenizer()

    assert parse_b2_retained_supports(None) == ()
    assert parse_b2_retained_supports("L0b, math_a0") == ("L0b", "math_a0")
    with pytest.raises(ValueError, match="L0c1 is report-only"):
        parse_b2_retained_supports("L0b,L0c1")
    with pytest.raises(ValueError, match="duplicate B2 retained support"):
        parse_b2_retained_supports("L0b,L0b")

    support_sets = build_b2_retained_support_sets(
        ("L0b", "math_a0"),
        tok=tok,
        max_len=TINY_ARCH["max_len"],
        support_batch_sizes={"L0b": 8, "math_a0": 16},
        curriculum_seed=17,
        device=torch.device("cpu"),
    )

    assert support_sets["L0b"]["proof"]["schema"] == B2_RETAINED_SUPPORT_SCHEMA_VERSION
    assert support_sets["L0b"]["proof"]["support_role"] == "retained_true_prior"
    assert support_sets["L0b"]["proof"]["report_only"] is False
    assert support_sets["L0b"]["proof"]["replay_ce_veto"] is True
    assert support_sets["L0b"]["proof"]["target_parent_kl"] is False
    assert support_sets["math_a0"]["proof"]["expected_count"] == 1255
    assert support_sets["math_a0"]["proof"]["batch_size"] == 16


def test_b2_full_coverage_tracker_counts_disjoint_cycles():
    tracker = new_b2_full_coverage_tracker({"math_a0": 4})

    update_b2_full_coverage_tracker(
        tracker,
        support="math_a0",
        row_ids=("0", "1", "2", "3"),
    )
    one_pass = snapshot_b2_full_coverage_tracker(tracker)["math_a0"]
    assert one_pass["coverage_cycles"] == 1
    assert one_pass["coverage_gate_met"] is True
    assert one_pass["rows_seen_current_cycle"] == 0
    assert one_pass["rows_seen_total"] == 4

    update_b2_full_coverage_tracker(
        tracker,
        support="math_a0",
        row_ids=("0", "1"),
    )
    one_and_half_passes = snapshot_b2_full_coverage_tracker(tracker)["math_a0"]
    assert one_and_half_passes["coverage_cycles"] == 1
    assert one_and_half_passes["rows_seen_current_cycle"] == 2
    assert one_and_half_passes["rows_seen_total"] == 6

    update_b2_full_coverage_tracker(
        tracker,
        support="math_a0",
        row_ids=("2", "3"),
    )
    two_passes = snapshot_b2_full_coverage_tracker(tracker)["math_a0"]
    assert two_passes["coverage_cycles"] == 2
    assert two_passes["rows_seen_current_cycle"] == 0
    assert b2_full_coverage_gate_met({"math_a0": two_passes}) is True


def test_b2_full_precoverage_snapshot_does_not_stop_and_l0c1_is_report_only():
    state = new_b2_full_verdict_state()
    start_reports = {
        "L0b": _b2_prior_report(0),
        "math_a0": _b2_prior_report(0),
        "L0c1": _b2_prior_report(0),
    }
    current_reports = {
        "L0b": _b2_prior_report(40),
        "math_a0": _b2_prior_report(40),
        "L0c1": _b2_prior_report(
            40,
            strict_count=7,
            parsed_count=7,
            strict_failures=("l0c1-a", "l0c1-b", "l0c1-c"),
            parsed_failures=("l0c1-a", "l0c1-b", "l0c1-c"),
            source="close_wrapper",
        ),
    }
    target_audit = {
        "step": 40,
        "strict_exact": "90/90",
        "strict_exact_count": 90,
        "parsed_exact": "90/90",
        "parsed_exact_count": 90,
        "acquired": True,
    }
    coverage = {
        "L0b": {"coverage_cycles": 0, "rows_total": 4},
        "math_a0": {"coverage_cycles": 0, "rows_total": 4},
    }

    assert b2_full_required_snapshot_names(
        state,
        target_audit=target_audit,
        coverage_by_support=coverage,
    ) == ["first_audited_target_ge_90"]

    snapshot = build_b2_full_prior_snapshot(
        snapshot_name="first_audited_target_ge_90",
        step=40,
        target_audit=target_audit,
        coverage_by_support=coverage,
        start_reports=start_reports,
        current_reports=current_reports,
    )

    assert snapshot["schema"] == B2_FULL_VERDICT_SCHEMA_VERSION
    assert snapshot["target_gate_met"] is True
    assert snapshot["coverage_gate_met"] is False
    assert snapshot["combined_stop_pass"] is False
    assert snapshot["retained_true_priors_no_new_broad_cluster"] is True
    assert snapshot["stop_support_status"] == {"L0b": True, "math_a0": True}
    assert snapshot["deltas"]["L0c1"]["no_new_broad_cluster"] is False
    assert "L0c1" not in snapshot["stop_support_status"]


def test_b2_full_snapshot_state_dedupes_same_step_combined_stop_and_terminal():
    state = new_b2_full_verdict_state()
    start_reports = {
        "L0b": _b2_prior_report(0),
        "math_a0": _b2_prior_report(0),
        "L0c1": _b2_prior_report(0),
    }
    current_reports = {
        "L0b": _b2_prior_report(80),
        "math_a0": _b2_prior_report(80),
        "L0c1": _b2_prior_report(80),
    }
    target_audit = {
        "step": 80,
        "strict_exact": "90/90",
        "strict_exact_count": 90,
        "parsed_exact": "90/90",
        "parsed_exact_count": 90,
        "acquired": True,
    }
    coverage = {
        "L0b": {"coverage_cycles": 1, "rows_total": 4},
        "math_a0": {"coverage_cycles": 1, "rows_total": 4},
    }
    names = b2_full_required_snapshot_names(
        state,
        target_audit=target_audit,
        coverage_by_support=coverage,
    )
    assert names == ["first_audited_target_ge_90", "first_covered_target_ge_90"]

    snapshot = build_b2_full_prior_snapshot(
        snapshot_name="runtime_prior_snapshot",
        step=80,
        target_audit=target_audit,
        coverage_by_support=coverage,
        start_reports=start_reports,
        current_reports=current_reports,
    )
    record_b2_full_prior_snapshot(state, snapshot_names=names, snapshot=snapshot)

    assert state["prior_audit_count"] == 1
    assert state["snapshot_steps"]["first_audited_target_ge_90"] == 80
    assert state["snapshot_steps"]["first_covered_target_ge_90"] == 80
    assert state["combined_stop"]["triggered"] is True
    assert state["combined_stop"]["step"] == 80
    assert state["first_audited_target_ge_90"]["snapshot_name"] == "first_audited_target_ge_90"
    assert state["first_covered_target_ge_90"]["snapshot_name"] == "first_covered_target_ge_90"

    terminal_snapshot = build_b2_full_prior_snapshot(
        snapshot_name="terminal",
        step=160,
        target_audit=target_audit,
        coverage_by_support={
            "L0b": {"coverage_cycles": 2, "rows_total": 4},
            "math_a0": {"coverage_cycles": 2, "rows_total": 4},
        },
        start_reports=start_reports,
        current_reports={
            "L0b": _b2_prior_report(160),
            "math_a0": _b2_prior_report(160),
            "L0c1": _b2_prior_report(160),
        },
    )
    receipt = finalize_b2_full_verdict_state(state, terminal_snapshot=terminal_snapshot)

    assert receipt["prior_audit_count"] == 2
    assert receipt["snapshot_steps"]["terminal"] == 160
    assert receipt["math_a0_coverage_cycles"] == 2
    assert receipt["l0b_coverage_cycles"] == 2
    assert receipt["terminal"]["combined_stop_pass"] is True
    assert receipt["verdict"] == "RETAINS"


def test_b2_full_cli_flag_defaults_off_and_support_validation_is_preload(tmp_path):
    args = build_arg_parser().parse_args([])
    assert args.b2_full_verdict_mode is False
    assert args.oracle_screen_mode is None
    assert args.oracle_screen_max_sampled_candidates == 8
    assert args.global_cap_contract == GLOBAL_CAP_CONTRACT_OFF
    assert args.tie_rule_mode == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    assert args.matched_continued_training_horizon_steps == 0
    assert args.max_silent_phase_seconds is None
    assert resolve_max_silent_phase_seconds(
        allow_gpu_launch=args.allow_gpu_launch,
        max_silent_phase_seconds=args.max_silent_phase_seconds,
    ) is None
    assert build_arg_parser().parse_args(["--b2-full-verdict-mode"]).b2_full_verdict_mode is True
    gpu_args = build_arg_parser().parse_args(["--allow-gpu-launch"])
    assert resolve_max_silent_phase_seconds(
        allow_gpu_launch=gpu_args.allow_gpu_launch,
        max_silent_phase_seconds=gpu_args.max_silent_phase_seconds,
    ) == C2P2_DEFAULT_GPU_SILENT_PHASE_TIMEOUT_SECONDS
    override_args = build_arg_parser().parse_args(
        ["--allow-gpu-launch", "--max-silent-phase-seconds", "0"]
    )
    assert resolve_max_silent_phase_seconds(
        allow_gpu_launch=override_args.allow_gpu_launch,
        max_silent_phase_seconds=override_args.max_silent_phase_seconds,
    ) is None
    oracle_args = build_arg_parser().parse_args(
        ["--oracle-screen-mode", ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY]
    )
    assert oracle_args.oracle_screen_mode == ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY
    oracle_budget_args = build_arg_parser().parse_args(
        [
            "--oracle-screen-mode",
            ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
            "--oracle-screen-max-sampled-candidates",
            "32",
        ]
    )
    assert oracle_budget_args.oracle_screen_max_sampled_candidates == 32
    pivot_args = build_arg_parser().parse_args(
        [
            "--oracle-screen-mode",
            ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
            "--oracle-screen-max-sampled-candidates",
            "32",
        ]
    )
    assert pivot_args.oracle_screen_mode == ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT
    assert pivot_args.oracle_screen_max_sampled_candidates == 32

    with pytest.raises(ValueError, match="requires retained supports"):
        run_c2p1_probe(
            parent=tmp_path / "missing.pt",
            scratch_root=tmp_path,
            enabled=True,
            b2_full_verdict_mode=True,
            audit_interval=20,
            prior_audit_supports="L0b,math_a0,L0c1",
            b2_retained_supports="L0b",
        )

    with pytest.raises(ValueError, match="requires audit_interval"):
        run_c2p1_probe(
            parent=tmp_path / "missing.pt",
            scratch_root=tmp_path,
            enabled=True,
            b2_full_verdict_mode=True,
            audit_interval=0,
            prior_audit_supports="L0b,math_a0,L0c1",
            b2_retained_supports="L0b,math_a0",
        )


def test_strict_exact_stop_state_honors_matched_continued_training_horizon():
    first_step, token = update_strict_exact_stop_state(
        step=10,
        audit_report={"acquired": True},
        stop_on_strict_exact=True,
        matched_continued_training_horizon_steps=50,
        first_strict_exact_step=None,
    )
    assert first_step == 10
    assert token is None

    first_step, token = update_strict_exact_stop_state(
        step=59,
        audit_report={"acquired": True},
        stop_on_strict_exact=True,
        matched_continued_training_horizon_steps=50,
        first_strict_exact_step=first_step,
    )
    assert first_step == 10
    assert token is None

    first_step, token = update_strict_exact_stop_state(
        step=60,
        audit_report={"acquired": True},
        stop_on_strict_exact=True,
        matched_continued_training_horizon_steps=50,
        first_strict_exact_step=first_step,
    )
    assert first_step == 10
    assert token == "strict_exact_acquired_matched_horizon"

    first_step, token = update_strict_exact_stop_state(
        step=4,
        audit_report={"acquired": True},
        stop_on_strict_exact=True,
        matched_continued_training_horizon_steps=0,
        first_strict_exact_step=None,
    )
    assert first_step == 4
    assert token == "strict_exact_acquired"


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


def test_phase_progress_silent_phase_guard_breaches_before_phase_exit(tmp_path: Path):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    last_active_path = tmp_path / "last_active_phase.json"
    progress = PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=5.0,
        last_active_phase_path=last_active_path,
        arm_faulthandler_timer=False,
        clock=clock,
    )

    with progress.phase("step_update", step=1):
        clock.now = 5.5
        with pytest.raises(C2PhaseTimeout) as excinfo:
            progress.check_stale_active_phase()

    payload = excinfo.value.payload
    assert payload["phase"] == "step_update"
    assert payload["bound_kind"] == "silent_phase"
    assert payload["failure_class"] == "LIVENESS_FAILURE"
    assert payload["step"] == 1
    last_active = json.loads(last_active_path.read_text(encoding="utf-8"))
    assert last_active["phase"] == "step_update"
    assert last_active["budget_seconds"] == 5.0
    assert last_active["failure_class"] == "LIVENESS_FAILURE"


def test_phase_progress_silent_phase_guard_allows_normal_progress(tmp_path: Path):
    class FakeClock:
        def __init__(self) -> None:
            self.now = 0.0

        def __call__(self) -> float:
            return self.now

    clock = FakeClock()
    progress = PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=5.0,
        last_active_phase_path=tmp_path / "last_active_phase.json",
        arm_faulthandler_timer=False,
        clock=clock,
    )

    with progress.phase("step_update", step=1):
        clock.now = 4.0
        assert progress.check_stale_active_phase() is None


def test_phase_progress_arms_faulthandler_timer_without_firing(monkeypatch, tmp_path: Path):
    calls = []

    def fake_cancel() -> None:
        calls.append(("cancel",))

    def fake_dump(timeout: float, *, repeat: bool, exit: bool) -> None:
        calls.append(("dump", timeout, repeat, exit))

    monkeypatch.setattr(probe_module.faulthandler, "cancel_dump_traceback_later", fake_cancel)
    monkeypatch.setattr(probe_module.faulthandler, "dump_traceback_later", fake_dump)
    progress = PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=7.0,
        last_active_phase_path=tmp_path / "last_active_phase.json",
    )

    with progress.phase("step_update", step=1):
        assert ("dump", 7.0, False, True) in calls

    assert calls[-1] == ("cancel",)


def test_phase_progress_fails_closed_when_faulthandler_timer_cannot_arm(monkeypatch, tmp_path: Path):
    body_entered = {"value": False}
    calls = []

    def fake_cancel() -> None:
        calls.append(("cancel",))
        return None

    def fake_dump(timeout: float, *, repeat: bool, exit: bool) -> None:
        calls.append(("dump", timeout, repeat, exit))
        raise RuntimeError("timer unavailable")

    monkeypatch.setattr(probe_module.faulthandler, "cancel_dump_traceback_later", fake_cancel)
    monkeypatch.setattr(probe_module.faulthandler, "dump_traceback_later", fake_dump)
    progress = PhaseProgress(
        enabled=False,
        device=torch.device("cpu"),
        silent_phase_timeout_seconds=7.0,
        last_active_phase_path=tmp_path / "last_active_phase.json",
    )

    with pytest.raises(RuntimeError, match="failed to arm silent phase faulthandler guard"):
        with progress.phase("cuda_memory_reset"):
            body_entered["value"] = True

    assert body_entered["value"] is False
    assert progress._phase_stack == []
    assert ("cancel",) in calls
    assert ("dump", 7.0, False, True) in calls


def test_register_probe_faulthandler_enable_failure_fails_closed():
    def fake_enable(*, all_threads: bool) -> None:
        raise RuntimeError("enable unavailable")

    with pytest.raises(RuntimeError, match="failed to enable faulthandler for probe"):
        register_probe_faulthandler(
            enable_fn=fake_enable,
            is_enabled_fn=lambda: False,
        )


def test_tiny_step0_receipt_write_uses_guarded_phase(monkeypatch, tmp_path: Path):
    calls = []
    timer_armed = {"value": False}
    receipt_write_armed = []
    original_write_text = Path.write_text

    def fake_cancel() -> None:
        calls.append(("cancel",))
        timer_armed["value"] = False

    def fake_dump(timeout: float, *, repeat: bool, exit: bool) -> None:
        calls.append(("dump", timeout, repeat, exit))
        timer_armed["value"] = True

    def guarded_write_text(self, data, *args, **kwargs):
        if self.name == "receipt.json":
            receipt_write_armed.append(bool(timer_armed["value"]))
        return original_write_text(self, data, *args, **kwargs)

    monkeypatch.setattr(probe_module.faulthandler, "cancel_dump_traceback_later", fake_cancel)
    monkeypatch.setattr(probe_module.faulthandler, "dump_traceback_later", fake_dump)
    monkeypatch.setattr(Path, "write_text", guarded_write_text)
    parent = tmp_path / "tiny_parent.pt"
    scratch_root = tmp_path / "scratch"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=scratch_root,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=0,
        batch_size=2,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        max_silent_phase_seconds=11.0,
        enabled=True,
    )

    events = [
        event
        for event in receipt["phase_telemetry"]["events"]
        if event["phase"] == "receipt_write"
    ]
    assert [event["event"] for event in events] == ["start", "end"]
    last_active = json.loads((scratch_root / "last_active_phase.json").read_text(encoding="utf-8"))
    assert last_active["phase"] == "receipt_write"
    assert last_active["budget_seconds"] == 11.0
    assert ("dump", 11.0, False, True) in calls
    assert receipt_write_armed == [True]
    disk_receipt = json.loads((scratch_root / "receipt.json").read_text(encoding="utf-8"))
    assert disk_receipt["receipt_path"] == str(scratch_root / "receipt.json")


def test_register_probe_faulthandler_reports_signal_paths_without_signalling():
    calls = []
    enabled = {"value": False}

    def fake_is_enabled() -> bool:
        return bool(enabled["value"])

    def fake_enable(*, all_threads: bool) -> None:
        calls.append(("enable", all_threads))
        enabled["value"] = True

    def fake_register(sig, *, all_threads: bool, chain: bool) -> None:
        calls.append(("register", int(sig), all_threads, chain))

    report = register_probe_faulthandler(
        enable_fn=fake_enable,
        register_fn=fake_register,
        is_enabled_fn=fake_is_enabled,
    )

    assert report["schema"] == C2P2_FAULTHANDLER_SCHEMA_VERSION
    assert report["enabled_before"] is False
    assert report["enabled_after"] is True
    assert ("enable", True) in calls
    assert report["signals"]["SIGABRT"]["status"] == "handled_by_faulthandler_enable"
    if getattr(probe_module.signal, "SIGQUIT", None) is not None:
        assert report["signals"]["SIGQUIT"]["status"] == "registered"
        assert any(call[0] == "register" for call in calls)


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
    for tensor_stats in step_result["tensor_stats"].values():
        assert tensor_stats["bounded_accumulator_fresh_for_exact_shadow"] is False
        assert tensor_stats["bounded_accumulator_rebuilt_for_parity"] is False
        assert tensor_stats["bounded_decode_parity_checked"] is False
    assert receipt["checkpoint_payload"]["checkpoint_written"] is False
    for tensor_summary in receipt["checkpoint_payload"]["tensor_summaries"].values():
        assert tensor_summary["bounded_accumulator_fresh_for_exact_shadow"] is True
        assert tensor_summary["bounded_accumulator_rebuilt_for_parity"] is True
        assert tensor_summary["bounded_decode_parity_checked"] is True
        assert tensor_summary["exact_shadow_matches_bounded_decode"] is True
    assert Path(receipt["receipt_path"]).exists()


def test_direct_oracle_screen_runner_keeps_parent_state_unmutated_and_emits_compact_loss_delta():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    q_before = {
        key: tensor_state.q_levels.clone()
        for key, tensor_state in states.items()
    }
    acc_before = {
        key: tensor_state.exact_accumulator_shadow.clone()
        for key, tensor_state in states.items()
    }

    receipt = run_candidate_set_viability_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
    )

    assert receipt["candidate_count"] >= receipt["sampled_candidate_count"] > 0
    assert receipt["branch_classification"] in ORACLE_SCREEN_BRANCHES
    assert receipt["compact_summary"]["local_loss_delta_deciles"]["best_local_loss_delta"] is not None
    wider_inputs = receipt["wider_screen_interpretation_inputs"]
    assert wider_inputs["max_sampled_candidates"] in ORACLE_SCREEN_ALLOWED_MAX_SAMPLED_CANDIDATES
    assert wider_inputs["oracle_best_current_rank_position"] is not None
    assert wider_inputs["oracle_best_current_sampled_rank_position"] is not None
    assert wider_inputs["oracle_best_current_rank_fraction"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_rank_position"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_sampled_rank_position"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_rank_fraction"] is not None
    assert wider_inputs["current_vs_oracle_top1_gap"] is not None
    assert (
        wider_inputs["current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta"]
        is not None
    )
    assert wider_inputs["ce_improving_candidate_fraction"] is not None
    assert wider_inputs["oracle_best_current_rank_fraction"] <= 1.0
    assert wider_inputs["oracle_best_deterministic_hash_rank_fraction"] <= 1.0
    assert receipt["compact_summary"]["wider_screen_interpretation_inputs"] == wider_inputs
    assert receipt["non_persistence"]["q_persisted"] is False
    assert receipt["non_persistence"]["checkpoint_written"] is False
    for key, tensor_state in states.items():
        assert torch.equal(tensor_state.q_levels, q_before[key])
        assert torch.equal(tensor_state.exact_accumulator_shadow, acc_before[key])


def test_direct_oracle_screen_runner_budget_overrun_classifies_infeasible():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)

    receipt = run_candidate_set_viability_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_seconds=-1.0,
    )

    assert receipt["candidate_count"] > 0
    assert receipt["sampled_candidate_count"] == 0
    assert receipt["oracle_feasible"] is False
    assert receipt["budget_exceeded"] is True
    assert receipt["branch_classification"] == BRANCH_ORACLE_INFEASIBLE_OR_TOO_EXPENSIVE


def test_oracle_sampled_rank_fraction_uses_sampled_order_not_raw_rank():
    sampled_candidates = [
        {
            "candidate_id": "oracle-best",
            "current_rank_position": 300_000,
            "deterministic_hash_rank_position": 500_000,
            "local_loss_delta": -0.50,
        },
        {
            "candidate_id": "runner-up",
            "current_rank_position": 400_000,
            "deterministic_hash_rank_position": 600_000,
            "local_loss_delta": -0.25,
        },
    ]
    current_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            int(candidate["current_rank_position"]),
            str(candidate["candidate_id"]),
        ),
    )
    deterministic_top = sorted(
        sampled_candidates,
        key=lambda candidate: (
            int(candidate["deterministic_hash_rank_position"]),
            str(candidate["candidate_id"]),
        ),
    )

    current_position = _sampled_rank_position(current_top, "oracle-best")
    deterministic_position = _sampled_rank_position(deterministic_top, "oracle-best")
    current_fraction = _sampled_rank_fraction(current_position, len(sampled_candidates))
    deterministic_fraction = _sampled_rank_fraction(
        deterministic_position,
        len(sampled_candidates),
    )

    assert current_position == 0
    assert deterministic_position == 0
    assert current_fraction == 0.5
    assert deterministic_fraction == 0.5
    assert current_fraction <= 1.0
    assert deterministic_fraction <= 1.0


def test_credit_ranking_pivot_poor_rank_threshold_uses_zero_based_sampled_position():
    assert _pivot_poor_rank_position_threshold(32) == 8
    assert _sampled_rank_fraction(7, 32) == 0.25
    assert _pivot_is_poor_rank_position(7, 32) is False
    assert _pivot_is_poor_rank_position(8, 32) is True


def test_credit_ranking_pivot_tie_band_ambiguity_requires_high_regret_spread():
    assert _pivot_tie_band_is_ambiguous(
        oracle_best_candidate_present=True,
        band_candidate_count=2,
        regret_spread_ratio=0.20,
    ) is False
    assert _pivot_tie_band_is_ambiguous(
        oracle_best_candidate_present=True,
        band_candidate_count=2,
        regret_spread_ratio=0.25,
    ) is False
    assert _pivot_tie_band_is_ambiguous(
        oracle_best_candidate_present=True,
        band_candidate_count=2,
        regret_spread_ratio=0.30,
    ) is True


def test_tiny_oracle_screen_mode_returns_compact_non_persistent_receipt(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=8), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_oracle_screen",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=8,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        oracle_screen_mode=ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY,
        oracle_screen_max_sampled_candidates=32,
    )

    assert receipt["gpu_launched"] is False
    assert receipt["checkpoint_written"] is False
    assert receipt["parent_hash_before"] == parent_sha
    assert receipt["parent_hash_after"] == parent_sha
    assert receipt["parent_hash_unchanged"] is True
    assert receipt["forward_backward_update_executed"] is False
    assert receipt["science_arm"] is None
    assert receipt["oracle_screen_mode"] == ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY
    assert receipt["stop_reason"] == "oracle_screen_completed"
    assert receipt["step_reports"] == {}
    assert receipt["audit_reports"] == {}
    oracle = receipt["oracle_screen"]
    assert oracle["mode"] == ORACLE_SCREEN_MODE_CANDIDATE_SET_VIABILITY
    assert oracle["same_candidate_set_required"] is True
    assert oracle["screen_rows"] == 8
    assert oracle["candidate_count"] >= oracle["sampled_candidate_count"]
    assert oracle["sampled_candidate_count"] <= oracle["max_sampled_candidates"]
    assert oracle["max_sampled_candidates"] == 32
    assert oracle["max_seconds"] == oracle_screen_budget_max_seconds(32)
    assert oracle["branch_classification"] in ORACLE_SCREEN_BRANCHES
    assert oracle["non_persistence"]["q_persisted"] is False
    assert oracle["non_persistence"]["checkpoint_written"] is False
    assert oracle["non_persistence"]["pt_writes_allowed"] is False
    assert oracle["non_persistence"]["screen_state_mutated"] is False
    assert oracle["compact_summary"]["candidate_count"] == oracle["candidate_count"]
    assert oracle["compact_summary"]["sampled_candidate_count"] == oracle["sampled_candidate_count"]
    wider_inputs = oracle["wider_screen_interpretation_inputs"]
    assert wider_inputs["max_sampled_candidates"] == 32
    assert wider_inputs["oracle_best_current_rank_position"] is not None
    assert wider_inputs["oracle_best_current_sampled_rank_position"] is not None
    assert wider_inputs["oracle_best_current_rank_fraction"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_rank_position"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_sampled_rank_position"] is not None
    assert wider_inputs["oracle_best_deterministic_hash_rank_fraction"] is not None
    assert wider_inputs["current_vs_oracle_top1_gap"] is not None
    assert (
        wider_inputs["current_vs_oracle_top1_gap_denominator_abs_oracle_top1_local_loss_delta"]
        is not None
    )
    assert wider_inputs["ce_improving_candidate_fraction"] is not None
    assert wider_inputs["oracle_best_current_rank_fraction"] <= 1.0
    assert wider_inputs["oracle_best_deterministic_hash_rank_fraction"] <= 1.0
    assert oracle["compact_summary"]["wider_screen_interpretation_inputs"] == wider_inputs
    assert Path(receipt["receipt_path"]).exists()


def test_direct_credit_ranking_pivot_runner_emits_compact_stage_a_and_stage_b_smoke():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    receipt = run_credit_ranking_pivot_measurement_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_sampled_candidates=32,
        max_seconds=oracle_screen_budget_max_seconds(32),
    )

    assert receipt["mode"] == ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT
    assert receipt["same_candidate_set_required"] is True
    assert receipt["candidate_count"] >= receipt["sampled_candidate_count"] > 0
    assert receipt["max_sampled_candidates"] == 32
    assert receipt["oracle_feasible"] is True
    assert receipt["branch_classification"] in {
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
        PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    }
    compact = receipt["compact_summary"]
    assert set(compact) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "score_family_metrics",
        "stage_a_null_guard",
        "tie_band_ambiguity",
        "local_apply_magnitude_smoke",
        "telemetry",
    }
    assert compact["candidate_count"] == receipt["candidate_count"]
    assert compact["sampled_candidate_count"] == receipt["sampled_candidate_count"]
    assert compact["sampled_candidate_table"]
    first_row = compact["sampled_candidate_table"][0]
    assert "deterministic_hash_rank_position" not in first_row
    assert set(first_row["score_rank_positions"]) == {
        "S_vote_margin",
        "S_vote_only",
        "S_margin_only",
        "S_current",
    }
    metrics_by_score = compact["score_family_metrics"]["metrics_by_score_id"]
    assert metrics_by_score["S_vote_margin"]["score_id"] == "S_vote_margin"
    assert metrics_by_score["S_vote_margin"][
        "oracle_best_sampled_rank_position_poor_threshold"
    ] == _pivot_poor_rank_position_threshold(receipt["sampled_candidate_count"])
    assert compact["stage_a_null_guard"]["score_id"] == "S_vote_margin"
    assert isinstance(compact["stage_a_null_guard"]["passes"], bool)
    assert compact["telemetry"]["deterministic_hash_control_only"] is True
    assert set(compact["telemetry"]["binary_top_k_ce_improving_capture"]) == {
        "S_vote_margin",
        "S_vote_only",
        "S_margin_only",
        "S_current",
    }
    smoke = compact["local_apply_magnitude_smoke"]
    assert compact["tie_band_ambiguity"]["ambiguous_if_regret_spread_ratio_gt"] == 0.25
    assert smoke["current_spec_is_non_definitive_without_live_full_cap"] is True
    assert smoke["definitive_b_requires_follow_on"] is True
    assert {
        variant["variant_id"]
        for variant in smoke["variants"]
    } == {
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_TOP1,
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_PREFIX_CAP_1024,
        PIVOT_MEASUREMENT_LOCAL_APPLY_VARIANT_CURRENT_SPEC,
    }
    assert receipt["non_persistence"]["q_persisted"] is False
    assert receipt["non_persistence"]["checkpoint_written"] is False


def test_direct_credit_ranking_pivot_runner_budget_overrun_fails_closed_ambiguous():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    receipt = run_credit_ranking_pivot_measurement_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_sampled_candidates=32,
        max_seconds=-1.0,
    )

    assert receipt["budget_exceeded"] is True
    assert receipt["oracle_feasible"] is False
    assert receipt["branch_classification"] == BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH


def test_tiny_credit_ranking_pivot_mode_returns_compact_non_persistent_receipt(
    tmp_path: Path,
):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=8), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_credit_ranking_pivot",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=8,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        oracle_screen_mode=ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT,
        oracle_screen_max_sampled_candidates=32,
    )

    assert receipt["science_arm"] is None
    assert receipt["oracle_screen_mode"] == ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT
    assert receipt["stop_reason"] == "oracle_screen_completed"
    oracle = receipt["oracle_screen"]
    assert oracle["mode"] == ORACLE_SCREEN_MODE_CREDIT_RANKING_PIVOT_MEASUREMENT
    assert oracle["same_candidate_set_required"] is True
    assert oracle["candidate_count"] >= oracle["sampled_candidate_count"] > 0
    assert oracle["max_sampled_candidates"] == 32
    assert oracle["branch_classification"] in {
        BRANCH_PREREGISTERED_CHEAP_LEARNER_FEATURE_FAMILY_CANNOT_PREDICT_REGRET,
        PIVOT_MEASUREMENT_PREDICTIVE_SEED_LABEL,
        BRANCH_MEASUREMENT_AMBIGUOUS_TIE_BAND_ALIASING,
        BRANCH_MEASUREMENT_AMBIGUOUS_NO_BRANCH,
    }
    assert (
        oracle["compact_summary"]["local_apply_magnitude_smoke"][
            "current_spec_is_non_definitive_without_live_full_cap"
        ]
        is True
    )
    assert oracle["non_persistence"]["q_persisted"] is False
    assert oracle["non_persistence"]["checkpoint_written"] is False
    assert Path(receipt["receipt_path"]).exists()


def test_candidate_weighted_grad_proxies_preserve_invocation_pairing_order():
    inputs = [
        torch.tensor([[[1.0, 10.0]]], dtype=torch.float32),
        torch.tensor([[[100.0, 1000.0]]], dtype=torch.float32),
    ]
    grad_outputs = [
        torch.tensor([[[2.0, 20.0]]], dtype=torch.float32),
        torch.tensor([[[3.0, 30.0]]], dtype=torch.float32),
    ]

    proxies = candidate_weighted_grad_proxies_from_captures(
        inputs,
        grad_outputs,
        flat_indices=[0, 3],
        weight_shape=(2, 2),
    )

    assert proxies.tolist() == pytest.approx(
        [
            (1.0 * 3.0) + (100.0 * 2.0),
            (10.0 * 30.0) + (1000.0 * 20.0),
        ]
    )


def test_direct_activation_credit_scale_smoke_runner_emits_device_resident_telemetry():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    receipt = run_activation_credit_scale_smoke_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_sampled_candidates=8,
        max_seconds=oracle_screen_budget_max_seconds(8),
    )

    assert receipt["mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE
    assert receipt["same_candidate_set_required"] is True
    assert receipt["branch_classification"] is None
    assert receipt["sampled_candidate_count"] > 0
    assert receipt["capture_device_mode"] == "device_resident"
    assert receipt["gather_device"] == "cpu"
    _assert_finite_non_negative(receipt["background_candidate_generation_seconds"])
    _assert_finite_non_negative(receipt["observer_forward_backward_seconds"])
    _assert_finite_non_negative(receipt["grad_proxy_accumulation_seconds"])
    _assert_finite_non_negative(receipt["binning_emission_seconds"])
    compact = receipt["compact_summary"]
    assert set(compact) == {
        "target_tie_band_id",
        "target_band_candidate_count",
        "grad_proxy_candidate_count",
        "magnitude_bin_threshold",
        "magnitude_bin_histogram",
        "magnitude_bin_degenerate",
        "singleton_magnitude_source_count",
        "sampled_target_band_rows",
    }
    assert compact["target_tie_band_id"] == "voteabs=4|marginabs=4"
    assert receipt["non_persistence"]["q_persisted"] is False
    assert receipt["non_persistence"]["checkpoint_written"] is False


def test_direct_activation_credit_measurement_runner_emits_compact_family_metrics():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    receipt = run_activation_credit_measurement_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_sampled_candidates=32,
        max_seconds=oracle_screen_budget_max_seconds(32),
    )

    assert receipt["mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT
    assert receipt["same_candidate_set_required"] is True
    assert receipt["candidate_count"] >= receipt["sampled_candidate_count"] > 0
    assert receipt["max_sampled_candidates"] == 32
    assert receipt["branch_classification"] in {
        BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
        BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
        BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
    }
    compact = receipt["compact_summary"]
    assert set(compact) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    }
    assert compact["target_tie_band"]["target_tie_band_id"] == "voteabs=4|marginabs=4"
    assert compact["target_tie_band"][
        "fresh_confirmation_seed_required_for_persistent_followup"
    ] == 71
    assert compact["family_metrics"]["primary_family_id"] == "F_align_magbin"
    assert compact["family_metrics"]["topology_control_family_id"] == (
        "F_topology_lane_head_row_block128"
    )
    first_row = compact["sampled_candidate_table"][0]
    assert {
        "candidate_delta_sign",
        "credit_sign",
        "credit_magnitude_bin",
        "signed_alignment",
        "topology_row_block_128",
        "activation_feature_valid",
    }.issubset(first_row)
    target_band_rows = [
        row for row in compact["sampled_candidate_table"] if bool(row["in_target_tie_band"])
    ]
    assert target_band_rows
    assert any(int(row["candidate_delta_sign"]) != 0 for row in target_band_rows)
    assert any(bool(row["activation_feature_valid"]) for row in target_band_rows)
    assert compact["telemetry"]["capture_device_mode"] == "device_resident"
    assert compact["telemetry"]["gather_device"] == "cpu"
    assert receipt["non_persistence"]["q_persisted"] is False
    assert receipt["non_persistence"]["checkpoint_written"] is False


def test_tiny_activation_credit_modes_return_compact_non_persistent_receipts(
    tmp_path: Path,
):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=8), parent)
    parent_sha = file_sha256(parent)

    smoke_receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_activation_credit_smoke",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=4,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        oracle_screen_mode=ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE,
        oracle_screen_max_sampled_candidates=8,
    )
    assert smoke_receipt["science_arm"] is None
    assert smoke_receipt["oracle_screen_mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE
    assert smoke_receipt["stop_reason"] == "oracle_screen_completed"
    smoke_oracle = smoke_receipt["oracle_screen"]
    assert smoke_oracle["mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_SCALE_SMOKE
    assert smoke_oracle["branch_classification"] is None
    assert smoke_oracle["capture_device_mode"] == "device_resident"
    assert smoke_oracle["non_persistence"]["q_persisted"] is False
    assert Path(smoke_receipt["receipt_path"]).exists()

    full_receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_activation_credit_full",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=8,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        oracle_screen_mode=ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT,
        oracle_screen_max_sampled_candidates=32,
    )
    assert full_receipt["science_arm"] is None
    assert full_receipt["oracle_screen_mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT
    assert full_receipt["stop_reason"] == "oracle_screen_completed"
    full_oracle = full_receipt["oracle_screen"]
    assert full_oracle["mode"] == ORACLE_SCREEN_MODE_ACTIVATION_CREDIT_MEASUREMENT
    assert full_oracle["branch_classification"] in {
        BRANCH_ACTIVATION_CREDIT_CANDIDATE_SIGNAL,
        BRANCH_ACTIVATION_CREDIT_MISSING_SIGNAL_DEEPER_THAN_FIRST_ORDER_CREDIT_STORAGE,
        BRANCH_ACTIVATION_CREDIT_AMBIGUOUS_NO_BRANCH,
    }
    assert full_oracle["compact_summary"]["target_tie_band"][
        "fresh_confirmation_seed_required_for_persistent_followup"
    ] == 71
    assert full_oracle["non_persistence"]["q_persisted"] is False
    assert Path(full_receipt["receipt_path"]).exists()


def test_within_tie_band_null_fraction_helpers_are_directional():
    assert _fraction_gte_observed([0.10, 0.25, 0.50], 0.25) == pytest.approx(2.0 / 3.0)
    assert _fraction_lte_observed([0.10, 0.25, 0.50], 0.25) == pytest.approx(2.0 / 3.0)


def test_within_tie_band_family_metrics_emit_histogram_and_one_sided_null_guards():
    target_band_candidates = [
        {
            "candidate_id": "a",
            "state_key": "layer0.weight",
            "transition_class": "q0|dir1",
            "current_rank_quartile_within_state": 0,
            "flat_index_quartile": 0,
            "current_rank_position": 0,
            "local_loss_delta": -1.0,
        },
        {
            "candidate_id": "b",
            "state_key": "layer0.weight",
            "transition_class": "q0|dir1",
            "current_rank_quartile_within_state": 0,
            "flat_index_quartile": 1,
            "current_rank_position": 1,
            "local_loss_delta": -0.6,
        },
        {
            "candidate_id": "c",
            "state_key": "layer1.weight",
            "transition_class": "q0|dir1",
            "current_rank_quartile_within_state": 1,
            "flat_index_quartile": 2,
            "current_rank_position": 2,
            "local_loss_delta": -0.2,
        },
        {
            "candidate_id": "d",
            "state_key": "layer2.weight",
            "transition_class": "q-1|dir1",
            "current_rank_quartile_within_state": 2,
            "flat_index_quartile": 3,
            "current_rank_position": 3,
            "local_loss_delta": 0.1,
        },
    ]

    metrics = _within_tie_band_family_metrics(
        target_band_candidates=target_band_candidates,
        family_id=WITHIN_TIE_BAND_PRIMARY_FAMILY_ID,
        oracle_best_candidate=target_band_candidates[0],
        oracle_top1_delta=-1.0,
    )

    assert metrics["family_id"] == WITHIN_TIE_BAND_PRIMARY_FAMILY_ID
    assert metrics["bucket_count"] == 3
    assert metrics["bucket_cardinality_histogram"] == {"1": 2, "2": 1}
    assert metrics["singleton_bucket_count"] == 2
    assert metrics["oracle_best_bucket_candidate_count"] == 2
    assert metrics["oracle_best_bucket_fraction"] == pytest.approx(0.5)
    assert metrics["oracle_best_bucket_regret_capture_ratio"] > 0.5
    assert 0.0 <= metrics["matched_hash_null_fraction_gte_observed_bucket_fraction"] <= 1.0
    assert 0.0 <= metrics["matched_hash_null_fraction_lte_observed_regret_capture_ratio"] <= 1.0


def test_direct_within_tie_band_runner_emits_compact_family_metrics():
    model, batch, eligible, states = _tiny_forward_fixture(batch_size=8)
    receipt = run_within_tie_band_discriminator_oracle_screen(
        model=model,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch.device("cpu"),
        max_abs_per_tensor=4096,
        extras=model.compute_train_extra_args(1, 1),
        max_sampled_candidates=32,
        max_seconds=oracle_screen_budget_max_seconds(32),
    )

    assert receipt["mode"] == ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR
    assert receipt["same_candidate_set_required"] is True
    assert receipt["candidate_count"] >= receipt["sampled_candidate_count"] > 0
    assert receipt["max_sampled_candidates"] == 32
    assert receipt["oracle_feasible"] is True
    assert receipt["branch_classification"] in {
        BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
        BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
        BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    }
    compact = receipt["compact_summary"]
    assert set(compact) == {
        "candidate_count",
        "sampled_candidate_count",
        "sampled_candidate_table",
        "target_tie_band",
        "family_metrics",
        "telemetry",
    }
    assert compact["target_tie_band"]["target_tie_band_id"] == "voteabs=4|marginabs=4"
    assert compact["telemetry"]["deterministic_hash_control_only"] is True
    first_row = compact["sampled_candidate_table"][0]
    assert {
        "current_q_level",
        "pre_accumulator_i16",
        "new_acc_i32_signed",
        "proposal_direction",
        "threshold_residual_signed",
        "current_rank_quartile_within_state",
        "flat_index_quartile",
        "transition_class",
    }.issubset(first_row)
    if compact["target_tie_band"]["band_candidate_count"] == 0:
        assert compact["family_metrics"]["metrics_by_family_id"] == {}
        assert receipt["branch_classification"] == BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH
    else:
        primary = compact["family_metrics"]["metrics_by_family_id"][
            WITHIN_TIE_BAND_PRIMARY_FAMILY_ID
        ]
        assert "matched_hash_null_fraction_gte_observed_bucket_fraction" in primary
        assert "matched_hash_null_fraction_lte_observed_regret_capture_ratio" in primary
        assert "bucket_cardinality_histogram" in primary
        assert "singleton_bucket_count" in primary


def test_tiny_within_tie_band_mode_returns_compact_non_persistent_receipt(
    tmp_path: Path,
):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=8), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_within_tie_band",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=8,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        oracle_screen_mode=ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR,
        oracle_screen_max_sampled_candidates=32,
    )

    assert receipt["science_arm"] is None
    assert receipt["oracle_screen_mode"] == ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR
    assert receipt["stop_reason"] == "oracle_screen_completed"
    oracle = receipt["oracle_screen"]
    assert oracle["mode"] == ORACLE_SCREEN_MODE_WITHIN_TIE_BAND_DISCRIMINATOR
    assert oracle["same_candidate_set_required"] is True
    assert oracle["branch_classification"] in {
        BRANCH_WITHIN_TIE_BAND_LEARNER_FEATURES_SEPARATE_REGRET,
        BRANCH_WITHIN_TIE_BAND_NEEDS_NEW_LEARNER_STATE,
        BRANCH_WITHIN_TIE_BAND_AMBIGUOUS_NO_BRANCH,
    }
    assert oracle["non_persistence"]["q_persisted"] is False
    assert oracle["non_persistence"]["checkpoint_written"] is False
    assert Path(receipt["receipt_path"]).exists()


def test_tiny_b2_retained_support_receipt_is_cpu_only_and_target_excluded_from_pc(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_b2",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=2,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        b2_retained_supports=["L0b"],
        b2_parent_consistency_weight=1.0,
        b2_pc_aux_mode="telemetry",
        b2_l0b_batch_size=4,
    )

    b2 = receipt["b2_retention"]
    assert receipt["gpu_launched"] is False
    assert receipt["parent_hash_before"] == parent_sha
    assert receipt["parent_hash_after"] == parent_sha
    assert receipt["parent_hash_unchanged"] is True
    assert b2["schema"] == B2_RETAINED_SUPPORT_SCHEMA_VERSION
    assert b2["enabled"] is True
    assert b2["requested_supports"] == ["L0b"]
    assert b2["prior_batches_fed_to_bounded_steps"] is True
    assert b2["replay_ce_veto"] is True
    assert b2["pc_aux_enabled"] is True
    assert b2["pc_aux_mode"] == "telemetry"
    assert b2["target_parent_kl"] is False
    assert b2["target_rows_excluded_from_pc"] is True
    assert b2["support_proofs"]["L0b"]["support_hash16"] == "89174273d21845bc"
    assert b2["support_proofs"]["L0b"]["report_only"] is False
    assert b2["coverage_by_support"]["L0b"]["rows_seen"] == 4
    assert b2["coverage_by_support"]["L0b"]["coverage_cycle_complete"] is False

    step_b2 = receipt["step_reports"]["1"]["b2_retained_support"]
    assert step_b2["enabled"] is True
    assert step_b2["replay_ce_veto_generated"] is True
    assert step_b2["pc_aux_generated"] is True
    assert step_b2["target_parent_kl"] is False
    assert step_b2["target_rows_excluded_from_pc"] is True
    for tensor_stats in receipt["step_reports"]["1"]["step_result"]["tensor_stats"].values():
        assert "replay_ce_veto_count" in tensor_stats
        assert tensor_stats["pc_aux_mode"] == "telemetry"
        assert tensor_stats["pc_aux_veto_count"] == 0
    assert receipt["step_reports"]["1"]["optimizer_identity_proof"]["optimizer_checks"][
        "eligible_optimizer_state_entries"
    ] == 0


def test_tiny_prior_audit_is_report_only_and_preserves_state_hash_parity(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=90), parent)
    parent_sha = file_sha256(parent)
    common = {
        "parent": parent,
        "parent_sha256": parent_sha,
        "device": "cpu",
        "eligible_scope": "first-bitlinear",
        "steps": 1,
        "batch_size": 90,
        "max_len": TINY_ARCH["max_len"],
        "curriculum_seed": 17,
        "enabled": True,
    }

    off_receipt = run_c2p1_probe(
        **common,
        scratch_root=tmp_path / "scratch_off",
    )
    on_receipt = run_c2p1_probe(
        **common,
        scratch_root=tmp_path / "scratch_on",
        prior_audit_supports=["L0c1"],
    )

    prior_audit = on_receipt["prior_audit"]
    assert off_receipt["prior_audit"]["enabled"] is False
    assert prior_audit["schema"] == B1_PRIOR_AUDIT_SCHEMA_VERSION
    assert prior_audit["enabled"] is True
    assert prior_audit["requested_supports"] == ["L0c1"]
    assert prior_audit["prior_batches_fed_to_bounded_steps"] is False
    assert prior_audit["direct_kl"] is False
    assert prior_audit["replay_pc"] == "OUT"
    assert prior_audit["target_parent_kl"] is False
    assert prior_audit["support_proofs"]["L0c1"]["support_hash16"] == "7bc8cd771daab878"
    assert prior_audit["support_proofs"]["L0c1"]["expected_count"] == 121
    assert prior_audit["support_proofs"]["L0c1"]["close_wrapper_report_only"] == {
        "direct_kl": False,
        "replay_pc": "OUT",
        "target_parent_kl": False,
    }
    assert prior_audit["start_reports"]["L0c1"]["phase"] == "prior_audit0"
    assert prior_audit["final_reports"]["L0c1"]["phase"] == "prior_final_audit"
    assert prior_audit["start_reports"]["L0c1"]["support_rows_audited"] == 121
    assert prior_audit["final_reports"]["L0c1"]["support_rows_audited"] == 121
    assert prior_audit["deltas"]["L0c1"]["broad_cluster_classification"] in {
        "no-new-broad-cluster",
        "broad-cluster",
    }
    assert "parent_baseline_vs_final" in prior_audit["deltas"]["L0c1"]

    assert _state_parity_subset(off_receipt) == _state_parity_subset(on_receipt)


def test_explicit_global_cap_off_keeps_tiny_state_parity(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=32), parent)
    parent_sha = file_sha256(parent)
    common = {
        "parent": parent,
        "parent_sha256": parent_sha,
        "device": "cpu",
        "eligible_scope": "first-bitlinear",
        "steps": 1,
        "batch_size": 32,
        "max_len": TINY_ARCH["max_len"],
        "curriculum_seed": 17,
        "enabled": True,
    }

    implicit_off = run_c2p1_probe(
        **common,
        scratch_root=tmp_path / "scratch_implicit_off",
    )
    explicit_off = run_c2p1_probe(
        **common,
        scratch_root=tmp_path / "scratch_explicit_off",
        global_cap_contract=GLOBAL_CAP_CONTRACT_OFF,
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    )

    assert implicit_off["global_cap_contract"]["name"] == GLOBAL_CAP_CONTRACT_OFF
    assert implicit_off["global_cap_contract"]["enabled"] is False
    assert explicit_off["global_cap_contract"]["name"] == GLOBAL_CAP_CONTRACT_OFF
    assert explicit_off["tie_rule_mode"] == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    assert _state_parity_subset(implicit_off) == _state_parity_subset(explicit_off)


def test_tiny_exact_global_cap_receipt_exposes_banked_faithful_contract(tmp_path: Path):
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(batch_size=16), parent)
    parent_sha = file_sha256(parent)

    receipt = run_c2p1_probe(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=tmp_path / "scratch_global_cap_exact",
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=16,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    )

    assert receipt["global_cap_contract"]["name"] == (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    )
    assert receipt["global_cap_contract"]["finite_schedule_source"] == [512, 512, 256, 256]
    assert receipt["global_cap_contract"]["long_run_translation"] == (
        "steps 1..2 cap=512; steps >=3 cap=256"
    )
    assert receipt["tie_rule_mode"] == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    summary = receipt["step_reports"]["1"]["step_result"]["global_summary"]
    assert summary["global_rate_cap_enabled"] is True
    assert summary["global_rate_cap_contract_name"] == (
        C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME
    )
    assert summary["global_tie_rule_mode"] == EXACT_GLOBAL_CAP_TIE_RULE_MODE
    assert summary["global_rate_cap_cap"] == 512


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
