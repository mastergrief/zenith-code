"""Tests for the HRM-Text-1.58 credit bridge diagnostic helpers."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch


def _import_credit_bridge():
    repo_root = Path(__file__).resolve().parents[3]
    script_path = repo_root / "scripts" / "hrm_text_158_credit_bridge.py"
    spec = importlib.util.spec_from_file_location("_test_hrm_text_158_credit_bridge", str(script_path))
    assert spec is not None and spec.loader is not None
    if "_test_hrm_text_158_credit_bridge" in sys.modules:
        return sys.modules["_test_hrm_text_158_credit_bridge"]
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_test_hrm_text_158_credit_bridge"] = mod
    spec.loader.exec_module(mod)
    return mod


def _count_accumulator(bridge, label: str, *, scale: int = 1, q0: int = 3, mixed: bool = True):
    counts = bridge.CountAccumulator(label=label)
    counts.denom = 16 * scale
    counts.agree = 8 * scale
    counts.buckets_global = [
        bridge.BucketCounts(
            fp_pos=6 * scale,
            fp_neg=4 * scale,
            int_pos=5 * scale,
            int_neg=3 * scale,
            int_zero=2 * scale,
        )
    ]
    counts.buckets_rowq = [
        bridge.BucketCounts(fp_pos=3 * scale, fp_neg=2 * scale, int_pos=2 * scale, int_neg=2 * scale, int_zero=1),
        bridge.BucketCounts(fp_pos=2 * scale, fp_neg=3 * scale, int_pos=3 * scale, int_neg=1 * scale, int_zero=1),
    ]
    counts.q_stats = {
        "-1": {"denom": 4 * scale if mixed else 0, "agree": 2 * scale if mixed else 0},
        "0": {"denom": q0 * scale, "agree": max(0, q0 * scale - 1)},
        "1": {"denom": 5 * scale if mixed else 0, "agree": 2 * scale if mixed else 0},
    }
    return counts


def _variant_count_sets(bridge):
    out = {}
    for idx, variant in enumerate(bridge.CREDIT_VARIANTS, start=1):
        out[variant] = {
            "global": _count_accumulator(bridge, f"{variant}_global", scale=idx + 2, q0=idx + 2),
            "families": {
                f"{variant}_family": _count_accumulator(bridge, f"{variant}_family", scale=idx + 1, q0=idx + 1)
            },
            "aggregate64": {},
            "invocations": {
                f"{variant}_small_easy": _count_accumulator(bridge, f"{variant}_small_easy", scale=1, q0=0, mixed=False),
                f"{variant}_mixed_q0": _count_accumulator(bridge, f"{variant}_mixed_q0", scale=idx + 1, q0=idx + 3),
            },
        }
    return out


def test_project_fp_gradient_to_admissible_ternary_moves() -> None:
    bridge = _import_credit_bridge()
    q = torch.tensor([[-1, -1, 0, 0, 1, 1]], dtype=torch.int8)
    grad = torch.tensor([[-2.0, 3.0, -0.5, 0.25, -4.0, 6.0]])

    moves = bridge.project_fp_gradient_to_moves(grad, q)

    assert moves.tolist() == [[1, 0, 1, -1, 0, -1]]


def test_project_integer_credit_includes_zero_revival() -> None:
    bridge = _import_credit_bridge()
    q = torch.tensor([[-1, -1, 0, 0, 1, 1]], dtype=torch.int8)
    credit = torch.tensor([[4, -3, 5, -6, 7, -8]], dtype=torch.int32)

    moves = bridge.project_integer_credit_to_moves(credit, q)

    assert moves.tolist() == [[1, 0, 1, -1, 0, -1]]


def test_full_magnitude_ceiling_projects_identically_to_fp() -> None:
    bridge = _import_credit_bridge()
    q = torch.tensor([[-1, 0, 1]], dtype=torch.int8)
    grad = torch.tensor([[-2.0, 3.0, 4.0]])

    fp_moves = bridge.project_fp_gradient_to_moves(grad, q)
    ceiling_moves = bridge.project_integer_credit_to_moves(-grad, q)

    assert ceiling_moves.tolist() == fp_moves.tolist()


def test_pow2_bucket_weights_signed_credit_before_projection() -> None:
    bridge = _import_credit_bridge()
    grad = torch.tensor([[[1.0], [-1.0]]])
    inputs = torch.tensor([[[1.0], [100.0]]])

    strict = bridge.strict_sign_credit(grad, inputs)
    pow2 = bridge.pow2_bucket_credit(grad, inputs)

    assert strict.item() == 0
    assert pow2.item() > 0


def test_fp16_groupwise_mean_abs_can_flip_strict_cancellation() -> None:
    bridge = _import_credit_bridge()
    grad = torch.tensor([[[1.0, 1.0], [-1.0, 100.0]]])
    inputs = torch.ones(1, 2, 1)

    strict = bridge.strict_sign_credit(grad, inputs)
    fp16_groupwise = bridge.fp16_groupwise_credit(grad, inputs, group_size=128)

    assert strict[0, 0].item() == 0
    assert fp16_groupwise[0, 0].item() > 0


def test_row_q_preserving_null_uses_q_buckets() -> None:
    bridge = _import_credit_bridge()
    fp = torch.tensor([[1, 1, -1, -1], [1, -1, 1, -1]], dtype=torch.int8)
    im = torch.tensor([[1, -1, -1, 0], [-1, -1, 1, 0]], dtype=torch.int8)
    q = torch.tensor([[-1, -1, 0, 0], [1, 1, 0, 0]], dtype=torch.int8)
    mask = fp != 0

    buckets = bridge.row_q_bucket_counts(fp, im, q, mask)

    assert sum(b.total for b in buckets) == 8
    assert len(buckets) == 4
    assert all(b.total == 2 for b in buckets)


def test_simulated_null_is_deterministic_for_seed() -> None:
    bridge = _import_credit_bridge()
    buckets = [bridge.BucketCounts(fp_pos=6, fp_neg=4, int_pos=5, int_neg=3, int_zero=2)]

    a = bridge.simulate_permutation_null(buckets, permutations=32, seed=17)
    b = bridge.simulate_permutation_null(buckets, permutations=32, seed=17)

    assert a == b
    assert 0.0 <= a["mean"] <= 1.0
    assert a["p95"] <= a["p99"]


def test_cpu_sampler_gpu_aggregation_replay_matches_cpu_locked() -> None:
    bridge = _import_credit_bridge()
    buckets = [
        bridge.BucketCounts(fp_pos=6, fp_neg=4, int_pos=5, int_neg=3, int_zero=2),
        bridge.BucketCounts(fp_pos=9, fp_neg=7, int_pos=8, int_neg=4, int_zero=4),
    ]

    cpu = bridge.simulate_permutation_null(
        buckets,
        permutations=32,
        seed=17,
        backend=bridge.NULL_BACKEND_CPU_LOCKED,
    )
    replay_cpu = bridge.simulate_permutation_null(
        buckets,
        permutations=32,
        seed=17,
        backend=bridge.NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
        aggregation_device="cpu",
        profile=True,
    )

    assert {k: replay_cpu[k] for k in ("mean", "p95", "p99")} == {k: cpu[k] for k in ("mean", "p95", "p99")}
    assert replay_cpu["backend"] == bridge.NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY
    assert replay_cpu["timing_seconds"]["cpu_sampler"] >= 0.0
    assert replay_cpu["timing_seconds"]["aggregation"] >= 0.0

    if torch.cuda.is_available():
        replay_cuda = bridge.simulate_permutation_null(
            buckets,
            permutations=32,
            seed=17,
            backend=bridge.NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY,
            aggregation_device="cuda",
        )
        assert {k: replay_cuda[k] for k in ("mean", "p95", "p99")} == {k: cpu[k] for k in ("mean", "p95", "p99")}


def test_gpu_native_counts_pmf_analytic_parity_cpu_and_cuda_if_available() -> None:
    bridge = _import_credit_bridge()

    cpu = bridge.run_analytic_pmf_parity("cpu")

    assert cpu["pass"]
    assert cpu["max_tv_distance"] <= bridge.GPU_NATIVE_PMF_TV_BOUND
    assert cpu["max_cdf_delta"] <= bridge.GPU_NATIVE_PMF_CDF_BOUND
    assert cpu["max_omitted_mass"] <= 3 * bridge.GPU_NATIVE_PMF_CAPTURED_MASS_EPS
    assert cpu["reference_candidate_independent"]
    assert cpu["reference_pmf_function"] != cpu["candidate_pmf_function"]
    assert {fixture["q_level"] for fixture in cpu["fixtures"]} == {-1, 0, 1}
    assert any(fixture["support_policy"]["mass_trimmed_bucket_count"] > 0 for fixture in cpu["fixtures"])
    assert all(fixture["scipy_cross_check"]["checked_distribution_count"] > 0 for fixture in cpu["fixtures"])

    if torch.cuda.is_available():
        cuda = bridge.run_analytic_pmf_parity("cuda")
        assert cuda["pass"]
        assert cuda["max_tv_distance"] <= bridge.GPU_NATIVE_PMF_TV_BOUND
        assert cuda["max_cdf_delta"] <= bridge.GPU_NATIVE_PMF_CDF_BOUND


def test_sparse_pmf_metrics_compare_union_support() -> None:
    bridge = _import_credit_bridge()

    metrics = bridge._sparse_pmf_distance_metrics(
        {0: 0.5, 1: 0.5},
        {1: 0.5, 2: 0.5},
        omitted_mass_bound=0.0,
    )

    assert metrics["tv_distance_core"] == 0.5
    assert metrics["tv_distance"] == 0.5
    assert metrics["max_cdf_delta_core"] == 0.5
    assert metrics["max_cdf_delta"] == 0.5
    assert metrics["max_pmf_delta"] == 0.5


def test_reference_pmf_uses_bounded_dense_total_window() -> None:
    bridge = _import_credit_bridge()
    bucket = bridge.BucketCounts(fp_pos=500, fp_neg=9500, int_pos=300, int_neg=200, int_zero=0)

    reference = bridge._joint_match_pmf_reference_scipy_vectorized_sparse(
        bucket,
        support_guard=256,
        captured_mass_eps=bridge.GPU_NATIVE_PMF_CAPTURED_MASS_EPS,
    )

    assert reference["reference_mode"] == "vectorized_chunked_exact"
    assert reference["max_support_size"] > 256
    assert reference["max_stage_window_size"] <= 256
    assert reference["materialized_window_size"] <= 256
    assert len(reference["pmf"]) <= reference["materialized_window_size"]
    assert reference["omitted_mass_bound"] <= bridge.GPU_NATIVE_PMF_CAPTURED_MASS_EPS


def test_vectorized_reference_matches_scalar_loop_on_q0_structured_bucket() -> None:
    bridge = _import_credit_bridge()
    bucket = bridge.BucketCounts(fp_pos=17, fp_neg=13, int_pos=11, int_neg=9, int_zero=10)

    scalar = bridge._joint_match_pmf_reference_scalar_loop_sparse(bucket)
    vectorized = bridge._joint_match_pmf_reference_scipy_vectorized_sparse(bucket)
    metrics = bridge._sparse_pmf_distance_metrics(
        scalar["pmf"],
        vectorized["pmf"],
        omitted_mass_bound=scalar["omitted_mass_bound"] + vectorized["omitted_mass_bound"],
    )

    assert scalar["reference_mode"] == "scalar_loop_exact"
    assert vectorized["reference_mode"] == "vectorized_chunked_exact"
    assert vectorized["joint_work"]["chunking_mode"] == "batched_conditional_scatter"
    assert metrics["tv_distance"] <= 1e-12
    assert metrics["max_cdf_delta"] <= 1e-12


def test_stage1_budget_fallback_self_limits_without_losing_science_tier() -> None:
    bridge = _import_credit_bridge()

    parity = bridge.run_analytic_pmf_parity("cpu", reference_joint_work_budget=5_000)
    fallback = [fixture for fixture in parity["fixtures"] if fixture["reference_mode"] == "bounded_sampled"]

    assert parity["pass"]
    assert parity["exact_backend_certified"] is False
    assert parity["bounded_reference_certified"] is True
    assert parity["explicit_backend_validated_for_science"] is True
    assert parity["default_flip_eligible"] is False
    assert parity["q0_exact_coverage"]["present"] is True
    assert "q_zero_skew_tail_full_support" in parity["q0_exact_coverage"]["fixture_names"]
    assert fallback
    assert all(fixture["bounded_certified"] for fixture in fallback)
    assert all(fixture["science_unblock_eligible"] for fixture in fallback)
    assert all(fixture["sampling_aware_metrics"]["sampling_cdf_delta_bound"] <= bridge.GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND for fixture in fallback)


def test_gpu_native_counts_pmf_preserves_bucket_metadata() -> None:
    bridge = _import_credit_bridge()
    buckets = [
        bridge.BucketCounts(fp_pos=6, fp_neg=4, int_pos=5, int_neg=3, int_zero=2),
        bridge.BucketCounts(fp_pos=30, fp_neg=20, int_pos=25, int_neg=15, int_zero=10),
    ]

    out = bridge.simulate_permutation_null(
        buckets,
        permutations=64,
        seed=17,
        backend=bridge.NULL_BACKEND_GPU_NATIVE_COUNTS_PMF,
        aggregation_device="cpu",
        profile=True,
    )

    assert out["backend"] == bridge.NULL_BACKEND_GPU_NATIVE_COUNTS_PMF
    assert out["input_bucket_metadata"] == out["candidate_batching_metadata"]
    assert out["input_bucket_metadata"]["bucket_count"] == len(buckets)
    assert out["input_bucket_metadata"]["sum_total"] == sum(bucket.total for bucket in buckets)
    assert out["support_policy"]["max_omitted_mass"] <= bridge.GPU_NATIVE_PMF_CAPTURED_MASS_EPS
    assert out["timing_seconds"]["gpu_sampler"] >= 0.0


def test_real_analytic_corpus_names_required_full_subset_extremes() -> None:
    bridge = _import_credit_bridge()
    variant_count_sets = _variant_count_sets(bridge)

    fixtures, manifest = bridge.collect_real_analytic_pmf_fixtures(
        variant_count_sets,
        max_invocations_per_variant=1,
    )
    winners = manifest["real_winners"]

    assert {
        "max_denominator",
        "max_support_size",
        "max_q0_denominator",
        "skew_tail_heavy",
        "global_permutation",
        "row_q_preserving",
    } <= set(winners)
    assert {"global_permutation", "row_q_preserving"} <= set(manifest["null_kind_coverage"])
    assert all(fixture["source"] == "real_full_subset" for fixture in fixtures)

    parity = bridge.run_analytic_pmf_parity(
        "cpu",
        variant_count_sets,
        max_invocations_per_variant=1,
    )
    assert parity["pass"]
    assert parity["corpus_manifest"]["real_full_subset"]["deduped_fixture_count"] == len(fixtures)


def test_tiny_empirical_selector_prefers_q0_mixed_and_caps_invocations() -> None:
    bridge = _import_credit_bridge()
    variant_count_sets = _variant_count_sets(bridge)

    items, manifest = bridge.collect_tiny_empirical_items(variant_count_sets, max_items=1)

    assert len(items) == 1
    assert manifest["selected_count"] == 1
    coverage = manifest["selected"][0]["coverage"]
    assert coverage["has_q0"]
    assert coverage["mixed_q"]
    assert manifest["selection_rule"].startswith("prefer q0+mixed-q")


def test_expected_invocation_schedule_bp_steps_5() -> None:
    bridge = _import_credit_bridge()

    assert bridge.expected_grad_rec_indices("H", bp_steps=5) == {0, 1}
    assert bridge.expected_grad_rec_indices("L", bp_steps=5) == {3, 4, 5}
    assert bridge.expected_forward_calls("H") == 2
    assert bridge.expected_forward_calls("L") == 6


def test_cached_native_flag_guard_rejects_cached_bitlinear() -> None:
    bridge = _import_credit_bridge()
    from calm.hrm_text_158 import BitLinear

    bl = BitLinear(in_features=4, out_features=3, bias=False)
    target = bridge.TargetInfo(
        name="model.H_level.core.layers.0.attn.o_proj",
        level="H",
        layer=0,
        proj="o_proj",
        module=bl,
    )
    bridge.assert_runtime_bitlinear_flags([target])
    bl._cached_active = True

    try:
        bridge.assert_runtime_bitlinear_flags([target])
    except bridge.DiagnosticInvalid as exc:
        assert "_cached_active" in str(exc)
    else:
        raise AssertionError("cached BitLinear guard did not fail")


def test_schedule_excluded_no_grad_requires_exact_96() -> None:
    bridge = _import_credit_bridge()
    excluded = [{"label": str(i), "reason": "schedule_excluded_no_grad"} for i in range(96)]

    bridge.assert_schedule_excluded_no_grad_count(excluded)

    try:
        bridge.assert_schedule_excluded_no_grad_count(excluded[:-1])
    except bridge.DiagnosticInvalid as exc:
        assert "96" in str(exc)
    else:
        raise AssertionError("schedule-excluded count guard did not fail")


def test_prereg_locks_variant_folds(tmp_path: Path) -> None:
    bridge = _import_credit_bridge()
    args = bridge.parse_args(
        [
            "--out-dir",
            str(tmp_path),
            "--ckpt",
            "dummy.pt",
            "--device",
            "cpu",
            "--prereg-only",
        ]
    )

    prereg = bridge.build_prereg(
        args=args,
        ckpt_path=Path("dummy.pt"),
        checkpoint_sha256_before="abc123",
    )

    locked = "\n".join(prereg["locked_tightenings"])
    assert len(prereg["locked_tightenings"]) >= 11
    assert "recurrence-aware" in locked
    assert "prefix/response" in locked
    assert "q=-1/q=0/q=+1" in locked
    assert "row/output-channel-preserving" in locked
    assert "global>=0.65" in locked
    assert "all variants reuse slice-1 null seed labels" in locked
    assert "strict variant is a reproduction sentinel" in locked
    assert "fp16_groupwise uses local group size 128" in locked
    assert "assert len(schedule_excluded_no_grad)==96" in locked
    assert "cached/native" in locked
    assert "F1: analytic PMF parity" in locked
    assert "F2b: reference joint accumulation" in locked
    assert "F4: row-q hard proof" in locked
    assert "F5: no live CPU speed denominator" in locked
    assert "gpu_native_counts_pmf" in prereg["null_backend"]["candidate"]
    assert prereg["strict_reproduction"]["expected_global_agreement"] == bridge.STRICT_REPRODUCTION_EXPECTED
    assert prereg["strict_reproduction"]["tolerance_abs"] == bridge.STRICT_REPRODUCTION_TOL
    assert prereg["null_label_scheme"] == "slice1_shared_seed_labels"
    assert prereg["null_backend"]["default"] == bridge.NULL_BACKEND_CPU_LOCKED
    assert prereg["null_backend"]["oracle"] == bridge.NULL_BACKEND_CPU_LOCKED
    assert prereg["null_backend"]["candidate"] == bridge.NULL_BACKEND_GPU_NATIVE_COUNTS_PMF
    assert prereg["null_backend"]["intended_default_if_all_gates_pass"] == bridge.NULL_BACKEND_GPU_NATIVE_COUNTS_PMF
    assert prereg["null_backend"]["speedup_floor"] == bridge.DEFAULT_NULL_SPEEDUP_FLOOR
    assert prereg["null_backend"]["analytic_tv_bound"] == bridge.GPU_NATIVE_PMF_TV_BOUND
    assert prereg["null_backend"]["analytic_cdf_bound"] == bridge.GPU_NATIVE_PMF_CDF_BOUND
    assert prereg["null_backend"]["captured_mass_epsilon"] == bridge.GPU_NATIVE_PMF_CAPTURED_MASS_EPS
    assert prereg["null_backend"]["reference_joint_work_budget"] == bridge.GPU_NATIVE_REFERENCE_JOINT_WORK_BUDGET
    assert prereg["null_backend"]["reference_chunk_cell_budget"] == bridge.GPU_NATIVE_REFERENCE_CHUNK_CELL_BUDGET
    assert prereg["null_backend"]["bounded_sample_cdf_bound"] == bridge.GPU_NATIVE_BOUNDED_SAMPLE_CDF_BOUND
    assert "explicit_backend_validated_for_science" in prereg["null_backend"]["stage1_certification_tiers"]
    assert "default_flip_eligible" in prereg["null_backend"]["stage1_certification_tiers"]
    assert prereg["null_backend"]["reference_pmf_function"] == bridge.REFERENCE_PMF_FUNCTION
    assert prereg["null_backend"]["candidate_pmf_function"] == bridge.CANDIDATE_PMF_FUNCTION
    assert prereg["null_backend"]["reference_candidate_independent_required"] is True
    assert prereg["null_backend"]["scipy_cross_check_required_where_feasible"] is True
    assert "max_denominator" in prereg["null_backend"]["stage1_real_corpus_required_winners"]
    assert prereg["null_backend"]["stage2_candidate_full_max_seconds"] == bridge.GPU_NATIVE_STAGE2_MAX_SECONDS
    assert "does not erase Stage1+2" in prereg["null_backend"]["stage3_semantics"]
    assert prereg["null_backend"]["cited_cpu_subset_containment_required"] is True
    assert prereg["null_backend"]["speed_cpu_repeats"] == bridge.DEFAULT_NULL_SPEED_CPU_REPEATS
    assert prereg["null_backend"]["speed_candidate_repeats"] == bridge.DEFAULT_NULL_SPEED_CANDIDATE_REPEATS
    assert prereg["credit_variants"]["order"] == list(bridge.CREDIT_VARIANTS)
    assert "slice-1 labels" in prereg["credit_variants"]["null_seed_label_scheme"]
    assert prereg["credit_variants"]["fp16_groupwise"]["group_size"] == 128
    assert prereg["credit_variants"]["fp16_groupwise"]["scale_stat"] == "mean_abs"
    assert prereg["credit_terminal_labels"] == list(bridge.CREDIT_TERMINAL_LABELS)
    assert prereg["terminal_labels"] == list(bridge.GPU_NATIVE_NULL_TERMINAL_LABELS)
    assert "gpu_native_null_parity_explicit_validated_default_deferred" in prereg["terminal_labels"]
