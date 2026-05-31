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
    assert "Fold A" in locked
    assert "Fold G" in locked
    assert "cpu_sampler_gpu_aggregation_replay" in locked
    assert prereg["strict_reproduction"]["expected_global_agreement"] == bridge.STRICT_REPRODUCTION_EXPECTED
    assert prereg["strict_reproduction"]["tolerance_abs"] == bridge.STRICT_REPRODUCTION_TOL
    assert prereg["null_label_scheme"] == "slice1_shared_seed_labels"
    assert prereg["null_backend"]["default"] == bridge.NULL_BACKEND_CPU_LOCKED
    assert prereg["null_backend"]["candidate"] == bridge.NULL_BACKEND_CPU_SAMPLER_GPU_AGGREGATION_REPLAY
    assert prereg["null_backend"]["speedup_floor"] == bridge.DEFAULT_NULL_SPEEDUP_FLOOR
    assert prereg["credit_variants"]["order"] == list(bridge.CREDIT_VARIANTS)
    assert "slice-1 labels" in prereg["credit_variants"]["null_seed_label_scheme"]
    assert prereg["credit_variants"]["fp16_groupwise"]["group_size"] == 128
    assert prereg["credit_variants"]["fp16_groupwise"]["scale_stat"] == "mean_abs"
    assert prereg["credit_terminal_labels"] == list(bridge.CREDIT_TERMINAL_LABELS)
    assert prereg["terminal_labels"] == list(bridge.NULL_PARITY_TERMINAL_LABELS)
