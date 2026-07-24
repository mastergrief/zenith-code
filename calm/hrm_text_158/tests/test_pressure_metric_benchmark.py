"""Production-path / runtime / warmup / proof / formal negatives (PLAN_v6 rev4).

Real-checkpoint tests (load_and_patch_runtime + train-step + probes) are
GPU-lane work per workflow.md §"Full-GPU for trainer-loop work"; they run only
with PM_REAL_CKPT_TESTS=1 (device via PM_REAL_CKPT_DEVICE, default cuda:0) and
are excluded from the default developer suite. Synthetic-runtime tests cover
the same production install/mutation path in milliseconds.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.pressure_metric_benchmark import (
    aggregate_replicate_gate_flags,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
    ProofValidationError,
    _validate_replicate_rows,
    load_and_validate_paired_proof,
    require_cuda_proof_device,
    sha256_file,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    PARENT_SHA256,
    PAIRED_N,
    PLAN_SHA256,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_warmup_runtime import (
    assert_final_probe_surface_tracks_q,
    prove_q_forward_coupling_on_runtime,
    run_hotpath_warmup_throwaway,
    run_one_diagnostic_loop,
)
from calm.hrm_text_158.native_full_stack.screen_model_runtime import (
    _install_fixed_qscale_forwards,
    assert_q_levels_coupled,
    load_and_patch_runtime,
)

REPO = Path(__file__).resolve().parents[3]
PARENT = (
    REPO
    / "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c2K2add50s_seed0017_replay80_n12k_lr5e5_pc1p0_"
    "rsL0b1math1r1b2m1idfull1k1to4pin1k5to8pin1L0c1pin1_ceL0c1x3_anchorsv1r3_"
    "from_L0c2K2add120k5to8_step01000_final_step00750.pt"
)

# Real-ckpt lane: opt-in only (GPU test-operator lane), never the default suite.
REAL_CKPT_ENABLED = os.environ.get("PM_REAL_CKPT_TESTS") == "1"
REAL_CKPT_DEVICE = os.environ.get("PM_REAL_CKPT_DEVICE", "cuda:0")
real_ckpt = pytest.mark.skipif(
    not REAL_CKPT_ENABLED or not PARENT.is_file(),
    reason="real-ckpt lane: set PM_REAL_CKPT_TESTS=1 (GPU test-operator lane)",
)


def _tiny_patched_runtime():
    """Minimal production install path (same _install_fixed_qscale_forwards)."""
    mod = BitLinear(4, 3, bias=False)
    with torch.no_grad():
        mod.weight.copy_(torch.randn(3, 4) * 0.1)
    modules = {"w": mod}
    scale = mod.weight.detach().float().abs().mean().clamp(min=1e-5)
    frozen = {"w": scale.detach().cpu().to(torch.float32).reshape(())}
    q = {
        "w": (mod.weight.detach().float() / scale)
        .round()
        .clamp(-1, 1)
        .to(torch.int8)
        .cpu()
    }
    _install_fixed_qscale_forwards(modules, q, frozen)
    rt = {
        "m": torch.nn.Module(),  # unused
        "modules": modules,
        "eligible": ["w"],
        "q_levels": q,
        "frozen_scales": frozen,
    }
    return rt, mod


def test_q_forward_coupling_production_install_path():
    """B1: mutating the EXACT captured q_levels dict changes FixedQScale forward."""
    from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import begin_credit_step

    rt, mod = _tiny_patched_runtime()
    assert_q_levels_coupled(rt, rt["q_levels"])
    clone = {k: v.clone() for k, v in rt["q_levels"].items()}
    with pytest.raises(RuntimeError, match="q-forward decoupling"):
        assert_q_levels_coupled(rt, clone)
    x = torch.randn(2, 4)
    with torch.no_grad():
        begin_credit_step(["w"])
        y0 = mod.forward(x).clone()
        # In-place
        flat = rt["q_levels"]["w"].view(-1)
        old = int(flat[0].item())
        flat[0] = 0 if old != 0 else 1
        begin_credit_step(["w"])
        y1 = mod.forward(x).clone()
        flat[0] = old
        # Loop-style writeback reassignment into SAME dict
        q_new = rt["q_levels"]["w"].clone()
        q_new.view(-1)[0] = 0 if old != 0 else 1
        rt["q_levels"]["w"] = q_new
        begin_credit_step(["w"])
        y2 = mod.forward(x).clone()
    assert not torch.equal(y0, y1)
    assert not torch.equal(y0, y2)


@real_ckpt
def test_q_forward_coupling_on_real_runtime():
    rt = load_and_patch_runtime(ckpt_path=str(PARENT), device=REAL_CKPT_DEVICE)
    out = prove_q_forward_coupling_on_runtime(rt)
    assert out["ok"] is True
    assert out["forward_changed"] is True
    assert out["writeback_changed"] is True
    assert out["q_levels_is_rt"] is True


@real_ckpt
def test_probe_order_through_real_path():
    trace: list[str] = []
    result = run_one_diagnostic_loop(
        ckpt_path=str(PARENT),
        device=REAL_CKPT_DEVICE,
        steps=1,
        batch=2,
        topk=64,
        telemetry=True,
        skip_probes=False,
        seed=0,
        warmup_enable=True,
        probe_order_trace=trace,
        formal_mode=False,
    )
    assert trace == ["step0", "final"]
    assert result["probes"]["step0_taken_before_train"] is True
    assert result["q_levels_is_rt"] is True
    assert result["warmup"]["hot_path_executed"] is True
    assert result["warmup"]["ids_differ"] is True
    assert result["warmup"]["post_warmup_reload"] is True
    # Final probe surface tracks q when writebacks occur
    surface = assert_final_probe_surface_tracks_q(result)
    assert surface["ok"] is True


@real_ckpt
def test_warmup_noop_refuses_evidence():
    ev = run_hotpath_warmup_throwaway(
        ckpt_path=str(PARENT),
        device=REAL_CKPT_DEVICE,
        batch=2,
        n_steps=1,
        seed=0,
        enable=False,
    )
    assert ev["non_mutating_warmup"] is False
    assert ev["hot_path_executed"] is False


def test_cpu_proof_requires_override():
    with pytest.raises(SystemExit, match="CUDA"):
        require_cuda_proof_device("cpu", diagnostic_override=False)
    assert require_cuda_proof_device("cpu", diagnostic_override=True) is False


def test_formal_refuses_skip_probes(tmp_path):
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
        run_formal_diagnostic,
    )

    with pytest.raises(SystemExit, match="REFUSES skip_probes"):
        run_formal_diagnostic(
            ckpt_path=str(PARENT) if PARENT.is_file() else str(tmp_path / "x.pt"),
            device="cuda",
            steps=150,
            batch=8,
            topk=1024,
            seed=0,
            telemetry=True,
            skip_probes=True,
            paired_proof_json="",
            paired_proof_sha256="",
            repo_root=str(REPO),
            output_json=None,
        )


def test_formal_refuses_telemetry_disabled(tmp_path):
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
        run_formal_diagnostic,
    )

    with pytest.raises(SystemExit, match="telemetry-disabled"):
        run_formal_diagnostic(
            ckpt_path=str(PARENT) if PARENT.is_file() else str(tmp_path / "x.pt"),
            device="cuda",
            steps=150,
            batch=8,
            topk=1024,
            seed=0,
            telemetry=False,
            skip_probes=False,
            paired_proof_json="",
            paired_proof_sha256="",
            repo_root=str(REPO),
            output_json=None,
        )


def test_proof_empty_replicate_map_refuses():
    """Direct production validator path — empty map must refuse (no CUDA gate)."""
    with pytest.raises(ProofValidationError, match="empty"):
        _validate_replicate_rows(
            replicates={},
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def test_proof_missing_sha_required(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("{}")
    with pytest.raises(SystemExit, match="REQUIRED"):
        load_and_validate_paired_proof(
            path=str(p),
            expected_sha256="",
            repo_root=str(REPO),
            formal_device="cuda:0",
            formal_batch=8,
            formal_topk=1024,
            formal_steps=150,
        )


def test_proof_sha_mismatch(tmp_path):
    p = tmp_path / "proof.json"
    p.write_text("{}")
    with pytest.raises(SystemExit, match="mismatch"):
        load_and_validate_paired_proof(
            path=str(p),
            expected_sha256="deadbeef",
            repo_root=str(REPO),
            formal_device="cuda:0",
            formal_batch=8,
            formal_topk=1024,
            formal_steps=150,
        )


def _minimal_rep_row(**overrides):
    from calm.hrm_text_158.native_full_stack.family_classifier import ARM0
    from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
        PAIRED_STEPS,
    )

    # Matching A/B per-index hashes (identical synthetic identity).
    flip_h = "11" * 32
    q_h = "22" * 32
    app_h = "33" * 32
    row = {
        "rep_index": 0,
        "order": "AB",
        "flavor": "A",
        "parent_sha": PARENT_SHA256,
        "q_sha_before": "abc123",
        "source_hashes": {},
        "geometry": {
            "steps": PAIRED_STEPS,
            "batch": 8,
            "topk": 1024,
            "arm": ARM0,
            "device": "cuda:0",
        },
        "seed": 0,
        "warmup": {"non_mutating_warmup": True, "hot_path_executed": True},
        "measurements": {"n_flips": 1, "q_changed_count": 1, "credited_mass": 1},
        "wall_ms_per_step": 1.0,
        "two_tier_threshold_assert_pass": True,
        "flip_count_sha256": flip_h,
        "q_final_sha256": q_h,
        "applied_identity_sha256": app_h,
        "flip_count_equal": True,
        "q_final_equal": True,
        "applied_identity_equal": True,
    }
    row.update(overrides)
    return row


def _full_replicate_pack(*, wall_ms=1.0, arm=None, meas=None, warmup=None):
    from calm.hrm_text_158.native_full_stack.family_classifier import ARM0

    use_arm = ARM0 if arm is None else arm

    def _rows(order, flavor):
        out = []
        for i in range(PAIRED_N):
            kw = {
                "rep_index": i,
                "order": order,
                "flavor": flavor,
                "seed": i,
                "geometry": {
                    "steps": 25,
                    "batch": 8,
                    "topk": 1024,
                    "arm": use_arm,
                    "device": "cuda:0",
                },
            }
            if wall_ms is not None:
                kw["wall_ms_per_step"] = wall_ms
            if meas is not None:
                kw["measurements"] = meas
            if warmup is not None:
                kw["warmup"] = warmup
            out.append(_minimal_rep_row(**kw))
        return out

    ab_a, ab_b = _rows("AB", "A"), _rows("AB", "B")
    ba_a, ba_b = _rows("BA", "A"), _rows("BA", "B")
    for i in range(PAIRED_N):
        ab_b[i]["q_sha_before"] = ab_a[i]["q_sha_before"]
        ba_b[i]["q_sha_before"] = ba_a[i]["q_sha_before"]
    return {"AB_A": ab_a, "AB_B": ab_b, "BA_A": ba_a, "BA_B": ba_b}


def test_tampered_ab_artifact_detected(tmp_path, monkeypatch):
    """Validator re-hashes named A/B files — tamper must refuse (CUDA gated)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    uninst = tmp_path / "u.json"
    inst = tmp_path / "i.json"
    uninst.write_text('{"ok": true}')
    inst.write_text('{"ok": true}')
    reps = _full_replicate_pack()
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
        source_file_hashes,
    )

    live = source_file_hashes(str(REPO))
    for rows in reps.values():
        for row in rows:
            row["source_hashes"] = dict(live)
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof_contract import (
        bind_amendment_into_summary,
        load_live_amendment,
    )

    proof = {
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "is_proof": True,
        "accepted": True,
        "N": PAIRED_N,
        "steps": 25,
        "batch": 8,
        "topk": 1024,
        "device": "cuda:0",
        "two_tier_threshold_assert_pass": True,
        "protocol": "AB_BA_median_of_N",
        # rev5: must be internally consistent with replicate walls (all 1.0 →
        # recomputed overhead 0.0) or the recompute guard fires first.
        "overhead_frac_AB": 0.0,
        "overhead_frac_BA": 0.0,
        "determinism_prefix_match": True,
        "determinism_per_index_match": True,
        "warmup_ok_all_replicates": True,
        "replicates": reps,
        "artifact_paths": {
            "uninstrumented_25": str(uninst),
            "instrumented_25": str(inst),
        },
        "determinism_reference_sha256s": {"uninstrumented_25": "wrong_hash"},
        "instrumented_sha256s": {"instrumented_25": sha256_file(str(inst))},
        "source_hashes": live,
    }
    amendment, amendment_sha = load_live_amendment(str(REPO))
    bind_amendment_into_summary(
        proof, amendment_sha256=amendment_sha, amendment=amendment
    )
    p = tmp_path / "proof.json"
    blob = json.dumps(proof)
    p.write_text(blob)
    sha = hashlib.sha256(blob.encode()).hexdigest()
    with pytest.raises(ProofValidationError, match="uninstrumented artifact sha"):
        load_and_validate_paired_proof(
            path=str(p),
            expected_sha256=sha,
            repo_root=str(REPO),
            formal_device="cuda:0",
            formal_batch=8,
            formal_topk=1024,
            formal_steps=150,
        )


def test_proof_wrong_nested_geometry_refuses():
    rows_a = []
    rows_b = []
    for i in range(PAIRED_N):
        rows_a.append(
            _minimal_rep_row(
                rep_index=i,
                order="AB",
                flavor="A",
                seed=i,
                geometry={
                    "steps": 25,
                    "batch": 8,
                    "topk": 1024,
                    "arm": "WRONG_ARM",
                    "device": "cuda:0",
                },
            )
        )
        rows_b.append(
            _minimal_rep_row(
                rep_index=i,
                order="AB",
                flavor="B",
                seed=i,
                q_sha_before=rows_a[-1]["q_sha_before"],
            )
        )
    ba_a = [
        _minimal_rep_row(rep_index=i, order="BA", flavor="A", seed=i)
        for i in range(PAIRED_N)
    ]
    ba_b = [
        _minimal_rep_row(
            rep_index=i,
            order="BA",
            flavor="B",
            seed=i,
            q_sha_before=ba_a[i]["q_sha_before"],
        )
        for i in range(PAIRED_N)
    ]
    with pytest.raises(ProofValidationError, match="geometry.arm"):
        _validate_replicate_rows(
            replicates={"AB_A": rows_a, "AB_B": rows_b, "BA_A": ba_a, "BA_B": ba_b},
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def test_proof_missing_per_rep_counters_refuses():
    def _rows(order, flavor, meas=None):
        out = []
        for i in range(PAIRED_N):
            out.append(
                _minimal_rep_row(
                    rep_index=i,
                    order=order,
                    flavor=flavor,
                    seed=i,
                    measurements=meas
                    if meas is not None
                    else {"n_flips": 1, "q_changed_count": 1, "credited_mass": 1},
                )
            )
        return out

    bad = _rows("AB", "A", meas={"n_flips": 1})  # missing q_changed/credited
    good_b = _rows("AB", "B")
    for i in range(PAIRED_N):
        good_b[i]["q_sha_before"] = bad[i]["q_sha_before"]
    ba_a = _rows("BA", "A")
    ba_b = _rows("BA", "B")
    for i in range(PAIRED_N):
        ba_b[i]["q_sha_before"] = ba_a[i]["q_sha_before"]
    with pytest.raises(ProofValidationError, match="measurements"):
        _validate_replicate_rows(
            replicates={"AB_A": bad, "AB_B": good_b, "BA_A": ba_a, "BA_B": ba_b},
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def test_proof_missing_wall_ms_refuses():
    reps = _full_replicate_pack()
    del reps["AB_A"][0]["wall_ms_per_step"]
    with pytest.raises(ProofValidationError, match="wall_ms_per_step"):
        _validate_replicate_rows(
            replicates=reps,
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def test_warmup_and_threshold_require_every_replicate():
    """PRODUCTION reducer: mid-rep fail → aggregate false (not last-only)."""
    mid_fail = aggregate_replicate_gate_flags(
        warmup_flags=[True, False, True],
        threshold_flags=[True, True, True],
    )
    assert mid_fail["warmup_ok_all"] is False
    assert mid_fail["threshold_ok_all"] is True
    # Both orders must AND
    ab = mid_fail
    ba = aggregate_replicate_gate_flags(
        warmup_flags=[True, True, True],
        threshold_flags=[True, True, True],
    )
    assert (ab["warmup_ok_all"] and ba["warmup_ok_all"]) is False
    # Empty flags → refuse (fail-closed)
    empty = aggregate_replicate_gate_flags(warmup_flags=[], threshold_flags=[])
    assert empty["warmup_ok_all"] is False
    assert empty["threshold_ok_all"] is False


def test_proof_failed_non_last_warmup_refuses():
    """Proof validator refuses if ANY replicate warmup evidence fails (not last-only)."""

    def _pack(order):
        a, b = [], []
        for i in range(PAIRED_N):
            wu = {"non_mutating_warmup": True, "hot_path_executed": True}
            if order == "AB" and i == 1:  # non-last fail
                wu = {"non_mutating_warmup": False, "hot_path_executed": False}
            ra = _minimal_rep_row(
                rep_index=i, order=order, flavor="A", seed=i, warmup=wu
            )
            rb = _minimal_rep_row(
                rep_index=i,
                order=order,
                flavor="B",
                seed=i,
                q_sha_before=ra["q_sha_before"],
                warmup=wu
                if order != "AB" or i != 1
                else {
                    "non_mutating_warmup": True,
                    "hot_path_executed": True,
                },
            )
            a.append(ra)
            b.append(rb)
        return a, b

    ab_a, ab_b = _pack("AB")
    ba_a, ba_b = _pack("BA")
    with pytest.raises(ProofValidationError, match="warmup evidence"):
        _validate_replicate_rows(
            replicates={"AB_A": ab_a, "AB_B": ab_b, "BA_A": ba_a, "BA_B": ba_b},
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def _forged_summary_proof(tmp_path, reps, **top_overrides):
    """Consistent-by-default proof payload for consumer recompute negatives.

    Recompute runs BEFORE artifact re-hash, so dummy artifact paths suffice
    when the expected failure is a recompute disagreement.
    """
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
        source_file_hashes,
    )
    from calm.hrm_text_158.native_full_stack.pressure_metric_proof_contract import (
        bind_amendment_into_summary,
        load_live_amendment,
    )

    live = source_file_hashes(str(REPO))
    for rows in reps.values():
        for row in rows:
            row["source_hashes"] = dict(live)
    proof = {
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "is_proof": True,
        "accepted": True,
        "N": PAIRED_N,
        "steps": 25,
        "batch": 8,
        "topk": 1024,
        "device": "cuda:0",
        "two_tier_threshold_assert_pass": True,
        "protocol": "AB_BA_median_of_N",
        "overhead_frac_AB": 0.0,
        "overhead_frac_BA": 0.0,
        "determinism_prefix_match": True,
        "determinism_per_index_match": True,
        "warmup_ok_all_replicates": True,
        "replicates": reps,
        "artifact_paths": {
            "uninstrumented_25": str(tmp_path / "u.json"),
            "instrumented_25": str(tmp_path / "i.json"),
        },
        "determinism_reference_sha256s": {},
        "instrumented_sha256s": {},
        "source_hashes": live,
    }
    amendment, amendment_sha = load_live_amendment(str(REPO))
    bind_amendment_into_summary(
        proof, amendment_sha256=amendment_sha, amendment=amendment
    )
    proof.update(top_overrides)
    p = tmp_path / "proof.json"
    blob = json.dumps(proof)
    p.write_text(blob)
    return str(p), hashlib.sha256(blob.encode()).hexdigest()


def _load_proof(path, sha):
    return load_and_validate_paired_proof(
        path=path,
        expected_sha256=sha,
        repo_root=str(REPO),
        formal_device="cuda:0",
        formal_batch=8,
        formal_topk=1024,
        formal_steps=150,
    )


def test_proof_counter_mismatch_forged_determinism_refuses(tmp_path, monkeypatch):
    """A/B counter divergence must refuse even when the summary forges det=True."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reps = _full_replicate_pack()
    reps["AB_B"][1]["measurements"] = {
        "n_flips": 99,
        "q_changed_count": 99,
        "credited_mass": 99,
    }
    path, sha = _forged_summary_proof(tmp_path, reps)
    with pytest.raises(ProofValidationError, match="counters disagree"):
        _load_proof(path, sha)


def test_proof_high_walls_forged_low_overhead_refuses(tmp_path, monkeypatch):
    """Per-rep walls implying >bound overhead must refuse a forged low top-level."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reps = _full_replicate_pack()
    for key in ("AB_B", "BA_B"):
        for row in reps[key]:
            row["wall_ms_per_step"] = 2.0  # recomputed overhead = 1.0 >> 0.15
    path, sha = _forged_summary_proof(
        tmp_path, reps, overhead_frac_AB=0.01, overhead_frac_BA=0.01
    )
    with pytest.raises(ProofValidationError, match="disagrees with replicate recomputation"):
        _load_proof(path, sha)


def test_proof_consistent_high_overhead_refuses(tmp_path, monkeypatch):
    """Even an HONEST summary must refuse when recomputed overhead exceeds bound."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    reps = _full_replicate_pack()
    for key in ("AB_B", "BA_B"):
        for row in reps[key]:
            row["wall_ms_per_step"] = 2.0
    path, sha = _forged_summary_proof(
        tmp_path, reps, overhead_frac_AB=1.0, overhead_frac_BA=1.0
    )
    with pytest.raises(ProofValidationError, match="exceeds"):
        _load_proof(path, sha)


def test_proof_non_last_instrumented_threshold_false_refuses():
    """One non-last instrumented replicate with threshold=false must refuse."""
    reps = _full_replicate_pack()
    reps["AB_B"][1]["two_tier_threshold_assert_pass"] = False
    with pytest.raises(ProofValidationError, match="instrumented replicate threshold"):
        _validate_replicate_rows(
            replicates=reps,
            live_src={},
            seed0=0,
            batch=8,
            topk=1024,
            device="cuda:0",
        )


def test_assert_q_levels_coupled_rejects_clone_dict():
    rt, _mod = _tiny_patched_runtime()
    clone_dict = dict(rt["q_levels"])  # shallow dict clone (new mapping)
    with pytest.raises(RuntimeError, match="q-forward decoupling"):
        assert_q_levels_coupled(rt, clone_dict)
