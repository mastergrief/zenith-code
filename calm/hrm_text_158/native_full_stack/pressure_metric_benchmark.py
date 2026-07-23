"""Paired AB/BA orchestration only (PLAN_v6 rev4).

Owns: run_paired_benchmark (median-of-N both orders, AND warmup/threshold over
EVERY replicate, per-rep purity rows, emit three named receipts). Proof
validation/formal live in pressure_metric_proof.py; runtime/warmup in
pressure_metric_warmup_runtime.py.
Dependency: benchmark → warmup_runtime + proof helpers + telemetry.
Bound by PLAN_v6 sha 346b67d8…; rev4 re-scope 1784829182373.
"""
from __future__ import annotations

from typing import Any

from calm.hrm_text_158.native_full_stack.family_classifier import ARM0
from calm.hrm_text_158.native_full_stack.pressure_metric_proof import (
    emit_json,
    require_cuda_proof_device,
    sha256_file,
    source_file_hashes,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    FORMAL_BATCH,
    FORMAL_TOPK,
    OVERHEAD_BOUND,
    PAIRED_N,
    PAIRED_STEPS,
    PLAN_SHA256,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_warmup_runtime import (
    run_one_diagnostic_loop,
)

DEFAULT_UNINSTRUMENTED_25 = (
    "artifacts/acc_entropy/pressure_metric_uninstrumented_25_receipt.json"
)
DEFAULT_INSTRUMENTED_25 = (
    "artifacts/acc_entropy/pressure_metric_instrumented_25_receipt.json"
)
DEFAULT_PAIRED_TIMING = (
    "artifacts/acc_entropy/pressure_metric_paired_25_timing_summary.json"
)


def _median(xs: list[float]) -> float:
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def aggregate_replicate_gate_flags(
    *,
    warmup_flags: list[bool],
    threshold_flags: list[bool],
) -> dict[str, bool]:
    """PRODUCTION reducer: warmup/threshold OK only if EVERY replicate passes."""
    return {
        "warmup_ok_all": bool(warmup_flags) and all(warmup_flags),
        "threshold_ok_all": bool(threshold_flags) and all(threshold_flags),
    }


def _rep_record(
    *,
    result: dict[str, Any],
    rep_index: int,
    order: str,
    flavor: str,
    src_hashes: dict[str, str],
    steps: int,
    batch: int,
    topk: int,
    device: str,
) -> dict[str, Any]:
    return {
        "rep_index": int(rep_index),
        "order": order,
        "flavor": flavor,
        "parent_sha": result["parent_sha"],
        "q_sha_before": result["q_sha_before"],
        "source_hashes": src_hashes,
        "geometry": {
            "steps": int(steps),
            "batch": int(batch),
            "topk": int(topk),
            "arm": ARM0,
            "device": str(device),
        },
        "seed": int(result["seed"]),
        "warmup": result["warmup"],
        "device": str(device),
        "measurements": {
            "n_flips": int(result["measurements"]["n_flips"]),
            "q_changed_count": int(result["measurements"]["q_changed_count"]),
            "credited_mass": int(result["measurements"]["credited_mass"]),
        },
        "wall_ms_per_step": float(result["wall_ms_per_step"]),
        "two_tier_threshold_assert_pass": bool(
            result["store"].two_tier_threshold_assert_pass
            if result.get("store") is not None
            else False
        ),
    }


def run_paired_benchmark(
    *,
    ckpt_path: str,
    device: str,
    batch: int,
    topk: int,
    seed: int,
    paired_n: int,
    repo_root: str,
    uninstrumented_25_json: str | None = None,
    instrumented_25_json: str | None = None,
    paired_timing_json: str | None = None,
    output_json: str | None = None,
    diagnostic_override: bool = False,
) -> int:
    """AB/BA median-of-N paired purity+overhead smoke."""
    n = int(paired_n)
    steps = PAIRED_STEPS
    is_proof = require_cuda_proof_device(
        device, diagnostic_override=bool(diagnostic_override)
    )
    if n != PAIRED_N or batch != FORMAL_BATCH or topk != FORMAL_TOPK:
        if not diagnostic_override:
            raise SystemExit(
                f"paired-benchmark geometry must be N={PAIRED_N} batch={FORMAL_BATCH} "
                f"topk={FORMAL_TOPK}; pass --diagnostic-override for NON-PROOF"
            )
        is_proof = False

    src_hashes = source_file_hashes(repo_root)

    def _run_order(order: str) -> dict[str, Any]:
        a_walls: list[float] = []
        b_walls: list[float] = []
        a_counters: list[dict[str, int]] = []
        b_counters: list[dict[str, int]] = []
        a_reps: list[dict[str, Any]] = []
        b_reps: list[dict[str, Any]] = []
        warmup_flags: list[bool] = []
        threshold_flags: list[bool] = []
        a_last = b_last = None
        for i in range(n):
            seed_i = int(seed) + i

            def _one(tel: bool) -> dict[str, Any]:
                return run_one_diagnostic_loop(
                    ckpt_path=ckpt_path,
                    device=device,
                    steps=steps,
                    batch=int(batch),
                    topk=int(topk),
                    telemetry=tel,
                    skip_probes=True,
                    seed=seed_i,
                    warmup_enable=True,
                    formal_mode=False,
                )

            if order == "AB":
                a, b = _one(False), _one(True)
            else:
                b, a = _one(True), _one(False)

            # AND over EVERY replicate (not last-only)
            for r in (a, b):
                warmup_flags.append(bool(r["warmup"].get("non_mutating_warmup")))
                thr = bool(
                    r["store"].two_tier_threshold_assert_pass
                    if r.get("store") is not None
                    else (False if r["telemetry"] else True)  # uninstrumented N/A→True
                )
                if r["telemetry"]:
                    threshold_flags.append(thr)

            a_walls.append(a["wall_ms_per_step"])
            b_walls.append(b["wall_ms_per_step"])
            a_counters.append(
                {
                    "n_flips": a["measurements"]["n_flips"],
                    "q_changed_count": a["measurements"]["q_changed_count"],
                    "credited_mass": a["measurements"]["credited_mass"],
                }
            )
            b_counters.append(
                {
                    "n_flips": b["measurements"]["n_flips"],
                    "q_changed_count": b["measurements"]["q_changed_count"],
                    "credited_mass": b["measurements"]["credited_mass"],
                }
            )
            a_reps.append(
                _rep_record(
                    result=a,
                    rep_index=i,
                    order=order,
                    flavor="A",
                    src_hashes=src_hashes,
                    steps=steps,
                    batch=batch,
                    topk=topk,
                    device=device,
                )
            )
            b_reps.append(
                _rep_record(
                    result=b,
                    rep_index=i,
                    order=order,
                    flavor="B",
                    src_hashes=src_hashes,
                    steps=steps,
                    batch=batch,
                    topk=topk,
                    device=device,
                )
            )
            a_last, b_last = a, b

        med_a = _median(a_walls)
        med_b = _median(b_walls)
        overhead = (med_b - med_a) / max(med_a, 1e-9)
        prefix_ok = all(a_counters[i] == b_counters[i] for i in range(n))
        return {
            "order": order,
            "median_wall_ms_per_step_A": med_a,
            "median_wall_ms_per_step_B": med_b,
            "overhead_frac": float(overhead),
            "determinism_prefix_match": bool(prefix_ok),
            "a_counters": a_counters,
            "b_counters": b_counters,
            "a_last": a_last,
            "b_last": b_last,
            "a_reps": a_reps,
            "b_reps": b_reps,
            **aggregate_replicate_gate_flags(
                warmup_flags=warmup_flags,
                threshold_flags=threshold_flags,
            ),
        }

    ab = _run_order("AB")
    ba = _run_order("BA")

    uninst_path = uninstrumented_25_json or DEFAULT_UNINSTRUMENTED_25
    inst_path = instrumented_25_json or DEFAULT_INSTRUMENTED_25
    timing_path = paired_timing_json or DEFAULT_PAIRED_TIMING

    a_ref = ab["a_last"]
    b_ref = ab["b_last"]
    assert a_ref is not None and b_ref is not None

    uninst = {
        "screen": "censor_null_pressure_metric_diagnostic/paired_uninstrumented_25",
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "protocol": "AB_BA_median_of_N",
        "is_proof": is_proof,
        "N": n,
        "steps": steps,
        "batch": int(batch),
        "topk": int(topk),
        "device": str(device),
        "telemetry": False,
        "measurements": a_ref["measurements"],
        "route_counters": a_ref["route_counters"],
        "banked_sha": a_ref["banked_sha"],
        "frozen_scale_sha": a_ref["frozen_scale_sha"],
        "q_sha_before": a_ref["q_sha_before"],
        "parent_sha": a_ref["parent_sha"],
        "wall_ms_per_step": a_ref["wall_ms_per_step"],
        "source_hashes": src_hashes,
        "purity": {
            "per_rep_reload": True,
            "non_mutating_warmup": a_ref["warmup"]["non_mutating_warmup"],
            "warmup_evidence": a_ref["warmup"],
            "seed": a_ref["seed"],
        },
    }
    inst = {
        "screen": "censor_null_pressure_metric_diagnostic/paired_instrumented_25",
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "protocol": "AB_BA_median_of_N",
        "is_proof": is_proof,
        "N": n,
        "steps": steps,
        "batch": int(batch),
        "topk": int(topk),
        "device": str(device),
        "telemetry": True,
        "measurements": b_ref["measurements"],
        "route_counters": b_ref["route_counters"],
        "banked_sha": b_ref["banked_sha"],
        "frozen_scale_sha": b_ref["frozen_scale_sha"],
        "q_sha_before": b_ref["q_sha_before"],
        "parent_sha": b_ref["parent_sha"],
        "wall_ms_per_step": b_ref["wall_ms_per_step"],
        "two_tier_threshold_assert_pass": bool(
            b_ref["store"].two_tier_threshold_assert_pass if b_ref["store"] else False
        ),
        "source_hashes": src_hashes,
        "purity": {
            "per_rep_reload": True,
            "non_mutating_warmup": b_ref["warmup"]["non_mutating_warmup"],
            "warmup_evidence": b_ref["warmup"],
            "seed": b_ref["seed"],
        },
    }

    # Write A/B first — external sha authority (no embedded self-hash circularity)
    emit_json(uninst, uninst_path)
    emit_json(inst, inst_path)
    uninst_sha = sha256_file(uninst_path)
    inst_sha = sha256_file(inst_path)

    overhead_ab = float(ab["overhead_frac"])
    overhead_ba = float(ba["overhead_frac"])
    det_ok = bool(ab["determinism_prefix_match"] and ba["determinism_prefix_match"])
    warmup_ok = bool(ab["warmup_ok_all"] and ba["warmup_ok_all"])
    threshold_ok = bool(ab["threshold_ok_all"] and ba["threshold_ok_all"])
    accepted = (
        is_proof
        and det_ok
        and warmup_ok
        and threshold_ok
        and overhead_ab <= OVERHEAD_BOUND
        and overhead_ba <= OVERHEAD_BOUND
    )

    timing = {
        "screen": "censor_null_pressure_metric_diagnostic/paired_timing_summary",
        "plan_sha256": PLAN_SHA256,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "protocol": "AB_BA_median_of_N",
        "is_proof": is_proof,
        "N": n,
        "steps": steps,
        "batch": int(batch),
        "topk": int(topk),
        "device": str(device),
        "overhead_bound_frac": OVERHEAD_BOUND,
        "overhead_frac_AB": overhead_ab,
        "overhead_frac_BA": overhead_ba,
        "determinism_prefix_match": det_ok,
        "two_tier_threshold_assert_pass": threshold_ok,
        "warmup_ok_all_replicates": warmup_ok,
        "accepted": bool(accepted),
        "source_hashes": src_hashes,
        "determinism_reference_sha256s": {"uninstrumented_25": uninst_sha},
        "instrumented_sha256s": {"instrumented_25": inst_sha},
        "replicates": {
            "AB_A": ab["a_reps"],
            "AB_B": ab["b_reps"],
            "BA_A": ba["a_reps"],
            "BA_B": ba["b_reps"],
        },
        "AB": {
            "median_wall_ms_per_step_A": ab["median_wall_ms_per_step_A"],
            "median_wall_ms_per_step_B": ab["median_wall_ms_per_step_B"],
            "overhead_frac": overhead_ab,
            "determinism_prefix_match": ab["determinism_prefix_match"],
            "a_counters": ab["a_counters"],
            "b_counters": ab["b_counters"],
        },
        "BA": {
            "median_wall_ms_per_step_A": ba["median_wall_ms_per_step_A"],
            "median_wall_ms_per_step_B": ba["median_wall_ms_per_step_B"],
            "overhead_frac": overhead_ba,
            "determinism_prefix_match": ba["determinism_prefix_match"],
            "a_counters": ba["a_counters"],
            "b_counters": ba["b_counters"],
        },
        "artifact_paths": {
            "uninstrumented_25": uninst_path,
            "instrumented_25": inst_path,
            "paired_timing_summary": timing_path,
        },
        # No embedded self-hash — external/final sha is sha256_file(timing_path)
    }
    emit_json(timing, timing_path)
    if output_json:
        emit_json(timing, output_json)
    print(
        f"[pressure-metric] paired-benchmark accepted={accepted} is_proof={is_proof} "
        f"overhead_AB={overhead_ab:.4f} overhead_BA={overhead_ba:.4f} "
        f"det={det_ok} warmup_all={warmup_ok} thr_all={threshold_ok}",
        flush=True,
    )
    return 0 if accepted else 2
