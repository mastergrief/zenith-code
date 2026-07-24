"""Receipt schema + assembly + JSON emission for forgetting-mechanism screen.

Owns: receipt schema skeleton, measurements/probe receipt assembly, JSON emission.
Does NOT import model/runtime or the train execution loop.

PLAN_v10 identity: PLAN_SHA256 is a frozen constant = sha256 of
artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10.json, updated in the
same edit as plan content (plan JSON does NOT self-reference its own sha).
Authority: defect-cycle chain + r2 dispatch; launch_authority_dispatch is a
distinct packet-time field (not the retired v9 dispatch overload).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    retention_ok,
)
from calm.hrm_text_158.native_full_stack.forgetting_laws import (
    H_TRAJECTORY_EVERY,
    entropy_bits,
)
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    build_phase1_probe_sets,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    summarize_demand_totals,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    count_censored_active_episodes,
    lifetime_censored_frac,
    lifetime_quantiles,
    never_convert_metrics,
)


def compact_r1_surface_from_store(store: Any) -> dict[str, Any]:
    """Compact R1 measurements from accepted DeviceLifecycleStore / PressureTelemetryStore."""
    return {
        "demand": summarize_demand_totals(list(store.per_step_ratios)),
        "deferred_survival": dict(store.survival_summary()),
    }


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().cpu().contiguous().numpy().tobytes()
    ).hexdigest()

# Frozen constant = sha256(PLAN_v10.json); updated same-edit as plan content.
# Plan artifact must not embed this value (no self-reference).
PLAN_PATH = "artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10.json"
PLAN_SHA256 = "2cb92e50bc40f5def493864189968fd62cc21a888ad7f896f2e723c1a194805c"
# Retired PLAN_v9 identity — reject if still present on live receipts.
PLAN_V9_SHA256 = "07a02afff92cef7b2c6cee46a761a1e46b6b3422df911f8b4d4f63d41157e7a5"
AUTHORITY_DISPATCH_V9 = "1784812148229-f466bc29"
# Current v10 implement authority (defect-cycle r2 dispatch).
AUTHORITY_DISPATCH = "1784893417123-0300fdf7"
DEFECT_CYCLE_AUTHORITY = "1784892185413-a4f0e9bb"
# Packet-time launch authority — distinct field; filled at GPU packet time only.
LAUNCH_AUTHORITY_DISPATCH = None
COMMIT_SURFACE_FILES = [
    "artifacts/acc_entropy/forgetting_mechanism_screen_PLAN_v10.json",
    "calm/hrm_text_158/native_full_stack/forgetting_mechanism_screen_reducers.py",
    "calm/hrm_text_158/native_full_stack/forgetting_screen_v10_contract.py",
    "scripts/hrm_text_158_forgetting_mechanism_screen.py",
    "calm/hrm_text_158/tests/test_forgetting_screen_v10_contract.py",
    "calm/hrm_text_158/tests/test_forgetting_mechanism_screen_reducers.py",
    "calm/hrm_text_158/native_full_stack/screen_run_loop.py",
    "calm/hrm_text_158/native_full_stack/screen_receipt_output.py",
]


def _receipt_schema_skeleton() -> dict:
    return {
        "screen": "forgetting_mechanism_screen/v1",
        "plan_sha256": PLAN_SHA256,
        "plan_path": PLAN_PATH,
        "authority_dispatch": AUTHORITY_DISPATCH,
        "defect_cycle_authority": DEFECT_CYCLE_AUTHORITY,
        "launch_authority_dispatch": LAUNCH_AUTHORITY_DISPATCH,
        "fixed_qscale_credit_seam": True,
        "begin_credit_step_required": True,
        "dW_formula": "grad_output.reshape(-1,out).T @ act.reshape(-1,in)",
        "commit_surface_files": list(COMMIT_SURFACE_FILES),
        "measurements": {
            "H_bits_per_weight": None,
            "H_trajectory": [],
            "n_flips": None,
            "lifetime_censored_frac": None,
            "q_changed_count": None,
        },
        "probes": {
            "acquisition_n": 64,
            "retention_n": 64,
            "acquisition_selection_sha256": None,
            "identity_selection_sha256": None,
            "acq_step0_count": None,
            "acq_final_count": None,
            "acq_delta_count": None,
            "retention_step0_count": None,
            "retention_final_count": None,
            "retention_ok": None,
            "excluded_hit_count": None,
        },
        "banked_sha": {},
        "route_counters": {},
        "limits": [
            "design-family screen only — no forgetting-law ship",
            "no sub-2 claim",
        ],
    }


def emit_receipt_json(receipt: dict, output_json: str | None) -> None:
    from calm.hrm_text_158.native_full_stack.phase_receipt_contracts import (
        sanitize_receipt_for_strict_json,
    )

    payload = sanitize_receipt_for_strict_json(receipt)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"[forget-mech] wrote {output_json}", flush=True)
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


def run_schema_only(args: argparse.Namespace) -> int:
    """CPU schema smoke — probe pins, no ckpt load."""
    receipt = _receipt_schema_skeleton()
    receipt["schema_only"] = True
    receipt["cli"] = {
        "steps": int(args.steps),
        "batch": int(args.batch),
        "device": str(args.device),
        "arm": str(args.arm),
        "aggregate_phase1": False,
        "h_trajectory_every": H_TRAJECTORY_EVERY,
    }
    probes = build_phase1_probe_sets()
    receipt["probes"].update(
        {
            "acquisition_selection_sha256": probes["acquisition_selection_sha256"],
            "identity_selection_sha256": probes["identity_selection_sha256"],
            "math_a0_parent_support_hash": probes["math_a0_parent_support_hash"],
            "identity_parent_support_hash": probes["identity_parent_support_hash"],
            "acquisition_n": probes["acquisition_n"],
            "retention_n": probes["retention_n"],
            "disjoint_acq_ret_math": True,
        }
    )
    emit_receipt_json(receipt, args.output_json)
    return 0


def assemble_arm_receipt(
    *,
    args: argparse.Namespace,
    device: str,
    sha_before: str,
    scale_sha_before: str,
    q_sha_before: str,
    frozen_scales: dict[str, torch.Tensor],
    q_levels: dict[str, torch.Tensor],
    ckpt_path: str,
    probe_sets: dict[str, Any],
    acq_step0,
    ret_step0,
    acq_final,
    ret_final,
    loop_out: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the per-arm/Phase-0 screen receipt from loop + probe results."""
    acc = loop_out["acc"]
    episode_start = loop_out["episode_start"]
    flip_count = loop_out["flip_count"]
    lifetimes = loop_out["lifetimes"]
    credited_mass = loop_out["credited_mass"]
    n_flips = loop_out["n_flips"]
    q_changed_count = loop_out["q_changed_count"]
    n_applied_drains = loop_out["n_applied_drains"]
    excluded_hit_count = loop_out["excluded_hit_count"]
    H_trajectory = list(loop_out["H_trajectory"])
    train_route_counters = loop_out["train_route_counters"]

    sha_after = _sha256_file(ckpt_path)
    scale_sha_after = hashlib.sha256(
        b"".join(_sha_tensor(frozen_scales[n]).encode() for n in sorted(frozen_scales))
    ).hexdigest()
    q_sha_after = hashlib.sha256(
        b"".join(_sha_tensor(q_levels[n]).encode() for n in sorted(q_levels))
    ).hexdigest()

    final_acc = torch.cat([a.flatten() for a in acc.values()])
    final_ep = torch.cat([e.flatten() for e in episode_start.values()])
    final_fc = torch.cat([c.flatten() for c in flip_count.values()])
    n_censored = count_censored_active_episodes(final_acc, final_ep)
    lcf = lifetime_censored_frac(n_flips, n_censored)
    b_metrics = never_convert_metrics(final_acc, final_fc)
    quant = lifetime_quantiles(lifetimes)
    H = entropy_bits(final_acc)
    # Ensure final is present even if steps < every (should_record covers final).
    if not H_trajectory or H_trajectory[-1]["step"] != int(args.steps):
        H_trajectory.append(
            {
                "step": int(args.steps),
                "H_bits_per_weight": float(H),
                "support": "pooled_named_parameter_acc_flatten",
                "denominator": "acc.numel()",
                "estimator": "shannon_unique_counts",
            }
        )

    probes_out: dict[str, Any] = {
        "acquisition_n": probe_sets["acquisition_n"],
        "retention_n": probe_sets["retention_n"],
        "acquisition_selection_sha256": probe_sets["acquisition_selection_sha256"],
        "identity_selection_sha256": probe_sets["identity_selection_sha256"],
        "math_a0_parent_support_hash": probe_sets["math_a0_parent_support_hash"],
        "identity_parent_support_hash": probe_sets["identity_parent_support_hash"],
        "excluded_hit_count": int(excluded_hit_count),
        "skipped": bool(args.skip_probes),
    }
    if not args.skip_probes:
        assert acq_step0 is not None and acq_final is not None
        assert ret_step0 is not None and ret_final is not None
        acq_delta = int(acq_final) - int(acq_step0)
        ret_ok = retention_ok(final_count=int(ret_final), step0_count=int(ret_step0))
        probes_out.update(
            {
                "acq_step0_count": int(acq_step0),
                "acq_final_count": int(acq_final),
                "acq_delta_count": int(acq_delta),
                "retention_step0_count": int(ret_step0),
                "retention_final_count": int(ret_final),
                "retention_ok": bool(ret_ok),
            }
        )

    receipt = _receipt_schema_skeleton()
    receipt.update(
        {
            "schema_only": False,
            "correctness_smoke": bool(args.correctness_smoke),
            "arm": str(args.arm),
            "steps": int(args.steps),
            "batch": int(args.batch),
            "topk": int(args.topk),
            "device": device,
            "banked_sha": {
                "before": sha_before,
                "after": sha_after,
                "match": sha_before == sha_after,
            },
            "frozen_scale_sha": {
                "before": scale_sha_before,
                "after": scale_sha_after,
                "match": scale_sha_before == scale_sha_after,
            },
            "q_sha": {"before": q_sha_before, "after": q_sha_after},
            "route_counters": dict(train_route_counters),
            "measurements": {
                "n_flips": n_flips,
                "n_applied_drains": n_applied_drains,
                "q_changed_count": q_changed_count,
                "credited_mass": credited_mass,
                "lifetime_censored_frac": lcf,
                "p50_flip_lifetime": quant.get("p50"),
                "never_convert_frac": b_metrics["never_convert_frac"],
                "H_bits_per_weight": H,
                "H_trajectory": H_trajectory,
                "H_estimator": {
                    "name": "pooled_empirical_shannon",
                    "support": "all named parameter tensors concatenated flatten",
                    "denominator": "acc.numel() across pooled weights",
                    "units": "bits_per_weight",
                    "trajectory_every": H_TRAJECTORY_EVERY,
                },
            },
            "probes": probes_out,
            "asserts": {
                "banked_sha_stable": sha_before == sha_after,
                "frozen_scale_sha_stable": scale_sha_before == scale_sha_after,
                "dynamic_bitlinear_unused": (
                    int(train_route_counters.get("n_bitlinear_dynamic_forwards", -1))
                    == 0
                ),
            },
        }
    )
    # Compact R1 surface from live DeviceLifecycleStore observer (required under v10).
    pt = loop_out.get("pressure_telemetry")
    if pt is not None:
        r1 = compact_r1_surface_from_store(pt)
        receipt["measurements"]["demand"] = r1["demand"]
        receipt["measurements"]["deferred_survival"] = r1["deferred_survival"]
    return receipt
