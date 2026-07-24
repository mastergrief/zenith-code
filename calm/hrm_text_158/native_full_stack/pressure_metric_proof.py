"""Proof emission finalization + validation/consumption + formal facade (PLAN_v6 rev4).

Owns: finalize_paired_timing_summary, load_and_validate_paired_proof (exact
replicate key set, row VALUES, artifact file re-hash), run_formal_diagnostic.
Dependency: proof → warmup_runtime + receipt + telemetry + proof_contract.
Bound by PLAN_v6 sha 346b67d8…; fork-2 PLAN_v2 LIVE amendment.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import ARM0
from calm.hrm_text_158.native_full_stack.pressure_metric_proof_contract import (
    SOURCE_FILES,
    ContractError,
    recompute_per_index_determinism,
    sha256_file,
    validate_proof_against_live_amendment,
    validate_replicate_per_index_fields,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_receipt import (
    build_diagnostic_receipt,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    AUTHORITY_DISPATCH,
    FORMAL_BATCH,
    FORMAL_STEPS,
    FORMAL_TOPK,
    OVERHEAD_BOUND,
    PAIRED_N,
    PAIRED_STEPS,
    PARENT_SHA256,
    PLAN_SHA256,
    sanitize_receipt_for_strict_json,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_warmup_runtime import (
    run_one_diagnostic_loop,
)

REQUIRED_REPLICATE_KEYS = ("AB_A", "AB_B", "BA_A", "BA_B")


class ProofValidationError(ContractError):
    pass


def emit_json(receipt: dict, output_json: str | None) -> None:
    payload = sanitize_receipt_for_strict_json(receipt)
    if output_json:
        os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, allow_nan=False)
        print(f"[pressure-metric] wrote {output_json}", flush=True)
    else:
        print(json.dumps(payload, indent=2, allow_nan=False))


def source_file_hashes(repo_root: str) -> dict[str, str]:
    out = {}
    for rel in SOURCE_FILES:
        p = os.path.join(repo_root, rel)
        if os.path.isfile(p):
            out[rel] = sha256_file(p)
    return out


def require_cuda_proof_device(device: str, *, diagnostic_override: bool) -> bool:
    """Return is_proof eligibility. CUDA required for proof; CPU → override only."""
    dev = str(device)
    cuda_ok = dev.startswith("cuda") and torch.cuda.is_available()
    if cuda_ok:
        return True
    if diagnostic_override:
        return False  # NON-PROOF
    raise SystemExit(
        "proof-eligible paired/formal require CUDA device + availability; "
        "pass --diagnostic-override for CPU NON-PROOF"
    )

def _validate_replicate_rows(
    *,
    replicates: Mapping[str, Any],
    live_src: Mapping[str, str],
    seed0: int,
    batch: int,
    topk: int,
    device: str,
) -> None:
    if not isinstance(replicates, Mapping) or not replicates:
        raise ProofValidationError("paired-proof replicates empty/missing")
    if set(replicates.keys()) != set(REQUIRED_REPLICATE_KEYS):
        raise ProofValidationError(
            f"paired-proof replicate keys must be exactly {REQUIRED_REPLICATE_KEYS}, "
            f"got {sorted(replicates.keys())}"
        )
    for key in REQUIRED_REPLICATE_KEYS:
        rows = replicates[key]
        if not isinstance(rows, list) or len(rows) != PAIRED_N:
            raise ProofValidationError(f"paired-proof replicates[{key}] incomplete")
        expect_order, expect_flavor = key.split("_")
        for i, row in enumerate(rows):
            if not isinstance(row, Mapping):
                raise ProofValidationError(f"paired-proof replicate row not mapping: {key}[{i}]")
            for fld in (
                "rep_index", "order", "flavor", "parent_sha", "q_sha_before",
                "source_hashes", "geometry", "seed", "warmup",
                "measurements", "wall_ms_per_step",
                "two_tier_threshold_assert_pass",
            ):
                if fld not in row:
                    raise ProofValidationError(
                        f"paired-proof replicate missing {fld} in {key}[{i}]"
                    )
            if int(row["rep_index"]) != i:
                raise ProofValidationError(f"paired-proof rep_index mismatch {key}[{i}]")
            if row["order"] != expect_order or row["flavor"] != expect_flavor:
                raise ProofValidationError(
                    f"paired-proof order/flavor mismatch {key}[{i}]"
                )
            if row["parent_sha"] != PARENT_SHA256:
                raise ProofValidationError("paired-proof replicate parent_sha mismatch")
            qsha = row["q_sha_before"]
            if not isinstance(qsha, str) or not qsha:
                raise ProofValidationError("paired-proof q_sha_before empty")
            geom = row["geometry"]
            if not isinstance(geom, Mapping):
                raise ProofValidationError("paired-proof geometry not mapping")
            for gk, gv in (
                ("steps", PAIRED_STEPS),
                ("batch", batch),
                ("topk", topk),
                ("arm", ARM0),
                ("device", device),
            ):
                if geom.get(gk) != gv:
                    raise ProofValidationError(
                        f"paired-proof geometry.{gk} mismatch in {key}[{i}]"
                    )
            if int(row["seed"]) != int(seed0) + i:
                raise ProofValidationError(f"paired-proof seed sequence mismatch {key}[{i}]")
            # source hashes vs live
            src = row["source_hashes"] or {}
            for rel, h in live_src.items():
                if src.get(rel) != h:
                    raise ProofValidationError(
                        f"paired-proof source hash mismatch {rel} in {key}[{i}]"
                    )
            # warmup evidence
            wu = row["warmup"] or {}
            if wu.get("non_mutating_warmup") is not True or wu.get("hot_path_executed") is not True:
                raise ProofValidationError(
                    f"paired-proof warmup evidence fail in {key}[{i}]"
                )
            # measurements counters
            meas = row["measurements"] or {}
            for mk in ("n_flips", "q_changed_count", "credited_mass"):
                if mk not in meas or not isinstance(meas[mk], int):
                    raise ProofValidationError(
                        f"paired-proof measurements.{mk} missing in {key}[{i}]"
                    )
            wall = row["wall_ms_per_step"]
            if isinstance(wall, bool) or not isinstance(wall, (int, float)) or float(wall) <= 0:
                raise ProofValidationError(
                    f"paired-proof wall_ms_per_step invalid in {key}[{i}]"
                )
            # rev5: every INSTRUMENTED (flavor B) replicate must individually
            # carry a true threshold assert — the producer records it per-rep.
            thr = row["two_tier_threshold_assert_pass"]
            if not isinstance(thr, bool):
                raise ProofValidationError(
                    f"paired-proof two_tier_threshold_assert_pass non-bool in {key}[{i}]"
                )
            if expect_flavor == "B" and thr is not True:
                raise ProofValidationError(
                    f"paired-proof instrumented replicate threshold fail in {key}[{i}]"
                )
        # Paired A/B q_sha equality within same order+rep (AB_A[i] == AB_B[i] seeds)
        # For AB_A vs AB_B and BA_A vs BA_B at same rep_index: same seed ⇒ same q_sha_before
    for order in ("AB", "BA"):
        a_rows = replicates[f"{order}_A"]
        b_rows = replicates[f"{order}_B"]
        for i in range(PAIRED_N):
            if a_rows[i]["q_sha_before"] != b_rows[i]["q_sha_before"]:
                raise ProofValidationError(
                    f"paired-proof q_sha_before A/B mismatch at {order}[{i}]"
                )


def _median(xs: list[float]) -> float:
    """Exact mirror of the producer median (pressure_metric_benchmark)."""
    ys = sorted(xs)
    n = len(ys)
    if n == 0:
        return float("nan")
    mid = n // 2
    if n % 2:
        return float(ys[mid])
    return float(0.5 * (ys[mid - 1] + ys[mid]))


def _close(a: float, b: float) -> bool:
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-12)


def recompute_summary_from_replicates(
    replicates: Mapping[str, Any],
) -> dict[str, Any]:
    """Pure consumer-side recomputation over the exact four replicate families.

    Mirrors the producer's aggregation law: per-order medians of
    wall_ms_per_step, overhead fraction, per-index A/B counter equality
    (determinism), AND-over-instrumented threshold, AND-over-all warmup.
    """
    out: dict[str, Any] = {}
    det_all = True
    thr_all = True
    warm_all = True
    for order in ("AB", "BA"):
        a_rows = replicates[f"{order}_A"]
        b_rows = replicates[f"{order}_B"]
        med_a = _median([float(r["wall_ms_per_step"]) for r in a_rows])
        med_b = _median([float(r["wall_ms_per_step"]) for r in b_rows])
        overhead = (med_b - med_a) / max(med_a, 1e-9)
        det_order = all(
            {
                k: int(a_rows[i]["measurements"][k])
                for k in ("n_flips", "q_changed_count", "credited_mass")
            }
            == {
                k: int(b_rows[i]["measurements"][k])
                for k in ("n_flips", "q_changed_count", "credited_mass")
            }
            for i in range(len(a_rows))
        )
        det_all = det_all and det_order
        thr_all = thr_all and all(
            r["two_tier_threshold_assert_pass"] is True for r in b_rows
        )
        warm_all = warm_all and all(
            (r["warmup"] or {}).get("non_mutating_warmup") is True
            and (r["warmup"] or {}).get("hot_path_executed") is True
            for r in list(a_rows) + list(b_rows)
        )
        out[f"overhead_frac_{order}"] = float(overhead)
        out[f"median_A_{order}"] = med_a
        out[f"median_B_{order}"] = med_b
    out["determinism_prefix_match"] = bool(det_all)
    out["two_tier_threshold_assert_pass"] = bool(thr_all)
    out["warmup_ok_all_replicates"] = bool(warm_all)
    out["determinism_per_index_match"] = bool(
        recompute_per_index_determinism(replicates)
    )
    return out


def load_and_validate_paired_proof(
    *,
    path: str,
    expected_sha256: str,
    repo_root: str,
    formal_device: str,
    formal_batch: int,
    formal_topk: int,
    formal_steps: int,
    seed0: int = 0,
) -> dict[str, Any]:
    """Fail-closed proof consume — ANY mismatch → refuse formal run."""
    if not expected_sha256:
        raise ProofValidationError(
            "--paired-proof-sha256 REQUIRED for formal diagnostic"
        )
    got = sha256_file(path)
    if got != expected_sha256:
        raise ProofValidationError(
            f"paired-proof sha mismatch: got {got}, expected {expected_sha256}"
        )
    with open(path, encoding="utf-8") as f:
        proof = json.load(f)

    if proof.get("plan_sha256") != PLAN_SHA256:
        raise ProofValidationError("paired-proof plan_sha256 mismatch")
    if proof.get("authority_dispatch") != AUTHORITY_DISPATCH:
        raise ProofValidationError("paired-proof authority_dispatch mismatch")
    if proof.get("is_proof") is not True:
        raise ProofValidationError("paired-proof is_proof!=true (NON-PROOF)")
    if proof.get("accepted") is not True:
        raise ProofValidationError("paired-proof accepted!=true")
    if int(proof.get("N", -1)) != PAIRED_N:
        raise ProofValidationError(f"paired-proof N!={PAIRED_N}")
    if int(proof.get("steps", -1)) != PAIRED_STEPS:
        raise ProofValidationError(f"paired-proof steps!={PAIRED_STEPS}")
    if int(proof.get("batch", -1)) != FORMAL_BATCH:
        raise ProofValidationError("paired-proof batch mismatch")
    if int(proof.get("topk", -1)) != FORMAL_TOPK:
        raise ProofValidationError("paired-proof topk mismatch")
    if str(proof.get("device")) != str(formal_device):
        raise ProofValidationError(
            f"paired-proof device {proof.get('device')} != formal {formal_device}"
        )
    if not str(formal_device).startswith("cuda") or not torch.cuda.is_available():
        raise ProofValidationError("formal requires CUDA device + availability")
    if int(formal_batch) != FORMAL_BATCH or int(formal_topk) != FORMAL_TOPK:
        raise ProofValidationError("formal geometry must be batch=8 topk=1024")
    if int(formal_steps) != FORMAL_STEPS:
        raise ProofValidationError(f"formal steps must be {FORMAL_STEPS}")
    if proof.get("two_tier_threshold_assert_pass") is not True:
        raise ProofValidationError("paired-proof threshold assert fail")

    live = source_file_hashes(repo_root)
    replicates = proof.get("replicates") or {}
    _validate_replicate_rows(
        replicates=replicates,
        live_src=live,
        seed0=int(seed0),
        batch=FORMAL_BATCH,
        topk=FORMAL_TOPK,
        device=str(formal_device),
    )
    try:
        validate_proof_against_live_amendment(proof, repo_root=repo_root)
        validate_replicate_per_index_fields(replicates)
    except ContractError as exc:
        raise ProofValidationError(str(exc)) from exc

    # rev5: NEVER trust the summary's top-level acceptance fields — recompute
    # every one of them from the bound replicate rows and reject disagreement.
    recomputed = recompute_summary_from_replicates(replicates)
    for fld in ("overhead_frac_AB", "overhead_frac_BA"):
        try:
            summary_val = float(proof[fld])
        except (TypeError, ValueError, KeyError):
            raise ProofValidationError(f"paired-proof {fld} unparseable")
        if not _close(recomputed[fld], summary_val):
            raise ProofValidationError(
                f"paired-proof {fld} disagrees with replicate recomputation: "
                f"summary={summary_val} recomputed={recomputed[fld]}"
            )
        if recomputed[fld] > OVERHEAD_BOUND:
            raise ProofValidationError(
                f"paired-proof recomputed {fld}={recomputed[fld]:.4f} exceeds "
                f"bound {OVERHEAD_BOUND}"
            )
    if recomputed["determinism_prefix_match"] is not True:
        raise ProofValidationError(
            "paired-proof replicate counters disagree A/B — recomputed "
            "determinism_prefix_match=false (summary field untrusted)"
        )
    if recomputed.get("determinism_per_index_match") is not True:
        raise ProofValidationError(
            "paired-proof per-index determinism fail (flip/q/applied identity)"
        )
    if proof.get("determinism_prefix_match") is not True:
        raise ProofValidationError("paired-proof determinism_prefix_match!=true")
    if recomputed["two_tier_threshold_assert_pass"] is not True:
        raise ProofValidationError(
            "paired-proof recomputed threshold-all-instrumented=false"
        )
    if recomputed["warmup_ok_all_replicates"] is not True:
        raise ProofValidationError("paired-proof recomputed warmup-all=false")
    if proof.get("warmup_ok_all_replicates") is not True:
        raise ProofValidationError("paired-proof warmup_ok_all_replicates!=true")
    recomputed_accepted = (
        proof.get("is_proof") is True
        and recomputed["determinism_prefix_match"] is True
        and recomputed.get("determinism_per_index_match") is True
        and recomputed["warmup_ok_all_replicates"] is True
        and recomputed["two_tier_threshold_assert_pass"] is True
        and recomputed["overhead_frac_AB"] <= OVERHEAD_BOUND
        and recomputed["overhead_frac_BA"] <= OVERHEAD_BOUND
    )
    if recomputed_accepted is not True or proof.get("accepted") is not True:
        raise ProofValidationError(
            "paired-proof accepted disagrees with replicate recomputation"
        )

    # Artifact truth: re-hash named A/B receipt FILES
    arts = proof.get("artifact_paths") or {}
    uninst = arts.get("uninstrumented_25")
    inst = arts.get("instrumented_25")
    if not uninst or not inst or not os.path.isfile(uninst) or not os.path.isfile(inst):
        raise ProofValidationError("paired-proof A/B artifact files missing")
    uninst_sha = sha256_file(uninst)
    inst_sha = sha256_file(inst)
    det_refs = proof.get("determinism_reference_sha256s") or {}
    inst_refs = proof.get("instrumented_sha256s") or {}
    if det_refs.get("uninstrumented_25") != uninst_sha:
        raise ProofValidationError("paired-proof uninstrumented artifact sha mismatch")
    if inst_refs.get("instrumented_25") != inst_sha:
        raise ProofValidationError("paired-proof instrumented artifact sha mismatch")

    return {
        "path": str(path),
        "sha256": got,  # external/final sha of the timing summary file
        "protocol": proof.get("protocol"),
        "overhead_frac_AB": proof.get("overhead_frac_AB"),
        "overhead_frac_BA": proof.get("overhead_frac_BA"),
        "determinism_prefix_match": proof.get("determinism_prefix_match"),
        "accepted": proof.get("accepted"),
        "plan_sha256": proof.get("plan_sha256"),
        "authority_dispatch": proof.get("authority_dispatch"),
        "device": proof.get("device"),
        "N": proof.get("N"),
        "steps": proof.get("steps"),
        "batch": proof.get("batch"),
        "topk": proof.get("topk"),
        "is_proof": proof.get("is_proof"),
        "two_tier_threshold_assert_pass": proof.get("two_tier_threshold_assert_pass"),
        "source_hashes": live,
        "determinism_reference_sha256s": det_refs,
        "instrumented_sha256s": inst_refs,
        "artifact_paths": arts,
        "replicates": proof.get("replicates"),
        "recomputed_from_replicates": recomputed,
    }


def run_formal_diagnostic(
    *,
    ckpt_path: str,
    device: str,
    steps: int,
    batch: int,
    topk: int,
    seed: int,
    telemetry: bool,
    skip_probes: bool,
    paired_proof_json: str,
    paired_proof_sha256: str,
    repo_root: str,
    output_json: str | None,
) -> int:
    if skip_probes or not telemetry:
        raise SystemExit("formal mode REFUSES skip_probes / telemetry-disabled")
    require_cuda_proof_device(device, diagnostic_override=False)

    paired_proof = load_and_validate_paired_proof(
        path=paired_proof_json,
        expected_sha256=paired_proof_sha256,
        repo_root=repo_root,
        formal_device=device,
        formal_batch=batch,
        formal_topk=topk,
        formal_steps=steps,
        seed0=int(seed),
    )
    paired_ok = (
        paired_proof.get("accepted") is True
        and paired_proof.get("determinism_prefix_match") is True
        and float(paired_proof["overhead_frac_AB"]) <= OVERHEAD_BOUND
        and float(paired_proof["overhead_frac_BA"]) <= OVERHEAD_BOUND
    )

    result = run_one_diagnostic_loop(
        ckpt_path=ckpt_path,
        device=device,
        steps=int(steps),
        batch=int(batch),
        topk=int(topk),
        telemetry=True,
        skip_probes=False,
        seed=int(seed),
        warmup_enable=True,
        formal_mode=True,
    )
    if result["warmup"].get("non_mutating_warmup") is not True:
        raise SystemExit("formal run: warmup evidence failed")
    if result["store"] is None:
        raise SystemExit("formal run: store missing")

    receipt = build_diagnostic_receipt(
        store=result["store"],
        measurements=result["measurements"],
        probes=result["probes"],
        route_counters=result["route_counters"],
        banked_sha=result["banked_sha"],
        frozen_scale_sha=result["frozen_scale_sha"],
        paired_determinism_cost_ok=bool(paired_ok),
        paired_proof=paired_proof,
        expected_parent_sha=result["parent_sha"],
        steps=int(steps),
        require_probes=True,
    )
    receipt["steps"] = int(steps)
    receipt["batch"] = int(batch)
    receipt["topk"] = int(topk)
    receipt["arm"] = ARM0
    receipt["telemetry"] = True
    receipt["warmup"] = result["warmup"]
    emit_json(receipt, output_json)
    return 0
