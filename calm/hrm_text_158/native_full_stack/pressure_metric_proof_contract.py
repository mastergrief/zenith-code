"""LIVE proof-contract / amendment validator (fork-2 shared GPU baseline).

Owns: LIVE amendment JSON load + sha bind, shared-GPU baseline refusal of old
CPU proofs, R-i2 denominator contract, per-index determinism field helpers /
consumer recompute validation. Consumed by pressure_metric_proof (thin facade).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    OVERHEAD_BOUND,
)

LIVE_AMENDMENT_RELPATH = (
    "artifacts/acc_entropy/pressure_metric_proof_contract_amendment_fork2_LIVE.json"
)
LIVE_AMENDMENT_SCHEMA = "pressure_metric_proof_contract_amendment_fork2_LIVE/v1"
BASELINE_VARIABLE = "shared_A_B_GPU_selection_entrypoint"
DENOMINATOR_LAW = "same_run_shared_baseline_uninstrumented_A_median_wall_per_order"

SOURCE_FILES = (
    "calm/hrm_text_158/native_full_stack/pressure_metric_telemetry.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_lifecycle.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_classifier.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_readiness.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_receipt.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_warmup_runtime.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_proof.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_proof_contract.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_benchmark.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_loop_bridge.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_lifecycle_derisk.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_gpu_selection_derisk.py",
    "calm/hrm_text_158/native_full_stack/pressure_metric_fork2_geometry.py",
    "calm/hrm_text_158/native_full_stack/screen_execution_loop.py",
    "calm/hrm_text_158/native_full_stack/screen_model_runtime.py",
    "scripts/hrm_text_158_censor_null_pressure_metric_diagnostic.py",
    "scripts/hrm_text_158_fork2_integration_additive_smoke.py",
)

PER_INDEX_HASH_FIELDS = (
    "flip_count_sha256",
    "q_final_sha256",
    "applied_identity_sha256",
)


class ContractError(SystemExit):
    """Fail-closed proof-contract / amendment error."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_tensor_sha256(t: torch.Tensor) -> str:
    arr = t.detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
    return sha256_bytes(arr.tobytes())


def canonical_arm_dict_sha256(arms: Mapping[str, torch.Tensor]) -> str:
    """Hash named tensors in sorted arm-name order (compact; never emit arrays)."""
    h = hashlib.sha256()
    for name in sorted(arms.keys()):
        h.update(name.encode("utf-8"))
        h.update(b"\0")
        arr = arms[name].detach().to(dtype=torch.int64, device="cpu").contiguous().numpy()
        h.update(arr.tobytes())
    return h.hexdigest()


def applied_identity_sha256_from_frames(frames: Sequence[bytes]) -> str:
    """Ordered per-step selection identity (step-framed flat-idx sequence)."""
    h = hashlib.sha256()
    for fr in frames:
        h.update(fr)
    return h.hexdigest()


def per_index_fields_from_loop_out(loop_out: Mapping[str, Any]) -> dict[str, str]:
    return {
        "flip_count_sha256": canonical_arm_dict_sha256(loop_out["flip_count"]),
        "q_final_sha256": canonical_arm_dict_sha256(loop_out["q_levels"]),
        "applied_identity_sha256": applied_identity_sha256_from_frames(
            list(loop_out.get("selection_frames") or [])
        ),
    }


def compare_per_index(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> dict[str, bool]:
    return {
        "flip_count_equal": a.get("flip_count_sha256") == b.get("flip_count_sha256"),
        "q_final_equal": a.get("q_final_sha256") == b.get("q_final_sha256"),
        "applied_identity_equal": a.get("applied_identity_sha256")
        == b.get("applied_identity_sha256"),
    }


def per_index_all_equal(flags: Mapping[str, bool]) -> bool:
    return all(
        bool(flags.get(k))
        for k in ("flip_count_equal", "q_final_equal", "applied_identity_equal")
    )


def load_live_amendment(repo_root: str) -> tuple[dict[str, Any], str]:
    path = os.path.join(repo_root, LIVE_AMENDMENT_RELPATH)
    if not os.path.isfile(path):
        raise ContractError(f"LIVE amendment missing: {LIVE_AMENDMENT_RELPATH}")
    with open(path, encoding="utf-8") as f:
        doc = json.load(f)
    digest = sha256_file(path)
    validate_live_amendment_doc(doc)
    return doc, digest


def validate_live_amendment_doc(doc: Mapping[str, Any]) -> None:
    if doc.get("schema") != LIVE_AMENDMENT_SCHEMA:
        raise ContractError("LIVE amendment schema mismatch")
    if doc.get("status") != "LIVE":
        raise ContractError("LIVE amendment status!=LIVE")
    if doc.get("variable") != BASELINE_VARIABLE:
        raise ContractError("LIVE amendment baseline variable mismatch")
    den = doc.get("overhead_denominator") or {}
    if den.get("law") != DENOMINATOR_LAW:
        raise ContractError("LIVE amendment denominator law mismatch (R-i2)")
    if float(den.get("bound", -1)) != float(OVERHEAD_BOUND):
        raise ContractError("LIVE amendment overhead bound mismatch")
    if doc.get("old_cpu_baseline_proof_semantics") != "REFUSED":
        raise ContractError("LIVE amendment must REFUSE old CPU baseline proofs")


def refuse_old_cpu_baseline_proof(proof: Mapping[str, Any]) -> None:
    """Old CPU-baseline proofs are not consumable under the LIVE amendment."""
    baseline = proof.get("paired_baseline") or proof.get("baseline") or {}
    if isinstance(baseline, str):
        tag = baseline
    else:
        tag = str(baseline.get("variable") or baseline.get("id") or "")
    if "cpu" in tag.lower() and "shared" not in tag.lower():
        raise ContractError("old CPU-baseline proof REFUSED by LIVE amendment")
    if proof.get("shared_gpu_baseline") is False:
        raise ContractError("proof marked shared_gpu_baseline=false — REFUSED")
    # Explicit legacy marker
    if proof.get("legacy_cpu_selection_baseline") is True:
        raise ContractError("legacy_cpu_selection_baseline proof REFUSED")


def bind_amendment_into_summary(
    summary: dict[str, Any], *, amendment_sha256: str, amendment: Mapping[str, Any]
) -> None:
    summary["proof_contract_amendment_sha256"] = str(amendment_sha256)
    summary["proof_contract_amendment_schema"] = amendment.get("schema")
    summary["paired_baseline"] = {
        "variable": amendment.get("variable"),
        "registered_tiebreak": amendment.get("registered_tiebreak"),
        "new_paired_baseline": amendment.get("new_paired_baseline"),
    }
    summary["overhead_denominator"] = dict(amendment.get("overhead_denominator") or {})
    summary["shared_gpu_baseline"] = True


def validate_proof_against_live_amendment(
    proof: Mapping[str, Any], *, repo_root: str
) -> dict[str, Any]:
    amendment, digest = load_live_amendment(repo_root)
    refuse_old_cpu_baseline_proof(proof)
    got = proof.get("proof_contract_amendment_sha256")
    if got != digest:
        raise ContractError(
            f"proof amendment sha mismatch: got {got}, live {digest}"
        )
    if proof.get("shared_gpu_baseline") is not True:
        raise ContractError("proof missing shared_gpu_baseline=true")
    den = proof.get("overhead_denominator") or {}
    if den.get("law") != DENOMINATOR_LAW:
        raise ContractError("proof denominator law mismatch (R-i2)")
    if float(den.get("bound", -1)) != float(OVERHEAD_BOUND):
        raise ContractError("proof overhead bound mismatch")
    return {"amendment": amendment, "amendment_sha256": digest}


def validate_replicate_per_index_fields(replicates: Mapping[str, Any]) -> None:
    """Consumer: require bound per-index fields; recompute A/B equality."""
    for order in ("AB", "BA"):
        a_rows = replicates.get(f"{order}_A") or []
        b_rows = replicates.get(f"{order}_B") or []
        if len(a_rows) != len(b_rows):
            raise ContractError(f"per-index replicate length mismatch {order}")
        for i, (a, b) in enumerate(zip(a_rows, b_rows)):
            for fld in PER_INDEX_HASH_FIELDS:
                if not isinstance(a.get(fld), str) or not a.get(fld):
                    raise ContractError(f"missing {fld} in {order}_A[{i}]")
                if not isinstance(b.get(fld), str) or not b.get(fld):
                    raise ContractError(f"missing {fld} in {order}_B[{i}]")
            flags = compare_per_index(a, b)
            # Prefer producer-recorded booleans when present; else recompute.
            for bk, rk in (
                ("flip_count_equal", "flip_count_equal"),
                ("q_final_equal", "q_final_equal"),
                ("applied_identity_equal", "applied_identity_equal"),
            ):
                recorded = a.get(bk)
                if recorded is not None and bool(recorded) != flags[rk]:
                    raise ContractError(
                        f"recorded {bk} disagrees with hash recompute at {order}[{i}]"
                    )
                recorded_b = b.get(bk)
                if recorded_b is not None and bool(recorded_b) != flags[rk]:
                    raise ContractError(
                        f"recorded {bk} disagrees with hash recompute at {order}_B[{i}]"
                    )
            if not per_index_all_equal(flags):
                raise ContractError(
                    f"per-index determinism fail at {order}[{i}]: {flags}"
                )


def recompute_per_index_determinism(replicates: Mapping[str, Any]) -> bool:
    try:
        validate_replicate_per_index_fields(replicates)
        return True
    except ContractError:
        return False
