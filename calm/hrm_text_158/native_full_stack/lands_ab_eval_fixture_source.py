"""Frozen fixture / descriptor / recarry / source-pin readers (IMPLEMENT_v3 seam b).

IO only for pinned read-only artifacts. No measurement / apply.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    weighted_grad_from_captures,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    FIXTURE_RECIPE_NAME,
    PARITY_FIXTURE_DESCRIPTOR_SHA256,
    RANK_SPEC_DIGEST_EXPECTED,
    RECARRY_RECEIPT_PATH,
    RECARRY_RECEIPT_SHA256,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    require_canonical_rank_spec,
    tensor_sha256,
)
from calm.hrm_text_158.native_full_stack.optimizer_credit_state import (
    OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256,
)
from calm.hrm_text_158.native_full_stack.recarry_measurement_evidence import (
    _load_3c_harness,
)


def repo_root_from_here() -> Path:
    return Path(__file__).resolve().parents[3]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



# Production source pins (TSA/BDL refreshed under DW_INJECTIVE PLAN_v7 pin addendum)
DEFAULT_SOURCE_PINS: dict[str, str] = {
    "artifacts/acc_entropy/optimizer_credit_state_projected_moves_recarry_measurement_receipt_v1.json": "783f279986ebaa9bd7d170b5996146a319e9c8f1980939ec8ee49ac4b5d5db2f",
    "artifacts/acc_entropy/optimizer_credit_state_sparse_live_carrier_production_landing_FINAL_SNAPSHOT_v9.json": "e2d0c18dcf91d3fd13a197c61b110ad479bc2db2272623ee17d35f89ca4f203a",
    "artifacts/acc_entropy/optimizer_credit_state_sparse_live_carrier_production_landing_GPU_SMOKE_packet_v4.json": "191176e2afa39d6148d7388ed504216629636ddad66df9271849ae105739253a",
    "bin/watch-wrap": "a19f1c5fe88fb3dcbf00ab442047576708f75272210e9a0cc94ed9369bf45d4b",
    "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py": "ae7a213cb8581153a756c131cf03786f7a04462b6508171b7e1e92a5f5fe3707",
    "calm/hrm_text_158/native_full_stack/recarry_measurement_evidence.py": "b82d30ffeee7121a3a07c19d9f6345173e7d6f9effbaeccfb90bf3907a06e1b0",
    "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py": "3c6c3db2ea25233f0c842fca47d6539ed7376ee7a2bf969062d0b39bac958fa0",
}

def verify_source_pins(
    pins: Mapping[str, str],
    *,
    repo_root: Path | None = None,
    require_exact_default_set: bool = True,
) -> dict[str, Any]:
    """Re-hash pins; scope_creep if incomplete/extra/substituted expected or live drift.

    When require_exact_default_set (science path default), pins must equal
    DEFAULT_SOURCE_PINS key+expected-hash exactly before live rehash.
    """
    root = repo_root or repo_root_from_here()
    results: dict[str, dict[str, str]] = {}
    drift = False
    if require_exact_default_set:
        if set(pins.keys()) != set(DEFAULT_SOURCE_PINS.keys()):
            missing = sorted(set(DEFAULT_SOURCE_PINS) - set(pins))
            extra = sorted(set(pins) - set(DEFAULT_SOURCE_PINS))
            drift = True
            results["__set__"] = {
                "status": "SET_MISMATCH",
                "expected": f"missing={missing} extra={extra}",
                "actual": ",".join(sorted(pins)),
            }
        for rel, expected in sorted(DEFAULT_SOURCE_PINS.items()):
            got = pins.get(rel)
            if got is None:
                results[rel] = {"status": "MISSING_PIN", "expected": expected, "actual": ""}
                drift = True
                continue
            if got != expected:
                results[rel] = {
                    "status": "SUBSTITUTED_EXPECTED_HASH",
                    "expected": expected,
                    "actual": str(got),
                }
                drift = True
                continue
            path = root / rel
            if not path.exists():
                results[rel] = {"status": "MISSING", "expected": expected, "actual": ""}
                drift = True
                continue
            actual = sha256_file(path)
            if actual != expected:
                results[rel] = {"status": "DRIFT", "expected": expected, "actual": actual}
                drift = True
            else:
                results[rel] = {"status": "OK", "expected": expected, "actual": actual}
        return {"pins": results, "scope_creep": drift, "required_set": "DEFAULT_SOURCE_PINS"}

    for rel, expected in sorted(pins.items()):
        path = root / rel
        if not path.exists():
            results[rel] = {"status": "MISSING", "expected": expected, "actual": ""}
            drift = True
            continue
        actual = sha256_file(path)
        if actual != expected:
            results[rel] = {"status": "DRIFT", "expected": expected, "actual": actual}
            drift = True
        else:
            results[rel] = {"status": "OK", "expected": expected, "actual": actual}
    return {"pins": results, "scope_creep": drift}


def load_seed158_static_fixture() -> dict[str, Any]:
    if (
        OPTIMIZER_CREDIT_STATE_3C_C1_PARITY_FIXTURE_DESCRIPTOR_SHA256
        != PARITY_FIXTURE_DESCRIPTOR_SHA256
    ):
        raise ValueError("parity_fixture_descriptor_sha256_pin_mismatch")
    harness = _load_3c_harness()
    captures, q_flat, weight_shape, _eligible, model = harness._dry_run_fixture()
    weighted_grad = weighted_grad_from_captures(
        captures["inputs"], captures["grad_outputs"], weight_shape=weight_shape
    )
    q_levels = q_flat.reshape(weight_shape).to(torch.int8).contiguous()
    rank_spec = require_canonical_rank_spec()
    return {
        "fixture_recipe_name": FIXTURE_RECIPE_NAME,
        "parity_fixture_descriptor_sha256": PARITY_FIXTURE_DESCRIPTOR_SHA256,
        "weighted_grad": weighted_grad,
        "q_levels": q_levels,
        "weight_shape": tuple(int(x) for x in weight_shape),
        "model": model,
        "rank_spec": rank_spec,
        "weighted_grad_sha256": tensor_sha256(weighted_grad),
        "q_levels_sha256": tensor_sha256(q_levels),
    }


def verify_recarry_receipt_ro(repo_root: Path | None = None) -> dict[str, Any]:
    root = repo_root or repo_root_from_here()
    path = root / RECARRY_RECEIPT_PATH
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != RECARRY_RECEIPT_SHA256:
        raise ValueError(
            f"recarry_receipt_pin_mismatch: expected={RECARRY_RECEIPT_SHA256} actual={digest}"
        )
    data = json.loads(raw.decode("utf-8"))
    if data.get("fixture_recipe_name") != FIXTURE_RECIPE_NAME:
        raise ValueError("recarry_receipt_fixture_recipe_mismatch")
    if data.get("parity_fixture_descriptor_sha256") != PARITY_FIXTURE_DESCRIPTOR_SHA256:
        raise ValueError("recarry_receipt_descriptor_mismatch")
    if data.get("rank_spec_to_live_dict_canonical_sha256") != RANK_SPEC_DIGEST_EXPECTED:
        raise ValueError("recarry_receipt_rank_spec_mismatch")
    return {
        "path": str(RECARRY_RECEIPT_PATH),
        "sha256": digest,
        "events_equal": bool(data.get("events_equal")),
        "compositional_reduction_holds": bool(data.get("compositional_reduction_holds")),
        "fused_event_count": int(data.get("fused_event_count", 0)),
        "weighted_grad_sha256": data.get("weighted_grad_sha256"),
    }
