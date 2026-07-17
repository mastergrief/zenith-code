"""Inversion/calibration/weight/isolation proofs (D2c3 S2)."""
from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_integrity_proofs import (
    INTEGRITY,
    assert_zero_cross_arm_storage_overlap,
    hash_arm_state_manifest,
    within_state_alias_topology,
)

INVERT_DIR_FIELDS = ("applied_directions", "replay_veto_directions")
PLAN_V6_MUTABLE_ARMS = ("capture_disposable", "calibration_shadow", "prod", "inv", "noop")


class ArmProofError(RuntimeError):
    pass


def _tensor_content_sha(t: Any) -> str:
    arr = t.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(f"{arr.dtype}|{tuple(arr.shape)}|".encode())
    h.update(arr.numpy().tobytes())
    return h.hexdigest()


def hash_current_weights_tensors(weights: Mapping[str, Any]) -> str:
    h = hashlib.sha256()
    for key in sorted(map(str, weights)):
        arr = weights[key].detach().cpu().contiguous()
        h.update(f"{key}|{arr.dtype}|{tuple(arr.shape)}|".encode())
        h.update(arr.numpy().tobytes())
    return h.hexdigest()


def canonical_invert_plans_v4(plans_by_key: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, plan in plans_by_key.items():
        kwargs = {}
        for name in INVERT_DIR_FIELDS:
            if not hasattr(plan, name):
                raise ArmProofError(f"invert_missing_direction_field:{name}")
            t = getattr(plan, name)
            if t is None or not hasattr(t, "contiguous"):
                raise ArmProofError(f"invert_malformed_direction_field:{name}")
            kwargs[name] = (-t).contiguous()
        if hasattr(plan, "__dataclass_fields__"):
            out[key] = replace(plan, **kwargs)
        else:
            data = dict(getattr(plan, "__dict__", {}))
            data.update(kwargs)
            try:
                out[key] = type(plan)(**data)
            except TypeError as exc:
                raise ArmProofError(f"invert_rebuild_failed:{exc}") from exc
    return out


def calibrate_capture_vs_public(capture_states: Mapping[str, Any], public_states: Mapping[str, Any]) -> dict[str, Any]:
    if set(capture_states) != set(public_states):
        return {"ok": False, "pass": False, "reason": "key_mismatch"}
    per_key: dict[str, Any] = {}
    for key in sorted(capture_states):
        c, p = capture_states[key], public_states[key]
        cq = c.q_levels.detach().cpu().contiguous()
        pq = p.q_levels.detach().cpu().contiguous()
        ca = getattr(c, "exact_accumulator_shadow", None)
        pa = getattr(p, "exact_accumulator_shadow", None)
        q_ok = tuple(cq.shape) == tuple(pq.shape) and cq.dtype == pq.dtype and bool((cq == pq).all().item())
        if ca is None or pa is None:
            acc_ok = ca is pa
            ca_sha = pa_sha = None
        else:
            ca_t = ca.detach().cpu().contiguous()
            pa_t = pa.detach().cpu().contiguous()
            acc_ok = (
                tuple(ca_t.shape) == tuple(pa_t.shape)
                and ca_t.dtype == pa_t.dtype
                and bool((ca_t == pa_t).all().item())
            )
            ca_sha, pa_sha = _tensor_content_sha(ca_t), _tensor_content_sha(pa_t)
        cq_sha, pq_sha = _tensor_content_sha(cq), _tensor_content_sha(pq)
        content_ok = q_ok and acc_ok and cq_sha == pq_sha and ca_sha == pa_sha
        per_key[key] = {
            "q_ok": q_ok, "acc_ok": acc_ok, "content_ok": content_ok,
            "q_sha_capture": cq_sha, "q_sha_public": pq_sha,
            "acc_sha_capture": ca_sha, "acc_sha_public": pa_sha,
        }
        if not content_ok:
            return {"ok": False, "pass": False, "reason": "state_mismatch", "per_key": per_key}
    return {"ok": True, "pass": True, "per_key": per_key}


def arm_hash_map(arms: Mapping[str, Any], keys: Sequence[str]) -> dict[str, str]:
    return {k: hash_arm_state_manifest(arms[k]) for k in keys}


def mutable_arms_for_isolation(arms: Mapping[str, Any]) -> dict[str, Any]:
    missing = [k for k in PLAN_V6_MUTABLE_ARMS if k not in arms]
    if missing:
        raise ArmProofError(f"isolation_arms_missing:{missing}")
    return {k: arms[k] for k in PLAN_V6_MUTABLE_ARMS}


def run_isolation_sentinel_checkpoint(arms, *, base=None, label=""):
    overlap = assert_zero_cross_arm_storage_overlap(arms, base=base)
    return {
        "label": label,
        "overlap": overlap,
        "topology": {k: within_state_alias_topology(v) for k, v in arms.items()},
        "hashes": {k: hash_arm_state_manifest(v) for k, v in arms.items()},
        "classifier_on_fail": INTEGRITY,
    }


__all__ = [
    "ArmProofError",
    "INVERT_DIR_FIELDS",
    "PLAN_V6_MUTABLE_ARMS",
    "arm_hash_map",
    "calibrate_capture_vs_public",
    "canonical_invert_plans_v4",
    "hash_current_weights_tensors",
    "mutable_arms_for_isolation",
    "run_isolation_sentinel_checkpoint",
]
