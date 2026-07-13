"""Zero-seed forget + A1 shared resume serialize/load (exact vs zero-strip).

A1 control integrity (load-bearing):
- E/R0/RW share ONE serializer/loader/resume orchestration.
- Sole policy delta = EXACT_PRESERVE (E) vs ZERO_STRIP (R0/RW).
- E: serialize → discard live → load; missing/mismatched exact state → CONTROL_INVALID.
- R0/RW artifacts must physically omit/zero exact shadow + bounded payload + backlog.
- E artifact lives under E arm root and is inaccessible to R0/RW loaders.
"""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
    backlog_content_sha256,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY,
    ArmId,
    FailClosedClass,
    ResumePolicy,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_arm_ops import (
    rehydrate_z_zeros,
)


ARTIFACT_FILENAME = "resume_artifact.json"
EXACT_SHADOW_KEY = "exact_accumulator_shadow_i16"
BOUNDED_PAYLOAD_KEY = "bounded_accumulator_payload"
BACKLOG_KEY = "deferred_backlog"


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha_tensor(t: torch.Tensor) -> str:
    return _sha_bytes(t.detach().cpu().contiguous().numpy().tobytes())


def _canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def empty_deferred_backlog() -> dict[str, dict[int, dict[str, int]]]:
    return {}


def assert_pre_W_zeroed_identity(
    *,
    r0_acc_sha: str,
    rw_acc_sha: str,
    r0_backlog_sha: str,
    rw_backlog_sha: str,
) -> str:
    if r0_acc_sha != rw_acc_sha or r0_backlog_sha != rw_backlog_sha:
        raise AssertionError(PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY)
    return PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY


def apply_zero_seed_forget_state(state: BoundedDeltaTensorState) -> BoundedDeltaTensorState:
    """Strip exact shadow + zero bounded payload (via rehydrate_z_zeros)."""

    return rehydrate_z_zeros(state)


def apply_zero_seed_forget(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    deferred_backlog: Mapping[str, Any] | None,
) -> tuple[dict[str, BoundedDeltaTensorState], dict[str, dict[int, dict[str, int]]]]:
    next_states = {
        key: apply_zero_seed_forget_state(state) for key, state in sorted(tensor_states.items())
    }
    _ = deferred_backlog  # discarded — authoritative post-forget backlog is empty
    return next_states, empty_deferred_backlog()


def _serialize_bounded_payload(state: BoundedDeltaTensorState) -> dict[str, Any]:
    b = state.bounded_accumulator
    return {
        "logical_shape": list(b.logical_shape),
        "hot_exact_indices": list(b.hot_exact_indices),
        "cold_exception_indices": list(b.cold_exception_indices),
        "cold_default_value": int(b.cold_default_value),
        # compact digest of packed values for absence/presence proofs
        "payload_sha256": _sha_bytes(_canonical_json({
            "hot": list(b.hot_exact_indices),
            "cold": list(b.cold_exception_indices),
            "default": int(b.cold_default_value),
            "shape": list(b.logical_shape),
        }).encode("utf-8")),
    }


def _serialize_state_exact(state: BoundedDeltaTensorState) -> dict[str, Any]:
    if state.exact_accumulator_shadow is None:
        raise ValueError(
            f"{FailClosedClass.CONTROL_INVALID.value}: E exact-preserve requires "
            "exact_accumulator_shadow at serialize"
        )
    shadow = state.exact_accumulator_shadow.detach().cpu().contiguous()
    return {
        "state_key": state.state_key,
        "q_levels_i8": state.q_levels.detach().cpu().contiguous().flatten().tolist(),
        "q_shape": list(state.q_levels.shape),
        "frozen_scale": float(state.frozen_scale.detach().cpu().reshape(()).item()),
        EXACT_SHADOW_KEY: shadow.flatten().tolist(),
        "exact_shadow_sha256": _sha_tensor(shadow),
        BOUNDED_PAYLOAD_KEY: _serialize_bounded_payload(state),
        "q_sha256": _sha_tensor(state.q_levels),
    }


def _serialize_state_zero_strip(state: BoundedDeltaTensorState) -> dict[str, Any]:
    # Physically omit exact shadow field; bounded payload forced to zeroed encoding.
    zeroed = rehydrate_z_zeros(state)
    return {
        "state_key": state.state_key,
        "q_levels_i8": zeroed.q_levels.detach().cpu().contiguous().flatten().tolist(),
        "q_shape": list(zeroed.q_levels.shape),
        "frozen_scale": float(zeroed.frozen_scale.detach().cpu().reshape(()).item()),
        # EXACT_SHADOW_KEY intentionally ABSENT
        BOUNDED_PAYLOAD_KEY: {
            **_serialize_bounded_payload(zeroed),
            "forced_zero": True,
        },
        "q_sha256": _sha_tensor(zeroed.q_levels),
        "exact_shadow_physically_absent": True,
        "bounded_forced_zero": True,
    }


@dataclass(frozen=True)
class ResumeArtifact:
    arm: ArmId
    policy: ResumePolicy
    pre_cut_source_sha256: str
    states: dict[str, dict[str, Any]]
    deferred_backlog: dict[str, Any]
    rng_metadata: dict[str, Any]
    non_accumulator_metadata: dict[str, Any]

    def to_json_obj(self) -> dict[str, Any]:
        return {
            "schema": "forgotten_accum_resume_artifact/v1",
            "arm": self.arm.value,
            "policy": self.policy.value,
            "pre_cut_source_sha256": self.pre_cut_source_sha256,
            "states": self.states,
            BACKLOG_KEY: self.deferred_backlog,
            "rng_metadata": self.rng_metadata,
            "non_accumulator_metadata": self.non_accumulator_metadata,
        }

    def raw_bytes(self) -> bytes:
        return (_canonical_json(self.to_json_obj()) + "\n").encode("utf-8")

    def content_sha256(self) -> str:
        return _sha_bytes(self.raw_bytes())


def pre_cut_source_sha256(
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    deferred_backlog: Mapping[str, Any] | None,
    rng_metadata: Mapping[str, Any] | None = None,
) -> str:
    parts: list[str] = []
    for key, state in sorted(tensor_states.items()):
        parts.append(key)
        parts.append(_sha_tensor(state.q_levels))
        if state.exact_accumulator_shadow is not None:
            parts.append(_sha_tensor(state.exact_accumulator_shadow))
        parts.append(_serialize_bounded_payload(state)["payload_sha256"])
    parts.append(backlog_content_sha256(deferred_backlog))
    parts.append(_canonical_json(dict(rng_metadata or {})))
    return _sha_bytes("|".join(parts).encode("utf-8"))


def build_resume_artifact(
    *,
    arm: ArmId,
    policy: ResumePolicy,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    deferred_backlog: Mapping[str, Any] | None,
    rng_metadata: Mapping[str, Any] | None = None,
    non_accumulator_metadata: Mapping[str, Any] | None = None,
    pre_cut_source_sha256_value: str | None = None,
) -> ResumeArtifact:
    if arm is ArmId.E and policy is not ResumePolicy.EXACT_PRESERVE:
        raise ValueError("E requires EXACT_PRESERVE policy")
    if arm in (ArmId.R0, ArmId.RW) and policy is not ResumePolicy.ZERO_STRIP:
        raise ValueError("R0/RW require ZERO_STRIP policy")

    source_sha = pre_cut_source_sha256_value or pre_cut_source_sha256(
        tensor_states, deferred_backlog, rng_metadata
    )
    states_out: dict[str, dict[str, Any]] = {}
    for key, state in sorted(tensor_states.items()):
        if policy is ResumePolicy.EXACT_PRESERVE:
            states_out[key] = _serialize_state_exact(state)
        else:
            states_out[key] = _serialize_state_zero_strip(state)

    if policy is ResumePolicy.EXACT_PRESERVE:
        backlog = copy.deepcopy(dict(deferred_backlog or {}))
    else:
        backlog = empty_deferred_backlog()

    return ResumeArtifact(
        arm=arm,
        policy=policy,
        pre_cut_source_sha256=source_sha,
        states=states_out,
        deferred_backlog=backlog,
        rng_metadata=dict(rng_metadata or {}),
        non_accumulator_metadata=dict(non_accumulator_metadata or {}),
    )


def arm_artifact_path(arm_root: Path) -> Path:
    return Path(arm_root) / ARTIFACT_FILENAME


def write_resume_artifact(arm_root: Path, artifact: ResumeArtifact) -> Path:
    arm_root = Path(arm_root)
    arm_root.mkdir(parents=True, exist_ok=True)
    path = arm_artifact_path(arm_root)
    path.write_bytes(artifact.raw_bytes())
    return path


def assert_r0_rw_physical_absence(raw: bytes | str | Mapping[str, Any]) -> None:
    """R0/RW manifests/byte scan: exact shadow/bounded non-zero/backlog must be absent/zero."""

    if isinstance(raw, (bytes, bytearray)):
        text = bytes(raw).decode("utf-8")
        obj = json.loads(text)
        raw_text = text
    elif isinstance(raw, str):
        obj = json.loads(raw)
        raw_text = raw
    else:
        obj = dict(raw)
        raw_text = _canonical_json(obj)

    if EXACT_SHADOW_KEY in raw_text:
        raise AssertionError(
            "ZERO_STRIP artifact must physically omit exact_accumulator_shadow_i16 bytes"
        )
    if obj.get(BACKLOG_KEY):
        raise AssertionError("ZERO_STRIP artifact backlog must be empty")
    for state in obj.get("states", {}).values():
        if EXACT_SHADOW_KEY in state:
            raise AssertionError("exact shadow key present in ZERO_STRIP state")
        if not state.get("exact_shadow_physically_absent"):
            raise AssertionError("exact_shadow_physically_absent flag missing")
        bounded = state.get(BOUNDED_PAYLOAD_KEY) or {}
        if not bounded.get("forced_zero"):
            raise AssertionError("bounded payload not forced_zero")
        if list(bounded.get("hot_exact_indices") or []) or list(
            bounded.get("cold_exception_indices") or []
        ):
            raise AssertionError("bounded payload not physically zeroed")


def load_resume_artifact(
    arm_root: Path,
    *,
    expected_arm: ArmId,
    expected_policy: ResumePolicy,
    allowed_artifact_roots: Mapping[ArmId, Path] | None = None,
) -> tuple[dict[str, BoundedDeltaTensorState], dict[str, dict[int, dict[str, int]]], dict[str, Any]]:
    """Shared loader. Isolates E artifact: R0/RW cannot open E's arm root."""

    arm_root = Path(arm_root).resolve()
    if allowed_artifact_roots is not None:
        allowed = {k: Path(v).resolve() for k, v in allowed_artifact_roots.items()}
        if expected_arm not in allowed:
            raise PermissionError(f"arm {expected_arm} not in allowed roots")
        if arm_root != allowed[expected_arm]:
            raise PermissionError(
                f"arm {expected_arm} refused path {arm_root}; "
                f"expected isolated root {allowed[expected_arm]}"
            )
        # Explicit cross-arm isolation: R0/RW must not resolve to E's root
        if expected_arm in (ArmId.R0, ArmId.RW):
            e_root = allowed.get(ArmId.E)
            if e_root is not None and arm_root == e_root:
                raise PermissionError("R0/RW cannot load E control artifact")

    path = arm_artifact_path(arm_root)
    raw = path.read_bytes()
    obj = json.loads(raw.decode("utf-8"))
    if obj.get("arm") != expected_arm.value:
        raise ValueError(f"{FailClosedClass.CONTROL_INVALID.value}: arm mismatch")
    if obj.get("policy") != expected_policy.value:
        raise ValueError(f"{FailClosedClass.CONTROL_INVALID.value}: policy mismatch")

    if expected_policy is ResumePolicy.ZERO_STRIP:
        assert_r0_rw_physical_absence(raw)
    else:
        # EXACT_PRESERVE: every state must carry exact shadow
        for key, state_obj in obj.get("states", {}).items():
            if EXACT_SHADOW_KEY not in state_obj:
                raise ValueError(
                    f"{FailClosedClass.CONTROL_INVALID.value}: missing exact shadow "
                    f"for state {key} — no fallback to in-memory clone or zeroed"
                )

    loaded: dict[str, BoundedDeltaTensorState] = {}
    for key, state_obj in obj["states"].items():
        q = torch.tensor(state_obj["q_levels_i8"], dtype=torch.int8).view(
            *state_obj["q_shape"]
        )
        scale = float(state_obj["frozen_scale"])
        if expected_policy is ResumePolicy.EXACT_PRESERVE:
            acc = torch.tensor(state_obj[EXACT_SHADOW_KEY], dtype=torch.int16).view_as(q)
            loaded[key] = make_bounded_tensor_state(key, q, scale, acc)
            # Integrity: re-hash must match serialized
            if _sha_tensor(acc) != state_obj["exact_shadow_sha256"]:
                raise ValueError(
                    f"{FailClosedClass.CONTROL_INVALID.value}: exact shadow hash mismatch"
                )
        else:
            zeros = torch.zeros_like(q, dtype=torch.int16)
            loaded[key] = make_bounded_tensor_state(key, q, scale, zeros)

    backlog_raw = obj.get(BACKLOG_KEY) or {}
    # JSON object keys are strings; restore int flat_index keys used by runtime.
    backlog: dict[str, dict[int, dict[str, int]]] = {}
    for state_key, by_index in dict(backlog_raw).items():
        inner: dict[int, dict[str, int]] = {}
        for flat_index, entry in dict(by_index).items():
            inner[int(flat_index)] = {
                str(k): int(v) for k, v in dict(entry).items()
            }
        backlog[str(state_key)] = inner
    meta = {
        "pre_cut_source_sha256": obj["pre_cut_source_sha256"],
        "rng_metadata": obj.get("rng_metadata") or {},
        "non_accumulator_metadata": obj.get("non_accumulator_metadata") or {},
        "artifact_sha256": _sha_bytes(raw),
        "policy": expected_policy.value,
        "arm": expected_arm.value,
    }
    return loaded, backlog, meta


def serialize_discard_load(
    *,
    arm: ArmId,
    policy: ResumePolicy,
    live_states: Mapping[str, BoundedDeltaTensorState],
    live_backlog: Mapping[str, Any] | None,
    arm_root: Path,
    allowed_artifact_roots: Mapping[ArmId, Path],
    rng_metadata: Mapping[str, Any] | None = None,
    non_accumulator_metadata: Mapping[str, Any] | None = None,
    pre_cut_source_sha256_value: str,
) -> tuple[dict[str, BoundedDeltaTensorState], dict[str, dict[int, dict[str, int]]], dict[str, Any]]:
    """A1 path: serialize under policy → discard live authority → load via shared loader."""

    artifact = build_resume_artifact(
        arm=arm,
        policy=policy,
        tensor_states=live_states,
        deferred_backlog=live_backlog,
        rng_metadata=rng_metadata,
        non_accumulator_metadata=non_accumulator_metadata,
        pre_cut_source_sha256_value=pre_cut_source_sha256_value,
    )
    write_resume_artifact(arm_root, artifact)
    # Discard live authority — callers must not reuse live_states after this.
    del live_states
    return load_resume_artifact(
        arm_root,
        expected_arm=arm,
        expected_policy=policy,
        allowed_artifact_roots=allowed_artifact_roots,
    )
