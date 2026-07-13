"""U/E/R0/RW arm orchestration (CPU-facing; thin).

Reuse Phase-A flip-defer facade via runner kwarg schedule. Formal 4-arm GPU
run is a later gated packet — this module owns manifests, A1 resume path,
FUTURE_STREAM step accounting, and RW flip_application_deferred schedule.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    FUTURE_STREAM_MATCHED_BUDGET,
    PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY,
    W_REWARM_STEPS,
    ArmId,
    ResumePolicy,
    build_all_arm_manifests,
    flip_defer_schedule,
    policy_for_arm,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_zero_seed import (
    assert_pre_W_zeroed_identity,
    pre_cut_source_sha256,
    serialize_discard_load,
)


@dataclass(frozen=True)
class FutureStreamBudget:
    """Matched post-cut budget: RW rewarm steps ARE the first W post-cut train steps."""

    t_cut: int
    W: int
    runway_end: int
    label: str = FUTURE_STREAM_MATCHED_BUDGET

    def post_cut_train_steps(self) -> int:
        return int(self.runway_end) - int(self.t_cut)

    def rw_rewarm_step_indices(self) -> tuple[int, ...]:
        # 1-based post-cut indices
        return tuple(range(1, int(self.W) + 1))

    def assert_matched(self) -> None:
        if self.post_cut_train_steps() < int(self.W):
            raise AssertionError("FUTURE_STREAM: runway too short for W rewarm steps")


@dataclass
class ArmResumeResult:
    arm: ArmId
    tensor_states: dict[str, BoundedDeltaTensorState]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]
    meta: dict[str, Any]


def isolated_arm_roots(experiment_root: Path) -> dict[ArmId, Path]:
    root = Path(experiment_root)
    return {arm: (root / "arms" / arm.value).resolve() for arm in ArmId}


def resume_arm_from_live_cut(
    *,
    arm: ArmId,
    live_states: Mapping[str, BoundedDeltaTensorState],
    live_backlog: Mapping[str, Any] | None,
    experiment_root: Path,
    rng_metadata: Mapping[str, Any] | None = None,
    non_accumulator_metadata: Mapping[str, Any] | None = None,
    shared_pre_cut_source_sha256: str | None = None,
) -> ArmResumeResult:
    """A1: shared serialize→discard→load. U does not resume (no-op identity)."""

    if arm is ArmId.U:
        raise ValueError("U is uninterrupted — no resume path")

    policy = policy_for_arm(arm)
    assert policy is not None
    roots = isolated_arm_roots(experiment_root)
    source_sha = shared_pre_cut_source_sha256 or pre_cut_source_sha256(
        live_states, live_backlog, rng_metadata
    )
    states, backlog, meta = serialize_discard_load(
        arm=arm,
        policy=policy,
        live_states=live_states,
        live_backlog=live_backlog,
        arm_root=roots[arm],
        allowed_artifact_roots=roots,
        rng_metadata=rng_metadata,
        non_accumulator_metadata=non_accumulator_metadata,
        pre_cut_source_sha256_value=source_sha,
    )
    return ArmResumeResult(arm=arm, tensor_states=states, deferred_backlog=backlog, meta=meta)


def prove_r0_rw_same_zero_seed(
    r0: ArmResumeResult,
    rw: ArmResumeResult,
) -> str:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_flip_deferral_reducers import (
        backlog_content_sha256,
    )
    import hashlib

    def acc_sha(states: Mapping[str, BoundedDeltaTensorState]) -> str:
        h = hashlib.sha256()
        for key, st in sorted(states.items()):
            shadow = st.exact_accumulator_shadow
            if shadow is None:
                raise AssertionError("expected shadow after make_bounded_tensor_state")
            h.update(key.encode())
            h.update(shadow.detach().cpu().contiguous().numpy().tobytes())
        return h.hexdigest()

    return assert_pre_W_zeroed_identity(
        r0_acc_sha=acc_sha(r0.tensor_states),
        rw_acc_sha=acc_sha(rw.tensor_states),
        r0_backlog_sha=backlog_content_sha256(r0.deferred_backlog),
        rw_backlog_sha=backlog_content_sha256(rw.deferred_backlog),
    )


def all_else_identical_manifests() -> dict[str, dict[str, Any]]:
    manifests = build_all_arm_manifests()
    identity_blobs = {k: v.identity.as_dict() for k, v in manifests.items()}
    first = next(iter(identity_blobs.values()))
    for arm, blob in identity_blobs.items():
        if blob != first:
            raise AssertionError(f"identity drift on arm {arm}")
    return {k: v.as_dict() for k, v in manifests.items()}


def rw_flip_defer_flags_for_post_cut_window() -> list[bool]:
    """Flags for post-cut steps 1..W+2 (includes W+1 ordinary release step)."""

    return [flip_defer_schedule(ArmId.RW, post_cut_step_index=i) for i in range(1, W_REWARM_STEPS + 3)]


__all__ = [
    "FutureStreamBudget",
    "ArmResumeResult",
    "isolated_arm_roots",
    "resume_arm_from_live_cut",
    "prove_r0_rw_same_zero_seed",
    "all_else_identical_manifests",
    "rw_flip_defer_flags_for_post_cut_window",
    "PRE_W_ZEROED_ACC_AND_BACKLOG_IDENTITY",
    "ResumePolicy",
]
