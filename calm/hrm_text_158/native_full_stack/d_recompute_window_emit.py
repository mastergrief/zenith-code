"""Default-off D recompute-window instrumentation log emitter.

Captures replay-exact production constants plus bounded per-step vote/acc/q
snapshots so an offline CPU analyzer can measure per-lane K* and inclusive bpw.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec, VoteUpdateState

if TYPE_CHECKING:
    from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
        StratifiedSelectorManifest,
    )

D_RECOMPUTE_WINDOW_LOG_FILENAME = "recompute_window_log.jsonl"
D_RECOMPUTE_WINDOW_SCHEMA_VERSION = "hrm_text_158_recompute_window_log/v1"
D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0 = "hrm_text_158_recompute_window_log/v0"

BOOTSTRAP_KNOWN_ZERO = "known_zero"
BOOTSTRAP_KNOWN_SATURATED_POSITIVE = "known_saturated_positive"
BOOTSTRAP_KNOWN_SATURATED_NEGATIVE = "known_saturated_negative"
ALLOWED_BOOTSTRAP_STATES: frozenset[str] = frozenset(
    {
        BOOTSTRAP_KNOWN_ZERO,
        BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
        BOOTSTRAP_KNOWN_SATURATED_NEGATIVE,
    }
)

FORBIDDEN_BOOTSTRAP_FIELDS: frozenset[str] = frozenset(
    {
        "acc_before_stored",
        "stored_acc_before",
        "persisted_acc_before",
    }
)

MAX_INSTRUMENTED_STATE_KEYS = 2
DEFAULT_SAMPLED_LANE_COUNT = 64


class DObserverShadowUnavailableError(ValueError):
    """Fail-closed when exact_accumulator_shadow is missing or invalid for sampling."""


class DObserverOutOfRangeShadowError(ValueError):
    """Fail-closed when a sampled shadow lane violates production clip bounds."""

REQUIRED_REPLAY_CONSTANT_KEYS: tuple[str, ...] = (
    "threshold_abs",
    "accumulator_clip_min",
    "accumulator_clip_max",
    "decay_numerator",
    "decay_denominator",
    "max_abs_per_tensor",
    "fraction_per_tensor",
    "residual_subtracts_threshold_on_flip",
    "q_format",
    "acc_format",
)


@dataclass(frozen=True)
class ReplayConstants:
    threshold_abs: int
    accumulator_clip_min: int
    accumulator_clip_max: int
    decay_numerator: int = 1
    decay_denominator: int = 1
    max_abs_per_tensor: int = 2**31 - 1
    fraction_per_tensor: float = 1.0
    residual_subtracts_threshold_on_flip: bool = True
    q_format: str = "int16"
    acc_format: str = "int32_carry_storage"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_vote_update_spec(cls, spec: VoteUpdateSpec) -> ReplayConstants:
        return cls(
            threshold_abs=int(spec.threshold_abs),
            accumulator_clip_min=int(spec.accumulator_clip_min),
            accumulator_clip_max=int(spec.accumulator_clip_max),
            decay_numerator=int(spec.decay_numerator),
            decay_denominator=int(spec.decay_denominator),
            max_abs_per_tensor=int(spec.max_abs_per_tensor),
            fraction_per_tensor=float(spec.fraction_per_tensor),
        )


def default_production_replay_constants() -> ReplayConstants:
    return ReplayConstants.from_vote_update_spec(
        VoteUpdateSpec(
            threshold_abs=10,
            accumulator_clip_min=-127,
            accumulator_clip_max=127,
        )
    )


def _lane_vector_hash(values: Sequence[int]) -> str:
    payload = json.dumps([int(value) for value in values], separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _digest_mapping(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _optional_raw_global_summary_int(
    global_summary: Mapping[str, Any] | None,
    key: str,
) -> int | None:
    if global_summary is None:
        return None
    if key not in global_summary:
        return None
    value = global_summary.get(key)
    if value is None:
        return None
    return int(value)


def read_global_rate_cap_accepted_count(record: Mapping[str, Any]) -> int | None:
    if "global_rate_cap_accepted_count" not in record:
        return None
    value = record.get("global_rate_cap_accepted_count")
    if value is None:
        return None
    return int(value)


def read_global_rate_cap_deferred_count(record: Mapping[str, Any]) -> int | None:
    if "global_rate_cap_deferred_count" not in record:
        return None
    value = record.get("global_rate_cap_deferred_count")
    if value is None:
        return None
    return int(value)


def validate_bootstrap_record(record: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    for field in FORBIDDEN_BOOTSTRAP_FIELDS:
        if field in record:
            failures.append(f"forbidden_bootstrap_field:{field}")
    bootstrap = record.get("bootstrap_state")
    if bootstrap is not None and str(bootstrap) not in ALLOWED_BOOTSTRAP_STATES:
        failures.append(f"invalid_bootstrap_state:{bootstrap}")
    saturated_sign = record.get("saturated_sign_proof")
    if bootstrap in (
        BOOTSTRAP_KNOWN_SATURATED_POSITIVE,
        BOOTSTRAP_KNOWN_SATURATED_NEGATIVE,
    ):
        if saturated_sign not in ("positive", "negative"):
            failures.append("saturated_sign_proof_required")
        if bootstrap == BOOTSTRAP_KNOWN_SATURATED_POSITIVE and saturated_sign != "positive":
            failures.append("saturated_sign_mismatch_positive")
        if bootstrap == BOOTSTRAP_KNOWN_SATURATED_NEGATIVE and saturated_sign != "negative":
            failures.append("saturated_sign_mismatch_negative")
    return failures


def _accumulator_i32_flat(state: Any) -> torch.Tensor:
    """Test-only parity oracle: decoded bounded carrier via rebuild_if_stale=True."""

    if isinstance(state, BoundedDeltaTensorState):
        return (
            state.decoded_accumulators(rebuild_if_stale=True)
            .detach()
            .cpu()
            .flatten()
            .to(torch.int32)
        )
    if isinstance(state, VoteUpdateState):
        return state.accumulators.detach().cpu().flatten().to(torch.int32)
    raise TypeError(f"unsupported tensor state type {type(state)!r}")


def _shadow_numel(state: Any) -> int:
    if isinstance(state, BoundedDeltaTensorState):
        if state.exact_accumulator_shadow is not None:
            return int(state.exact_accumulator_shadow.numel())
        return int(state.q_levels.numel())
    if isinstance(state, VoteUpdateState):
        return int(state.accumulators.numel())
    raise TypeError(f"unsupported tensor state type {type(state)!r}")


def _sample_accumulator_lanes(
    state: Any,
    lane_indices: Sequence[int],
    *,
    replay_constants: ReplayConstants,
) -> list[int]:
    clip_min = int(replay_constants.accumulator_clip_min)
    clip_max = int(replay_constants.accumulator_clip_max)
    if isinstance(state, VoteUpdateState):
        flat = state.accumulators.detach().cpu().flatten().to(torch.int32)
        values = [int(flat[int(index)].item()) for index in lane_indices]
    elif isinstance(state, BoundedDeltaTensorState):
        shadow = state.exact_accumulator_shadow
        if shadow is None:
            raise DObserverShadowUnavailableError("exact_accumulator_shadow is None")
        if shadow.dtype != torch.int16:
            raise DObserverShadowUnavailableError(
                f"exact_accumulator_shadow must be torch.int16, got {shadow.dtype}"
            )
        if tuple(shadow.shape) != tuple(state.q_levels.shape):
            raise DObserverShadowUnavailableError(
                "exact_accumulator_shadow shape must match q_levels.shape"
            )
        flat = shadow.detach().cpu().flatten()
        values = [int(flat[int(index)].item()) for index in lane_indices]
    else:
        raise TypeError(f"unsupported tensor state type {type(state)!r}")
    for value in values:
        if int(value) < clip_min or int(value) > clip_max:
            raise DObserverOutOfRangeShadowError(
                "sampled accumulator lane "
                f"{int(value)} outside production clip [{clip_min}, {clip_max}]"
            )
    return values


def _q_i16_flat(state: Any) -> torch.Tensor:
    if isinstance(state, BoundedDeltaTensorState):
        return state.q_levels.detach().cpu().flatten().to(torch.int16)
    if isinstance(state, VoteUpdateState):
        return state.q_levels.detach().cpu().flatten().to(torch.int16)
    raise TypeError(f"unsupported tensor state type {type(state)!r}")


def select_instrumentation_state_keys(
    tensor_states: Mapping[str, Any],
    *,
    max_keys: int = MAX_INSTRUMENTED_STATE_KEYS,
) -> list[str]:
    ranked = sorted(
        tensor_states.keys(),
        key=lambda key: int(_shadow_numel(tensor_states[key])),
    )
    return list(ranked[: int(max_keys)])


def _sample_lane_indices(numel: int, *, max_lanes: int = DEFAULT_SAMPLED_LANE_COUNT) -> list[int]:
    if numel <= max_lanes:
        return list(range(numel))
    stride = max(1, numel // int(max_lanes))
    indices = list(range(0, numel, stride))[: int(max_lanes)]
    if (numel - 1) not in indices:
        indices.append(numel - 1)
    return sorted(set(indices))


def _carry_after_scalar_for_emit(
    acc: int,
    vote: int,
    *,
    replay: ReplayConstants,
) -> int:
    decayed = int(acc)
    if int(replay.decay_numerator) != 1 or int(replay.decay_denominator) != 1:
        decayed = (int(acc) * int(replay.decay_numerator)) // int(replay.decay_denominator)
    value = int(decayed) + int(vote)
    return max(int(replay.accumulator_clip_min), min(int(replay.accumulator_clip_max), value))


_TERNARY_Q_LEVELS: frozenset[int] = frozenset({-1, 0, 1})


def _derive_lane_flip_residual(
    *,
    q_before: int,
    q_after: int,
    acc_before: int,
    acc_after: int,
    vote: int,
    replay: ReplayConstants,
) -> tuple[bool, int | None, int | None, str]:
    """Derive per-lane flip residual observability from production q/acc transitions.

    Production apply path (vote_update.py:1105-1113):
      q_after = (q_before + direction).clamp(-1, 1)
      acc_after = clamp(carry(acc_before, vote) - direction * threshold)
    Valid single-step ternary flips are 0↔±1 with |q_delta|==1; -1↔+1 is impossible.
    """

    q_before_i = int(q_before)
    q_after_i = int(q_after)
    if q_before_i not in _TERNARY_Q_LEVELS or q_after_i not in _TERNARY_Q_LEVELS:
        return False, None, None, "absent"

    expected_carry = _carry_after_scalar_for_emit(
        int(acc_before),
        int(vote),
        replay=replay,
    )
    if int(expected_carry) == int(acc_after):
        return False, None, None, "not_applicable"

    q_delta = q_after_i - q_before_i
    if q_delta in (-1, 1):
        direction = int(q_delta)
        return True, direction, int(replay.threshold_abs), "present"
    if q_delta == 0:
        return False, None, None, "absent"
    return False, None, None, "absent"


def build_step_log_entry(
    *,
    step: int,
    state_key: str,
    replay_constants: ReplayConstants,
    acc_before: Sequence[int],
    acc_after: Sequence[int],
    q_before: Sequence[int],
    q_after: Sequence[int],
    vote_lanes: Sequence[int],
    lane_indices: Sequence[int],
    resume_generation: int = 0,
    cap_order_digest: str | None = None,
    applied_order_digest: str | None = None,
    vote_source_digest: str | None = None,
    flip_residual_applied: bool = False,
    flip_direction: int | None = None,
    flip_threshold: int | None = None,
    flip_residual_applied_lanes: Sequence[bool] | None = None,
    flip_direction_lanes: Sequence[int | None] | None = None,
    flip_threshold_lanes: Sequence[int | None] | None = None,
    residual_authority_lanes: Sequence[str] | None = None,
    backlog_depth: int | None = None,
    horizon_steps_available: int | None = None,
    global_rate_cap_accepted_count: int | None = None,
    global_rate_cap_deferred_count: int | None = None,
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "schema_version": D_RECOMPUTE_WINDOW_SCHEMA_VERSION,
        "step": int(step),
        "state_key": str(state_key),
        "resume_generation": int(resume_generation),
        "replay_constants": replay_constants.to_dict(),
        "lane_indices": [int(index) for index in lane_indices],
        "vote_lanes": [int(value) for value in vote_lanes],
        "acc_before_lanes": [int(value) for value in acc_before],
        "acc_after_lanes": [int(value) for value in acc_after],
        "q_before_lanes": [int(value) for value in q_before],
        "q_after_lanes": [int(value) for value in q_after],
        "acc_before_sha256": _lane_vector_hash(acc_before),
        "acc_after_sha256": _lane_vector_hash(acc_after),
        "q_before_sha256": _lane_vector_hash(q_before),
        "q_after_sha256": _lane_vector_hash(q_after),
        "flip_residual_applied": bool(flip_residual_applied),
        "flip_direction": None if flip_direction is None else int(flip_direction),
        "flip_threshold": None if flip_threshold is None else int(flip_threshold),
        "flip_residual_applied_lanes": (
            [bool(value) for value in flip_residual_applied_lanes]
            if flip_residual_applied_lanes is not None
            else [bool(flip_residual_applied)] * len(lane_indices)
        ),
        "flip_direction_lanes": (
            [None if value is None else int(value) for value in flip_direction_lanes]
            if flip_direction_lanes is not None
            else [flip_direction] * len(lane_indices)
        ),
        "flip_threshold_lanes": (
            [None if value is None else int(value) for value in flip_threshold_lanes]
            if flip_threshold_lanes is not None
            else [flip_threshold] * len(lane_indices)
        ),
        "residual_authority_lanes": (
            [str(value) for value in residual_authority_lanes]
            if residual_authority_lanes is not None
            else ["not_applicable"] * len(lane_indices)
        ),
        "cap_order_digest": cap_order_digest,
        "applied_order_digest": applied_order_digest,
        "vote_source_digest": vote_source_digest,
        "backlog_depth": backlog_depth,
        "horizon_steps_available": horizon_steps_available,
        "global_rate_cap_accepted_count": (
            None
            if global_rate_cap_accepted_count is None
            else int(global_rate_cap_accepted_count)
        ),
        "global_rate_cap_deferred_count": (
            None
            if global_rate_cap_deferred_count is None
            else int(global_rate_cap_deferred_count)
        ),
        "bootstrap_state": None,
        "saturated_sign_proof": None,
    }
    return entry


def append_recompute_window_log_chunk(log_path: Path | str, entry: Mapping[str, Any]) -> None:
    failures = validate_bootstrap_record(entry)
    if failures:
        raise ValueError(
            "bootstrap validation failed: " + ",".join(failures)
        )
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(entry), sort_keys=True, separators=(",", ":")))
        handle.write("\n")


def initialize_recompute_window_log_for_probe_session(log_path: Path | str) -> Path:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    return path


def iter_recompute_window_log_records(log_path: Path | str) -> list[dict[str, Any]]:
    path = Path(log_path)
    if not path.is_file():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        records.append(json.loads(line))
    return records


def maybe_emit_d_recompute_window_step_records(
    *,
    enabled: bool,
    log_path: Path | None,
    step: int,
    pre_update_states: Mapping[str, Any],
    post_update_states: Mapping[str, Any],
    votes_by_key: Mapping[str, torch.Tensor],
    replay_constants: ReplayConstants,
    global_summary: Mapping[str, Any] | None = None,
    resume_generation: int = 0,
    selector_manifest: StratifiedSelectorManifest | None = None,
) -> None:
    if not enabled or log_path is None:
        return
    if selector_manifest is not None:
        from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
            STRESS_TAIL_POLICY_HORIZON_FIXED,
            sample_lanes_for_key,
            select_instrumentation_state_keys_from_manifest,
        )

        state_keys = select_instrumentation_state_keys_from_manifest(
            pre_update_states,
            selector_manifest,
        )
        manifest_entries = selector_manifest.entry_by_key()
        stress_tail_policy = str(selector_manifest.manifest_spec.get("stress_tail_policy") or "")
    else:
        state_keys = select_instrumentation_state_keys(pre_update_states)
        manifest_entries = None
        stress_tail_policy = ""
    cap_digest = None
    applied_digest = None
    if global_summary is not None:
        cap_digest = _digest_mapping(
            {
                "global_rate_cap_cap": global_summary.get("global_rate_cap_cap"),
                "global_rate_cap_saturated": global_summary.get("global_rate_cap_saturated"),
                "global_rate_cap_enabled": global_summary.get("global_rate_cap_enabled"),
            }
        )
        applied_digest = _digest_mapping(
            {
                "global_rate_cap_accepted_count": global_summary.get(
                    "global_rate_cap_accepted_count"
                ),
                "global_rate_cap_deferred_count": global_summary.get(
                    "global_rate_cap_deferred_count"
                ),
                "q_changed_count": global_summary.get("q_changed_count"),
            }
        )
    for state_key in state_keys:
        pre_state = pre_update_states[state_key]
        post_state = post_update_states[state_key]
        q_before_tensor = _q_i16_flat(pre_state)
        q_after_tensor = _q_i16_flat(post_state)
        votes_tensor = votes_by_key[state_key].detach().cpu().flatten().to(torch.int32)
        vote_lane_values = [int(votes_tensor[index].item()) for index in range(votes_tensor.numel())]
        if manifest_entries is not None:
            lane_indices = sample_lanes_for_key(
                pre_state,
                manifest_entry=manifest_entries[state_key],
                vote_values=(
                    vote_lane_values
                    if stress_tail_policy != STRESS_TAIL_POLICY_HORIZON_FIXED
                    else None
                ),
                replay_constants=replay_constants,
                stress_tail_policy=stress_tail_policy or None,
            )
        else:
            lane_indices = _sample_lane_indices(int(_shadow_numel(pre_state)))
        vote_lanes = [int(votes_tensor[index].item()) for index in lane_indices]
        acc_before_lanes = _sample_accumulator_lanes(
            pre_state,
            lane_indices,
            replay_constants=replay_constants,
        )
        acc_after_lanes = _sample_accumulator_lanes(
            post_state,
            lane_indices,
            replay_constants=replay_constants,
        )
        q_before_lanes = [int(q_before_tensor[index].item()) for index in lane_indices]
        q_after_lanes = [int(q_after_tensor[index].item()) for index in lane_indices]
        flip_residual_applied_lanes: list[bool] = []
        flip_direction_lanes: list[int | None] = []
        flip_threshold_lanes: list[int | None] = []
        residual_authority_lanes: list[str] = []
        for position, lane_index in enumerate(lane_indices):
            flip_applied, direction, threshold, authority = _derive_lane_flip_residual(
                q_before=q_before_lanes[position],
                q_after=q_after_lanes[position],
                acc_before=acc_before_lanes[position],
                acc_after=acc_after_lanes[position],
                vote=vote_lanes[position],
                replay=replay_constants,
            )
            flip_residual_applied_lanes.append(bool(flip_applied))
            flip_direction_lanes.append(direction)
            flip_threshold_lanes.append(threshold)
            residual_authority_lanes.append(str(authority))
        entry = build_step_log_entry(
            step=int(step),
            state_key=str(state_key),
            replay_constants=replay_constants,
            acc_before=acc_before_lanes,
            acc_after=acc_after_lanes,
            q_before=q_before_lanes,
            q_after=q_after_lanes,
            vote_lanes=vote_lanes,
            lane_indices=lane_indices,
            resume_generation=int(resume_generation),
            cap_order_digest=cap_digest,
            applied_order_digest=applied_digest,
            vote_source_digest=_lane_vector_hash(vote_lanes),
            flip_residual_applied=any(flip_residual_applied_lanes),
            flip_residual_applied_lanes=flip_residual_applied_lanes,
            flip_direction_lanes=flip_direction_lanes,
            flip_threshold_lanes=flip_threshold_lanes,
            residual_authority_lanes=residual_authority_lanes,
            backlog_depth=int(global_summary.get("deferred_backlog_size", 0))
            if global_summary is not None
            else None,
            horizon_steps_available=int(step),
            global_rate_cap_accepted_count=_optional_raw_global_summary_int(
                global_summary,
                "global_rate_cap_accepted_count",
            ),
            global_rate_cap_deferred_count=_optional_raw_global_summary_int(
                global_summary,
                "global_rate_cap_deferred_count",
            ),
        )
        append_recompute_window_log_chunk(log_path, entry)
