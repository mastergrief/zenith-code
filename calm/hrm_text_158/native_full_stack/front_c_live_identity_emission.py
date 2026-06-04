"""Default-off live Front-C identity artifact emission helpers.

This module is intentionally separate from the locked Front-C scaffold/adapter:
it converts cloned CPU learner observations into the single self-contained
identity artifact that the Stage-1a adapter already validates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.front_c_identity_emitter import (
    FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
    FRONT_C_DENSE_DECISION_SOURCE,
    FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
    FRONT_C_RUN_DERIVED_ARTIFACT,
    FRONT_C_SPARSE_DECISION_SOURCE,
    FRONT_C_STATE_LAYOUT_HASH_SEMANTICS,
    classify_front_c_saved_audit_root,
    front_c_report_from_identity_artifact,
    validate_front_c_identity_artifact,
)
from calm.hrm_text_158.native_full_stack.front_c_projection import (
    FrontCDecisionPath,
    FrontCDecisionSurfaceStep,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    apply_integer_vote_update_reference,
    plan_integer_vote_update_reference,
)


FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION = (
    "hrm_text_158_front_c/v0.live_identity_emission"
)
FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION = (
    "hrm_text_158_front_c/v0.live_identity_observation_cloned_cpu"
)


FrontCIdentity = tuple[str, int]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _ids_from_indices(state_key: str, indices: torch.Tensor | Sequence[int]) -> set[FrontCIdentity]:
    if isinstance(indices, torch.Tensor):
        raw = indices.detach().cpu().to(torch.int64).flatten().tolist()
    else:
        raw = [int(index) for index in indices]
    return {(str(state_key), int(index)) for index in raw}


def _identity_dicts(identities: Sequence[FrontCIdentity]) -> list[dict[str, int | str]]:
    return [
        {"state_key": state_key, "flat_index": int(flat_index)}
        for state_key, flat_index in sorted((str(k), int(i)) for k, i in identities)
    ]


def _direction_dicts(values: Mapping[FrontCIdentity, int]) -> list[dict[str, int | str]]:
    return [
        {"state_key": state_key, "flat_index": int(flat_index), "direction": int(direction)}
        for (state_key, flat_index), direction in sorted(values.items())
    ]


def _backlog_keys(backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None) -> set[FrontCIdentity]:
    out: set[FrontCIdentity] = set()
    for state_key, by_index in (backlog or {}).items():
        for flat_index in by_index:
            out.add((str(state_key), int(flat_index)))
    return out


def _layout_state_entry(state_key: str, q_levels: torch.Tensor) -> dict[str, Any]:
    shape = [int(dim) for dim in q_levels.shape]
    layout_payload = {
        "state_key": str(state_key),
        "logical_shape": shape,
        "state_key_semantics": FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
        "flat_index_semantics": FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
    }
    return {
        "state_key": str(state_key),
        "logical_shape": shape,
        "eligible_weight_count": int(q_levels.numel()),
        "state_layout_sha256": _sha256_json(layout_payload),
    }


def _state_metadata(
    states_by_key: Mapping[str, VoteUpdateState],
    *,
    timeline_steps: Sequence[int],
) -> dict[str, Any]:
    states = [
        _layout_state_entry(state_key, state.q_levels)
        for state_key, state in sorted(states_by_key.items())
    ]
    layout_hash_by_key = {
        str(state["state_key"]): str(state["state_layout_sha256"])
        for state in states
    }
    metadata_payload = {
        "state_key_semantics": FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
        "flat_index_semantics": FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
        "state_hash_semantics": FRONT_C_STATE_LAYOUT_HASH_SEMANTICS,
        "states": states,
    }
    return {
        **metadata_payload,
        "state_layout_metadata_sha256": _sha256_json(metadata_payload),
        "step_state_layout_sha256": {
            str(int(step)): dict(layout_hash_by_key)
            for step in sorted({int(step) for step in timeline_steps})
        },
    }


def _q_ledger_payload(states_by_key: Mapping[str, VoteUpdateState]) -> dict[str, Any]:
    return {
        "regime_name": "front_c_live_identity_base3_q",
        "logical_shapes": [
            [int(dim) for dim in state.q_levels.shape]
            for _, state in sorted(states_by_key.items())
        ],
        "scale_count": len(states_by_key),
        "accumulator_bits_per_weight": 0.0,
    }


@dataclass(frozen=True)
class _StepPaths:
    surface: FrontCDecisionSurfaceStep
    dense_path: FrontCDecisionPath
    sparse_path: FrontCDecisionPath
    dense_diagnostics: dict[str, Any]
    sparse_diagnostics: dict[str, Any]


def _current_threshold_ids(
    state_key: str,
    state: VoteUpdateState,
    spec: VoteUpdateSpec,
) -> set[FrontCIdentity]:
    flat_q = state.q_levels.flatten().to(torch.int16)
    flat_acc = state.accumulators.flatten().to(torch.int32)
    threshold = int(spec.threshold_abs)
    current = ((flat_acc >= threshold) & (flat_q < 1)) | (
        (flat_acc <= -threshold) & (flat_q > -1)
    )
    return _ids_from_indices(state_key, torch.nonzero(current, as_tuple=False).flatten())


def _path_from_inputs(
    states_by_key: Mapping[str, VoteUpdateState],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    specs_by_key: Mapping[str, VoteUpdateSpec],
    *,
    label: str,
) -> tuple[FrontCDecisionPath, dict[str, Any]]:
    q_directions: dict[FrontCIdentity, int] = {}
    replay_veto_ids: set[FrontCIdentity] = set()
    q_changed_ids: set[FrontCIdentity] = set()
    stats_by_key: dict[str, Any] = {}

    for state_key, state in sorted(states_by_key.items()):
        result = apply_integer_vote_update_reference(
            state,
            inputs_by_key[state_key],
            specs_by_key[state_key],
        )
        plan = result.plan
        for raw_index, raw_direction in zip(
            plan.applied_indices.detach().cpu().to(torch.int64).tolist(),
            plan.applied_directions.detach().cpu().to(torch.int16).tolist(),
        ):
            q_directions[(state_key, int(raw_index))] = int(raw_direction)
        replay_veto_ids |= _ids_from_indices(state_key, plan.replay_ce_veto_indices)
        changed = torch.nonzero(
            result.q_levels.flatten() != state.q_levels.flatten(),
            as_tuple=False,
        ).flatten()
        q_changed_ids |= _ids_from_indices(state_key, changed)
        stats_by_key[state_key] = dict(result.stats)

    path = FrontCDecisionPath(
        label=label,
        q_flip_directions=_direction_dicts(q_directions),
        accepted_under_global_cap_keys=(),
        deferred_under_global_cap_keys=(),
        backlog_keys=(),
        replay_veto_decision_keys=_identity_dicts(replay_veto_ids),
    )
    return path, {
        "q_changed_count": len(q_changed_ids),
        "q_changed_sha256": _sha256_json(_identity_dicts(q_changed_ids)),
        "global_cap_used": False,
        "local_stats_by_key": stats_by_key,
    }


def _surface_from_exact_path(
    step: int,
    states_by_key: Mapping[str, VoteUpdateState],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    specs_by_key: Mapping[str, VoteUpdateSpec],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    *,
    cap_frontier_width: int,
) -> tuple[FrontCDecisionSurfaceStep, set[FrontCIdentity]]:
    current_threshold: set[FrontCIdentity] = set()
    active_next: set[FrontCIdentity] = set()
    ranking_exact: set[FrontCIdentity] = set()
    cap_frontier: set[FrontCIdentity] = set()
    replay_veto: set[FrontCIdentity] = set()

    for state_key, state in sorted(states_by_key.items()):
        spec = specs_by_key[state_key]
        plan = plan_integer_vote_update_reference(state, inputs_by_key[state_key], spec)
        current_threshold |= _current_threshold_ids(state_key, state, spec)
        candidates = _ids_from_indices(state_key, plan.candidate_indices)
        active_next |= candidates
        ranking_exact |= candidates
        cap_frontier |= _ids_from_indices(
            state_key,
            plan.pre_veto_selected_indices[: int(cap_frontier_width)],
        )
        replay_veto |= _ids_from_indices(state_key, plan.replay_ce_veto_indices)

    backlog = _backlog_keys(deferred_backlog)
    decision_relevant = active_next | ranking_exact | cap_frontier | replay_veto | backlog
    eligible = sum(int(state.q_levels.numel()) for state in states_by_key.values())
    return (
        FrontCDecisionSurfaceStep(
            step=int(step),
            eligible_weight_count=eligible,
            current_magnitude_threshold_keys=_identity_dicts(current_threshold),
            active_next_step_keys=_identity_dicts(active_next),
            ranking_sensitive_exact_keys=_identity_dicts(ranking_exact),
            global_cap_frontier_keys=_identity_dicts(cap_frontier),
            backlog_carry_keys=_identity_dicts(backlog),
            replay_veto_residual_keys=_identity_dicts(replay_veto),
        ),
        decision_relevant,
    )


def _sparse_states_from_active_set(
    states_by_key: Mapping[str, VoteUpdateState],
    active_ids: set[FrontCIdentity],
) -> dict[str, VoteUpdateState]:
    sparse: dict[str, VoteUpdateState] = {}
    for state_key, state in sorted(states_by_key.items()):
        hot = tuple(
            int(flat_index)
            for key, flat_index in sorted(active_ids)
            if key == state_key
        )
        encoded = encode_sparse_active_set_accumulator(state, hot_exact_indices=hot)
        sparse[state_key] = VoteUpdateState(
            q_levels=state.q_levels.detach().cpu().clone().contiguous(),
            accumulators=encoded,
        )
    return sparse


def encode_sparse_active_set_accumulator(
    state: VoteUpdateState,
    *,
    hot_exact_indices: Sequence[int],
) -> torch.Tensor:
    """Encode/decode the sparse active-set accumulator without touching q."""

    from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
        decode_bounded_accumulator_to_i16,
        encode_budget_capped_hybrid_reference,
    )

    encoded = encode_budget_capped_hybrid_reference(
        state,
        hot_exact_indices=tuple(int(index) for index in hot_exact_indices),
        cold_default_value=0,
    )
    return decode_bounded_accumulator_to_i16(encoded).detach().cpu().clone().contiguous()


def build_front_c_live_step_paths(
    *,
    step: int,
    states_by_key: Mapping[str, VoteUpdateState],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    specs_by_key: Mapping[str, VoteUpdateSpec],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    global_cap_used: bool = False,
    cap_frontier_width: int = 1,
) -> _StepPaths:
    """Build dense and independently-derived sparse path records for one step."""

    if global_cap_used:
        raise ValueError("Front-C path-b emission must not enable or consume global cap")
    if set(states_by_key) != set(inputs_by_key) or set(states_by_key) != set(specs_by_key):
        raise ValueError("states, inputs, and specs must have identical keys")

    surface, active_ids = _surface_from_exact_path(
        int(step),
        states_by_key,
        inputs_by_key,
        specs_by_key,
        deferred_backlog,
        cap_frontier_width=cap_frontier_width,
    )
    dense_path, dense_diag = _path_from_inputs(
        states_by_key,
        inputs_by_key,
        specs_by_key,
        label="front_c_dense_int16_reference",
    )
    sparse_states = _sparse_states_from_active_set(states_by_key, active_ids)
    sparse_path, sparse_diag = _path_from_inputs(
        sparse_states,
        inputs_by_key,
        specs_by_key,
        label="front_c_sparse_encode_decode_reference",
    )
    sparse_diag.update(
        {
            "sparse_active_set_count": len(active_ids),
            "sparse_active_set_sha256": _sha256_json(_identity_dicts(active_ids)),
            "sparse_active_set_source": "dense_oracle_active_ids",
            "sparse_policy_selector_claimed": False,
            "sparse_decision_equivalence_scope": (
                "conditional_on_dense_oracle_active_set_encode_decode"
            ),
            "sparse_derivation": "base3_q_plus_sparse_active_set_accumulator_encode_decode",
        },
    )
    return _StepPaths(
        surface=surface,
        dense_path=dense_path,
        sparse_path=sparse_path,
        dense_diagnostics=dense_diag,
        sparse_diagnostics=sparse_diag,
    )


@dataclass
class FrontCLiveIdentityCollector:
    artifact_path: Path
    emission_interval: int = 0
    audit_interval: int = 0
    cap_frontier_width: int = 1
    tensor_metadata_bits_per_state: int = 64
    bucket_metadata_bits: int = 64
    guardrail_metadata_bits: int = 64
    value_bits_per_row: int = 16
    flag_bits_per_row: int = 2

    def __post_init__(self) -> None:
        self.artifact_path = Path(self.artifact_path)
        self.emission_interval = int(self.emission_interval)
        self.audit_interval = int(self.audit_interval)
        self.cap_frontier_width = int(self.cap_frontier_width)
        self._step_rows: dict[int, FrontCDecisionSurfaceStep] = {}
        self._states_by_key: dict[str, VoteUpdateState] = {}
        self._latest_dense_path: FrontCDecisionPath | None = None
        self._latest_sparse_path: FrontCDecisionPath | None = None
        self._dense_paths_by_step: dict[int, FrontCDecisionPath] = {}
        self._sparse_paths_by_step: dict[int, FrontCDecisionPath] = {}
        self._diagnostics: dict[str, Any] = {
            "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
            "global_cap_used": False,
            "global_cap_honest_false": True,
            "collector_return_ignored_by_learner": True,
            "observation_source": FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION,
            "step_diagnostics": {},
        }

    def should_collect_step(self, step: int, *, total_steps: int) -> bool:
        step_i = int(step)
        if step_i <= 0:
            return False
        if step_i == 1 or step_i == int(total_steps):
            return True
        if self.audit_interval > 0 and step_i % self.audit_interval == 0:
            return True
        return self.emission_interval > 0 and step_i % self.emission_interval == 0

    def record_step0(self, states_by_key: Mapping[str, Any]) -> None:
        coerced = {
            str(key): VoteUpdateState(
                q_levels=value.q_levels.detach().cpu().clone().contiguous(),
                accumulators=value.exact_accumulator_shadow.detach().cpu().clone().contiguous(),
            )
            for key, value in sorted(states_by_key.items())
        }
        self._states_by_key = coerced
        eligible = sum(int(state.q_levels.numel()) for state in coerced.values())
        self._step_rows[0] = FrontCDecisionSurfaceStep(
            step=0,
            eligible_weight_count=eligible,
        )

    def record_step_observation(self, *, step: int, observation: Mapping[str, Any]) -> None:
        if observation.get("schema") != FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unexpected Front-C observation schema")
        if bool(observation.get("global_cap_used", False)):
            raise ValueError("Front-C path-b emission must record global_cap_used=false")
        states_by_key = dict(observation["states_by_key"])
        inputs_by_key = dict(observation["inputs_by_key"])
        specs_by_key = dict(observation["specs_by_key"])
        self._states_by_key = states_by_key
        paths = build_front_c_live_step_paths(
            step=int(step),
            states_by_key=states_by_key,
            inputs_by_key=inputs_by_key,
            specs_by_key=specs_by_key,
            deferred_backlog=observation.get("deferred_backlog", {}),
            global_cap_used=False,
            cap_frontier_width=self.cap_frontier_width,
        )
        self._step_rows[int(step)] = paths.surface
        self._latest_dense_path = paths.dense_path
        self._latest_sparse_path = paths.sparse_path
        self._dense_paths_by_step[int(step)] = paths.dense_path
        self._sparse_paths_by_step[int(step)] = paths.sparse_path
        self._diagnostics["step_diagnostics"][str(int(step))] = {
            "dense": paths.dense_diagnostics,
            "dense_q_flip_directions": paths.dense_path.to_dict()["q_flip_directions"],
            "sparse": paths.sparse_diagnostics,
            "sparse_q_flip_directions": paths.sparse_path.to_dict()["q_flip_directions"],
            "observation": {
                "states_sha256": {
                    key: {
                        "q_levels": _tensor_sha256(state.q_levels),
                        "accumulators": _tensor_sha256(state.accumulators),
                    }
                    for key, state in sorted(states_by_key.items())
                },
                "votes_sha256": {
                    key: _tensor_sha256(inputs.votes)
                    for key, inputs in sorted(inputs_by_key.items())
                },
            },
        }

    def _selected_timeline(self, audit_reports: Mapping[str, Any] | None) -> list[FrontCDecisionSurfaceStep]:
        rows = dict(self._step_rows)
        if not rows:
            raise ValueError("Front-C identity collector has no timeline rows")
        selected_steps: set[int] = {0} if 0 in rows else set()
        positive_steps = sorted(step for step in rows if step > 0)
        if positive_steps:
            selected_steps.add(positive_steps[0])
            selected_steps.add(positive_steps[-1])
            acquired_steps: list[int] = []
            for raw_step, report in sorted(
                (audit_reports or {}).items(),
                key=lambda item: int(item[0]),
            ):
                step_i = int(raw_step)
                if bool(dict(report).get("acquired", False)) or int(
                    dict(report).get("strict_exact_count", 0),
                ) >= 90:
                    acquired_steps.append(step_i)
            missing_acquired_steps = [step for step in acquired_steps if step not in rows]
            if missing_acquired_steps:
                self._diagnostics["timeline_selection_error"] = {
                    "reason": "acquired_audit_step_not_collected",
                    "missing_acquired_steps": [int(step) for step in missing_acquired_steps],
                    "collected_steps": sorted(int(step) for step in rows),
                    "collector_audit_interval": int(self.audit_interval),
                    "collector_emission_interval": int(self.emission_interval),
                }
                raise ValueError(
                    "Front-C identity emission acquired audit step was not collected: "
                    f"missing {tuple(missing_acquired_steps)}; set collection to audit rows"
                )
            if acquired_steps:
                acquired_step = acquired_steps[0]
                selected_steps.add(acquired_step)
            else:
                selected_steps.update(positive_steps[-2:])
        return [rows[step] for step in sorted(selected_steps)]

    def _selected_q_flip_receipt(
        self,
        timeline: Sequence[FrontCDecisionSurfaceStep],
    ) -> dict[str, Any]:
        selected_steps = [int(row.step) for row in timeline]
        per_step: dict[str, Any] = {}
        union_identities: set[FrontCIdentity] = set()
        total_q_flip_events = 0
        for step in selected_steps:
            dense_path = self._dense_paths_by_step.get(step)
            directions = {} if dense_path is None else dict(dense_path.q_flip_directions)
            identities = set(directions)
            union_identities |= identities
            total_q_flip_events += len(directions)
            per_step[str(step)] = {
                "q_flip_count": len(directions),
                "q_flip_directions": _direction_dicts(directions),
                "q_flip_directions_sha256": _sha256_json(_direction_dicts(directions)),
            }
        return {
            "selected_timeline_steps": selected_steps,
            "event_delta_count": int(total_q_flip_events),
            "event_delta_count_semantics": (
                "total_q_flip_direction_rows_across_selected_timeline"
            ),
            "event_delta_unique_identity_count": len(union_identities),
            "event_delta_union_sha256": _sha256_json(_identity_dicts(union_identities)),
            "selected_step_q_flip_receipts": per_step,
        }

    def _metadata_bit_receipt(
        self,
        timeline: Sequence[FrontCDecisionSurfaceStep],
    ) -> dict[str, Any]:
        state_count = len(self._states_by_key)
        q_flip_receipt = self._selected_q_flip_receipt(timeline)
        q_flip_count = int(q_flip_receipt["event_delta_count"])
        return {
            "tensor_metadata_bits": state_count * int(self.tensor_metadata_bits_per_state),
            "bucket_metadata_bits": int(self.bucket_metadata_bits),
            "scale_metadata_bits": 0,
            "scale_metadata_zero_reason": (
                "accumulator sparse-active-set path has no extra accumulator scale; "
                "q frozen scales are charged by q_ledger.scale_count"
            ),
            "guardrail_metadata_bits": int(self.guardrail_metadata_bits),
            "event_delta_count": q_flip_count,
            "event_delta_zero_reason": "no q flip events observed" if q_flip_count == 0 else "",
            "value_bits_per_row": int(self.value_bits_per_row),
            "flag_bits_per_row": int(self.flag_bits_per_row),
            **q_flip_receipt,
        }

    def build_payload(
        self,
        *,
        audit_reports: Mapping[str, Any] | None = None,
        prior_audit_start_reports: Mapping[str, Any] | None = None,
        prior_audit_final_reports: Mapping[str, Any] | None = None,
        steps_completed: int | None = None,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        if self._latest_dense_path is None or self._latest_sparse_path is None:
            raise ValueError("Front-C identity collector needs at least one step observation")
        timeline = self._selected_timeline(audit_reports)
        state_metadata = _state_metadata(
            self._states_by_key,
            timeline_steps=[step.step for step in timeline],
        )
        bit_receipt = self._metadata_bit_receipt(timeline)
        source_artifact_id = _sha256_json(
            {
                "timeline_steps": [int(step.step) for step in timeline],
                "dense": self._latest_dense_path.to_dict(),
                "sparse": self._latest_sparse_path.to_dict(),
                "selected_step_q_flip_receipts": bit_receipt[
                    "selected_step_q_flip_receipts"
                ],
                "steps_completed": steps_completed,
                "stop_reason": stop_reason,
            },
        )
        payload = {
            "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
            "timeline": [step.to_dict() for step in timeline],
            "dense_decision_path": self._latest_dense_path.to_dict(),
            "sparse_decision_path": self._latest_sparse_path.to_dict(),
            "state_metadata": state_metadata,
            "decision_path_derivation": {
                "artifact_class": FRONT_C_RUN_DERIVED_ARTIFACT,
                "dense_source": FRONT_C_DENSE_DECISION_SOURCE,
                "sparse_source": FRONT_C_SPARSE_DECISION_SOURCE,
                "independent_sparse_derivation": True,
                "sparse_active_set_source": "dense_oracle_active_ids",
                "sparse_policy_selector_claimed": False,
                "sparse_decision_equivalence_scope": (
                    "conditional_on_dense_oracle_active_set_encode_decode"
                ),
                "source_artifact_id": source_artifact_id,
                "state_layout_metadata_sha256": state_metadata[
                    "state_layout_metadata_sha256"
                ],
            },
            "q_ledger": _q_ledger_payload(self._states_by_key),
            "value_bits_per_row": bit_receipt["value_bits_per_row"],
            "flag_bits_per_row": bit_receipt["flag_bits_per_row"],
            "tensor_metadata_bits": bit_receipt["tensor_metadata_bits"],
            "bucket_metadata_bits": bit_receipt["bucket_metadata_bits"],
            "scale_metadata_bits": bit_receipt["scale_metadata_bits"],
            "guardrail_metadata_bits": bit_receipt["guardrail_metadata_bits"],
            "event_delta_count": bit_receipt["event_delta_count"],
            "prior_audit": {
                "requested_supports": ["L0b", "math_a0", "L0c1"],
                "start_reports": dict(prior_audit_start_reports or {}),
                "final_reports": dict(prior_audit_final_reports or {}),
            },
            "diagnostics": {
                **self._diagnostics,
                "steps_completed": None if steps_completed is None else int(steps_completed),
                "stop_reason": "" if stop_reason is None else str(stop_reason),
                "metadata_bit_receipt": bit_receipt,
                "audit_report_steps": sorted(str(key) for key in (audit_reports or {})),
            },
        }
        return payload

    def finalize(
        self,
        *,
        audit_reports: Mapping[str, Any] | None = None,
        prior_audit_start_reports: Mapping[str, Any] | None = None,
        prior_audit_final_reports: Mapping[str, Any] | None = None,
        steps_completed: int | None = None,
        stop_reason: str | None = None,
    ) -> dict[str, Any]:
        payload = self.build_payload(
            audit_reports=audit_reports,
            prior_audit_start_reports=prior_audit_start_reports,
            prior_audit_final_reports=prior_audit_final_reports,
            steps_completed=steps_completed,
            stop_reason=stop_reason,
        )
        self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
        self.artifact_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        validation = validate_front_c_identity_artifact(payload)
        report = front_c_report_from_identity_artifact(payload)
        inventory = classify_front_c_saved_audit_root(self.artifact_path)
        return {
            "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": hashlib.sha256(
                self.artifact_path.read_bytes(),
            ).hexdigest(),
            "identity_validation": validation.to_dict(),
            "front_c_report": report.to_dict(),
            "inventory": inventory.to_dict(),
            "single_self_contained_artifact": True,
            "global_cap_used": False,
            "gpu_launched": False,
            "pt_artifact_written": False,
        }


__all__ = [
    "FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION",
    "FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION",
    "FrontCLiveIdentityCollector",
    "build_front_c_live_step_paths",
    "encode_sparse_active_set_accumulator",
]
