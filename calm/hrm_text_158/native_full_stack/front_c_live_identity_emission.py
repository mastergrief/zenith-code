"""Default-off live Front-C identity artifact emission helpers.

This module is intentionally separate from the locked Front-C scaffold/adapter:
it converts cloned CPU learner observations into the single self-contained
identity artifact that the Stage-1a adapter already validates.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Iterator, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.front_c_identity_emitter import (
    FRONT_C_CANONICAL_STATE_KEY_SEMANTICS,
    FRONT_C_DENSE_DECISION_SOURCE,
    FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS,
    FRONT_C_RUN_DERIVED_ARTIFACT,
    FRONT_C_SPARSE_DECISION_SOURCE,
    FRONT_C_STATE_LAYOUT_HASH_SEMANTICS,
    classify_front_c_identity_payload,
    front_c_report_from_identity_artifact,
    validate_front_c_identity_artifact,
)
from calm.hrm_text_158.native_full_stack.front_c_projection import (
    FrontCDecisionPath,
    FrontCDecisionSurfaceStep,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdatePlan,
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
FRONT_C_IDENTITY_SCOPE_EXACT = "full_exact_identity_emission"
FRONT_C_IDENTITY_SCOPE_BOUNDED_FRONTIER = "bounded_frontier_summary"
FRONT_C_IDENTITY_SCOPE_BOUNDED_SPARSE = "bounded_sparse_oracle_sample"
FRONT_C_SPARSE_EQUIVALENCE_EXACT = (
    "conditional_on_dense_oracle_active_set_encode_decode"
)
FRONT_C_SPARSE_EQUIVALENCE_BOUNDED = "bounded_sparse_oracle_sample_nonclaim"
FRONT_C_LIVE_TIMING_SCHEMA_VERSION = "hrm_text_158_front_c/v0.live_identity_timing"
DEFAULT_MAX_EXACT_IDENTITY_KEYS = 100_000
DEFAULT_SPARSE_ORACLE_MAX_ACTIVE_IDS = 100_000
FRONT_C_ORACLE_FULL_REBUILD_ENV = "HRM_TEXT_158_FRONT_C_ORACLE_FULL_REBUILD"
FRONT_C_INDEPENDENT_ORACLE_ENV = "HRM_TEXT_158_FRONT_C_INDEPENDENT_ORACLE"


FrontCIdentity = tuple[str, int]


@dataclass(frozen=True)
class _IdentityUniverse:
    sources_by_key: tuple[tuple[str, torch.Tensor], ...] = ()
    extra_identities: frozenset[FrontCIdentity] = frozenset()


@dataclass(frozen=True)
class _BoundedIdentitySelection:
    identities: tuple[FrontCIdentity, ...]
    diagnostics: dict[str, Any]


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _new_timing_diagnostics(*, phase: str) -> dict[str, Any]:
    return {
        "schema": FRONT_C_LIVE_TIMING_SCHEMA_VERSION,
        "phase": str(phase),
        "durations_seconds": {},
    }


def _record_duration(timing: dict[str, Any], key: str, start: float) -> None:
    timing.setdefault("durations_seconds", {})[str(key)] = max(
        0.0,
        time.perf_counter() - start,
    )


def _ensure_duration(
    timing: dict[str, Any],
    key: str,
    *,
    duration_seconds: float = 0.0,
) -> None:
    timing.setdefault("durations_seconds", {}).setdefault(
        str(key),
        max(0.0, float(duration_seconds)),
    )


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


def _identity_row_json_bytes(state_key: str, flat_index: int) -> bytes:
    state_json = json.dumps(
        str(state_key),
        ensure_ascii=True,
        separators=(",", ":"),
    )
    return f'{{"flat_index":{int(flat_index)},"state_key":{state_json}}}'.encode("utf-8")


def _identity_dicts_sha256(identities: Iterable[FrontCIdentity]) -> str:
    h = hashlib.sha256()
    h.update(b"[")
    first = True
    for state_key, flat_index in sorted((str(k), int(i)) for k, i in identities):
        if not first:
            h.update(b",")
        h.update(_identity_row_json_bytes(state_key, int(flat_index)))
        first = False
    h.update(b"]")
    return h.hexdigest()


def _sorted_unique_i64_indices(indices: torch.Tensor | Sequence[int]) -> torch.Tensor:
    if isinstance(indices, torch.Tensor):
        flat = indices.detach().cpu().to(torch.int64).flatten().contiguous()
    else:
        flat = torch.tensor([int(index) for index in indices], dtype=torch.int64)
    if int(flat.numel()) <= 1:
        return flat.clone().contiguous()
    if bool(torch.all(flat[1:] >= flat[:-1]).item()):
        return torch.unique_consecutive(flat).contiguous()
    return torch.unique(flat, sorted=True).to(torch.int64).cpu().contiguous()


def _empty_i64_indices() -> torch.Tensor:
    return torch.empty(0, dtype=torch.int64)


def _sorted_difference_i64(base: torch.Tensor, remove: torch.Tensor) -> torch.Tensor:
    base = _sorted_unique_i64_indices(base)
    remove = _sorted_unique_i64_indices(remove)
    if int(base.numel()) == 0 or int(remove.numel()) == 0:
        return base.clone().contiguous()
    pos = torch.searchsorted(remove, base)
    in_bounds = pos < int(remove.numel())
    matched = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any().item()):
        matched[in_bounds] = remove[pos[in_bounds]] == base[in_bounds]
    return base[~matched].to(torch.int64).cpu().contiguous()


def _sorted_probe_present_i64(source: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    source = _sorted_unique_i64_indices(source)
    probe = _sorted_unique_i64_indices(probe)
    if int(source.numel()) == 0 or int(probe.numel()) == 0:
        return _empty_i64_indices()
    pos = torch.searchsorted(source, probe)
    in_bounds = pos < int(source.numel())
    matched = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any().item()):
        matched[in_bounds] = source[pos[in_bounds]] == probe[in_bounds]
    return probe[matched].to(torch.int64).cpu().contiguous()


def _sorted_probe_missing_i64(source: torch.Tensor, probe: torch.Tensor) -> torch.Tensor:
    source = _sorted_unique_i64_indices(source)
    probe = _sorted_unique_i64_indices(probe)
    if int(probe.numel()) == 0:
        return probe
    if int(source.numel()) == 0:
        return probe.clone().contiguous()
    pos = torch.searchsorted(source, probe)
    in_bounds = pos < int(source.numel())
    matched = torch.zeros_like(in_bounds, dtype=torch.bool)
    if bool(in_bounds.any().item()):
        matched[in_bounds] = source[pos[in_bounds]] == probe[in_bounds]
    return probe[~matched].to(torch.int64).cpu().contiguous()


def _sorted_union_i64(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left = _sorted_unique_i64_indices(left)
    right = _sorted_unique_i64_indices(right)
    if int(left.numel()) == 0:
        return right.clone().contiguous()
    if int(right.numel()) == 0:
        return left.clone().contiguous()
    return torch.unique(torch.cat((left, right)), sorted=True).to(torch.int64).cpu().contiguous()


def _merge_index_delta(
    base: torch.Tensor,
    *,
    touched: torch.Tensor,
    active_after_touch: torch.Tensor,
) -> torch.Tensor:
    return _sorted_union_i64(
        _sorted_difference_i64(base, touched),
        active_after_touch,
    )


def _identity_universe_from_sources(
    sources_by_key: Mapping[str, torch.Tensor | Sequence[int]],
    *,
    extra_identities: Iterable[FrontCIdentity] = (),
) -> _IdentityUniverse:
    sources = tuple(
        (str(state_key), _sorted_unique_i64_indices(indices))
        for state_key, indices in sorted(sources_by_key.items())
    )
    extras = frozenset((str(state_key), int(flat_index)) for state_key, flat_index in extra_identities)
    return _IdentityUniverse(sources_by_key=sources, extra_identities=extras)


def _identity_universe_from_identities(identities: Iterable[FrontCIdentity]) -> _IdentityUniverse:
    return _IdentityUniverse(
        extra_identities=frozenset((str(state_key), int(flat_index)) for state_key, flat_index in identities),
    )


def _source_contains(source: torch.Tensor | None, flat_index: int) -> bool:
    if source is None or int(source.numel()) == 0:
        return False
    needle = torch.tensor(int(flat_index), dtype=torch.int64)
    pos = int(torch.searchsorted(source, needle).item())
    return pos < int(source.numel()) and int(source[pos].item()) == int(flat_index)


def _identity_universe_contains(universe: _IdentityUniverse, identity: FrontCIdentity) -> bool:
    state_key, flat_index = str(identity[0]), int(identity[1])
    if (state_key, flat_index) in universe.extra_identities:
        return True
    source_map = dict(universe.sources_by_key)
    return _source_contains(source_map.get(state_key), flat_index)


def _identity_universe_count(universe: _IdentityUniverse) -> int:
    source_map = dict(universe.sources_by_key)
    count = sum(int(indices.numel()) for indices in source_map.values())
    for state_key, flat_index in universe.extra_identities:
        if not _source_contains(source_map.get(state_key), int(flat_index)):
            count += 1
    return count


def _extra_indices_by_key(universe: _IdentityUniverse) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    source_map = dict(universe.sources_by_key)
    for state_key, flat_index in universe.extra_identities:
        if _source_contains(source_map.get(state_key), int(flat_index)):
            continue
        out.setdefault(str(state_key), []).append(int(flat_index))
    for values in out.values():
        values.sort()
    return out


def _iter_merged_indices(source: torch.Tensor | None, extras: Sequence[int]) -> Iterator[int]:
    extra_pos = 0
    if source is not None:
        chunk_size = 8192
        for start in range(0, int(source.numel()), chunk_size):
            for raw_index in source[start : start + chunk_size].tolist():
                index = int(raw_index)
                while extra_pos < len(extras) and int(extras[extra_pos]) < index:
                    yield int(extras[extra_pos])
                    extra_pos += 1
                if extra_pos < len(extras) and int(extras[extra_pos]) == index:
                    extra_pos += 1
                yield index
    while extra_pos < len(extras):
        yield int(extras[extra_pos])
        extra_pos += 1


def _iter_identity_universe(universe: _IdentityUniverse) -> Iterator[FrontCIdentity]:
    source_map = dict(universe.sources_by_key)
    extras_by_key = _extra_indices_by_key(universe)
    for state_key in sorted(set(source_map) | set(extras_by_key)):
        for flat_index in _iter_merged_indices(
            source_map.get(state_key),
            extras_by_key.get(state_key, ()),
        ):
            yield (str(state_key), int(flat_index))


def _identity_universe_sha256(universe: _IdentityUniverse) -> str:
    h = hashlib.sha256()
    h.update(b"[")
    first = True
    for state_key, flat_index in _iter_identity_universe(universe):
        if not first:
            h.update(b",")
        h.update(_identity_row_json_bytes(state_key, int(flat_index)))
        first = False
    h.update(b"]")
    return h.hexdigest()


def _select_bounded_identity_universe(
    surface_name: str,
    universe: _IdentityUniverse,
    *,
    max_keys: int,
    priority_ids: Iterable[FrontCIdentity] | None = None,
) -> _BoundedIdentitySelection:
    limit = max(0, int(max_keys))
    full_count = _identity_universe_count(universe)
    bounded = full_count > limit
    if not bounded:
        identities = tuple(_iter_identity_universe(universe))
    else:
        selected: list[FrontCIdentity] = []
        seen: set[FrontCIdentity] = set()
        for raw_identity in sorted(
            (str(state_key), int(flat_index))
            for state_key, flat_index in (priority_ids or ())
        ):
            if len(selected) >= limit:
                break
            if raw_identity in seen:
                continue
            if not _identity_universe_contains(universe, raw_identity):
                continue
            selected.append(raw_identity)
            seen.add(raw_identity)
        for identity in _iter_identity_universe(universe):
            if len(selected) >= limit:
                break
            if identity in seen:
                continue
            selected.append(identity)
            seen.add(identity)
        identities = tuple(sorted(selected))
    diagnostics = {
        "surface": str(surface_name),
        "full_identity_count": int(full_count),
        "emitted_identity_count": len(identities),
        "identity_cap": limit,
        "bounded": bool(bounded),
    }
    if bounded:
        diagnostics["cap_reason"] = (
            f"{surface_name} full identity count {full_count} exceeds cap {limit}"
        )
    return _BoundedIdentitySelection(identities=identities, diagnostics=diagnostics)


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
    surface_diagnostics: dict[str, Any]
    emitted_decision_relevant_ids: frozenset[FrontCIdentity] = frozenset()
    full_active_identity_universe: _IdentityUniverse | None = None
    identity_emission_scope: str = FRONT_C_IDENTITY_SCOPE_EXACT
    full_identity_emission_claimed: bool = True
    full_sparse_equivalence_claimed: bool = True
    bounded_nonclaim_reasons: tuple[str, ...] = ()
    timing_diagnostics: dict[str, Any] | None = None


def _current_threshold_indices(
    state: VoteUpdateState,
    spec: VoteUpdateSpec,
) -> torch.Tensor:
    flat_q = state.q_levels.flatten().to(torch.int16)
    flat_acc = state.accumulators.flatten().to(torch.int32)
    threshold = int(spec.threshold_abs)
    current = ((flat_acc >= threshold) & (flat_q < 1)) | (
        (flat_acc <= -threshold) & (flat_q > -1)
    )
    return _sorted_unique_i64_indices(torch.nonzero(current, as_tuple=False).flatten())


def _threshold_indices_from_q_acc_rows(
    q_levels: torch.Tensor,
    accumulators: torch.Tensor,
    spec: VoteUpdateSpec,
    rows: torch.Tensor,
) -> torch.Tensor:
    rows = _sorted_unique_i64_indices(rows)
    if int(rows.numel()) == 0:
        return rows
    flat_q = q_levels.detach().cpu().flatten().to(torch.int16)
    flat_acc = accumulators.detach().cpu().flatten().to(torch.int32)
    valid = (rows >= 0) & (rows < int(flat_q.numel()))
    rows = rows[valid]
    if int(rows.numel()) == 0:
        return rows.to(torch.int64).cpu().contiguous()
    threshold = int(spec.threshold_abs)
    q_values = flat_q[rows]
    acc_values = flat_acc[rows]
    active = ((acc_values >= threshold) & (q_values < 1)) | (
        (acc_values <= -threshold) & (q_values > -1)
    )
    return rows[active].to(torch.int64).cpu().contiguous()


def _current_threshold_ids(
    state_key: str,
    state: VoteUpdateState,
    spec: VoteUpdateSpec,
) -> set[FrontCIdentity]:
    return _ids_from_indices(state_key, _current_threshold_indices(state, spec))


def _bounded_identity_set(
    surface_name: str,
    identities: set[FrontCIdentity],
    *,
    max_keys: int,
    priority_ids: set[FrontCIdentity] | None = None,
) -> tuple[set[FrontCIdentity], dict[str, Any]]:
    selection = _select_bounded_identity_universe(
        surface_name,
        _identity_universe_from_identities(identities),
        max_keys=max_keys,
        priority_ids=priority_ids,
    )
    return set(selection.identities), selection.diagnostics


def _q_acc_entry_parts(value: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    if isinstance(value, Mapping):
        return (
            value["q_levels"],
            value["accumulators"],
            dict(value.get("stats", {})),
        )
    q_levels, accumulators, stats = value
    return q_levels, accumulators, dict(stats)


def _plan_priority_ids(
    plans_by_key: Mapping[str, VoteUpdatePlan],
    *,
    cap_frontier_width: int,
) -> set[FrontCIdentity]:
    priority: set[FrontCIdentity] = set()
    for state_key, plan in sorted(plans_by_key.items()):
        priority |= _ids_from_indices(state_key, plan.applied_indices)
        priority |= _ids_from_indices(state_key, plan.replay_ce_veto_indices)
        priority |= _ids_from_indices(
            state_key,
            plan.pre_veto_selected_indices[: int(cap_frontier_width)],
        )
    return priority


def _path_from_reused_plans(
    states_by_key: Mapping[str, VoteUpdateState],
    plans_by_key: Mapping[str, VoteUpdatePlan],
    q_acc_by_key: Mapping[str, Any],
    *,
    label: str,
    active_ids: set[FrontCIdentity] | None = None,
) -> tuple[FrontCDecisionPath, dict[str, Any]]:
    q_directions: dict[FrontCIdentity, int] = {}
    replay_veto_ids: set[FrontCIdentity] = set()
    q_changed_ids: set[FrontCIdentity] = set()
    stats_by_key: dict[str, Any] = {}
    active_filter = None if active_ids is None else set(active_ids)

    for state_key, state in sorted(states_by_key.items()):
        plan = plans_by_key[state_key]
        q_after, _, stats = _q_acc_entry_parts(q_acc_by_key[state_key])
        for raw_index, raw_direction in zip(
            plan.applied_indices.detach().cpu().to(torch.int64).tolist(),
            plan.applied_directions.detach().cpu().to(torch.int16).tolist(),
        ):
            identity = (str(state_key), int(raw_index))
            if active_filter is None or identity in active_filter:
                q_directions[identity] = int(raw_direction)
        replay_ids = _ids_from_indices(state_key, plan.replay_ce_veto_indices)
        if active_filter is not None:
            replay_ids &= active_filter
        replay_veto_ids |= replay_ids
        changed = torch.nonzero(
            q_after.detach().cpu().flatten().to(torch.int8)
            != state.q_levels.detach().cpu().flatten().to(torch.int8),
            as_tuple=False,
        ).flatten()
        changed_ids = _ids_from_indices(state_key, changed)
        if active_filter is not None:
            changed_ids &= active_filter
        q_changed_ids |= changed_ids
        stats_by_key[state_key] = dict(stats)

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
        "derivation": "reused_vote_update_plan_and_q_acc_result",
        "active_filter_count": None if active_filter is None else len(active_filter),
    }


def _surface_from_reused_plans(
    step: int,
    states_by_key: Mapping[str, VoteUpdateState],
    specs_by_key: Mapping[str, VoteUpdateSpec],
    plans_by_key: Mapping[str, VoteUpdatePlan],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    *,
    cap_frontier_width: int,
    max_exact_identity_keys: int,
    current_threshold_indices_by_key: Mapping[str, torch.Tensor] | None = None,
) -> tuple[
    FrontCDecisionSurfaceStep,
    set[FrontCIdentity],
    _IdentityUniverse,
    dict[str, Any],
    bool,
]:
    surface_timing: dict[str, float] = {}
    cap_frontier: set[FrontCIdentity] = set()
    replay_veto: set[FrontCIdentity] = set()
    candidate_indices_by_key: dict[str, torch.Tensor] = {}
    current_threshold_indices: dict[str, torch.Tensor] = {}

    phase_start = time.perf_counter()
    for state_key, state in sorted(states_by_key.items()):
        plan = plans_by_key[state_key]
        if current_threshold_indices_by_key is None:
            current_threshold_indices[str(state_key)] = _current_threshold_indices(
                state,
                specs_by_key[state_key],
            )
        else:
            current_threshold_indices[str(state_key)] = _sorted_unique_i64_indices(
                current_threshold_indices_by_key.get(str(state_key), _empty_i64_indices()),
            )
        candidate_indices_by_key[str(state_key)] = plan.candidate_indices
        cap_frontier |= _ids_from_indices(
            state_key,
            plan.pre_veto_selected_indices[: int(cap_frontier_width)],
        )
        replay_veto |= _ids_from_indices(state_key, plan.replay_ce_veto_indices)
    surface_timing[
        "current_threshold_scan"
        if current_threshold_indices_by_key is None
        else "current_threshold_index_read"
    ] = max(0.0, time.perf_counter() - phase_start)

    backlog = _backlog_keys(deferred_backlog)
    phase_start = time.perf_counter()
    priority = _plan_priority_ids(plans_by_key, cap_frontier_width=cap_frontier_width) | backlog
    surface_timing["priority_build"] = max(0.0, time.perf_counter() - phase_start)
    candidate_universe = _identity_universe_from_sources(candidate_indices_by_key)
    decision_universe = _IdentityUniverse(
        sources_by_key=candidate_universe.sources_by_key,
        extra_identities=frozenset(cap_frontier | replay_veto | backlog),
    )
    surface_universes = {
        "current_magnitude_threshold_keys": _identity_universe_from_sources(current_threshold_indices),
        "active_next_step_keys": candidate_universe,
        "ranking_sensitive_exact_keys": candidate_universe,
        "global_cap_frontier_keys": _identity_universe_from_identities(cap_frontier),
        "backlog_carry_keys": _identity_universe_from_identities(backlog),
        "replay_veto_residual_keys": _identity_universe_from_identities(replay_veto),
    }
    emitted: dict[str, tuple[FrontCIdentity, ...]] = {}
    surface_diagnostics: dict[str, Any] = {}
    bounded = False
    selection_total = 0.0
    for name, universe in surface_universes.items():
        phase_start = time.perf_counter()
        selection = _select_bounded_identity_universe(
            name,
            universe,
            max_keys=max_exact_identity_keys,
            priority_ids=priority,
        )
        selection_duration = max(0.0, time.perf_counter() - phase_start)
        selection_total += selection_duration
        emitted[name] = selection.identities
        surface_diagnostics[name] = selection.diagnostics
        surface_diagnostics[name]["selection_duration_seconds"] = selection_duration
        bounded = bounded or bool(selection.diagnostics["bounded"])

    emitted_decision_relevant: set[FrontCIdentity] = set()
    for name in (
        "active_next_step_keys",
        "ranking_sensitive_exact_keys",
        "global_cap_frontier_keys",
        "replay_veto_residual_keys",
        "backlog_carry_keys",
    ):
        emitted_decision_relevant.update(emitted[name])
    eligible = sum(int(state.q_levels.numel()) for state in states_by_key.values())
    surface_diagnostics["decision_relevant_identity_count"] = _identity_universe_count(
        decision_universe,
    )
    surface_diagnostics["emitted_decision_relevant_identity_count"] = len(emitted_decision_relevant)
    surface_diagnostics["identity_emission_bounded"] = bool(bounded)
    surface_timing["bounded_selection_by_surface"] = selection_total
    surface_diagnostics["surface_build_subtimers_seconds"] = surface_timing
    return (
        FrontCDecisionSurfaceStep(
            step=int(step),
            eligible_weight_count=eligible,
            current_magnitude_threshold_keys=emitted["current_magnitude_threshold_keys"],
            active_next_step_keys=emitted["active_next_step_keys"],
            ranking_sensitive_exact_keys=emitted["ranking_sensitive_exact_keys"],
            global_cap_frontier_keys=emitted["global_cap_frontier_keys"],
            backlog_carry_keys=emitted["backlog_carry_keys"],
            replay_veto_residual_keys=emitted["replay_veto_residual_keys"],
        ),
        emitted_decision_relevant,
        decision_universe,
        surface_diagnostics,
        bounded,
    )


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
    plans_by_key: Mapping[str, VoteUpdatePlan] | None = None,
    q_acc_by_key: Mapping[str, Any] | None = None,
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
    global_cap_used: bool = False,
    cap_frontier_width: int = 1,
    max_exact_identity_keys: int = DEFAULT_MAX_EXACT_IDENTITY_KEYS,
    sparse_oracle_max_active_ids: int = DEFAULT_SPARSE_ORACLE_MAX_ACTIVE_IDS,
    current_threshold_indices_by_key: Mapping[str, torch.Tensor] | None = None,
    include_full_active_hash: bool = False,
) -> _StepPaths:
    """Build dense and independently-derived sparse path records for one step."""

    if global_cap_used:
        raise ValueError("Front-C path-b emission must not enable or consume global cap")
    if set(states_by_key) != set(inputs_by_key) or set(states_by_key) != set(specs_by_key):
        raise ValueError("states, inputs, and specs must have identical keys")
    if plans_by_key is not None and set(states_by_key) != set(plans_by_key):
        raise ValueError("states and plans_by_key must have identical keys")
    if q_acc_by_key is not None and set(states_by_key) != set(q_acc_by_key):
        raise ValueError("states and q_acc_by_key must have identical keys")

    if plans_by_key is not None and q_acc_by_key is not None:
        step_timing = _new_timing_diagnostics(phase="build_front_c_live_step_paths")
        step_timing["path_source"] = "reused_observer_plan"
        phase_start = time.perf_counter()
        surface, active_ids, full_active_ids, surface_diag, surface_bounded = (
            _surface_from_reused_plans(
                int(step),
                states_by_key,
                specs_by_key,
                plans_by_key,
                deferred_backlog,
                cap_frontier_width=cap_frontier_width,
                max_exact_identity_keys=max_exact_identity_keys,
                current_threshold_indices_by_key=current_threshold_indices_by_key,
            )
        )
        _record_duration(step_timing, "surface_from_reused_plans", phase_start)
        step_timing["durations_seconds"]["collect_surface_build"] = step_timing[
            "durations_seconds"
        ]["surface_from_reused_plans"]
        surface_subtimers = dict(surface_diag.get("surface_build_subtimers_seconds", {}))
        _ensure_duration(
            step_timing,
            "current_threshold_scan",
            duration_seconds=float(surface_subtimers.get("current_threshold_scan", 0.0)),
        )
        _ensure_duration(
            step_timing,
            "current_threshold_index_read",
            duration_seconds=float(surface_subtimers.get("current_threshold_index_read", 0.0)),
        )
        _ensure_duration(
            step_timing,
            "priority_build",
            duration_seconds=float(surface_subtimers.get("priority_build", 0.0)),
        )
        _ensure_duration(
            step_timing,
            "bounded_selection_by_surface",
            duration_seconds=float(surface_subtimers.get("bounded_selection_by_surface", 0.0)),
        )
        phase_start = time.perf_counter()
        dense_path, dense_diag = _path_from_reused_plans(
            states_by_key,
            plans_by_key,
            q_acc_by_key,
            label="front_c_dense_int16_reference",
        )
        _record_duration(step_timing, "dense_from_reused_plans", phase_start)
        phase_start = time.perf_counter()
        priority = _plan_priority_ids(
            plans_by_key,
            cap_frontier_width=cap_frontier_width,
        )
        sparse_active_ids = set(active_ids)
        full_active_count = _identity_universe_count(full_active_ids)
        sparse_subset_diag = {
            "surface": "sparse_active_set",
            "full_identity_count": full_active_count,
            "emitted_identity_count": len(sparse_active_ids),
            "identity_cap": int(sparse_oracle_max_active_ids),
            "bounded": False,
        }
        sparse_bounded = False
        sparse_exact_oracle_ran = False
        if full_active_count > int(sparse_oracle_max_active_ids):
            sparse_selection = _select_bounded_identity_universe(
                "sparse_active_set",
                full_active_ids,
                max_keys=sparse_oracle_max_active_ids,
                priority_ids=priority,
            )
            sparse_active_ids = set(sparse_selection.identities)
            sparse_subset_diag = sparse_selection.diagnostics
            sparse_bounded = True
        full_identity = not surface_bounded
        _record_duration(step_timing, "sparse_active_set_select_or_bound", phase_start)
        if not surface_bounded and not sparse_bounded:
            sparse_active_ids = set(_iter_identity_universe(full_active_ids))
            sparse_path_mode = "exact_sparse_encode_decode"
            phase_start = time.perf_counter()
            sparse_states = _sparse_states_from_active_set(states_by_key, sparse_active_ids)
            _record_duration(step_timing, "sparse_path_materialize", phase_start)
            phase_start = time.perf_counter()
            sparse_path, sparse_diag = _path_from_inputs(
                sparse_states,
                inputs_by_key,
                specs_by_key,
                label="front_c_sparse_encode_decode_reference",
            )
            _record_duration(step_timing, "sparse_encode_decode_or_bounded_filter", phase_start)
            sparse_exact_oracle_ran = True
            sparse_subset_diag["emitted_identity_count"] = len(sparse_active_ids)
        else:
            sparse_path_mode = "bounded_reused_plan_filter"
            _ensure_duration(step_timing, "sparse_path_materialize")
            phase_start = time.perf_counter()
            sparse_path, sparse_diag = _path_from_reused_plans(
                states_by_key,
                plans_by_key,
                q_acc_by_key,
                label="front_c_sparse_encode_decode_reference",
                active_ids=sparse_active_ids,
            )
            _record_duration(step_timing, "sparse_encode_decode_or_bounded_filter", phase_start)
        full_sparse = not sparse_bounded and not surface_bounded and sparse_exact_oracle_ran
        scope = FRONT_C_IDENTITY_SCOPE_EXACT
        reasons: list[str] = []
        if surface_bounded:
            scope = FRONT_C_IDENTITY_SCOPE_BOUNDED_FRONTIER
            reasons.append("surface_identity_emission_exceeded_cap")
        elif sparse_bounded:
            scope = FRONT_C_IDENTITY_SCOPE_BOUNDED_SPARSE
            reasons.append("sparse_active_set_exceeded_cap")
        full_active_sha256 = ""
        phase_start = time.perf_counter()
        if bool(include_full_active_hash):
            full_active_sha256 = _identity_universe_sha256(full_active_ids)
        _record_duration(step_timing, "sparse_active_set_full_hash_oracle", phase_start)
        sparse_diag.update(
            {
                "sparse_active_set_count": len(sparse_active_ids),
                "sparse_active_set_full_count": full_active_count,
                "sparse_active_set_sha256": _identity_dicts_sha256(sparse_active_ids),
                "sparse_active_set_full_sha256": full_active_sha256,
                "sparse_active_set_full_hash_computed": bool(include_full_active_hash),
                "sparse_active_set_full_hash_semantics": (
                    "oracle_debug_only_empty_when_not_computed"
                ),
                "sparse_active_set_source": (
                    "dense_oracle_active_ids"
                    if full_sparse
                    else "bounded_reused_plan_active_ids"
                ),
                "sparse_active_set_bounding": sparse_subset_diag,
                "sparse_policy_selector_claimed": False,
                "sparse_decision_equivalence_scope": (
                    FRONT_C_SPARSE_EQUIVALENCE_EXACT
                    if full_sparse
                    else FRONT_C_SPARSE_EQUIVALENCE_BOUNDED
                ),
                "sparse_derivation": (
                    "base3_q_plus_sparse_active_set_accumulator_encode_decode"
                    if full_sparse
                    else "reused_vote_update_plan_bounded_nonclaim_no_sparse_oracle"
                ),
                "sparse_exact_oracle_ran": bool(sparse_exact_oracle_ran),
                "full_sparse_equivalence_claimed": bool(full_sparse),
            },
        )
        step_timing["sparse_path_mode"] = sparse_path_mode
        step_timing["full_identity_emission_claimed"] = bool(full_identity)
        step_timing["full_sparse_equivalence_claimed"] = bool(full_sparse)
        dense_diag["dense_source"] = "reused_vote_update_plan"
        dense_diag["full_identity_emission_claimed"] = bool(full_identity)
        return _StepPaths(
            surface=surface,
            dense_path=dense_path,
            sparse_path=sparse_path,
            dense_diagnostics=dense_diag,
            sparse_diagnostics=sparse_diag,
            surface_diagnostics=surface_diag,
            emitted_decision_relevant_ids=frozenset(active_ids),
            full_active_identity_universe=full_active_ids,
            identity_emission_scope=scope,
            full_identity_emission_claimed=bool(full_identity),
            full_sparse_equivalence_claimed=bool(full_sparse),
            bounded_nonclaim_reasons=tuple(reasons),
            timing_diagnostics=step_timing,
        )

    step_timing = _new_timing_diagnostics(phase="build_front_c_live_step_paths")
    step_timing["path_source"] = "reference_recompute_compatibility_fallback"
    phase_start = time.perf_counter()
    surface, active_ids = _surface_from_exact_path(
        int(step),
        states_by_key,
        inputs_by_key,
        specs_by_key,
        deferred_backlog,
        cap_frontier_width=cap_frontier_width,
    )
    _record_duration(step_timing, "surface_from_reused_plans", phase_start)
    phase_start = time.perf_counter()
    dense_path, dense_diag = _path_from_inputs(
        states_by_key,
        inputs_by_key,
        specs_by_key,
        label="front_c_dense_int16_reference",
    )
    _record_duration(step_timing, "dense_from_reused_plans", phase_start)
    phase_start = time.perf_counter()
    active_ids = set(active_ids)
    _record_duration(step_timing, "sparse_active_set_select_or_bound", phase_start)
    phase_start = time.perf_counter()
    sparse_states = _sparse_states_from_active_set(states_by_key, active_ids)
    _record_duration(step_timing, "sparse_path_materialize", phase_start)
    phase_start = time.perf_counter()
    sparse_path, sparse_diag = _path_from_inputs(
        sparse_states,
        inputs_by_key,
        specs_by_key,
        label="front_c_sparse_encode_decode_reference",
    )
    _record_duration(step_timing, "sparse_encode_decode_or_bounded_filter", phase_start)
    step_timing["sparse_path_mode"] = "exact_sparse_encode_decode"
    step_timing["full_identity_emission_claimed"] = True
    step_timing["full_sparse_equivalence_claimed"] = True
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
        surface_diagnostics={
            "identity_emission_bounded": False,
            "derivation": "reference_recompute_compatibility_fallback",
        },
        emitted_decision_relevant_ids=frozenset(active_ids),
        full_active_identity_universe=_identity_universe_from_identities(active_ids),
        timing_diagnostics=step_timing,
    )


def _surface_diagnostics_contract(diagnostics: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in diagnostics.items():
        if key == "surface_build_subtimers_seconds":
            continue
        if isinstance(value, Mapping):
            out[key] = {
                str(k): v
                for k, v in value.items()
                if str(k) != "selection_duration_seconds"
            }
        else:
            out[str(key)] = value
    return out


def _assert_legacy_surface_oracle_match(
    *,
    step: int,
    paths: _StepPaths,
    states_by_key: Mapping[str, VoteUpdateState],
    inputs_by_key: Mapping[str, VoteUpdateInputs],
    specs_by_key: Mapping[str, VoteUpdateSpec],
    deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    cap_frontier_width: int,
    max_exact_identity_keys: int,
    sparse_oracle_max_active_ids: int,
    include_full_active_hash: bool = False,
) -> dict[str, Any]:
    oracle_paths = build_front_c_live_step_paths(
        step=int(step),
        states_by_key=states_by_key,
        inputs_by_key=inputs_by_key,
        specs_by_key=specs_by_key,
        deferred_backlog=deferred_backlog,
        global_cap_used=False,
        cap_frontier_width=cap_frontier_width,
        sparse_oracle_max_active_ids=sparse_oracle_max_active_ids,
        max_exact_identity_keys=max_exact_identity_keys,
    )
    oracle_surface = oracle_paths.surface
    oracle_active = set(oracle_paths.emitted_decision_relevant_ids)
    oracle_full_active = (
        oracle_paths.full_active_identity_universe
        or _identity_universe_from_identities(oracle_active)
    )
    oracle_timing = dict(oracle_paths.timing_diagnostics or {})
    oracle_source = str(oracle_timing.get("path_source", ""))
    actual_full_active = paths.full_active_identity_universe or _identity_universe_from_identities(())
    full_active_count = _identity_universe_count(actual_full_active)
    oracle_full_active_count = _identity_universe_count(oracle_full_active)
    expected_full_identity = paths.surface.to_dict() == oracle_surface.to_dict()
    expected_full_sparse = expected_full_identity and (
        oracle_full_active_count <= max(0, int(sparse_oracle_max_active_ids))
    )
    dense_q_flips = paths.dense_path.to_dict()["q_flip_directions"]
    sparse_q_flips = paths.sparse_path.to_dict()["q_flip_directions"]
    q_flip_parity_scope = (
        "dense_sparse_exact_q_flip_receipt"
        if paths.full_sparse_equivalence_claimed
        else "bounded_sparse_nonclaim_q_flip_receipt_parity_not_claimed"
    )
    checks = {
        "surface_rows": paths.surface.to_dict() == oracle_surface.to_dict(),
        "emitted_decision_relevant": (
            _identity_dicts(tuple(paths.emitted_decision_relevant_ids))
            == _identity_dicts(tuple(oracle_active))
        ),
        "full_active_count": full_active_count == oracle_full_active_count,
        "full_identity_flag": paths.full_identity_emission_claimed == expected_full_identity,
        "full_sparse_flag": paths.full_sparse_equivalence_claimed == expected_full_sparse,
        "q_flip_receipt_parity": (
            dense_q_flips == sparse_q_flips
            if paths.full_sparse_equivalence_claimed
            else True
        ),
        "q_flip_receipt_hash_parity": (
            _sha256_json(dense_q_flips) == _sha256_json(sparse_q_flips)
            if paths.full_sparse_equivalence_claimed
            else True
        ),
        "sparse_active_set_full_count": int(
            paths.sparse_diagnostics.get("sparse_active_set_full_count", -1),
        )
        == oracle_full_active_count,
        "oracle_source_independent": (
            oracle_source == "reference_recompute_compatibility_fallback"
        ),
    }
    actual_full_active_sha256 = ""
    oracle_full_active_sha256 = ""
    if bool(include_full_active_hash):
        actual_full_active_sha256 = _identity_universe_sha256(actual_full_active)
        oracle_full_active_sha256 = _identity_universe_sha256(oracle_full_active)
        checks.update(
            {
                "full_active_sha256": (
                    actual_full_active_sha256 == oracle_full_active_sha256
                ),
                "sparse_active_set_full_hash_computed": bool(
                    paths.sparse_diagnostics.get(
                        "sparse_active_set_full_hash_computed",
                        False,
                    ),
                ),
                "sparse_active_set_full_sha256": (
                    str(paths.sparse_diagnostics.get("sparse_active_set_full_sha256", ""))
                    == oracle_full_active_sha256
                ),
            },
        )
    else:
        checks.update(
            {
                "sparse_active_set_full_hash_not_computed": not bool(
                    paths.sparse_diagnostics.get(
                        "sparse_active_set_full_hash_computed",
                        False,
                    ),
                ),
                "sparse_active_set_full_sha256_empty": str(
                    paths.sparse_diagnostics.get("sparse_active_set_full_sha256", ""),
                )
                == "",
            },
        )
    if not all(checks.values()):
        failed = sorted(name for name, ok in checks.items() if not ok)
        raise AssertionError(
            "Front-C exact-reference oracle mismatch at step "
            f"{int(step)}: {failed}"
        )
    return {
        "enabled": True,
        "step": int(step),
        "oracle_source": "exact_reference_recompute_no_reused_plan",
        "oracle_path_source": oracle_source,
        "live_path_source": str(
            (paths.timing_diagnostics or {}).get("path_source", ""),
        ),
        "checks": checks,
        "full_active_count": full_active_count,
        "oracle_full_active_count": oracle_full_active_count,
        "full_active_hash_computed": bool(include_full_active_hash),
        "full_active_sha256": actual_full_active_sha256,
        "oracle_full_active_sha256": oracle_full_active_sha256,
        "full_active_hash_control": (
            "enabled_by_full_rebuild_or_legacy_control"
            if bool(include_full_active_hash)
            else "disabled_for_independent_oracle_perf_path"
        ),
        "emitted_decision_relevant_count": len(paths.emitted_decision_relevant_ids),
        "expected_full_identity_emission_claimed": expected_full_identity,
        "expected_full_sparse_equivalence_claimed": expected_full_sparse,
        "q_flip_receipt_parity_scope": q_flip_parity_scope,
        "dense_q_flip_receipt_sha256": _sha256_json(dense_q_flips),
        "sparse_q_flip_receipt_sha256": _sha256_json(sparse_q_flips),
        "oracle_reference_recompute_timing_seconds": dict(
            oracle_timing.get("durations_seconds", {}),
        ),
    }


def _truthy_env(name: str) -> bool:
    return str(os.environ.get(name, "")).strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class FrontCLiveIdentityCollector:
    artifact_path: Path
    emission_interval: int = 0
    audit_interval: int = 0
    cap_frontier_width: int = 1
    max_exact_identity_keys: int = DEFAULT_MAX_EXACT_IDENTITY_KEYS
    sparse_oracle_max_active_ids: int = DEFAULT_SPARSE_ORACLE_MAX_ACTIVE_IDS
    tensor_metadata_bits_per_state: int = 64
    bucket_metadata_bits: int = 64
    guardrail_metadata_bits: int = 64
    value_bits_per_row: int = 16
    flag_bits_per_row: int = 2
    legacy_oracle_compare: bool = False
    independent_oracle_compare: bool = False
    full_active_hash_oracle: bool = False

    def __post_init__(self) -> None:
        self.artifact_path = Path(self.artifact_path)
        self.emission_interval = int(self.emission_interval)
        self.audit_interval = int(self.audit_interval)
        self.cap_frontier_width = int(self.cap_frontier_width)
        self.max_exact_identity_keys = int(self.max_exact_identity_keys)
        self.sparse_oracle_max_active_ids = int(self.sparse_oracle_max_active_ids)
        self.tensor_metadata_bits_per_state = int(self.tensor_metadata_bits_per_state)
        self.bucket_metadata_bits = int(self.bucket_metadata_bits)
        self.guardrail_metadata_bits = int(self.guardrail_metadata_bits)
        self.value_bits_per_row = int(self.value_bits_per_row)
        self.flag_bits_per_row = int(self.flag_bits_per_row)
        self.full_active_hash_oracle = bool(
            self.full_active_hash_oracle
            or self.legacy_oracle_compare
            or _truthy_env(FRONT_C_ORACLE_FULL_REBUILD_ENV),
        )
        self.independent_oracle_compare = bool(
            self.independent_oracle_compare
            or _truthy_env(FRONT_C_INDEPENDENT_ORACLE_ENV)
            or self.full_active_hash_oracle,
        )
        self.legacy_oracle_compare = self.independent_oracle_compare
        self._step_rows: dict[int, FrontCDecisionSurfaceStep] = {}
        self._states_by_key: dict[str, VoteUpdateState] = {}
        self._current_threshold_indices_by_key: dict[str, torch.Tensor] = {}
        self._pending_threshold_add_indices_by_key: dict[str, torch.Tensor] = {}
        self._pending_threshold_remove_indices_by_key: dict[str, torch.Tensor] = {}
        self._current_threshold_count_by_key: dict[str, int] = {}
        self._current_threshold_initialized = False
        self._latest_dense_path: FrontCDecisionPath | None = None
        self._latest_sparse_path: FrontCDecisionPath | None = None
        self._dense_paths_by_step: dict[int, FrontCDecisionPath] = {}
        self._sparse_paths_by_step: dict[int, FrontCDecisionPath] = {}
        self._identity_emission_scope = FRONT_C_IDENTITY_SCOPE_EXACT
        self._full_identity_emission_claimed = True
        self._full_sparse_equivalence_claimed = True
        self._bounded_nonclaim_reasons: list[str] = []
        self._diagnostics: dict[str, Any] = {
            "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
            "global_cap_used": False,
            "global_cap_honest_false": True,
            "collector_return_ignored_by_learner": True,
            "observation_source": FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION,
            "identity_emission_scope": self._identity_emission_scope,
            "full_identity_emission_claimed": True,
            "full_sparse_equivalence_claimed": True,
            "max_exact_identity_keys": self.max_exact_identity_keys,
            "sparse_oracle_max_active_ids": self.sparse_oracle_max_active_ids,
            "legacy_oracle_compare_enabled": self.independent_oracle_compare,
            "independent_oracle_compare_enabled": self.independent_oracle_compare,
            "full_active_hash_oracle_enabled": self.full_active_hash_oracle,
            "collection_mode": "collection_cadence_current_threshold_rebuild",
            "observe_only_diagnostics": {},
            "collection_current_threshold_rebuild_by_step": {},
            "touched_count_by_step": {},
            "carried_threshold_count_by_step": {},
            "observe_only_duration_by_step": {},
            "touch_ratio_alarm_by_step": {},
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

    def _ensure_current_threshold_index(
        self,
        *,
        states_by_key: Mapping[str, VoteUpdateState],
        specs_by_key: Mapping[str, VoteUpdateSpec],
    ) -> dict[str, Any]:
        start = time.perf_counter()
        if self._current_threshold_initialized:
            return {
                "initialized": False,
                "duration_seconds": 0.0,
                "carried_threshold_count": sum(self._current_threshold_count_by_key.values()),
            }
        self._current_threshold_indices_by_key = {
            str(state_key): _current_threshold_indices(
                state,
                specs_by_key[str(state_key)],
            )
            for state_key, state in sorted(states_by_key.items())
        }
        self._pending_threshold_add_indices_by_key = {
            str(state_key): _empty_i64_indices()
            for state_key in self._current_threshold_indices_by_key
        }
        self._pending_threshold_remove_indices_by_key = {
            str(state_key): _empty_i64_indices()
            for state_key in self._current_threshold_indices_by_key
        }
        self._current_threshold_count_by_key = {
            str(state_key): int(indices.numel())
            for state_key, indices in self._current_threshold_indices_by_key.items()
        }
        self._current_threshold_initialized = True
        duration = max(0.0, time.perf_counter() - start)
        per_state = dict(sorted(self._current_threshold_count_by_key.items()))
        return {
            "initialized": True,
            "duration_seconds": duration,
            "carried_threshold_count": sum(per_state.values()),
            "per_state_carried_threshold_count": per_state,
        }

    def _materialize_current_threshold_indices_for_collect(
        self,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        start = time.perf_counter()
        out: dict[str, torch.Tensor] = {}
        per_state: dict[str, Any] = {}
        for key in sorted(self._current_threshold_indices_by_key):
            base = self._current_threshold_indices_by_key.get(key, _empty_i64_indices())
            pending_add = self._pending_threshold_add_indices_by_key.get(
                key,
                _empty_i64_indices(),
            )
            pending_remove = self._pending_threshold_remove_indices_by_key.get(
                key,
                _empty_i64_indices(),
            )
            materialized = _merge_index_delta(
                base,
                touched=pending_remove,
                active_after_touch=pending_add,
            )
            out[key] = materialized
            self._current_threshold_indices_by_key[key] = materialized
            self._pending_threshold_add_indices_by_key[key] = _empty_i64_indices()
            self._pending_threshold_remove_indices_by_key[key] = _empty_i64_indices()
            self._current_threshold_count_by_key[key] = int(materialized.numel())
            per_state[key] = {
                "materialized_count": int(materialized.numel()),
                "pending_add_count_before": int(pending_add.numel()),
                "pending_remove_count_before": int(pending_remove.numel()),
            }
        duration = max(0.0, time.perf_counter() - start)
        return out, {
            "materialized_for_collect": True,
            "duration_seconds": duration,
            "carried_threshold_count": sum(self._current_threshold_count_by_key.values()),
            "per_state": per_state,
        }

    def _touched_threshold_indices_by_key(
        self,
        *,
        states_by_key: Mapping[str, VoteUpdateState],
        plans_by_key: Mapping[str, VoteUpdatePlan],
        deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        backlog_by_key = {
            str(state_key): _sorted_unique_i64_indices(tuple(int(index) for index in by_index))
            for state_key, by_index in (deferred_backlog or {}).items()
        }
        touched_by_key: dict[str, torch.Tensor] = {}
        per_state: dict[str, Any] = {}
        for state_key in sorted(states_by_key):
            plan = plans_by_key[str(state_key)]
            pieces: list[torch.Tensor] = []

            def add_piece(indices: torch.Tensor) -> int:
                selected = _sorted_unique_i64_indices(indices)
                if int(selected.numel()) > 0:
                    pieces.append(selected)
                return int(selected.numel())

            candidate_count = add_piece(plan.candidate_indices)
            pre_veto_count = add_piece(plan.pre_veto_selected_indices)
            applied_count = add_piece(plan.applied_indices)
            replay_count = add_piece(plan.replay_ce_veto_indices)
            pc_negative_count = add_piece(plan.pc_aux_negative_indices)
            pc_veto_count = add_piece(plan.pc_aux_veto_indices)
            backlog_indices = backlog_by_key.get(str(state_key), _empty_i64_indices())
            if int(backlog_indices.numel()) > 0:
                pieces.append(backlog_indices)
            backlog_count = int(backlog_indices.numel())
            touched = (
                _sorted_unique_i64_indices(torch.cat(pieces))
                if pieces
                else _empty_i64_indices()
            )
            touched_by_key[str(state_key)] = touched
            applied_veto_delta_count = applied_count + replay_count + pc_veto_count
            per_state[str(state_key)] = {
                "touched_count": int(touched.numel()),
                "candidate_count": candidate_count,
                "pre_veto_count": pre_veto_count,
                "applied_count": applied_count,
                "replay_ce_veto_count": replay_count,
                "pc_aux_negative_count": pc_negative_count,
                "pc_aux_veto_count": pc_veto_count,
                "backlog_count": backlog_count,
                "applied_veto_delta_count": applied_veto_delta_count,
                "touched_semantics": (
                    "candidate/pre_veto/applied/replay_veto/pc_negative/"
                    "pc_veto/backlog rows only; prior threshold rows are not included"
                ),
            }
        return touched_by_key, per_state

    def _update_current_threshold_index(
        self,
        *,
        states_by_key: Mapping[str, VoteUpdateState],
        specs_by_key: Mapping[str, VoteUpdateSpec],
        plans_by_key: Mapping[str, VoteUpdatePlan],
        q_acc_by_key: Mapping[str, Any],
        deferred_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    ) -> dict[str, Any]:
        update_start = time.perf_counter()
        phase_start = time.perf_counter()
        touched_by_key, touched_diag = self._touched_threshold_indices_by_key(
            states_by_key=states_by_key,
            plans_by_key=plans_by_key,
            deferred_backlog=deferred_backlog,
        )
        touched_duration = max(0.0, time.perf_counter() - phase_start)
        phase_start = time.perf_counter()
        next_threshold_indices: dict[str, torch.Tensor] = {}
        per_state: dict[str, Any] = {}
        total_touched = 0
        total_carried = 0
        ratio_alarm = False
        for state_key in sorted(states_by_key):
            key = str(state_key)
            touched = touched_by_key.get(key, _empty_i64_indices())
            q_after, acc_after, _ = _q_acc_entry_parts(q_acc_by_key[key])
            active_after_touch = _threshold_indices_from_q_acc_rows(
                q_after,
                acc_after,
                specs_by_key[key],
                touched,
            )
            inactive_after_touch = _sorted_difference_i64(touched, active_after_touch)
            base = self._current_threshold_indices_by_key.get(key, _empty_i64_indices())
            pending_add = self._pending_threshold_add_indices_by_key.get(
                key,
                _empty_i64_indices(),
            )
            pending_remove = self._pending_threshold_remove_indices_by_key.get(
                key,
                _empty_i64_indices(),
            )
            touched_in_base = _sorted_probe_present_i64(base, touched)
            touched_in_pending_remove = _sorted_probe_present_i64(
                pending_remove,
                touched_in_base,
            )
            touched_in_pending_add = _sorted_probe_present_i64(pending_add, touched)
            old_active_touched_count = (
                int(touched_in_base.numel())
                - int(touched_in_pending_remove.numel())
                + int(touched_in_pending_add.numel())
            )
            active_missing_from_base = _sorted_probe_missing_i64(base, active_after_touch)
            inactive_present_in_base = _sorted_probe_present_i64(base, inactive_after_touch)
            pending_add = _sorted_union_i64(pending_add, active_missing_from_base)
            pending_add = _sorted_difference_i64(pending_add, inactive_after_touch)
            pending_remove = _sorted_difference_i64(pending_remove, active_after_touch)
            pending_remove = _sorted_union_i64(pending_remove, inactive_present_in_base)
            self._pending_threshold_add_indices_by_key[key] = pending_add
            self._pending_threshold_remove_indices_by_key[key] = pending_remove
            previous_count = int(
                self._current_threshold_count_by_key.get(key, int(base.numel())),
            )
            carried_count = (
                previous_count
                - old_active_touched_count
                + int(active_after_touch.numel())
            )
            self._current_threshold_count_by_key[key] = carried_count
            next_threshold_indices[key] = base
            eligible_count = int(q_after.numel())
            touched_count = int(touched.numel())
            applied_veto_delta_count = int(
                touched_diag[key].get("applied_veto_delta_count", 0),
            )
            state_alarm = bool(
                touched_count > 5 * applied_veto_delta_count
                or (eligible_count > 0 and touched_count > 0.10 * eligible_count),
            )
            ratio_alarm = ratio_alarm or state_alarm
            total_touched += touched_count
            total_carried += carried_count
            per_state[key] = {
                **dict(touched_diag[key]),
                "eligible_count": eligible_count,
                "active_after_touch_count": int(active_after_touch.numel()),
                "carried_threshold_count": carried_count,
                "old_active_touched_count": old_active_touched_count,
                "pending_add_count": int(pending_add.numel()),
                "pending_remove_count": int(pending_remove.numel()),
                "base_carried_count": int(base.numel()),
                "update_mode": "pending_overlay_touched_bounded",
                "touch_ratio_alarm": state_alarm,
                "touch_over_eligible_ratio": (
                    float(touched_count) / float(eligible_count)
                    if eligible_count
                    else 0.0
                ),
                "touch_over_applied_veto_delta_ratio": (
                    float(touched_count) / float(applied_veto_delta_count)
                    if applied_veto_delta_count
                    else None
                ),
            }
        self._current_threshold_indices_by_key.update(next_threshold_indices)
        merge_duration = max(0.0, time.perf_counter() - phase_start)
        total_duration = max(0.0, time.perf_counter() - update_start)
        return {
            "schema": FRONT_C_LIVE_TIMING_SCHEMA_VERSION,
            "touched_count": total_touched,
            "carried_threshold_count": total_carried,
            "touch_ratio_alarm": ratio_alarm,
            "per_state": per_state,
            "durations_seconds": {
                "touched_set_build": touched_duration,
                "carried_index_delta_merge": merge_duration,
                "observe_carried_index_update": total_duration,
                "carried_index_update": total_duration,
            },
        }

    def record_step_observation(
        self,
        *,
        step: int,
        observation: Mapping[str, Any],
        collect: bool = True,
    ) -> None:
        record_start = time.perf_counter()
        if observation.get("schema") != FRONT_C_LIVE_OBSERVATION_SCHEMA_VERSION:
            raise ValueError("unexpected Front-C observation schema")
        if bool(observation.get("global_cap_used", False)):
            raise ValueError("Front-C path-b emission must record global_cap_used=false")
        if not bool(collect):
            return
        states_by_key = dict(observation["states_by_key"])
        inputs_by_key = dict(observation["inputs_by_key"])
        specs_by_key = dict(observation["specs_by_key"])
        plans_by_key = dict(observation.get("plans_by_key", {}))
        q_acc_by_key = dict(observation.get("q_acc_by_key", {}))
        if not plans_by_key or not q_acc_by_key:
            raise ValueError("Front-C carried identity observer requires plans_by_key and q_acc_by_key")
        missing_plans = sorted(set(states_by_key) - set(plans_by_key))
        missing_q_acc = sorted(set(states_by_key) - set(q_acc_by_key))
        if missing_plans or missing_q_acc:
            raise ValueError(
                "Front-C carried identity observer missing plan/q_acc rows: "
                f"plans={missing_plans}, q_acc={missing_q_acc}",
            )
        self._states_by_key = states_by_key
        paths = build_front_c_live_step_paths(
            step=int(step),
            states_by_key=states_by_key,
            inputs_by_key=inputs_by_key,
            specs_by_key=specs_by_key,
            plans_by_key=plans_by_key,
            q_acc_by_key=q_acc_by_key,
            deferred_backlog=observation.get("deferred_backlog", {}),
            global_cap_used=False,
            cap_frontier_width=self.cap_frontier_width,
            max_exact_identity_keys=self.max_exact_identity_keys,
            sparse_oracle_max_active_ids=self.sparse_oracle_max_active_ids,
            current_threshold_indices_by_key=None,
            include_full_active_hash=self.full_active_hash_oracle,
        )
        step_key = str(int(step))
        self._diagnostics["observe_only_duration_by_step"][step_key] = 0.0
        surface_subtimers = dict(
            paths.surface_diagnostics.get("surface_build_subtimers_seconds", {}),
        )
        current_threshold_surface = paths.surface_diagnostics.get(
            "current_magnitude_threshold_keys",
            {},
        )
        collection_rebuild = {
            "schema": FRONT_C_LIVE_TIMING_SCHEMA_VERSION,
            "source": "pre_step_q_acc_scan",
            "current_threshold_scan_seconds": float(
                surface_subtimers.get("current_threshold_scan", 0.0),
            ),
            "current_threshold_count": int(
                dict(current_threshold_surface).get("full_identity_count", 0),
            ),
            "carried_index_used": False,
        }
        self._diagnostics["collection_current_threshold_rebuild_by_step"][
            step_key
        ] = collection_rebuild
        if (
            not paths.full_identity_emission_claimed
            or not paths.full_sparse_equivalence_claimed
        ):
            self._identity_emission_scope = paths.identity_emission_scope
            self._full_identity_emission_claimed = (
                self._full_identity_emission_claimed
                and paths.full_identity_emission_claimed
            )
            self._full_sparse_equivalence_claimed = (
                self._full_sparse_equivalence_claimed
                and paths.full_sparse_equivalence_claimed
            )
            self._bounded_nonclaim_reasons.extend(paths.bounded_nonclaim_reasons)
            self._diagnostics["identity_emission_scope"] = self._identity_emission_scope
            self._diagnostics["full_identity_emission_claimed"] = (
                self._full_identity_emission_claimed
            )
            self._diagnostics["full_sparse_equivalence_claimed"] = (
                self._full_sparse_equivalence_claimed
            )
            self._diagnostics["bounded_nonclaim_reasons"] = sorted(
                set(self._bounded_nonclaim_reasons),
            )
        self._step_rows[int(step)] = paths.surface
        self._latest_dense_path = paths.dense_path
        self._latest_sparse_path = paths.sparse_path
        self._dense_paths_by_step[int(step)] = paths.dense_path
        self._sparse_paths_by_step[int(step)] = paths.sparse_path
        timing = dict(
            paths.timing_diagnostics
            or _new_timing_diagnostics(phase="record_step_observation"),
        )
        timing["durations_seconds"] = dict(timing.get("durations_seconds", {}))
        timing["collection_mode"] = "collection_cadence_current_threshold_rebuild"
        legacy_oracle = {"enabled": False}
        if self.independent_oracle_compare:
            oracle_start = time.perf_counter()
            legacy_oracle = _assert_legacy_surface_oracle_match(
                step=int(step),
                paths=paths,
                states_by_key=states_by_key,
                inputs_by_key=inputs_by_key,
                specs_by_key=specs_by_key,
                deferred_backlog=observation.get("deferred_backlog", {}),
                cap_frontier_width=self.cap_frontier_width,
                max_exact_identity_keys=self.max_exact_identity_keys,
                sparse_oracle_max_active_ids=self.sparse_oracle_max_active_ids,
                include_full_active_hash=self.full_active_hash_oracle,
            )
            _record_duration(timing, "independent_oracle_reference_recompute", oracle_start)
        step_diagnostics = {
            "collect": True,
            "dense": paths.dense_diagnostics,
            "dense_q_flip_directions": paths.dense_path.to_dict()["q_flip_directions"],
            "sparse": paths.sparse_diagnostics,
            "sparse_q_flip_directions": paths.sparse_path.to_dict()["q_flip_directions"],
            "surface": paths.surface_diagnostics,
            "collection_current_threshold_rebuild": collection_rebuild,
            "carried_index_materialize_for_collect": {
                "materialized_for_collect": False,
                "duration_seconds": 0.0,
                "disabled_reason": "collection_cadence_rebuild_no_carried_index",
            },
            "carried_index_update": {
                "enabled": False,
                "duration_seconds": 0.0,
                "disabled_reason": "observer_not_installed_on_non_collected_steps",
            },
            "legacy_oracle": legacy_oracle,
            "identity_emission_scope": paths.identity_emission_scope,
            "full_identity_emission_claimed": paths.full_identity_emission_claimed,
            "full_sparse_equivalence_claimed": paths.full_sparse_equivalence_claimed,
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
        _record_duration(timing, "record_step_observation_total", record_start)
        step_diagnostics["timing"] = timing
        self._diagnostics["step_diagnostics"][str(int(step))] = step_diagnostics

    def _selected_timeline(self, audit_reports: Mapping[str, Any] | None) -> list[FrontCDecisionSurfaceStep]:
        rows = dict(self._step_rows)
        if not rows:
            raise ValueError("Front-C identity collector has no timeline rows")
        selected_steps: set[int] = {0} if 0 in rows else set()
        positive_steps = sorted(step for step in rows if step > 0)
        if positive_steps:
            selected_steps.add(positive_steps[0])
            selected_steps.add(positive_steps[-1])
            if self.audit_interval > 0:
                selected_steps.update(
                    step
                    for step in positive_steps
                    if step % int(self.audit_interval) == 0
                )
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
                "sparse_active_set_source": (
                    "dense_oracle_active_ids"
                    if self._full_sparse_equivalence_claimed
                    else "bounded_reused_plan_active_ids"
                ),
                "sparse_policy_selector_claimed": False,
                "sparse_decision_equivalence_scope": (
                    FRONT_C_SPARSE_EQUIVALENCE_EXACT
                    if self._full_sparse_equivalence_claimed
                    else FRONT_C_SPARSE_EQUIVALENCE_BOUNDED
                ),
                "identity_emission_scope": self._identity_emission_scope,
                "full_identity_emission_claimed": self._full_identity_emission_claimed,
                "full_sparse_equivalence_claimed": self._full_sparse_equivalence_claimed,
                "bounded_nonclaim_reasons": sorted(set(self._bounded_nonclaim_reasons)),
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
        finalize_start = time.perf_counter()
        finalize_timing = _new_timing_diagnostics(phase="front_c_finalize")
        phase_start = time.perf_counter()
        payload = self.build_payload(
            audit_reports=audit_reports,
            prior_audit_start_reports=prior_audit_start_reports,
            prior_audit_final_reports=prior_audit_final_reports,
            steps_completed=steps_completed,
            stop_reason=stop_reason,
        )
        _record_duration(finalize_timing, "build_payload", phase_start)
        payload["diagnostics"]["finalize_timing"] = finalize_timing
        phase_start = time.perf_counter()
        validation = validate_front_c_identity_artifact(payload)
        inventory = classify_front_c_identity_payload(
            payload,
            matched_artifact_path=str(self.artifact_path),
        )
        _record_duration(finalize_timing, "identity_validate", phase_start)
        bounded_nonclaim = (
            self._identity_emission_scope.startswith("bounded_")
            or not self._full_identity_emission_claimed
            or not self._full_sparse_equivalence_claimed
        )

        def persist_final_payload() -> None:
            finalize_timing["artifact_write_position"] = (
                "post_identity_validate_and_front_c_report_or_skip"
            )
            finalize_timing["authoritative"] = True
            finalize_timing["authoritative_timing_location"] = (
                "front_c_finalize_receipt.front_c_finalize_timing"
            )
            artifact_timing = {
                **finalize_timing,
                "authoritative": False,
                "artifact_embedded_timing_caveat": (
                    "Self-contained artifact timing is a pre-persist diagnostic "
                    "snapshot. Use front_c_finalize_receipt.front_c_finalize_timing "
                    "as the authoritative source for artifact_write and finalize_total; "
                    "the artifact cannot embed the cost of the write that serializes "
                    "those fields."
                ),
                "excluded_duration_keys_due_to_self_reference": [
                    "artifact_write",
                    "finalize_total",
                ],
                "durations_seconds": dict(finalize_timing["durations_seconds"]),
            }
            payload["diagnostics"]["finalize_timing"] = artifact_timing
            self.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            persist_start = time.perf_counter()
            self.artifact_path.write_text(
                json.dumps(payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            _record_duration(finalize_timing, "artifact_write", persist_start)
            _record_duration(finalize_timing, "finalize_total", finalize_start)

        phase_start = time.perf_counter()
        if bounded_nonclaim:
            _record_duration(finalize_timing, "front_c_report_or_skip", phase_start)
            persist_final_payload()
            return {
                "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
                "artifact_path": str(self.artifact_path),
                "artifact_sha256": hashlib.sha256(
                    self.artifact_path.read_bytes(),
                ).hexdigest(),
                "identity_validation": validation.to_dict(),
                "front_c_finalize_timing": finalize_timing,
                "front_c_report_skipped_bounded_nonclaim": {
                    "reason": "bounded_identity_artifact_is_structurally_nonclaimable",
                    "identity_emission_scope": self._identity_emission_scope,
                    "full_identity_emission_claimed": self._full_identity_emission_claimed,
                    "full_sparse_equivalence_claimed": self._full_sparse_equivalence_claimed,
                    "bounded_nonclaim_reasons": sorted(set(self._bounded_nonclaim_reasons)),
                },
                "inventory": inventory.to_dict(),
                "single_self_contained_artifact": True,
                "global_cap_used": False,
                "gpu_launched": False,
                "pt_artifact_written": False,
            }
        report = front_c_report_from_identity_artifact(payload)
        _record_duration(finalize_timing, "front_c_report_or_skip", phase_start)
        persist_final_payload()
        return {
            "schema": FRONT_C_LIVE_IDENTITY_EMISSION_SCHEMA_VERSION,
            "artifact_path": str(self.artifact_path),
            "artifact_sha256": hashlib.sha256(
                self.artifact_path.read_bytes(),
            ).hexdigest(),
            "identity_validation": validation.to_dict(),
            "front_c_finalize_timing": finalize_timing,
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
