"""Deterministic stratified selector manifest for D recompute-window instrumentation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    _shadow_numel,
    default_production_replay_constants,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateState

STRATIFIED_SELECTOR_SCHEMA_VERSION = (
    "hrm_text_158_d_recompute_window_stratified_selector/v0"
)

COVERAGE_TIER_REPRESENTATIVE = "REPRESENTATIVE"
COVERAGE_TIER_PILOT = "PILOT_NOT_REPRESENTATIVE_SIZING"

MIN_REPRESENTATIVE_KEYS = 12
MAX_REPRESENTATIVE_KEYS = 18
PILOT_MIN_KEYS = 8
LANES_PER_KEY = 32
UNIFORM_LANE_COUNT = 16
STRESS_LANE_COUNT = 16
PILOT_MIN_LANES_PER_KEY = 16

STRESS_TAIL_POLICY_HORIZON_FIXED = "horizon_fixed_warmup_calibrated_v0"
STRESS_TAIL_POLICY_DYNAMIC = "per_step_vote_aware_v0"
DEFAULT_CALIBRATION_WARMUP_STEPS = 5

EXCLUDED_KEY_FRAGMENTS: tuple[str, ...] = (
    ".embed",
    "embeddings",
    "lm_head",
    ".norm",
    "rotary",
    "pos_embed",
    ".pos.",
)

DEPTH_TERCILE_LABELS: tuple[str, ...] = ("low", "mid", "high")
NUMEL_BAND_LABELS: tuple[str, ...] = ("small", "median", "large")

ROLE_PATTERNS: tuple[tuple[str, str], ...] = (
    (".attn.q_proj", "attn_q"),
    (".attn.k_proj", "attn_k"),
    (".attn.v_proj", "attn_v"),
    (".attn.gqkv_proj", "attn_gqkv"),
    (".attn.o_proj", "attn_o"),
    (".mlp.gate_up_proj", "mlp_gate_up"),
    (".mlp.down_proj", "mlp_down"),
)

_LAYER_IDX_RE = re.compile(r"\.layers\.(\d+)\.")


@dataclass(frozen=True)
class CalibrationWarmupStep:
    step: int
    observations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]]


@dataclass(frozen=True)
class StratifiedKeyEntry:
    state_key: str
    level: str
    layer_idx: int
    role: str
    depth_tercile: str
    numel_band: str
    numel: int
    uniform_lanes: tuple[int, ...]
    stress_tail_lanes: tuple[int, ...]
    lane_indices: tuple[int, ...]
    stratum_weight: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "state_key": self.state_key,
            "level": self.level,
            "layer_idx": int(self.layer_idx),
            "role": self.role,
            "depth_tercile": self.depth_tercile,
            "numel_band": self.numel_band,
            "numel": int(self.numel),
            "uniform_lanes": [int(index) for index in self.uniform_lanes],
            "stress_tail_lanes": [int(index) for index in self.stress_tail_lanes],
            "lane_indices": [int(index) for index in self.lane_indices],
            "stratum_weight": float(self.stratum_weight),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StratifiedKeyEntry:
        return cls(
            state_key=str(payload["state_key"]),
            level=str(payload["level"]),
            layer_idx=int(payload["layer_idx"]),
            role=str(payload["role"]),
            depth_tercile=str(payload["depth_tercile"]),
            numel_band=str(payload["numel_band"]),
            numel=int(payload["numel"]),
            uniform_lanes=tuple(int(index) for index in payload["uniform_lanes"]),
            stress_tail_lanes=tuple(int(index) for index in payload["stress_tail_lanes"]),
            lane_indices=tuple(int(index) for index in payload["lane_indices"]),
            stratum_weight=float(payload["stratum_weight"]),
        )


@dataclass(frozen=True)
class StratifiedSelectorManifest:
    schema_version: str
    manifest_sha256: str
    coverage_tier: str
    selected_key_count: int
    stratum_weights: dict[str, float]
    entries: tuple[StratifiedKeyEntry, ...]
    manifest_spec: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "manifest_sha256": self.manifest_sha256,
            "coverage_tier": self.coverage_tier,
            "selected_key_count": int(self.selected_key_count),
            "stratum_weights": {
                str(key): float(value) for key, value in sorted(self.stratum_weights.items())
            },
            "entries": [entry.to_dict() for entry in self.entries],
            "manifest_spec": dict(self.manifest_spec),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> StratifiedSelectorManifest:
        entries = tuple(
            StratifiedKeyEntry.from_dict(entry)
            for entry in payload.get("entries", ())
        )
        return cls(
            schema_version=str(payload["schema_version"]),
            manifest_sha256=str(payload["manifest_sha256"]),
            coverage_tier=str(payload["coverage_tier"]),
            selected_key_count=int(payload["selected_key_count"]),
            stratum_weights={
                str(key): float(value)
                for key, value in dict(payload.get("stratum_weights", {})).items()
            },
            entries=entries,
            manifest_spec=dict(payload.get("manifest_spec", {})),
        )

    def entry_by_key(self) -> dict[str, StratifiedKeyEntry]:
        return {entry.state_key: entry for entry in self.entries}


def _manifest_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _is_excluded_state_key(state_key: str) -> bool:
    lowered = str(state_key).lower()
    return any(fragment in lowered for fragment in EXCLUDED_KEY_FRAGMENTS)


def _is_eligible_tensor_state(state: Any) -> bool:
    return isinstance(state, (BoundedDeltaTensorState, VoteUpdateState))


def _parse_level(state_key: str) -> str | None:
    if ".H_level." in state_key:
        return "H"
    if ".L_level." in state_key:
        return "L"
    return None


def _parse_layer_idx(state_key: str) -> int | None:
    match = _LAYER_IDX_RE.search(state_key)
    if match is None:
        return None
    return int(match.group(1))


def _parse_role(state_key: str) -> str | None:
    for pattern, role in ROLE_PATTERNS:
        if pattern in state_key:
            return role
    return None


def _tercile_label(value: int, sorted_values: Sequence[int]) -> str:
    if not sorted_values:
        return DEPTH_TERCILE_LABELS[0]
    unique_sorted = sorted(set(int(item) for item in sorted_values))
    if len(unique_sorted) == 1:
        return DEPTH_TERCILE_LABELS[1]
    lo = unique_sorted[0]
    hi = unique_sorted[-1]
    span = max(1, hi - lo)
    ratio = (int(value) - lo) / float(span)
    if ratio < 1.0 / 3.0:
        return DEPTH_TERCILE_LABELS[0]
    if ratio < 2.0 / 3.0:
        return DEPTH_TERCILE_LABELS[1]
    return DEPTH_TERCILE_LABELS[2]


def _numel_band(numel: int, band_values: Sequence[int]) -> str:
    unique_sorted = sorted(set(int(item) for item in band_values))
    if len(unique_sorted) <= 1:
        return NUMEL_BAND_LABELS[1]
    lo = unique_sorted[0]
    hi = unique_sorted[-1]
    span = max(1, hi - lo)
    ratio = (int(numel) - lo) / float(span)
    if ratio < 1.0 / 3.0:
        return NUMEL_BAND_LABELS[0]
    if ratio < 2.0 / 3.0:
        return NUMEL_BAND_LABELS[1]
    return NUMEL_BAND_LABELS[2]


def _accumulator_lane_values(
    state: Any,
    *,
    replay_constants: ReplayConstants,
) -> list[int]:
    clip_min = int(replay_constants.accumulator_clip_min)
    clip_max = int(replay_constants.accumulator_clip_max)
    if isinstance(state, VoteUpdateState):
        flat = state.accumulators.detach().cpu().flatten().to(torch.int32)
        return [int(flat[index].item()) for index in range(flat.numel())]
    if isinstance(state, BoundedDeltaTensorState):
        shadow = state.exact_accumulator_shadow
        if shadow is None:
            raise ValueError(f"exact_accumulator_shadow missing for {state.state_key!r}")
        flat = shadow.detach().cpu().flatten()
        values = [int(flat[index].item()) for index in range(flat.numel())]
        for value in values:
            if value < clip_min or value > clip_max:
                raise ValueError(
                    f"shadow lane {value} outside clip [{clip_min}, {clip_max}] "
                    f"for {state.state_key!r}"
                )
        return values
    raise TypeError(f"unsupported tensor state type {type(state)!r}")


def _uniform_stride_indices(numel: int, *, count: int) -> list[int]:
    if numel <= 0:
        return []
    if numel <= count:
        return list(range(numel))
    stride = max(1, numel // int(count))
    indices = list(range(0, numel, stride))[: int(count)]
    if (numel - 1) not in indices:
        indices.append(numel - 1)
    return sorted(set(indices))[: int(count)]


def _near_threshold_distance(value: int, *, threshold_abs: int) -> int:
    threshold = int(threshold_abs)
    return min(
        abs(int(value) - threshold),
        abs(int(value) + threshold),
        abs(int(value)),
    )


def _stress_tail_score(
    acc: int,
    vote: int,
    *,
    threshold_abs: int,
    index: int,
) -> tuple[int, int, int, int, int]:
    """Deterministic stress-tail ranking keyed by |acc|, |vote|, near-threshold.

    Descending sort uses:
    1. max(|acc|, |vote|) — either high current magnitude or update pressure
    2. |vote| — update-pressure tie-break / displacement
    3. |acc| — current-magnitude secondary
    4. near-threshold proximity (closer is higher)
    5. stable index tie-break
    """
    acc_abs = abs(int(acc))
    vote_abs = abs(int(vote))
    return (
        max(acc_abs, vote_abs),
        vote_abs,
        acc_abs,
        -_near_threshold_distance(acc, threshold_abs=int(threshold_abs)),
        -int(index),
    )


def _vote_lane_values(
    vote_values: Sequence[int] | None,
    *,
    numel: int,
) -> list[int]:
    if vote_values is None:
        return [0] * int(numel)
    if len(vote_values) != int(numel):
        raise ValueError(
            f"vote_values length {len(vote_values)} != numel {numel}"
        )
    return [int(value) for value in vote_values]


def _stress_tail_indices(
    state: Any,
    *,
    numel: int,
    exclude: set[int],
    count: int,
    replay_constants: ReplayConstants,
    vote_values: Sequence[int] | None = None,
) -> list[int]:
    if count <= 0 or numel <= 0:
        return []
    acc_values = _accumulator_lane_values(state, replay_constants=replay_constants)
    votes = _vote_lane_values(vote_values, numel=numel)
    threshold = int(replay_constants.threshold_abs)
    candidates = [index for index in range(numel) if index not in exclude]
    scored: list[tuple[tuple[int, int, int, int, int], int]] = []
    for index in candidates:
        acc = int(acc_values[index])
        vote = int(votes[index])
        score = _stress_tail_score(
            acc,
            vote,
            threshold_abs=threshold,
            index=index,
        )
        scored.append((score, index))
    scored.sort(reverse=True)
    return [index for _, index in scored[: int(count)]]


def rank_dynamic_stress_tail_lanes(
    state: Any,
    *,
    numel: int,
    exclude: set[int],
    count: int,
    replay_constants: ReplayConstants,
    vote_values: Sequence[int] | None = None,
) -> list[int]:
    """Per-step dynamic stress-tail ranking (calibration helper / pressure annotation)."""
    return _stress_tail_indices(
        state,
        numel=numel,
        exclude=exclude,
        count=count,
        replay_constants=replay_constants,
        vote_values=vote_values,
    )


def extract_calibration_observation(
    state_key: str,
    state: Any,
    vote_tensor: torch.Tensor,
    *,
    replay_constants: ReplayConstants,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    acc_values = tuple(
        _accumulator_lane_values(state, replay_constants=replay_constants)
    )
    votes_flat = vote_tensor.detach().cpu().flatten().to(torch.int32)
    vote_values = tuple(int(votes_flat[index].item()) for index in range(votes_flat.numel()))
    if len(acc_values) != len(vote_values):
        raise ValueError(
            f"calibration observation length mismatch for {state_key!r}: "
            f"acc={len(acc_values)} vote={len(vote_values)}"
        )
    return acc_values, vote_values


def _max_stress_score(
    left: tuple[int, int, int, int, int],
    right: tuple[int, int, int, int, int],
) -> tuple[int, int, int, int, int]:
    return left if left >= right else right


def _aggregate_lane_stress_scores_for_key(
    state_key: str,
    *,
    numel: int,
    uniform_lanes: Sequence[int],
    calibration_samples: Sequence[CalibrationWarmupStep],
    replay_constants: ReplayConstants,
) -> dict[int, tuple[int, int, int, int, int]]:
    exclude = {int(index) for index in uniform_lanes}
    threshold = int(replay_constants.threshold_abs)
    aggregated: dict[int, tuple[int, int, int, int, int]] = {}
    for sample in calibration_samples:
        observation = sample.observations.get(state_key)
        if observation is None:
            continue
        acc_values, vote_values = observation
        if len(acc_values) != int(numel) or len(vote_values) != int(numel):
            raise ValueError(
                f"calibration sample step {sample.step} length mismatch for "
                f"{state_key!r}: acc={len(acc_values)} vote={len(vote_values)} "
                f"expected numel={numel}"
            )
        for index in range(int(numel)):
            if index in exclude:
                continue
            score = _stress_tail_score(
                int(acc_values[index]),
                int(vote_values[index]),
                threshold_abs=threshold,
                index=index,
            )
            prior = aggregated.get(index)
            aggregated[index] = score if prior is None else _max_stress_score(prior, score)
    return aggregated


def _fixed_stress_lanes_from_calibration(
    state_key: str,
    *,
    numel: int,
    uniform_lanes: Sequence[int],
    calibration_samples: Sequence[CalibrationWarmupStep],
    replay_constants: ReplayConstants,
    count: int = STRESS_LANE_COUNT,
) -> list[int]:
    aggregated = _aggregate_lane_stress_scores_for_key(
        state_key,
        numel=numel,
        uniform_lanes=uniform_lanes,
        calibration_samples=calibration_samples,
        replay_constants=replay_constants,
    )
    if not aggregated:
        raise ValueError(
            f"no calibration observations for {state_key!r}; cannot derive fixed stress lanes"
        )
    scored = sorted(
        ((score, index) for index, score in aggregated.items()),
        reverse=True,
    )
    return [int(index) for _, index in scored[: int(count)]]


def sample_lanes_for_key(
    state: Any,
    *,
    manifest_entry: StratifiedKeyEntry,
    vote_values: Sequence[int] | None = None,
    replay_constants: ReplayConstants | None = None,
    stress_tail_policy: str | None = None,
) -> list[int]:
    replay = replay_constants or default_production_replay_constants()
    numel = int(_shadow_numel(state))
    if numel != int(manifest_entry.numel):
        uniform = _uniform_stride_indices(numel, count=UNIFORM_LANE_COUNT)
    else:
        uniform = list(manifest_entry.uniform_lanes)

    if stress_tail_policy == STRESS_TAIL_POLICY_HORIZON_FIXED:
        if numel != int(manifest_entry.numel):
            raise ValueError(
                "horizon-fixed selector manifest numel mismatch for "
                f"{manifest_entry.state_key!r}: live={numel} "
                f"manifest={int(manifest_entry.numel)}; "
                "fixed stress lanes cannot be rebound dynamically"
            )
        stress = list(manifest_entry.stress_tail_lanes)
    else:
        stress = rank_dynamic_stress_tail_lanes(
            state,
            numel=numel,
            exclude=set(uniform),
            count=STRESS_LANE_COUNT,
            replay_constants=replay,
            vote_values=vote_values,
        )
    combined = list(uniform) + [index for index in stress if index not in set(uniform)]
    return [int(index) for index in combined]


def _build_lane_entry(
    state_key: str,
    state: Any,
    *,
    level: str,
    layer_idx: int,
    role: str,
    depth_tercile: str,
    numel_band: str,
    replay_constants: ReplayConstants,
) -> StratifiedKeyEntry:
    numel = int(_shadow_numel(state))
    uniform = _uniform_stride_indices(numel, count=UNIFORM_LANE_COUNT)
    stress = _stress_tail_indices(
        state,
        numel=numel,
        exclude=set(uniform),
        count=STRESS_LANE_COUNT,
        replay_constants=replay_constants,
    )
    lane_indices = list(uniform) + [index for index in stress if index not in set(uniform)]
    return StratifiedKeyEntry(
        state_key=str(state_key),
        level=str(level),
        layer_idx=int(layer_idx),
        role=str(role),
        depth_tercile=str(depth_tercile),
        numel_band=str(numel_band),
        numel=numel,
        uniform_lanes=tuple(int(index) for index in uniform),
        stress_tail_lanes=tuple(int(index) for index in stress),
        lane_indices=tuple(int(index) for index in lane_indices),
        stratum_weight=0.0,
    )


def _eligible_candidates(
    tensor_states: Mapping[str, Any],
) -> list[tuple[str, Any, str, int, str, int]]:
    candidates: list[tuple[str, Any, str, int, str, int]] = []
    for state_key in sorted(tensor_states.keys()):
        if _is_excluded_state_key(state_key):
            continue
        state = tensor_states[state_key]
        if not _is_eligible_tensor_state(state):
            continue
        level = _parse_level(state_key)
        layer_idx = _parse_layer_idx(state_key)
        role = _parse_role(state_key)
        if level is None or layer_idx is None or role is None:
            continue
        numel = int(_shadow_numel(state))
        candidates.append((str(state_key), state, level, layer_idx, role, numel))
    return candidates


def _round_robin_select_keys(
    candidates: Sequence[tuple[str, Any, str, int, str, int]],
    *,
    min_keys: int,
    max_keys: int,
    replay_constants: ReplayConstants,
) -> list[StratifiedKeyEntry]:
    if not candidates:
        return []

    layers_by_level: dict[str, list[int]] = {"H": [], "L": []}
    numel_by_level_role: dict[tuple[str, str], list[int]] = {}
    for _, _, level, layer_idx, role, numel in candidates:
        layers_by_level.setdefault(level, []).append(int(layer_idx))
        numel_by_level_role.setdefault((level, role), []).append(int(numel))

    annotated: list[tuple[tuple[str, str, str, str], StratifiedKeyEntry]] = []
    for state_key, state, level, layer_idx, role, numel in candidates:
        depth_tercile = _tercile_label(layer_idx, layers_by_level[level])
        numel_band = _numel_band(numel, numel_by_level_role[(level, role)])
        entry = _build_lane_entry(
            state_key,
            state,
            level=level,
            layer_idx=layer_idx,
            role=role,
            depth_tercile=depth_tercile,
            numel_band=numel_band,
            replay_constants=replay_constants,
        )
        stratum = (level, depth_tercile, role, numel_band)
        annotated.append((stratum, entry))

    strata = sorted({stratum for stratum, _ in annotated})
    keys_by_stratum: dict[tuple[str, str, str, str], list[StratifiedKeyEntry]] = {
        stratum: [] for stratum in strata
    }
    for stratum, entry in annotated:
        keys_by_stratum[stratum].append(entry)
    for stratum in strata:
        keys_by_stratum[stratum] = sorted(
            keys_by_stratum[stratum],
            key=lambda item: item.state_key,
        )

    selected: list[StratifiedKeyEntry] = []
    stratum_idx = 0
    guard = 0
    target_max = min(int(max_keys), len(annotated))
    while len(selected) < target_max and guard < len(annotated) * len(strata) + 1:
        guard += 1
        if not strata:
            break
        stratum = strata[stratum_idx % len(strata)]
        bucket = keys_by_stratum[stratum]
        if bucket:
            selected.append(bucket.pop(0))
        stratum_idx += 1
        if all(not keys_by_stratum[item] for item in strata):
            break

    if len(selected) < int(min_keys) and len(annotated) >= int(min_keys):
        selected_keys = {entry.state_key for entry in selected}
        for _, entry in sorted(annotated, key=lambda item: item[1].state_key):
            if entry.state_key in selected_keys:
                continue
            selected.append(entry)
            selected_keys.add(entry.state_key)
            if len(selected) >= int(min_keys):
                break

    selected = sorted(selected, key=lambda item: item.state_key)[: int(max_keys)]
    total_numel = sum(int(entry.numel) for entry in selected) or 1
    weighted: list[StratifiedKeyEntry] = []
    for entry in selected:
        weighted.append(
            StratifiedKeyEntry(
                state_key=entry.state_key,
                level=entry.level,
                layer_idx=entry.layer_idx,
                role=entry.role,
                depth_tercile=entry.depth_tercile,
                numel_band=entry.numel_band,
                numel=entry.numel,
                uniform_lanes=entry.uniform_lanes,
                stress_tail_lanes=entry.stress_tail_lanes,
                lane_indices=entry.lane_indices,
                stratum_weight=float(entry.numel) / float(total_numel),
            )
        )
    return weighted


def _coverage_tier(entries: Sequence[StratifiedKeyEntry]) -> str:
    if len(entries) < PILOT_MIN_KEYS:
        return COVERAGE_TIER_PILOT
    for entry in entries:
        if len(entry.uniform_lanes) < PILOT_MIN_LANES_PER_KEY:
            return COVERAGE_TIER_PILOT
        if len(entry.stress_tail_lanes) < PILOT_MIN_LANES_PER_KEY:
            return COVERAGE_TIER_PILOT
        overlap = set(entry.uniform_lanes) & set(entry.stress_tail_lanes)
        if overlap:
            return COVERAGE_TIER_PILOT
        if len(entry.lane_indices) < LANES_PER_KEY:
            return COVERAGE_TIER_PILOT
    if len(entries) < MIN_REPRESENTATIVE_KEYS:
        return COVERAGE_TIER_PILOT
    return COVERAGE_TIER_REPRESENTATIVE


def _finalize_manifest_with_calibrated_stress_lanes(
    manifest: StratifiedSelectorManifest,
    *,
    calibration_samples: Sequence[CalibrationWarmupStep],
    replay_constants: ReplayConstants,
    stress_tail_policy: str,
    measurement_start_step: int = 1,
) -> StratifiedSelectorManifest:
    spec = dict(manifest.manifest_spec)
    spec["stress_tail_policy"] = str(stress_tail_policy)
    spec["calibration_warmup_steps"] = len(calibration_samples)
    spec["measurement_start_step"] = int(measurement_start_step)
    spec["calibration_discarded_before_measurement"] = True

    new_entries: list[StratifiedKeyEntry] = []
    for entry in manifest.entries:
        fixed_stress = _fixed_stress_lanes_from_calibration(
            entry.state_key,
            numel=entry.numel,
            uniform_lanes=entry.uniform_lanes,
            calibration_samples=calibration_samples,
            replay_constants=replay_constants,
        )
        lane_indices = list(entry.uniform_lanes) + [
            index for index in fixed_stress if index not in set(entry.uniform_lanes)
        ]
        new_entries.append(
            StratifiedKeyEntry(
                state_key=entry.state_key,
                level=entry.level,
                layer_idx=entry.layer_idx,
                role=entry.role,
                depth_tercile=entry.depth_tercile,
                numel_band=entry.numel_band,
                numel=entry.numel,
                uniform_lanes=entry.uniform_lanes,
                stress_tail_lanes=tuple(int(index) for index in fixed_stress),
                lane_indices=tuple(int(index) for index in lane_indices),
                stratum_weight=entry.stratum_weight,
            )
        )

    coverage_tier = _coverage_tier(new_entries)
    stratum_weights = {entry.state_key: float(entry.stratum_weight) for entry in new_entries}
    body_without_digest = {
        "schema_version": manifest.schema_version,
        "coverage_tier": coverage_tier,
        "selected_key_count": len(new_entries),
        "stratum_weights": stratum_weights,
        "entries": [entry.to_dict() for entry in new_entries],
        "manifest_spec": spec,
    }
    manifest_sha256 = _manifest_digest(body_without_digest)
    return StratifiedSelectorManifest(
        schema_version=manifest.schema_version,
        manifest_sha256=manifest_sha256,
        coverage_tier=coverage_tier,
        selected_key_count=len(new_entries),
        stratum_weights=stratum_weights,
        entries=tuple(new_entries),
        manifest_spec=spec,
    )


def build_calibrated_stratified_selector_manifest(
    tensor_states: Mapping[str, Any],
    *,
    calibration_samples: Sequence[CalibrationWarmupStep],
    manifest_spec: Mapping[str, Any] | None = None,
    replay_constants: ReplayConstants | None = None,
    measurement_start_step: int = 1,
) -> StratifiedSelectorManifest:
    replay = replay_constants or default_production_replay_constants()
    base = build_stratified_selector_manifest(
        tensor_states,
        manifest_spec=manifest_spec,
        replay_constants=replay,
    )
    return _finalize_manifest_with_calibrated_stress_lanes(
        base,
        calibration_samples=calibration_samples,
        replay_constants=replay,
        stress_tail_policy=STRESS_TAIL_POLICY_HORIZON_FIXED,
        measurement_start_step=measurement_start_step,
    )


def build_stratified_selector_manifest(
    tensor_states: Mapping[str, Any],
    *,
    manifest_spec: Mapping[str, Any] | None = None,
    replay_constants: ReplayConstants | None = None,
) -> StratifiedSelectorManifest:
    replay = replay_constants or default_production_replay_constants()
    spec = dict(manifest_spec or {})
    min_keys = int(spec.get("min_keys", MIN_REPRESENTATIVE_KEYS))
    max_keys = int(spec.get("max_keys", MAX_REPRESENTATIVE_KEYS))
    candidates = _eligible_candidates(tensor_states)
    entries = _round_robin_select_keys(
        candidates,
        min_keys=min_keys,
        max_keys=max_keys,
        replay_constants=replay,
    )
    coverage_tier = _coverage_tier(entries)
    stratum_weights = {
        entry.state_key: float(entry.stratum_weight) for entry in entries
    }
    body_without_digest = {
        "schema_version": STRATIFIED_SELECTOR_SCHEMA_VERSION,
        "coverage_tier": coverage_tier,
        "selected_key_count": len(entries),
        "stratum_weights": stratum_weights,
        "entries": [entry.to_dict() for entry in entries],
        "manifest_spec": spec,
    }
    manifest_sha256 = _manifest_digest(body_without_digest)
    return StratifiedSelectorManifest(
        schema_version=STRATIFIED_SELECTOR_SCHEMA_VERSION,
        manifest_sha256=manifest_sha256,
        coverage_tier=coverage_tier,
        selected_key_count=len(entries),
        stratum_weights=stratum_weights,
        entries=tuple(entries),
        manifest_spec=spec,
    )


def select_instrumentation_state_keys_from_manifest(
    tensor_states: Mapping[str, Any],
    manifest: StratifiedSelectorManifest,
) -> list[str]:
    selected: list[str] = []
    for entry in manifest.entries:
        if entry.state_key not in tensor_states:
            raise KeyError(
                f"selector manifest key {entry.state_key!r} missing from tensor_states"
            )
        selected.append(entry.state_key)
    return selected


def save_stratified_selector_manifest(
    manifest: StratifiedSelectorManifest,
    path: Path | str,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = manifest.to_dict()
    target.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    return target


def load_stratified_selector_manifest(path: Path | str) -> StratifiedSelectorManifest:
    target = Path(path)
    payload = json.loads(target.read_text(encoding="utf-8"))
    manifest = StratifiedSelectorManifest.from_dict(payload)
    recomputed = _manifest_digest(
        {
            "schema_version": manifest.schema_version,
            "coverage_tier": manifest.coverage_tier,
            "selected_key_count": manifest.selected_key_count,
            "stratum_weights": manifest.stratum_weights,
            "entries": [entry.to_dict() for entry in manifest.entries],
            "manifest_spec": manifest.manifest_spec,
        }
    )
    if recomputed != manifest.manifest_sha256:
        raise ValueError(
            "selector manifest sha256 mismatch: "
            f"expected {manifest.manifest_sha256}, got {recomputed}"
        )
    return manifest
