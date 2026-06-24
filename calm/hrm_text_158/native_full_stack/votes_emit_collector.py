"""Per-step votes observables emitter for dynamics-proof runs."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import BoundedDeltaTensorState
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    build_compact_within_tie_band_sampled_table_rows,
    build_within_tie_band_candidate_universe_from_votes,
)
from calm.hrm_text_158.native_full_stack.two_tier_step_orchestrator import (
    derive_warmup_apply_tags_from_applied_abs_new_acc,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    frozen_threshold_semantics_block,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    plan_integer_vote_update_reference,
)


VOTES_EMIT_SCHEMA_VERSION = "hrm_text_158_votes_emit/v0"
VOTES_EMIT_MAX_SAMPLED_ROWS = 32
VOTES_EMIT_ENABLED_ENV = "HRM_TEXT_158_VOTES_EMIT_ENABLED"
VOTES_EMIT_ROOT_ENV = "HRM_TEXT_158_VOTES_EMIT_ROOT"


def votes_emit_enabled(*, enabled: bool | None = None) -> bool:
    if enabled is not None:
        return bool(enabled)
    import os

    return str(os.environ.get(VOTES_EMIT_ENABLED_ENV, "")).strip() == "1"


def _canonical_json(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _deterministic_sampled_candidates(
    universe: Mapping[str, Any],
    *,
    max_sampled_rows: int = VOTES_EMIT_MAX_SAMPLED_ROWS,
) -> list[dict[str, Any]]:
    candidate_by_id = dict(universe["candidate_by_id"])
    sampled_ids = list(universe["sampled_ids"])
    sampled_candidates = [dict(candidate_by_id[str(candidate_id)]) for candidate_id in sampled_ids]
    sampled_candidates.sort(
        key=lambda candidate: (
            int(candidate["current_rank_position"]),
            int(candidate["flat_index"]),
            str(candidate["state_key"]),
            str(candidate["candidate_id"]),
        )
    )
    return sampled_candidates[: int(max_sampled_rows)]


def _preview_applied_flip_count(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    two_tier_carry_w6_enabled: bool,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None,
    local_selection_ordering_seed: int,
    optimizer_step_index: int,
) -> int:
    total = 0
    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        inputs = VoteUpdateInputs(
            votes=votes_by_key[state_key],
            local_loss_delta=(
                local_loss_delta_by_key[state_key]
                if local_loss_delta_by_key is not None
                else None
            ),
        )
        plan = plan_integer_vote_update_reference(
            vote_state,
            inputs,
            vote_specs_by_key[state_key],
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(optimizer_step_index),
            two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        )
        total += int(plan.applied_indices.numel())
    return int(total)


def _preview_warmup_tags(
    *,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    two_tier_carry_w6_enabled: bool,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None,
    local_selection_ordering_seed: int,
    optimizer_step_index: int,
) -> dict[str, Any]:
    applied_abs_values: list[int] = []
    for state_key, state in sorted(tensor_states.items()):
        vote_state = state.vote_update_state()
        inputs = VoteUpdateInputs(
            votes=votes_by_key[state_key],
            local_loss_delta=(
                local_loss_delta_by_key[state_key]
                if local_loss_delta_by_key is not None
                else None
            ),
        )
        plan = plan_integer_vote_update_reference(
            vote_state,
            inputs,
            vote_specs_by_key[state_key],
            local_selection_ordering_seed=int(local_selection_ordering_seed),
            local_selection_ordering_step=int(optimizer_step_index),
            two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        )
        flat_new_acc = plan.new_acc_i32.detach().cpu().flatten()
        applied_indices = [
            int(value) for value in plan.applied_indices.detach().cpu().tolist()
        ]
        for flat_index in applied_indices:
            applied_abs_values.append(int(abs(int(flat_new_acc[int(flat_index)].item()))))
    return derive_warmup_apply_tags_from_applied_abs_new_acc(applied_abs_values)


def build_votes_emit_step_record(
    *,
    optimizer_step_index: int,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    max_abs_per_tensor: int,
    two_tier_carry_w6_enabled: bool = False,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None = None,
    local_selection_ordering_seed: int = 0,
) -> dict[str, Any]:
    universe = build_within_tie_band_candidate_universe_from_votes(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        max_sampled_candidates=int(VOTES_EMIT_MAX_SAMPLED_ROWS),
    )
    sampled_candidates = _deterministic_sampled_candidates(universe)
    for candidate in sampled_candidates:
        candidate.setdefault("candidate_loss", 0.0)
        candidate.setdefault("local_loss_delta", 0.0)
        candidate.setdefault(
            "regret_vs_target_tie_band_oracle_top1_local_loss_delta",
            None,
        )
    sampled_candidate_table = build_compact_within_tie_band_sampled_table_rows(
        sampled_candidates
    )
    warmup_tags = _preview_warmup_tags(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        local_loss_delta_by_key=local_loss_delta_by_key,
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        optimizer_step_index=int(optimizer_step_index),
    )
    applied_flip_count = _preview_applied_flip_count(
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        local_loss_delta_by_key=local_loss_delta_by_key,
        local_selection_ordering_seed=int(local_selection_ordering_seed),
        optimizer_step_index=int(optimizer_step_index),
    )
    table_hash = _sha256_text(_canonical_json(sampled_candidate_table))
    return {
        "schema_version": VOTES_EMIT_SCHEMA_VERSION,
        "optimizer_step_index": int(optimizer_step_index),
        "warmup_apply_class": str(warmup_tags["warmup_apply_class"]),
        "effective_apply_threshold_abs": warmup_tags.get("effective_apply_threshold_abs"),
        "applied_flip_count": int(applied_flip_count),
        "threshold_semantics": frozen_threshold_semantics_block(),
        "sampled_candidate_table": sampled_candidate_table,
        "sampled_candidate_count": int(len(sampled_candidate_table)),
        "source_table_hash": table_hash,
    }


class VotesEmitCollector:
    """Write per-step votes sidecars under {root}/votes_emit/v1."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.emit_root = self.root / "votes_emit" / "v1"
        self.per_step_dir = self.emit_root / "per_step"
        self.per_step_dir.mkdir(parents=True, exist_ok=True)
        self._step_hashes: dict[str, str] = {}
        self._emit_timings_ms: list[float] = []

    def emit_step(
        self,
        record: Mapping[str, Any],
        *,
        optimizer_step_index: int,
    ) -> dict[str, Any]:
        start = time.perf_counter()
        step_name = f"{int(optimizer_step_index):05d}"
        step_path = self.per_step_dir / f"{step_name}.json"
        payload = dict(record)
        canonical = _canonical_json(payload)
        step_hash = _sha256_text(canonical)
        step_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        self._step_hashes[step_name] = step_hash
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        self._emit_timings_ms.append(float(elapsed_ms))
        manifest = self.write_manifest()
        return {
            "step_path": str(step_path),
            "step_hash": step_hash,
            "manifest_path": str(self.emit_root / "manifest.json"),
            "manifest_hash": manifest["manifest_sha256"],
            "emit_elapsed_ms": float(elapsed_ms),
        }

    def write_manifest(self) -> dict[str, Any]:
        emit_timings_ms = [float(value) for value in self._emit_timings_ms]
        stable_manifest = {
            "schema_version": VOTES_EMIT_SCHEMA_VERSION,
            "per_step_hashes": dict(sorted(self._step_hashes.items())),
            "step_count": int(len(self._step_hashes)),
        }
        manifest_sha256 = _sha256_text(_canonical_json(stable_manifest))
        manifest = {
            **stable_manifest,
            "emit_timings_ms": emit_timings_ms,
            "emit_sample_count": int(len(emit_timings_ms)),
            "manifest_sha256": manifest_sha256,
        }
        manifest_path = self.emit_root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return manifest

    @property
    def emit_timings_ms(self) -> tuple[float, ...]:
        return tuple(self._emit_timings_ms)


def maybe_emit_votes_step_record(
    *,
    root: Path | None,
    enabled: bool,
    optimizer_step_index: int,
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    votes_by_key: Mapping[str, torch.Tensor],
    vote_specs_by_key: Mapping[str, VoteUpdateSpec],
    max_abs_per_tensor: int,
    collector: VotesEmitCollector | None = None,
    two_tier_carry_w6_enabled: bool = False,
    local_loss_delta_by_key: Mapping[str, torch.Tensor] | None = None,
    local_selection_ordering_seed: int = 0,
) -> dict[str, Any] | None:
    if not bool(enabled) or root is None:
        return None
    record = build_votes_emit_step_record(
        optimizer_step_index=int(optimizer_step_index),
        tensor_states=tensor_states,
        votes_by_key=votes_by_key,
        vote_specs_by_key=vote_specs_by_key,
        max_abs_per_tensor=int(max_abs_per_tensor),
        two_tier_carry_w6_enabled=bool(two_tier_carry_w6_enabled),
        local_loss_delta_by_key=local_loss_delta_by_key,
        local_selection_ordering_seed=int(local_selection_ordering_seed),
    )
    active_collector = collector or VotesEmitCollector(Path(root))
    return active_collector.emit_step(record, optimizer_step_index=int(optimizer_step_index))
