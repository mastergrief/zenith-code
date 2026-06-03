"""Slice 2B global-rate-cap reference over Slice 2A vote-update plans.

This module is intentionally CPU/reference glue. It performs cross-tensor
selection, cap-bounded apply, and deferred-backlog accounting over already-built
``VoteUpdatePlan`` objects. It is not a trainer integration point, functional
veto, bad-pressure drain, rank-bucket credit generation, or GPU kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import copy
import hashlib
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdatePlan,
    VoteUpdateState,
    _safe_ratio,
)


CAP_ORDERING_HASH_SEED = 17
SCRATCH_S1_GLOBAL_CAP_START = 512
SCRATCH_S1_GLOBAL_CAP_MAX = 1024
SCRATCH_S1_GLOBAL_CAP_ANNEAL_STEP = 256
DEFERRED_NON_SCOPE = "deferred_non_scope"
CPU_GLUE_NOT_KERNEL_NOTE = (
    "global-rate-cap selection is thin cross-tensor CPU/control-flow glue; "
    "it has no GPU receipt by design in Slice 2B"
)


class GlobalRateCapOrderingMode(str, Enum):
    MARGIN = "margin"
    HASH_SHUFFLE = "hash_shuffle"
    ROUND_ROBIN = "round_robin"


@dataclass(frozen=True)
class GlobalRateCapSpec:
    cap: int
    step: int
    ordering_mode: GlobalRateCapOrderingMode | str = GlobalRateCapOrderingMode.MARGIN
    ordering_seed: int = CAP_ORDERING_HASH_SEED
    functional_veto_policy: str = DEFERRED_NON_SCOPE
    bad_pressure_drain_policy: str = DEFERRED_NON_SCOPE
    mutate_outputs: bool = True

    @property
    def normalized_ordering_mode(self) -> GlobalRateCapOrderingMode:
        return GlobalRateCapOrderingMode(self.ordering_mode)

    def validate(self) -> None:
        _ = self.normalized_ordering_mode
        if self.cap < 0:
            raise ValueError(f"global cap must be >= 0, got {self.cap}")
        if self.step < 0:
            raise ValueError(f"step must be >= 0 for ordering, got {self.step}")
        if self.functional_veto_policy != DEFERRED_NON_SCOPE:
            raise NotImplementedError(
                "functional-window veto is deferred_non_scope in Slice 2B"
            )
        if self.bad_pressure_drain_policy != DEFERRED_NON_SCOPE:
            raise NotImplementedError(
                "bad-pressure drain is deferred_non_scope in Slice 2B"
            )


@dataclass(frozen=True)
class GlobalRateCapTensorInput:
    state_key: str
    state: VoteUpdateState
    plan: VoteUpdatePlan


@dataclass(frozen=True)
class GlobalRateCapRow:
    state_key: str
    flat_index: int
    local_pos: int
    global_flat_index: int
    abs_new_acc: int
    threshold_abs: int
    margin_abs_over_threshold: int

    def identity(self) -> tuple[str, int]:
        return (self.state_key, self.flat_index)


@dataclass(frozen=True)
class GlobalRateCapTensorResult:
    state_key: str
    q_levels: torch.Tensor
    accumulators: torch.Tensor
    stats: dict[str, Any]


@dataclass(frozen=True)
class GlobalRateCapResult:
    tensor_results: list[GlobalRateCapTensorResult]
    step_summary: dict[str, Any]
    rows: list[GlobalRateCapRow]
    accepted_rows: list[GlobalRateCapRow]
    deferred_rows: list[GlobalRateCapRow]
    deferred_backlog: dict[str, dict[int, dict[str, int]]]


def scratch_s1_global_cap_for_step(step: int) -> int:
    step_i = int(step)
    if step_i < 1:
        raise ValueError(f"S1 global cap step must be >=1, got {step}")
    phase = (step_i - 1) // SCRATCH_S1_GLOBAL_CAP_ANNEAL_STEP
    return min(
        SCRATCH_S1_GLOBAL_CAP_MAX,
        SCRATCH_S1_GLOBAL_CAP_START + phase * SCRATCH_S1_GLOBAL_CAP_START,
    )


def scratch_s1_global_cap_contract() -> dict[str, Any]:
    return {
        "schema": "scratch_s1_global_rate_cap_contract/v1",
        "helper": "scratch_s1_global_cap_for_step",
        "active_runtime_control": True,
        "start": SCRATCH_S1_GLOBAL_CAP_START,
        "max": SCRATCH_S1_GLOBAL_CAP_MAX,
        "anneal_step": SCRATCH_S1_GLOBAL_CAP_ANNEAL_STEP,
        "formula": "min(1024, 512 + floor((step - 1) / 256) * 512)",
        "applied_by": "apply_global_rate_cap_reference(..., mutate_outputs=True)",
        "per_step_assertions": [
            "global_rate_cap_applied_count <= global_rate_cap_cap",
            "q_changed_count == global_rate_cap_applied_count",
            "demand_gt_cap_implies_saturated_and_deferred",
        ],
    }


def tensor_offsets_for_vote_update_states(
    inputs: list[GlobalRateCapTensorInput],
) -> dict[str, int]:
    if not inputs:
        raise ValueError("tensor offsets require at least one tensor input")
    offsets: dict[str, int] = {}
    cursor = 0
    for item in inputs:
        if item.state_key in offsets:
            raise ValueError(f"duplicate state_key {item.state_key!r}")
        offsets[item.state_key] = cursor
        cursor += int(item.state.q_levels.numel())
    return offsets


def _tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _row_global_index_sha(rows: list[GlobalRateCapRow]) -> str:
    values = torch.tensor([row.global_flat_index for row in rows], dtype=torch.int64)
    return _tensor_sha256(values)


def _deferred_age_summary(
    deferred_backlog: dict[str, dict[int, dict[str, int]]],
    *,
    step: int,
) -> dict[str, int]:
    entries = [
        entry
        for by_index in deferred_backlog.values()
        for entry in by_index.values()
    ]
    if not entries:
        return {
            "deferred_backlog_size": 0,
            "deferred_backlog_max_age_steps": 0,
            "deferred_backlog_max_defer_count": 0,
        }
    return {
        "deferred_backlog_size": len(entries),
        "deferred_backlog_max_age_steps": max(
            int(step) - int(entry["first_step"]) for entry in entries
        ),
        "deferred_backlog_max_defer_count": max(
            int(entry["defer_count"]) for entry in entries
        ),
    }


def validate_global_rate_cap_inputs(inputs: list[GlobalRateCapTensorInput]) -> None:
    if not inputs:
        raise ValueError("global rate cap requires at least one tensor input")
    seen: set[str] = set()
    for item in inputs:
        if not item.state_key:
            raise ValueError("state_key must be non-empty")
        if item.state_key in seen:
            raise ValueError(f"duplicate state_key {item.state_key!r}")
        seen.add(item.state_key)
        if item.state.q_levels.shape != item.state.accumulators.shape:
            raise ValueError(f"q/accumulator shape mismatch for {item.state_key}")
        if item.plan.q_i16.shape != item.state.q_levels.shape:
            raise ValueError(f"plan q shape mismatch for {item.state_key}")
        if item.plan.new_acc_i32.shape != item.state.accumulators.shape:
            raise ValueError(f"plan accumulator shape mismatch for {item.state_key}")
        if item.plan.applied_indices.numel() != item.plan.applied_directions.numel():
            raise ValueError(f"applied index/direction mismatch for {item.state_key}")
        if item.plan.applied_indices.numel() != item.plan.applied_thresholds.numel():
            raise ValueError(f"applied index/threshold mismatch for {item.state_key}")


def global_rate_cap_priority_rows(
    inputs: list[GlobalRateCapTensorInput],
    *,
    tensor_offsets: dict[str, int],
) -> list[GlobalRateCapRow]:
    validate_global_rate_cap_inputs(inputs)
    rows: list[GlobalRateCapRow] = []
    for item in inputs:
        if item.state_key not in tensor_offsets:
            raise ValueError(f"missing tensor offset for {item.state_key!r}")
        offset = int(tensor_offsets[item.state_key])
        flat_new_acc = item.plan.new_acc_i32.flatten()
        indices = item.plan.applied_indices.to(torch.int64)
        thresholds = item.plan.applied_thresholds.to(torch.int64)
        for local_pos, raw_idx in enumerate(indices.tolist()):
            flat_index = int(raw_idx)
            abs_new_acc = int(flat_new_acc[flat_index].abs().item())
            threshold_abs = int(thresholds[local_pos].item())
            rows.append(
                GlobalRateCapRow(
                    state_key=item.state_key,
                    flat_index=flat_index,
                    local_pos=int(local_pos),
                    global_flat_index=offset + flat_index,
                    abs_new_acc=abs_new_acc,
                    threshold_abs=threshold_abs,
                    margin_abs_over_threshold=abs_new_acc - threshold_abs,
                )
            )
    rows.sort(key=lambda row: (-row.abs_new_acc, row.global_flat_index))
    return rows


def _cap_hash_order_int(
    row: GlobalRateCapRow,
    *,
    global_step: int,
    seed: int,
) -> int:
    h = hashlib.sha256()
    h.update(
        (
            f"{int(seed)}|{int(global_step)}|{row.state_key}|"
            f"{int(row.global_flat_index)}|{int(row.flat_index)}|{int(row.local_pos)}"
        ).encode("utf-8")
    )
    return int.from_bytes(h.digest()[:16], "big", signed=False)


def _cap_round_robin_rows(
    rows: list[GlobalRateCapRow],
    *,
    global_step: int,
) -> list[GlobalRateCapRow]:
    grouped: dict[str, list[GlobalRateCapRow]] = {}
    for row in rows:
        grouped.setdefault(row.state_key, []).append(row)
    state_keys = sorted(grouped)
    if not state_keys:
        return []
    phase = int(global_step) % len(state_keys)
    ordered_state_keys = state_keys[phase:] + state_keys[:phase]
    ordered: list[GlobalRateCapRow] = []
    cursor = 0
    while len(ordered) < len(rows):
        progressed = False
        for state_key in ordered_state_keys:
            bucket = grouped[state_key]
            if cursor >= len(bucket):
                continue
            ordered.append(bucket[cursor])
            progressed = True
        if not progressed:
            break
        cursor += 1
    return ordered


def order_global_rate_cap_rows(
    rows: list[GlobalRateCapRow],
    *,
    mode: GlobalRateCapOrderingMode | str,
    global_step: int,
    seed: int = CAP_ORDERING_HASH_SEED,
) -> list[GlobalRateCapRow]:
    normalized = GlobalRateCapOrderingMode(mode)
    if normalized == GlobalRateCapOrderingMode.MARGIN:
        return list(rows)
    if normalized == GlobalRateCapOrderingMode.HASH_SHUFFLE:
        return sorted(
            rows,
            key=lambda row: (
                _cap_hash_order_int(row, global_step=global_step, seed=seed),
                row.global_flat_index,
            ),
        )
    if normalized == GlobalRateCapOrderingMode.ROUND_ROBIN:
        return _cap_round_robin_rows(rows, global_step=global_step)
    raise AssertionError(f"unreachable cap ordering mode {normalized!r}")


def select_global_rate_cap_rows(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    tensor_offsets: dict[str, int] | None = None,
) -> tuple[list[GlobalRateCapRow], list[GlobalRateCapRow], list[GlobalRateCapRow]]:
    spec.validate()
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    rows = global_rate_cap_priority_rows(inputs, tensor_offsets=offsets)
    rows = order_global_rate_cap_rows(
        rows,
        mode=spec.normalized_ordering_mode,
        global_step=spec.step,
        seed=spec.ordering_seed,
    )
    cap = max(0, int(spec.cap))
    return rows, rows[:cap], rows[cap:]


def _rows_by_key(rows: list[GlobalRateCapRow]) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {}
    for row in rows:
        out.setdefault(row.state_key, set()).add(int(row.flat_index))
    return out


def _apply_threshold_residual(
    new_acc_i32: torch.Tensor,
    indices: torch.Tensor,
    directions: torch.Tensor,
    thresholds: torch.Tensor,
) -> None:
    if indices.numel() == 0:
        return
    residual = new_acc_i32[indices] - directions.to(torch.int32) * thresholds
    low = -thresholds + 1
    high = thresholds - 1
    new_acc_i32[indices] = torch.minimum(torch.maximum(residual, low), high)


def apply_global_rate_cap_reference(
    inputs: list[GlobalRateCapTensorInput],
    spec: GlobalRateCapSpec,
    *,
    deferred_backlog: dict[str, dict[int, dict[str, int]]] | None = None,
    tensor_offsets: dict[str, int] | None = None,
) -> GlobalRateCapResult:
    """Apply cap-bounded global selection to copies of q/acc tensors.

    Slice 2A ``VoteUpdatePlan.applied_*`` fields are local post-veto,
    pre-global-cap candidates. This function is the first place those candidates
    become final global-applied mutations.
    """

    spec.validate()
    offsets = tensor_offsets or tensor_offsets_for_vote_update_states(inputs)
    rows, accepted_rows, deferred_rows = select_global_rate_cap_rows(
        inputs,
        spec,
        tensor_offsets=offsets,
    )
    accepted_by_key = _rows_by_key(accepted_rows)
    deferred_by_key = _rows_by_key(deferred_rows)
    backlog = copy.deepcopy(deferred_backlog or {})
    accepted_from_prior_deferred = 0
    for row in accepted_rows:
        state_backlog = backlog.get(row.state_key, {})
        if row.flat_index in state_backlog:
            accepted_from_prior_deferred += 1
            del state_backlog[row.flat_index]
    for row in deferred_rows:
        state_backlog = backlog.setdefault(row.state_key, {})
        entry = state_backlog.setdefault(
            row.flat_index,
            {"first_step": int(spec.step), "last_deferred_step": int(spec.step), "defer_count": 0},
        )
        entry["last_deferred_step"] = int(spec.step)
        entry["defer_count"] = int(entry.get("defer_count", 0)) + 1

    tensor_results: list[GlobalRateCapTensorResult] = []
    total_q_changed = 0
    for item in inputs:
        accepted_set = accepted_by_key.get(item.state_key, set())
        deferred_set = deferred_by_key.get(item.state_key, set())
        plan = item.plan
        q_i16 = plan.q_i16.flatten().clone()
        new_acc_i32 = plan.new_acc_i32.flatten().clone().to(torch.int32)
        pre_cap_indices = plan.applied_indices.to(torch.int64)
        pre_cap_directions = plan.applied_directions.to(torch.int16)
        pre_cap_thresholds = plan.applied_thresholds.to(torch.int32)
        accepted_mask = torch.tensor(
            [int(idx) in accepted_set for idx in pre_cap_indices.tolist()],
            dtype=torch.bool,
        )
        accepted_indices = pre_cap_indices[accepted_mask]
        accepted_directions = pre_cap_directions[accepted_mask]
        accepted_thresholds = pre_cap_thresholds[accepted_mask]
        if spec.mutate_outputs and accepted_indices.numel() > 0:
            q_i16[accepted_indices] = (
                q_i16[accepted_indices] + accepted_directions
            ).clamp(-1, 1)
            _apply_threshold_residual(
                new_acc_i32,
                accepted_indices,
                accepted_directions,
                accepted_thresholds,
            )

        replay_indices = plan.replay_ce_veto_indices.to(torch.int64)
        replay_directions = plan.replay_veto_directions.to(torch.int16)
        replay_thresholds = plan.replay_veto_thresholds.to(torch.int32)
        _apply_threshold_residual(
            new_acc_i32,
            replay_indices,
            replay_directions,
            replay_thresholds,
        )

        if spec.mutate_outputs:
            q_out = q_i16.view_as(item.state.q_levels).to(torch.int8).contiguous()
            acc_out = new_acc_i32.view_as(item.state.accumulators).to(torch.int16).contiguous()
        else:
            q_out = item.state.q_levels.detach().clone().contiguous()
            acc_out = item.state.accumulators.detach().clone().contiguous()
        q_changed = int((q_out != item.state.q_levels).sum().item())
        total_q_changed += q_changed
        deferred_indices = torch.tensor(sorted(deferred_set), dtype=torch.int64)
        stats = dict(plan.stats)
        stats.update(
            {
                "scope": "global_rate_cap_reference",
                "two_b_input_name": "2A applied_indices are local_post_veto_pre_global_cap_candidates",
                "global_rate_cap_enabled": True,
                "global_rate_cap_cap": int(spec.cap),
                "global_rate_cap_ordering_mode": spec.normalized_ordering_mode.value,
                "global_rate_cap_ordering_seed": int(spec.ordering_seed),
                "functional_veto_policy": DEFERRED_NON_SCOPE,
                "bad_pressure_drain_policy": DEFERRED_NON_SCOPE,
                "cpu_glue_not_kernel": True,
                "global_rate_cap_would_accept_count": int(accepted_indices.numel()),
                "ternary_mutation_enabled": bool(spec.mutate_outputs),
                "ternary_mutation_frozen": not bool(spec.mutate_outputs),
                "flip_count": int(accepted_indices.numel()) if spec.mutate_outputs else 0,
                "post_veto_applied_flip_count": (
                    int(accepted_indices.numel()) if spec.mutate_outputs else 0
                ),
                "global_rate_cap_accepted_count": int(accepted_indices.numel()),
                "global_rate_cap_applied_count": (
                    int(accepted_indices.numel()) if spec.mutate_outputs else 0
                ),
                "global_rate_cap_deferred_count": int(deferred_indices.numel()),
                "global_rate_cap_deferred_indices_sha256": _tensor_sha256(deferred_indices),
                "global_rate_cap_accepted_indices_sha256": _tensor_sha256(
                    accepted_indices.to(torch.int64)
                ),
                "global_rate_cap_accepted_indices_sample": [
                    int(x) for x in accepted_indices[:16].detach().cpu().tolist()
                ],
                "global_rate_cap_deferred_indices_sample": [
                    int(x) for x in deferred_indices[:16].detach().cpu().tolist()
                ],
                "post_veto_applied_indices": [
                    int(x) for x in accepted_indices.detach().cpu().tolist()
                ] if spec.mutate_outputs else [],
                "global_rate_cap_accepted_indices": [
                    int(x) for x in accepted_indices.detach().cpu().tolist()
                ],
                "global_rate_cap_deferred_indices": [
                    int(x) for x in deferred_indices.detach().cpu().tolist()
                ],
                "q_changed_count": q_changed,
            }
        )
        tensor_results.append(
            GlobalRateCapTensorResult(
                state_key=item.state_key,
                q_levels=q_out,
                accumulators=acc_out,
                stats=stats,
            )
        )

    accepted_count = len(accepted_rows)
    deferred_count = len(deferred_rows)
    age_summary = _deferred_age_summary(backlog, step=spec.step)
    step_summary = {
        "global_rate_cap_enabled": True,
        "global_rate_cap_cap": int(spec.cap),
        "global_rate_cap_ordering_mode": spec.normalized_ordering_mode.value,
        "global_rate_cap_ordering_seed": int(spec.ordering_seed),
        "global_rate_cap_ordering_summary": {
            "schema_version": "global_rate_cap_ordering/v1",
            "mode": spec.normalized_ordering_mode.value,
            "default_margin_behavior_equivalent": (
                spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
            ),
            "seed": int(spec.ordering_seed),
            "global_step": int(spec.step),
            "order_key": (
                "highest_abs_new_acc_then_lower_global_flat_index"
                if spec.normalized_ordering_mode == GlobalRateCapOrderingMode.MARGIN
                else "sha256(seed|global_step|state_key|global_flat_index|flat_index|local_pos)"
                if spec.normalized_ordering_mode == GlobalRateCapOrderingMode.HASH_SHUFFLE
                else "state_key_round_robin_preserving_margin_order_within_state_key"
            ),
            "full_demand_count": len(rows),
            "selected_count": accepted_count,
            "deferred_count": deferred_count,
            "global_indices_sha256": {
                "full_demand": _row_global_index_sha(rows),
                "cap_selected": _row_global_index_sha(accepted_rows),
                "cap_deferred": _row_global_index_sha(deferred_rows),
            },
        },
        "functional_veto_policy": DEFERRED_NON_SCOPE,
        "bad_pressure_drain_policy": DEFERRED_NON_SCOPE,
        "cpu_glue_not_kernel": True,
        "cpu_glue_not_kernel_note": CPU_GLUE_NOT_KERNEL_NOTE,
        "ternary_mutation_enabled": bool(spec.mutate_outputs),
        "ternary_mutation_frozen": not bool(spec.mutate_outputs),
        "global_pre_cap_would_apply_count": len(rows),
        "global_rate_cap_accepted_count": accepted_count,
        "global_rate_cap_applied_count": accepted_count if spec.mutate_outputs else 0,
        "global_rate_cap_deferred_count": deferred_count,
        "global_rate_cap_saturated": len(rows) > int(spec.cap),
        "global_rate_cap_fill_ratio": _safe_ratio(accepted_count, int(spec.cap)),
        "global_deferred_ratio": _safe_ratio(deferred_count, len(rows)),
        "accepted_from_prior_deferred_count": accepted_from_prior_deferred,
        "accepted_fresh_count": accepted_count - accepted_from_prior_deferred,
        "q_changed_count": total_q_changed,
        **age_summary,
    }
    return GlobalRateCapResult(
        tensor_results=tensor_results,
        step_summary=step_summary,
        rows=rows,
        accepted_rows=accepted_rows,
        deferred_rows=deferred_rows,
        deferred_backlog=backlog,
    )
