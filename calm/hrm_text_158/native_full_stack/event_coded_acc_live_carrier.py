"""V4-LIVE event-coded accumulator carrier (CPU synthetic / standalone)."""
from __future__ import annotations

import copy
from collections.abc import Callable, Iterable, Iterator, Mapping, MutableMapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.two_tier_carry_reducers import (
    DEFAULT_CROSSING_THRESHOLD_ABS,
    carry_self_update_row,
    crossing_bool_w6,
    encode_post_flip_residual,
    vectorized_carry_self_update_row,
    vectorized_crosses_threshold,
)

DEFAULT_WATCH_BAND = 3
DEFAULT_COLD_DEFAULT = 0
DEFAULT_DECAY_NUMERATOR = 1
DEFAULT_DECAY_DENOMINATOR = 1
DEFAULT_VERDICT_NUMEL = 1024


def promotion_carry_threshold(*, threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS) -> int:
    return int(threshold_abs) - int(DEFAULT_WATCH_BAND)


def merge_hot_table_arrays(
    base_indices: np.ndarray,
    base_values: np.ndarray,
    remove_indices: np.ndarray,
    update_indices: np.ndarray,
    update_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Sorted-unique hot table: apply removes then last-wins updates (vectorized)."""

    idx = np.asarray(base_indices, dtype=np.int32)
    val = np.asarray(base_values, dtype=np.int16)
    remove_indices = np.asarray(remove_indices, dtype=np.int32).reshape(-1)
    update_indices = np.asarray(update_indices, dtype=np.int32).reshape(-1)
    update_values = np.asarray(update_values, dtype=np.int16).reshape(-1)
    if remove_indices.size:
        keep = ~np.isin(idx, remove_indices)
        idx = idx[keep]
        val = val[keep]
    if update_indices.size:
        if remove_indices.size:
            upd_keep = ~np.isin(update_indices, remove_indices)
            update_indices = update_indices[upd_keep]
            update_values = update_values[upd_keep]
        if update_indices.size:
            merged_idx = np.concatenate([idx, update_indices])
            merged_val = np.concatenate([val, update_values])
            order = np.argsort(merged_idx, kind="mergesort")
            merged_idx = merged_idx[order]
            merged_val = merged_val[order]
            uniq = np.ones(merged_idx.shape[0], dtype=bool)
            uniq[:-1] = merged_idx[1:] != merged_idx[:-1]
            idx = merged_idx[uniq]
            val = merged_val[uniq]
    return idx, val


def hot_risk_proxy_indices(
    carries: Mapping[int, int],
    *,
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS,
) -> set[int]:
    """Numeric near-threshold proxy for synthetic CPU sweeps (not regret/tie-band)."""

    promote_at = promotion_carry_threshold(threshold_abs=threshold_abs)
    return {int(index) for index, carry in carries.items() if abs(int(carry)) >= promote_at}


@dataclass
class StepSurfaceRecord:
    step_index: int
    crossing_indices: tuple[int, ...]
    applied_indices: tuple[int, ...]
    backlog_indices: tuple[int, ...]
    q_levels: dict[int, int]
    hot_exact_row_count: int
    promotion_count: int
    demotion_on_decay_count: int
    demotion_on_crossing_count: int


class _PackedHotTable:
    """Sorted unique int32 index + int16 value hot table with copy-on-write."""

    __slots__ = ("_indices", "_values", "_writable")

    def __init__(
        self,
        indices: np.ndarray | None = None,
        values: np.ndarray | None = None,
        *,
        shared: bool = False,
    ) -> None:
        self._indices = (
            np.array([], dtype=np.int32)
            if indices is None
            else np.ascontiguousarray(indices, dtype=np.int32)
        )
        self._values = (
            np.array([], dtype=np.int16)
            if values is None
            else np.ascontiguousarray(values, dtype=np.int16)
        )
        if self._indices.shape != self._values.shape:
            raise ValueError("hot index/value shape mismatch")
        self._writable = not shared
        self._validate_invariant()

    @classmethod
    def empty(cls) -> _PackedHotTable:
        return cls()

    @classmethod
    def from_arrays(cls, indices: np.ndarray, values: np.ndarray) -> _PackedHotTable:
        return cls(
            np.ascontiguousarray(indices, dtype=np.int32),
            np.ascontiguousarray(values, dtype=np.int16),
        )

    @classmethod
    def from_dict(cls, hot_exact: Mapping[int, int]) -> _PackedHotTable:
        if not hot_exact:
            return cls.empty()
        keys = sorted(int(k) for k in hot_exact)
        indices = np.array(keys, dtype=np.int32)
        values = np.array([int(hot_exact[k]) for k in keys], dtype=np.int16)
        return cls(indices, values)

    def _validate_invariant(self) -> None:
        if self._indices.size == 0:
            return
        if not np.all(self._indices[1:] > self._indices[:-1]):
            raise ValueError("hot indices must be strictly sorted and unique")

    def _ensure_writable(self) -> None:
        if self._writable:
            return
        self._indices = self._indices.copy()
        self._values = self._values.copy()
        self._writable = True

    def fork(self) -> _PackedHotTable:
        return _PackedHotTable(self._indices, self._values, shared=True)

    def __len__(self) -> int:
        return int(self._indices.size)

    def contains(self, flat_index: int) -> bool:
        idx = int(flat_index)
        pos = int(np.searchsorted(self._indices, idx))
        return pos < self._indices.size and int(self._indices[pos]) == idx

    def get(self, flat_index: int, default: int) -> int:
        idx = int(flat_index)
        pos = int(np.searchsorted(self._indices, idx))
        if pos < self._indices.size and int(self._indices[pos]) == idx:
            return int(self._values[pos])
        return int(default)

    def set_lane(self, flat_index: int, value: int) -> None:
        signed = int(value)
        if signed < -32768 or signed > 32767:
            raise ValueError("hot_exact value must fit int16")
        idx = int(flat_index)
        if idx < 0 or idx >= 2**31:
            raise ValueError("hot index must be non-negative and < 2**31")
        self._ensure_writable()
        pos = int(np.searchsorted(self._indices, idx))
        if pos < self._indices.size and int(self._indices[pos]) == idx:
            self._values[pos] = np.int16(signed)
            return
        self._indices = np.insert(self._indices, pos, idx)
        self._values = np.insert(self._values, pos, np.int16(signed))

    def remove_lane(self, flat_index: int) -> None:
        idx = int(flat_index)
        pos = int(np.searchsorted(self._indices, idx))
        if pos >= self._indices.size or int(self._indices[pos]) != idx:
            return
        self._ensure_writable()
        self._indices = np.delete(self._indices, pos)
        self._values = np.delete(self._values, pos)

    def indices_array(self) -> np.ndarray:
        return self._indices

    def values_array(self) -> np.ndarray:
        return self._values

    def replace_arrays(self, indices: np.ndarray, values: np.ndarray) -> None:
        self._indices = np.ascontiguousarray(indices, dtype=np.int32)
        self._values = np.ascontiguousarray(values, dtype=np.int16)
        self._writable = True
        self._validate_invariant()

    def to_dict(self) -> dict[int, int]:
        return {int(i): int(v) for i, v in zip(self._indices.tolist(), self._values.tolist())}


class _HotExactView(MutableMapping[int, int]):
    __slots__ = ("_table", "_invalidate_cache")

    def __init__(
        self,
        table: _PackedHotTable,
        *,
        invalidate_cache: Callable[[], None] | None = None,
    ) -> None:
        self._table = table
        self._invalidate_cache = invalidate_cache

    def _touch(self) -> None:
        if self._invalidate_cache is not None:
            self._invalidate_cache()

    def __getitem__(self, key: int) -> int:
        idx = int(key)
        if not self._table.contains(idx):
            raise KeyError(key)
        return self._table.get(idx, 0)

    def __setitem__(self, key: int, value: int) -> None:
        self._table.set_lane(int(key), int(value))
        self._touch()

    def __delitem__(self, key: int) -> None:
        idx = int(key)
        if not self._table.contains(idx):
            raise KeyError(key)
        self._table.remove_lane(idx)
        self._touch()

    def __iter__(self) -> Iterator[int]:
        return iter(self._table.indices_array().tolist())

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: object) -> bool:
        if not isinstance(key, int):
            return False
        return self._table.contains(int(key))

    def get(self, key: int, default: int | None = None) -> int | None:
        idx = int(key)
        if self._table.contains(idx):
            return self._table.get(idx, 0)
        return default

    def pop(self, key: int, default: int | None = None) -> int | None:
        idx = int(key)
        if not self._table.contains(idx):
            if default is not None:
                return default
            raise KeyError(key)
        value = self._table.get(idx, 0)
        self._table.remove_lane(idx)
        self._touch()
        return value

    def keys(self) -> Iterable[int]:
        return iter(self._table.indices_array().tolist())

    def values(self) -> Iterable[int]:
        return (int(v) for v in self._table.values_array().tolist())

    def items(self) -> Iterable[tuple[int, int]]:
        return zip(self.keys(), self.values())


class _HotStepJournal:
    """Deferred hot-table mutations — one sorted rebuild per apply_step."""

    __slots__ = ("_base_indices", "_base_values", "_updates", "_removes")

    def __init__(self, table: _PackedHotTable) -> None:
        self._base_indices = table.indices_array()
        self._base_values = table.values_array()
        self._updates: dict[int, int] = {}
        self._removes: set[int] = set()

    def contains(self, flat_index: int) -> bool:
        idx = int(flat_index)
        if idx in self._removes:
            return False
        if idx in self._updates:
            return True
        pos = int(np.searchsorted(self._base_indices, idx))
        return pos < self._base_indices.size and int(self._base_indices[pos]) == idx

    def set_lane(self, flat_index: int, value: int) -> None:
        idx = int(flat_index)
        signed = int(value)
        if signed < -32768 or signed > 32767:
            raise ValueError("hot_exact value must fit int16")
        if idx < 0 or idx >= 2**31:
            raise ValueError("hot index must be non-negative and < 2**31")
        self._removes.discard(idx)
        self._updates[idx] = signed

    def remove_lane(self, flat_index: int) -> None:
        idx = int(flat_index)
        self._updates.pop(idx, None)
        self._removes.add(idx)

    def set_lanes_from_arrays(self, indices: np.ndarray, values: np.ndarray) -> None:
        for idx, val in zip(indices.tolist(), values.tolist()):
            self.set_lane(int(idx), int(val))

    def remove_lanes_from_array(self, indices: np.ndarray) -> None:
        for idx in indices.tolist():
            self.remove_lane(int(idx))

    def finalize(self) -> tuple[np.ndarray, np.ndarray]:
        if not self._updates and not self._removes:
            return self._base_indices, self._base_values
        if self._base_indices.size == 0:
            if not self._updates:
                return self._base_indices, self._base_values
            keys = np.array(sorted(self._updates), dtype=np.int32)
            vals = np.array([self._updates[int(k)] for k in keys], dtype=np.int16)
            return keys, vals
        if self._removes:
            remove_arr = np.array(sorted(self._removes), dtype=np.int32)
            keep = ~np.isin(self._base_indices, remove_arr)
        else:
            keep = np.ones(self._base_indices.shape[0], dtype=bool)
        base_idx = self._base_indices[keep]
        base_val = self._base_values[keep]
        if not self._updates:
            return base_idx, base_val
        upd_keys = np.array(sorted(self._updates), dtype=np.int32)
        upd_vals = np.array([self._updates[int(k)] for k in upd_keys], dtype=np.int16)
        merged_idx = np.concatenate([base_idx, upd_keys])
        merged_val = np.concatenate([base_val, upd_vals])
        order = np.argsort(merged_idx, kind="mergesort")
        merged_idx = merged_idx[order]
        merged_val = merged_val[order]
        if merged_idx.size == 0:
            return merged_idx, merged_val
        uniq = np.ones(merged_idx.shape[0], dtype=bool)
        uniq[:-1] = merged_idx[1:] != merged_idx[:-1]
        return merged_idx[uniq], merged_val[uniq]


def _union_sorted_int32(*parts: np.ndarray) -> np.ndarray:
    arrays = [np.ascontiguousarray(part, dtype=np.int32) for part in parts if part.size]
    if not arrays:
        return np.array([], dtype=np.int32)
    out = arrays[0]
    for extra in arrays[1:]:
        if extra.size == 0:
            continue
        extra = np.unique(extra)
        pos = np.searchsorted(out, extra)
        mask = np.ones(extra.size, dtype=bool)
        in_range = pos < out.size
        if in_range.any():
            mask[in_range] &= out[pos[in_range]] != extra[in_range]
        new_extra = extra[mask]
        if new_extra.size:
            out = np.sort(np.concatenate([out, new_extra]))
    return out


@dataclass
class EventCodedAccLiveState:
    logical_numel: int
    cold_default: int = DEFAULT_COLD_DEFAULT
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS
    demotion_band: int = 3
    _hot: _PackedHotTable = field(default_factory=_PackedHotTable.empty, repr=False, compare=False)
    events: list[EventCodedAccEvent] = field(default_factory=list)
    backlog: set[int] = field(default_factory=set)
    q_levels: dict[int, int] = field(default_factory=dict)
    step_records: list[StepSurfaceRecord] = field(default_factory=list)
    dense_accumulator_materialized_numel: int = 0
    _hot_packed_bytes_cache: bytes | None = field(default=None, repr=False, compare=False)

    def _invalidate_packed_caches(self) -> None:
        self._hot_packed_bytes_cache = None

    def hot_packed_bytes(self) -> bytes:
        if self._hot_packed_bytes_cache is None:
            from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
                encode_hot_exact_rows_from_arrays,
            )

            self._hot_packed_bytes_cache = encode_hot_exact_rows_from_arrays(
                self._hot.indices_array(),
                self._hot.values_array(),
            )
        return self._hot_packed_bytes_cache

    def hot_lane_indices_tensor(self) -> torch.Tensor:
        """Read-only packed hot lane indices (no dict/list round-trip)."""
        idx = self._hot.indices_array()
        if idx.size == 0:
            return torch.empty(0, dtype=torch.int64)
        return torch.from_numpy(np.ascontiguousarray(idx, dtype=np.int64))

    def hot_lane_values_tensor(self) -> torch.Tensor:
        """Read-only packed hot lane values aligned to hot_lane_indices_tensor()."""
        val = self._hot.values_array()
        if val.size == 0:
            return torch.empty(0, dtype=torch.int32)
        return torch.from_numpy(np.ascontiguousarray(val, dtype=np.int32))

    @classmethod
    def with_hot_exact(
        cls,
        logical_numel: int,
        hot_exact: Mapping[int, int],
        **kwargs: Any,
    ) -> EventCodedAccLiveState:
        return cls(
            logical_numel=int(logical_numel),
            _hot=_PackedHotTable.from_dict(hot_exact),
            **kwargs,
        )

    @property
    def hot_exact(self) -> _HotExactView:
        return _HotExactView(self._hot, invalidate_cache=self._invalidate_packed_caches)

    def cow_copy(self) -> EventCodedAccLiveState:
        return EventCodedAccLiveState(
            logical_numel=int(self.logical_numel),
            cold_default=int(self.cold_default),
            threshold_abs=int(self.threshold_abs),
            demotion_band=int(self.demotion_band),
            _hot=self._hot.fork(),
            events=list(self.events),
            backlog=set(self.backlog),
            q_levels=dict(self.q_levels),
            step_records=list(self.step_records),
            dense_accumulator_materialized_numel=int(self.dense_accumulator_materialized_numel),
        )

    def reconstruct_lane(self, flat_index: int) -> int:
        return int(self._hot.get(int(flat_index), int(self.cold_default)))

    def q_level(self, flat_index: int) -> int:
        return int(self.q_levels.get(int(flat_index), 0))

    def _should_promote(
        self,
        flat_index: int,
        *,
        post_carry: int,
        vote_touched: bool,
        hot_risk_proxy: set[int],
        hot_journal: _HotStepJournal | None = None,
    ) -> bool:
        contains = (
            hot_journal.contains(int(flat_index))
            if hot_journal is not None
            else self._hot.contains(int(flat_index))
        )
        if contains:
            return True
        if vote_touched:
            return True
        if int(flat_index) in hot_risk_proxy:
            return True
        promote_at = promotion_carry_threshold(threshold_abs=self.threshold_abs)
        return abs(int(post_carry)) >= int(promote_at)

    def _maybe_demote(
        self,
        flat_index: int,
        *,
        post_carry: int,
        fired_crossing: bool,
        hot_risk_proxy: set[int],
        vote_touched: bool,
        hot_journal: _HotStepJournal | None = None,
    ) -> bool:
        contains = (
            hot_journal.contains(int(flat_index))
            if hot_journal is not None
            else self._hot.contains(int(flat_index))
        )
        if not contains:
            return False
        if fired_crossing:
            if hot_journal is not None:
                hot_journal.remove_lane(int(flat_index))
            else:
                self._hot.remove_lane(int(flat_index))
            return True
        if vote_touched:
            return False
        if int(flat_index) in hot_risk_proxy:
            return False
        if int(flat_index) in self.backlog:
            return False
        if abs(int(post_carry)) >= int(self.demotion_band):
            return False
        if hot_journal is not None:
            hot_journal.remove_lane(int(flat_index))
        else:
            self._hot.remove_lane(int(flat_index))
        return True

    def apply_step(
        self,
        step_index: int,
        *,
        votes: Mapping[int, int],
        hot_risk_override: Iterable[int] | None = None,
    ) -> StepSurfaceRecord:
        vote_map = {int(k): int(v) for k, v in votes.items()}
        touched_arr = (
            np.array(sorted(vote_map), dtype=np.int32)
            if vote_map
            else np.array([], dtype=np.int32)
        )
        hot_indices = self._hot.indices_array()
        hot_values = self._hot.values_array()
        if hot_risk_override is not None:
            proxy_arr = np.unique(
                np.array([int(item) for item in hot_risk_override], dtype=np.int32)
            )
            active_sorted = _union_sorted_int32(hot_indices, touched_arr, proxy_arr)
        else:
            active_sorted = _union_sorted_int32(hot_indices, touched_arr)
        if active_sorted.size == 0:
            record = StepSurfaceRecord(
                step_index=int(step_index),
                crossing_indices=(),
                applied_indices=(),
                backlog_indices=tuple(sorted(self.backlog)),
                q_levels=dict(self.q_levels),
                hot_exact_row_count=len(self._hot),
                promotion_count=0,
                demotion_on_decay_count=0,
                demotion_on_crossing_count=0,
            )
            self.step_records.append(record)
            return record

        cold = int(self.cold_default)
        pre_arr = np.full(active_sorted.shape[0], cold, dtype=np.int32)
        if hot_indices.size:
            pos_in_hot = np.searchsorted(hot_indices, active_sorted)
            in_hot = pos_in_hot < hot_indices.size
            if in_hot.any():
                matched = hot_indices[pos_in_hot[in_hot]] == active_sorted[in_hot]
                hot_hits = np.zeros(active_sorted.shape[0], dtype=bool)
                hot_hits[np.where(in_hot)[0][matched]] = True
                pre_arr[hot_hits] = hot_values[pos_in_hot[hot_hits]].astype(np.int32)
        if touched_arr.size:
            pass  # touched lanes not in hot already default to cold_default in pre_arr
        promote_at = promotion_carry_threshold(threshold_abs=self.threshold_abs)
        if hot_risk_override is not None:
            # Oracle replaces proxy with the override set entirely (no near-threshold union).
            proxy_indices = proxy_arr
        else:
            if hot_indices.size:
                hot_proxy_mask = np.abs(hot_values.astype(np.int32)) >= int(promote_at)
                proxy_indices = (
                    hot_indices[hot_proxy_mask] if hot_proxy_mask.any() else np.empty(0, dtype=np.int32)
                )
            else:
                proxy_indices = np.empty(0, dtype=np.int32)
            if touched_arr.size:
                # touched_arr is unioned into active_sorted above, so positions are in-bounds.
                touched_positions = np.searchsorted(active_sorted, touched_arr)
                valid = (touched_positions < active_sorted.size) & (
                    active_sorted[touched_positions] == touched_arr
                )
                if valid.any():
                    pos = touched_positions[valid]
                    idx = touched_arr[valid]
                    near = np.abs(pre_arr[pos].astype(np.int32)) >= int(promote_at)
                    if near.any():
                        extra = idx[near].astype(np.int32)
                        proxy_indices = (
                            _union_sorted_int32(proxy_indices, extra)
                            if proxy_indices.size
                            else np.sort(extra)
                        )
        touched_set = set(vote_map)
        vote_arr = np.zeros(active_sorted.shape[0], dtype=np.int32)
        if touched_arr.size:
            vote_positions = np.searchsorted(active_sorted, touched_arr)
            vote_valid = (vote_positions < active_sorted.size) & (
                active_sorted[vote_positions] == touched_arr
            )
            if vote_valid.any():
                vote_arr[vote_positions[vote_valid]] = np.array(
                    [int(vote_map[int(i)]) for i in touched_arr[vote_valid]],
                    dtype=np.int32,
                )
        post_arr = vectorized_carry_self_update_row(
            pre_arr,
            vote_arr,
            decay_numerator=DEFAULT_DECAY_NUMERATOR,
            decay_denominator=DEFAULT_DECAY_DENOMINATOR,
        )
        if hot_indices.size:
            pos_in_hot = np.searchsorted(hot_indices, active_sorted)
            in_hot = np.zeros(active_sorted.shape[0], dtype=bool)
            hot_match = pos_in_hot < hot_indices.size
            if hot_match.any():
                matched = hot_indices[pos_in_hot[hot_match]] == active_sorted[hot_match]
                in_hot[np.where(hot_match)[0][matched]] = True
        else:
            in_hot = np.zeros(active_sorted.shape[0], dtype=bool)
        vote_touched_mask = (
            np.isin(active_sorted, touched_arr)
            if touched_arr.size
            else np.zeros(active_sorted.shape[0], dtype=bool)
        )
        in_proxy_mask = (
            np.isin(active_sorted, proxy_indices)
            if proxy_indices.size
            else np.zeros(active_sorted.shape[0], dtype=bool)
        )
        promote_mask = (
            in_hot
            | vote_touched_mask
            | in_proxy_mask
            | (np.abs(post_arr) >= int(promote_at))
        )
        promotion_count = int(np.sum(promote_mask & ~in_hot))

        q_arr = np.zeros(active_sorted.shape[0], dtype=np.int32)
        if self.q_levels:
            q_keys = np.array(sorted(self.q_levels), dtype=np.int32)
            q_vals = np.array([self.q_levels[int(k)] for k in q_keys], dtype=np.int32)
            q_pos = np.searchsorted(q_keys, active_sorted)
            q_hit = q_pos < q_keys.size
            if q_hit.any():
                q_match = q_keys[q_pos[q_hit]] == active_sorted[q_hit]
                q_arr[np.where(q_hit)[0][q_match]] = q_vals[q_pos[q_hit][q_match]]

        cross_mask = vectorized_crosses_threshold(
            post_arr,
            q_arr,
            threshold_abs=int(self.threshold_abs),
        )
        backlog_arr = (
            np.array(sorted(self.backlog), dtype=np.int32)
            if self.backlog
            else np.array([], dtype=np.int32)
        )
        in_backlog = (
            np.isin(active_sorted, backlog_arr)
            if backlog_arr.size
            else np.zeros(active_sorted.shape[0], dtype=bool)
        )
        journal_has = in_hot | promote_mask
        demotion_decay_mask = (
            journal_has
            & ~cross_mask
            & ~vote_touched_mask
            & ~in_proxy_mask
            & ~in_backlog
            & (np.abs(post_arr) < int(self.demotion_band))
        )
        demotion_cross_mask = cross_mask & journal_has
        update_mask = journal_has & ~cross_mask & ~demotion_decay_mask

        crossing_indices = [int(x) for x in active_sorted[cross_mask]]
        applied_indices = list(crossing_indices)
        demotion_on_crossing = int(np.sum(demotion_cross_mask))
        demotion_on_decay = int(np.sum(demotion_decay_mask))

        for flat_index, post in zip(active_sorted[cross_mask], post_arr[cross_mask]):
            idx = int(flat_index)
            direction = 1 if int(post) >= 0 else 0
            residual_mag = min(abs(int(post)), int(self.threshold_abs) - 1)
            self.events.append(
                EventCodedAccEvent(
                    flat_index=idx,
                    direction=int(direction),
                    residual_mag=int(residual_mag),
                    event_type=1,
                )
            )
            self.q_levels[idx] = 1 if int(post) >= 0 else -1

        remove_idx = active_sorted[demotion_cross_mask | demotion_decay_mask]
        write_mask = promote_mask | update_mask
        upd_idx = active_sorted[write_mask]
        upd_val = post_arr[write_mask].astype(np.int16)
        new_idx, new_val = merge_hot_table_arrays(
            hot_indices,
            hot_values,
            remove_idx,
            upd_idx,
            upd_val,
        )
        self._hot.replace_arrays(new_idx, new_val)
        self._invalidate_packed_caches()

        record = StepSurfaceRecord(
            step_index=int(step_index),
            crossing_indices=tuple(crossing_indices),
            applied_indices=tuple(applied_indices),
            backlog_indices=tuple(sorted(self.backlog)),
            q_levels=dict(self.q_levels),
            hot_exact_row_count=len(self._hot),
            promotion_count=int(promotion_count),
            demotion_on_decay_count=int(demotion_on_decay),
            demotion_on_crossing_count=int(demotion_on_crossing),
        )
        self.step_records.append(record)
        return record

    def to_checkpoint_payload(self):
        return pack_event_coded_acc_checkpoint_v1(
            logical_numel=int(self.logical_numel),
            events=tuple(self.events),
            backlog_indices=tuple(sorted(self.backlog)),
            hot_exact_indices=self._hot.indices_array(),
            hot_exact_values=self._hot.values_array(),
        )


def apply_step_dict_reference(
    carrier: EventCodedAccLiveState,
    step_index: int,
    *,
    votes: Mapping[int, int],
    hot_risk_override: Iterable[int] | None = None,
) -> StepSurfaceRecord:
    """Frozen dict/deepcopy oracle — independent deepcopy baseline for equivalence tests."""

    oracle = copy.deepcopy(_carrier_as_dict_state(carrier))
    return _apply_step_dict_impl(
        oracle,
        int(step_index),
        votes=dict(votes),
        hot_risk_override=hot_risk_override,
    )


def _carrier_as_dict_state(carrier: EventCodedAccLiveState) -> EventCodedAccLiveState:
    return EventCodedAccLiveState(
        logical_numel=int(carrier.logical_numel),
        cold_default=int(carrier.cold_default),
        threshold_abs=int(carrier.threshold_abs),
        demotion_band=int(carrier.demotion_band),
        _hot=_PackedHotTable.from_dict(carrier._hot.to_dict()),
        events=list(carrier.events),
        backlog=set(carrier.backlog),
        q_levels=dict(carrier.q_levels),
        step_records=list(carrier.step_records),
        dense_accumulator_materialized_numel=int(carrier.dense_accumulator_materialized_numel),
    )


def _apply_step_dict_impl(
    carrier: EventCodedAccLiveState,
    step_index: int,
    *,
    votes: Mapping[int, int],
    hot_risk_override: Iterable[int] | None = None,
) -> StepSurfaceRecord:
    """Original dict-based apply_step semantics (used only by the frozen oracle)."""

    hot_dict = carrier._hot.to_dict()
    vote_map = {int(k): int(v) for k, v in votes.items()}
    touched = set(vote_map)
    pre_carries = {index: carrier.reconstruct_lane(index) for index in hot_dict}
    pre_carries.update({index: carrier.reconstruct_lane(index) for index in touched})
    proxy = (
        set(int(item) for item in hot_risk_override)
        if hot_risk_override is not None
        else hot_risk_proxy_indices(pre_carries, threshold_abs=carrier.threshold_abs)
    )
    active = set(hot_dict) | touched | proxy
    post_carries: dict[int, int] = {}
    promotion_count = 0
    for flat_index in active:
        pre = carrier.reconstruct_lane(flat_index)
        vote = int(vote_map.get(flat_index, 0))
        post = carry_self_update_row(
            pre,
            vote,
            decay_numerator=DEFAULT_DECAY_NUMERATOR,
            decay_denominator=DEFAULT_DECAY_DENOMINATOR,
        )
        post_carries[int(flat_index)] = int(post)
        if carrier._should_promote(
            flat_index,
            post_carry=post,
            vote_touched=flat_index in touched,
            hot_risk_proxy=proxy,
        ):
            if flat_index not in hot_dict:
                promotion_count += 1
            carrier._hot.set_lane(int(flat_index), int(post))
            hot_dict[int(flat_index)] = int(post)

    crossing_indices: list[int] = []
    applied_indices: list[int] = []
    demotion_on_crossing = 0
    demotion_on_decay = 0
    for flat_index, post in sorted(post_carries.items()):
        q = carrier.q_level(flat_index)
        fired = crossing_bool_w6(post, q, threshold_abs=carrier.threshold_abs)
        if fired:
            crossing_indices.append(int(flat_index))
            direction = 1 if int(post) >= 0 else 0
            residual_mag = min(abs(int(post)), int(carrier.threshold_abs) - 1)
            carrier.events.append(
                EventCodedAccEvent(
                    flat_index=int(flat_index),
                    direction=int(direction),
                    residual_mag=int(residual_mag),
                    event_type=1,
                )
            )
            encode_post_flip_residual(
                1 if direction else -1,
                residual_mag,
                threshold_abs=carrier.threshold_abs,
            )
            new_q = 1 if int(post) >= 0 else -1
            carrier.q_levels[int(flat_index)] = int(new_q)
            applied_indices.append(int(flat_index))
            if carrier._maybe_demote(
                flat_index,
                post_carry=post,
                fired_crossing=True,
                hot_risk_proxy=proxy,
                vote_touched=flat_index in touched,
            ):
                demotion_on_crossing += 1
                hot_dict.pop(int(flat_index), None)
        elif carrier._maybe_demote(
            flat_index,
            post_carry=post,
            fired_crossing=False,
            hot_risk_proxy=proxy,
            vote_touched=flat_index in touched,
        ):
            demotion_on_decay += 1
            hot_dict.pop(int(flat_index), None)
        elif carrier._hot.contains(int(flat_index)):
            carrier._hot.set_lane(int(flat_index), int(post))
            hot_dict[int(flat_index)] = int(post)

    record = StepSurfaceRecord(
        step_index=int(step_index),
        crossing_indices=tuple(crossing_indices),
        applied_indices=tuple(applied_indices),
        backlog_indices=tuple(sorted(carrier.backlog)),
        q_levels=dict(carrier.q_levels),
        hot_exact_row_count=len(carrier._hot),
        promotion_count=int(promotion_count),
        demotion_on_decay_count=int(demotion_on_decay),
        demotion_on_crossing_count=int(demotion_on_crossing),
    )
    carrier.step_records.append(record)
    return record


@dataclass(frozen=True)
class DenseOracleState:
    logical_numel: int
    threshold_abs: int = DEFAULT_CROSSING_THRESHOLD_ABS
    cold_default: int = DEFAULT_COLD_DEFAULT
    accumulators: list[int] = field(default_factory=list)
    q_levels: list[int] = field(default_factory=list)
    step_records: list[StepSurfaceRecord] = field(default_factory=list)

    @classmethod
    def zeros(cls, logical_numel: int) -> DenseOracleState:
        return cls(
            logical_numel=int(logical_numel),
            accumulators=[0] * int(logical_numel),
            q_levels=[0] * int(logical_numel),
        )

    def apply_step(self, step_index: int, *, votes: Mapping[int, int]) -> StepSurfaceRecord:
        vote_map = {int(k): int(v) for k, v in votes.items()}
        crossing_indices: list[int] = []
        applied_indices: list[int] = []
        for flat_index in range(int(self.logical_numel)):
            pre = int(self.accumulators[flat_index])
            vote = int(vote_map.get(flat_index, 0))
            post = carry_self_update_row(
                pre,
                vote,
                decay_numerator=DEFAULT_DECAY_NUMERATOR,
                decay_denominator=DEFAULT_DECAY_DENOMINATOR,
            )
            self.accumulators[flat_index] = int(post)
            q = int(self.q_levels[flat_index])
            if crossing_bool_w6(post, q, threshold_abs=self.threshold_abs):
                crossing_indices.append(int(flat_index))
                self.q_levels[flat_index] = 1 if int(post) >= 0 else -1
                applied_indices.append(int(flat_index))
        record = StepSurfaceRecord(
            step_index=int(step_index),
            crossing_indices=tuple(crossing_indices),
            applied_indices=tuple(applied_indices),
            backlog_indices=(),
            q_levels={index: int(value) for index, value in enumerate(self.q_levels)},
            hot_exact_row_count=int(self.logical_numel),
            promotion_count=0,
            demotion_on_decay_count=0,
            demotion_on_crossing_count=0,
        )
        self.step_records.append(record)
        return record


def _decisive_q_snapshot(record: StepSurfaceRecord) -> dict[int, int]:
    """Compare q only on decisive lanes (crossing/applied), not full-tensor oracle keys."""

    indices = set(record.applied_indices) | set(record.crossing_indices)
    return {int(index): int(record.q_levels.get(int(index), 0)) for index in sorted(indices)}


def observed_surfaces_dict(record: StepSurfaceRecord) -> dict[str, Any]:
    return {
        "crossing_flat_indices": [int(value) for value in record.crossing_indices],
        "applied_flat_indices": [int(value) for value in record.applied_indices],
        "decisive_q_snapshot": {
            str(key): int(value) for key, value in _decisive_q_snapshot(record).items()
        },
    }


def decisive_surface_drift_count(
    carrier_records: Sequence[StepSurfaceRecord],
    oracle_records: Sequence[StepSurfaceRecord],
) -> int:
    drift = 0
    for carrier, oracle in zip(carrier_records, oracle_records):
        if carrier.crossing_indices != oracle.crossing_indices:
            drift += 1
        if carrier.applied_indices != oracle.applied_indices:
            drift += 1
        if _decisive_q_snapshot(carrier) != _decisive_q_snapshot(oracle):
            drift += 1
    return int(drift)


def decisive_surface_drift_details(
    carrier_records: Sequence[StepSurfaceRecord],
    oracle_records: Sequence[StepSurfaceRecord],
) -> list[dict[str, object]]:
    """Per-step mismatch dump for harness root-cause classification."""

    details: list[dict[str, object]] = []
    for carrier, oracle in zip(carrier_records, oracle_records):
        mismatches: list[str] = []
        if carrier.crossing_indices != oracle.crossing_indices:
            mismatches.append("crossing_indices")
        if carrier.applied_indices != oracle.applied_indices:
            mismatches.append("applied_indices")
        if _decisive_q_snapshot(carrier) != _decisive_q_snapshot(oracle):
            mismatches.append("decisive_q_snapshot")
        if mismatches:
            details.append(
                {
                    "step_index": int(carrier.step_index),
                    "mismatch_fields": tuple(mismatches),
                    "carrier_crossing": tuple(carrier.crossing_indices),
                    "oracle_crossing": tuple(oracle.crossing_indices),
                    "carrier_applied": tuple(carrier.applied_indices),
                    "oracle_applied": tuple(oracle.applied_indices),
                    "carrier_q": _decisive_q_snapshot(carrier),
                    "oracle_q": _decisive_q_snapshot(oracle),
                }
            )
    return details
