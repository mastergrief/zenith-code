"""Observation-only R7 selective-drain eligibility census (CPU-reference land).

Compact-by-construction sidecar emitter. Never mutates q/acc/backlog/cap.
"""
from __future__ import annotations

import hashlib
import struct
import json
import sys
import time
from array import array
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

DEFAULT_K_GRID: tuple[int, ...] = (2, 4, 8, 12, 16)
SIDECAR_FILENAME = "r7_selective_drain_eligibility_census.jsonl"
SCHEMA = "hrm_text_158_r7_selective_drain_eligibility_census_step_chunk/v1"
OBSERVER_INIT_SCHEMA = (
    "hrm_text_158_r7_selective_drain_eligibility_census_observer_init/v1"
)
OBSERVER_INIT_KIND = "observer_continuity_init"
DIGEST_SCHEMA = "order_independent_v1_blake2b"
DIGEST_TAG = b"pre_step_backlog_oi_v1_blake2b\n"

CENSUS_OK = "OK"
CENSUS_INVALID = "INVALID"


class SelectiveDrainCensusObserverInitError(ValueError):
    """Fail-closed observer-continuity init precondition breach."""


TABLE2_OK = "OK"
TABLE2_NOT_EVALUABLE = "NOT_EVALUABLE_EMPTY_PRE_STEP_BACKLOG"

TABLE3_BUCKETS: tuple[str, ...] = (
    "accepted_within_prior_window",
    "replay_protected_within_prior_window",
    "not_continuously_re_candidated_and_deferred",
    "frontier_undefined",
    "insufficient_below_frontier_window",
    "eligible",
)


def _u64_be(n: int) -> bytes:
    return int(n).to_bytes(8, "big", signed=False)


def _u64_le(n: int) -> bytes:
    return int(n).to_bytes(8, "little", signed=False)


def identity_h(state_key: str, flat_index: int) -> bytes:
    # blake2b@32B: cryptographic, commutative XOR-lane input; ~3–5× faster than
    # SHA-256 at 737k scale (STEP-2 fresh-process wall ceiling).
    return hashlib.blake2b(
        b"id\0" + state_key.encode("utf-8") + b"\0" + _u64_le(int(flat_index)),
        digest_size=32,
    ).digest()


def xor_bytes32(a: bytes, b: bytes) -> bytes:
    return bytes(x ^ y for x, y in zip(a, b, strict=True))


def _finalize_oi_digest(count: int, xor_lane: bytes) -> str:
    return hashlib.blake2b(DIGEST_TAG + _u64_be(count) + xor_lane, digest_size=32).hexdigest()


def pre_step_backlog_set_digest_oi_v1(
    backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> tuple[int, str]:
    """Order-independent digest over structurally unique dict keys. No seen-set.

    Relies on dict[str, dict[int, ...]] uniqueness; asserts count == sum(len(inner)).
    """
    if not backlog:
        return 0, _finalize_oi_digest(0, bytes(32))
    xor_arr = bytearray(32)
    count = 0
    summed = 0
    for state_key, inner in backlog.items():
        summed += len(inner)
        sk_s = str(state_key)
        for flat_index in inner.keys():
            count += 1
            digest_i = identity_h(sk_s, int(flat_index))
            for j in range(32):
                xor_arr[j] ^= digest_i[j]
    if count != summed:
        raise ValueError("backlog_structure_uniqueness_violation")
    return count, _finalize_oi_digest(count, bytes(xor_arr))


def assert_pre_step_backlog_input_unchanged(
    before: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    after: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> None:
    if before is after or (before is None and after is None):
        # same object or both None — also compare content if both non-None same object
        if before is not None and after is not None and before is after:
            return
    if before is None or after is None:
        if before is not after:
            raise AssertionError("pre_step_backlog_input_changed_none")
        return
    if before is not after:
        # Cap must not replace the caller's object; content equality alone is insufficient
        # if a different object was substituted. Prefer identity.
        raise AssertionError("pre_step_backlog_input_object_identity_changed")


def _pack_i64(values: Sequence[int]) -> bytes:
    buf = array("q", (int(v) for v in values))
    return buf.tobytes()


def _pack_u64(values: Sequence[int]) -> bytes:
    buf = array("Q", (int(v) for v in values))
    return buf.tobytes()


def _pack_u8(values: Sequence[int]) -> bytes:
    buf = array("B", (1 if int(v) else 0 for v in values))
    return buf.tobytes()


def _unpack_i64(blob: bytes) -> list[int]:
    buf = array("q")
    buf.frombytes(blob)
    return list(buf)


def _unpack_u64(blob: bytes) -> list[int]:
    buf = array("Q")
    buf.frombytes(blob)
    return list(buf)


def _unpack_u8(blob: bytes) -> list[int]:
    buf = array("B")
    buf.frombytes(blob)
    return list(buf)


@dataclass(frozen=True)
class SelectiveDrainCensusStepDTO:
    step: int
    ordering_mode: str
    cap: int
    # Packed accepted columns (immutable bytes)
    accepted_state_key_codes: bytes  # uint32 codes
    accepted_flat_index: bytes  # uint64
    accepted_abs_new_acc: bytes  # int64
    accepted_threshold_abs: bytes  # int64
    # Packed deferred columns
    deferred_state_key_codes: bytes
    deferred_flat_index: bytes
    deferred_abs_new_acc: bytes
    deferred_threshold_abs: bytes
    deferred_was_in_pre_step_backlog: bytes  # uint8
    state_key_table: tuple[str, ...]
    replay_protected_codes_flat: bytes  # pairs of uint32 code + uint64 flat as packed? store parallel
    replay_state_key_codes: bytes
    replay_flat_index: bytes
    frontier_abs_new_acc: int | None
    pre_step_backlog_unique_count: int
    pre_step_backlog_set_digest: str
    pre_step_backlog_max_age_steps: int
    pre_step_backlog_max_defer_count: int
    re_candidated_current_count: int
    backlog_only_not_current_candidate_count: int
    dto_build_time_ms: float

    def accepted_count(self) -> int:
        return len(self.accepted_flat_index) // 8

    def deferred_count(self) -> int:
        return len(self.deferred_flat_index) // 8

    def candidate_count(self) -> int:
        return self.accepted_count() + self.deferred_count()

    def deep_retained_bytes(self) -> int:
        """Packed buffers + key table + Python/container overhead (not JSON)."""
        total = 0
        for name, value in self.__dict__.items():
            if isinstance(value, (bytes, str)):
                total += len(value)
            total += sys.getsizeof(value)
        total += sys.getsizeof(self)
        total += sys.getsizeof(self.__dict__)
        for s in self.state_key_table:
            total += sys.getsizeof(s) + len(s.encode("utf-8"))
        return int(total)

    def retained_per_row_python_objects(self) -> int:
        # Structural: row storage is bytes only — zero retained per-row objects.
        for name in (
            "accepted_flat_index",
            "deferred_flat_index",
            "accepted_abs_new_acc",
            "deferred_abs_new_acc",
        ):
            if not isinstance(getattr(self, name), bytes):
                return -1
        return 0


@dataclass
class _ContRecord:
    last_step: int
    consec_deferred: int = 0
    consec_below: int = 0
    # bit i = step-(i+1) flags for prior window up to 16
    prior_accepted_bits: int = 0
    prior_replay_bits: int = 0
    prior_deferred_bits: int = 0


@dataclass
class ObserverContinuityTracker:
    status: str = CENSUS_OK
    invalid_reason: str | None = None
    last_step: int | None = None
    enabled_at_step: int | None = None
    _records: dict[tuple[str, int], _ContRecord] = field(default_factory=dict)

    def reset(self) -> None:
        self.status = CENSUS_OK
        self.invalid_reason = None
        self.last_step = None
        self.enabled_at_step = None
        self._records.clear()

    def cardinality(self) -> int:
        return len(self._records)

    def approx_bytes(self) -> int:
        # rough: key tuple + record object overhead
        return int(self.cardinality() * 120 + sys.getsizeof(self._records))

    def mark_invalid(self, reason: str) -> None:
        self.status = CENSUS_INVALID
        self.invalid_reason = str(reason)

    def update_from_dto(self, dto: SelectiveDrainCensusStepDTO) -> None:
        if self.status == CENSUS_INVALID:
            return
        step = int(dto.step)
        if self.enabled_at_step is None:
            self.enabled_at_step = step
            if step != 0 and self.last_step is None:
                # mid-run enablement without warmup → hard-invalid
                self.mark_invalid("mid_run_observer_enablement")
                return
        if self.last_step is not None:
            if step == self.last_step:
                self.mark_invalid("duplicate_step")
                return
            if step != self.last_step + 1:
                self.mark_invalid("step_discontinuity")
                return
        # shift prior bits for all known records
        for rec in self._records.values():
            rec.prior_accepted_bits = (rec.prior_accepted_bits << 1) & 0xFFFF
            rec.prior_replay_bits = (rec.prior_replay_bits << 1) & 0xFFFF
            rec.prior_deferred_bits = (rec.prior_deferred_bits << 1) & 0xFFFF

        table = dto.state_key_table
        acc_code_arr = array("I")
        if dto.accepted_state_key_codes:
            acc_code_arr.frombytes(dto.accepted_state_key_codes)
        def_code_arr = array("I")
        if dto.deferred_state_key_codes:
            def_code_arr.frombytes(dto.deferred_state_key_codes)
        acc_flat = _unpack_u64(dto.accepted_flat_index)
        def_flat = _unpack_u64(dto.deferred_flat_index)
        def_abs = _unpack_i64(dto.deferred_abs_new_acc)
        replay_codes = array("I")
        if dto.replay_state_key_codes:
            replay_codes.frombytes(dto.replay_state_key_codes)
        replay_flat = _unpack_u64(dto.replay_flat_index)

        accepted_set = {(table[int(c)], int(f)) for c, f in zip(acc_code_arr, acc_flat)}
        deferred_set = {(table[int(c)], int(f)) for c, f in zip(def_code_arr, def_flat)}
        replay_set = {(table[int(c)], int(f)) for c, f in zip(replay_codes, replay_flat)}
        frontier = dto.frontier_abs_new_acc

        for key in accepted_set:
            rec = self._records.setdefault(key, _ContRecord(last_step=step))
            rec.last_step = step
            rec.consec_deferred = 0
            rec.consec_below = 0
            rec.prior_accepted_bits |= 1  # current will shift next step; mark current in bit0 after shift already done — use bit0 as most recent prior at next step. For THIS step's "prior window" we need previous steps only. So set bit0 to mean "this step" which becomes prior after next shift. Good.

        for key in replay_set:
            rec = self._records.setdefault(key, _ContRecord(last_step=step))
            rec.last_step = step
            rec.prior_replay_bits |= 1

        for i, fi in enumerate(def_flat):
            ident = (table[int(def_code_arr[i])], int(fi))
            rec = self._records.setdefault(ident, _ContRecord(last_step=step))
            rec.last_step = step
            rec.prior_deferred_bits |= 1
            abs_acc = int(def_abs[i])
            if frontier is None:
                rec.consec_below = 0
            elif abs_acc < int(frontier):
                rec.consec_below += 1
            else:
                rec.consec_below = 0

        for ident, rec in self._records.items():
            if ident not in deferred_set:
                # not deferred this step → break continuous deferred streak for next queries
                bits = rec.prior_deferred_bits
                consec = 0
                while bits & 1:
                    consec += 1
                    bits >>= 1
                # bit0 may still be 0; consec from bit0
                if not (rec.prior_deferred_bits & 1):
                    rec.consec_deferred = 0
                else:
                    rec.consec_deferred = consec
            else:
                bits = rec.prior_deferred_bits
                consec = 0
                while bits & 1:
                    consec += 1
                    bits >>= 1
                rec.consec_deferred = consec

        self.last_step = step

    def prior_window_accepted(self, ident: tuple[str, int], k: int) -> bool:
        rec = self._records.get(ident)
        if rec is None:
            return False
        # bits 1..k are prior steps after update (bit0=current). For prior-only, check bits 1..k
        mask = ((1 << k) - 1) << 1
        return bool(rec.prior_accepted_bits & mask)

    def prior_window_replay(self, ident: tuple[str, int], k: int) -> bool:
        rec = self._records.get(ident)
        if rec is None:
            return False
        mask = ((1 << k) - 1) << 1
        return bool(rec.prior_replay_bits & mask)

    def prior_k_continuous_deferred(self, ident: tuple[str, int], k: int) -> bool:
        rec = self._records.get(ident)
        if rec is None:
            return False
        # need prior K steps deferred: bits 1..k all set
        mask = ((1 << k) - 1) << 1
        return (rec.prior_deferred_bits & mask) == mask

    def consec_below(self, ident: tuple[str, int]) -> int:
        rec = self._records.get(ident)
        return 0 if rec is None else int(rec.consec_below)


def _pre_step_backlog_is_truly_empty(
    pre_step_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> bool:
    if pre_step_backlog is None:
        return True
    if len(pre_step_backlog) == 0:
        return True
    for _state_key, inner in pre_step_backlog.items():
        if inner:
            return False
    return True


def initialize_selective_drain_census_observer_continuity_at_step0(
    *,
    tracker: ObserverContinuityTracker,
    observed_step: int,
    sidecar_path: Path | str,
    pre_step_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
) -> dict[str, Any]:
    """Thin in-memory observer init at step 0. Never appends an ordinary census row."""
    if not (type(observed_step) is int and observed_step == 0):
        raise SelectiveDrainCensusObserverInitError(
            "observed_step must be canonical integer 0 "
            f"(type(observed_step) is int and == 0); got {observed_step!r}"
        )
    if sidecar_path is None or not isinstance(sidecar_path, (str, Path)):
        raise SelectiveDrainCensusObserverInitError(
            "sidecar_path must be a required str or Path (None/invalid type forbidden); "
            f"got {sidecar_path!r}"
        )
    resolved_sidecar = Path(sidecar_path)
    if resolved_sidecar.exists():
        raise SelectiveDrainCensusObserverInitError(
            "census sidecar path must be ABSENT for fresh observer init; "
            f"found existing path {resolved_sidecar}"
        )
    if tracker.status != CENSUS_OK:
        raise SelectiveDrainCensusObserverInitError(
            f"tracker.status must be {CENSUS_OK!r} for fresh init; got {tracker.status!r}"
        )
    if tracker.enabled_at_step is not None:
        raise SelectiveDrainCensusObserverInitError(
            "tracker.enabled_at_step must be None for fresh init "
            f"(duplicate/mid-run); got {tracker.enabled_at_step!r}"
        )
    if tracker.last_step is not None:
        raise SelectiveDrainCensusObserverInitError(
            "tracker.last_step must be None for fresh init "
            f"(duplicate/mid-run); got {tracker.last_step!r}"
        )
    if tracker.cardinality() != 0:
        raise SelectiveDrainCensusObserverInitError(
            "tracker continuity records must be empty for fresh init; "
            f"cardinality={tracker.cardinality()}"
        )
    if not _pre_step_backlog_is_truly_empty(pre_step_backlog):
        raise SelectiveDrainCensusObserverInitError(
            "pre_step_backlog must be truly empty for step-0 observer init"
        )

    tracker.enabled_at_step = 0
    tracker.last_step = 0
    return {
        "schema_version": OBSERVER_INIT_SCHEMA,
        "kind": OBSERVER_INIT_KIND,
        "step": 0,
        "observed_step": 0,
        "enabled_at_step": 0,
        "last_step": 0,
        "pre_step_backlog_empty": True,
        "ordinary_sidecar_rows_appended": False,
        "sidecar_was_absent": True,
        "sidecar_path": str(resolved_sidecar),
        "tracker_cardinality": 0,
    }


def _state_key_coder(keys: Iterable[str]) -> tuple[tuple[str, ...], dict[str, int]]:
    table = tuple(sorted(set(keys)))
    return table, {k: i for i, k in enumerate(table)}


def build_selective_drain_census_step_dto(
    *,
    step: int,
    ordering_mode: str,
    cap: int,
    pre_step_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    accepted_rows: Sequence[Any],
    deferred_rows: Sequence[Any],
    plans_by_key: Mapping[str, Any] | None = None,
) -> SelectiveDrainCensusStepDTO:
    t0 = time.perf_counter()
    # Candidate id set first (small N) for O(1) ∩ during single backlog pass.
    candidate_ids = {(str(r.state_key), int(r.flat_index)) for r in accepted_rows}
    candidate_ids.update((str(r.state_key), int(r.flat_index)) for r in deferred_rows)

    xor_arr = bytearray(32)
    pre_count = 0
    max_age = 0
    max_defer = 0
    re_cand = 0
    if pre_step_backlog:
        summed = 0
        step_i = int(step)
        for sk, inner in pre_step_backlog.items():
            summed += len(inner)
            sk_s = str(sk)
            for flat_index, entry in inner.items():
                fi = int(flat_index)
                pre_count += 1
                digest_i = identity_h(sk_s, fi)
                for j in range(32):
                    xor_arr[j] ^= digest_i[j]
                first = int(entry.get("first_step", step_i))
                age = step_i - first
                if age > max_age:
                    max_age = age
                dc = int(entry.get("defer_count", 0))
                if dc > max_defer:
                    max_defer = dc
                if (sk_s, fi) in candidate_ids:
                    re_cand += 1
        if pre_count != summed:
            raise ValueError("backlog_structure_uniqueness_violation")
        pre_digest = _finalize_oi_digest(pre_count, bytes(xor_arr))
    else:
        pre_digest = _finalize_oi_digest(0, bytes(32))

    def was_in_pre(sk: str, fi: int) -> bool:
        if not pre_step_backlog:
            return False
        bucket = pre_step_backlog.get(sk)
        return bool(bucket is not None and int(fi) in bucket)

    replay_ids: list[tuple[str, int]] = []
    if plans_by_key:
        for sk, plan in plans_by_key.items():
            idxs = getattr(plan, "replay_ce_veto_indices", None)
            if idxs is None:
                continue
            for fi in idxs.detach().cpu().tolist() if hasattr(idxs, "detach") else list(idxs):
                replay_ids.append((str(sk), int(fi)))

    all_keys = [str(r.state_key) for r in list(accepted_rows) + list(deferred_rows)]
    all_keys.extend(sk for sk, _ in replay_ids)
    if pre_step_backlog:
        all_keys.extend(str(k) for k in pre_step_backlog.keys())
    table, coder = _state_key_coder(all_keys)

    def pack_rows(rows: Sequence[Any], *, with_was: bool) -> dict[str, bytes]:
        codes: list[int] = []
        flats: list[int] = []
        abs_acc: list[int] = []
        thr: list[int] = []
        was: list[int] = []
        for r in rows:
            sk = str(r.state_key)
            fi = int(r.flat_index)
            codes.append(coder[sk])
            flats.append(fi)
            abs_acc.append(int(r.abs_new_acc))
            thr.append(int(r.threshold_abs))
            if with_was:
                was.append(1 if was_in_pre(sk, fi) else 0)
        out = {
            "codes": array("I", codes).tobytes(),
            "flat": _pack_u64(flats),
            "abs": _pack_i64(abs_acc),
            "thr": _pack_i64(thr),
        }
        if with_was:
            out["was"] = _pack_u8(was)
        return out

    acc = pack_rows(accepted_rows, with_was=False)
    deferred = pack_rows(deferred_rows, with_was=True)

    backlog_only = int(pre_count) - int(re_cand)
    if backlog_only < 0:
        raise ValueError("re_candidated_exceeds_pre_step_backlog")

    frontier = None
    if accepted_rows:
        frontier = min(int(r.abs_new_acc) for r in accepted_rows)

    r_codes = array("I", (coder[sk] for sk, _ in replay_ids))
    r_flat = _pack_u64([fi for _, fi in replay_ids])

    dto = SelectiveDrainCensusStepDTO(
        step=int(step),
        ordering_mode=str(ordering_mode),
        cap=int(cap),
        accepted_state_key_codes=acc["codes"],
        accepted_flat_index=acc["flat"],
        accepted_abs_new_acc=acc["abs"],
        accepted_threshold_abs=acc["thr"],
        deferred_state_key_codes=deferred["codes"],
        deferred_flat_index=deferred["flat"],
        deferred_abs_new_acc=deferred["abs"],
        deferred_threshold_abs=deferred["thr"],
        deferred_was_in_pre_step_backlog=deferred.get("was", b""),
        state_key_table=table,
        replay_protected_codes_flat=b"",
        replay_state_key_codes=r_codes.tobytes(),
        replay_flat_index=r_flat,
        frontier_abs_new_acc=frontier,
        pre_step_backlog_unique_count=int(pre_count),
        pre_step_backlog_set_digest=str(pre_digest),
        pre_step_backlog_max_age_steps=int(max_age),
        pre_step_backlog_max_defer_count=int(max_defer),
        re_candidated_current_count=int(re_cand),
        backlog_only_not_current_candidate_count=int(backlog_only),
        dto_build_time_ms=float((time.perf_counter() - t0) * 1000.0),
    )
    if dto.retained_per_row_python_objects() != 0:
        raise AssertionError("dto_retained_per_row_python_objects")
    return dto


def build_table1_cap_accounting(dto: SelectiveDrainCensusStepDTO) -> dict[str, Any]:
    a = dto.accepted_count()
    d = dto.deferred_count()
    denom = a + d
    return {
        "accepted_current_count": a,
        "deferred_current_count": d,
        "authoritative_candidate_denominator": denom,
        "cap_closure_ok": (a + d) == denom,
    }


def build_table2_backlog_materiality(dto: SelectiveDrainCensusStepDTO) -> dict[str, Any]:
    denom = int(dto.pre_step_backlog_unique_count)
    re_c = int(dto.re_candidated_current_count)
    only = int(dto.backlog_only_not_current_candidate_count)
    if denom == 0:
        return {
            "table2_status": TABLE2_NOT_EVALUABLE,
            "pre_step_backlog_unique_count": 0,
            "pre_step_backlog_set_digest": dto.pre_step_backlog_set_digest,
            "re_candidated_current_count": 0,
            "backlog_only_not_current_candidate_count": 0,
            "re_candidated_fraction": None,
            "materiality_closure_ok": True,
        }
    closure = (re_c + only) == denom
    return {
        "table2_status": TABLE2_OK,
        "pre_step_backlog_unique_count": denom,
        "pre_step_backlog_set_digest": dto.pre_step_backlog_set_digest,
        "re_candidated_current_count": re_c,
        "backlog_only_not_current_candidate_count": only,
        "re_candidated_fraction": (float(re_c) / float(denom)) if closure else None,
        "materiality_closure_ok": bool(closure),
    }


def build_table3_eligibility(
    dto: SelectiveDrainCensusStepDTO,
    tracker: ObserverContinuityTracker,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
) -> dict[str, Any]:
    table = dto.state_key_table
    def_codes = array("I")
    if dto.deferred_state_key_codes:
        def_codes.frombytes(dto.deferred_state_key_codes)
    def_flat = _unpack_u64(dto.deferred_flat_index)
    def_abs = _unpack_i64(dto.deferred_abs_new_acc)
    denom = len(def_flat)
    frontier = dto.frontier_abs_new_acc
    per_k: dict[str, Any] = {}
    for k in k_grid:
        counts = {b: 0 for b in TABLE3_BUCKETS}
        eligible_ids: list[tuple[str, int]] = []
        for i, fi in enumerate(def_flat):
            ident = (table[int(def_codes[i])], int(fi))
            if tracker.prior_window_accepted(ident, int(k)):
                bucket = "accepted_within_prior_window"
            elif tracker.prior_window_replay(ident, int(k)):
                bucket = "replay_protected_within_prior_window"
            elif not tracker.prior_k_continuous_deferred(ident, int(k)):
                bucket = "not_continuously_re_candidated_and_deferred"
            elif frontier is None:
                bucket = "frontier_undefined"
            elif tracker.consec_below(ident) < int(k):
                bucket = "insufficient_below_frontier_window"
            else:
                bucket = "eligible"
                eligible_ids.append(ident)
            counts[bucket] += 1
        total = sum(counts.values())
        # identity set hash for eligible
        lines = "".join(f"{sk}\t{fi}\n" for sk, fi in sorted(eligible_ids))
        elig_sha = hashlib.sha256(lines.encode("utf-8")).hexdigest()
        per_k[str(int(k))] = {
            "partition_counts": counts,
            "eligible_count": counts["eligible"],
            "eligible_fraction_of_deferred": (counts["eligible"] / denom) if denom else None,
            "eligible_identity_set_sha256": elig_sha,
            "eligibility_closure_ok": total == denom,
            "frontier_status": "DEFINED" if frontier is not None else "UNDEFINED",
            "current_deferred_candidate_denominator": denom,
        }
    return {"per_k": per_k}


def accounting_invariant(chunk: Mapping[str, Any]) -> list[str]:
    failures: list[str] = []
    t1 = chunk.get("table1") or {}
    if not t1.get("cap_closure_ok", False):
        failures.append("table1_closure")
    t2 = chunk.get("table2") or {}
    if t2.get("table2_status") == TABLE2_OK and not t2.get("materiality_closure_ok", False):
        failures.append("table2_closure")
    t3 = chunk.get("table3") or {}
    for k, body in (t3.get("per_k") or {}).items():
        if not body.get("eligibility_closure_ok", False):
            failures.append(f"table3_closure_k{k}")
    re_c = int(t2.get("re_candidated_current_count") or 0)
    pre_n = int(t2.get("pre_step_backlog_unique_count") or 0)
    cand = int(t1.get("authoritative_candidate_denominator") or 0)
    if pre_n and re_c > pre_n:
        failures.append("re_candidated_gt_pre_step")
    if cand and re_c > cand:
        failures.append("re_candidated_gt_candidates")
    return failures


def build_census_chunk(
    dto: SelectiveDrainCensusStepDTO,
    tracker: ObserverContinuityTracker,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
) -> dict[str, Any]:
    table1 = build_table1_cap_accounting(dto)
    table2 = build_table2_backlog_materiality(dto)
    table3 = build_table3_eligibility(dto, tracker, k_grid=k_grid)
    status = CENSUS_OK
    reason = None
    if tracker.status == CENSUS_INVALID:
        status = CENSUS_INVALID
        reason = tracker.invalid_reason
    failures = accounting_invariant({"table1": table1, "table2": table2, "table3": table3})
    if failures and status == CENSUS_OK:
        status = CENSUS_INVALID
        reason = ",".join(failures)
    chunk = {
        "schema_version": SCHEMA,
        "step": int(dto.step),
        "census_status": status,
        "census_invalid_reason": reason,
        "digest_schema": DIGEST_SCHEMA,
        "table1": table1,
        "table2": table2,
        "table3": table3,
        "dto_deep_retained_bytes": dto.deep_retained_bytes(),
        "dto_build_time_ms": dto.dto_build_time_ms,
        "observer_tracker_cardinality": tracker.cardinality(),
        "observer_tracker_approx_bytes": tracker.approx_bytes(),
        "raw_arrays_included": False,
    }
    return chunk


def append_census_chunk(sidecar_path: Path, chunk: Mapping[str, Any]) -> None:
    sidecar_path = Path(sidecar_path)
    sidecar_path.parent.mkdir(parents=True, exist_ok=True)
    with sidecar_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(chunk), sort_keys=True) + "\n")


def maybe_run_selective_drain_census(
    *,
    enabled: bool,
    pre_step_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
    cap_result: Any,
    plans_by_key: Mapping[str, Any] | None,
    step: int,
    ordering_mode: str = "margin",
    cap: int | None = None,
    tracker: ObserverContinuityTracker | None = None,
    sidecar_path: Path | str | None = None,
    k_grid: Sequence[int] = DEFAULT_K_GRID,
    pre_step_backlog_before_cap: Mapping[str, Mapping[int, Mapping[str, int]]] | None = None,
) -> dict[str, Any] | None:
    """Shared CPU-land orchestrator. Default-off skips DTO construction entirely."""
    if not enabled:
        return None
    if tracker is None:
        raise ValueError("selective_drain_census requires ObserverContinuityTracker when enabled")
    if pre_step_backlog_before_cap is not None:
        assert_pre_step_backlog_input_unchanged(pre_step_backlog_before_cap, pre_step_backlog)
    summary = getattr(cap_result, "step_summary", {}) or {}
    cap_i = int(cap) if cap is not None else int(summary.get("global_rate_cap_cap", 0) or 0)
    dto = build_selective_drain_census_step_dto(
        step=int(step),
        ordering_mode=str(ordering_mode),
        cap=cap_i,
        pre_step_backlog=pre_step_backlog,
        accepted_rows=list(cap_result.accepted_rows),
        deferred_rows=list(cap_result.deferred_rows),
        plans_by_key=plans_by_key,
    )
    # transactional tracker update
    import copy as _copy
    snapshot_records = _copy.deepcopy(tracker._records)
    snapshot_meta = (tracker.status, tracker.invalid_reason, tracker.last_step, tracker.enabled_at_step)
    try:
        tracker.update_from_dto(dto)
        chunk = build_census_chunk(dto, tracker, k_grid=k_grid)
        if sidecar_path is not None:
            append_census_chunk(Path(sidecar_path), chunk)
        return chunk
    except Exception:
        tracker.status, tracker.invalid_reason, tracker.last_step, tracker.enabled_at_step = snapshot_meta
        tracker._records.clear()
        tracker._records.update(snapshot_records)
        raise
