"""Pure B=64 block-occupancy builder for R7 census instrumentation.

Stdlib-only. No observer/learner/probe/plan-type/GPU/IO imports.
Accumulator values arrive as int32 little-endian bytes (not Python int lists).
"""

from __future__ import annotations

import base64
import hashlib
import struct
from array import array
from dataclasses import dataclass
from typing import Mapping, Sequence

SCHEMA_VERSION = "hrm_text_158_r7_block_occupancy_B64/v1"
DEFAULT_B = 64
DEFAULT_K = 12
BINARY_ENCODING = "base64"
# Soft compact ceiling per step (all states combined, encoded JSON payload body).
COMPACT_SIZE_CEILING_BYTES = 64 * 1024


class BlockOccupancyError(ValueError):
    """Fail-closed occupancy construction / validation error."""


class BlockOccupancyMissingObservablesError(BlockOccupancyError):
    """Required plan/geometry observables absent or inconsistent."""


def _require_canonical_int(value: object, *, name: str) -> int:
    if type(value) is not int:
        raise BlockOccupancyError(f"{name} must be canonical int, got {type(value)!r}")
    return value


def _block_len(numel: int, block_index: int, B: int) -> int:
    start = block_index * B
    if start >= numel:
        raise BlockOccupancyError(f"block_index {block_index} out of range for numel={numel}")
    return min(B, numel - start)


def _acc_array_from_le_bytes(acc_i32_le: bytes, *, logical_numel: int) -> array:
    if not isinstance(acc_i32_le, (bytes, bytearray, memoryview)):
        raise BlockOccupancyMissingObservablesError(
            f"acc_i32_le must be bytes-like, got {type(acc_i32_le)!r}"
        )
    raw = bytes(acc_i32_le)
    expected = int(logical_numel) * 4
    if len(raw) != expected:
        raise BlockOccupancyMissingObservablesError(
            f"acc_i32_le length {len(raw)} != logical_numel*4 ({expected})"
        )
    out = array("i")
    out.frombytes(raw)
    if len(out) != int(logical_numel):
        raise BlockOccupancyMissingObservablesError(
            f"acc array length {len(out)} != logical_numel {logical_numel}"
        )
    return out


def _u8_bytes(counts: Sequence[int]) -> bytes:
    try:
        return bytes(int(c) for c in counts)
    except ValueError as exc:
        raise BlockOccupancyError(f"u8 count out of range: {exc}") from exc


def _bitmap_from_noneligible_nonzero(noneligible_nonzero: Sequence[int]) -> bytes:
    n = len(noneligible_nonzero)
    out = bytearray((n + 7) // 8)
    for b, nzn in enumerate(noneligible_nonzero):
        if int(nzn) == 0:
            out[b >> 3] |= 1 << (b & 7)
    return bytes(out)


def _eoe_block_ids_from_bitmap(bitmap: bytes, n_blocks: int) -> tuple[int, ...]:
    ids: list[int] = []
    for b in range(n_blocks):
        if bitmap[b >> 3] & (1 << (b & 7)):
            ids.append(b)
    return tuple(ids)


def _eoe_set_sha256(block_ids: Sequence[int]) -> str:
    payload = b"".join(struct.pack("<I", int(b)) for b in block_ids)
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True, slots=True)
class PerStateOccupancySource:
    """Normalized per-state source. acc_i32_le is int32 LE bytes (immutable)."""

    state_key: str
    logical_numel: int
    acc_i32_le: bytes
    q_numel: int


@dataclass(frozen=True, slots=True)
class BlockOccupancyInput:
    """Pure-module input. Eligible ids are (state_key, flat) for K only."""

    per_state: tuple[PerStateOccupancySource, ...]
    eligible_ids_k: tuple[tuple[str, int], ...]
    k: int = DEFAULT_K
    B: int = DEFAULT_B


@dataclass(frozen=True, slots=True)
class PerStateBlockOccupancy:
    state_key: str
    logical_numel: int
    n_blocks: int
    tail_len: int
    per_block_eligible_u8: bytes
    per_block_noneligible_nonzero_u8: bytes
    per_block_empty_u8: bytes
    fully_eoe_block_bitmap: bytes
    fully_eoe_count: int
    fully_eoe_set_sha256: str
    already_fully_empty_count: int
    poisoned_count: int
    set_hash_ok: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "state_key": self.state_key,
            "logical_numel": self.logical_numel,
            "n_blocks": self.n_blocks,
            "tail_len": self.tail_len,
            "per_block_eligible_u8_b64": base64.b64encode(self.per_block_eligible_u8).decode("ascii"),
            "per_block_noneligible_nonzero_u8_b64": base64.b64encode(
                self.per_block_noneligible_nonzero_u8
            ).decode("ascii"),
            "per_block_empty_u8_b64": base64.b64encode(self.per_block_empty_u8).decode("ascii"),
            "fully_eoe_block_bitmap_b64": base64.b64encode(self.fully_eoe_block_bitmap).decode(
                "ascii"
            ),
            "fully_eoe_count": self.fully_eoe_count,
            "fully_eoe_set_sha256": self.fully_eoe_set_sha256,
            "already_fully_empty_count": self.already_fully_empty_count,
            "poisoned_count": self.poisoned_count,
            "set_hash_ok": self.set_hash_ok,
        }


@dataclass(frozen=True, slots=True)
class BlockOccupancyResult:
    schema_version: str
    B: int
    k: int
    event_coded_live: bool
    binary_encoding: str
    per_state: tuple[PerStateBlockOccupancy, ...]
    compact_payload_bytes: int

    def to_chunk_dict(self) -> dict[str, object]:
        body = {
            "schema_version": self.schema_version,
            "B": self.B,
            "k": self.k,
            "event_coded_live": self.event_coded_live,
            "binary_encoding": self.binary_encoding,
            "per_state": [ps.to_dict() for ps in self.per_state],
            "compact_payload_bytes": self.compact_payload_bytes,
        }
        return body


def _eligible_sets_by_key(
    eligible_ids_k: Sequence[tuple[str, int]],
    *,
    numel_by_key: Mapping[str, int],
) -> dict[str, set[int]]:
    out: dict[str, set[int]] = {sk: set() for sk in numel_by_key}
    seen: set[tuple[str, int]] = set()
    for item in eligible_ids_k:
        if not isinstance(item, tuple) or len(item) != 2:
            raise BlockOccupancyError(f"eligible id must be (state_key, flat), got {item!r}")
        sk, fi = item
        sk_s = str(sk)
        fi_i = _require_canonical_int(fi, name="eligible.flat")
        if sk_s not in numel_by_key:
            raise BlockOccupancyMissingObservablesError(
                f"eligible state_key {sk_s!r} missing from per_state sources"
            )
        numel = numel_by_key[sk_s]
        if fi_i < 0 or fi_i >= numel:
            raise BlockOccupancyError(
                f"eligible flat {fi_i} out of range for {sk_s} numel={numel}"
            )
        key = (sk_s, fi_i)
        if key in seen:
            raise BlockOccupancyError(f"duplicate eligible id {key}")
        seen.add(key)
        out[sk_s].add(fi_i)
    return out


def _build_one_state(
    src: PerStateOccupancySource,
    *,
    eligible: set[int],
    B: int,
) -> PerStateBlockOccupancy:
    sk = str(src.state_key)
    numel = _require_canonical_int(src.logical_numel, name="logical_numel")
    q_numel = _require_canonical_int(src.q_numel, name="q_numel")
    if numel <= 0:
        raise BlockOccupancyMissingObservablesError(f"{sk}: logical_numel must be > 0")
    if numel != q_numel:
        raise BlockOccupancyMissingObservablesError(
            f"{sk}: new_acc numel {numel} != q_numel {q_numel}"
        )
    acc = _acc_array_from_le_bytes(src.acc_i32_le, logical_numel=numel)
    n_blocks = (numel + B - 1) // B
    tail_len = numel % B
    elig_counts = [0] * n_blocks
    nzn_counts = [0] * n_blocks
    empty_counts = [0] * n_blocks
    already_empty = 0
    poisoned = 0
    for fi in range(numel):
        b = fi // B
        if fi in eligible:
            elig_counts[b] += 1
        elif int(acc[fi]) != 0:
            nzn_counts[b] += 1
        else:
            empty_counts[b] += 1
    for b in range(n_blocks):
        blen = _block_len(numel, b, B)
        total = elig_counts[b] + nzn_counts[b] + empty_counts[b]
        if total != blen:
            raise BlockOccupancyError(
                f"{sk} block {b}: sum {total} != block_len {blen}"
            )
        if elig_counts[b] > blen or nzn_counts[b] > blen or empty_counts[b] > blen:
            raise BlockOccupancyError(f"{sk} block {b}: u8 count exceeds block_len")
        if nzn_counts[b] > 0:
            poisoned += 1
        elif elig_counts[b] == 0 and empty_counts[b] == blen:
            already_empty += 1
    elig_u8 = _u8_bytes(elig_counts)
    nzn_u8 = _u8_bytes(nzn_counts)
    empty_u8 = _u8_bytes(empty_counts)
    bitmap = _bitmap_from_noneligible_nonzero(nzn_counts)
    eoe_ids = _eoe_block_ids_from_bitmap(bitmap, n_blocks)
    eoe_sha = _eoe_set_sha256(eoe_ids)
    recomputed_ids = tuple(b for b, nzn in enumerate(nzn_counts) if nzn == 0)
    set_hash_ok = eoe_ids == recomputed_ids and eoe_sha == _eoe_set_sha256(recomputed_ids)
    if not set_hash_ok:
        raise BlockOccupancyError(f"{sk}: fully_eoe set/hash inconsistency")
    return PerStateBlockOccupancy(
        state_key=sk,
        logical_numel=numel,
        n_blocks=n_blocks,
        tail_len=tail_len,
        per_block_eligible_u8=elig_u8,
        per_block_noneligible_nonzero_u8=nzn_u8,
        per_block_empty_u8=empty_u8,
        fully_eoe_block_bitmap=bitmap,
        fully_eoe_count=len(eoe_ids),
        fully_eoe_set_sha256=eoe_sha,
        already_fully_empty_count=already_empty,
        poisoned_count=poisoned,
        set_hash_ok=True,
    )


def build_block_occupancy_B64(
    occupancy_input: BlockOccupancyInput,
    *,
    k: int | None = None,
    B: int | None = None,
) -> BlockOccupancyResult:
    if not isinstance(occupancy_input, BlockOccupancyInput):
        raise BlockOccupancyError(
            f"occupancy_input must be BlockOccupancyInput, got {type(occupancy_input)!r}"
        )
    k_i = DEFAULT_K if k is None else _require_canonical_int(k, name="k")
    B_i = DEFAULT_B if B is None else _require_canonical_int(B, name="B")
    if k_i != _require_canonical_int(occupancy_input.k, name="input.k"):
        raise BlockOccupancyError("k mismatch between args and input")
    if B_i != _require_canonical_int(occupancy_input.B, name="input.B"):
        raise BlockOccupancyError("B mismatch between args and input")
    if B_i != 64:
        raise BlockOccupancyError(f"only B=64 supported, got {B_i}")
    if not occupancy_input.per_state:
        raise BlockOccupancyMissingObservablesError("per_state empty")
    numel_by_key: dict[str, int] = {}
    for src in occupancy_input.per_state:
        sk = str(src.state_key)
        if sk in numel_by_key:
            raise BlockOccupancyError(f"duplicate state_key in per_state: {sk}")
        numel_by_key[sk] = _require_canonical_int(src.logical_numel, name="logical_numel")
    eligible_by_key = _eligible_sets_by_key(
        occupancy_input.eligible_ids_k, numel_by_key=numel_by_key
    )
    per_state: list[PerStateBlockOccupancy] = []
    payload = 0
    for src in occupancy_input.per_state:
        built = _build_one_state(
            src, eligible=eligible_by_key[str(src.state_key)], B=B_i
        )
        per_state.append(built)
        payload += (
            len(built.per_block_eligible_u8)
            + len(built.per_block_noneligible_nonzero_u8)
            + len(built.per_block_empty_u8)
            + len(built.fully_eoe_block_bitmap)
        )
    if payload > COMPACT_SIZE_CEILING_BYTES:
        raise BlockOccupancyError(
            f"compact payload {payload} exceeds ceiling {COMPACT_SIZE_CEILING_BYTES}"
        )
    return BlockOccupancyResult(
        schema_version=SCHEMA_VERSION,
        B=B_i,
        k=k_i,
        event_coded_live=False,
        binary_encoding=BINARY_ENCODING,
        per_state=tuple(per_state),
        compact_payload_bytes=int(payload),
    )


def rebuild_bitmap_from_chunk_state(state_dict: Mapping[str, object]) -> bytes:
    """Test/helper: rebuild eoe bitmap from noneligible_nonzero b64 field."""
    raw = base64.b64decode(str(state_dict["per_block_noneligible_nonzero_u8_b64"]))
    return _bitmap_from_noneligible_nonzero(list(raw))
