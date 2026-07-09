"""Event-coded accumulator checkpoint codec (V4 saved-byte carrier)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA,
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
    EVENT_CODED_ACC_METADATA_HEADER_BYTES,
    PACKED_EVENT_CODED_ACC_FORMAT,
)


@dataclass(frozen=True)
class EventCodedAccEvent:
  flat_index: int
  direction: int
  residual_mag: int
  event_type: int


@dataclass(frozen=True)
class PackedEventCodedAccState:
  events_packed: torch.Tensor
  backlog_packed: torch.Tensor
  logical_numel: int
  event_count: int
  backlog_entry_count: int
  schema: str = EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA
  format: str = PACKED_EVENT_CODED_ACC_FORMAT
  hot_exact_packed: torch.Tensor = field(
      default_factory=lambda: torch.tensor([], dtype=torch.uint8)
  )
  hot_exact_row_count: int = 0

  @property
  def metadata_bytes(self) -> int:
    return int(EVENT_CODED_ACC_METADATA_HEADER_BYTES)


def _encode_varint(value: int) -> bytes:
  if int(value) < 0:
    raise ValueError("varint value must be non-negative")
  out = bytearray()
  current = int(value)
  while True:
    byte = current & 0x7F
    current >>= 7
    if current:
      out.append(byte | 0x80)
    else:
      out.append(byte)
      break
  return bytes(out)


def _decode_varint(data: bytes, offset: int) -> tuple[int, int]:
  value = 0
  shift = 0
  index = int(offset)
  while index < len(data):
    byte = int(data[index])
    index += 1
    value |= (byte & 0x7F) << shift
    if not (byte & 0x80):
      return int(value), int(index)
    shift += 7
    if shift > 63:
      raise ValueError("varint overflow")
  raise ValueError("truncated varint")


def _pack_event_flags(*, direction: int, residual_mag: int, event_type: int) -> int:
  if int(direction) not in (0, 1):
    raise ValueError("direction must be 0 or 1")
  if not (0 <= int(residual_mag) <= 15):
    raise ValueError("residual_mag must fit in 4 bits")
  if not (0 <= int(event_type) <= 3):
    raise ValueError("event_type must fit in 2 bits")
  return (
    ((int(event_type) & 0x3) << 5)
    | ((int(residual_mag) & 0xF) << 1)
    | (int(direction) & 0x1)
  )


def _unpack_event_flags(flags: int) -> tuple[int, int, int]:
  value = int(flags) & 0xFF
  direction = value & 0x1
  residual_mag = (value >> 1) & 0xF
  event_type = (value >> 5) & 0x3
  return int(direction), int(residual_mag), int(event_type)


def encode_event_coded_acc_events(events: Sequence[EventCodedAccEvent]) -> bytes:
  payload = bytearray()
  for event in events:
    payload.extend(_encode_varint(int(event.flat_index)))
    payload.append(
      _pack_event_flags(
        direction=int(event.direction),
        residual_mag=int(event.residual_mag),
        event_type=int(event.event_type),
      )
    )
  return bytes(payload)


def decode_event_coded_acc_events(
  events_packed: torch.Tensor,
  *,
  event_count: int,
) -> tuple[EventCodedAccEvent, ...]:
  if events_packed.dtype != torch.uint8:
    raise ValueError("events_packed must be torch.uint8")
  if events_packed.ndim != 1:
    raise ValueError("events_packed must be 1-D")
  data = bytes(events_packed.detach().cpu().contiguous().tolist())
  expected = int(event_count)
  decoded: list[EventCodedAccEvent] = []
  offset = 0
  while offset < len(data):
    flat_index, offset = _decode_varint(data, offset)
    if offset >= len(data):
      raise ValueError("packed byte length must match format-specific ceiling")
    direction, residual_mag, event_type = _unpack_event_flags(data[offset])
    offset += 1
    decoded.append(
      EventCodedAccEvent(
        flat_index=int(flat_index),
        direction=int(direction),
        residual_mag=int(residual_mag),
        event_type=int(event_type),
      )
    )
  if len(decoded) != expected:
    raise ValueError("event_count mismatch for events_packed payload")
  return tuple(decoded)


def encode_hot_exact_rows(
    indices: Sequence[int],
    values: Sequence[int],
) -> bytes:
  if isinstance(indices, np.ndarray) and isinstance(values, np.ndarray):
    return encode_hot_exact_rows_from_arrays(indices, values)
  if len(indices) != len(values):
    raise ValueError("hot_exact index/value count mismatch")
  payload = bytearray()
  for flat_index, value in zip(indices, values):
    payload.extend(_encode_varint(int(flat_index)))
    signed = int(value)
    if signed < -32768 or signed > 32767:
      raise ValueError("hot_exact value must fit int16")
    payload.extend(int(signed).to_bytes(2, byteorder="little", signed=True))
  return bytes(payload)


def encode_hot_exact_rows_from_arrays(
    indices: np.ndarray,
    values: np.ndarray,
) -> bytes:
  idx = np.ascontiguousarray(indices, dtype=np.uint32).ravel()
  val = np.ascontiguousarray(values, dtype=np.int16).ravel()
  if idx.size != val.size:
    raise ValueError("hot_exact index/value count mismatch")
  payload = bytearray()
  append = payload.append
  extend = payload.extend
  for i in range(idx.size):
    v = int(idx[i])
    while v > 0x7F:
      append((v & 0x7F) | 0x80)
      v >>= 7
    append(v)
    signed = int(val[i])
    if signed < -32768 or signed > 32767:
      raise ValueError("hot_exact value must fit int16")
    extend(int(signed).to_bytes(2, byteorder="little", signed=True))
  return bytes(payload)


def decode_hot_exact_rows(
  hot_exact_packed: torch.Tensor,
  *,
  hot_exact_row_count: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
  if hot_exact_packed.dtype != torch.uint8:
    raise ValueError("hot_exact_packed must be torch.uint8")
  if hot_exact_packed.ndim != 1:
    raise ValueError("hot_exact_packed must be 1-D")
  expected = int(hot_exact_row_count)
  if expected == 0:
    if int(hot_exact_packed.numel()) != 0:
      raise ValueError("hot_exact_row_count mismatch for hot_exact_packed payload")
    return (), ()
  data = bytes(hot_exact_packed.detach().cpu().contiguous().tolist())
  indices: list[int] = []
  values: list[int] = []
  offset = 0
  while offset < len(data) and len(indices) < expected:
    flat_index, offset = _decode_varint(data, offset)
    if offset + 2 > len(data):
      raise ValueError("packed byte length must match format-specific ceiling")
    signed = int.from_bytes(data[offset : offset + 2], byteorder="little", signed=True)
    offset += 2
    indices.append(int(flat_index))
    values.append(int(signed))
  if len(indices) != expected:
    raise ValueError("hot_exact_row_count mismatch for hot_exact_packed payload")
  if offset != len(data):
    raise ValueError("packed byte length must match format-specific ceiling")
  return tuple(indices), tuple(values)


def encode_event_coded_backlog_indices(indices: Sequence[int]) -> bytes:
  payload = bytearray()
  for flat_index in indices:
    payload.extend(_encode_varint(int(flat_index)))
  return bytes(payload)


def decode_event_coded_backlog_indices(
  backlog_packed: torch.Tensor,
  *,
  backlog_entry_count: int,
) -> tuple[int, ...]:
  if backlog_packed.dtype != torch.uint8:
    raise ValueError("backlog_packed must be torch.uint8")
  if backlog_packed.ndim != 1:
    raise ValueError("backlog_packed must be 1-D")
  data = bytes(backlog_packed.detach().cpu().contiguous().tolist())
  expected = int(backlog_entry_count)
  if expected == 0:
    if len(data) != 0:
      raise ValueError("backlog_entry_count mismatch for backlog_packed payload")
    return ()
  decoded: list[int] = []
  offset = 0
  while offset < len(data):
    flat_index, offset = _decode_varint(data, offset)
    decoded.append(int(flat_index))
  if len(decoded) != expected:
    raise ValueError("backlog_entry_count mismatch for backlog_packed payload")
  return tuple(decoded)


def pack_event_coded_acc_checkpoint_reference(
  *,
  logical_numel: int,
  events: Sequence[EventCodedAccEvent],
  backlog_indices: Sequence[int] | None = None,
) -> PackedEventCodedAccState:
  logical = int(logical_numel)
  if logical <= 0:
    raise ValueError("logical_numel must be positive")
  event_list = tuple(events)
  backlog = tuple(int(item) for item in (backlog_indices or ()))
  events_bytes = encode_event_coded_acc_events(event_list)
  backlog_bytes = encode_event_coded_backlog_indices(backlog)
  return PackedEventCodedAccState(
    events_packed=torch.tensor(list(events_bytes), dtype=torch.uint8),
    backlog_packed=torch.tensor(list(backlog_bytes), dtype=torch.uint8),
    logical_numel=logical,
    event_count=len(event_list),
    backlog_entry_count=len(backlog),
    schema=EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA,
  )


def pack_event_coded_acc_checkpoint_v1(
  *,
  logical_numel: int,
  events: Sequence[EventCodedAccEvent],
  backlog_indices: Sequence[int] | None = None,
  hot_exact_indices: Sequence[int] | None = None,
  hot_exact_values: Sequence[int] | None = None,
) -> PackedEventCodedAccState:
  logical = int(logical_numel)
  if logical <= 0:
    raise ValueError("logical_numel must be positive")
  event_list = tuple(events)
  events_bytes = encode_event_coded_acc_events(event_list)
  return pack_event_coded_acc_checkpoint_v1_from_packed_events(
    logical_numel=logical,
    events_bytes=events_bytes,
    event_count=len(event_list),
    backlog_indices=backlog_indices,
    hot_exact_indices=hot_exact_indices,
    hot_exact_values=hot_exact_values,
  )


def pack_event_coded_acc_checkpoint_v1_from_packed_events(
  *,
  logical_numel: int,
  events_bytes: bytes | bytearray,
  event_count: int,
  backlog_indices: Sequence[int] | None = None,
  hot_exact_indices: Sequence[int] | None = None,
  hot_exact_values: Sequence[int] | None = None,
) -> PackedEventCodedAccState:
  """Build a V1 checkpoint from already-packed event bytes (no EventCodedAccEvent materialization)."""

  logical = int(logical_numel)
  if logical <= 0:
    raise ValueError("logical_numel must be positive")
  count = int(event_count)
  if count < 0:
    raise ValueError("event_count must be non-negative")
  packed = bytes(events_bytes)
  backlog = tuple(int(item) for item in (backlog_indices or ()))
  if isinstance(hot_exact_indices, np.ndarray) and isinstance(hot_exact_values, np.ndarray):
    hot_indices = tuple(int(item) for item in hot_exact_indices.tolist())
    hot_values = tuple(int(item) for item in hot_exact_values.tolist())
    hot_bytes = encode_hot_exact_rows_from_arrays(hot_exact_indices, hot_exact_values)
  else:
    hot_indices = tuple(int(item) for item in (hot_exact_indices or ()))
    hot_values = tuple(int(item) for item in (hot_exact_values or ()))
    hot_bytes = encode_hot_exact_rows(hot_indices, hot_values)
  if len(hot_indices) != len(hot_values):
    raise ValueError("hot_exact index/value count mismatch")
  backlog_bytes = encode_event_coded_backlog_indices(backlog)
  return PackedEventCodedAccState(
    events_packed=torch.tensor(list(packed), dtype=torch.uint8),
    backlog_packed=torch.tensor(list(backlog_bytes), dtype=torch.uint8),
    logical_numel=logical,
    event_count=count,
    backlog_entry_count=len(backlog),
    schema=EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
    hot_exact_packed=torch.tensor(list(hot_bytes), dtype=torch.uint8),
    hot_exact_row_count=len(hot_indices),
  )


def unpack_event_coded_acc_checkpoint_reference(
  state: PackedEventCodedAccState | Any,
) -> tuple[tuple[EventCodedAccEvent, ...], tuple[int, ...]]:
  schema_tag = str(getattr(state, "schema", ""))
  if schema_tag not in (
      EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA,
      EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1,
  ):
    raise ValueError(
      f"unknown event-coded acc checkpoint schema {schema_tag!r}; "
      f"expected {EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA!r} or "
      f"{EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1!r}"
    )
  if str(getattr(state, "format", "")) != PACKED_EVENT_CODED_ACC_FORMAT:
    raise ValueError(f"unknown event-coded acc format {getattr(state, 'format', '')!r}")
  events = decode_event_coded_acc_events(
    state.events_packed,
    event_count=int(state.event_count),
  )
  backlog = decode_event_coded_backlog_indices(
    state.backlog_packed,
    backlog_entry_count=int(state.backlog_entry_count),
  )
  return events, backlog


def unpack_event_coded_acc_checkpoint_v1(
  state: PackedEventCodedAccState | Any,
) -> tuple[
  tuple[EventCodedAccEvent, ...],
  tuple[int, ...],
  tuple[int, ...],
  tuple[int, ...],
]:
  schema_tag = str(getattr(state, "schema", ""))
  if schema_tag != EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1:
    raise ValueError(
      f"unknown event-coded acc checkpoint schema {schema_tag!r}; "
      f"expected {EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA_V1!r}"
    )
  events, backlog = unpack_event_coded_acc_checkpoint_reference(state)
  hot_exact_packed = getattr(state, "hot_exact_packed")
  hot_exact_row_count = int(getattr(state, "hot_exact_row_count", 0))
  hot_indices, hot_values = decode_hot_exact_rows(
    hot_exact_packed,
    hot_exact_row_count=hot_exact_row_count,
  )
  return events, backlog, hot_indices, hot_values
