"""Event-coded accumulator checkpoint codec (V4 saved-byte carrier)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch

from calm.hrm_text_158.native_full_stack.persistent_state_budget import (
    EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA,
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


def unpack_event_coded_acc_checkpoint_reference(
  state: PackedEventCodedAccState | Any,
) -> tuple[tuple[EventCodedAccEvent, ...], tuple[int, ...]]:
  schema_tag = str(getattr(state, "schema", ""))
  if schema_tag != EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA:
    raise ValueError(
      f"unknown event-coded acc checkpoint schema {schema_tag!r}; "
      f"expected {EVENT_CODED_ACC_CHECKPOINT_PAYLOAD_SCHEMA!r}"
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
