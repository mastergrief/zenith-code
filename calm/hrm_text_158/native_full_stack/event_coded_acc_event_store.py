"""Codec-byte live event store for EventCodedAccLiveState (lazy-decode facade).

Frozen plan: ai-room 1783547263583 + 1783547408000 + 1783547629738 (+1 1783547797755).
Append-one-encode is concat-associative with encode_event_coded_acc_events (codec L99-110).
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import Any

from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    decode_event_coded_acc_events,
    encode_event_coded_acc_events,
)
import torch


class EventCodedAccEventStore:
    """Packed live event buffer with copy-on-write bytes and lazy decode."""

    __slots__ = (
        "_buf",
        "_count",
        "_shared",
        "_materialize_count",
    )

    def __init__(
        self,
        *,
        buf: bytearray | bytes | None = None,
        count: int = 0,
        shared: bool = False,
    ) -> None:
        if buf is None:
            self._buf = bytearray()
            self._count = 0
            self._shared = False
        elif isinstance(buf, bytearray):
            self._buf = buf
            self._count = int(count)
            self._shared = bool(shared)
        else:
            self._buf = bytearray(buf)
            self._count = int(count)
            self._shared = False
        self._materialize_count = 0

    @classmethod
    def empty(cls) -> EventCodedAccEventStore:
        return cls()

    @classmethod
    def from_events(cls, events: Sequence[EventCodedAccEvent] | Iterable[EventCodedAccEvent]) -> EventCodedAccEventStore:
        event_list = tuple(events)
        if not event_list:
            return cls.empty()
        packed = encode_event_coded_acc_events(event_list)
        return cls(buf=bytearray(packed), count=len(event_list), shared=False)

    @classmethod
    def from_packed_bytes(cls, events_bytes: bytes | bytearray, *, event_count: int) -> EventCodedAccEventStore:
        return cls(buf=bytearray(events_bytes), count=int(event_count), shared=False)

    def __len__(self) -> int:
        return int(self._count)

    def __bool__(self) -> bool:
        return int(self._count) > 0

    @property
    def materialize_count(self) -> int:
        return int(self._materialize_count)

    def reset_materialize_count(self) -> None:
        self._materialize_count = 0

    def _ensure_writable(self) -> None:
        if self._shared:
            self._buf = bytearray(self._buf)
            self._shared = False

    def append(self, event: EventCodedAccEvent) -> int:
        """Append one event via per-event encode; returns encoded byte length."""

        encoded = encode_event_coded_acc_events((event,))
        self._ensure_writable()
        self._buf.extend(encoded)
        self._count += 1
        return len(encoded)

    def encode_bytes(self) -> bytes:
        """Return packed codec bytes without materializing EventCodedAccEvent objects."""

        return bytes(self._buf)

    def cow_copy(self) -> EventCodedAccEventStore:
        """Share buffer until next write (copy-on-write), like _PackedHotTable.fork()."""

        self._shared = True
        return EventCodedAccEventStore(
            buf=self._buf,
            count=int(self._count),
            shared=True,
        )

    def __iter__(self) -> Iterator[EventCodedAccEvent]:
        self._materialize_count += 1
        if self._count == 0:
            return iter(())
        packed = torch.tensor(list(self._buf), dtype=torch.uint8)
        decoded = decode_event_coded_acc_events(packed, event_count=int(self._count))
        return iter(decoded)

    def as_tuple(self) -> tuple[EventCodedAccEvent, ...]:
        self._materialize_count += 1
        if self._count == 0:
            return ()
        packed = torch.tensor(list(self._buf), dtype=torch.uint8)
        return decode_event_coded_acc_events(packed, event_count=int(self._count))

    def view(self, *, on_append: Any) -> EventCodedAccEventsView:
        return EventCodedAccEventsView(self, on_append=on_append)


class EventCodedAccEventsView:
    """List-like view over EventCodedAccEventStore with coherent .append."""

    __slots__ = ("_store", "_on_append")

    def __init__(self, store: EventCodedAccEventStore, *, on_append: Any) -> None:
        self._store = store
        self._on_append = on_append

    def __len__(self) -> int:
        return len(self._store)

    def __bool__(self) -> bool:
        return bool(self._store)

    def __iter__(self) -> Iterator[EventCodedAccEvent]:
        return iter(self._store)

    def __getitem__(self, index: int) -> EventCodedAccEvent:
        # Materialize only the requested path via full decode (rare); keep simple.
        return self._store.as_tuple()[int(index)]

    def append(self, event: EventCodedAccEvent) -> None:
        self._on_append(event)

    def as_tuple(self) -> tuple[EventCodedAccEvent, ...]:
        return self._store.as_tuple()
