"""Bounded-delta runner post-step hook contract (generic; NOT Fork-B classifier).

Owns the event dataclass + backlog clone ownership helpers. Probe wires this
thinly; science drivers consume the event. NEVER imports probe/CLI/GPU/launch.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class BoundedDeltaPostStepEvent:
    """Synchronous post-step event emitted AFTER carry_backlog update.

    ``states`` is the LIVE mapping owned by the runner — callers that need
    immutability MUST deep-clone tensors/state themselves (see Fork-B
    ``clone_f_in_memory``). ``carry_backlog`` is whatever the runner currently
    holds (may be None when r7 carry is off); ownership helpers below clone
    defensively when seeding or snapshotting backlog.
    """

    step: int
    states: Mapping[str, Any]
    carry_backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None
    step_batch_metadata: Mapping[str, Any] | None = None


def clone_deferred_backlog(
    backlog: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> dict[str, dict[int, dict[str, int]]]:
    """Deep-clone deferred backlog; facade-local (no learner alias export).

    Byte/field-equivalent to ``bounded_delta_learner._clone_backlog_for_front_c``
    (verified by characterization tests against a nested fixture). Do NOT edit
    the learner or re-export an alias under this gate.
    """

    return {
        str(state_key): {
            int(flat_index): {
                str(field): int(value)
                for field, value in dict(entry).items()
            }
            for flat_index, entry in dict(by_index).items()
        }
        for state_key, by_index in dict(backlog or {}).items()
    }


def seed_initial_deferred_backlog(
    initial: Mapping[str, Mapping[int, Mapping[str, int]]] | None,
) -> dict[str, dict[int, dict[str, int]]] | None:
    """Defensive clone for runner S2 seed. None stays None."""

    if initial is None:
        return None
    return clone_deferred_backlog(initial)


def invoke_post_step_hook(
    hook: Callable[[BoundedDeltaPostStepEvent], None] | None,
    event: BoundedDeltaPostStepEvent,
) -> None:
    """Synchronous invoke. NO try/except swallow — FAIL CLOSED on raise."""

    if hook is None:
        return
    hook(event)


__all__ = [
    "BoundedDeltaPostStepEvent",
    "clone_deferred_backlog",
    "invoke_post_step_hook",
    "seed_initial_deferred_backlog",
]
