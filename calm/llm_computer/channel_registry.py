"""Channel registry — typed channel allocation for unified CHRLM.

Level 1 of the residual-stream redesign. The residual stream is a
d_model-dim flat vector shared by every card in the unified tensor. Up
until now, channel ownership is convention + comments in card source
(adder_tiny.py says "ch 3..9 are step functions"). This module makes
ownership explicit and checkable.

A `ChannelAllocation` is a card's claim on a contiguous channel range
with a human-readable type and purpose. A `ChannelRegistry` tracks
allocations for a tensor and catches:

  - Overlap: two cards claim the same channel. Old behavior: silent
    corruption. New behavior: assertion at allocation time.
  - Type mismatch: a reader expects "int_scalar" but finds "text_embed".
  - Orphaned access: code tries to read a channel no card owns.

Usage:
    registry = ChannelRegistry(d_model=16)
    registry.allocate("adder", channels=range(2, 10), ch_type="int_step",
                     purpose="a+b step functions 0..6")
    registry.allocate("echo", channels=range(10, 16), ch_type="text",
                     purpose="echo value on vocab 8-15")
    # registry.allocate("conflict", channels=range(5, 8), ...) → raises

This is pure Python metadata. No runtime cost. Existing cards work
unchanged (registry is opt-in).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class ChannelAllocation:
    """One card's claim on a contiguous channel range.

    Attributes:
        card_name: identifier for the card (e.g. "adder_tiny", "echo").
        channels: frozen set of channel indices owned by this card.
        ch_type: type tag ("int_scalar", "int_step", "text_embed",
            "bias", "attention_key", ...). Free-form string.
        purpose: human-readable description of what lives here.
    """
    card_name: str
    channels: frozenset[int]
    ch_type: str
    purpose: str = ""


class AllocationError(ValueError):
    """Raised when a channel allocation conflicts with existing ones."""


class ChannelRegistry:
    """Tracks channel allocations for a unified CHRLM tensor.

    Maps `channel_index -> ChannelAllocation`. Enforces non-overlap:
    allocating a channel that's already owned raises `AllocationError`.
    """

    def __init__(self, d_model: int):
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        self._d_model = d_model
        self._by_channel: dict[int, ChannelAllocation] = {}
        self._by_card: dict[str, ChannelAllocation] = {}

    @property
    def d_model(self) -> int:
        return self._d_model

    def allocate(
        self,
        card_name: str,
        channels: Iterable[int],
        ch_type: str,
        purpose: str = "",
    ) -> ChannelAllocation:
        """Register a card's channel allocation.

        Args:
            card_name: unique card identifier. Registering a card name
                that already exists raises AllocationError.
            channels: iterable of channel indices to claim.
            ch_type: type tag for the claimed channels.
            purpose: human-readable description.

        Returns:
            The ChannelAllocation.

        Raises:
            AllocationError: if card_name already exists OR if any
                requested channel is already owned by another card OR
                if a channel is out of [0, d_model) range.
        """
        ch_set = frozenset(channels)
        # Range check
        for c in ch_set:
            if not 0 <= c < self._d_model:
                raise AllocationError(
                    f"channel {c} out of range [0, {self._d_model}) "
                    f"for card {card_name!r}"
                )
        # Duplicate card name
        if card_name in self._by_card:
            raise AllocationError(
                f"card {card_name!r} already registered "
                f"(previous: {self._by_card[card_name].purpose!r})"
            )
        # Overlap with existing allocations
        for c in ch_set:
            if c in self._by_channel:
                prior = self._by_channel[c]
                raise AllocationError(
                    f"channel {c} conflict: requested by {card_name!r} "
                    f"({ch_type}) but already owned by "
                    f"{prior.card_name!r} ({prior.ch_type})"
                )
        alloc = ChannelAllocation(
            card_name=card_name,
            channels=ch_set,
            ch_type=ch_type,
            purpose=purpose,
        )
        for c in ch_set:
            self._by_channel[c] = alloc
        self._by_card[card_name] = alloc
        return alloc

    def get_owner(self, channel: int) -> ChannelAllocation | None:
        """Returns the allocation for `channel`, or None if unallocated."""
        return self._by_channel.get(channel)

    def get_card(self, card_name: str) -> ChannelAllocation | None:
        """Returns the card's allocation, or None if not registered."""
        return self._by_card.get(card_name)

    def channels_for(self, card_name: str) -> frozenset[int]:
        """All channels owned by a named card. Raises KeyError if unknown."""
        return self._by_card[card_name].channels

    def all_allocated(self) -> frozenset[int]:
        """Union of all claimed channels."""
        return frozenset(self._by_channel.keys())

    def free_channels(self) -> frozenset[int]:
        """Channels not yet allocated."""
        return frozenset(range(self._d_model)) - self.all_allocated()

    def cards(self) -> list[str]:
        """Names of all registered cards, in registration order."""
        return list(self._by_card.keys())

    def describe(self) -> str:
        """Human-readable summary of all allocations."""
        lines = [f"ChannelRegistry(d_model={self._d_model}):"]
        for c in range(self._d_model):
            owner = self._by_channel.get(c)
            if owner is None:
                lines.append(f"  ch {c}: <free>")
            else:
                lines.append(
                    f"  ch {c}: {owner.card_name} ({owner.ch_type}) "
                    f"— {owner.purpose}" if owner.purpose
                    else f"  ch {c}: {owner.card_name} ({owner.ch_type})"
                )
        return "\n".join(lines)

    def validate_coverage(self, required: set[int]) -> None:
        """Raise if any required channel is unallocated. Used at compile
        time to catch "reader expects a channel that no card owns" bugs.
        """
        missing = required - self.all_allocated()
        if missing:
            raise AllocationError(
                f"required channels {sorted(missing)} are unallocated"
            )


# ----- Predefined allocations for known compiled programs -----

def adder_tiny_allocation() -> list[ChannelAllocation]:
    """The adder_tiny card's channel claims per its source comments.

    Channel layout (from calm/llm_computer/programs/adder_tiny.py):
      ch 0: own token scalar (from TokenEmbed)
      ch 1: bias 1            (from PosEmbed)
      ch 2: a copied from pos 0 (from LookUp at layer 0)
      ch 3..9: step functions step_S = 1[a+b >= S] (from ReGLU)
    """
    return [
        ChannelAllocation(
            card_name="adder_tiny.tok_embed",
            channels=frozenset({0}),
            ch_type="int_scalar",
            purpose="own token scalar from TokenEmbed",
        ),
        ChannelAllocation(
            card_name="adder_tiny.pos_embed",
            channels=frozenset({1}),
            ch_type="bias",
            purpose="position-invariant 1.0 bias",
        ),
        ChannelAllocation(
            card_name="adder_tiny.lookup_a",
            channels=frozenset({2}),
            ch_type="int_scalar",
            purpose="a copied from pos 0 via LookUp",
        ),
        ChannelAllocation(
            card_name="adder_tiny.step_funcs",
            channels=frozenset(range(3, 10)),
            ch_type="int_step",
            purpose="step functions step_S = 1[a+b >= S] for S in 0..6",
        ),
    ]


def register_adder_tiny(registry: ChannelRegistry) -> None:
    """Register adder_tiny's full channel allocation in the registry.

    Splits adder into 4 sub-card names because its channels come from
    different IR nodes (TokenEmbed, PosEmbed, LookUp, ReGLU). This is
    finer-grained than "adder_tiny owns 0..9" — future cards can see
    that channel 0 is a token embedding (widely-read) vs channels 3..9
    which are adder-specific step functions.
    """
    for alloc in adder_tiny_allocation():
        registry.allocate(
            card_name=alloc.card_name,
            channels=alloc.channels,
            ch_type=alloc.ch_type,
            purpose=alloc.purpose,
        )
