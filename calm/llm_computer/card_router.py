"""Dispatch PT output to compute cards via a routing registry.

Session 32 chained-demo (commit f5455f6) hardcoded a single adapter:
PT writes log-probs to residual channels [2400:2480], adder_tiny reads
those channels, argmax-decodes to integer (a, b), runs adder. This
only works for one operator on one card.

The router generalizes that wiring. Routes are declarative:

    router = CardRouter(id_to_char=_ID_TO_CHAR)
    router.register(Route(
        source_ch=(2400, 2480),
        operator="+",
        target_card_slot=adder_slot,
        translator=lambda ops: torch.tensor([[ops[0], ops[1]]]),
    ))
    # Now adder_slot.card_input_fn = router.make_card_input_fn(0)

At forward time, each target card's input_fn reads the PT's output
channels from Gemma's residual, decodes to a character string via the
PT vocab, finds its registered operator, parses operands, and applies
the per-route translator to produce the target card's input tensor.

The router is pure Python; it composes with the existing CardSlot +
preservation-masking machinery in gemma_substrate.py. No changes
required to GemmaSubstrate or CardSlot.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import torch


@dataclass
class Route:
    """One PT output channel range → one compute card slot.

    `source_ch=(lo, hi)` identifies the residual channels where the source
    CardSlot wrote its log-probs (matches PT's d_card = vocab_size).

    `operator` is a literal substring matched in the argmax-decoded PT
    output. The first matching route wins.

    `translator` converts a list of parsed integer operands into the
    target card's input tensor (card-specific shape; e.g. adder_tiny
    wants `(1, 2)` two-token input).

    `fallback_operands` is used when the operator match fires but
    operand parsing fails (non-numeric operand, malformed expression).
    Keeps the chain alive instead of crashing mid-forward.
    """

    source_ch: tuple[int, int]
    operator: str
    target_card_slot: Any  # CardSlot — avoids circular import
    translator: Callable[[list[int]], torch.Tensor]
    fallback_operands: list[int] = field(default_factory=lambda: [0, 0])


class CardRouter:
    """Dispatcher from PT output to compute cards.

    The router is stateless per-forward: it reads the residual each
    time, decodes, parses, dispatches. Holds only the route registry
    and a PT vocab map.
    """

    def __init__(self, id_to_char: dict[int, str]):
        self.id_to_char = id_to_char
        self.routes: list[Route] = []

    def register(self, route: Route) -> int:
        """Append a route and return its index (stable for
        make_card_input_fn)."""
        self.routes.append(route)
        return len(self.routes) - 1

    def decode_pt_output(self, h: torch.Tensor,
                        ch_lo: int, ch_hi: int) -> str:
        """Read PT output channels from residual, argmax per position,
        return a clean character string (specials stripped)."""
        pt_log_probs = h[..., ch_lo:ch_hi]  # (B, S, V)
        tokens = pt_log_probs.argmax(dim=-1)  # (B, S)
        # Walk the first batch; specials like <bos> <sep> filtered out.
        chars = []
        for t in tokens[0].tolist():
            c = self.id_to_char.get(int(t), "")
            if c and not (c.startswith("<") and c.endswith(">")):
                chars.append(c)
        return "".join(chars)

    @staticmethod
    def _parse_operands(text: str, operator: str) -> Optional[list[int]]:
        """Find `operator` in `text`, return integer operands before/after.
        Ignores trailing '=' and surrounding whitespace. Returns None if
        the split doesn't yield two integers."""
        idx = text.find(operator)
        if idx < 0:
            return None
        a_str = text[:idx]
        b_str = text[idx + len(operator):]
        # Strip '=', whitespace, and any non-digit/non-sign leading junk.
        a_match = re.search(r"-?\d+", a_str)
        b_match = re.search(r"-?\d+", b_str)
        if a_match is None or b_match is None:
            return None
        try:
            return [int(a_match.group()), int(b_match.group())]
        except ValueError:
            return None

    def route_forward(self, h: torch.Tensor, route_idx: int) -> torch.Tensor:
        """Execute one route: read PT output, parse, translate, return
        the target card's input tensor on h's device."""
        route = self.routes[route_idx]
        text = self.decode_pt_output(h, *route.source_ch)
        operands = self._parse_operands(text, route.operator)
        if operands is None:
            operands = list(route.fallback_operands)
        tensor = route.translator(operands)
        if isinstance(tensor, torch.Tensor) and tensor.device != h.device:
            tensor = tensor.to(h.device)
        return tensor

    def make_card_input_fn(self, route_idx: int) -> Callable:
        """Returns a function suitable as CardSlot.card_input_fn.
        Closure over `self` and `route_idx` so the route can be looked up
        at forward time (supports post-install route edits)."""
        def fn(h: torch.Tensor) -> torch.Tensor:
            return self.route_forward(h, route_idx)
        return fn
