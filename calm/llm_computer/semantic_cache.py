"""Semantic cache — reuse intermediate residual states across similar
prompts.

HullKVCache is positional (position → (K, V)). A SEMANTIC cache is
keyed on the RESIDUAL STATE itself, letting us reuse intermediate
computations across prompts that share structure but differ in position.

Use case: the compiled adder_tiny processes `[a, b]` at positions 0-1.
If we've already cached the layer-0 output for input `[3, 2]`, the
next call with `[3, 2]` can skip layer 0 entirely.

MVP design:
  - Fingerprint a residual tensor by hashing (rounded to N decimal
    places for float fuzziness tolerance).
  - `SemanticCache.get(residual) → cached_output_or_None`
  - `SemanticCache.put(residual, output)` stores for reuse.
  - `SemanticCache.stats()` reports hit rate and utilization.

Not in MVP: cross-prompt approximate matching (nearest-neighbor lookup
on residual states). Only exact-match caching. Future: use L2 distance
or learned hashing.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional

import torch


def fingerprint(
    residual: torch.Tensor,
    precision: int = 4,
    layer_idx: int = 0,
) -> bytes:
    """Compute a deterministic hash of a residual tensor.

    Rounds to `precision` decimal places before hashing so tiny numeric
    differences don't produce different fingerprints (useful for mild
    floating-point drift between runs). Includes layer_idx so layer 0's
    cache doesn't collide with layer 1's.
    """
    rounded = torch.round(residual * (10 ** precision)).to(torch.int64)
    return (
        f"L{layer_idx}|".encode()
        + rounded.cpu().numpy().tobytes()
    )


@dataclass
class CacheStats:
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    size: int = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / max(1, total)


class SemanticCache:
    """LRU cache keyed by residual-state fingerprints.

    Use at intermediate layers: given (layer_idx, input_residual),
    look up cached output. On miss, compute + store.
    """

    def __init__(self, max_entries: int = 1024, precision: int = 4):
        self.max_entries = max_entries
        self.precision = precision
        self._cache: OrderedDict[bytes, torch.Tensor] = OrderedDict()
        self._stats = CacheStats()

    def get(
        self,
        residual: torch.Tensor,
        layer_idx: int = 0,
    ) -> Optional[torch.Tensor]:
        """Lookup by residual fingerprint. Returns the cached output
        tensor (detached clone) or None on miss."""
        key = fingerprint(residual, self.precision, layer_idx)
        if key in self._cache:
            self._stats.hits += 1
            # LRU: move to end
            self._cache.move_to_end(key)
            return self._cache[key].clone()
        self._stats.misses += 1
        return None

    def put(
        self,
        residual: torch.Tensor,
        output: torch.Tensor,
        layer_idx: int = 0,
    ) -> None:
        """Store (residual, output) association. Evicts LRU if full."""
        key = fingerprint(residual, self.precision, layer_idx)
        self._cache[key] = output.detach().clone()
        self._cache.move_to_end(key)
        while len(self._cache) > self.max_entries:
            self._cache.popitem(last=False)  # remove oldest (LRU)
            self._stats.evictions += 1
        self._stats.size = len(self._cache)

    def stats(self) -> CacheStats:
        return CacheStats(
            hits=self._stats.hits,
            misses=self._stats.misses,
            evictions=self._stats.evictions,
            size=len(self._cache),
        )

    def clear(self) -> None:
        self._cache.clear()
        # Preserve stats counters across clears (they're lifetime metrics)
        self._stats.size = 0


def cached_forward(
    cache: SemanticCache,
    layer_idx: int,
    input_residual: torch.Tensor,
    compute_fn,
) -> torch.Tensor:
    """Decorator-style: compute via `compute_fn(input_residual)` on miss,
    return cached output on hit. `compute_fn` must be pure — same input
    ALWAYS produces same output (no side effects, no randomness).
    """
    cached = cache.get(input_residual, layer_idx)
    if cached is not None:
        return cached
    output = compute_fn(input_residual)
    cache.put(input_residual, output, layer_idx)
    return output
