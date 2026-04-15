"""Keyed residual memory — Level 3 bus redesign.

The compile-to-weights parabolic-key mechanism from `read_by_key` maps
integer keys `k` to 2D points `(2k, -k²)` such that dot-product
attention exactly retrieves the value at any queried key. That's
runtime-trivial math, but `read_by_key` baked it into compiled weights.

Level 3 extracts the mechanism into a RUNTIME primitive. Any tensor
can allocate a "keyed memory region" — a pair of channel slices (keys,
values) — and write/read named slots at any sequence position. This
gives cards a symbolic addressing interface on top of the positional
residual stream.

Analogy: single-stream residual is assembly (variables are raw memory
addresses). Typed channels (Level 1) adds variable types. Multi-stream
(Level 2) adds stack frames. Keyed memory (Level 3) adds named
variables with scoping. Each level buys more compositional power.

MVP primitives:
  - `KeyRegistry(name -> key_id)` — maps string keys to integer IDs.
  - `write_keyed_slot(x, pos, key_id, value, key_ch, value_ch)` —
    writes a (key, value) pair into residual at the given position.
  - `read_by_key_attention(x, query_key_id, key_ch, value_ch)` —
    returns (B, S, V) attention-lookup where each query position
    retrieves the value written under `query_key_id`.
  - `KeyedMemoryConfig` — declarative config for the key/value channel
    layout.

Differentiability: the read operation is a standard softmax attention
over a contrived (Q, K, V) layout, fully differentiable. Values can
be trainable; keys are typically fixed (determined by key_id).

Compositional usage pattern:
  Card A writes {"sum": a+b} into keyed memory.
  Card B reads "sum" via read_by_key_attention.
  Neither card knows physical channel indices — only the key name.

This is only a START. True keyed memory would support variable-length
key names, garbage collection, nested scopes. MVP ships the core
math; future rounds layer on the compositional conveniences.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


# ----- Key ID math (parabolic, d_head=2) -----

def parabolic_key(key_id: int, key_channels: slice) -> tuple[float, float]:
    """Return the (2D) parabolic key vector for a given integer key_id.

    Writes go into a 2-channel key region: the first channel gets
    `2*key_id`, the second gets `-key_id ** 2`. Dot-product attention
    `q · k_i` peaks exactly at i == query_id because:

        q · k_i = 2q·i - i²  →  derivative w.r.t. i is 2(q - i) = 0 at i=q

    This is the same math `read_by_key.py` uses at compile time, now
    exposed as a runtime helper.
    """
    start, stop, _ = key_channels.indices(10**9)
    assert stop - start == 2, (
        f"key_channels must be a 2-channel slice; got width {stop - start}"
    )
    return (2.0 * key_id, -float(key_id) ** 2)


# ----- Configuration -----

@dataclass(frozen=True)
class KeyedMemoryConfig:
    """Specifies the residual-channel layout for a keyed memory region.

    Attributes:
        key_channels: 2-channel slice holding parabolic key (2k, -k²).
        value_channels: N-channel slice holding the value payload.
        max_key_id: max integer key_id supported (for range-checking).
    """
    key_channels: slice
    value_channels: slice
    max_key_id: int = 256

    @property
    def value_dim(self) -> int:
        start, stop, _ = self.value_channels.indices(10**9)
        return stop - start

    def validate(self, d_model: int) -> None:
        k_start, k_stop, _ = self.key_channels.indices(d_model)
        v_start, v_stop, _ = self.value_channels.indices(d_model)
        assert 0 <= k_start < k_stop <= d_model, (
            f"key_channels out of range for d_model={d_model}"
        )
        assert 0 <= v_start < v_stop <= d_model, (
            f"value_channels out of range for d_model={d_model}"
        )
        # No overlap
        assert k_stop <= v_start or v_stop <= k_start, (
            f"key_channels and value_channels overlap"
        )
        assert k_stop - k_start == 2, (
            "key region must be exactly 2 channels for parabolic keys"
        )


# ----- Symbolic name → integer key_id -----

class KeyRegistry:
    """Maps string key names to integer IDs with optional nested scopes.

    Scopes are a stack of namespaces. `push_scope("math")` adds a new
    frame; `register("sum")` inside that frame binds "sum" to a fresh
    key_id in the current scope. `resolve("sum")` walks the stack from
    top (innermost) to bottom (outermost) and returns the first match —
    classic lexical scoping semantics.

    GC: `unregister(name)` frees a key's ID for reuse. Freed IDs are
    recycled FIFO on the next `register()` call.

    Backward compatibility: with no scopes pushed, the registry behaves
    as a single flat namespace (the "global scope").
    """

    def __init__(self, max_key_id: int = 256):
        self._max_key_id = max_key_id
        # One dict per scope frame; index 0 is the global scope.
        self._scopes: list[dict[str, int]] = [{}]
        self._by_id: dict[int, tuple[int, str]] = {}  # id -> (scope_idx, name)
        # Start at 1 so key_id=0 (parabolic key = (0,0)) means "unwritten"
        # unambiguously. Without this, the first registered name collides
        # with every zero-init position during read-by-attention.
        self._next_id = 1
        self._free_ids: list[int] = []  # recycled ids (FIFO)

    # --- Scope management ---

    def push_scope(self, name: str = "") -> int:
        """Push a new scope frame. Returns its depth (0 = global)."""
        self._scopes.append({})
        return len(self._scopes) - 1

    def pop_scope(self) -> None:
        """Pop the innermost scope, unregistering all its names and
        freeing their IDs. Popping the global scope (depth 0) is a no-op."""
        if len(self._scopes) <= 1:
            return  # never pop global
        frame = self._scopes.pop()
        for name, key_id in frame.items():
            self._by_id.pop(key_id, None)
            self._free_ids.append(key_id)

    def scope_depth(self) -> int:
        """Current scope depth; 0 is global."""
        return len(self._scopes) - 1

    # --- Registration ---

    def register(self, name: str) -> int:
        """Register `name` in the current (innermost) scope. Idempotent
        if `name` already exists in the SAME scope; shadowing across
        scopes creates a new binding per scope."""
        current = self._scopes[-1]
        if name in current:
            return current[name]
        # Allocate an ID: recycle if any free, else take next_id.
        if self._free_ids:
            key_id = self._free_ids.pop(0)
        else:
            if self._next_id >= self._max_key_id:
                raise ValueError(
                    f"key registry full (max_key_id={self._max_key_id})"
                )
            key_id = self._next_id
            self._next_id += 1
        current[name] = key_id
        self._by_id[key_id] = (len(self._scopes) - 1, name)
        return key_id

    def unregister(self, name: str) -> int | None:
        """Remove `name` from the innermost scope that contains it. Its
        ID is recycled. Returns the freed ID, or None if name not found.
        """
        for depth in range(len(self._scopes) - 1, -1, -1):
            frame = self._scopes[depth]
            if name in frame:
                key_id = frame.pop(name)
                self._by_id.pop(key_id, None)
                self._free_ids.append(key_id)
                return key_id
        return None

    # --- Lookup ---

    def resolve(self, name: str) -> int:
        """Lexical lookup: walk scope stack from innermost to outermost.
        Returns the first matching ID. Raises KeyError if not found."""
        for depth in range(len(self._scopes) - 1, -1, -1):
            if name in self._scopes[depth]:
                return self._scopes[depth][name]
        raise KeyError(name)

    def id_of(self, name: str) -> int:
        """Alias for resolve — kept for backward compatibility."""
        return self.resolve(name)

    def name_of(self, key_id: int) -> str:
        """Returns the name bound to `key_id` in its defining scope."""
        return self._by_id[key_id][1]

    def __contains__(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except KeyError:
            return False

    def names(self) -> list[str]:
        """All visible names (innermost wins on shadowing)."""
        seen: dict[str, None] = {}
        for frame in reversed(self._scopes):
            for name in frame:
                if name not in seen:
                    seen[name] = None
        return list(seen.keys())

    def used_ids(self) -> list[int]:
        """All currently-allocated IDs, sorted."""
        return sorted(self._by_id.keys())

    def free_id_count(self) -> int:
        """Number of recycled IDs available for reuse."""
        return len(self._free_ids)


# ----- Write and read primitives -----

def write_keyed_slot(
    residual: torch.Tensor,
    position: int,
    key_id: int,
    value: torch.Tensor,
    cfg: KeyedMemoryConfig,
) -> torch.Tensor:
    """Write (key_id, value) into the residual stream at `position`.

    Returns a NEW tensor (same shape as residual) with the specified
    position's key and value slots set. Channels outside the keyed
    memory region are preserved unchanged.

    Args:
        residual: (B, S, d_model) residual stream.
        position: sequence position to write into, in [0, S).
        key_id: integer key ID (must be < cfg.max_key_id).
        value: shape (B, cfg.value_dim) or (cfg.value_dim,).
        cfg: keyed memory config.

    Returns:
        Updated residual (B, S, d_model).
    """
    assert 0 <= key_id < cfg.max_key_id, (
        f"key_id {key_id} out of range [0, {cfg.max_key_id})"
    )
    B, S, D = residual.shape
    cfg.validate(D)
    assert 0 <= position < S, f"position {position} out of range"
    # Broadcast value to (B, value_dim). Accepts (value_dim,),
    # (1, value_dim), or (B, value_dim).
    if value.dim() == 1:
        value = value.unsqueeze(0)
    if value.size(0) == 1 and B > 1:
        value = value.expand(B, -1)
    assert value.shape == (B, cfg.value_dim), (
        f"value shape {value.shape} doesn't match (B={B}, "
        f"value_dim={cfg.value_dim})"
    )
    # Build key vector
    k0, k1 = parabolic_key(key_id, cfg.key_channels)
    # Clone to avoid in-place on autograd leaf
    out = residual.clone()
    k_start, k_stop, _ = cfg.key_channels.indices(D)
    v_start, v_stop, _ = cfg.value_channels.indices(D)
    out[:, position, k_start] = k0
    out[:, position, k_start + 1] = k1
    out[:, position, v_start:v_stop] = value
    return out


def read_by_key_attention(
    residual: torch.Tensor,
    query_key_id: int,
    cfg: KeyedMemoryConfig,
) -> torch.Tensor:
    """Soft-attention lookup: return values where keys ≈ query_key_id.

    Uses parabolic-key math: build query vector q = (query_key_id, 1),
    compute scores q · k for every position's 2-channel key, softmax
    over positions, return softmax-weighted values.

    Dot-product q · (2k, -k²) = 2·query_id·k - k². At k == query_id
    this equals query_id² (the max over non-negative integer k).
    Softmax picks out the matching position.

    Returns:
        (B, cfg.value_dim) — one retrieved value per batch element.
    """
    assert 0 <= query_key_id < cfg.max_key_id
    B, S, D = residual.shape
    cfg.validate(D)
    k_start, k_stop, _ = cfg.key_channels.indices(D)
    v_start, v_stop, _ = cfg.value_channels.indices(D)

    # Extract keys and values at every position
    keys = residual[:, :, k_start:k_stop]       # (B, S, 2)
    values = residual[:, :, v_start:v_stop]     # (B, S, value_dim)

    # Query: (query_id, 1) so that q·(2k, -k²) = 2·q·k - k²
    q = torch.tensor(
        [float(query_key_id), 1.0],
        device=residual.device, dtype=residual.dtype,
    )  # (2,)
    scores = torch.einsum("bsd,d->bs", keys, q)  # (B, S)
    weights = F.softmax(scores, dim=-1)          # (B, S)
    return torch.einsum("bs,bsd->bd", weights, values)


def read_by_key_hard(
    residual: torch.Tensor,
    query_key_id: int,
    cfg: KeyedMemoryConfig,
) -> torch.Tensor:
    """Hard-argmax version of read_by_key_attention. Non-differentiable
    but exact: returns the value at the position whose key has the
    highest dot-product with the query. Use when you need bit-exact
    retrieval (e.g. compiled-program-style lookups)."""
    assert 0 <= query_key_id < cfg.max_key_id
    B, S, D = residual.shape
    cfg.validate(D)
    k_start, k_stop, _ = cfg.key_channels.indices(D)
    v_start, v_stop, _ = cfg.value_channels.indices(D)

    keys = residual[:, :, k_start:k_stop]
    values = residual[:, :, v_start:v_stop]
    q = torch.tensor(
        [float(query_key_id), 1.0],
        device=residual.device, dtype=residual.dtype,
    )
    scores = torch.einsum("bsd,d->bs", keys, q)  # (B, S)
    idx = scores.argmax(dim=-1)  # (B,)
    return values[torch.arange(B, device=residual.device), idx]


def write_by_name(
    residual: torch.Tensor,
    position: int,
    key_name: str,
    value: torch.Tensor,
    registry: KeyRegistry,
    cfg: KeyedMemoryConfig,
) -> torch.Tensor:
    """Symbolic write: `write_by_name(x, 1, "sum", v, registry, cfg)`.

    Registers `name` in `registry` if not already present.
    """
    if key_name not in registry:
        registry.register(key_name)
    return write_keyed_slot(
        residual, position, registry.id_of(key_name), value, cfg,
    )


def read_by_name(
    residual: torch.Tensor,
    key_name: str,
    registry: KeyRegistry,
    cfg: KeyedMemoryConfig,
    hard: bool = False,
) -> torch.Tensor:
    """Symbolic read: `read_by_name(x, "sum", registry, cfg)` → (B, V)."""
    query_id = registry.id_of(key_name)
    if hard:
        return read_by_key_hard(residual, query_id, cfg)
    return read_by_key_attention(residual, query_id, cfg)
