"""Call stack — LIFO working memory for recursive reasoning.

Keyed memory (keyed_memory.py) is read-anytime-write-anytime. Real
reasoning often has RECURSIVE structure: solve subproblem → subproblem
pushes its context → sub-subproblem runs → pop back to parent context.
Standard transformers model this implicitly via attention; a substrate
with an explicit stack makes recursion legible.

MVP design:
  - A contiguous region of the residual stream reserved for the stack.
  - Each "frame" is `frame_size` channels. `max_depth` frames total.
  - Top-of-stack pointer lives in a separate channel (scalar).
  - `push(x, frame)` writes frame at TOS, increments TOS.
  - `pop(x) → (x_after, frame)` decrements TOS, reads frame.
  - `peek(x, offset=0)` reads frame at TOS - offset without popping.

This is RUNTIME callable Python utility on a residual tensor — not yet
a compiled program in the gate-graph IR. Next round: compile push/pop
into weights so the substrate can autonomously use the stack.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CallStackConfig:
    """Layout of the call-stack region within a residual tensor.

    Attributes:
        stack_channels: slice of the residual stream that holds frames.
            Size must equal max_depth * frame_size.
        tos_channel: single channel index holding top-of-stack pointer
            as a scalar (interpretable as an integer 0..max_depth).
        frame_size: number of channels per frame.
        max_depth: maximum number of frames the stack can hold.
    """
    stack_channels: slice
    tos_channel: int
    frame_size: int
    max_depth: int

    def validate(self, d_model: int) -> None:
        start, stop, _ = self.stack_channels.indices(d_model)
        required = self.frame_size * self.max_depth
        assert stop - start == required, (
            f"stack_channels width {stop - start} != "
            f"frame_size * max_depth = {required}"
        )
        assert 0 <= self.tos_channel < d_model
        # TOS must be outside stack region to avoid clobber
        assert not (start <= self.tos_channel < stop), (
            f"tos_channel {self.tos_channel} lies inside stack region "
            f"[{start}, {stop}) — must be a separate channel"
        )


def _frame_slice(cfg: CallStackConfig, depth: int) -> slice:
    """Residual-channel slice for the frame at given depth (0 = bottom)."""
    start, _stop, _step = cfg.stack_channels.indices(10**9)
    frame_start = start + depth * cfg.frame_size
    return slice(frame_start, frame_start + cfg.frame_size)


def get_tos(residual: torch.Tensor, cfg: CallStackConfig,
            position: int = 0) -> int:
    """Read the top-of-stack pointer. Reads from `position` (usually 0)."""
    val = residual[:, position, cfg.tos_channel]
    # Batch 0 value, rounded to int
    return int(val[0].round().item())


def set_tos(residual: torch.Tensor, cfg: CallStackConfig,
            new_tos: int, position: int = 0) -> torch.Tensor:
    """Returns a new residual with TOS set to new_tos at position.

    Does NOT mutate the input. Rest of residual preserved.
    """
    assert 0 <= new_tos <= cfg.max_depth, (
        f"new_tos {new_tos} out of range [0, {cfg.max_depth}]"
    )
    out = residual.clone()
    out[:, position, cfg.tos_channel] = float(new_tos)
    return out


def push(
    residual: torch.Tensor,
    frame: torch.Tensor,
    cfg: CallStackConfig,
    position: int = 0,
) -> torch.Tensor:
    """Push a frame onto the stack at `position`.

    Args:
        residual: (B, S, d_model).
        frame: shape (B, frame_size) or (frame_size,) broadcast.
        cfg: stack layout.
        position: which sequence position to modify (usually 0).

    Returns:
        New residual with frame written to the slot at TOS and TOS
        incremented. Raises IndexError if stack would overflow.
    """
    B, S, D = residual.shape
    cfg.validate(D)
    tos = get_tos(residual, cfg, position)
    if tos >= cfg.max_depth:
        raise IndexError(
            f"stack overflow: tos={tos} >= max_depth={cfg.max_depth}"
        )
    # Broadcast frame to (B, frame_size)
    if frame.dim() == 1:
        frame = frame.unsqueeze(0).expand(B, -1)
    assert frame.shape == (B, cfg.frame_size), (
        f"frame shape {frame.shape} != ({B}, {cfg.frame_size})"
    )
    # Write frame at depth=tos, then increment TOS
    out = residual.clone()
    slot = _frame_slice(cfg, tos)
    out[:, position, slot] = frame
    out[:, position, cfg.tos_channel] = float(tos + 1)
    return out


def pop(
    residual: torch.Tensor,
    cfg: CallStackConfig,
    position: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pop a frame off the stack.

    Returns (new_residual, popped_frame) where popped_frame is
    (B, frame_size). Raises IndexError if stack is empty.
    """
    B, S, D = residual.shape
    cfg.validate(D)
    tos = get_tos(residual, cfg, position)
    if tos <= 0:
        raise IndexError(f"stack underflow: tos={tos}")
    # Read the frame at depth tos-1
    new_tos = tos - 1
    slot = _frame_slice(cfg, new_tos)
    popped = residual[:, position, slot].clone()
    # Zero out the slot and decrement TOS
    out = residual.clone()
    out[:, position, slot] = 0.0
    out[:, position, cfg.tos_channel] = float(new_tos)
    return out, popped


def peek(
    residual: torch.Tensor,
    cfg: CallStackConfig,
    offset: int = 0,
    position: int = 0,
) -> torch.Tensor:
    """Read the frame at TOS - 1 - offset without popping. Returns
    (B, frame_size). offset=0 reads the most recent; offset=1 reads the
    one below; etc. Raises IndexError if nothing at that depth."""
    cfg.validate(residual.shape[-1])
    tos = get_tos(residual, cfg, position)
    target_depth = tos - 1 - offset
    if target_depth < 0:
        raise IndexError(
            f"peek past stack bottom: tos={tos} offset={offset}"
        )
    slot = _frame_slice(cfg, target_depth)
    return residual[:, position, slot].clone()


def depth(residual: torch.Tensor, cfg: CallStackConfig,
          position: int = 0) -> int:
    """Number of frames currently on the stack."""
    return get_tos(residual, cfg, position)
