"""Channel masking — prevent a trainable layer from writing to compiled
program output channels via gradient hooks.

Problem this solves: in a unified CHRLM, compiled programs allocate
specific residual-stream channels (adder_tiny writes step functions to
channels 3-9; head reads from those channels). Trainable layers that come
after a compiled layer add their residual contributions to ALL channels
via their `W_out` (attention projection) and `ff_out` (FFN projection).
Even small gradient updates accumulate and corrupt the compiled
program's output — observed in minibench_unified as adder regressing
16/16 → 15/16 in 100 steps at lr=1e-3.

Solution: zero-out the rows of `W_out[layer]` and `ff_out[layer]` that
correspond to protected channels at setup time, and register gradient
hooks that zero the gradient on those rows every backward pass. This
guarantees:
  - forward pass: trainable layer's residual write to protected channels
    is always 0 (zero weight row → zero output column)
  - backward pass: gradient to protected rows is zeroed before the
    optimizer step, so they stay at 0 forever.

Leaves unprotected channels fully trainable. Trainable layer can still
read from protected channels (via W_qkv and ff_in input weights), write
to unprotected channels, and attention/FFN gates still operate on all
channels. Only the residual WRITE path to protected channels is blocked.

Layer indexing: apply this to a trainable layer that comes AFTER the
compiled layer in the forward-pass order. Don't apply to the compiled
layer itself (it's already frozen).

Caveat: this only protects residual-stream channel writes. If a compiled
program depends on a specific attention pattern that a trainable layer's
Q/K projections can disturb via shared residual reads, additional
protections may be needed. For the session-28 adder case (pure
residual-channel writes read by a frozen head), channel masking on
W_out and ff_out is sufficient.
"""

from __future__ import annotations

from typing import Iterable

import torch
from torch.utils.hooks import RemovableHandle

from calm.llm_computer.model import Small2DTransformer


def protect_residual_channels(
    model: Small2DTransformer,
    layer_idx: int,
    protected_channels: Iterable[int],
) -> list[RemovableHandle]:
    """Block a trainable layer from writing to specified residual channels.

    Zeros the corresponding rows of `W_out[layer_idx]` and
    `ff_out[layer_idx]` and registers gradient hooks that keep those rows
    at zero through optimization.

    Args:
        model: unified substrate model. Must be a Small2DTransformer.
        layer_idx: index of the TRAINABLE layer to constrain. Should come
            after the compiled layer in forward order. Applying to the
            compiled layer itself is redundant (compiled layers are frozen).
        protected_channels: residual-stream channel indices that compiled
            programs own. The trainable layer will not be able to write
            to these channels.

    Returns:
        List of RemovableHandle — call `.remove()` on each to tear down
        the gradient hooks (e.g. for post-training inference).

    Raises:
        IndexError: if any channel is out of `[0, d_model)` range.
    """
    d_model = model.config.d_model
    channels = list(protected_channels)
    for c in channels:
        if not 0 <= c < d_model:
            raise IndexError(
                f"protected channel {c} out of range [0, {d_model})"
            )

    # Build per-weight masks. W_out and ff_out have shape (d_model, _in).
    # Row i of weight writes to output channel i. A mask of shape
    # (d_model, 1) broadcasts across the input dimension.
    mask = torch.ones(d_model, 1)
    for c in channels:
        mask[c, 0] = 0.0

    handles: list[RemovableHandle] = []
    with torch.no_grad():
        for weight in (
            model.W_out[layer_idx].weight,
            model.ff_out[layer_idx].weight,
        ):
            # Zero the protected rows at init so forward pass contribution
            # starts at zero too.
            for c in channels:
                weight[c, :].zero_()
            # Register backward hook that zeros the gradient on protected
            # rows. Capture mask per-tensor via default arg so each hook
            # binds to its own broadcast shape.
            w_mask = mask.to(weight.device, weight.dtype).expand_as(weight)
            h = weight.register_hook(
                lambda g, m=w_mask: g * m
            )
            handles.append(h)

    return handles


def compiled_output_channels_adder_tiny() -> tuple[int, ...]:
    """Residual channels written by the compiled adder_tiny program.

    Per adder_tiny.py channel layout comment:
      ch 0: own token scalar (from embedding, not layer)
      ch 1: bias 1 (from embedding, not layer)
      ch 2: a scalar (from LookUp in layer 0 attention)
      ch 3..9: step functions (from layer 0 FFN ReGLU)

    Channels 2..9 are the ones the layer writes (embeddings come from
    tok/pos, already frozen separately via freeze_embeddings_and_head).
    Channels 3..9 are what the LM head reads to compute sum logits. For
    minimal interference, protecting 3..9 is sufficient; protecting 2..9
    is stronger (protects the attention-output channel too).

    Returns the stronger set (2..9) by default; caller can slice to 3..9
    if they want to let trainable layers read/write channel 2 freely.
    """
    return tuple(range(2, 10))
