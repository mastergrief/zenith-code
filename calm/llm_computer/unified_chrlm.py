"""Unified CHRLM — one Small2DTransformer .pt with compiled + trained fused.

Session-27's fusion MVP (commit `e9f5ecb`) showed empty layer 0 + compiled
adder at layer 1 survives side-by-side in one model. This module
generalizes the pattern: install N compiled programs into N specified
layers of a larger Small2DTransformer, freeze those layers, train the
remaining layers on mixed LM + HRM data.

Result: ONE `.pt` file that runs exact arithmetic (compiled layers),
natural language generation (trained layers), and structure parsing
(trained layers with HRM mode prefix) in a single forward pass — without
routing between separate tensors.

Usage:
    from calm.llm_computer.unified_chrlm import (
        UnifiedCHRLMConfig, build_unified_chrlm,
        install_compiled_program, freeze_layer_params,
    )
    from calm.llm_computer.programs.adder_tiny import build_adder_tiny

    # 4-layer substrate: 1 compiled, 3 trained
    cfg = UnifiedCHRLMConfig(d_model=32, n_heads=16, n_layers=4, d_ffn=64,
                             max_len=32, vocab_size=16)
    model = build_unified_chrlm(cfg)
    install_compiled_program(model, build_adder_tiny, target_layer=0)
    freeze_layer_params(model, layer_idx=0)

    # `model` is now a single nn.Module. Train its unfrozen params normally.
    # Layer 0 remains exactly the compiled adder; layers 1-3 are trainable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Optional

import torch
import torch.nn as nn

from calm.llm_computer.channel_registry import (
    ChannelAllocation, ChannelRegistry, MultiStreamChannelRegistry,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.multi_stream import (
    MultiStreamConfig, MultiStreamTransformer,
)


@dataclass
class UnifiedCHRLMConfig(Small2DConfig):
    """Config for a unified CHRLM tensor. Inherits Small2DConfig.

    `compiled_layers` lists the layer indices occupied by compiled programs.
    Those layers' weights are loaded from the compiled source and marked
    frozen before training begins.
    """
    compiled_layers: tuple[int, ...] = ()


def build_unified_chrlm(cfg: UnifiedCHRLMConfig) -> Small2DTransformer:
    """Instantiate an empty unified CHRLM substrate. Zero-initialized until
    `install_compiled_program` copies in compiled weights."""
    assert cfg.d_head == 2, f"substrate invariant d_head=2, got {cfg.d_head}"
    # Zero-initialize all parameters so compiled regions load cleanly.
    model = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def install_compiled_program(
    model: Small2DTransformer,
    program_builder: Callable[..., Small2DTransformer],
    *,
    target_layer: int,
    builder_kwargs: Optional[dict] = None,
) -> None:
    """Compile a program and copy its weights into `model` at `target_layer`.

    `program_builder` is a factory (e.g. `build_adder_tiny`) that returns a
    compiled Small2DTransformer. The source program's layer-0 weights are
    copied into `model`'s `target_layer`. Embedding + head tensors copy
    whole (they're compiled globally, not per-layer).

    Assumes source and model share the same vocab / d_model / max_len /
    n_heads. If they differ, a specialized adapter must be written (out
    of scope here — the session-27 pattern uses dimension-matched builders).
    """
    builder_kwargs = builder_kwargs or {}
    src = program_builder(**builder_kwargs)
    src_cfg = src.config
    dst_cfg = model.config
    assert src_cfg.d_model == dst_cfg.d_model, (
        f"d_model mismatch: src={src_cfg.d_model} dst={dst_cfg.d_model}. "
        "Recompile the source program at the unified d_model, or build an "
        "adapter layer."
    )
    assert src_cfg.n_heads == dst_cfg.n_heads, (
        f"n_heads mismatch: src={src_cfg.n_heads} dst={dst_cfg.n_heads}"
    )
    assert src_cfg.max_len <= dst_cfg.max_len, (
        f"source max_len {src_cfg.max_len} > dst {dst_cfg.max_len}"
    )

    with torch.no_grad():
        # Token embedding: OR in (zero-init dst means this is a simple copy)
        model.tok.weight[: src_cfg.vocab_size] += src.tok.weight

        # Positional embedding: same.
        model.pos.weight[: src_cfg.max_len] += src.pos.weight

        # Per-layer weights: copy src layer 0 into dst target_layer.
        # Source programs are single-layer; if a program uses multiple layers,
        # it must be installed into a contiguous layer range (not supported
        # here yet — session-27 programs are single-layer compiles).
        if src_cfg.n_layers != 1:
            raise NotImplementedError(
                f"multi-layer compiled programs not yet supported "
                f"(src has {src_cfg.n_layers} layers)"
            )
        model.W_qkv[target_layer].weight += src.W_qkv[0].weight
        model.W_out[target_layer].weight += src.W_out[0].weight
        model.ff_in[target_layer].weight += src.ff_in[0].weight
        model.ff_out[target_layer].weight += src.ff_out[0].weight

        # LM head: add compiled head contributions. Multiple compiled programs
        # can contribute to the head (see _apply_linear_head's += accumulation
        # fix from session 27 commit `dee8c42`).
        model.head.weight[: src_cfg.vocab_size] += src.head.weight


def freeze_layer_params(model: Small2DTransformer, *,
                         layer_idx: int) -> int:
    """Mark all weights in `layer_idx` as requires_grad=False.

    Returns the number of parameters frozen. Use this after installing a
    compiled program to protect its weights from gradient updates during
    training of the surrounding layers.
    """
    frozen = 0
    for linear in (model.W_qkv[layer_idx], model.W_out[layer_idx],
                   model.ff_in[layer_idx], model.ff_out[layer_idx]):
        for p in linear.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def freeze_embeddings_and_head(model: Small2DTransformer) -> int:
    """Freeze token embedding, positional embedding, and LM head.

    Use when compiled programs' global tensors must stay exact while
    surrounding trained layers update. Returns number of params frozen.
    """
    frozen = 0
    for module in (model.tok, model.pos, model.head):
        for p in module.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def trainable_param_count(model: Small2DTransformer) -> int:
    """Count how many parameters are still trainable."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def install_compiled_with_registry(
    model: Small2DTransformer,
    program_builder: Callable[..., Small2DTransformer],
    *,
    target_layer: int,
    registry: ChannelRegistry,
    allocations: Iterable[ChannelAllocation],
    builder_kwargs: Optional[dict] = None,
) -> None:
    """install_compiled_program + L1 channel registry integration.

    Installs the compiled program as before, AND registers each supplied
    ChannelAllocation into the registry. Raises AllocationError if any
    of the allocations conflict with existing registry entries (e.g.
    another compiled program already claims those channels).

    Use `allocations` to declare which channels the compiled program
    owns. For adder_tiny, use `adder_tiny_allocation()` from
    `channel_registry.py`.
    """
    # Register FIRST so allocation failures don't leave a partial install.
    for alloc in allocations:
        registry.allocate(
            card_name=alloc.card_name,
            channels=alloc.channels,
            ch_type=alloc.ch_type,
            purpose=alloc.purpose,
        )
    install_compiled_program(
        model, program_builder, target_layer=target_layer,
        builder_kwargs=builder_kwargs,
    )


def install_compiled_in_stream(
    ms_model: MultiStreamTransformer,
    program_builder: Callable[..., Small2DTransformer],
    *,
    stream_name: str,
    target_layer: int = 0,
    builder_kwargs: Optional[dict] = None,
    registry: Optional[MultiStreamChannelRegistry] = None,
    allocations: Optional[Iterable[ChannelAllocation]] = None,
) -> None:
    """Install a compiled program into a specific stream of a
    multi-stream unified tensor.

    Each stream is an independent `_StreamStack` with its own tok/pos
    embeddings, per-layer parameters. This function builds the compiled
    source via `program_builder` and copies weights into the named
    stream's embedding + layer slots. Other streams are unchanged.

    Assumes source builder's d_model/n_heads/max_len/vocab match the
    target stream's spec. Raises AssertionError otherwise.

    If `registry` and `allocations` are supplied, registers the
    compiled program's channel claims in the stream's registry.
    """
    builder_kwargs = builder_kwargs or {}
    src = program_builder(**builder_kwargs)
    src_cfg = src.config
    stream_spec = ms_model.config.stream_by_name(stream_name)
    assert src_cfg.d_model == stream_spec.d_model, (
        f"d_model mismatch: src={src_cfg.d_model} "
        f"stream {stream_name!r}={stream_spec.d_model}"
    )
    assert src_cfg.n_heads == stream_spec.n_heads, (
        f"n_heads mismatch: src={src_cfg.n_heads} "
        f"stream {stream_name!r}={stream_spec.n_heads}"
    )
    assert src_cfg.max_len <= ms_model.config.max_len, (
        f"src max_len {src_cfg.max_len} > stream max_len "
        f"{ms_model.config.max_len}"
    )
    assert src_cfg.n_layers == 1, (
        "multi-layer compiled programs not yet supported for stream install"
    )

    stream = ms_model.streams[stream_name]
    with torch.no_grad():
        stream.tok.weight[: src_cfg.vocab_size] += src.tok.weight
        stream.pos.weight[: src_cfg.max_len] += src.pos.weight
        stream.W_qkv[target_layer].weight += src.W_qkv[0].weight
        stream.W_out[target_layer].weight += src.W_out[0].weight
        stream.ff_in[target_layer].weight += src.ff_in[0].weight
        stream.ff_out[target_layer].weight += src.ff_out[0].weight
        # Shared head: multi-stream head reads concat(stream_finals);
        # the stream's output slice of the head is offset by
        # sum(d_model) for streams preceding this one.
        offset = 0
        for s in ms_model.config.streams:
            if s.name == stream_name:
                break
            offset += s.d_model
        # src.head.weight is (V, d_model). Project into (V, total_d)
        # at the stream's offset.
        ms_model.head.weight[: src_cfg.vocab_size, offset: offset + src_cfg.d_model] += \
            src.head.weight

    if registry is not None and allocations is not None:
        for alloc in allocations:
            registry.allocate(
                stream_name=stream_name,
                card_name=alloc.card_name,
                channels=alloc.channels,
                ch_type=alloc.ch_type,
                purpose=alloc.purpose,
            )


def freeze_stream_layer(
    ms_model: MultiStreamTransformer,
    stream_name: str,
    layer_idx: int,
) -> int:
    """Freeze a specific layer of a specific stream. Multi-stream
    analog of freeze_layer_params.
    """
    frozen = 0
    stream = ms_model.streams[stream_name]
    for linear in (stream.W_qkv[layer_idx], stream.W_out[layer_idx],
                   stream.ff_in[layer_idx], stream.ff_out[layer_idx]):
        for p in linear.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def freeze_stream_embeddings(
    ms_model: MultiStreamTransformer,
    stream_name: str,
) -> int:
    """Freeze the token + pos embeddings of a specific stream."""
    frozen = 0
    stream = ms_model.streams[stream_name]
    for module in (stream.tok, stream.pos):
        for p in module.parameters():
            p.requires_grad = False
            frozen += p.numel()
    return frozen


def verify_compiled_preserved(
    source_behavior_fn: Callable[[Small2DTransformer], int],
    model: Small2DTransformer,
    expected: int,
) -> bool:
    """Run `source_behavior_fn(model)` (e.g. exhaustive adder test) and
    check it matches `expected`. Use this as a pre-training baseline AND
    a post-training regression check.
    """
    result = source_behavior_fn(model)
    return result == expected
