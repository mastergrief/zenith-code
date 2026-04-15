"""Unified tensor substrate — one Small2DTransformer hosts Gemma + HRMs
+ compiled cards + working memory via channel and sub-head partitioning.

Architecture:

  d_model = upscaled from Gemma's native 2560 to 4096 (or similar 2×-ish)
  n_heads = d_model / 2 sub-heads at d_head=2
  d_ffn = upscaled proportionally

Partitioning strategy (all layers, partitioned differently per layer type):

  Channels:
    0..gemma_d_model-1           Gemma's residual (trained, frozen)
    gemma_d_model..+hrm_io_width Per-HRM I/O channels
    +working_memory_width         Keyed memory + call stack regions
    remainder                     Free for trainable adapters / future

  Sub-heads (per SWA layer):
    0..gemma_swa_sub_heads        Gemma's attention (grouped, n_groups=8,
                                   group_size=128 = 1024 sub-heads)
    +hrm_sub_heads                Per-HRM attention (each HRM takes 16)
    +compiled_sub_heads           Compiled card attention heads
    remainder                     Free

  Sub-heads (per full attention layer):
    0..gemma_full_sub_heads        Gemma's attention (grouped, n_groups=8,
                                   group_size=256 = 2048 sub-heads)
    (no room for extras on full layers — they saturate the substrate's
    sub-heads. HRMs and cards contribute via FFN only on these layers.)

This MVP ships the CONFIG + PARTITIONING primitives. Actual Gemma weight
install (involves tq4 block repacking for the upscaled shape) is a
separate step with its own failure modes. Tests use zero-init tensors
to verify partitioning logic without requiring real weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

from calm.llm_computer.gemma4_config import Gemma4Config
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)


@dataclass
class UnifiedTensorConfig:
    """Config for the unified substrate hosting Gemma + HRMs + cards.

    Designed around Gemma 4 E4B dimensions by default but parameterized
    so smaller test configs work identically.
    """
    # Gemma backbone
    gemma_d_model: int = 2560
    gemma_n_heads: int = 8
    gemma_n_kv_heads: int = 2
    gemma_n_layers: int = 42
    gemma_d_ffn: int = 10240
    gemma_swa_head_dim: int = 256
    gemma_full_head_dim: int = 512
    gemma_vocab_size: int = 262144
    gemma_max_position: int = 131072
    gemma_full_layer_indices: tuple[int, ...] = field(
        default_factory=lambda: tuple(range(5, 42, 6)),
    )

    # HRM specialists — each occupies a slice
    hrm_specialists: tuple[str, ...] = field(
        default_factory=lambda: (
            "math", "nl", "word", "gsm", "multi", "router",
        ),
    )
    hrm_d_model: int = 32         # SubstrateHRM native width
    hrm_n_heads: int = 16         # SubstrateHRM native heads (d_head=2)

    # Compiled cards — how much substrate to reserve
    n_compiled_sub_heads: int = 128

    # Working memory regions (in channels)
    keyed_memory_channels: int = 512
    call_stack_channels: int = 256
    card_scratchpad_channels: int = 256

    def __post_init__(self):
        # Compute both sizing constraints and pick the larger.
        # (1) Sub-head requirement: the largest layer type (typically full
        # attention) determines substrate_n_heads.
        full_sub_heads = self.gemma_n_heads * (self.gemma_full_head_dim // 2)
        swa_gemma_sub_heads = self.gemma_n_heads * (self.gemma_swa_head_dim // 2)
        hrm_total_sub_heads = len(self.hrm_specialists) * self.hrm_n_heads
        swa_extras = hrm_total_sub_heads + self.n_compiled_sub_heads
        swa_total_sub_heads = swa_gemma_sub_heads + swa_extras
        n_heads_from_sub_heads = max(full_sub_heads, swa_total_sub_heads)

        # (2) Channel requirement: sum of all channel allocations.
        total_channels = (
            self.gemma_d_model
            + len(self.hrm_specialists) * self.hrm_d_model
            + self.keyed_memory_channels
            + self.call_stack_channels
            + self.card_scratchpad_channels
        )
        # d_model must be >= total_channels AND equal 2 × n_heads (d_head=2)
        n_heads_from_channels = (total_channels + 1) // 2  # ceil div

        # Pick the binding constraint
        self._substrate_n_heads = max(n_heads_from_sub_heads, n_heads_from_channels)
        self._substrate_d_model = self._substrate_n_heads * 2
        # Round substrate_d_model up to a multiple of HEAD_DIM=256 so
        # tq4 block alignment holds. This may over-allocate by up to 255
        # channels but keeps the tq4 substrate compatible.
        HEAD_DIM = 256
        if self._substrate_d_model % HEAD_DIM != 0:
            rounded = ((self._substrate_d_model + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM
            self._substrate_d_model = rounded
            self._substrate_n_heads = rounded // 2
        # FFN sized proportionally, rounded to tq4 block multiple (256)
        self._substrate_d_ffn = int(
            self.gemma_d_ffn * (self._substrate_d_model / self.gemma_d_model)
        )
        # Round UP to multiple of HEAD_DIM
        self._substrate_d_ffn = max(
            HEAD_DIM,
            ((self._substrate_d_ffn + HEAD_DIM - 1) // HEAD_DIM) * HEAD_DIM,
        )

        # Channel allocation map
        cursor = 0
        self.gemma_channels = (cursor, cursor + self.gemma_d_model)
        cursor += self.gemma_d_model
        self.hrm_channels: dict[str, tuple[int, int]] = {}
        for name in self.hrm_specialists:
            self.hrm_channels[name] = (cursor, cursor + self.hrm_d_model)
            cursor += self.hrm_d_model
        self.keyed_memory_range = (cursor, cursor + self.keyed_memory_channels)
        cursor += self.keyed_memory_channels
        self.call_stack_range = (cursor, cursor + self.call_stack_channels)
        cursor += self.call_stack_channels
        self.card_scratchpad_range = (
            cursor, cursor + self.card_scratchpad_channels
        )
        cursor += self.card_scratchpad_channels
        self.free_channels = (cursor, self._substrate_d_model)
        # Sanity: cursor must be within substrate
        assert cursor <= self._substrate_d_model, (
            f"channel allocations ({cursor}) overflow substrate d_model "
            f"({self._substrate_d_model})"
        )

        # Sub-head allocation (SWA layers)
        sh_cursor = 0
        self.gemma_swa_sub_heads_range = (
            sh_cursor, sh_cursor + swa_gemma_sub_heads,
        )
        sh_cursor += swa_gemma_sub_heads
        self.hrm_sub_heads: dict[str, tuple[int, int]] = {}
        for name in self.hrm_specialists:
            self.hrm_sub_heads[name] = (
                sh_cursor, sh_cursor + self.hrm_n_heads,
            )
            sh_cursor += self.hrm_n_heads
        self.compiled_sub_heads_range = (
            sh_cursor, sh_cursor + self.n_compiled_sub_heads,
        )
        sh_cursor += self.n_compiled_sub_heads
        self.free_sub_heads = (sh_cursor, self._substrate_n_heads)
        assert sh_cursor <= self._substrate_n_heads, (
            f"SWA sub-head allocations ({sh_cursor}) overflow substrate "
            f"n_heads ({self._substrate_n_heads})"
        )

        # Sub-head allocation (full attention layers) — Gemma saturates
        self.gemma_full_sub_heads_range = (0, full_sub_heads)

    @property
    def substrate_d_model(self) -> int:
        return self._substrate_d_model

    @property
    def substrate_n_heads(self) -> int:
        return self._substrate_n_heads

    @property
    def substrate_d_ffn(self) -> int:
        return self._substrate_d_ffn

    def build_grouped_config(self) -> GroupedSmall2DConfig:
        """Generate a GroupedSmall2DConfig with per-layer modes matching
        Gemma 4's SWA/full alternation, with SWA layers sized for the
        extra HRM+card sub-heads and full layers sized to Gemma's full
        attention requirement only."""
        layer_modes = []
        layer_n_groups = []
        layer_group_sizes = []

        # On SWA layers: grouped with gemma_n_heads=8 groups × 128 group_size
        # The substrate has extra sub-heads beyond 1024; Gemma only uses
        # the first 1024 via the grouping.
        # But n_groups × group_size must == substrate_n_heads.
        # With substrate_n_heads possibly 2048+, Gemma's SWA grouping
        # (8, 128) would only cover 1024 sub-heads.
        #
        # Solution: use "single" mode for EVERY layer, but on Gemma's
        # channels only. For the MVP we accept this: compiled cards and
        # HRMs run in single mode on their sub-heads, Gemma runs in
        # single mode on its sub-heads (equivalent to treating each
        # sub-head as its own head).
        #
        # This isn't the EXACT Gemma decomposition (grouped mode is), but
        # it's sufficient for the partitioning test. Real grouped mode
        # requires per-layer n_heads variation which Small2DConfig doesn't
        # support — that's a deeper refactor.
        for i in range(self.gemma_n_layers):
            layer_modes.append("single")
            layer_n_groups.append(1)
            layer_group_sizes.append(self._substrate_n_heads)

        return GroupedSmall2DConfig(
            vocab_size=self.gemma_vocab_size,
            d_model=self._substrate_d_model,
            n_heads=self._substrate_n_heads,
            n_layers=self.gemma_n_layers,
            d_ffn=self._substrate_d_ffn,
            max_len=self.gemma_max_position,
            use_hard_max=False,
            layer_modes=tuple(layer_modes),
            layer_n_groups=tuple(layer_n_groups),
            layer_group_sizes=tuple(layer_group_sizes),
        )

    def describe(self) -> str:
        lines = [
            f"UnifiedTensorConfig:",
            f"  substrate: d_model={self.substrate_d_model} "
            f"n_heads={self.substrate_n_heads} d_ffn={self.substrate_d_ffn}",
            f"  (Gemma native: d_model={self.gemma_d_model})",
            f"  channel map:",
            f"    Gemma       : {self.gemma_channels[0]}..{self.gemma_channels[1]-1}",
        ]
        for name, (lo, hi) in self.hrm_channels.items():
            lines.append(f"    HRM '{name}' : {lo}..{hi-1}")
        lines += [
            f"    keyed_mem   : {self.keyed_memory_range[0]}..{self.keyed_memory_range[1]-1}",
            f"    call_stack  : {self.call_stack_range[0]}..{self.call_stack_range[1]-1}",
            f"    card_scratch: {self.card_scratchpad_range[0]}..{self.card_scratchpad_range[1]-1}",
            f"    FREE        : {self.free_channels[0]}..{self.free_channels[1]-1}",
            f"  SWA sub-head map:",
            f"    Gemma       : {self.gemma_swa_sub_heads_range[0]}..{self.gemma_swa_sub_heads_range[1]-1}",
        ]
        for name, (lo, hi) in self.hrm_sub_heads.items():
            lines.append(f"    HRM '{name}' : {lo}..{hi-1}")
        lines += [
            f"    compiled    : {self.compiled_sub_heads_range[0]}..{self.compiled_sub_heads_range[1]-1}",
            f"    FREE        : {self.free_sub_heads[0]}..{self.free_sub_heads[1]-1}",
            f"  full attn sub-heads: {self.gemma_full_sub_heads_range[0]}..{self.gemma_full_sub_heads_range[1]-1} (Gemma only)",
        ]
        return "\n".join(lines)


def build_unified_substrate(cfg: UnifiedTensorConfig) -> GroupedSmall2DTransformer:
    """Instantiate the unified substrate with zeroed weights. Caller
    then installs Gemma, HRM, and compiled card weights into their
    respective channel/sub-head ranges."""
    substrate_cfg = cfg.build_grouped_config()
    model = GroupedSmall2DTransformer(substrate_cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()
    return model


def install_padded_weight(
    target_module: nn.Linear,
    source_weight: torch.Tensor,
    row_offset: int = 0,
    col_offset: int = 0,
) -> None:
    """Install `source_weight` into a corner of `target_module.weight`.

    target_module.weight shape: (out_features, in_features).
    source_weight shape: (src_out, src_in).
    After: target[row_offset:row_offset+src_out, col_offset:col_offset+src_in]
        = source_weight.
    """
    src_out, src_in = source_weight.shape
    tgt = target_module.weight
    assert row_offset + src_out <= tgt.shape[0], (
        f"row overflow: {row_offset}+{src_out} > {tgt.shape[0]}"
    )
    assert col_offset + src_in <= tgt.shape[1], (
        f"col overflow: {col_offset}+{src_in} > {tgt.shape[1]}"
    )
    with torch.no_grad():
        tgt[row_offset:row_offset+src_out, col_offset:col_offset+src_in] = source_weight
