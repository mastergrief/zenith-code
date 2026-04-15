"""Load Gemma E4B weights into ONE Small2DTransformer tensor.

This is the final piece of the "Gemma in unified CHRLM tensor" path.
Uses the grouped attention decomposition to host Gemma's d_head=256
(SWA) and d_head=512 (full) attention inside a single Small2DTransformer
with d_head=2 throughout.

Design:
  - d_model = Gemma's d_model (2560 for E4B)
  - n_heads = d_model / 2 = 1280 sub-heads
  - d_head = 2 (substrate invariant preserved)
  - Per-layer mode:
    - SWA layers: grouped with n_groups=Gemma_n_heads=8, group_size=128
      (equivalent d_head = 256)
    - Full attention layers: grouped with n_groups=8, group_size=256
      (equivalent d_head = 512)

Per-layer complications of Gemma 4 (handled separately):
  - Sliding window on SWA layers → needs sliding_window_mask
  - RoPE per-group (not per-sub-head) → needs grouped RoPE application
  - Per-head Q/K norms, post-attn-norm, layer_output_scale — copied as
    Gemma4Layer does; this MVP skips these for initial validation
  - Per-layer token embeddings (Gemma 4 specific) — skipped
  - Token embedding + output head — Gemma's tied embedding

MVP scope:
  - Weight reshape from GGUF's (d_model, n_heads × d_head_gemma) layout
    into sub-head layout
  - Installation into a GroupedSmall2DTransformer at matching dims
  - Per-layer mode config derived from Gemma4Config's full_attention_layers
  - FORWARD-PASS PARITY with Gemma4Stream on a few tokens (architectural
    gate — if grouped decomposition matches reference Gemma forward,
    we've won)

Not in this MVP: full bit-exact match against llama.cpp serving
(requires Q6_K embeddings + per-layer embedding injection + all norm
details). We validate against Gemma4Stream which is our own PyTorch
reference.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from calm.llm_computer.gemma4_config import Gemma4Config
from calm.llm_computer.grouped_attention import gemma_to_grouped_weights
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)


def substrate_config_for_gemma4(
    gemma_cfg: Gemma4Config,
) -> GroupedSmall2DConfig:
    """Derive a GroupedSmall2DConfig that hosts Gemma 4 E4B in one
    Small2DTransformer tensor.

    All layers use grouped mode (no single-mode layers initially).
    SWA layers: 8 groups × 128 sub-heads = Gemma d_head=256 equivalent.
    Full layers: 8 groups × 256 sub-heads = Gemma d_head=512 equivalent.

    Problem: n_heads on the substrate must equal n_groups × group_size
    PER LAYER. But Small2DConfig has ONE n_heads for the whole model.
    Solution: pick n_heads = max(SWA sub-heads, full sub-heads) = 2048
    for full attention layers. SWA layers then have "unused" sub-head
    slots 1024..2047 (empty, ignored in their grouped computation).

    Actually simpler: choose n_heads based on d_model / 2. If d_model
    is 2560, n_heads = 1280. For SWA that's 1280 / 128 = 10 groups
    (but Gemma only has 8) — so 8 groups × 128 = 1024, 256 slots free.
    For full attention 1280 / 256 = 5 groups — NOT divisible by 8.

    So d_model=2560 can't host full attention as 8 groups of 256 in
    d_head=2. We need d_model = 8 × 256 × 2 = 4096 for full layers.

    Clean option: use d_model=4096 (upsize), or accept that full
    attention layers must have fewer groups. The fully-clean option
    is extending d_model, but that requires random-initializing the
    extra channels (Gemma never trained them).

    For the MVP, we match Gemma's d_model=2560 exactly and only
    implement SWA layers (35 of 42) via grouped mode. Full attention
    layers (7 of 42) stay on standard d_head attention — they'd need
    a different substrate or d_model=4096 to fit perfectly.

    This means our unified tensor hosts ~83% of Gemma's layers via the
    substrate decomposition. The remaining 7 layers are still in one
    state_dict but compute standard d_head=512 attention.

    (Future: extend to d_model=5120 to give both SWA and full enough
    sub-heads. Or go fully d_model=4096 and random-init extra channels
    for SWA layers.)
    """
    # SWA config: n_groups=8, group_size=128 → 1024 sub-heads
    n_sub_heads_swa = gemma_cfg.n_heads * (gemma_cfg.swa_head_dim // 2)
    # We size the substrate for SWA since most layers are SWA
    n_heads_substrate = n_sub_heads_swa  # 1024 sub-heads of d_head=2
    d_model_substrate = n_heads_substrate * 2  # 2048 — less than Gemma's 2560!
    # Actually we keep Gemma's d_model=2560 and pad sub-heads:
    d_model_substrate = gemma_cfg.d_model  # 2560
    # → n_heads = d_model / 2 = 1280
    n_heads_substrate = d_model_substrate // 2

    # Per-layer modes
    layer_modes = []
    layer_n_groups = []
    layer_group_sizes = []
    for i in range(gemma_cfg.n_layers):
        if i in gemma_cfg.full_attention_layers:
            # Full attention: would need group_size=256 for d_head=512.
            # n_groups * group_size must == n_heads=1280. 1280 / 256 = 5,
            # but Gemma has n_heads=8. Incompatible.
            # For this MVP, mark as 'single' and accept suboptimal output
            # — substrate-native handling of full attention at d_model=2560
            # requires either d_model=4096 or a different decomposition.
            layer_modes.append("single")
            layer_n_groups.append(1)
            layer_group_sizes.append(n_heads_substrate)
        else:
            # SWA: n_groups=8, group_size=128 = 1024 sub-heads
            # but n_heads_substrate=1280 (for d_model=2560)
            # Pad: 1280 / 8 = 160 group_size. But Gemma's d_head=256 → 128
            # Mismatch. We need n_heads % 8 == 0 and (n_heads / 8) * 2 == 256
            # i.e. n_heads = 8 * 128 = 1024, d_model = 2048.
            # Conclusion: d_model=2560 doesn't cleanly host SWA either.
            # Use single mode for this MVP.
            layer_modes.append("single")
            layer_n_groups.append(1)
            layer_group_sizes.append(n_heads_substrate)

    return GroupedSmall2DConfig(
        vocab_size=gemma_cfg.vocab_size,
        d_model=d_model_substrate,
        n_heads=n_heads_substrate,
        n_layers=gemma_cfg.n_layers,
        d_ffn=gemma_cfg.d_ffn,
        max_len=gemma_cfg.max_position,
        use_hard_max=False,
        layer_modes=tuple(layer_modes),
        layer_n_groups=tuple(layer_n_groups),
        layer_group_sizes=tuple(layer_group_sizes),
    )


def substrate_config_for_gemma_swa_only(
    gemma_cfg: Gemma4Config,
) -> GroupedSmall2DConfig:
    """Alternative: size the substrate at d_model=2048 which cleanly
    hosts SWA attention (8 × 128 sub-heads × 2 = 2048). Full attention
    layers need different substrate. For demonstrating SWA decomposition
    standalone.

    d_model = 2048, n_heads = 1024, d_head = 2.
    Every layer: grouped mode, n_groups=8, group_size=128.
    Equivalent to 8-head d_head=256 attention — SWA-compatible.

    NOTE: this does NOT match Gemma's d_model=2560 exactly. Loading
    Gemma weights would require either:
      (a) Projecting Gemma's 2560 channels down to 2048 (lossy)
      (b) Only loading the first 2048 channels (discards info)
      (c) Accepting this as a proof of concept, not a Gemma loader.

    Option (c) is what we ship here — proof that the substrate CAN
    host Gemma-equivalent SWA attention in one tensor. Gemma's actual
    d_model=2560 would require a d_model=5120 substrate to fit both
    attention types cleanly; that's a 2× resource cost not worth it
    without a real scaling reason.
    """
    # Use Gemma's n_heads and SWA d_head, but force d_model to fit
    n_sub_heads = gemma_cfg.n_heads * (gemma_cfg.swa_head_dim // 2)
    d_model = n_sub_heads * 2
    layer_modes = tuple(["grouped"] * gemma_cfg.n_layers)
    layer_n_groups = tuple([gemma_cfg.n_heads] * gemma_cfg.n_layers)
    layer_group_sizes = tuple([gemma_cfg.swa_head_dim // 2] * gemma_cfg.n_layers)
    return GroupedSmall2DConfig(
        vocab_size=gemma_cfg.vocab_size,
        d_model=d_model,
        n_heads=n_sub_heads,
        n_layers=gemma_cfg.n_layers,
        d_ffn=gemma_cfg.d_ffn,
        max_len=gemma_cfg.max_position,
        use_hard_max=False,
        layer_modes=layer_modes,
        layer_n_groups=layer_n_groups,
        layer_group_sizes=layer_group_sizes,
    )


def install_gemma_attention_weights_swa(
    substrate: GroupedSmall2DTransformer,
    layer_idx: int,
    q_proj: torch.Tensor,
    k_proj: torch.Tensor,
    v_proj: torch.Tensor,
    o_proj: torch.Tensor,
    n_heads_gemma: int,
    d_head_gemma: int,
) -> None:
    """Install Gemma's Q/K/V/O weights for one SWA layer into the
    substrate via grouped decomposition.

    Gemma stores:
        q_proj: (d_model, n_heads × d_head_gemma)  — or (out, in)? check GGUF
        k_proj, v_proj: (d_model, n_kv_heads × d_head_gemma) — GQA
        o_proj: (n_heads × d_head_gemma, d_model)

    Substrate expects:
        W_qkv[layer]: Linear(d_model, 3 × d_model) — stacks Q, K, V
            (each size d_model = n_heads × d_head_substrate × 2)
        W_out[layer]: Linear(d_model, d_model)

    The reshape:
        Gemma q_proj (d_model, n_heads × d_head_gemma) →
            reshape to (d_model, n_heads × group_size × 2) →
            equivalent to our sub-head layout via gemma_to_grouped_weights

    For GQA: k, v have fewer heads than q. The substrate's W_qkv stores
    Q, K, V each of size d_model. K and V from Gemma need to be
    broadcast (repeated) across n_heads/n_kv_heads to fill d_model.

    This function handles the reshape + assignment for one SWA layer.
    MVP-level — doesn't handle RoPE configuration, norm installation,
    or GQA broadcast (those are separate concerns).
    """
    raise NotImplementedError(
        "Direct single-tensor Gemma loading deferred: Gemma's d_model=2560 "
        "doesn't cleanly divide into the sub-head decomposition needed for "
        "BOTH SWA (group_size=128) and full (group_size=256) layers. Use "
        "substrate_config_for_gemma_swa_only() for SWA-only demonstration "
        "at d_model=2048, or accept the multi-stream UnifiedCHRLMCard "
        "approach which doesn't have this constraint."
    )
