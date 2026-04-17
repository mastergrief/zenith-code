"""HubInjectionCard — L23 H1/H4 forced-attention hub as a reusable facade.

Promotes R43's inline ForcedAttentionOutput into a class. One install
per substrate covers arithmetic (via R28-adjacent content routing),
SV agreement (R42), comparison (R43a), and counting (R43b) — the
empirical 4-for-1 hub validated at 32/34 argmax matches across three
capabilities in R43.

Mechanism (Rounds 38-43):
- L23 H1 and H4 are a shared content-carrier pair. H4 reads the
  primary content token (subject / a-operand / winner), H1 reads the
  secondary (distractor / b-operand / loser).
- Gemma's own Q projection routes attention per-task — no hand-
  coded task dispatch. The facade just forces attention to be
  one-hot at the position Gemma itself would attend strongest to.
- Per-forward two-phase:
    (1) custom loop stops at L23, captures attn_q input, computes
        softmax over L23's own K/Q to find each head's argmax
        position (owns-own KV for L23 since 23 < n_layer_kv_from_start
        = 24);
    (2) gemma.forward runs with layer.attn_output wrapped so H1/H4
        slices of the last query position are overwritten with the
        V-cache value at the argmax position. Other heads pass through.

attn_output input layout for a global d_head=512 layer:
  (B, S, 8 * 512 = 4096), heads concatenated along last dim.
  H1 slice: cols [512, 1024). H4 slice: cols [2048, 2560).
  KV grouping (GQA 8Q/2KV): H1 → KV group 0, H4 → KV group 1.

Usage:

    card = HubInjectionCard()
    card.install(gemma)
    logits = card.forward(token_ids)          # with injection
    logits = card.forward(token_ids, inject=False)  # clean baseline
    card.detach()
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F

from calm.llm_computer.gemma_substrate import (
    GemmaSubstrate,
    KVCache,
    _apply_rope,
    _rms_norm,
)


class _AttnOutputWrapper:
    """Wraps layer.attn_output to force H1/H4 slices at the last query
    position to the V-cache value at specified positions. Other heads
    and earlier query positions pass through unchanged.

    Reads V from kv_cache.v_cache[kv_src_layer]. For L23 (owns-own KV)
    kv_src_layer = 23; for SWA layers that share an earlier layer's KV
    the caller supplies the correct source.
    """

    def __init__(self, inner, kv_cache: KVCache, kv_src_layer: int,
                 h1_slice: slice, h4_slice: slice,
                 kv_group_h1: int, kv_group_h4: int,
                 pos_h1: int, pos_h4: int):
        self.inner = inner
        self.kv_cache = kv_cache
        self.kv_src_layer = kv_src_layer
        self.h1_slice = h1_slice
        self.h4_slice = h4_slice
        self.kv_group_h1 = kv_group_h1
        self.kv_group_h4 = kv_group_h4
        self.pos_h1 = pos_h1
        self.pos_h4 = pos_h4
        # Forward attrs the substrate / Triton path might introspect.
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        v_cache = self.kv_cache.v_cache[self.kv_src_layer]
        v_h1 = v_cache[0, self.kv_group_h1, self.pos_h1, :].to(
            x.dtype).to(x.device)
        v_h4 = v_cache[0, self.kv_group_h4, self.pos_h4, :].to(
            x.dtype).to(x.device)
        x_mod = x.clone()
        x_mod[0, -1, self.h1_slice] = v_h1
        x_mod[0, -1, self.h4_slice] = v_h4
        return self.inner(x_mod)


class _InputCapture:
    """Transparent shim around attn_q that records its last input."""

    def __init__(self, inner):
        self.inner = inner
        self.captured: Optional[torch.Tensor] = None
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        self.captured = x.detach().clone()
        return self.inner(x)


class HubInjectionCard:
    """L23 H1/H4 hub — forced one-hot attention at natural top positions.

    Parameters mirror R43's empirical setup by default (L23, heads 1 & 4,
    d_head=512). Heads and slices are parameterised so this class can
    be re-used if a future round localises another hub at different
    indices.
    """

    DEFAULT_TARGET_LAYER = 23
    DEFAULT_HEADS = (1, 4)
    DEFAULT_D_HEAD = 512      # L23 is global; d_head=512
    DEFAULT_N_HEADS_Q = 8
    DEFAULT_N_HEADS_KV = 2    # Gemma 4 E4B GQA

    def __init__(
        self,
        target_layer: int = DEFAULT_TARGET_LAYER,
        heads: tuple[int, int] = DEFAULT_HEADS,
        d_head: int = DEFAULT_D_HEAD,
        n_heads_q: int = DEFAULT_N_HEADS_Q,
        n_heads_kv: int = DEFAULT_N_HEADS_KV,
    ):
        self.target_layer = target_layer
        self.heads = heads
        self.d_head = d_head
        self.n_heads_q = n_heads_q
        self.n_heads_kv = n_heads_kv
        h1, h4 = heads
        self.h1_slice = slice(h1 * d_head, (h1 + 1) * d_head)
        self.h4_slice = slice(h4 * d_head, (h4 + 1) * d_head)
        heads_per_group = n_heads_q // n_heads_kv
        self.kv_group_h1 = h1 // heads_per_group
        self.kv_group_h4 = h4 // heads_per_group
        self._installed_on: Optional[GemmaSubstrate] = None

    # --- Public API ---

    def install(self, gemma: GemmaSubstrate) -> None:
        """Register with a GemmaSubstrate. Currently stores the
        back-reference; injection happens per-forward via .forward()."""
        if self._installed_on is not None:
            raise RuntimeError(
                f"HubInjectionCard already installed on "
                f"{self._installed_on!r}; detach() first")
        self._installed_on = gemma

    def detach(self) -> None:
        self._installed_on = None

    def natural_positions(self, token_ids: torch.Tensor) -> dict[int, int]:
        """Return {head_idx: top_attended_position} at target_layer for
        the given prompt. Positions identify Gemma's own argmax over
        attention weights at the last query position."""
        if self._installed_on is None:
            raise RuntimeError("install() first")
        return self._compute_natural_positions(token_ids)

    def forward(
        self,
        token_ids: torch.Tensor,
        *,
        inject: bool = True,
        positions: Optional[dict[int, int]] = None,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Run gemma.forward with optional L23 H1/H4 injection.

        Args:
            token_ids: (B=1, S) prompt token ids on target device
            inject: if False, runs gemma.forward unmodified (baseline)
            positions: override the natural-top detection. Dict
                {head_idx: position}. If None and inject=True, positions
                are computed from Gemma's own Q/K pattern.
            device: forward device (matches gemma.forward)
        """
        if self._installed_on is None:
            raise RuntimeError("install() first")
        if not inject:
            return self._baseline_forward(token_ids, device=device)

        if positions is None:
            positions = self._compute_natural_positions(token_ids)
        pos_h1 = int(positions[self.heads[0]])
        pos_h4 = int(positions[self.heads[1]])
        return self._forced_forward(token_ids, pos_h1, pos_h4,
                                    device=device)

    # --- Internals ---

    def _baseline_forward(self, token_ids, *, device):
        m = self._installed_on
        cache = KVCache(m.config.n_layers, device=device)
        return m.forward(token_ids, device=device, kv_cache=cache,
                          start_pos=0)

    def _forced_forward(self, token_ids, pos_h1, pos_h4, *, device):
        m = self._installed_on
        cache = KVCache(m.config.n_layers, device=device)
        target = m.layers[self.target_layer]
        saved = target.attn_output
        target.attn_output = _AttnOutputWrapper(
            saved, cache, self.target_layer,
            h1_slice=self.h1_slice, h4_slice=self.h4_slice,
            kv_group_h1=self.kv_group_h1, kv_group_h4=self.kv_group_h4,
            pos_h1=pos_h1, pos_h4=pos_h4,
        )
        try:
            return m.forward(token_ids, device=device, kv_cache=cache,
                              start_pos=0)
        finally:
            target.attn_output = saved

    def _compute_natural_positions(
        self, token_ids: torch.Tensor,
    ) -> dict[int, int]:
        """Run forward up to target_layer, capture attn_q input, replay
        the Q/K dot-product to find each head's argmax position at the
        last query position. Mirrors R42/R43 get_natural_top_positions.
        """
        m = self._installed_on
        cfg = m.config
        device = token_ids.device
        B, S = token_ids.shape
        assert B == 1, "HubInjectionCard currently supports B=1 only"
        cache = KVCache(cfg.n_layers, device=str(device))

        # Embedding + per-layer-embed (same math as gemma.forward).
        h = m.token_embd[token_ids].to(device) * math.sqrt(cfg.d_model)
        m._per_layer_embd = None
        if m.per_layer_token_embd is not None:
            pl_embd = (m.per_layer_token_embd[token_ids]
                       * math.sqrt(cfg.d_per_layer))
            pl_embd = pl_embd.reshape(
                B, S, cfg.n_layers, cfg.d_per_layer)
            if m.per_layer_model_proj is not None:
                h_proj = h @ m.per_layer_model_proj
                h_proj = h_proj * (1.0 / math.sqrt(cfg.d_model))
                h_proj = h_proj.reshape(
                    B, S, cfg.n_layers, cfg.d_per_layer)
                if m.per_layer_proj_norm_w is not None:
                    h_proj = _rms_norm(
                        h_proj, m.per_layer_proj_norm_w,
                        cfg.rms_norm_eps)
                pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
            m._per_layer_embd = [
                pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

        # Run layers up to and including target, capturing attn_q input.
        target = m.layers[self.target_layer]
        cap = _InputCapture(target.attn_q)
        target.attn_q = cap
        try:
            with torch.no_grad():
                for i, layer in enumerate(m.layers):
                    h = m._forward_layer(
                        h, layer, i, kv_cache=cache, start_pos=0)
                    if i == self.target_layer:
                        break
        finally:
            target.attn_q = cap.inner

        # Replay Q/K at target layer to derive attention weights.
        with torch.no_grad():
            x_attn = cap.captured
            q_raw = cap.inner(x_attn)
            d_head_q = q_raw.shape[-1] // cfg.n_heads_q
            q = q_raw.reshape(1, S, cfg.n_heads_q, d_head_q).transpose(
                1, 2)
            if target.attn_q_norm_w is not None:
                q = _rms_norm(
                    q, target.attn_q_norm_w, cfg.rms_norm_eps)
            is_global = d_head_q > cfg.d_head
            freqs = (m.rope_freqs_global if is_global
                     else m.rope_freqs_swa)
            q = _apply_rope(q, freqs[:S])

            kv_src = cfg.kv_source_layer(
                self.target_layer, is_swa=not is_global)
            if kv_src == self.target_layer:
                # Owns-own KV — recompute K from the captured attn_q
                # input (= post-attn-norm residual). This is identical
                # to the K the layer wrote into the cache, minus RoPE.
                k_raw = target.attn_k(x_attn)
                d_head_kv = k_raw.shape[-1] // cfg.n_heads_kv
                k_new = k_raw.reshape(
                    1, S, cfg.n_heads_kv, d_head_kv).transpose(1, 2)
                if target.attn_k_norm_w is not None:
                    k_new = _rms_norm(
                        k_new, target.attn_k_norm_w, cfg.rms_norm_eps)
                k = _apply_rope(k_new, freqs[:S])
            else:
                k = cache.k_cache[kv_src].float()[..., :S, :]
            if cfg.n_heads_kv < cfg.n_heads_q:
                repeat = cfg.n_heads_q // cfg.n_heads_kv
                k = k.repeat_interleave(repeat, dim=1)

            positions: dict[int, int] = {}
            for head in self.heads:
                q_h = q[0, head, -1, :]
                k_h = k[0, head, :, :]
                scores = (q_h.unsqueeze(0) @ k_h.T).squeeze(0)
                weights = F.softmax(scores, dim=-1)
                positions[head] = int(weights.argmax().item())
        return positions
