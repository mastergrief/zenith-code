"""
HRM — Hierarchical Reasoning Model.

Latent-space reasoning via nested recurrent loops. L-module handles
fast low-level computation, H-module handles slow high-level planning.
Each H-cycle resets L-module context, enabling deeper reasoning than
standard RNNs.

Architecture:
  Input → Embed → [L(z_L, z_H+x) × L_cycles → H(z_H, z_L)] × H_cycles → Head → Output

Based on: arxiv.org/abs/2506.21734 (Wang et al., 2025)
Built from scratch — no dependency on the original repo.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HRMConfig:
    """Configuration for an HRM model (legacy single-tower + encoder-decoder)."""
    vocab_size: int = 128
    hidden_size: int = 64
    num_heads: int = 4
    expansion: float = 2.67
    L_layers: int = 1
    H_layers: int = 1
    L_cycles: int = 8
    H_cycles: int = 8
    max_seq_len: int = 64
    # Encoder-decoder extras (ignored by legacy HRM)
    decoder_layers: int = 2
    max_dec_len: int = 16
    dropout: float = 0.0
    rms_norm_eps: float = 1e-6

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def param_count(self) -> int:
        """Estimate total parameter count."""
        block_params = 12 * self.hidden_size ** 2  # approx per block
        total_blocks = self.L_layers + self.H_layers
        embed_params = 2 * self.vocab_size * self.hidden_size
        return total_blocks * block_params + embed_params


# --- Components ---

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE)."""

    def __init__(self, dim: int, max_len: int = 512, theta: float = 10000.0):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        t = torch.arange(max_len).float()
        emb = torch.outer(t, freqs)
        self.register_buffer("cos", emb.cos(), persistent=False)
        self.register_buffer("sin", emb.sin(), persistent=False)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.cos[:seq_len], self.sin[:seq_len]


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings to queries/keys."""
    # x: (batch, heads, seq, head_dim)
    d = x.shape[-1]
    x1, x2 = x[..., :d // 2], x[..., d // 2:]
    cos = cos[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    sin = sin[:x.shape[-2]].unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with RoPE.

    `is_causal` is a constructor flag:
      - True  → autoregressive decoder self-attn (each position sees 0..i)
      - False → bidirectional encoder self-attn (each position sees all)
    """

    def __init__(self, hidden_size: int, num_heads: int, is_causal: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.is_causal = is_causal
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x).reshape(B, S, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # 3 × (B, H, S, D)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=self.is_causal)
        out = attn.transpose(1, 2).reshape(B, S, D)
        return self.out(out)


class CrossAttention(nn.Module):
    """Cross-attention: queries from x, keys/values from memory.

    No RoPE — query and key sequences come from different spaces, so
    positional modulation across them is meaningless. Memory dimensions
    equal hidden_size (encoder and decoder share hidden_size).
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.q = nn.Linear(hidden_size, hidden_size, bias=False)
        self.kv = nn.Linear(hidden_size, 2 * hidden_size, bias=False)
        self.out = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        B, S_q, D = x.shape
        S_kv = memory.shape[1]
        q = self.q(x).reshape(B, S_q, self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(memory).reshape(B, S_kv, 2, self.num_heads, self.head_dim)
        k, v = kv.permute(2, 0, 3, 1, 4)  # 2 × (B, H, S_kv, D_head)
        attn = F.scaled_dot_product_attention(q, k, v)
        out = attn.transpose(1, 2).reshape(B, S_q, D)
        return self.out(out)


class SwiGLU(nn.Module):
    """SwiGLU feedforward: gate(x) * up(x), then down."""

    def __init__(self, hidden_size: int, expansion: float = 2.67):
        super().__init__()
        inner = int(hidden_size * expansion)
        self.gate = nn.Linear(hidden_size, inner, bias=False)
        self.up = nn.Linear(hidden_size, inner, bias=False)
        self.down = nn.Linear(inner, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    """Single transformer block: attention + SwiGLU + post-norms."""

    def __init__(self, config: HRMConfig, is_causal: bool = True):
        super().__init__()
        self.attn = Attention(config.hidden_size, config.num_heads, is_causal=is_causal)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        # Post-norm (matches HRM paper)
        x = self.norm1(x + self.attn(x, cos, sin))
        x = self.norm2(x + self.mlp(x))
        return x


class ReasoningModule(nn.Module):
    """Stack of transformer blocks with input injection (additive)."""

    def __init__(self, config: HRMConfig, num_layers: int, is_causal: bool = True):
        super().__init__()
        self.layers = nn.ModuleList([
            TransformerBlock(config, is_causal=is_causal) for _ in range(num_layers)
        ])

    def forward(self, hidden: torch.Tensor, injection: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = hidden + injection  # input injection (additive, like HRM paper)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return x


class DecoderBlock(nn.Module):
    """Decoder block: causal self-attn + cross-attn + SwiGLU + post-norms."""

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.self_attn = Attention(config.hidden_size, config.num_heads, is_causal=True)
        self.cross_attn = CrossAttention(config.hidden_size, config.num_heads)
        self.mlp = SwiGLU(config.hidden_size, config.expansion)
        self.norm1 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm2 = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.norm3 = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x: torch.Tensor, memory: torch.Tensor,
                cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.self_attn(x, cos, sin))
        x = self.norm2(x + self.cross_attn(x, memory))
        x = self.norm3(x + self.mlp(x))
        return x


# --- Main Model ---

class HRM(nn.Module):
    """Hierarchical Reasoning Model.

    Two recurrent modules in nested loops:
      - L-module: fast, low-level computation (runs L_cycles times per H-step)
      - H-module: slow, high-level planning (runs H_cycles times total)
      - z_L updates are conditioned on z_H (H guides L)
      - z_H updates are conditioned on z_L (L informs H)

    Forward pass: no gradient through the recurrent iterations
    (detached — only the final iteration is differentiable).
    This matches the original HRM approach.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config

        # Input/output
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Positional encoding
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)

        # Reasoning modules
        self.L_module = ReasoningModule(config, config.L_layers)
        self.H_module = ReasoningModule(config, config.H_layers)

        # Initial hidden states (learned)
        self.z_H_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass with nested L/H recurrent loops.

        Args:
            input_ids: (batch, seq_len) token indices

        Returns:
            logits: (batch, seq_len, vocab_size)
        """
        B, S = input_ids.shape
        cos, sin = self.rope(S)

        # Input embedding
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)

        # Initialize hidden states (broadcast to batch × seq)
        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        # Nested recurrent loop (detached — no grad through iterations)
        with torch.no_grad():
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    # Skip the very last iteration (will be done with grad below)
                    if h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1:
                        break
                    z_L = self.L_module(z_L, z_H + x, cos, sin)

                if h_step < self.config.H_cycles - 1:
                    z_H = self.H_module(z_H, z_L, cos, sin)

        # Final iteration with gradient (for training)
        z_L = self.L_module(z_L, z_H + x, cos, sin)
        z_H = self.H_module(z_H, z_L, cos, sin)

        # Output
        logits = self.head(z_H)
        return logits

    def param_count(self) -> int:
        """Actual parameter count."""
        return sum(p.numel() for p in self.parameters())


# --- Encoder-Decoder Architecture (Option A from plan) ---

class HRMEncoder(nn.Module):
    """Bidirectional encoder with nested L/H recurrent reasoning.

    This is what the HRM paper was designed for — refine a hidden
    representation over a *full* input via H-cycles × L-cycles iterations.
    Outputs the final z_H as encoder memory for the decoder to cross-attend
    to.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.head_dim, config.max_seq_len)
        self.L_module = ReasoningModule(config, config.L_layers, is_causal=False)
        self.H_module = ReasoningModule(config, config.H_layers, is_causal=False)
        self.z_H_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)
        self.z_L_init = nn.Parameter(torch.randn(config.hidden_size) * 0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return encoder memory (B, S_enc, D) = final z_H."""
        B, S = input_ids.shape
        cos, sin = self.rope(S)
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        z_H = self.z_H_init.unsqueeze(0).unsqueeze(0).expand(B, S, -1)
        z_L = self.z_L_init.unsqueeze(0).unsqueeze(0).expand(B, S, -1)

        with torch.no_grad():
            for h_step in range(self.config.H_cycles):
                for l_step in range(self.config.L_cycles):
                    if h_step == self.config.H_cycles - 1 and l_step == self.config.L_cycles - 1:
                        break
                    z_L = self.L_module(z_L, z_H + x, cos, sin)
                if h_step < self.config.H_cycles - 1:
                    z_H = self.H_module(z_H, z_L, cos, sin)

        z_L = self.L_module(z_L, z_H + x, cos, sin)
        z_H = self.H_module(z_H, z_L, cos, sin)
        return z_H


class HRMDecoder(nn.Module):
    """Causal decoder stack. No recurrence — pure transformer decoder.

    The reasoning happened in the encoder (nested L/H loops); the decoder
    just generates answer tokens autoregressively, cross-attending to
    encoder memory.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.rope = RotaryEmbedding(config.head_dim, max(config.max_dec_len, 16))
        self.blocks = nn.ModuleList([DecoderBlock(config) for _ in range(config.decoder_layers)])
        self.norm_out = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Return logits (B, S_dec, vocab)."""
        S = input_ids.shape[1]
        cos, sin = self.rope(S)
        x = self.embed(input_ids) * math.sqrt(self.config.hidden_size)
        for block in self.blocks:
            x = block(x, memory, cos, sin)
        x = self.norm_out(x)
        return self.head(x)


class HRMSeq2Seq(nn.Module):
    """Encoder-decoder HRM: bidirectional recurrent encoder + causal decoder.

    Training: teacher-force decoder on shifted answer tokens; loss only on
    answer positions.
    Inference: encode prompt once, generate answer autoregressively.
    """

    def __init__(self, config: HRMConfig):
        super().__init__()
        self.config = config
        self.encoder = HRMEncoder(config)
        self.decoder = HRMDecoder(config)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.trunc_normal_(module.weight, std=0.02)

    def forward(self, encoder_ids: torch.Tensor, decoder_ids: torch.Tensor) -> torch.Tensor:
        """Full forward: encoder → decoder → logits (B, S_dec, vocab)."""
        memory = self.encoder(encoder_ids)
        return self.decoder(decoder_ids, memory)

    def encode(self, encoder_ids: torch.Tensor) -> torch.Tensor:
        """Encode only — useful for cached-memory inference."""
        return self.encoder(encoder_ids)

    def decode_step(self, decoder_ids: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        """Decode with cached memory — used in autoregressive inference loop."""
        return self.decoder(decoder_ids, memory)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())
