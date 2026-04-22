"""Copy-augmented DeltaNet — PT backbone swapped for DeltaNet recurrence.

Round 6a hybrid: pointer-copy transducer with DeltaNet sequence-mixing
instead of softmax attention. The copy path (copy_gate + pointer attn
over input positions) is unchanged; only the backbone producing
hidden states `h` is swapped.

Hypothesis (vs Round-5c null): PT's copy mechanism is a parallel path
that bypasses the backbone's generate-side output. When DeltaNet's
hidden states are noisy (as Round 5c showed on random-KV recall), the
copy gate leans on the copy path and downweights the generate path.
For digit-heavy structured outputs (PT's canonical workload), ~70-90%
of emitted tokens are copies — so backbone noise on the generate path
is partially shielded. Whether this is enough to preserve PT's 95-100%
autoregressive accuracy is the binary Round-6a gate.

Architecture — inherits `DeltaNetSmall2DTransformer`, which itself
inherits `Small2DTransformer`. The copy machinery (copy_gate,
copy_q_proj, copy_k_proj) is additive on top of DeltaNet's hidden
states. Forward:

    x = self._forward_backbone(idx)          # DeltaNet recurrence
    gen_logits = self.head(x)                # generate-path logits
    copy_scores over input prefix            # unchanged from PT
    blended = p_copy · copy_dist + (1-p_copy) · gen_probs
    return log(blended + ε)

State dict compatibility: this is NOT a drop-in load from a plain PT
checkpoint — the copy machinery is compatible but the backbone layers
differ (DeltaNet has β_head Linear per layer; softmax PT doesn't). So
a fresh training run is required. Regression tests live against the
existing PT checkpoint separately.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.delta_rule import DeltaNetConfig, DeltaNetSmall2DTransformer


@dataclass
class CopyAugmentedDeltaConfig(DeltaNetConfig):
    """DeltaNetConfig + copy mechanism parameters (mirrors CopyAugmentedConfig)."""
    n_copy_heads: int = 4
    sep_token_id: int = 3
    # Copy-gate bias init: -2.0 (sigmoid 0.12) favors generation at start;
    # 0.0 (sigmoid 0.5) is neutral; +1.0 (sigmoid 0.73) favors copy.
    # For copy-dominant tasks (args literally in prompt), higher init lets
    # the copy path establish before gen path overwhelms. See Round 1
    # diagnostic (gate plateaued at 0.193) and Round 4 finding ("content
    # is wrong, not form" → gen path dominance is the real issue).
    copy_gate_bias_init: float = -2.0


class CopyAugmentedDeltaNet(DeltaNetSmall2DTransformer):
    """DeltaNet backbone + pointer-copy decode mechanism."""

    def __init__(self, config: CopyAugmentedDeltaConfig):
        super().__init__(config)
        self.copy_config = config
        d = config.d_model

        self.copy_gate = nn.Linear(d, 1, bias=True)
        nn.init.constant_(self.copy_gate.bias, config.copy_gate_bias_init)

        copy_dim = config.n_copy_heads * config.d_head
        self.copy_q_proj = nn.Linear(d, copy_dim, bias=False)
        self.copy_k_proj = nn.Linear(d, copy_dim, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns log-probs (B, S, vocab)."""
        B, S = idx.shape
        cfg = self.config

        # DeltaNet backbone produces per-position hidden states.
        x = self._forward_backbone(idx)

        # Generation distribution.
        gen_logits = self.head(x)

        # Copy mechanism (identical to CopyAugmentedTransformer).
        sep_id = self.copy_config.sep_token_id
        prefix_mask = self._build_prefix_mask(idx, sep_id)

        p_copy = torch.sigmoid(self.copy_gate(x))
        # Expose for diagnostics (eval_dt_checkpoint reads this to compute
        # avg copy-gate usage — 0 = pure generation, 1 = pure copy).
        self.last_p_copy = p_copy.detach()

        n_ch = self.copy_config.n_copy_heads
        dh = cfg.d_head
        cq = self.copy_q_proj(x).reshape(B, S, n_ch, dh)
        ck = self.copy_k_proj(x).reshape(B, S, n_ch, dh)

        copy_scores = torch.einsum("bihd,bjhd->bhij", cq, ck)

        causal = torch.triu(
            torch.ones(S, S, dtype=torch.bool, device=idx.device), diagonal=1,
        )
        prefix_block = ~prefix_mask.unsqueeze(1).unsqueeze(1).expand_as(copy_scores)
        copy_scores = copy_scores.masked_fill(causal, float("-inf"))
        copy_scores = copy_scores.masked_fill(prefix_block, float("-inf"))

        copy_scores_avg = copy_scores.mean(dim=1)
        copy_attn = F.softmax(copy_scores_avg, dim=-1)

        copy_logits = torch.zeros_like(gen_logits)
        src_tokens = idx.unsqueeze(1).expand(B, S, S)
        copy_logits.scatter_add_(2, src_tokens, copy_attn)

        gen_probs = F.softmax(gen_logits, dim=-1)
        blended = p_copy * copy_logits + (1 - p_copy) * gen_probs

        # Expose copy-path probability distribution for auxiliary loss
        # (R26). NOT detached — caller may backprop through it.
        self._last_copy_logits_grad = copy_logits

        return torch.log(blended + 1e-10)

    def decode_greedy_cached(
        self,
        prefix_ids: torch.Tensor,
        max_gen: int = 30,
        eos_token: int | None = None,
    ) -> torch.Tensor:
        """Cached autoregressive greedy decode. Returns (B, L_gen) of ids.

        Closes the inference gap vs plain-PT by:
        1. Running the prefix forward once (prefill), saving per-layer
           DeltaNet state + cached copy-K for prefix positions.
        2. Each decode step processes ONE new token using the cached
           state (O(1) per step vs uncached O(L) forward).

        Produces identical output to repeated `forward(idx)` calls up
        to float32 numerical epsilon (both paths use the same delta
        rule; cached just skips the recompute of prior positions).
        """
        cfg = self.config
        copy_cfg = self.copy_config
        device = prefix_ids.device
        B, S = prefix_ids.shape
        assert B == 1, "cached decode supports batch=1 for now"

        sep_id = copy_cfg.sep_token_id
        n_layers = cfg.n_layers
        n_heads = cfg.n_heads
        d_head = cfg.d_head
        d_model = cfg.d_model
        n_ch = copy_cfg.n_copy_heads
        dh_copy = d_head

        # --- Prefill: full forward, capture per-layer state ---
        pos_idx = torch.arange(S, device=device)
        x = self.tok(prefix_ids) + self.pos(pos_idx)

        # Running buffers: prefix x-per-layer-input (needed for copy K later),
        # plus per-layer S_state after processing prefix.
        layer_states: list[torch.Tensor] = []
        # We'll also remember the residual x at EVERY prefix position after ALL
        # layers, for computing copy-K over prefix via copy_k_proj(x).
        for layer in range(n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, n_heads, d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q_flat = q.transpose(1, 2).reshape(B, S, d_model)
            k_flat = k.transpose(1, 2).reshape(B, S, d_model)
            v_flat = v.transpose(1, 2).reshape(B, S, d_model)

            if cfg.use_silu_feat:
                q_feat = F.silu(q_flat)
                k_feat = F.silu(k_flat)
            else:
                q_feat = q_flat
                k_feat = k_flat
            if cfg.use_l2_norm:
                q_feat = F.normalize(q_feat, p=2, dim=-1, eps=1e-6)
                k_feat = F.normalize(k_feat, p=2, dim=-1, eps=1e-6)

            beta = torch.sigmoid(self.beta_head[layer](x))

            S_state = torch.zeros(B, d_model, d_model, device=device, dtype=x.dtype)
            if getattr(cfg, "use_chunkwise", False):
                S_state, delta_out = self._delta_chunkwise(
                    S_state, q_feat, k_feat, v_flat, beta,
                    chunk_size=getattr(cfg, "chunk_size", 32),
                )
            else:
                reads = []
                for t in range(S):
                    S_state, out_t = self._delta_step(
                        S_state, q_feat[:, t, :], k_feat[:, t, :],
                        v_flat[:, t, :], beta[:, t, :],
                    )
                    reads.append(out_t)
                delta_out = torch.stack(reads, dim=1)

            x = x + self.W_out[layer](delta_out)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)

            layer_states.append(S_state)

        # Cache copy-K over the ENTIRE prefix.
        # Only positions before <sep> are copyable; we mask scores later.
        cached_copy_k = self.copy_k_proj(x).reshape(B, S, n_ch, dh_copy)

        # Prefix mask (which prefix positions are eligible to copy from).
        prefix_mask = self._build_prefix_mask(prefix_ids, sep_id)  # (B, S)

        # Prefix token ids — scatter_add target for copy distribution.
        prefix_ids_buf = prefix_ids  # (B, S)

        # --- Phase 1 output: argmax at last prefix position ---
        current_x = x[:, -1:, :]  # (B, 1, D)
        last_id = self._predict_next_token(
            current_x, cached_copy_k, prefix_ids_buf, prefix_mask,
        )

        # --- Decode loop: process one new token at a time ---
        gen_ids: list[int] = []
        if eos_token is not None and int(last_id.item()) == eos_token:
            return torch.tensor([gen_ids], dtype=torch.long, device=device)
        gen_ids.append(int(last_id.item()))

        current_pos = S  # next position index
        for _ in range(max_gen - 1):
            if eos_token is not None and gen_ids[-1] == eos_token:
                gen_ids.pop()
                break

            # Embed last generated token.
            new_id = torch.tensor([[gen_ids[-1]]], dtype=torch.long, device=device)
            pos_t = torch.tensor([current_pos], device=device)
            new_x = self.tok(new_id) + self.pos(pos_t).unsqueeze(0)  # (B, 1, D)
            current_pos += 1

            # One-step through all layers, updating cached S per layer.
            for layer in range(n_layers):
                qkv = self.W_qkv[layer](new_x)
                qkv = qkv.reshape(B, 1, 3, n_heads, d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                q_flat = q.transpose(1, 2).reshape(B, 1, d_model)
                k_flat = k.transpose(1, 2).reshape(B, 1, d_model)
                v_flat = v.transpose(1, 2).reshape(B, 1, d_model)

                if cfg.use_silu_feat:
                    q_feat = F.silu(q_flat)
                    k_feat = F.silu(k_flat)
                else:
                    q_feat = q_flat
                    k_feat = k_flat
                if cfg.use_l2_norm:
                    q_feat = F.normalize(q_feat, p=2, dim=-1, eps=1e-6)
                    k_feat = F.normalize(k_feat, p=2, dim=-1, eps=1e-6)

                beta_t = torch.sigmoid(self.beta_head[layer](new_x))

                S_prev = layer_states[layer]
                S_new, out_t = self._delta_step(
                    S_prev, q_feat[:, 0, :], k_feat[:, 0, :],
                    v_flat[:, 0, :], beta_t[:, 0, :],
                )
                layer_states[layer] = S_new
                delta_out_t = out_t.unsqueeze(1)  # (B, 1, D)

                new_x = new_x + self.W_out[layer](delta_out_t)
                gate, val = self.ff_in[layer](new_x).chunk(2, dim=-1)
                new_x = new_x + self.ff_out[layer](F.relu(gate) * val)

            # Predict next token from new_x using cached copy K.
            # (Decode-phase positions are NOT added to copy-K cache since they're
            # past <sep> — prefix_mask already zeros them out in the uncached path.)
            next_id = self._predict_next_token(
                new_x, cached_copy_k, prefix_ids_buf, prefix_mask,
            )
            gen_ids.append(int(next_id.item()))

        # Strip trailing eos if present.
        if eos_token is not None and gen_ids and gen_ids[-1] == eos_token:
            gen_ids = gen_ids[:-1]
        return torch.tensor([gen_ids], dtype=torch.long, device=device)

    def _predict_next_token(
        self,
        x_last: torch.Tensor,             # (B, 1, D)
        cached_copy_k: torch.Tensor,       # (B, L_prefix, n_copy_heads, d_head)
        prefix_ids: torch.Tensor,          # (B, L_prefix)
        prefix_mask: torch.Tensor,         # (B, L_prefix)
    ) -> torch.Tensor:
        """Compute log-probs at one position using cached copy K; return argmax id."""
        B = x_last.shape[0]
        cfg = self.config
        cfg_copy = self.copy_config

        gen_logits = self.head(x_last)                              # (B, 1, V)
        p_copy = torch.sigmoid(self.copy_gate(x_last))              # (B, 1, 1)

        n_ch = cfg_copy.n_copy_heads
        dh = cfg.d_head
        cq = self.copy_q_proj(x_last).reshape(B, 1, n_ch, dh)       # (B, 1, H, dh)
        # Score: cq @ cached_copy_k^T across L_prefix — per-head-then-avg.
        copy_scores = torch.einsum("bihd,bjhd->bhij", cq, cached_copy_k)
        # Mask non-prefix positions.
        prefix_block = ~prefix_mask.unsqueeze(1).unsqueeze(1).expand_as(copy_scores)
        copy_scores = copy_scores.masked_fill(prefix_block, float("-inf"))
        copy_scores_avg = copy_scores.mean(dim=1)                   # (B, 1, L_prefix)
        copy_attn = F.softmax(copy_scores_avg, dim=-1)              # (B, 1, L_prefix)

        copy_logits = torch.zeros_like(gen_logits)
        src_tokens = prefix_ids.unsqueeze(1)                         # (B, 1, L_prefix)
        copy_logits.scatter_add_(2, src_tokens, copy_attn)

        gen_probs = F.softmax(gen_logits, dim=-1)
        blended = p_copy * copy_logits + (1 - p_copy) * gen_probs
        log_probs = torch.log(blended + 1e-10)

        return log_probs[0, 0].argmax()

    @staticmethod
    def _build_prefix_mask(idx: torch.Tensor, sep_id: int) -> torch.Tensor:
        """Positions before first <sep> marked True (copyable input prefix)."""
        B, S = idx.shape
        is_sep = (idx == sep_id)
        has_sep = is_sep.any(dim=1)
        sep_pos = is_sep.float().argmax(dim=1)
        sep_pos = torch.where(has_sep, sep_pos, torch.tensor(S, device=idx.device))
        positions = torch.arange(S, device=idx.device).unsqueeze(0)
        return positions < sep_pos.unsqueeze(1)


def build_copy_augmented_delta(
    vocab_size: int = 80, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    n_copy_heads: int = 4, sep_token_id: int = 3,
    use_hard_max: bool = False,
    use_softmax_attn: bool = False,
    copy_gate_bias_init: float = -2.0,
) -> CopyAugmentedDeltaNet:
    """Build a CopyAugmentedDeltaNet mirroring PT's default sizing."""
    cfg = CopyAugmentedDeltaConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, sep_token_id=sep_token_id,
        use_hard_max=use_hard_max,
        use_delta_net=True, use_softmax_attn=use_softmax_attn,
        copy_gate_bias_init=copy_gate_bias_init,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return CopyAugmentedDeltaNet(cfg)
