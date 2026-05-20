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

        # Slice 2: re-apply LeCun init now that copy_gate / copy_q_proj /
        # copy_k_proj exist. Scoped via `_apply_lecun_init_to` per co_lead
        # audit msg 1779305159197 — broad walker would re-init the H bank
        # too when Slice 8 flag is on, scrambling RNG consumption.
        # Bias-preserving — `copy_gate.bias` stays at copy_gate_bias_init.
        if getattr(config, "use_lecun_init", False):
            self._apply_lecun_init_to([
                self.copy_gate, self.copy_q_proj, self.copy_k_proj,
            ])

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
        # v17: expose copy_attn (B, S_target, S_source) for pointer
        # supervision loss that trains WHICH source position to attend
        # to, not just which output char to emit.
        self._last_copy_attn_grad = copy_attn

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

        # Slice 5 / co_lead audit msg 1779303378368: cached decode is a
        # FLAT prefill + flat per-token loop. It does NOT honor
        # `_forward_backbone` control flow — no `_run_l_loop`,
        # `_delta_layer_stack` wrapping, `n_iterations` outer loop,
        # `h_cycles` outer loop, z_init swap, input injection,
        # loop-index embed, prefix_lm mask, or H-boundary RMSNorm.
        #
        # Without a hard guard, a card trained with Slice 1-5 control-flow
        # flags would silently get a degraded forward at facade-install
        # inference time (training-time `forward` ≠ product-path
        # `decode_greedy_cached`). That's the worst kind of regression.
        #
        # Allowlist: `use_gated_attention` is mirrored in BOTH cached-path
        # branches (per Slice 1) and is safe. Everything else in the
        # following blocklist forces NotImplementedError.
        _CACHED_DECODE_BLOCKED = (
            ("h_cycles", lambda v: v > 1),
            ("n_iterations", lambda v: v > 1),
            ("use_input_injection", bool),
            ("use_z_init", bool),
            ("use_loop_index", bool),
            ("use_prefix_lm", bool),
            ("use_softmax_attn", bool),
            ("use_h_rmsnorm", bool),
            # Slice 6: cached decode would need to keep last (k-1) hidden/QKV
            # values per layer in streaming state to mirror the causal conv.
            # Until that's implemented, hard-block the flag.
            ("use_short_conv", bool),
            # Slice 8: H layer stack is only meaningful when h_cycles > 1
            # (already blocked), but block flag-on too as a defensive
            # signal — cached path has no notion of an H weight bank.
            ("use_h_layer_stack", bool),
        )
        _blocked = []
        for _attr, _pred in _CACHED_DECODE_BLOCKED:
            _v = getattr(cfg, _attr, False)
            if _pred(_v):
                _blocked.append(f"{_attr}={_v!r}")
        if _blocked:
            raise NotImplementedError(
                "decode_greedy_cached is a flat-layer-pass codepath that "
                "does NOT honor _forward_backbone control flow. Refusing "
                "to run with non-flat config: " + ", ".join(_blocked) +
                ". Use the full-forward path `model(idx)` for inference, "
                "or extend decode_greedy_cached to mirror _forward_backbone "
                "before installing a card trained with these flags."
            )

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

            # Mirror _delta_layer_stack's gated sequence-mixer output for
            # parity with non-cached forward. Without this, training-time
            # forward (uses _delta_layer_stack with gate) and product-path
            # cached decode (this code) silently diverge under use_gated_attention.
            if self.attn_gate_proj is not None:
                delta_gate = torch.sigmoid(self.attn_gate_proj[layer](x))  # (B, S, D)
                delta_out = delta_gate * delta_out

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

                # Mirror gated sequence-mixer output for parity with non-cached path.
                if self.attn_gate_proj is not None:
                    delta_gate_t = torch.sigmoid(self.attn_gate_proj[layer](new_x))
                    delta_out_t = delta_gate_t * delta_out_t

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

    def _compute_prefix_mask(self, idx: torch.Tensor) -> torch.Tensor:
        """Override Small2DTransformer hook to expose the prefix block via the
        existing sep-token-id mechanism. Returned mask is (B, S) bool — True
        at positions before the first sep, i.e. the prompt/copyable region.

        Only consulted by `_attention` when `config.use_prefix_lm` is True;
        otherwise this hook is never called.
        """
        return self._build_prefix_mask(idx, self.copy_config.sep_token_id)


def build_copy_augmented_delta(
    vocab_size: int = 80, d_model: int = 64, n_heads: int = 32,
    n_layers: int = 4, d_ffn: int = 128, max_len: int = 96,
    n_copy_heads: int = 4, sep_token_id: int = 3,
    use_hard_max: bool = False,
    use_softmax_attn: bool = False,
    copy_gate_bias_init: float = -2.0,
    # HRM-Text-derived flags (default off → existing checkpoints unaffected):
    use_chunkwise: bool = False,
    n_iterations: int = 1,
    use_loop_index: bool = False,
    use_input_injection: bool = False,
    use_gated_attention: bool = False,
    use_z_init: bool = False,
    use_lecun_init: bool = False,
    use_prefix_lm: bool = False,
    h_cycles: int = 1,
    use_h_rmsnorm: bool = False,
    use_short_conv: bool = False,
    use_h_layer_stack: bool = False,
) -> CopyAugmentedDeltaNet:
    """Build a CopyAugmentedDeltaNet mirroring PT's default sizing.

    Optional HRM-Text-derived flags must be plumbed through here so that
    `load_dt_checkpoint` can reconstruct a trained-with-flag model with the
    correct architecture. Without this, a checkpoint with `use_gated_attention=True`
    would silently load as default-off and fail param-shape checks.
    """
    cfg = CopyAugmentedDeltaConfig(
        vocab_size=vocab_size, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads, sep_token_id=sep_token_id,
        use_hard_max=use_hard_max,
        use_delta_net=True, use_softmax_attn=use_softmax_attn,
        copy_gate_bias_init=copy_gate_bias_init,
        use_chunkwise=use_chunkwise,
        n_iterations=n_iterations,
        use_loop_index=use_loop_index,
        use_input_injection=use_input_injection,
        use_gated_attention=use_gated_attention,
        use_z_init=use_z_init,
        use_lecun_init=use_lecun_init,
        use_prefix_lm=use_prefix_lm,
        h_cycles=h_cycles,
        use_h_rmsnorm=use_h_rmsnorm,
        use_short_conv=use_short_conv,
        use_h_layer_stack=use_h_layer_stack,
    )
    assert cfg.d_head == 2, f"d_head must be 2, got {cfg.d_head}"
    return CopyAugmentedDeltaNet(cfg)
