"""DeltaNet subclass of Small2DTransformer — generalized Householder recurrence.

Round 5 of the runtime-weight-addition research track. Succeeds the Round-4
null (fast_weights delta+gate stuck at 10.5-12.2% on n=10 recall) by
implementing DeltaNet per Yang et al. 2024 (NeurIPS, arXiv:2406.06484).

Update rule:

    S_t = S_{t-1} (I - β_t k_t k_t^T) + β_t v_t k_t^T
        = S_{t-1} - β_t (S_{t-1} k_t - v_t) k_t^T      (SGD form)

Key differences from fast_weights.py's `use_delta_rule=True`:

  (1) No multiplicative decay λ. Only the specific k_t direction is
      erased; all other directions are preserved perfectly. The paper's
      ablation: when β_t = 1 and ‖k_t‖ = 1, `I − k_t k_t^T` is a clean
      projection matrix. The Round-4 variant had λ=0.95 which quietly
      decayed ALL stored bindings by ~40% over 10 steps.

  (2) L2 normalization on K and Q. Ensures eigenvalues of
      `I − β_t k_t k_t^T` land in [0, 1]. Paper ablation: L1→L2 alone
      worth +2pp zero-shot and ~3× on FDA.

  (3) SiLU feature map on K and Q (paper: Qin 2022 / Dao-Gu 2024).

  (4) Learned data-dependent β_t = σ(W_β x_t). Write strength is per-
      position and per-sample, not a fixed hyperparameter.

Read-after-write ordering: `o_t = S_t q_t` (state reflects the current
position's write). This is the paper's convention; contrast fast_weights
which uses read-before-write (S_{t-1} q_t). Makes a material difference
at n=query_at_position when the model wants to recall what it JUST wrote.

The recurrent form below is O(L·d²) — fine at L≤64 where this test runs.
The paper's contribution is the chunkwise parallel form (see
RESEARCH/DELTA-RULE/02_Chunkwise_Parallel_Algorithm.md) which matters
at L≥2K; not needed for this round.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.model import Small2DConfig, Small2DTransformer


class _LayerBank:
    """Per-layer weight references for one pass through `_delta_layer_stack`.

    Slice 8: factored so the L stack (existing `self.W_qkv` etc.) and the
    opt-in H stack (`self.H_W_qkv` etc.) can share the same forward code.
    Holds ModuleList references — not Modules themselves — so state-dict
    keys still point at the named submodules of the parent transformer.
    """

    __slots__ = (
        "W_qkv", "W_out", "ff_in", "ff_out", "beta_head",
        "attn_gate_proj", "short_conv_q", "short_conv_k", "short_conv_v",
    )

    def __init__(self, W_qkv, W_out, ff_in, ff_out, beta_head,
                 attn_gate_proj=None,
                 short_conv_q=None, short_conv_k=None, short_conv_v=None):
        self.W_qkv = W_qkv
        self.W_out = W_out
        self.ff_in = ff_in
        self.ff_out = ff_out
        self.beta_head = beta_head
        self.attn_gate_proj = attn_gate_proj
        self.short_conv_q = short_conv_q
        self.short_conv_k = short_conv_k
        self.short_conv_v = short_conv_v


@dataclass
class DeltaNetConfig(Small2DConfig):
    """Small2DConfig + DeltaNet hyperparameters.

    Architecture matches FastWeightConfig so Round 4 baseline vs Round 5
    DeltaNet is apples-to-apples (same d_model, heads, layers, head dim).

    use_softmax_attn=False (default) matches the paper's Figure-2 architecture:
    DeltaNet REPLACES self-attention entirely. When True, softmax attention
    runs in parallel with DeltaNet (Round-5a ablation; regressed at n≥5
    because the optimizer routed through softmax and left DeltaNet
    untrained).
    """
    use_delta_net: bool = True
    use_softmax_attn: bool = False  # paper-canonical: DeltaNet replaces attn
    use_short_conv: bool = False    # paper's short 1D causal depthwise conv applied to
                                     # Q/K/V AFTER the QKV projection and BEFORE feature-map
                                     # + L2-norm (Slice 6, Tier B). kernel_size=4 fixed.
                                     # Adds ~3 × kernel_size × d_model params per layer.
                                     # Cached-decode blocklist includes this flag (per-token
                                     # cached path can't mirror causal conv without keeping
                                     # the last k-1 hidden/QKV values in streaming state).
    use_l2_norm: bool = True        # L2 on K/Q per paper; ablation: try False
    use_silu_feat: bool = True      # SiLU feature map on K/Q per paper
    use_chunkwise: bool = False     # chunkwise parallel form (UT transform, paper §4)
    chunk_size: int = 32            # C in paper — 32 is sweet spot at seq≤128
    n_delta_heads: int = 1          # split delta state into H parallel (D/H, D/H) states
                                     # (H=1 matches single-head baseline bit-for-bit)
    n_iterations: int = 1           # D5 refinement loop — iterate layer stack N times
                                     # per forward pass (ARC-Prize finding: +13pp from 0→1)
    use_loop_index: bool = False    # Add deterministic per-iteration embedding in D5 loop
    # HRM-Text-derived (default off → bit-equivalence preserved):
    use_input_injection: bool = False  # Re-add the initial input residual at every
                                       # D5 iteration boundary (per Sapient HRM-Text
                                       # `core(hidden_states + input_injection)`). Skip
                                       # injection on iteration 0 — x already IS input.
                                       # Only meaningful when n_iterations > 1.
    use_z_init: bool = False           # Replace iteration-0 hidden state with a learned
                                       # per-channel z_init parameter (HRM-Text z_L_init).
                                       # When set, iter 0 starts from z_init.expand(B, S, D)
                                       # instead of (tok+pos) embedding; embedding must be
                                       # re-supplied via use_input_injection (pairing rule).
                                       # When z_init AND injection both on, iter-0 ALSO
                                       # injects (the skip-iter-0 guard releases because
                                       # x no longer "already has the embed"). Only fires
                                       # when n_iterations > 1.
    # use_gated_attention inherits from Small2DConfig. On DT path
    # (use_softmax_attn=False) the gate is applied to delta_out (sequence-mixer
    # output) before adding to residual. On softmax path it gates attn_output.
    # use_lecun_init inherits from Small2DConfig. Re-init applies to all
    # Linear weights AFTER subclass construction adds beta_head / copy_* layers.
    # use_prefix_lm inherits from Small2DConfig. On DT path (delta-only), it's
    # inert — DeltaNet recurrence has no explicit attention mask. On softmax
    # path (use_softmax_attn=True) it relaxes causal mask within the prefix.
    use_h_rmsnorm: bool = False        # Slice 5: per-channel RMSNorm at the H-cycle
                                       # hand-off (`z_H = h_norm(z_L)`) to stabilize
                                       # magnitude when h_cycles × n_iter × use_input_injection
                                       # compounds beyond toy-substrate ceiling. Default off
                                       # → bit-equivalent to Slice 4 hand-off. Fires only when
                                       # h_cycles > 1 (Slice 4 baseline guard preserved).
                                       # Per co_lead audit msg 1779303378368: RMSNorm chosen
                                       # over LayerNorm (no mean centering for register-like
                                       # channels) and over scalar τ (per-channel skew matters).
                                       # CAVEAT: rescales WHOLE residual at H boundary, which
                                       # shifts substrate-card reserved channels. Existing
                                       # recall-card installs (R22 MQAR etc.) on h_cycles=1
                                       # are unaffected; new cards opting into h_cycles>1
                                       # accept normalized residuals at train AND inference.
    use_h_layer_stack: bool = False    # Slice 8 (Tier B): separate H/L weight banks.
                                       # When True AND h_cycles > 1, the H boundary
                                       # runs a FULL H layer stack with its own
                                       # weights (W_qkv, W_out, ff_in, ff_out,
                                       # beta_head, attn_gate_proj, short_conv_q/k/v)
                                       # before the optional RMSNorm:
                                       #   z_H_raw = _run_h_layer_stack(z_L)
                                       #   z_H = h_norm(z_H_raw) if rmsnorm else z_H_raw
                                       # Doubles DT param count when on. Default off →
                                       # Slice 4-5 hand-off (`z_H = z_L` or `h_norm(z_L)`)
                                       # path preserved bit-equivalently.
    h_cycles: int = 1                  # HRM-Text H/L hierarchy: outer-loop cycles.
                                       # The existing `n_iterations` field becomes
                                       # the L-cycles (inner). When h_cycles > 1,
                                       # `_forward_backbone` runs `h_cycles` outer
                                       # iterations of the L loop. Each H cycle
                                       # HANDS OFF L's converged state to the next
                                       # H cycle: z_H ← z_L_final. (Pure hand-off,
                                       # NOT residual add — residual add explodes
                                       # magnitude without a LayerNorm to stabilize.
                                       # Architectural distinction vs flat L is
                                       # preserved because the H boundary skips
                                       # iter-0 injection that flat L would have.)
                                       # h_cycles=1 (default) special-cases to the
                                       # flat Slice 1-3 path — bit-equivalent baseline.


class DeltaNetSmall2DTransformer(Small2DTransformer):
    """Small2DTransformer + per-layer DeltaNet recurrence alongside attention.

    Adds a β_t linear head per layer (d_model + 1 params per layer).
    Runs DeltaNet recurrence in parallel with softmax attention; outputs
    are summed into the residual stream, same pattern as fast_weights.py.
    """

    SHORT_CONV_KERNEL = 4  # paper-canonical kernel size for the short-conv mixer

    def __init__(self, config: DeltaNetConfig):
        super().__init__(config)
        self.beta_head = nn.ModuleList([
            nn.Linear(config.d_model, 1, bias=True)
            for _ in range(config.n_layers)
        ])
        # Initialize β bias to 0 so σ(0) = 0.5 at init — balanced read/write
        # strength before training. Paper initializes differently for LM
        # training; 0.5 works for this short-sequence synthetic.
        for h in self.beta_head:
            with torch.no_grad():
                h.bias.fill_(0.0)

        # Slice 6: per-layer depthwise causal short conv on Q/K/V.
        # Allocated as three ModuleLists (one per Q/K/V) so each axis can
        # learn its own local-mixing kernel. groups=d_model → depthwise
        # (per-channel kernel, no cross-channel mixing — keeps register-
        # like channel semantics intact). bias=False per Mamba/DeltaNet
        # convention. Forward uses `F.pad(x, (k-1, 0))` for left-pad
        # causality, then conv with padding=0 → output length unchanged.
        if getattr(config, "use_short_conv", False):
            k = self.SHORT_CONV_KERNEL
            self.short_conv_q = nn.ModuleList([
                nn.Conv1d(config.d_model, config.d_model,
                          kernel_size=k, padding=0,
                          groups=config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
            self.short_conv_k = nn.ModuleList([
                nn.Conv1d(config.d_model, config.d_model,
                          kernel_size=k, padding=0,
                          groups=config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
            self.short_conv_v = nn.ModuleList([
                nn.Conv1d(config.d_model, config.d_model,
                          kernel_size=k, padding=0,
                          groups=config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
        else:
            self.short_conv_q = None
            self.short_conv_k = None
            self.short_conv_v = None

        # z_init: learned per-channel initial state for the D5 recurrence.
        # When use_z_init=True AND n_iterations>1, iter 0 starts from
        # z_init.expand(B, S, D) instead of the token+position embedding.
        # Small init (std=0.02) keeps gradient stable at step 0.
        #
        # NOTE: init through a LOCAL generator so the global RNG stream is
        # not disturbed. Without this, downstream subclass parameters
        # (e.g. CopyAugmentedDeltaNet.copy_gate / copy_q_proj / copy_k_proj
        # constructed after super().__init__() returns) would get different
        # seeded values vs the use_z_init=False case — which would break
        # n_iters=1 bit-equivalence tests where the flag is meant to be
        # invisible to forward.
        if getattr(config, "use_z_init", False):
            self.z_init = nn.Parameter(torch.zeros(config.d_model))
            _gen = torch.Generator(device="cpu")
            _gen.manual_seed(0x5A0B2D1F)  # arbitrary fixed seed for z_init
            with torch.no_grad():
                init_vals = torch.empty_like(self.z_init.data, device="cpu")
                init_vals.normal_(mean=0.0, std=0.02, generator=_gen)
                self.z_init.data.copy_(init_vals)
        else:
            self.z_init = None

        # Slice 2: re-apply LeCun init AFTER subclass construction so the
        # newly-added beta_head Linears (and any further subclass layers
        # added via super().__init__() chains) get LeCun-normal too.
        # Bias-preserving — beta_head.bias stays at 0 (from above) and
        # subclass biases like copy_gate_bias_init remain intact.
        if getattr(config, "use_lecun_init", False):
            self._apply_lecun_init()

        # Slice 3: backprop-warmup runtime knob. When set to k < n_iters,
        # only the LAST k iterations of the D5 loop are differentiable;
        # earlier iters run under torch.no_grad() and their output enters
        # the gradient-tracked region as a detached tensor. None = full
        # gradient flow (default — backward unchanged). Set via the
        # context manager `model.bp_warmup_ctx(k)`. NOT persisted to
        # checkpoints — it's a training-step-dependent runtime concern.
        self._bp_warmup_active_iters: int | None = None

        # Slice 5: H-boundary RMSNorm. Single shared module — applied at
        # the H-cycle hand-off in `_forward_backbone` when both flag and
        # h_cycles > 1 conditions are met. Stays None when flag off →
        # zero overhead, zero state-dict drift for Slice 1-4 checkpoints.
        if getattr(config, "use_h_rmsnorm", False):
            self.h_norm = nn.RMSNorm(config.d_model)
        else:
            self.h_norm = None

        # Slice 8: separate H/L weight banks. When `use_h_layer_stack=True`,
        # allocate parallel ModuleLists mirroring the L stack (same shapes
        # so `_delta_layer_stack` can run unchanged code against either
        # bank). Doubles DT param count when on. Zero state-dict drift
        # when off (modules simply not created → not present in state_dict).
        #
        # RNG isolation: save/restore global torch.Generator state around
        # H-bank allocation so the downstream subclass-constructed modules
        # (CopyAugmentedDeltaNet.copy_gate / copy_q_proj / copy_k_proj
        # after super().__init__() returns) get the SAME seeded values
        # whether this flag is on or off. Without this, the n_iters=1
        # bit-equivalence guard at h_cycles=1 (which never invokes the H
        # bank) would silently break because downstream params would
        # diverge between flag-on and flag-off builds at the same seed.
        # Same hazard pattern caught in Slice 2 (z_init local generator).
        if getattr(config, "use_h_layer_stack", False):
            _saved_rng = torch.get_rng_state()
            self.H_W_qkv = nn.ModuleList([
                nn.Linear(config.d_model, 3 * config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
            self.H_W_out = nn.ModuleList([
                nn.Linear(config.d_model, config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
            self.H_ff_in = nn.ModuleList([
                nn.Linear(config.d_model, 2 * config.d_ffn, bias=False)
                for _ in range(config.n_layers)
            ])
            self.H_ff_out = nn.ModuleList([
                nn.Linear(config.d_ffn, config.d_model, bias=False)
                for _ in range(config.n_layers)
            ])
            self.H_beta_head = nn.ModuleList([
                nn.Linear(config.d_model, 1, bias=True)
                for _ in range(config.n_layers)
            ])
            for h in self.H_beta_head:
                with torch.no_grad():
                    h.bias.fill_(0.0)
            # Slice 1 + Slice 6 per-layer modules mirror on the H bank if their
            # respective flags are also on. Otherwise None (forward gated).
            if getattr(config, "use_gated_attention", False):
                self.H_attn_gate_proj = nn.ModuleList([
                    nn.Linear(config.d_model, config.d_model, bias=False)
                    for _ in range(config.n_layers)
                ])
            else:
                self.H_attn_gate_proj = None
            if getattr(config, "use_short_conv", False):
                k = self.SHORT_CONV_KERNEL
                self.H_short_conv_q = nn.ModuleList([
                    nn.Conv1d(config.d_model, config.d_model,
                              kernel_size=k, padding=0,
                              groups=config.d_model, bias=False)
                    for _ in range(config.n_layers)
                ])
                self.H_short_conv_k = nn.ModuleList([
                    nn.Conv1d(config.d_model, config.d_model,
                              kernel_size=k, padding=0,
                              groups=config.d_model, bias=False)
                    for _ in range(config.n_layers)
                ])
                self.H_short_conv_v = nn.ModuleList([
                    nn.Conv1d(config.d_model, config.d_model,
                              kernel_size=k, padding=0,
                              groups=config.d_model, bias=False)
                    for _ in range(config.n_layers)
                ])
            else:
                self.H_short_conv_q = None
                self.H_short_conv_k = None
                self.H_short_conv_v = None
            # Restore RNG so downstream subclass-constructed modules get the
            # same seeded values as the flag-off case.
            torch.set_rng_state(_saved_rng)
        else:
            self.H_W_qkv = None
            self.H_W_out = None
            self.H_ff_in = None
            self.H_ff_out = None
            self.H_beta_head = None
            self.H_attn_gate_proj = None
            self.H_short_conv_q = None
            self.H_short_conv_k = None
            self.H_short_conv_v = None

        # Build the lazy bank handles. These are plain Python objects holding
        # references to the ModuleList submodules — NOT registered modules
        # themselves, so they don't double-count parameters or pollute
        # state_dict keys.
        self._l_bank = _LayerBank(
            W_qkv=self.W_qkv, W_out=self.W_out,
            ff_in=self.ff_in, ff_out=self.ff_out,
            beta_head=self.beta_head,
            attn_gate_proj=self.attn_gate_proj,
            short_conv_q=self.short_conv_q,
            short_conv_k=self.short_conv_k,
            short_conv_v=self.short_conv_v,
        )
        if getattr(config, "use_h_layer_stack", False):
            self._h_bank = _LayerBank(
                W_qkv=self.H_W_qkv, W_out=self.H_W_out,
                ff_in=self.H_ff_in, ff_out=self.H_ff_out,
                beta_head=self.H_beta_head,
                attn_gate_proj=self.H_attn_gate_proj,
                short_conv_q=self.H_short_conv_q,
                short_conv_k=self.H_short_conv_k,
                short_conv_v=self.H_short_conv_v,
            )
        else:
            self._h_bank = None

        # Slice 8: re-apply LeCun init AFTER H bank allocation so the H
        # Linear weights ALSO get LeCun-normal when both flags are on.
        # Slice 2's lecun-init pass earlier in DeltaNet.__init__ ran
        # BEFORE the H bank existed, so H weights would have default init
        # without this second pass. Bias-preserving as always (beta_head
        # bias stays at 0; copy_gate.bias in CopyAugmentedDeltaNet stays
        # at -2.0).
        if getattr(config, "use_lecun_init", False) and self._h_bank is not None:
            self._apply_lecun_init()

    @staticmethod
    def _delta_step(
        S: torch.Tensor,
        q_t: torch.Tensor,
        k_t: torch.Tensor,
        v_t: torch.Tensor,
        beta_t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """One DeltaNet update.

        Args:
            S: (B, D, D) hidden state before the write.
            q_t, k_t, v_t: (B, D) — k_t and q_t assumed L2-normalized
                and SiLU-mapped. v_t is raw (no feature map).
            beta_t: (B, 1) in [0, 1] — data-dependent write strength.

        Returns:
            S_new: (B, D, D)
            out: (B, D) — `S_new @ q_t` (read-after-write).
        """
        # Householder / SGD form: S_new = S - β·(S@k - v) k^T
        v_old = torch.einsum("bij,bj->bi", S, k_t)           # (B, D)
        delta = (v_old - v_t) * beta_t                        # (B, D)
        update = torch.einsum("bi,bj->bij", delta, k_t)       # (B, D, D)
        S_new = S - update
        out = torch.einsum("bij,bj->bi", S_new, q_t)
        return S_new, out

    @staticmethod
    def _delta_chunkwise(
        S: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Chunkwise parallel delta-rule (paper §3-4 — UT transform).

        Args:
            S:     (B, D, D) initial state (typically zeros).
            Q/K/V: (B, L, D) per-position queries, keys, values.
                   K/Q are assumed L2-normalized + SiLU (caller applies).
            beta:  (B, L, 1) per-position write strength in (0, 1).
            chunk_size: C — number of tokens processed in parallel per
                   chunk. Trade-off: bigger C = more parallelism, more
                   SRAM per chunk. 32 is sweet spot at L≤128; bump to
                   64 or 128 for longer sequences.

        Returns:
            S_final: (B, D, D) final state after all positions processed.
            reads:   (B, L, D) per-position read-after-write outputs,
                     matches `_delta_step` one-position-at-a-time output
                     to float32 numerical epsilon.

        Math (for chunk [t] of size C starting from prior state S):
            K_c K_c^T                                  (C, C)
            A = I + tril(diag(β) K K^T, -1)            (C, C) lower-tri
            T = A^-1 · diag(β)                          (C, C) tri solve
            W = T K,  U = T V                          (C, D) each
            U' = U - W S^T                              (C, D) — prior-state adjusted
            O = Q S^T + (Q K^T ⊙ M_causal) U'          (C, D) output
            S_next = S + U'^T K                        (D, D) state update
        """
        B, L, D = Q.shape
        device = Q.device
        dtype = Q.dtype

        # Precompute the strict-lower-triangular mask once (outside chunk loop).
        # For the causal Q K^T mask we want lower-triangular INCLUDING diagonal
        # (position r reads its own k_r after write at position r).
        C = chunk_size

        reads_chunks = []
        for start in range(0, L, C):
            end = min(start + C, L)
            Cc = end - start

            Q_c = Q[:, start:end, :]                        # (B, Cc, D)
            K_c = K[:, start:end, :]                        # (B, Cc, D)
            V_c = V[:, start:end, :]                        # (B, Cc, D)
            beta_c = beta[:, start:end, :].squeeze(-1)      # (B, Cc)

            # K K^T intra-chunk Gram matrix.
            Kkt = torch.matmul(K_c, K_c.transpose(-2, -1))  # (B, Cc, Cc)

            # A = I + tril(diag(β) K K^T, -1)
            # Row r of (diag(β) K K^T) is β_r · K_r K^T. Strict lower tri only.
            eye_Cc = torch.eye(Cc, device=device, dtype=dtype).unsqueeze(0).expand(B, Cc, Cc)
            strict_tril = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=-1,
            )
            A_mat = eye_Cc + (beta_c.unsqueeze(-1) * Kkt) * strict_tril

            # T · A = diag(β)  →  T = A^-1 · diag(β)
            # Using torch.linalg.solve_triangular: solves A X = RHS for X.
            rhs = torch.diag_embed(beta_c)                  # (B, Cc, Cc)
            T_mat = torch.linalg.solve_triangular(A_mat, rhs, upper=False)

            W_c = torch.matmul(T_mat, K_c)                  # (B, Cc, D)
            U_c = torch.matmul(T_mat, V_c)                  # (B, Cc, D)

            # Prior-state adjustment. S^T is (B, D, D).
            S_t = S.transpose(-2, -1)                        # (B, D, D)
            U_prime = U_c - torch.matmul(W_c, S_t)           # (B, Cc, D)

            # Intra-chunk output with causal (lower-tri incl. diagonal) mask.
            causal_incl_diag = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=0,
            )
            Qkt = torch.matmul(Q_c, K_c.transpose(-2, -1))   # (B, Cc, Cc)
            Qkt_masked = Qkt * causal_incl_diag
            O_c = (
                torch.matmul(Q_c, S_t)                        # prior-state reads
                + torch.matmul(Qkt_masked, U_prime)           # intra-chunk reads
            )                                                 # (B, Cc, D)
            reads_chunks.append(O_c)

            # State update: S_next = S + (U')^T K
            S = S + torch.matmul(U_prime.transpose(-2, -1), K_c)  # (B, D, D)

        reads = torch.cat(reads_chunks, dim=1)               # (B, L, D)
        return S, reads

    @staticmethod
    def _delta_chunkwise_multihead(
        S: torch.Tensor,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        beta: torch.Tensor,
        chunk_size: int = 32,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Multi-head chunkwise delta rule.

        Same math as `_delta_chunkwise`, with a leading head dim that
        broadcasts through all the matmuls. H heads each maintain their
        own (D_h, D_h) state; aggregate storage is H·D_h² = D·D_h
        (smaller than single-head's D² for D_h < D).

        Args:
            S:     (B, H, D_h, D_h)
            Q/K/V: (B, H, L, D_h)
            beta:  (B, L, 1) — shared across heads, reshaped internally to (B, 1, L)
            chunk_size: same as single-head.

        Returns:
            S_final: (B, H, D_h, D_h)
            reads:   (B, H, L, D_h)
        """
        B, H, L, D_h = Q.shape
        device = Q.device
        dtype = Q.dtype
        C = chunk_size

        reads_chunks = []
        for start in range(0, L, C):
            end = min(start + C, L)
            Cc = end - start

            Q_c = Q[:, :, start:end, :]                             # (B, H, Cc, D_h)
            K_c = K[:, :, start:end, :]                             # (B, H, Cc, D_h)
            V_c = V[:, :, start:end, :]                             # (B, H, Cc, D_h)
            # beta shared across heads — broadcast to (B, 1, Cc, 1).
            beta_c_full = beta[:, start:end, :]                     # (B, Cc, 1)
            beta_c = beta_c_full.squeeze(-1).unsqueeze(1)           # (B, 1, Cc)

            Kkt = torch.matmul(K_c, K_c.transpose(-2, -1))          # (B, H, Cc, Cc)

            eye_Cc = torch.eye(Cc, device=device, dtype=dtype)
            eye_Cc = eye_Cc.view(1, 1, Cc, Cc).expand(B, H, Cc, Cc)
            strict_tril = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=-1,
            )
            A_mat = eye_Cc + (beta_c.unsqueeze(-1) * Kkt) * strict_tril

            # diag(β) broadcast to (B, 1, Cc, Cc) then expanded to (B, H, ...)
            diag_beta = torch.diag_embed(beta_c.expand(B, H, Cc))   # (B, H, Cc, Cc)
            T_mat = torch.linalg.solve_triangular(A_mat, diag_beta, upper=False)

            W_c = torch.matmul(T_mat, K_c)                          # (B, H, Cc, D_h)
            U_c = torch.matmul(T_mat, V_c)                          # (B, H, Cc, D_h)

            S_t = S.transpose(-2, -1)                                # (B, H, D_h, D_h)
            U_prime = U_c - torch.matmul(W_c, S_t)                   # (B, H, Cc, D_h)

            causal_incl_diag = torch.tril(
                torch.ones(Cc, Cc, device=device, dtype=dtype), diagonal=0,
            )
            Qkt = torch.matmul(Q_c, K_c.transpose(-2, -1))           # (B, H, Cc, Cc)
            Qkt_masked = Qkt * causal_incl_diag
            O_c = (
                torch.matmul(Q_c, S_t)
                + torch.matmul(Qkt_masked, U_prime)
            )                                                         # (B, H, Cc, D_h)
            reads_chunks.append(O_c)

            S = S + torch.matmul(U_prime.transpose(-2, -1), K_c)     # (B, H, D_h, D_h)

        reads = torch.cat(reads_chunks, dim=2)                       # (B, H, L, D_h)
        return S, reads

    @staticmethod
    def _apply_short_conv(x: torch.Tensor, conv: nn.Conv1d) -> torch.Tensor:
        """Slice 6: apply a causal depthwise short conv to (B, S, D) → (B, S, D).

        Left-pads by kernel_size-1 (causality: position t sees [t-k+1, t]),
        runs Conv1d on (B, D, S), transposes back. Output length matches
        input. Per-channel mixing only (Conv1d with groups=D).
        """
        k = conv.kernel_size[0]
        x_t = x.transpose(1, 2)            # (B, D, S)
        x_pad = F.pad(x_t, (k - 1, 0))     # causal left-pad
        out = conv(x_pad)                   # (B, D, S)
        return out.transpose(1, 2)         # (B, S, D)

    def bp_warmup_ctx(self, active_iters: int | None):
        """Context manager that sets `_bp_warmup_active_iters` for the scope
        and restores it on exit. Use during training to ramp the number of
        differentiable D5 iterations:

            with model.bp_warmup_ctx(k):
                out = model(idx)
                loss = ...
                loss.backward()
        """
        outer = self

        class _Ctx:
            def __enter__(self_inner):
                self_inner._saved = outer._bp_warmup_active_iters
                outer._bp_warmup_active_iters = active_iters
                return outer

            def __exit__(self_inner, exc_type, exc_val, exc_tb):
                outer._bp_warmup_active_iters = self_inner._saved
                return False

        return _Ctx()

    @staticmethod
    def _loop_index_embedding(
        iteration: int,
        d_model: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Sinusoidal depth embedding for differentiating D5 iterations."""
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            / d_model
        )
        angles = float(iteration) * freqs
        emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=0)
        if emb.numel() < d_model:
            emb = F.pad(emb, (0, d_model - emb.numel()))
        return emb[:d_model].to(dtype=dtype)

    def _forward_backbone(self, idx: torch.Tensor) -> torch.Tensor:
        """Run the DeltaNet backbone; return pre-head hidden states x (B, S, D).

        Factored out so subclasses (e.g. CopyAugmentedDeltaNet) can use the
        DeltaNet recurrence without the final vocab projection. Has the same
        use_delta_net=False fallback as forward().
        """
        if not getattr(self.config, "use_delta_net", True):
            # Vanilla path — replicate Small2DTransformer.forward minus the head
            B, S = idx.shape
            cfg = self.config
            pos_idx = torch.arange(S, device=idx.device)
            x = self.tok(idx) + self.pos(pos_idx)
            prefix_mask_v = (
                self._compute_prefix_mask(idx)
                if getattr(cfg, "use_prefix_lm", False)
                else None
            )
            for layer in range(cfg.n_layers):
                qkv = self.W_qkv[layer](x)
                qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
                q, k, v = qkv.permute(2, 0, 3, 1, 4)
                attn = self._attention(
                    q, k, v, hard_max=cfg.use_hard_max,
                    prefix_mask=prefix_mask_v,
                )
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
                x = x + self.W_out[layer](attn)
                gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
                x = x + self.ff_out[layer](F.relu(gate) * val)
            return x

        B, S = idx.shape
        cfg = self.config
        pos_idx = torch.arange(S, device=idx.device)
        x_embed = self.tok(idx) + self.pos(pos_idx)

        # D5 refinement loop: iterate the full layer stack n_iterations times
        # per forward pass, sharing weights across iterations. Each iteration
        # rebuilds DeltaNet state from scratch; the residual stream x carries
        # over between iterations. n_iterations=1 matches baseline exactly
        # (the outer loop runs once and the inner body is bit-identical).
        n_iters = max(1, getattr(cfg, "n_iterations", 1))
        use_loop_index = bool(getattr(cfg, "use_loop_index", False)) and n_iters > 1
        # HRM-Text-derived input injection: re-anchor the original input at every
        # iteration boundary so deep recurrence doesn't dilute the input signal.
        # Only meaningful when n_iters > 1 (otherwise injection is a no-op).
        use_input_injection = (
            bool(getattr(cfg, "use_input_injection", False)) and n_iters > 1
        )
        # z_init swap: iter-0 hidden state is the learned z_init parameter
        # (broadcast over (B, S)) instead of the embedding. Only fires when
        # n_iters > 1 — preserves n=1 bit-equivalence baseline. When z_init
        # IS active, iter-0 also needs the embedding via injection (the
        # skip-iter-0 guard releases since x no longer "already is" embed).
        use_z_init = (
            bool(getattr(cfg, "use_z_init", False))
            and self.z_init is not None
            and n_iters > 1
        )
        # Slice 3: prefix_lm mask. Inert on delta-only DT path (no softmax
        # attention to mask); _delta_layer_stack only consults it when
        # cfg.use_softmax_attn is True. Computed once per forward.
        prefix_mask = (
            self._compute_prefix_mask(idx)
            if getattr(cfg, "use_prefix_lm", False)
            else None
        )
        # Slice 3: bp warmup. Default None = full gradient flow.
        bp_active = self._bp_warmup_active_iters
        if bp_active is None or bp_active >= n_iters:
            detach_until = 0  # All iters differentiable.
        else:
            bp_active = max(1, bp_active)
            detach_until = n_iters - bp_active  # iters [0, detach_until) are no_grad

        input_residual = x_embed if use_input_injection else None

        # Slice 4: H/L hierarchy. h_cycles=1 (default) takes the flat-loop
        # path → bit-equivalent to Slice 1-3. h_cycles > 1 wraps the L loop
        # in an outer H loop with a residual skip add between H cycles.
        h_cycles = max(1, getattr(cfg, "h_cycles", 1))

        def _run_l_loop(x_in: torch.Tensor) -> torch.Tensor:
            """Run l_cycles (= n_iters) inner iterations starting from x_in.
            All existing Slice 1-3 flags apply at L-iter granularity."""
            x_l = x_in
            for iteration in range(n_iters):
                # Inject the embedding when:
                #   (a) injection flag is on AND iteration > 0, OR
                #   (b) injection flag is on AND z_init replaced x at iter 0 (need
                #       to recover the embed signal because z_init alone has none).
                inject_now = use_input_injection and (iteration > 0 or use_z_init)
                if inject_now:
                    x_l = x_l + input_residual
                if use_loop_index:
                    loop_emb = self._loop_index_embedding(
                        iteration, cfg.d_model, device=x_l.device, dtype=x_l.dtype,
                    )
                    x_l = x_l + loop_emb.view(1, 1, cfg.d_model)
                if iteration < detach_until:
                    with torch.no_grad():
                        x_l = self._delta_layer_stack(x_l, cfg, B, S, prefix_mask=prefix_mask)
                    x_l = x_l.detach()
                else:
                    x_l = self._delta_layer_stack(x_l, cfg, B, S, prefix_mask=prefix_mask)
            return x_l

        if use_z_init:
            x = self.z_init.view(1, 1, cfg.d_model).expand(B, S, cfg.d_model)
        else:
            x = x_embed

        if h_cycles == 1:
            # Bit-equivalent baseline path: run L loop once, return directly.
            # No residual add; preserves Slice 1-3 behavior exactly.
            return _run_l_loop(x)

        # H/L hierarchy: z_H carries across H cycles; each H cycle resets
        # z_L from z_H and runs n_iters L iters; H hands off L's converged
        # state via `z_H = z_L_final`. (See class docstring on why this
        # is NOT a residual add — magnitude stability without LayerNorm.)
        #
        # Slice 5: when `use_h_rmsnorm=True`, the hand-off is normalized:
        # `z_H = h_norm(z_L)`. This is the gate that unlocks the full
        # `h_cycles × n_iters × use_input_injection` regime which would
        # otherwise NaN on toy d_model=8 (Slice 4 explicit punt). Flag-off
        # behavior is bit-identical to Slice 4 (no h_norm call).
        use_h_norm = (
            bool(getattr(cfg, "use_h_rmsnorm", False))
            and self.h_norm is not None
        )
        # Slice 8: full H layer stack at the hand-off when flag on.
        # Seam migrates from `z_H = h_norm(z_L)` to
        # `z_H = h_norm(H_stack(z_L))` — H bank actually transforms the
        # carry, not just normalizes it. RMSNorm wraps the H stack output.
        use_h_stack = (
            bool(getattr(cfg, "use_h_layer_stack", False))
            and self._h_bank is not None
        )
        z_H = x
        for _h_step in range(h_cycles):
            z_L = z_H
            z_L = _run_l_loop(z_L)
            if use_h_stack:
                z_H_raw = self._run_h_layer_stack(
                    z_L, cfg, B, S, prefix_mask=prefix_mask,
                )
            else:
                z_H_raw = z_L
            z_H = self.h_norm(z_H_raw) if use_h_norm else z_H_raw
        return z_H

    def _delta_layer_stack(self, x: torch.Tensor, cfg, B: int, S: int,
                           prefix_mask: torch.Tensor | None = None,
                           bank: _LayerBank | None = None) -> torch.Tensor:
        """One pass through all layers. Extracted for D5 refinement-loop reuse.

        Slice 8: accepts a `bank` arg for separate H/L weight banks. When
        bank is None, uses `self._l_bank` (the original modules). The H
        path (`_run_h_layer_stack`) passes `self._h_bank`. The forward body
        below references `bank.W_qkv[layer]` etc. instead of `self.W_qkv[layer]`
        so the same code works for both stacks.

        `prefix_mask` is forwarded to the softmax-attention call when
        `cfg.use_softmax_attn=True`. On delta-only DT path it has no effect.
        """
        if bank is None:
            bank = self._l_bank
        for layer in range(cfg.n_layers):
            qkv = bank.W_qkv[layer](x)                         # (B, S, 3D)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)               # 3 × (B, H, S, Dh)

            if cfg.use_softmax_attn:
                attn = self._attention(
                    q, k, v, hard_max=cfg.use_hard_max,
                    prefix_mask=prefix_mask,
                )
                attn = attn.transpose(1, 2).reshape(B, S, cfg.d_model)
            else:
                attn = None

            # Flatten heads back to (B, S, D) for the DeltaNet recurrence,
            # which treats the whole d_model vector as one "head" (the
            # sub-head invariant lives in the attention path only).
            q_flat = q.transpose(1, 2).reshape(B, S, cfg.d_model)
            k_flat = k.transpose(1, 2).reshape(B, S, cfg.d_model)
            v_flat = v.transpose(1, 2).reshape(B, S, cfg.d_model)

            # Slice 6: paper-canonical short causal depthwise conv on Q/K/V
            # AFTER projection, BEFORE feature-map + L2-norm. Applied only
            # to the DeltaNet-side q_flat/k_flat/v_flat — the softmax
            # attention path (when use_softmax_attn=True) uses the
            # head-shaped q/k/v directly above and is NOT short-conv'd.
            # Per-channel local mixing (groups=d_model) — no cross-channel.
            if bank.short_conv_q is not None:
                q_flat = self._apply_short_conv(q_flat, bank.short_conv_q[layer])
                k_flat = self._apply_short_conv(k_flat, bank.short_conv_k[layer])
                v_flat = self._apply_short_conv(v_flat, bank.short_conv_v[layer])

            # Paper feature-map + L2-norm on K and Q. V untouched.
            # Both ablatable — the paper's d_head=128 regime may not transfer
            # to the substrate's d_head=2 invariant.
            if cfg.use_silu_feat:
                q_feat = F.silu(q_flat)
                k_feat = F.silu(k_flat)
            else:
                q_feat = q_flat
                k_feat = k_flat
            if cfg.use_l2_norm:
                q_feat = F.normalize(q_feat, p=2, dim=-1, eps=1e-6)
                k_feat = F.normalize(k_feat, p=2, dim=-1, eps=1e-6)

            # Per-position learned β_t ∈ (0, 1).
            beta = torch.sigmoid(bank.beta_head[layer](x))     # (B, S, 1)

            n_dh = getattr(cfg, "n_delta_heads", 1)
            if n_dh > 1 and getattr(cfg, "use_chunkwise", False):
                # Multi-head path — reshape q/k/v/S per n_delta_heads, run mh chunkwise.
                assert cfg.d_model % n_dh == 0, (
                    f"d_model ({cfg.d_model}) must be divisible by n_delta_heads ({n_dh})"
                )
                D_h = cfg.d_model // n_dh
                q_mh = q_feat.reshape(B, S, n_dh, D_h).transpose(1, 2)  # (B, H, S, D_h)
                k_mh = k_feat.reshape(B, S, n_dh, D_h).transpose(1, 2)
                v_mh = v_flat.reshape(B, S, n_dh, D_h).transpose(1, 2)
                S_mh = torch.zeros(
                    B, n_dh, D_h, D_h, device=x.device, dtype=x.dtype,
                )
                S_mh, delta_out_mh = self._delta_chunkwise_multihead(
                    S_mh, q_mh, k_mh, v_mh, beta,
                    chunk_size=getattr(cfg, "chunk_size", 32),
                )
                # (B, H, S, D_h) → (B, S, H, D_h) → (B, S, D)
                delta_out = delta_out_mh.transpose(1, 2).reshape(B, S, cfg.d_model)
                S_state = S_mh  # unused after this — we don't read it out
            elif getattr(cfg, "use_chunkwise", False):
                S_state = torch.zeros(
                    B, cfg.d_model, cfg.d_model,
                    device=x.device, dtype=x.dtype,
                )
                S_state, delta_out = self._delta_chunkwise(
                    S_state, q_feat, k_feat, v_flat, beta,
                    chunk_size=getattr(cfg, "chunk_size", 32),
                )
            else:
                S_state = torch.zeros(
                    B, cfg.d_model, cfg.d_model,
                    device=x.device, dtype=x.dtype,
                )
                reads = []
                for t in range(S):
                    S_state, out_t = self._delta_step(
                        S_state,
                        q_feat[:, t, :],
                        k_feat[:, t, :],
                        v_flat[:, t, :],
                        beta[:, t, :],
                    )
                    reads.append(out_t)
                delta_out = torch.stack(reads, dim=1)          # (B, S, D)

            # HRM-Text-derived gated sequence-mixer output. On DT canonical
            # path (use_softmax_attn=False, attn=None) the gate is applied to
            # delta_out before W_out. On softmax+DT (use_softmax_attn=True),
            # the softmax attn output is already gated inside _attention via
            # Small2DTransformer's parent path; we still apply the delta gate
            # here so both mixers get gated.
            if bank.attn_gate_proj is not None:
                delta_gate = torch.sigmoid(bank.attn_gate_proj[layer](x))  # (B, S, D)
                gated_delta = delta_gate * delta_out
            else:
                gated_delta = delta_out

            if attn is not None:
                x = x + bank.W_out[layer](attn) + bank.W_out[layer](gated_delta)
            else:
                x = x + bank.W_out[layer](gated_delta)
            gate, val = bank.ff_in[layer](x).chunk(2, dim=-1)
            x = x + bank.ff_out[layer](F.relu(gate) * val)

        return x

    def _run_h_layer_stack(self, z_L: torch.Tensor, cfg, B: int, S: int,
                           prefix_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Slice 8: one pass through the H weight bank. Used at H-cycle
        hand-off when `use_h_layer_stack=True` (and `_h_bank is not None`).
        Calls `_delta_layer_stack` with `bank=self._h_bank` — same forward
        code, different weights.
        """
        return self._delta_layer_stack(
            z_L, cfg, B, S, prefix_mask=prefix_mask, bank=self._h_bank,
        )

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (B, S). Returns logits (B, S, vocab)."""
        return self.head(self._forward_backbone(idx))
