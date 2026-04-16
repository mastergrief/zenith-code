"""Round-22A: compiled adder INSIDE Gemma-like sub-heads — same layer.

The Level 3 proof: "Gemma-like" softmax sub-heads and compiled hard_max
sub-heads coexist in the SAME layer, processed in ONE forward pass with
per-sub-head attention mode partition. No separate layers for cards.

Architecture (d_model=20, n_sub_heads=10, 1 layer):
  Sub-heads 0..4:  "Gemma-like" (random trained weights, softmax attention)
  Sub-heads 5..9:  compiled adder_tiny (hard_max attention)
  Channels  0..9:  Gemma-like residual slice
  Channels 10..19: adder channels (own=10, bias=11, copy_a=12, steps=13..19)

The forward computes Q/K/V from ONE shared W_qkv matrix, then PARTITIONS
by sub-head range: softmax on [0..5), hard_max on [5..10). W_out sums
both partitions' contributions back to the residual. FFN runs once on
the full residual (adder step neurons only fire on adder channels; Gemma
neurons only on Gemma channels — because weights are zero in the other
region).

Tests:
  (a) Adder sub-heads produce correct a+b for all (a,b) ∈ [0,3]² = 16/16.
  (b) Gemma-like channels are IDENTICAL to a baseline where adder
      sub-heads are zeroed — proving compiled sub-heads don't disturb
      the language model's computation.

This is ONE tensor, ONE layer, ONE forward, TWO attention modes.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


# --- Layout constants ---
D_MODEL = 20
N_HEADS = D_MODEL // 2  # 10
N_LAYERS = 1
D_FFN = 24  # Gemma neurons [0, 8) + adder neurons [8, 22)
MAX_LEN = 4
VOCAB = 8
MAX_OPERAND = 3
MAX_SUM = MAX_OPERAND * 2  # 6

# Partition: Gemma-like sub-heads [0, 5), adder [5, 10)
GEMMA_SH_END = 5
ADDER_SH_START = 5

# Channel partition: Gemma [0, 10), adder [10, 20)
GEMMA_CH_END = 10
ADDER_CH_OFF = 10

# Adder channel layout (offset by ADDER_CH_OFF)
CH_OWN = ADDER_CH_OFF + 0   # 10
CH_BIAS = ADDER_CH_OFF + 1   # 11
CH_COPY_A = ADDER_CH_OFF + 2  # 12
CH_STEP_BASE = ADDER_CH_OFF + 3  # 13..19 (7 step channels)


def build_adder_native() -> Small2DTransformer:
    """Compile adder_tiny at its NATIVE d_model=10 (5 sub-heads). The
    installer will place it at the right offset in the d_model=20
    substrate — sub-heads [5, 10), channels [10, 20)."""
    from calm.llm_computer.programs.adder_tiny import build_adder_tiny
    return build_adder_tiny(vocab_size=VOCAB, max_len=MAX_LEN)


def build_gemma_standin() -> Small2DTransformer:
    """Random-init 'Gemma-like' sub-heads at channels [0, 10). Only
    populates its own channel/sub-head rectangle; adder region stays
    zero."""
    cfg = Small2DConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ffn=D_FFN, max_len=MAX_LEN,
        use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.zero_()
        # Populate ONLY Gemma's sub-head rectangle
        # tok: rows [0, VOCAB), cols [0, GEMMA_CH_END)
        m.tok.weight[:, :GEMMA_CH_END].normal_(0, 0.1)
        # pos: cols [0, GEMMA_CH_END)
        m.pos.weight[:, :GEMMA_CH_END].normal_(0, 0.1)
        # W_qkv: Gemma Q rows [0, 2*GEMMA_SH_END), cols [0, GEMMA_CH_END)
        # K at [D_MODEL, D_MODEL + 2*GEMMA_SH_END), V at [2*D_MODEL, ...]
        for seg_off in [0, D_MODEL, 2 * D_MODEL]:
            m.W_qkv[0].weight[
                seg_off:seg_off + 2 * GEMMA_SH_END,
                :GEMMA_CH_END,
            ].normal_(0, 0.1)
        # W_out: rows [0, GEMMA_CH_END), cols [0, 2*GEMMA_SH_END)
        m.W_out[0].weight[:GEMMA_CH_END, :2 * GEMMA_SH_END].normal_(0, 0.1)
        # ff_in: Gemma uses first 8 gate + first 8 val neurons, cols [0, GEMMA_CH_END)
        GEMMA_FFN = 8
        m.ff_in[0].weight[:GEMMA_FFN, :GEMMA_CH_END].normal_(0, 0.1)
        m.ff_in[0].weight[D_FFN:D_FFN + GEMMA_FFN, :GEMMA_CH_END].normal_(0, 0.1)
        # ff_out: rows [0, GEMMA_CH_END), cols [0, GEMMA_FFN)
        m.ff_out[0].weight[:GEMMA_CH_END, :GEMMA_FFN].normal_(0, 0.1)
        # head: rows [0, VOCAB), cols [0, GEMMA_CH_END)
        m.head.weight[:, :GEMMA_CH_END].normal_(0, 0.1)
    return m


def build_substrate_with_adder(gemma: Small2DTransformer,
                               adder: Small2DTransformer) -> Small2DTransformer:
    """Build d_model=20 substrate: copy Gemma stand-in weights directly,
    install adder at (ch_off=10, sh_off=5) via the standard card
    installer. The installer handles sub-head/channel offset mapping."""
    from calm.llm_computer.card_installer import CardSlot, install_compiled_card
    cfg = Small2DConfig(
        vocab_size=VOCAB, d_model=D_MODEL, n_heads=N_HEADS,
        n_layers=N_LAYERS, d_ffn=D_FFN, max_len=MAX_LEN,
        use_hard_max=False,
    )
    substrate = Small2DTransformer(cfg)
    # Start from Gemma's weights (which are zero outside Gemma's rectangle)
    with torch.no_grad():
        for (_, ps), (_, pg) in zip(
            substrate.named_parameters(), gemma.named_parameters(),
        ):
            ps.copy_(pg)
    # Install adder at its offset — this correctly shifts the compiled
    # sub-heads, channels, FFN neurons, and tok/head entries.
    install_compiled_card(substrate, adder, CardSlot(
        ch_off=ADDER_CH_OFF,       # 10
        sh_off=ADDER_SH_START,     # 5
        ffn_off=8,                 # after Gemma's 8 FFN neurons
        tok_off=0,                 # share tok range (adder operands 0..7)
        layer_off=0,               # SAME LAYER as Gemma
    )
    )
    return substrate


def partitioned_forward(model: Small2DTransformer,
                         idx: torch.Tensor) -> torch.Tensor:
    """Forward pass with per-sub-head attention partition:
      sub-heads [0, GEMMA_SH_END): softmax
      sub-heads [ADDER_SH_START, N_HEADS): hard_max (argmax)

    Everything else (embedding, FFN, head) is standard.
    """
    B, S = idx.shape
    cfg = model.config
    pos_idx = torch.arange(S, device=idx.device)
    x = model.tok(idx) + model.pos(pos_idx)
    mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device),
                      diagonal=1)

    for layer in range(cfg.n_layers):
        qkv = model.W_qkv[layer](x)  # (B, S, 3*D)
        qkv = qkv.reshape(B, S, 3, N_HEADS, 2)
        q, k, v = qkv.permute(2, 0, 3, 1, 4)  # (B, H, S, 2)

        # --- Partition attention by sub-head range ---

        # Gemma partition: softmax
        q_g, k_g, v_g = (q[:, :GEMMA_SH_END],
                          k[:, :GEMMA_SH_END],
                          v[:, :GEMMA_SH_END])
        scores_g = torch.einsum("bhid,bhjd->bhij", q_g, k_g)
        scores_g = scores_g.masked_fill(mask, float("-inf"))
        weights_g = F.softmax(scores_g, dim=-1)
        attn_g = torch.einsum("bhij,bhjd->bhid", weights_g, v_g)

        # Adder partition: hard_max (argmax, first-tie)
        q_a, k_a, v_a = (q[:, ADDER_SH_START:],
                          k[:, ADDER_SH_START:],
                          v[:, ADDER_SH_START:])
        scores_a = torch.einsum("bhid,bhjd->bhij", q_a, k_a)
        scores_a = scores_a.masked_fill(mask, float("-inf"))
        idx_a = scores_a.argmax(dim=-1, keepdim=True)
        weights_a = torch.zeros_like(scores_a)
        weights_a.scatter_(-1, idx_a, 1.0)
        attn_a = torch.einsum("bhij,bhjd->bhid", weights_a, v_a)

        # Concat back along head dim
        attn = torch.cat([attn_g, attn_a], dim=1)  # (B, H, S, 2)
        attn = attn.transpose(1, 2).reshape(B, S, D_MODEL)
        x = x + model.W_out[layer](attn)

        # FFN (standard, shared — each partition only fires its neurons)
        gate, val = model.ff_in[layer](x).chunk(2, dim=-1)
        x = x + model.ff_out[layer](F.relu(gate) * val)

    return model.head(x)


if __name__ == "__main__":
    import itertools

    print("[R22A] building Gemma stand-in (random sub-heads 0..4)...")
    gemma = build_gemma_standin()

    print("[R22A] building compiled adder (native d_model=10, 5 sub-heads)...")
    adder = build_adder_native()
    print(f"  adder d_model={adder.config.d_model}, n_heads={adder.config.n_heads}")

    print("[R22A] installing adder at (ch_off=10, sh_off=5) via card_installer...")
    substrate = build_substrate_with_adder(gemma, adder)
    print(f"  d_model={D_MODEL}, n_heads={N_HEADS}, params={substrate.param_count()}")

    # --- CHECK (a): adder sub-heads produce correct sums ---
    print("\n[R22A] CHECK (a) — adder computes correctly at its sub-heads")
    ok = 0
    total = 0
    for a, b in itertools.product(range(MAX_OPERAND + 1), repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            logits = partitioned_forward(substrate, x)
        pred = int(logits[0, 1].argmax().item())
        expected = a + b
        if pred == expected:
            ok += 1
        else:
            print(f"  [✗] {a} + {b}: got {pred}, expected {expected}")
        total += 1
    print(f"  adder: {ok}/{total} — {'PASS' if ok == total else 'FAIL'}")

    # --- CHECK (b): Gemma channels undisturbed by adder sub-heads ---
    print("\n[R22A] CHECK (b) — Gemma residual undisturbed")
    # Baseline: forward with ONLY gemma sub-heads (no adder installed)
    gemma_only = Small2DTransformer(gemma.config)
    with torch.no_grad():
        for (_, ps), (_, pg) in zip(
            gemma_only.named_parameters(), gemma.named_parameters()
        ):
            ps.copy_(pg)

    x_probe = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        # Both use partitioned_forward — in gemma_only, adder sub-heads
        # have zero Q/K/V so hard_max produces zero attention output.
        logits_merged = partitioned_forward(substrate, x_probe)
        logits_baseline = partitioned_forward(gemma_only, x_probe)

    # Compare Gemma's channels (0..9) contribution to residual.
    # Since adder's W_out writes ONLY to channels [10, 20) and Gemma's
    # W_out writes ONLY to channels [0, 10), the Gemma channels should
    # be identical regardless of adder's presence.
    #
    # But logits are head @ residual — head has entries for BOTH ranges.
    # To compare Gemma's portion: look at residual directly.
    # Rerun to get pre-head residual:
    def get_residual(model, idx):
        B, S = idx.shape
        pos_idx = torch.arange(S)
        x = model.tok(idx) + model.pos(pos_idx)
        mask = torch.triu(torch.ones(S, S, dtype=torch.bool), diagonal=1)
        for layer in range(model.config.n_layers):
            qkv = model.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, N_HEADS, 2)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            q_g, k_g, v_g = q[:, :GEMMA_SH_END], k[:, :GEMMA_SH_END], v[:, :GEMMA_SH_END]
            scores_g = torch.einsum("bhid,bhjd->bhij", q_g, k_g)
            scores_g = scores_g.masked_fill(mask, float("-inf"))
            weights_g = F.softmax(scores_g, dim=-1)
            attn_g = torch.einsum("bhij,bhjd->bhid", weights_g, v_g)
            q_a, k_a, v_a = q[:, ADDER_SH_START:], k[:, ADDER_SH_START:], v[:, ADDER_SH_START:]
            scores_a = torch.einsum("bhid,bhjd->bhij", q_a, k_a)
            scores_a = scores_a.masked_fill(mask, float("-inf"))
            idx_a = scores_a.argmax(dim=-1, keepdim=True)
            weights_a = torch.zeros_like(scores_a)
            weights_a.scatter_(-1, idx_a, 1.0)
            attn_a = torch.einsum("bhij,bhjd->bhid", weights_a, v_a)
            attn = torch.cat([attn_g, attn_a], dim=1)
            attn = attn.transpose(1, 2).reshape(B, S, D_MODEL)
            x = x + model.W_out[layer](attn)
            gate, val = model.ff_in[layer](x).chunk(2, dim=-1)
            x = x + model.ff_out[layer](F.relu(gate) * val)
        return x

    with torch.no_grad():
        res_merged = get_residual(substrate, x_probe)
        res_baseline = get_residual(gemma_only, x_probe)

    gemma_ch_merged = res_merged[:, :, :GEMMA_CH_END]
    gemma_ch_baseline = res_baseline[:, :, :GEMMA_CH_END]
    diff = (gemma_ch_merged - gemma_ch_baseline).abs().max().item()
    ok_b = diff < 1e-6
    print(f"  max |gemma_channels_merged - gemma_channels_baseline| = {diff:.2e}")
    print(f"  {'PASS' if ok_b else 'FAIL'} — compiled sub-heads "
          f"{'DO NOT' if ok_b else 'DO'} disturb Gemma's computation")

    # --- Summary ---
    all_ok = (ok == total) and ok_b
    print(f"\n[R22A] OVERALL: {'PASS' if all_ok else 'FAIL'}")
    print(f"[R22A] Level 3 — compiled program INSIDE Gemma's attention layer:")
    print(f"[R22A]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    if all_ok:
        print("[R22A]   one tensor, one layer, one forward, two attention modes,")
        print("[R22A]   zero cross-talk between compiled and trained sub-heads.")
