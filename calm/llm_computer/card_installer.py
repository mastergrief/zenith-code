"""Install a compiled Small2DTransformer card into a GroupedSmall2DTransformer
substrate via weight corner-patching.

A compiled "card" — `compiled_router`, `adder`, `dispatched`, etc. — is a
Small2DTransformer whose weights were populated by `compile.py` from a
gate-graph IR. Its d_model, n_heads, d_ffn, n_layers are much smaller
than the substrate's.

Installation places:
  - card's Q/K/V rows for sub-heads `[0, H_c)` at substrate sub-head
    range `[sh_off, sh_off + H_c)` — each sub-head occupies 2 rows of
    the QKV segment (d_head=2).
  - card's W_out rows at substrate channels `[ch_off, ch_off + D_c)`,
    cols at substrate sub-head range (same offset mapping).
  - card's FFN neurons `[0, d_ffn_c)` at substrate FFN slots
    `[ffn_off, ffn_off + d_ffn_c)`.
  - card's tok/pos embedding entries: restricted to the card's vocab
    range `[0, V_c)` writing into substrate channels `[ch_off, ch_off + D_c)`.
  - card's head entries: restricted to card's vocab slots, reading
    from substrate channels.

The substrate is expected to already be zero-initialized (compiled
programs rely on unreferenced weights being exactly zero). Gemma or
other cards can coexist so long as their channel / sub-head / FFN
allocations do not overlap.

Per-layer `hard_max` must be enabled on the substrate for each layer
the card occupies (compiled cards rely on argmax attention, not softmax).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.model import Small2DTransformer


@dataclass
class CardSlot:
    """Where to install a compiled card within the substrate."""
    # Channel range in substrate residual [ch_off, ch_off + card.d_model)
    ch_off: int
    # Sub-head range in substrate [sh_off, sh_off + card.n_heads)
    sh_off: int
    # FFN slot range in substrate [ffn_off, ffn_off + card.d_ffn)
    ffn_off: int
    # Token vocab range in substrate tok/head/pos embeddings for this card.
    # Card tokens [0, vocab_c) map to substrate token ids
    # [tok_off, tok_off + vocab_c). Default 0 = share top of vocab.
    tok_off: int = 0
    # Substrate layer offset. Card's layer l maps to substrate's layer
    # `layer_off + l`. Used when the substrate hosts Gemma on layers
    # [0, N_G) (grouped / softmax) and a compiled card on layers
    # [N_G, N_G + N_C) (single / hard_max) — the two attention modes
    # cannot share a layer, so separate allocations keep them disjoint.
    layer_off: int = 0


def install_compiled_card(
    substrate: GroupedSmall2DTransformer,
    card: Small2DTransformer,
    slot: CardSlot,
) -> None:
    """Corner-patch a compiled card's weights into the substrate.

    Preconditions (caller's responsibility):
      - substrate weights are zero-initialized OR all other occupants'
        allocations are disjoint from this card's slot.
      - substrate n_layers >= card.n_layers.
      - substrate.config.layer_hard_max[l] is True for all l in range(card.n_layers).
      - slot.ch_off + card.d_model <= substrate.d_model.
      - slot.sh_off + card.n_heads <= substrate.n_heads.
      - slot.ffn_off + card.d_ffn <= substrate.d_ffn.
      - slot.tok_off + card.vocab_size <= substrate.vocab_size.
    """
    s_cfg = substrate.config
    c_cfg = card.config
    D_s = s_cfg.d_model
    D_c = c_cfg.d_model
    H_c = c_cfg.n_heads
    F_c = c_cfg.d_ffn
    V_c = c_cfg.vocab_size

    ch_off = slot.ch_off
    sh_off = slot.sh_off
    ffn_off = slot.ffn_off
    tok_off = slot.tok_off
    layer_off = slot.layer_off

    # Bounds — fail fast with a readable message.
    assert layer_off + c_cfg.n_layers <= s_cfg.n_layers, (
        f"card needs {c_cfg.n_layers} layers starting at {layer_off}, "
        f"substrate has {s_cfg.n_layers}"
    )
    assert c_cfg.d_head == s_cfg.d_head == 2, (
        f"d_head mismatch card={c_cfg.d_head} substrate={s_cfg.d_head}"
    )
    assert ch_off + D_c <= D_s
    assert sh_off + H_c <= s_cfg.n_heads
    assert ffn_off + F_c <= s_cfg.d_ffn
    assert tok_off + V_c <= s_cfg.vocab_size

    with torch.no_grad():
        # tok: card.tok.weight (V_c, D_c) → substrate.tok.weight
        # rows [tok_off, tok_off + V_c), cols [ch_off, ch_off + D_c).
        substrate.tok.weight[
            tok_off : tok_off + V_c, ch_off : ch_off + D_c,
        ] = card.tok.weight

        # pos: card.pos.weight (max_len_c, D_c) → substrate.pos.weight
        # rows [0, max_len_c), cols [ch_off, ch_off + D_c).
        # Positions share across cards (same positional signal); writes are
        # additive across channels so as long as different cards touch
        # different channel ranges, they coexist.
        pos_rows = min(c_cfg.max_len, s_cfg.max_len)
        substrate.pos.weight[
            :pos_rows, ch_off : ch_off + D_c,
        ] = card.pos.weight[:pos_rows]

        for l in range(c_cfg.n_layers):
            s_l = layer_off + l
            # W_qkv: card (3*D_c, D_c) → substrate (3*D_s, D_s)
            # Q segment: rows [0, D_c) of card → rows [2*sh_off, 2*sh_off + 2*H_c)
            #            = [2*sh_off, 2*sh_off + D_c) of substrate
            #            (since D_c = 2 * H_c)
            # K segment: card rows [D_c, 2*D_c) → substrate rows
            #            [D_s + 2*sh_off, D_s + 2*sh_off + D_c)
            # V segment: card rows [2*D_c, 3*D_c) → substrate rows
            #            [2*D_s + 2*sh_off, 2*D_s + 2*sh_off + D_c)
            assert D_c == 2 * H_c, (
                f"card d_model {D_c} != 2 * n_heads {H_c}"
            )
            qkv_s = substrate.W_qkv[s_l].weight
            qkv_c = card.W_qkv[l].weight
            q_row_s = 2 * sh_off
            k_row_s = D_s + 2 * sh_off
            v_row_s = 2 * D_s + 2 * sh_off
            qkv_s[q_row_s : q_row_s + D_c, ch_off : ch_off + D_c] = \
                qkv_c[0:D_c, :]
            qkv_s[k_row_s : k_row_s + D_c, ch_off : ch_off + D_c] = \
                qkv_c[D_c : 2 * D_c, :]
            qkv_s[v_row_s : v_row_s + D_c, ch_off : ch_off + D_c] = \
                qkv_c[2 * D_c : 3 * D_c, :]

            # W_out: card (D_c, D_c) → substrate (D_s, D_s)
            # rows [ch_off, ch_off + D_c), cols [2*sh_off, 2*sh_off + D_c)
            substrate.W_out[s_l].weight[
                ch_off : ch_off + D_c, 2 * sh_off : 2 * sh_off + D_c,
            ] = card.W_out[l].weight

            # ff_in: card (2*F_c, D_c) → substrate (2*F_s, D_s)
            # Substrate packs gate [0, F_s) then val [F_s, 2*F_s).
            # Card packs gate [0, F_c) then val [F_c, 2*F_c).
            F_s = s_cfg.d_ffn
            ff_in_s = substrate.ff_in[s_l].weight
            ff_in_c = card.ff_in[l].weight
            # Gate region
            ff_in_s[ffn_off : ffn_off + F_c, ch_off : ch_off + D_c] = \
                ff_in_c[0:F_c, :]
            # Val region
            ff_in_s[F_s + ffn_off : F_s + ffn_off + F_c,
                    ch_off : ch_off + D_c] = ff_in_c[F_c : 2 * F_c, :]

            # ff_out: card (D_c, F_c) → substrate (D_s, F_s)
            # rows [ch_off, ch_off + D_c), cols [ffn_off, ffn_off + F_c)
            substrate.ff_out[s_l].weight[
                ch_off : ch_off + D_c, ffn_off : ffn_off + F_c,
            ] = card.ff_out[l].weight

        # head: card (V_c, D_c) → substrate (V_s, D_s)
        # rows [tok_off, tok_off + V_c), cols [ch_off, ch_off + D_c)
        substrate.head.weight[
            tok_off : tok_off + V_c, ch_off : ch_off + D_c,
        ] = card.head.weight


def build_card_hosting_substrate(
    card: Small2DTransformer,
    *,
    extra_channels: int = 0,
    extra_sub_heads: int = 0,
    extra_ffn: int = 0,
    extra_vocab: int = 0,
) -> tuple[GroupedSmall2DTransformer, CardSlot]:
    """Convenience: build a zero-init substrate sized to host `card` plus
    some padding, with hard_max on every layer the card uses.

    Returns (substrate, slot) where slot installs the card at offset (0, 0, 0, 0).
    Extra room stays zero — further cards can be installed into it.
    """
    c = card.config
    d_model = c.d_model + extra_channels
    n_heads = c.n_heads + extra_sub_heads
    d_ffn = c.d_ffn + extra_ffn
    vocab = c.vocab_size + extra_vocab
    # d_model must equal 2 * n_heads (d_head=2 invariant)
    if d_model != 2 * n_heads:
        # Prefer to grow d_model to match even sub-head count
        d_model = 2 * n_heads
    sub_cfg = GroupedSmall2DConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=c.n_layers,
        d_ffn=d_ffn,
        max_len=c.max_len,
        use_hard_max=False,
        layer_modes=tuple(["single"] * c.n_layers),
        layer_hard_max=tuple([True] * c.n_layers),
    )
    substrate = GroupedSmall2DTransformer(sub_cfg)
    with torch.no_grad():
        for p in substrate.parameters():
            p.zero_()
    slot = CardSlot(ch_off=0, sh_off=0, ffn_off=0, tok_off=0)
    install_compiled_card(substrate, card, slot)
    return substrate, slot
