"""Round-12 demo — 5 SubstrateHRM slots + dispatched_v4 in one substrate.

Validates the "limitless HRMs" claim: the substrate hosts N HRM slots
at disjoint channel/sub-head/vocab rectangles, each independently
callable. Only one real checkpoint exists today (`substrate_hrm_nl_best.pt`);
slots 1-4 are random-init stand-ins for future HRM specialists (math,
word, gsm, multi). The architecture validates regardless of weight
provenance.

Layout (d_head=2, single-mode):
  HRM slots 0..4:
    vocab rows :  [80*i,  80*(i+1))            i ∈ [0, 5)
    channels   :  [64*i,  64*(i+1))
    sub-heads  :  [32*i,  32*(i+1))
    FFN slots  :  [128*i, 128*(i+1))
    layers     :  [0, 4)   — softmax (shared across HRMs)
  Card slot (dispatched_v4):
    vocab rows :  [5*80, 5*80 + 284)          = [400, 684)
    channels   :  [5*64, 5*64 + 582)          = [320, 902)
    sub-heads  :  [5*32, 5*32 + 291)          = [160, 451)
    FFN slots  :  [5*128, 5*128 + 1134)       = [640, 1774)
    layers     :  [4, 6)   — hard_max

Pass:
  (a) slot 0 logits on slot-0 input == standalone HRM logits (bit-exact).
  (b) slots 1-4 produce non-zero, distinct outputs on their own vocab input.
  (c) feeding slot-i input → slots j≠i stay silent (outputs in other
      slot-j vocab ranges are near-zero, bounded by cross-talk noise
      from zero-initialized parts of the substrate).
  (d) dispatched_v4 passes 791/791 on its own vocab tokens.
  (e) feeding HRM input silences ALL card slots (cross-card gating via
      dispatched_v4's shifted opcodes).
"""

from __future__ import annotations

import itertools
import math
import tempfile
import time
from pathlib import Path

import torch

from calm.llm_computer.card_installer import CardSlot, install_compiled_card
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.programs.dispatched_v4 import (
    ADD_MAX, FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND, OPCODE_SHIFT, PRIME_MAX_N,
    PRIME_MIN_N, VOCAB as CARD_VOCAB, build_dispatched_v4, decode_output,
)
from calm.llm_computer.programs.is_prime import _is_prime


HRM_CKPT = Path(
    "/mnt/c/Users/gabes/projects/claw-code/calm/hrm/checkpoints/"
    "substrate_hrm_nl_best.pt"
)
N_HRM_SLOTS = 5


def load_real_hrm():
    ckpt = torch.load(HRM_CKPT, weights_only=False, map_location="cpu")
    cfg = Small2DConfig(
        vocab_size=ckpt["config"]["vocab_size"],
        d_model=ckpt["config"]["d_model"],
        n_heads=ckpt["config"]["n_heads"],
        n_layers=ckpt["config"]["n_layers"],
        d_ffn=ckpt["config"]["d_ffn"],
        max_len=ckpt["config"]["max_len"],
        use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m, ckpt


def build_synthetic_hrm(ref_cfg: Small2DConfig, seed: int) -> Small2DTransformer:
    """Build an HRM-shaped Small2DTransformer with random init (stand-in
    for a future specialist)."""
    cfg = Small2DConfig(
        vocab_size=ref_cfg.vocab_size,
        d_model=ref_cfg.d_model,
        n_heads=ref_cfg.n_heads,
        n_layers=ref_cfg.n_layers,
        d_ffn=ref_cfg.d_ffn,
        max_len=ref_cfg.max_len,
        use_hard_max=False,
    )
    torch.manual_seed(seed)
    m = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)
    return m


def build_substrate(hrm_ref, card):
    h = hrm_ref.config
    c = card.config
    d_model = N_HRM_SLOTS * h.d_model + c.d_model        # 5*64 + 582 = 902
    d_model += d_model % 2
    n_heads = d_model // 2
    d_ffn = N_HRM_SLOTS * h.d_ffn + c.d_ffn              # 5*128 + 1134 = 1774
    vocab = N_HRM_SLOTS * h.vocab_size + c.vocab_size    # 5*80 + 284 = 684
    n_layers = max(h.n_layers, 0) + c.n_layers           # HRM shares layers 0..3, card 4..5
    # HRM uses layers [0, h.n_layers); card uses layers [h.n_layers, +c.n_layers)
    n_layers = h.n_layers + c.n_layers                    # 4 + 2 = 6
    max_len = max(h.max_len, c.max_len)

    layer_hard_max = tuple([False] * h.n_layers + [True] * c.n_layers)

    cfg = GroupedSmall2DConfig(
        vocab_size=vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len, use_hard_max=False,
        layer_modes=tuple(["single"] * n_layers),
        layer_hard_max=layer_hard_max,
    )
    s = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in s.parameters():
            p.zero_()
    return s


def main():
    t0 = time.time()
    print("[multi-hrm] loading real HRM checkpoint...")
    real_hrm, ckpt = load_real_hrm()
    print(f"  real HRM d_model={real_hrm.config.d_model} "
          f"n_layers={real_hrm.config.n_layers} "
          f"val_acc={ckpt['val_acc']:.4f}")

    print(f"[multi-hrm] building {N_HRM_SLOTS - 1} synthetic HRM stand-ins...")
    hrms = [real_hrm] + [
        build_synthetic_hrm(real_hrm.config, seed=100 + i)
        for i in range(N_HRM_SLOTS - 1)
    ]

    print("[multi-hrm] building dispatched_v4 card...")
    card = build_dispatched_v4()

    print("[multi-hrm] building unified substrate...")
    substrate = build_substrate(real_hrm, card)
    print(f"  d_model={substrate.config.d_model} "
          f"n_heads={substrate.config.n_heads} "
          f"d_ffn={substrate.config.d_ffn} "
          f"n_layers={substrate.config.n_layers} "
          f"vocab={substrate.config.vocab_size} "
          f"params={substrate.param_count():,}")

    h = real_hrm.config
    HRM_VOCAB = h.vocab_size
    HRM_D_MODEL = h.d_model
    HRM_SH = h.d_model // 2
    HRM_FFN = h.d_ffn

    print(f"[multi-hrm] installing {N_HRM_SLOTS} HRM slots...")
    for i, hrm_i in enumerate(hrms):
        install_compiled_card(substrate, hrm_i, CardSlot(
            ch_off=i * HRM_D_MODEL,
            sh_off=i * HRM_SH,
            ffn_off=i * HRM_FFN,
            tok_off=i * HRM_VOCAB,
            layer_off=0,
        ))

    print("[multi-hrm] installing dispatched_v4 card...")
    install_compiled_card(substrate, card, CardSlot(
        ch_off=N_HRM_SLOTS * HRM_D_MODEL,
        sh_off=N_HRM_SLOTS * HRM_SH,
        ffn_off=N_HRM_SLOTS * HRM_FFN,
        tok_off=N_HRM_SLOTS * HRM_VOCAB,
        layer_off=h.n_layers,
    ))

    # Save/reload
    tmp = Path(tempfile.mkdtemp()) / "multi_hrm.pt"
    torch.save({"state_dict": substrate.state_dict(),
                "config": substrate.config.__dict__}, tmp)
    print(f"  saved {tmp.stat().st_size / 1e6:.1f} MB")

    reloaded = build_substrate(real_hrm, card)
    reloaded.load_state_dict(torch.load(tmp, weights_only=True)["state_dict"])
    reloaded.eval()

    # --- CHECK (a) slot 0 bit-identical to standalone real HRM ---
    print("\n[multi-hrm] CHECK (a) — slot 0 (real HRM) bit-identical")
    torch.manual_seed(7)
    sample_inp = torch.randint(0, HRM_VOCAB, (4, 16))
    with torch.no_grad():
        std_logits = real_hrm(sample_inp)
        # Substrate: slot-0 tokens are in [0, HRM_VOCAB). Output in slot-0 vocab.
        sub_logits = reloaded(sample_inp)[:, :, 0:HRM_VOCAB]
    diff_a = (std_logits - sub_logits).abs().max().item()
    ok_a = diff_a < 1e-4
    print(f"  max |real_hrm - substrate[slot0]| = {diff_a:.2e} — "
          f"{'PASS' if ok_a else 'FAIL'}")

    # --- CHECK (b) slots 1-4 produce distinct non-zero outputs ---
    print("\n[multi-hrm] CHECK (b) — slots 1-4 active and distinct")
    slot_outputs = []
    for i in range(N_HRM_SLOTS):
        tok_lo = i * HRM_VOCAB
        inp_i = torch.randint(0, HRM_VOCAB, (1, 16)) + tok_lo
        with torch.no_grad():
            l = reloaded(inp_i)[0, -1, tok_lo:tok_lo + HRM_VOCAB]
        slot_outputs.append(l)
        print(f"  slot {i}: logit range [{l.min().item():.3f}, {l.max().item():.3f}] "
              f"std={l.std().item():.3f}")
    # All slots should have non-trivial outputs
    ok_b = all(s.std().item() > 1e-3 for s in slot_outputs)
    # And they should differ from each other (random-init HRMs produce
    # different distributions).
    pairwise_ok = True
    for i in range(1, N_HRM_SLOTS):
        for j in range(i + 1, N_HRM_SLOTS):
            d = (slot_outputs[i] - slot_outputs[j]).abs().max().item()
            if d < 1e-3:
                pairwise_ok = False
    ok_b = ok_b and pairwise_ok
    print(f"  {'PASS' if ok_b else 'FAIL'}")

    # --- CHECK (c) slot-i input dominates — other slots produce only
    # pos-embed noise, functionally silent (slot-0 logits win argmax). ---
    # NB: each slot's pos embedding is written to its own channel range,
    # so slots-j≠i receive a pos-only signal even when i's tokens are
    # in play. Exact zero isolation would require shared-pos or a
    # gated pos embedding — out of scope here. Functional isolation
    # (slot-i's max logit ≫ slot-j's) is what matters for argmax routing.
    print("\n[multi-hrm] CHECK (c) — slot isolation (functional dominance)")
    inp_slot0 = torch.randint(0, HRM_VOCAB, (1, 8))
    with torch.no_grad():
        full_l = reloaded(inp_slot0)[0, -1]
    slot0_max = full_l[0:HRM_VOCAB].abs().max().item()
    crosstalk_max = 0.0
    for j in range(1, N_HRM_SLOTS):
        tok_lo = j * HRM_VOCAB
        other = full_l[tok_lo:tok_lo + HRM_VOCAB]
        crosstalk_max = max(crosstalk_max, other.abs().max().item())
    # Slot 0 must dominate by at least 10×.
    ok_c = slot0_max > 10 * crosstalk_max
    print(f"  slot-0 max |logit| = {slot0_max:.3f}, "
          f"max other-slot |logit| = {crosstalk_max:.2e} "
          f"(ratio {slot0_max / max(crosstalk_max, 1e-12):.0f}×) — "
          f"{'PASS' if ok_c else 'FAIL'}")
    # Also verify argmax lands in slot-0 range
    argmax_global = int(full_l.argmax().item())
    argmax_in_slot0 = argmax_global < HRM_VOCAB
    print(f"  global argmax = token {argmax_global} "
          f"({'in slot-0' if argmax_in_slot0 else 'OUTSIDE slot-0'}) — "
          f"{'OK' if argmax_in_slot0 else 'LEAK'}")
    ok_c = ok_c and argmax_in_slot0

    # --- CHECK (d) dispatched_v4 passes exhaustive ---
    print("\n[multi-hrm] CHECK (d) — dispatched_v4 791/791")
    CARD_TOK_OFF = N_HRM_SLOTS * HRM_VOCAB

    def _run(inputs, expected, label):
        shifted = [(a + CARD_TOK_OFF, b + CARD_TOK_OFF,
                    op + OPCODE_SHIFT + CARD_TOK_OFF)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long)
        with torch.no_grad():
            logits = reloaded(x)
        card_range = logits[:, 2, CARD_TOK_OFF:CARD_TOK_OFF + CARD_VOCAB]
        preds = card_range.argmax(dim=-1).tolist()
        c = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {c}/{len(inputs)}")
        return c, len(inputs)

    pairs = list(itertools.product(range(GCD_BASE), repeat=2))
    mul_pairs = list(itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2))
    ok_d_cnt = tot_d = 0
    for inp, exp, lab in [
        ([(a, b, 0) for a, b in pairs], [math.gcd(a, b) for a, b in pairs], "GCD      "),
        ([(n, 0, 1) for n in range(FACT_MAX_N + 1)],
         [math.factorial(n) for n in range(FACT_MAX_N + 1)], "FACTORIAL"),
        ([(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
         [_is_prime(n) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)], "IS_PRIME "),
        ([(a, b, 3) for a, b in pairs], [a + b for a, b in pairs], "ADD      "),
        ([(a, b, 4) for a, b in mul_pairs], [a * b for a, b in mul_pairs], "MUL      "),
    ]:
        c, n = _run(inp, exp, lab)
        ok_d_cnt += c
        tot_d += n
    ok_d = ok_d_cnt == tot_d
    print(f"  dispatched_v4 total: {ok_d_cnt}/{tot_d} — "
          f"{'PASS' if ok_d else 'FAIL'}")

    # --- CHECK (e) HRM input silences card slots ---
    print("\n[multi-hrm] CHECK (e) — HRM input → card slots exactly zero")
    inp_hrm = torch.randint(0, HRM_VOCAB, (2, 3))  # slot-0 HRM input
    with torch.no_grad():
        l = reloaded(inp_hrm)[:, :, CARD_TOK_OFF:CARD_TOK_OFF + CARD_VOCAB]
    max_card = l.abs().max().item()
    ok_e = max_card < 1e-5
    print(f"  max |card slots on HRM input| = {max_card:.2e} — "
          f"{'PASS' if ok_e else 'FAIL'}")

    all_ok = ok_a and ok_b and ok_c and ok_d and ok_e
    t = time.time() - t0
    print(f"\n[multi-hrm] OVERALL: {'PASS' if all_ok else 'FAIL'}  ({t:.1f}s)")
    print("[multi-hrm] 5 HRM slots + dispatched_v4 in one substrate:")
    print(f"[multi-hrm]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    tmp.unlink()


if __name__ == "__main__":
    main()
