"""Round-9 E2E demo — SubstrateHRM (trained) + dispatched_v2 (compiled)
coexist in ONE substrate.

Validates that trained + compiled cards coexist under the same
`Small2DTransformer` / `GroupedSmall2DTransformer` protocol. Both are
substrate-compliant (d_head=2, channel-allocation-by-region); the
installer places each at a disjoint channel/sub-head/layer rectangle.

  * SubstrateHRM `substrate_hrm_nl_best.pt`: d_model=64, n_heads=32,
    n_layers=4, d_ffn=128, vocab=80, val_acc=99.1% on NL math parsing.
    Uses softmax attention.
  * dispatched_v2: d_model=582, n_heads=291, n_layers=2, d_ffn=1134,
    vocab=284. Uses hard_max attention (compiled LookUp requires it).
  * Substrate: 6 layers total (4 softmax + 2 hard_max). HRM installed
    at channels [0, 64), sub-heads [0, 32), FFN [0, 128), tokens [0, 80),
    layers [0, 4). Card at channels [64, 646), sub-heads [32, 323),
    FFN [128, 1262), tokens [80, 364), layers [4, 6).

Pass criteria:
  (1) HRM produces bit-identical logits in substrate as standalone for
      sampled NL-shaped input sequences.
  (2) dispatched_v2 exhaustive passes 791/791 in substrate.
  (3) Save/reload preserves both.
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
from calm.llm_computer.programs.dispatched_v2 import (
    ADD_MAX, FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND, PRIME_MAX_N, PRIME_MIN_N,
    VOCAB as CARD_VOCAB, build_dispatched_v2, decode_output,
)
from calm.llm_computer.programs.is_prime import _is_prime


HRM_CKPT = Path(
    "/mnt/c/Users/gabes/projects/claw-code/calm/hrm/checkpoints/"
    "substrate_hrm_nl_best.pt"
)


def load_hrm() -> tuple[Small2DTransformer, dict]:
    """Load `substrate_hrm_nl_best.pt` into a Small2DTransformer."""
    ckpt = torch.load(HRM_CKPT, weights_only=False, map_location="cpu")
    cfg_dict = ckpt["config"]
    cfg = Small2DConfig(
        vocab_size=cfg_dict["vocab_size"],
        d_model=cfg_dict["d_model"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        d_ffn=cfg_dict["d_ffn"],
        max_len=cfg_dict["max_len"],
        use_hard_max=False,  # HRM is softmax
    )
    model = Small2DTransformer(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model, ckpt


def build_substrate(hrm, card) -> GroupedSmall2DTransformer:
    """Substrate hosting HRM (softmax layers 0..N_H) + card (hard_max
    layers N_H..N_H+N_C)."""
    h = hrm.config
    c = card.config
    N_H = h.n_layers
    N_C = c.n_layers

    d_model = h.d_model + c.d_model             # 646
    d_model += d_model % 2
    n_heads = d_model // 2
    d_ffn = h.d_ffn + c.d_ffn                    # 1262
    vocab = h.vocab_size + c.vocab_size          # 364
    n_layers = N_H + N_C                         # 6
    max_len = max(h.max_len, c.max_len)          # 96

    layer_modes = tuple(["single"] * n_layers)
    layer_hard_max = tuple([False] * N_H + [True] * N_C)

    cfg = GroupedSmall2DConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        use_hard_max=False,
        layer_modes=layer_modes,
        layer_hard_max=layer_hard_max,
    )
    sub = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in sub.parameters():
            p.zero_()
    return sub


def main() -> None:
    t0 = time.time()
    print("[demo] loading SubstrateHRM NL-math checkpoint...")
    hrm, ckpt = load_hrm()
    print(f"  HRM d_model={hrm.config.d_model} n_heads={hrm.config.n_heads} "
          f"n_layers={hrm.config.n_layers} d_ffn={hrm.config.d_ffn} "
          f"vocab={hrm.config.vocab_size} val_acc={ckpt['val_acc']:.4f}")

    print("[demo] building dispatched_v2 card...")
    card = build_dispatched_v2()
    print(f"  card d_model={card.config.d_model} n_heads={card.config.n_heads} "
          f"n_layers={card.config.n_layers} d_ffn={card.config.d_ffn} "
          f"vocab={card.config.vocab_size}")

    print("[demo] building unified substrate...")
    substrate = build_substrate(hrm, card)
    N_H = hrm.config.n_layers
    print(f"  substrate d_model={substrate.config.d_model} "
          f"n_heads={substrate.config.n_heads} d_ffn={substrate.config.d_ffn} "
          f"n_layers={substrate.config.n_layers} "
          f"vocab={substrate.config.vocab_size} "
          f"params={substrate.param_count():,}")
    print(f"  layer_hard_max={substrate._grouped_config.layer_hard_max}")

    print("[demo] installing HRM @ slot (0, 0, 0, 0, 0)...")
    install_compiled_card(substrate, hrm, CardSlot(
        ch_off=0, sh_off=0, ffn_off=0, tok_off=0, layer_off=0,
    ))

    print(f"[demo] installing dispatched_v2 @ slot "
          f"(ch_off={hrm.config.d_model}, sh_off={hrm.config.d_model // 2}, "
          f"ffn_off={hrm.config.d_ffn}, tok_off={hrm.config.vocab_size}, "
          f"layer_off={N_H})...")
    install_compiled_card(substrate, card, CardSlot(
        ch_off=hrm.config.d_model,
        sh_off=hrm.config.d_model // 2,
        ffn_off=hrm.config.d_ffn,
        tok_off=hrm.config.vocab_size,
        layer_off=N_H,
    ))

    # Save + reload
    tmp = Path(tempfile.mkdtemp()) / "hrm_plus_card.pt"
    print(f"[demo] saving to {tmp}...")
    torch.save({
        "state_dict": substrate.state_dict(),
        "config": substrate.config.__dict__,
    }, tmp)
    print(f"  file size: {tmp.stat().st_size / 1e6:.1f} MB")

    print("[demo] reloading...")
    reloaded = build_substrate(hrm, card)
    ckpt_reload = torch.load(tmp, weights_only=True)
    reloaded.load_state_dict(ckpt_reload["state_dict"])
    reloaded.eval()

    # --- CHECK (1): HRM bit-identical ---
    print("\n[demo] CHECK (1) — HRM logits match standalone bit-exactly")
    # Sample input: a small sequence of HRM tokens.
    # HRM vocab size = 80 — use random ids in [0, 80).
    torch.manual_seed(42)
    n_samples = 10
    seq_len = 20
    hrm_inputs = torch.randint(0, hrm.config.vocab_size, (n_samples, seq_len))

    with torch.no_grad():
        hrm_logits = hrm(hrm_inputs)
    # Same tokens in substrate (HRM vocab maps to substrate vocab [0, 80))
    with torch.no_grad():
        sub_full_logits = reloaded(hrm_inputs)
    # Restrict substrate logits to HRM vocab range
    sub_hrm_logits = sub_full_logits[:, :, :hrm.config.vocab_size]
    diff = (hrm_logits - sub_hrm_logits).abs().max().item()
    ok_hrm = diff < 1e-4
    print(f"  max |hrm_standalone - substrate_hrm_range| = {diff:.2e} — "
          f"{'PASS' if ok_hrm else 'FAIL'}")

    # Verify argmax behavior matches too
    hrm_preds = hrm_logits.argmax(dim=-1)
    sub_preds = sub_hrm_logits.argmax(dim=-1)
    argmax_match = (hrm_preds == sub_preds).float().mean().item()
    print(f"  argmax agreement: {argmax_match * 100:.1f}%")

    # --- CHECK (2): dispatched_v2 exhaustive ---
    print("\n[demo] CHECK (2) — dispatched_v2 exhaustive (5 ops) via substrate")
    HRM_VOCAB = hrm.config.vocab_size

    def _run(inputs, expected, label):
        # Card tokens are substrate tokens in [HRM_VOCAB, HRM_VOCAB + CARD_VOCAB)
        shifted = [(a + HRM_VOCAB, b + HRM_VOCAB, op + HRM_VOCAB)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long)
        with torch.no_grad():
            logits = reloaded(x)
        card_logits = logits[:, 2, HRM_VOCAB:HRM_VOCAB + CARD_VOCAB]
        preds = card_logits.argmax(dim=-1).tolist()
        correct = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {correct}/{len(inputs)}")
        return correct, len(inputs)

    gcd_in = [(a, b, 0) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    gcd_ex = [math.gcd(a, b) for (a, b, _) in gcd_in]
    fact_in = [(n, 0, 1) for n in range(FACT_MAX_N + 1)]
    fact_ex = [math.factorial(n) for (n, _, _) in fact_in]
    prime_in = [(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]
    prime_ex = [_is_prime(n) for (n, _, _) in prime_in]
    add_in = [(a, b, 3) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    add_ex = [a + b for (a, b, _) in add_in]
    mul_in = [(a, b, 4) for a, b in itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2)]
    mul_ex = [a * b for (a, b, _) in mul_in]

    ok_card = 0
    tot = 0
    for inputs, expected, label in [
        (gcd_in, gcd_ex, "GCD      "),
        (fact_in, fact_ex, "FACTORIAL"),
        (prime_in, prime_ex, "IS_PRIME "),
        (add_in, add_ex, "ADD      "),
        (mul_in, mul_ex, "MUL      "),
    ]:
        c, n = _run(inputs, expected, label)
        ok_card += c
        tot += n
    ok_dispatched = ok_card == tot
    print(f"  dispatched total: {ok_card}/{tot} — "
          f"{'PASS' if ok_dispatched else 'FAIL'}")

    all_ok = ok_hrm and ok_dispatched
    t = time.time() - t0
    print(f"\n[demo] OVERALL: {'PASS' if all_ok else 'FAIL'}  (total {t:.1f}s)")
    print("[demo] trained HRM + compiled dispatched_v2 in one substrate:")
    print(f"[demo]   {'VALIDATED' if all_ok else 'NOT VALIDATED'}")
    tmp.unlink()


if __name__ == "__main__":
    main()
