"""Round 15: does L35 matter across arithmetic, or just 17×23?

Extends Round 14's activation patching. Run each arithmetic prompt,
ablate L35 alone, measure how much it changes the first-digit choice.

If L35 is the "arithmetic selection" circuit, ablating should
systematically flip the correct digit to a wrong one across many
test pairs. If L35 is prompt-specific, only 17×23 flips.

Test set: 10 2-digit × pairs. For each:
  - Baseline (no ablation): top digit logit rank
  - L35 ablated: does the top digit change? Is the correct digit's
    logit decreased, and by how much? Does a different digit rise?
"""

from __future__ import annotations

import math
import os
import re
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    last = normed[:, -1:, :]
    logits = m.token_embd.output_logits(last)
    return torch.tanh(logits / 30.0) * 30.0


def forward_with_ablation(m, token_ids, ablate_layer=None):
    from calm.llm_computer.gemma_substrate import KVCache
    cfg = m.config
    S = token_ids.shape[1]
    cache = KVCache(cfg.n_layers, device="cuda")

    h = m.token_embd[token_ids].to("cuda") * math.sqrt(cfg.d_model)
    m._per_layer_embd = None
    if m.per_layer_token_embd is not None:
        pl_embd = m.per_layer_token_embd[token_ids] * math.sqrt(cfg.d_per_layer)
        pl_embd = pl_embd.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
        if m.per_layer_model_proj is not None:
            h_proj = h @ m.per_layer_model_proj * (1.0 / math.sqrt(cfg.d_model))
            h_proj = h_proj.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
            if m.per_layer_proj_norm_w is not None:
                h_proj = _rms_norm(h_proj, m.per_layer_proj_norm_w, cfg.rms_norm_eps)
            pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
        m._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            h_before = h.clone() if i == ablate_layer else None
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == ablate_layer:
                h = h_before
    return project_to_logits(m, h)


DIGITS = "0123456789"
DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}


def first_digit_of(product: int) -> str:
    return str(product)[0]


# Pairs chosen so the correct first-digit of the product spans 0-9
# to test whether L35 favors SOMETHING specific to the correct answer,
# or just happens to matter for 17×23.
PAIRS = [
    (17, 23),  # =391, first digit '3' — Round 14's finding
    (34, 12),  # =408, '4' — Gemma passes baseline
    (47, 19),  # =893, '8' — Gemma wrongly says 903 normally
    (13, 27),  # =351, '3'
    (21, 38),  # =798, '7'
    (45, 15),  # =675, '6' — Gemma wrongly says 705 normally
    (11, 11),  # =121, '1'
    (29, 17),  # =493, '4'
    (32, 25),  # =800, '8'
    (16, 31),  # =496, '4'
]


def digit_logits(logits):
    """Return tensor of length 10 with logits for digits 0-9."""
    out = torch.zeros(10)
    for d, tok_id in DIGIT_IDS.items():
        out[int(d)] = logits[0, -1, tok_id].item()
    return out


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[l35-gen] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 16))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    print("\n=== ablate L35 on 10 2-digit × prompts ===\n")
    print(f"{'a,b':>8} {'product':>8} {'correct':>8} "
          f"{'base_top':>10} {'abl_top':>10} "
          f"{'Δcorrect':>10} {'KL':>6}")

    flips_to_correct = 0
    flips_to_wrong = 0
    no_flip = 0

    for a, b in PAIRS:
        product = a * b
        correct_d = first_digit_of(product)
        prompt = f"{a} times {b} equals "
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")

        # Baseline
        base_logits = forward_with_ablation(m, token_ids, ablate_layer=None)
        base_top = int(base_logits[0, -1].argmax())
        base_dlog = digit_logits(base_logits)
        base_correct_logit = base_dlog[int(correct_d)].item()

        # L35 ablated
        abl_logits = forward_with_ablation(m, token_ids, ablate_layer=35)
        abl_top = int(abl_logits[0, -1].argmax())
        abl_dlog = digit_logits(abl_logits)
        abl_correct_logit = abl_dlog[int(correct_d)].item()

        # KL
        base_probs = F.softmax(base_logits[0, -1], dim=-1)
        abl_probs = F.softmax(abl_logits[0, -1], dim=-1)
        kl = (base_probs * (base_probs.clamp_min(1e-20).log()
                            - abl_probs.clamp_min(1e-20).log())).sum().item()

        base_lbl = tok.id_to_token.get(base_top, '?')
        abl_lbl = tok.id_to_token.get(abl_top, '?')

        # What the flip is
        if base_top == abl_top:
            no_flip += 1
            flip_type = "—"
        elif abl_top == DIGIT_IDS.get(correct_d, -1):
            flips_to_correct += 1
            flip_type = "→✓"
        else:
            flips_to_wrong += 1
            flip_type = "→✗"

        print(f"{a:>3},{b:<3}   {product:>4}     {correct_d:>3}    "
              f"{base_lbl!r:>8}  {abl_lbl!r:>8} "
              f"{abl_correct_logit - base_correct_logit:>+10.2f} "
              f"{kl:>6.2f}  {flip_type}")

    print(f"\nSummary across 10 arithmetic pairs:")
    print(f"  L35 ablation → no flip:         {no_flip}")
    print(f"  L35 ablation → correct digit:   {flips_to_correct}")
    print(f"  L35 ablation → wrong digit:     {flips_to_wrong}")


if __name__ == "__main__":
    sys.exit(main())
