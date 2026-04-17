"""Round 47.1: multi-step arithmetic layer sweep.

Mirrors R16 (which localized single-step `a*b` to L22-L30, peak L23)
but on multi-step `a*b+c` prompts. 42 layers × 10 prompts = 420
forwards. For each (layer, prompt), measure Δ at the expected
final-answer first-digit logit when layer's contribution is zeroed.

Hypothesis: multi-step arithmetic is computed in a single forward pass
on direct-answer prompts, and localizes to a specific layer cluster —
probably later than L22-L30 since step 2 happens AFTER step 1 writes
its intermediate. If we find a concentrated cluster (≥ 4 layers with
mean Δ < -1.0), the circuit is tractable; per-head probing follows
in R47.2.

If diffuse OR if baseline doesn't attempt (emits '\\n' or 'The' etc.),
flag: Gemma may be doing multi-step via sequential chain-of-thought,
not single-forward — a qualitatively different compile target.

Prompt format: '{a} times {b} plus {c} equals ' — mirrors R16's
structure so any localization difference is attributable to the
added +c step, not prompt framing.
"""

from __future__ import annotations

import math
import os
import sys

import torch


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


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

# 10 multi-step triples: (a, b, c) → a*b+c, varied first-digit answers.
TRIPLES = [
    (17, 23, 5),   # 396, '3'
    (47, 19, 23),  # 916, '9'
    (37, 14, 50),  # 568, '5'
    (13, 27, 8),   # 359, '3'
    (21, 38, 15),  # 813, '8'
    (11, 11, 10),  # 131, '1'
    (29, 17, 4),   # 497, '4'
    (32, 25, 7),   # 807, '8'
    (16, 31, 12),  # 508, '5'
    (34, 12, 5),   # 413, '4'
]


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r47.1] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Verify baseline — does Gemma attempt? Filter out prompts where
    # baseline argmax isn't a digit.
    deltas = torch.zeros(m.config.n_layers, len(TRIPLES))
    baseline_correct = torch.zeros(len(TRIPLES))
    baselines = []
    usable = []

    print(f"\n=== baseline forwards (is Gemma attempting?) ===")
    print(f"{'triple':>20} {'expected':>10} {'correct_d':>10} "
          f"{'base_argmax':>14} {'base_L_corr':>12} {'base_L_arg':>12}")
    for j, (a, b, c) in enumerate(TRIPLES):
        prompt = f"{a} times {b} plus {c} equals "
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        base_logits = forward_with_ablation(m, token_ids, ablate_layer=None)
        base_correct = base_logits[0, -1, DIGIT_IDS[correct_d]].item()
        base_argmax = int(base_logits[0, -1].argmax())
        base_argmax_tok = tok.id_to_token.get(base_argmax, '?')
        base_argmax_logit = base_logits[0, -1, base_argmax].item()
        baselines.append((token_ids, correct_d, base_correct, answer,
                          prompt, base_argmax_tok))
        baseline_correct[j] = base_correct

        # Usable: baseline argmax is SOME digit (Gemma is attempting)
        # OR the correct digit is within top-10.
        top10 = base_logits[0, -1].topk(10).indices.tolist()
        top10_toks = [tok.id_to_token.get(i, '?') for i in top10]
        attempts = any(t.lstrip('▁').isdigit() for t in top10_toks)
        if attempts:
            usable.append(j)

        print(f"{f'{a}×{b}+{c}={answer}':>20} "
              f"{str(answer):>10} {correct_d!r:>10} "
              f"{base_argmax_tok!r:>14} "
              f"{base_correct:>+12.2f} {base_argmax_logit:>+12.2f}")

    print(f"\n  usable prompts (digit in top-10): {len(usable)}/{len(TRIPLES)}")
    if len(usable) < 5:
        print(f"\n  ✗ Gemma isn't attempting multi-step in single forward.")
        print(f"    Multi-step may require sequential chain-of-thought.")
        print(f"    Flag for R47-alt: probe with answer-in-prompt format.")
        return 0

    print(f"\n=== ablation sweep: 42 layers × {len(TRIPLES)} triples ===")
    for L in range(m.config.n_layers):
        for j, (token_ids, correct_d, base_correct,
                 answer, prompt, _) in enumerate(baselines):
            abl_logits = forward_with_ablation(m, token_ids, ablate_layer=L)
            abl_correct = abl_logits[0, -1, DIGIT_IDS[correct_d]].item()
            deltas[L, j] = abl_correct - base_correct
        if L % 5 == 0:
            mean_d = deltas[L, usable].mean().item() if usable else float('nan')
            print(f"  L{L:>2}: mean Δ(correct, usable) = {mean_d:+.3f}")

    # --- Summary ---
    mean_all = deltas.mean(dim=1)
    mean_usable = deltas[:, usable].mean(dim=1) if usable else mean_all

    print(f"\n========== LAYER AVERAGES ({len(usable)} usable triples) ==========")
    print(f"{'L':>3} {'mean_Δ':>10} {'std':>8} {'#hurts':>8}  "
          f"({'hurts = Δ < -0.5':>18})")
    for L in range(m.config.n_layers):
        mu = mean_usable[L].item()
        std = deltas[L, usable].std().item() if usable else 0.0
        hurts = int((deltas[L, usable] < -0.5).sum().item()) if usable else 0
        marker = " ←" if mu < -1.0 or hurts >= len(usable) * 0.7 else ""
        print(f"{L:>3} {mu:>+10.3f} {std:>8.3f}   {hurts:>2}/{len(usable)}{marker}")

    # --- Cluster detection ---
    concentrated_layers = [
        L for L in range(m.config.n_layers)
        if mean_usable[L].item() < -1.0
    ]
    hurts_majority = [
        L for L in range(m.config.n_layers)
        if int((deltas[L, usable] < -0.5).sum().item())
        >= len(usable) * 0.7
    ]

    print(f"\n  layers with mean Δ < -1.0 (strongly load-bearing): "
          f"{concentrated_layers}")
    print(f"  layers hurting ≥70% of triples: {hurts_majority}")

    # R16 comparison
    print(f"\n  R16 (single-step a*b) found:")
    print(f"    L22-L30 cluster, L23 peak mean Δ=-10.18 (hurts 10/10)")
    print(f"\n  R47.1 (multi-step a*b+c) found:")
    if concentrated_layers:
        peak = min(range(m.config.n_layers),
                    key=lambda L: mean_usable[L].item())
        print(f"    Cluster: {concentrated_layers}")
        print(f"    Peak:    L{peak} mean Δ={mean_usable[peak].item():+.3f}")

    # Gate
    print(f"\n========== R47.1 GATE ==========")
    gate_ok = len(concentrated_layers) >= 4
    print(f"  cluster size ≥ 4 load-bearing layers: "
          f"{'PASS' if gate_ok else 'FAIL'} "
          f"({len(concentrated_layers)} layers)")

    if gate_ok:
        print(f"\n  ✓ Multi-step arithmetic localizes to a concentrated")
        print(f"    cluster. R47.2 (per-head ablation at peak layer) is")
        print(f"    the next step.")
    else:
        print(f"\n  ~ Multi-step is diffuse in single forward.")
        print(f"    Possibilities:")
        print(f"      (a) Gemma does multi-step via sequential")
        print(f"          chain-of-thought (multi-forward), not single-pass")
        print(f"      (b) Distributed across too many layers to compile")
        print(f"      (c) Prompt framing doesn't activate the circuit")
        print(f"    Next: try alternate prompt formats, or pivot to")
        print(f"    Route B (composed compiled cards).")

    # Save raw deltas for R47.2
    torch.save({
        "deltas": deltas.cpu(),
        "triples": TRIPLES,
        "usable": usable,
        "mean_usable": mean_usable.cpu(),
    }, "/tmp/r47_1_deltas.pt")
    print(f"\n  saved raw data: /tmp/r47_1_deltas.pt")
    return 0 if gate_ok else 1


if __name__ == "__main__":
    sys.exit(main())
