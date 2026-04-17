"""Round 47.3: multi-step layer sweep with prompt format that resists
copy-last-operand shortcut.

R47.2 discovered Gemma's baseline argmax on '{a} times {b} plus {c}
equals ' prompts often just echoes c (the last operand) instead of
computing. This may have contaminated R47.1's layer cluster — could
be circuits doing copy-c rather than step-2 composition.

Hypothesis: with a prompt format where c is NOT the last operand and
the continuation trigger is an explicit answer cue, we'll see:
  (a) baseline argmax matching correct first digit on ≥ 3/10 triples
  (b) a cleaner layer cluster (or no cluster → confirms Gemma doesn't
      multi-step in single forward)

Prompt: 'What is ({a} * {b}) + {c}? Answer: '
  - operand c is mid-prompt, not trailing
  - '?' and 'Answer:' prime direct-answer emission
  - parens emphasize grouping (tier 1 evaluation)

42 layers × 10 triples = 420 forwards ≈ 15 min.

Distinguishes:
  - Cluster at L22-L29 + L33-L41 with correct argmax ≥ 3/10:
      real two-stage circuit, R47.4+ per-head at L37/L40
  - Cluster with 0 correct argmax:
      R47.1's signal was copy-c, multi-step not in single forward
  - No cluster:
      diffuse or not-in-single-forward
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

# Same triples as R47.1 for comparability.
TRIPLES = [
    (17, 23, 5),   # 396 '3'
    (47, 19, 23),  # 916 '9'
    (37, 14, 50),  # 568 '5'
    (13, 27, 8),   # 359 '3'
    (21, 38, 15),  # 813 '8'
    (11, 11, 10),  # 131 '1'
    (29, 17, 4),   # 497 '4'
    (32, 25, 7),   # 807 '8'
    (16, 31, 12),  # 508 '5'
    (34, 12, 5),   # 413 '4'
]


def build_prompt(a: int, b: int, c: int) -> str:
    return f"What is ({a} * {b}) + {c}? Answer: "


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r47.3] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    deltas = torch.zeros(m.config.n_layers, len(TRIPLES))
    baselines = []
    n_argmax_correct = 0
    n_argmax_copy_c = 0

    print(f"\n=== baseline forwards (clean prompt format) ===")
    print(f"{'triple':>20} {'correct_d':>10} {'base_argmax':>14} "
          f"{'c_digit':>8} {'argmax_is':>12}")
    for j, (a, b, c) in enumerate(TRIPLES):
        prompt = build_prompt(a, b, c)
        answer = a * b + c
        correct_d = str(answer)[0]
        c_first_digit = str(c)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        base_logits = forward_with_ablation(m, token_ids, ablate_layer=None)
        base_correct = base_logits[0, -1, DIGIT_IDS[correct_d]].item()
        base_argmax = int(base_logits[0, -1].argmax())
        base_argmax_tok = tok.id_to_token.get(base_argmax, '?')
        base_argmax_stripped = base_argmax_tok.lstrip('▁')

        is_correct = base_argmax_stripped == correct_d
        is_copy_c = base_argmax_stripped == c_first_digit and not is_correct
        classification = ("CORRECT" if is_correct
                           else "copy_c" if is_copy_c
                           else "other")
        if is_correct:
            n_argmax_correct += 1
        if is_copy_c:
            n_argmax_copy_c += 1

        baselines.append((token_ids, correct_d, base_correct, answer,
                            prompt, is_correct))
        print(f"{f'{a}×{b}+{c}={answer}':>20} {correct_d!r:>10} "
              f"{base_argmax_tok!r:>14} {c_first_digit!r:>8} "
              f"{classification:>12}")

    print(f"\n  argmax correct:  {n_argmax_correct}/10  "
          f"(Gemma truly answering)")
    print(f"  argmax copy-c:   {n_argmax_copy_c}/10  "
          f"(shortcut echo of +c)")
    print(f"  argmax other:    {10 - n_argmax_correct - n_argmax_copy_c}/10")

    if n_argmax_correct == 0:
        print(f"\n  ✗ Gemma never picks correct digit as argmax on this "
              f"format either.")
        print(f"    Multi-step in single forward is unlikely. Sweep still")
        print(f"    runs so we can see if ANY cluster emerges for later"
              f" diagnosis.")

    print(f"\n=== ablation sweep: 42 layers × {len(TRIPLES)} triples ===")
    for L in range(m.config.n_layers):
        for j, (token_ids, correct_d, base_correct,
                 _, _, _) in enumerate(baselines):
            abl_logits = forward_with_ablation(m, token_ids, ablate_layer=L)
            abl_correct = abl_logits[0, -1, DIGIT_IDS[correct_d]].item()
            deltas[L, j] = abl_correct - base_correct
        if L % 5 == 0:
            mean_d = deltas[L].mean().item()
            print(f"  L{L:>2}: mean Δ = {mean_d:+.3f}")

    print(f"\n========== LAYER AVERAGES (clean format) ==========")
    print(f"{'L':>3} {'mean_Δ':>10} {'#hurts':>8}  (hurts = Δ < -0.5)")
    mean_all = deltas.mean(dim=1)
    for L in range(m.config.n_layers):
        mu = mean_all[L].item()
        hurts = int((deltas[L] < -0.5).sum().item())
        marker = " ←" if mu < -1.0 or hurts >= 7 else ""
        print(f"{L:>3} {mu:>+10.3f}   {hurts:>2}/10{marker}")

    concentrated = [
        L for L in range(m.config.n_layers)
        if mean_all[L].item() < -1.0
    ]
    hurts_majority = [
        L for L in range(m.config.n_layers)
        if int((deltas[L] < -0.5).sum().item()) >= 7
    ]

    print(f"\n  layers with mean Δ < -1.0: {concentrated}")
    print(f"  layers hurting ≥70% of triples: {hurts_majority}")

    # Comparison with R47.1 results
    print(f"\n  Comparison with R47.1 (prev prompt format):")
    print(f"    R47.1 cluster: [0, 5, 7, 13, 21, 22, 23, 24, 25, 26, 27,")
    print(f"                    29, 33, 34, 35, 37, 39, 40, 41]")
    print(f"    R47.3 cluster: {concentrated}")
    print(f"    shared:   "
          f"{sorted(set(concentrated) & {0,5,7,13,21,22,23,24,25,26,27,29,33,34,35,37,39,40,41})}")
    print(f"    new:      "
          f"{sorted(set(concentrated) - {0,5,7,13,21,22,23,24,25,26,27,29,33,34,35,37,39,40,41})}")
    print(f"    dropped:  "
          f"{sorted({0,5,7,13,21,22,23,24,25,26,27,29,33,34,35,37,39,40,41} - set(concentrated))}")

    print(f"\n========== R47.3 GATE ==========")
    real_circuit = n_argmax_correct >= 3 and len(concentrated) >= 4
    print(f"  baseline argmax correct ≥ 3/10: "
          f"{'PASS' if n_argmax_correct >= 3 else 'FAIL'} "
          f"({n_argmax_correct}/10)")
    print(f"  cluster size ≥ 4 layers:        "
          f"{'PASS' if len(concentrated) >= 4 else 'FAIL'} "
          f"({len(concentrated)} layers)")
    if real_circuit:
        print(f"\n  ✓ Real single-forward multi-step circuit confirmed.")
        print(f"    R47.4: per-head ablation at strongest new-layer peak.")
    elif len(concentrated) >= 4 and n_argmax_correct == 0:
        print(f"\n  ~ Cluster present but Gemma never gets argmax-correct.")
        print(f"    R47.1's cluster was likely copy-c, not real composition.")
        print(f"    Gemma doesn't do multi-step in single forward.")
        print(f"    Recommend: pivot to Route B (compose compiled cards).")
    else:
        print(f"\n  ✗ Neither gate passes — Gemma single-forward multi-step")
        print(f"    is not localized in a compile-friendly shape.")

    torch.save({
        "deltas": deltas.cpu(),
        "triples": TRIPLES,
        "n_argmax_correct": n_argmax_correct,
        "n_argmax_copy_c": n_argmax_copy_c,
        "mean_all": mean_all.cpu(),
    }, "/tmp/r47_3_clean_deltas.pt")
    print(f"\n  saved raw data: /tmp/r47_3_clean_deltas.pt")


if __name__ == "__main__":
    sys.exit(main())
