"""Round 14: activation patching — causal probe for failure localization.

For each layer L, replace that layer's output residual with the
INPUT residual (effectively zeroing L's contribution), run the rest
of the forward, measure how much the final logits change.

On a failure case (17×23 → 401), layers whose removal causes the
output to change MOST are the layers causally responsible for the
failure. Layers whose removal does nothing are pass-through for this
computation.

Measure per layer:
  - KL divergence between ablated logits and baseline logits
  - Argmax flip (does the top token change?)
  - Specifically: does the logit for '4' (wrong start) drop or
    does the logit for '3' (right start) rise?

If the failure is localized to a small number of layers, Phase 2+
(SAE on those layers, circuit identification) has a concrete target.
If it's spread uniformly across layers, interpretability on this
model is harder than expected.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    """Apply output_norm + head + softcap. Returns (B, 1, vocab)."""
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    last = normed[:, -1:, :]
    logits = m.token_embd.output_logits(last)
    cap = 30.0
    return torch.tanh(logits / cap) * cap


def forward_with_ablation(m, token_ids, ablate_layer=None):
    """Run Gemma's forward pass. If ablate_layer is not None, that
    layer's contribution is zeroed (we skip the layer by passing h
    through unchanged)."""
    from calm.llm_computer.gemma_substrate import KVCache
    cfg = m.config
    S = token_ids.shape[1]
    cache = KVCache(cfg.n_layers, device="cuda")

    # Embedding
    h = m.token_embd[token_ids].to("cuda")
    h = h * math.sqrt(cfg.d_model)

    # Per-layer embedding
    m._per_layer_embd = None
    if m.per_layer_token_embd is not None:
        d_pl = cfg.d_per_layer * cfg.n_layers
        pl_embd = m.per_layer_token_embd[token_ids]
        pl_embd = pl_embd * math.sqrt(cfg.d_per_layer)
        pl_embd = pl_embd.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
        if m.per_layer_model_proj is not None:
            h_proj = h @ m.per_layer_model_proj
            h_proj = h_proj * (1.0 / math.sqrt(cfg.d_model))
            h_proj = h_proj.reshape(1, S, cfg.n_layers, cfg.d_per_layer)
            if m.per_layer_proj_norm_w is not None:
                h_proj = _rms_norm(h_proj, m.per_layer_proj_norm_w, cfg.rms_norm_eps)
            pl_embd = (pl_embd + h_proj) * (1.0 / math.sqrt(2.0))
        m._per_layer_embd = [pl_embd[:, :, i, :] for i in range(cfg.n_layers)]

    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            # Always run the full forward (populates KV cache so later
            # shared-KV layers can read it). If ablating, discard the
            # layer's contribution by resetting h to its pre-layer value.
            h_before = h.clone() if i == ablate_layer else None
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == ablate_layer:
                h = h_before

    return project_to_logits(m, h)


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[patching] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    DIGIT_IDS = {
        '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
        '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
        '▁': 236743,
    }

    # Failure case: 17 × 23 = 391, Gemma says 401.
    # The wrong digit '4' gets picked when Gemma eventually emits the
    # first answer digit. But the immediate-next token is '\n'. We
    # need to trace TWO STEPS: the `\n` commitment, then the `4` vs `3`
    # commitment on the next forward.
    #
    # For this probe, simplify: use a prompt that forces Gemma to
    # emit the digit IMMEDIATELY (no `\n` prefix). Then layer-layer
    # ablation tells us where the digit choice is made.

    # Two prompts: one where Gemma gets it right, one where it gets wrong.
    prompts = [
        ("17 times 23 equals ", "direct"),  # baseline gets '3' (right)
        ("what is 17 times 23? Answer with just the number.", "verbose"),
        # verbose fails — emits \n then 401
    ]

    for prompt, label in prompts:
        print(f"\n\n{'#'*70}")
        print(f"## PROMPT: {prompt!r}  ({label})")
        print(f"{'#'*70}")
        _run_ablation_sweep(m, tok, prompt, DIGIT_IDS)


def _run_ablation_sweep(m, tok, prompt, DIGIT_IDS):
    token_ids = torch.tensor([tok.encode(prompt)], device="cuda")

    print(f"\n=== baseline forward ===")
    base_logits = forward_with_ablation(m, token_ids, ablate_layer=None)
    base_probs = F.softmax(base_logits[0, -1], dim=-1)
    base_top5 = torch.topk(base_logits[0, -1], k=5)
    print(f"  top-5: {[(tok.id_to_token.get(i.item(), '?'), v.item()) for v, i in zip(base_top5.values, base_top5.indices)]}")
    print(f"  P('3'): {base_probs[DIGIT_IDS['3']].item():.4f}  (correct first digit of 391)")
    print(f"  P('4'): {base_probs[DIGIT_IDS['4']].item():.4f}  (wrong first digit of 401)")

    base_logit_3 = base_logits[0, -1, DIGIT_IDS['3']].item()
    base_logit_4 = base_logits[0, -1, DIGIT_IDS['4']].item()
    base_argmax = int(base_logits[0, -1].argmax())

    print(f"\n=== per-layer ablation sweep (42 layers) ===")
    print(f"{'L':>3} {'argmax_change':>14} {'Δlogit_3':>10} {'Δlogit_4':>10} {'KL_div':>8}")

    results = []
    for L in range(m.config.n_layers):
        logits = forward_with_ablation(m, token_ids, ablate_layer=L)
        argmax = int(logits[0, -1].argmax())
        logit_3 = logits[0, -1, DIGIT_IDS['3']].item()
        logit_4 = logits[0, -1, DIGIT_IDS['4']].item()
        probs = F.softmax(logits[0, -1], dim=-1)
        kl = (base_probs * (base_probs.clamp_min(1e-20).log() - probs.clamp_min(1e-20).log())).sum().item()

        results.append({
            "layer": L,
            "argmax_flip": argmax != base_argmax,
            "argmax_new": argmax,
            "d_logit_3": logit_3 - base_logit_3,
            "d_logit_4": logit_4 - base_logit_4,
            "kl": kl,
        })
        flip_str = f"✗ → {tok.id_to_token.get(argmax, '?')!r}" if argmax != base_argmax else "—"
        print(f"{L:>3} {flip_str:>14} {logit_3 - base_logit_3:>+10.3f} "
              f"{logit_4 - base_logit_4:>+10.3f} {kl:>8.3f}")

    # Summary
    print(f"\n========== SUMMARY ==========")
    print(f"  baseline argmax: {tok.id_to_token.get(base_argmax, '?')!r}")
    print(f"  baseline P('3'): {base_probs[DIGIT_IDS['3']].item():.4f}")
    print(f"  baseline P('4'): {base_probs[DIGIT_IDS['4']].item():.4f}")

    # Layers whose ablation flipped the argmax
    flipped = [r for r in results if r["argmax_flip"]]
    print(f"\n  layers whose ablation flipped argmax: {len(flipped)}")
    for r in flipped:
        print(f"    L{r['layer']:>2} → argmax now "
              f"{tok.id_to_token.get(r['argmax_new'], '?')!r}, KL={r['kl']:.2f}")

    # Top layers by KL divergence
    print(f"\n  top 10 layers by KL divergence (most impactful):")
    top_kl = sorted(results, key=lambda r: -r["kl"])[:10]
    for r in top_kl:
        print(f"    L{r['layer']:>2} KL={r['kl']:.3f} "
              f"Δ3={r['d_logit_3']:+.2f} Δ4={r['d_logit_4']:+.2f}")

    # Top layers where removing helps 3 (correct) over 4 (wrong)
    print(f"\n  top 5 layers where ablation INCREASES 3 MORE than 4 (favors correct):")
    top_helpful = sorted(results, key=lambda r: -(r["d_logit_3"] - r["d_logit_4"]))[:5]
    for r in top_helpful:
        print(f"    L{r['layer']:>2} Δ3-Δ4={r['d_logit_3'] - r['d_logit_4']:+.3f} "
              f"(Δ3={r['d_logit_3']:+.2f}, Δ4={r['d_logit_4']:+.2f})")

    print(f"\n  top 5 layers where ablation INCREASES 4 MORE than 3 (these layers were helping the CORRECT answer; removing them hurts):")
    top_hurts = sorted(results, key=lambda r: -(r["d_logit_4"] - r["d_logit_3"]))[:5]
    for r in top_hurts:
        print(f"    L{r['layer']:>2} Δ4-Δ3={r['d_logit_4'] - r['d_logit_3']:+.3f} "
              f"(Δ3={r['d_logit_3']:+.2f}, Δ4={r['d_logit_4']:+.2f})")


if __name__ == "__main__":
    sys.exit(main())
