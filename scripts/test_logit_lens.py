"""Round 13: logit lens on Gemma 4 E4B.

For each layer L, compute what Gemma would emit if it stopped at L:
  project_to_logits(L) = softcap(head(output_norm(residual_at_L)))
  top_k = argmax over vocab

Contrast trajectories on failure vs success cases:
  FAIL: "what is 17 times 23? Answer with just the number." → 401 (Gemma)
  PASS: "what is 34 times 12? Answer with just the number." → 408 (Gemma)

What we're looking for:
  - Layer where the WRONG answer token (e.g. '4' for 17×23) first dominates
  - Whether the correct answer ('3' for 391) ever appears at any layer
  - Whether failure-case trajectory is distinguishable from success-case
    by layer-level structure

If predictions crystallize cleanly at a specific layer range, tracing
is tractable on this model — worth investing in SAE + circuit work.
If the trajectory is opaque, we know interpretability work on this
model is harder than expected.
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


def project_residual_to_logits(m, h):
    """Apply output_norm + head + softcap to residual h.
    Returns (B, S, vocab) logits."""
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    last = normed[:, -1:, :]  # only need last position for the lens
    logits = m.token_embd.output_logits(last)  # (B, 1, vocab)
    cap = 30.0
    logits = torch.tanh(logits / cap) * cap
    return logits


def top_k_tokens(tok, logits, k=5):
    """Return list of (token_string, logit_value) for top-k tokens
    at the LAST position."""
    last = logits[0, -1]  # (vocab,)
    top = torch.topk(last, k=k)
    result = []
    for i, (score, idx) in enumerate(zip(top.values.tolist(), top.indices.tolist())):
        lbl = tok.id_to_token.get(idx, f"?{idx}")
        result.append((lbl, score, idx))
    return result


def run_logit_lens(m, tok, prompt, track_tokens=None):
    """Run forward pass with hooks that capture each layer's residual.
    Print top-5 tokens predicted at each layer after position -1.
    If track_tokens is given, also print those tokens' rank/logit at
    each layer even if they're not top-5."""
    from calm.llm_computer.gemma_substrate import KVCache
    ids = tok.encode(prompt)
    S = len(ids)
    print(f"\n{'='*70}")
    print(f"prompt: {prompt!r}")
    print(f"  ids  : {ids}")
    print(f"  labels: {[tok.id_to_token.get(i, '?') for i in ids]}")
    print(f"{'='*70}")

    # Manual forward that captures residual after each layer
    cfg = m.config
    token_ids = torch.tensor([ids], device="cuda")
    cache = KVCache(cfg.n_layers, device="cuda")

    # Token embedding (mirrors gemma_substrate.forward)
    h = m.token_embd[token_ids].to("cuda")
    h = h * math.sqrt(cfg.d_model)

    # Per-layer embedding setup (mirrors forward)
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

    # Print lens after embedding + each layer
    print(f"\n{'layer':<8} {'top-5 tokens (label, logit)':<60}",
          end="")
    if track_tokens:
        print(f"  {'tracked':<20}")
    else:
        print()

    with torch.no_grad():
        # Logit lens on pre-layer-0 embedding
        logits = project_residual_to_logits(m, h)
        tops = top_k_tokens(tok, logits)
        tops_str = " ".join(f"{lbl!r:>6}({v:.1f})" for lbl, v, _ in tops)
        out = f"{'embed':<8} {tops_str:<60}"
        if track_tokens:
            tracked_str = track_tokens_str(tok, logits, track_tokens)
            out += f"  {tracked_str}"
        print(out)

        for i, layer in enumerate(m.layers):
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            logits = project_residual_to_logits(m, h)
            tops = top_k_tokens(tok, logits)
            tops_str = " ".join(f"{lbl!r:>6}({v:.1f})" for lbl, v, _ in tops)
            out = f"L{i:<7} {tops_str:<60}"
            if track_tokens:
                tracked_str = track_tokens_str(tok, logits, track_tokens)
                out += f"  {tracked_str}"
            print(out)

    return h


def track_tokens_str(tok, logits, track_tokens):
    """Format tracked-token info: label=logit for each tracked token."""
    last = logits[0, -1]
    parts = []
    for label, tok_id in track_tokens:
        v = last[tok_id].item()
        # Also compute rank
        rank = int((last > v).sum().item()) + 1
        parts.append(f"{label}:{v:.1f}(r{rank})")
    return " ".join(parts)


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[logit-lens] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Token IDs for digits — track where '3' (right for 391) vs
    # '4' (wrong start for 401) dominates.
    DIGIT_IDS = {
        '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
        '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
        '▁': 236743,  # space/answer-marker
    }

    # FAIL case: 17 × 23 = 391 (Gemma says 401)
    run_logit_lens(
        m, tok,
        "what is 17 times 23? Answer with just the number.",
        track_tokens=[
            ("▁", DIGIT_IDS['▁']),
            ("3", DIGIT_IDS['3']),  # correct first digit (391)
            ("4", DIGIT_IDS['4']),  # wrong first digit (401)
            ("9", DIGIT_IDS['9']),  # correct second digit (391)
            ("1", DIGIT_IDS['1']),  # correct third digit / wrong third
        ],
    )

    # PASS case: 34 × 12 = 408 (Gemma correct)
    run_logit_lens(
        m, tok,
        "what is 34 times 12? Answer with just the number.",
        track_tokens=[
            ("▁", DIGIT_IDS['▁']),
            ("4", DIGIT_IDS['4']),  # correct first digit (408)
            ("0", DIGIT_IDS['0']),
            ("8", DIGIT_IDS['8']),
        ],
    )


if __name__ == "__main__":
    sys.exit(main())
