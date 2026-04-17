"""Round 49.3: position-wise L24 FFN ablation.

R47.3 found L24 mean Δ=-17.23 (full-layer ablation). R47.4 said diffuse
at heads. R48.1 said diffuse at neurons. R49.1 said rank 34 in SVD.
R49.2 said K=1 at POSITION -1 suffices. Contradiction?

Resolution: R47.3's -17.23 comes from L24's contributions at
non-last positions. R49.2 only zeroed/projected at position -1, so
earlier-position contributions survived. This round tests that.

Hypothesis: ablating L24 FFN at all EXCEPT-last positions will
recreate most of the -17.23 signal. Ablating only at last position
alone will have near-zero effect (consistent with R49.2).

Test conditions (10 held-out multi-step prompts, clean format):
  (a) baseline: no ablation
  (b) last-only: zero L24 FFN output at position -1
  (c) except-last: zero L24 FFN output at positions [0 .. S-2]
  (d) all: zero L24 FFN output everywhere

Predictions:
  (a) baseline logit ≈ 26.4 (R49.2 finding)
  (b) last-only Δ ≈ 0 (per R49.2's K=1 result)
  (c) except-last Δ ≈ -16 (most of the full-layer signal)
  (d) all Δ ≈ -17 (full-layer ablation)

If confirmed: the composition computation lives at L24 FFN applied
to earlier token positions (operand tokens, question-mark token,
answer: cue). Downstream layers' attention at position -1 reads
this via the residual stream.

Cost: 4 conditions × 10 prompts = 40 forwards ≈ 1 min.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")
TARGET_LAYER = 24


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class FFNPositionAblator:
    """Zero L24 ffn_down output at specified positions."""
    def __init__(self, inner, ablate_positions):
        """ablate_positions: set of ints, or string 'all'."""
        self.inner = inner
        self.positions = ablate_positions
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)  # (B=1, S, d_model)
        S = out.shape[1]
        if self.positions == "all":
            out = out.clone()
            out[0, :, :] = 0
        elif self.positions == "except_last":
            out = out.clone()
            out[0, :S-1, :] = 0
        elif self.positions == "last_only":
            out = out.clone()
            out[0, -1, :] = 0
        else:
            out = out.clone()
            for p in self.positions:
                if 0 <= p < S:
                    out[0, p, :] = 0
        return out


def forward_with_ablation(m, token_ids, ablator=None):
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

    target = m.layers[TARGET_LAYER]
    original = target.ffn_down
    if ablator is not None:
        target.ffn_down = ablator
    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.ffn_down = original


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

# Same 10 triples as R47.1/R47.3/R47.4 for direct comparison.
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
    print("[r49.3] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    conditions = [
        ("baseline", None),
        ("last_only", "last_only"),
        ("except_last", "except_last"),
        ("all", "all"),
    ]

    # Compute baselines first (no ablation)
    baselines = []
    for a, b, c in TRIPLES:
        prompt = build_prompt(a, b, c)
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_ablation(m, token_ids, ablator=None)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        baselines.append({
            "a": a, "b": b, "c": c, "answer": answer,
            "correct_d": correct_d, "token_ids": token_ids,
            "S": token_ids.shape[1], "base_correct": base_correct,
        })

    print(f"\n=== position-wise L{TARGET_LAYER} FFN ablation ===\n")
    print(f"{'condition':>14} {'mean logit':>12} {'mean Δ':>10} "
          f"{'correct_argmax':>16}")

    condition_stats = {}
    for cond_name, cond_mode in conditions:
        sum_logit = 0.0
        sum_delta = 0.0
        n_correct = 0
        details = []
        for b in baselines:
            if cond_mode is None:
                logits = forward_with_ablation(m, b["token_ids"], ablator=None)
            else:
                ablator = FFNPositionAblator(
                    m.layers[TARGET_LAYER].ffn_down, cond_mode)
                logits = forward_with_ablation(m, b["token_ids"],
                                                  ablator=ablator)
            correct_logit = logits[0, -1, DIGIT_IDS[b["correct_d"]]].item()
            delta = correct_logit - b["base_correct"]
            argmax = int(logits[0, -1].argmax())
            argmax_tok = tok.id_to_token.get(argmax, '?')
            if argmax_tok.lstrip('▁') == b["correct_d"]:
                n_correct += 1
            sum_logit += correct_logit
            sum_delta += delta
            details.append((b, correct_logit, delta))

        mean_logit = sum_logit / len(baselines)
        mean_delta = sum_delta / len(baselines)
        condition_stats[cond_name] = {
            "mean_logit": mean_logit,
            "mean_delta": mean_delta,
            "n_correct": n_correct,
            "details": details,
        }
        print(f"{cond_name:>14} {mean_logit:>12.2f} {mean_delta:>+10.2f}   "
              f"{n_correct}/{len(baselines)}")

    # Interpretation
    print(f"\n========== R49.3 ANALYSIS ==========")
    full_delta = condition_stats["all"]["mean_delta"]
    last_delta = condition_stats["last_only"]["mean_delta"]
    except_delta = condition_stats["except_last"]["mean_delta"]

    print(f"  full L24 FFN ablation:          Δ = {full_delta:+.2f}")
    print(f"  last-position-only ablation:    Δ = {last_delta:+.2f}")
    print(f"  non-last positions ablation:    Δ = {except_delta:+.2f}")
    print(f"  additive check (last + except): "
          f"{last_delta + except_delta:+.2f}  (should ≈ full)")

    # Gate
    signal_at_last = abs(last_delta) > 2.0
    signal_at_non_last = abs(except_delta) > 5.0
    print(f"\n========== R49.3 GATE ==========")
    if not signal_at_last and signal_at_non_last:
        print(f"  ✓ Composition computation localized to NON-LAST positions.")
        print(f"    L24 FFN at position -1 is ~null for composition.")
        print(f"    L24 FFN at operand/question positions carries the signal.")
        print(f"    Next R49.4: per-position ablation to find WHICH")
        print(f"    position(s) matter (operand tokens? question mark? etc.)")
    elif signal_at_last and signal_at_non_last:
        print(f"  ~ Composition is at BOTH last and non-last positions.")
        print(f"    Need per-position ablation to decompose.")
    elif signal_at_last and not signal_at_non_last:
        print(f"  ✗ Unexpected: last-position carries the signal but R49.2")
        print(f"    said K=1 projection at last position preserves accuracy.")
        print(f"    Investigate projection vs zero difference.")
    else:
        print(f"  ✗ Neither last nor non-last shows strong signal.")
        print(f"    R47.3 full-layer Δ may have been attention-dominated, not FFN.")

    torch.save({
        "condition_stats": {k: {kk: vv for kk, vv in v.items() if kk != "details"}
                              for k, v in condition_stats.items()},
        "triples": TRIPLES,
    }, "/tmp/r49_3_positions.pt")
    print(f"\n  saved: /tmp/r49_3_positions.pt")


if __name__ == "__main__":
    sys.exit(main())
