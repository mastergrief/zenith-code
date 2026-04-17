"""Round 49.4: decompose L24's contribution across the three pathways.

R49.3 showed L24 FFN contributes ~0 to the correct-digit logit
while R47.3 showed full-layer ablation of L24 gives Δ=-17.23. The
residual is unexplained. Candidate: the per-layer-embedding pathway
that Gemma 4 E4B adds AFTER attn + FFN.

Gemma's per-layer path at layer L:
  proj_out = layer.proj(gelu(layer.inp_gate(h)) * per_layer_embd[L])
  h = h + proj_out

This is NOT attention and NOT FFN. It's a gated cross-projection
from a layer-specific embedding table. We never ablated it.

Test conditions:
  (a) baseline (no ablation)
  (b) zero L24 attn_output
  (c) zero L24 ffn_down output (all positions)
  (d) zero L24 proj output (per-layer-embd pathway)
  (e) zero all three (= full-layer ablation, reference -17.23)

Predictions:
  (b) attn-only:        small negative (R47.4 heads sum ~-0.25)
  (c) ffn-only:         small negative (R49.3 showed ~-0.59)
  (d) per-layer-only:   ≈ -16 if per-layer carries composition
  (e) all-three:        ≈ -17.23 (reproduces R47.3)

Check additivity: (b)+(c)+(d) ≈ (e). If significantly different,
there's non-linear interaction between pathways.

Cost: 5 conditions × 10 prompts = 50 forwards ≈ 1.5 min.
"""

from __future__ import annotations

import math
import os
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


class ZeroOutput:
    """Wraps a linear so its output is zeros. Pathway-ablate helper."""
    def __init__(self, inner):
        self.inner = inner
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        return torch.zeros_like(out)


def forward_with_pathway_ablation(m, token_ids, mode=None):
    """mode: None | 'attn' | 'ffn' | 'proj' | 'all' | 'layer'."""
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
    orig_attn = target.attn_output
    orig_ffn = target.ffn_down
    orig_proj = target.proj if hasattr(target, 'proj') else None

    if mode == "attn":
        target.attn_output = ZeroOutput(orig_attn)
    elif mode == "ffn":
        target.ffn_down = ZeroOutput(orig_ffn)
    elif mode == "proj":
        if orig_proj is not None:
            target.proj = ZeroOutput(orig_proj)
    elif mode == "all":
        target.attn_output = ZeroOutput(orig_attn)
        target.ffn_down = ZeroOutput(orig_ffn)
        if orig_proj is not None:
            target.proj = ZeroOutput(orig_proj)
    # None / "layer" → no ablation (baseline)

    try:
        with torch.no_grad():
            # For "layer" mode, we still want to reproduce R47.3's
            # full-layer ablation semantics: run the layer, then reset
            # h. Use the same approach as R47.3's forward_with_ablation.
            if mode == "layer":
                for i, layer in enumerate(m.layers):
                    h_before = h.clone() if i == TARGET_LAYER else None
                    h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
                    if i == TARGET_LAYER:
                        h = h_before
            else:
                for i, layer in enumerate(m.layers):
                    h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
        return project_to_logits(m, h)
    finally:
        target.attn_output = orig_attn
        target.ffn_down = orig_ffn
        if orig_proj is not None:
            target.proj = orig_proj


DIGIT_IDS = {
    '0': 236771, '1': 236770, '2': 236778, '3': 236800, '4': 236812,
    '5': 236810, '6': 236825, '7': 236832, '8': 236828, '9': 236819,
}

TRIPLES = [
    (17, 23, 5), (47, 19, 23), (37, 14, 50), (13, 27, 8), (21, 38, 15),
    (11, 11, 10), (29, 17, 4), (32, 25, 7), (16, 31, 12), (34, 12, 5),
]


def build_prompt(a, b, c):
    return f"What is ({a} * {b}) + {c}? Answer: "


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r49.4] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    # Verify layer 24 has per-layer pathway
    target = m.layers[TARGET_LAYER]
    has_proj = hasattr(target, 'proj') and target.proj is not None
    has_inp_gate = hasattr(target, 'inp_gate') and target.inp_gate is not None
    print(f"  L{TARGET_LAYER} has proj={has_proj}, has inp_gate={has_inp_gate}")

    # Baselines
    baselines = []
    for a, b, c in TRIPLES:
        prompt = build_prompt(a, b, c)
        answer = a * b + c
        correct_d = str(answer)[0]
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        logits = forward_with_pathway_ablation(m, token_ids, mode=None)
        base_correct = logits[0, -1, DIGIT_IDS[correct_d]].item()
        baselines.append({
            "a": a, "b": b, "c": c, "answer": answer,
            "correct_d": correct_d, "token_ids": token_ids,
            "base_correct": base_correct,
        })

    modes = [
        ("baseline", None),
        ("zero attn_out", "attn"),
        ("zero ffn_out", "ffn"),
        ("zero proj_out (per-layer)", "proj"),
        ("zero all three", "all"),
        ("full-layer (R47.3 style)", "layer"),
    ]

    print(f"\n=== L{TARGET_LAYER} pathway decomposition ===\n")
    print(f"{'condition':>30} {'mean logit':>12} {'mean Δ':>10} "
          f"{'correct_argmax':>16}")

    mean_deltas = {}
    for label, mode in modes:
        sum_logit = 0.0
        sum_delta = 0.0
        n_correct = 0
        for b in baselines:
            logits = forward_with_pathway_ablation(
                m, b["token_ids"], mode=mode)
            correct_logit = logits[0, -1, DIGIT_IDS[b["correct_d"]]].item()
            delta = correct_logit - b["base_correct"]
            argmax = int(logits[0, -1].argmax())
            argmax_tok = tok.id_to_token.get(argmax, '?')
            if argmax_tok.lstrip('▁') == b["correct_d"]:
                n_correct += 1
            sum_logit += correct_logit
            sum_delta += delta
        mean_logit = sum_logit / len(baselines)
        mean_delta = sum_delta / len(baselines)
        mean_deltas[label] = mean_delta
        print(f"{label:>30} {mean_logit:>12.2f} {mean_delta:>+10.2f}   "
              f"{n_correct}/{len(baselines)}")

    # Analysis
    d_attn = mean_deltas["zero attn_out"]
    d_ffn = mean_deltas["zero ffn_out"]
    d_proj = mean_deltas["zero proj_out (per-layer)"]
    d_all = mean_deltas["zero all three"]
    d_layer = mean_deltas["full-layer (R47.3 style)"]

    print(f"\n========== R49.4 DECOMPOSITION ==========")
    print(f"  attn pathway alone:          Δ = {d_attn:+.2f}")
    print(f"  ffn pathway alone:           Δ = {d_ffn:+.2f}")
    print(f"  per-layer-embd pathway alone: Δ = {d_proj:+.2f}")
    print(f"  all three zeroed:            Δ = {d_all:+.2f}")
    print(f"  full-layer (R47.3 style):    Δ = {d_layer:+.2f}")
    print(f"  additive check (attn+ffn+proj): {d_attn+d_ffn+d_proj:+.2f} "
          f"(vs all-three {d_all:+.2f})")

    # Gate
    print(f"\n========== R49.4 GATE ==========")
    per_layer_carries = abs(d_proj) > 5.0
    additive = abs((d_attn + d_ffn + d_proj) - d_all) < 2.0

    if per_layer_carries:
        print(f"  ✓ per-layer-embd pathway carries the composition signal")
        print(f"    ({d_proj:+.2f} vs full-layer {d_layer:+.2f})")
        if additive:
            print(f"  ✓ pathways are approximately additive — compile target")
            print(f"    is specifically the proj matmul at L{TARGET_LAYER}")
        else:
            print(f"  ~ non-additive interactions between pathways")
        print(f"\n    Next R49.5: rank analysis of per-layer proj output at")
        print(f"    L{TARGET_LAYER} (mirror R49.1 but for the proj pathway).")
    elif abs(d_layer) > 10:
        print(f"  ~ Full-layer Δ is large ({d_layer:+.2f}) but no single")
        print(f"    pathway carries it (attn={d_attn:+.2f}, ffn={d_ffn:+.2f},")
        print(f"    proj={d_proj:+.2f}). Interaction effects dominate.")
    else:
        print(f"  ✗ Full-layer Δ ({d_layer:+.2f}) is smaller than R47.3 "
              f"predicted.")
        print(f"    Methodology inconsistency — investigate.")

    torch.save({
        "mean_deltas": mean_deltas,
        "triples": TRIPLES,
    }, "/tmp/r49_4_decomp.pt")
    print(f"\n  saved: /tmp/r49_4_decomp.pt")


if __name__ == "__main__":
    sys.exit(main())
