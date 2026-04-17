"""Round 49.1: L24 FFN activation rank — is the diffuse circuit low-rank
in a different basis?

R48.1 showed L24 FFN is diffuse at neuron level (top 25% of neurons = 6%
of full Δ). That rules out compact per-neuron intervention but is fully
consistent with SUPERPOSITION — many neurons each encoding a fraction
of a small number of features.

Hypothesis: the composition circuit lives in a LOW-RANK subspace of
the residual stream. Specifically, L24's FFN contribution (shape
d_model=2560) across varied multi-step prompts should have effective
rank << 10240 and potentially << 100.

Method:
  1. Run N multi-step prompts through Gemma.
  2. At each run, capture L24 ffn_down output at last query position
     (d_model=2560 vector).
  3. Stack into matrix A ∈ R^(N × 2560).
  4. SVD → singular values σ_1 ≥ σ_2 ≥ ... ≥ σ_min(N, d_model).
  5. Effective rank = smallest K such that sum(σ_i² for i≤K) / sum(σ_i²)
     ≥ 0.9.

Gates:
  rank < 50:    extremely low-rank, clean rank-K compile target
  rank 50-200:  moderately low-rank, compile via top-K SVD
  rank > 500:   circuit is genuinely high-rank, not compressible

Also: compare against L23 FFN (single-step hub, R16) — expect L23
to ALSO be low-rank (it's a concentrated circuit). If L23 is rank ~20
and L24 is rank ~80, we've quantified the extra composition capacity.

Cost: 100 forwards (no ablation sweep) ≈ 3 min. Plus a few-ms SVD.
"""

from __future__ import annotations

import math
import os
import random
import sys

import torch


GGUF_PATH = os.path.expanduser("~/models/gemma-4-E4B-it-tq4-aligned.gguf")


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def project_to_logits(m, h):
    normed = _rms_norm(h, m.output_norm_w, m.config.rms_norm_eps)
    return torch.tanh(m.token_embd.output_logits(normed[:, -1:, :]) / 30.0) * 30.0


class FFNOutputCapture:
    """Wraps ffn_down to record its last-position output."""
    def __init__(self, inner):
        self.inner = inner
        self.captured: list[torch.Tensor] = []
        self.in_features = inner.in_features
        self.out_features = inner.out_features
        self._gpu_qs = getattr(inner, "_gpu_qs", None)

    def __call__(self, x):
        out = self.inner(x)
        # out: (B=1, S, d_model). Take last position.
        self.captured.append(out[0, -1, :].detach().clone().cpu())
        return out


def forward_and_capture(m, token_ids, layers_to_capture: list[int]):
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

    # Install capture wrappers
    captures: dict[int, FFNOutputCapture] = {}
    originals: dict[int, object] = {}
    for L in layers_to_capture:
        originals[L] = m.layers[L].ffn_down
        captures[L] = FFNOutputCapture(originals[L])
        m.layers[L].ffn_down = captures[L]

    try:
        with torch.no_grad():
            for i, layer in enumerate(m.layers):
                h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
    finally:
        for L, orig in originals.items():
            m.layers[L].ffn_down = orig

    return {L: cap.captured[-1] for L, cap in captures.items()}


def sample_multi_step(rng, n: int) -> list[tuple[str, int]]:
    """Generate n multi-step (a, b, c) prompts with a*b < 999 and c < 99.
    Returns list of (prompt, correct_answer). """
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        c = rng.randint(1, 99)
        prompt = f"What is ({a} * {b}) + {c}? Answer: "
        out.append((prompt, a * b + c))
    return out


def sample_single_step(rng, n: int) -> list[tuple[str, int]]:
    """Generate n single-step a*b prompts for comparison."""
    out = []
    for _ in range(n):
        while True:
            a = rng.randint(2, 29)
            b = rng.randint(2, 29)
            if a * b < 1000:
                break
        prompt = f"{a} times {b} equals "
        out.append((prompt, a * b))
    return out


def effective_rank(A: torch.Tensor, threshold: float = 0.90) -> int:
    """Smallest K such that top-K singular values capture ≥ threshold
    of the total squared-singular-value sum."""
    # A: (N, d). SVD → S ∈ R^min(N, d)
    S = torch.linalg.svdvals(A.float())
    energy = (S ** 2).cumsum(dim=0)
    total = energy[-1]
    ratio = energy / total
    K = int((ratio >= threshold).nonzero()[0].item()) + 1
    return K, S.tolist()


def main():
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    enable_triton_tq4(True)
    print("[r49.1] loading substrate...")
    m = GemmaSubstrate.from_gguf(GGUF_PATH, max_len=256)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(GGUF_PATH)

    N_PROMPTS = 100
    rng = random.Random(42)
    multi_prompts = sample_multi_step(rng, N_PROMPTS)
    single_prompts = sample_single_step(rng, N_PROMPTS)

    # Capture FFN outputs at L24 (composition peak) and L23 (single-step hub)
    LAYERS = [23, 24]

    print(f"\n=== capturing L{LAYERS} FFN output across {N_PROMPTS} "
          f"multi-step prompts ===")
    multi_acts = {L: [] for L in LAYERS}
    for i, (prompt, _) in enumerate(multi_prompts):
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        caps = forward_and_capture(m, token_ids, LAYERS)
        for L in LAYERS:
            multi_acts[L].append(caps[L])
        if i % 25 == 0:
            print(f"  {i+1}/{N_PROMPTS}")

    print(f"\n=== capturing across {N_PROMPTS} single-step prompts ===")
    single_acts = {L: [] for L in LAYERS}
    for i, (prompt, _) in enumerate(single_prompts):
        token_ids = torch.tensor([tok.encode(prompt)], device="cuda")
        caps = forward_and_capture(m, token_ids, LAYERS)
        for L in LAYERS:
            single_acts[L].append(caps[L])
        if i % 25 == 0:
            print(f"  {i+1}/{N_PROMPTS}")

    # Analyze rank per layer, task
    print(f"\n========== RANK ANALYSIS ==========")
    print(f"{'task':>12} {'layer':>6} {'mean ||a||':>12} "
          f"{'rank90':>8} {'rank95':>8} {'rank99':>8}")

    results = {}
    for task_name, acts in [("multi", multi_acts), ("single", single_acts)]:
        for L in LAYERS:
            A = torch.stack(acts[L])  # (N, d_model)
            mean_norm = A.norm(dim=1).mean().item()
            rank90, S = effective_rank(A, 0.90)
            rank95, _ = effective_rank(A, 0.95)
            rank99, _ = effective_rank(A, 0.99)
            results[(task_name, L)] = {
                "A": A, "rank90": rank90, "rank95": rank95,
                "rank99": rank99, "S": S, "mean_norm": mean_norm,
            }
            print(f"{task_name:>12} L{L:>4} {mean_norm:>12.2f} "
                  f"{rank90:>8} {rank95:>8} {rank99:>8}")

    # Cross-task subspace alignment: how much of multi-step variance is
    # captured by the top-K single-step directions?
    print(f"\n========== COMPOSITION SUBSPACE (multi - single) ==========")
    print(f"How much of multi-step activation variance projects onto the")
    print(f"top-K single-step PCA directions? Residual = composition-specific.")
    print(f"")
    print(f"{'layer':>6} {'top-K':>8} {'captured_by_single':>22} "
          f"{'composition-residual':>22}")
    for L in LAYERS:
        A_multi = results[("multi", L)]["A"].float()
        A_single = results[("single", L)]["A"].float()
        # Get single-step top directions via SVD
        _, _, Vt_single = torch.linalg.svd(A_single, full_matrices=False)
        # Project multi onto first K single-step directions for varied K
        for K in [5, 20, 50, results[("single", L)]["rank90"]]:
            if K > Vt_single.shape[0]:
                continue
            V_single_K = Vt_single[:K]  # (K, d)
            # Project: A_multi_proj = A_multi @ V_single_K.T @ V_single_K
            proj = A_multi @ V_single_K.T @ V_single_K  # (N, d)
            captured = (proj.norm(dim=1) ** 2).sum()
            total = (A_multi.norm(dim=1) ** 2).sum()
            captured_ratio = (captured / total).item()
            resid_ratio = 1.0 - captured_ratio
            print(f"L{L:>5} {K:>8} {captured_ratio*100:>21.1f}% "
                  f"{resid_ratio*100:>21.1f}%")

    # Gate
    print(f"\n========== R49.1 GATE ==========")
    l24_rank90 = results[("multi", 24)]["rank90"]
    l24_rank95 = results[("multi", 24)]["rank95"]
    low_rank = l24_rank90 < 50
    mid_rank = 50 <= l24_rank90 < 200
    print(f"  L24 multi-step rank@90%: {l24_rank90}")
    print(f"  L24 multi-step rank@95%: {l24_rank95}")

    if low_rank:
        print(f"\n  ✓ L24 FFN contribution is EXTREMELY low-rank despite")
        print(f"    neuron-diffuse. Superposition confirmed. Compile")
        print(f"    target: rank-{l24_rank90} linear approximation.")
        print(f"    Next R49.2: compile + verify rank-K approximation.")
    elif mid_rank:
        print(f"\n  ~ L24 is moderately low-rank. Compile via top-{l24_rank90}")
        print(f"    SVD, but approximation error may be significant.")
    else:
        print(f"\n  ✗ L24 multi-step rank > 200 — genuinely high-rank,")
        print(f"    not compressible. Reverse-engineering this circuit")
        print(f"    requires more sophisticated tools (SAE, ACDC).")

    # Save everything for R49.2
    torch.save({
        "multi_acts_L23": results[("multi", 23)]["A"].cpu(),
        "multi_acts_L24": results[("multi", 24)]["A"].cpu(),
        "single_acts_L23": results[("single", 23)]["A"].cpu(),
        "single_acts_L24": results[("single", 24)]["A"].cpu(),
        "multi_prompts": multi_prompts,
        "single_prompts": single_prompts,
        "ranks": {
            k: {"rank90": v["rank90"], "rank95": v["rank95"],
                "rank99": v["rank99"]}
            for k, v in results.items()
        },
    }, "/tmp/r49_1_l24_rank.pt")
    print(f"\n  saved: /tmp/r49_1_l24_rank.pt")


if __name__ == "__main__":
    sys.exit(main())
