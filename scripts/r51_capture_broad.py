"""R51.1: capture L24 input + residual contribution across 6 domains.

Daemon-compatible. Assumes `m` (GemmaSubstrate) and `tok` (GemmaTokenizer)
are pre-bound in the global namespace by `bin/gemma_daemon.py`. Does NOT
load Gemma itself.

For each of 3000 prompts (500 per domain: multi / single / trans / code
/ creative / factual), runs a Gemma forward at sequence length S and
captures BOTH the residual state entering L24 (h_before) AND L24's own
contribution (h_after - h_before), at every position except the last.
Each kept position is labelled with its source domain.

h_before is the student's INPUT during R51.3 training; contribution is
the TARGET. Saving them as aligned [N_total, 2560] tensors means the
training loop can load one file and index with a single domain_id mask
for stratified sampling.

Saves a dict to /tmp/r51_captures_broad.pt:
    X_in           [N_total, 2560]  fp32 cpu     residual entering L24
    X_out          [N_total, 2560]  fp32 cpu     L24's contribution (delta)
    domain_ids     [N_total]        int8 cpu     indices into DOMAIN_NAMES
    prompt_ids     [N_total]        int32 cpu    row -> prompt index (0..n-1)
    prompt_lens    [n_prompts]      int32 cpu    kept positions per prompt
    prompts        list[str]                     the prompt strings (in order)
    DOMAIN_NAMES   list[str]                     ["multi", ..., "factual"]
    prompt_counts  dict[str, int]                prompts per domain
    positions_per_domain dict[str, int]          kept positions per domain

prompt_ids / prompt_lens preserve sequence boundaries so R51.3 training can
reconstitute per-prompt sequences of shape [S_i, 2560] rather than treating
every position independently (which would collapse the student's attention).

Estimated runtime: ~300ms/prompt * 3000 = ~15 min on the daemon (longer
prompts in trans/creative push above the original 100ms/prompt estimate).
"""

from __future__ import annotations

import math
import random

import torch

from calm.llm_computer.r51.prompt_bank import build_broad_corpus


TARGET_LAYER = 24
PER_DOMAIN = 500
OUT_PATH = "/tmp/r51_captures_broad.pt"
DOMAIN_NAMES = ["multi", "single", "trans", "code", "creative", "factual"]
_DOMAIN_TO_ID = {name: i for i, name in enumerate(DOMAIN_NAMES)}


def _rms_norm(x, weight, eps=1e-6):
    rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return x / rms * weight


def forward_and_capture_l24(m, token_ids):
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

    h_before = None
    contribution = None
    with torch.no_grad():
        for i, layer in enumerate(m.layers):
            if i == TARGET_LAYER:
                h_before = h.clone().detach()
            h = m._forward_layer(h, layer, i, kv_cache=cache, start_pos=0)
            if i == TARGET_LAYER:
                contribution = (h - h_before).detach()
                break
    return h_before, contribution


def main():
    assert "m" in globals(), "daemon contract: `m` must be pre-bound"
    assert "tok" in globals(), "daemon contract: `tok` must be pre-bound"

    rng = random.Random(42)
    prompts, prompt_counts = build_broad_corpus(rng, per_domain=PER_DOMAIN)
    total = len(prompts)
    print(f"[r51.1] corpus: {total} prompts, {prompt_counts}", flush=True)

    in_parts: list[torch.Tensor] = []
    out_parts: list[torch.Tensor] = []
    domain_id_chunks: list[torch.Tensor] = []
    prompt_id_chunks: list[torch.Tensor] = []
    prompt_lens_list: list[int] = []
    kept_prompts: list[str] = []
    positions_per_domain: dict[str, int] = {name: 0 for name in DOMAIN_NAMES}
    in_norm_sums: dict[str, float] = {name: 0.0 for name in DOMAIN_NAMES}
    out_norm_sums: dict[str, float] = {name: 0.0 for name in DOMAIN_NAMES}
    domain_norm_counts: dict[str, int] = {name: 0 for name in DOMAIN_NAMES}
    skipped = 0

    with torch.no_grad():
        for i, (prompt, label) in enumerate(prompts):
            ids = tok.encode(prompt)
            if len(ids) < 2:
                skipped += 1
                continue
            token_ids = torch.tensor([ids], device="cuda")
            h_before, contrib = forward_and_capture_l24(m, token_ids)
            S = contrib.shape[1]
            if S <= 1:
                skipped += 1
                continue
            kept_in = h_before[0, : S - 1, :].detach().cpu().float()
            kept_out = contrib[0, : S - 1, :].detach().cpu().float()
            in_parts.append(kept_in)
            out_parts.append(kept_out)
            n_kept = kept_in.shape[0]
            dom_id = _DOMAIN_TO_ID[label]
            pr_id = len(kept_prompts)
            kept_prompts.append(prompt)
            prompt_lens_list.append(n_kept)
            domain_id_chunks.append(
                torch.full((n_kept,), dom_id, dtype=torch.int8)
            )
            prompt_id_chunks.append(
                torch.full((n_kept,), pr_id, dtype=torch.int32)
            )
            positions_per_domain[label] += n_kept
            in_norm_sums[label] += kept_in.norm(dim=1).sum().item()
            out_norm_sums[label] += kept_out.norm(dim=1).sum().item()
            domain_norm_counts[label] += n_kept

            if (i + 1) % 250 == 0:
                print(f"  {i+1}/{total} processed, skipped={skipped}",
                      flush=True)
            if (i + 1) % 500 == 0:
                torch.cuda.empty_cache()

    X_in = torch.cat(in_parts, dim=0).contiguous()
    X_out = torch.cat(out_parts, dim=0).contiguous()
    domain_ids = torch.cat(domain_id_chunks, dim=0).contiguous()
    prompt_ids = torch.cat(prompt_id_chunks, dim=0).contiguous()
    prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.int32)

    torch.save(
        {
            "X_in": X_in,
            "X_out": X_out,
            "domain_ids": domain_ids,
            "prompt_ids": prompt_ids,
            "prompt_lens": prompt_lens,
            "prompts": kept_prompts,
            "DOMAIN_NAMES": DOMAIN_NAMES,
            "prompt_counts": prompt_counts,
            "positions_per_domain": positions_per_domain,
        },
        OUT_PATH,
    )

    print("", flush=True)
    print(f"[r51.1] saved {OUT_PATH}", flush=True)
    print(f"  X_in  shape: {tuple(X_in.shape)}  dtype={X_in.dtype}",
          flush=True)
    print(f"  X_out shape: {tuple(X_out.shape)}  dtype={X_out.dtype}",
          flush=True)
    print(f"  domain_ids shape: {tuple(domain_ids.shape)}  "
          f"dtype={domain_ids.dtype}", flush=True)
    print(f"  total skipped prompts: {skipped}", flush=True)
    print("", flush=True)
    print("  per-domain summary:", flush=True)
    print(f"    {'domain':<10} {'prompts':>8} {'positions':>10} "
          f"{'mean||in||':>12} {'mean||out||':>13}", flush=True)
    for name in DOMAIN_NAMES:
        pc = prompt_counts.get(name, 0)
        pos = positions_per_domain[name]
        n = domain_norm_counts[name]
        mean_in = (in_norm_sums[name] / n) if n > 0 else 0.0
        mean_out = (out_norm_sums[name] / n) if n > 0 else 0.0
        print(f"    {name:<10} {pc:>8} {pos:>10} "
              f"{mean_in:>12.3f} {mean_out:>13.3f}",
              flush=True)


if __name__ == "__main__":
    main()
else:
    main()
