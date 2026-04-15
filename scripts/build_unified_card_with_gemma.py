"""Build a unified CHRLM card containing Gemma 4 E4B + compiled adder +
trained substrate stream — all in one .pt file.

This is the end-to-end demo of "Gemma lives IN the unified CHRLM."
Before this, they were separate modules routed at the application
level. After this, they're streams inside one nn.Module with one
state_dict on disk.

Pipeline:
  1. Load Gemma 4 E4B from its existing tq4 GGUF
  2. Build a compiled adder_tiny stream (d_head=2)
  3. Instantiate a trainable substrate stream
  4. Wrap all three in UnifiedCHRLMCard
  5. Save as single .pt
  6. Reload and verify everything works

Output: /tmp/unified_chrlm_card.pt (~5 GB — Gemma dominates)

Not in this demo (deferred):
  - Cross-stream joins actually wired into forward
  - Shared vocab / head reconciliation
  - Training signal across streams

Purpose: prove the CONTAINMENT works. Composition work follows.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from calm.llm_computer.gemma4_stream import load_gemma4_stream_from_gguf
from calm.llm_computer.unified_card import (
    UnifiedCHRLMCard, add_forward_residual_to_gemma4,
)


def build_compiled_adder_stream():
    """A tiny stream containing the compiled adder. Its .forward returns
    logits for the adder task (a+b at position 1)."""
    import torch.nn as nn
    from scripts.experiment_fast_weights_fusion import (
        build_adder_tiny_small2d,
    )
    # adder_tiny is already a Small2DTransformer; wrap it with a residual
    # method so UnifiedCHRLMCard can call forward_all on it.
    adder = build_adder_tiny_small2d(target_layer=0, n_layers=1)
    # Monkey-patch forward_residual (runs up to but not including head)
    import types
    def forward_residual(self, input_ids):
        # Small2DTransformer forward: embed + N layers + head
        # We want everything before head.
        B, S = input_ids.shape
        pos_idx = torch.arange(S, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos_idx)
        cfg = self.config
        for layer in range(cfg.n_layers):
            import torch.nn.functional as F
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, cfg.n_heads, cfg.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k) / (cfg.d_head ** 0.5)
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            if cfg.use_hard_max:
                idx_max = scores.argmax(dim=-1, keepdim=True)
                w = torch.zeros_like(scores)
                w.scatter_(-1, idx_max, 1.0)
            else:
                w = F.softmax(scores, dim=-1)
            attn = torch.einsum("bhij,bhjd->bhid", w, v).transpose(1, 2).reshape(B, S, cfg.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return x
    adder.forward_residual = types.MethodType(forward_residual, adder)
    return adder


def build_substrate_stream():
    """A tiny trained-substrate placeholder. Real use would load
    SubstrateLM or SubstrateHRLM weights here."""
    from calm.llm_computer.model import Small2DConfig, Small2DTransformer
    cfg = Small2DConfig(
        vocab_size=16, d_model=16, n_heads=8, n_layers=2,
        d_ffn=32, max_len=4, use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    import types, torch.nn.functional as F
    def forward_residual(self, input_ids):
        B, S = input_ids.shape
        pos_idx = torch.arange(S, device=input_ids.device)
        x = self.tok(input_ids) + self.pos(pos_idx)
        for layer in range(self.config.n_layers):
            qkv = self.W_qkv[layer](x)
            qkv = qkv.reshape(B, S, 3, self.config.n_heads, self.config.d_head)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k) / (self.config.d_head ** 0.5)
            mask = torch.triu(torch.ones(S, S, dtype=torch.bool, device=x.device), diagonal=1)
            scores = scores.masked_fill(mask, float("-inf"))
            w = F.softmax(scores, dim=-1)
            attn = torch.einsum("bhij,bhjd->bhid", w, v).transpose(1, 2).reshape(B, S, self.config.d_model)
            x = x + self.W_out[layer](attn)
            gate, val = self.ff_in[layer](x).chunk(2, dim=-1)
            x = x + self.ff_out[layer](F.relu(gate) * val)
        return x
    m.forward_residual = types.MethodType(forward_residual, m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", default="/home/gabe/models/gemma-4-E4B-it-tq4-aligned.gguf")
    ap.add_argument("--out", default="/tmp/unified_chrlm_card.pt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip-gemma", action="store_true",
                    help="don't load Gemma (for fast structural test)")
    args = ap.parse_args()

    print("=== building unified CHRLM card ===", flush=True)
    print(f"  out: {args.out}", flush=True)

    streams = {}

    # 1. Compiled adder (fast)
    print("\n--- building compiled adder stream ---", flush=True)
    streams["adder"] = build_compiled_adder_stream()
    print(f"  adder: {sum(p.numel() for p in streams['adder'].parameters())} params",
          flush=True)

    # 2. Substrate (fast)
    print("\n--- building substrate stream ---", flush=True)
    streams["substrate"] = build_substrate_stream()
    print(f"  substrate: {sum(p.numel() for p in streams['substrate'].parameters())} params",
          flush=True)

    # 3. Gemma 4 E4B (slow — ~90s load)
    if not args.skip_gemma:
        print(f"\n--- loading Gemma 4 E4B from {args.gguf} ---", flush=True)
        t0 = time.time()
        gemma = load_gemma4_stream_from_gguf(args.gguf, device=args.device)
        add_forward_residual_to_gemma4(gemma)
        print(f"  gemma: {sum(p.numel() for p in gemma.parameters()):,} params "
              f"loaded in {time.time()-t0:.1f}s", flush=True)
        streams["gemma"] = gemma

    # 4. Wrap
    card = UnifiedCHRLMCard(streams)
    print(f"\n=== unified card built ===", flush=True)
    print(f"  streams: {card.stream_names()}", flush=True)
    print(f"  total params: {card.total_param_count():,}", flush=True)
    for name, n in card.per_stream_param_counts().items():
        print(f"    {name}: {n:,}", flush=True)

    # 5. Verify each stream forwards
    print("\n--- verifying each stream ---", flush=True)
    adder_input = torch.tensor([[1, 2]], dtype=torch.long)
    with torch.no_grad():
        adder_out = card.forward(adder_input, stream="adder")
    print(f"  adder(1,2): argmax at pos 1 = {adder_out[0, 1, :7].argmax().item()} "
          f"(expected 3)", flush=True)

    subs_input = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    with torch.no_grad():
        subs_out = card.forward(subs_input, stream="substrate")
    print(f"  substrate: output shape {tuple(subs_out.shape)}", flush=True)

    if "gemma" in streams:
        gemma_input = torch.tensor([[1, 100, 200, 42]], dtype=torch.long)
        with torch.no_grad():
            gemma_out = card.forward(gemma_input, stream="gemma")
        print(f"  gemma: output shape {tuple(gemma_out.shape)} "
              f"finite={torch.isfinite(gemma_out).all().item()}", flush=True)

    # 6. Save as one file
    print(f"\n--- saving to {args.out} ---", flush=True)
    t0 = time.time()
    card.save(args.out)
    size_gb = Path(args.out).stat().st_size / (1024**3)
    print(f"  wrote {size_gb:.2f} GB in {time.time()-t0:.1f}s", flush=True)

    # 7. Reload and verify
    print("\n--- reloading from file ---", flush=True)
    builders = {
        "adder":     build_compiled_adder_stream,
        "substrate": build_substrate_stream,
    }
    if "gemma" in streams:
        builders["gemma"] = lambda: load_gemma4_stream_from_gguf(
            args.gguf, device=args.device,
        )

    t0 = time.time()
    card2 = UnifiedCHRLMCard.load(args.out, builders)
    print(f"  loaded in {time.time()-t0:.1f}s", flush=True)
    print(f"  streams: {card2.stream_names()}", flush=True)

    # Adder still works after reload
    with torch.no_grad():
        rt_out = card2.forward(adder_input, stream="adder")
    print(f"  reloaded adder(1,2): {rt_out[0, 1, :7].argmax().item()} (expected 3)",
          flush=True)

    print("\n=== SUCCESS — unified CHRLM card built, saved, reloaded ===",
          flush=True)
    print(f"  one .pt, multiple streams, shared state_dict", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
