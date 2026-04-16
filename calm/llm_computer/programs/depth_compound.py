"""Depth-compounding: 3 compiled operations chaining across layers
on the SAME sub-head, via residual channels.

One "software branch" — sub-heads [0, 5) — executes a 3-stage pipeline:

  Layer 0:  a + b → channel SUM           (addition)
  Layer 1:  SUM * 2 → channel DOUBLE      (scaling, reads layer 0's output)
  Layer 2:  DOUBLE >= 8 ? → head slot      (classification, reads layer 1's output)

Three independently-compiled stages, merged via weight addition into
ONE Small2DTransformer. Each stage only populates ITS layer's weights;
other layers are zero. The residual stream carries intermediate values
between stages — exactly like registers in a CPU pipeline.

Expected behavior: output slot 1 ("large") when a + b >= 4 (because
2*(a+b) >= 8), output slot 0 ("small") otherwise. For a, b ∈ [0, 3]:
6/16 cases are "large", 10/16 are "small".

This is the proof that compiled programs COMPOUND across depth (layers),
not just width (sub-heads). One sub-head thread, three operations,
one forward pass. Each layer builds on the previous layer's output.
"""

from __future__ import annotations

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


D_MODEL = 12
N_HEADS = D_MODEL // 2
N_LAYERS = 3
D_FFN = 4
MAX_LEN = 4
VOCAB = 8
MAX_OPERAND = 3

# Channel assignments (the "registers" of this pipeline)
CH_OWN = 0       # token scalar
CH_BIAS = 1      # constant 1
CH_COPY_A = 2    # a copied from pos 0
CH_SUM = 3       # a + b (written by layer 0)
CH_DOUBLE = 4    # 2 * (a + b) (written by layer 1)
CH_IS_LARGE = 5  # 1 if DOUBLE >= 8 (written by layer 2)


def build_stage_0_adder() -> Small2DTransformer:
    """Layer 0: compute a + b → CH_SUM. Uses LookUp + 1 ReGLU."""
    graph = GateGraph(vocab_size=VOCAB)
    graph.add(TokenEmbed(
        name="tok", entries=[(k, CH_OWN, float(k)) for k in range(VOCAB)],
    ))
    graph.add(PosEmbed(
        name="bias", entries=[(p, CH_BIAS, 1.0) for p in range(MAX_LEN)],
    ))
    graph.add(LookUp(
        name="copy_a", layer=0,
        v_source_channels=[CH_OWN], out_channels=[CH_COPY_A],
    ))
    graph.add(ReGLU(
        name="sum_write", layer=0,
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_A, 1.0), (CH_OWN, 1.0)],
        output_channel=CH_SUM,
        output_coef=1.0,
    ))
    graph.add(LinearHead(name="zero_head", entries=[]))
    return compile_program(
        graph, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ffn=D_FFN, max_len=MAX_LEN, vocab_size=VOCAB,
    )


def build_stage_1_doubler() -> Small2DTransformer:
    """Layer 1: read CH_SUM, write 2*SUM → CH_DOUBLE.
    No tok/pos — stage 0 already set up bias. Only FFN weights."""
    graph = GateGraph(vocab_size=VOCAB)
    graph.add(TokenEmbed(name="tok_1", entries=[]))
    graph.add(PosEmbed(name="pos_1", entries=[]))  # empty — stage 0 owns bias
    graph.add(ReGLU(
        name="double", layer=1,
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_SUM, 2.0)],
        output_channel=CH_DOUBLE,
        output_coef=1.0,
    ))
    graph.add(LinearHead(name="zero_head_1", entries=[]))
    return compile_program(
        graph, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ffn=D_FFN, max_len=MAX_LEN, vocab_size=VOCAB,
    )


def build_stage_2_classifier() -> Small2DTransformer:
    """Layer 2: read CH_DOUBLE, classify DOUBLE >= 8 → head slots.
    No tok/pos — stage 0 owns bias."""
    graph = GateGraph(vocab_size=VOCAB)
    graph.add(TokenEmbed(name="tok_2", entries=[]))
    graph.add(PosEmbed(name="pos_2", entries=[]))  # empty
    # step_8(DOUBLE) = ReLU(DOUBLE - 7) - ReLU(DOUBLE - 8)
    graph.add(ReGLU(
        name="step_hi", layer=2,
        gate=[(CH_DOUBLE, 1.0), (CH_BIAS, -7.0)],
        val=[(CH_BIAS, 1.0)],
        output_channel=CH_IS_LARGE,
        output_coef=1.0,
    ))
    graph.add(ReGLU(
        name="step_lo", layer=2,
        gate=[(CH_DOUBLE, 1.0), (CH_BIAS, -8.0)],
        val=[(CH_BIAS, 1.0)],
        output_channel=CH_IS_LARGE,
        output_coef=-1.0,
    ))
    graph.add(LinearHead(
        name="classify_head",
        entries=[
            (0, CH_BIAS, 1.0),        # slot 0 = 1 (always)
            (1, CH_IS_LARGE, 2.0),    # slot 1 = 2 if large, 0 if not
        ],
    ))
    return compile_program(
        graph, d_model=D_MODEL, n_heads=N_HEADS, n_layers=N_LAYERS,
        d_ffn=D_FFN, max_len=MAX_LEN, vocab_size=VOCAB,
    )


def merge_stages(*stages: Small2DTransformer) -> Small2DTransformer:
    """Merge N stages into one model via weight addition. Each stage only
    populates its own layer's weights — other layers are zero from
    compile_program's zero-init. Addition is conflict-free."""
    cfg = stages[0].config
    merged = Small2DTransformer(Small2DConfig(
        vocab_size=cfg.vocab_size, d_model=cfg.d_model, n_heads=cfg.n_heads,
        n_layers=cfg.n_layers, d_ffn=cfg.d_ffn, max_len=cfg.max_len,
        use_hard_max=cfg.use_hard_max,
    ))
    with torch.no_grad():
        for p in merged.parameters():
            p.zero_()
        for stage in stages:
            for (_, pm), (_, ps) in zip(
                merged.named_parameters(), stage.named_parameters()
            ):
                pm.add_(ps)
    return merged


if __name__ == "__main__":
    import itertools

    print("[depth] building 3-stage compiled pipeline...")
    s0 = build_stage_0_adder()
    s1 = build_stage_1_doubler()
    s2 = build_stage_2_classifier()
    print(f"  stage 0 (adder):      layer 0, writes CH_SUM={CH_SUM}")
    print(f"  stage 1 (doubler):    layer 1, reads CH_SUM, writes CH_DOUBLE={CH_DOUBLE}")
    print(f"  stage 2 (classifier): layer 2, reads CH_DOUBLE, writes CH_IS_LARGE={CH_IS_LARGE}")

    print("[depth] merging stages into ONE model (weight addition)...")
    model = merge_stages(s0, s1, s2)
    print(f"  d_model={D_MODEL}, n_layers={N_LAYERS}, params={model.param_count()}")

    # Exhaustive test
    print("\n[depth] exhaustive test: a+b >= 4 → slot 1, else → slot 0")
    ok = 0
    total = 0
    for a, b in itertools.product(range(MAX_OPERAND + 1), repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        pred = int(logits[0, 1].argmax().item())
        s = a + b
        expected = 1 if s >= 4 else 0
        mark = "✓" if pred == expected else "✗"
        if pred != expected:
            print(f"  [{mark}] a={a} b={b} sum={s} double={2*s} "
                  f"pred={pred} expected={expected}")
        ok += (pred == expected)
        total += 1
    print(f"  result: {ok}/{total} — {'PASS' if ok == total else 'FAIL'}")

    # Trace intermediate channels to show the pipeline
    print("\n[depth] pipeline trace for a=2, b=3 (sum=5, double=10, large=YES):")
    x = torch.tensor([[2, 3]], dtype=torch.long)
    with torch.no_grad():
        cfg = model.config
        pos_idx = torch.arange(2)
        res = model.tok(x) + model.pos(pos_idx)
        mask = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
        for layer in range(N_LAYERS):
            qkv = model.W_qkv[layer](res)
            qkv = qkv.reshape(1, 2, 3, N_HEADS, 2)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            scores = scores.masked_fill(mask, float("-inf"))
            idx_hm = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx_hm, 1.0)
            attn = torch.einsum("bhij,bhjd->bhid", weights, v)
            attn = attn.transpose(1, 2).reshape(1, 2, D_MODEL)
            res = res + model.W_out[layer](attn)
            import torch.nn.functional as F
            gate, val = model.ff_in[layer](res).chunk(2, dim=-1)
            res = res + model.ff_out[layer](F.relu(gate) * val)
            # Print pipeline registers at pos 1 after each layer
            r = res[0, 1]
            print(f"  after layer {layer}: "
                  f"CH_SUM={r[CH_SUM].item():.1f} "
                  f"CH_DOUBLE={r[CH_DOUBLE].item():.1f} "
                  f"CH_IS_LARGE={r[CH_IS_LARGE].item():.1f}")

    print(f"\n[depth] 3 stages × 1 sub-head thread × {N_LAYERS} layers = "
          f"compound program")
    print("[depth] each layer builds on previous layer's residual output")
    print("[depth] like a CPU pipeline: layer = instruction, "
          "channel = register, sub-head = thread")
