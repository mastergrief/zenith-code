"""Gate-graph → Small2DTransformer weights compiler.

Layer 1 scope: compiles graphs that use only `TokenInput` → `TokenOutput`
with an optional linear transformation matrix. Enough for `add_one` and
`identity`. Attention / FFN primitives are added in later layers as we
exercise them.

Compilation recipe (Layer 1):
  - Token embedding = identity (tok[k] = e_k).
  - Position embedding = zero.
  - All attention Q/K/V/out matrices = zero → attention residual
    contribution is zero at every position.
  - All FFN in/out matrices = zero → FFN residual contribution is zero.
  - Final linear head = the `TokenOutput.matrix` (or identity if None).

Result: the forward pass reduces to
    logits[j] = head.weight @ (tok(idx[j]) + pos(j))
             = head.weight @ e_{idx[j]} + 0
             = head.weight[:, idx[j]]

So `head.weight[:, k]` column is the logits produced for input token k.
This is exactly what lets a hand-defined `TokenOutput.matrix` implement
any deterministic token → token function.
"""

from __future__ import annotations

import torch

from calm.llm_computer.gate_graph import GateGraph, TokenInput, TokenOutput
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


def compile_graph(graph: GateGraph,
                  max_len: int = 32,
                  n_layers: int = 2) -> Small2DTransformer:
    """Compile a GateGraph into a Small2DTransformer with set weights.

    Layer 1 assumptions:
      - Exactly one `TokenInput` and one `TokenOutput`.
      - `vocab_size` taken from the graph.
      - `d_model == vocab_size` so token embedding can be identity.
      - `n_heads = vocab_size // 2` to force `d_head = 2`.
    """
    inputs = graph.inputs()
    outputs = graph.outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("Layer 1 compiler expects exactly 1 input and 1 output")
    token_in = inputs[0]
    token_out = outputs[0]

    vocab = graph.vocab_size or token_in.vocab_size or token_out.vocab_size
    if vocab <= 0:
        raise ValueError("graph vocab_size must be > 0")
    if vocab % 2 != 0:
        raise ValueError("Layer 1 requires even vocab (d_head=2 constraint)")

    cfg = Small2DConfig(
        vocab_size=vocab,
        d_model=vocab,
        n_heads=vocab // 2,
        n_layers=n_layers,
        d_ffn=vocab,
        max_len=max_len,
        use_hard_max=True,
    )
    assert cfg.d_head == 2, f"expected d_head=2, got {cfg.d_head}"

    model = Small2DTransformer(cfg)
    with torch.no_grad():
        # Zero everything.
        for p in model.parameters():
            p.zero_()
        # Identity token embedding: tok[k] = e_k.
        model.tok.weight.copy_(torch.eye(vocab))
        # Position embedding stays zero (Layer 1 doesn't use position info).
        # Head: use the TokenOutput.matrix, else identity.
        if token_out.matrix is not None:
            if token_out.matrix.shape != (vocab, vocab):
                raise ValueError(
                    f"TokenOutput.matrix shape {tuple(token_out.matrix.shape)} "
                    f"!= (vocab={vocab}, vocab={vocab})"
                )
            model.head.weight.copy_(token_out.matrix)
        else:
            model.head.weight.copy_(torch.eye(vocab))
    return model
