"""Gate-graph → Small2DTransformer weights compiler.

Walks the hardware-node subset of a `GateGraph` and populates every weight
tensor of a `Small2DTransformer` accordingly. No training — the weights
ARE the compiled program.

Session 26 scope: handles `TokenEmbed`, `PosEmbed`, `LookUp` (copy-from-
pos-0 form), `ReGLU`, `LinearHead`, and the legacy `TokenInput`/
`TokenOutput` shorthand. Every hand-wired program in `programs/*.py`
can be rewritten as a gate graph and produce bit-identical weights.

The IR is fully declarative: every node names residual channels by index,
so layout is explicit (no implicit `d_model`-splitting magic). The caller
supplies `d_model`, `n_heads`, `n_layers`, `d_ffn`, and the compiler wires
the tensors one entry at a time.

Not yet done (deferred to Round 4 Layer 3+):
  - Parabolic-key lookups (exact past-position selection by key).
  - Automatic slot allocation / interval coloring — today the caller
    picks residual-channel indices.
  - MILP gate scheduling across layers — today the caller picks layers.
"""

from __future__ import annotations

from typing import Optional, Sequence

import torch

from calm.llm_computer.gate_graph import (
    ChannelLC, GateGraph, LinearHead, LookUp, LookUpExact, Node, PosEmbed,
    ReGLU, TokenEmbed, TokenInput, TokenOutput,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


def compile_program(
    graph: GateGraph,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ffn: int,
    max_len: int = 32,
    vocab_size: Optional[int] = None,
    use_hard_max: bool = True,
) -> Small2DTransformer:
    """Build a Small2DTransformer and populate its weights from `graph`.

    `d_head = d_model // n_heads` is asserted to be 2 (the paper's
    architectural constraint).

    Every weight starts at zero; the graph walks contribute only what
    they declare. Unreachable weights stay zero, which is analytically
    correct for the compiled program.
    """
    vocab = vocab_size or graph.vocab_size
    if vocab <= 0:
        raise ValueError("compile_program requires a positive vocab_size")

    cfg = Small2DConfig(
        vocab_size=vocab,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        use_hard_max=use_hard_max,
    )
    assert cfg.d_head == 2, f"expected d_head=2, got {cfg.d_head}"

    model = Small2DTransformer(cfg)
    # Per-layer ReGLU neuron counter. Each ReGLU in the graph consumes the
    # next free neuron slot at its declared layer, in insertion order.
    reglu_counts = [0] * n_layers
    # Per-layer LookUp head counter (same idea for attention heads).
    head_counts = [0] * n_layers

    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        for node in graph.nodes:
            if isinstance(node, TokenEmbed):
                _apply_token_embed(model, node)
            elif isinstance(node, PosEmbed):
                _apply_pos_embed(model, node)
            elif isinstance(node, LookUp):
                _apply_lookup(model, node, cfg, head_counts)
            elif isinstance(node, LookUpExact):
                _apply_lookup_exact(model, node, cfg, head_counts)
            elif isinstance(node, ReGLU):
                _apply_reglu(model, node, cfg, reglu_counts)
            elif isinstance(node, LinearHead):
                _apply_linear_head(model, node)
            elif isinstance(node, (TokenInput, TokenOutput)):
                # Legacy Layer-1 shorthand — TokenOutput.matrix populates
                # the head directly, everything else stays zero.
                if isinstance(node, TokenOutput) and node.matrix is not None:
                    if node.matrix.shape != (vocab, d_model):
                        raise ValueError(
                            f"TokenOutput.matrix shape {tuple(node.matrix.shape)} "
                            f"!= (vocab={vocab}, d_model={d_model})"
                        )
                    model.head.weight.copy_(node.matrix)
            # Other nodes (Const/BinOp/Delegate/Result) are compute-only
            # and don't contribute to weights.

    return model


# ----- helpers: one per hardware-node kind ----------------------------


def _apply_token_embed(model: Small2DTransformer, node: TokenEmbed) -> None:
    for k, ch, coef in node.entries:
        model.tok.weight[k, ch] = coef


def _apply_pos_embed(model: Small2DTransformer, node: PosEmbed) -> None:
    for p, ch, coef in node.entries:
        model.pos.weight[p, ch] = coef


def _apply_lookup(model: Small2DTransformer, node: LookUp, cfg: Small2DConfig,
                   head_counts: list) -> None:
    """Copy-from-pos-0 form: keys/queries zero → first-tie argmax picks
    past position 0. V projection copies the designated residual channels
    into the attention output; W_out routes that attention output into
    the designated output channels.

    Each declared `(v_source_channel, out_channel)` pair consumes one
    attention head at `node.layer`. Heads are allocated sequentially via
    `head_counts[layer]`. The head picks up residual[src_ch] into its
    v-dim 0, and W_out routes dim (2 * head_idx) → out_ch.
    """
    if len(node.v_source_channels) != len(node.out_channels):
        raise ValueError(
            f"LookUp: v_source_channels ({len(node.v_source_channels)}) "
            f"and out_channels ({len(node.out_channels)}) must match"
        )
    D = cfg.d_model
    v_chunk_start = 2 * D
    for src_ch, out_ch in zip(node.v_source_channels, node.out_channels):
        head_idx = head_counts[node.layer]
        if head_idx >= cfg.n_heads:
            raise ValueError(
                f"LookUp at layer {node.layer}: head {head_idx} >= n_heads {cfg.n_heads}"
            )
        model.W_qkv[node.layer].weight[v_chunk_start + 2 * head_idx, src_ch] = 1.0
        model.W_out[node.layer].weight[out_ch, 2 * head_idx] = 1.0
        head_counts[node.layer] += 1


def _apply_lookup_exact(model: Small2DTransformer, node: LookUpExact,
                         cfg: Small2DConfig, head_counts: list) -> None:
    """Parabolic-key attention: q · k_j = key² - (j - key)², argmax at j=key.

    One head per `(value_source, out_channel)` pair. Per-head wiring:
      - q[0] = residual[query_key_channel]
      - q[1] = residual[bias_channel]                 (constant 1)
      - k[0] = residual[pos_key0_channel]             (carries 2p)
      - k[1] = residual[pos_key1_channel]             (carries -p²)
      - v[0] = residual[value_source_channel]
      - W_out[out_channel, 2*head_idx] = 1.0          (route attn[0] out)

    d_head = 2 is the paper's architectural constraint, enforced in
    `compile_program`.
    """
    if len(node.value_source_channels) != len(node.out_channels):
        raise ValueError(
            f"LookUpExact: value_source_channels "
            f"({len(node.value_source_channels)}) and out_channels "
            f"({len(node.out_channels)}) must match"
        )
    D = cfg.d_model
    q_start = 0
    k_start = D
    v_start = 2 * D
    for v_src, out_ch in zip(node.value_source_channels, node.out_channels):
        head_idx = head_counts[node.layer]
        if head_idx >= cfg.n_heads:
            raise ValueError(
                f"LookUpExact at layer {node.layer}: head {head_idx} >= "
                f"n_heads {cfg.n_heads}"
            )
        # q projection (with coefs so callers can scale).
        model.W_qkv[node.layer].weight[q_start + 2 * head_idx + 0,
                                        node.query_key_channel] = node.query_key_coef
        model.W_qkv[node.layer].weight[q_start + 2 * head_idx + 1,
                                        node.bias_channel] = node.bias_coef
        # k projection (coefs let semantic-keyed use scale 2.0 on a scalar
        # key channel rather than requiring a precomputed 2*key table).
        model.W_qkv[node.layer].weight[k_start + 2 * head_idx + 0,
                                        node.pos_key0_channel] = node.pos_key0_coef
        model.W_qkv[node.layer].weight[k_start + 2 * head_idx + 1,
                                        node.pos_key1_channel] = node.pos_key1_coef
        # v projection: pulls the requested value channel into v[0].
        model.W_qkv[node.layer].weight[v_start + 2 * head_idx + 0,
                                        v_src] = 1.0
        # W_out: route attn[2*head] → out_ch.
        model.W_out[node.layer].weight[out_ch, 2 * head_idx] = 1.0
        head_counts[node.layer] += 1


def _apply_reglu(model: Small2DTransformer, node: ReGLU, cfg: Small2DConfig,
                  reglu_counts: list) -> None:
    """One FFN neuron consuming `reglu_counts[layer]` (allocated sequentially).

    ff_in outputs `[gate_0..gate_{d_ffn-1}, val_0..val_{d_ffn-1}]` (chunk
    layout from the model's forward pass). ff_in[layer].weight[neuron]
    gates and ff_in[layer].weight[d_ffn + neuron] vals; ff_out[layer] sums
    via ReGLU products into the output channel.
    """
    layer = node.layer
    neuron = reglu_counts[layer]
    if neuron >= cfg.d_ffn:
        raise ValueError(
            f"ReGLU at layer {layer}: neuron index {neuron} >= d_ffn {cfg.d_ffn}"
        )
    for ch, coef in node.gate:
        model.ff_in[layer].weight[neuron, ch] += coef
    for ch, coef in node.val:
        model.ff_in[layer].weight[cfg.d_ffn + neuron, ch] += coef
    model.ff_out[layer].weight[node.output_channel, neuron] += node.output_coef
    reglu_counts[layer] += 1


def _apply_linear_head(model: Small2DTransformer, node: LinearHead) -> None:
    # Accumulate like ReGLU — multiple program-level contributions to the
    # same logit slot must sum, not overwrite. Programs with unique entries
    # (add_one, adder, adder_tiny, threshold, read_by_key) are unaffected;
    # programs that naturally collide (gcd and other binary-LUTs where
    # several input pairs share an output value) rely on this accumulation.
    for k, ch, coef in node.entries:
        model.head.weight[k, ch] += coef


# ----- backward-compat: pre-session-26 API (add_one still uses this) --


def compile_graph(graph: GateGraph,
                  max_len: int = 32,
                  n_layers: int = 2) -> Small2DTransformer:
    """Layer-1 compiler — exists for backward compat.

    Assumes exactly 1 `TokenInput` + 1 `TokenOutput` with optional linear
    transformation matrix. Attention/FFN zeroed. New code should prefer
    `compile_program` with explicit hardware nodes.
    """
    inputs = graph.inputs()
    outputs = graph.outputs()
    if len(inputs) != 1 or len(outputs) != 1:
        raise ValueError("compile_graph expects exactly 1 input and 1 output")
    token_in = inputs[0]
    token_out = outputs[0]

    vocab = graph.vocab_size or token_in.vocab_size or token_out.vocab_size
    if vocab <= 0:
        raise ValueError("graph vocab_size must be > 0")
    if vocab % 2 != 0:
        raise ValueError("compile_graph requires even vocab (d_head=2 constraint)")

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
        for p in model.parameters():
            p.zero_()
        model.tok.weight.copy_(torch.eye(vocab))
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
