"""Program builder — facade + import system for compiled neural programs.

A programming language for transformer weights:
  * `StdLib` — layer-0 facade that sets up shared primitives (tok embed,
    pos embed, LookUp copies). Exports named channels.
  * `CompiledOp` — a single compiled operation that declares imports
    (channels it reads) and exports (channels it writes). Each op becomes
    ReGLU neurons at a specific layer.
  * `HeadSpec` — head wiring mapping channels → output vocab slots.
  * `build_program(stdlib, ops, head, d_model, vocab, ...)` — the linker.
    Resolves imports to channel numbers, auto-schedules layers (each op
    runs one layer after its deepest import), compiles all gates into one
    Small2DTransformer, returns a working model.

Import resolution is channel-number lookup. Layer scheduling is
topological sort on the import graph. The linker IS merge_cards() under
the hood — each op's weights are zero outside its layer, so addition
is conflict-free.

Usage:
    stdlib = StdLib(vocab_size=8, max_len=4, exports={
        "a": 3, "b": 4, "bias": 1,
    }, copy_pairs=[("a", 0, 3)], token_channel=0, bias_channel=1)

    adder = CompiledOp(
        name="adder",
        imports={"x": "a", "y": "b", "bias": "bias"},
        gate=lambda ch: [(ch["bias"], 1.0)],
        val=lambda ch: [(ch["x"], 1.0), (ch["y"], 1.0)],
        export_channel=5,
        export_name="sum",
    )

    model = build_program(stdlib, [adder, ...], head, d_model=16, ...)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer


@dataclass
class StdLib:
    """Layer-0 facade — sets up shared primitives that all programs import.

    Exports: dict mapping name → channel index.
    copy_pairs: list of (export_name, source_pos, out_channel) — each
        becomes a LookUp sub-head copying from source_pos's token channel.
    token_channel: which channel holds the token scalar (via TokenEmbed).
    bias_channel: which channel holds the constant 1 (via PosEmbed).
    """
    vocab_size: int
    max_len: int
    exports: Dict[str, int]
    copy_pairs: List[Tuple[str, int, int]] = field(default_factory=list)
    token_channel: int = 0
    bias_channel: int = 1


@dataclass
class CompiledOp:
    """One compiled operation — declares imports and exports.

    imports: dict mapping local_name → stdlib_or_op export name.
        At build time, each name is resolved to a channel number.
    gate: callable(resolved_channels) → ChannelLC for the ReGLU gate.
    val: callable(resolved_channels) → ChannelLC for the ReGLU val.
    export_channel: substrate channel to write output.
    export_name: name other ops can import this output by.
    output_coef: ReGLU output coefficient (default 1.0).
    n_neurons: how many ReGLU neuron PAIRS (hi/lo) for step functions.
        Default 1 = a single ReGLU. For step functions, provide
        `step_thresholds` instead.
    step_thresholds: if set, generates step-function pairs at each
        threshold (hi/lo per threshold, all writing to export_channel).
    """
    name: str
    imports: Dict[str, str]
    gate: Optional[Callable] = None
    val: Optional[Callable] = None
    export_channel: int = 0
    export_name: str = ""
    output_coef: float = 1.0
    step_thresholds: Optional[List[int]] = None


@dataclass
class HeadSpec:
    """Head wiring: maps channel values → output vocab slots."""
    entries: List[Tuple[int, int, float]]  # (slot, channel, coef)


def _resolve_imports(
    op: CompiledOp,
    channel_map: Dict[str, int],
) -> Dict[str, int]:
    """Resolve op's imports to channel numbers via the global channel map."""
    resolved = {}
    for local_name, source_name in op.imports.items():
        if source_name not in channel_map:
            raise KeyError(
                f"op '{op.name}' imports '{source_name}' (as '{local_name}') "
                f"but it's not exported by any prior op or stdlib. "
                f"Available: {list(channel_map.keys())}"
            )
        resolved[local_name] = channel_map[source_name]
    return resolved


def _schedule_layers(
    ops: List[CompiledOp],
    channel_map: Dict[str, int],
    export_to_op: Dict[str, str],
    op_layers: Dict[str, int],
) -> Dict[str, int]:
    """Topological schedule: each op runs one layer after its deepest
    dependency. StdLib is layer 0. First op with no op-dependencies is
    layer 1."""
    for op in ops:
        max_dep_layer = 0  # stdlib is layer 0
        for _, source_name in op.imports.items():
            if source_name in export_to_op:
                dep_op_name = export_to_op[source_name]
                if dep_op_name not in op_layers:
                    raise ValueError(
                        f"op '{op.name}' depends on '{dep_op_name}' which "
                        f"hasn't been scheduled yet — ops must be in "
                        f"dependency order"
                    )
                max_dep_layer = max(max_dep_layer, op_layers[dep_op_name])
        op_layers[op.name] = max_dep_layer + 1
    return op_layers


def build_program(
    stdlib: StdLib,
    ops: List[CompiledOp],
    head: HeadSpec,
    d_model: int,
    d_ffn: int = 8,
) -> Small2DTransformer:
    """The linker: resolve imports, schedule layers, compile, merge.

    Returns a single Small2DTransformer containing all ops + stdlib +
    head wiring, ready for forward().
    """
    # Build channel map: name → channel index
    channel_map = dict(stdlib.exports)
    export_to_op: Dict[str, str] = {}
    for op in ops:
        if op.export_name:
            if op.export_name in channel_map:
                raise ValueError(
                    f"duplicate export '{op.export_name}' from op '{op.name}'"
                )
            channel_map[op.export_name] = op.export_channel
            export_to_op[op.export_name] = op.name

    # Schedule layers
    op_layers: Dict[str, int] = {}
    _schedule_layers(ops, channel_map, export_to_op, op_layers)
    n_layers = max(op_layers.values()) + 1 if op_layers else 1

    print(f"  [linker] channel map: {channel_map}")
    print(f"  [linker] layer schedule: {op_layers}")
    print(f"  [linker] n_layers: {n_layers}")

    n_heads = d_model // 2

    # Build the gate graph for the ENTIRE program
    graph = GateGraph(vocab_size=stdlib.vocab_size)

    # StdLib: tok embed + pos embed + LookUps
    graph.add(TokenEmbed(
        name="stdlib_tok",
        entries=[(k, stdlib.token_channel, float(k))
                 for k in range(stdlib.vocab_size)],
    ))
    pos_entries = [(p, stdlib.bias_channel, 1.0)
                   for p in range(stdlib.max_len)]
    graph.add(PosEmbed(name="stdlib_pos", entries=pos_entries))

    for export_name, source_pos, out_ch in stdlib.copy_pairs:
        graph.add(LookUp(
            name=f"stdlib_copy_{export_name}",
            layer=0,
            v_source_channels=[stdlib.token_channel],
            out_channels=[out_ch],
        ))

    # Compile each op's ReGLU neurons at its scheduled layer
    neuron_counter = 0
    for op in ops:
        resolved = _resolve_imports(op, channel_map)
        layer = op_layers[op.name]

        if op.step_thresholds is not None:
            # Step-function pair per threshold
            for t in op.step_thresholds:
                gate_lc = op.gate(resolved, t) if op.gate else []
                val_lc = op.val(resolved) if op.val else [(stdlib.bias_channel, 1.0)]
                graph.add(ReGLU(
                    name=f"{op.name}_step_{t}_hi",
                    layer=layer,
                    gate=gate_lc,
                    val=val_lc,
                    output_channel=op.export_channel,
                    output_coef=1.0,
                ))
                # lo: same gate but threshold shifted by 1
                gate_lo = op.gate(resolved, t, lo=True) if op.gate else []
                graph.add(ReGLU(
                    name=f"{op.name}_step_{t}_lo",
                    layer=layer,
                    gate=gate_lo,
                    val=val_lc,
                    output_channel=op.export_channel,
                    output_coef=-1.0,
                ))
                neuron_counter += 2
        else:
            # Single ReGLU
            gate_lc = op.gate(resolved) if op.gate else [(stdlib.bias_channel, 1.0)]
            val_lc = op.val(resolved) if op.val else [(stdlib.bias_channel, 1.0)]
            graph.add(ReGLU(
                name=f"{op.name}_main",
                layer=layer,
                gate=gate_lc,
                val=val_lc,
                output_channel=op.export_channel,
                output_coef=op.output_coef,
            ))
            neuron_counter += 1

    # Head wiring
    graph.add(LinearHead(name="program_head", entries=head.entries))

    # Auto-schedule and compile
    from calm.llm_computer.schedule import auto_schedule
    n_layers_actual = auto_schedule(graph)
    d_ffn_actual = max(d_ffn, neuron_counter * 2)

    return compile_program(
        graph,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers_actual,
        d_ffn=d_ffn_actual,
        max_len=stdlib.max_len,
        vocab_size=stdlib.vocab_size,
    )


# --- Convenience: step-function gate factories ---

def step_gate_ge(input_ch_name: str, bias_name: str = "bias"):
    """Returns a gate factory for step functions: 1[input >= threshold].
    The factory is called with (resolved_channels, threshold, lo=False)."""
    def factory(ch, threshold=0, lo=False):
        t = threshold if not lo else threshold + 1
        return [(ch[input_ch_name], 1.0), (ch[bias_name], -(t - 1))]
    return factory


def simple_val(channel_name: str, coef: float = 1.0):
    """Returns a val factory: reads one channel."""
    def factory(ch):
        return [(ch[channel_name], coef)]
    return factory


def sum_val(*names_and_coefs):
    """Returns a val factory: linear combination of named channels."""
    def factory(ch):
        return [(ch[n], c) for n, c in names_and_coefs]
    return factory


def bias_val():
    """Val factory: just reads bias (constant 1)."""
    def factory(ch):
        return [(ch["bias"], 1.0)]
    return factory
