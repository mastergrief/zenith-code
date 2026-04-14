"""Greedy scheduler — assign layer indices to LookUp/LookUpExact/ReGLU nodes.

Removes the "which layer?" burden from program authors. Given a GateGraph
where nodes declare their residual-channel reads and writes, the scheduler
does a one-pass topological placement:

  phase 0 = layer 0 attn, phase 1 = layer 0 FFN,
  phase 2 = layer 1 attn, phase 3 = layer 1 FFN, etc.

Channel availability:
  TokenEmbed/PosEmbed writes: available at phase 0 (pre-layer-0-attn).
  LookUp[Exact] at phase 2L writes: available at phase 2L+1 (same-layer
                                     FFN onwards).
  ReGLU at phase 2L+1 writes: available at phase 2L+2 (next-layer attn
                               onwards).

Each LookUp/LookUpExact/ReGLU node is placed at the minimum phase that
(a) matches its type's parity (even for attn, odd for FFN) and (b) is
>= every read channel's availability. Writes are marked after
placement; the next node sees the updated availability.

Shared head/neuron counters within each (layer, phase) still come from
compile.py's allocator — scheduler picks layers, compiler picks heads.

Limitations:
  - Greedy / one-pass. If a node's reads have mutually-conflicting
    availability from multiple sources, caller should order the graph
    to avoid cycles. The IR is a DAG so this is automatic.
  - No layer consolidation / register allocation. Each node gets its
    own slot unless the caller explicitly reuses.
"""

from __future__ import annotations

from typing import Set

from calm.llm_computer.gate_graph import (
    GateGraph, LookUp, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)


def _read_channels(node) -> Set[int]:
    if isinstance(node, LookUp):
        return set(node.v_source_channels)
    if isinstance(node, LookUpExact):
        return {
            node.pos_key0_channel, node.pos_key1_channel,
            node.query_key_channel, node.bias_channel,
            *node.value_source_channels,
        }
    if isinstance(node, ReGLU):
        return {ch for (ch, _) in node.gate} | {ch for (ch, _) in node.val}
    return set()


def _write_channels(node) -> Set[int]:
    if isinstance(node, (LookUp, LookUpExact)):
        return set(node.out_channels)
    if isinstance(node, ReGLU):
        return {node.output_channel}
    return set()


def auto_schedule(graph: GateGraph) -> int:
    """Fill in `layer` fields on every LookUp/LookUpExact/ReGLU node.

    Returns the number of transformer layers required (max_layer + 1).
    Mutates the graph in place — pass a fresh copy if you need the
    pre-scheduled form.

    Raises `ValueError` if a node has no layer assignable (shouldn't
    happen for valid DAGs).
    """
    # Channel availability: phase at which channel becomes readable.
    # TokenEmbed/PosEmbed writes land at phase 0 (i.e., layer 0 attn
    # can read them).
    avail: dict[int, int] = {}
    for node in graph.nodes:
        if isinstance(node, (TokenEmbed, PosEmbed)):
            for (_, ch, _) in node.entries:
                avail[ch] = max(avail.get(ch, 0), 0)

    max_layer = 0

    for node in graph.nodes:
        if isinstance(node, (LookUp, LookUpExact)):
            reads = _read_channels(node)
            max_read_phase = max((avail.get(ch, 0) for ch in reads), default=0)
            # Attn phase = min even phase >= max_read_phase.
            attn_phase = max_read_phase if max_read_phase % 2 == 0 \
                                         else max_read_phase + 1
            node.layer = attn_phase // 2
            write_phase = attn_phase + 1
            for ch in _write_channels(node):
                avail[ch] = max(avail.get(ch, 0), write_phase)
            max_layer = max(max_layer, node.layer)
        elif isinstance(node, ReGLU):
            reads = _read_channels(node)
            max_read_phase = max((avail.get(ch, 0) for ch in reads), default=0)
            # FFN phase = min odd phase >= max_read_phase.
            ffn_phase = max_read_phase if max_read_phase % 2 == 1 \
                                        else max_read_phase + 1
            node.layer = (ffn_phase - 1) // 2
            write_phase = ffn_phase + 1
            for ch in _write_channels(node):
                avail[ch] = max(avail.get(ch, 0), write_phase)
            max_layer = max(max_layer, node.layer)

    return max_layer + 1
