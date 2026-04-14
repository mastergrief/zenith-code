"""Gate-graph IR.

Two flavors of node coexist in the same IR to support two compile targets:

**Compute nodes** (Round 4 Layer 2 — integration #3 via trace-as-program):
  - `Const` — integer constant.
  - `BinOp` — `add`, `sub`, `mul`, `div` on two source nodes.
  - `Delegate` — route a named function call (e.g. `gcd`, `factorial`) to
    a verified backend. The interpreter resolves this via `safe_eval`;
    a future compile target could route it to a different executor.
  - `Result` — a named output of the graph; holds a reference to the
    source node whose value is the final answer.

**Hardware nodes** (Round 4 Layer 1 — direct compile to transformer weights):
  - `TokenInput` / `TokenOutput` — for the Small2DTransformer pipeline.
  - `TokenEmbed` / `PosEmbed` — populate embedding tables.
  - `LookUp` — one attention head (first-tie "copy-from-pos-0" form for
    session 26; parabolic-key lookups in a later pass). Specifies which
    residual channels to read as the V projection and which channels to
    write the attention output into.
  - `ReGLU` — one FFN neuron: `out_channel += coef · val · ReLU(gate)`,
    where `gate` and `val` are linear combinations of residual channels.
    Section 4c/4d of RESEARCH/03.
  - `LinearHead` — final head weights, one `(token, channel, coef)`
    triple per wiring.

The compute-node subset is self-contained and independent of the
hardware-node subset. Today's interpreter walks only the compute nodes;
the `compile_program` path walks the hardware nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# --- Base ---

@dataclass
class Node:
    """Base node. Subclasses mark specific roles in the gate graph."""
    name: str


# --- Compute nodes (math / logic domain) ---

@dataclass
class Const(Node):
    """Integer constant."""
    value: int = 0


@dataclass
class BinOp(Node):
    """Binary arithmetic op on two source nodes."""
    op: str = "add"  # one of: "add", "sub", "mul", "div"
    left: Optional[Node] = None
    right: Optional[Node] = None


@dataclass
class Delegate(Node):
    """Delegate to a named backend function.

    `fn_name` is a bare Python-style identifier (`"gcd"`, `"factorial"`,
    `"is_prime"`, etc.) resolved by the interpreter via `safe_eval` —
    exactly the same function registry the HRM's `<call>/<end_call>`
    machinery routes to.

    `args` are other nodes whose values are evaluated first and passed
    as integer arguments. Function output type is whatever the backend
    returns (int, bool, str).
    """
    fn_name: str = ""
    args: List[Node] = field(default_factory=list)


@dataclass
class Result(Node):
    """Named graph output — holds the node whose value is the final answer."""
    source: Optional[Node] = None


# --- Hardware nodes (Small2DTransformer direct-compile path) ---

@dataclass
class TokenInput(Node):
    vocab_size: int = 0


@dataclass
class TokenOutput(Node):
    vocab_size: int = 0
    source: Optional[Node] = None
    matrix: Optional[torch.Tensor] = None


# Linear combination over residual channels: list of (channel_index, coef).
# The compiler reads this as "sum_i coef_i * residual[channel_i]".
ChannelLC = List[Tuple[int, float]]


@dataclass
class TokenEmbed(Node):
    """Per-token entries `(token_id, channel, coef)` — tok.weight[k, ch] = coef."""
    entries: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class PosEmbed(Node):
    """Per-position entries `(position, channel, coef)` — pos.weight[p, ch] = coef."""
    entries: List[Tuple[int, int, float]] = field(default_factory=list)


@dataclass
class LookUp(Node):
    """Attention head — copy-from-pos-0 form.

    Keys / queries are all zero, so hard-max argmax with first-tie
    semantics deterministically picks past position 0. Values get read
    from the residual via `v_source_channels`, written to the output
    residual via `out_channels` (same length).

    Session 26 ships just this form because it's what `copy_past` needs.
    Parabolic-key exact lookups (RESEARCH/03 §4a) land in a later pass.
    """
    layer: int = 0
    v_source_channels: List[int] = field(default_factory=list)
    out_channels: List[int] = field(default_factory=list)


@dataclass
class ReGLU(Node):
    """One FFN neuron: `output_channel += output_coef · val · ReLU(gate)`.

    `gate` and `val` are linear combinations of residual channels. Section
    4c/4d of RESEARCH/03 — a gated ReLU computes exact step functions on
    integer inputs (`1[z ≥ 0] = ReLU(z+1) − ReLU(z)`) and exact products
    (`a · b = a · ReLU(b) − a · ReLU(−b)` for signed ints).
    """
    layer: int = 0
    gate: ChannelLC = field(default_factory=list)
    val: ChannelLC = field(default_factory=list)
    output_channel: int = 0
    output_coef: float = 1.0


@dataclass
class LinearHead(Node):
    """Final head entries `(token_id, channel, coef)` — head.weight[k, ch] = coef."""
    entries: List[Tuple[int, int, float]] = field(default_factory=list)


# --- Container ---

@dataclass
class GateGraph:
    """A gate graph — collection of nodes + metadata.

    Nodes can be added via `add()`, looked up via `get()` by name. The
    graph is append-only; topo order follows insertion order if every
    node's sources were added before the node itself.
    """
    nodes: List[Node] = field(default_factory=list)
    vocab_size: int = 0
    _by_name: Dict[str, Node] = field(default_factory=dict)

    def add(self, node: Node) -> Node:
        if node.name in self._by_name:
            raise ValueError(f"duplicate node name: {node.name}")
        self.nodes.append(node)
        self._by_name[node.name] = node
        return node

    def get(self, name: str) -> Node:
        return self._by_name[name]

    def outputs(self) -> List[TokenOutput]:
        return [n for n in self.nodes if isinstance(n, TokenOutput)]

    def inputs(self) -> List[TokenInput]:
        return [n for n in self.nodes if isinstance(n, TokenInput)]

    def results(self) -> List[Result]:
        return [n for n in self.nodes if isinstance(n, Result)]
