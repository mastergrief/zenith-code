"""Auto-scheduler tests.

Build each program's GateGraph with layer fields unset, run
auto_schedule, compare against the hand-picked layering. The
scheduler should produce the same or equivalent placement for every
program we've authored so far.
"""

from __future__ import annotations

import itertools

import torch

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.schedule import auto_schedule


def _build_threshold_ir_no_layer(V=8, T=4, max_len=16):
    """Same as threshold_ir but with layer=0 implicit (scheduler assigns)."""
    g = GateGraph(vocab_size=V)
    g.add(TokenEmbed(name="input_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias_channel",
                   entries=[(p, 1, 1.0) for p in range(max_len)]))
    g.add(ReGLU(name="step_hi", gate=[(0, 1.0), (1, -(T - 1))],
                val=[(1, 1.0)], output_channel=2, output_coef=1.0))
    g.add(ReGLU(name="step_lo", gate=[(0, 1.0), (1, -T)],
                val=[(1, 1.0)], output_channel=2, output_coef=-1.0))
    g.add(LinearHead(name="read_step", entries=[(1, 2, 1.0)]))
    return g


def _build_copy_past_ir_no_layer(V=8):
    g = GateGraph(vocab_size=V)
    g.add(TokenEmbed(name="tok_lower",
                     entries=[(k, k, 1.0) for k in range(V)]))
    g.add(LookUp(name="copy_from_pos_0",
                 v_source_channels=list(range(V)),
                 out_channels=[V + k for k in range(V)]))
    g.add(LinearHead(name="read_upper",
                     entries=[(j, V + j, 1.0) for j in range(V)]))
    return g


def _build_read_by_key_no_layer(V=4, max_len=5):
    """Same program as programs/read_by_key.py but layers unset."""
    g = GateGraph(vocab_size=V)
    g.add(TokenEmbed(name="key_scalar",
                     entries=[(k, 0, float(k)) for k in range(V)]))
    g.add(PosEmbed(name="bias_and_pos",
                   entries=([(p, 1, 1.0) for p in range(max_len)]
                            + [(p, 2, float(p)) for p in range(max_len)])))
    # square_neg — consumes ch 0, writes ch 3. Depends only on tok embed.
    g.add(ReGLU(name="square_neg",
                gate=[(0, 1.0)], val=[(0, -1.0)],
                output_channel=3, output_coef=1.0))
    # retrieve — reads ch 0, 1, 2, 3. Depends on ReGLU (ch 3).
    g.add(LookUpExact(
        name="retrieve_by_key",
        pos_key0_channel=0, pos_key0_coef=2.0,
        pos_key1_channel=3, pos_key1_coef=1.0,
        query_key_channel=0, query_key_coef=1.0,
        bias_channel=1, bias_coef=1.0,
        value_source_channels=[2], out_channels=[4],
    ))
    # step functions — read ch 4 (retrieved), written by LookUpExact.
    for S in range(V):
        g.add(ReGLU(name=f"step_{S}_hi",
                    gate=[(4, 1.0), (1, -(S - 1))], val=[(1, 1.0)],
                    output_channel=5 + S, output_coef=1.0))
        g.add(ReGLU(name=f"step_{S}_lo",
                    gate=[(4, 1.0), (1, -S)], val=[(1, 1.0)],
                    output_channel=5 + S, output_coef=-1.0))
    head_entries = []
    for k in range(V):
        head_entries.append((k, 5 + k, 1.0))
        if k + 1 < V:
            head_entries.append((k, 5 + k + 1, -1.0))
    g.add(LinearHead(name="decode", entries=head_entries))
    return g


def test_schedule_threshold_single_layer():
    g = _build_threshold_ir_no_layer()
    n_layers = auto_schedule(g)
    assert n_layers == 1, f"threshold should need 1 layer, got {n_layers}"
    for node in g.nodes:
        if isinstance(node, ReGLU):
            assert node.layer == 0


def test_schedule_copy_past_single_layer():
    g = _build_copy_past_ir_no_layer()
    n_layers = auto_schedule(g)
    assert n_layers == 1
    for node in g.nodes:
        if isinstance(node, LookUp):
            assert node.layer == 0


def test_schedule_read_by_key_two_layers():
    """square_neg (layer 0 FFN) → retrieve (layer 1 attn) → decode ReGLUs (layer 1 FFN)."""
    g = _build_read_by_key_no_layer()
    n_layers = auto_schedule(g)
    assert n_layers == 2, f"read_by_key should need 2 layers, got {n_layers}"
    for node in g.nodes:
        if isinstance(node, ReGLU) and node.name == "square_neg":
            assert node.layer == 0, f"square_neg should be layer 0, got {node.layer}"
        elif isinstance(node, LookUpExact):
            assert node.layer == 1, f"retrieve should be layer 1, got {node.layer}"
        elif isinstance(node, ReGLU) and node.name.startswith("step_"):
            assert node.layer == 1, f"{node.name} should be layer 1, got {node.layer}"


def test_schedule_read_by_key_end_to_end():
    """Scheduler + compile + run. Should produce the same 96/96 as the
    hand-layered version in programs/read_by_key.py."""
    V = 4
    max_len = V + 1
    g = _build_read_by_key_no_layer(V=V, max_len=max_len)
    n_layers = auto_schedule(g)
    model = compile_program(
        g, d_model=10, n_heads=5, n_layers=n_layers,
        d_ffn=2 * V,  # layer 1 FFN has 2V neurons; layer 0 has 1
        max_len=max_len, vocab_size=V,
    )
    for perm in itertools.permutations(range(V)):
        for q in range(V):
            inp = list(perm) + [q]
            x = torch.tensor([inp], dtype=torch.long)
            with torch.no_grad():
                got = int(model(x)[0, V].argmax().item())
            expected = perm.index(q)
            assert got == expected, \
                f"auto-scheduled read_by_key: keys={perm} q={q} got={got} exp={expected}"


if __name__ == "__main__":
    test_schedule_threshold_single_layer()
    print("[ok] threshold → 1 layer")
    test_schedule_copy_past_single_layer()
    print("[ok] copy_past → 1 layer")
    test_schedule_read_by_key_two_layers()
    print("[ok] read_by_key → 2 layers (square_neg @ 0, retrieve @ 1, decode @ 1)")
    test_schedule_read_by_key_end_to_end()
    print("[ok] auto-scheduled read_by_key: 96/96 (perm, q) cases")
