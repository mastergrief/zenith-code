"""isa — tiny accumulator ISA compiled on the substrate.

Stage 3 of the UTM path: compile a small instruction set. This MVP
supports 2 opcodes (INC, DEC) plus HALT on out-of-bounds. The input
sequence is:

    [OP, V]     OP ∈ {INC=8, DEC=9}, V ∈ [0, V_MAX]

Autoregressive generation emits:

    [OP, V, f(V, OP), f(f(V, OP), OP), ..., HALT]

where f(v, INC) = v + 1 (HALT if v = V_MAX)
      f(v, DEC) = v - 1 (HALT if v = 0)

Construction: at every query position i ≥ 1 the model reads own_scalar
(= last accumulator) at ch 0 and copy-from-pos-0 into ch 2 (= OP).
Computes combined key = 2 * accumulator + (op - INC) ∈ [0, 2·V_MAX + 1].
Step-diff decode on key selects a unique (v, op) pair; the head maps
each pair to its output token via LinearHead accumulation.

Scaling to 8 opcodes: change key encoding to N * v + (op - OP_BASE) for
N-opcode ISAs; each op adds another factor in the combined vocabulary.
Unchanged: the pattern (LookUp for OP, own_scalar for state, combined
key via ReGLU, step-diff decode, LinearHead dispatch).
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


V_MAX = 7          # accumulator range [0, V_MAX]
INC = 8            # opcode token
DEC = 9            # opcode token
HALT = 10
VOCAB = 11         # 0..V_MAX + INC, DEC, HALT

N_OPCODES = 2
KEY_MAX = (V_MAX + 1) * N_OPCODES - 1   # 15 for V_MAX=7, N_OPCODES=2


def _target_token(v: int, op: int) -> int:
    """The output token emitted for accumulator v under opcode op."""
    if op == INC:
        return v + 1 if v < V_MAX else HALT
    if op == DEC:
        return v - 1 if v > 0 else HALT
    raise ValueError(op)


def build_isa(max_len: int = 32) -> Small2DTransformer:
    V = VOCAB
    g = GateGraph(vocab_size=V)

    # ch 0: own_scalar (token value at each position)
    g.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    # ch 1: bias
    g.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    # Layer 0 attn: LookUp copy-from-pos-0 fills ch 2 = OP at every position.
    g.add(LookUp(
        name="copy_op",
        v_source_channels=[0],
        out_channels=[2],
    ))
    # Layer 0 FFN: key = N_OPCODES * own_scalar + (op - INC)
    #            = 2 * acc + op_bit
    #            where op_bit = 0 for INC, 1 for DEC.
    g.add(ReGLU(
        name="key_scale_acc",
        gate=[(1, 1.0)],
        val=[(0, 1.0)],
        output_channel=3,
        output_coef=float(N_OPCODES),
    ))
    g.add(ReGLU(
        name="key_add_op",
        gate=[(1, 1.0)],
        val=[(2, 1.0)],
        output_channel=3,
        output_coef=1.0,
    ))
    g.add(ReGLU(
        name="key_sub_inc",
        gate=[(1, 1.0)],
        val=[(1, 1.0)],
        output_channel=3,
        output_coef=-float(INC),
    ))

    # Layer 1 FFN: step decode on key ∈ [0, KEY_MAX].
    for S in range(KEY_MAX + 2):
        g.add(ReGLU(
            name=f"step_{S}_hi",
            gate=[(3, 1.0), (1, -(S - 1))],
            val=[(1, 1.0)],
            output_channel=4 + S,
            output_coef=1.0,
        ))
        g.add(ReGLU(
            name=f"step_{S}_lo",
            gate=[(3, 1.0), (1, -S)],
            val=[(1, 1.0)],
            output_channel=4 + S,
            output_coef=-1.0,
        ))
    # Head: for each (v, op) with combined key k = 2*v + (op - INC),
    # logits[_target_token(v, op)] += step_k - step_{k+1}.
    head_entries = []
    for v in range(V_MAX + 1):
        for i, op in enumerate((INC, DEC)):
            k = N_OPCODES * v + i
            target = _target_token(v, op)
            head_entries.append((target, 4 + k, 1.0))
            if k + 1 <= KEY_MAX + 1:
                head_entries.append((target, 4 + k + 1, -1.0))
    g.add(LinearHead(name="isa_head", entries=head_entries))

    n_layers = auto_schedule(g)
    d_model = 4 + (KEY_MAX + 2)            # 20 for KEY_MAX=15
    if d_model % 2 != 0:
        d_model += 1
    n_heads = d_model // 2
    d_ffn = 2 * (KEY_MAX + 2)              # 32
    return compile_program(
        g,
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn,
        max_len=max_len, vocab_size=V,
    )


def run_isa(op: int, start_value: int, max_steps: int = 30, model=None):
    """Execute ISA via autoregressive loop. Returns full token trace."""
    import torch
    if op not in (INC, DEC):
        raise ValueError(f"unsupported opcode: {op}")
    if not (0 <= start_value <= V_MAX):
        raise ValueError(f"start_value must be in [0, {V_MAX}], got {start_value}")
    if model is None:
        model = build_isa()

    seq = [op, start_value]
    for _ in range(max_steps):
        x = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(logits[0, -1, :].argmax().item())
        seq.append(next_tok)
        if next_tok == HALT:
            break
    return seq


def simulate_expected(op: int, start_value: int):
    """Reference Python implementation for test comparison."""
    out = [op, start_value]
    v = start_value
    while True:
        new_v = _target_token(v, op)
        out.append(new_v)
        if new_v == HALT:
            return out
        v = new_v


if __name__ == "__main__":
    model = build_isa()
    print(f"[isa] built, {model.param_count():,} params, "
          f"{N_OPCODES} opcodes × [0, {V_MAX}] = {(V_MAX + 1) * N_OPCODES} (v, op) pairs")

    all_ok = True
    for op, op_name in ((INC, "INC"), (DEC, "DEC")):
        for v in range(V_MAX + 1):
            seq = run_isa(op, v, model=model)
            expected = simulate_expected(op, v)
            ok = seq == expected
            all_ok = all_ok and ok
            mark = "✓" if ok else "✗"
            print(f"  {mark} {op_name:3} {v}  →  {seq}  (exp {expected})")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
