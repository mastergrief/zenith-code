"""compiled_router — opcode-gated dispatch between ADD and MUL.

Round 1 of the "compiled routing replaces LoRA" thesis. Same template
as `dispatched.py` (gcd/factorial/is_prime), narrowed to arithmetic and
with operands in `[0, 9]`. Proves the gate-graph IR expresses ADD/MUL
dispatch declaratively; no training.

Program: input `[a, b, opcode]` at positions 0, 1, 2.
  - opcode=0 (ADD): slot == a + b, in `[0, 18]`
  - opcode=1 (MUL): slot == MUL_SLOT_BASE + (a * b), in `[19, 100]`

Residual channel layout (d_model = 130):
  0:   own_scalar          (TokenEmbed: tok[k, 0] = k)
  1:   bias                (PosEmbed: 1 at every pos)
  2:   pos_k0              (PosEmbed: 2p)
  3:   pos_k1              (PosEmbed: -p²)
  4:   target_pos          (PosEmbed: 1 at pos 2 only)
  5:   copy_a              (LookUp from pos 0)
  6:   copy_b              (LookUpExact from pos 1)
  7..9 (3):  opcode_step_k for k ∈ [0, 2]   (layer 0 FFN)
  10..28 (19): gated_add_step_k (layer 1 FFN: is_op_0 · step_k(a + b))
  29..128 (100): gated_mul_step_k (layer 1 FFN: is_op_1 · step_k(10a + b))
  129: spare (padding to even d_model for d_head=2)

ADD uses the adder_tiny template — step_k is over the linear combination
(a + b), so we don't need a separate key channel. MUL uses key = 10a + b
to index a 100-entry lookup table; each (a, b) maps to a unique key, the
head wires that key → product slot.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


MAX_OPERAND = 9
ADD_MAX = MAX_OPERAND * 2           # 18
MUL_MAX = MAX_OPERAND * MAX_OPERAND  # 81
MUL_KEY_BASE = 10                    # key = 10·a + b, keys ∈ [0, 99]
MUL_KEY_MAX = MUL_KEY_BASE * MAX_OPERAND + MAX_OPERAND  # 99

ADD_SLOT_BASE = 0                    # slots 0..18 = a+b
MUL_SLOT_BASE = ADD_MAX + 1          # 19; slots 19..100 = a*b offset by 19
VOCAB = MUL_SLOT_BASE + MUL_MAX + 1  # 101

CH_OWN = 0
CH_BIAS = 1
CH_POS_K0 = 2
CH_POS_K1 = 3
CH_TARGET_POS = 4
CH_COPY_A = 5
CH_COPY_B = 6
CH_OPCODE_STEP_BASE = 7                              # 3 channels: 7, 8, 9
CH_GATED_ADD_BASE = CH_OPCODE_STEP_BASE + 3          # 10
CH_GATED_MUL_BASE = CH_GATED_ADD_BASE + (ADD_MAX + 1)  # 29
D_MODEL_CORE = CH_GATED_MUL_BASE + (MUL_KEY_MAX + 1)  # 129
D_MODEL = D_MODEL_CORE + (D_MODEL_CORE % 2)          # 130 (even for d_head=2)


def _is_op_val(k: int):
    """is_op_k = opcode_step_k - opcode_step_{k+1} as a ChannelLC."""
    return [
        (CH_OPCODE_STEP_BASE + k, 1.0),
        (CH_OPCODE_STEP_BASE + k + 1, -1.0),
    ]


def build_router(max_len: int = 3) -> Small2DTransformer:
    V = VOCAB
    graph = GateGraph(vocab_size=V)

    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, CH_OWN, float(k)) for k in range(V)],
    ))

    pos_entries = []
    for p in range(max_len):
        pos_entries.append((p, CH_BIAS, 1.0))
        pos_entries.append((p, CH_POS_K0, float(2 * p)))
        pos_entries.append((p, CH_POS_K1, -float(p * p)))
    pos_entries.append((2, CH_TARGET_POS, 1.0))
    graph.add(PosEmbed(name="pos_consts", entries=pos_entries))

    # Layer 0 attn: copy a from pos 0 (first-tie LookUp) + b from pos 1 (exact).
    graph.add(LookUp(
        name="copy_a",
        v_source_channels=[CH_OWN],
        out_channels=[CH_COPY_A],
    ))
    graph.add(LookUpExact(
        name="copy_b",
        pos_key0_channel=CH_POS_K0, pos_key0_coef=1.0,
        pos_key1_channel=CH_POS_K1, pos_key1_coef=1.0,
        query_key_channel=CH_TARGET_POS, query_key_coef=1.0,
        bias_channel=CH_BIAS, bias_coef=1.0,
        value_source_channels=[CH_OWN],
        out_channels=[CH_COPY_B],
    ))

    # Layer 0 FFN: opcode_step_k = 1[opcode >= k] on CH_OWN, for k ∈ [0, 2].
    # The opcode token sits at pos 2; spurious values at pos 0/1 are ignored.
    for k in range(3):
        graph.add(ReGLU(
            name=f"opcode_step_{k}_hi",
            gate=[(CH_OWN, 1.0), (CH_BIAS, -(k - 1))],
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_OPCODE_STEP_BASE + k,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"opcode_step_{k}_lo",
            gate=[(CH_OWN, 1.0), (CH_BIAS, -k)],
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_OPCODE_STEP_BASE + k,
            output_coef=-1.0,
        ))

    # Layer 1 FFN: gated_add_step_k = step_k(a + b) · is_op_0, k ∈ [0, 18].
    is_op_0 = _is_op_val(0)
    for k in range(ADD_MAX + 1):
        ch = CH_GATED_ADD_BASE + k
        graph.add(ReGLU(
            name=f"add_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_COPY_B, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_0,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"add_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_COPY_B, 1.0), (CH_BIAS, -k)],
            val=is_op_0,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # Layer 1 FFN: gated_mul_step_k = step_k(10a + b) · is_op_1, k ∈ [0, 99].
    is_op_1 = _is_op_val(1)
    for kk in range(MUL_KEY_MAX + 1):
        ch = CH_GATED_MUL_BASE + kk
        graph.add(ReGLU(
            name=f"mul_gated_{kk}_hi",
            gate=[(CH_COPY_A, float(MUL_KEY_BASE)), (CH_COPY_B, 1.0),
                  (CH_BIAS, -(kk - 1))],
            val=is_op_1,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"mul_gated_{kk}_lo",
            gate=[(CH_COPY_A, float(MUL_KEY_BASE)), (CH_COPY_B, 1.0),
                  (CH_BIAS, -kk)],
            val=is_op_1,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # Head: step-diff entries pick one slot per opcode.
    head_entries = []
    # ADD: slot = a + b, channel = CH_GATED_ADD_BASE + (a + b). Same as adder.
    for s in range(ADD_MAX + 1):
        head_entries.append((ADD_SLOT_BASE + s, CH_GATED_ADD_BASE + s, 1.0))
        if s + 1 <= ADD_MAX:
            head_entries.append(
                (ADD_SLOT_BASE + s, CH_GATED_ADD_BASE + s + 1, -1.0)
            )
    # MUL: for each (a, b), slot = MUL_SLOT_BASE + a*b, channel = 10a+b.
    for a in range(MAX_OPERAND + 1):
        for b in range(MAX_OPERAND + 1):
            kk = MUL_KEY_BASE * a + b
            m = a * b
            slot = MUL_SLOT_BASE + m
            head_entries.append((slot, CH_GATED_MUL_BASE + kk, 1.0))
            if kk + 1 <= MUL_KEY_MAX:
                head_entries.append((slot, CH_GATED_MUL_BASE + kk + 1, -1.0))
    graph.add(LinearHead(name="router_head", entries=head_entries))

    n_layers = auto_schedule(graph)
    n_heads = D_MODEL // 2
    d_ffn = 2 * (ADD_MAX + 1) + 2 * (MUL_KEY_MAX + 1) + 2 * 3  # layer 1 dominates

    return compile_program(
        graph,
        d_model=D_MODEL,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


def decode_output(opcode: int, slot: int) -> int:
    """Map raw output slot back to logical value."""
    if opcode == 0:
        return slot
    if opcode == 1:
        return slot - MUL_SLOT_BASE
    raise ValueError(f"unknown opcode: {opcode}")


def run_program(model, opcode: int, a: int, b: int) -> int:
    import torch
    x = torch.tensor([[a, b, opcode]], dtype=torch.long)
    with torch.no_grad():
        slot = int(model(x)[0, 2].argmax().item())
    return decode_output(opcode, slot)


if __name__ == "__main__":
    import itertools
    import time
    import torch

    t0 = time.time()
    model = build_router()
    t_build = time.time() - t0
    print(f"[router] built in {t_build:.1f}s, {model.param_count():,} params")

    pairs = list(itertools.product(range(MAX_OPERAND + 1), repeat=2))

    add_inputs = [(a, b, 0) for a, b in pairs]
    x = torch.tensor(add_inputs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    add_correct = sum(
        1 for p, (a, b, _) in zip(preds, add_inputs)
        if decode_output(0, p) == a + b
    )
    print(f"  ADD {add_correct}/{len(add_inputs)}")

    mul_inputs = [(a, b, 1) for a, b in pairs]
    x = torch.tensor(mul_inputs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    mul_correct = sum(
        1 for p, (a, b, _) in zip(preds, mul_inputs)
        if decode_output(1, p) == a * b
    )
    print(f"  MUL {mul_correct}/{len(mul_inputs)}")

    total = len(add_inputs) + len(mul_inputs)
    correct = add_correct + mul_correct
    print(f"\n[router] total: {correct}/{total} — "
          f"{'PASS' if correct == total else 'FAIL'}")
