"""dispatched_v2 — five-op unified compiled card with opcode dispatch.

Extends `dispatched.py` (GCD / FACTORIAL / IS_PRIME) with two more
arithmetic ops — ADD and MUL — all gated by a single opcode token.
This is the Round-4/5 validation of the "compiled routing replaces LoRA"
thesis: ONE compiled card handles five CALM-backend operations, the
opcode selects which one fires, inactive ops output exactly zero (the
`is_op_k · step_k(...)` gating pattern from `dispatched.py` section 4).

Program: input `[a, b, opcode]` at positions 0, 1, 2. Output at pos 2:
  - opcode=0 (GCD):       slot ∈ [0, 15]   = gcd(a, b)
  - opcode=1 (FACTORIAL): slot = FACT_SLOT_BASE + n where a = n ∈ [0, 8]
  - opcode=2 (IS_PRIME):  slot = PRIME_SLOT_TRUE / PRIME_SLOT_FALSE
  - opcode=3 (ADD):       slot = ADD_SLOT_BASE + (a + b), a+b ∈ [0, 30]
  - opcode=4 (MUL):       slot = MUL_SLOT_BASE + (a * b), a*b ∈ [0, 225]

For unary ops (FACTORIAL, IS_PRIME) pass b = 0.

Residual channel layout (d_model = 582):
  0:   own_scalar
  1:   bias
  2:   pos_k0             (2p)
  3:   pos_k1             (-p²)
  4:   target_pos         (1 at pos 2 only)
  5:   copy_a             (LookUp from pos 0)
  6:   copy_b             (LookUpExact from pos 1)
  7:   key                (layer 0 FFN: 16·copy_a + copy_b; reused by GCD and MUL)
  8..13 (6):   opcode_step_k, k ∈ [0, 5]     (layer 0 FFN, on CH_OWN)
  14..269 (256): gated_gcd_step_k           (layer 1 FFN: is_op_0 · step_k(key))
  270..278 (9):  gated_fact_step_k          (layer 1 FFN: is_op_1 · step_k(copy_a))
  279..293 (15): gated_prime_step_k         (layer 1 FFN: is_op_2 · step_k(copy_a))
  294..324 (31): gated_add_step_k           (layer 1 FFN: is_op_3 · step_k(copy_a + copy_b))
  325..580 (256): gated_mul_step_k          (layer 1 FFN: is_op_4 · step_k(key))
  581: spare (padding to even d_model)

Same `is_op_k = opcode_step_k - opcode_step_{k+1}` trick as dispatched.py
— when opcode=3 the only non-zero `is_op_k` is `is_op_3`, so only the
ADD gated channels carry a non-zero step-function; all other ops' channels
stay exactly zero. Head wiring picks step-diffs per op across its gated
channel range.
"""

from __future__ import annotations

import math

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, LookUp, LookUpExact, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.programs.is_prime import _is_prime
from calm.llm_computer.schedule import auto_schedule


# --- Operand / op parameters ---
GCD_MAX = 15
GCD_BASE = GCD_MAX + 1              # 16 — key multiplier
GCD_MAX_KEY = GCD_BASE * GCD_BASE - 1  # 255
FACT_MAX_N = 8
PRIME_MIN_N = 2
PRIME_MAX_N = 15
ADD_MAX = GCD_MAX + GCD_MAX         # 30
MUL_MAX_OPERAND = GCD_MAX            # 15
MUL_MAX_KEY = GCD_MAX_KEY            # 255 (reuses the 16·a+b key)
MUL_MAX_PRODUCT = MUL_MAX_OPERAND * MUL_MAX_OPERAND  # 225

N_OPS = 5  # 0=GCD, 1=FACT, 2=PRIME, 3=ADD, 4=MUL

# --- Slot allocation (token IDs in the output vocabulary) ---
GCD_SLOT_BASE = 0                                        # [0, 15]
FACT_SLOT_BASE = GCD_MAX + 1                             # 16, [16, 24]
PRIME_SLOT_FALSE = FACT_SLOT_BASE + FACT_MAX_N + 1       # 25
PRIME_SLOT_TRUE = PRIME_SLOT_FALSE + 1                    # 26
ADD_SLOT_BASE = PRIME_SLOT_TRUE + 1                       # 27, [27, 57]
MUL_SLOT_BASE = ADD_SLOT_BASE + ADD_MAX + 1              # 58, [58, 283]
VOCAB = MUL_SLOT_BASE + MUL_MAX_PRODUCT + 1              # 284

# --- Channel layout ---
CH_OWN = 0
CH_BIAS = 1
CH_POS_K0 = 2
CH_POS_K1 = 3
CH_TARGET_POS = 4
CH_COPY_A = 5
CH_COPY_B = 6
CH_KEY = 7                                               # 16·a + b
CH_OPCODE_STEP_BASE = 8                                  # 6 channels
CH_GATED_GCD_BASE = CH_OPCODE_STEP_BASE + (N_OPS + 1)    # 14
CH_GATED_FACT_BASE = CH_GATED_GCD_BASE + (GCD_MAX_KEY + 1)  # 270
CH_GATED_PRIME_BASE = CH_GATED_FACT_BASE + (FACT_MAX_N + 1)  # 279
CH_GATED_ADD_BASE = CH_GATED_PRIME_BASE + (PRIME_MAX_N - PRIME_MIN_N + 2)  # 294
CH_GATED_MUL_BASE = CH_GATED_ADD_BASE + (ADD_MAX + 1)    # 325
D_MODEL_CORE = CH_GATED_MUL_BASE + (MUL_MAX_KEY + 1)     # 581
D_MODEL = D_MODEL_CORE + (D_MODEL_CORE % 2)              # 582


def _is_op_val(k: int):
    """is_op_k = opcode_step_k - opcode_step_{k+1} as a ChannelLC."""
    return [
        (CH_OPCODE_STEP_BASE + k, 1.0),
        (CH_OPCODE_STEP_BASE + k + 1, -1.0),
    ]


def build_dispatched_v2(max_len: int = 3) -> Small2DTransformer:
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

    # Layer 0 attn: copy_a from pos 0 + copy_b from pos 1.
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

    # Layer 0 FFN: KEY = 16·copy_a + copy_b, opcode_step_k.
    graph.add(ReGLU(
        name="key_scale_a",
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_A, 1.0)],
        output_channel=CH_KEY,
        output_coef=float(GCD_BASE),
    ))
    graph.add(ReGLU(
        name="key_add_b",
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_B, 1.0)],
        output_channel=CH_KEY,
        output_coef=1.0,
    ))

    for k in range(N_OPS + 1):
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

    # --- Layer 1 FFN: gated step functions per op ---
    is_op_gcd = _is_op_val(0)
    is_op_fact = _is_op_val(1)
    is_op_prime = _is_op_val(2)
    is_op_add = _is_op_val(3)
    is_op_mul = _is_op_val(4)

    # GCD: step_k(key) · is_op_0, k ∈ [0, GCD_MAX_KEY]
    for k in range(GCD_MAX_KEY + 1):
        ch = CH_GATED_GCD_BASE + k
        graph.add(ReGLU(
            name=f"gcd_gated_{k}_hi",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_gcd,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"gcd_gated_{k}_lo",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -k)],
            val=is_op_gcd,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # FACTORIAL: step_k(copy_a) · is_op_1, k ∈ [0, FACT_MAX_N]
    for k in range(FACT_MAX_N + 1):
        ch = CH_GATED_FACT_BASE + k
        graph.add(ReGLU(
            name=f"fact_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_fact,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"fact_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -k)],
            val=is_op_fact,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # IS_PRIME: step_k(copy_a) · is_op_2, k ∈ [PRIME_MIN_N, PRIME_MAX_N + 1]
    for k in range(PRIME_MIN_N, PRIME_MAX_N + 2):
        ch = CH_GATED_PRIME_BASE + (k - PRIME_MIN_N)
        graph.add(ReGLU(
            name=f"prime_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_prime,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"prime_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -k)],
            val=is_op_prime,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # ADD: step_k(copy_a + copy_b) · is_op_3, k ∈ [0, ADD_MAX]
    for k in range(ADD_MAX + 1):
        ch = CH_GATED_ADD_BASE + k
        graph.add(ReGLU(
            name=f"add_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_COPY_B, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_add,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"add_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_COPY_B, 1.0), (CH_BIAS, -k)],
            val=is_op_add,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # MUL: step_k(key) · is_op_4, k ∈ [0, MUL_MAX_KEY]
    # The KEY channel holds 16a+b. For each (a, b) there's a unique key,
    # which head wiring will map to the correct product slot.
    for k in range(MUL_MAX_KEY + 1):
        ch = CH_GATED_MUL_BASE + k
        graph.add(ReGLU(
            name=f"mul_gated_{k}_hi",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_mul,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"mul_gated_{k}_lo",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -k)],
            val=is_op_mul,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # --- Head wiring (accumulating step-diffs per op) ---
    head_entries = []

    # GCD
    for A in range(GCD_BASE):
        for B in range(GCD_BASE):
            kk = GCD_BASE * A + B
            g = math.gcd(A, B)
            head_entries.append((g, CH_GATED_GCD_BASE + kk, 1.0))
            if kk + 1 <= GCD_MAX_KEY:
                head_entries.append((g, CH_GATED_GCD_BASE + kk + 1, -1.0))

    # FACTORIAL
    for n in range(FACT_MAX_N + 1):
        slot = FACT_SLOT_BASE + n
        head_entries.append((slot, CH_GATED_FACT_BASE + n, 1.0))
        if n + 1 <= FACT_MAX_N:
            head_entries.append((slot, CH_GATED_FACT_BASE + n + 1, -1.0))

    # IS_PRIME
    for n in range(PRIME_MIN_N, PRIME_MAX_N + 1):
        slot = PRIME_SLOT_TRUE if _is_prime(n) else PRIME_SLOT_FALSE
        ch = CH_GATED_PRIME_BASE + (n - PRIME_MIN_N)
        head_entries.append((slot, ch, 1.0))
        head_entries.append((slot, ch + 1, -1.0))

    # ADD: slot = ADD_SLOT_BASE + (a + b); step-diff on gated_add_step_{a+b}.
    for s in range(ADD_MAX + 1):
        slot = ADD_SLOT_BASE + s
        head_entries.append((slot, CH_GATED_ADD_BASE + s, 1.0))
        if s + 1 <= ADD_MAX:
            head_entries.append((slot, CH_GATED_ADD_BASE + s + 1, -1.0))

    # MUL: for each (a, b), slot = MUL_SLOT_BASE + a*b, channel = key = 16a+b.
    # Multiple (a, b) can share a product; head's `+=` accumulation handles it.
    for A in range(MUL_MAX_OPERAND + 1):
        for B in range(MUL_MAX_OPERAND + 1):
            kk = GCD_BASE * A + B
            prod = A * B
            slot = MUL_SLOT_BASE + prod
            head_entries.append((slot, CH_GATED_MUL_BASE + kk, 1.0))
            if kk + 1 <= MUL_MAX_KEY:
                head_entries.append((slot, CH_GATED_MUL_BASE + kk + 1, -1.0))

    graph.add(LinearHead(name="dispatched_v2_head", entries=head_entries))

    n_layers = auto_schedule(graph)
    n_heads = D_MODEL // 2
    # Upper bound of concurrent ReGLU neurons on layer 1 (all 5 ops).
    d_ffn = (
        2 * (GCD_MAX_KEY + 1)
        + 2 * (FACT_MAX_N + 1)
        + 2 * (PRIME_MAX_N - PRIME_MIN_N + 2)
        + 2 * (ADD_MAX + 1)
        + 2 * (MUL_MAX_KEY + 1)
    )
    return compile_program(
        graph,
        d_model=D_MODEL,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


def decode_output(opcode: int, slot: int):
    """Map raw output slot back to logical value for the given opcode."""
    if opcode == 0:
        return slot                                       # gcd value
    if opcode == 1:
        return math.factorial(slot - FACT_SLOT_BASE)
    if opcode == 2:
        return slot == PRIME_SLOT_TRUE
    if opcode == 3:
        return slot - ADD_SLOT_BASE
    if opcode == 4:
        return slot - MUL_SLOT_BASE
    raise ValueError(f"unknown opcode: {opcode}")


def run_program(model, opcode: int, a: int, b: int = 0):
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
    model = build_dispatched_v2()
    t_build = time.time() - t0
    print(f"[dispatched_v2] built in {t_build:.1f}s, "
          f"{model.param_count():,} params (d_model={D_MODEL}, vocab={VOCAB})")

    # Exhaustive per op.
    def _run(inputs, expected, label):
        x = torch.tensor(inputs, dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
        correct = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {correct}/{len(inputs)}")
        return correct, len(inputs)

    gcd_inputs = [(a, b, 0) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    gcd_expected = [math.gcd(a, b) for (a, b, _) in gcd_inputs]
    fact_inputs = [(n, 0, 1) for n in range(FACT_MAX_N + 1)]
    fact_expected = [math.factorial(n) for (n, _, _) in fact_inputs]
    prime_inputs = [(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]
    prime_expected = [_is_prime(n) for (n, _, _) in prime_inputs]
    add_inputs = [(a, b, 3) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    add_expected = [a + b for (a, b, _) in add_inputs]
    mul_inputs = [(a, b, 4) for a, b in itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2)]
    mul_expected = [a * b for (a, b, _) in mul_inputs]

    t0 = time.time()
    ok = 0
    tot = 0
    c, n = _run(gcd_inputs, gcd_expected, "GCD      ")
    ok += c; tot += n
    c, n = _run(fact_inputs, fact_expected, "FACTORIAL")
    ok += c; tot += n
    c, n = _run(prime_inputs, prime_expected, "IS_PRIME ")
    ok += c; tot += n
    c, n = _run(add_inputs, add_expected, "ADD      ")
    ok += c; tot += n
    c, n = _run(mul_inputs, mul_expected, "MUL      ")
    ok += c; tot += n
    t_run = time.time() - t0
    print(f"\n[dispatched_v2] total: {ok}/{tot} — "
          f"{'PASS' if ok == tot else 'FAIL'} "
          f"(run time {t_run:.1f}s)")
