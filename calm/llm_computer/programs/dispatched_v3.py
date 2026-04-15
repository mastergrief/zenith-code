"""dispatched_v3 — 9-op unified compiled card (5 from v2 + 4 new).

Extends `dispatched_v2.py` with four more arithmetic/logical ops:
  * MOD(a, b)   — a mod b (b > 0 required; b=0 returns 0 by convention)
  * MIN(a, b)   — min(a, b)
  * MAX(a, b)   — max(a, b)
  * DIFF(a, b)  — |a - b|

All four share the `KEY = 16·a + b` lookup pattern already used by GCD
and MUL. Each op adds one gated step-function bank of 256 channels and
64 head entries (one per (a, b) key in [0, 255]).

Same `is_op_k · step_k(KEY)` gating: inactive ops contribute exactly
zero. Proves the dispatched pattern scales linearly with op count —
adding a new compute backend to a compiled card is a ~20-line change
(channel allocation + per-op ReGLU block + head entries).

Program: input `[a, b, opcode]` at positions 0, 1, 2. Opcodes 0..8:
  0=GCD, 1=FACT, 2=PRIME, 3=ADD, 4=MUL, 5=MOD, 6=MIN, 7=MAX, 8=DIFF.
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


# Inherit numeric constraints from v2
GCD_MAX = 15
GCD_BASE = GCD_MAX + 1                     # 16 — key multiplier
GCD_MAX_KEY = GCD_BASE * GCD_BASE - 1      # 255
FACT_MAX_N = 8
PRIME_MIN_N = 2
PRIME_MAX_N = 15
ADD_MAX = GCD_MAX + GCD_MAX                # 30
MUL_MAX_OPERAND = GCD_MAX
MUL_MAX_KEY = GCD_MAX_KEY
MUL_MAX_PRODUCT = MUL_MAX_OPERAND * MUL_MAX_OPERAND  # 225
# New ops all use operand range [0, 15] (= GCD_MAX), shared KEY = 16a+b.
MOD_MAX_OPERAND = GCD_MAX
MOD_MAX_RESULT = GCD_MAX - 1              # mod b-1 at b=GCD_MAX
MIN_MAX_RESULT = GCD_MAX
MAX_MAX_RESULT = GCD_MAX
DIFF_MAX_RESULT = GCD_MAX

N_OPS = 9
# 0=GCD, 1=FACT, 2=PRIME, 3=ADD, 4=MUL, 5=MOD, 6=MIN, 7=MAX, 8=DIFF

# --- Slot allocation ---
GCD_SLOT_BASE = 0                                          # [0, 15]
FACT_SLOT_BASE = GCD_MAX + 1                               # 16, [16, 24]
PRIME_SLOT_FALSE = FACT_SLOT_BASE + FACT_MAX_N + 1         # 25
PRIME_SLOT_TRUE = PRIME_SLOT_FALSE + 1                      # 26
ADD_SLOT_BASE = PRIME_SLOT_TRUE + 1                         # 27, [27, 57]
MUL_SLOT_BASE = ADD_SLOT_BASE + ADD_MAX + 1                # 58, [58, 283]
MOD_SLOT_BASE = MUL_SLOT_BASE + MUL_MAX_PRODUCT + 1        # 284, [284, 298]
MIN_SLOT_BASE = MOD_SLOT_BASE + MOD_MAX_RESULT + 1         # 299, [299, 314]
MAX_SLOT_BASE = MIN_SLOT_BASE + MIN_MAX_RESULT + 1         # 315, [315, 330]
DIFF_SLOT_BASE = MAX_SLOT_BASE + MAX_MAX_RESULT + 1        # 331, [331, 346]
VOCAB = DIFF_SLOT_BASE + DIFF_MAX_RESULT + 1               # 347

# --- Channel layout ---
CH_OWN = 0
CH_BIAS = 1
CH_POS_K0 = 2
CH_POS_K1 = 3
CH_TARGET_POS = 4
CH_COPY_A = 5
CH_COPY_B = 6
CH_KEY = 7                                                  # 16·a + b
CH_OPCODE_STEP_BASE = 8                                     # N_OPS+1 = 10 channels
CH_GATED_GCD_BASE = CH_OPCODE_STEP_BASE + (N_OPS + 1)       # 18
CH_GATED_FACT_BASE = CH_GATED_GCD_BASE + (GCD_MAX_KEY + 1)  # 274
CH_GATED_PRIME_BASE = CH_GATED_FACT_BASE + (FACT_MAX_N + 1)  # 283
CH_GATED_ADD_BASE = CH_GATED_PRIME_BASE + (PRIME_MAX_N - PRIME_MIN_N + 2)  # 298
CH_GATED_MUL_BASE = CH_GATED_ADD_BASE + (ADD_MAX + 1)       # 329
CH_GATED_MOD_BASE = CH_GATED_MUL_BASE + (MUL_MAX_KEY + 1)   # 585
CH_GATED_MIN_BASE = CH_GATED_MOD_BASE + (GCD_MAX_KEY + 1)   # 841
CH_GATED_MAX_BASE = CH_GATED_MIN_BASE + (GCD_MAX_KEY + 1)   # 1097
CH_GATED_DIFF_BASE = CH_GATED_MAX_BASE + (GCD_MAX_KEY + 1)  # 1353
D_MODEL_CORE = CH_GATED_DIFF_BASE + (GCD_MAX_KEY + 1)       # 1609
D_MODEL = D_MODEL_CORE + (D_MODEL_CORE % 2)                 # 1610


def _is_op_val(k: int):
    return [
        (CH_OPCODE_STEP_BASE + k, 1.0),
        (CH_OPCODE_STEP_BASE + k + 1, -1.0),
    ]


def _add_key_gated_op(
    graph: GateGraph,
    name_prefix: str,
    is_op: list,
    base_channel: int,
) -> None:
    """Add a 256-step gated bank indexed by CH_KEY — for GCD/MUL/MOD/MIN/MAX/DIFF.

    Each of the 256 ReGLU pairs writes `step_k(KEY) · is_op` to the
    corresponding gated channel. When the op's opcode isn't active,
    is_op = 0 and the entire bank outputs zero.
    """
    for k in range(GCD_MAX_KEY + 1):
        ch = base_channel + k
        graph.add(ReGLU(
            name=f"{name_prefix}_gated_{k}_hi",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"{name_prefix}_gated_{k}_lo",
            gate=[(CH_KEY, 1.0), (CH_BIAS, -k)],
            val=is_op,
            output_channel=ch,
            output_coef=-1.0,
        ))


def _add_key_gated_head_entries(
    head_entries: list,
    slot_fn,
    base_slot: int,
    base_channel: int,
) -> None:
    """For each (a, b), slot = base_slot + slot_fn(a, b); entry writes
    `+1` at gated[key] and `-1` at gated[key+1] (step-diff one-hot)."""
    for A in range(GCD_BASE):
        for B in range(GCD_BASE):
            kk = GCD_BASE * A + B
            value = slot_fn(A, B)
            if value is None:
                continue
            slot = base_slot + value
            head_entries.append((slot, base_channel + kk, 1.0))
            if kk + 1 <= GCD_MAX_KEY:
                head_entries.append((slot, base_channel + kk + 1, -1.0))


def build_dispatched_v3(max_len: int = 3) -> Small2DTransformer:
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

    # Layer 0 attn
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

    # Layer 0 FFN: KEY = 16a+b, opcode_step_k
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

    # Layer 1 FFN: per-op gated banks.
    is_op_gcd = _is_op_val(0)
    is_op_fact = _is_op_val(1)
    is_op_prime = _is_op_val(2)
    is_op_add = _is_op_val(3)
    is_op_mul = _is_op_val(4)
    is_op_mod = _is_op_val(5)
    is_op_min = _is_op_val(6)
    is_op_max = _is_op_val(7)
    is_op_diff = _is_op_val(8)

    _add_key_gated_op(graph, "gcd", is_op_gcd, CH_GATED_GCD_BASE)

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

    _add_key_gated_op(graph, "mul", is_op_mul, CH_GATED_MUL_BASE)
    _add_key_gated_op(graph, "mod", is_op_mod, CH_GATED_MOD_BASE)
    _add_key_gated_op(graph, "min", is_op_min, CH_GATED_MIN_BASE)
    _add_key_gated_op(graph, "max", is_op_max, CH_GATED_MAX_BASE)
    _add_key_gated_op(graph, "diff", is_op_diff, CH_GATED_DIFF_BASE)

    # --- Head wiring ---
    head_entries = []

    # GCD
    _add_key_gated_head_entries(
        head_entries, lambda a, b: math.gcd(a, b),
        GCD_SLOT_BASE, CH_GATED_GCD_BASE,
    )

    # FACT
    for n in range(FACT_MAX_N + 1):
        slot = FACT_SLOT_BASE + n
        head_entries.append((slot, CH_GATED_FACT_BASE + n, 1.0))
        if n + 1 <= FACT_MAX_N:
            head_entries.append((slot, CH_GATED_FACT_BASE + n + 1, -1.0))

    # PRIME
    for n in range(PRIME_MIN_N, PRIME_MAX_N + 1):
        slot = PRIME_SLOT_TRUE if _is_prime(n) else PRIME_SLOT_FALSE
        ch = CH_GATED_PRIME_BASE + (n - PRIME_MIN_N)
        head_entries.append((slot, ch, 1.0))
        head_entries.append((slot, ch + 1, -1.0))

    # ADD
    for s in range(ADD_MAX + 1):
        slot = ADD_SLOT_BASE + s
        head_entries.append((slot, CH_GATED_ADD_BASE + s, 1.0))
        if s + 1 <= ADD_MAX:
            head_entries.append((slot, CH_GATED_ADD_BASE + s + 1, -1.0))

    # MUL
    _add_key_gated_head_entries(
        head_entries, lambda a, b: a * b,
        MUL_SLOT_BASE, CH_GATED_MUL_BASE,
    )

    # MOD — b=0 returns 0 by convention (undefined; caller should avoid)
    _add_key_gated_head_entries(
        head_entries, lambda a, b: (a % b) if b > 0 else 0,
        MOD_SLOT_BASE, CH_GATED_MOD_BASE,
    )

    # MIN
    _add_key_gated_head_entries(
        head_entries, lambda a, b: min(a, b),
        MIN_SLOT_BASE, CH_GATED_MIN_BASE,
    )

    # MAX
    _add_key_gated_head_entries(
        head_entries, lambda a, b: max(a, b),
        MAX_SLOT_BASE, CH_GATED_MAX_BASE,
    )

    # DIFF
    _add_key_gated_head_entries(
        head_entries, lambda a, b: abs(a - b),
        DIFF_SLOT_BASE, CH_GATED_DIFF_BASE,
    )

    graph.add(LinearHead(name="dispatched_v3_head", entries=head_entries))

    n_layers = auto_schedule(graph)
    n_heads = D_MODEL // 2
    # Layer 1 has all gated banks concurrently.
    d_ffn = (
        2 * (GCD_MAX_KEY + 1)    # GCD
        + 2 * (FACT_MAX_N + 1)    # FACT
        + 2 * (PRIME_MAX_N - PRIME_MIN_N + 2)  # PRIME
        + 2 * (ADD_MAX + 1)       # ADD
        + 5 * 2 * (GCD_MAX_KEY + 1)  # MUL, MOD, MIN, MAX, DIFF
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
    if opcode == 0:
        return slot
    if opcode == 1:
        return math.factorial(slot - FACT_SLOT_BASE)
    if opcode == 2:
        return slot == PRIME_SLOT_TRUE
    if opcode == 3:
        return slot - ADD_SLOT_BASE
    if opcode == 4:
        return slot - MUL_SLOT_BASE
    if opcode == 5:
        return slot - MOD_SLOT_BASE
    if opcode == 6:
        return slot - MIN_SLOT_BASE
    if opcode == 7:
        return slot - MAX_SLOT_BASE
    if opcode == 8:
        return slot - DIFF_SLOT_BASE
    raise ValueError(f"unknown opcode: {opcode}")


if __name__ == "__main__":
    import itertools
    import time
    import torch

    t0 = time.time()
    model = build_dispatched_v3()
    t_build = time.time() - t0
    print(f"[dispatched_v3] built in {t_build:.1f}s, "
          f"{model.param_count():,} params (d_model={D_MODEL}, vocab={VOCAB})")

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

    t0 = time.time()
    ok = 0
    tot = 0

    pairs = list(itertools.product(range(GCD_BASE), repeat=2))

    gcd_in = [(a, b, 0) for a, b in pairs]
    gcd_ex = [math.gcd(a, b) for (a, b, _) in gcd_in]
    c, n = _run(gcd_in, gcd_ex, "GCD      "); ok += c; tot += n

    fact_in = [(n, 0, 1) for n in range(FACT_MAX_N + 1)]
    fact_ex = [math.factorial(n) for (n, _, _) in fact_in]
    c, n = _run(fact_in, fact_ex, "FACTORIAL"); ok += c; tot += n

    prime_in = [(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]
    prime_ex = [_is_prime(n) for (n, _, _) in prime_in]
    c, n = _run(prime_in, prime_ex, "IS_PRIME "); ok += c; tot += n

    add_in = [(a, b, 3) for a, b in pairs]
    add_ex = [a + b for (a, b, _) in add_in]
    c, n = _run(add_in, add_ex, "ADD      "); ok += c; tot += n

    mul_in = [(a, b, 4) for a, b in pairs]
    mul_ex = [a * b for (a, b, _) in mul_in]
    c, n = _run(mul_in, mul_ex, "MUL      "); ok += c; tot += n

    # MOD: skip b=0 (undefined)
    mod_in = [(a, b, 5) for a, b in pairs if b > 0]
    mod_ex = [a % b for (a, b, _) in mod_in]
    c, n = _run(mod_in, mod_ex, "MOD      "); ok += c; tot += n

    min_in = [(a, b, 6) for a, b in pairs]
    min_ex = [min(a, b) for (a, b, _) in min_in]
    c, n = _run(min_in, min_ex, "MIN      "); ok += c; tot += n

    max_in = [(a, b, 7) for a, b in pairs]
    max_ex = [max(a, b) for (a, b, _) in max_in]
    c, n = _run(max_in, max_ex, "MAX      "); ok += c; tot += n

    diff_in = [(a, b, 8) for a, b in pairs]
    diff_ex = [abs(a - b) for (a, b, _) in diff_in]
    c, n = _run(diff_in, diff_ex, "DIFF     "); ok += c; tot += n

    t_run = time.time() - t0
    print(f"\n[dispatched_v3] total: {ok}/{tot} — "
          f"{'PASS' if ok == tot else 'FAIL'}  (run {t_run:.1f}s)")
