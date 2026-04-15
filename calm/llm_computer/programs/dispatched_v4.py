"""dispatched_v4 — dispatched_v2 with shifted opcodes for cross-card gating.

Same 5 ops as dispatched_v2 (GCD, FACT, PRIME, ADD, MUL) but opcode
tokens are shifted by +1: valid opcodes are [1, N_OPS] instead of
[0, N_OPS-1]. Token 0 is reserved as "no op" — when pos-2 token is 0
(or any value outside [1, N_OPS]), ALL gated step banks output exactly
zero, and the card contributes zero to every one of its head slots.

This is the cross-card routing primitive from Round 10 (Z): a card
installed next to another card (e.g., HRM, a different dispatched
instance) only fires when pos-2 token falls in its valid opcode range.
For any input where another card is the intended recipient, this card
is silent. Cross-card dispatch emerges from the shift + the shared
vocabulary convention — no new layers, no new channels.

Threshold details:
  * opcode_step_k = 1[CH_OWN >= k] (step function, k ∈ [1, N_OPS+1])
  * is_op_k = step_{k+1} - step_{k+2} = 1 iff CH_OWN == k+1
  * User-facing opcode `op` ∈ [0, N_OPS-1] maps to pos-2 token `op+1`.

Decoding is unchanged — the user-facing opcode is `op`, the card just
expects token `op + 1` at position 2.
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

# Re-use dispatched_v2's numeric constants.
from calm.llm_computer.programs.dispatched_v2 import (
    ADD_MAX, ADD_SLOT_BASE, CH_BIAS, CH_COPY_A, CH_COPY_B, CH_GATED_ADD_BASE,
    CH_GATED_FACT_BASE, CH_GATED_GCD_BASE, CH_GATED_MUL_BASE,
    CH_GATED_PRIME_BASE, CH_KEY, CH_OPCODE_STEP_BASE, CH_OWN, CH_POS_K0,
    CH_POS_K1, CH_TARGET_POS, D_MODEL, FACT_MAX_N, FACT_SLOT_BASE, GCD_BASE,
    GCD_MAX_KEY, MUL_MAX_KEY, MUL_MAX_OPERAND, MUL_SLOT_BASE, N_OPS,
    PRIME_MAX_N, PRIME_MIN_N, PRIME_SLOT_FALSE, PRIME_SLOT_TRUE, VOCAB,
)


# Opcode offset: user-facing opcode k is passed as token k + OPCODE_SHIFT.
OPCODE_SHIFT = 1


def _is_op_val(k: int):
    """Use the shifted step pair — is_op_k fires when CH_OWN = k + SHIFT."""
    return [
        (CH_OPCODE_STEP_BASE + k, 1.0),
        (CH_OPCODE_STEP_BASE + k + 1, -1.0),
    ]


def build_dispatched_v4(max_len: int = 3) -> Small2DTransformer:
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

    # KEY = GCD_BASE · copy_a + copy_b (shared between GCD, MUL).
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

    # Opcode step bank — thresholds SHIFTED by OPCODE_SHIFT compared to
    # dispatched_v2. Threshold for opcode_step_k is k + SHIFT.
    for k in range(N_OPS + 1):
        threshold = k + OPCODE_SHIFT
        graph.add(ReGLU(
            name=f"opcode_step_{k}_hi",
            gate=[(CH_OWN, 1.0), (CH_BIAS, -(threshold - 1))],
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_OPCODE_STEP_BASE + k,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"opcode_step_{k}_lo",
            gate=[(CH_OWN, 1.0), (CH_BIAS, -threshold)],
            val=[(CH_BIAS, 1.0)],
            output_channel=CH_OPCODE_STEP_BASE + k,
            output_coef=-1.0,
        ))

    # Gated step banks (unchanged from v2).
    is_op_gcd = _is_op_val(0)
    is_op_fact = _is_op_val(1)
    is_op_prime = _is_op_val(2)
    is_op_add = _is_op_val(3)
    is_op_mul = _is_op_val(4)

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

    # Head — same entries as v2 (copy from v2 exactly).
    head_entries = []
    for A in range(GCD_BASE):
        for B in range(GCD_BASE):
            kk = GCD_BASE * A + B
            g = math.gcd(A, B)
            head_entries.append((g, CH_GATED_GCD_BASE + kk, 1.0))
            if kk + 1 <= GCD_MAX_KEY:
                head_entries.append((g, CH_GATED_GCD_BASE + kk + 1, -1.0))
    for n in range(FACT_MAX_N + 1):
        slot = FACT_SLOT_BASE + n
        head_entries.append((slot, CH_GATED_FACT_BASE + n, 1.0))
        if n + 1 <= FACT_MAX_N:
            head_entries.append((slot, CH_GATED_FACT_BASE + n + 1, -1.0))
    for n in range(PRIME_MIN_N, PRIME_MAX_N + 1):
        slot = PRIME_SLOT_TRUE if _is_prime(n) else PRIME_SLOT_FALSE
        ch = CH_GATED_PRIME_BASE + (n - PRIME_MIN_N)
        head_entries.append((slot, ch, 1.0))
        head_entries.append((slot, ch + 1, -1.0))
    for s in range(ADD_MAX + 1):
        slot = ADD_SLOT_BASE + s
        head_entries.append((slot, CH_GATED_ADD_BASE + s, 1.0))
        if s + 1 <= ADD_MAX:
            head_entries.append((slot, CH_GATED_ADD_BASE + s + 1, -1.0))
    for A in range(MUL_MAX_OPERAND + 1):
        for B in range(MUL_MAX_OPERAND + 1):
            kk = GCD_BASE * A + B
            prod = A * B
            slot = MUL_SLOT_BASE + prod
            head_entries.append((slot, CH_GATED_MUL_BASE + kk, 1.0))
            if kk + 1 <= MUL_MAX_KEY:
                head_entries.append((slot, CH_GATED_MUL_BASE + kk + 1, -1.0))
    graph.add(LinearHead(name="dispatched_v4_head", entries=head_entries))

    n_layers = auto_schedule(graph)
    n_heads = D_MODEL // 2
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
    """Decode slot → value for a given user-facing opcode ∈ [0, N_OPS-1].

    (Pos-2 token at inference time is `opcode + OPCODE_SHIFT`.)"""
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
    raise ValueError(f"unknown opcode: {opcode}")


if __name__ == "__main__":
    import itertools
    import time
    import torch

    t0 = time.time()
    model = build_dispatched_v4()
    print(f"[dispatched_v4] built in {time.time() - t0:.1f}s, "
          f"{model.param_count():,} params")

    # CHECK (1) — valid inputs still produce correct outputs.
    # Input tokens are now (a, b, opcode + OPCODE_SHIFT).
    def _run(inputs, expected, label):
        shifted_inputs = [(a, b, op + OPCODE_SHIFT) for (a, b, op) in inputs]
        x = torch.tensor(shifted_inputs, dtype=torch.long)
        with torch.no_grad():
            preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
        correct = sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        print(f"  {label}: {correct}/{len(inputs)}")
        return correct, len(inputs)

    ok = tot = 0
    pairs = list(itertools.product(range(GCD_BASE), repeat=2))
    c, n = _run([(a, b, 0) for a, b in pairs],
                [math.gcd(a, b) for a, b in pairs], "GCD      "); ok += c; tot += n
    c, n = _run([(n, 0, 1) for n in range(FACT_MAX_N + 1)],
                [math.factorial(n) for n in range(FACT_MAX_N + 1)],
                "FACTORIAL"); ok += c; tot += n
    c, n = _run([(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
                [_is_prime(n) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
                "IS_PRIME "); ok += c; tot += n
    c, n = _run([(a, b, 3) for a, b in pairs],
                [a + b for a, b in pairs], "ADD      "); ok += c; tot += n
    mul_pairs = list(itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2))
    c, n = _run([(a, b, 4) for a, b in mul_pairs],
                [a * b for a, b in mul_pairs], "MUL      "); ok += c; tot += n
    print(f"  valid-input total: {ok}/{tot} — "
          f"{'PASS' if ok == tot else 'FAIL'}")

    # CHECK (2) — invalid opcode at pos 2 ⇒ all card slots ≈ 0.
    print("\n[dispatched_v4] CROSS-CARD GATING check:")
    print("  feeding opcode token 0 (no-op) — expect all slots silent...")
    silent_inputs = torch.tensor([
        [5, 3, 0],   # opcode token 0 = "not my input"
        [0, 0, 0],
        [7, 9, 0],
        [15, 15, 0],
    ], dtype=torch.long)
    with torch.no_grad():
        silent_logits = model(silent_inputs)[:, 2, :]
    max_silent = silent_logits.abs().max().item()
    mean_silent = silent_logits.abs().mean().item()
    ok_silent = max_silent < 1e-5
    print(f"  max |logit| = {max_silent:.2e}, mean |logit| = {mean_silent:.2e} — "
          f"{'PASS' if ok_silent else 'FAIL'}")

    # Also invalid opcodes above range.
    print("  feeding opcode token N_OPS+1 = 6 (out of range) — expect silent...")
    out_of_range = torch.tensor([
        [5, 3, 6], [7, 2, 7], [9, 4, 10],
    ], dtype=torch.long)
    with torch.no_grad():
        oor_logits = model(out_of_range)[:, 2, :]
    max_oor = oor_logits.abs().max().item()
    ok_oor = max_oor < 1e-5
    print(f"  max |logit| = {max_oor:.2e} — "
          f"{'PASS' if ok_oor else 'FAIL'}")

    all_ok = (ok == tot) and ok_silent and ok_oor
    print(f"\n[dispatched_v4] OVERALL: {'PASS' if all_ok else 'FAIL'}")
