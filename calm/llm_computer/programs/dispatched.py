"""dispatched — weight-level opcode dispatcher for gcd / factorial / is_prime.

Program: input `[a, b, opcode]` at positions 0, 1, 2. Output at position 2:
  - opcode=0 (gcd):       argmax slot == gcd(a, b), in [0, 15]
  - opcode=1 (factorial): argmax slot == FACT_SLOT_BASE + n for a = n;
                          caller decodes: value = factorial(slot - FACT_SLOT_BASE)
  - opcode=2 (is_prime):  slot == PRIME_SLOT_TRUE  if a is prime
                          slot == PRIME_SLOT_FALSE otherwise
For opcode 1 and 2 only `a` is used (pass b=0).

Weight-level routing without extra layers: each gated step ReGLU combines
the step-function shape with the opcode gate directly — `val` holds
`is_op_k` (a linear combo of opcode_step channels), `gate` holds the
step-function shape. Since `is_op_k ∈ {0, 1}` and ReLU preserves it,
`val · ReLU(gate) = is_op_k · step_k` in one ReGLU pair.

Residual channel layout (d_model = 292):
  0: own_scalar  (TokenEmbed: tok[k, 0] = k)
  1: bias        (PosEmbed: 1 at every pos)
  2: pos_k0      (PosEmbed: 2p at each pos j)
  3: pos_k1      (PosEmbed: -j² at each pos j)
  4: target_pos  (PosEmbed: 1 at pos 2 only — the "fetch-from-pos-1" query key)
  5: copy_a      (LookUp copy-from-pos-0 into ch 5)
  6: copy_b      (LookUpExact selects pos 1 via parabolic key, writes b)
  7: key_gcd     (layer 0 FFN: 16·copy_a + copy_b)
  8..11 (4):     opcode_step_k for k ∈ [0, 3]  (layer 0 FFN, on ch 0)
  12..267 (256): gated_gcd_step_k     (layer 1 FFN: is_op_0 · step_k(key_gcd))
  268..276 (9):  gated_fact_step_k    (layer 1 FFN: is_op_1 · step_k(copy_a))
  277..291 (15): gated_prime_step_k   (layer 1 FFN: is_op_2 · step_k(copy_a),
                                        k ∈ [PRIME_MIN_N, PRIME_MAX_N + 1])

All three sub-programs share the head. Only the opcode-matched gated
channels carry nonzero step-diff contributions; the others zero out.
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


GCD_MAX = 15
GCD_BASE = GCD_MAX + 1              # 16
GCD_MAX_KEY = GCD_BASE * GCD_BASE - 1   # 255
FACT_MAX_N = 8
PRIME_MIN_N = 2
PRIME_MAX_N = 15

# Output-slot encoding (compact vocab). Caller decodes the logical value.
FACT_SLOT_BASE = 16                 # slots 16..24 = factorial(0..8)
PRIME_SLOT_FALSE = 25
PRIME_SLOT_TRUE = 26
VOCAB = 27

# Channel layout constants.
CH_OWN = 0
CH_BIAS = 1
CH_POS_K0 = 2
CH_POS_K1 = 3
CH_TARGET_POS = 4
CH_COPY_A = 5
CH_COPY_B = 6
CH_KEY_GCD = 7
CH_OPCODE_STEP_BASE = 8            # 4 channels: opcode_step_0..3
CH_GATED_GCD_BASE = 12             # 256 channels: gated_gcd_step_0..255
CH_GATED_FACT_BASE = 12 + 256      # 9 channels: gated_fact_step_0..8
CH_GATED_PRIME_BASE = 12 + 256 + 9 # 15 channels: gated_prime_step_2..16
D_MODEL = 12 + 256 + 9 + 15        # 292 (even → d_head=2 holds)


def _is_op_val(k: int):
    """Return a ChannelLC for is_op_k = opcode_step_k - opcode_step_{k+1}."""
    return [
        (CH_OPCODE_STEP_BASE + k, 1.0),
        (CH_OPCODE_STEP_BASE + k + 1, -1.0),
    ]


def build_dispatched(max_len: int = 3) -> Small2DTransformer:
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
    # target_pos: 1 at query pos 2 only (to select pos 1 via LookUpExact).
    pos_entries.append((2, CH_TARGET_POS, 1.0))
    graph.add(PosEmbed(name="pos_consts", entries=pos_entries))

    # Layer 0 attn: copy_a from pos 0 (first-tie LookUp) + copy_b from pos 1
    # (LookUpExact with target=1).
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

    # Layer 0 FFN: key_gcd = 16·copy_a + copy_b.
    graph.add(ReGLU(
        name="key_scale_a",
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_A, 1.0)],
        output_channel=CH_KEY_GCD,
        output_coef=float(GCD_BASE),
    ))
    graph.add(ReGLU(
        name="key_add_b",
        gate=[(CH_BIAS, 1.0)],
        val=[(CH_COPY_B, 1.0)],
        output_channel=CH_KEY_GCD,
        output_coef=1.0,
    ))

    # Layer 0 FFN: opcode_step_k = 1[opcode >= k] for k ∈ [0, 3] (on ch 0 at
    # pos 2 — only pos 2 holds the opcode). At pos 0/1 this channel holds
    # spurious values but we don't query those positions.
    for k in range(4):
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

    # Layer 1 FFN: gated step functions — each pair embeds is_op_k in `val`.
    # gated_gcd_step_k: step_k(key_gcd) · is_op_0, for k ∈ [0, GCD_MAX_KEY].
    is_op_0 = _is_op_val(0)
    is_op_1 = _is_op_val(1)
    is_op_2 = _is_op_val(2)
    for k in range(GCD_MAX_KEY + 1):
        ch = CH_GATED_GCD_BASE + k
        graph.add(ReGLU(
            name=f"gcd_gated_{k}_hi",
            gate=[(CH_KEY_GCD, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_0,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"gcd_gated_{k}_lo",
            gate=[(CH_KEY_GCD, 1.0), (CH_BIAS, -k)],
            val=is_op_0,
            output_channel=ch,
            output_coef=-1.0,
        ))
    # gated_fact_step_k: step_k(copy_a) · is_op_1, for k ∈ [0, FACT_MAX_N].
    for k in range(FACT_MAX_N + 1):
        ch = CH_GATED_FACT_BASE + k
        graph.add(ReGLU(
            name=f"fact_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_1,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"fact_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -k)],
            val=is_op_1,
            output_channel=ch,
            output_coef=-1.0,
        ))
    # gated_prime_step_k: step_k(copy_a) · is_op_2, for k ∈ [PRIME_MIN_N, PRIME_MAX_N + 1].
    for k in range(PRIME_MIN_N, PRIME_MAX_N + 2):
        ch = CH_GATED_PRIME_BASE + (k - PRIME_MIN_N)
        graph.add(ReGLU(
            name=f"prime_gated_{k}_hi",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -(k - 1))],
            val=is_op_2,
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"prime_gated_{k}_lo",
            gate=[(CH_COPY_A, 1.0), (CH_BIAS, -k)],
            val=is_op_2,
            output_channel=ch,
            output_coef=-1.0,
        ))

    # Head wiring: one step-diff entry per output case. compile.py's `+=`
    # accumulation means conflicting +1/-1 contributions on the same
    # (slot, channel) naturally sum.
    head_entries = []
    # gcd: slot = gcd(A, B), channel = gated_gcd_step_{16A+B}.
    for A in range(GCD_BASE):
        for B in range(GCD_BASE):
            kk = GCD_BASE * A + B
            g = math.gcd(A, B)
            head_entries.append((g, CH_GATED_GCD_BASE + kk, 1.0))
            if kk + 1 <= GCD_MAX_KEY:
                head_entries.append((g, CH_GATED_GCD_BASE + kk + 1, -1.0))
    # factorial: slot = FACT_SLOT_BASE + n, channel = gated_fact_step_n.
    for n in range(FACT_MAX_N + 1):
        slot = FACT_SLOT_BASE + n
        head_entries.append((slot, CH_GATED_FACT_BASE + n, 1.0))
        if n + 1 <= FACT_MAX_N:
            head_entries.append((slot, CH_GATED_FACT_BASE + n + 1, -1.0))
    # is_prime: slot = PRIME_SLOT_TRUE / PRIME_SLOT_FALSE.
    for n in range(PRIME_MIN_N, PRIME_MAX_N + 1):
        slot = PRIME_SLOT_TRUE if _is_prime(n) else PRIME_SLOT_FALSE
        ch = CH_GATED_PRIME_BASE + (n - PRIME_MIN_N)
        head_entries.append((slot, ch, 1.0))
        head_entries.append((slot, ch + 1, -1.0))
    graph.add(LinearHead(name="dispatched_head", entries=head_entries))

    n_layers = auto_schedule(graph)

    n_heads = D_MODEL // 2
    d_ffn = 2 * (GCD_MAX_KEY + 1) + 2 * (FACT_MAX_N + 1) + 2 * (PRIME_MAX_N - PRIME_MIN_N + 2)
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
    """Decode the output slot into the logical value given the opcode."""
    if opcode == 0:
        return slot                              # gcd value
    if opcode == 1:
        return math.factorial(slot - FACT_SLOT_BASE)
    if opcode == 2:
        return slot == PRIME_SLOT_TRUE
    raise ValueError(f"unknown opcode: {opcode}")


def run_program(model, opcode: int, a: int, b: int = 0):
    """Convenience: construct input, run, decode. Useful for demo / tests."""
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
    model = build_dispatched()
    t_build = time.time() - t0
    print(f"[dispatched] built in {t_build:.1f}s, {model.param_count():,} params")

    # Exhaustive check per sub-program.
    print("\n[dispatched] gcd exhaustive...")
    gcd_inputs = [(a, b, 0) for a, b in itertools.product(range(GCD_BASE), repeat=2)]
    x = torch.tensor(gcd_inputs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    expected = [math.gcd(a, b) for a, b, _ in gcd_inputs]
    gcd_correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"  gcd {gcd_correct}/{len(gcd_inputs)}")

    print("[dispatched] factorial exhaustive...")
    fact_inputs = [(n, 0, 1) for n in range(FACT_MAX_N + 1)]
    x = torch.tensor(fact_inputs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    fact_correct = sum(
        1 for p, (n, _, _) in zip(preds, fact_inputs)
        if decode_output(1, p) == math.factorial(n)
    )
    print(f"  factorial {fact_correct}/{len(fact_inputs)}")

    print("[dispatched] is_prime exhaustive...")
    prime_inputs = [(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]
    x = torch.tensor(prime_inputs, dtype=torch.long)
    with torch.no_grad():
        preds = model(x)[:, 2, :].argmax(dim=-1).tolist()
    prime_correct = sum(
        1 for p, (n, _, _) in zip(preds, prime_inputs)
        if decode_output(2, p) == _is_prime(n)
    )
    print(f"  is_prime {prime_correct}/{len(prime_inputs)}")

    total = len(gcd_inputs) + len(fact_inputs) + len(prime_inputs)
    correct = gcd_correct + fact_correct + prime_correct
    print(f"\n[dispatched] total: {correct}/{total}")

    # Mixed-opcode demo
    print("\n=== Mixed-opcode demo ===")
    demos = [
        (0, 12, 15, math.gcd(12, 15)),
        (0, 7, 13, math.gcd(7, 13)),
        (1, 5, 0, math.factorial(5)),
        (1, 0, 0, math.factorial(0)),
        (1, 8, 0, math.factorial(8)),
        (2, 7, 0, True),
        (2, 9, 0, False),
        (2, 2, 0, True),
    ]
    for opcode, a, b, expected in demos:
        got = run_program(model, opcode, a, b)
        mark = "✓" if got == expected else "✗"
        op_name = {0: "gcd", 1: "factorial", 2: "is_prime"}[opcode]
        arg = f"{a}, {b}" if opcode == 0 else f"{a}"
        print(f"  {mark} {op_name}({arg}) = {got}  (expected {expected})")
