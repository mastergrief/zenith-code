"""countdown — autoregressive compute loop via Small2DTransformer substrate.

Stage 2 of the UTM roadmap: *the substrate runs iterative programs via
autoregressive decoding*. Each forward pass is one tick of a machine;
the generated token sequence is the tape.

This program implements: decrement-until-zero.
  Input:  [V]              (starting value V ∈ [0, V_MAX])
  Output: [V, V-1, V-2, ..., 0, HALT]   (via autoregressive loop)

At each position i the compiled head outputs:
  - token  (current_value - 1)   if current_value ≥ 1
  - token  HALT                  if current_value == 0

The Python-level autoregressive runner feeds each output token back as
the next position's input and stops on HALT. No new IR primitives — the
existing step-diff decode handles both the subtraction and the HALT
signal via separate head entries.

Significance: this is the same machinery a full UTM needs — an
instruction, a state register (own_scalar here), and a halt condition
baked into the ISA. Scaling up (more instructions, a proper tape, an
instruction pointer) is adding more compiled subgraphs and wiring them
through dispatched.py's opcode routing.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


V_MAX = 8            # maximum register value
HALT = V_MAX + 1     # token id for HALT = 9
VOCAB = V_MAX + 2    # 0..V_MAX + HALT


def build_countdown(max_len: int = 16) -> Small2DTransformer:
    V = VOCAB
    g = GateGraph(vocab_size=V)

    # ch 0 = own_scalar (token value, maps 0..V_MAX to their integer value;
    # HALT maps to V_MAX+1 as a sentinel — once HALT is the token, the
    # runner stops before we'd execute another step on it).
    g.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    g.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))

    # Step decode: step_S = 1[value >= S] for S in [0, V_MAX + 1].
    # Need step_{V_MAX+1} for cancellation at the top of the range.
    for S in range(V_MAX + 2):
        g.add(ReGLU(
            name=f"step_{S}_hi",
            gate=[(0, 1.0), (1, -(S - 1))],
            val=[(1, 1.0)],
            output_channel=2 + S,
            output_coef=1.0,
        ))
        g.add(ReGLU(
            name=f"step_{S}_lo",
            gate=[(0, 1.0), (1, -S)],
            val=[(1, 1.0)],
            output_channel=2 + S,
            output_coef=-1.0,
        ))

    # Head wiring:
    #   For value v ∈ [1, V_MAX]: logits[v - 1] += step_v - step_{v+1}
    #     → emits the decremented value.
    #   For value 0: logits[HALT] += step_0 - step_1
    #     → emits HALT when register reaches zero.
    head_entries = []
    for v in range(1, V_MAX + 1):
        head_entries.append((v - 1, 2 + v, 1.0))
        head_entries.append((v - 1, 2 + v + 1, -1.0))
    # HALT emit
    head_entries.append((HALT, 2 + 0, 1.0))
    head_entries.append((HALT, 2 + 1, -1.0))
    g.add(LinearHead(name="countdown_head", entries=head_entries))

    n_layers = auto_schedule(g)
    d_model = 2 + (V_MAX + 2)        # 12
    if d_model % 2 != 0:
        d_model += 1                  # pad for d_head=2
    n_heads = d_model // 2
    d_ffn = 2 * (V_MAX + 2)           # 20
    return compile_program(
        g,
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn,
        max_len=max_len, vocab_size=V,
    )


def run_countdown(start_value: int, max_steps: int = 20, model=None):
    """Execute the autoregressive loop. Returns the full sequence."""
    import torch
    if model is None:
        model = build_countdown()
    if not (0 <= start_value <= V_MAX):
        raise ValueError(f"start_value must be in [0, {V_MAX}], got {start_value}")

    seq = [start_value]
    for _ in range(max_steps):
        x = torch.tensor([seq], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        next_tok = int(logits[0, -1, :].argmax().item())
        seq.append(next_tok)
        if next_tok == HALT:
            break
    return seq


if __name__ == "__main__":
    model = build_countdown()
    print(f"[countdown] built, {model.param_count():,} params")

    all_ok = True
    for v in range(V_MAX + 1):
        seq = run_countdown(v, model=model)
        expected = list(range(v, -1, -1)) + [HALT]
        ok = seq == expected
        all_ok = all_ok and ok
        mark = "✓" if ok else "✗"
        print(f"  {mark} start={v}  →  {seq}  (expected {expected})")
    print(f"\noverall: {'PASS' if all_ok else 'FAIL'}")
