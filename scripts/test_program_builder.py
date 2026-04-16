"""Round-25 test: facade + import system builds a 4-stage compound program.

The pipeline (all via imports, no hardcoded channel numbers in ops):

  stdlib → exports: a (channel 3), b (channel 4), bias (channel 1)
  adder  → imports(a, b, bias) → exports: sum (channel 5)
  doubler → imports(sum, bias) → exports: doubled (channel 6)
  classifier → imports(doubled, bias) → exports: is_large (channel 7)
  head → reads is_large + bias → slots 0 (small) / 1 (large)

Expected: slot 1 iff a + b >= 4 (because 2*(a+b) >= 8).
"""

from __future__ import annotations

import itertools

import torch

from calm.llm_computer.program_builder import (
    CompiledOp, HeadSpec, StdLib, bias_val, build_program,
    simple_val, step_gate_ge, sum_val,
)


def main():
    print("[R25] defining stdlib (layer 0 facade)...")
    stdlib = StdLib(
        vocab_size=8,
        max_len=4,
        exports={
            "own": 0,    # token scalar
            "bias": 1,   # constant 1
            "a": 3,      # operand a (copied from pos 0)
            "b": 4,      # operand b = own at pos 1 (alias, same as own at pos 1)
        },
        copy_pairs=[
            ("a", 0, 3),  # LookUp: copy pos 0's token → channel 3
        ],
        token_channel=0,
        bias_channel=1,
    )
    # Note: "b" is just CH_OWN at pos 1. The token embed writes own=0
    # for all tokens. We actually need "b" to come from CH_OWN directly.
    # Fix: "b" maps to channel 0 (own). The adder reads (a=ch3, b=ch0).
    stdlib.exports["b"] = 0  # b = own token at pos 1

    print("[R25] defining compiled ops (each declares imports + exports)...")

    adder = CompiledOp(
        name="adder",
        imports={"a": "a", "b": "b", "bias": "bias"},
        gate=lambda ch: [(ch["bias"], 1.0)],
        val=lambda ch: [(ch["a"], 1.0), (ch["b"], 1.0)],
        export_channel=5,
        export_name="sum",
    )

    doubler = CompiledOp(
        name="doubler",
        imports={"sum": "sum", "bias": "bias"},
        gate=lambda ch: [(ch["bias"], 1.0)],
        val=lambda ch: [(ch["sum"], 2.0)],
        export_channel=6,
        export_name="doubled",
    )

    classifier = CompiledOp(
        name="classifier",
        imports={"doubled": "doubled", "bias": "bias"},
        step_thresholds=[8],
        gate=lambda ch, t=0, lo=False: [
            (ch["doubled"], 1.0),
            (ch["bias"], -(t - 1) if not lo else -t),
        ],
        val=lambda ch: [(ch["bias"], 1.0)],
        export_channel=7,
        export_name="is_large",
    )

    head = HeadSpec(entries=[
        (0, 1, 1.0),   # slot 0 = bias = 1 (always)
        (1, 7, 2.0),   # slot 1 = 2 * is_large (2 if large, 0 if not)
    ])

    print("[R25] building program (linker resolves imports, schedules layers)...")
    model = build_program(
        stdlib, [adder, doubler, classifier], head,
        d_model=16, d_ffn=8,
    )
    print(f"  d_model={model.config.d_model}, n_layers={model.config.n_layers}, "
          f"params={model.param_count()}")

    # Exhaustive test
    print("\n[R25] exhaustive test: a+b >= 4 → slot 1, else slot 0")
    ok = total = 0
    for a, b in itertools.product(range(4), repeat=2):
        x = torch.tensor([[a, b]], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        pred = int(logits[0, 1].argmax().item())
        expected = 1 if a + b >= 4 else 0
        if pred == expected:
            ok += 1
        else:
            print(f"  [✗] a={a} b={b} sum={a+b} pred={pred} expected={expected}")
        total += 1
    print(f"  result: {ok}/{total} — {'PASS' if ok == total else 'FAIL'}")

    # Trace
    print("\n[R25] pipeline trace for a=3, b=2:")
    x = torch.tensor([[3, 2]], dtype=torch.long)
    with torch.no_grad():
        import torch.nn.functional as F
        cfg = model.config
        pos_idx = torch.arange(2)
        res = model.tok(x) + model.pos(pos_idx)
        mask = torch.triu(torch.ones(2, 2, dtype=torch.bool), diagonal=1)
        for layer in range(cfg.n_layers):
            qkv = model.W_qkv[layer](res)
            qkv = qkv.reshape(1, 2, 3, cfg.n_heads, 2)
            q, k, v = qkv.permute(2, 0, 3, 1, 4)
            scores = torch.einsum("bhid,bhjd->bhij", q, k)
            scores = scores.masked_fill(mask, float("-inf"))
            idx_hm = scores.argmax(dim=-1, keepdim=True)
            weights = torch.zeros_like(scores)
            weights.scatter_(-1, idx_hm, 1.0)
            attn = torch.einsum("bhij,bhjd->bhid", weights, v)
            attn = attn.transpose(1, 2).reshape(1, 2, cfg.d_model)
            res = res + model.W_out[layer](attn)
            gate, val = model.ff_in[layer](res).chunk(2, dim=-1)
            res = res + model.ff_out[layer](F.relu(gate) * val)
            r = res[0, 1]
            print(f"  layer {layer}: sum={r[5].item():.1f} "
                  f"doubled={r[6].item():.1f} is_large={r[7].item():.1f}")
    print(f"  expected: sum=5, doubled=10, is_large=1 (10 >= 8)")

    # Test import resolution: bad import should fail
    print("\n[R25] testing import resolution error handling...")
    bad_op = CompiledOp(
        name="bad",
        imports={"x": "nonexistent_export"},
        gate=lambda ch: [(ch["x"], 1.0)],
        val=lambda ch: [(ch["x"], 1.0)],
        export_channel=8,
        export_name="bad_out",
    )
    try:
        build_program(stdlib, [bad_op], head, d_model=16)
        print("  [✗] should have raised KeyError")
    except KeyError as e:
        print(f"  [✓] correctly caught: {e}")


if __name__ == "__main__":
    main()
