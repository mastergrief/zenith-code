"""Hand-written `threshold` weights — exercises FFN via the paper's step function.

Program: output `1` if input_token >= 4, else `0`.

The step function primitive (doc 03 §4c) turns a ReLU-activated FFN into
an exact 0/1 indicator on integer inputs:

    1[z ≥ 0] = ReLU(z + 1) − ReLU(z)

Implementation uses two ReGLU neurons in a single FFN layer:
  - neuron 0: gate = input - 3, val = +1  →  ReLU(input - 3)
  - neuron 1: gate = input - 4, val = -1  →  -ReLU(input - 4)
  - ff_out sums: ReLU(input - 3) - ReLU(input - 4) = 1[input ≥ 4]

Residual layout (d_model = 4):
  - dim 0: input scalar (via tok[k][0] = k)
  - dim 1: always 1 (via pos[p][1] = 1 — the "bias channel" that lets
           FFN produce constant shifts without a bias parameter)
  - dim 2: step-function output (written by FFN)
  - dim 3: unused

Head reads only dim 2:
  - step = 1 → logits = [0, 1, 0, 0, ...] → argmax = 1
  - step = 0 → logits all zero → first-tie argmax = 0

This is the first program in the library that uses a FFN as actual compute.
Residual wiring (add_one), attention (copy_past), and position embedding
(increment_counter) were the prior primitives.
"""

from __future__ import annotations

import torch

from calm.llm_computer.model import Small2DTransformer, Small2DConfig


def build_threshold(
    vocab_size: int = 8,
    threshold_value: int = 4,
) -> Small2DTransformer:
    V = vocab_size
    T = threshold_value

    cfg = Small2DConfig(
        vocab_size=V,
        d_model=4,           # [input_scalar, bias, step_out, unused]
        n_heads=2,           # d_head = 2
        n_layers=1,
        d_ffn=2,             # two ReGLU neurons for the step function
        max_len=16,
        use_hard_max=True,
    )
    assert cfg.d_head == 2

    model = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in model.parameters():
            p.zero_()

        # Token embedding: dim 0 carries input scalar k.
        for k in range(V):
            model.tok.weight[k, 0] = float(k)

        # Position embedding: dim 1 is a constant bias channel.
        for p in range(cfg.max_len):
            model.pos.weight[p, 1] = 1.0

        # Attention layer 0: zero (we don't need attention for this program).

        # FFN layer 0 — compute 1[input >= T] = ReLU(input-(T-1)) - ReLU(input-T)
        # ff_in: Linear(d_model=4, 2*d_ffn=4). Output: [gate_0, gate_1, val_0, val_1].
        #   gate_0 = input - (T-1): read dim 0 coef 1, dim 1 coef -(T-1)
        #   gate_1 = input - T:     read dim 0 coef 1, dim 1 coef -T
        #   val_0  = +1:            read dim 1 coef +1
        #   val_1  = -1:            read dim 1 coef -1
        model.ff_in[0].weight[0, 0] = 1.0
        model.ff_in[0].weight[0, 1] = -(T - 1)
        model.ff_in[0].weight[1, 0] = 1.0
        model.ff_in[0].weight[1, 1] = -T
        model.ff_in[0].weight[2, 1] = 1.0
        model.ff_in[0].weight[3, 1] = -1.0

        # ff_out: Linear(d_ffn=2, d_model=4). Output: add to residual dim 2.
        #   dim 2 += ff_out_weight[2, 0] * ReLU(input-(T-1)) * val_0
        #          + ff_out_weight[2, 1] * ReLU(input-T) * val_1
        # With val already absorbed in F.relu(gate) * val, we need:
        #   dim 2 += 1 * ReLU(input-(T-1)) + 1 * (-ReLU(input-T))
        #         = ReLU(input-(T-1)) - ReLU(input-T)
        #         = 1[input >= T]
        model.ff_out[0].weight[2, 0] = 1.0
        model.ff_out[0].weight[2, 1] = 1.0

        # Head: logits[1] = residual[2]; logits[j] = 0 for j != 1.
        # When step = 0: all logits zero → first-tie argmax = 0.
        # When step = 1: logits[1] = 1 > 0 → argmax = 1.
        model.head.weight[1, 2] = 1.0

    return model


def run_threshold(model: Small2DTransformer, input_token: int) -> int:
    with torch.no_grad():
        x = torch.tensor([[input_token]], dtype=torch.long)
        logits = model(x)
        return int(logits[0, 0].argmax().item())


if __name__ == "__main__":
    V = 8
    T = 4
    model = build_threshold(vocab_size=V, threshold_value=T)
    print(f"[threshold] built Small2DTransformer, {model.param_count():,} params, T={T}")
    all_ok = True
    for k in range(V):
        got = run_threshold(model, k)
        expected = 1 if k >= T else 0
        status = "ok" if got == expected else "FAIL"
        all_ok = all_ok and (got == expected)
        print(f"  [{status}] input={k} → output={got} (expected {expected})")
    print(f"overall: {'PASS' if all_ok else 'FAIL'}")
