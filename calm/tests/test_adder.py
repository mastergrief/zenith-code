"""
Exhaustive correctness sweep for the Phase 1 single-digit adder.

Compiles the adder twice (once reading TENS, once reading ONES -- see
the long comment in calm/compiler.py:_build_ffn for why the v1
construction needs two compiles), then runs all 100 (d1, d2) pairs
and asserts the argmax of the output logits matches Python's `d1+d2`
split into decimal digits.

This is the "proof the substrate works" test: if the transformer
forward pass on hand-compiled weights produces exact arithmetic on
all 100 cases, we've validated that parabolic-key attention +
step-function ReLU arithmetic give bit-exact compute at small scale.
"""

from __future__ import annotations

import numpy as np

from calm.compiler import build_adder, TOK_PLUS, TOK_EQ, S_TENS, S_ONES
from calm.transformer import forward


def _build_both():
    cfg, w_tens = build_adder(read_slot=S_TENS)
    _, w_ones = build_adder(read_slot=S_ONES)
    return cfg, w_tens, w_ones


def test_adder_exhaustive():
    cfg, w_tens, w_ones = _build_both()

    fails = []
    for d1 in range(10):
        for d2 in range(10):
            total = d1 + d2
            expect_tens, expect_ones = total // 10, total % 10

            # Tens digit: predict the token AFTER '=' by reading the
            # logits at position 3 (the '=' token's slot).
            prompt_tens = np.array([d1, TOK_PLUS, d2, TOK_EQ], dtype=np.int64)
            logits_tens = forward(prompt_tens, cfg, w_tens)
            got_tens = int(np.argmax(logits_tens[-1]))

            # Ones digit: feed the correct tens back in and read the
            # logits at position 4.
            prompt_ones = np.array(
                [d1, TOK_PLUS, d2, TOK_EQ, expect_tens], dtype=np.int64
            )
            logits_ones = forward(prompt_ones, cfg, w_ones)
            got_ones = int(np.argmax(logits_ones[-1]))

            if (got_tens, got_ones) != (expect_tens, expect_ones):
                fails.append(
                    (d1, d2, got_tens, got_ones, expect_tens, expect_ones)
                )

    assert not fails, (
        f"{len(fails)}/100 cases failed; first: "
        f"d1={fails[0][0]} d2={fails[0][1]} "
        f"got=({fails[0][2]},{fails[0][3]}) "
        f"expected=({fails[0][4]},{fails[0][5]})"
    )


if __name__ == "__main__":
    test_adder_exhaustive()
    print("OK: 100/100 pass")
