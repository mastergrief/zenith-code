# CALM — Compiled Arithmetic/Logic in Model weights

Exploration of the "transformer-as-computer" direction (Percepta-style).
We hand-compile tiny deterministic programs into transformer weights,
run them through a reference NumPy forward pass, and validate bit-exact
correctness against a Python reference.

Phase 1 target: a 1-layer transformer that adds two single-digit
integers (0-9) + 0-9 → 0-18, represented as tokens, via parabolic-key
attention + step-function ReLU arithmetic. No training, no
backpropagation, no gradients — the weights come straight from the
compiler.

## Files

- `transformer.py` — minimal NumPy forward pass (embedding → attention →
  FFN → output projection). No layer norm, no batching, no KV cache.
  ~100 LOC, designed to be readable end-to-end.
- `compiler.py` — the CALM compiler. `build_adder()` returns a dict of
  NumPy arrays (the transformer weights) + a `TransformerConfig` that
  together implement single-digit addition.
- `tests/test_adder.py` — exhaustive correctness sweep: runs all 100
  (d1, d2) pairs through the compiled transformer and asserts the
  output digits match Python's `a + b`.

## Design notes

See the compiler source. Key ideas:

1. **Parabolic-key attention** makes softmax attention behave as an
   exact memory lookup. Query `[1, -2t, t²]`, key `[k², k, 1]` gives
   score `-(k-t)²`, which peaks at `k=t` with enough sharpness that
   softmax puts ~100% of its weight on the target position.
2. **Step functions via ReLU differences**: `ReLU(x-9) - ReLU(x-10)`
   is exactly 0 for integer x < 10 and exactly 1 for integer x ≥ 10.
   Second differences (`ReLU(x-(k-1)) - 2·ReLU(x-k) + ReLU(x-(k+1))`)
   are exact indicator functions for `x == k`.
3. **Position-multiplexed output**: the FFN writes `tens` at position 3
   (the `=` token) and `ones` at position 4 (after the tens digit has
   been emitted). The output projection is position-independent; the
   multiplexing happens inside the FFN via the exact `is_pos_k`
   indicators.
4. **Parabolic output projection**: digit token `i`'s logit is
   `2i·out_digit - i²`. This is a parabola in `i` that peaks at
   `i = out_digit`, so argmax recovers the correct digit.

## Running

```bash
cd /mnt/c/Users/gabes/projects/claw-code
python3 -m calm.tests.test_adder
```
