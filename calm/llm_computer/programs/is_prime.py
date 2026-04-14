"""is_prime — compile-time primality check via gate-graph IR.

Program: pos 0 = n ∈ [MIN_N, MAX_N]. Output at pos 0: argmax selects
token 1 if n is prime, else token 0.

Construction: single-input step-diff decode into a binary head.
  - TokenEmbed ch 0 = token scalar
  - PosEmbed ch 1 = bias 1
  - Layer 0 FFN: step ReGLUs decode n into indicators in channels
    [2, 2 + MAX_N - MIN_N + 1]; step_k = 1[n >= k] for k in
    [MIN_N, MAX_N + 1]
  - LinearHead: for each n, logits[prime(n) ? 1 : 0] += step_n - step_{n+1}

The +1/-1 entries at adjacent channels accumulate in the head via `+=`
(see compile.py), so consecutive primes or consecutive composites sum
cleanly into the same logit slot.
"""

from __future__ import annotations

from calm.llm_computer.compile import compile_program
from calm.llm_computer.gate_graph import (
    GateGraph, LinearHead, PosEmbed, ReGLU, TokenEmbed,
)
from calm.llm_computer.model import Small2DTransformer
from calm.llm_computer.schedule import auto_schedule


MIN_N = 2
MAX_N = 100
VOCAB = MAX_N + 1  # 101 — covers inputs; outputs are token 0 or 1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0:
        return False
    for d in range(3, int(n ** 0.5) + 1, 2):
        if n % d == 0:
            return False
    return True


def build_is_prime(max_len: int = 2) -> Small2DTransformer:
    V = VOCAB
    graph = GateGraph(vocab_size=V)

    graph.add(TokenEmbed(
        name="own_scalar",
        entries=[(k, 0, float(k)) for k in range(V)],
    ))
    graph.add(PosEmbed(
        name="bias",
        entries=[(p, 1, 1.0) for p in range(max_len)],
    ))
    # step_k = 1[n >= k] for k in [MIN_N, MAX_N + 1]. The +1 extra is for
    # the step-diff cancellation at n=MAX_N.
    for k in range(MIN_N, MAX_N + 2):
        ch = 2 + (k - MIN_N)
        graph.add(ReGLU(
            name=f"step_{k}_hi",
            gate=[(0, 1.0), (1, -(k - 1))],
            val=[(1, 1.0)],
            output_channel=ch,
            output_coef=1.0,
        ))
        graph.add(ReGLU(
            name=f"step_{k}_lo",
            gate=[(0, 1.0), (1, -k)],
            val=[(1, 1.0)],
            output_channel=ch,
            output_coef=-1.0,
        ))
    head_entries = []
    for n in range(MIN_N, MAX_N + 1):
        ch = 2 + (n - MIN_N)
        target = 1 if _is_prime(n) else 0
        head_entries.append((target, ch, 1.0))
        head_entries.append((target, ch + 1, -1.0))
    graph.add(LinearHead(name="is_prime_head", entries=head_entries))

    n_layers = auto_schedule(graph)

    num_steps = MAX_N - MIN_N + 2   # 100
    d_model = 2 + num_steps
    if d_model % 2 != 0:
        d_model += 1
    n_heads = d_model // 2
    d_ffn = 2 * num_steps
    return compile_program(
        graph,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ffn=d_ffn,
        max_len=max_len,
        vocab_size=V,
    )


if __name__ == "__main__":
    import time
    import torch

    t0 = time.time()
    model = build_is_prime()
    t_build = time.time() - t0
    print(f"[is_prime] built in {t_build:.1f}s, {model.param_count():,} params")

    inputs = torch.tensor([[n] for n in range(MIN_N, MAX_N + 1)], dtype=torch.long)
    t0 = time.time()
    with torch.no_grad():
        logits = model(inputs)
        preds = logits[:, 0, :].argmax(dim=-1).tolist()
    t_run = time.time() - t0
    expected = [1 if _is_prime(n) else 0 for n in range(MIN_N, MAX_N + 1)]
    correct = sum(1 for p, e in zip(preds, expected) if p == e)
    print(f"[is_prime] ran in {t_run:.3f}s")
    print(f"[is_prime] {correct}/{len(expected)} correct")
    if correct != len(expected):
        for n, p, e in zip(range(MIN_N, MAX_N + 1), preds, expected):
            if p != e:
                print(f"  [FAIL] is_prime({n}) = {p} (expected {e})")
    else:
        primes = [n for n, e in zip(range(MIN_N, MAX_N + 1), expected) if e]
        print(f"  primes in [{MIN_N}, {MAX_N}]: {primes}")
