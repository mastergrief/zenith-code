"""Round-13 GPU validation — run capstone demo on CUDA.

Hypothesis: pushing `HybridGroupedSmall2DTransformer` or
`GroupedSmall2DTransformer` through `.to("cuda")` preserves correctness
(bit-identical up to FP32 cross-device nondeterminism), and accelerates
the exhaustive 791-case dispatched_v4 pass.

Tests:
  (1) Hybrid substrate FP32 card install — CUDA == CPU (no quant).
  (2) Round 9 HRM + dispatched_v2 — HRM bit-identical on CUDA.
  (3) Perf: wall-clock for 791 exhaustive on CPU vs CUDA.

Memory budget: RTX 4070 8 GB. The Round 9 substrate is 25M params × 4
bytes = 100 MB — fits trivially.
"""

from __future__ import annotations

import itertools
import math
import time

import torch

from calm.llm_computer.card_installer import CardSlot, install_compiled_card
from calm.llm_computer.grouped_small2d import (
    GroupedSmall2DConfig, GroupedSmall2DTransformer,
)
from calm.llm_computer.model import Small2DConfig, Small2DTransformer
from calm.llm_computer.programs.dispatched_v4 import (
    FACT_MAX_N, GCD_BASE, MUL_MAX_OPERAND, OPCODE_SHIFT, PRIME_MAX_N,
    PRIME_MIN_N, VOCAB as CARD_VOCAB, build_dispatched_v4, decode_output,
)
from calm.llm_computer.programs.is_prime import _is_prime
from pathlib import Path


HRM_CKPT = Path(
    "/mnt/c/Users/gabes/projects/claw-code/calm/hrm/checkpoints/"
    "substrate_hrm_nl_best.pt"
)


def load_hrm():
    ckpt = torch.load(HRM_CKPT, weights_only=False, map_location="cpu")
    cfg = Small2DConfig(
        vocab_size=ckpt["config"]["vocab_size"],
        d_model=ckpt["config"]["d_model"],
        n_heads=ckpt["config"]["n_heads"],
        n_layers=ckpt["config"]["n_layers"],
        d_ffn=ckpt["config"]["d_ffn"],
        max_len=ckpt["config"]["max_len"],
        use_hard_max=False,
    )
    m = Small2DTransformer(cfg)
    m.load_state_dict(ckpt["model_state_dict"])
    m.eval()
    return m


def build_substrate(hrm, card):
    h = hrm.config
    c = card.config
    d_model = h.d_model + c.d_model
    d_model += d_model % 2
    cfg = GroupedSmall2DConfig(
        vocab_size=h.vocab_size + c.vocab_size,
        d_model=d_model,
        n_heads=d_model // 2,
        n_layers=h.n_layers + c.n_layers,
        d_ffn=h.d_ffn + c.d_ffn,
        max_len=max(h.max_len, c.max_len),
        use_hard_max=False,
        layer_modes=tuple(["single"] * (h.n_layers + c.n_layers)),
        layer_hard_max=tuple([False] * h.n_layers + [True] * c.n_layers),
    )
    s = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in s.parameters():
            p.zero_()
    install_compiled_card(s, hrm, CardSlot(0, 0, 0, 0, 0))
    install_compiled_card(s, card, CardSlot(
        ch_off=h.d_model, sh_off=h.d_model // 2,
        ffn_off=h.d_ffn, tok_off=h.vocab_size, layer_off=h.n_layers,
    ))
    return s


def run_dispatched_exhaustive(model, hrm_vocab: int, device: str,
                              warmup: bool = True) -> tuple[int, float]:
    """Run 791-case exhaustive dispatched test on `device`. Return
    (correct_count, wall_seconds)."""
    pairs = list(itertools.product(range(GCD_BASE), repeat=2))
    mul_pairs = list(itertools.product(range(MUL_MAX_OPERAND + 1), repeat=2))
    test_suite = [
        ([(a, b, 0) for a, b in pairs],
         [math.gcd(a, b) for a, b in pairs]),
        ([(n, 0, 1) for n in range(FACT_MAX_N + 1)],
         [math.factorial(n) for n in range(FACT_MAX_N + 1)]),
        ([(n, 0, 2) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)],
         [_is_prime(n) for n in range(PRIME_MIN_N, PRIME_MAX_N + 1)]),
        ([(a, b, 3) for a, b in pairs],
         [a + b for a, b in pairs]),
        ([(a, b, 4) for a, b in mul_pairs],
         [a * b for a, b in mul_pairs]),
    ]
    if warmup and device == "cuda":
        # Single warmup forward to eliminate CUDA kernel-compile overhead.
        x = torch.tensor([[0, 0, 0]], dtype=torch.long, device=device)
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
    correct = 0
    total = 0
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for inputs, expected in test_suite:
        shifted = [(a + hrm_vocab, b + hrm_vocab,
                    op + OPCODE_SHIFT + hrm_vocab)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        card_logits = logits[:, 2, hrm_vocab:hrm_vocab + CARD_VOCAB]
        preds = card_logits.argmax(dim=-1).tolist()
        correct += sum(
            1 for p, (args, exp) in zip(preds, zip(inputs, expected))
            if decode_output(args[2], p) == exp
        )
        total += len(inputs)
    if device == "cuda":
        torch.cuda.synchronize()
    t = time.time() - t0
    return correct, total, t


def main():
    print("[gpu] CUDA device:", torch.cuda.get_device_name(0))

    hrm = load_hrm()
    card = build_dispatched_v4()
    sub_cpu = build_substrate(hrm, card)
    sub_cpu.eval()
    print(f"[gpu] substrate params: {sub_cpu.param_count():,}")

    # ---- Perf / correctness on CPU ----
    print("\n[gpu] CPU run (baseline):")
    c_cpu, t_all, cpu_time = run_dispatched_exhaustive(
        sub_cpu, hrm.config.vocab_size, "cpu", warmup=False,
    )
    print(f"  dispatched_v4: {c_cpu}/{t_all} in {cpu_time:.3f}s")

    # ---- Move to CUDA ----
    print("\n[gpu] moving substrate to CUDA...")
    sub_gpu = sub_cpu.to("cuda")
    print(f"  VRAM used: {torch.cuda.memory_allocated() / 1e6:.1f} MB")

    # ---- Correctness on CUDA ----
    print("\n[gpu] CUDA run:")
    c_gpu, t_all2, gpu_time = run_dispatched_exhaustive(
        sub_gpu, hrm.config.vocab_size, "cuda", warmup=True,
    )
    print(f"  dispatched_v4: {c_gpu}/{t_all2} in {gpu_time:.3f}s")

    ok_correctness = (c_cpu == t_all) and (c_gpu == t_all2)
    print(f"\n[gpu] correctness CPU=GPU: {'PASS' if ok_correctness else 'FAIL'}")

    # ---- HRM bit-identical on CUDA ----
    print("\n[gpu] HRM bit-identical check:")
    torch.manual_seed(7)
    x = torch.randint(0, hrm.config.vocab_size, (4, 16))
    x_cuda = x.to("cuda")
    hrm_cuda = hrm.to("cuda")
    with torch.no_grad():
        std_logits = hrm_cuda(x_cuda)
        sub_logits = sub_gpu(x_cuda)[:, :, :hrm.config.vocab_size]
    diff_gpu = (std_logits - sub_logits).abs().max().item()
    ok_hrm = diff_gpu < 1e-4
    print(f"  max |hrm_cuda - substrate_cuda[hrm_range]| = {diff_gpu:.2e} — "
          f"{'PASS' if ok_hrm else 'FAIL'}")

    # ---- CPU vs CUDA logits agree ----
    # sub_cpu.to("cuda") moves in-place, so sub_cpu is now on cuda.
    # Rebuild a fresh CPU substrate from the same weights for comparison.
    print("\n[gpu] CPU vs CUDA logit agreement:")
    sub_cpu_fresh = build_substrate(hrm.to("cpu"), card)  # rebuild on cpu
    sub_cpu_fresh.eval()
    with torch.no_grad():
        l_cpu = sub_cpu_fresh(x)
        l_gpu = sub_gpu(x_cuda).to("cpu")
    diff_cpu_gpu = (l_cpu - l_gpu).abs().max().item()
    # FP32 matmul on CUDA uses different reduction order than CPU; expect
    # small numerical drift, not exact match. 1e-3 relative is typical.
    ok_agree = diff_cpu_gpu < 1e-2
    print(f"  max |l_cpu - l_gpu| = {diff_cpu_gpu:.2e} — "
          f"{'PASS' if ok_agree else 'FAIL'}")

    # ---- Speedup ----
    speedup = cpu_time / gpu_time if gpu_time > 0 else float("inf")
    print(f"\n[gpu] perf: CPU {cpu_time:.3f}s | GPU {gpu_time:.3f}s | "
          f"speedup {speedup:.1f}x")

    all_ok = ok_correctness and ok_hrm and ok_agree
    print(f"\n[gpu] OVERALL: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    main()
