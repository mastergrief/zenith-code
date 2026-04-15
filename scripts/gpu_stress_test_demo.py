"""Round-16 GPU stress test — scale hybrid substrate to 8GB VRAM ceiling.

Pushes the multi-HRM substrate (Round 12 setup) through 3 sizes:
  * N=5  HRM slots  (~50M params)
  * N=20 HRM slots  (~200M params)
  * N=50 HRM slots  (~1B params)

Each size includes 1 dispatched_v4 card for baseline arithmetic check.
For each: measure VRAM, forward time for 100 samples CPU + GPU, speedup.

Validates the "limitless HRMs" claim quantitatively — at 50 HRMs the
substrate hosts ~50 specialist modules plus arithmetic backends in one
model on a consumer GPU.
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
# Values match the real HRM checkpoint shape.
HRM_VOCAB = 80
HRM_D = 64
HRM_SH = HRM_D // 2
HRM_FFN = 128
HRM_LAYERS = 4


def build_synthetic_hrm(seed: int) -> Small2DTransformer:
    cfg = Small2DConfig(
        vocab_size=HRM_VOCAB, d_model=HRM_D, n_heads=HRM_SH,
        n_layers=HRM_LAYERS, d_ffn=HRM_FFN, max_len=96,
        use_hard_max=False,
    )
    torch.manual_seed(seed)
    m = Small2DTransformer(cfg)
    with torch.no_grad():
        for p in m.parameters():
            p.normal_(0, 0.02)
    return m


def build_test_substrate(n_hrm_slots: int, card) -> GroupedSmall2DTransformer:
    d_model = n_hrm_slots * HRM_D + card.config.d_model
    d_model += d_model % 2
    n_heads = d_model // 2
    d_ffn = n_hrm_slots * HRM_FFN + card.config.d_ffn
    vocab = n_hrm_slots * HRM_VOCAB + card.config.vocab_size
    n_layers = HRM_LAYERS + card.config.n_layers
    cfg = GroupedSmall2DConfig(
        vocab_size=vocab, d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=96, use_hard_max=False,
        layer_modes=tuple(["single"] * n_layers),
        layer_hard_max=tuple(
            [False] * HRM_LAYERS + [True] * card.config.n_layers
        ),
    )
    s = GroupedSmall2DTransformer(cfg)
    with torch.no_grad():
        for p in s.parameters():
            p.zero_()
    # Install N synthetic HRMs at their slots (shares layers 0..3)
    for i in range(n_hrm_slots):
        hrm = build_synthetic_hrm(seed=100 + i)
        install_compiled_card(s, hrm, CardSlot(
            ch_off=i * HRM_D, sh_off=i * HRM_SH,
            ffn_off=i * HRM_FFN, tok_off=i * HRM_VOCAB,
            layer_off=0,
        ))
    # Install card at the tail
    install_compiled_card(s, card, CardSlot(
        ch_off=n_hrm_slots * HRM_D,
        sh_off=n_hrm_slots * HRM_SH,
        ffn_off=n_hrm_slots * HRM_FFN,
        tok_off=n_hrm_slots * HRM_VOCAB,
        layer_off=HRM_LAYERS,
    ))
    return s


def run_card_exhaustive(model, n_hrm_slots: int, device: str,
                        warmup: bool = True) -> tuple[int, int, float]:
    card_tok_off = n_hrm_slots * HRM_VOCAB
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
        x = torch.tensor([[card_tok_off, card_tok_off, card_tok_off]],
                         dtype=torch.long, device=device)
        with torch.no_grad():
            model(x)
        torch.cuda.synchronize()
    correct = 0
    total = 0
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for inputs, expected in test_suite:
        shifted = [(a + card_tok_off, b + card_tok_off,
                    op + OPCODE_SHIFT + card_tok_off)
                   for (a, b, op) in inputs]
        x = torch.tensor(shifted, dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(x)
        card_logits = logits[:, 2, card_tok_off:card_tok_off + CARD_VOCAB]
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


def main() -> None:
    if not torch.cuda.is_available():
        print("[stress] CUDA not available")
        return
    print(f"[stress] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[stress] free VRAM: "
          f"{torch.cuda.mem_get_info(0)[0] / 1e9:.1f} GB\n")

    card = build_dispatched_v4()

    sizes = [5, 20, 50]
    print(f"{'N_HRMs':>8} {'params':>13} {'CPU':>10} {'GPU':>10} "
          f"{'speedup':>8} {'VRAM':>8} {'ok':>5}")
    print("-" * 72)

    for n in sizes:
        try:
            # Build on CPU
            substrate = build_test_substrate(n, card)
            params = substrate.param_count()

            # CPU run
            ok_cpu, tot_cpu, cpu_t = run_card_exhaustive(
                substrate, n, "cpu", warmup=False,
            )

            # Move to GPU
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()
            substrate = substrate.to("cuda")
            mem_after = torch.cuda.memory_allocated()
            vram_gb = (mem_after - mem_before) / 1e9

            ok_gpu, tot_gpu, gpu_t = run_card_exhaustive(
                substrate, n, "cuda", warmup=True,
            )
            speedup = cpu_t / gpu_t if gpu_t > 0 else float("inf")
            status = "PASS" if (ok_cpu == tot_cpu and ok_gpu == tot_gpu) else "FAIL"
            print(f"{n:>8} {params:>13,} {cpu_t:>9.3f}s {gpu_t:>9.3f}s "
                  f"{speedup:>7.1f}× {vram_gb:>6.2f}GB {status:>5}")

            # Release GPU memory for next iter
            del substrate
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"{n:>8}  (OOM)")
            torch.cuda.empty_cache()
            break
        except Exception as e:
            print(f"{n:>8}  ERROR: {e}")
            import traceback
            traceback.print_exc()
            break


if __name__ == "__main__":
    main()
