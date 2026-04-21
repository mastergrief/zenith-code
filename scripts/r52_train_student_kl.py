"""R52.1 — train R51Student on Gemma L24 via KL-divergence on final logits.

Hypothesis: MSE on residuals (R51.3) reaches 92.6% variance-explained
but preserves token-space behavior poorly (R51.5: 0.19/0.34 mean-prefix
match on dual gate). MSE averages error across all 2560 channels,
washing out sharp digit-selector / content-reader directions. Training
the same 1.25M-param student with forward KL on Gemma's final next-token
logits targets token-space preservation directly, letting the student
fail to reconstruct arbitrary channels as long as whatever drives the
head's argmax is preserved.

Daemon-compatible. Assumes `m` (GemmaSubstrate) and `tok` are pre-bound.
Does NOT load Gemma. Does NOT modify `calm/llm_computer/r51/install.py`.

Install path: monkey-patches `m._forward_layer` so that when called
with `layer_idx == target_layer` (default 24), we return
`h_before + student(h_before)` and skip Gemma's native L24 compute.
GRAD-ENABLED variant of the install — autograd flows back from the
final logits through the head, layers 25..41, the student's forward,
and into the student's parameters.

Triton kernels are disabled for training (autograd through Triton is
uncertain); the PyTorch fast path (`_tq4_linear_kernel`) uses standard
torch ops and is autograd-compatible. Re-enabled in the finally block.

Loss: F.kl_div(log_softmax(student_logits), softmax(teacher_logits),
                reduction="batchmean") with log_target=False. This is
forward KL from student's perspective — KL(teacher || student),
the standard distillation loss; positive, decreases to near-zero at
convergence.

CLI:
    --captures PATH      Default /tmp/r52_teacher_logits.pt
    --out PATH           Default calm/llm_computer/r51/checkpoints/r52_student_kl.pt
    --steps N            Default 20 (smoke); full run N >= 1000
    --batch-size K       Prompts per grad step (accumulated). Default 4
    --eval-every N       Val eval cadence. Default 10
    --lr                 Default 1e-3
    --warmup-steps       Default 200
    --grad-clip          Default 1.0
    --val-frac           Default 0.1
    --target-layer       Default 24
    --seed               Default 42
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from calm.llm_computer.r51 import R51Student, R51StudentConfig


DEFAULT_CAPTURES = "/tmp/r52_teacher_logits.pt"
DEFAULT_OUT = "calm/llm_computer/r51/checkpoints/r52_student_kl.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--captures", type=str, default=DEFAULT_CAPTURES)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ffn", type=int, default=512)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--steps", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=10)
    p.add_argument("--target-layer", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_device)
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    return p.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stratified_split(
    labels: list, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Per-domain deterministic shuffle; last val_frac per domain → val."""
    rng = random.Random(seed)
    by_domain: dict = {}
    for p_idx, dom in enumerate(labels):
        by_domain.setdefault(dom, []).append(p_idx)
    train_ids: list[int] = []
    val_ids: list[int] = []
    for dom in sorted(by_domain.keys()):
        ids = list(by_domain[dom])
        rng.shuffle(ids)
        n_val = max(1, int(round(len(ids) * val_frac)))
        val_ids.extend(ids[-n_val:])
        train_ids.extend(ids[:-n_val])
    rng.shuffle(train_ids)
    rng.shuffle(val_ids)
    return train_ids, val_ids


def install_student_with_grad(m, student, target_layer: int = 24):
    """Grad-enabled monkey-patch variant of install_r51_student.

    Differs from calm/llm_computer/r51/install.py::install_r51_student by
    NOT wrapping the student call in torch.no_grad(). The returned callable
    restores the original _forward_layer when called.
    """
    original = m._forward_layer
    device = next(student.parameters()).device
    dtype = next(student.parameters()).dtype

    def patched(h, layer, layer_idx, kv_cache=None, start_pos=0):
        if layer_idx != target_layer:
            return original(h, layer, layer_idx, kv_cache=kv_cache,
                            start_pos=start_pos)
        h_before = h
        x = h_before.to(device=device, dtype=dtype)
        delta = student(x)
        return h_before + delta.to(dtype=h_before.dtype)

    m._forward_layer = patched

    def restore():
        m._forward_layer = original

    return restore


def lr_at_step(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


def forward_one_prompt_grad(m, token_ids_cuda: torch.Tensor) -> torch.Tensor:
    """Run Gemma forward with a fresh KV cache. Returns last-position
    logits [vocab]. Grad flows via the monkey-patched L24 → student."""
    from calm.llm_computer.gemma_substrate import KVCache

    cfg = m.config
    cache = KVCache(cfg.n_layers, device="cuda")
    logits = m.forward(
        token_ids_cuda, device="cuda", kv_cache=cache, start_pos=0
    )
    return logits[0, 0]


def compute_kl_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor
) -> torch.Tensor:
    """Forward KL: KL(teacher || student). F.kl_div expects input = log_q,
    target = p, reduction='batchmean' divides by outer (vocab-sized)
    batch-dim = 1 here, so effectively sum over vocab. Returns scalar."""
    log_p_student = F.log_softmax(student_logits, dim=-1)
    p_teacher = F.softmax(teacher_logits, dim=-1)
    return F.kl_div(log_p_student, p_teacher, reduction="sum")


def eval_val(
    m,
    student,
    val_ids: list[int],
    token_ids_list: list,
    teacher_logits_all: torch.Tensor,
    labels: list,
    domain_names: list,
) -> tuple[float, dict]:
    student.eval()
    per_dom_sum: dict = {n: 0.0 for n in domain_names}
    per_dom_count: dict = {n: 0 for n in domain_names}
    total_sum = 0.0
    total_count = 0
    with torch.no_grad():
        for p_idx in val_ids:
            ids = token_ids_list[p_idx].to("cuda", dtype=torch.long).unsqueeze(0)
            with torch.amp.autocast(device_type="cuda",
                                    dtype=torch.bfloat16):
                s_logits = forward_one_prompt_grad(m, ids)
            t_logits = teacher_logits_all[p_idx].to(
                "cuda", dtype=torch.float32
            )
            loss = compute_kl_loss(s_logits.float(), t_logits)
            v = float(loss.item())
            dom = labels[p_idx]
            per_dom_sum[dom] += v
            per_dom_count[dom] += 1
            total_sum += v
            total_count += 1
    student.train()
    per_dom = {
        n: (per_dom_sum[n] / per_dom_count[n])
        if per_dom_count[n] > 0
        else float("nan")
        for n in domain_names
    }
    total = total_sum / total_count if total_count > 0 else float("nan")
    return total, per_dom


def main() -> None:
    assert "m" in globals(), "daemon contract: `m` must be pre-bound"
    args = parse_args()
    set_seed(args.seed)

    # Triton tq4 autograd wrapper with MATCHED Triton backward kernel.
    # Forward streams tq4 bytes (no materialized W); backward uses
    # tq4_backward_triton kernel — same tq4 access pattern + fp32
    # reduction, so forward/backward are self-consistent despite
    # diverging from PyTorch fast path by ~6e-5 per linear (different
    # reduction order). Training on Triton-Gemma stays consistent.
    from calm.llm_computer.gemma_substrate import enable_triton_tq4
    from calm.llm_computer.tq4_autograd import (
        install_tq4_autograd, restore_tq4_autograd,
    )
    enable_triton_tq4(True)
    install_tq4_autograd()
    print("[r52.1] Triton tq4 ENABLED + autograd.Function (w/ Triton backward)",
          flush=True)

    # torch.compile stays OFF — "Not enough SMs for max_autotune_gemm"
    # on 4070M falls back slower than uncompiled. Explicit reset in
    # case a prior script enabled it.
    import calm.llm_computer.gemma_substrate as _gs
    _gs._compiled_tq4_linear = None
    print("[r52.1] torch.compile for tq4 DISABLED + module state reset",
          flush=True)

    # Speed fix 2 — pre-dequant the Q6_K output head to FP16 once. The head
    # is 262144 x 2560 Q6_K = 1.34 GB in FP16, static during training. The
    # original GpuQ6KEmbedding.output_logits prefers q6k_matvec_triton (no
    # grad), and the naive PyTorch fallback re-dequants all 262144 rows
    # every forward — ~80% of per-step cost at the 91 s/step measured.
    # Pre-dequant once, then `h @ HEAD.T` is a single matmul with autograd.
    q6k_embd = m.token_embd
    original_output_logits = q6k_embd.output_logits
    print("[r52.1] pre-dequanting Q6_K head to FP16 (~1.34 GB)...", flush=True)
    _t0 = time.time()
    bpr = q6k_embd.blocks_per_row
    # Chunked pre-dequant — full-vocab dequant (262144 * bpr block indices
    # + intermediate workspace) OOMs at 8 GB with Gemma resident. Chunk
    # over vocab rows, dequant each to FP16, stitch at the end.
    chunk_rows = 16384
    HEAD_FP16 = torch.empty(
        q6k_embd.vocab_size, q6k_embd.d_model,
        device="cuda", dtype=torch.float16,
    )
    for start in range(0, q6k_embd.vocab_size, chunk_rows):
        end = min(start + chunk_rows, q6k_embd.vocab_size)
        n_rows = end - start
        row_starts = torch.arange(start, end, device="cuda") * bpr
        offsets = torch.arange(bpr, device="cuda")
        block_idx = (
            row_starts.unsqueeze(1) + offsets.unsqueeze(0)
        ).flatten()
        chunk = q6k_embd._dequant_blocks(block_idx).reshape(
            n_rows, q6k_embd.d_model
        )
        HEAD_FP16[start:end] = chunk.to(torch.float16)
        del chunk, block_idx, row_starts, offsets
    torch.cuda.empty_cache()
    print(
        f"[r52.1] head FP16: {tuple(HEAD_FP16.shape)} "
        f"{HEAD_FP16.element_size() * HEAD_FP16.numel() / 1e9:.2f} GB "
        f"in {time.time() - _t0:.1f}s",
        flush=True,
    )

    def _output_logits_static(h):
        # h: (B, 1, d_model) fp32. HEAD_FP16: (V, d_model).
        return (h.to(torch.float16) @ HEAD_FP16.T).to(torch.float32)

    q6k_embd.output_logits = _output_logits_static
    print("[r52.1] patched token_embd.output_logits to static FP16 head",
          flush=True)

    # Inspect m.parameters() — Gemma's linears are MmapTq4Linear, NOT
    # nn.Module subclasses. If m.parameters() is empty, Gemma weights
    # can't track grads by design, so nothing to freeze.
    m_params = list(m.parameters()) if hasattr(m, "parameters") else []
    print(
        f"[r52.1] m.parameters() count: {len(m_params)}  "
        f"(0 = no autograd-tracked weights on Gemma, as expected)",
        flush=True,
    )
    if m_params:
        for p in m_params:
            p.requires_grad_(False)
        print(f"[r52.1] froze {len(m_params)} Gemma params", flush=True)

    caps_path = Path(args.captures)
    if not caps_path.exists():
        raise FileNotFoundError(f"captures not found: {caps_path}")
    print(f"[r52.1] loading captures: {caps_path}", flush=True)
    caps = torch.load(str(caps_path), weights_only=False, map_location="cpu")

    teacher_logits_all: torch.Tensor = caps["teacher_logits"]  # fp16 cpu
    token_ids_list: list = caps["token_ids_list"]
    labels: list = caps["labels"]
    domain_names: list = list(caps["DOMAIN_NAMES"])
    n_prompts = teacher_logits_all.shape[0]
    vocab = teacher_logits_all.shape[1]

    print(
        f"[r52.1] teacher_logits={tuple(teacher_logits_all.shape)} "
        f"n_prompts={n_prompts} n_domains={len(domain_names)} vocab={vocab}",
        flush=True,
    )

    # Stratified split.
    train_ids, val_ids = stratified_split(labels, args.val_frac, args.seed)
    # Guard: with tiny smoke corpora (6 prompts) val_frac=0.1 rounds up
    # to 1 val per domain, leaving 0 train prompts in that domain.
    # For smoke tests that's fine — we still have non-empty train_ids
    # overall.
    if len(train_ids) == 0:
        # Smoke fallback: train on all, val on all (no real stratification).
        train_ids = list(range(n_prompts))
        val_ids = list(range(n_prompts))
        print(
            "[r52.1] WARNING: empty train split (tiny corpus), "
            "using all prompts for both train + val",
            flush=True,
        )

    train_dom = {n: 0 for n in domain_names}
    val_dom = {n: 0 for n in domain_names}
    for p in train_ids:
        train_dom[labels[p]] = train_dom.get(labels[p], 0) + 1
    for p in val_ids:
        val_dom[labels[p]] = val_dom.get(labels[p], 0) + 1
    print(f"[r52.1] split: train={len(train_ids)} val={len(val_ids)}",
          flush=True)
    print(f"[r52.1] train per-domain: {train_dom}", flush=True)
    print(f"[r52.1] val per-domain:   {val_dom}", flush=True)

    # Build student.
    cfg = R51StudentConfig(
        d_io=2560,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_len=args.max_len,
        dropout=0.0,
    )
    device = args.device
    student = R51Student(cfg).to(device)
    student.train()
    n_params = student.param_count()
    print(
        f"[r52.1] student: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"d_ffn={cfg.d_ffn} max_len={cfg.max_len} "
        f"params={n_params:,} ({n_params/1e6:.2f}M) device={device}",
        flush=True,
    )

    opt = torch.optim.Adam(
        student.parameters(), lr=args.lr, betas=(0.9, 0.999)
    )

    # Install student on L24 (grad-enabled) — install ONCE, detach in finally.
    restore_layer = install_student_with_grad(m, student, args.target_layer)
    print(
        f"[r52.1] installed student on L{args.target_layer} (grad-enabled)",
        flush=True,
    )

    # Filter out prompts whose sequence length > student.max_len.
    ok_ids = [
        p for p in range(n_prompts)
        if token_ids_list[p].shape[0] <= cfg.max_len
    ]
    n_filtered = n_prompts - len(ok_ids)
    if n_filtered > 0:
        ok_set = set(ok_ids)
        train_ids = [p for p in train_ids if p in ok_set]
        val_ids = [p for p in val_ids if p in ok_set]
        print(
            f"[r52.1] filtered {n_filtered} prompts longer than "
            f"max_len={cfg.max_len}; train={len(train_ids)} val={len(val_ids)}",
            flush=True,
        )

    train_history: list = []
    best_val_total = float("inf")
    best_step = 0
    best_per_domain: dict = {}
    initial_loss = None
    final_loss = None

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + 1)
    train_ema = float("nan")
    step_times: list = []
    t0 = time.time()

    try:
        for step in range(1, args.steps + 1):
            t_step = time.time()

            if len(train_ids) < args.batch_size:
                batch = [rng.choice(train_ids) for _ in range(args.batch_size)]
            else:
                batch = rng.sample(train_ids, args.batch_size)

            lr_now = lr_at_step(step, args.warmup_steps, args.lr)
            for g in opt.param_groups:
                g["lr"] = lr_now

            opt.zero_grad(set_to_none=True)
            batch_loss_sum = 0.0
            for p_idx in batch:
                ids = token_ids_list[p_idx].to(
                    "cuda", dtype=torch.long
                ).unsqueeze(0)
                # BF16 autocast — tensor cores + regularization noise
                # help convergence. Triton autograd uses custom_fwd
                # cast_inputs=fp32 so Triton runs fp32 internally while
                # downstream ops auto-cast to bf16.
                with torch.amp.autocast(device_type="cuda",
                                        dtype=torch.bfloat16):
                    s_logits = forward_one_prompt_grad(m, ids)
                t_logits = teacher_logits_all[p_idx].to(
                    "cuda", dtype=torch.float32
                )
                loss = compute_kl_loss(s_logits.float(), t_logits)
                (loss / args.batch_size).backward()
                batch_loss_sum += float(loss.item())

            torch.nn.utils.clip_grad_norm_(
                student.parameters(), args.grad_clip
            )
            opt.step()

            loss_val = batch_loss_sum / args.batch_size
            if step == 1:
                initial_loss = loss_val
                train_ema = loss_val
            elif step <= 10:
                train_ema = 0.7 * train_ema + 0.3 * loss_val
            else:
                train_ema = 0.98 * train_ema + 0.02 * loss_val
            final_loss = loss_val

            step_times.append(time.time() - t_step)
            if len(step_times) > 50:
                step_times.pop(0)

            # Periodic cache clear — fragmentation insurance on tight VRAM.
            # HEAD_FP16 + Gemma + activations sit at ~7 GB resident on 8 GB,
            # so freeing transient allocations helps keep a steady footprint
            # across 2000+ step runs. ~1-2ms per call, negligible.
            if step % 50 == 0:
                torch.cuda.empty_cache()

            if step % 5 == 0 or step == 1:
                ms = 1000.0 * sum(step_times) / max(1, len(step_times))
                print(
                    f"step {step} loss={loss_val:.4f} "
                    f"ema={train_ema:.4f} lr={lr_now:.5f} "
                    f"ms/step={ms:.0f}",
                    flush=True,
                )

            do_eval = (step % args.eval_every == 0) or (step == args.steps)
            if do_eval and len(val_ids) > 0:
                val_total, per_dom = eval_val(
                    m, student, val_ids, token_ids_list,
                    teacher_logits_all, labels, domain_names,
                )
                per_dom_str = " ".join(
                    f"val_{n}={per_dom[n]:.4f}" for n in domain_names
                )
                print(
                    f"[eval step {step}] train_ema={train_ema:.4f} "
                    f"val_total={val_total:.4f} {per_dom_str}",
                    flush=True,
                )
                train_history.append({
                    "step": step,
                    "train_ema": train_ema,
                    "val_total": val_total,
                    "per_domain": dict(per_dom),
                })
                if val_total < best_val_total:
                    best_val_total = val_total
                    best_step = step
                    best_per_domain = dict(per_dom)
                    ckpt = {
                        "state_dict": student.state_dict(),
                        "config": asdict(cfg),
                        "train_history": train_history,
                        "best_val_total": best_val_total,
                        "best_step": best_step,
                        "per_domain_val_final": best_per_domain,
                        "seed": args.seed,
                        "captures_path": str(caps_path),
                        "loss_type": "kl_forward",
                    }
                    tmp_path = str(out_path) + ".tmp"
                    torch.save(ckpt, tmp_path)
                    os.replace(tmp_path, out_path)
    finally:
        restore_layer()
        restore_tq4_autograd()
        q6k_embd.output_logits = original_output_logits
        try:
            del HEAD_FP16
        except NameError:
            pass
        torch.cuda.empty_cache()
        print(
            "[r52.1] detached student; Triton tq4 autograd restored; "
            "output_logits restored; HEAD_FP16 freed",
            flush=True,
        )

    total_time = time.time() - t0
    ckpt_bytes = out_path.stat().st_size if out_path.exists() else 0
    ckpt_mb = ckpt_bytes / (1024 * 1024)

    print("", flush=True)
    print("=" * 72, flush=True)
    print(
        f"[r52.1] training complete in {total_time:.1f}s "
        f"({args.steps} steps, batch={args.batch_size})",
        flush=True,
    )
    print(f"  initial loss: {initial_loss:.4f}", flush=True)
    print(f"  final loss:   {final_loss:.4f}", flush=True)
    if initial_loss is not None and final_loss is not None:
        ratio = final_loss / max(1e-9, initial_loss)
        print(f"  ratio final/initial: {ratio:.3f}", flush=True)
    print(f"  best_val_total = {best_val_total:.4f} at step {best_step}",
          flush=True)
    if best_per_domain:
        print("  best per-domain val KL:", flush=True)
        for name in domain_names:
            v = best_per_domain.get(name, float("nan"))
            print(f"    {name:<10} {v:.4f}", flush=True)
    print(f"  checkpoint: {out_path} ({ckpt_mb:.2f} MB)", flush=True)
    print("=" * 72, flush=True)

    # Smoke self-test assertion: loss must decrease.
    if initial_loss is not None and final_loss is not None:
        if final_loss < 0.9 * initial_loss:
            print(
                f"[r52.1] SMOKE PASS: final < 0.9 * initial "
                f"({final_loss:.4f} < {0.9 * initial_loss:.4f})",
                flush=True,
            )
        else:
            print(
                f"[r52.1] SMOKE FAIL: final >= 0.9 * initial "
                f"({final_loss:.4f} vs {0.9 * initial_loss:.4f}) — loss not "
                f"decreasing enough",
                flush=True,
            )


if __name__ == "__main__":
    main()
else:
    main()
