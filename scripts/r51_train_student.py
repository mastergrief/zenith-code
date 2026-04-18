"""R51.3 — train R51Student on broad L24 captures.

Supervised MSE regression from Gemma L24 input (X_in) to L24 residual
contribution (X_out). Prompt-level stratified train/val split across 6
domains. Padding + mask collation so student self-attention sees exactly
the S positions Gemma did per prompt. Best-val checkpointing.
"""

from __future__ import annotations

import argparse
import os
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn as nn

from calm.llm_computer.r51 import R51Student, R51StudentConfig


DEFAULT_CAPTURES = "/tmp/r51_captures_broad.pt"
DEFAULT_OUT = "calm/llm_computer/r51/checkpoints/r51_student.pt"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--captures", type=str, default=DEFAULT_CAPTURES)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--d-model", type=int, default=128)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--d-ffn", type=int, default=512)
    p.add_argument("--max-len", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--val-frac", type=float, default=0.1)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    default_device = "cuda" if torch.cuda.is_available() else "cpu"
    p.add_argument("--device", type=str, default=default_device)
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_prompt_index(prompt_ids: torch.Tensor, n_prompts: int) -> tuple:
    """Return (starts, lengths) arrays indexed by prompt id. O(N) scan."""
    pid = prompt_ids.to(torch.int64).numpy()
    starts = [-1] * n_prompts
    lengths = [0] * n_prompts
    prev = -1
    for i, p in enumerate(pid):
        if p != prev:
            starts[p] = i
            prev = p
        lengths[p] += 1
    return starts, lengths


def stratified_split(
    prompt_domains: list, val_frac: float, seed: int
) -> tuple[list[int], list[int]]:
    """Per-domain deterministic shuffle, last val_frac per domain goes to val."""
    rng = random.Random(seed)
    by_domain: dict[int, list[int]] = {}
    for p_idx, dom in enumerate(prompt_domains):
        by_domain.setdefault(int(dom), []).append(p_idx)
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


def collate(
    p_ids: list[int],
    X_in: torch.Tensor,
    X_out: torch.Tensor,
    starts: list[int],
    lengths: list[int],
    prompt_domains: list,
    d_io: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    K = len(p_ids)
    S_max = max(lengths[p] for p in p_ids)
    x_batch = torch.zeros(K, S_max, d_io, dtype=X_in.dtype)
    y_batch = torch.zeros(K, S_max, d_io, dtype=X_out.dtype)
    mask = torch.zeros(K, S_max, dtype=torch.float32)
    dom_batch = torch.zeros(K, dtype=torch.int64)
    for k, p in enumerate(p_ids):
        s = starts[p]
        L = lengths[p]
        x_batch[k, :L] = X_in[s : s + L]
        y_batch[k, :L] = X_out[s : s + L]
        mask[k, :L] = 1.0
        dom_batch[k] = int(prompt_domains[p])
    return x_batch, y_batch, mask, dom_batch


def masked_mse(
    pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Average per valid (position, feature) element."""
    diff2 = (pred - target).pow(2)
    m = mask.unsqueeze(-1)
    denom = mask.sum() * pred.shape[-1]
    return (diff2 * m).sum() / denom.clamp_min(1.0)


def lr_at_step(step: int, warmup: int, base_lr: float) -> float:
    if warmup <= 0:
        return base_lr
    if step < warmup:
        return base_lr * (step + 1) / warmup
    return base_lr


@torch.no_grad()
def eval_val(
    model: nn.Module,
    val_ids: list[int],
    X_in: torch.Tensor,
    X_out: torch.Tensor,
    starts: list[int],
    lengths: list[int],
    prompt_domains: list,
    domain_names: list,
    batch_size: int,
    device: str,
    d_io: int,
) -> tuple[float, dict[str, float]]:
    model.eval()
    n_domains = len(domain_names)
    per_dom_sum = [0.0] * n_domains
    per_dom_count = [0.0] * n_domains
    total_sum = 0.0
    total_count = 0.0
    for i in range(0, len(val_ids), batch_size):
        batch_ids = val_ids[i : i + batch_size]
        x, y, mask, doms = collate(
            batch_ids, X_in, X_out, starts, lengths, prompt_domains, d_io
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        pred = model(x)
        diff2 = (pred - y).pow(2)
        m_expand = mask.unsqueeze(-1)
        per_batch_per_prompt = (diff2 * m_expand).sum(dim=(1, 2))
        weight_per_prompt = mask.sum(dim=1) * d_io
        for j, dom in enumerate(doms.tolist()):
            w = weight_per_prompt[j].item()
            s = per_batch_per_prompt[j].item()
            per_dom_sum[dom] += s
            per_dom_count[dom] += w
            total_sum += s
            total_count += w
    per_dom = {
        domain_names[d]: (per_dom_sum[d] / per_dom_count[d])
        if per_dom_count[d] > 0
        else float("nan")
        for d in range(n_domains)
    }
    total = total_sum / total_count if total_count > 0 else float("nan")
    model.train()
    return total, per_dom


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    caps_path = Path(args.captures)
    if not caps_path.exists():
        raise FileNotFoundError(f"captures not found: {caps_path}")
    print(f"[r51.3] loading captures: {caps_path}", flush=True)
    caps = torch.load(str(caps_path), weights_only=False, map_location="cpu")

    X_in: torch.Tensor = caps["X_in"]
    X_out: torch.Tensor = caps["X_out"]
    prompt_ids: torch.Tensor = caps["prompt_ids"]
    prompt_lens: torch.Tensor = caps["prompt_lens"]
    domain_ids: torch.Tensor = caps["domain_ids"]
    domain_names: list = list(caps["DOMAIN_NAMES"])
    n_prompts = int(prompt_lens.shape[0])

    d_io = int(X_in.shape[1])
    print(
        f"[r51.3] X_in={tuple(X_in.shape)} X_out={tuple(X_out.shape)} "
        f"n_prompts={n_prompts} n_domains={len(domain_names)}",
        flush=True,
    )

    starts, lengths = build_prompt_index(prompt_ids, n_prompts)

    prompt_domains: list = [0] * n_prompts
    dom_np = domain_ids.to(torch.int64).numpy()
    for p in range(n_prompts):
        s = starts[p]
        if s >= 0:
            prompt_domains[p] = int(dom_np[s])

    S_max_actual = max(lengths)
    print(
        f"[r51.3] per-prompt S: min={min(l for l in lengths if l>0)} "
        f"max={S_max_actual} mean={sum(lengths)/max(1,n_prompts):.1f}",
        flush=True,
    )
    if S_max_actual > args.max_len:
        raise ValueError(
            f"capture S_max {S_max_actual} exceeds --max-len {args.max_len}; "
            f"increase --max-len"
        )

    train_ids, val_ids = stratified_split(
        prompt_domains, args.val_frac, args.seed
    )
    train_dom_counts = {n: 0 for n in domain_names}
    val_dom_counts = {n: 0 for n in domain_names}
    for p in train_ids:
        train_dom_counts[domain_names[prompt_domains[p]]] += 1
    for p in val_ids:
        val_dom_counts[domain_names[prompt_domains[p]]] += 1
    print(
        f"[r51.3] split: train={len(train_ids)} val={len(val_ids)}",
        flush=True,
    )
    print(f"[r51.3] train per-domain: {train_dom_counts}", flush=True)
    print(f"[r51.3] val per-domain:   {val_dom_counts}", flush=True)

    cfg = R51StudentConfig(
        d_io=d_io,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_ffn=args.d_ffn,
        max_len=args.max_len,
        dropout=0.0,
    )
    device = args.device
    model = R51Student(cfg).to(device)
    n_params = model.param_count()
    print(
        f"[r51.3] student: d_model={cfg.d_model} n_layers={cfg.n_layers} "
        f"d_ffn={cfg.d_ffn} max_len={cfg.max_len} params={n_params:,} "
        f"({n_params/1e6:.2f}M) device={device}",
        flush=True,
    )

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

    train_history: list = []
    best_val_total = float("inf")
    best_step = 0
    best_per_domain: dict = {}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed + 1)
    train_ema = float("nan")
    step_times: list = []
    t0 = time.time()

    model.train()
    for step in range(1, args.steps + 1):
        t_step = time.time()

        if len(train_ids) < args.batch_size:
            batch_ids = [
                rng.choice(train_ids) for _ in range(args.batch_size)
            ]
        else:
            batch_ids = rng.sample(train_ids, args.batch_size)

        x, y, mask, _ = collate(
            batch_ids, X_in, X_out, starts, lengths, prompt_domains, d_io
        )
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        lr_now = lr_at_step(step, args.warmup_steps, args.lr)
        for g in opt.param_groups:
            g["lr"] = lr_now

        pred = model(x)
        loss = masked_mse(pred, y, mask)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()

        loss_val = float(loss.detach().item())
        if step <= 10:
            train_ema = loss_val if step == 1 else 0.7 * train_ema + 0.3 * loss_val
        else:
            train_ema = 0.98 * train_ema + 0.02 * loss_val

        step_times.append(time.time() - t_step)
        if len(step_times) > 50:
            step_times.pop(0)

        if step % 10 == 0 or step == 1:
            ms = 1000.0 * sum(step_times) / max(1, len(step_times))
            print(
                f"step {step} train_ema={train_ema:.4f} "
                f"lr={lr_now:.5f} ms/step={ms:.0f}",
                flush=True,
            )

        do_eval = (step % args.eval_every == 0) or (step == args.steps)
        if do_eval:
            val_total, per_dom = eval_val(
                model,
                val_ids,
                X_in,
                X_out,
                starts,
                lengths,
                prompt_domains,
                domain_names,
                args.batch_size,
                device,
                d_io,
            )
            per_dom_str = " ".join(
                f"val_{n}={per_dom[n]:.4f}" for n in domain_names
            )
            print(
                f"[eval step {step}] train_ema={train_ema:.4f} "
                f"val_total={val_total:.4f} {per_dom_str}",
                flush=True,
            )
            train_history.append(
                {
                    "step": step,
                    "train_ema": train_ema,
                    "val_total": val_total,
                    "per_domain": dict(per_dom),
                }
            )
            if val_total < best_val_total:
                best_val_total = val_total
                best_step = step
                best_per_domain = dict(per_dom)
                ckpt = {
                    "state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "train_history": train_history,
                    "best_val_total": best_val_total,
                    "best_step": best_step,
                    "per_domain_val_final": best_per_domain,
                    "seed": args.seed,
                    "captures_path": str(caps_path),
                }
                tmp_path = str(out_path) + ".tmp"
                torch.save(ckpt, tmp_path)
                os.replace(tmp_path, out_path)

    total_time = time.time() - t0
    ckpt_bytes = out_path.stat().st_size if out_path.exists() else 0
    ckpt_mb = ckpt_bytes / (1024 * 1024)

    print("", flush=True)
    print("=" * 72, flush=True)
    print(
        f"[r51.3] training complete in {total_time:.1f}s "
        f"({args.steps} steps)",
        flush=True,
    )
    print(f"  best_val_total = {best_val_total:.4f} at step {best_step}",
          flush=True)
    if best_per_domain:
        print("  best per-domain val MSE:", flush=True)
        for name in domain_names:
            v = best_per_domain.get(name, float("nan"))
            print(f"    {name:<10} {v:.4f}", flush=True)
    print(f"  checkpoint: {out_path} ({ckpt_mb:.2f} MB)", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
