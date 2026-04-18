"""R52.0: capture Gemma's native end-of-prompt logits across the R51.1
broad corpus for KL-divergence distillation.

Daemon-compatible. Assumes `m` (GemmaSubstrate) and `tok` (GemmaTokenizer)
are pre-bound globals by `bin/gemma_daemon.py`. Does NOT load Gemma.

For each of 3000 prompts (500 per domain: multi / single / trans / code
/ creative / factual, seed=42 — same ordering as R51.1 so token_ids
correspond row-for-row with /tmp/r51_captures_broad.pt prompts), runs
one Gemma forward and captures:

    logits_last[vocab] = m.forward(token_ids)[0, 0]

in fp16 on CPU (halves disk — 3000 * 262144 * 2 B ≈ 1.5 GB).

Saves a dict to /tmp/r52_teacher_logits.pt:
    teacher_logits    [N, vocab]     fp16 cpu     next-token logits
    token_ids_list    list[Tensor]   int32 cpu    variable-length per prompt
    prompts           list[str]                   prompt strings (in order)
    labels            list[str]                   domain label per prompt
    DOMAIN_NAMES      list[str]                   ["multi", ..., "factual"]
    prompt_counts     dict[str, int]              prompts per domain

Estimated runtime: ~2s/prompt * 3000 ≈ 100 min on the warm daemon
(head materialization over 262144 vocab is the dominant cost).

CLI:
    --limit N            Capture only first N prompts (smoke).
                         Default 0 = full corpus.
    --per-domain N       Per-domain sample target. Default 500.
    --out PATH           Output path. Default /tmp/r52_teacher_logits.pt.

Running the smoke case:
    # Edit this script's default --per-domain to 1 (or pass --per-domain 1)
    # for a 6-prompt subset, then run via daemon.
"""

from __future__ import annotations

import argparse
import random
import sys

import torch

from calm.llm_computer.r51.prompt_bank import build_broad_corpus


DEFAULT_OUT = "/tmp/r52_teacher_logits.pt"
DOMAIN_NAMES = ["multi", "single", "trans", "code", "creative", "factual"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--per-domain", type=int, default=500)
    p.add_argument("--limit", type=int, default=0,
                   help="Cap number of prompts (0 = no cap).")
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    # argparse handles its own argv; daemon execs us via exec(compile(...))
    # with no argv, so fall back gracefully.
    argv = sys.argv[1:] if len(sys.argv) > 1 else []
    return p.parse_args(argv)


def main() -> None:
    assert "m" in globals(), "daemon contract: `m` must be pre-bound"
    assert "tok" in globals(), "daemon contract: `tok` must be pre-bound"

    args = parse_args()

    rng = random.Random(42)
    prompts, prompt_counts = build_broad_corpus(
        rng, per_domain=args.per_domain
    )
    total = len(prompts)
    if args.limit > 0:
        prompts = prompts[: args.limit]
        total = len(prompts)
    print(
        f"[r52.0] corpus: {total} prompts (per_domain={args.per_domain}, "
        f"limit={args.limit}), prompt_counts={prompt_counts}",
        flush=True,
    )

    from calm.llm_computer.gemma_substrate import KVCache

    kept_logits: list[torch.Tensor] = []
    kept_token_ids: list[torch.Tensor] = []
    kept_prompts: list[str] = []
    kept_labels: list[str] = []
    skipped = 0

    cfg = m.config

    with torch.no_grad():
        for i, (prompt, label) in enumerate(prompts):
            ids = tok.encode(prompt)
            if len(ids) < 2:
                skipped += 1
                continue
            token_ids = torch.tensor([ids], device="cuda", dtype=torch.long)
            cache = KVCache(cfg.n_layers, device="cuda")
            logits = m.forward(
                token_ids, device="cuda", kv_cache=cache, start_pos=0
            )  # [1, 1, vocab]
            logits_last = logits[0, 0].detach().to("cpu", dtype=torch.float16)
            kept_logits.append(logits_last)
            # Store token ids as int32 to keep mem down — vocab < 2^31.
            kept_token_ids.append(
                torch.tensor(ids, dtype=torch.int32, device="cpu")
            )
            kept_prompts.append(prompt)
            kept_labels.append(label)

            if (i + 1) % 250 == 0:
                print(
                    f"  {i+1}/{total} processed, skipped={skipped}",
                    flush=True,
                )
            if (i + 1) % 500 == 0:
                torch.cuda.empty_cache()

    teacher_logits = torch.stack(kept_logits, dim=0).contiguous()

    payload = {
        "teacher_logits": teacher_logits,
        "token_ids_list": kept_token_ids,
        "prompts": kept_prompts,
        "labels": kept_labels,
        "DOMAIN_NAMES": DOMAIN_NAMES,
        "prompt_counts": prompt_counts,
    }
    torch.save(payload, args.out)

    size_mb = teacher_logits.numel() * 2 / (1024 * 1024)
    print("", flush=True)
    print(f"[r52.0] saved {args.out}", flush=True)
    print(
        f"  teacher_logits shape: {tuple(teacher_logits.shape)}  "
        f"dtype={teacher_logits.dtype}  size={size_mb:.1f} MB",
        flush=True,
    )
    print(f"  n_prompts kept: {len(kept_prompts)}  skipped: {skipped}",
          flush=True)

    # Per-domain summary
    per_dom: dict[str, int] = {n: 0 for n in DOMAIN_NAMES}
    for lab in kept_labels:
        per_dom[lab] = per_dom.get(lab, 0) + 1
    print("  per-domain kept:", flush=True)
    for name in DOMAIN_NAMES:
        print(f"    {name:<10} {per_dom.get(name, 0):>6}", flush=True)

    # Peek at first logit row for sanity
    first = teacher_logits[0].float()
    print(
        f"  first prompt: {kept_prompts[0]!r}",
        flush=True,
    )
    print(
        f"  first logits: min={first.min().item():.3f} "
        f"max={first.max().item():.3f} "
        f"argmax={int(first.argmax())} "
        f"mean={first.mean().item():.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
else:
    main()
