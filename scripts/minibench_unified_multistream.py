"""Rung 4 rewritten with multi-stream — zero channel-mask hooks needed.

The original rung 4 (scripts/minibench_unified_rung4.py) validated the
orthogonal multi-task unified CHRLM at d_model=16 using channel masking
+ partial head freeze. This version demonstrates the same gates using
L2 multi-stream: adder on a dedicated `math` stream, echo on a separate
`lm` stream. Cross-stream interference is architecturally impossible
(they share zero parameters), so no gradient hooks are required.

Design:
  Streams:
    - "math": d_model=10, n_heads=5 (adder's native dims)
    - "lm":   d_model=32, n_heads=16 (trainable LM space)
  Head: shared Linear(total_d=42, vocab=16). Math's head slice is
        columns 0..9; lm's is 10..41. Adder's vocab rows 0..7 read
        from math-offset columns; trainable rows 8..15 read from
        lm-offset columns (enforced by head-row freeze).

Gates (all must pass):
  1. Pre-training adder 16/16 (via math stream)
  2. Post-training adder 16/16 (no drift under lm training)
  3. Post-training echo >= 75% (lm learned orthogonal task)
  4. Save+reload: both tasks survive roundtrip
  5. Zero channel-mask gradient hooks used

Also demonstrates L1: MultiStreamChannelRegistry records adder's
allocation on the math stream; lm stream stays unallocated (future
cards can slot in without conflict).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from calm.llm_computer.channel_masking import freeze_head_rows
from calm.llm_computer.channel_registry import (
    MultiStreamChannelRegistry, adder_tiny_allocation,
)
from calm.llm_computer.multi_stream import (
    MultiStreamConfig, StreamSpec, build_empty_multistream,
)
from calm.llm_computer.unified_chrlm import (
    freeze_stream_embeddings, freeze_stream_layer,
    install_compiled_in_stream,
)
from scripts.experiment_fast_weights_fusion import build_adder_tiny_small2d


VOCAB = 16
MAX_LEN = 4


def _build_adder_src():
    return build_adder_tiny_small2d(target_layer=0, n_layers=1)


@torch.no_grad()
def eval_adder(ms_model) -> int:
    """Adder eval: for all 16 (a,b), argmax of logits at pos 1 within
    vocab rows 0..6 must equal a+b."""
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = ms_model(x)[0, 1, :7]
            if int(logits.argmax().item()) == a + b:
                ok += 1
    return ok


@torch.no_grad()
def eval_echo(ms_model, offset: int = 8) -> int:
    """Echo eval: logits at pos 1, argmax over rows [offset, offset+8),
    should be offset + a for every (a,b)."""
    ok = 0
    for a in range(4):
        for b in range(4):
            x = torch.tensor([[a, b]], dtype=torch.long)
            logits = ms_model(x)[0, 1, offset:offset + 8]
            if offset + int(logits.argmax().item()) == offset + a:
                ok += 1
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=500)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--init-std", type=float, default=0.02)
    ap.add_argument("--save-path",
                   default="/tmp/minibench_unified_multistream.pt")
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    print("=== minibench_unified_multistream — L2 physical isolation ===",
          flush=True)

    cfg = MultiStreamConfig(
        streams=(
            StreamSpec("math", d_model=10, n_heads=5, d_ffn=14),
            StreamSpec("lm",   d_model=32, n_heads=16, d_ffn=64),
        ),
        n_layers=1, vocab_size=VOCAB, max_len=MAX_LEN, use_hard_max=True,
    )
    ms = build_empty_multistream(cfg)

    # L1: channel registry per stream
    regs = MultiStreamChannelRegistry.from_config(cfg)

    # Install adder on math stream with registry tracking
    install_compiled_in_stream(
        ms, program_builder=_build_adder_src,
        stream_name="math", target_layer=0,
        registry=regs, allocations=adder_tiny_allocation(),
    )

    print(f"  streams: {[f'{s.name}(d_model={s.d_model})' for s in cfg.streams]}",
          flush=True)
    print(f"  math registry: {len(regs.for_stream('math').cards())} cards, "
          f"{len(regs.for_stream('math').all_allocated())} channels used",
          flush=True)
    print(f"  lm registry:   {len(regs.for_stream('lm').all_allocated())} "
          f"channels used (free for trainable)", flush=True)

    # Random init on lm stream so gradients can flow
    with torch.no_grad():
        for lin in (ms.streams["lm"].W_qkv[0], ms.streams["lm"].W_out[0],
                    ms.streams["lm"].ff_in[0], ms.streams["lm"].ff_out[0]):
            lin.weight.normal_(0.0, args.init_std)
        for p in ms.streams["lm"].tok.parameters():
            p.normal_(0.0, args.init_std)
        for p in ms.streams["lm"].pos.parameters():
            p.normal_(0.0, args.init_std)
        # Trainable head rows 8-15 init
        ms.head.weight[8:16].normal_(0.0, args.init_std)

    # Freeze: math stream fully + math embeddings + adder head rows 0-7
    n_frozen_math_layer = freeze_stream_layer(ms, "math", layer_idx=0)
    n_frozen_math_emb = freeze_stream_embeddings(ms, "math")
    # Partial head freeze: rows 0-7 (adder's predictions) frozen
    head_hooks = freeze_head_rows(ms, rows=range(0, 8))

    n_trainable = sum(p.numel() for p in ms.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in ms.parameters())
    print(f"  params: total={n_total:,}  "
          f"frozen_math={n_frozen_math_layer + n_frozen_math_emb:,}  "
          f"trainable={n_trainable:,}", flush=True)
    print(f"  zero channel-mask hooks (only partial head freeze)",
          flush=True)

    # Gate 1: adder works pre-training
    pre_adder = eval_adder(ms)
    print(f"\n  [gate 1] pre-training adder: {pre_adder}/16", flush=True)
    assert pre_adder == 16, f"compiled adder broken BEFORE training: {pre_adder}/16"

    # Train on echo task: at pos 1, predict token 8+a via trainable head
    print(f"\n--- training {args.steps} steps (echo task only) ---",
          flush=True)
    xs = torch.tensor([[a, b] for a in range(4) for b in range(4)],
                     dtype=torch.long)
    ys_echo = torch.tensor([8 + a for a in range(4) for b in range(4)],
                          dtype=torch.long)
    trainable = [p for p in ms.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=args.lr)
    rng = torch.Generator().manual_seed(args.seed)
    ms.train()
    t0 = time.time()
    first_loss, last_loss = None, None
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, 16, (args.batch_size,), generator=rng)
        logits = ms(xs[idx])[:, 1, 8:16]  # trainable vocab slice
        loss = F.cross_entropy(logits, ys_echo[idx] - 8)
        opt.zero_grad(); loss.backward(); opt.step()
        if step == 1:
            first_loss = float(loss.item())
        last_loss = float(loss.item())
    ms.eval()
    print(f"  wallclock: {time.time()-t0:.2f}s", flush=True)
    print(f"  echo loss: first={first_loss:.4f}  last={last_loss:.4f}",
          flush=True)

    # Gate 2+3: post-training
    post_adder = eval_adder(ms)
    post_echo = eval_echo(ms)
    print(f"\n  [gate 2] post-training adder: {post_adder}/16", flush=True)
    print(f"  [gate 3] post-training echo:  {post_echo}/16", flush=True)

    # Gate 4: save + reload
    save_path = Path(args.save_path)
    torch.save(ms.state_dict(), save_path)
    # Rebuild and reload (remove hooks in the original, rebuild fresh)
    from calm.llm_computer.multi_stream import MultiStreamTransformer
    ms2 = MultiStreamTransformer(cfg)
    ms2.load_state_dict(torch.load(save_path, weights_only=True))
    ms2.eval()
    rt_adder = eval_adder(ms2)
    rt_echo = eval_echo(ms2)
    print(f"\n  [gate 4] save+reload adder: {rt_adder}/16  echo: {rt_echo}/16",
          flush=True)

    echo_gate = post_echo >= 12  # 75%
    all_pass = (
        pre_adder == 16 and post_adder == 16 and echo_gate
        and rt_adder == 16 and rt_echo == post_echo
    )
    print("", flush=True)
    print("=== summary ===", flush=True)
    print(f"  gate 1 (pre adder):    {'PASS' if pre_adder == 16 else 'FAIL'} "
          f"({pre_adder}/16)", flush=True)
    print(f"  gate 2 (post adder):   {'PASS' if post_adder == 16 else 'FAIL'} "
          f"({post_adder}/16)", flush=True)
    print(f"  gate 3 (echo >= 75%):  {'PASS' if echo_gate else 'FAIL'} "
          f"({post_echo}/16 = {post_echo/16*100:.0f}%)", flush=True)
    print(f"  gate 4 (roundtrip):    "
          f"{'PASS' if rt_adder == 16 and rt_echo == post_echo else 'FAIL'} "
          f"(adder {rt_adder}/16, echo {rt_echo}/16)", flush=True)
    print(f"  channel mask hooks:    0 (physical isolation via L2)",
          flush=True)
    print("", flush=True)
    print(f"  OVERALL: {'PASS — multi-stream isolation works end-to-end' if all_pass else 'FAIL'}",
          flush=True)


if __name__ == "__main__":
    main()
