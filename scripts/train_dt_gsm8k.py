"""Train a (RDT-v2 or baseline) Delta-Transducer card on real GSM8k.

S0b2 of the rdt-v2 first-flag-enabled-card arc (board task
`1779311831769-1d1e02e5`). Codex audit chain `1779311799556` →
`1779312982222` → `1779313349390` → `1779313584790` locked the contract:

- Corpus: real GSM8k via `datasets` library parquet backend
  (HF datasets-server rate-limits paged fetches, see scripts/preflight).
- Split: train+val from `train` (last 10% deterministic held-out); `test`
  is OOV check + final A/B only.
- Tokenizer: `calm.llm_computer.gsm8k_tokenizer.Gsm8kTokenizer.from_corpus`
  on train+val only. 98-token vocab; normalizer v2 applied at train,
  eval, inference.
- Hard-fail at startup if any corpus char is OOV vs declared vocab.
- Target: `<bos> question <sep> {integer} <eos>` (final-integer-only).
- Loss: F.nll_loss on log-probs, masked to positions `> sep_pos`.
- max_len=512 (1.46% truncation tail).
- Tier-A+B flag bundle exposed via the S0a CLI plumbing pattern.
- Checkpoint shape compatible with `dt_install.load_dt_checkpoint`:
  `model_state` + `config` (incl. `gsm8k_char_vocab` and
  `gsm8k_normalizer_version` metadata).

Usage:
    PYTHONPATH=. python3 -u scripts/train_dt_gsm8k.py \\
        --epochs 30 --batch-size 32 --max-len 512 \\
        --d-model 64 --n-heads 32 --n-layers 4 --d-ffn 128 \\
        --use-loop-index --use-input-injection --use-z-init \\
        --use-lecun-init --use-gated-attention --use-short-conv \\
        --use-h-rmsnorm --use-h-layer-stack \\
        --n-iterations 2 --h-cycles 2

Baseline (flags off) and Core-H/L (codex's locked first-card config)
share this entry-point; the `--use-*` flags pick the variant.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from calm.llm_computer.copy_augmented_delta import build_copy_augmented_delta
from calm.llm_computer.gsm8k_tokenizer import (
    NORMALIZER_VERSION,
    Gsm8kTokenizer,
)


DEFAULT_CHECKPOINT = Path("calm/hrm/checkpoints/dt_gsm8k_best.pt")


def load_gsm8k_splits(val_frac: float = 0.10) -> tuple[list[dict], list[dict], list[dict]]:
    """Load GSM8k via the `datasets` lib parquet backend.

    Returns (train, val, test). Train is 90% (deterministic head); val is
    10% (deterministic tail of train). Test is the full HF test split.
    """
    import re

    from datasets import load_dataset
    out: dict[str, list[dict]] = {"train": [], "test": []}
    for split in ("train", "test"):
        ds = load_dataset("openai/gsm8k", "main", split=split)
        for i, r in enumerate(ds):
            gt = r["answer"]
            m = re.search(r"####\s*(-?[\d,]+)", gt)
            if not m:
                continue
            try:
                expected = int(m.group(1).replace(",", "").strip())
            except ValueError:
                continue
            out[split].append({
                "id": f"gsm8k_{split}_{i}",
                "question": r["question"],
                "expected": expected,
                "answer_raw": gt,
            })
    full_train = out["train"]
    n_val = int(len(full_train) * val_frac)
    train = full_train[:-n_val] if n_val else full_train
    val = full_train[-n_val:] if n_val else []
    return train, val, out["test"]


class Gsm8kDataset(Dataset):
    """Yields `(ids, sep_pos, length)` per row.

    Rows exceeding `max_len` are dropped (truncation rate measured by the
    preflight; ~1.46% at max_len=512). Dropping > truncating prevents the
    last-chars-of-question being silently amputated.
    """

    def __init__(self, rows: list[dict], tok: Gsm8kTokenizer, max_len: int):
        self.tok = tok
        self.max_len = max_len
        self.items: list[tuple[list[int], int]] = []
        n_dropped = 0
        for r in rows:
            ids, sep_pos = tok.encode_example(r["question"], r["expected"])
            if len(ids) > max_len:
                n_dropped += 1
                continue
            self.items.append((ids, sep_pos))
        self.n_dropped = n_dropped

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def collate(batch, pad_id: int, max_len: int, fixed_shape: bool = False):
    """Right-pad to the batch's longest sequence; build a target-position
    mask (1 for positions whose NEXT-token prediction loss counts).

    Slice 13e.1 (TRM-1.58 throughput-to-signal track):
      `fixed_shape=False` (default, preserved baseline): pads to per-batch
      max_L. Variable shape per batch — precludes CUDA graph capture at
      training-step granularity.

      `fixed_shape=True`: pads ALL batches to fixed `max_len`. Tail
      positions in `pad` are `pad_id`; tail positions in `loss_mask` are
      False. Invariant: the last real loss position remains `L-2` for any
      example with `L > sep_pos + 1` (matches `_masked_shifted_nll` which
      uses `targets=ids[:, 1:]` and `mask=mask[:, :-1]`). Per codex audit
      msg `1779383623566-0b118cd2`: tail-pad regions are pad_id AND
      loss_mask=False, so model loss is semantically identical to
      `fixed_shape=False` on the same examples.
    """
    seq_lens = [len(ids) for ids, _ in batch]
    if fixed_shape:
        max_L = max_len
    else:
        max_L = max(seq_lens)
    B = len(batch)
    pad = torch.full((B, max_L), pad_id, dtype=torch.long)
    sep_positions = torch.zeros(B, dtype=torch.long)
    # Loss is on positions where the model PREDICTS a target token. The
    # model produces log-probs at positions [0..L-1] predicting input[1..L].
    # So we want mask[t] = True iff (t+1) is a target-side position
    # (sep_pos < t+1 < L, i.e. t >= sep_pos).
    loss_mask = torch.zeros(B, max_L, dtype=torch.bool)
    for i, (ids, sep_pos) in enumerate(batch):
        L = len(ids)
        pad[i, :L] = torch.tensor(ids, dtype=torch.long)
        sep_positions[i] = sep_pos
        # Loss positions: sep_pos <= t < L-1 (predicts ids[t+1] which is
        # within the target span, including the final <eos>).
        if L > sep_pos + 1:
            loss_mask[i, sep_pos:L - 1] = True
    return pad, loss_mask, sep_positions


def _masked_shifted_nll(log_probs: torch.Tensor, ids: torch.Tensor,
                        mask: torch.Tensor) -> torch.Tensor:
    """Canonical trainer NLL: predict ids[:, 1:] from log_probs[:, :-1],
    mask[:, :-1]. Returns scalar mean over target positions (mask=True).

    Factored so the final-NLL path and each per-iter NLL path (deep
    supervision) use identical shift + mask semantics. The mask MUST be
    the target-position mask built by `collate` — prompt positions stay
    at 0, so prompt-token predictions never contribute to the loss.
    """
    log_probs = log_probs[:, :-1].contiguous()
    targets = ids[:, 1:].contiguous()
    mask = mask[:, :-1].contiguous()
    B, L, V = log_probs.shape
    nll_per = F.nll_loss(
        log_probs.reshape(B * L, V),
        targets.reshape(B * L),
        reduction="none",
    ).reshape(B, L)
    denom = mask.float().sum().clamp(min=1.0)
    return (nll_per * mask.float()).sum() / denom


def autoreg_decode_integer(model, tok: Gsm8kTokenizer, question: str,
                           max_new: int = 16, device: str = "cuda") -> str:
    """Greedy autoreg from `<bos> question <sep>` to first `<eos>`.

    Returns the decoded target string (post-decode; no normalization).

    NOTE: this is the legacy single-forward-per-token decoder. It does
    NOT use the segment loop / carry / halt-head — appropriate for
    vanilla NLL training (`use_carry=False AND use_halt_head=False`).
    For HRM-Text-trained models (`use_carry=True AND use_halt_head=True`),
    use `autoreg_decode_integer_hrm` — single-forward decode is an
    inference-path mismatch for those models. See Slice 13h.
    """
    model.eval()
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    ids = torch.tensor([prefix], dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= model.config.max_len:
                break
            log_probs = model(ids)
            next_id = int(log_probs[0, -1].argmax().item())
            if next_id == tok.eos_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)
    return tok.decode(ids[0, len(prefix):].tolist(), stop_at_eos=True)


def autoreg_decode_integer_hrm(
    model, tok: Gsm8kTokenizer, question: str,
    *, m_max: int = 4, max_new: int = 16,
    eval_min_segments: int = 1,
    device: str = "cuda",
) -> dict:
    """HRM-Text source-faithful ACT-greedy inference (Slice 13h).

    Per token, runs outer segment loop with carry across segments for
    the SAME prefix, carry=None at the start of each token. Halts on
    ``Q_halt > Q_continue`` (gated by ``eval_min_segments``) or at
    ``seg == m_max`` (forced). Emits from the halted segment's
    final-position logits.

    Per HRM-Text §5:222-255 inference contract + codex audit
    msg 1779391827108-b211c8d8 (scope-lock for Slice 13h).

    Args:
      eval_min_segments: minimum segments before halt-head can fire
                         (deterministic analog of train-side m_min;
                         default 1 = head decides every segment, no
                         stochastic exploration at eval).

    Returns:
      dict with:
        decoded: str — the post-sep token sequence
        segs_per_token: list[int] — segments used to emit each token
        halt_histogram: list[int] len m_max — count of halts per seg idx
        qh_sum / qc_sum / n_emits: aggregation primitives for callers
    """
    model.eval()
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    ids = torch.tensor([prefix], dtype=torch.long, device=device)

    segs_per_token: list[int] = []
    halt_hist = [0] * m_max
    qh_sum = 0.0
    qc_sum = 0.0
    n_emits = 0

    with torch.no_grad():
        for _ in range(max_new):
            if ids.shape[1] >= model.config.max_len:
                break
            carry = None
            final_log_probs = None
            seg_emitted = 0
            for seg_m in range(m_max):
                seg = seg_m + 1
                out = model(ids, carry=carry, return_carry=True)
                if isinstance(out, tuple):
                    log_probs, carry_m = out
                else:
                    log_probs, carry_m = out, None
                q_pair = model.last_q_pair  # (1, 2)
                q_halt = float(q_pair[0, 0])
                q_continue = float(q_pair[0, 1])
                halt = (q_halt > q_continue) and (seg >= eval_min_segments)
                forced = (seg == m_max)
                if halt or forced:
                    final_log_probs = log_probs
                    seg_emitted = seg
                    halt_hist[seg - 1] += 1
                    qh_sum += q_halt
                    qc_sum += q_continue
                    n_emits += 1
                    break
                carry = carry_m.detach() if carry_m is not None else None

            segs_per_token.append(seg_emitted)
            next_id = int(final_log_probs[0, -1].argmax().item())
            if next_id == tok.eos_id:
                break
            ids = torch.cat([ids, torch.tensor([[next_id]], device=device)], dim=1)

    decoded = tok.decode(ids[0, len(prefix):].tolist(), stop_at_eos=True)
    return {
        "decoded": decoded,
        "segs_per_token": segs_per_token,
        "halt_histogram": halt_hist,
        "qh_sum": qh_sum,
        "qc_sum": qc_sum,
        "n_emits": n_emits,
    }


def autoreg_eval(model, tok: Gsm8kTokenizer, val_rows: list[dict],
                 cap: int, device: str) -> tuple[float, int, int]:
    """Returns (accuracy, n_correct, n_evaluated). Scores via
    `surface_gsm8k.score_row` for parity with the b-v Step 2 A/B harness.

    Legacy decoder; appropriate only for vanilla-NLL trained models
    (no carry, no segment loop). For HRM-Text-trained models, use
    `autoreg_eval_hrm` to avoid the Slice 13h inference-path mismatch.
    """
    from scripts.bv_step2.surface_gsm8k import score_row

    n_eval = min(cap, len(val_rows))
    n_correct = 0
    for r in val_rows[:n_eval]:
        generated = autoreg_decode_integer(model, tok, r["question"], device=device)
        _, correct = score_row(generated, r)
        if correct:
            n_correct += 1
    acc = n_correct / max(n_eval, 1)
    return acc, n_correct, n_eval


def autoreg_eval_hrm(
    model, tok: Gsm8kTokenizer, val_rows: list[dict],
    cap: int, device: str,
    *, m_max: int = 4, eval_min_segments: int = 1,
) -> dict:
    """HRM-Text source-faithful eval with full telemetry (Slice 13h).

    Uses the carry-aware outer-segment loop decoder. Reports both the
    legacy parsed-numeric metric (via `surface_gsm8k.score_row` — same
    as `autoreg_eval`) AND the exact-string token-sequence match
    (same semantics as train-side `full_answer_correct`). Folds in the
    13f.4 reward-alignment probe at zero marginal cost.

    Returns dict with:
      acc_parsed, acc_exact: floats (0..1)
      n_correct_parsed, n_correct_exact: ints
      n_evaluated: int
      avg_segs_per_token: float
      halt_histogram: list[int] len m_max
      qh_mean, qc_mean: floats
    """
    from scripts.bv_step2.surface_gsm8k import score_row

    n_eval = min(cap, len(val_rows))
    n_correct_parsed = 0
    n_correct_exact = 0
    total_segs = 0
    total_tokens = 0
    halt_hist = [0] * m_max
    qh_sum_total = 0.0
    qc_sum_total = 0.0
    n_emits_total = 0

    for r in val_rows[:n_eval]:
        result = autoreg_decode_integer_hrm(
            model, tok, r["question"],
            m_max=m_max, max_new=16,
            eval_min_segments=eval_min_segments, device=device,
        )
        _, parsed_correct = score_row(result["decoded"], r)
        if parsed_correct:
            n_correct_parsed += 1
        # Exact-string match: decoded == str(expected_int). The trainer's
        # target is `<sep> {digits} <eos>`; decode strips bos+question+sep
        # prefix and stops at eos, so decoded should == str(expected) for
        # exact match.
        if result["decoded"] == str(r["expected"]):
            n_correct_exact += 1
        for s in result["segs_per_token"]:
            total_segs += s
            total_tokens += 1
        for i in range(m_max):
            halt_hist[i] += result["halt_histogram"][i]
        qh_sum_total += result["qh_sum"]
        qc_sum_total += result["qc_sum"]
        n_emits_total += result["n_emits"]

    return {
        "acc_parsed": n_correct_parsed / max(n_eval, 1),
        "acc_exact": n_correct_exact / max(n_eval, 1),
        "n_correct_parsed": n_correct_parsed,
        "n_correct_exact": n_correct_exact,
        "n_evaluated": n_eval,
        "avg_segs_per_token": total_segs / max(total_tokens, 1),
        "halt_histogram": halt_hist,
        "qh_mean": qh_sum_total / max(n_emits_total, 1),
        "qc_mean": qc_sum_total / max(n_emits_total, 1),
    }


def _build_ckpt_config(
    m, tok: Gsm8kTokenizer, *,
    max_len: int, d_model: int, n_heads: int, n_layers: int,
    d_ffn: int, n_copy_heads: int, aux_weight: float,
) -> dict:
    """Slice 13l helper: shared ckpt-blob config dict for both epoch-end
    saves AND step-N saves (--save-at-step). Single source of truth to
    prevent config drift between save sites per codex msg 1779444785341."""
    return {
        "vocab_size": tok.vocab_size,
        "max_len": max_len,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "d_ffn": d_ffn,
        "n_copy_heads": n_copy_heads,
        "copy_gate_bias_init": -2.0,
        "use_chunkwise": getattr(m.config, "use_chunkwise", True),
        "chunk_size": getattr(m.config, "chunk_size", 32),
        "n_iterations": getattr(m.config, "n_iterations", 1),
        "use_loop_index": getattr(m.config, "use_loop_index", False),
        "use_input_injection": getattr(m.config, "use_input_injection", False),
        "use_gated_attention": getattr(m.config, "use_gated_attention", False),
        "use_z_init": getattr(m.config, "use_z_init", False),
        "use_lecun_init": getattr(m.config, "use_lecun_init", False),
        "use_prefix_lm": getattr(m.config, "use_prefix_lm", False),
        "use_softmax_attn": getattr(m.config, "use_softmax_attn", False),
        "use_softmax_only": getattr(m.config, "use_softmax_only", False),
        "h_cycles": getattr(m.config, "h_cycles", 1),
        "use_h_rmsnorm": getattr(m.config, "use_h_rmsnorm", False),
        "use_short_conv": getattr(m.config, "use_short_conv", False),
        "use_h_layer_stack": getattr(m.config, "use_h_layer_stack", False),
        "use_halt_head": getattr(m.config, "use_halt_head", False),
        "use_carry": getattr(m.config, "use_carry", False),
        "use_pre_rmsnorm": getattr(m.config, "use_pre_rmsnorm", False),
        "use_ternary_bulk": getattr(m.config, "use_ternary_bulk", False),
        "gsm8k_char_vocab": tok.vocab_as_list(),
        "gsm8k_normalizer_version": tok.normalizer_version,
        "aux_weight": float(aux_weight),
        "loss_mode": (
            "final_plus_per_iter_mean"
            if aux_weight > 0.0 else "final_only"
        ),
    }


def train(
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    d_model: int = 64,
    n_heads: int = 32,
    n_layers: int = 4,
    d_ffn: int = 128,
    max_len: int = 512,
    n_copy_heads: int = 4,
    seed: int = 42,
    eval_every: int = 1,
    eval_cap: int = 100,
    device: str | None = None,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    n_train_cap: int | None = None,
    n_val_cap: int | None = None,
    # rdt-v2 Tier A+B build-time flags (S0a CLI plumbing pattern).
    use_chunkwise: bool = True,
    n_iterations: int = 1,
    use_loop_index: bool = False,
    use_input_injection: bool = False,
    use_gated_attention: bool = False,
    use_z_init: bool = False,
    use_lecun_init: bool = False,
    use_prefix_lm: bool = False,
    use_softmax_attn: bool = False,
    use_softmax_only: bool = False,
    h_cycles: int = 1,
    use_h_rmsnorm: bool = False,
    use_short_conv: bool = False,
    use_h_layer_stack: bool = False,
    use_halt_head: bool = False,
    use_carry: bool = False,
    chunk_size: int = 32,
    # Slice 12: per-layer Pre-RMSNorm flag. Fixes S2 NaN root cause
    # (residual magnitude blow-up through L stack at the Core-H/L
    # flag composition). Default False → bit-equivalent to Slice 1-11.
    use_pre_rmsnorm: bool = False,
    # TRM-1.58 (Slice 13) — native W1.58A8 BitNet-style ternary bulk
    # projections from step zero. Flips W_qkv/W_out/ff_in/ff_out + H-bank
    # mirrors + copy_k_proj to TernaryLinear with BF16 master + STE.
    # Mechanism-critical projections (copy_gate/copy_q/beta/attn_gate/
    # RMSNorm/halt/embeddings/head) stay FP. Default False → bit-equivalent
    # to pre-Slice-13 path; flag-on enters TRM-1.58 training regime.
    use_ternary_bulk: bool = False,
    # Deep-supervision aux loss (S0c-aux per codex audit `1779314708107`).
    # 0.0 = final-NLL only (baseline, bit-equivalent to pre-S0c-aux path).
    # >0 = enable per-iter NLL via `return_per_iter=True`.
    aux_weight: float = 0.0,
    # TRM-1.58 throughput-to-signal track Slice 13e.1: fixed-shape padding.
    # When True, every batch pads to `max_len` (default 512) instead of the
    # per-batch max — unlocks training-step CUDA graph capture at fixed
    # shapes. Default False preserves baseline collate behavior. Loss/logits
    # at non-pad positions are semantically identical (causal mask + tail
    # loss_mask=False); only the per-step wall-clock changes when graph
    # capture also lands.
    fixed_shape_padding: bool = False,
    # TRM-1.58 throughput-to-signal track Slice 13e.2: CUDA graph capture
    # of the training step (forward + loss + backward + grad-clip). The
    # finite-check and opt.step REMAIN OUTSIDE the captured region per
    # codex guardrail 2 (preserve training semantics — NaN tripwire must
    # fire BEFORE opt.step, not after). Requires `fixed_shape_padding=True`
    # (variable shapes preclude capture). Adds `capturable=True` to AdamW.
    # Hard requirement: aux_weight > 0 (deep-supervision path); the
    # final-NLL-only branch isn't captured in this slice (separate later).
    graph_capture: bool = False,
    # TRM-1.58 Slice 13f.2: source-faithful HRM-Text/Sapient ACT training.
    # When True, training step runs M_max outer segments per HRM-Text §4-5
    # (RESEARCH/HRM/03_Training_Procedure.md:177-187, :222-255). Each segment
    # = one forward with detached carry; loss = NLL + BCE(Q̂, Ĝ) where
    # Ĝ_halt = full-answer reward and Ĝ_continue = stop-grad max-Q of next
    # segment (NO-grad lookahead per codex guardrail to avoid autograd-vs-
    # opt.step version-error). Per-example active mask drops halting examples
    # from subsequent segments. Requires use_halt_head=True. INCOMPATIBLE
    # with graph_capture (future slice).
    use_hrm_act: bool = False,
    m_max: int = 4,
    m_min_epsilon: float = 0.1,  # HRM-Text §5:234-236 exploration probability
    # Slice 13i.1: deterministic deeper-M_min warmup curriculum.
    # During the first `m_min_warmup_epochs` epochs, force m_min to a
    # constant `m_min_warmup_value` on every batch (override the
    # stochastic m_min_epsilon draw). Anneals back to the source-faithful
    # epsilon-stochastic schedule after the warmup. Default 0 = no
    # warmup, preserves current behavior.
    m_min_warmup_epochs: int = 0,
    m_min_warmup_value: int = 4,
    # Slice 13i.1: continue-biased Q-head init. When True, patches the
    # halt_head bias so Q_continue > Q_halt before training starts.
    # Inverts the default policy from "halt at seg 1" to "continue to
    # M_max" — the model must LEARN to halt rather than learn-to-not-halt.
    # Halt bias = -1.0, Continue bias = +1.0 → sigmoid Qh ~ 0.27, Qc ~ 0.73.
    q_init_bias_continue: bool = False,
    # Slice 13l: --save-at-step N mid-training ckpt hook. Repeatable —
    # pass a sequence of step indices, ckpts saved to
    # `<stem>_step{N:05d}.pt` when step_idx ∈ save_at_steps (HRM segment-
    # loop path only). Default None = disabled, no extra saves. Closes
    # the gap that bit 13j/13k where slope-based aborts left no usable
    # ckpt to probe. Codex msg 1779446584981 spec: minimal API, single
    # kwarg, frozenset-once dedupe, positive-int validation.
    save_at_steps: list[int] | None = None,
    # Interior-batch loss/grad logging (NaN-diagnostic). 0 = disabled (default,
    # preserves prior log shape). N>0 = print `[ep E step S] loss=X grad_norm=Y`
    # every N batches AND early-exit on first non-finite loss with diagnostic
    # context (step idx, last finite loss, last grad_norm).
    log_every: int = 0,
) -> None:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    if use_softmax_only:
        print("[gsm8k] Slice 13k: SOFTMAX-ONLY mixer mode ENABLED "
              "(delta path skipped; softmax_attn auto-forced ON)")

    # Slice 13l: validate + dedupe save_at_steps once at train entry.
    # Codex msg 1779446584981 spec: minimal API, frozenset-once dedupe,
    # positive-int validation.
    if save_at_steps is not None:
        for s in save_at_steps:
            if not isinstance(s, int) or s <= 0:
                raise ValueError(
                    f"save_at_steps entries must be positive ints; got {s!r}"
                )
        save_at_steps_set = frozenset(save_at_steps)
        print(f"[gsm8k] Slice 13l: save_at_steps ENABLED → "
              f"{sorted(save_at_steps_set)}")
    else:
        save_at_steps_set = frozenset()

    print(f"[gsm8k] loading splits via `datasets` lib...")
    full_train, full_val, test_rows = load_gsm8k_splits(val_frac=0.10)
    print(f"[gsm8k] splits: train={len(full_train)}  val={len(full_val)}  test={len(test_rows)}")

    # Vocab MUST be built from the full train+val so the OOV gate (and
    # checkpoint metadata) is locked at the canonical 98-token shape
    # regardless of smoke-test caps. Caps apply to training data only.
    print(f"[gsm8k] building tokenizer from full train+val (normalizer {NORMALIZER_VERSION})...")
    tok = Gsm8kTokenizer.from_corpus(full_train + full_val)
    print(f"[gsm8k] vocab: {tok.vocab_size} tokens")

    train_rows = full_train[:n_train_cap] if n_train_cap is not None else full_train
    val_rows = full_val[:n_val_cap] if n_val_cap is not None else full_val
    if n_train_cap is not None or n_val_cap is not None:
        print(f"[gsm8k] applied caps: train={len(train_rows)}/{len(full_train)}  "
              f"val={len(val_rows)}/{len(full_val)}")

    # Hard-fail at startup on OOV (codex's smallest-S0 gate, mirrored here).
    print(f"[gsm8k] OOV check on test split (must pass; declared vocab is "
          f"locked at train time)...")
    tok.assert_corpus_covered(test_rows, label="test")
    print(f"[gsm8k] OOV check PASS — test split covered by train+val vocab")

    train_ds = Gsm8kDataset(train_rows, tok, max_len=max_len)
    val_ds = Gsm8kDataset(val_rows, tok, max_len=max_len)
    print(f"[gsm8k] usable rows after max_len={max_len} drop: "
          f"train={len(train_ds)} (dropped {train_ds.n_dropped}) "
          f"val={len(val_ds)} (dropped {val_ds.n_dropped})")

    loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, drop_last=True,
        collate_fn=lambda b: collate(b, tok.pad_id, max_len,
                                      fixed_shape=fixed_shape_padding),
    )
    if fixed_shape_padding:
        print(f"[gsm8k] Slice 13e.1: fixed-shape padding ENABLED — every "
              f"batch padded to (B={batch_size}, max_len={max_len})")

    print(f"[gsm8k] building model (d_model={d_model}, layers={n_layers}, "
          f"vocab={tok.vocab_size})...")
    m = build_copy_augmented_delta(
        vocab_size=tok.vocab_size,
        d_model=d_model, n_heads=n_heads,
        n_layers=n_layers, d_ffn=d_ffn, max_len=max_len,
        n_copy_heads=n_copy_heads,
        sep_token_id=tok.sep_id,
        use_chunkwise=use_chunkwise,
        n_iterations=n_iterations,
        use_loop_index=use_loop_index,
        use_input_injection=use_input_injection,
        use_gated_attention=use_gated_attention,
        use_z_init=use_z_init,
        use_lecun_init=use_lecun_init,
        use_prefix_lm=use_prefix_lm,
        use_softmax_attn=use_softmax_attn,
        use_softmax_only=use_softmax_only,
        h_cycles=h_cycles,
        use_h_rmsnorm=use_h_rmsnorm,
        use_short_conv=use_short_conv,
        use_h_layer_stack=use_h_layer_stack,
        use_halt_head=use_halt_head,
        use_carry=use_carry,
        use_pre_rmsnorm=use_pre_rmsnorm,
        use_ternary_bulk=use_ternary_bulk,
    ).to(device)
    m.config.chunk_size = chunk_size
    m.max_len = max_len
    print(f"[gsm8k] params: {sum(p.numel() for p in m.parameters()):,}")
    print(f"[gsm8k] config: n_iter={m.config.n_iterations}  h_cycles={m.config.h_cycles}  "
          f"chunkwise={m.config.use_chunkwise}  layer_stack={m.config.use_h_layer_stack}")

    # Slice 13e.2: `capturable=True` enables CUDA graph capture of
    # opt.step's internal state updates. We don't actually capture opt.step
    # in this slice (it runs uncaptured after the finite tripwire fires),
    # but capturable=True is harmless without capture and ready for a
    # future slice that may want to include opt.step in the graph.
    opt = torch.optim.AdamW(
        m.parameters(), lr=lr, weight_decay=1e-4,
        capturable=graph_capture,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # Slice 13e.2: graph-capture setup. Static buffers + captured handle
    # initialized lazily on the first batch (so we can pin shapes from
    # actual data instead of assuming).
    if use_hrm_act:
        if not getattr(m.config, "use_halt_head", False):
            raise ValueError(
                "use_hrm_act=True requires use_halt_head=True — the HRM-Text "
                "training contract reads Q-values from the model's halt_head "
                "(reshaped to Linear(d, 2) per Slice 13f.2)."
            )
        if not getattr(m.config, "use_carry", False):
            raise ValueError(
                "use_hrm_act=True requires use_carry=True — HRM-Text segments "
                "share x; state z_H detached between segments. See "
                "RESEARCH/HRM/03_Training_Procedure.md:171-187."
            )
        if graph_capture:
            raise ValueError(
                "use_hrm_act=True + graph_capture=True not yet supported "
                "(captures one forward; HRM segment loop runs M_max forwards "
                "per training step). Separate slice."
            )
        print(f"[gsm8k] Slice 13f.2: HRM-Text per-segment training ENABLED "
              f"(M_max={m_max}, m_min_epsilon={m_min_epsilon})")

        # Slice 13i.1: continue-biased Q-head init. Patch halt_head.bias
        # so Q_continue > Q_halt at startup (sigmoid Qh~0.27, Qc~0.73).
        # Forces the model to LEARN to halt rather than learn-to-not-halt;
        # addresses the shallow-halt zero-attractor observed in 13f.3b
        # (all 5 saved epochs had avg_segs/tok=1.00, halt_hist concentrated
        # at bin 0). Per codex audit 1779432805671 + 1779432871832 gate.
        if q_init_bias_continue:
            with torch.no_grad():
                # halt_head is Linear(d_model, 2). bias is shape (2,).
                # Index 0 = Q_halt, Index 1 = Q_continue.
                m.halt_head.bias.zero_()
                m.halt_head.bias[0] = -1.0  # Q_halt sigmoid → ~0.27
                m.halt_head.bias[1] = +1.0  # Q_continue sigmoid → ~0.73
            print(f"[gsm8k] Slice 13i.1: continue-biased Q init INSTALLED "
                  f"(halt bias=[-1.0, +1.0] → sigmoid Qh~0.27, Qc~0.73)")

        # Slice 13i.1: deterministic deeper-M_min warmup curriculum.
        if m_min_warmup_epochs > 0:
            print(f"[gsm8k] Slice 13i.1: M_min warmup ENABLED "
                  f"(constant m_min={m_min_warmup_value} for first "
                  f"{m_min_warmup_epochs} epochs, then anneal to "
                  f"epsilon-stochastic at epsilon={m_min_epsilon})")

    if graph_capture:
        if not fixed_shape_padding:
            raise ValueError(
                "graph_capture=True requires fixed_shape_padding=True — "
                "variable per-batch shapes preclude CUDA graph capture"
            )
        if aux_weight <= 0.0:
            raise ValueError(
                "graph_capture=True requires aux_weight > 0.0 — Slice 13e.2 "
                "captures the deep-supervision branch only; final-NLL-only "
                "capture is a separate later slice"
            )
        print(f"[gsm8k] Slice 13e.2: graph-capture training step ENABLED — "
              f"first batch will trigger capture")
        _graph_ids_buf = torch.zeros(batch_size, max_len, dtype=torch.long,
                                      device=device)
        _graph_mask_buf = torch.zeros(batch_size, max_len, dtype=torch.bool,
                                       device=device)
        _captured_g = None      # filled on first batch
        _captured_outs = None   # dict of output tensors: loss, final_nll, aux_nll, per_iter_nlls

    best_acc = -1.0
    best_ep = -1
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    last_finite_loss = float("nan")
    last_grad_norm = float("nan")
    for ep in range(1, epochs + 1):
        m.train()
        t0 = time.time()
        total_loss = 0.0
        n_batches = 0
        for ids, mask, _sep in loader:
            ids = ids.to(device)
            mask = mask.to(device)

            # Slice 13f.2: HRM-Text per-segment training contract.
            # Source: RESEARCH/HRM/03_Training_Procedure.md:171-255.
            # Active mask per example (NOT batch-shared per codex audit
            # msg 1779386734917-9df1f5d5). Per-segment backward+opt.step
            # with detached carry. NO-grad lookahead for non-terminal Q
            # target (avoids autograd-vs-opt.step version-error trap).
            if use_hrm_act:
                from calm.llm_computer.copy_augmented_delta import (
                    full_answer_correct, compute_hrm_segment_loss,
                    hrm_boundary_q_continue_target,
                )
                B = ids.shape[0]
                active = torch.ones(B, dtype=torch.bool, device=device)
                carry = None
                # M_min: Slice 13i.1 warmup curriculum + HRM-Text §5:234-236.
                # During warmup epochs, force constant deeper m_min to give
                # the recurrence + Q-head dense training signal at depth.
                # After warmup, anneal back to source-faithful stochastic draw.
                if m_min_warmup_epochs > 0 and ep <= m_min_warmup_epochs:
                    m_min = m_min_warmup_value
                elif torch.rand(1).item() < m_min_epsilon:
                    m_min = int(torch.randint(2, m_max + 1, (1,)).item())
                else:
                    m_min = 1
                step_idx = n_batches + 1
                seg_losses = []
                seg_count = 0
                for seg_m in range(m_max):
                    seg = seg_m + 1  # 1-based per HRM source rule
                    if not active.any():
                        break
                    seg_count = seg_m + 1
                    # Grad-tracked forward
                    out = m(ids, return_carry=True, carry=carry)
                    if isinstance(out, tuple):
                        log_probs_m, carry_m = out
                    else:
                        log_probs_m, carry_m = out, None
                    q_pair = m.last_q_pair
                    if q_pair is None:
                        raise RuntimeError(
                            "use_hrm_act=True but last_q_pair is None — "
                            "model not built with use_halt_head=True"
                        )
                    reward_m = full_answer_correct(log_probs_m, ids, mask)

                    # Per-example halt decision
                    q_halt = q_pair[..., 0]
                    q_continue = q_pair[..., 1]
                    halt_decision = (q_halt > q_continue) & (seg >= m_min)
                    forced_halt = (seg == m_max)
                    terminal = active & (halt_decision | torch.full_like(
                        halt_decision, forced_halt))
                    needs_continue = active & ~terminal

                    # Lookahead Q-target ONLY if any active example continues.
                    # Boundary rule lives in hrm_boundary_q_continue_target
                    # (HRM-Text §5:248-250): at seg+1==m_max the lookahead
                    # segment is itself forced-halt, so its Q_continue is
                    # illegal — bootstrap from Q_halt only.
                    if needs_continue.any() and seg < m_max:
                        with torch.no_grad():
                            out_next = m(ids, return_carry=True,
                                          carry=carry_m.detach() if carry_m is not None else None)
                            if isinstance(out_next, tuple):
                                _, _ = out_next
                            q_next = m.last_q_pair  # (B, 2)
                        g_continue_lookahead = hrm_boundary_q_continue_target(
                            q_next, seg, m_max)
                    else:
                        g_continue_lookahead = torch.zeros_like(q_halt)

                    g_halt = reward_m.float()
                    g_continue = torch.where(terminal,
                                              torch.zeros_like(g_continue_lookahead),
                                              g_continue_lookahead)

                    total_seg, nll_seg, bce_seg = compute_hrm_segment_loss(
                        log_probs_m, q_pair, g_halt, g_continue, ids, mask, active,
                    )

                    # Finite tripwire (preserved per codex guardrail 2)
                    if not torch.isfinite(total_seg).all():
                        print(f"[NaN-DETECT] ep={ep} step={step_idx} seg={seg} "
                              f"nll={float(nll_seg.detach())} bce={float(bce_seg.detach())} "
                              f"NON-FINITE TOTAL — STOP", flush=True)
                        sys.exit(2)

                    opt.zero_grad()
                    total_seg.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        m.parameters(), max_norm=1.0)
                    if not math.isfinite(float(grad_norm)):
                        print(f"[NaN-DETECT] ep={ep} step={step_idx} seg={seg} "
                              f"grad_norm={float(grad_norm)} NON-FINITE — STOP",
                              flush=True)
                        sys.exit(2)
                    opt.step()

                    # Per-segment telemetry over ACTIVE rows only (the
                    # only rows whose Q-pair contributed gradient this seg).
                    # Slice 13f.3 instrumentation: halt-decision histogram
                    # (raw q_halt > q_continue, ungated by m_min so we see
                    # the head's signal directly) + Q mean for both axes.
                    active_f = active.float()
                    denom = active_f.sum().clamp_min(1.0)
                    q_halt_mean = float((q_halt.detach() * active_f).sum() / denom)
                    q_continue_mean = float((q_continue.detach() * active_f).sum() / denom)
                    raw_halt = ((q_halt > q_continue) & active).sum().item()
                    seg_losses.append((float(total_seg.detach()),       # 0 total
                                        float(nll_seg.detach()),         # 1 nll
                                        float(bce_seg.detach()),         # 2 bce
                                        int(active.sum().item()),        # 3 active
                                        int((reward_m & active).sum().item()),  # 4 reward
                                        q_halt_mean,                     # 5 Qh mean
                                        q_continue_mean,                 # 6 Qc mean
                                        int(raw_halt)))                  # 7 raw halt

                    # Drop terminal examples for next segment
                    active = active & ~terminal
                    carry = carry_m.detach() if carry_m is not None else None

                # Aggregate logging
                avg_seg_total = sum(s[0] for s in seg_losses) / len(seg_losses)
                total_loss += avg_seg_total
                n_batches += 1
                last_finite_loss = avg_seg_total
                last_grad_norm = float(grad_norm)

                if log_every > 0 and (step_idx == 1 or step_idx % log_every == 0):
                    nll_str = ",".join(f"{s[1]:.3f}" for s in seg_losses)
                    bce_str = ",".join(f"{s[2]:.3f}" for s in seg_losses)
                    rew_str = ",".join(f"{s[4]}/{s[3]}" for s in seg_losses)
                    qh_str = ",".join(f"{s[5]:.3f}" for s in seg_losses)
                    qc_str = ",".join(f"{s[6]:.3f}" for s in seg_losses)
                    halt_str = ",".join(f"{s[7]}/{s[3]}" for s in seg_losses)
                    print(f"[ep {ep:3d} step {step_idx:5d}] HRM seg_count={seg_count}/{m_max} "
                          f"m_min={m_min} loss_avg={avg_seg_total:.4f} "
                          f"per_seg_nll=[{nll_str}] per_seg_bce=[{bce_str}] "
                          f"per_seg_reward=[{rew_str}] "
                          f"per_seg_Qh=[{qh_str}] per_seg_Qc=[{qc_str}] "
                          f"per_seg_raw_halt=[{halt_str}] "
                          f"grad_norm={last_grad_norm:.4f}",
                          flush=True)
                # Slice 13l: --save-at-step N mid-training ckpt hook (repeatable).
                # Closes the gap that bit 13j/13k where slope-based aborts
                # left no usable ckpt for the forced-depth probe gate.
                # Saves a per-step ckpt to `<stem>_step{N:05d}.pt` then
                # continues training (does NOT terminate; allows the run
                # to either continue or be killed externally).
                if step_idx in save_at_steps_set:
                    step_ckpt_path = Path(checkpoint_path).with_name(
                        Path(checkpoint_path).stem + f"_step{step_idx:05d}.pt"
                    )
                    step_ckpt_blob = {
                        "model_state": m.state_dict(),
                        "config": _build_ckpt_config(
                            m=m, tok=tok, max_len=max_len, d_model=d_model,
                            n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn,
                            n_copy_heads=n_copy_heads, aux_weight=aux_weight,
                        ),
                        "epoch": ep,
                        "step": step_idx,
                        "val_acc": float("nan"),
                        "n_train": len(train_ds),
                        "n_val": len(val_ds),
                    }
                    torch.save(step_ckpt_blob, step_ckpt_path)
                    print(f"[ep {ep:3d} step {step_idx:5d}] "
                          f"Slice 13l save_at_step: saved {step_ckpt_path}",
                          flush=True)
                continue  # skip the standard graph_capture/aux_weight branches below

            # Slice 13e.2: graph-capture training-step branch.
            # Captured region: opt.zero_grad + forward + loss + backward +
            # grad-clip. opt.step REMAINS UNCAPTURED so the finite tripwire
            # can fire BEFORE the optimizer poisons weights. First batch
            # warms up + captures; subsequent batches just copy-buf + replay.
            if graph_capture:
                _graph_ids_buf.copy_(ids)
                _graph_mask_buf.copy_(mask)
                if _captured_g is None:
                    # First batch: side-stream warmup with FULL training step
                    # (forward+backward+clip+opt.step × 3) to initialize Adam
                    # state on the side stream. Without this, opt.zero_grad
                    # inside the captured region fails because Adam state
                    # isn't allocated. Cost: batch 0 effectively gets 4 extra
                    # opt.steps before batch 1 sees the model — negligible at
                    # 30-epoch × ~800-step run (0.017% drift).
                    _s = torch.cuda.Stream()
                    _s.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(_s):
                        for _ in range(3):
                            opt.zero_grad()
                            _fp, _pi = m(_graph_ids_buf, return_per_iter=True)
                            _fn = _masked_shifted_nll(_fp, _graph_ids_buf, _graph_mask_buf)
                            _pn = [_masked_shifted_nll(lp, _graph_ids_buf, _graph_mask_buf) for lp in _pi]
                            _an = sum(_pn) / len(_pn)
                            _ll = _fn + aux_weight * _an
                            _ll.backward()
                            torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                            opt.step()  # init Adam state on side stream
                            del _ll, _fn, _pn, _an, _fp, _pi
                    torch.cuda.current_stream().wait_stream(_s)
                    torch.cuda.synchronize()
                    # Capture forward+backward+clip (NO opt.step — finite-
                    # tripwire and opt.step stay uncaptured per codex
                    # guardrail 2 in msg 1779384319796-f0f31547)
                    _captured_g = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(_captured_g):
                        opt.zero_grad()
                        _fp, _pi = m(_graph_ids_buf, return_per_iter=True)
                        _fn = _masked_shifted_nll(_fp, _graph_ids_buf, _graph_mask_buf)
                        _pn = [_masked_shifted_nll(lp, _graph_ids_buf, _graph_mask_buf) for lp in _pi]
                        _an = sum(_pn) / len(_pn)
                        _ll = _fn + aux_weight * _an
                        _ll.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                    _captured_outs = {
                        'loss': _ll, 'final_nll': _fn, 'aux_nll': _an,
                        'per_iter_nlls': _pn,
                    }
                    print(f"[gsm8k] Slice 13e.2: training-step graph captured "
                          f"(forward+loss+backward+grad-clip; opt.step + "
                          f"finite-tripwire stay uncaptured)")
                else:
                    _captured_g.replay()
                # Expose captured outputs to the rest of the loop as if from
                # a normal forward — finite-check + opt.step run BELOW.
                final_log_probs = None  # not needed downstream when captured
                per_iter_list = None
                loss = _captured_outs['loss']
                final_nll = _captured_outs['final_nll']
                aux_nll = _captured_outs['aux_nll']
                per_iter_nlls = _captured_outs['per_iter_nlls']
            elif aux_weight > 0.0:
                # Deep supervision (S11 / S10b seam, per codex audit
                # `1779314284912` / `1779314708107`): forward returns
                # `(final_log_probs, per_iter_log_probs_list)`. Total loss
                # = final_NLL + aux_weight * mean(per_iter_NLL).
                #
                # per_iter[-1] equals final at h_cycles>=1 (see
                # test_rdt_v2_slice11.py:130-142), so the final term is
                # deliberately double-weighted (1 + aux_weight/k) on the
                # last cycle. Documented choice — keeps the mean simple
                # and matches the Slice 11 deep-supervision tests at
                # :164-200.
                final_log_probs, per_iter_list = m(ids, return_per_iter=True)
                final_nll = _masked_shifted_nll(final_log_probs, ids, mask)
                per_iter_nlls = [
                    _masked_shifted_nll(lp, ids, mask) for lp in per_iter_list
                ]
                aux_nll = sum(per_iter_nlls) / len(per_iter_nlls)
                loss = final_nll + aux_weight * aux_nll
            else:
                # Final-NLL only (baseline, bit-equivalent to pre-S0c-aux
                # path when called via _masked_shifted_nll on the same
                # inputs).
                final_nll = _masked_shifted_nll(m(ids), ids, mask)
                loss = final_nll
                per_iter_nlls = None
                aux_nll = None

            # Finite-tripwire (NaN-diagnostic per codex audit
            # `1779353017204-69c791ba`). Check ALL loss components BEFORE
            # backward — NaN-poisoned grads corrupt weights silently.
            # Exit nonzero with full component readout on first violation;
            # do NOT save or eval a poisoned model.
            step_idx = n_batches + 1
            loss_item = float(loss.detach())
            finite_loss = math.isfinite(loss_item)
            final_item = float(final_nll.detach())
            finite_final = math.isfinite(final_item)
            if per_iter_nlls is not None:
                pi_items = [float(p.detach()) for p in per_iter_nlls]
                aux_item = float(aux_nll.detach())
                finite_aux = math.isfinite(aux_item)
                finite_pi = all(math.isfinite(p) for p in pi_items)
            else:
                pi_items = None
                aux_item = None
                finite_aux = True
                finite_pi = True

            if not (finite_loss and finite_final and finite_aux and finite_pi):
                print(f"[NaN-DETECT] ep={ep} step={step_idx} loss={loss_item} "
                      f"finite_loss={finite_loss}",
                      flush=True)
                print(f"[NaN-DETECT] final_nll={final_item} "
                      f"finite_final={finite_final}",
                      flush=True)
                if pi_items is not None:
                    print(f"[NaN-DETECT] aux_nll={aux_item} "
                          f"finite_aux={finite_aux} per_iter={pi_items} "
                          f"finite_pi={finite_pi}",
                          flush=True)
                print(f"[NaN-DETECT] last_finite_loss={last_finite_loss} "
                      f"last_grad_norm={last_grad_norm}",
                      flush=True)
                sys.exit(2)

            last_finite_loss = loss_item
            if graph_capture:
                # Slice 13e.2: backward + grad-clip already ran inside the
                # captured region. Compute grad_norm from p.grad for the
                # tripwire below (cheap GPU reduction; not in capture).
                last_grad_norm = float(
                    torch.sqrt(sum(
                        p.grad.detach().pow(2).sum()
                        for p in m.parameters() if p.grad is not None
                    ))
                )
            else:
                opt.zero_grad()
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(m.parameters(), max_norm=1.0)
                last_grad_norm = float(grad_norm)

            # Grad-norm finite gate (per codex audit `1779353275232-793dc65e`).
            # clip_grad_norm_ divides but does NOT filter — NaN/inf grads
            # survive clipping and poison weights via opt.step(). Refuse step
            # on non-finite grad norm; same diagnostic readout as loss tripwire.
            if not math.isfinite(last_grad_norm):
                print(f"[NaN-DETECT] ep={ep} step={step_idx} "
                      f"grad_norm={last_grad_norm} "
                      f"loss={loss_item}",
                      flush=True)
                print(f"[NaN-DETECT] final_nll={final_item} "
                      f"finite_final={finite_final}",
                      flush=True)
                if pi_items is not None:
                    print(f"[NaN-DETECT] aux_nll={aux_item} "
                          f"per_iter={pi_items}",
                          flush=True)
                print(f"[NaN-DETECT] last_finite_loss={last_finite_loss} "
                      f"(grad explosion: loss was finite but grads diverged)",
                      flush=True)
                sys.exit(2)

            opt.step()
            total_loss += loss_item
            n_batches += 1

            # Interior log: step 1 AND every log_every batches. Decomposes
            # loss into final_nll + aux_nll + per-iter NLLs so divergence
            # source is visible per step.
            if log_every > 0 and (step_idx == 1 or step_idx % log_every == 0):
                if pi_items is not None:
                    pi_str = ",".join(f"{p:.4f}" for p in pi_items)
                    print(f"[ep {ep:3d} step {step_idx:5d}] "
                          f"loss={loss_item:.4f}  final_nll={final_item:.4f}  "
                          f"aux_nll={aux_item:.4f}  per_iter=[{pi_str}]  "
                          f"grad_norm={last_grad_norm:.4f}",
                          flush=True)
                else:
                    print(f"[ep {ep:3d} step {step_idx:5d}] "
                          f"loss={loss_item:.4f}  grad_norm={last_grad_norm:.4f}",
                          flush=True)
        sched.step()
        avg_loss = total_loss / max(n_batches, 1)
        epoch_secs = time.time() - t0
        print(f"[ep {ep:3d}] loss={avg_loss:.4f}  lr={sched.get_last_lr()[0]:.2e}  "
              f"time={epoch_secs:.1f}s")

        if ep % eval_every == 0 or ep == epochs:
            # Slice 13h: route to source-faithful HRM eval when the model
            # was trained with carry + halt-head. Otherwise legacy decoder.
            use_hrm_eval = bool(
                getattr(m.config, "use_carry", False)
                and getattr(m.config, "use_halt_head", False)
            )
            if use_hrm_eval:
                er = autoreg_eval_hrm(
                    m, tok, val_rows, cap=eval_cap, device=device,
                    m_max=m_max if use_hrm_act else 4,
                )
                acc = er["acc_parsed"]   # gate on parsed (legacy-compatible)
                n_c = er["n_correct_parsed"]
                n_e = er["n_evaluated"]
                halt_str = ",".join(str(h) for h in er["halt_histogram"])
                print(
                    f"[ep {ep:3d}] val_acc_parsed={er['acc_parsed']:.3f} "
                    f"({er['n_correct_parsed']}/{n_e})  "
                    f"val_acc_exact={er['acc_exact']:.3f} "
                    f"({er['n_correct_exact']}/{n_e})  "
                    f"avg_segs/tok={er['avg_segs_per_token']:.2f}  "
                    f"halt_hist=[{halt_str}]  "
                    f"Qh={er['qh_mean']:.3f}  Qc={er['qc_mean']:.3f}"
                )
            else:
                acc, n_c, n_e = autoreg_eval(
                    m, tok, val_rows, cap=eval_cap, device=device)
                print(f"[ep {ep:3d}] val_acc={acc:.3f} ({n_c}/{n_e})")

            # Slice 13h checkpoint hygiene: build the ckpt blob once and
            # save it unconditionally as `<name>_last.pt` EVERY epoch
            # (so post-collapse epochs aren't silently dropped); also
            # save as `<name>.pt` (best) only when val_acc beats best.
            ckpt_blob = {
                "model_state": m.state_dict(),
                "config": _build_ckpt_config(
                    m=m, tok=tok, max_len=max_len, d_model=d_model,
                    n_heads=n_heads, n_layers=n_layers, d_ffn=d_ffn,
                    n_copy_heads=n_copy_heads, aux_weight=aux_weight,
                ),
                "epoch": ep,
                "val_acc": acc,
                "n_train": len(train_ds),
                "n_val": len(val_ds),
            }

            # Per-epoch ckpt: `<stem>_ep{N}.pt`. Each epoch a distinct file
            # so 13i.0-style cross-checkpoint diagnostics have stable inputs.
            # (Slice 13h originally saved single overwritten `_last.pt`;
            # codex audit 1779391827108 + 13f.3b workaround revealed the
            # gap. This is the proper fix for future runs.)
            ep_path = Path(checkpoint_path).with_name(
                Path(checkpoint_path).stem + f"_ep{ep:03d}.pt"
            )
            torch.save(ckpt_blob, ep_path)
            # Also save a `_last.pt` symlink-equivalent (overwritten each
            # epoch) for backward-compat with downstream tooling that
            # expects a single "current" pointer.
            last_path = Path(checkpoint_path).with_name(
                Path(checkpoint_path).stem + "_last.pt"
            )
            torch.save(ckpt_blob, last_path)

            # best.pt: gate on (loss finite AND acc finite AND acc > best).
            # Co-lead audit `1779353017204-69c791ba`: `acc > best_acc` lets
            # `acc=0.0` save when `best_acc=-1`, so guard requires clean metrics.
            if (
                math.isfinite(avg_loss)
                and math.isfinite(acc)
                and acc > best_acc
            ):
                best_acc = acc
                best_ep = ep
                torch.save(ckpt_blob, checkpoint_path)
                print(f"[ep {ep:3d}] saved best to {checkpoint_path}")

    print(f"\nBest: epoch {best_ep}  val_acc={best_acc:.3f}")
    print(f"Saved: {checkpoint_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--n-layers", type=int, default=4)
    ap.add_argument("--d-ffn", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--n-copy-heads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval-every", type=int, default=1)
    ap.add_argument("--eval-cap", type=int, default=100)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--checkpoint-path", type=str, default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--n-train-cap", type=int, default=None,
                    help="Cap training rows (smoke test). None = full set.")
    ap.add_argument("--n-val-cap", type=int, default=None)
    # rdt-v2 Tier A+B flag bundle. Defaults preserve baseline behavior.
    ap.add_argument("--no-chunkwise", dest="use_chunkwise",
                    action="store_false", default=True)
    ap.add_argument("--n-iterations", type=int, default=1)
    ap.add_argument("--use-loop-index", action="store_true")
    ap.add_argument("--use-input-injection", action="store_true")
    ap.add_argument("--use-gated-attention", action="store_true")
    ap.add_argument("--use-z-init", action="store_true")
    ap.add_argument("--use-lecun-init", action="store_true")
    ap.add_argument("--use-prefix-lm", action="store_true")
    ap.add_argument("--use-softmax-attn", action="store_true",
                    help="Slice 13j hybrid: enable softmax attention path "
                         "alongside DeltaNet (parallel residual add at "
                         "delta_rule.py:1384). Required for --use-prefix-lm "
                         "to have any effect on the active code path. "
                         "MUTEX with --use-softmax-only (different arms).")
    ap.add_argument("--use-softmax-only", action="store_true",
                    help="Slice 13k: softmax-ONLY mixer mode. Skips the "
                         "DeltaNet recurrence entirely (no compute, no "
                         "memory). Auto-forces --use-softmax-attn ON so the "
                         "softmax path actually runs. H/L stack + carry + "
                         "halt_head outer-loop semantics preserved. MUTEX "
                         "with --use-softmax-attn alone (hybrid arm).")
    ap.add_argument("--h-cycles", type=int, default=1)
    ap.add_argument("--use-h-rmsnorm", action="store_true")
    ap.add_argument("--use-short-conv", action="store_true")
    ap.add_argument("--use-h-layer-stack", action="store_true")
    ap.add_argument("--use-halt-head", action="store_true")
    ap.add_argument("--use-carry", action="store_true")
    ap.add_argument("--chunk-size", type=int, default=32)
    # Slice 12: per-layer Pre-RMSNorm flag (L-stack residual stability fix).
    ap.add_argument("--use-ternary-bulk", action="store_true",
                    help="TRM-1.58 (Slice 13): native W1.58A8 ternary forward "
                         "weights for bulk projections + STE backward")
    ap.add_argument("--fixed-shape-padding", action="store_true",
                    help="TRM-1.58 throughput-to-signal Slice 13e.1: pad "
                         "every batch to fixed --max-len instead of per-batch "
                         "max_L. Required for CUDA graph capture of the "
                         "training step (Slice 13e.2).")
    ap.add_argument("--graph-capture", action="store_true",
                    help="TRM-1.58 throughput-to-signal Slice 13e.2: capture "
                         "forward+loss+backward+grad-clip as a CUDA graph; "
                         "replay per step. opt.step + finite-tripwire stay "
                         "uncaptured for training-semantics preservation. "
                         "Requires --fixed-shape-padding + aux-weight > 0.")
    ap.add_argument("--use-hrm-act", action="store_true",
                    help="TRM-1.58 Slice 13f.2: source-faithful HRM-Text/Sapient "
                         "ACT training contract. M_max-segment outer loop with "
                         "per-segment NLL + BCE(Q̂, Ĝ) loss, detached carry "
                         "between segments. Requires --use-halt-head + --use-carry. "
                         "Incompatible with --graph-capture (future slice).")
    ap.add_argument("--m-max", type=int, default=4,
                    help="HRM-Text M_max — segments per training step "
                         "(default 4). HRM paper uses 8.")
    ap.add_argument("--m-min-epsilon", type=float, default=0.1,
                    help="HRM-Text §5:234-236 exploration probability: with "
                         "probability epsilon, M_min is sampled uniform from "
                         "{2..M_max}; otherwise M_min=1.")
    # Slice 13i.1: deterministic deeper-M_min warmup curriculum + Q-init bias.
    ap.add_argument("--m-min-warmup-epochs", type=int, default=0,
                    help="Slice 13i.1: deterministic deeper-M_min warmup. "
                         "For the first N epochs, force constant m_min="
                         "--m-min-warmup-value on every batch (override "
                         "stochastic epsilon draw). After warmup, anneal "
                         "back to epsilon-stochastic source-faithful schedule. "
                         "Default 0 = no warmup, preserves current behavior.")
    ap.add_argument("--m-min-warmup-value", type=int, default=4,
                    help="Slice 13i.1: m_min value used during warmup epochs "
                         "(default 4 = M_max; forces all batches through full "
                         "M_max segments during warmup).")
    ap.add_argument("--q-init-bias-continue", action="store_true",
                    help="Slice 13i.1: continue-biased Q-head init. After "
                         "model build, patches halt_head.bias to [-1.0, +1.0] "
                         "(sigmoid Qh~0.27, Qc~0.73). Inverts default policy "
                         "from 'halt at seg 1' to 'continue to M_max'; addresses "
                         "shallow-halt zero-attractor observed in 13f.3b.")
    ap.add_argument("--save-at-step", type=int, action="append", default=None,
                    help="Slice 13l: mid-training ckpt save hook (repeatable). "
                         "Pass multiple times (e.g. `--save-at-step 100 "
                         "--save-at-step 200`) to save at multiple step indices "
                         "in one trajectory. Saves to `<stem>_step{N:05d}.pt` "
                         "when step_idx is in the supplied set. Closes the gap "
                         "that bit 13j/13k where slope-based aborts left no "
                         "usable ckpt for forced-depth probe gates. HRM "
                         "segment-loop path only; defaults to None (off). "
                         "Positive ints only; duplicates deduped via frozenset.")
    ap.add_argument("--use-pre-rmsnorm", action="store_true",
                    help="Allocate per-layer RMSNorm before sequence-mixer "
                         "AND before FFN in both L and H banks. Fixes S2 NaN "
                         "root cause (residual magnitude blow-up). Default OFF "
                         "preserves Slice 1-11 bit-equivalence.")
    # S0c-aux: deep-supervision aux loss (per codex audit `1779314708107`).
    ap.add_argument("--aux-weight", type=float, default=0.0,
                    help="Weight on mean(per_iter_NLL) added to final_NLL. "
                         "0.0 = final-only baseline (bit-equivalent). "
                         ">0 enables `return_per_iter=True` deep supervision.")
    # NaN-diagnostic interior logging (per codex audit `1779353017204-69c791ba`).
    ap.add_argument("--log-every", type=int, default=0,
                    help="Print interior loss/grad components every N batches "
                         "(step 1 always logged when >0). Triggers "
                         "finite-tripwire on every step regardless of value. "
                         "0 = disabled (preserves prior log shape).")
    args = ap.parse_args()
    # Slice 13k mutex per codex msg 1779442478419: --use-softmax-only and
    # --use-softmax-attn are different comparator arms. Hybrid runs softmax
    # in parallel with DeltaNet; softmax-only skips DeltaNet entirely.
    # Silently accepting both poisons later provenance.
    # (softmax-only DOES auto-set softmax_attn=True internally, but here
    # we check what the USER supplied at the CLI.)
    if args.use_softmax_only and args.use_softmax_attn:
        raise ValueError(
            "--use-softmax-only and --use-softmax-attn are MUTEX comparator "
            "arms. Hybrid (softmax-attn alone, parallel with DeltaNet) and "
            "softmax-only (DeltaNet skipped) are different research arms; "
            "do not enable both. softmax-only auto-forces softmax_attn=True "
            "internally, so you do not need to pass both."
        )
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        d_model=args.d_model, n_heads=args.n_heads,
        n_layers=args.n_layers, d_ffn=args.d_ffn,
        max_len=args.max_len,
        n_copy_heads=args.n_copy_heads,
        seed=args.seed,
        eval_every=args.eval_every,
        eval_cap=args.eval_cap,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        n_train_cap=args.n_train_cap,
        n_val_cap=args.n_val_cap,
        use_chunkwise=args.use_chunkwise,
        n_iterations=args.n_iterations,
        use_loop_index=args.use_loop_index,
        use_input_injection=args.use_input_injection,
        use_gated_attention=args.use_gated_attention,
        use_z_init=args.use_z_init,
        use_lecun_init=args.use_lecun_init,
        use_prefix_lm=args.use_prefix_lm,
        use_softmax_attn=args.use_softmax_attn,
        use_softmax_only=args.use_softmax_only,
        h_cycles=args.h_cycles,
        use_h_rmsnorm=args.use_h_rmsnorm,
        use_short_conv=args.use_short_conv,
        use_h_layer_stack=args.use_h_layer_stack,
        use_halt_head=args.use_halt_head,
        use_carry=args.use_carry,
        use_pre_rmsnorm=args.use_pre_rmsnorm,
        use_ternary_bulk=args.use_ternary_bulk,
        fixed_shape_padding=args.fixed_shape_padding,
        graph_capture=args.graph_capture,
        use_hrm_act=args.use_hrm_act,
        m_max=args.m_max,
        m_min_epsilon=args.m_min_epsilon,
        m_min_warmup_epochs=args.m_min_warmup_epochs,
        m_min_warmup_value=args.m_min_warmup_value,
        q_init_bias_continue=args.q_init_bias_continue,
        save_at_steps=args.save_at_step,
        chunk_size=args.chunk_size,
        aux_weight=args.aux_weight,
        log_every=args.log_every,
    )
