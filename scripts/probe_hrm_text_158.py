"""HRM-Text-1.58 Phase 1 Slice 2 — probe.

Cap=50 GSM8k val eval + canonical 17×23=391 row.

Per task #51 + codex msg 1779452208756 (Phase 1 Slice 2 +1 implement):
- D1.9 deferred: full-prefix re-forward greedy decoding (no KV cache).
  Acceptable for probe-speed per codex residual risk note.
- 17×23 prompt is the GSM8k-text form ("what is 17 times 23?") to
  stay in tokenizer vocab; result REPORTED (not asserted) per
  codex correction 1: "If step200 decodes wrong, that is a receipt
  and a binary next-decision, not an automatic implementation failure."
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch

from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer
from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID

# Use HRM-Text-1.58 trainer's inlined neutral loader (no RDT/Delta import).
# Phase 1 guardrail: no `scripts.train_dt_gsm8k` import in HRM-Text-1.58 path.
from scripts.train_hrm_text_158 import load_gsm8k_splits

# Phase 3 curriculum imports (codex msg 1779462307554 Phase A +1)
from calm.hrm_text_158.curriculum import (
    BROAD_NORMALIZER_VERSION,
    BroadTokenizer,
    RUNG_NAMES,
    RungProbeResult,
    make_rung_examples,
    r1b4v2_one_digit_audit_rows,
    r1b5_one_digit_audit_rows,
)

# Per-rung audit registry (codex msg 1779523412979-ff88b885).
# Each entry maps rung name -> callable(seed) -> list of 9 audit rows.
# When `probe_curriculum` runs, it iterates this registry against the
# `rungs` argument so every audit-eligible rung present gets a keyed
# audit in result.one_digit_audits (no silent retention drops).
ONE_DIGIT_AUDIT_REGISTRY = {
    "R1b4v2": r1b4v2_one_digit_audit_rows,
    "R1b5": r1b5_one_digit_audit_rows,
}


def _build_model_from_ckpt(ckpt: dict, device: str) -> tuple[LMHead, object]:
    """Reconstruct model + tokenizer from ckpt blob.

    Auto-detects tokenizer type from `gsm8k_normalizer_version`:
    - `byte_utf8_v1` -> BroadTokenizer (Phase 3 curriculum)
    - anything else  -> Gsm8kTokenizer (Phase 1/2 GSM8k path)
    """
    config = ckpt["config"]
    normalizer_version = config["gsm8k_normalizer_version"]
    if normalizer_version == BROAD_NORMALIZER_VERSION:
        tok = BroadTokenizer()
        # Sanity: ckpt vocab must equal BroadTokenizer's deterministic vocab
        if list(config["gsm8k_char_vocab"]) != tok.vocab_as_list():
            raise ValueError(
                f"BroadTokenizer ckpt vocab mismatch: ckpt has "
                f"{len(config['gsm8k_char_vocab'])} entries; "
                f"BroadTokenizer constant vocab has {tok.vocab_size}"
            )
    else:
        tok = Gsm8kTokenizer.from_metadata(
            vocab_list=config["gsm8k_char_vocab"],
            normalizer_version=normalizer_version,
        )
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=config["max_seq_len"],
        n_layers=config["n_layers"],
        hidden_size=config["hidden_size"],
        num_heads=config["num_heads"],
        expansion=config["expansion"],
        H_cycles=config["H_cycles"],
        L_cycles=config["L_cycles"],
        half_layers=config["half_layers"],
        bp_warmup_ratio=config["bp_warmup_ratio"],
        bp_min_steps=config["bp_min_steps"],
        bp_max_steps=config["bp_max_steps"],
        norm_type=config["norm_type"],
        norm_eps=config["norm_eps"],
        rope_theta=config["rope_theta"],
        attn_type=config["attn_type"],
        init_type=config["init_type"],
        pos_emb_type=config["pos_emb_type"],
        # CRITICAL per codex msg 1779457628632: load ternary flag from ckpt
        # config blob. Without this, ternary ckpts reconstruct as FP models
        # (state_dict keys match because BitLinear and LinearInit both use
        # `weight`/`bias`), but inference runs FP LinearInit over master
        # weights -- silently wrong probe results, false A/B.
        use_ternary_bulk=config.get("use_ternary_bulk", False),
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=config["vocab_size"])).to(device)
    load_result = m.load_state_dict(ckpt["model_state"], strict=True)
    print(f"[probe] state_dict loaded: missing={len(load_result.missing_keys)} unexpected={len(load_result.unexpected_keys)}", flush=True)
    m.eval()
    return m, tok


def _decode_greedy_no_cache(
    m: LMHead,
    tok,
    question: str,
    *,
    max_gen: int = 8,
    max_seq_len: int,
    device: str,
) -> tuple[str, bool, bool]:
    """Greedy decode by re-forwarding the full prefix at each step.

    Per D1.9 deferral: no KV cache. Each step adds 1 token, re-runs
    full forward. Quadratic in generated length but acceptable for
    short probe answers.

    Returns (decoded_string, too_long_flag, finite_flag).
    - too_long: True iff prefix `<bos> question <sep>` exceeds max_seq_len.
    - finite: True iff ALL logits at every decode step are finite. Per
      codex msg 1779463196431 rule 4 — catches NaN/Inf propagation in
      logits rather than the prior no-op string-comparison sentinel.
    """
    # Build prefix: <bos> question <sep>
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    if len(prefix) >= max_seq_len:
        # Cannot decode any new token (no room past sep). Row is unevaluable.
        return "", True, True
    sep_pos = 1 + len(q_ids)
    out_tokens: list[int] = []
    cur = list(prefix)
    finite = True
    for _ in range(max_gen):
        if len(cur) >= max_seq_len:
            break
        ids = torch.tensor([cur], dtype=torch.long, device=device)
        sep_pos_t = torch.tensor([sep_pos], dtype=torch.long, device=device)
        pos = torch.arange(ids.shape[1], dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad():
            new_carry, logits = m(
                None,
                {"inputs": ids, "sep_positions": sep_pos_t, "position_ids": pos},
            )
        # Finite-check across the FULL logits tensor (not just last-pos):
        # catches NaN/Inf anywhere in the head output, which would indicate
        # ternary-bulk or recurrence divergence.
        if not bool(torch.isfinite(logits).all().item()):
            finite = False
            break
        # Greedy argmax at last position
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id == tok.eos_id:
            break
        out_tokens.append(next_id)
        cur.append(next_id)
    return tok.decode(out_tokens, stop_at_eos=False), False, finite


def _decode_greedy_cached(
    m: LMHead,
    tok,
    question: str,
    *,
    max_gen: int = 8,
    max_seq_len: int,
    device: str,
) -> tuple[str, bool, bool]:
    """Greedy decode using KV cache (T2 γ1 per codex msg 1779530833485-eb9296ca).

    Single-row (B=1). Prefill processes the full prompt under existing
    PrefixLM mask, populating the cache. Decode loop appends one position
    per step with attn_mask=None, is_causal=False (cache truncation
    enforces causality).

    Returns (decoded_string, too_long_flag, finite_flag). Contract matches
    `_decode_greedy_no_cache` so the caller (probe loops) can swap freely.
    """
    from calm.hrm_text_158.kv_cache import KVCache

    hrm = m.model
    sample_attn = hrm.H_level.core.layers[0].attn
    sample_w = sample_attn.gqkv_proj.weight
    cache = KVCache(
        max_seq_len=max_seq_len,
        num_kv_heads=sample_attn.num_key_value_heads,
        head_dim=sample_attn.head_dim,
        dtype=sample_w.dtype,
        device=device,
    )

    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    if len(prefix) >= max_seq_len:
        return "", True, True
    sep_pos = 1 + len(q_ids)

    # Prefill: full prompt forward, populates cache at length len(prefix)
    prefill_ids = torch.tensor([prefix], dtype=torch.long, device=device)
    prefill_pos = torch.arange(len(prefix), dtype=torch.long, device=device).unsqueeze(0)
    prefill_sep = torch.tensor([sep_pos], dtype=torch.long, device=device)
    finite = True
    with torch.no_grad():
        _, logits = m(
            None,
            {"inputs": prefill_ids, "sep_positions": prefill_sep, "position_ids": prefill_pos},
            kv_cache=cache,
        )
    if not bool(torch.isfinite(logits).all().item()):
        return "", False, False

    next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
    out_tokens: list[int] = []
    if next_id == tok.eos_id:
        return tok.decode(out_tokens, stop_at_eos=False), False, finite
    out_tokens.append(next_id)
    current_position = len(prefix)

    # Decode loop: single-token append per step
    for _ in range(max_gen - 1):
        if current_position >= max_seq_len:
            break
        decode_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
        decode_pos = torch.tensor([[current_position]], dtype=torch.long, device=device)
        # sep_positions unused by the cached-decode branch (no mask) but
        # passed through for kwarg-shape compatibility with the assert
        # path in the prefill branch (which is not entered here since
        # cache is active AND S==1 triggers the decode branch).
        decode_sep = prefill_sep
        with torch.no_grad():
            _, logits = m(
                None,
                {"inputs": decode_ids, "sep_positions": decode_sep, "position_ids": decode_pos},
                kv_cache=cache,
            )
        if not bool(torch.isfinite(logits).all().item()):
            finite = False
            break
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id == tok.eos_id:
            break
        out_tokens.append(next_id)
        current_position += 1

    return tok.decode(out_tokens, stop_at_eos=False), False, finite


def _decode_greedy_batched_cached(
    m: LMHead,
    tok,
    questions: list[str],
    *,
    max_gen: int = 8,
    max_seq_len: int,
    device: str,
) -> list[tuple[str, bool, bool]]:
    """R4a batched cached decode for a chunk of same-prefix-length rows.

    Per codex +1 R4a at msg 1779534977172-88a0cb6c. All questions in
    `questions` MUST tokenize to identical prefix length (caller is
    `_run_rows_batched` which groups by exact `len(prefix)`).

    One chunk-local `KVCache(batch_size=B)` per call. Cache buffers shape
    `(B, n_kv, max_seq_len, head_dim)`. No padding. Rows that hit EOS
    drop out of `active` mask; inactive rows continue to consume cache
    slots (lockstep position advance) but their emitted tokens are not
    appended. Cross-row attention is impossible inside SDPA so inactive
    rows cannot leak into live ones.

    Returns a list of (decoded_string, too_long_flag, finite_flag) in the
    same order as `questions`. `too_long=True` cannot happen here since
    the caller pre-filters that case; the return tuple shape is kept for
    contract parity with `_decode_greedy_cached`.
    """
    from calm.hrm_text_158.kv_cache import KVCache

    B = len(questions)
    if B == 0:
        return []
    hrm = m.model
    sample_attn = hrm.H_level.core.layers[0].attn
    sample_w = sample_attn.gqkv_proj.weight

    # Encode + assemble prefixes; verify same-length invariant.
    prefixes: list[list[int]] = []
    for q in questions:
        q_ids = tok.encode(q)
        prefixes.append([tok.bos_id] + q_ids + [tok.sep_id])
    prefix_len = len(prefixes[0])
    for p in prefixes:
        if len(p) != prefix_len:
            raise ValueError(
                "all questions in a batched chunk must share encoded prefix length; "
                f"got {[len(x) for x in prefixes]}"
            )
    if prefix_len >= max_seq_len:
        # Caller should have filtered this case; surface defensively.
        return [("", True, True) for _ in questions]
    sep_pos = prefix_len - 1  # 1 + len(q_ids); sep is at last prefill position

    cache = KVCache(
        max_seq_len=max_seq_len,
        num_kv_heads=sample_attn.num_key_value_heads,
        head_dim=sample_attn.head_dim,
        dtype=sample_w.dtype,
        device=device,
        batch_size=B,
    )

    # Prefill: stacked (B, prefix_len) tensor.
    prefill_ids = torch.tensor(prefixes, dtype=torch.long, device=device)
    prefill_pos = (
        torch.arange(prefix_len, dtype=torch.long, device=device)
        .unsqueeze(0)
        .expand(B, prefix_len)
        .contiguous()
    )
    prefill_sep = torch.full((B,), sep_pos, dtype=torch.long, device=device)
    with torch.no_grad():
        _, logits = m(
            None,
            {"inputs": prefill_ids, "sep_positions": prefill_sep, "position_ids": prefill_pos},
            kv_cache=cache,
        )

    # Per-row finite tracking. Logits last position: (B, V).
    finite_per_row = torch.isfinite(logits[:, -1, :]).all(dim=-1)  # (B,)
    # If any row went non-finite on prefill, that row gets ("", False, False);
    # decode proceeds for the rest (still in lockstep but their tokens are
    # discarded since the row is recorded as not-finite — keeps cache shapes uniform).
    next_ids = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)

    out_tokens: list[list[int]] = [[] for _ in range(B)]
    row_finite: list[bool] = [bool(finite_per_row[b].item()) for b in range(B)]
    # active: row will continue contributing tokens until EOS or max_gen.
    active = [row_finite[b] for b in range(B)]
    for b in range(B):
        if not active[b]:
            continue
        nid = int(next_ids[b].item())
        if nid == tok.eos_id:
            active[b] = False
        else:
            out_tokens[b].append(nid)

    current_position = prefix_len

    # Decode loop: lockstep, all rows advance one slot per step.
    for _ in range(max_gen - 1):
        if current_position >= max_seq_len:
            break
        if not any(active):
            break
        decode_ids = next_ids.unsqueeze(-1)  # (B, 1)
        decode_pos = torch.full(
            (B, 1), current_position, dtype=torch.long, device=device
        )
        decode_sep = prefill_sep
        with torch.no_grad():
            _, logits = m(
                None,
                {"inputs": decode_ids, "sep_positions": decode_sep, "position_ids": decode_pos},
                kv_cache=cache,
            )
        step_finite = torch.isfinite(logits[:, -1, :]).all(dim=-1)  # (B,)
        next_ids = torch.argmax(logits[:, -1, :], dim=-1)  # (B,)
        for b in range(B):
            if not active[b]:
                continue
            if not bool(step_finite[b].item()):
                row_finite[b] = False
                active[b] = False
                continue
            nid = int(next_ids[b].item())
            if nid == tok.eos_id:
                active[b] = False
            else:
                out_tokens[b].append(nid)
        current_position += 1

    results: list[tuple[str, bool, bool]] = []
    for b in range(B):
        if not row_finite[b]:
            # Match scalar-path failure shape: empty decoded, finite=False
            results.append(("", False, False))
        else:
            results.append((tok.decode(out_tokens[b], stop_at_eos=False), False, True))
    return results


def _run_rows_batched(
    m: LMHead,
    tok,
    questions: list[str],
    *,
    max_gen: int,
    max_seq_len: int,
    device: str,
    batch_size: int,
) -> tuple[list[tuple[str, bool, bool]], dict[int, int]]:
    """R4a batched runner: group rows by exact encoded prefix length,
    chunk each group ≤ batch_size, decode in batch, return per-row results
    in the original input order plus a chunk-size histogram.

    Pre-filters rows with `len(prefix) >= max_seq_len` (same `too_long`
    semantics as the scalar path) so they are never sent into a batched
    chunk and don't distort group-length statistics.
    """
    n = len(questions)
    results: list[Optional[tuple[str, bool, bool]]] = [None] * n
    # Encode + classify each row.
    prefix_lens: list[Optional[int]] = []
    too_long_indices: list[int] = []
    for i, q in enumerate(questions):
        q_ids = tok.encode(q)
        plen = 1 + len(q_ids) + 1
        if plen >= max_seq_len:
            results[i] = ("", True, True)
            too_long_indices.append(i)
            prefix_lens.append(None)
        else:
            prefix_lens.append(plen)

    # Group by exact prefix length (skip too_long rows).
    groups: dict[int, list[int]] = {}
    for i, plen in enumerate(prefix_lens):
        if plen is None:
            continue
        groups.setdefault(plen, []).append(i)

    chunk_size_hist: dict[int, int] = {}
    # Iterate groups deterministically (sorted by length) so receipts are
    # reproducible. Per-group: split into chunks ≤ batch_size.
    for plen in sorted(groups.keys()):
        indices = groups[plen]
        for start in range(0, len(indices), batch_size):
            chunk_indices = indices[start:start + batch_size]
            chunk_qs = [questions[i] for i in chunk_indices]
            chunk_out = _decode_greedy_batched_cached(
                m, tok, chunk_qs,
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
            )
            for orig_i, out in zip(chunk_indices, chunk_out):
                results[orig_i] = out
            csz = len(chunk_indices)
            chunk_size_hist[csz] = chunk_size_hist.get(csz, 0) + 1

    # Defensive: every result slot must be filled.
    for i, r in enumerate(results):
        if r is None:
            raise RuntimeError(f"batched runner left row {i} unfilled (bug)")
    return results, chunk_size_hist  # type: ignore[return-value]


def _parse_int(s: str) -> Optional[int]:
    """Extract first signed-int from a string; None if none found."""
    out: list[str] = []
    started = False
    for c in s:
        if c == "-" and not started:
            out.append(c)
            started = True
        elif c.isdigit():
            out.append(c)
            started = True
        else:
            if started:
                break
    if not out or out == ["-"]:
        return None
    try:
        return int("".join(out))
    except ValueError:
        return None


def probe(
    ckpt_path: str,
    eval_cap: int = 50,
    max_gen: int = 8,
    device: str | None = None,
    splits_loader=load_gsm8k_splits,
) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", "?")
    print(f"[probe] ckpt step={step}", flush=True)
    m, tok = _build_model_from_ckpt(ckpt, device)
    max_seq_len = ckpt["config"]["max_seq_len"]
    n_params = sum(p.numel() for p in m.parameters())
    print(f"[probe] params: {n_params:,} max_seq_len={max_seq_len}", flush=True)

    # Load val rows
    _, full_val, _ = splits_loader(val_frac=0.10)
    val_rows = full_val[:eval_cap]
    print(f"[probe] cap={eval_cap} val_rows={len(val_rows)}", flush=True)

    # Per-row probe
    parsed_correct = 0
    exact_correct = 0
    too_long_count = 0
    start_t = time.time()
    rows_out: list[dict] = []
    for i, r in enumerate(val_rows):
        expected = int(r["expected"])
        decoded, too_long, _finite = _decode_greedy_no_cache(
            m, tok, r["question"], max_gen=max_gen, max_seq_len=max_seq_len, device=device
        )
        parsed = _parse_int(decoded)
        is_parsed_correct = (parsed == expected) and not too_long
        is_exact_correct = (decoded == str(expected)) and not too_long
        parsed_correct += int(is_parsed_correct)
        exact_correct += int(is_exact_correct)
        too_long_count += int(too_long)
        rows_out.append({
            "i": i,
            "question": r["question"][:60],
            "expected": expected,
            "decoded": decoded,
            "parsed": parsed,
            "parsed_ok": is_parsed_correct,
            "exact_ok": is_exact_correct,
            "too_long": too_long,
        })

    elapsed = time.time() - start_t
    print(f"[probe] cap={eval_cap}: parsed={parsed_correct}/{eval_cap}={parsed_correct/eval_cap:.3f}  "
          f"exact={exact_correct}/{eval_cap}={exact_correct/eval_cap:.3f}  "
          f"too_long={too_long_count}/{eval_cap}  t={elapsed:.1f}s",
          flush=True)

    # Canonical 17×23=391 row (per codex correction 1: report only, no hard-assert)
    print(f"[probe] === canonical 17×23 row ===", flush=True)
    canonical_q = "what is 17 times 23?"
    canonical_expected = 391
    canonical_decoded, canonical_too_long, _canonical_finite = _decode_greedy_no_cache(
        m, tok, canonical_q, max_gen=max_gen, max_seq_len=max_seq_len, device=device
    )
    canonical_parsed = _parse_int(canonical_decoded)
    canonical_parsed_ok = (canonical_parsed == canonical_expected) and not canonical_too_long
    canonical_exact_ok = (canonical_decoded == str(canonical_expected)) and not canonical_too_long
    print(f"[probe]   question: {canonical_q!r}", flush=True)
    print(f"[probe]   expected: {canonical_expected}", flush=True)
    print(f"[probe]   decoded:  {canonical_decoded!r}", flush=True)
    print(f"[probe]   parsed:   {canonical_parsed}", flush=True)
    print(f"[probe]   parsed_ok: {canonical_parsed_ok}", flush=True)
    print(f"[probe]   exact_ok:  {canonical_exact_ok}", flush=True)
    print(f"[probe]   too_long:  {canonical_too_long}", flush=True)

    return {
        "ckpt_path": ckpt_path,
        "step": step,
        "n_params": n_params,
        "eval_cap": eval_cap,
        "parsed_correct": parsed_correct,
        "exact_correct": exact_correct,
        "too_long_count": too_long_count,
        "canonical_17x23": {
            "question": canonical_q,
            "expected": canonical_expected,
            "decoded": canonical_decoded,
            "parsed": canonical_parsed,
            "parsed_ok": canonical_parsed_ok,
            "exact_ok": canonical_exact_ok,
            "too_long": canonical_too_long,
        },
        "rows": rows_out,
        "elapsed_sec": elapsed,
    }


def probe_curriculum(
    ckpt_path: str,
    rungs: list[str],
    eval_cap: int = 200,
    max_gen: int = 8,
    device: str | None = None,
    output_json: str | None = None,
    use_cached_ternary_infer: bool = False,
    use_kv_cache_decode: bool = False,
    use_batched_probe_eval: bool = False,
    probe_batch_size: int = 16,
) -> RungProbeResult:
    """Phase 3 curriculum-mode probe (codex msg 1779462307554 Phase A receipt
    requirement).

    Probes the ckpt against:
    - each rung's held_out (via make_rung_examples(split="held_out"))
    - canonical 17×23=391 multiplication probe

    Returns a RungProbeResult dataclass; optionally writes to `output_json`.

    Exact and parsed metrics both reported per codex rule 7 — exact is the
    primary curriculum metric (`decoded.strip() == str(expected)`).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not rungs:
        raise ValueError("--curriculum-rungs requires at least one rung name")
    for r in rungs:
        if r not in RUNG_NAMES or r == "R7":
            raise ValueError(f"Invalid curriculum rung {r!r}; valid synthetic rungs: "
                             f"{[x for x in RUNG_NAMES if x != 'R7']}")
    # R4a fail-fast: --use-batched-probe-eval requires --use-kv-cache-decode.
    # Checked BEFORE ckpt load per codex msg 1779535251889-85062157 fail-fast rule.
    if use_batched_probe_eval and not use_kv_cache_decode:
        raise ValueError(
            "--use-batched-probe-eval requires --use-kv-cache-decode "
            "(batched path is built on top of the γ1 KV cache contract)"
        )
    if probe_batch_size < 1:
        raise ValueError(f"--probe-batch-size must be >= 1, got {probe_batch_size}")

    print(f"[probe-curriculum] loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", -1)
    print(f"[probe-curriculum] ckpt step={step}", flush=True)
    m, tok = _build_model_from_ckpt(ckpt, device)
    config = ckpt["config"]
    max_seq_len = config["max_seq_len"]
    n_params = sum(p.numel() for p in m.parameters())

    # T1 (α): cached ternary inference path. Codex msg 1779528934673-1c8bedf3.
    # Called AFTER `_build_model_from_ckpt` already set `m.eval()`.
    # Freezing in eval mode avoids the `train()` override clearing the cache.
    if use_cached_ternary_infer:
        from calm.hrm_text_158.bit_linear import freeze_bitlinears_for_inference
        n_frozen = freeze_bitlinears_for_inference(m)
        print(f"[probe-curriculum] cached-ternary-infer: froze {n_frozen} BitLinear modules", flush=True)

    # T2 γ1: KV cache decode (codex msg 1779530833485-eb9296ca). Substitutes
    # _decode_greedy_cached for _decode_greedy_no_cache at all call sites
    # below. Same signature; cached path uses single-row B=1 KV cache.
    decode_fn = _decode_greedy_cached if use_kv_cache_decode else _decode_greedy_no_cache
    if use_kv_cache_decode:
        print(f"[probe-curriculum] kv-cache-decode: ENABLED (single-row B=1)", flush=True)

    # R4a batched probe/eval (codex msg 1779534977172-88a0cb6c). Groups rows
    # by exact encoded prefix length; chunks each group ≤ probe_batch_size;
    # one chunk-local KVCache(batch_size=actual_chunk_B) per chunk. No padding.
    # Dependency check is at top of function (fail-fast before ckpt load).
    chunk_size_hist_agg: dict[int, int] = {}
    if use_batched_probe_eval:
        print(
            f"[probe-curriculum] batched-probe-eval: ENABLED "
            f"(batch_size={probe_batch_size}, exact-prefix-length grouping)",
            flush=True,
        )

    # Identify rung being trained (informational; not always the most recent rung)
    trained_rung = config.get("curriculum_rung", "?")
    replay_ratio = config.get("replay_ratio", 0.0)
    print(f"[probe-curriculum] trained_rung={trained_rung} replay_ratio={replay_ratio} "
          f"n_params={n_params:,} max_seq_len={max_seq_len}", flush=True)

    result = RungProbeResult(
        rung=trained_rung if trained_rung != "?" else (rungs[-1] if rungs else "?"),
        ckpt_path=ckpt_path,
        step=int(step) if isinstance(step, int) else 0,
        n_params=n_params,
    )

    start_t = time.time()
    overall_finite = True
    for r in rungs:
        # Curriculum seed pulls from ckpt config to ensure probe uses the
        # SAME generator state that train used (else held_out has different
        # rows -> meaningless probe)
        curriculum_seed = config.get("curriculum_seed", 42)
        rows = make_rung_examples(r, n=eval_cap, seed=curriculum_seed, split="held_out")
        rung_cap = len(rows)
        parsed_ok = 0
        exact_ok = 0
        too_long = 0
        rung_finite = True
        if use_batched_probe_eval:
            batched_results, hist = _run_rows_batched(
                m, tok,
                [ex["question"] for ex in rows],
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
                batch_size=probe_batch_size,
            )
            for sz, c in hist.items():
                chunk_size_hist_agg[sz] = chunk_size_hist_agg.get(sz, 0) + c
            row_outputs = batched_results
        else:
            row_outputs = [
                decode_fn(
                    m, tok, ex["question"], max_gen=max_gen,
                    max_seq_len=max_seq_len, device=device,
                )
                for ex in rows
            ]
        for ex, (decoded, tl, fin) in zip(rows, row_outputs):
            expected = ex["expected"]
            parsed = _parse_int(decoded)
            is_parsed = (parsed == expected) and not tl
            is_exact = (decoded.strip() == str(expected)) and not tl
            parsed_ok += int(is_parsed)
            exact_ok += int(is_exact)
            too_long += int(tl)
            if not fin:
                rung_finite = False
        # codex msg 1779463196431 rule 3: EXACT is the primary curriculum
        # metric. `rung_accuracy` reports exact_ok/cap so G1/G2 gates
        # downstream see strict string equality, not parsed-int (which
        # would let "391xyz" count as correct). Parsed metric remains
        # in `rung_parsed` for comparison.
        exact_acc = exact_ok / rung_cap if rung_cap else 0.0
        parsed_acc = parsed_ok / rung_cap if rung_cap else 0.0
        result.rung_accuracy[r] = exact_acc
        result.rung_parsed[r] = parsed_ok
        result.rung_exact[r] = exact_ok
        result.rung_too_long[r] = too_long
        result.rung_cap[r] = rung_cap
        overall_finite = overall_finite and rung_finite
        print(f"[probe-curriculum] {r}: exact={exact_ok}/{rung_cap}={exact_acc:.3f} (primary) "
              f"parsed={parsed_ok}/{rung_cap}={parsed_acc:.3f} "
              f"too_long={too_long}/{rung_cap} finite={rung_finite}", flush=True)

    # Canonical 17×23=391 (R3 mastery falsifier, codex rule 3 -- HARD R3 advance gate)
    canonical_q = "what is 17 times 23?"
    canonical_expected = 391
    if use_batched_probe_eval:
        canonical_results, canonical_hist = _run_rows_batched(
            m, tok, [canonical_q],
            max_gen=max_gen, max_seq_len=max_seq_len, device=device,
            batch_size=probe_batch_size,
        )
        for sz, c in canonical_hist.items():
            chunk_size_hist_agg[sz] = chunk_size_hist_agg.get(sz, 0) + c
        canonical_decoded, canonical_too_long, canonical_finite = canonical_results[0]
    else:
        canonical_decoded, canonical_too_long, canonical_finite = decode_fn(
            m, tok, canonical_q, max_gen=max_gen, max_seq_len=max_seq_len, device=device,
        )
    overall_finite = overall_finite and canonical_finite
    canonical_parsed = _parse_int(canonical_decoded)
    result.canonical_17x23 = {
        "question": canonical_q,
        "expected": canonical_expected,
        "decoded": canonical_decoded,
        "parsed": canonical_parsed,
        "parsed_ok": bool(canonical_parsed == canonical_expected and not canonical_too_long),
        # codex rule 7: exact (string equality) is the primary curriculum metric
        "exact_ok": bool(canonical_decoded.strip() == str(canonical_expected) and not canonical_too_long),
        "too_long": canonical_too_long,
    }
    # Per-rung one_digit exhaustive audits (codex msg 1779523412979-ff88b885).
    # For every audit-eligible rung present in `rungs`, run the rung's
    # 9-row finite-domain audit accessor and store keyed under
    # `result.one_digit_audits[rung_name]`. Required so multi-rung
    # probes (e.g. R1b5 with R1b4v2 retention) don't silently drop
    # prior-rung retention signal.
    #
    # Backcompat: `result.one_digit_audit` (singular legacy field)
    # mirrors `one_digit_audits["R1b4v2"]` when present, for older
    # receipt readers.
    curriculum_seed = config.get("curriculum_seed", 42)
    for audit_rung, accessor in ONE_DIGIT_AUDIT_REGISTRY.items():
        if audit_rung not in rungs:
            continue
        audit_rows = accessor(seed=curriculum_seed)
        audit_exact = 0
        audit_parsed = 0
        audit_too_long = 0
        audit_finite = True
        audit_row_results: list[dict] = []
        if use_batched_probe_eval:
            audit_decoded_rows, audit_hist = _run_rows_batched(
                m, tok,
                [ex["question"] for ex in audit_rows],
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
                batch_size=probe_batch_size,
            )
            for sz, c in audit_hist.items():
                chunk_size_hist_agg[sz] = chunk_size_hist_agg.get(sz, 0) + c
            audit_row_outputs = audit_decoded_rows
        else:
            audit_row_outputs = [
                decode_fn(
                    m, tok, ex["question"], max_gen=max_gen,
                    max_seq_len=max_seq_len, device=device,
                )
                for ex in audit_rows
            ]
        for ex, (decoded, tl, fin) in zip(audit_rows, audit_row_outputs):
            expected = ex["expected"]
            parsed = _parse_int(decoded)
            is_parsed = (parsed == expected) and not tl
            is_exact = (decoded.strip() == str(expected)) and not tl
            audit_exact += int(is_exact)
            audit_parsed += int(is_parsed)
            audit_too_long += int(tl)
            if not fin:
                audit_finite = False
            audit_row_results.append({
                "question": ex["question"],
                "expected": expected,
                "decoded": decoded,
                "parsed": parsed,
                "exact_ok": bool(is_exact),
                "parsed_ok": bool(is_parsed),
                "too_long": bool(tl),
            })
        overall_finite = overall_finite and audit_finite
        audit_record = {
            "exact": audit_exact,
            "parsed": audit_parsed,
            "too_long": audit_too_long,
            "cap": len(audit_rows),
            "finite": audit_finite,
            "rows": audit_row_results,
        }
        result.one_digit_audits[audit_rung] = audit_record
        # Backcompat alias: mirror R1b4v2 record into legacy singular field.
        if audit_rung == "R1b4v2":
            result.one_digit_audit = audit_record
        print(f"[probe-curriculum] {audit_rung} one_digit audit: exact={audit_exact}/"
              f"{len(audit_rows)} parsed={audit_parsed}/{len(audit_rows)} "
              f"too_long={audit_too_long}/{len(audit_rows)} finite={audit_finite}",
              flush=True)

    result.elapsed_sec = time.time() - start_t
    result.finite = overall_finite
    if use_batched_probe_eval:
        result.batched_chunk_size_hist = dict(chunk_size_hist_agg)
        print(
            f"[probe-curriculum] batched chunk-size histogram: "
            f"{sorted(chunk_size_hist_agg.items())}",
            flush=True,
        )
    print(f"[probe-curriculum] canonical 17×23: decoded={canonical_decoded!r} "
          f"parsed={canonical_parsed} parsed_ok={result.canonical_17x23['parsed_ok']} "
          f"exact_ok={result.canonical_17x23['exact_ok']}", flush=True)
    print(f"[probe-curriculum] elapsed={result.elapsed_sec:.1f}s finite={result.finite}",
          flush=True)

    if output_json is not None:
        out_path = Path(output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w") as f:
            json.dump(asdict(result), f, indent=2, sort_keys=True)
        print(f"[probe-curriculum] wrote {out_path}", flush=True)

    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HRM-Text-1.58 probe.")
    ap.add_argument("--ckpt-path", type=str, required=True)
    ap.add_argument("--eval-cap", type=int, default=50)
    ap.add_argument("--max-gen", type=int, default=8)
    # Phase 3 curriculum probe flags (codex msg 1779462307554 Phase A)
    ap.add_argument("--curriculum-rungs", type=str, default=None,
                    help="Comma-separated rung names to probe (e.g. 'R0,R1,R2'). "
                         "When set, runs the curriculum-mode probe instead of GSM8k.")
    ap.add_argument("--probe-output-json", type=str, default=None,
                    help="Path to write RungProbeResult JSON (curriculum mode).")
    # T1 (α) inference-only flag per codex msg 1779528934673-1c8bedf3.
    # Caches BitLinear `w_q * scale` once at model freeze to skip per-call
    # re-quantization. Name distinct from future `--use-native-ternary-infer`
    # which is reserved for a true packed-ternary matmul kernel (T1 β).
    ap.add_argument("--use-cached-ternary-infer", action="store_true",
                    help="Cache BitLinear quantized weights once at model load "
                         "for inference-only F.linear dispatch. Skips per-call "
                         "quantize-and-materialize (~30%% wall-clock per T0). "
                         "Inference-only; training STE/backward unchanged.")
    # T2 γ1 inference-only KV cache decode flag per codex msg 1779530833485-eb9296ca.
    ap.add_argument("--use-kv-cache-decode", action="store_true",
                    help="Single-row (B=1) KV cache decode. Prefill with PrefixLM "
                         "mask, then single-token append per decode step with "
                         "attn_mask=None, is_causal=False. Targets the ~83%% non-BL "
                         "bucket from T2 re-profile. Inference-only; training path "
                         "unchanged. Composable with --use-cached-ternary-infer.")
    # R4a batched probe/eval flags per codex msg 1779534977172-88a0cb6c.
    ap.add_argument("--use-batched-probe-eval", action="store_true",
                    help="Group probe rows by exact encoded prefix length and "
                         "decode each chunk with a B=N KVCache. Requires "
                         "--use-kv-cache-decode (fails fast otherwise). No "
                         "padding; chunk-local cache per exact-length group. "
                         "Preserves original row order in result records.")
    ap.add_argument("--probe-batch-size", type=int, default=16,
                    help="Max chunk size when --use-batched-probe-eval is set. "
                         "Default 16. Groups larger than this are split into "
                         "multiple chunks; smaller groups run at their natural "
                         "size (no padding).")
    args = ap.parse_args()

    if args.curriculum_rungs is not None:
        rungs = [r.strip() for r in args.curriculum_rungs.split(",") if r.strip()]
        probe_curriculum(
            args.ckpt_path,
            rungs=rungs,
            eval_cap=args.eval_cap,
            max_gen=args.max_gen,
            output_json=args.probe_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
        )
    else:
        probe(args.ckpt_path, eval_cap=args.eval_cap, max_gen=args.max_gen)
