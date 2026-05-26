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
import re
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
    r1b6_one_digit_audit_rows,
    r1b7_one_digit_audit_rows,
    r1b8_one_digit_audit_rows,
    r1b9_one_digit_audit_rows,
    r1b10_one_digit_audit_rows,
)

# Per-rung audit registry (codex msg 1779523412979-ff88b885 + R1b6 added
# per codex msg 1779545956176-4a8cfc3e after gabe greenlight relay
# 1779545575582-7c52a912 of verbatim "ok implement, full prov"; R1b7
# added per codex msg 1779547753761-5711d790 under durable gabe
# provenance relay 1779547541812; R1b8 added per codex msg
# 1779550489408-f40f66ab after R1b7 commit 682659b ADVANCED + A0
# exhaustive audit PASS; R1b9 added per codex msg 1779554293017-3ba4b4ee
# after R-C diagnostic PASS msg 1779554256972, parent = R1b3-repair
# candidate banked as new chain head; R1b10 audit accessor preserved
# but R1b10 is PARKED / diagnosis-only per codex msg 1779558351771-055c2265
# after three failed promotion attempts from R1b9 chain head (R1b9
# remains math chain head). The R1b10 keyed audit fires only when
# R1b10 is explicitly in `--curriculum-rungs`; default math probe
# does NOT exercise it).
# Each entry maps rung name -> callable(seed) -> list of 9 audit rows.
# When `probe_curriculum` runs, it iterates this registry against the
# `rungs` argument so every audit-eligible rung present gets a keyed
# audit in result.one_digit_audits (no silent retention drops).
ONE_DIGIT_AUDIT_REGISTRY = {
    "R1b4v2": r1b4v2_one_digit_audit_rows,
    "R1b5": r1b5_one_digit_audit_rows,
    "R1b6": r1b6_one_digit_audit_rows,
    "R1b7": r1b7_one_digit_audit_rows,
    "R1b8": r1b8_one_digit_audit_rows,
    "R1b9": r1b9_one_digit_audit_rows,
    "R1b10": r1b10_one_digit_audit_rows,
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


def merge_watch_rows(passed_rows: list[dict] | None,
                     config_rows: list[dict] | None) -> list[dict]:
    """Union watch rows from --watch-rows-json (``passed_rows``) with a
    checkpoint's ``config.watch_rows`` (accepted-exception metadata written at
    bank time). Deduped by ``(question, int(expected))``; passed rows win on
    collision. Pure — unit-tested without a model. (codex msg 1779691762976:
    accepted-exception policy needs mechanical visibility, not prose.)"""
    merged = list(passed_rows or [])
    seen = {(r["question"], int(r["expected"])) for r in merged}
    for r in (config_rows or []):
        k = (r["question"], int(r["expected"]))
        if k not in seen:
            merged.append(r)
            seen.add(k)
    return merged


def format_watch_line(result: dict) -> str:
    """One ``[probe-watch]`` line for a decoded watch-row result (grep-able by
    the audit watcher). Includes ``source_rung`` when present so the accepted
    exception is mechanically keyed to its rung (codex msg 1779692376889)."""
    sr = result.get("source_rung")
    sr_part = f" source_rung={sr}" if sr else ""
    return (f"[probe-watch] {result.get('key', '?')} {result['question']!r} "
            f"expected={result['expected']} decoded={result.get('decoded')!r} "
            f"parsed={result.get('parsed')} parsed_ok={result.get('parsed_ok')}{sr_part}")


def watch_aggregate(watch_results: list[dict]) -> dict:
    """Aggregate over decoded watch rows: parsed-ok count + total. Empty ⇒
    vacuously all-ok (no watch rows is not a failure)."""
    n = len(watch_results)
    n_ok = sum(1 for w in watch_results if w.get("parsed_ok"))
    return {"n_total": n, "n_parsed_ok": n_ok,
            "all_parsed_ok": (n_ok == n) if n else True}


def _l0c_watch_transform(row: dict) -> dict:
    """Map a watch row onto the exhaustive-L0c surface for support lookup:
    a math-surface question (`what is <expr>?`) becomes `<expr> equals what?`
    preserving key/expected/source_rung; already-L0c or foreign questions
    pass through unchanged (matched directly, or reported NOT_IN_ACTIVE by
    design). codex msg 1779693537447 Q2 — without this the banked math-surface
    watch row would falsely report NOT_IN_ACTIVE in the L0c audit."""
    from calm.hrm_text_158.curriculum.language_supports import _math_q_to_l0c
    q = row.get("question", "")
    if q.startswith("what is ") and q.endswith("?"):
        new = dict(row)
        new["question"] = _math_q_to_l0c(q)
        return new
    return row


def probe_exhaustive_finite_supports(
    ckpt_path: str,
    *,
    max_gen: int = 8,
    device: str | None = None,
    output_json: str | None = None,
    watch_rows: list[dict] | None = None,
    use_cached_ternary_infer: bool = False,
    use_kv_cache_decode: bool = False,
    use_batched_probe_eval: bool = False,
    probe_batch_size: int = 32,
    support_builder=None,
    expected_aggregate: int | None = None,
    label: str = "probe-exhaustive",
    watch_row_transform=None,
) -> dict:
    """Exhaustive finite-support audit for the active math chain
    (currently R0..R1b9; aggregate 1255). Per codex msg
    1779552750209-3218959b after R1b8 commit 1a14a09. Promoted from
    /tmp helper to committed tooling because sampled probes can hide
    cluster regressions and boundary singletons that exhaustive audit
    catches deterministically. R1b9 added per codex msg
    1779554293017-3ba4b4ee. R1b10 is PARKED / diagnosis-only per codex
    msg 1779558351771-055c2265 and is NOT in the default active math
    chain (still reachable via explicit per-rung probes).

    Iterates `build_exhaustive_supports()` per rung, decodes via the
    faststack path (cached ternary + KV + batched eval when enabled),
    aggregates totals, captures up to first 20 holes per rung with
    replayable detail, runs optional `watch_rows` boundary checks, and
    optionally writes JSON.

    Returns dict with `ckpt_path`, `ckpt_step`, `device`, per-rung
    `results`, `aggregate`, `watch_rows`, `elapsed_s`, `active_rungs`,
    plus flag echo for receipt reproducibility.
    """
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        EXHAUSTIVE_ACTIVE_RUNGS,
        EXHAUSTIVE_EXPECTED_AGGREGATE,
        build_exhaustive_supports,
    )

    # Parametrized so the same audit serves math A0 (default) AND exhaustive
    # L0c (codex msg 1779693537447): support_builder/expected_aggregate/label
    # default to byte-identical math-A0 behavior; watch_row_transform maps the
    # config watch row onto the support surface (identity for math).
    if support_builder is None:
        support_builder = build_exhaustive_supports
    if expected_aggregate is None:
        expected_aggregate = EXHAUSTIVE_EXPECTED_AGGREGATE
    _label = f"[{label}] "  # trailing space: prints render "[label] <content>"

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if use_batched_probe_eval and not use_kv_cache_decode:
        raise ValueError(
            "--use-batched-probe-eval requires --use-kv-cache-decode "
            "(batched path is built on top of the γ1 KV cache contract)"
        )
    if probe_batch_size < 1:
        raise ValueError(f"--probe-batch-size must be >= 1, got {probe_batch_size}")

    watch_rows = list(watch_rows) if watch_rows else []

    print(f"{_label}loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", -1)
    print(f"{_label}ckpt step={step}", flush=True)
    m, tok = _build_model_from_ckpt(ckpt, device)
    max_seq_len = ckpt["config"]["max_seq_len"]

    if use_cached_ternary_infer:
        from calm.hrm_text_158.bit_linear import freeze_bitlinears_for_inference
        n_frozen = freeze_bitlinears_for_inference(m)
        print(f"{_label}cached-ternary-infer: froze {n_frozen} "
              f"BitLinear modules", flush=True)

    # Decode-path dispatch — mirror probe_curriculum (codex msg 1779553066144
    # Blocker 2: prior implementation always called _run_rows_batched; flags
    # were ignored). Selection:
    #   - use_batched_probe_eval (requires use_kv_cache_decode): batched KV path
    #   - elif use_kv_cache_decode:                                row-by-row KV cache
    #   - else:                                                    row-by-row no cache
    if use_batched_probe_eval:
        dispatch_path = "batched_kv_cache"
    elif use_kv_cache_decode:
        dispatch_path = "scalar_kv_cache"
    else:
        dispatch_path = "scalar_no_cache"
    print(f"{_label}decode dispatch: {dispatch_path}", flush=True)

    decode_fn = _decode_greedy_cached if use_kv_cache_decode else _decode_greedy_no_cache

    supports = support_builder()
    print(f"{_label}active rungs: {list(supports.keys())}", flush=True)

    # Auto-source accepted-exception watch rows from the chain-head ckpt
    # `config.watch_rows` (written at bank time) so the accepted-exception
    # policy is mechanically surfaced every audit, not prose-dependent
    # (codex msg 1779691762976 / bank metadata). Merged with any
    # --watch-rows-json rows; deduped.
    _cfg_watch = ckpt.get("config", {}).get("watch_rows") or []
    if _cfg_watch:
        watch_rows = merge_watch_rows(watch_rows, _cfg_watch)
        print(f"{_label}watch_rows: +{len(_cfg_watch)} from ckpt "
              f"config.watch_rows (merged total {len(watch_rows)})", flush=True)

    # Map watch rows onto the SUPPORT surface before lookup: identity for
    # math A0; for exhaustive L0c, `what is <expr>?` -> `<expr> equals what?`
    # so the math-surface config watch row matches the language-surface
    # support rows instead of falsely reporting NOT_IN_ACTIVE_SUPPORT
    # (codex msg 1779693537447 Q2).
    if watch_row_transform is not None and watch_rows:
        watch_rows = [watch_row_transform(r) for r in watch_rows]

    # Build a flat lookup for watch-row population.
    watch_lookup: dict[tuple[str, int], dict] = {}
    if watch_rows:
        watch_lookup = {(row["question"], int(row["expected"])): row for row in watch_rows}
    watch_results: list[dict] = []

    def _parse_int(text: str) -> int | None:
        # Mirror production parser: cap digit-run at 12 chars to defeat
        # post-bias loop residue (compute_facades.md POST_BIAS_BUDGET).
        capped = re.sub(r"(\d{12})\d+", r"\1", text)
        m_ = re.search(r"-?\d+", capped)
        return int(m_.group(0)) if m_ else None

    def _decode_rows(qs: list[str]) -> list[tuple[str, bool, bool]]:
        """Dispatch decode according to selected path."""
        if use_batched_probe_eval:
            per_row, _hist = _run_rows_batched(
                m, tok, qs,
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
                batch_size=probe_batch_size,
            )
            return per_row
        # Row-by-row path (scalar): use decode_fn per row
        return [
            decode_fn(m, tok, q, max_gen=max_gen,
                      max_seq_len=max_seq_len, device=device)
            for q in qs
        ]

    results: dict[str, dict] = {}
    finite_all = True
    t0 = time.time()
    for rung, rows in supports.items():
        questions = [q for q, _ in rows]
        expected_list = [e for _, e in rows]
        rt0 = time.time()
        per_row = _decode_rows(questions)
        rt_elapsed = time.time() - rt0
        holes: list[dict] = []
        finite_rung = True
        too_long = 0
        exact = 0
        parsed_correct = 0
        for q, exp, (decoded, tl, fin) in zip(questions, expected_list, per_row):
            if not fin:
                finite_rung = False
                finite_all = False
            # Codex msg 1779553066144 Blocker 1: n_exact / exact_ok must be
            # strict string equality (matches probe_curriculum's primary
            # `decoded.strip() == str(expected)`). Parsed correctness is
            # reported separately as `n_parsed_correct` per rung.
            exact_match = (not tl) and (decoded.strip() == str(exp))
            parsed = _parse_int(decoded) if not tl else None
            parsed_match = (not tl) and parsed == exp
            wk = (q, exp)
            if wk in watch_lookup:
                src = watch_lookup[wk]
                watch_results.append({
                    "key": src["key"],
                    "question": q,
                    "expected": exp,
                    # Ground-truth rung where the row was actually decoded (the
                    # exhaustive-audit loop var) — keys the accepted exception
                    # to R1b2 and makes moved-hole classification mechanical
                    # rather than prose (codex msg 1779692376889 fix 1). Falls
                    # back to the config row's source_rung if absent.
                    "source_rung": rung or src.get("source_rung"),
                    "decoded": decoded,
                    "parsed": parsed,
                    "too_long": tl,
                    "finite": fin,
                    "exact_ok": exact_match,        # strict
                    "parsed_ok": parsed_match,      # lenient (separate report)
                })
            if tl:
                too_long += 1
                holes.append({
                    "question": q, "expected": exp, "decoded": decoded,
                    "parsed": None, "exact_ok": False, "parsed_ok": False,
                    "too_long": True, "finite": fin,
                })
                continue
            if parsed_match:
                parsed_correct += 1
            if exact_match:
                exact += 1
            else:
                holes.append({
                    "question": q, "expected": exp, "decoded": decoded,
                    "parsed": parsed,
                    "exact_ok": False,
                    "parsed_ok": parsed_match,
                    "too_long": False, "finite": fin,
                })
        n_total = len(rows)
        results[rung] = {
            "n_total": n_total,
            "n_exact": exact,                       # strict (primary)
            "n_parsed_correct": parsed_correct,     # lenient (separate)
            "rate": exact / n_total if n_total else 1.0,
            "n_holes": len(holes),                  # strict holes
            "n_too_long": too_long,
            "finite": finite_rung,
            "holes_first20": holes[:20],
            "elapsed_s": round(rt_elapsed, 3),
        }
        print(f"{_label}{rung:8s} {exact}/{n_total} = "
              f"{results[rung]['rate']:.4f} (strict) parsed={parsed_correct}/{n_total} "
              f"holes={len(holes)} too_long={too_long} "
              f"finite={finite_rung} t={rt_elapsed:.2f}s", flush=True)

    total_elapsed = time.time() - t0
    agg_total = sum(r["n_total"] for r in results.values())
    agg_exact = sum(r["n_exact"] for r in results.values())
    agg_parsed = sum(r["n_parsed_correct"] for r in results.values())
    agg_holes = sum(r["n_holes"] for r in results.values())
    aggregate = {
        "n_total": agg_total,
        "n_exact": agg_exact,                       # strict (primary)
        "n_parsed_correct": agg_parsed,             # lenient (separate)
        "rate": agg_exact / agg_total if agg_total else 1.0,
        "n_holes": agg_holes,
        "finite": finite_all,
        "expected_aggregate": expected_aggregate,
    }
    print(f"{_label}AGGREGATE strict={agg_exact}/{agg_total} = "
          f"{aggregate['rate']:.4f} parsed={agg_parsed}/{agg_total} "
          f"holes={agg_holes} finite={finite_all} "
          f"elapsed={total_elapsed:.1f}s", flush=True)

    # Watch-row surfacing: emit one grep-able [probe-watch] line per decoded
    # watch row + a WATCH AGGREGATE. Non-blocking — accepted exceptions are
    # reported, never fail the audit by themselves (codex msg 1779691762976 /
    # gabe self-repair hypothesis 1779692099947: track watch-row status each
    # rung receipt).
    watch_agg = watch_aggregate(watch_results)
    if watch_rows:
        for w in watch_results:
            print(format_watch_line(w), flush=True)
        _found = {(w["question"], int(w["expected"])) for w in watch_results}
        for r in watch_rows:
            if (r["question"], int(r["expected"])) not in _found:
                _sr = r.get("source_rung")
                _sr_part = f" source_rung={_sr}" if _sr else ""
                print(f"[probe-watch] {r.get('key', '?')} {r['question']!r} "
                      f"expected={r['expected']} NOT_IN_ACTIVE_SUPPORT{_sr_part}",
                      flush=True)
        print(f"[probe-watch] WATCH AGGREGATE parsed_ok="
              f"{watch_agg['n_parsed_ok']}/{watch_agg['n_total']}", flush=True)

    output = {
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(step) if step != -1 else None,
        "label": label,  # self-describing surface (e.g. probe-l0c-exhaustive)
        "device": device,
        "active_rungs": list(supports.keys()),
        "flags": {
            "use_cached_ternary_infer": use_cached_ternary_infer,
            "use_kv_cache_decode": use_kv_cache_decode,
            "use_batched_probe_eval": use_batched_probe_eval,
            "probe_batch_size": probe_batch_size,
            "max_gen": max_gen,
        },
        "dispatch_path": dispatch_path,
        "results": results,
        "aggregate": aggregate,
        "watch_rows": watch_results,
        "watch_aggregate": watch_agg,
        "elapsed_s": round(total_elapsed, 2),
    }
    if output_json:
        # Codex msg 1779553066144 polish: mkdir -p parent like probe_curriculum.
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"{_label}wrote {output_json}", flush=True)
    return output


def probe_language_finite_supports(
    ckpt_path: str,
    *,
    audit_seed: int | None = None,
    max_gen: int = 8,
    device: str | None = None,
    output_json: str | None = None,
    use_cached_ternary_infer: bool = False,
    use_kv_cache_decode: bool = False,
    use_batched_probe_eval: bool = False,
    probe_batch_size: int = 32,
    supports_builder=None,
    expected_aggregate: int | None = None,
    expected_aggregate_fn=None,
    surface: str = "language",
) -> dict:
    """Language-axis finite-support audit (codex msg 1779559495228-f863199b
    +1 implement; slice 2 probe integration per msg 1779560726491-971f67d5).

    Parallel surface to `probe_exhaustive_finite_supports`. Iterates
    `build_language_supports()` per active language rung (currently L0a),
    decodes via the faststack path, aggregates totals, and emits a
    per-source-rung breakdown in audit JSON. Does NOT touch the math
    A0 export; math aggregate stays at 1255 in `probe_exhaustive_finite_supports`.

    Audit seed handling (codex 1779560443281 nit): defaults to the
    ckpt's stored `curriculum_seed` config field; explicit
    `audit_seed` arg overrides. Warns on mismatch. NO hardcoded
    `seed=42` default.

    Returns dict with `ckpt_path`, `ckpt_step`, `device`,
    `audit_seed`, per-rung `results` (each result has `n_total`,
    `n_exact`, `n_parsed_correct`, `by_source_rung`,
    `holes_first20`), `aggregate`, `elapsed_s`, plus flag echo.
    """
    from calm.hrm_text_158.curriculum.language_supports import (
        LANGUAGE_ACTIVE_RUNGS,
        LANGUAGE_EXPECTED_AGGREGATE,
        build_language_supports,
        language_source_rung_buckets,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if use_batched_probe_eval and not use_kv_cache_decode:
        raise ValueError(
            "--use-batched-probe-eval requires --use-kv-cache-decode "
            "(batched path is built on top of the γ1 KV cache contract)"
        )
    if probe_batch_size < 1:
        raise ValueError(f"--probe-batch-size must be >= 1, got {probe_batch_size}")

    print(f"[probe-language] loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", -1)
    print(f"[probe-language] ckpt step={step}", flush=True)

    # Audit seed resolution per codex msg 1779560443281: prefer ckpt's
    # stored curriculum_seed (matches the seed used to draw the L0a
    # train/held partition during training). NO hardcoded 42 default.
    config = ckpt["config"]
    ckpt_curriculum_seed = config.get("curriculum_seed")
    if audit_seed is None:
        if ckpt_curriculum_seed is None:
            raise ValueError(
                "language audit seed cannot be resolved: ckpt config has no "
                "'curriculum_seed' field and no explicit --language-audit-seed "
                "was passed. Pass --language-audit-seed N or use a ckpt that "
                "stores curriculum_seed (Phase 3 ckpts do this by default)."
            )
        audit_seed = int(ckpt_curriculum_seed)
        print(f"[probe-language] audit_seed={audit_seed} (from ckpt.curriculum_seed)", flush=True)
    else:
        if (
            ckpt_curriculum_seed is not None
            and int(ckpt_curriculum_seed) != int(audit_seed)
        ):
            print(
                f"[probe-language] WARN: --language-audit-seed={audit_seed} differs "
                f"from ckpt.curriculum_seed={ckpt_curriculum_seed}; audit will "
                f"use {audit_seed} but row sampling may not match training partition.",
                flush=True,
            )
        else:
            print(f"[probe-language] audit_seed={audit_seed} (explicit override)", flush=True)

    # Seed-aware expected aggregate (F.4d STEP-0 fix): K-band surfaces have
    # seed-dependent counts (L0c2-K1 = 24 at seed-42, 29 at seed-17). When the
    # caller passes expected_aggregate_fn, evaluate it at the RESOLVED audit_seed
    # so the reported expected matches the actual built rows — no more static
    # seed-42 expected=24 against a 29-row seed-17 audit (header/manifest match).
    if expected_aggregate_fn is not None:
        expected_aggregate = int(expected_aggregate_fn(audit_seed))
        print(f"[probe-language] expected_aggregate={expected_aggregate} "
              f"(seed-aware, audit_seed={audit_seed})", flush=True)

    m, tok = _build_model_from_ckpt(ckpt, device)
    max_seq_len = ckpt["config"]["max_seq_len"]

    if use_cached_ternary_infer:
        from calm.hrm_text_158.bit_linear import freeze_bitlinears_for_inference
        n_frozen = freeze_bitlinears_for_inference(m)
        print(f"[probe-language] cached-ternary-infer: froze {n_frozen} "
              f"BitLinear modules", flush=True)

    # Decode-path dispatch — mirror probe_exhaustive_finite_supports.
    if use_batched_probe_eval:
        dispatch_path = "batched_kv_cache"
    elif use_kv_cache_decode:
        dispatch_path = "scalar_kv_cache"
    else:
        dispatch_path = "scalar_no_cache"
    print(f"[probe-language] decode dispatch: {dispatch_path}", flush=True)

    decode_fn = _decode_greedy_cached if use_kv_cache_decode else _decode_greedy_no_cache

    # supports_builder override (Slice F.1): --l0c1-audit passes
    # build_l0c1_support to audit the standalone 121-row L0c1 precursor
    # surface; default keeps the canonical build_language_supports (690).
    _supports_builder = supports_builder if supports_builder is not None else build_language_supports
    supports = _supports_builder(seed=audit_seed)
    print(f"[probe-language] surface={surface} audited rungs: {list(supports.keys())}", flush=True)

    def _parse_int(text: str) -> int | None:
        capped = re.sub(r"(\d{12})\d+", r"\1", text)
        m_ = re.search(r"-?\d+", capped)
        return int(m_.group(0)) if m_ else None

    def _identity_n(question: str, expected: int) -> int | None:
        m_ = re.fullmatch(r"\s*(\d+) equals what\?\s*", question)
        if not m_:
            return None
        n_ = int(m_.group(1))
        return n_ if n_ == int(expected) else None

    def _decode_class(
        decoded: str,
        *,
        parsed: int | None,
        expected: int,
        exact_ok: bool,
        too_long: bool,
        finite: bool,
    ) -> str:
        if exact_ok:
            return "exact_copy"
        stripped = decoded.strip()
        if too_long or not finite or not stripped:
            return "empty"
        if parsed is None:
            return "format_only"
        if expected >= 10 and parsed == expected % 10:
            return "ones_only"
        if expected >= 10 and parsed == expected // 10:
            return "tens_only"
        if expected >= 10 and 0 <= parsed <= 9:
            return "single_digit_other"
        digits = re.sub(r"\D", "", stripped)
        if len(digits) >= 2 and len(set(digits)) == 1:
            return "constant"
        return "other_wrong"

    def _decode_rows(qs: list[str]) -> list[tuple[str, bool, bool]]:
        if use_batched_probe_eval:
            per_row, _hist = _run_rows_batched(
                m, tok, qs,
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
                batch_size=probe_batch_size,
            )
            return per_row
        return [
            decode_fn(m, tok, q, max_gen=max_gen,
                      max_seq_len=max_seq_len, device=device)
            for q in qs
        ]

    results: dict[str, dict] = {}
    finite_all = True
    t0 = time.time()
    for rung, rows in supports.items():
        # rows are (question, expected, source_rung) triples
        questions = [q for q, _e, _s in rows]
        expecteds = [e for _q, e, _s in rows]
        sources = [s for _q, _e, s in rows]
        rt0 = time.time()
        per_row = _decode_rows(questions)
        rt_elapsed = time.time() - rt0

        # Per-source-rung accumulators
        bucket_keys = language_source_rung_buckets(rung)
        by_source: dict[str, dict] = {
            b: {"n_total": 0, "n_exact": 0, "n_parsed_correct": 0, "n_holes": 0}
            for b in bucket_keys
        }

        holes: list[dict] = []
        rows_all: list[dict] = []
        finite_rung = True
        too_long = 0
        exact = 0
        parsed_correct = 0
        for q, exp, src, (decoded, tl, fin) in zip(questions, expecteds, sources, per_row):
            if not fin:
                finite_rung = False
                finite_all = False
            exact_match = (not tl) and (decoded.strip() == str(exp))
            parsed = _parse_int(decoded) if not tl else None
            parsed_match = (not tl) and parsed == exp
            decode_class = _decode_class(
                decoded,
                parsed=parsed,
                expected=exp,
                exact_ok=exact_match,
                too_long=tl,
                finite=fin,
            )
            row_record = {
                "question": q,
                "expected": exp,
                "decoded": decoded,
                "parsed": parsed,
                "exact_ok": exact_match,
                "parsed_ok": parsed_match,
                "too_long": tl,
                "finite": fin,
                "source_rung": src,
                "bucket": src,
                "decode_class": decode_class,
            }
            n_ = _identity_n(q, exp)
            if n_ is not None:
                row_record.update({
                    "n": n_,
                    "tens": n_ // 10,
                    "ones": n_ % 10,
                })
            rows_all.append(row_record)
            by_source[src]["n_total"] += 1
            if tl:
                too_long += 1
                by_source[src]["n_holes"] += 1
                holes.append(row_record)
                continue
            if parsed_match:
                parsed_correct += 1
                by_source[src]["n_parsed_correct"] += 1
            if exact_match:
                exact += 1
                by_source[src]["n_exact"] += 1
            else:
                by_source[src]["n_holes"] += 1
                holes.append(row_record)
        n_total = len(rows)
        results[rung] = {
            "n_total": n_total,
            "n_exact": exact,                       # strict (primary)
            "n_parsed_correct": parsed_correct,     # lenient (separate)
            "rate": exact / n_total if n_total else 1.0,
            "n_holes": len(holes),
            "n_too_long": too_long,
            "finite": finite_rung,
            "by_source_rung": by_source,
            "rows_all": rows_all,
            "holes_first20": holes[:20],
            "elapsed_s": round(rt_elapsed, 3),
        }
        print(f"[probe-language] {rung:6s} {exact}/{n_total} = "
              f"{results[rung]['rate']:.4f} (strict) parsed={parsed_correct}/{n_total} "
              f"holes={len(holes)} too_long={too_long} "
              f"finite={finite_rung} t={rt_elapsed:.2f}s", flush=True)
        # Per-source-rung sub-line
        for b in bucket_keys:
            bs = by_source[b]
            print(f"[probe-language]   bucket {b:14s} "
                  f"strict={bs['n_exact']}/{bs['n_total']} "
                  f"parsed={bs['n_parsed_correct']}/{bs['n_total']} "
                  f"holes={bs['n_holes']}", flush=True)

    total_elapsed = time.time() - t0
    agg_total = sum(r["n_total"] for r in results.values())
    agg_exact = sum(r["n_exact"] for r in results.values())
    agg_parsed = sum(r["n_parsed_correct"] for r in results.values())
    agg_holes = sum(r["n_holes"] for r in results.values())
    aggregate = {
        "n_total": agg_total,
        "n_exact": agg_exact,
        "n_parsed_correct": agg_parsed,
        "rate": agg_exact / agg_total if agg_total else 1.0,
        "n_holes": agg_holes,
        "finite": finite_all,
        "expected_aggregate": (
            expected_aggregate if expected_aggregate is not None
            else LANGUAGE_EXPECTED_AGGREGATE
        ),
    }
    print(f"[probe-language] {surface.upper()} AGGREGATE strict={agg_exact}/{agg_total} = "
          f"{aggregate['rate']:.4f} parsed={agg_parsed}/{agg_total} "
          f"holes={agg_holes} finite={finite_all} "
          f"elapsed={total_elapsed:.1f}s", flush=True)

    # Codex msg 1779560820500 tightening: JSON records BOTH audit_seed
    # (actually used) and ckpt_curriculum_seed (what training used),
    # plus an explicit `seed_mismatch` flag so receipts can be
    # filter-grepped for mismatch incidents.
    output = {
        "surface": surface,
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(step) if step != -1 else None,
        "device": device,
        "audit_seed": int(audit_seed),
        "ckpt_curriculum_seed": (
            int(ckpt_curriculum_seed)
            if ckpt_curriculum_seed is not None
            else None
        ),
        "seed_mismatch": (
            ckpt_curriculum_seed is not None
            and int(ckpt_curriculum_seed) != int(audit_seed)
        ),
        "active_language_rungs": list(supports.keys()),
        "flags": {
            "use_cached_ternary_infer": use_cached_ternary_infer,
            "use_kv_cache_decode": use_kv_cache_decode,
            "use_batched_probe_eval": use_batched_probe_eval,
            "probe_batch_size": probe_batch_size,
            "max_gen": max_gen,
        },
        "dispatch_path": dispatch_path,
        "results": results,
        "aggregate": aggregate,
        "elapsed_s": round(total_elapsed, 2),
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[probe-language] wrote {output_json}", flush=True)
    return output


def probe_anchor_finite_supports(
    ckpt_path: str,
    *,
    anchor_set_override: str | None = None,
    max_gen: int = 8,
    device: str | None = None,
    output_json: str | None = None,
    use_cached_ternary_infer: bool = False,
    use_kv_cache_decode: bool = False,
    use_batched_probe_eval: bool = False,
    probe_batch_size: int = 32,
) -> dict:
    """Retention-anchor V0 Slice C finite-support audit (codex msg
    1779566905283-8ba63fe9 +1 implement with corrected baseline gate).

    Parallel surface to `probe_exhaustive_finite_supports` (math A0 = 1255)
    and `probe_language_finite_supports` (language L0a = 230). Iterates the
    full anchor set (currently `math_fragile_v1` = 21 entries), decodes via
    faststack path, aggregates strict + parsed totals, emits per-source-rung
    breakdown AND per-anchor-row records keyed by anchor_id (not question
    text, because the natural-dup of `what is 0 plus 0?` appears under both
    R1_zero_left and R1_zero_right buckets).

    Aggregate `expected_aggregate=21` is kept SEPARATE from math 1255 and
    language 230; no blended single-number aggregate.

    Anchor-set resolution per codex msg 1779566905283:
    - Default (no override): use ckpt's stored `retention_anchor_set` if
      present, else fall back to `math_fragile_v1`. Fallback source is
      printed/logged plainly so receipts are unambiguous.
    - Explicit `anchor_set_override`: use that name; if ckpt has a recorded
      set and it differs, WARN + record both values + flag mismatch.
    - JSON always records `anchor_set` (resolved), `ckpt_anchor_set` (what
      ckpt has, or None), `anchor_set_mismatch` (bool).

    Returns dict with `ckpt_path`, `ckpt_step`, `device`, `anchor_set`,
    `ckpt_anchor_set`, `anchor_set_mismatch`, `active_anchor_buckets`,
    per-set `results` (each with `n_total`, `n_exact`,
    `n_parsed_correct`, `by_source_rung`, `rows` keyed by anchor_id,
    `holes_first20`), `aggregate`, `elapsed_s`, flag echo.
    """
    from calm.hrm_text_158.curriculum.retention_anchors import (
        RETENTION_ANCHOR_SETS,
        anchor_set_source_rung_buckets,
        load_anchor_set,
    )

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if use_batched_probe_eval and not use_kv_cache_decode:
        raise ValueError(
            "--use-batched-probe-eval requires --use-kv-cache-decode "
            "(batched path is built on top of the γ1 KV cache contract)"
        )
    if probe_batch_size < 1:
        raise ValueError(f"--probe-batch-size must be >= 1, got {probe_batch_size}")

    print(f"[probe-anchor] loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", -1)
    print(f"[probe-anchor] ckpt step={step}", flush=True)

    # Anchor-set resolution per codex msg 1779566905283: prefer ckpt's
    # stored `retention_anchor_set`; fall back to math_fragile_v1 when
    # absent; explicit override emits WARN on mismatch. Source must be
    # printed plainly so receipts are unambiguous (codex tightening).
    config = ckpt["config"]
    ckpt_anchor_set = config.get("retention_anchor_set")
    if anchor_set_override is None:
        if ckpt_anchor_set is not None and ckpt_anchor_set != "none":
            anchor_set = str(ckpt_anchor_set)
            print(
                f"[probe-anchor] anchor_set={anchor_set} "
                f"(source=ckpt.retention_anchor_set)",
                flush=True,
            )
        else:
            anchor_set = "math_fragile_v1"
            print(
                f"[probe-anchor] anchor_set={anchor_set} "
                f"(source=fallback; ckpt has no retention_anchor_set recorded — "
                f"this is a baseline audit, NOT a trained-anchor check)",
                flush=True,
            )
    else:
        anchor_set = str(anchor_set_override)
        if (
            ckpt_anchor_set is not None
            and ckpt_anchor_set != "none"
            and str(ckpt_anchor_set) != anchor_set
        ):
            print(
                f"[probe-anchor] WARN: --anchor-set={anchor_set} differs from "
                f"ckpt.retention_anchor_set={ckpt_anchor_set!r}; audit will use "
                f"{anchor_set} but rows may not match the set this ckpt was "
                f"trained against.",
                flush=True,
            )
        else:
            print(
                f"[probe-anchor] anchor_set={anchor_set} "
                f"(source=explicit override)",
                flush=True,
            )

    if anchor_set not in RETENTION_ANCHOR_SETS:
        raise ValueError(
            f"unknown anchor_set {anchor_set!r}; valid: "
            f"{tuple(RETENTION_ANCHOR_SETS)}"
        )

    anchor_rows = load_anchor_set(anchor_set)

    m, tok = _build_model_from_ckpt(ckpt, device)
    max_seq_len = ckpt["config"]["max_seq_len"]

    if use_cached_ternary_infer:
        from calm.hrm_text_158.bit_linear import freeze_bitlinears_for_inference
        n_frozen = freeze_bitlinears_for_inference(m)
        print(f"[probe-anchor] cached-ternary-infer: froze {n_frozen} "
              f"BitLinear modules", flush=True)

    if use_batched_probe_eval:
        dispatch_path = "batched_kv_cache"
    elif use_kv_cache_decode:
        dispatch_path = "scalar_kv_cache"
    else:
        dispatch_path = "scalar_no_cache"
    print(f"[probe-anchor] decode dispatch: {dispatch_path}", flush=True)

    decode_fn = _decode_greedy_cached if use_kv_cache_decode else _decode_greedy_no_cache

    def _parse_int(text: str) -> int | None:
        capped = re.sub(r"(\d{12})\d+", r"\1", text)
        m_ = re.search(r"-?\d+", capped)
        return int(m_.group(0)) if m_ else None

    def _decode_rows(qs: list[str]) -> list[tuple[str, bool, bool]]:
        if use_batched_probe_eval:
            per_row, _hist = _run_rows_batched(
                m, tok, qs,
                max_gen=max_gen, max_seq_len=max_seq_len, device=device,
                batch_size=probe_batch_size,
            )
            return per_row
        return [
            decode_fn(m, tok, q, max_gen=max_gen,
                      max_seq_len=max_seq_len, device=device)
            for q in qs
        ]

    t0 = time.time()
    questions = [r.question for r in anchor_rows]
    expecteds = [r.expected for r in anchor_rows]
    sources = [r.source_rung for r in anchor_rows]
    ids = [r.anchor_id for r in anchor_rows]
    per_row = _decode_rows(questions)

    # Per-source-rung accumulators (canonical bucket order)
    bucket_keys = anchor_set_source_rung_buckets(anchor_set)
    by_source: dict[str, dict] = {
        b: {"n_total": 0, "n_exact": 0, "n_parsed_correct": 0, "n_holes": 0}
        for b in bucket_keys
    }

    # Per-anchor-row records keyed by anchor_id (full 21 rows preserved,
    # including the natural-dup of `what is 0 plus 0?`).
    rows_out: list[dict] = []
    holes: list[dict] = []
    finite_all = True
    too_long = 0
    exact = 0
    parsed_correct = 0

    for aid, q, exp, src, (decoded, tl, fin) in zip(
        ids, questions, expecteds, sources, per_row
    ):
        if not fin:
            finite_all = False
        exact_match = (not tl) and (decoded.strip() == str(exp))
        parsed = _parse_int(decoded) if not tl else None
        parsed_match = (not tl) and parsed == exp
        by_source[src]["n_total"] += 1
        row_record = {
            "anchor_id": aid,
            "question": q,
            "expected": exp,
            "decoded": decoded,
            "parsed": parsed,
            "exact_ok": bool(exact_match),
            "parsed_ok": bool(parsed_match),
            "too_long": bool(tl),
            "finite": bool(fin),
            "source_rung": src,
        }
        rows_out.append(row_record)
        if tl:
            too_long += 1
            by_source[src]["n_holes"] += 1
            holes.append(row_record)
            continue
        if parsed_match:
            parsed_correct += 1
            by_source[src]["n_parsed_correct"] += 1
        if exact_match:
            exact += 1
            by_source[src]["n_exact"] += 1
        else:
            by_source[src]["n_holes"] += 1
            holes.append(row_record)

    n_total = len(anchor_rows)
    set_result = {
        "n_total": n_total,
        "n_exact": exact,
        "n_parsed_correct": parsed_correct,
        "rate": exact / n_total if n_total else 1.0,
        "n_holes": len(holes),
        "n_too_long": too_long,
        "finite": finite_all,
        "by_source_rung": by_source,
        "rows": rows_out,
        "holes_first20": holes[:20],
    }
    total_elapsed = time.time() - t0
    set_result["elapsed_s"] = round(total_elapsed, 3)

    print(f"[probe-anchor] {anchor_set:18s} {exact}/{n_total} = "
          f"{set_result['rate']:.4f} (strict) parsed={parsed_correct}/{n_total} "
          f"holes={len(holes)} too_long={too_long} "
          f"finite={finite_all} t={total_elapsed:.2f}s", flush=True)
    for b in bucket_keys:
        bs = by_source[b]
        print(f"[probe-anchor]   bucket {b:14s} "
              f"strict={bs['n_exact']}/{bs['n_total']} "
              f"parsed={bs['n_parsed_correct']}/{bs['n_total']} "
              f"holes={bs['n_holes']}", flush=True)

    aggregate = {
        "n_total": n_total,
        "n_exact": exact,
        "n_parsed_correct": parsed_correct,
        "rate": exact / n_total if n_total else 1.0,
        "n_holes": len(holes),
        "finite": finite_all,
        "expected_aggregate": 21,  # math_fragile_v1 fixed-size set
    }
    print(f"[probe-anchor] ANCHOR AGGREGATE strict={exact}/{n_total} = "
          f"{aggregate['rate']:.4f} parsed={parsed_correct}/{n_total} "
          f"holes={len(holes)} finite={finite_all} "
          f"elapsed={total_elapsed:.1f}s", flush=True)

    output = {
        "ckpt_path": str(ckpt_path),
        "ckpt_step": int(step) if step != -1 else None,
        "device": device,
        "anchor_set": anchor_set,
        "ckpt_anchor_set": (
            str(ckpt_anchor_set)
            if ckpt_anchor_set is not None and ckpt_anchor_set != "none"
            else None
        ),
        "anchor_set_mismatch": (
            ckpt_anchor_set is not None
            and ckpt_anchor_set != "none"
            and str(ckpt_anchor_set) != anchor_set
        ),
        "active_anchor_buckets": bucket_keys,
        "flags": {
            "use_cached_ternary_infer": use_cached_ternary_infer,
            "use_kv_cache_decode": use_kv_cache_decode,
            "use_batched_probe_eval": use_batched_probe_eval,
            "probe_batch_size": probe_batch_size,
            "max_gen": max_gen,
        },
        "dispatch_path": dispatch_path,
        "results": {anchor_set: set_result},
        "aggregate": aggregate,
        "elapsed_s": round(total_elapsed, 2),
    }
    if output_json:
        Path(output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(output, f, indent=2)
        print(f"[probe-anchor] wrote {output_json}", flush=True)
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="HRM-Text-1.58 probe.")
    from calm.hrm_text_158.curriculum.retention_anchors import (
        RETENTION_ANCHOR_SETS as _ANCHOR_SETS,
    )
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
    # Exhaustive finite-support audit flags per codex msg 1779552750209-3218959b
    # after R1b8 commit 1a14a09 where A0 exhaustive caught the digit-7 cluster
    # + 0-plus-N cluster + R1b2 boundary that sampled probes hid. Promoted
    # from /tmp helper to committed tooling.
    ap.add_argument("--exhaustive-finite-supports", action="store_true",
                    help="Run exhaustive finite-support audit on the active "
                         "math chain (R0..R1b9, aggregate 1255; R1b10 PARKED) "
                         "instead of sampled per-rung "
                         "probe. Conflicts with --curriculum-rungs. Per-rung "
                         "supports built via "
                         "calm.hrm_text_158.curriculum.exhaustive_supports."
                         "build_exhaustive_supports.")
    ap.add_argument("--watch-rows-json", type=str, default=None,
                    help="Optional path to JSON file listing watch-rows for "
                         "boundary checking under --exhaustive-finite-supports. "
                         "Each entry must be {key: str, question: str, "
                         "expected: int}. Schema validated BEFORE ckpt load "
                         "(fails loud per codex 1779552750209 guardrail).")
    ap.add_argument("--audit-output-json", type=str, default=None,
                    help="Path to write exhaustive audit JSON (per-rung + "
                         "aggregate + watch_rows). Required for --exhaustive-"
                         "finite-supports if you want machine-readable output.")
    # Language-axis audit flags per codex msg 1779559495228-f863199b +1 implement
    # L0a (first language rung). Slice 2 per codex msg 1779560726491. Parallel
    # to --exhaustive-finite-supports; emits a separate `language` section in
    # the audit JSON. Math A0 export remains pure.
    ap.add_argument("--language-supports", action="store_true",
                    help="Run language-axis finite-support audit on the active "
                         "language rungs (currently L0a = `what's <math>?` "
                         "paraphrase wrapper over R0..R1b9 primitives, 230 "
                         "rows). Emits per-source-rung breakdown. Conflicts "
                         "with --curriculum-rungs and --exhaustive-finite-supports.")
    ap.add_argument("--language-audit-seed", type=int, default=None,
                    help="Explicit seed for language-axis support sampling. "
                         "Defaults to ckpt's stored `curriculum_seed` config "
                         "(matches training-side L0a partition). If ckpt has "
                         "no curriculum_seed AND this flag is omitted, the "
                         "probe fails BEFORE ckpt load. Mismatch with ckpt "
                         "seed warns and records both values in audit JSON.")
    # Retention-anchor V0 Slice C (codex msg 1779566905283-8ba63fe9 +1
    # implement). Parallel surface to --exhaustive-finite-supports and
    # --language-supports. Emits an `anchor` JSON section with
    # aggregate.expected_aggregate=21 (separate from math 1255 + language 230).
    ap.add_argument("--anchor-audit", action="store_true",
                    help="Run retention-anchor V0 finite-support audit on the "
                         "current anchor set (currently `math_fragile_v1` = 21 "
                         "entries: R1b2 known-fragile row + R1 zero-left + R1 "
                         "zero-right). Emits per-source-rung breakdown + per-"
                         "anchor-row records keyed by anchor_id. Conflicts with "
                         "--curriculum-rungs, --exhaustive-finite-supports, "
                         "--language-supports.")
    ap.add_argument("--anchor-set", type=str, default=None,
                    choices=sorted(_ANCHOR_SETS),
                    help="Explicit anchor-set override for --anchor-audit. "
                         "Default resolution uses ckpt's stored "
                         "`retention_anchor_set` config field; falls back to "
                         "`math_fragile_v1` when ckpt has none (fallback source "
                         "is printed plainly so receipts are unambiguous). "
                         "Mismatch with ckpt's recorded set warns and records "
                         "both values in audit JSON.")
    # L0c1 precursor audit (codex msg 1779636434289-de29e525 +1 Slice F.1).
    # SEPARATE surface from --language-supports: audits the one_digit-stratum
    # subset of L0c (121 rows) and emits surface='l0c1' JSON with aggregate 121,
    # NOT blended into the canonical 690 language aggregate.
    ap.add_argument("--l0c1-audit", action="store_true",
                    help="Run the standalone L0c1 precursor finite-support "
                         "audit (one_digit-stratum subset of L0c, 121 rows, "
                         "same `<expr> equals what?` template). Emits a "
                         "separate surface='l0c1' JSON section (aggregate 121); "
                         "NOT blended into --language-supports / the 690 "
                         "language aggregate. Conflicts with --language-supports, "
                         "--exhaustive-finite-supports, --anchor-audit, "
                         "--curriculum-rungs.")
    # L0c2 bounded-2-digit stair-step audit (F.4-audit). SEPARATE surface like
    # L0c1: audits the 230-row F.4a hard subset, emits surface='l0c2' JSON
    # (aggregate 230) with a per-(source_rung:operator) COMPOSITE-bucket
    # breakdown so the R1b2:minus / `10 minus 1 -> 9` failure class stays
    # visible; NOT blended into the canonical 690 language aggregate.
    ap.add_argument("--l0c2-audit", action="store_true",
                    help="Run the standalone L0c2 bounded-2-digit stair-step "
                         "finite-support audit (230 rows, all 2-digit-hard, "
                         "same `<expr> equals what?` template). Emits a separate "
                         "surface='l0c2' JSON section (aggregate 230) with 12 "
                         "composite source_rung:operator buckets (keeps the "
                         "R1b2:minus failure class visible); NOT blended into "
                         "--language-supports / the 690 aggregate. Conflicts with "
                         "the other audit modes.")
    # F.4d K-magnitude band audits: three explicit surfaces over the existing
    # L0c2 pool, never blended into the canonical language aggregate.
    ap.add_argument("--l0c2k1-audit", action="store_true",
                    help="Run the standalone L0c2-K1 audit surface (result "
                         "magnitude 10-19 plus the 10 minus 1 -> 9 singleton; "
                         "seed-42 aggregate 24). Conflicts with other modes.")
    ap.add_argument("--l0c2k2-audit", action="store_true",
                    help="Run the standalone L0c2-K2 audit surface (result "
                         "magnitude 20-49; seed-42 aggregate 79). Conflicts "
                         "with other modes.")
    ap.add_argument("--l0c2k2-addition-full-audit", action="store_true",
                    help="Run the standalone L0c2-K2-addition-full coverage "
                         "audit: 240 trainable acquisition rows over "
                         "'<a> plus <k> equals what?' for results 20-49 and "
                         "k=1..8. Emits surface='l0c2k2additionfull'. "
                         "Conflicts with other modes.")
    ap.add_argument("--l0c2k2-addition-120-audit", action="store_true",
                    help="Run the standalone L0c2-K2-addition-120 coverage "
                         "audit: 120 trainable acquisition rows (2x-density "
                         "k=1..4 subset of the 240 surface) over '<a> plus <k> "
                         "equals what?' for results 20-49 and k=1..4. Emits "
                         "surface='l0c2k2addition120'. Conflicts with other modes.")
    ap.add_argument("--l0c2k2-addition-120-k5to8-audit", action="store_true",
                    help="Run the standalone L0c2-K2-addition-120-k5to8 coverage "
                         "audit: 120 trainable acquisition rows (SECOND 2x-density "
                         "atom, k=5..8 subset of the 240 surface, DISJOINT from the "
                         "banked k=1..4 atom) over '<a> plus <k> equals what?' for "
                         "results 20-49 and k=5..8. Emits "
                         "surface='l0c2k2addition120k5to8'. Conflicts with other modes.")
    ap.add_argument("--l0c2k2-addition-heldout-50s-audit", action="store_true",
                    help="Run the trained-OUT L0c2-K2 addition heldout-50s "
                         "diagnostic audit: 80 non-gating rows over results "
                         "50-59 and k=1..8. Audit-visible only; not a rung, "
                         "not retained. Emits surface='l0c2k2additionheldout50s'. "
                         "Conflicts with other modes.")
    ap.add_argument("--l0c2k3-audit", action="store_true",
                    help="Run the standalone L0c2-K3 audit surface (result "
                         "magnitude 50-99; seed-42 aggregate 127). Conflicts "
                         "with other modes.")
    # F.4d-edge held-generalization micro-slice audit. SEPARATE dense 65-row
    # surface (NOT a filter of L0c2); two finite sub-surfaces (train 52 / held
    # 13) so the gate reports per surface, held bucket axis = legacy(4)/fresh(9).
    ap.add_argument("--l0c2k1-edge-audit", action="store_true",
                    help="Run the standalone L0c2-K1-edge audit surface: dense "
                         "same-template held-generalization micro-slice (65 rows "
                         "= 52 train + 13 held). Emits surface='l0c2k1edge' with "
                         "two sub-surfaces (train 52/52, held 13/13; held buckets "
                         "legacy/fresh). Counts are seed-independent. Conflicts "
                         "with the other audit modes.")
    # F.4d-identity: standalone suffix-copy precursor audit. Mirrors the
    # K1-edge pattern with train/held sub-surfaces (70 / 20), never blended
    # into the canonical language aggregate.
    ap.add_argument("--l0c2k1-identity-audit", action="store_true",
                    help="Run the standalone L0c2-K1-identity-2digit audit "
                         "surface: 90 identity rows over '<n> equals what?' "
                         "(70 train + 20 held). Emits surface='l0c2k1identity' "
                         "with train/held sub-surfaces. Conflicts with the "
                         "other audit modes.")
    # F.4d-identity-full: full-density 90/90 coverage audit for the
    # emission-primitive rung L0c2-K1-identity-2digit-full (all identities
    # 10..99 trained, no held sub-surface). Emits surface='l0c2k1identityfull';
    # the aggregate token L0C2K1IDENTITYFULL is trailing-space anchored so it
    # never cross-matches L0C2K1IDENTITY (same discipline as K1EDGE vs K1).
    ap.add_argument("--l0c2k1-identity-full-audit", action="store_true",
                    help="Run the full-density L0c2-K1-identity-2digit-full "
                         "coverage audit: all 90 identity rows over "
                         "'<n> equals what?' (10..99, train-only, no held). "
                         "Emits surface='l0c2k1identityfull' (aggregate 90). "
                         "Conflicts with the other audit modes.")
    # Exhaustive-L0c language-density audit (codex msg 1779693537447 / Slice:
    # language-to-math-density). The `<expr> equals what?` wrapper over the
    # FULL math-A0 set (1255). Reuses the exhaustive audit machinery via
    # support_builder + watch_row_transform; emits label `probe-l0c-exhaustive`
    # (aggregate 1255), with the config watch row mapped onto the L0c surface.
    ap.add_argument("--l0c-exhaustive-audit", action="store_true",
                    help="Run the exhaustive-L0c language-density audit: the "
                         "`<expr> equals what?` wrapper over the full math-A0 "
                         "exhaustive set (1255 rows). Per-source-rung breakdown "
                         "parallel to math A0; watch rows mapped to L0c surface "
                         "so config.watch_rows decode (not NOT_IN_ACTIVE). "
                         "Conflicts with the other audit modes.")
    args = ap.parse_args()

    # Pre-checks BEFORE ckpt load (codex 1779552750209 guardrail: fail loud
    # on bad CLI/schema, never after expensive ckpt load).
    if args.exhaustive_finite_supports and args.curriculum_rungs is not None:
        raise SystemExit(
            "ERROR: --exhaustive-finite-supports conflicts with "
            "--curriculum-rungs (mutually exclusive); pass only one."
        )
    if args.language_supports and args.curriculum_rungs is not None:
        raise SystemExit(
            "ERROR: --language-supports conflicts with --curriculum-rungs "
            "(mutually exclusive); pass only one."
        )
    if args.language_supports and args.exhaustive_finite_supports:
        raise SystemExit(
            "ERROR: --language-supports conflicts with --exhaustive-finite-supports "
            "(mutually exclusive — math and language are separate probe modes). "
            "Run them as two separate invocations and combine the JSON output "
            "in the receipt."
        )
    # Slice C: 3 mutex checks for --anchor-audit (codex msg 1779566905283).
    if args.anchor_audit and args.curriculum_rungs is not None:
        raise SystemExit(
            "ERROR: --anchor-audit conflicts with --curriculum-rungs "
            "(mutually exclusive); pass only one."
        )
    if args.anchor_audit and args.exhaustive_finite_supports:
        raise SystemExit(
            "ERROR: --anchor-audit conflicts with --exhaustive-finite-supports "
            "(mutually exclusive — anchor and math A0 are separate probe modes). "
            "Run them as two separate invocations and combine the JSON output "
            "in the receipt."
        )
    if args.anchor_audit and args.language_supports:
        raise SystemExit(
            "ERROR: --anchor-audit conflicts with --language-supports "
            "(mutually exclusive — anchor and language L0a are separate probe modes). "
            "Run them as two separate invocations and combine the JSON output "
            "in the receipt."
        )
    # Slice F.1: 4 mutex checks for --l0c1-audit (codex msg 1779636434289).
    if args.l0c1_audit and args.curriculum_rungs is not None:
        raise SystemExit(
            "ERROR: --l0c1-audit conflicts with --curriculum-rungs "
            "(mutually exclusive); pass only one."
        )
    if args.l0c1_audit and args.exhaustive_finite_supports:
        raise SystemExit(
            "ERROR: --l0c1-audit conflicts with --exhaustive-finite-supports "
            "(mutually exclusive — L0c1 precursor and math A0 are separate probe "
            "modes). Run them separately and combine the JSON in the receipt."
        )
    if args.l0c1_audit and args.language_supports:
        raise SystemExit(
            "ERROR: --l0c1-audit conflicts with --language-supports "
            "(mutually exclusive — L0c1 is a SEPARATE precursor surface, NOT "
            "blended into the 690 language aggregate). Run them separately."
        )
    if args.l0c1_audit and args.anchor_audit:
        raise SystemExit(
            "ERROR: --l0c1-audit conflicts with --anchor-audit "
            "(mutually exclusive — separate probe modes). Run them separately."
        )
    # F.4-audit: mutex checks for --l0c2-audit (mirror --l0c1-audit; L0c2 is a
    # SEPARATE bounded-2-digit surface, never blended into the 690 aggregate).
    if args.l0c2_audit and args.curriculum_rungs is not None:
        raise SystemExit(
            "ERROR: --l0c2-audit conflicts with --curriculum-rungs "
            "(mutually exclusive); pass only one."
        )
    if args.l0c2_audit and args.exhaustive_finite_supports:
        raise SystemExit(
            "ERROR: --l0c2-audit conflicts with --exhaustive-finite-supports "
            "(mutually exclusive — separate probe modes). Run them separately."
        )
    if args.l0c2_audit and args.language_supports:
        raise SystemExit(
            "ERROR: --l0c2-audit conflicts with --language-supports "
            "(mutually exclusive — L0c2 is a SEPARATE surface, NOT blended into "
            "the 690 language aggregate). Run them separately."
        )
    if args.l0c2_audit and args.anchor_audit:
        raise SystemExit(
            "ERROR: --l0c2-audit conflicts with --anchor-audit "
            "(mutually exclusive — separate probe modes). Run them separately."
        )
    if args.l0c2_audit and args.l0c1_audit:
        raise SystemExit(
            "ERROR: --l0c2-audit conflicts with --l0c1-audit "
            "(mutually exclusive — two separate bounded surfaces). Run them "
            "separately and combine JSON in the receipt."
        )
    _l0c2k_flags = [
        ("--l0c2k1-audit", args.l0c2k1_audit),
        ("--l0c2k2-audit", args.l0c2k2_audit),
        ("--l0c2k2-addition-full-audit", args.l0c2k2_addition_full_audit),
        ("--l0c2k2-addition-120-audit", args.l0c2k2_addition_120_audit),
        ("--l0c2k2-addition-120-k5to8-audit", args.l0c2k2_addition_120_k5to8_audit),
        ("--l0c2k2-addition-heldout-50s-audit", args.l0c2k2_addition_heldout_50s_audit),
        ("--l0c2k3-audit", args.l0c2k3_audit),
        ("--l0c2k1-edge-audit", args.l0c2k1_edge_audit),
        ("--l0c2k1-identity-audit", args.l0c2k1_identity_audit),
        ("--l0c2k1-identity-full-audit", args.l0c2k1_identity_full_audit),
    ]
    for _flag, _on in _l0c2k_flags:
        if not _on:
            continue
        _conflicts = [
            ("--curriculum-rungs", args.curriculum_rungs is not None),
            ("--exhaustive-finite-supports", args.exhaustive_finite_supports),
            ("--language-supports", args.language_supports),
            ("--anchor-audit", args.anchor_audit),
            ("--l0c1-audit", args.l0c1_audit),
            ("--l0c2-audit", args.l0c2_audit),
            ("--l0c-exhaustive-audit", args.l0c_exhaustive_audit),
        ] + [(_other_flag, _other_on) for _other_flag, _other_on in _l0c2k_flags if _other_flag != _flag]
        _hit = [name for name, hit in _conflicts if hit]
        if _hit:
            raise SystemExit(
                f"ERROR: {_flag} conflicts with {', '.join(_hit)} "
                "(mutually exclusive — separate L0c2 K-band audit surfaces). "
                "Run them separately and combine JSON in the receipt."
            )
    # --l0c-exhaustive-audit is mutually exclusive with every other audit mode
    # (codex msg 1779694143993): the dispatch order (anchor -> l0c1 -> language
    # -> l0c_exhaustive -> exhaustive -> curriculum) would otherwise let a
    # co-passed mode silently win. Fail fast BEFORE ckpt load.
    if args.l0c_exhaustive_audit:
        _l0ce_conflicts = [
            ("--curriculum-rungs", args.curriculum_rungs is not None),
            ("--language-supports", args.language_supports),
            ("--exhaustive-finite-supports", args.exhaustive_finite_supports),
            ("--anchor-audit", args.anchor_audit),
            ("--l0c1-audit", args.l0c1_audit),
            ("--l0c2-audit", args.l0c2_audit),
            ("--l0c2k1-audit", args.l0c2k1_audit),
            ("--l0c2k2-audit", args.l0c2k2_audit),
            ("--l0c2k2-addition-full-audit", args.l0c2k2_addition_full_audit),
            ("--l0c2k2-addition-heldout-50s-audit", args.l0c2k2_addition_heldout_50s_audit),
            ("--l0c2k3-audit", args.l0c2k3_audit),
            ("--l0c2k1-edge-audit", args.l0c2k1_edge_audit),
            ("--l0c2k1-identity-audit", args.l0c2k1_identity_audit),
            ("--l0c2k1-identity-full-audit", args.l0c2k1_identity_full_audit),
        ]
        _l0ce_hit = [name for name, on in _l0ce_conflicts if on]
        if _l0ce_hit:
            raise SystemExit(
                f"ERROR: --l0c-exhaustive-audit conflicts with "
                f"{', '.join(_l0ce_hit)} (mutually exclusive — separate probe "
                "modes). Run them separately and combine JSON in the receipt."
            )
    if args.use_batched_probe_eval and not args.use_kv_cache_decode:
        # Mirror existing pre-check from probe_curriculum so exhaustive mode
        # fails consistently before ckpt load.
        raise SystemExit(
            "ERROR: --use-batched-probe-eval requires --use-kv-cache-decode "
            "(fails fast for --curriculum-rungs, --exhaustive-finite-supports, "
            "and --language-supports modes)."
        )

    if args.anchor_audit:
        probe_anchor_finite_supports(
            args.ckpt_path,
            anchor_set_override=args.anchor_set,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
        )
    elif args.l0c1_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c1_support,
            L0C1_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c1_support,
            expected_aggregate=L0C1_EXPECTED_COUNT,
            surface="l0c1",
        )
    elif args.l0c2_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2_support,
            L0C2_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2_support,
            expected_aggregate=L0C2_AUDIT_EXPECTED_COUNT,
            surface="l0c2",
        )
    elif args.l0c2k1_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k1_support,
            l0c2_band_audit_expected_count,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k1_support,
            expected_aggregate_fn=lambda s: l0c2_band_audit_expected_count(s, "K1"),
            surface="l0c2k1",
        )
    elif args.l0c2k2_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k2_support,
            l0c2_band_audit_expected_count,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k2_support,
            expected_aggregate_fn=lambda s: l0c2_band_audit_expected_count(s, "K2"),
            surface="l0c2k2",
        )
    elif args.l0c2k2_addition_full_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k2_addition_full_support,
            L0C2K2_ADDITION_FULL_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k2_addition_full_support,
            expected_aggregate=L0C2K2_ADDITION_FULL_AUDIT_EXPECTED_COUNT,
            surface="l0c2k2additionfull",
        )
    elif args.l0c2k2_addition_120_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k2_addition_120_support,
            L0C2K2_ADDITION_120_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k2_addition_120_support,
            expected_aggregate=L0C2K2_ADDITION_120_AUDIT_EXPECTED_COUNT,
            surface="l0c2k2addition120",
        )
    elif args.l0c2k2_addition_120_k5to8_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k2_addition_120_k5to8_support,
            L0C2K2_ADDITION_120_K5TO8_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k2_addition_120_k5to8_support,
            expected_aggregate=L0C2K2_ADDITION_120_K5TO8_AUDIT_EXPECTED_COUNT,
            surface="l0c2k2addition120k5to8",
        )
    elif args.l0c2k2_addition_heldout_50s_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k2_addition_heldout_50s_support,
            L0C2K2_ADDITION_HELDOUT_50S_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k2_addition_heldout_50s_support,
            expected_aggregate=L0C2K2_ADDITION_HELDOUT_50S_AUDIT_EXPECTED_COUNT,
            surface="l0c2k2additionheldout50s",
        )
    elif args.l0c2k3_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k3_support,
            l0c2_band_audit_expected_count,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k3_support,
            expected_aggregate_fn=lambda s: l0c2_band_audit_expected_count(s, "K3"),
            surface="l0c2k3",
        )
    elif args.l0c2k1_edge_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k1_edge_support,
            L0C2K1_EDGE_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k1_edge_support,
            expected_aggregate=L0C2K1_EDGE_AUDIT_EXPECTED_COUNT,
            surface="l0c2k1edge",
        )
    elif args.l0c2k1_identity_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k1_identity_support,
            L0C2K1_IDENTITY_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k1_identity_support,
            expected_aggregate=L0C2K1_IDENTITY_AUDIT_EXPECTED_COUNT,
            surface="l0c2k1identity",
        )
    elif args.l0c2k1_identity_full_audit:
        from calm.hrm_text_158.curriculum.language_supports import (
            build_l0c2k1_identity_full_support,
            L0C2K1_IDENTITY_FULL_AUDIT_EXPECTED_COUNT,
        )
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            supports_builder=build_l0c2k1_identity_full_support,
            expected_aggregate=L0C2K1_IDENTITY_FULL_AUDIT_EXPECTED_COUNT,
            surface="l0c2k1identityfull",
        )
    elif args.language_supports:
        probe_language_finite_supports(
            args.ckpt_path,
            audit_seed=args.language_audit_seed,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
        )
    elif args.l0c_exhaustive_audit:
        # Exhaustive L0c language-density audit (codex msg 1779693537447):
        # same machinery as math A0 via support_builder + watch_row_transform.
        # count 1255, label probe-l0c-exhaustive; watch row mapped to L0c
        # surface so the banked config watch row decodes (not NOT_IN_ACTIVE).
        from calm.hrm_text_158.curriculum.exhaustive_supports import (
            validate_watch_rows,
        )
        from calm.hrm_text_158.curriculum.language_supports import (
            build_exhaustive_l0c_supports,
            L0C_EXHAUSTIVE_EXPECTED_COUNT,
        )
        watch_rows = []
        if args.watch_rows_json:
            with open(args.watch_rows_json) as f:
                watch_rows = validate_watch_rows(json.load(f))
        probe_exhaustive_finite_supports(
            args.ckpt_path,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            watch_rows=watch_rows,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
            support_builder=build_exhaustive_l0c_supports,
            expected_aggregate=L0C_EXHAUSTIVE_EXPECTED_COUNT,
            label="probe-l0c-exhaustive",
            watch_row_transform=_l0c_watch_transform,
        )
    elif args.exhaustive_finite_supports:
        # Validate watch-rows JSON schema BEFORE ckpt load
        from calm.hrm_text_158.curriculum.exhaustive_supports import (
            validate_watch_rows,
        )
        watch_rows: list[dict] = []
        if args.watch_rows_json:
            with open(args.watch_rows_json) as f:
                watch_rows = validate_watch_rows(json.load(f))
        probe_exhaustive_finite_supports(
            args.ckpt_path,
            max_gen=args.max_gen,
            output_json=args.audit_output_json,
            watch_rows=watch_rows,
            use_cached_ternary_infer=args.use_cached_ternary_infer,
            use_kv_cache_decode=args.use_kv_cache_decode,
            use_batched_probe_eval=args.use_batched_probe_eval,
            probe_batch_size=args.probe_batch_size,
        )
    elif args.curriculum_rungs is not None:
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
