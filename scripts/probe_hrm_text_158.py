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

    print(f"[probe-curriculum] loading ckpt: {ckpt_path}", flush=True)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", -1)
    print(f"[probe-curriculum] ckpt step={step}", flush=True)
    m, tok = _build_model_from_ckpt(ckpt, device)
    config = ckpt["config"]
    max_seq_len = config["max_seq_len"]
    n_params = sum(p.numel() for p in m.parameters())

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
        for ex in rows:
            decoded, tl, fin = _decode_greedy_no_cache(
                m, tok, ex["question"], max_gen=max_gen,
                max_seq_len=max_seq_len, device=device,
            )
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
    canonical_decoded, canonical_too_long, canonical_finite = _decode_greedy_no_cache(
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
        for ex in audit_rows:
            decoded, tl, fin = _decode_greedy_no_cache(
                m, tok, ex["question"], max_gen=max_gen,
                max_seq_len=max_seq_len, device=device,
            )
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
    args = ap.parse_args()

    if args.curriculum_rungs is not None:
        rungs = [r.strip() for r in args.curriculum_rungs.split(",") if r.strip()]
        probe_curriculum(
            args.ckpt_path,
            rungs=rungs,
            eval_cap=args.eval_cap,
            max_gen=args.max_gen,
            output_json=args.probe_output_json,
        )
    else:
        probe(args.ckpt_path, eval_cap=args.eval_cap, max_gen=args.max_gen)
