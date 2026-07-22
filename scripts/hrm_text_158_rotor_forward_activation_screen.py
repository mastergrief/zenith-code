"""Phase 1 — rotor forward-activation tolerance screen (ternary-rotor lane).

Plan: zenith-code `.claude/MEMORY/ternary-rotor.md` §Phase 1.

Hypothesis: HRM-Text-1.58 forward passes tolerate rotated 3-bit (then 2-bit)
activation fake-quantization at the residual seams with ≤ε degradation on
banked-support probes. Inference only; NO training; NO model edits — the
quantizer is injected through the existing `activation_codec_seam` kwarg.

Conditions (one variable = seam-set × width; candidate-gen path unchanged):
  baseline      — no seam
  t3_post_mlp   — 3-bit at residual.post_mlp only
  t3_all        — 3-bit at residual.post_attn + residual.post_mlp
  t2_post_mlp   — 2-bit at residual.post_mlp only
  t2_all        — 2-bit at residual.post_attn + residual.post_mlp
(z_L/z_H recurrent-state and attn.gqkv.* seams pass through untouched — they
are separate surfaces: Phase 2 covers KV.)

`--surface kv` runs the Phase 2 screen instead: rotated K (post-rope,
matching KV-cache storage semantics) + V at 3-bit / 2-bit, with the
analogous prereg branches (kv_at_2bit / kv_at_3bit_packing_mandatory /
kv_park_null). Same thresholds, same eval set.

Evaluation set (fixed, deterministic): 6 rows per active exhaustive rung
(seed-17 stratified sample of the A0 supports) + the canonical 17×23 row.
Metrics per condition: strict-exact, parsed-correct, too_long/finite, and
teacher-forced mean answer-token CE (response tokens only).

PRE-REGISTERED branch classifier (decided BEFORE launch; the all-seams
variant is the binding read for each width):
  clean(cond) :=  strict_exact(cond) >= strict_exact(baseline) - 2 rows
                  AND mean_ce(cond) - mean_ce(baseline) <= 0.10 nats
  branch a: clean(t2_all)                      -> "phase2_at_2bit"
  branch b: !clean(t2_all) AND clean(t3_all)   -> "3bit_lane_packing_mandatory_for_sub2"
  branch c: !clean(t3_all)                     -> "representation_not_viable_at_hrm_scale_park_null"

Usage:
  PYTHONPATH=. python3 scripts/hrm_text_158_rotor_forward_activation_screen.py \
    --ckpt-path calm/hrm/checkpoints/<banked-chain-head>.pt \
    --output-json artifacts/rotor/phase1_forward_activation_screen_receipt.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.rotor_runtime_quant import (  # noqa: E402
    rotor_bits_ledger,
    rotor_fake_quant,
)

ROWS_PER_RUNG = 6
SAMPLE_SEED = 17
STRICT_EXACT_DROP_MAX_ROWS = 2
CE_DELTA_MAX_NATS = 0.10

RESIDUAL_ALL = ("residual.post_attn", "residual.post_mlp")
RESIDUAL_POST_MLP = ("residual.post_mlp",)
KV_ALL = ("attn.gqkv.key_post_rope", "attn.gqkv.value")

# Per-surface condition sets. Binding (all-seams) condition per width is the
# one named in the branch classifier below.
SURFACE_CONDITIONS = {
    "residual": {
        "baseline": None,
        "t3_post_mlp": (3, RESIDUAL_POST_MLP),
        "t3_all": (3, RESIDUAL_ALL),
        "t2_post_mlp": (2, RESIDUAL_POST_MLP),
        "t2_all": (2, RESIDUAL_ALL),
    },
    # Phase 2: rotated K (post-rope, matching KV-cache storage) + V at
    # head_dim=128 = exactly one rotation group per head.
    "kv": {
        "baseline": None,
        "t3_kv": (3, KV_ALL),
        "t2_kv": (2, KV_ALL),
    },
    # Phase 4a: THE sub-2 KV read. 4-level codes are 2.0 bits flat — no
    # scale packing clears <2.0 scale-inclusive; the only sub-2 route is
    # 3-level (ternary) codes base-3 packed in the Q1_75 geometry
    # (26B codes + 2B fp16 scale per 128 = exactly 1.75 bpw).
    "kv_ternary": {
        "baseline": None,
        "tt_kv": ("ternary", KV_ALL),
    },
}

# Ordered branch ladder per surface: first clean binding wins; last entry is
# the park branch if none are clean.
SURFACE_BRANCHES = {
    "residual": (
        ("t2_all", "phase2_at_2bit"),
        ("t3_all", "3bit_lane_packing_mandatory_for_sub2"),
        (None, "representation_not_viable_at_hrm_scale_park_null"),
    ),
    "kv": (
        ("t2_kv", "kv_at_2bit"),
        ("t3_kv", "kv_at_3bit_packing_mandatory_for_sub2"),
        (None, "kv_representation_not_viable_park_null"),
    ),
    "kv_ternary": (
        ("tt_kv", "kv_sub2_at_ternary_1p75_q175_geometry"),
        (None, "kv_sub2_blocked_2bit_floor_park"),
    ),
}


def _load_probe_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probe_hrm_text_158.py")
    spec = importlib.util.spec_from_file_location("probe_hrm_text_158", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_seam(bits: int, families: tuple[str, ...]):
    fam = set(families)
    fired = {"count": 0}

    def seam(family, tensor, **_kwargs):
        if family in fam:
            fired["count"] += 1
            return rotor_fake_quant(tensor, bits=bits)
        return tensor

    return seam, fired


def _build_eval_rows() -> list[dict]:
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )
    import random

    supports = build_exhaustive_supports()
    rng = random.Random(SAMPLE_SEED)
    rows: list[dict] = []
    for rung in sorted(supports.keys()):
        pool = list(supports[rung])
        picks = rng.sample(pool, min(ROWS_PER_RUNG, len(pool)))
        for q, expected in picks:
            rows.append({"rung": rung, "question": q, "expected": int(expected)})
    rows.append({"rung": "canonical", "question": "what is 17 times 23?",
                 "expected": 391})
    return rows


def _decode_greedy_with_seam(m, tok, question: str, *, max_gen: int,
                             max_seq_len: int, device: str, extras: dict):
    """Greedy no-cache decode, mirroring probe `_decode_greedy_no_cache`,
    with seq_info extras (the seam) forwarded into the model call."""
    q_ids = tok.encode(question)
    prefix = [tok.bos_id] + q_ids + [tok.sep_id]
    if len(prefix) >= max_seq_len:
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
        pos = torch.arange(ids.shape[1], dtype=torch.long,
                           device=device).unsqueeze(0)
        with torch.no_grad():
            _carry, logits = m(
                None,
                {"inputs": ids, "sep_positions": sep_pos_t,
                 "position_ids": pos},
                **extras,
            )
        if not bool(torch.isfinite(logits).all().item()):
            finite = False
            break
        next_id = int(torch.argmax(logits[0, -1], dim=-1).item())
        if next_id == tok.eos_id:
            break
        out_tokens.append(next_id)
        cur.append(next_id)
    return tok.decode(out_tokens, stop_at_eos=False), False, finite


def _answer_ce(m, tok, question: str, expected: int, *, max_seq_len: int,
               device: str, extras: dict) -> float | None:
    """Teacher-forced mean CE (nats) over the gold answer tokens only."""
    q_ids = tok.encode(question)
    a_ids = tok.encode(str(expected))
    seq = [tok.bos_id] + q_ids + [tok.sep_id] + a_ids
    if len(seq) > max_seq_len:
        return None
    sep_pos = 1 + len(q_ids)
    ids = torch.tensor([seq], dtype=torch.long, device=device)
    sep_pos_t = torch.tensor([sep_pos], dtype=torch.long, device=device)
    pos = torch.arange(ids.shape[1], dtype=torch.long,
                       device=device).unsqueeze(0)
    with torch.no_grad():
        _carry, logits = m(
            None,
            {"inputs": ids, "sep_positions": sep_pos_t, "position_ids": pos},
            **extras,
        )
    # logits[t] predicts token t+1; answer tokens occupy positions
    # sep_pos+1 .. sep_pos+len(a_ids) so their predictors are sep_pos ..
    logp = torch.log_softmax(logits[0].float(), dim=-1)
    total = 0.0
    for j, tok_id in enumerate(a_ids):
        total += -float(logp[sep_pos + j, tok_id].item())
    return total / max(len(a_ids), 1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--max-gen", type=int, default=8)
    ap.add_argument("--device", default=None)
    ap.add_argument("--surface", choices=sorted(SURFACE_CONDITIONS),
                    default="residual")
    args = ap.parse_args()
    conditions = SURFACE_CONDITIONS[args.surface]
    branch_ladder = SURFACE_BRANCHES[args.surface]

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe = _load_probe_module()

    print(f"[rotor-p1] loading ckpt: {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", "?")
    m, tok = probe._build_model_from_ckpt(ckpt, device)
    m.eval()
    max_seq_len = ckpt["config"]["max_seq_len"]
    print(f"[rotor-p1] ckpt step={step} device={device} "
          f"max_seq_len={max_seq_len}", flush=True)

    rows = _build_eval_rows()
    print(f"[rotor-p1] eval rows: {len(rows)} "
          f"({ROWS_PER_RUNG}/rung seed={SAMPLE_SEED} + canonical)", flush=True)

    # Bits-ledger receipt for the screened surface shape (accounting, not a
    # claim): residual stream = hidden 512/token; KV = head_dim 128/head.
    if args.surface.startswith("kv"):
        ledger_surface, ledger_n = "attention_kv_attention_buffers", 128
    else:
        ledger_surface, ledger_n = "activations_residuals", 512
    if args.surface == "kv_ternary":
        ledger2 = rotor_bits_ledger(ledger_n, "ternary",
                                    surface=ledger_surface)
        ledger3 = rotor_bits_ledger(ledger_n, "ternary",
                                    surface=ledger_surface,
                                    scale_dtype="int8")
    else:
        ledger2 = rotor_bits_ledger(ledger_n, 2, surface=ledger_surface)
        ledger3 = rotor_bits_ledger(ledger_n, 3, surface=ledger_surface)

    results: dict[str, dict] = {}
    for cond, spec in conditions.items():
        if spec is None:
            extras: dict = {}
            fired = {"count": 0}
        else:
            bits, families = spec
            seam, fired = _make_seam(bits, families)
            extras = {"activation_codec_seam": seam}
        t0 = time.time()
        strict = parsed = too_long = nonfinite = 0
        ce_sum = 0.0
        ce_n = 0
        row_details = []
        for r in rows:
            decoded, tl, finite = _decode_greedy_with_seam(
                m, tok, r["question"], max_gen=args.max_gen,
                max_seq_len=max_seq_len, device=device, extras=extras)
            ce = _answer_ce(m, tok, r["question"], r["expected"],
                            max_seq_len=max_seq_len, device=device,
                            extras=extras)
            ok_exact = (decoded == str(r["expected"])) and not tl
            ok_parsed = (probe._parse_int(decoded) == r["expected"]) and not tl
            strict += int(ok_exact)
            parsed += int(ok_parsed)
            too_long += int(tl)
            nonfinite += int(not finite)
            if ce is not None:
                ce_sum += ce
                ce_n += 1
            row_details.append({
                "rung": r["rung"], "question": r["question"],
                "expected": r["expected"], "decoded": decoded,
                "exact_ok": ok_exact, "parsed_ok": ok_parsed, "ce": ce,
            })
        mean_ce = ce_sum / max(ce_n, 1)
        results[cond] = {
            "strict_exact": strict, "parsed": parsed, "n": len(rows),
            "too_long": too_long, "nonfinite": nonfinite,
            "mean_answer_ce_nats": mean_ce,
            "seam_fire_count": fired["count"],
            "elapsed_s": round(time.time() - t0, 1),
            "rows": row_details,
        }
        print(f"[rotor-p1] {cond:12s} strict={strict}/{len(rows)} "
              f"parsed={parsed}/{len(rows)} ce={mean_ce:.4f} "
              f"nonfinite={nonfinite} seam_fires={fired['count']} "
              f"t={results[cond]['elapsed_s']}s", flush=True)

    base = results["baseline"]

    def _clean(cond: str) -> bool:
        c = results[cond]
        return (c["strict_exact"] >= base["strict_exact"]
                - STRICT_EXACT_DROP_MAX_ROWS
                and c["mean_answer_ce_nats"] - base["mean_answer_ce_nats"]
                <= CE_DELTA_MAX_NATS
                and c["nonfinite"] == 0)

    branch = branch_ladder[-1][1]
    for cond_name, branch_name in branch_ladder[:-1]:
        if _clean(cond_name):
            branch = branch_name
            break

    print(f"[rotor-p1] === PREREG BRANCH: {branch} ===", flush=True)
    for cond in conditions:
        if cond == "baseline":
            continue
        d_strict = results[cond]["strict_exact"] - base["strict_exact"]
        d_ce = results[cond]["mean_answer_ce_nats"] - base["mean_answer_ce_nats"]
        print(f"[rotor-p1]   {cond:12s} d_strict={d_strict:+d} "
              f"d_ce={d_ce:+.4f} clean={_clean(cond)}", flush=True)

    receipt = {
        "screen": f"rotor_runtime_surface_tolerance/{args.surface}_v1",
        "surface": args.surface,
        "ckpt_path": args.ckpt_path,
        "ckpt_step": step,
        "device": device,
        "sample": {"rows_per_rung": ROWS_PER_RUNG, "seed": SAMPLE_SEED,
                   "n_rows": len(rows)},
        "prereg": {
            "strict_exact_drop_max_rows": STRICT_EXACT_DROP_MAX_ROWS,
            "ce_delta_max_nats": CE_DELTA_MAX_NATS,
            "branches": [b for _c, b in branch_ladder],
        },
        "bits_ledger": {"turbo2": ledger2.as_dict(),
                        "turbo3": ledger3.as_dict()},
        "conditions": {k: {kk: vv for kk, vv in v.items() if kk != "rows"}
                       for k, v in results.items()},
        "row_details": {k: v["rows"] for k, v in results.items()},
        "branch_verdict": branch,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"[rotor-p1] receipt -> {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
