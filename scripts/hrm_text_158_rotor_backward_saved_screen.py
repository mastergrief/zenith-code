"""Phase 3a — rotor backward-saved-tensor gradient-fidelity screen.

Plan: zenith-code `.claude/MEMORY/ternary-rotor.md` §Phase 3 (preterminal
screen before any training run — fastest-science: cheapest decisive read).

GACT-style semantics via `torch.autograd.graph.saved_tensors_hooks`:
the FORWARD is exact; tensors STASHED for backward are rotor
quantize->dequantized at pack time, so the backward pass consumes
quantized saved activations. This isolates the Phase 3 variable
(saved-tensor precision) from the Phase 1 variable (forward precision).

Saved-tensor filter (documented in the receipt): quantize only floating
tensors with dim == 3 and last dim % 128 == 0 — i.e. block-level
activations (B,T,512)/(B,T,intermediate). 2-D tensors (weights, effective
ternary weights) and non-128-aligned tensors pass through exact.

CLASSIFIED EXCLUSION (bisect receipt, 2026-07-22): 4-D SDPA-saved q/k/v
MUST NOT be blanket-quantized behind the fused attention kernel — its
backward recomputes attention scores from saved q/k against the EXACT
forward logsumexp, so perturbed scores blow up as exp(score - lse)
(measured: 3-bit dim4-only med_cos 0.650, mean rel_l2 7.4e3, vs dim3-only
med_cos 0.994, rel 0.18). Quantized attention internals are the Phase 2
KV surface lever — a rotor-aware attention kernel that quantizes THEN
computes (lse-consistent), never a saved-tensor swap.

Measurement: response-only CE loss summed over a fixed seed-17 row set,
ONE backward per condition, then per-parameter gradient cosine + relative
L2 error vs the FP-saved control.

PRE-REGISTERED branch classifier (--mode blanket, the original screen):
  proceed(width) := median_cosine >= 0.99 AND min_cosine >= 0.95
  branch a: proceed(2bit)              -> "phase3_run_at_2bit"
  branch b: !proceed(2bit) AND proceed(3bit) -> "phase3_run_at_3bit"
  branch c: !proceed(3bit)             -> "backward_saved_park_null"

--mode 3b — Phase 3b quantize-narrow + remat-wide screen (designed from the
blanket-mode null: the 1536-wide SwiGLU-saved family carried ALL residual
corruption; the 512-wide family was clean at 3-bit). Every SwiGLU module is
checkpoint-wrapped (torch.utils.checkpoint, use_reentrant=False) so the wide
intermediates are RECOMPUTED in backward from the saved block input instead
of stored; the pack hook rotor-quantizes only 512-wide dim-3 saves. The
trainer-production integration point for the same semantics is the Tier-1
seam remat codec (`activation_residuals_m1_remat.py`) — this screen decides
viability before that integration is built.

PRE-REGISTERED (--mode 3b):
  validation: remat_only condition (wrap, no quant) must be lossless —
    median_cosine >= 0.9999 (recompute from exact input is deterministic);
    a miss is a HARNESS failure, not a science verdict.
  proceed(width) := median_cosine >= 0.99 AND min_cosine >= 0.95
  branch a: proceed(2bit)  -> "phase3b_run_at_2bit"
  branch b: !proceed(2bit) AND proceed(3bit) -> "phase3b_run_at_3bit"
  branch c: !proceed(3bit) -> "phase3b_narrow_quant_insufficient_park"
The full Phase 3 training run (slice + replay + 90/90 gates) happens ONLY
under a later dispatched launch packet; this screen makes no
banking/readiness claim.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.rotor_runtime_quant import (  # noqa: E402
    SavedTensorRotorCodec,
    rotor_fake_quant,
    unwrap_swiglu,
    wrap_swiglu_with_checkpoint,
)

N_ROWS = 16
SAMPLE_SEED = 17
MEDIAN_COS_MIN = 0.99
MIN_COS_MIN = 0.95


def _load_probe_module():
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "probe_hrm_text_158.py")
    spec = importlib.util.spec_from_file_location("probe_hrm_text_158", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_rows() -> list[dict]:
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )
    import random

    supports = build_exhaustive_supports()
    rng = random.Random(SAMPLE_SEED)
    flat = [(rung, q, int(e)) for rung in sorted(supports)
            for q, e in supports[rung]]
    picks = rng.sample(flat, N_ROWS)
    return [{"rung": r, "question": q, "expected": e} for r, q, e in picks]


# Codec + SwiGLU wrap/unwrap live in the facade (promoted on second use —
# trainer integration imports the same implementations):
# SavedTensorRotorCodec, wrap_swiglu_with_checkpoint, unwrap_swiglu.
_SavedTensorRotorCodec = SavedTensorRotorCodec
_wrap_swiglu_with_checkpoint = wrap_swiglu_with_checkpoint
_unwrap_swiglu = unwrap_swiglu


def _loss_and_grads(m, tok, rows, *, max_seq_len: int, device: str,
                    codec: _SavedTensorRotorCodec | None):
    from calm.hrm_text_158.lm_head import IGNORE_LABEL_ID

    m.zero_grad(set_to_none=True)
    total_loss = None
    ctx = (torch.autograd.graph.saved_tensors_hooks(codec.pack, codec.unpack)
           if codec is not None else _nullcontext())
    with ctx:
        for r in rows:
            q_ids = tok.encode(r["question"])
            a_ids = tok.encode(str(r["expected"])) + [tok.eos_id]
            seq = [tok.bos_id] + q_ids + [tok.sep_id] + a_ids
            if len(seq) > max_seq_len:
                continue
            sep_pos = 1 + len(q_ids)
            labels = [IGNORE_LABEL_ID] * (sep_pos + 1) + a_ids
            ids = torch.tensor([seq], dtype=torch.long, device=device)
            lab = torch.tensor([labels], dtype=torch.long, device=device)
            sep_t = torch.tensor([sep_pos], dtype=torch.long, device=device)
            pos = torch.arange(ids.shape[1], dtype=torch.long,
                               device=device).unsqueeze(0)
            _carry, loss, _metrics = m(
                None,
                {"inputs": ids, "labels": lab, "sep_positions": sep_t,
                 "position_ids": pos},
            )
            total_loss = loss if total_loss is None else total_loss + loss
        if codec is not None:
            # Forward-phase snapshot: backward-time checkpoint recompute
            # re-fires pack on transient tensors; only forward saves count
            # toward the stored-bytes ledger.
            codec.forward_quantized = codec.quantized
            codec.forward_passed = codec.passed
            codec.forward_quantized_values = codec.quantized_values
            codec.forward_passed_values = codec.passed_values
        total_loss.backward()
    grads = {n: p.grad.detach().clone() for n, p in m.named_parameters()
             if p.grad is not None}
    return float(total_loss.item()), grads


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def _compare(base: dict, cand: dict) -> dict:
    per_param = {}
    for name, g0 in base.items():
        g1 = cand[name]
        f0, f1 = g0.flatten().float(), g1.flatten().float()
        denom = (f0.norm() * f1.norm()).clamp(min=1e-30)
        cos = float((f0 @ f1) / denom)
        rel = float((f0 - f1).norm() / f0.norm().clamp(min=1e-30))
        per_param[name] = {"cosine": cos, "rel_l2": rel}
    cosines = [v["cosine"] for v in per_param.values()]
    worst = sorted(per_param.items(), key=lambda kv: kv[1]["cosine"])[:5]
    return {
        "median_cosine": statistics.median(cosines),
        "min_cosine": min(cosines),
        "mean_rel_l2": statistics.mean(v["rel_l2"] for v in per_param.values()),
        "n_params": len(per_param),
        "worst5": [{"name": n, **v} for n, v in worst],
        "per_param": per_param,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", required=True)
    ap.add_argument("--output-json", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--mode", choices=("blanket", "3b"), default="blanket")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    probe = _load_probe_module()

    print(f"[rotor-p3a] loading ckpt: {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location=device, weights_only=False)
    step = ckpt.get("step", "?")
    m, tok = probe._build_model_from_ckpt(ckpt, device)
    m.train()
    max_seq_len = ckpt["config"]["max_seq_len"]
    rows = _build_rows()
    print(f"[rotor-p3a] ckpt step={step} device={device} rows={len(rows)}",
          flush=True)

    loss0, grads0 = _loss_and_grads(m, tok, rows, max_seq_len=max_seq_len,
                                    device=device, codec=None)
    print(f"[rotor-p3a] control loss={loss0:.6f} "
          f"params_with_grad={len(grads0)}", flush=True)

    if args.mode == "3b":
        n_wrapped = _wrap_swiglu_with_checkpoint(m)
        print(f"[rotor-p3a] 3b: checkpoint-wrapped {n_wrapped} SwiGLU "
              f"modules (wide family rematerialized)", flush=True)
        conditions = [("remat_only", None), ("3bit", 3), ("2bit", 2)]
        narrow_only = True
    else:
        conditions = [("3bit", 3), ("2bit", 2)]
        narrow_only = False

    results = {}
    for tag, bits in conditions:
        codec = _SavedTensorRotorCodec(bits, narrow_only=narrow_only)
        loss, grads = _loss_and_grads(m, tok, rows, max_seq_len=max_seq_len,
                                      device=device, codec=codec)
        cmp = _compare(grads0, grads)
        results[tag] = {
            "loss": loss,
            "loss_delta_vs_control": loss - loss0,
            "saved_tensors_quantized_fwd": codec.forward_quantized,
            "saved_tensors_passed_fwd": codec.forward_passed,
            "saved_values_quantized_fwd": codec.forward_quantized_values,
            "saved_values_passed_fwd": codec.forward_passed_values,
            **{k: v for k, v in cmp.items() if k != "per_param"},
            "per_param": cmp["per_param"],
        }
        print(f"[rotor-p3a] {tag:10s} loss={loss:.6f} "
              f"(d={loss - loss0:+.2e}) q_fwd={codec.forward_quantized} "
              f"pass_fwd={codec.forward_passed} "
              f"med_cos={cmp['median_cosine']:.5f} "
              f"min_cos={cmp['min_cosine']:.5f} "
              f"rel_l2={cmp['mean_rel_l2']:.4f}", flush=True)
        for w in cmp["worst5"]:
            print(f"[rotor-p3a]   worst: {w['name']} cos={w['cosine']:.5f} "
                  f"rel={w['rel_l2']:.4f}", flush=True)

    if args.mode == "3b":
        _unwrap_swiglu(m)

    def _proceed(tag: str) -> bool:
        c = results[tag]
        return (c["median_cosine"] >= MEDIAN_COS_MIN
                and c["min_cosine"] >= MIN_COS_MIN)

    remat_lossless = None
    if args.mode == "3b":
        remat_lossless = results["remat_only"]["median_cosine"] >= 0.9999
        if not remat_lossless:
            branch = "3b_harness_failure_remat_not_lossless"
        elif _proceed("2bit"):
            branch = "phase3b_run_at_2bit"
        elif _proceed("3bit"):
            branch = "phase3b_run_at_3bit"
        else:
            branch = "phase3b_narrow_quant_insufficient_park"
        print(f"[rotor-p3a] remat_only lossless: {remat_lossless}",
              flush=True)
    else:
        if _proceed("2bit"):
            branch = "phase3_run_at_2bit"
        elif _proceed("3bit"):
            branch = "phase3_run_at_3bit"
        else:
            branch = "backward_saved_park_null"
    print(f"[rotor-p3a] === PREREG BRANCH: {branch} ===", flush=True)

    receipt = {
        "screen": f"rotor_backward_saved_gradient_fidelity/{args.mode}_v1",
        "mode": args.mode,
        "remat_only_lossless": remat_lossless,
        "ckpt_path": args.ckpt_path,
        "ckpt_step": step,
        "device": device,
        "sample": {"n_rows": N_ROWS, "seed": SAMPLE_SEED},
        "prereg": {"median_cos_min": MEDIAN_COS_MIN,
                   "min_cos_min": MIN_COS_MIN},
        "control_loss": loss0,
        "conditions": {k: {kk: vv for kk, vv in v.items()
                           if kk != "per_param"}
                       for k, v in results.items()},
        "per_param": {k: v["per_param"] for k, v in results.items()},
        "branch_verdict": branch,
    }
    if args.output_json:
        os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
        with open(args.output_json, "w") as f:
            json.dump(receipt, f, indent=2)
        print(f"[rotor-p3a] receipt -> {args.output_json}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
