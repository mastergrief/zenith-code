"""Vote-lifetime statistics screen — classify-before-build for q/acc forgetting.

Twin of ``hrm_text_158_acc_entropy_screen.py`` on the same from-clean-parent
vote/credit stream. Bound by PLAN_v1 sha 3b3848c8… + riders 1/2/3.

Developer checks only from plan-dev: py_compile, CPU-static reducer tests,
``--steps 5`` CPU shape smoke. Formal 150-step GPU run is Claude/test-operator.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (  # noqa: E402
    project_s1_gradient_to_moves,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (  # noqa: E402
    CROSSING_THRESHOLD_ABS,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (  # noqa: E402
    accumulate_hazard,
    accumulate_survival,
    age_rate_partition,
    apply_drain_resets,
    classify_forgetting_family,
    count_censored_active_episodes,
    empty_hazard_table,
    empty_survival_table,
    histogram_lifetimes,
    lifetime_censored_frac,
    lifetime_quantiles,
    never_convert_metrics,
    survival_fractions,
    update_episode_starts,
)

CLIP = 127
TOPK_PER_STEP = 1024
BULK_SUFFIXES = (
    "gqkv_proj.weight",
    "o_proj.weight",
    "gate_up_proj.weight",
    "down_proj.weight",
)
EXPECTED_PARENT_SHA256 = (
    "2d9b9f6746e66cec9e7e39d65e8171151e836daca99df6b56fb488d8a6f2403b"
)
PLAN_SHA256 = (
    "3b3848c8b6ff818ccecce4a8ea5aac9c13a3d8b2ebc00becdd5dea81bc91fae0"
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_probe_and_screen():
    base = os.path.dirname(os.path.abspath(__file__))
    out = []
    for name in ("probe_hrm_text_158", "hrm_text_158_rotor_backward_saved_screen"):
        spec = importlib.util.spec_from_file_location(
            name, os.path.join(base, name + ".py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        out.append(mod)
    return out


def _arm_drain_mask_topk(
    arms_acc: dict[str, torch.Tensor],
    *,
    topk: int,
) -> dict[str, torch.Tensor]:
    """Global top-K crossers by |acc| across eligible tensors (armC)."""
    flat_abs = []
    shapes = []
    for n, a in arms_acc.items():
        flat_abs.append(a.abs().flatten())
        shapes.append((n, a.numel(), a.shape))
    allabs = torch.cat(flat_abs)
    crosser_idx = torch.nonzero(
        allabs >= CROSSING_THRESHOLD_ABS, as_tuple=False
    ).flatten()
    sel = torch.zeros_like(allabs, dtype=torch.bool)
    if crosser_idx.numel():
        k = min(int(topk), int(crosser_idx.numel()))
        top = crosser_idx[allabs[crosser_idx].argsort(descending=True)[:k]]
        sel[top] = True
    masks: dict[str, torch.Tensor] = {}
    off = 0
    for n, nn, shape in shapes:
        masks[n] = sel[off : off + nn].view(shape)
        off += nn
    return masks


def _receipt_schema_skeleton() -> dict:
    return {
        "screen": "vote_lifetime_screen/v1",
        "plan_sha256": PLAN_SHA256,
        "binding_riders": [
            "1784802561699-5dc9da8f",
            "1784802776203-87cc3967",
            "1784802863548-ded13e80",
        ],
        "measurements": {
            "a_flip_lifetime": {},
            "b_never_convert": {},
            "c_age_hazard": {},
            "d_acc_survival": {},
            "right_censor": {},
        },
        "classifier": {},
        "banked_sha": {},
        "limits": [],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-path", default=None)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument(
        "--device",
        default="cpu",
        help="cpu for developer shape smoke; cuda for formal science run",
    )
    ap.add_argument("--topk", type=int, default=TOPK_PER_STEP)
    ap.add_argument("--output-json", default=None)
    ap.add_argument(
        "--schema-only",
        action="store_true",
        help="Emit receipt skeleton without loading ckpt (import/CLI smoke)",
    )
    args = ap.parse_args()

    if args.schema_only:
        receipt = _receipt_schema_skeleton()
        receipt["schema_only"] = True
        if args.output_json:
            os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(receipt, f, indent=2)
            print(f"[vote-life] schema-only -> {args.output_json}", flush=True)
        else:
            print(json.dumps(receipt, indent=2))
        return 0

    if not args.ckpt_path:
        raise SystemExit("--ckpt-path is required unless --schema-only")

    device = str(args.device)
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("cuda requested but torch.cuda.is_available() is False")

    sha_before = _sha256_file(args.ckpt_path)
    if sha_before != EXPECTED_PARENT_SHA256:
        raise SystemExit(
            f"banked sha mismatch before load: got {sha_before}, "
            f"expected {EXPECTED_PARENT_SHA256}"
        )

    torch.set_num_threads(8)
    probe, scr = _load_probe_and_screen()
    from calm.hrm_text_158.curriculum.exhaustive_supports import (
        build_exhaustive_supports,
    )

    print(f"[vote-life] loading ckpt ({device}): {args.ckpt_path}", flush=True)
    ckpt = torch.load(args.ckpt_path, map_location="cpu", weights_only=False)
    m, tok = probe._build_model_from_ckpt(ckpt, device)
    m.train()

    eligible = {
        n: p for n, p in m.named_parameters() if n.endswith(BULK_SUFFIXES)
    }
    q_levels: dict[str, torch.Tensor] = {}
    with torch.no_grad():
        for n, p in eligible.items():
            # Instrumentation path is CPU-only: detach q to CPU even when the
            # model lives on --device cuda (gate-2 BLOCK 1784803494954).
            scale = p.detach().float().abs().mean().clamp(min=1e-5)
            q_levels[n] = (
                (p.detach().float() / scale).round().clamp(-1, 1).to(torch.int8).cpu()
            )

    # From-clean-parent: empty acc + empty backlog; frozen q (all CPU).
    acc = {n: torch.zeros_like(q, dtype=torch.int16) for n, q in q_levels.items()}
    episode_start = {
        n: torch.zeros_like(q, dtype=torch.int32) for n, q in q_levels.items()
    }
    flip_count = {
        n: torch.zeros_like(q, dtype=torch.int32) for n, q in q_levels.items()
    }

    lifetimes: list[int] = []
    hazard = empty_hazard_table()
    survival = empty_survival_table()
    credited_mass = 0
    n_flips = 0

    pool = [
        (q, int(e))
        for rows in build_exhaustive_supports().values()
        for q, e in rows
    ]
    t0 = time.time()
    for step in range(1, int(args.steps) + 1):
        rng = random.Random(1000 + step)
        batch_rows = [
            {"rung": "mix", "question": q, "expected": e}
            for q, e in rng.sample(pool, int(args.batch))
        ]
        _loss, grads = scr._loss_and_grads(
            m, tok, batch_rows, max_seq_len=384, device=device, codec=None
        )
        moves = {}
        for n in eligible:
            g_cpu = grads[n].detach().cpu()
            q_cpu = q_levels[n]
            if g_cpu.device.type != "cpu" or q_cpu.device.type != "cpu":
                raise RuntimeError(
                    f"vote-lifetime projection requires CPU tensors; "
                    f"got grad={g_cpu.device} q={q_cpu.device} for {n}"
                )
            if g_cpu.shape != q_cpu.shape:
                raise RuntimeError(
                    f"grad/q shape mismatch for {n}: {tuple(g_cpu.shape)} vs "
                    f"{tuple(q_cpu.shape)}"
                )
            moves[n] = project_s1_gradient_to_moves(g_cpu, q_cpu)
        credited_mass += int(
            sum(int(mv.abs().sum().item()) for mv in moves.values())
        )

        # Keep episode/acc instrumentation on CPU (compact int maps).
        drain_inputs: dict[str, torch.Tensor] = {}
        for n, mv in moves.items():
            prev = acc[n]
            new = (
                (prev.to(torch.int32) + mv.to(torch.int32))
                .clamp(-CLIP, CLIP)
                .to(torch.int16)
            )
            episode_start[n] = update_episode_starts(
                prev, new, episode_start[n], step
            )
            acc[n] = new
            drain_inputs[n] = new

        masks = _arm_drain_mask_topk(drain_inputs, topk=int(args.topk))

        # Hazard/survival exposure BEFORE drain, using pre-drain ages.
        for n, a in acc.items():
            active = (a != 0) & (episode_start[n] > 0)
            if not bool(active.any()):
                continue
            ages = (step - episode_start[n][active]).to(torch.int64)
            abs_a = a[active].abs().to(torch.int64)
            flip_here = masks[n][active]
            accumulate_hazard(hazard, ages, flip_here)
            accumulate_survival(survival, ages, abs_a)

        for n in list(acc.keys()):
            drained = int(masks[n].sum().item())
            new_acc, new_ep, lt = apply_drain_resets(
                acc[n], episode_start[n], masks[n], step
            )
            if drained:
                flip_count[n] = flip_count[n] + masks[n].to(torch.int32)
                n_flips += drained
            if lt:
                lifetimes.extend(lt)
            acc[n] = new_acc
            episode_start[n] = new_ep

        if step % 10 == 0 or step == int(args.steps):
            print(
                f"[vote-life] step {step:4d} "
                f"({time.time() - t0:.1f}s) flips={n_flips} "
                f"credited_mass={credited_mass}",
                flush=True,
            )

    # End-of-window metrics on armC state.
    final_acc = torch.cat([a.flatten() for a in acc.values()])
    final_ep = torch.cat([e.flatten() for e in episode_start.values()])
    final_fc = torch.cat([c.flatten() for c in flip_count.values()])
    n_censored = count_censored_active_episodes(final_acc, final_ep)
    lcf = lifetime_censored_frac(n_flips, n_censored)
    b_metrics = never_convert_metrics(final_acc, final_fc)
    quant = lifetime_quantiles(lifetimes)
    hist = histogram_lifetimes(lifetimes)
    old_rate, young_rate = age_rate_partition(hazard)
    classifier = classify_forgetting_family(
        n_flips=n_flips,
        credited_mass=float(credited_mass),
        p50_flip_lifetime=quant["p50"],
        never_convert_frac=float(b_metrics["never_convert_frac"]),
        age_rate_old=old_rate,
        age_rate_young=young_rate,
        lifetime_censored_frac_value=lcf,
    )

    sha_after = _sha256_file(args.ckpt_path)
    if sha_after != sha_before:
        raise SystemExit(
            f"banked sha mutated during screen: before={sha_before} after={sha_after}"
        )

    receipt = {
        "screen": "vote_lifetime_screen/v1",
        "plan_sha256": PLAN_SHA256,
        "binding_riders": [
            "1784802561699-5dc9da8f",
            "1784802776203-87cc3967",
            "1784802863548-ded13e80",
        ],
        "ckpt_path": args.ckpt_path,
        "device": device,
        "steps": int(args.steps),
        "batch": int(args.batch),
        "topk_per_step": int(args.topk),
        "law": {
            "projection": "project_s1_gradient_to_moves (imported)",
            "vote_quantum": 1,
            "crossing_threshold_abs": int(CROSSING_THRESHOLD_ABS),
            "clip": CLIP,
            "episode_semantics": (
                "zero->nonzero start; reset on zero-return / sign-reversal / "
                "applied drain; lifetime=flip_step-episode_start (excl)"
            ),
        },
        "banked_sha": {
            "before": sha_before,
            "after": sha_after,
            "expected": EXPECTED_PARENT_SHA256,
            "match": sha_before == EXPECTED_PARENT_SHA256 == sha_after,
        },
        "measurements": {
            "a_flip_lifetime": {
                "n_flips": n_flips,
                "histogram": hist,
                "quantiles": quant,
            },
            "b_never_convert": b_metrics,
            "c_age_hazard": {
                "bins": hazard,
                "age_rate_old_gt32": old_rate,
                "age_rate_young_le8": young_rate,
            },
            "d_acc_survival": survival_fractions(survival),
            "right_censor": {
                "n_censored_active_episodes": n_censored,
                "lifetime_censored_frac": lcf,
                "note": (
                    "includes post-flip active episodes; distinct from "
                    "never_convert_frac"
                ),
                "optional_censored_abs_mass": float(
                    final_acc[final_ep > 0].abs().sum().item()
                ),
            },
            "credited_mass": credited_mass,
        },
        "classifier": classifier,
        "limits": [
            "frozen weights / frozen q (no coupled weight motion)",
            "armC topK=1024 is bracket analog of global applied-rate cap",
            "selects DESIGN family only — no forgetting law / no sub-2 claim",
            "instrumentation path forces CPU q+grad transfers each step "
            "(acceptable for 150-step screen; not a training hot loop)",
        ],
        "elapsed_s": round(time.time() - t0, 2),
    }

    if args.output_json:
        out_dir = os.path.dirname(args.output_json)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(receipt, f, indent=2)
        print(f"[vote-life] receipt -> {args.output_json}", flush=True)
    else:
        print(json.dumps({k: receipt[k] for k in ("classifier", "banked_sha")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
