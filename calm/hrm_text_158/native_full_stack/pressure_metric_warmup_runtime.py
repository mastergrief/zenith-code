"""Diagnostic runtime load/reset/rebind + REAL disposable HOT-PATH warmup (PLAN_v6 rev4).

Owns: load measured/throwaway runtimes, assert q-forward coupling, hot-path warmup
(separately loaded/patched throwaway runtime running actual fwd/bwd/credit/q-acc
observer steps, discarded), then measured runtime loaded/reset AFTER warmup,
and the single-run diagnostic loop (step0 probes BEFORE train).

Dependency: warmup_runtime → model_runtime + execution_loop + lifecycle + telemetry.
Bound by PLAN_v6 sha 346b67d8…; rev4 re-scope 1784829182373.
"""
from __future__ import annotations

import time
from typing import Any

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import ARM0
from calm.hrm_text_158.native_full_stack.forgetting_laws import entropy_bits
from calm.hrm_text_158.native_full_stack.phase_probe_sets import build_phase1_probe_sets
from calm.hrm_text_158.native_full_stack.pressure_metric_lifecycle import (
    PressureTelemetryStore,
)
from calm.hrm_text_158.native_full_stack.pressure_metric_telemetry import (
    hash_q_dict,
    hash_scale_dict,
)
from calm.hrm_text_158.native_full_stack.screen_execution_loop import run_train_loop
from calm.hrm_text_158.native_full_stack.screen_model_runtime import (
    _exact_match_count,
    _loss_and_credit,
    assert_q_levels_coupled,
    load_and_patch_runtime,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    lifetime_censored_frac,
)


def cuda_sync(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def build_pool():
    from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports

    return [
        (q, int(e), rung)
        for rung, rows in build_exhaustive_supports().items()
        for q, e in rows
    ]


def sha256_file(path: str) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_hotpath_warmup_throwaway(
    *,
    ckpt_path: str,
    device: str,
    batch: int = 8,
    n_steps: int = 1,
    seed: int = 0,
    enable: bool = True,
) -> dict[str, Any]:
    """Load a SEPARATE throwaway runtime, run real train steps, discard it.

    Measured runtime must be loaded AFTER this returns. enable=False → evidence
    false (formal/paired must refuse).
    """
    if not enable:
        return {
            "non_mutating_warmup": False,
            "hot_path_executed": False,
            "throwaway_runtime_id": None,
            "n_steps": 0,
            "n_fixed_qscale_forwards": 0,
            "stop_reason": "warmup_noop",
        }

    torch.manual_seed(int(seed) + 99991)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed) + 99991)

    throwaway = load_and_patch_runtime(ckpt_path=ckpt_path, device=device)
    throwaway_id = id(throwaway["m"])
    q_levels = throwaway["q_levels"]  # EXACT captured dict — no clone
    assert_q_levels_coupled(throwaway, q_levels)

    pool = build_pool()
    probe_sets = build_phase1_probe_sets()
    acq_set = set(probe_sets["acquisition"])
    store = PressureTelemetryStore.from_q_levels(q_levels, steps=int(n_steps))

    cuda_sync(device)
    loop_out = run_train_loop(
        m=throwaway["m"],
        tok=throwaway["tok"],
        eligible=throwaway["eligible"],
        q_levels=q_levels,
        pool=pool,
        acq_set=acq_set,
        arm=ARM0,
        steps=int(n_steps),
        batch=int(batch),
        topk=1024,
        max_seq_len=throwaway["max_seq_len"],
        device=device,
        correctness_smoke=False,
        pressure_telemetry=store,
    )
    cuda_sync(device)

    n_fixed = int(loop_out["train_route_counters"].get("n_fixed_qscale_forwards", 0))
    n_flips = int(loop_out.get("n_flips", 0))
    credited = int(loop_out.get("credited_mass", 0))
    q_changed = int(loop_out.get("q_changed_count", 0))
    n_credit_grads = int(
        loop_out["train_route_counters"].get("n_credit_grads_present", 0)
    )
    observer_ok = ("margin_trajectory" in loop_out) or (
        "pressure_telemetry" in loop_out
    )
    # Real hot path: FixedQScale forwards ran, credit route present, observer
    # telemetry attached. Rejects the old mean()-only deepcopy "warmup".
    hot = n_fixed > 0 and n_credit_grads > 0 and observer_ok
    evidence = {
        "non_mutating_warmup": True,  # measured not yet loaded
        "hot_path_executed": bool(hot),
        "throwaway_runtime_id": throwaway_id,
        "n_steps": int(n_steps),
        "n_fixed_qscale_forwards": n_fixed,
        "n_flips": n_flips,
        "credited_mass": credited,
        "q_changed_count": q_changed,
        "n_credit_grads_present": n_credit_grads,
        "observer_telemetry": bool(observer_ok),
        "stop_reason": None if hot else "warmup_hot_path_incomplete",
    }
    # Discard throwaway (drop refs)
    del throwaway, q_levels, store, loop_out
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return evidence


def run_one_diagnostic_loop(
    *,
    ckpt_path: str,
    device: str,
    steps: int,
    batch: int,
    topk: int,
    telemetry: bool,
    skip_probes: bool,
    seed: int,
    warmup_enable: bool = True,
    probe_order_trace: list[str] | None = None,
    formal_mode: bool = False,
) -> dict[str, Any]:
    """Hot-path warmup on throwaway → load measured AFTER → coupled-q train loop."""
    if formal_mode:
        if skip_probes:
            raise SystemExit("formal mode REFUSES skip_probes")
        if not telemetry:
            raise SystemExit("formal mode REFUSES telemetry-disabled")

    # 1) Throwaway hot-path warmup FIRST
    cuda_sync(device)
    warmup = run_hotpath_warmup_throwaway(
        ckpt_path=ckpt_path,
        device=device,
        batch=int(batch),
        n_steps=1,
        seed=int(seed),
        enable=bool(warmup_enable),
    )
    cuda_sync(device)

    # 2) Load measured runtime AFTER warmup (clean parent)
    torch.manual_seed(int(seed))
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    rt = load_and_patch_runtime(ckpt_path=ckpt_path, device=device)
    m = rt["m"]
    tok = rt["tok"]
    eligible = rt["eligible"]
    # B1 FIX: mutate the EXACT dict captured by installed forwards — NO clone
    q_levels = rt["q_levels"]
    assert_q_levels_coupled(rt, q_levels)
    frozen_scales = rt["frozen_scales"]
    max_seq_len = rt["max_seq_len"]
    banked_before = rt["sha_before"]
    scale_sha_before = rt["scale_sha_before"]
    q_sha_before = rt["q_sha_before"]
    measured_id = id(m)

    warmup["measured_runtime_id"] = measured_id
    warmup["ids_differ"] = (
        warmup.get("throwaway_runtime_id") is not None
        and warmup["throwaway_runtime_id"] != measured_id
    )
    warmup["post_warmup_reload"] = True
    if warmup_enable:
        warmup["non_mutating_warmup"] = bool(
            warmup.get("hot_path_executed") and warmup.get("ids_differ")
        )
        if not warmup["non_mutating_warmup"]:
            warmup["stop_reason"] = warmup.get("stop_reason") or "warmup_evidence_fail"

    probe_sets = build_phase1_probe_sets()
    acq_set = set(probe_sets["acquisition"])
    pool = build_pool()

    # step0 probes BEFORE training
    probes: dict[str, Any] = {}
    if not skip_probes:
        if probe_order_trace is not None:
            probe_order_trace.append("step0")
        acq_step0 = _exact_match_count(
            m, tok, probe_sets["acquisition"], max_seq_len=max_seq_len, device=device
        )
        ret_step0 = _exact_match_count(
            m, tok, probe_sets["retention"], max_seq_len=max_seq_len, device=device
        )
        probes["acq_step0_count"] = int(acq_step0)
        probes["ret_step0_count"] = int(ret_step0)
        probes["step0_taken_before_train"] = True
        probes["q_sha_at_step0"] = hash_q_dict(q_levels)

    store = None
    if telemetry:
        store = PressureTelemetryStore.from_q_levels(q_levels, steps=int(steps))

    cuda_sync(device)
    t0 = time.perf_counter()
    loop_out = run_train_loop(
        m=m,
        tok=tok,
        eligible=eligible,
        q_levels=q_levels,
        pool=pool,
        acq_set=acq_set,
        arm=ARM0,
        steps=int(steps),
        batch=int(batch),
        topk=int(topk),
        max_seq_len=max_seq_len,
        device=device,
        correctness_smoke=False,
        pressure_telemetry=store,
    )
    cuda_sync(device)
    wall_s = time.perf_counter() - t0

    if not skip_probes:
        if probe_order_trace is not None:
            probe_order_trace.append("final")
        acq_final = _exact_match_count(
            m, tok, probe_sets["acquisition"], max_seq_len=max_seq_len, device=device
        )
        ret_final = _exact_match_count(
            m, tok, probe_sets["retention"], max_seq_len=max_seq_len, device=device
        )
        probes.update(
            {
                "acq_final_count": int(acq_final),
                "acq_delta_count": int(acq_final) - int(probes["acq_step0_count"]),
                "ret_final_count": int(ret_final),
                "retention_ok": int(ret_final) >= int(probes["ret_step0_count"]) - 2,
                "acquisition_selection_sha256": probe_sets[
                    "acquisition_selection_sha256"
                ],
                "identity_selection_sha256": probe_sets["identity_selection_sha256"],
                "q_sha_at_final": hash_q_dict(q_levels),
            }
        )

    n_flips = int(loop_out["n_flips"])
    lifetimes = list(loop_out["lifetimes"])
    n_cens = 0
    for n, a in loop_out["acc"].items():
        ep = loop_out["episode_start"][n]
        n_cens += int(((a != 0) & (ep > 0)).sum().item())
    lcf = float(
        lifetime_censored_frac(n_flips=n_flips, n_censored_active_episodes=n_cens)
    )
    H_final = float(
        entropy_bits(torch.cat([a.flatten() for a in loop_out["acc"].values()]))
    )

    banked_after = sha256_file(ckpt_path)
    scale_sha_after = hash_scale_dict(frozen_scales)

    measurements = {
        "n_flips": n_flips,
        "q_changed_count": int(loop_out["q_changed_count"]),
        "credited_mass": int(loop_out["credited_mass"]),
        "n_applied_drains": int(loop_out["n_applied_drains"]),
        "lifetime_censored_frac": lcf,
        "p50_flip_lifetime": (
            float(sorted(lifetimes)[len(lifetimes) // 2]) if lifetimes else None
        ),
        "H_bits_per_weight": H_final,
        "H_trajectory": loop_out["H_trajectory"],
        "margin_trajectory": loop_out.get("margin_trajectory") or [],
        "episode_trajectory": loop_out.get("episode_trajectory") or [],
    }

    return {
        "rt": rt,
        "loop_out": loop_out,
        "store": store,
        "measurements": measurements,
        "probes": probes,
        "banked_sha": {
            "before": banked_before,
            "after": banked_after,
            "match": banked_before == banked_after,
        },
        "frozen_scale_sha": {
            "before": scale_sha_before,
            "after": scale_sha_after,
            "match": scale_sha_before == scale_sha_after,
        },
        "frozen_scales": frozen_scales,
        "q_sha_before": q_sha_before,
        "wall_s": float(wall_s),
        "wall_ms_per_step": float(wall_s) * 1000.0 / float(max(1, steps)),
        "route_counters": loop_out["train_route_counters"],
        "parent_sha": banked_before,
        "steps": int(steps),
        "batch": int(batch),
        "topk": int(topk),
        "seed": int(seed),
        "telemetry": bool(telemetry),
        "warmup": warmup,
        "device": str(device),
        "arm": ARM0,
        "q_levels_is_rt": q_levels is rt["q_levels"],
    }


def prove_q_forward_coupling_on_runtime(rt: dict[str, Any]) -> dict[str, Any]:
    """PRODUCTION-PATH: mutate captured q_levels → FixedQScale forward output changes.

    Covers BOTH in-place mutation AND loop-style ``q_levels[n] = new_q``
    writeback (the path that previously decoupled clones from forwards).
    """
    from calm.hrm_text_158.native_full_stack.fixed_qscale_credit import begin_credit_step

    assert_q_levels_coupled(rt, rt["q_levels"])
    m = rt["m"]
    modules = rt["modules"]
    q_levels = rt["q_levels"]
    pname = rt["eligible"][0]
    mod = modules[pname]
    device = next(m.parameters()).device
    in_f = int(mod.weight.shape[1])
    x = torch.randn(2, in_f, device=device, dtype=torch.float32)
    with torch.no_grad():
        begin_credit_step([pname])
        y0 = mod.forward(x).detach().cpu().clone()
        # 1) In-place mutate CAPTURED dict entry
        q = q_levels[pname]
        flat = q.view(-1)
        idx = 0
        old = int(flat[idx].item())
        flat[idx] = 0 if old != 0 else 1
        begin_credit_step([pname])
        y1 = mod.forward(x).detach().cpu().clone()
        flat[idx] = old
        # 2) Loop-style writeback: reassign a NEW tensor into the SAME dict
        q_new = q.clone()
        q_new.view(-1)[idx] = 0 if old != 0 else 1
        q_levels[pname] = q_new
        begin_credit_step([pname])
        y2 = mod.forward(x).detach().cpu().clone()
        q_levels[pname] = q  # restore original tensor object
    in_place_changed = not bool(torch.equal(y0, y1))
    writeback_changed = not bool(torch.equal(y0, y2))
    if not in_place_changed or not writeback_changed:
        raise RuntimeError(
            "q-forward coupling proof FAILED: in-place or writeback update of "
            "rt['q_levels'] did not change FixedQScale forward output "
            f"(in_place={in_place_changed}, writeback={writeback_changed})"
        )
    return {
        "ok": True,
        "param": pname,
        "forward_changed": True,
        "in_place_changed": True,
        "writeback_changed": True,
        "q_levels_is_rt": q_levels is rt["q_levels"],
    }


def assert_final_probe_surface_tracks_q(result: dict[str, Any]) -> dict[str, Any]:
    """Fail-closed: when train writeback moves q, final probe surface must move too."""
    if not result.get("q_levels_is_rt"):
        raise RuntimeError("final probe surface check: q_levels not coupled to rt")
    probes = result.get("probes") or {}
    q0 = probes.get("q_sha_at_step0")
    qf = probes.get("q_sha_at_final")
    q_changed = int((result.get("measurements") or {}).get("q_changed_count", 0))
    if q_changed > 0:
        if not q0 or not qf:
            raise RuntimeError(
                "final probe surface check: missing q_sha_at_step0/final after q moves"
            )
        if q0 == qf:
            raise RuntimeError(
                "final probe surface check FAILED: q_changed_count>0 but "
                "q_sha_at_step0 == q_sha_at_final (forward/probes stuck on parent q)"
            )
    return {
        "ok": True,
        "q_changed_count": q_changed,
        "q_sha_moved": bool(q0 and qf and q0 != qf),
        "q_levels_is_rt": True,
    }
