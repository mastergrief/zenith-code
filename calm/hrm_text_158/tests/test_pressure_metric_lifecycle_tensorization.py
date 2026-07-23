"""Oracle + micro-perf for vectorized `_close_events_masked` (dispatch 1784837556458)."""

import json
import time
from types import MethodType

import torch as _torch

from calm.hrm_text_158.native_full_stack.pressure_metric_lifecycle import (
    DurableAggregates as _DA,
    PressureTelemetryStore as _PTS,
)


_AGG_KEYS = (
    "N_events_evaluable",
    "N_events_censored_insufficient_followup",
    "N_survived_applied_within_H",
    "N_never_applied_within_H",
    "N_events_evaluable_early",
    "N_events_evaluable_late",
    "N_never_applied_within_H_early",
    "N_never_applied_within_H_late",
)


def _oracle_close_event(
    agg: _DA,
    *,
    first_step: int,
    applied_after: int,
    now_step: int,
    reason: str,
    steps: int,
    H: int,
) -> None:
    """INDEPENDENT FROZEN COPY of pre-change scalar close-event arithmetic.

    Lives only in this test file. Must NOT import, call, or share any production
    vectorized close helper or extracted shared pure arithmetic — common-mode /
    tautological equivalence is a gate bounce.
    """
    if first_step > steps - H:
        agg.N_events_censored_insufficient_followup += 1
        return
    if reason == "window_end" and applied_after == 0 and (now_step - first_step) < H:
        if now_step >= steps and (first_step + H) > steps:
            agg.N_events_censored_insufficient_followup += 1
            return

    agg.N_events_evaluable += 1
    mid = steps // 2
    bucket = None
    if first_step > steps - H:
        bucket = None
    elif 1 <= first_step <= mid:
        bucket = "early"
    elif mid < first_step <= steps - H:
        bucket = "late"
    if bucket == "early":
        agg.N_events_evaluable_early += 1
    elif bucket == "late":
        agg.N_events_evaluable_late += 1

    survived = applied_after > 0 and 0 < (applied_after - first_step) <= H
    if survived:
        agg.N_survived_applied_within_H += 1
    else:
        if applied_after == 0 and (now_step - first_step) >= H:
            agg.N_never_applied_within_H += 1
            if bucket == "early":
                agg.N_never_applied_within_H_early += 1
            elif bucket == "late":
                agg.N_never_applied_within_H_late += 1
        elif applied_after > 0 and (applied_after - first_step) > H:
            agg.N_never_applied_within_H += 1
            if bucket == "early":
                agg.N_never_applied_within_H_early += 1
            elif bucket == "late":
                agg.N_never_applied_within_H_late += 1
        elif reason in ("horizon_expired", "residual_clear", "residual_restart") and applied_after == 0:
            if (now_step - first_step) >= H:
                agg.N_never_applied_within_H += 1
                if bucket == "early":
                    agg.N_never_applied_within_H_early += 1
                elif bucket == "late":
                    agg.N_never_applied_within_H_late += 1
            else:
                agg.N_events_evaluable -= 1
                if bucket == "early":
                    agg.N_events_evaluable_early -= 1
                elif bucket == "late":
                    agg.N_events_evaluable_late -= 1
                agg.N_events_censored_insufficient_followup += 1


def _scalar_close_events_masked(
    self: _PTS,
    *,
    first: _torch.Tensor,
    after: _torch.Tensor,
    close_mask: _torch.Tensor,
    now_step: int,
    reason: str,
) -> None:
    """Oracle path: per-index pure-Python close + identical tracker zeroing."""
    if not bool(close_mask.any()):
        return
    steps = int(self.steps)
    H = int(self.follow_up_horizon)
    idxs = close_mask.nonzero(as_tuple=False)
    for idx in idxs:
        idx_t = tuple(int(x) for x in idx.tolist())
        fs = int(first[idx_t].item())
        aa = int(after[idx_t].item())
        _oracle_close_event(
            self.aggregates,
            first_step=fs,
            applied_after=aa,
            now_step=int(now_step),
            reason=reason,
            steps=steps,
            H=H,
        )
        first[idx_t] = 0
        after[idx_t] = 0


def _clone_store(src: _PTS) -> _PTS:
    dst = _PTS(steps=src.steps, follow_up_horizon=src.follow_up_horizon, threshold=src.threshold)
    for n in src.first_deferral_step:
        dst.first_deferral_step[n] = src.first_deferral_step[n].clone()
        dst.applied_after_deferral_step[n] = src.applied_after_deferral_step[n].clone()
        dst.episode_generation[n] = src.episode_generation[n].clone()
    dst.aggregates = _DA(**src.aggregates.as_dict())
    dst.per_step_ratios = [dict(x) for x in src.per_step_ratios]
    dst.two_tier_threshold_assert_pass = src.two_tier_threshold_assert_pass
    return dst


def _assert_store_equiv(vec: _PTS, ora: _PTS, *, label: str) -> None:
    va = vec.aggregates.as_dict()
    oa = ora.aggregates.as_dict()
    for k in _AGG_KEYS:
        assert va[k] == oa[k], f"{label}: agg[{k}] vec={va[k]} ora={oa[k]}"
    for n in vec.first_deferral_step:
        assert _torch.equal(vec.first_deferral_step[n], ora.first_deferral_step[n]), (
            f"{label}: first_deferral_step[{n}] mismatch"
        )
        assert _torch.equal(
            vec.applied_after_deferral_step[n], ora.applied_after_deferral_step[n]
        ), f"{label}: applied_after_deferral_step[{n}] mismatch"
        assert _torch.equal(vec.episode_generation[n], ora.episode_generation[n]), (
            f"{label}: episode_generation[{n}] mismatch"
        )


def _oracle_process_pre_writeback(
    self: _PTS,
    *,
    candidate_masks,
    applied_masks,
    step: int,
    n_candidates: int,
    n_applied: int,
) -> None:
    """R5: INDEPENDENT FROZEN COPY of eccab77 process_pre_writeback three-loop body.

    Taken pre-2A from pressure_metric_lifecycle.py:247-296. Must NOT call production
    process_pre_writeback or any 2A short-circuit helper — only self._close_events_masked
    (bound to the scalar close oracle on oracle stores).
    """
    t = int(step)
    H = int(self.follow_up_horizon)
    for n, cand in candidate_masks.items():
        applied = applied_masks[n]
        deferred = cand & ~applied
        first = self.first_deferral_step[n]
        new_def = deferred & (first == 0)
        if bool(new_def.any()):
            first[new_def] = t

    for n, applied in applied_masks.items():
        first = self.first_deferral_step[n]
        after = self.applied_after_deferral_step[n]
        hit = applied & (first > 0) & (after == 0) & (first < t)
        if bool(hit.any()):
            after[hit] = t

    for n in list(self.first_deferral_step.keys()):
        first = self.first_deferral_step[n]
        after = self.applied_after_deferral_step[n]
        open_ev = first > 0
        if not bool(open_ev.any()):
            continue
        just_survived = open_ev & (after == t) & ((after - first) <= H) & ((after - first) > 0)
        self._close_events_masked(
            first=first,
            after=after,
            close_mask=just_survived,
            now_step=t,
            reason="survived",
        )
        expired = (first > 0) & (after == 0) & ((t - first) >= H)
        self._close_events_masked(
            first=first,
            after=after,
            close_mask=expired,
            now_step=t,
            reason="horizon_expired",
        )

    ratio = float(n_candidates) / float(max(1, n_applied))
    self.per_step_ratios.append(
        {
            "step": t,
            "candidate_crossers_before_cap": int(n_candidates),
            "applied_count": int(n_applied),
            "demand_applied_ratio": ratio,
            "deferred_count": max(0, int(n_candidates) - int(n_applied)),
        }
    )


def _bind_oracle(store: _PTS) -> _PTS:
    """Bind BOTH frozen phase oracle and scalar close oracle (R5)."""
    store._close_events_masked = MethodType(_scalar_close_events_masked, store)  # type: ignore[method-assign]
    store.process_pre_writeback = MethodType(_oracle_process_pre_writeback, store)  # type: ignore[method-assign]
    return store


def _seeded_store(shapes: dict[str, tuple[int, ...]], *, steps: int, H: int = 32) -> _PTS:
    q = {n: _torch.zeros(shape, dtype=_torch.int8) for n, shape in shapes.items()}
    return _PTS.from_q_levels(q, steps=steps, follow_up_horizon=H)


def test_close_events_masked_matches_oracle_adversarial_boundaries():
    """Direct masked close: vectorized vs pure-Python on adversarial (fs,aa) grids."""
    for steps, H in ((25, 32), (150, 32)):
        mid = steps // 2
        cases = [
            (steps - H, 0, steps, "window_end"),
            (steps - H + 1, 0, steps, "window_end"),
            (1, 1 + H, steps, "window_end"),
            (1, 1 + H + 1, steps, "window_end"),
            (mid, 0, steps, "window_end"),
            (mid + 1, 0, steps, "window_end"),
            (5, 0, steps, "window_end"),
            (5, 10, 10, "survived"),
            (5, 0, 5 + H, "horizon_expired"),
            (5, 0, 5 + H - 1, "residual_clear"),
            (5, 0, 5 + H - 1, "residual_restart"),
            (steps - H + 1, 0, steps // 2, "survived"),
        ]
        n = len(cases)
        for reason_filter in sorted({c[3] for c in cases}):
            idxs = [i for i, c in enumerate(cases) if c[3] == reason_filter]
            if not idxs:
                continue
            vec = _seeded_store({"w": (n,)}, steps=steps, H=H)
            ora = _bind_oracle(_clone_store(vec))
            first_v = vec.first_deferral_step["w"]
            after_v = vec.applied_after_deferral_step["w"]
            first_o = ora.first_deferral_step["w"]
            after_o = ora.applied_after_deferral_step["w"]
            mask = _torch.zeros(n, dtype=_torch.bool)
            for i in idxs:
                fs, aa, t, reason = cases[i]
                first_v[i] = fs
                after_v[i] = aa
                first_o[i] = fs
                after_o[i] = aa
                mask[i] = True
            t = cases[idxs[0]][2]
            # use per-case now_step only when uniform; otherwise close one-by-one
            nows = {cases[i][2] for i in idxs}
            if len(nows) == 1:
                vec._close_events_masked(
                    first=first_v, after=after_v, close_mask=mask, now_step=t, reason=reason_filter
                )
                ora._close_events_masked(
                    first=first_o, after=after_o, close_mask=mask, now_step=t, reason=reason_filter
                )
            else:
                for i in idxs:
                    fs, aa, t_i, reason = cases[i]
                    m = _torch.zeros(n, dtype=_torch.bool)
                    m[i] = True
                    first_v[i] = fs
                    after_v[i] = aa
                    first_o[i] = fs
                    after_o[i] = aa
                    vec._close_events_masked(
                        first=first_v, after=after_v, close_mask=m, now_step=t_i, reason=reason
                    )
                    ora._close_events_masked(
                        first=first_o, after=after_o, close_mask=m, now_step=t_i, reason=reason
                    )
            _assert_store_equiv(vec, ora, label=f"adv steps={steps} reason={reason_filter}")


def test_lifecycle_phase_oracle_equivalence_randomized():
    """Property: after each phase + finalize, 8 counters AND tracker tensors match oracle."""
    rng = _torch.Generator().manual_seed(15801)
    geometries = ((25, 32), (150, 32))
    for steps, H in geometries:
        for trial in range(8):
            shapes = {
                "a": (7,),
                "b": (3, 5),
            }
            vec = _seeded_store(shapes, steps=steps, H=H)
            ora = _bind_oracle(_clone_store(vec))
            # random schedule over steps
            T = min(steps, 40 if steps == 150 else steps)
            for t in range(1, T + 1):
                cand_masks = {}
                app_masks = {}
                n_c = 0
                n_a = 0
                for name, shape in shapes.items():
                    cand = _torch.rand(shape, generator=rng) < 0.35
                    app = cand & (_torch.rand(shape, generator=rng) < 0.45)
                    cand_masks[name] = cand
                    app_masks[name] = app
                    n_c += int(cand.sum().item())
                    n_a += int(app.sum().item())
                vec.process_pre_writeback(
                    candidate_masks=cand_masks,
                    applied_masks=app_masks,
                    step=t,
                    n_candidates=n_c,
                    n_applied=n_a,
                )
                ora.process_pre_writeback(
                    candidate_masks=cand_masks,
                    applied_masks=app_masks,
                    step=t,
                    n_candidates=n_c,
                    n_applied=n_a,
                )
                _assert_store_equiv(vec, ora, label=f"pre_wb trial={trial} t={t} steps={steps}")

                # occasional residual close + generation rollover
                if t % 7 == 0 and n_a > 0:
                    rz = {
                        name: (_torch.rand(app_masks[name].shape, generator=rng) < 0.5)
                        & app_masks[name]
                        for name in shapes
                    }
                    vec.close_before_writeback_resets(
                        applied_masks=app_masks, step=t, residual_zero=rz
                    )
                    ora.close_before_writeback_resets(
                        applied_masks=app_masks, step=t, residual_zero=rz
                    )
                    _assert_store_equiv(vec, ora, label=f"residual trial={trial} t={t}")
                    before = {
                        name: _torch.randint(0, 5, shapes[name], dtype=_torch.int32, generator=rng)
                        for name in shapes
                    }
                    after = {name: before[name] + app_masks[name].to(_torch.int32) for name in shapes}
                    vec.roll_tracker_after_writeback(
                        applied_masks=app_masks,
                        episode_start_before=before,
                        episode_start_after=after,
                        step=t,
                    )
                    ora.roll_tracker_after_writeback(
                        applied_masks=app_masks,
                        episode_start_before=before,
                        episode_start_after=after,
                        step=t,
                    )
                    _assert_store_equiv(vec, ora, label=f"rollover trial={trial} t={t}")

            vec.finalize_window(final_step=steps)
            ora.finalize_window(final_step=steps)
            _assert_store_equiv(vec, ora, label=f"finalize trial={trial} steps={steps}")


def test_residual_zero_vs_restart_and_generation_rollover_trackers():
    """Explicit amendment cases: residual reason split + roll_tracker zeroing."""
    st = _seeded_store({"w": (4,)}, steps=100, H=32)
    none = _torch.zeros(4, dtype=_torch.bool)
    # defer idx0 and idx1
    st.process_pre_writeback(
        candidate_masks={"w": _torch.tensor([True, True, False, False])},
        applied_masks={"w": none},
        step=2,
        n_candidates=2,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 2
    assert int(st.first_deferral_step["w"][1].item()) == 2
    applied = _torch.tensor([True, True, False, False])
    rz = {"w": _torch.tensor([True, False, False, False])}
    st.close_before_writeback_resets(applied_masks={"w": applied}, step=3, residual_zero=rz)
    assert int(st.first_deferral_step["w"][0].item()) == 0
    assert int(st.first_deferral_step["w"][1].item()) == 0
    # reopen deferrals for rollover case
    st.process_pre_writeback(
        candidate_masks={"w": _torch.tensor([True, False, False, False])},
        applied_masks={"w": none},
        step=5,
        n_candidates=1,
        n_applied=0,
    )
    applied1 = _torch.tensor([True, False, False, False])
    st.close_before_writeback_resets(
        applied_masks={"w": applied1},
        step=6,
        residual_zero={"w": none},
    )
    gen0 = int(st.episode_generation["w"][0].item())
    st.roll_tracker_after_writeback(
        applied_masks={"w": applied1},
        episode_start_before={"w": _torch.tensor([5, 0, 0, 0], dtype=_torch.int32)},
        episode_start_after={"w": _torch.tensor([6, 0, 0, 0], dtype=_torch.int32)},
        step=6,
    )
    assert int(st.episode_generation["w"][0].item()) == gen0 + 1
    assert int(st.first_deferral_step["w"][0].item()) == 0
    assert int(st.applied_after_deferral_step["w"][0].item()) == 0


def test_receipt_facing_structures_have_no_raw_per_event_arrays():
    st = _seeded_store({"w": (8,)}, steps=50, H=32)
    for t in range(1, 11):
        cand = _torch.tensor([True, True, False, False, False, False, False, False])
        app = _torch.tensor([True, False, False, False, False, False, False, False])
        st.process_pre_writeback(
            candidate_masks={"w": cand},
            applied_masks={"w": app},
            step=t,
            n_candidates=2,
            n_applied=1,
        )
    st.finalize_window(final_step=50)
    surv = st.survival_summary()
    blob = json.dumps(surv)
    assert "first_deferral_step" not in blob
    assert "applied_after_deferral_step" not in blob
    assert "episode_generation" not in blob
    for row in st.per_step_ratios:
        assert set(row.keys()) <= {
            "step",
            "candidate_crossers_before_cap",
            "applied_count",
            "demand_applied_ratio",
            "deferred_count",
        }


def test_oracle_is_independent_frozen_copy_not_shared_helper():
    """Acceptance #1: oracle must not call production vector close helpers."""
    import inspect

    # Body only (strip docstring) must not name production helpers.
    src = inspect.getsource(_oracle_close_event)
    body = src.split('"""', 2)[-1]
    assert "_close_events_masked" not in body
    assert "PressureTelemetryStore" not in body
    names = set(_oracle_close_event.__code__.co_names)
    assert "_close_events_masked" not in names
    assert "_close_event" not in names


def test_named_fixture_window_end_outer_true_inner_false_fallthrough():
    """Acceptance #2a: window_end outer-true / inner-false → evaluable, neither survived nor never."""
    steps, H = 150, 32
    # fs=100, aa=0, t=120: (t-fs)=20 < H (outer true); t < steps (inner false) → FALL THROUGH
    vec = _seeded_store({"w": (1,)}, steps=steps, H=H)
    ora = _bind_oracle(_clone_store(vec))
    for st in (vec, ora):
        st.first_deferral_step["w"][0] = 100
        st.applied_after_deferral_step["w"][0] = 0
    before = vec.aggregates.as_dict()
    mask = _torch.tensor([True])
    for st in (vec, ora):
        st._close_events_masked(
            first=st.first_deferral_step["w"],
            after=st.applied_after_deferral_step["w"],
            close_mask=mask,
            now_step=120,
            reason="window_end",
        )
    _assert_store_equiv(vec, ora, label="window_end_fallthrough")
    after = vec.aggregates.as_dict()
    assert after["N_events_evaluable"] == before["N_events_evaluable"] + 1
    assert after["N_events_censored_insufficient_followup"] == before[
        "N_events_censored_insufficient_followup"
    ]
    assert after["N_survived_applied_within_H"] == before["N_survived_applied_within_H"]
    assert after["N_never_applied_within_H"] == before["N_never_applied_within_H"]
    # mid=75, steps-H=118 → fs=100 is late cohort
    assert after["N_events_evaluable_late"] == before["N_events_evaluable_late"] + 1


def test_named_fixture_residual_early_censor_net_decrement():
    """Acceptance #2b: residual early-censor path nets evaluable±0 and censor+1."""
    steps, H = 150, 32
    # fs=5, aa=0, t=10, residual_clear: (t-fs)<H → +evaluable then -evaluable, +censor
    vec = _seeded_store({"w": (1,)}, steps=steps, H=H)
    ora = _bind_oracle(_clone_store(vec))
    for st in (vec, ora):
        st.first_deferral_step["w"][0] = 5
        st.applied_after_deferral_step["w"][0] = 0
    before = vec.aggregates.as_dict()
    mask = _torch.tensor([True])
    for st in (vec, ora):
        st._close_events_masked(
            first=st.first_deferral_step["w"],
            after=st.applied_after_deferral_step["w"],
            close_mask=mask,
            now_step=10,
            reason="residual_clear",
        )
    _assert_store_equiv(vec, ora, label="residual_early_censor_net_dec")
    after = vec.aggregates.as_dict()
    assert after["N_events_evaluable"] == before["N_events_evaluable"]
    assert after["N_events_evaluable_early"] == before["N_events_evaluable_early"]
    assert after["N_events_censored_insufficient_followup"] == before[
        "N_events_censored_insufficient_followup"
    ] + 1
    assert after["N_never_applied_within_H"] == before["N_never_applied_within_H"]
    assert after["N_survived_applied_within_H"] == before["N_survived_applied_within_H"]


def test_finalize_window_19m_micro_perf_cpu():
    """CPU-static smoke: ~19M open events finalize must complete in <5s.

    Binding perf-receipt fields (acceptance #3):
    - tensor shape: (19_000_000,) single-arm int32 trackers
    - open fraction: 0.90
    - warmup policy: none — single timed finalize; no prior warmup call
    - elapsed: asserted < 5.0s (printed via assertion message on fail)
    - allocation caveat: from_q_levels already residencies 3×N int32 tensors;
      finalize allocates transient bool class-masks (~O(N) each) freed after the
      call; no persistent growth beyond zeroed trackers
    - <5s does NOT waive the paired ≤0.15 overhead gate
    """
    n = 19_000_000
    shape = (n,)
    open_frac = 0.90
    warmup_policy = "none"
    st = _seeded_store({"w": shape}, steps=150, H=32)
    first = st.first_deferral_step["w"]
    after = st.applied_after_deferral_step["w"]
    open_n = int(open_frac * n)
    first[:open_n] = 10
    after[:open_n] = 0
    # no warmup finalize
    t0 = time.perf_counter()
    st.finalize_window(final_step=150)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"finalize 19M took {elapsed:.3f}s (>=5s); "
        f"shape={shape} open_frac={open_frac} warmup={warmup_policy}"
    )
    assert int(st.first_deferral_step["w"].sum().item()) == 0
    assert (
        st.aggregates.N_events_evaluable
        + st.aggregates.N_events_censored_insufficient_followup
        == open_n
    )
    # stash for receipt consumers reading test stdout / attributes
    test_finalize_window_19m_micro_perf_cpu.last_perf = {  # type: ignore[attr-defined]
        "tensor_shape": shape,
        "open_fraction": open_frac,
        "warmup_policy": warmup_policy,
        "elapsed_s": elapsed,
        "allocation_caveat": (
            "3xN int32 trackers resident pre-call; transient bool masks O(N) during "
            "finalize; no persistent growth beyond zeroed trackers"
        ),
        "does_not_waive_paired_0_15": True,
    }


def test_phase_oracle_is_independent_frozen_copy():
    """R5: phase oracle must not call production process_pre or 2A short-circuit helpers."""
    import inspect

    src = inspect.getsource(_oracle_process_pre_writeback)
    body = src.split('"""', 2)[-1]
    assert "hits_by_arm" not in body
    assert "had_hit" not in body
    assert "t <= H" not in body and "t<=H" not in body
    names = set(_oracle_process_pre_writeback.__code__.co_names)
    # May call _close_events_masked on self (oracle-bound); must not name production short-circuit locals
    assert "hits_by_arm" not in names


def test_R1_hit_mask_retained_before_after_mutation():
    """R1: pre-mutation hit is the close_mask input; mutating first would self-reference."""
    # Defer at step 2, apply at step 5 within H → survived via retained hit
    st = _seeded_store({"w": (2,)}, steps=150, H=32)
    none = _torch.zeros(2, dtype=_torch.bool)
    st.process_pre_writeback(
        candidate_masks={"w": _torch.tensor([True, False])},
        applied_masks={"w": none},
        step=2,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 2
    assert int(st.applied_after_deferral_step["w"][0].item()) == 0
    st.process_pre_writeback(
        candidate_masks={"w": _torch.tensor([True, False])},
        applied_masks={"w": _torch.tensor([True, False])},
        step=5,
        n_candidates=1,
        n_applied=1,
    )
    # closed as survived; trackers zeroed
    assert int(st.first_deferral_step["w"][0].item()) == 0
    assert int(st.applied_after_deferral_step["w"][0].item()) == 0
    assert st.aggregates.N_survived_applied_within_H >= 1


def test_R2_sentinel_deferral_at_step_1_sets_first_eq_1():
    """R2: live deferral at step 1 sets first==1; sentinel first==0 never collides."""
    st = _seeded_store({"w": (2,)}, steps=150, H=32)
    assert int(st.first_deferral_step["w"][0].item()) == 0  # sentinel
    st.process_pre_writeback(
        candidate_masks={"w": _torch.tensor([True, False])},
        applied_masks={"w": _torch.zeros(2, dtype=_torch.bool)},
        step=1,
        n_candidates=1,
        n_applied=0,
    )
    assert int(st.first_deferral_step["w"][0].item()) == 1
    assert int(st.first_deferral_step["w"][1].item()) == 0


def test_R4_step_eq_H_expiry_structurally_empty():
    """R4: at step==H, expired path is structurally empty (no horizon closes)."""
    H = 32
    steps = 150
    vec = _seeded_store({"w": (4,)}, steps=steps, H=H)
    ora = _bind_oracle(_clone_store(vec))
    none = _torch.zeros(4, dtype=_torch.bool)
    # defer all at step 1; never apply → at step H still open, not yet expired
    for st in (vec, ora):
        st.process_pre_writeback(
            candidate_masks={"w": _torch.ones(4, dtype=_torch.bool)},
            applied_masks={"w": none},
            step=1,
            n_candidates=4,
            n_applied=0,
        )
    before_censor = vec.aggregates.N_events_censored_insufficient_followup
    before_never = vec.aggregates.N_never_applied_within_H
    before_eval = vec.aggregates.N_events_evaluable
    for st in (vec, ora):
        st.process_pre_writeback(
            candidate_masks={"w": none},
            applied_masks={"w": none},
            step=H,
            n_candidates=0,
            n_applied=0,
        )
    _assert_store_equiv(vec, ora, label="R4_step_eq_H")
    # no horizon_expired closes at t==H (t-first = H-1 < H)
    assert int(vec.first_deferral_step["w"].sum().item()) == 4  # still open
    assert vec.aggregates.N_never_applied_within_H == before_never
    assert vec.aggregates.N_events_evaluable == before_eval
    assert vec.aggregates.N_events_censored_insufficient_followup == before_censor


def test_R4_step_eq_H_plus_1_expiry_can_fire():
    """R4: at step==H+1, deferred-at-1 never-applied events expire."""
    H = 32
    steps = 150
    vec = _seeded_store({"w": (4,)}, steps=steps, H=H)
    ora = _bind_oracle(_clone_store(vec))
    none = _torch.zeros(4, dtype=_torch.bool)
    for st in (vec, ora):
        st.process_pre_writeback(
            candidate_masks={"w": _torch.ones(4, dtype=_torch.bool)},
            applied_masks={"w": none},
            step=1,
            n_candidates=4,
            n_applied=0,
        )
        st.process_pre_writeback(
            candidate_masks={"w": none},
            applied_masks={"w": none},
            step=H + 1,
            n_candidates=0,
            n_applied=0,
        )
    _assert_store_equiv(vec, ora, label="R4_step_eq_H_plus_1")
    assert int(vec.first_deferral_step["w"].sum().item()) == 0  # closed
    assert vec.aggregates.N_never_applied_within_H >= 4


def test_late_apply_aa_fs_H_plus_1_persists_through_no_hit_step():
    """R5 late-apply persistence: aa-fs==H+1 stays open through next no-hit step."""
    H = 32
    steps = 150
    vec = _seeded_store({"w": (2,)}, steps=steps, H=H)
    ora = _bind_oracle(_clone_store(vec))
    none = _torch.zeros(2, dtype=_torch.bool)
    # defer at step 1
    for st in (vec, ora):
        st.process_pre_writeback(
            candidate_masks={"w": _torch.tensor([True, False])},
            applied_masks={"w": none},
            step=1,
            n_candidates=1,
            n_applied=0,
        )
    # Manually set a late-apply marker aa = 1+(H+1) without closing (simulates
    # applied_after recorded but not yet closed by just_survived because aa-fs > H).
    # Production hit path only sets after when after==0; just_survived requires
    # 0 < aa-fs <= H. So late-apply with aa-fs==H+1 stays open until finalize/residual.
    for st in (vec, ora):
        st.applied_after_deferral_step["w"][0] = 1 + (H + 1)  # aa-fs == H+1
    # next no-hit step must NOT clear/skip the late event
    for st in (vec, ora):
        st.process_pre_writeback(
            candidate_masks={"w": none},
            applied_masks={"w": none},
            step=1 + (H + 1) + 1,
            n_candidates=0,
            n_applied=0,
        )
    _assert_store_equiv(vec, ora, label="late_apply_persist")
    assert int(vec.first_deferral_step["w"][0].item()) == 1  # still open
    assert int(vec.applied_after_deferral_step["w"][0].item()) == 1 + (H + 1)
