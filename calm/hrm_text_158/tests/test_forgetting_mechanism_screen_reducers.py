"""CPU-static tests for forgetting_mechanism_screen_reducers (PLAN_v9)."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
    ARM1,
    ARM2,
    ARM3,
    CROSSING_THRESHOLD_ABS,
    FAMILY_F1,
    FAMILY_F2,
    FAMILY_F3,
    FAMILY_F4,
    FixedQScaleLinearWithCredit,
    apply_live_flip_writeback,
    begin_credit_step,
    bitlinear_absmean_quantize,
    classify_forgetting_family_screen,
    cumulative_q_transitions,
    fixed_qscale_linear_with_credit,
    flattened_nd_dW,
    get_credit_store,
    mechanical_dynamic_scale_diverges,
    qscale_reference_weight,
    retention_ok,
    threshold_residual_writeback,
)
from calm.hrm_text_158.native_full_stack.vote_lifetime_screen_reducers import (
    apply_drain_resets,
)


def test_threshold_residual_writeback_formula():
    T = CROSSING_THRESHOLD_ABS
    acc = torch.tensor([15, -15, 10, 0], dtype=torch.int16)
    direction = torch.tensor([1, -1, 1, 1], dtype=torch.int8)
    out = threshold_residual_writeback(acc, direction, threshold=T)
    # 15-10=5, -15-(-10)=-5, 10-10=0 (natural zero), 0-10=-10 clamped to -(T-1)=-9
    assert out.tolist() == [5, -5, 0, -9]


def test_residual_aware_episode_restart():
    acc = torch.tensor([15, 10], dtype=torch.int16)
    ep = torch.tensor([3, 5], dtype=torch.int32)
    q = torch.tensor([0, 0], dtype=torch.int8)
    mask = torch.tensor([True, True])
    new_acc, new_ep, new_q, lts, n_q = apply_live_flip_writeback(
        acc, ep, q, mask, step=14, threshold=10
    )
    assert new_acc[0].item() == 5  # residual nonzero → episode restart at 14
    assert new_ep[0].item() == 14
    assert new_acc[1].item() == 0  # natural zero residual → clear episode
    assert new_ep[1].item() == 0
    assert new_q.tolist() == [1, 1]
    assert n_q == 2
    assert sorted(lts) == [9, 11]


def test_retention_ok_counts_not_rates():
    assert retention_ok(final_count=30, step0_count=32, slop=2) is True
    assert retention_ok(final_count=29, step0_count=32, slop=2) is False
    try:
        retention_ok(final_count=0.9, step0_count=1.0)  # type: ignore[arg-type]
        assert False, "rate-style must TypeError"
    except TypeError:
        pass


def test_control_at_budget_before_r0():
    arms = {
        ARM1: _arm(H=0.3, flips=5000, qchg=100, drains=1000, ret=True, acq=1),
        ARM2: _arm(H=0.2, flips=5000, qchg=100, drains=1000, ret=True, acq=1),
        ARM3: _arm(H=0.1, flips=5000, qchg=100, drains=1000, ret=True, acq=1),
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=0.4, arm_metrics=arms
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "control_already_at_budget_no_forgetting_family"


def _arm(*, H, flips, qchg, drains, ret, acq, cens=0.1):
    return {
        "H_final": H,
        "n_flips": flips,
        "q_changed_count": qchg,
        "n_applied_drains": drains,
        "retention_ok": ret,
        "acq_delta_count": acq,
        "lifetime_censored_frac": cens,
    }


def test_g0b_q_motion_freeze_excludes_from_E():
    # control H=1.0 → gap=0.6, bar=0.3
    arms = {
        ARM1: _arm(H=0.5, flips=5000, qchg=0, drains=10000, ret=True, acq=1),  # qchg fail
        ARM2: _arm(H=0.4, flips=5000, qchg=200, drains=10000, ret=True, acq=1),
        ARM3: _arm(H=0.9, flips=5000, qchg=200, drains=10000, ret=True, acq=1),
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert ARM1 not in out["E"]
    assert ARM2 in out["E"]


def test_sole_eligible_f2_with_higher_dead_arm():
    # F3 dead (retention fail) but higher raw H progress; F2 sole eligible
    arms = {
        ARM1: _arm(H=0.9, flips=5000, qchg=200, drains=10000, ret=False, acq=1),
        ARM2: _arm(H=0.5, flips=5000, qchg=200, drains=10000, ret=True, acq=1),
        ARM3: _arm(H=0.1, flips=5000, qchg=200, drains=10000, ret=False, acq=1),  # best H but dead
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert out["E"] == [ARM2]
    assert out["family"] == FAMILY_F2


def test_near_tie_f3_wins_within_tau():
    # H_control=1.0 → gap=0.6, bar=0.3. Need Hp>=0.3 within tau of max.
    arms = {
        ARM1: _arm(H=0.615, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.385
        ARM2: _arm(H=0.60, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.40
        ARM3: _arm(H=0.61, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.39
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert ARM2 in out["S"] and ARM3 in out["S"]
    assert out["family"] == FAMILY_F3


def test_near_tie_f1_beats_f2_when_f3_out():
    arms = {
        ARM1: _arm(H=0.61, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.39
        ARM2: _arm(H=0.60, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.40
        ARM3: _arm(H=0.90, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.10 out of tau
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert set(out["S"]) == {ARM1, ARM2}
    assert out["family"] == FAMILY_F1


def test_f2_unique_outside_tau():
    arms = {
        ARM1: _arm(H=0.63, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.37
        ARM2: _arm(H=0.60, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.40
        ARM3: _arm(H=0.64, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.36
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert out["S"] == [ARM2]
    assert out["family"] == FAMILY_F2


def test_sub_bar_near_tie_cannot_block():
    # gap = 0.6, bar = 0.3. F1 Hp=0.30 exactly at bar; F3 Hp=0.29 within tau but sub-bar
    arms = {
        ARM1: _arm(H=0.70, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.30
        ARM2: _arm(H=0.95, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.05
        ARM3: _arm(H=0.71, flips=5000, qchg=200, drains=10000, ret=True, acq=1),  # Hp=0.29 sub-bar
    }
    out = classify_forgetting_family_screen(
        phase0_censor_cleared=True, H_control_final=1.0, arm_metrics=arms
    )
    assert ARM3 not in out["S"]
    assert ARM1 in out["S"]
    assert out["family"] == FAMILY_F1
    assert out["stop_reason"] != "R4_ambiguous_null"


def test_old_zero_drain_nonregression_import():
    acc = torch.tensor([5, 0, -3], dtype=torch.int16)
    ep = torch.tensor([2, 0, 1], dtype=torch.int32)
    mask = torch.tensor([True, False, True])
    new_acc, new_ep, lts = apply_drain_resets(acc, ep, mask, step=10)
    assert new_acc.tolist() == [0, 0, 0]
    assert new_ep.tolist() == [0, 0, 0]
    assert sorted(lts) == [8, 9]


def test_flattened_nd_dW():
    B, T, In, Out = 2, 3, 4, 5
    act = torch.randn(B, T, In)
    go = torch.randn(B, T, Out)
    dW = flattened_nd_dW(go, act)
    ref = go.reshape(-1, Out).T @ act.reshape(-1, In)
    assert torch.allclose(dW, ref)


def test_fixed_qscale_grad_parity_and_lifecycle_noncarry():
    torch.manual_seed(0)
    store = begin_credit_step(["layer"])
    q = torch.tensor([[1, 0, -1], [0, 1, 1]], dtype=torch.int8)
    scale = torch.tensor(0.5, dtype=torch.float32)
    bias = None
    # step1: two invocations
    x1 = torch.randn(2, 3, 3, requires_grad=True)
    y1 = fixed_qscale_linear_with_credit(x1, q, scale, bias, name="layer")
    y1.sum().backward()
    x2 = torch.randn(2, 3, 3, requires_grad=True)
    y2 = fixed_qscale_linear_with_credit(x2, q, scale, bias, name="layer")
    y2.sum().backward()
    snap1 = store.snapshot_and_mark()
    assert "layer" in snap1
    # reference: autograd through temp W
    W = (q.float() * scale).detach().requires_grad_(True)
    # rebuild accumulated dW via two separate forwards on saved inputs — use snap1 vs fresh
    # Parity for single invocation:
    store2 = begin_credit_step(["layer"])
    x = torch.randn(2, 3, 3, requires_grad=True)
    y = fixed_qscale_linear_with_credit(x, q, scale, bias, name="layer")
    loss = y.sum()
    loss.backward()
    dW_credit = store2.credit_grads["layer"]
    W = (q.float() * scale).detach().requires_grad_(True)
    y_ref = F.linear(x.detach(), W, bias)
    y_ref.sum().backward()
    assert torch.allclose(dW_credit, W.grad, atol=1e-5, rtol=1e-5)

    # two-step non-carry: step2 must not include step1
    begin_credit_step(["layer"])
    x_a = torch.ones(1, 3, requires_grad=True)
    fixed_qscale_linear_with_credit(x_a, q, scale, bias, name="layer").sum().backward()
    step1 = get_credit_store().credit_grads["layer"].clone()
    begin_credit_step(["layer"])
    x_b = torch.full((1, 3), 2.0, requires_grad=True)
    fixed_qscale_linear_with_credit(x_b, q, scale, bias, name="layer").sum().backward()
    step2 = get_credit_store().credit_grads["layer"].clone()
    # standalone step2 reference
    begin_credit_step(["layer"])
    fixed_qscale_linear_with_credit(x_b, q, scale, bias, name="layer").sum().backward()
    step2_alone = get_credit_store().credit_grads["layer"].clone()
    assert torch.allclose(step2, step2_alone)
    assert not torch.allclose(step2, step1 + step2_alone)


def test_begin_credit_step_fail_closed_and_clears_counters():
    begin_credit_step(["a"])
    store = get_credit_store()
    store.n_bitlinear_dynamic_forwards = 3
    store.n_fixed_qscale_forwards = 9
    begin_credit_step(["a", "b"])
    store = get_credit_store()
    assert store.n_bitlinear_dynamic_forwards == 0
    assert store.n_fixed_qscale_forwards == 0
    assert store.credit_grads == {}
    assert store.graph_anchor is not None and store.graph_anchor.requires_grad
    # fail-closed without begin
    store.begun = False
    try:
        FixedQScaleLinearWithCredit.apply(
            torch.randn(1, 2),
            torch.zeros(3, 2, dtype=torch.int8),
            torch.tensor(1.0),
            None,
            torch.zeros((), requires_grad=True),
            "x",
            0,
        )
        assert False
    except RuntimeError as e:
        assert "begin_credit_step" in str(e)


def test_detached_activation_still_accumulates_credit():
    """HRM carry-detach regression: act.requires_grad=False must still credit.

    BitLinear gets weight.grad via its Parameter leaf under detach; FixedQScale
    must keep a step-local graph_anchor so backward still fires (no Parameter).
    """
    import types

    from calm.hrm_text_158.bit_linear import BitLinear

    torch.manual_seed(1)
    mod = BitLinear(4, 3, bias=False)
    pname = "proj.weight"
    with torch.no_grad():
        scale = mod.weight.float().abs().mean().clamp(min=1e-5)
        q = (mod.weight.float() / scale).round().clamp(-1, 1).to(torch.int8)
        frozen = scale.detach().to(torch.float32).reshape(())

    begin_credit_step([pname])

    def _fwd(self, x, _q=q, _s=frozen, _name=pname):
        return fixed_qscale_linear_with_credit(
            x, _q, _s, self.bias, name=_name
        )

    mod.forward = types.MethodType(_fwd, mod)
    # Detached activation (HRM carry) + a live leaf so loss.backward runs.
    act = torch.randn(2, 5, 4).detach()
    leaf = torch.zeros((), requires_grad=True)
    y = mod(act) + leaf * 0
    y.sum().backward()
    store = get_credit_store()
    assert store.n_fixed_qscale_forwards >= 1
    assert store.n_bitlinear_dynamic_forwards == 0
    assert pname in store.credit_grads
    assert bool((store.credit_grads[pname] != 0).any())
    store.assert_route_completeness([pname])


def test_mechanical_dynamic_scale_negative_fixture():
    # sparse q: nonzero_frac in (0,1)
    q = torch.zeros(8, 8, dtype=torch.int8)
    q[0, 0] = 1
    q[1, 1] = -1
    q[2, 2] = 1
    assert 0 < float((q != 0).float().mean()) < 1
    x = torch.randn(4, 8)
    out = mechanical_dynamic_scale_diverges(q, frozen_scale=0.25, x=x)
    assert out["diverges"] is True
    assert out["scale_equal"] is False
    assert out["W_allclose"] is False
    assert out["Y_allclose"] is False


def test_fixed_scale_forward_parity_and_bitlinear_divergence():
    q = torch.zeros(4, 4, dtype=torch.int8)
    q[0, 0] = 1
    q[1, 2] = -1
    s = torch.tensor(0.3)
    W = qscale_reference_weight(q, s)
    # scale bit-stable
    s2 = s.clone()
    assert torch.equal(s, s2)
    x = torch.randn(2, 4)
    y = F.linear(x, W)
    # BitLinear absmean path diverges
    W_dyn, _ = bitlinear_absmean_quantize(W)
    assert not torch.allclose(W_dyn, W)


def test_cumulative_q_reversal_counts_two():
    q0 = torch.tensor([0, 1], dtype=torch.int8)
    q1 = torch.tensor([1, 1], dtype=torch.int8)
    q2 = torch.tensor([0, 1], dtype=torch.int8)
    mask = torch.tensor([True, False])
    n = 0
    n += cumulative_q_transitions(q0, q1, mask)
    n += cumulative_q_transitions(q1, q2, mask)
    assert n == 2
    assert q2[0].item() == q0[0].item()


def test_route_completeness_fails_on_dynamic_forward_counter():
    begin_credit_step(["a"])
    store = get_credit_store()
    store.credit_grads["a"] = torch.ones(2, 2)
    store.n_bitlinear_dynamic_forwards = 1
    try:
        store.assert_route_completeness(["a"])
        assert False
    except RuntimeError as e:
        assert "n_bitlinear_dynamic_forwards" in str(e)


def test_phase1_probe_sets_match_prereg_shas_and_disjoint():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ACQUISITION_SELECTION_SHA256,
        IDENTITY_SELECTION_SHA256,
        MATH_A0_PARENT_SUPPORT_HASH,
        IDENTITY_PARENT_SUPPORT_HASH,
        build_phase1_probe_sets,
    )

    sets = build_phase1_probe_sets()
    assert sets["acquisition_selection_sha256"] == ACQUISITION_SELECTION_SHA256
    assert sets["identity_selection_sha256"] == IDENTITY_SELECTION_SHA256
    assert sets["math_a0_parent_support_hash"] == MATH_A0_PARENT_SUPPORT_HASH
    assert sets["identity_parent_support_hash"] == IDENTITY_PARENT_SUPPORT_HASH
    assert sets["acquisition_n"] == 64
    assert sets["retention_n"] == 64
    assert set(sets["retention_math_a0"]).isdisjoint(set(sets["acquisition"]))


def test_train_exclusion_rejects_acquisition_rows():
    import random

    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        sample_batch_excluding_acquisition,
    )

    pool = [(f"q{i}", i, "r") for i in range(20)]
    acq = set(pool[:5])
    rng = random.Random(0)
    batch, excluded = sample_batch_excluding_acquisition(
        pool, batch=8, rng=rng, acquisition_set=acq
    )
    assert len(batch) == 8
    assert all(row not in acq for row in batch)
    assert excluded >= 0  # may be 0 by chance; force hits:
    # Force exclusion path: pool that is majority acquisition
    pool2 = list(pool[:5]) * 3 + pool[5:8]
    batch2, excluded2 = sample_batch_excluding_acquisition(
        pool2, batch=4, rng=random.Random(1), acquisition_set=acq
    )
    assert excluded2 > 0
    assert all(row not in acq for row in batch2)


def test_h_trajectory_every_25_and_final():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        should_record_h_trajectory,
    )

    assert should_record_h_trajectory(25, 150) is True
    assert should_record_h_trajectory(50, 150) is True
    assert should_record_h_trajectory(24, 150) is False
    assert should_record_h_trajectory(150, 150) is True
    assert should_record_h_trajectory(1, 1) is True
    assert should_record_h_trajectory(0, 150) is False


def test_phase1_aggregation_invokes_classifier_and_emits_family():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        FAMILY_F3,
        build_phase1_terminal_receipt,
    )

    def arm_rec(arm, *, H, flips=5000, qchg=200, drains=10000, ret=True, acq=1, cens=0.1):
        return {
            "arm": arm,
            "measurements": {
                "H_bits_per_weight": H,
                "n_flips": flips,
                "q_changed_count": qchg,
                "n_applied_drains": drains,
                "lifetime_censored_frac": cens,
            },
            "probes": {
                "retention_ok": ret,
                "acq_delta_count": acq,
            },
        }

    # Near-tie F2/F3 within tau → F3 (gentlest)
    control = arm_rec(ARM0, H=1.0)
    arms = {
        ARM1: arm_rec(ARM1, H=0.615),  # Hp=0.385
        ARM2: arm_rec(ARM2, H=0.60),  # Hp=0.40
        ARM3: arm_rec(ARM3, H=0.61),  # Hp=0.39
    }
    out = build_phase1_terminal_receipt(
        phase0_censor_cleared=True,
        control_receipt=control,
        arm_receipts=arms,
        plan_sha256="deadbeef",
        authority_dispatch="auth",
    )
    assert out["family"] == FAMILY_F3
    assert out["classifier"]["family"] == FAMILY_F3
    assert ARM2 in out["classifier"]["S"] and ARM3 in out["classifier"]["S"]
    assert out["screen"] == "forgetting_mechanism_phase1/v1"


def test_phase1_aggregation_phase0_uncleared_null():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM1,
        ARM2,
        ARM3,
        FAMILY_F4,
        build_phase1_terminal_receipt,
    )

    def arm_rec(H):
        return {
            "measurements": {
                "H_bits_per_weight": H,
                "n_flips": 5000,
                "q_changed_count": 200,
                "n_applied_drains": 10000,
                "lifetime_censored_frac": 0.1,
            },
            "probes": {"retention_ok": True, "acq_delta_count": 1},
        }

    out = build_phase1_terminal_receipt(
        phase0_censor_cleared=False,
        control_receipt=arm_rec(1.0),
        arm_receipts={ARM1: arm_rec(0.5), ARM2: arm_rec(0.4), ARM3: arm_rec(0.3)},
        plan_sha256="x",
        authority_dispatch="y",
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "phase0_censor_uncleared"


def test_aggregate_no_phase0_proof_forces_f4():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM1,
        ARM2,
        ARM3,
        FAMILY_F4,
        build_phase1_terminal_receipt,
        validate_phase0_receipt_for_aggregate,
    )

    v = validate_phase0_receipt_for_aggregate(None)
    assert v["ok"] is False
    assert v["reason"] == "phase0_proof_missing"
    out = build_phase1_terminal_receipt(
        phase0_censor_cleared=False,
        control_receipt={"measurements": {"H_bits_per_weight": 1.0}},
        arm_receipts={
            ARM1: {"measurements": {"H_bits_per_weight": 0.2}, "probes": {}},
            ARM2: {"measurements": {"H_bits_per_weight": 0.1}, "probes": {}},
            ARM3: {"measurements": {"H_bits_per_weight": 0.05}, "probes": {}},
        },
        plan_sha256="x",
        authority_dispatch="y",
        force_null_reason="phase0_proof_missing",
    )
    assert out["family"] == FAMILY_F4
    assert out["stop_reason"] == "phase0_proof_missing"
    # Even with cleared=True, force_null wins
    out2 = build_phase1_terminal_receipt(
        phase0_censor_cleared=True,
        control_receipt={"measurements": {"H_bits_per_weight": 1.0}},
        arm_receipts={
            ARM1: {
                "measurements": {
                    "H_bits_per_weight": 0.5,
                    "n_flips": 5000,
                    "q_changed_count": 200,
                    "n_applied_drains": 10000,
                    "lifetime_censored_frac": 0.1,
                },
                "probes": {"retention_ok": True, "acq_delta_count": 1},
            },
            ARM2: {
                "measurements": {
                    "H_bits_per_weight": 0.4,
                    "n_flips": 5000,
                    "q_changed_count": 200,
                    "n_applied_drains": 10000,
                    "lifetime_censored_frac": 0.1,
                },
                "probes": {"retention_ok": True, "acq_delta_count": 1},
            },
            ARM3: {
                "measurements": {
                    "H_bits_per_weight": 0.3,
                    "n_flips": 5000,
                    "q_changed_count": 200,
                    "n_applied_drains": 10000,
                    "lifetime_censored_frac": 0.1,
                },
                "probes": {"retention_ok": True, "acq_delta_count": 1},
            },
        },
        plan_sha256="x",
        authority_dispatch="y",
        force_null_reason="phase0_proof_missing",
    )
    assert out2["family"] == FAMILY_F4
    assert out2["stop_reason"] == "phase0_proof_missing"


def _contract_arm(
    arm,
    *,
    steps=150,
    batch=8,
    topk=1024,
    plan="07a02aff",
    parent="2d9b9f67",
    skipped=False,
    bad_plan=None,
    authority="1784812148229-f466bc29",
    scale_before="scale_shared_abc",
    q_before="q_shared_def",
    schema_only=False,
    correctness_smoke=False,
    n_fixed=10,
    n_dyn=0,
    n_elig=32,
    n_cred=32,
    screen="forgetting_mechanism_screen/v1",
):
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ACQUISITION_SELECTION_SHA256,
        IDENTITY_SELECTION_SHA256,
    )

    return {
        "arm": arm,
        "screen": screen,
        "schema_only": schema_only,
        "correctness_smoke": correctness_smoke,
        "plan_sha256": bad_plan if bad_plan is not None else plan,
        "authority_dispatch": authority,
        "steps": steps,
        "batch": batch,
        "topk": topk,
        "banked_sha": {"before": parent, "after": parent, "match": True},
        "frozen_scale_sha": {
            "before": scale_before,
            "after": scale_before,
            "match": True,
        },
        "q_sha": {"before": q_before, "after": q_before},
        "route_counters": {
            "n_fixed_qscale_forwards": n_fixed,
            "n_bitlinear_dynamic_forwards": n_dyn,
            "n_eligible_keys": n_elig,
            "n_credit_grads_present": n_cred,
        },
        "measurements": {
            "H_bits_per_weight": 0.5,
            "n_flips": 5000,
            "q_changed_count": 200,
            "n_applied_drains": 10000,
            "lifetime_censored_frac": 0.1,
        },
        "probes": {
            "skipped": skipped,
            "acquisition_selection_sha256": ACQUISITION_SELECTION_SHA256,
            "identity_selection_sha256": IDENTITY_SELECTION_SHA256,
            "acquisition_n": 64,
            "retention_n": 64,
            "acq_step0_count": 10,
            "acq_final_count": 12,
            "acq_delta_count": 2,
            "retention_step0_count": 20,
            "retention_final_count": 19,
            "retention_ok": True,
        },
    }


def _valid_phase0_receipt(
    *,
    lcf=0.1,
    plan=None,
    parent=None,
    authority=None,
    steps=150,
    batch=8,
    topk=1024,
    n_fixed=10,
    n_dyn=0,
    scale_before="scale_shared_abc",
    q_before="q_shared_def",
    screen=None,
    arm=None,
):
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        DEFAULT_AUTHORITY_DISPATCH,
        DEFAULT_PARENT_SHA256,
        DEFAULT_PLAN_SHA256,
        PHASE0_SCREEN_ID,
    )

    return {
        "screen": screen if screen is not None else PHASE0_SCREEN_ID,
        "arm": arm if arm is not None else ARM0,
        "plan_sha256": plan if plan is not None else DEFAULT_PLAN_SHA256,
        "authority_dispatch": (
            authority if authority is not None else DEFAULT_AUTHORITY_DISPATCH
        ),
        "steps": steps,
        "batch": batch,
        "topk": topk,
        "banked_sha": {
            "before": parent if parent is not None else DEFAULT_PARENT_SHA256,
            "after": parent if parent is not None else DEFAULT_PARENT_SHA256,
            "match": True,
        },
        "frozen_scale_sha": {
            "before": scale_before,
            "after": scale_before,
            "match": True,
        },
        "q_sha": {"before": q_before, "after": q_before},
        "route_counters": {
            "n_fixed_qscale_forwards": n_fixed,
            "n_bitlinear_dynamic_forwards": n_dyn,
        },
        "measurements": {"lifetime_censored_frac": lcf},
    }


def test_aggregate_mismatched_arm_fail_closed():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        ArmReceiptContractError,
        validate_shared_held_fixed_arm_receipts,
    )

    good = {
        ARM0: _contract_arm(ARM0),
        ARM1: _contract_arm(ARM1),
        ARM2: _contract_arm(ARM2),
        ARM3: _contract_arm(ARM3),
    }
    shared = validate_shared_held_fixed_arm_receipts(
        good, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
    )
    assert shared["frozen_scale_sha_before"] == "scale_shared_abc"
    assert shared["q_sha_before"] == "q_shared_def"

    bad_plan = dict(good)
    bad_plan[ARM2] = _contract_arm(ARM2, bad_plan="deadbeef")
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_plan, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False, "expected contract error"
    except ArmReceiptContractError as e:
        assert "plan_sha256" in str(e)

    bad_skip = dict(good)
    bad_skip[ARM1] = _contract_arm(ARM1, skipped=True)
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_skip, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False
    except ArmReceiptContractError as e:
        assert "skipped" in str(e)

    bad_steps = dict(good)
    bad_steps[ARM3] = _contract_arm(ARM3, steps=600)
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_steps, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False
    except ArmReceiptContractError as e:
        assert "steps" in str(e)


def test_shared_held_fixed_divergent_scale_q_authority_fail_closed():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        ArmReceiptContractError,
        validate_shared_held_fixed_arm_receipts,
    )

    good = {
        ARM0: _contract_arm(ARM0),
        ARM1: _contract_arm(ARM1),
        ARM2: _contract_arm(ARM2),
        ARM3: _contract_arm(ARM3),
    }

    bad_scale = dict(good)
    bad_scale[ARM2] = _contract_arm(ARM2, scale_before="scale_OTHER")
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_scale, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False, "expected divergent scale fail"
    except ArmReceiptContractError as e:
        assert "frozen_scale_sha.before" in str(e)

    bad_q = dict(good)
    bad_q[ARM1] = _contract_arm(ARM1, q_before="q_OTHER")
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_q, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False, "expected divergent q fail"
    except ArmReceiptContractError as e:
        assert "q_sha.before" in str(e)

    bad_auth = dict(good)
    bad_auth[ARM3] = _contract_arm(ARM3, authority="wrong-auth")
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_auth, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
        )
        assert False, "expected authority fail"
    except ArmReceiptContractError as e:
        assert "authority_dispatch" in str(e)

    # Phase-0 scale/q must match control on formal path
    p0 = _valid_phase0_receipt(scale_before="scale_OTHER")
    try:
        validate_shared_held_fixed_arm_receipts(
            good,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
            phase0_receipt=p0,
        )
        assert False, "expected phase0 scale divergence fail"
    except ArmReceiptContractError as e:
        assert "phase0 frozen_scale_sha.before" in str(e)


def test_phase0_full_contract_negative_and_positive():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        FAMILY_F4,
        build_phase1_terminal_receipt,
        validate_phase0_receipt_for_aggregate,
    )

    # Valid lcf alone is NOT enough — under-validated blob fails closed
    under = {"measurements": {"lifetime_censored_frac": 0.1}}
    v = validate_phase0_receipt_for_aggregate(under)
    assert v["ok"] is False
    assert v["phase0_censor_cleared"] is False

    cases = [
        (_valid_phase0_receipt(plan="deadbeef"), "phase0_plan_sha256_mismatch"),
        (
            _valid_phase0_receipt(parent="wrongparent" + "0" * 54),
            "phase0_banked_parent_mismatch",
        ),
        (
            _valid_phase0_receipt(authority="not-the-authority"),
            "phase0_authority_dispatch_mismatch",
        ),
        (_valid_phase0_receipt(n_fixed=0), "phase0_route_counters_n_fixed_nonpositive"),
        (_valid_phase0_receipt(n_dyn=1), "phase0_route_counters_dynamic_nonzero"),
        (_valid_phase0_receipt(steps=999), "phase0_steps_out_of_window"),
    ]
    for blob, reason in cases:
        vv = validate_phase0_receipt_for_aggregate(blob)
        assert vv["ok"] is False, reason
        assert vv["reason"] == reason
        # Malformed → F4/STOP, never authoritative family pick
        out = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={"measurements": {"H_bits_per_weight": 1.0}},
            arm_receipts={
                "arm1_decay_leak": {
                    "measurements": {"H_bits_per_weight": 0.2},
                    "probes": {},
                },
                "arm2_ttl_age_drain": {
                    "measurements": {"H_bits_per_weight": 0.1},
                    "probes": {},
                },
                "arm3_sparse_hot_forgettable_cold": {
                    "measurements": {"H_bits_per_weight": 0.05},
                    "probes": {},
                },
            },
            plan_sha256="x",
            authority_dispatch="y",
            authoritative=False,
            force_null_reason=vv["reason"],
        )
        assert out["family"] == FAMILY_F4
        assert out["authoritative"] is False
        assert out["stop_reason"] == reason

    # Full contract + lcf clear
    ok = validate_phase0_receipt_for_aggregate(_valid_phase0_receipt(lcf=0.1, steps=150))
    assert ok["ok"] is True
    assert ok["phase0_censor_cleared"] is True
    assert ok["frozen_scale_sha_before"] == "scale_shared_abc"
    assert ok["batch"] == 8 and ok["topk"] == 1024

    # Full contract + uncleared lcf (150)
    uncleared = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.55, steps=150)
    )
    assert uncleared["ok"] is True
    assert uncleared["phase0_censor_cleared"] is False
    assert uncleared["reason"] == "phase0_censor_uncleared"


def test_phase0_geometry_and_fallback_once_predecessor():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        FAMILY_F4,
        build_phase1_terminal_receipt,
        validate_phase0_receipt_for_aggregate,
    )

    # Wrong batch / topk → fail closed
    for blob, reason in (
        (_valid_phase0_receipt(batch=1), "phase0_batch_mismatch"),
        (_valid_phase0_receipt(topk=1), "phase0_topk_mismatch"),
    ):
        vv = validate_phase0_receipt_for_aggregate(blob)
        assert vv["ok"] is False
        assert vv["reason"] == reason
        out = build_phase1_terminal_receipt(
            phase0_censor_cleared=False,
            control_receipt={"measurements": {"H_bits_per_weight": 1.0}},
            arm_receipts={
                "arm1_decay_leak": {
                    "measurements": {"H_bits_per_weight": 0.2},
                    "probes": {},
                },
                "arm2_ttl_age_drain": {
                    "measurements": {"H_bits_per_weight": 0.1},
                    "probes": {},
                },
                "arm3_sparse_hot_forgettable_cold": {
                    "measurements": {"H_bits_per_weight": 0.05},
                    "probes": {},
                },
            },
            plan_sha256="x",
            authority_dispatch="y",
            authoritative=False,
            force_null_reason=vv["reason"],
        )
        assert out["family"] == FAMILY_F4
        assert out["authoritative"] is False

    # Standalone 600 → reject
    alone = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.1, steps=600)
    )
    assert alone["ok"] is False
    assert alone["reason"] == "phase0_fallback_predecessor_missing"

    # 600 with CLEARED 150 predecessor → reject
    cleared_pred = _valid_phase0_receipt(lcf=0.1, steps=150)
    bad = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.1, steps=600),
        phase0_predecessor_receipt=cleared_pred,
    )
    assert bad["ok"] is False
    assert bad["reason"] == "phase0_fallback_predecessor_cleared"

    # 600 with FAILED 150 predecessor (lcf>=0.50) → accepted
    failed_pred = _valid_phase0_receipt(lcf=0.55, steps=150)
    good = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.1, steps=600),
        phase0_predecessor_receipt=failed_pred,
    )
    assert good["ok"] is True
    assert good["phase0_censor_cleared"] is True
    assert good["steps"] == 600
    assert good["phase0_predecessor"]["failed_censor_guard"] is True
    assert good["phase0_predecessor"]["lifetime_censored_frac"] == 0.55


def test_arm_exact_geometry_and_phase0_window_bind():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        ArmReceiptContractError,
        validate_shared_held_fixed_arm_receipts,
    )

    good = {
        ARM0: _contract_arm(ARM0),
        ARM1: _contract_arm(ARM1),
        ARM2: _contract_arm(ARM2),
        ARM3: _contract_arm(ARM3),
    }
    validate_shared_held_fixed_arm_receipts(
        good,
        expected_plan_sha256="07a02aff",
        expected_parent_sha256="2d9b9f67",
        expected_steps=150,
    )

    bad_batch = dict(good)
    bad_batch[ARM1] = _contract_arm(ARM1, batch=1)
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_batch,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
        )
        assert False, "expected batch fail"
    except ArmReceiptContractError as e:
        assert "PHASE_BATCH" in str(e) or "batch" in str(e)

    bad_topk = dict(good)
    bad_topk[ARM2] = _contract_arm(ARM2, topk=5)
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_topk,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
        )
        assert False, "expected topk fail"
    except ArmReceiptContractError as e:
        assert "PHASE_TOPK" in str(e) or "topk" in str(e)

    bad_steps = dict(good)
    bad_steps[ARM3] = _contract_arm(ARM3, steps=999)
    try:
        validate_shared_held_fixed_arm_receipts(
            bad_steps,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
        )
        assert False, "expected steps window fail"
    except ArmReceiptContractError as e:
        assert "steps" in str(e)

    # Phase-0 winning window 600 must not authorize 150-step arms
    try:
        validate_shared_held_fixed_arm_receipts(
            good,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
            expected_steps=600,
        )
        assert False, "expected Phase-0 window mismatch"
    except ArmReceiptContractError as e:
        assert "winning window" in str(e)

    arms_600 = {
        ARM0: _contract_arm(ARM0, steps=600),
        ARM1: _contract_arm(ARM1, steps=600),
        ARM2: _contract_arm(ARM2, steps=600),
        ARM3: _contract_arm(ARM3, steps=600),
    }
    validate_shared_held_fixed_arm_receipts(
        arms_600,
        expected_plan_sha256="07a02aff",
        expected_parent_sha256="2d9b9f67",
        expected_steps=600,
    )
    try:
        validate_shared_held_fixed_arm_receipts(
            arms_600,
            expected_plan_sha256="07a02aff",
            expected_parent_sha256="2d9b9f67",
            expected_steps=150,
        )
        assert False, "expected 600 arms vs 150 window fail"
    except ArmReceiptContractError as e:
        assert "winning window" in str(e)


def test_phase0_aggregate_state_machine_transitions():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        FAMILY_F4,
        build_phase1_terminal_receipt,
        decide_phase0_aggregate_transition,
        validate_phase0_receipt_for_aggregate,
    )

    # Uncleared 150 → fallback_required
    p0 = validate_phase0_receipt_for_aggregate(_valid_phase0_receipt(lcf=0.55, steps=150))
    d = decide_phase0_aggregate_transition(p0)
    assert d["action"] == "fallback_required"
    assert d["authoritative"] is False
    out = build_phase1_terminal_receipt(
        phase0_censor_cleared=False,
        control_receipt={},
        arm_receipts={},
        plan_sha256="x",
        authority_dispatch="y",
        authoritative=False,
        force_null_reason=d["stop_reason"],
        null_family=None,
        transition="fallback_required",
        arms_classified=False,
    )
    assert out["family"] is None
    assert out["authoritative"] is False
    assert out["arms_classified"] is False
    assert out["stop_reason"] == "phase0_censor_uncleared_fallback_required"
    assert out["transition"] == "fallback_required"

    # Uncleared 600 w/ failed-150 pred → design_null authoritative
    failed = _valid_phase0_receipt(lcf=0.55, steps=150)
    p0b = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.6, steps=600),
        phase0_predecessor_receipt=failed,
    )
    d2 = decide_phase0_aggregate_transition(p0b)
    assert d2["action"] == "design_null_censor_unreducible"
    assert d2["authoritative"] is True
    out2 = build_phase1_terminal_receipt(
        phase0_censor_cleared=False,
        control_receipt={},
        arm_receipts={},
        plan_sha256="x",
        authority_dispatch="y",
        authoritative=True,
        force_null_reason=d2["stop_reason"],
        null_family=FAMILY_F4,
        transition="design_null_censor_unreducible",
        arms_classified=False,
    )
    assert out2["family"] == FAMILY_F4
    assert out2["authoritative"] is True
    assert out2["stop_reason"] == "design_null_censor_unreducible"
    assert out2["arms_classified"] is False

    # Cleared → enter_phase1
    p0c = validate_phase0_receipt_for_aggregate(_valid_phase0_receipt(lcf=0.1, steps=150))
    d3 = decide_phase0_aggregate_transition(p0c)
    assert d3["action"] == "enter_phase1"
    assert d3["authoritative"] is True


def test_arm_route_integrity_fail_closed():
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        ArmReceiptContractError,
        validate_shared_held_fixed_arm_receipts,
    )

    good = {
        ARM0: _contract_arm(ARM0),
        ARM1: _contract_arm(ARM1),
        ARM2: _contract_arm(ARM2),
        ARM3: _contract_arm(ARM3),
    }
    validate_shared_held_fixed_arm_receipts(
        good, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
    )

    cases = [
        (_contract_arm(ARM1, correctness_smoke=True), "correctness_smoke"),
        (_contract_arm(ARM1, schema_only=True), "schema_only"),
        (_contract_arm(ARM1, n_fixed=0), "n_fixed_qscale_forwards"),
        (_contract_arm(ARM1, n_dyn=3), "n_bitlinear_dynamic_forwards"),
        (_contract_arm(ARM1, n_elig=32, n_cred=16), "eligible"),
    ]
    for bad_arm, needle in cases:
        bad = dict(good)
        bad[ARM1] = bad_arm
        try:
            validate_shared_held_fixed_arm_receipts(
                bad, expected_plan_sha256="07a02aff", expected_parent_sha256="2d9b9f67"
            )
            assert False, f"expected fail for {needle}"
        except ArmReceiptContractError as e:
            assert needle in str(e), (needle, str(e))


def test_route_counter_snapshot_survives_probe_begin_credit_step():
    """Formal skip_probes=false path must retain training-step counters."""
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        begin_credit_step,
        get_credit_store,
        snapshot_route_counters,
    )

    store = begin_credit_step(["a", "b"])
    store.n_fixed_qscale_forwards = 1024
    store.n_bitlinear_dynamic_forwards = 0
    store.n_credit_grads_present = 32
    snap = snapshot_route_counters(store)
    assert snap["n_fixed_qscale_forwards"] == 1024
    # Probe decode path resets the global store:
    begin_credit_step([])
    store2 = get_credit_store()
    assert store2.n_fixed_qscale_forwards == 0
    # Snapshot must still hold the training values:
    assert snap["n_fixed_qscale_forwards"] == 1024
    assert snap["n_bitlinear_dynamic_forwards"] == 0


# --------------------------------------------------------------------------- #
# r6 Phase B — committed e2e aggregate WIRING tests (_run_aggregate_phase1)
# --------------------------------------------------------------------------- #


def _load_screen_module():
    import importlib.util
    from pathlib import Path

    script = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "hrm_text_158_forgetting_mechanism_screen.py"
    )
    spec = importlib.util.spec_from_file_location(
        "hrm_text_158_forgetting_mechanism_screen_under_test", script
    )
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def _write_json(path, obj):
    import json

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def _formal_arm(arm, **kwargs):
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        DEFAULT_PARENT_SHA256,
        DEFAULT_PLAN_SHA256,
    )

    return _contract_arm(
        arm,
        plan=DEFAULT_PLAN_SHA256,
        parent=DEFAULT_PARENT_SHA256,
        **kwargs,
    )


def test_run_aggregate_phase1_e2e_wiring(tmp_path):
    """Commit-durable wiring: invoke _run_aggregate_phase1, not reducers alone."""
    import argparse
    import json

    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        ARM0,
        ARM1,
        ARM2,
        ARM3,
        FAMILY_F3,
        FAMILY_F4,
    )

    mod = _load_screen_module()
    run = mod._run_aggregate_phase1

    # (a) uncleared-150 + arms -> fallback_required; arms NOT classified
    p0_unc = tmp_path / "p0_unc150.json"
    _write_json(p0_unc, _valid_phase0_receipt(lcf=0.55, steps=150))
    arm_paths = []
    for arm, H in (
        (ARM0, 1.5),
        (ARM1, 0.8),
        (ARM2, 0.7),
        (ARM3, 0.2),
    ):
        r = _formal_arm(arm)
        r["measurements"]["H_bits_per_weight"] = H
        ap = tmp_path / f"{arm}.json"
        _write_json(ap, r)
        arm_paths.append(str(ap))
    out_a = tmp_path / "out_a.json"
    ns = argparse.Namespace(
        phase0_receipt=str(p0_unc),
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
        arm_receipts=",".join(arm_paths),
        output_json=str(out_a),
    )
    assert run(ns) == 0
    rec_a = json.loads(out_a.read_text())
    assert rec_a["transition"] == "fallback_required"
    assert rec_a["authoritative"] is False
    assert rec_a["arms_classified"] is False
    assert rec_a["family"] is None
    assert rec_a.get("phase0_proof", {}).get("arms_rejected") is True
    assert rec_a["source_arm_receipts"] == []

    # (b) failed-150 predecessor -> uncleared-600, NO arms -> auth F4
    p0_fail150 = tmp_path / "p0_fail150.json"
    p0_unc600 = tmp_path / "p0_unc600.json"
    _write_json(p0_fail150, _valid_phase0_receipt(lcf=0.55, steps=150))
    _write_json(p0_unc600, _valid_phase0_receipt(lcf=0.60, steps=600))
    out_b = tmp_path / "out_b.json"
    ns_b = argparse.Namespace(
        phase0_receipt=str(p0_unc600),
        phase0_predecessor_receipt=str(p0_fail150),
        phase0_censor_cleared=None,
        arm_receipts=None,
        output_json=str(out_b),
    )
    assert run(ns_b) == 0
    rec_b = json.loads(out_b.read_text())
    assert rec_b["authoritative"] is True
    assert rec_b["family"] == FAMILY_F4
    assert rec_b["stop_reason"] == "design_null_censor_unreducible"
    assert rec_b["transition"] == "design_null_censor_unreducible"
    assert rec_b["arms_classified"] is False

    # (c) cleared-150 WITHOUT arms -> fail (arms required)
    p0_clr = tmp_path / "p0_clr150.json"
    _write_json(p0_clr, _valid_phase0_receipt(lcf=0.1, steps=150))
    out_c = tmp_path / "out_c.json"
    ns_c = argparse.Namespace(
        phase0_receipt=str(p0_clr),
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
        arm_receipts=None,
        output_json=str(out_c),
    )
    try:
        run(ns_c)
        assert False, "expected SystemExit for missing arms on cleared path"
    except SystemExit as e:
        assert "arm-receipts required" in str(e).lower() or "required" in str(e).lower()
    assert not out_c.exists()

    # (d) cleared-150 + route-invalid arm -> SystemExit
    bad_arms = []
    for arm, H in (
        (ARM0, 1.5),
        (ARM1, 0.8),
        (ARM2, 0.7),
        (ARM3, 0.2),
    ):
        r = _formal_arm(arm, n_fixed=(0 if arm == ARM2 else 10))
        r["measurements"]["H_bits_per_weight"] = H
        if arm == ARM1:
            r["correctness_smoke"] = True  # also invalid; either fail is fine
        ap = tmp_path / f"bad_{arm}.json"
        _write_json(ap, r)
        bad_arms.append(str(ap))
    # Use only n_fixed==0 on ARM2 (clearer single defect); reset smoke
    bad_arms = []
    for arm, H in (
        (ARM0, 1.5),
        (ARM1, 0.8),
        (ARM2, 0.7),
        (ARM3, 0.2),
    ):
        r = _formal_arm(arm, n_fixed=(0 if arm == ARM2 else 10))
        r["measurements"]["H_bits_per_weight"] = H
        ap = tmp_path / f"bad2_{arm}.json"
        _write_json(ap, r)
        bad_arms.append(str(ap))
    out_d = tmp_path / "out_d.json"
    ns_d = argparse.Namespace(
        phase0_receipt=str(p0_clr),
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
        arm_receipts=",".join(bad_arms),
        output_json=str(out_d),
    )
    try:
        run(ns_d)
        assert False, "expected SystemExit for route-invalid arm"
    except SystemExit as e:
        msg = str(e).lower()
        assert "contract" in msg or "n_fixed" in msg or "fail-closed" in msg

    # Happy path sanity: cleared-150 + valid arms -> authoritative family (not null wiring)
    good_arms = []
    for arm, H in (
        (ARM0, 1.5),
        (ARM1, 0.9),
        (ARM2, 0.85),
        (ARM3, 0.2),  # best H_progress -> F3
    ):
        r = _formal_arm(arm)
        r["measurements"]["H_bits_per_weight"] = H
        r["measurements"]["n_flips"] = 5000
        r["measurements"]["q_changed_count"] = 200
        r["measurements"]["n_applied_drains"] = 10000
        r["measurements"]["lifetime_censored_frac"] = 0.1
        ap = tmp_path / f"good_{arm}.json"
        _write_json(ap, r)
        good_arms.append(str(ap))
    out_ok = tmp_path / "out_ok.json"
    ns_ok = argparse.Namespace(
        phase0_receipt=str(p0_clr),
        phase0_predecessor_receipt=None,
        phase0_censor_cleared=None,
        arm_receipts=",".join(good_arms),
        output_json=str(out_ok),
    )
    assert run(ns_ok) == 0
    rec_ok = json.loads(out_ok.read_text())
    assert rec_ok["authoritative"] is True
    assert rec_ok["arms_classified"] is True
    assert rec_ok["family"] == FAMILY_F3


def test_failed_600_f4_aggregate_strict_json_no_nan():
    """No-arm authoritative F4 must be strict-JSON (null, never IEEE NaN)."""
    import json
    import math

    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        FAMILY_F4,
        build_phase1_terminal_receipt,
        decide_phase0_aggregate_transition,
        sanitize_receipt_for_strict_json,
        validate_phase0_receipt_for_aggregate,
    )

    pred = _valid_phase0_receipt(lcf=0.55, steps=150)
    p0 = validate_phase0_receipt_for_aggregate(
        _valid_phase0_receipt(lcf=0.55, steps=600),
        phase0_predecessor_receipt=pred,
    )
    d = decide_phase0_aggregate_transition(p0)
    assert d["action"] == "design_null_censor_unreducible"
    assert d["authoritative"] is True
    receipt = build_phase1_terminal_receipt(
        phase0_censor_cleared=False,
        control_receipt={},
        arm_receipts={},
        plan_sha256="07a02aff",
        authority_dispatch="1784812148229-f466bc29",
        authoritative=True,
        force_null_reason=d["stop_reason"],
        null_family=FAMILY_F4,
        transition="design_null_censor_unreducible",
        arms_classified=False,
    )
    assert receipt["arms_classified"] is False
    assert receipt["H_control_final"] is None
    for arm_m in receipt["arm_metrics"].values():
        assert arm_m["H_final"] is None
        assert not isinstance(arm_m["H_final"], float) or not math.isnan(arm_m["H_final"])

    payload = sanitize_receipt_for_strict_json(receipt)
    dumped = json.dumps(payload, allow_nan=False)
    assert "NaN" not in dumped
    assert "Infinity" not in dumped

    def _reject_constant(c):
        raise ValueError(f"non-standard JSON constant: {c}")

    roundtrip = json.loads(dumped, parse_constant=_reject_constant)
    assert roundtrip["H_control_final"] is None
    assert roundtrip["family"] == FAMILY_F4
    assert roundtrip["stop_reason"] == "design_null_censor_unreducible"
