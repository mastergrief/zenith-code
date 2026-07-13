"""CPU-static tests for Fork B resume-parity certificate facade (plan v2 Step 1)."""
from __future__ import annotations

import pytest
import torch

from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_certificate import (
    ArmId,
    CUTS_DEFAULT,
    DENSE_SHADOW_FIELD_PERSISTENT_BPW,
    NON_TARGET_SNAPSHOT_SCHEMA_FIELDS,
    PerCutResult,
    PreScienceClass,
    SCHEMA_ID,
    Z_BINDING_CUT_T,
    assert_cs_manifests_or_mismatch,
    assert_non_target_equality,
    build_non_target_snapshot,
    classify_terminal,
    clone_f_in_memory,
    comparison_stats_from_state,
    compute_s_accounting,
    estimate_bounded_bits,
    evolve_shadow_one_step,
    extract_comparison_surface,
    manifests_equal_outside_allowlist,
    non_target_schema_field_set,
    parent_seed_scope_tag,
    prepare_c_stale_for_save,
    prepare_s_refresh_for_save,
    rehydrate_from_bounded,
    rehydrate_z_zeros,
    snapshot_not_loadable_as_checkpoint_authority,
    surfaces_equal,
    z_decision_sensitive,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state as _make_state,
)


def _snap(**overrides):
    base = dict(
        rng_states={"torch": "abc"},
        exact_future_batch_sample_ids=(0, 1, 2, 3),
        loader_cursor={"idx": 0},
        rate_cap_backlog_schedule={"cap": 512, "backlog": 0, "step": 4},
        q_scales_weights_code_hash={"q": "q1", "code": "c1"},
        optimizer_empty_proof={"eligible_excluded": True},
        non_manipulated_manifest_fields={"phase": "x", "seed": 17},
    )
    base.update(overrides)
    return build_non_target_snapshot(**base)


def _toy_state():
    q = torch.zeros((4, 4), dtype=torch.int8)
    return _make_state("toy", q, 1.0, hot_exact_indices=())


def test_arm_id_exhaustive():
    assert {a.value for a in ArmId} == {"U", "F", "C", "S", "Z"}


def test_non_target_schema_frozen_keyset():
    assert non_target_schema_field_set() == frozenset(NON_TARGET_SNAPSHOT_SCHEMA_FIELDS)
    assert SCHEMA_ID == "fork_b_non_target_snapshot/v1"
    snap = _snap()
    assert snap.run_local_test_evidence_only is True
    assert snap.is_checkpoint_authority is False
    assert snap.contributes_persistent_bpw is False
    assert snapshot_not_loadable_as_checkpoint_authority(snap)


def test_non_target_hash_stable_and_sensitive():
    a = _snap()
    b = _snap()
    assert a.hash_bundle() == b.hash_bundle()
    c = _snap(rng_states={"torch": "CHANGED"})
    assert a.hash_bundle() != c.hash_bundle()
    d = _snap(exact_future_batch_sample_ids=(9, 9, 9, 9))
    assert a.hash_bundle() != d.hash_bundle()


def test_non_target_mismatch_raises():
    with pytest.raises(ValueError, match="NON_TARGET_STATE_MISMATCH"):
        assert_non_target_equality({"F": _snap(), "C": _snap(rng_states={"torch": "x"})})


def test_cs_manifest_allowlist_and_mismatch_stop():
    c = {"phase": "p", "seed": 1, "bounded_accumulator": {"hot_exact_values": (1,)}}
    s_ok = {"phase": "p", "seed": 1, "bounded_accumulator": {"hot_exact_values": (9,)}}
    ok, mism = manifests_equal_outside_allowlist(c, s_ok)
    assert ok and mism == ()
    assert_cs_manifests_or_mismatch(c, s_ok)
    s_bad = {"phase": "OTHER", "seed": 1, "bounded_accumulator": {"hot_exact_values": (9,)}}
    ok2, mism2 = manifests_equal_outside_allowlist(c, s_bad)
    assert not ok2 and "phase" in mism2[0]
    with pytest.raises(ValueError, match="NON_TARGET_STATE_MISMATCH"):
        assert_cs_manifests_or_mismatch(c, s_bad)


def test_s_accounting_variable_and_fixed_packed():
    var = compute_s_accounting(
        cut_t=16,
        pre_refresh_bounded_bits=100,
        post_refresh_bounded_bits=250,
        schema_metadata_delta_bits=5,
        fixed_size_packed_overwrite=False,
    )
    assert var.delta_bits == 155
    assert var.dense_shadow_field_persistent_bpw == 0
    packed = compute_s_accounting(
        cut_t=16,
        pre_refresh_bounded_bits=1000,
        post_refresh_bounded_bits=1000,
        schema_metadata_delta_bits=7,
        fixed_size_packed_overwrite=True,
    )
    assert packed.delta_bits == 7
    assert packed.fixed_size_packed_slab_delta_bits == 0


def test_arm_procedures_stale_vs_refresh_vs_z():
    state0 = _toy_state()
    live = evolve_shadow_one_step(state0, delta=5)
    assert live.bounded_accumulator_fresh_for_exact_shadow is False
    c_state = prepare_c_stale_for_save(live)
    assert c_state.bounded_accumulator_fresh_for_exact_shadow is False
    s_state = prepare_s_refresh_for_save(live)
    assert s_state.bounded_accumulator_fresh_for_exact_shadow is True
    # After "load" without shadow: rehydrate from bounded
    loaded_c = BoundedDeltaTensorState_no_shadow(c_state)
    rh_c = rehydrate_from_bounded(loaded_c)
    assert rh_c.exact_accumulator_shadow is not None
    z = rehydrate_z_zeros(live)
    assert int(z.exact_accumulator_shadow.abs().sum().item()) == 0
    f = clone_f_in_memory(live)
    assert torch.equal(f.exact_accumulator_shadow, live.exact_accumulator_shadow)


def BoundedDeltaTensorState_no_shadow(state):
    return BoundedDeltaTensorState(
        state_key=state.state_key,
        q_levels=state.q_levels,
        frozen_scale=state.frozen_scale,
        bounded_accumulator=state.bounded_accumulator,
        exact_accumulator_shadow=None,
        bounded_accumulator_fresh_for_exact_shadow=False,
    )


def test_z_decision_sensitivity_requires_gate_bearing_not_just_shadow_hash():
    u = {
        "q_sha256_after": "u",
        "applied_flat_indices_hash16": "a",
        "votes_sha256": "v",
        "global_rate_cap_accepted_indices_sha256": "g",
        "global_rate_cap_deferred_indices_sha256": "d",
        "global_rate_cap_applied_count": 1,
        "flip_count": 1,
        "q_changed_count": 0,
        "applied_selection_score_p50": 1.0,
        "applied_selection_score_p95": 2.0,
        "exact_accumulator_shadow_sha256_after": "SHADOW_U",
    }
    z_hash_only = dict(u)
    z_hash_only["exact_accumulator_shadow_sha256_after"] = "SHADOW_Z"
    # extract_comparison_surface / z_decision_sensitive ignore non-gate fields
    assert z_decision_sensitive(z_surface=z_hash_only, u_surface=u) is False
    z_break = dict(u)
    z_break["applied_flat_indices_hash16"] = "BROKEN"
    assert z_decision_sensitive(z_surface=z_break, u_surface=u, f_surface=u) is True


def test_classifier_z_t16_controls_aggregate():
    scope = parent_seed_scope_tag(
        parent_sha16="9b4e311a22787e7d",
        batch_seed=44,
        support_order_seed=43,
        ordering_seed=17,
    )
    base = {
        t: PerCutResult(
            cut_t=t,
            f_matches_u=True,
            z_decision_sensitive=(t != Z_BINDING_CUT_T),
            c_matches_u=True,
            s_matches_u=True,
            non_target_ok=True,
        )
        for t in CUTS_DEFAULT
    }
    out = classify_terminal(per_cut=base, parent_seed_scope=scope)
    assert out["pre_science"] == PreScienceClass.CONTROL_INVALID.value
    assert out["science_label"] is None

    ok = {
        t: PerCutResult(
            cut_t=t,
            f_matches_u=True,
            z_decision_sensitive=True,
            c_matches_u=True,
            s_matches_u=True,
            non_target_ok=True,
        )
        for t in CUTS_DEFAULT
    }
    out_ok = classify_terminal(per_cut=ok, parent_seed_scope=scope)
    assert out_ok["pre_science"] is None
    assert (
        out_ok["science_label"]
        == "CURRENT_PATH_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS"
    )


def test_classifier_at_all_tested_cuts_and_insufficient():
    scope = parent_seed_scope_tag(
        parent_sha16="9b4e311a22787e7d",
        batch_seed=44,
        support_order_seed=43,
        ordering_seed=17,
    )
    mixed = {
        4: PerCutResult(4, True, True, True, True, None, True),
        16: PerCutResult(16, True, True, False, True, None, True),
        28: PerCutResult(28, True, True, False, True, None, True),
    }
    out = classify_terminal(per_cut=mixed, parent_seed_scope=scope)
    assert out["science_label"] == "REFRESHED_BOUNDED_RECONSTRUCTABLE_AT_ALL_TESTED_CUTS"

    insuff = {
        4: PerCutResult(4, True, True, False, False, None, True),
        16: PerCutResult(16, True, True, False, False, None, True),
        28: PerCutResult(28, True, True, True, True, None, True),
    }
    out2 = classify_terminal(per_cut=insuff, parent_seed_scope=scope)
    assert out2["science_label"] == "TESTED_RECONSTRUCTIONS_INSUFFICIENT_AT_4+16"
    assert out2["dense_shadow_field_persistent_bpw"] == DENSE_SHADOW_FIELD_PERSISTENT_BPW
    assert "dense_int16_shadow_necessity" in out2["explicitly_not"]


def test_non_target_mismatch_pre_science_before_science():
    scope = "AT_TESTED_CUT_PARENT_SEED::x"
    bad = {
        t: PerCutResult(
            cut_t=t,
            f_matches_u=True,
            z_decision_sensitive=True,
            c_matches_u=True,
            s_matches_u=True,
            non_target_ok=False,
        )
        for t in CUTS_DEFAULT
    }
    out = classify_terminal(per_cut=bad, parent_seed_scope=scope)
    assert out["pre_science"] == PreScienceClass.NON_TARGET_STATE_MISMATCH.value
    assert out["science_label"] is None


def test_comparison_surface_helpers_and_estimate_bits():
    state = evolve_shadow_one_step(_toy_state())
    stats = comparison_stats_from_state(state, step_tag="1")
    surface = extract_comparison_surface(stats)
    assert surfaces_equal(surface, extract_comparison_surface(stats))
    assert estimate_bounded_bits(state) >= 64


def test_real_trainer_sub2_authority_checkpoint_roundtrip_strips_shadow(tmp_path):
    """CPU-static: C/S path uses REAL on-disk 2C4a save/load (not in-memory strip)."""
    import copy

    from calm.hrm_text_158.bit_linear import BitLinear
    from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_certificate import (
        prepare_c_stale_for_save,
        prepare_s_refresh_for_save,
        real_trainer_sub2_authority_checkpoint_roundtrip,
        rehydrate_from_bounded,
    )
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        derive_bounded_tensor_state_from_weight,
    )
    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        select_trainer_eligible_bitlinears,
    )

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    key = sorted(eligible)[0]
    state0 = derive_bounded_tensor_state_from_weight(
        key,
        eligible[key].weight.detach(),
        scale_eps=eligible[key]._SCALE_EPS,
    )
    assert state0.exact_accumulator_shadow is not None
    live = evolve_shadow_one_step(state0, delta=5)
    c_pre = prepare_c_stale_for_save(live)
    s_pre = prepare_s_refresh_for_save(live)

    c_model = copy.deepcopy(model)
    s_model = copy.deepcopy(model)
    c_rt = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=c_model,
        eligible_modules=select_trainer_eligible_bitlinears(
            c_model, use_ternary_bulk=True
        ),
        tensor_states={key: c_pre},
        checkpoint_path=tmp_path / "c_real.pt",
    )
    s_rt = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=s_model,
        eligible_modules=select_trainer_eligible_bitlinears(
            s_model, use_ternary_bulk=True
        ),
        tensor_states={key: s_pre},
        checkpoint_path=tmp_path / "s_real.pt",
    )
    assert c_rt["simulated"] is False and s_rt["simulated"] is False
    assert c_rt["path_class"].startswith("REAL_on_disk")
    assert c_rt["post_load_shadow_present"][key] is False
    assert s_rt["post_load_shadow_present"][key] is False
    c_hat = rehydrate_from_bounded(c_rt["loaded_states"][key])
    s_hat = rehydrate_from_bounded(s_rt["loaded_states"][key])
    assert c_hat.exact_accumulator_shadow is not None
    assert s_hat.exact_accumulator_shadow is not None
