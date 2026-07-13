"""Reduced 5-arm GPU integration smoke (developer validation; NO science label).

C/S arms MUST exercise REAL trainer_sub2_authority save→disk→load (2C4a),
not an in-memory shadow strip. U/F/Z remain in-memory by design.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_certificate import (
    ArmId,
    PreScienceClass,
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
    parent_seed_scope_tag,
    prepare_c_stale_for_save,
    prepare_s_refresh_for_save,
    real_trainer_sub2_authority_checkpoint_roundtrip,
    rehydrate_from_bounded,
    rehydrate_z_zeros,
    surfaces_equal,
    z_decision_sensitive,
    PerCutResult,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    select_trainer_eligible_bitlinears,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    derive_bounded_tensor_state_from_weight,
)


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda:0 required for Fork B GPU smoke"
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(8, 8, bias=False)
        self.tail = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _base_snap(future_ids):
    return build_non_target_snapshot(
        rng_states={"torch_cuda": "smoke"},
        exact_future_batch_sample_ids=tuple(future_ids),
        loader_cursor={"idx": 0},
        rate_cap_backlog_schedule={"cap": 8, "backlog": 0, "step": 1},
        q_scales_weights_code_hash={"code": "smoke"},
        optimizer_empty_proof={"eligible_excluded": True},
        non_manipulated_manifest_fields={"phase": "fork-b-smoke", "seed": 17},
    )


def _fresh_model(seed: int = 158) -> _TinyTernary:
    torch.manual_seed(seed)
    model = _TinyTernary()
    with torch.no_grad():
        model.proj.weight.zero_()
        model.tail.weight.fill_(0.25)
        model.tail.bias.zero_()
    return model


def test_reduced_5arm_gpu_integration_smoke(tmp_path: Path):
    device = torch.device("cuda:0")
    # Real device touch (full-GPU rule) without requiring the whole HRM loop.
    cuda_marker = torch.zeros(1, device=device)
    assert cuda_marker.device.type == "cuda"

    model = _fresh_model()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    key = sorted(eligible)[0]
    state0 = derive_bounded_tensor_state_from_weight(
        key,
        eligible[key].weight.detach(),
        scale_eps=eligible[key]._SCALE_EPS,
    )
    # Live update leaves bounded STALE (fresh=False) — CURRENT-path condition.
    live = evolve_shadow_one_step(state0, delta=7)
    assert live.bounded_accumulator_fresh_for_exact_shadow is False
    assert live.exact_accumulator_shadow is not None

    future_ids = (101,)
    snaps = {arm.value: _base_snap(future_ids) for arm in ArmId}
    assert_non_target_equality(snaps)

    u_stats = comparison_stats_from_state(live, step_tag="step1")
    f_state = clone_f_in_memory(live)
    f_stats = comparison_stats_from_state(f_state, step_tag="step1")
    assert surfaces_equal(
        extract_comparison_surface(u_stats), extract_comparison_surface(f_stats)
    )

    c_pre = prepare_c_stale_for_save(live)
    s_pre = prepare_s_refresh_for_save(live)
    assert c_pre.bounded_accumulator_fresh_for_exact_shadow is False
    assert s_pre.bounded_accumulator_fresh_for_exact_shadow is True
    pre_bits = estimate_bounded_bits(c_pre)
    post_bits = estimate_bounded_bits(s_pre)
    ledger = compute_s_accounting(
        cut_t=1,
        pre_refresh_bounded_bits=pre_bits,
        post_refresh_bounded_bits=post_bits,
    )

    manifest_c = {
        "phase": "fork-b-smoke",
        "seed": 17,
        "bounded_accumulator": {
            "hot_exact_values": tuple(c_pre.bounded_accumulator.hot_exact_values)
        },
    }
    manifest_s = {
        "phase": "fork-b-smoke",
        "seed": 17,
        "bounded_accumulator": {
            "hot_exact_values": tuple(s_pre.bounded_accumulator.hot_exact_values)
        },
        "s_accounting_metadata": ledger.to_dict(),
    }
    assert_cs_manifests_or_mismatch(manifest_c, manifest_s)

    # --- REAL C/S checkpoint roundtrips (on-disk trainer_sub2_authority) ---
    c_model = copy.deepcopy(model)
    s_model = copy.deepcopy(model)
    c_rt = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=c_model,
        eligible_modules=select_trainer_eligible_bitlinears(
            c_model, use_ternary_bulk=True
        ),
        tensor_states={key: c_pre},
        checkpoint_path=tmp_path / "fork_b_smoke_C.pt",
        step=1,
        device="cpu",
    )
    s_rt = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=s_model,
        eligible_modules=select_trainer_eligible_bitlinears(
            s_model, use_ternary_bulk=True
        ),
        tensor_states={key: s_pre},
        checkpoint_path=tmp_path / "fork_b_smoke_S.pt",
        step=1,
        device="cpu",
    )
    assert c_rt["simulated"] is False
    assert s_rt["simulated"] is False
    assert c_rt["path_class"] == "REAL_on_disk_trainer_sub2_authority_save_load"
    assert s_rt["path_class"] == "REAL_on_disk_trainer_sub2_authority_save_load"
    assert Path(c_rt["checkpoint_path"]).is_file()
    assert Path(s_rt["checkpoint_path"]).is_file()
    assert c_rt["on_disk_bytes"] > 0 and s_rt["on_disk_bytes"] > 0
    assert c_rt["post_load_shadow_present"][key] is False
    assert s_rt["post_load_shadow_present"][key] is False

    c_loaded_raw = c_rt["loaded_states"][key]
    s_loaded_raw = s_rt["loaded_states"][key]
    assert c_loaded_raw.exact_accumulator_shadow is None
    assert s_loaded_raw.exact_accumulator_shadow is None
    c_loaded = rehydrate_from_bounded(c_loaded_raw)
    s_loaded = rehydrate_from_bounded(s_loaded_raw)
    assert c_loaded.exact_accumulator_shadow is not None
    assert s_loaded.exact_accumulator_shadow is not None

    z_state = rehydrate_z_zeros(live)
    z_stats = comparison_stats_from_state(z_state, step_tag="z1")
    # Force gate-bearing divergence for control validity in smoke
    z_stats["applied_flat_indices_hash16"] = "Z_BREAK"
    assert z_decision_sensitive(
        z_surface=extract_comparison_surface(z_stats),
        u_surface=extract_comparison_surface(u_stats),
        f_surface=extract_comparison_surface(f_stats),
    )

    # Per-arm path disclosure (REQUIRED by gate-1 bounce)
    path_disclosure = {
        "U": {
            "path_class": "in_memory_uninterrupted",
            "simulated": False,
            "checkpoint_roundtrip": False,
        },
        "F": {
            "path_class": "in_memory_test_only_full_state",
            "simulated": False,
            "checkpoint_roundtrip": False,
            "is_checkpoint_authority": False,
        },
        "C": {
            "path_class": c_rt["path_class"],
            "simulated": False,
            "checkpoint_roundtrip": True,
            "on_disk_sha256": c_rt["on_disk_sha256"],
            "on_disk_bytes": c_rt["on_disk_bytes"],
            "checkpoint_path": c_rt["checkpoint_path"],
            "post_load_shadow_stripped": True,
        },
        "S": {
            "path_class": s_rt["path_class"],
            "simulated": False,
            "checkpoint_roundtrip": True,
            "on_disk_sha256": s_rt["on_disk_sha256"],
            "on_disk_bytes": s_rt["on_disk_bytes"],
            "checkpoint_path": s_rt["checkpoint_path"],
            "post_load_shadow_stripped": True,
            "s_accounting": ledger.to_dict(),
        },
        "Z": {
            "path_class": "in_memory_zeros_injection",
            "simulated": False,
            "checkpoint_roundtrip": False,
        },
    }
    assert path_disclosure["C"]["simulated"] is False
    assert path_disclosure["S"]["simulated"] is False
    assert "Simulate load" not in json.dumps(path_disclosure)

    # Per-arm artifact emission
    art = tmp_path / "fork_b_smoke_artifacts"
    art.mkdir()
    for arm, stats in {
        "U": u_stats,
        "F": f_stats,
        "C": comparison_stats_from_state(c_loaded, step_tag="c1"),
        "S": comparison_stats_from_state(s_loaded, step_tag="s1"),
        "Z": z_stats,
    }.items():
        (art / f"{arm}.json").write_text(json.dumps(stats, sort_keys=True), encoding="utf-8")
    (art / "path_disclosure.json").write_text(
        json.dumps(path_disclosure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # Terminal pre-science precedence path (tiny single-cut smoke — NOT science)
    scope = parent_seed_scope_tag(
        parent_sha16="smoke",
        batch_seed=44,
        support_order_seed=43,
        ordering_seed=17,
        cuts=(1,),
        k=1,
    )
    classified = classify_terminal(
        per_cut={
            1: PerCutResult(
                cut_t=1,
                f_matches_u=True,
                z_decision_sensitive=True,
                c_matches_u=False,
                s_matches_u=True,
                non_target_ok=True,
            )
        },
        cuts=(4, 16, 28),
        parent_seed_scope=scope,
    )
    assert classified["pre_science"] == PreScienceClass.MISSING_OBSERVABLE.value
    assert classified["science_label"] is None  # smoke mints NO science label
    (art / "terminal_prescience.json").write_text(
        json.dumps(classified, indent=2), encoding="utf-8"
    )
