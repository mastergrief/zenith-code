"""Characterization fixtures frozen BEFORE Fork B god-file extraction.

If post-extraction outputs differ from these fixtures → FAIL (never re-baseline).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    derive_bounded_tensor_state_from_weight,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_certificate import (
    CUTS_DEFAULT,
    PerCutResult,
    PreScienceClass,
    SCHEMA_ID,
    build_non_target_snapshot,
    classify_terminal,
    compute_s_accounting,
    evolve_shadow_one_step,
    non_target_schema_field_set,
    parent_seed_scope_tag,
    prepare_c_stale_for_save,
    prepare_s_refresh_for_save,
    real_trainer_sub2_authority_checkpoint_roundtrip,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    select_trainer_eligible_bitlinears,
)

FIX = Path(__file__).resolve().parent / "fixtures" / "fork_b_resume_parity_characterization_v0"


def _load(name: str) -> dict:
    return json.loads((FIX / name).read_text(encoding="utf-8"))


def test_fixture_index_present_and_complete():
    index = _load("FIXTURE_INDEX.json")
    assert set(index["fixtures"]) == {
        "classifier_outputs_fixture.json",
        "non_target_snapshot_fixture.json",
        "real_path_disclosure_fixture.json",
        "s_accounting_fixture.json",
    }


def test_non_target_snapshot_matches_frozen_fixture():
    frozen = _load("non_target_snapshot_fixture.json")
    snap = build_non_target_snapshot(
        rng_states={"torch": "abc"},
        exact_future_batch_sample_ids=(0, 1, 2, 3),
        loader_cursor={"idx": 0},
        rate_cap_backlog_schedule={"cap": 512, "backlog": 0, "step": 4},
        q_scales_weights_code_hash={"q": "q1", "code": "c1"},
        optimizer_empty_proof={"eligible_excluded": True},
        non_manipulated_manifest_fields={"phase": "x", "seed": 17},
    )
    assert SCHEMA_ID == frozen["schema_id"]
    assert sorted(non_target_schema_field_set()) == frozen["field_set"]
    # JSON fixture stores tuples as lists; compare via canonical hash + JSON dump.
    assert snap.hash_bundle() == frozen["snapshot_sha256"]
    live_json = json.loads(json.dumps(snap.to_dict(), sort_keys=True, default=str))
    assert live_json == frozen["snapshot"]


def test_s_accounting_matches_frozen_fixture():
    frozen = _load("s_accounting_fixture.json")
    var = compute_s_accounting(
        cut_t=16,
        pre_refresh_bounded_bits=100,
        post_refresh_bounded_bits=250,
        schema_metadata_delta_bits=5,
        fixed_size_packed_overwrite=False,
    ).to_dict()
    packed = compute_s_accounting(
        cut_t=16,
        pre_refresh_bounded_bits=1000,
        post_refresh_bounded_bits=1000,
        schema_metadata_delta_bits=7,
        fixed_size_packed_overwrite=True,
    ).to_dict()
    assert var == frozen["variable"]
    assert packed == frozen["fixed_packed"]


def test_classifier_outputs_match_frozen_fixture():
    frozen = _load("classifier_outputs_fixture.json")
    scope = parent_seed_scope_tag(
        parent_sha16="9b4e311a22787e7d",
        batch_seed=44,
        support_order_seed=43,
        ordering_seed=17,
    )
    assert scope == frozen["parent_seed_scope"]
    cases = {
        "missing_cuts": classify_terminal(
            per_cut={4: PerCutResult(4, True, True, True, True, None, True)},
            parent_seed_scope=scope,
        ),
        "non_target": classify_terminal(
            per_cut={
                t: PerCutResult(t, True, True, True, True, None, False)
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
        "control_invalid_f": classify_terminal(
            per_cut={
                t: PerCutResult(t, False, True, True, True, None, True)
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
        "control_invalid_z": classify_terminal(
            per_cut={
                t: PerCutResult(
                    t, True, True if t != 16 else False, True, True, None, True
                )
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
        "current_all": classify_terminal(
            per_cut={
                t: PerCutResult(t, True, True, True, True, None, True)
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
        "refreshed_all": classify_terminal(
            per_cut={
                t: PerCutResult(t, True, True, False, True, None, True)
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
        "insufficient": classify_terminal(
            per_cut={
                4: PerCutResult(4, True, True, False, False, None, True),
                16: PerCutResult(16, True, True, False, False, None, True),
                28: PerCutResult(28, True, True, True, True, None, True),
            },
            parent_seed_scope=scope,
        ),
        "infra": classify_terminal(
            per_cut={
                t: PerCutResult(
                    t,
                    True,
                    True,
                    True,
                    True,
                    PreScienceClass.INFRA_FAILURE.value,
                    True,
                )
                for t in CUTS_DEFAULT
            },
            parent_seed_scope=scope,
        ),
    }
    assert cases == frozen["cases"]


def test_real_path_disclosure_matches_frozen_fixture(tmp_path: Path):
    frozen = _load("real_path_disclosure_fixture.json")

    class _Tiny(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = BitLinear(8, 8, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    torch.manual_seed(158)
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
        checkpoint_path=tmp_path / "c.pt",
    )
    s_rt = real_trainer_sub2_authority_checkpoint_roundtrip(
        model=s_model,
        eligible_modules=select_trainer_eligible_bitlinears(
            s_model, use_ternary_bulk=True
        ),
        tensor_states={key: s_pre},
        checkpoint_path=tmp_path / "s.pt",
    )

    def disclosure_core(rt: dict, arm: str) -> dict:
        return {
            "arm": arm,
            "path_class": rt["path_class"],
            "simulated": rt["simulated"],
            "dense_int16_persistent_accumulator_saved": rt[
                "dense_int16_persistent_accumulator_saved"
            ],
            "post_load_shadow_present_all_false": (
                not any(rt["post_load_shadow_present"].values())
            ),
            "required_keys": sorted(
                k for k in rt.keys() if k not in {"loaded_states", "loaded_blob"}
            ),
        }

    got = {
        "C": disclosure_core(c_rt, "C"),
        "S": disclosure_core(s_rt, "S"),
        "required_path_class": "REAL_on_disk_trainer_sub2_authority_save_load",
    }
    assert got == frozen
