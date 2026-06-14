from __future__ import annotations

from unittest import mock

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    build_authoritative_checkpoint_payload,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    tensor_sha256,
    S1_PROJECTION_LAW,
    S1_RANK_BUCKET_VOTE_LAW,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from dataclasses import asdict
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack import bounded_delta_learner as learner_module


def _updater_config() -> dict:
    return {
        "rank_vote_spec": default_dry_run_rank_vote_spec().to_live_dict(),
        "vote_update_spec": asdict(
            VoteUpdateSpec(
                threshold_abs=1,
                accumulator_clip_min=-127,
                accumulator_clip_max=127,
                max_abs_per_tensor=32,
            )
        ),
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }



def test_build_authoritative_checkpoint_payload_noop_without_callback() -> None:
    state = make_bounded_tensor_state(
        "toy.proj",
        torch.zeros(4, dtype=torch.int8),
        0.5,
        torch.zeros(4, dtype=torch.int16),
    )
    baseline = build_authoritative_checkpoint_payload(
        {"toy.proj": state},
        step=1,
        updater_config=_updater_config(),
        dry_run=True,
        checkpoint_written=False,
    )
    with_callback = build_authoritative_checkpoint_payload(
        {"toy.proj": state},
        step=1,
        updater_config=_updater_config(),
        dry_run=True,
        checkpoint_written=False,
        on_tensor_export=lambda *_args: None,
    )
    assert baseline == with_callback


def test_export_bench_touches_sites_b_c_d_per_tensor_under_dry_run() -> None:
    tensor_count = 4
    states = {}
    for idx in range(tensor_count):
        numel = 256
        q = torch.zeros(numel, dtype=torch.int8)
        acc = torch.full((numel,), 7, dtype=torch.int16)
        acc[idx::tensor_count] = torch.arange(len(acc[idx::tensor_count]), dtype=torch.int16) - 2
        key = f"stress.{idx}"
        states[key] = make_bounded_tensor_state(
            key,
            q,
            0.5,
            acc,
            hot_exact_indices=(idx % numel, (idx + 1) % numel),
            cold_default_value=7,
        )

    sha_calls = 0
    decode_calls = 0
    canonical_calls = 0
    to_schema_calls: list[str] = []
    original_sha = tensor_sha256
    original_decode = decode_bounded_accumulator_to_i16
    original_canonical = learner_module._canonical_json
    original_to_schema = learner_module.BoundedDeltaTensorState.to_schema_dict

    def counting_sha(tensor: torch.Tensor) -> str:
        nonlocal sha_calls
        sha_calls += 1
        return original_sha(tensor)

    def counting_decode(*args, **kwargs):
        nonlocal decode_calls
        decode_calls += 1
        return original_decode(*args, **kwargs)

    def counting_canonical(value):
        nonlocal canonical_calls
        canonical_calls += 1
        return original_canonical(value)

    def counting_to_schema(self, **kwargs):
        to_schema_calls.append(self.state_key)
        return original_to_schema(self, **kwargs)

    with (
        mock.patch.object(learner_module, "tensor_sha256", side_effect=counting_sha),
        mock.patch.object(learner_module, "decode_bounded_accumulator_to_i16", side_effect=counting_decode),
        mock.patch.object(learner_module, "_canonical_json", side_effect=counting_canonical),
        mock.patch.object(learner_module.BoundedDeltaTensorState, "to_schema_dict", counting_to_schema),
    ):
        payload = build_authoritative_checkpoint_payload(
            states,
            step=1,
            updater_config=_updater_config(),
            dry_run=True,
            checkpoint_written=False,
        )

    assert len(to_schema_calls) == tensor_count
    assert len(set(to_schema_calls)) == tensor_count
    assert decode_calls == tensor_count
    assert sha_calls >= tensor_count
    assert canonical_calls >= 1  # summaries (D); updater_config_sha256 may add a second call
    for key in states:
        summary = payload["tensor_summaries"][key]
        assert summary["bounded_decode_parity_checked"] is True
        assert summary["q_sha256"]
    assert payload["authoritative_state_sha256"]


def test_export_bench_dry_run_short_circuit_would_fail_closed() -> None:
    state = make_bounded_tensor_state(
        "only",
        torch.zeros(8, dtype=torch.int8),
        0.5,
        torch.zeros(8, dtype=torch.int16),
    )

    def blocked_to_schema(self, **kwargs):
        raise AssertionError("dry_run must not skip to_schema_dict export")

    with mock.patch.object(
        learner_module.BoundedDeltaTensorState,
        "to_schema_dict",
        blocked_to_schema,
    ):
        try:
            build_authoritative_checkpoint_payload(
                {"only": state},
                step=1,
                updater_config=_updater_config(),
                dry_run=True,
                checkpoint_written=False,
            )
        except AssertionError as exc:
            assert "dry_run must not skip" in str(exc)
        else:
            raise AssertionError("expected dry_run export path to invoke to_schema_dict")


def test_probe_oracle_and_audit_checkpoint_paths_noop_without_callback() -> None:
    states = {
        f"toy.{idx}": make_bounded_tensor_state(
            f"toy.{idx}",
            torch.zeros(8, dtype=torch.int8),
            0.5,
            torch.zeros(8, dtype=torch.int16),
        )
        for idx in range(2)
    }
    updater_oracle = {
        "oracle_screen_mode": "feasibility",
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }
    updater_audit = {
        "scope": "b2_full_audit_summary",
        "projection_law": S1_PROJECTION_LAW,
        "vote_law": S1_RANK_BUCKET_VOTE_LAW,
    }
    for updater_config in (updater_oracle, updater_audit):
        baseline = build_authoritative_checkpoint_payload(
            states,
            step=3,
            updater_config=updater_config,
            oracle_receipt=None,
            dry_run=True,
            checkpoint_written=False,
        )
        with_callback = build_authoritative_checkpoint_payload(
            states,
            step=3,
            updater_config=updater_config,
            oracle_receipt=None,
            dry_run=True,
            checkpoint_written=False,
            on_tensor_export=lambda *_args: None,
        )
        assert baseline == with_callback


def test_trainer_sub2_authority_checkpoint_path_noop_without_callback() -> None:
    states = {
        f"mod.{idx}": make_bounded_tensor_state(
            f"mod.{idx}",
            torch.zeros(8, dtype=torch.int8),
            0.25,
            torch.zeros(8, dtype=torch.int16),
        )
        for idx in range(2)
    }
    updater_config = {
        "scope": "2C1_construction_counting_only",
        "eligible_scope": "all-bitlinear",
        "learner_update_called": False,
        "optimizer_step_called": False,
    }
    baseline = build_authoritative_checkpoint_payload(
        states,
        step=0,
        updater_config=updater_config,
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
    )
    with_callback = build_authoritative_checkpoint_payload(
        states,
        step=0,
        updater_config=updater_config,
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
        on_tensor_export=lambda *_args: None,
    )
    assert baseline == with_callback


def test_cold_exception_stress_fixture_has_many_exceptions() -> None:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    states, cold_exception_count = _build_synthetic_cold_exception_stress_states()
    assert cold_exception_count > 1000
    assert "cold_exception_stress" in states
