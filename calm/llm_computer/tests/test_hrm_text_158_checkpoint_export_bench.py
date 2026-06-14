from __future__ import annotations

import hashlib
from unittest import mock

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    _canonical_json,
    _cold_exception_indices_for_exact_preservation,
    build_authoritative_checkpoint_payload,
    default_dry_run_rank_vote_spec,
    make_bounded_tensor_state,
    tensor_sha256,
    validate_authoritative_resume_payload,
    S1_PROJECTION_LAW,
    S1_RANK_BUCKET_VOTE_LAW,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from dataclasses import asdict
from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    BoundedDeltaAccumulatorState,
    bounded_accumulator_decoded_sha256,
    decode_bounded_accumulator_to_i16,
)
from calm.hrm_text_158.native_full_stack import bounded_delta_learner as learner_module


def _reference_tensor_sha256(tensor: torch.Tensor) -> str:
    cpu = tensor.detach().cpu().contiguous()
    h = hashlib.sha256()
    h.update(str(cpu.dtype).encode("utf-8"))
    h.update(str(tuple(cpu.shape)).encode("utf-8"))
    h.update(cpu.numpy().tobytes())
    return h.hexdigest()


def _reference_cold_exception_indices(
    acc: torch.Tensor,
    *,
    hot_exact_indices: tuple[int, ...],
    cold_default_value: int,
) -> tuple[int, ...]:
    flat = acc.detach().cpu().flatten().to(torch.int16)
    hot = {int(idx) for idx in hot_exact_indices}
    return tuple(
        int(idx)
        for idx, value in enumerate(flat.tolist())
        if idx not in hot and int(value) != int(cold_default_value)
    )


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


def test_export_bench_touches_sites_b_c_per_tensor_under_dry_run() -> None:
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
    sparse_decode_sha_calls = 0
    canonical_calls = 0
    to_schema_calls: list[str] = []
    original_sha = tensor_sha256
    original_sparse_decode_sha = bounded_accumulator_decoded_sha256
    original_canonical = learner_module._canonical_json
    original_to_schema = learner_module.BoundedDeltaTensorState.to_schema_dict

    def counting_sha(tensor: torch.Tensor) -> str:
        nonlocal sha_calls
        sha_calls += 1
        return original_sha(tensor)

    def counting_sparse_decode_sha(*args, **kwargs):
        nonlocal sparse_decode_sha_calls
        sparse_decode_sha_calls += 1
        return original_sparse_decode_sha(*args, **kwargs)

    def counting_canonical(value):
        nonlocal canonical_calls
        canonical_calls += 1
        return original_canonical(value)

    def counting_to_schema(self, **kwargs):
        to_schema_calls.append(self.state_key)
        return original_to_schema(self, **kwargs)

    with (
        mock.patch.object(learner_module, "tensor_sha256", side_effect=counting_sha),
        mock.patch.object(
            learner_module,
            "bounded_accumulator_decoded_sha256",
            side_effect=counting_sparse_decode_sha,
        ),
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
    assert sparse_decode_sha_calls == tensor_count
    assert sha_calls >= tensor_count
    assert canonical_calls >= 1  # summaries (D deferred); updater_config_sha256 may add a second call
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


def test_phase1_site_b_tensor_sha256_matches_reference() -> None:
    tensors = [
        torch.zeros(4, dtype=torch.int8),
        torch.arange(2048, dtype=torch.int16),
        torch.randn(32, 64),
    ]
    for tensor in tensors:
        assert tensor_sha256(tensor) == _reference_tensor_sha256(tensor)


def test_phase1_site_a_cold_exception_indices_matches_reference() -> None:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    states, cold_exception_count = _build_synthetic_cold_exception_stress_states()
    assert cold_exception_count > 1000
    state = states["cold_exception_stress"]
    acc = state.exact_accumulator_shadow
    hot = tuple(range(0, min(128, acc.numel())))
    got = _cold_exception_indices_for_exact_preservation(
        acc,
        hot_exact_indices=hot,
        cold_default_value=7,
    )
    ref = _reference_cold_exception_indices(
        acc,
        hot_exact_indices=hot,
        cold_default_value=7,
    )
    assert got == ref
    assert len(got) > 1000


def test_phase1_site_c_decoded_sha256_matches_dense_decode() -> None:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    states, _ = _build_synthetic_cold_exception_stress_states()
    acc = states["cold_exception_stress"].bounded_accumulator
    dense_sha = tensor_sha256(decode_bounded_accumulator_to_i16(acc))
    sparse_sha = bounded_accumulator_decoded_sha256(acc)
    assert sparse_sha == dense_sha


def _minimal_bounded_state(**overrides: object) -> BoundedDeltaAccumulatorState:
    base = {
        "logical_shape": (8,),
        "cold_default_value": 0,
        "hot_exact_indices": (0,),
        "hot_exact_values": (1,),
        "cold_exception_indices": (),
        "cold_exception_values": (),
    }
    base.update(overrides)
    return BoundedDeltaAccumulatorState(**base)


def test_phase1_site_c_fail_closed_matches_dense_decoder() -> None:
    cases = [
        (
            {"raw_arrays_included": True},
            "bounded-delta compact state must not be marked raw-array-inclusive",
        ),
        (
            {"cold_exception_indices": (1, 2), "cold_exception_values": (3,)},
            "cold exceptions index/value count mismatch",
        ),
        (
            {"hot_exact_indices": (0, 1), "hot_exact_values": (2,)},
            "hot exact rows index/value count mismatch",
        ),
        (
            {"cold_exception_indices": (99,), "cold_exception_values": (3,)},
            "cold exceptions contains out-of-range index",
        ),
        (
            {"hot_exact_indices": (99,), "hot_exact_values": (3,)},
            "hot exact rows contains out-of-range index",
        ),
    ]
    for state_kwargs, message in cases:
        state = _minimal_bounded_state(**state_kwargs)
        with pytest.raises(ValueError, match=message):
            decode_bounded_accumulator_to_i16(state)
        with pytest.raises(ValueError, match=message):
            bounded_accumulator_decoded_sha256(state)


def _cold_exception_stress_payload() -> dict:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    states, _ = _build_synthetic_cold_exception_stress_states()
    return build_authoritative_checkpoint_payload(
        states,
        step=10,
        updater_config=_updater_config(),
        dry_run=True,
        checkpoint_written=False,
    )


def test_phase1_strict_payload_equivalence_cold_exception_stress() -> None:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    site_a_calls = 0
    site_c_calls = 0
    original_site_a = _cold_exception_indices_for_exact_preservation
    original_site_c = bounded_accumulator_decoded_sha256

    def counting_site_a(*args, **kwargs):
        nonlocal site_a_calls
        site_a_calls += 1
        return original_site_a(*args, **kwargs)

    def counting_site_c(*args, **kwargs):
        nonlocal site_c_calls
        site_c_calls += 1
        return original_site_c(*args, **kwargs)

    with (
        mock.patch.object(learner_module, "_cold_exception_indices_for_exact_preservation", counting_site_a),
        mock.patch.object(learner_module, "bounded_accumulator_decoded_sha256", counting_site_c),
    ):
        states, cold_exception_count = _build_synthetic_cold_exception_stress_states()
        assert cold_exception_count > 1000
        payload = build_authoritative_checkpoint_payload(
            states,
            step=10,
            updater_config=_updater_config(),
            dry_run=True,
            checkpoint_written=False,
        )

    assert site_a_calls >= 1
    assert site_c_calls >= 1
    validate_authoritative_resume_payload(payload)
    canonical = _canonical_json(payload)
    payload_repeat = _cold_exception_stress_payload()
    assert payload == payload_repeat
    assert payload["authoritative_state_sha256"] == payload_repeat["authoritative_state_sha256"]
    assert _canonical_json(payload_repeat) == canonical
    for key, summary in payload["tensor_summaries"].items():
        repeat_summary = payload_repeat["tensor_summaries"][key]
        assert summary["q_sha256"] == repeat_summary["q_sha256"]
        assert summary["bounded_accumulator_decoded_sha256"] == repeat_summary[
            "bounded_accumulator_decoded_sha256"
        ]
        assert summary["exact_accumulator_shadow_sha256"] == repeat_summary[
            "exact_accumulator_shadow_sha256"
        ]


def test_phase1_trainer_sub2_authority_path_cold_exception_equivalence() -> None:
    from scripts.hrm_text_158_checkpoint_export_bench import (
        _build_synthetic_cold_exception_stress_states,
    )

    states, cold_exception_count = _build_synthetic_cold_exception_stress_states()
    assert cold_exception_count > 1000
    updater_config = {
        "scope": "2C1_construction_counting_only",
        "eligible_scope": "all-bitlinear",
        "learner_update_called": False,
        "optimizer_step_called": False,
    }
    payload = build_authoritative_checkpoint_payload(
        states,
        step=0,
        updater_config=updater_config,
        oracle_receipt=None,
        dry_run=True,
        checkpoint_written=False,
    )
    validate_authoritative_resume_payload(payload)
    assert payload["tensor_summaries"]["cold_exception_stress"]["bounded_decode_parity_checked"] is True
