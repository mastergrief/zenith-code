"""CPU fixtures for R7 cap/defer pressure instrumentation and classifier (frozen v4)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_accumulator import (
    encode_budget_capped_hybrid_reference,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
    make_bounded_tensor_state,
)
from calm.hrm_text_158.native_full_stack.r7_cap_defer_pressure_instrumentation import (
    HIGH_PRESSURE_ABS,
    R7_STEP_CHUNK_SCHEMA_VERSION,
    build_step_chunk,
    pressure_mass_from_tensor_states,
    validate_accounting_invariant,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateState
from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
    BRANCH_ARTIFACT_INSUFFICIENT,
    BRANCH_CAP_DEFER,
    BRANCH_HARNESS_FAIL,
    BRANCH_MIXED,
    BRANCH_NO_PRESSURE_GROWTH,
    BRANCH_SCHEMA_FAIL,
    BRANCH_VOTE_AMPLITUDE,
    MIN_MEASURED_STEPS,
    build_classifier_from_chunks,
    select_branch,
)


def _summary(
    *,
    step: int = 1,
    candidate: int = 100,
    accepted: int = 40,
    deferred: int = 60,
    age: int = 0,
    backlog_size: int = 0,
    accepted_from_prior: int = 0,
) -> dict[str, object]:
    accepted_fresh = accepted - accepted_from_prior
    return {
        "global_rate_cap_enabled": True,
        "global_pre_cap_would_apply_count": candidate,
        "global_rate_cap_accepted_count": accepted,
        "global_rate_cap_deferred_count": deferred,
        "global_rate_cap_cap": 40,
        "global_rate_cap_saturated": candidate > 40,
        "q_changed_count": accepted,
        "deferred_backlog_size": backlog_size,
        "deferred_backlog_max_age_steps": age,
        "deferred_backlog_max_defer_count": 1 if backlog_size else 0,
        "accepted_from_prior_deferred_count": accepted_from_prior,
        "accepted_fresh_count": accepted_fresh,
    }


def _chunk(
    step: int,
    *,
    pressure_mass: int,
    pressure_mass_delta: int | None = None,
    summary: dict[str, object] | None = None,
) -> dict[str, object]:
    return build_step_chunk(
        step=step,
        global_summary=summary or _summary(step=step),
        pressure_mass=pressure_mass,
        pressure_mass_delta=pressure_mass_delta,
    )


def _chunks_for_branch(
    *,
    pressure_start: int,
    pressure_end: int,
    age: int,
    deferred_fraction: float,
    steps: int = MIN_MEASURED_STEPS,
) -> list[dict[str, object]]:
    chunks: list[dict[str, object]] = []
    for step in range(1, steps + 1):
        t = (step - 1) / max(steps - 1, 1)
        pressure = int(round(pressure_start + (pressure_end - pressure_start) * t))
        candidate = 100
        deferred = int(round(candidate * deferred_fraction))
        accepted = candidate - deferred
        chunks.append(
            _chunk(
                step,
                pressure_mass=pressure,
                pressure_mass_delta=None if step == 1 else pressure - chunks[-1]["pressure_mass"],
                summary=_summary(
                    step=step,
                    candidate=candidate,
                    accepted=accepted,
                    deferred=deferred,
                    age=age if step > 1 else 0,
                    backlog_size=deferred,
                ),
            )
        )
    return chunks


def test_accounting_invariant_accepts_source_partition() -> None:
    failures = validate_accounting_invariant(_summary(accepted_from_prior=5))
    assert failures == []


def test_accounting_invariant_rejects_partition_violation() -> None:
    summary = _summary()
    summary["global_rate_cap_deferred_count"] = 0
    failures = validate_accounting_invariant(summary)
    assert "candidate_partition_violation" in failures


def test_cap_defer_branch_only() -> None:
    chunks = _chunks_for_branch(
        pressure_start=100,
        pressure_end=250,
        age=2,
        deferred_fraction=0.20,
    )
    receipt = build_classifier_from_chunks(chunks=chunks)
    assert receipt["branch_selection"]["branch"] == BRANCH_CAP_DEFER


def test_vote_amplitude_branch_only() -> None:
    chunks = _chunks_for_branch(
        pressure_start=100,
        pressure_end=250,
        age=0,
        deferred_fraction=0.0,
    )
    for chunk in chunks:
        chunk["q_apply_count"] = 1
    receipt = build_classifier_from_chunks(chunks=chunks)
    assert receipt["branch_selection"]["branch"] == BRANCH_VOTE_AMPLITUDE


def test_branches_mutually_exclusive_cap_vs_vote() -> None:
    cap = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics={
            "steps_observed": MIN_MEASURED_STEPS,
            "pressure_growth_ratio": 2.0,
            "run_max_deferred_backlog_max_age_steps": 2,
            "run_mean_deferred_saturation": 0.20,
            "q_transition_mass_ratio": 0.01,
        },
    )
    vote = select_branch(
        harness_fail=False,
        schema_fail=False,
        metrics={
            "steps_observed": MIN_MEASURED_STEPS,
            "pressure_growth_ratio": 2.0,
            "run_max_deferred_backlog_max_age_steps": 0,
            "run_mean_deferred_saturation": 0.0,
            "q_transition_mass_ratio": 0.01,
        },
    )
    assert cap["branch"] == BRANCH_CAP_DEFER
    assert vote["branch"] == BRANCH_VOTE_AMPLITUDE
    assert cap["branch"] != vote["branch"]


def test_mixed_branch_when_pressure_grows_but_neither_predicate() -> None:
    receipt = build_classifier_from_chunks(
        chunks=_chunks_for_branch(
            pressure_start=100,
            pressure_end=200,
            age=0,
            deferred_fraction=0.05,
        )
    )
    assert receipt["branch_selection"]["branch"] == BRANCH_MIXED


def test_no_pressure_growth_branch() -> None:
    receipt = build_classifier_from_chunks(
        chunks=_chunks_for_branch(
            pressure_start=100,
            pressure_end=110,
            age=0,
            deferred_fraction=0.0,
        )
    )
    assert receipt["branch_selection"]["branch"] == BRANCH_NO_PRESSURE_GROWTH


def test_schema_fail_on_invariant_violation() -> None:
    chunk = _chunk(1, pressure_mass=100, summary=_summary())
    chunk["accounting_invariant_failures"] = ["candidate_partition_violation"]
    receipt = build_classifier_from_chunks(chunks=[chunk])
    assert receipt["branch_selection"]["branch"] == BRANCH_SCHEMA_FAIL


def test_artifact_insufficient_lt_eight_steps() -> None:
    receipt = build_classifier_from_chunks(
        chunks=_chunks_for_branch(
            pressure_start=100,
            pressure_end=200,
            age=2,
            deferred_fraction=0.20,
            steps=4,
        )
    )
    assert receipt["branch_selection"]["branch"] == BRANCH_ARTIFACT_INSUFFICIENT


def test_harness_fail_on_empty_sidecar() -> None:
    receipt = build_classifier_from_chunks(chunks=[], harness_fail=True)
    assert receipt["branch_selection"]["branch"] == BRANCH_HARNESS_FAIL


def test_default_off_chunk_compact_only() -> None:
    chunk = _chunk(1, pressure_mass=10)
    assert chunk["schema_version"] == R7_STEP_CHUNK_SCHEMA_VERSION
    assert chunk["raw_arrays_included"] is False
    assert "accumulator_lanes" not in chunk


def test_r7_backlog_carry_default_off_omits_backlog_arg() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        resolve_r7_deferred_backlog_vote_step_kwargs,
    )

    assert resolve_r7_deferred_backlog_vote_step_kwargs(
        r7_deferred_backlog_carry_enabled=False,
        carry_backlog=None,
    ) == {}


def test_r7_backlog_carry_off_rejects_non_none_outside_contract() -> None:
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        resolve_r7_deferred_backlog_vote_step_kwargs,
    )

    with pytest.raises(ValueError, match="carry_backlog is non-None"):
        resolve_r7_deferred_backlog_vote_step_kwargs(
            r7_deferred_backlog_carry_enabled=False,
            carry_backlog={"mod": {0: {"first_step": 1, "last_deferred_step": 1, "defer_count": 1}}},
        )


def test_pressure_mass_from_bounded_delta_tensor_state_production_type() -> None:
    q = torch.zeros((2, 3), dtype=torch.int8)
    acc = torch.tensor([[5, 10, 15], [20, -3, 12]], dtype=torch.int16)
    state = make_bounded_tensor_state("layer0", q, 1.0, acc)
    expected = int(torch.sum(acc.abs() >= HIGH_PRESSURE_ABS).item())
    assert expected == 4
    assert pressure_mass_from_tensor_states({"layer0": state}) == expected


def test_pressure_mass_from_vote_update_state_compat() -> None:
    q = torch.zeros(4, dtype=torch.int8)
    acc = torch.tensor([9, 10, -10, 3], dtype=torch.int16)
    state = VoteUpdateState(q_levels=q, accumulators=acc)
    assert pressure_mass_from_tensor_states({"x": state}) == 2


def test_pressure_mass_bounded_without_exact_shadow_decodes() -> None:
    q = torch.zeros((2,), dtype=torch.int8)
    acc = torch.tensor([5, 15], dtype=torch.int16)
    bounded = encode_budget_capped_hybrid_reference(
        VoteUpdateState(q_levels=q, accumulators=acc),
        hot_exact_indices=(0, 1),
    )
    state = BoundedDeltaTensorState(
        state_key="no_shadow",
        q_levels=q,
        frozen_scale=torch.tensor(1.0, dtype=torch.float32),
        bounded_accumulator=bounded,
        exact_accumulator_shadow=None,
        bounded_accumulator_fresh_for_exact_shadow=False,
    )
    assert pressure_mass_from_tensor_states({"no_shadow": state}) == 1


def test_classifier_receipt_json_roundtrip(tmp_path: Path) -> None:
    chunks = _chunks_for_branch(
        pressure_start=100,
        pressure_end=250,
        age=1,
        deferred_fraction=0.15,
    )
    sidecar = tmp_path / "r7_cap_defer_pressure_sidecar.jsonl"
    with sidecar.open("w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, sort_keys=True) + "\n")
    from calm.hrm_text_158.native_full_stack.r7_mechanism_classifier_probe import (
        build_classifier_probe_receipt,
    )

    receipt = build_classifier_probe_receipt(run_root=tmp_path, head_sha256="abc123")
    assert receipt["branch_selection"]["branch"] in {
        BRANCH_CAP_DEFER,
        BRANCH_MIXED,
        BRANCH_VOTE_AMPLITUDE,
    }
