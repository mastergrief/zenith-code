"""Multi-step CPU witness for R7 deferred backlog carry (frozen v4 semantics)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    GlobalRateCapSpec,
    GlobalRateCapTensorInput,
    apply_global_rate_cap_reference,
)
from calm.hrm_text_158.native_full_stack.vote_update import (
    VoteUpdateInputs,
    VoteUpdateSpec,
    VoteUpdateState,
    plan_integer_vote_update_reference,
)


def _spec() -> VoteUpdateSpec:
    return VoteUpdateSpec(
        threshold_abs=10,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        decay_numerator=1,
        decay_denominator=1,
        max_abs_per_tensor=64,
        fraction_per_tensor=1.0,
    )


def _tensor_input(
    state_key: str,
    q: list[int],
    acc: list[int],
    votes: list[int],
) -> GlobalRateCapTensorInput:
    state = VoteUpdateState(
        q_levels=torch.tensor(q, dtype=torch.int8),
        accumulators=torch.tensor(acc, dtype=torch.int16),
    )
    inputs = VoteUpdateInputs(votes=torch.tensor(votes, dtype=torch.int16))
    plan = plan_integer_vote_update_reference(state, inputs, _spec())
    return GlobalRateCapTensorInput(
        state_key=state_key,
        state=state,
        plan=plan,
        vote_inputs=inputs,
    )


def _partition_assertions(result) -> None:
    summary = result.step_summary
    candidate = int(summary["global_pre_cap_would_apply_count"])
    accepted = int(summary["global_rate_cap_accepted_count"])
    deferred = int(summary["global_rate_cap_deferred_count"])
    assert candidate == accepted + deferred
    accepted_from_prior = int(summary.get("accepted_from_prior_deferred_count", 0))
    assert 0 <= accepted_from_prior <= accepted
    assert int(summary["accepted_fresh_count"]) == accepted - accepted_from_prior


def run_multistep_backlog_carry_witness(*, carry_enabled: bool) -> dict[str, object]:
    item = _tensor_input(
        "synthetic.cap",
        [0, 0, 0],
        [0, 0, 0],
        [30, 30, 30],
    )
    offsets = {"synthetic.cap": 0}
    first = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=1, step=1),
        tensor_offsets=offsets,
    )
    assert int(first.step_summary["global_rate_cap_deferred_count"]) > 0
    assert first.deferred_backlog

    backlog = first.deferred_backlog if carry_enabled else None
    second = apply_global_rate_cap_reference(
        [item],
        GlobalRateCapSpec(cap=2, step=2),
        deferred_backlog=backlog,
        tensor_offsets=offsets,
    )

    witness: dict[str, object] = {
        "schema_version": "hrm_text_158_r7_multistep_backlog_carry_witness/v1",
        "carry_enabled": bool(carry_enabled),
        "step_n_deferred_count": int(first.step_summary["global_rate_cap_deferred_count"]),
        "step_n_plus_1_max_age_steps": int(
            second.step_summary.get("deferred_backlog_max_age_steps", 0)
        ),
        "step_n_plus_1_accepted_from_prior_deferred_count": int(
            second.step_summary.get("accepted_from_prior_deferred_count", 0)
        ),
        "positive_drain_observed": int(
            second.step_summary.get("accepted_from_prior_deferred_count", 0)
        )
        > 0,
        "cpu_no_model_forward": True,
    }

    _partition_assertions(first)
    _partition_assertions(second)

    if carry_enabled:
        assert witness["step_n_plus_1_max_age_steps"] >= 1
        assert int(witness["step_n_plus_1_accepted_from_prior_deferred_count"]) > 0
        accepted_identities = {(row.state_key, row.flat_index) for row in second.accepted_rows}
        assert ("synthetic.cap", 1) in accepted_identities
    else:
        assert witness["step_n_plus_1_max_age_steps"] == 0

    return witness


def test_multistep_backlog_carry_witness_passes_with_carry_on(tmp_path: Path) -> None:
    witness = run_multistep_backlog_carry_witness(carry_enabled=True)
    out = tmp_path / "r7_multistep_backlog_carry_witness.json"
    out.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    assert witness["carry_enabled"] is True
    assert int(witness["step_n_plus_1_max_age_steps"]) >= 1


def test_multistep_backlog_carry_witness_fails_carry_off() -> None:
    witness = run_multistep_backlog_carry_witness(carry_enabled=False)
    assert witness["carry_enabled"] is False
    assert int(witness["step_n_plus_1_max_age_steps"]) == 0
    with_carry = run_multistep_backlog_carry_witness(carry_enabled=True)
    assert int(with_carry["step_n_plus_1_max_age_steps"]) >= 1
