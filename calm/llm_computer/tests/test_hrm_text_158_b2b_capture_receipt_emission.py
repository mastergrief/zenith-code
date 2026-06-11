"""CPU fixture tests for B2b capture receipt-write emission (§2C + warmup tags)."""
from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.accumulator_policy_shadow_screen import (
    B2B_SEQUENTIAL_TRACE_SCHEMA,
    SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
)
from calm.hrm_text_158.native_full_stack.b2b_capture_receipt_emission import (
    FROZEN_THRESHOLD_SEMANTICS,
    WARMUP_APPLY_CLASS_CANONICAL,
    WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP,
    derive_step_warmup_apply_tags,
    enrich_b2b_trace_steps_at_receipt_write,
    finalize_b2b_capture_receipt,
    frozen_threshold_semantics_block,
    rewrite_b2b_trace_with_receipt_emissions,
)
from calm.hrm_text_158.native_full_stack.two_tier_threshold_semantics import (
    FROZEN_THRESHOLD_SEMANTICS as MODULE_FROZEN_THRESHOLD_SEMANTICS,
    THRESHOLD_CROSSCHECK_MISMATCH,
    frozen_threshold_semantics_block as module_frozen_threshold_semantics_block,
    resolve_threshold_crosscheck_authority,
)

PRE_B3_FROZEN_THRESHOLD_SEMANTICS = {
    "crossing_threshold_abs": 10,
    "crossing_threshold_source": "canonical_default_spec_accumulator_real_dynamics_verdict",
    "crossing_authority": "vote_update_spec",
    "residual_band_encoding": "threshold_minus_one",
    "row_fields_authority": "telemetry_not_crossing",
    "row_crosscheck_policy": "informational",
}
from calm.hrm_text_158.native_full_stack.oracle_screen_runner import (
    capture_b2b_sequential_pre_update_step,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    build_b2b_sequential_capture_receipt,
)


def _row(
    candidate_id: str,
    *,
    flat_index: int,
    new_acc: int,
) -> dict[str, object]:
    proposal_direction = 1 if int(new_acc) >= 0 else -1
    threshold = 10
    return {
        "candidate_id": candidate_id,
        "flat_index": flat_index,
        "pre_accumulator_i16": 0,
        "new_acc_i32_signed": new_acc,
        "vote_value": 1,
        "proposal_direction": proposal_direction,
        "current_q_level": 0,
        "in_target_tie_band": True,
        "threshold_residual_signed": int(new_acc) - proposal_direction * threshold,
        "proximity_to_threshold": abs(abs(int(new_acc)) - threshold),
        "local_loss_delta": -0.1,
        "current_rank_position": flat_index,
    }


def _step(
    step_index: int,
    rows: list[dict[str, object]],
    *,
    applied_flat_indices: list[int] | None = None,
    q_changed_count: int = 1,
) -> dict[str, object]:
    telemetry: dict[str, object] = {"q_changed_count": q_changed_count}
    if applied_flat_indices is not None:
        telemetry["applied_flip_flat_indices"] = list(applied_flat_indices)
    return {
        "optimizer_step_index": step_index,
        "source_kind": SOURCE_KIND_WITHIN_TIE_BAND_DISCRIMINATOR,
        "source_table_hash": "deadbeef",
        "sampled_candidate_table": rows,
        "post_update_telemetry": telemetry,
    }


def test_threshold_semantics_block_is_verbatim_frozen() -> None:
    block = frozen_threshold_semantics_block()
    assert block == FROZEN_THRESHOLD_SEMANTICS
    assert block == PRE_B3_FROZEN_THRESHOLD_SEMANTICS


def test_threshold_semantics_byte_parity_against_pre_b3_literal() -> None:
    emitted = finalize_b2b_capture_receipt({})["threshold_semantics"]
    assert emitted == PRE_B3_FROZEN_THRESHOLD_SEMANTICS


def test_threshold_semantics_reexport_parity_across_import_paths() -> None:
    assert FROZEN_THRESHOLD_SEMANTICS == MODULE_FROZEN_THRESHOLD_SEMANTICS
    assert frozen_threshold_semantics_block() == module_frozen_threshold_semantics_block()


def test_threshold_crosscheck_mismatch_resolves_informational_under_frozen_semantics() -> None:
    assert (
        resolve_threshold_crosscheck_authority(THRESHOLD_CROSSCHECK_MISMATCH)
        == "informational"
    )
    assert resolve_threshold_crosscheck_authority("passed") == "passed"


def test_finalize_b2b_capture_receipt_attaches_threshold_semantics() -> None:
    base = build_b2b_sequential_capture_receipt(
        capture_out=Path("/tmp/trace.ndjson"),
        steps_captured=1,
        min_steps_for_verdict=1,
        trace_hashes=["abc"],
        parent_hash_unchanged=True,
        max_sampled_candidates=32,
    )
    receipt = finalize_b2b_capture_receipt(base)
    assert receipt["threshold_semantics"] == frozen_threshold_semantics_block()


def test_subthreshold_derivation_applied_new_acc_four() -> None:
    step = _step(
        1,
        [_row("applied", flat_index=3, new_acc=4)],
        applied_flat_indices=[3],
    )
    tags = derive_step_warmup_apply_tags(step, applied_candidate_ids_by_step={})
    assert tags["warmup_apply_class"] == WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP
    assert tags["effective_apply_threshold_abs"] == 4


def test_canonical_derivation_applied_new_acc_at_least_ten() -> None:
    step = _step(
        2,
        [_row("applied", flat_index=5, new_acc=15)],
        applied_flat_indices=[5],
    )
    tags = derive_step_warmup_apply_tags(step, applied_candidate_ids_by_step={})
    assert tags["warmup_apply_class"] == WARMUP_APPLY_CLASS_CANONICAL
    assert tags["effective_apply_threshold_abs"] is None


def test_rewrite_b2b_trace_persists_warmup_tags(tmp_path: Path) -> None:
    trace_path = tmp_path / "trace.ndjson"
    step = _step(
        1,
        [_row("applied", flat_index=1, new_acc=4)],
        applied_flat_indices=[1],
    )
    trace_path.write_text(
        json.dumps({"schema": B2B_SEQUENTIAL_TRACE_SCHEMA}, sort_keys=True)
        + "\n"
        + json.dumps(step, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    enriched = rewrite_b2b_trace_with_receipt_emissions(trace_path)
    assert enriched[0]["warmup_apply_class"] == WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP
    assert enriched[0]["effective_apply_threshold_abs"] == 4
    disk_steps = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and "schema" not in line
    ]
    assert disk_steps[0]["warmup_apply_class"] == WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP


def test_enrich_trace_steps_uses_teacher_forced_applied_candidate_when_needed() -> None:
    rows = [
        _row("oracle", flat_index=1, new_acc=4),
        _row("decoy", flat_index=2, new_acc=20),
    ]
    rows[0]["local_loss_delta"] = -1.0
    rows[1]["local_loss_delta"] = -0.1
    step = _step(1, rows, q_changed_count=1)
    enriched = enrich_b2b_trace_steps_at_receipt_write([step])
    assert enriched[0]["warmup_apply_class"] == WARMUP_APPLY_CLASS_SUBTHRESHOLD_BOOTSTRAP
    assert enriched[0]["effective_apply_threshold_abs"] == 4


def test_loop_capture_path_untouched_by_emission_contract() -> None:
    capture_source = inspect.getsource(capture_b2b_sequential_pre_update_step)
    assert "warmup_apply_class" not in capture_source
    assert "threshold_semantics" not in capture_source
    assert "effective_apply_threshold_abs" not in capture_source
