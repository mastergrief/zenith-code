"""CPU smoke for probe event-coded sparse vote authority wiring + milestone emit."""
from __future__ import annotations

import copy
import json
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from unittest import mock

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.curriculum import BroadTokenizer
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    apply_bounded_delta_vote_step,
    make_event_coded_live_tensor_state,
)
from calm.hrm_text_158.native_full_stack.event_coded_vote_update_adapter import (
    C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY,
    carrier_content_sha256,
)
from calm.hrm_text_158.native_full_stack.global_rate_cap import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    GlobalRateCapSpec,
)
from calm.hrm_text_158.native_full_stack.vote_update import VoteUpdateSpec
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    EXACT_GLOBAL_CAP_TIE_RULE_MODE,
    MILESTONE_BUDGETED_PHASE_IDS,
    PHASE_MILESTONE_COUNTER_SCHEMA,
    PhaseMilestoneEmitter,
    _plan_integer_vote_update_for_control_arm_surfaces,
    _validate_event_coded_sparse_vote_authority_config,
    build_model_from_checkpoint,
    file_sha256,
    run_c2p1_probe,
    select_eligible_bitlinears,
)
from scripts.train_hrm_text_158 import _build_ckpt_config, SOURCE_PIN


TINY_ARCH = dict(
    max_len=64,
    hidden_size=64,
    n_layers=2,
    num_heads=2,
    expansion=4,
    H_cycles=1,
    L_cycles=1,
    half_layers=True,
    bp_warmup_ratio=0.2,
    bp_min_steps=1,
    bp_max_steps=2,
)


def _tiny_parent_blob(*, batch_size: int = 16, hidden_size: int = 64) -> dict:
    tok = BroadTokenizer()
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=TINY_ARCH["max_len"],
        n_layers=TINY_ARCH["n_layers"],
        hidden_size=int(hidden_size),
        num_heads=TINY_ARCH["num_heads"],
        expansion=TINY_ARCH["expansion"],
        H_cycles=TINY_ARCH["H_cycles"],
        L_cycles=TINY_ARCH["L_cycles"],
        half_layers=TINY_ARCH["half_layers"],
        bp_warmup_ratio=TINY_ARCH["bp_warmup_ratio"],
        bp_min_steps=TINY_ARCH["bp_min_steps"],
        bp_max_steps=TINY_ARCH["bp_max_steps"],
        use_ternary_bulk=True,
    )
    model = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=tok.vocab_size))
    return {
        "model_state": model.state_dict(),
        "config": _build_ckpt_config(
            model,
            tok,
            cfg,
            TINY_ARCH["max_len"],
            batch_size=batch_size,
            curriculum_rung="L0c1",
            curriculum_seed=17,
            replay_ratio=0.0,
            prior_rungs=[],
        ),
        "step": 50,
        "epoch": 1,
        "source_pin": SOURCE_PIN,
    }


def _state_parity_subset(receipt: dict) -> dict:
    checkpoint = receipt["checkpoint_payload"]
    return {
        "steps_completed": int(receipt["steps_completed"]),
        "stop_reason": receipt["stop_reason"],
        "step_report_keys": sorted(receipt["step_reports"]),
        "authoritative_state_sha256": checkpoint["authoritative_state_sha256"],
        "tensor_summaries": checkpoint["tensor_summaries"],
    }


def _milestone_line_counts(scratch_root: Path) -> dict[str, int]:
    root = scratch_root / "liveness_milestones"
    if not root.exists():
        return {}
    counts: dict[str, int] = {}
    for path in sorted(root.glob("*.jsonl")):
        counts[path.stem] = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return counts


def _sparse_authority_probe_kwargs(
    *,
    parent: Path,
    parent_sha: str,
    scratch_root: Path,
    steps: int = 2,
    hidden_size: int = 64,
) -> dict[str, Any]:
    return dict(
        parent=parent,
        parent_sha256=parent_sha,
        scratch_root=scratch_root,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=int(steps),
        batch_size=16,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        persistent_accumulator_event_coded_live=True,
        persistent_q_ternary_base3_codec=True,
        event_coded_sparse_vote_authority=True,
        global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        tie_rule_mode=EXACT_GLOBAL_CAP_TIE_RULE_MODE,
        d_diagnostic_compact_step_reports=True,
        d_live_carrier_snapshot_enabled=True,
        emit_progress=True,
    )


@contextmanager
def _dense_vote_alloc_guard(eligible_weight_shapes: set[tuple[int, int]]) -> Iterator[None]:
    original_zeros = torch.zeros

    def guarded_zeros(*size, **kwargs):
        dtype = kwargs.get("dtype", torch.get_default_dtype())
        if len(size) == 1 and isinstance(size[0], (tuple, list)):
            shape = tuple(int(dim) for dim in size[0])
        else:
            shape = tuple(int(dim) for dim in size)
        if len(shape) == 2 and shape in eligible_weight_shapes:
            if dtype == torch.float32:
                raise AssertionError("dense FP32 [O,I] allocation detected on sparse-authority path")
            if dtype == torch.int16:
                raise AssertionError("dense int16 [O,I] allocation detected on sparse-authority path")
        return original_zeros(*size, **kwargs)

    with mock.patch("torch.zeros", side_effect=guarded_zeros):
        yield


def _apply_sparse_authority_cap_oracle(
    states: dict[str, Any],
    sparse_by_key: dict[str, Any],
    vote_specs: dict[str, VoteUpdateSpec],
    cap: GlobalRateCapSpec,
    **extra_kwargs: Any,
):
    return apply_bounded_delta_vote_step(
        states,
        None,
        vote_specs,
        candidate_sparse_vote_events_by_key=sparse_by_key,
        global_cap_spec=cap,
        event_coded_sparse_vote_authority=True,
        **extra_kwargs,
    )


def test_sparse_authority_off_byte_identical_and_zero_milestone_files(tmp_path: Path) -> None:
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)
    common = dict(
        parent=parent,
        parent_sha256=parent_sha,
        device="cpu",
        eligible_scope="first-bitlinear",
        steps=1,
        batch_size=16,
        max_len=TINY_ARCH["max_len"],
        curriculum_seed=17,
        enabled=True,
        emit_progress=True,
    )

    implicit_off = run_c2p1_probe(scratch_root=tmp_path / "scratch_implicit_off", **common)
    explicit_off = run_c2p1_probe(
        scratch_root=tmp_path / "scratch_explicit_off",
        event_coded_sparse_vote_authority=False,
        **common,
    )

    assert _state_parity_subset(implicit_off) == _state_parity_subset(explicit_off)
    for scratch in (tmp_path / "scratch_implicit_off", tmp_path / "scratch_explicit_off"):
        assert not (scratch / "liveness_milestones").exists()
        assert _milestone_line_counts(scratch) == {}


def test_sparse_authority_on_full_step_through_receipt_write(tmp_path: Path) -> None:
    parent = tmp_path / "tiny_parent.pt"
    torch.save(_tiny_parent_blob(), parent)
    parent_sha = file_sha256(parent)
    scratch_root = tmp_path / "scratch_on"
    kwargs = _sparse_authority_probe_kwargs(
        parent=parent,
        parent_sha=parent_sha,
        scratch_root=scratch_root,
        steps=2,
    )

    model, _, _ = build_model_from_checkpoint(_tiny_parent_blob(), torch.device("cpu"))
    eligible = select_eligible_bitlinears(model, eligible_scope="first-bitlinear")
    weight_shapes = {
        tuple(int(dim) for dim in module.weight.shape)
        for module in eligible.values()
    }

    captured_apply: list[dict[str, Any]] = []
    original_apply = apply_bounded_delta_vote_step

    def _capturing_apply(states, votes_by_key, vote_specs, **apply_kwargs):
        with _dense_vote_alloc_guard(weight_shapes):
            pre_states = copy.deepcopy(states)
            result = original_apply(states, votes_by_key, vote_specs, **apply_kwargs)
        captured_apply.append(
            dict(
                states=pre_states,
                votes_by_key=votes_by_key,
                vote_specs=vote_specs,
                apply_kwargs=apply_kwargs,
                result=result,
            )
        )
        return result

    with (
        mock.patch(
            "scripts.hrm_text_158_bounded_delta_acquisition_probe.apply_bounded_delta_vote_step",
            side_effect=_capturing_apply,
        ),
        mock.patch(
            "scripts.hrm_text_158_bounded_delta_acquisition_probe._plan_integer_vote_update_for_control_arm_surfaces",
            side_effect=AssertionError(
                "control-arm planner must not run under sparse vote authority"
            ),
        ),
    ):
        receipt = run_c2p1_probe(**kwargs)

    disk_receipt = json.loads((scratch_root / "receipt.json").read_text(encoding="utf-8"))
    for payload in (receipt, disk_receipt):
        assert payload["event_coded_sparse_vote_authority"] is True
        assert payload["control_arm_index_surfaces_skipped_sparse_authority"] is True
        assert int(payload.get("C8_TRANSIENT_DENSE_COMPUTE_NUMEL", -1)) == 0
        assert payload["steps_completed"] == 2
        assert payload["stop_reason"] in {"steps", "max_steps_completed"}

    assert captured_apply, "expected sparse-authority apply captures"
    for capture in captured_apply:
        assert capture["votes_by_key"] is None
        assert capture["apply_kwargs"]["event_coded_sparse_vote_authority"] is True
        sparse_by_key = capture["apply_kwargs"]["candidate_sparse_vote_events_by_key"]
        assert sparse_by_key
        oracle = _apply_sparse_authority_cap_oracle(
            capture["states"],
            sparse_by_key,
            capture["vote_specs"],
            capture["apply_kwargs"]["global_cap_spec"],
            **{
                key: capture["apply_kwargs"][key]
                for key in (
                    "local_selection_ordering_mode",
                    "local_selection_ordering_seed",
                    "local_selection_ordering_step",
                )
                if key in capture["apply_kwargs"]
            },
        )
        probe_result = capture["result"]
        for state_key in capture["states"]:
            probe_carrier = probe_result.tensor_states[state_key].event_coded_live_carrier
            oracle_carrier = oracle.tensor_states[state_key].event_coded_live_carrier
            assert probe_carrier is not None
            assert oracle_carrier is not None
            assert carrier_content_sha256(probe_carrier) == carrier_content_sha256(
                oracle_carrier
            )

    for step_key in ("1", "2"):
        step_report = receipt["step_reports"][step_key]
        step_result = step_report["step_result"]
        assert step_result.get("control_arm_index_surfaces_skipped_sparse_authority") is True
        assert "global_summary" in step_result
        assert int(step_result["global_summary"].get(C8_TRANSIENT_DENSE_COMPUTE_NUMEL_KEY, -1)) == 0

    assert (scratch_root / "receipt.json").is_file()


def test_milestone_emit_split_and_numel_independence(tmp_path: Path) -> None:
    parent_small = tmp_path / "tiny_parent_small.pt"
    parent_large = tmp_path / "tiny_parent_large.pt"
    torch.save(_tiny_parent_blob(hidden_size=64), parent_small)
    torch.save(_tiny_parent_blob(hidden_size=256), parent_large)

    timings: list[float] = []
    counts: list[dict[str, int]] = []
    for parent, hidden_size in ((parent_small, 64), (parent_large, 256)):
        scratch_root = tmp_path / f"scratch_{hidden_size}"
        started = time.perf_counter()
        receipt = run_c2p1_probe(
            **_sparse_authority_probe_kwargs(
                parent=parent,
                parent_sha=file_sha256(parent),
                scratch_root=scratch_root,
                steps=2,
                hidden_size=hidden_size,
            )
        )
        timings.append(time.perf_counter() - started)
        assert receipt["steps_completed"] == 2
        counts.append(_milestone_line_counts(scratch_root))

    assert counts[0] == counts[1]
    expected_per_step = {
        "step_forward_backward": 2,
        "sparse_vote_construction": 2,
        "sparse_cap_apply": 2,
        "live_carrier_snapshot_emit": 2,
    }
    for phase_id, expected in expected_per_step.items():
        assert counts[0].get(phase_id) == expected
    assert counts[0].get("artifact_flush") == 1
    assert timings[1] < max(2.0 * timings[0], timings[0] + 30.0)

    emitter = PhaseMilestoneEmitter(
        tmp_path / "emitter_unit",
        enabled=True,
        device=torch.device("cpu"),
    )
    for _ in range(10_000):
        emitter.record_phase_complete(
            "sparse_vote_construction",
            optimizer_step_index=1,
            elapsed_since_phase_enter_seconds=0.001,
        )
    lines = (tmp_path / "emitter_unit" / "liveness_milestones" / "sparse_vote_construction.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(lines) == 10_000
    payload = json.loads(lines[-1])
    assert payload["schema"] == PHASE_MILESTONE_COUNTER_SCHEMA
    assert payload["milestone_counter"] == 10_000
    assert set(MILESTONE_BUDGETED_PHASE_IDS) == {
        "step_forward_backward",
        "sparse_vote_construction",
        "sparse_cap_apply",
        "live_carrier_snapshot_emit",
        "artifact_flush",
    }


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"persistent_accumulator_event_coded_live": False}, "requires --persistent-accumulator-event-coded-live"),
        ({"two_tier_carry_w6_enabled": True}, "two-tier-carry-w6"),
        ({"b2b_sequential_capture_enabled": True}, "b2b-sequential-capture"),
        ({"votes_emit_enabled": True}, "votes-emit-enabled"),
        ({"carrier_growth_enabled": True}, "carrier-growth-enabled"),
        ({"d_recompute_window_instrumentation_enabled": True}, "d-recompute-window-instrumentation"),
        (
            {"d_recompute_calibration_warmup_out": Path("/tmp/warmup.jsonl")},
            "d-recompute-calibration-warmup-out",
        ),
    ],
)
def test_sparse_authority_startup_fail_close(overrides: dict[str, Any], match: str) -> None:
    base = dict(
        event_coded_sparse_vote_authority=True,
        persistent_accumulator_event_coded_live=True,
        two_tier_carry_w6_enabled=False,
        b2b_sequential_capture_enabled=False,
        votes_emit_enabled=False,
        carrier_growth_enabled=False,
        d_recompute_window_instrumentation_enabled=False,
        d_recompute_calibration_warmup_out=None,
    )
    base.update(overrides)
    with pytest.raises(ValueError, match=match):
        _validate_event_coded_sparse_vote_authority_config(**base)


def test_control_arm_planner_requires_dense_votes_by_key() -> None:
    state = make_event_coded_live_tensor_state(
        "toy.proj",
        torch.tensor([0, 1, -1, 0], dtype=torch.int8),
        0.5,
    )
    spec = VoteUpdateSpec(
        threshold_abs=8,
        accumulator_clip_min=-127,
        accumulator_clip_max=127,
        max_abs_per_tensor=4,
    )
    with pytest.raises((TypeError, AttributeError, ValueError)):
        _plan_integer_vote_update_for_control_arm_surfaces(
            tensor_states={"toy.proj": state},
            votes_by_key=None,
            vote_specs_by_key={"toy.proj": spec},
            replay_ce_veto_votes_by_key=None,
            replay_ce_veto_moves_by_key=None,
            pc_aux_votes_by_key=None,
            pc_aux_moves_by_key=None,
            pc_aux_mode="telemetry",
            local_selection_ordering_mode="rank_bucket_current",
            local_selection_ordering_seed=17,
            local_selection_ordering_step=1,
        )
