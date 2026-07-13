"""Regression guards for Fork B resume-parity instrumentation (plan v2 §4)."""
from __future__ import annotations

import copy

import pytest

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_certificate import (
    build_non_target_snapshot,
    snapshot_not_loadable_as_checkpoint_authority,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    _roundtrip_payload_sha256,
    build_trainer_sub2_authority_checkpoint_blob,
    derive_trainer_sub2_authority_states,
    load_trainer_sub2_authority_checkpoint_blob,
    select_trainer_eligible_bitlinears,
)
import torch


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _make_blob():
    model = _TinyTernary()
    with torch.no_grad():
        model.proj.weight.zero_()
        model.tail.weight.fill_(0.25)
        model.tail.bias.zero_()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = derive_trainer_sub2_authority_states(eligible)
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible,
        tensor_states=states,
    )
    return model, eligible, blob


def test_shadow_persistence_reject_1098_1103_unchanged():
    _model, _eligible, blob = _make_blob()
    bad = copy.deepcopy(blob)
    sidecar = bad["trainer_sub2_authority"]
    sidecar["tensor_payloads"]["proj"]["exact_accumulator_shadow_saved"] = True
    sidecar_without_hash = dict(sidecar)
    sidecar_without_hash.pop("authoritative_state_payload_sha256", None)
    sidecar["authoritative_state_payload_sha256"] = _roundtrip_payload_sha256(
        sidecar_without_hash
    )
    fresh = _TinyTernary()
    with torch.no_grad():
        fresh.proj.weight.zero_()
    fresh_eligible = select_trainer_eligible_bitlinears(fresh, use_ternary_bulk=True)
    with pytest.raises(ValueError, match="dense exact accumulator shadows"):
        load_trainer_sub2_authority_checkpoint_blob(
            fresh,
            bad,
            eligible_modules=fresh_eligible,
        )


def test_run_local_snapshot_not_checkpoint_authority():
    snap = build_non_target_snapshot(
        rng_states={"torch": "x"},
        exact_future_batch_sample_ids=(1,),
        loader_cursor={"idx": 0},
        rate_cap_backlog_schedule={"cap": 1},
        q_scales_weights_code_hash={"code": "c"},
        optimizer_empty_proof={"ok": True},
        non_manipulated_manifest_fields={"phase": "p"},
    )
    assert snapshot_not_loadable_as_checkpoint_authority(snap)
    # Snapshot dict is not a trainer_sub2 authority blob
    assert "trainer_sub2_authority" not in snap.to_dict()
    assert snap.to_dict().get("is_checkpoint_authority") is False


def test_fork_b_probe_flags_default_absent_in_parser():
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser

    ns = build_arg_parser().parse_args([])
    # Non-Fork-B path: flags default off / None — probe behavior unchanged when absent
    assert ns.fork_b_resume_parity_certificate is False
    assert ns.fork_b_arm is None
    assert ns.fork_b_cut_t is None
    assert ns.fork_b_artifact_dir is None


def test_fork_b_probe_flags_parse_when_enabled(tmp_path):
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        _maybe_emit_fork_b_resume_parity_artifacts,
        build_arg_parser,
    )

    art = tmp_path / "fb"
    ns = build_arg_parser().parse_args(
        [
            "--fork-b-resume-parity-certificate",
            "--fork-b-arm",
            "C",
            "--fork-b-cut-t",
            "16",
            "--fork-b-artifact-dir",
            str(art),
        ]
    )
    assert ns.fork_b_resume_parity_certificate is True
    meta = _maybe_emit_fork_b_resume_parity_artifacts(
        args=ns,
        parent_receipt={"ok": True},
    )
    assert meta is not None
    assert meta["science_label"] is None
    assert meta["arm"] == "C"
    assert (art / "fork_b_arm_C_cut_16_scaffold.json").is_file()
