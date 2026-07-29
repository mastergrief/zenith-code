"""GPU-live nodes for LANDS-AB evaluation (PLAN_v6 / IMPLEMENT_v8).

FAIL-CLOSED without CUDA. Production-site instrumentation; branch-preserving
raw observations; O_EXCL to runtime-scratch only (never repo artifacts/).
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
    o_excl_write_json,
    runtime_scratch_raw_path,
    validate_raw_row_observation,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
    classify_phase_topology,
    synthesize_duplicate_start_events,
    synthesize_good_topology_events,
    synthesize_missing_coverage_events,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
    ENV_JSONL,
    emit_one_enforcer_cycle_to_memory_and_jsonl,
    load_jsonl_events,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_cuda_sites import (
    measure_b1_local_update_site,
    measure_b2_roundtrip_site,
    measure_b3_landing_site,
    measure_oracle_at_production_site,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    build_trainer_sub2_authority_local_update_receipt,
    build_trainer_sub2_authority_roundtrip_receipt,
    build_sparse_vote_authority_landing_receipt,
    save_trainer_sub2_live_checkpoint_envelope,
)


def _scratch_dir() -> Path:
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import resolve_run_scratch_dir
    return resolve_run_scratch_dir(create=True)


def _write_obs(gating_row: str, obs: dict, scratch: Path) -> str:
    """O_EXCL to unique runtime-scratch path — NO pre-delete, never artifacts/."""
    validate_raw_row_observation(obs)
    path = runtime_scratch_raw_path(
        scratch_dir=scratch, gating_row=gating_row, run_nonce=uuid.uuid4().hex[:12]
    )
    return o_excl_write_json(path, obs)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail(
            "CUDA required for lands_ab_eval gpu_live nodes (fail-closed; no CPU fallback)"
        )
    return torch.device("cuda")


class _Tiny(torch.nn.Module):
    def __init__(self, dim: int = 4, tag: int = 0) -> None:
        super().__init__()
        self.lin = BitLinear(dim, dim, bias=False)
        self._tag = int(tag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        if self._tag:
            y = y + 0.0 * float(self._tag)
        return y


def _batch(device: torch.device, dim: int = 4, seed: int = 0) -> dict:
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return {"x": torch.randn(2, dim, generator=g).to(device)}


def _loss(model, batch):
    return (model(batch["x"]) ** 2).mean()


def _output(model, batch):
    return model(batch["x"])


def _assert_transport(obs: dict) -> None:
    """Schema/transport only — do NOT assert S3/S4/S5 polarity True."""
    validate_raw_row_observation(obs)
    assert obs["science_claim"] is False
    assert obs["synthetic_only"] is False
    assert "measured_surfaces" in obs
    # each surface is a real bool (True or False — both legal)
    for k, v in obs["measured_surfaces"].items():
        assert type(v) is bool, k
    topo = obs.get("phase_topology") or {}
    assert topo.get("good_topology") is True, topo
    events = obs.get("phase_events") or []
    assert classify_phase_topology(events)["good_topology"] is True


def test_gpu_live_lands_ab_b1_apply_twin_s3_s4_s6():
    device = _require_cuda()
    torch.manual_seed(11)
    scratch = _scratch_dir()
    model = _Tiny(tag=1).to(device)
    obs = measure_b1_local_update_site(
        model=model,
        batch=_batch(device, seed=11),
        forward_loss_fn=_loss,
        device=device,
    )
    _assert_transport(obs)
    assert obs["gating_row"] == "G_CUDA_B1_APPLY"
    assert obs["site_tag"] == "B1_local_update"
    assert obs["production_site"] == "build_trainer_sub2_authority_local_update_receipt"
    sha = _write_obs("G_CUDA_B1_APPLY", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=b1_apply sha={sha[:12]}")


def test_gpu_live_lands_ab_b2_apply_twin_s3_s4_s6():
    device = _require_cuda()
    torch.manual_seed(22)
    scratch = _scratch_dir()

    def fresh():
        return _Tiny(tag=2).to(device)

    model = _Tiny(tag=2).to(device)
    obs = measure_b2_roundtrip_site(
        model=model,
        fresh_model_fn=fresh,
        batch=_batch(device, seed=22),
        forward_loss_fn=_loss,
        forward_output_fn=_output,
        device=device,
    )
    _assert_transport(obs)
    assert obs["gating_row"] == "G_CUDA_B2_APPLY"
    assert obs["site_tag"] == "B2_roundtrip"
    assert obs["production_site"] == "build_trainer_sub2_authority_roundtrip_receipt"
    sha = _write_obs("G_CUDA_B2_APPLY", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=b2_apply sha={sha[:12]}")


def test_gpu_live_lands_ab_b3_apply_twin_s3_s4_s6():
    device = _require_cuda()
    torch.manual_seed(33)
    scratch = _scratch_dir()

    def fresh():
        return _Tiny(tag=3).to(device)

    model = _Tiny(tag=3).to(device)
    obs = measure_b3_landing_site(
        model=model,
        fresh_model_fn=fresh,
        batch=_batch(device, seed=33),
        forward_loss_fn=_loss,
        forward_output_fn=_output,
        device=device,
    )
    _assert_transport(obs)
    assert obs["gating_row"] == "G_CUDA_B3_APPLY"
    assert obs["site_tag"] == "B3_landing"
    assert obs["production_site"] == "build_sparse_vote_authority_landing_receipt"
    sha = _write_obs("G_CUDA_B3_APPLY", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=b3_apply sha={sha[:12]}")


def test_gpu_live_lands_ab_oracle_b1_events_equal():
    device = _require_cuda()
    torch.manual_seed(101)
    scratch = _scratch_dir()
    model = _Tiny(tag=11).to(device)

    def runner():
        return build_trainer_sub2_authority_local_update_receipt(
            model=model,
            batch=_batch(device, seed=101),
            forward_loss_fn=_loss,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
        )

    obs = measure_oracle_at_production_site(
        gating_row="G_CUDA_ORACLE_B1",
        model=model,
        batch=_batch(device, seed=101),
        forward_loss_fn=_loss,
        device=device,
        site_tag="oracle_B1_local",
        production_site="build_trainer_sub2_authority_local_update_receipt",
        site_runner=runner,
    )
    _assert_transport(obs)
    assert obs["gating_row"] == "G_CUDA_ORACLE_B1"
    assert obs["metrics"]["oracle_mode_on_named_site"] is True
    assert obs["metrics"]["named_builder_returned_receipt"] is True
    assert obs["metrics"]["resolved_mode"] == "oracle_on"
    sha = _write_obs("G_CUDA_ORACLE_B1", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=oracle_b1 sha={sha[:12]}")


def test_gpu_live_lands_ab_oracle_b2_events_equal():
    device = _require_cuda()
    torch.manual_seed(202)
    scratch = _scratch_dir()
    model = _Tiny(tag=22).to(device)

    def fresh():
        return _Tiny(tag=22).to(device)

    def runner():
        return build_trainer_sub2_authority_roundtrip_receipt(
            model=model,
            fresh_model_fn=fresh,
            batch=_batch(device, seed=202),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
        )

    obs = measure_oracle_at_production_site(
        gating_row="G_CUDA_ORACLE_B2",
        model=model,
        batch=_batch(device, seed=202),
        forward_loss_fn=_loss,
        device=device,
        site_tag="oracle_B2_roundtrip_shape",
        production_site="build_trainer_sub2_authority_roundtrip_receipt",
        site_runner=runner,
    )
    _assert_transport(obs)
    assert obs["production_site"] == "build_trainer_sub2_authority_roundtrip_receipt"
    assert obs["metrics"]["oracle_mode_on_named_site"] is True
    assert obs["metrics"]["named_builder_returned_receipt"] is True
    assert obs["metrics"]["resolved_mode"] == "oracle_on"
    sha = _write_obs("G_CUDA_ORACLE_B2", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=oracle_b2 sha={sha[:12]}")


def test_gpu_live_lands_ab_oracle_b3_events_equal():
    device = _require_cuda()
    torch.manual_seed(303)
    scratch = _scratch_dir()
    model = _Tiny(tag=33).to(device)

    def fresh():
        return _Tiny(tag=33).to(device)

    def runner():
        p1 = save_trainer_sub2_live_checkpoint_envelope(
            model,
            use_ternary_bulk=True,
            step=0,
            config={"proof": "lands_ab_oracle_b3"},
            source_pin="lands_ab_oracle_b3",
            epoch=0,
        )
        return build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="lands_ab_oracle_b3",
            p1_checkpoint=p1,
            p1_envelope_bytes=b"{}",
            fresh_model_fn=fresh,
            batch=_batch(device, seed=303),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            parity_max_abs_diff_by_site={
                "cache_builder": 0.0,
                "main_kl": 0.0,
                "retained_fallback": 0.0,
            },
            use_ternary_bulk=True,
            device=device,
            sparse_vote_authority_mode=SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
        )

    obs = measure_oracle_at_production_site(
        gating_row="G_CUDA_ORACLE_B3",
        model=model,
        batch=_batch(device, seed=303),
        forward_loss_fn=_loss,
        device=device,
        site_tag="oracle_B3_landing_envelope",
        production_site="build_sparse_vote_authority_landing_receipt",
        site_runner=runner,
    )
    _assert_transport(obs)
    assert obs["production_site"] == "build_sparse_vote_authority_landing_receipt"
    assert obs["metrics"]["oracle_mode_on_named_site"] is True
    assert obs["metrics"]["named_builder_returned_receipt"] is True
    assert obs["metrics"]["resolved_mode"] == "oracle_on"
    sha = _write_obs("G_CUDA_ORACLE_B3", obs, scratch)
    assert len(sha) == 64
    print(f"device_type={device.type} node=oracle_b3 sha={sha[:12]}")


def test_phase_topology_characterizations_cpu_static():
    good = classify_phase_topology(synthesize_good_topology_events())
    assert good["good_topology"] is True
    assert good["detail"] == "good_topology"
    dup = classify_phase_topology(synthesize_duplicate_start_events())
    assert dup["good_topology"] is False
    assert dup["detail"] == "duplicate_start"
    miss = classify_phase_topology(synthesize_missing_coverage_events())
    assert miss["good_topology"] is False
    assert miss["detail"] == "missing_coverage"


def test_runtime_scratch_rejects_artifacts_path():
    with pytest.raises(ValueError, match="artifacts"):
        runtime_scratch_raw_path(
            scratch_dir=Path("artifacts/acc_entropy"),
            gating_row="G_CPU_STATIC_AB",
            run_nonce="x",
        )



def test_enforcer_jsonl_live_relay_characterization_cpu():
    """When ENV_JSONL set, one cycle writes enforcer-schema JSONL and classifies good."""
    import json
    import tempfile
    from pathlib import Path as P

    with tempfile.TemporaryDirectory() as td:
        jsonl = P(td) / "phase_events.jsonl"
        os.environ[ENV_JSONL] = str(jsonl)
        try:
            node_id = "G_CUDA_B1_APPLY"
            events = emit_one_enforcer_cycle_to_memory_and_jsonl(node_id)
            assert len(events) == 8
            topo = classify_phase_topology(
                events, expected_node_id=node_id, require_enforcer_fields=True
            )
            assert topo["good_topology"] is True
            lines = load_jsonl_events(jsonl)
            assert len(lines) == 8
            # schema: type, phase, node_id, ts_monotonic; duration_s on END
            starts = [e for e in lines if e["type"] == "PHASE_START"]
            ends = [e for e in lines if e["type"] == "PHASE_END"]
            assert len(starts) == 4 and len(ends) == 4
            for e in lines:
                assert e["node_id"] == node_id
                assert isinstance(e["ts_monotonic"], (int, float))
            for e in ends:
                assert "duration_s" in e
                assert float(e["duration_s"]) >= 0
            topo2 = classify_phase_topology(
                lines, expected_node_id=node_id, require_enforcer_fields=True
            )
            assert topo2["good_topology"] is True
        finally:
            os.environ.pop(ENV_JSONL, None)


def test_enforcer_jsonl_absent_env_is_noop():
    """Without ENV_JSONL, cycle still works in-memory and writes nothing required."""
    os.environ.pop(ENV_JSONL, None)
    events = emit_one_enforcer_cycle_to_memory_and_jsonl("G_CUDA_B2_APPLY")
    assert len(events) == 8
    assert classify_phase_topology(
        events, expected_node_id="G_CUDA_B2_APPLY", require_enforcer_fields=True
    )["good_topology"] is True


def test_phase_label_fidelity_document():
    """Step3: phase labels match enforcer vocabulary (forward_backward/update/emission/flush).

    LANDS-AB phase emitters use the same labels the enforcer budget classifier
    expects. This is a fidelity document + static assertion — not a native
    builder stream rewrite (TSA remains RO).
    """
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import PHASE_ORDER
    from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
        synthesize_good_topology_events,
        classify_phase_topology,
    )
    assert tuple(PHASE_ORDER) == (
        "forward_backward", "update", "emission", "flush"
    )
    events = synthesize_good_topology_events(node_id="G_CUDA_B1_APPLY")
    phases = [e["phase"] for e in events if e["type"] == "PHASE_START"]
    assert phases == list(PHASE_ORDER)
    assert classify_phase_topology(
        events, expected_node_id="G_CUDA_B1_APPLY", require_enforcer_fields=True
    )["good_topology"] is True
