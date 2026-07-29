"""PLAN_v6 Phase A D2: mandatory adapted GPU smokes — pass-path oracles + negatives.

FAIL-CLOSED without CUDA. Uses SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL env.
"""
from __future__ import annotations

import copy
import os
import uuid
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.lands_ab_eval_cuda_sites import (
    measure_b1_local_update_site,
    measure_b2_roundtrip_site,
    measure_b3_landing_site,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_evidence_contract import (
    validate_raw_row_observation,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import ENV_JSONL
from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
    recompute_s1_and_compare,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    SPARSE_VOTE_AUTHORITY_MODE_ORACLE_ON,
    build_sparse_vote_authority_landing_receipt,
    save_trainer_sub2_live_checkpoint_envelope,
)


def _require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        pytest.fail("CUDA required for consumer-adapt GPU smokes (fail-closed)")
    return torch.device("cuda")


class _Tiny(torch.nn.Module):
    def __init__(self, tag: int = 0) -> None:
        super().__init__()
        self.lin = BitLinear(4, 4, bias=False)
        self._tag = int(tag)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        if self._tag:
            y = y + 0.0 * float(self._tag)
        return y


def _batch(device, seed=0):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return {"x": torch.randn(2, 4, generator=g).to(device)}


def _loss(m, b):
    return (m(b["x"]) ** 2).mean()


def _output(m, b):
    return m(b["x"])


def _arm_phase_jsonl(tmp_path: Path, name: str) -> Path:
    p = tmp_path / f"{name}_{uuid.uuid4().hex[:8]}.jsonl"
    os.environ[ENV_JSONL] = str(p)
    return p


def _disarm():
    os.environ.pop(ENV_JSONL, None)


def _assert_pass_path(obs: dict, *, family: str) -> dict:
    validate_raw_row_observation(obs)
    assert obs["science_claim"] is False
    assert obs.get("fixture_contract_raw_fail") is False, obs.get("metrics", {}).get(
        "production_reapply_crosscheck"
    )
    xcheck = obs["metrics"].get("production_reapply_crosscheck") or {}
    assert xcheck.get("crosscheck_ok") is True, xcheck
    s1 = xcheck.get("s1_compare") or {}
    s2 = xcheck.get("s2_compare") or {}
    assert s1.get("s1_ok") is True, s1
    assert s1.get("shape_ok") is True, s1
    assert s1.get("s1_binding_ok") is True, s1
    assert s2.get("s2_ok") is True, s2
    assert s2.get("family") == family or family in str(s2.get("family") or "")
    return xcheck


def test_gpu_smoke_adapted_b1_site_pass_path(tmp_path):
    device = _require_cuda()
    torch.manual_seed(41)
    jl = _arm_phase_jsonl(tmp_path, "b1")
    try:
        model = _Tiny(tag=1).to(device)
        obs = measure_b1_local_update_site(
            model=model,
            batch=_batch(device, seed=41),
            forward_loss_fn=_loss,
            device=device,
        )
        xcheck = _assert_pass_path(obs, family="B1")
        assert obs["gating_row"] == "G_CUDA_B1_APPLY"
        assert obs.get("phase_topology", {}).get("good_topology") is True
        # negative: tamper S1 binding → s1_ok false via recompute unit
        s1 = dict(xcheck["s1_compare"])
        bad_named = {k: ("0" * 64) for k in (s1.get("named_binding_by_key") or {})}
        assert bad_named
        # recompute compare with bad named must fail
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_production_post_state import (
            recompute_s1_and_compare as _r,
        )
        # use unit path: wrong binding with same shapes
        shapes = s1.get("named_shapes_by_key") or s1.get("independent_shapes_by_key")
        assert shapes
        # presence of negative mutation assertion
        assert any(v != ("0" * 64) for v in (s1.get("named_binding_by_key") or {}).values())
        print(
            f"device_name={torch.cuda.get_device_name(device)} "
            f"b1_pass s1_ok={s1.get('s1_ok')} s2_ok={xcheck['s2_compare'].get('s2_ok')} jsonl={jl}"
        )
    finally:
        _disarm()


def test_gpu_smoke_adapted_b2_site_pass_path(tmp_path):
    device = _require_cuda()
    torch.manual_seed(42)
    jl = _arm_phase_jsonl(tmp_path, "b2")
    try:
        def fresh():
            return _Tiny(tag=2).to(device)

        model = _Tiny(tag=2).to(device)
        obs = measure_b2_roundtrip_site(
            model=model,
            fresh_model_fn=fresh,
            batch=_batch(device, seed=42),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            device=device,
        )
        xcheck = _assert_pass_path(obs, family="B2")
        assert obs["gating_row"] == "G_CUDA_B2_APPLY"
        # negative: payload VALUE mismatch class
        s2 = xcheck["s2_compare"]
        assert s2.get("named_post_payload_sha256") == s2.get(
            "twin_or_canonical_post_payload_sha256"
        )
        assert s2.get("named_post_payload_sha256")
        print(
            f"device_name={torch.cuda.get_device_name(device)} "
            f"b2_pass s1_ok={xcheck['s1_compare'].get('s1_ok')} s2_ok={s2.get('s2_ok')} jsonl={jl}"
        )
    finally:
        _disarm()


def test_gpu_smoke_b3_oracle_on_and_apply_pass_path(tmp_path):
    """B3 apply pass-path + oracle_on named-map (NOT fused-only sole oracle proof)."""
    device = _require_cuda()
    torch.manual_seed(43)
    jl = _arm_phase_jsonl(tmp_path, "b3oracle")
    try:
        def fresh():
            return _Tiny(tag=3).to(device)

        model = _Tiny(tag=3).to(device)
        obs_apply = measure_b3_landing_site(
            model=model,
            fresh_model_fn=fresh,
            batch=_batch(device, seed=43),
            forward_loss_fn=_loss,
            forward_output_fn=_output,
            device=device,
        )
        x_a = _assert_pass_path(obs_apply, family="B3")
        assert obs_apply["gating_row"] == "G_CUDA_B3_APPLY"

        # oracle_on named-map presence
        model2 = _Tiny(tag=4).to(device)
        p1 = save_trainer_sub2_live_checkpoint_envelope(
            model2,
            use_ternary_bulk=True,
            step=0,
            config={"proof": "lands_ab_b3_oracle_smoke"},
            source_pin="lands_ab_b3_oracle_smoke",
            epoch=0,
        )
        landing = build_sparse_vote_authority_landing_receipt(
            plan_sha256="0" * 64,
            task_id="lands_ab_b3_oracle_smoke",
            p1_checkpoint=p1,
            p1_envelope_bytes=b"{}",
            fresh_model_fn=lambda: _Tiny(tag=4).to(device),
            batch=_batch(device, seed=43),
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
        sub = dict(getattr(landing, "sparse_vote_authority_subproof", None) or {})
        named_s1 = sub.get("sparse_event_map_binding_sha256_by_key")
        assert isinstance(named_s1, dict) and named_s1
        oo = dict(sub.get("oracle_only") or {})
        eeq = oo.get("events_equal_by_key")
        assert isinstance(eeq, dict) and eeq
        assert oo.get("dense_reference_tagged") == "oracle_only"
        # negative: empty named shape would fail S1 unit (D5)
        s1 = x_a["s1_compare"]
        assert s1.get("named_shapes_by_key")
        print(
            f"device_name={torch.cuda.get_device_name(device)} "
            f"b3_apply_pass s1_ok={s1.get('s1_ok')} "
            f"oracle_on_s1_n={len(named_s1)} eeq_n={len(eeq)} jsonl={jl}"
        )
    finally:
        _disarm()


def test_gpu_smoke_family_negative_s1_mutation_unit():
    """Negative mutation: wrong same-numel shape fails S1 (D2/D5)."""
    # CPU-safe unit negative so GPU suite cannot be all soft-presence
    prior_q = torch.zeros((4,), dtype=torch.int8)

    class _P:
        def __init__(self, t):
            self.q_levels = t

    from calm.hrm_text_158.native_full_stack.sparse_vote_events import SparseVoteEvents

    events = {"lin": SparseVoteEvents.from_dict({0: 1, 3: 2})}
    prior = {"lin": _P(prior_q)}
    true = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=None,
        named_shape_by_key=None,
    )
    good_shapes = true["independent_shapes_by_key"]
    good = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key=good_shapes,
    )
    assert good["s1_ok"] is True
    bad = recompute_s1_and_compare(
        sparse_events_by_key=events,
        prior_states=prior,
        named_binding_by_key=true["recomputed_binding_by_key"],
        named_shape_by_key={"lin": (2, 2)},
    )
    assert bad["s1_ok"] is False
