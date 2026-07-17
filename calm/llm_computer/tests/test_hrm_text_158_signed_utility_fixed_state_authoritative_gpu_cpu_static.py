"""CPU-static DI tests for authoritative_gpu (D2c2/D2c5/D2c6)."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
import json

import pytest
import torch

from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
    ARM_FORK_NAMES, AuthoritativeGpuDeferredError, AuthoritativeGpuError, AuthoritativeGpuHooks,
    CALL_GRAPH_STEPS_V6, FORMAL_PHASE_BUDGETS, SMOKE_PHASE_BUDGETS, call_graph_steps,
    canonical_invert_plans_v4, compute_partition_leakage_compact, isolate_fork_arm_state,
    phase_budgets_for_packet, run_authoritative_gpu_call_graph,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    FORMAL_SOURCE_PIN_BASENAMES, rehash_path,
)
from calm.llm_computer.tests import hrm_text_158_signed_utility_authoritative_gpu_one_step_smoke as smoke

MOD = Path(__file__).resolve().parents[2] / "hrm_text_158/native_full_stack/signed_utility_fixed_state_authoritative_gpu.py"
STACK = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm_text_158/native_full_stack")
WATCH = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/bin/watch-wrap")
VOTE = STACK / "vote_update.py"
@dataclass
class _Plan:
    applied_indices: torch.Tensor
    applied_directions: torch.Tensor
    replay_veto_directions: torch.Tensor
    applied_thresholds: torch.Tensor
    candidate_indices: torch.Tensor
    pre_veto_selected_indices: torch.Tensor
    replay_ce_veto_indices: torch.Tensor
    replay_veto_thresholds: torch.Tensor
    pc_aux_negative_indices: torch.Tensor
    pc_aux_veto_indices: torch.Tensor
    q_i16: torch.Tensor
    new_acc_i32: torch.Tensor
def _plan(n=1, direction=1, *, new_acc=0, threshold=10):
    z = torch.zeros(n, dtype=torch.int64)
    d = torch.tensor([direction] * n, dtype=torch.int16)
    thr = torch.full((n,), int(threshold), dtype=torch.int32)
    empty = z[:0]
    return _Plan(
        applied_indices=torch.tensor([0] * n, dtype=torch.int64), applied_directions=d.clone(),
        replay_veto_directions=torch.zeros(0, dtype=torch.int16), applied_thresholds=thr.clone(),
        candidate_indices=z.clone(),
        pre_veto_selected_indices=z.clone(), replay_ce_veto_indices=empty.clone(),
        replay_veto_thresholds=thr[:0].clone(), pc_aux_negative_indices=empty.clone(),
        pc_aux_veto_indices=empty.clone(), q_i16=torch.zeros(4, dtype=torch.int16),
        new_acc_i32=torch.full((4,), int(new_acc), dtype=torch.int32),
    )
def _clone_state(st):
    return SimpleNamespace(
        q_levels=st.q_levels.clone(), exact_accumulator_shadow=st.exact_accumulator_shadow.clone(),
        frozen_scale=st.frozen_scale.clone(), state_key=st.state_key, vote_update_state=lambda: None,
    )
def _states():
    return {"k0": SimpleNamespace(
        q_levels=torch.zeros(4, dtype=torch.int8), exact_accumulator_shadow=torch.zeros(4, dtype=torch.int16),
        frozen_scale=torch.tensor(1.0), state_key="k0", vote_update_state=lambda: None,
    )}
def _formal_pins():
    pins = {n: {"absolute_path": str(STACK / n), "sha256": rehash_path(STACK / n)} for n in FORMAL_SOURCE_PIN_BASENAMES}
    pins["watch-wrap"] = {"absolute_path": str(WATCH), "sha256": rehash_path(WATCH)}
    pins["vote_update.py"] = {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)}
    return pins
def _packet(**over):
    p = {"authoritative_deferred": False, "pin_mode": "cpu_static_di",
         "parent_checkpoint": {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)},
         "source_pins": _formal_pins()}
    p.update(over)
    return p
def _batch(n, start, *, row_ids=None, prompts=None, targets=None, response_tokens=None):
    meta = {
        "row_ids": row_ids or [f"r{start + i}" for i in range(n)],
        "prompts": prompts or [f"prompt-{start + i}" for i in range(n)],
        "targets": targets or [f"target-{start + i}" for i in range(n)],
        "response_tokens": response_tokens or [[start + i, 1, 2] for i in range(n)],
    }
    return {"batch": {"x": torch.zeros(n, 1)}, "metadata": meta}
def _hooks(route_log: list[str], *, empty_apply=False, leak_surface=None, mutate_base_on_capture=False,
           bad_calib_acc=False, mutate_cal_input=False, weight_hash_mismatch=False):
    base = _states()

    def materialize(_p):
        route_log.append("materialize"); return SimpleNamespace(tensor_states=base, eligible_modules={}, model=None)

    def rebuild(_b):
        route_log.append("rebuild")
        batches = [_batch(32, 0), _batch(32, 100), _batch(26, 200)]
        if leak_surface:
            key = {"row":"row_ids","prompt":"prompts","target":"targets","response":"response_tokens"}[leak_surface]
            batches[1]["metadata"][key][0] = list(batches[0]["metadata"][key][0]) if key=="response_tokens" else batches[0]["metadata"][key][0]
        return batches

    def leakage(batches):
        route_log.append("leakage"); return compute_partition_leakage_compact(batches)

    def fork(_b):
        route_log.append("fork")
        return {k: {sk: _clone_state(sv) for sk, sv in _states().items()} for k in ARM_FORK_NAMES}

    def capture(_b, arms):
        route_log.append("capture")
        assert "capture_disposable" in arms
        arms["capture_disposable"]["k0"].q_levels[0] = 7
        if mutate_base_on_capture:
            arms["base"]["k0"].q_levels[0] = 9
        n = 0 if empty_apply else 1
        plans = {"k0": _plan(n, direction=1)}
        capture_states = {k: _clone_state(v) for k, v in arms["calibration_shadow"].items()}
        if n:
            # Match public residual: new_acc=0, thr=10, d=+1 → q=1, acc=-9
            capture_states["k0"].q_levels = capture_states["k0"].q_levels.clone()
            capture_states["k0"].q_levels[0] = capture_states["k0"].q_levels[0] + 1
            capture_states["k0"].exact_accumulator_shadow = plans["k0"].new_acc_i32.to(torch.int16).clone()
            capture_states["k0"].exact_accumulator_shadow[0] = -9
            if bad_calib_acc:
                capture_states["k0"].exact_accumulator_shadow = capture_states["k0"].exact_accumulator_shadow.clone()
                capture_states["k0"].exact_accumulator_shadow[0] = 3
        return plans, capture_states, 1

    def apply(states, plans):
        route_log.append("apply")
        out = {}
        for k, st in states.items():
            q = st.q_levels.clone().to(torch.int16)
            # Public-law residual: start from plan.new_acc_i32, then residual at applied indices
            plan = plans.get(k)
            if plan is None:
                acc = st.exact_accumulator_shadow.clone().to(torch.int32)
            else:
                acc = plan.new_acc_i32.detach().cpu().to(torch.int32).reshape(-1).clone()
                if int(plan.applied_indices.numel()) > 0:
                    for idx, direction, thr in zip(
                        plan.applied_indices.tolist(), plan.applied_directions.tolist(), plan.applied_thresholds.tolist()
                    ):
                        d, t = int(direction), int(thr)
                        v = int(q[int(idx)]) + d
                        q[int(idx)] = -1 if v < -1 else (1 if v > 1 else v)
                        residual = int(acc[int(idx)]) - d * t
                        lo, hi = -t + 1, t - 1
                        acc[int(idx)] = lo if residual < lo else (hi if residual > hi else residual)
            out[k] = SimpleNamespace(
                q_levels=q.to(torch.int8), exact_accumulator_shadow=acc.to(torch.int16),
                frozen_scale=st.frozen_scale.clone(), state_key=k,
            )
        if mutate_cal_input and states is not None:
            for st in states.values():
                st.q_levels = st.q_levels.clone(); st.q_levels[0] = st.q_levels[0] + 9
        return out, len(states)

    def invert(plans):
        route_log.append("invert"); return canonical_invert_plans_v4(plans)

    def eval_arm(arm, states, bundle, batches):
        route_log.append(f"eval:{arm}")
        wh = ("b" * 64) if (weight_hash_mismatch and arm == "noop_repeat") else ("a" * 64)
        return 2.0, 2, 1.0, wh

    return AuthoritativeGpuHooks(
        materialize=materialize, rebuild_support_batches=rebuild, leakage_report=leakage,
        fork_arm_states=fork, capture_plans=capture, public_apply=apply, invert_plans=invert,
        eval_arm_nll=eval_arm, phase_budgets=dict(SMOKE_PHASE_BUDGETS),
    )
def test_loc_and_steps_and_budgets():
    assert sum(1 for _ in MOD.open()) <= 350
    assert list(call_graph_steps()) == list(CALL_GRAPH_STEPS_V6)
    assert FORMAL_PHASE_BUDGETS["THREE_ARM_EVAL_NLL"] == 480.0
    assert phase_budgets_for_packet({"smoke_mode": True}) == {
        "MATERIALIZE": 120.0, "CAPTURE": 120.0, "CALIBRATE_EVAL": 60.0,
    }
    assert set(SMOKE_PHASE_BUDGETS) == {"MATERIALIZE", "CAPTURE", "CALIBRATE_EVAL"}
    assert phase_budgets_for_packet({})["CAPTURE_BACKWARD_VOTE"] == 180.0
    src = MOD.read_text(encoding="utf-8")
    assert "device_resident_if_cuda_else_cpu_legacy" not in src and "capture_disposable" in src
def test_deferred_true_still_fail_closed():
    with pytest.raises(AuthoritativeGpuDeferredError):
        run_authoritative_gpu_call_graph({"authoritative_deferred": True, "source_pins": _formal_pins()})
def test_ordered_route_science_via_injected_hooks():
    log: list[str] = []
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks(log))
    assert out["schema"].endswith("science_v4"), out
    assert out["route"][0] == "parse_packet_live_rehash_pins" and "emit_in_memory_payload" in out["route"]
    assert log[:4] == ["materialize", "rebuild", "leakage", "fork"]
    assert "capture" in log and "apply" in log and "eval:prod" in log
    assert out["hooks_injected"] is True and out["eval_batch_count"] == 2
    assert out["phase_budgets"]["THREE_ARM_EVAL_NLL"] == 480.0
    assert out["phase_budgets"]["CAPTURE_BACKWARD_VOTE"] == 180.0
def test_failure_ownership_leak_empty_parity_calib():
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], leak_surface="row"))
    assert out["reason"] == "leakage_overlap" and out["compact_diagnostics"]["row_id_overlap"] == 1
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], empty_apply=True))
    assert out["reason"] == "aggregate_applied_count_zero", out
    h = _hooks([])
    def invert(plans):
        return {k: _Plan(applied_indices=torch.tensor([1], dtype=torch.int64),
                         applied_directions=torch.tensor([-1], dtype=torch.int16),
                         replay_veto_directions=torch.tensor([-1], dtype=torch.int16),
                         applied_thresholds=p.applied_thresholds, candidate_indices=p.candidate_indices,
                         pre_veto_selected_indices=p.pre_veto_selected_indices,
                         replay_ce_veto_indices=p.replay_ce_veto_indices,
                         replay_veto_thresholds=p.replay_veto_thresholds,
                         pc_aux_negative_indices=p.pc_aux_negative_indices,
                         pc_aux_veto_indices=p.pc_aux_veto_indices, q_i16=p.q_i16, new_acc_i32=p.new_acc_i32)
                for k, p in plans.items()}
    bad = AuthoritativeGpuHooks(
        materialize=h.materialize, rebuild_support_batches=h.rebuild_support_batches,
        leakage_report=h.leakage_report, fork_arm_states=h.fork_arm_states, capture_plans=h.capture_plans,
        public_apply=h.public_apply, invert_plans=invert, eval_arm_nll=h.eval_arm_nll, phase_budgets=h.phase_budgets)
    out = run_authoritative_gpu_call_graph(_packet(), hooks=bad)
    assert out["classifier"] == "UNVERIFIED_ASYMMETRIC_INTERVENTION", out
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], bad_calib_acc=True))
    assert out["reason"] == "calibration_state_mismatch", out
def test_capture_disposable_isolation_and_base_untouched():
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], mutate_base_on_capture=True))
    assert "untouched_sentinel_drift" in out["reason"]
def test_formal_pins_required_and_schema_validated_payload():
    bad = _packet(); bad["source_pins"] = {
        "watch-wrap": {"absolute_path": str(WATCH), "sha256": rehash_path(WATCH)},
        "vote_update.py": {"absolute_path": str(VOTE), "sha256": rehash_path(VOTE)}}
    out = run_authoritative_gpu_call_graph(bad, hooks=_hooks([]))
    assert out["schema"].endswith("preflight_execution_receipt_v4")
    assert "formal_source_pins_missing" in out["observed"]["error"]
    out = run_authoritative_gpu_call_graph(
        _packet(pin_mode="formal", repo_root="/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"), hooks=_hooks([]))
    assert out["schema"].endswith("preflight_execution_receipt_v4")
    assert "formal_head_or_repo_root_missing" in out["observed"]["error"]
def test_smoke_ok_oexcl_and_d2c5_lossless(tmp_path: Path, capsys):
    pkt = _packet(smoke_mode=True, pin_mode="cpu_static_di"); receipt = tmp_path / "ok.json"
    out = smoke.run_smoke(receipt=receipt, packet=pkt, hooks=_hooks([]))
    assert out["status"] == "SMOKE_OK" and out["compact_is_authoritative"] is False
    ar = out["authoritative_result"]
    assert "capture_backward_vote" in ar["route"] and ar["schema"].endswith("science_v4")
    assert json.loads(receipt.read_text())["authoritative_result"] == ar
    assert smoke.smoke_preflight_cpu()["invokes_authoritative_entry"] is True
    with pytest.raises(FileExistsError):
        smoke.run_smoke(receipt=receipt, packet=pkt, hooks=_hooks([]))
    capsys.readouterr(); fail = tmp_path / "fail.json"
    out = smoke.run_smoke(receipt=fail, packet=pkt, hooks=_hooks([], leak_surface="row"))
    assert out["status"] == "SMOKE_FAIL" and out["authoritative_result"]["reason"] == "leakage_overlap"
    ar = out["authoritative_result"]
    assert ar["route"] and json.loads(fail.read_text())["authoritative_result"] == ar
    logged = capsys.readouterr().out
    assert logged.startswith("SMOKE_BEGIN\nSMOKE_FAIL") and "SMOKE_PHASE_" not in logged
    assert "reason=leakage_overlap" in logged and "failed_stage=partition" in logged
    pre = tmp_path / "pre.json"
    bad = _packet(smoke_mode=True, pin_mode="smoke",
                  repo_root="/mnt/c/Users/gabes/projects/claw-code-hrm-text-158", expected_head="0" * 40)
    out = smoke.run_smoke(receipt=pre, packet=bad, hooks=_hooks([]))
    ar = out["authoritative_result"]
    assert out["status"] == "SMOKE_FAIL" and ar["schema"].endswith("preflight_execution_receipt_v4")
    assert ar.get("route") == [] and json.loads(pre.read_text())["authoritative_result"] == ar
    assert "for step in CALL_GRAPH_STEPS_V6" not in Path(smoke.__file__).read_text(encoding="utf-8")
def test_resolve_capture_device_mode_exact_strings():
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
        resolve_capture_device_mode, AuthoritativeGpuError)
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY, AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT)
    assert resolve_capture_device_mode({}, "cuda:0") == AUTHORITATIVE_CAPTURE_MODE_DEVICE_RESIDENT
    assert resolve_capture_device_mode({"allow_cpu_legacy_eval": True}, "cpu") == AUTHORITATIVE_CAPTURE_MODE_CPU_LEGACY
    with pytest.raises(AuthoritativeGpuError): resolve_capture_device_mode({}, "cpu")
    assert "device_resident_if_cuda_else_cpu_legacy" not in MOD.read_text(encoding="utf-8")
def test_default_smoke_packet_hard_pins_expected_head(monkeypatch):
    src = Path(smoke.__file__).read_text(encoding="utf-8")
    assert "subprocess" not in src and "rev-parse" not in src
    assert smoke.D3A_EXPECTED_HEAD == "c93b68e9ddc3513866adc3f930a17eb80c6f5459"
    assert smoke.build_default_smoke_packet()["expected_head"] == smoke.D3A_EXPECTED_HEAD
    bad = _packet(smoke_mode=True, pin_mode="smoke",
                  repo_root="/mnt/c/Users/gabes/projects/claw-code-hrm-text-158", expected_head="0" * 40)
    out = run_authoritative_gpu_call_graph(bad, hooks=_hooks([]))
    assert out["schema"].endswith("preflight_execution_receipt_v4")
    assert "head_mismatch" in out["observed"]["error"] or "head_ne_upstream" in out["observed"]["error"]
def test_d2c4_calibration_and_weight_hash_integrity():
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], mutate_cal_input=True))
    assert "untouched_sentinel_drift" in out.get("reason", "") and not out["schema"].endswith("science_v4")
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([], weight_hash_mismatch=True))
    assert out.get("reason") == "noop_repeat_weight_hash_mismatch" and not out["schema"].endswith("science_v4")
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks([]))
    cal = out["observer_public_apply_calibration"]
    assert cal["calibration_input_unchanged"] is True
    assert cal["calibration_input_hash_pre"] == cal["calibration_input_hash_post"]
def test_d2c6_storage_isolation_red_green_and_routing():
    import hashlib, json
    from types import SimpleNamespace
    from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
        BoundedDeltaTensorState, make_bounded_tensor_state, make_live_shadow_tensor_state)
    from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_arm_ops import clone_f_in_memory
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_integrity_proofs import (
        IntegrityProofError, assert_zero_cross_arm_storage_overlap, enumerate_mutable_storage_spans,
        non_tensor_authority_manifest)
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import mutation_parity_report
    def value_manifest(st):
        parts = [json.dumps(non_tensor_authority_manifest(st), sort_keys=True)]
        for t in (st.q_levels, st.frozen_scale, st.exact_accumulator_shadow):
            parts.append(hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest())
        return hashlib.sha256("|".join(parts).encode()).hexdigest()
    q = torch.zeros(4, dtype=torch.int8); acc = torch.arange(4, dtype=torch.int16)
    fresh = make_bounded_tensor_state("k0", q, 0.25, acc)
    stale = BoundedDeltaTensorState(
        state_key=fresh.state_key, q_levels=fresh.q_levels.clone(), frozen_scale=fresh.frozen_scale.clone(),
        bounded_accumulator=fresh.bounded_accumulator, exact_accumulator_shadow=fresh.exact_accumulator_shadow.clone(),
        bounded_accumulator_fresh_for_exact_shadow=False,
        bounded_accumulator_rebuild_hot_exact_indices=(), bounded_accumulator_rebuild_cold_default_value=0)
    rebuilt = clone_f_in_memory(stale).with_fresh_bounded_accumulator()
    assert rebuilt.bounded_accumulator_fresh_for_exact_shadow is True
    assert non_tensor_authority_manifest(rebuilt) != non_tensor_authority_manifest(stale)
    with pytest.raises(AuthoritativeGpuError, match="unsupported_event_coded_live_carrier"):
        isolate_fork_arm_state(SimpleNamespace(event_coded_live_carrier=object(), exact_accumulator_shadow=acc))
    arms0 = {n: {"k0": isolate_fork_arm_state(stale)} for n in ARM_FORK_NAMES}
    vm0 = value_manifest(stale)
    for n, m in arms0.items():
        s = m["k0"]
        assert s.bounded_accumulator_fresh_for_exact_shadow is False
        assert s.bounded_accumulator_rebuild_hot_exact_indices == () and s.bounded_accumulator_rebuild_cold_default_value == 0
        assert s.bounded_accumulator == stale.bounded_accumulator and s.bounded_accumulator is not stale.bounded_accumulator
        assert non_tensor_authority_manifest(s) == non_tensor_authority_manifest(stale) and value_manifest(s) == vm0
        assert s.event_coded_live_carrier is None
        assert torch.equal(s.q_levels, stale.q_levels) and float(s.frozen_scale) == float(stale.frozen_scale)
    assert arms0["prod"]["k0"].bounded_accumulator is not arms0["inv"]["k0"].bounded_accumulator
    assert arms0["capture_disposable"]["k0"].bounded_accumulator is not arms0["calibration_shadow"]["k0"].bounded_accumulator
    mut = {k: arms0[k] for k in ("capture_disposable", "calibration_shadow", "prod", "inv", "noop")}
    assert_zero_cross_arm_storage_overlap(mut, base=arms0["base"])
    prod_bad = {"k0": make_live_shadow_tensor_state(stale, torch.ones(4, dtype=torch.int8), acc.clone())}
    inv_bad = {"k0": make_live_shadow_tensor_state(stale, torch.full((4,), -1, dtype=torch.int8), acc.clone())}
    with pytest.raises(IntegrityProofError, match="cross_arm_storage_overlap"):
        assert_zero_cross_arm_storage_overlap({"prod": prod_bad, "inv": inv_bad}, base={"k0": stale})
    p0, i0 = arms0["prod"]["k0"], arms0["inv"]["k0"]
    prod_ok = {"k0": make_live_shadow_tensor_state(p0, torch.ones(4, dtype=torch.int8), acc.clone())}
    inv_ok = {"k0": make_live_shadow_tensor_state(i0, torch.full((4,), -1, dtype=torch.int8), acc.clone())}
    assert_zero_cross_arm_storage_overlap({"prod": prod_ok, "inv": inv_ok}, base=arms0["base"])
    assert mutation_parity_report(arms0["base"], prod_ok, inv_ok)["pass"] is True
    src = MOD.read_text(encoding="utf-8")
    assert "isolate_fork_arm_state" in src and "with_fresh_bounded_accumulator" not in src
    assert "unsupported_event_coded_live_carrier_in_signed_utility_fork" in src
    seen, arm_ids, bh = [], {}, _hooks([])
    def fork(b):
        arms = bh.fork_arm_states(b)
        for n in ("base", "prod", "inv"): arm_ids[n] = id(arms[n])
        return arms
    def apply(states, plans):
        seen.append(id(states)); return bh.public_apply(states, plans)
    h = AuthoritativeGpuHooks(
        materialize=bh.materialize, rebuild_support_batches=bh.rebuild_support_batches,
        leakage_report=bh.leakage_report, fork_arm_states=fork, capture_plans=bh.capture_plans,
        public_apply=apply, invert_plans=bh.invert_plans, eval_arm_nll=bh.eval_arm_nll,
        phase_budgets=dict(SMOKE_PHASE_BUDGETS))
    out = run_authoritative_gpu_call_graph(_packet(), hooks=h)
    assert out["schema"].endswith("science_v4") and arm_ids["prod"] in seen and arm_ids["base"] not in seen
    assert sum(1 for _ in MOD.open()) <= 350

def test_d2c8_legal_subset_wired_and_claude_geometry_red():
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import ESTIMAND_NAME
    from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_reducers import mutation_parity_report
    route = []
    out = run_authoritative_gpu_call_graph(_packet(), hooks=_hooks(route))
    assert out["schema"].endswith("science_v4")
    assert out["estimand"] == ESTIMAND_NAME
    assert "legal_subset_filter" in out["route"]
    assert out["legal_subset"]["support_floors"]["pass"] is True
    assert out["mutation_parity"]["pass"] is True
    assert out["mutation_parity"]["frozen_scale"]["pass"] is True
    assert "changed_prod" not in out["mutation_parity"]["q_levels"]["per_key"]["k0"]
    # Unfiltered Claude geometry: q parity can hold while exact-acc fails
    base = _states()
    plan = _plan(1, direction=1, new_acc=10, threshold=10)
    prod, _ = _hooks([]).public_apply(base, {"k0": plan})
    inv, _ = _hooks([]).public_apply(_states(), {"k0": canonical_invert_plans_v4({"k0": plan})["k0"]})
    assert mutation_parity_report(base, prod, inv)["pass"] is False
    # Filtered path: Claude new_acc=10 drops all retained → support_degenerate
    base_hooks = _hooks([])

    def capture_claude(_b, arms):
        arms["capture_disposable"]["k0"].q_levels[0] = 7
        plans = {"k0": _plan(1, direction=1, new_acc=10, threshold=10)}
        capture_states = {k: _clone_state(v) for k, v in arms["calibration_shadow"].items()}
        capture_states["k0"].q_levels = capture_states["k0"].q_levels.clone()
        capture_states["k0"].q_levels[0] = 1
        # public apply: acc := new_acc (10), residual at idx0 → 0
        capture_states["k0"].exact_accumulator_shadow = torch.full((4,), 10, dtype=torch.int16)
        capture_states["k0"].exact_accumulator_shadow[0] = 0
        return plans, capture_states, 1

    h = AuthoritativeGpuHooks(
        materialize=base_hooks.materialize, rebuild_support_batches=base_hooks.rebuild_support_batches,
        leakage_report=base_hooks.leakage_report, fork_arm_states=base_hooks.fork_arm_states,
        capture_plans=capture_claude, public_apply=base_hooks.public_apply,
        invert_plans=base_hooks.invert_plans, eval_arm_nll=base_hooks.eval_arm_nll,
        phase_budgets=dict(SMOKE_PHASE_BUDGETS),
    )
    bad = run_authoritative_gpu_call_graph(_packet(), hooks=h)
    assert "legal_subset_support_degenerate" in bad.get("reason", ""), bad
    assert not bad["schema"].endswith("science_v4")
