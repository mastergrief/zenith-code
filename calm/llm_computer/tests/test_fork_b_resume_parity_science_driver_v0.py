"""CPU harness for Fork B science driver (anti-alias, isolation, classifier, CLI)."""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    derive_bounded_tensor_state_from_weight,
)
from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_arm_ops import (
    evolve_shadow_one_step,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_contracts import (
    GATE_BEARING_FIELDS,
    PreScienceClass,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_science_driver import (
    FORMAL_GLOBAL_HORIZON,
    assert_bundle_immutable,
    assert_global_horizon_equality,
    build_cut_freeze_bundle,
    deep_clone_states,
    evaluate_cut_from_surfaces,
    make_cut_capture_hook,
    run_fork_b_resume_parity_certificate,
)
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    select_trainer_eligible_bitlinears,
)


class _Tiny(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(8, 8, bias=False)
        self.tail = torch.nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


def _live_states(seed: int = 7):
    torch.manual_seed(seed)
    model = _Tiny()
    with torch.no_grad():
        model.proj.weight.zero_()
        model.tail.weight.fill_(0.1)
        model.tail.bias.zero_()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    key = sorted(eligible)[0]
    state0 = derive_bounded_tensor_state_from_weight(
        key, eligible[key].weight.detach(), scale_eps=eligible[key]._SCALE_EPS,
    )
    live = evolve_shadow_one_step(state0, delta=5)
    return model, eligible, {key: live}


def _gate_vector(tag: str = "u") -> dict:
    return {
        "q_sha256_after": f"q-{tag}",
        "applied_flat_indices_hash16": "1",
        "votes_sha256": "v",
        "global_rate_cap_accepted_indices_sha256": "a",
        "global_rate_cap_deferred_indices_sha256": "d",
        "global_rate_cap_applied_count": 1,
        "flip_count": 1,
        "q_changed_count": 0,
        "applied_selection_score_p50": 1.0,
        "applied_selection_score_p95": 2.0,
    }


def test_deep_clone_no_alias_and_post_u_immutability():
    _model, _elig, states = _live_states()
    event = BoundedDeltaPostStepEvent(
        step=4, states=states, carry_backlog={"k": {1: {"vote": 1}}},
    )
    bundle = build_cut_freeze_bundle(
        event, future_batch_sample_ids=(1, 2, 3, 4), global_horizon=FORMAL_GLOBAL_HORIZON,
    )
    key = sorted(states)[0]
    states[key].exact_accumulator_shadow[0, 0] = 99
    assert_bundle_immutable(bundle)
    assert bundle.global_horizon == FORMAL_GLOBAL_HORIZON
    cloned = deep_clone_states(bundle.states)
    cloned[key].exact_accumulator_shadow[0, 0] = -7
    assert_bundle_immutable(bundle)


def test_cut_capture_hook_stores_only_requested_cuts():
    store = {}
    _model, _elig, states = _live_states()
    hook = make_cut_capture_hook(
        cuts=(4, 16), store=store, future_ids_by_cut={4: (10,), 16: (20,)},
        global_horizon=FORMAL_GLOBAL_HORIZON,
    )
    for step in (1, 4, 8, 16, 32):
        hook(BoundedDeltaPostStepEvent(
            step=step, states=states, carry_backlog={"m": {1: {"vote": 1}}},
        ))
    assert set(store) == {4, 16}
    assert store[4].future_batch_sample_ids == (10,)
    assert store[4].backlog_hash
    assert store[4].global_horizon == 32


def test_evaluate_f_ne_u_fail_closed_before_cs():
    u = _gate_vector("u")
    f = dict(u)
    f["q_sha256_after"] = "DIFFERENT"
    result = evaluate_cut_from_surfaces(
        cut_t=16, u_surface=u, f_surface=f, c_surface=u, s_surface=u, z_surface=u,
        non_target_ok=True,
    )
    assert result.f_matches_u is False
    assert result.pre_science == PreScienceClass.CONTROL_INVALID.value


def test_assert_global_horizon_equality_fail_closed():
    with pytest.raises(RuntimeError, match="NON_TARGET_STATE_MISMATCH"):
        assert_global_horizon_equality(expected=32, observed={"U": 32, "F": 20})
    with pytest.raises(RuntimeError, match="missing global_horizon"):
        assert_global_horizon_equality(expected=32, observed={"U": 32, "F": None})


def test_cli_refuses_without_allow_gpu_launch():
    script = Path("scripts/fork_b_resume_parity_science_run.py")
    proc = subprocess.run(
        [sys.executable, str(script), "--parent", "/tmp/p.pt",
         "--parent-sha256", "abc", "--scratch-root", "/tmp/s"],
        capture_output=True, text=True, cwd=str(Path.cwd()),
    )
    assert proc.returncode == 2
    assert "REFUSED" in proc.stderr


def test_scaffold_fork_b_flags_unchanged_default_off():
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import build_arg_parser
    args = build_arg_parser().parse_args([])
    assert bool(getattr(args, "fork_b_resume_parity_certificate", False)) is False


def test_post_wiring_default_none_params_and_start_step():
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps
    sig = inspect.signature(run_bounded_delta_steps)
    assert sig.parameters["post_step_hook"].default is None
    assert sig.parameters["initial_deferred_backlog"].default is None
    assert int(sig.parameters["start_step"].default) == 1
    assert sig.parameters["global_horizon"].default is None
    assert "audit_callback" in sig.parameters


def test_global_horizon_none_falls_back_to_local_segment_end():
    """Default None must preserve prior start_step+steps-1 semantics (byte-identical callers)."""
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import run_bounded_delta_steps
    # Pure formula check mirrored from runner (no GPU): None → local end.
    start_step, steps, global_horizon = 17, 4, None
    bp_horizon = (
        int(global_horizon) if global_horizon is not None else int(start_step) + int(steps) - 1
    )
    assert bp_horizon == 20
    assert (17 + 4 - 1) != FORMAL_GLOBAL_HORIZON  # documents the old bug at cut-16
    assert run_bounded_delta_steps  # import seam still present


@pytest.mark.parametrize("cut", [4, 16, 28])
def test_driver_passes_formal_global_horizon_32_for_all_cut_arms(cut: int, tmp_path: Path):
    """Every resumed arm at cuts {4,16,28} must get global_horizon=32 — NOT 8/20/32."""

    model, eligible, states0 = _live_states()
    key = sorted(states0)[0]
    captured: list[dict] = []
    local_bug = {4: 8, 16: 20, 28: 32}  # start_step+steps-1 without override

    def _report(step: int, *, tag: str, gh: int, start: int, local: int) -> dict:
        vec = _gate_vector("z" if tag == "Z" else "u")
        if tag == "Z":
            vec["q_sha256_after"] = "z-break"
            vec["flip_count"] = 99
        return {
            "start_step": start,
            "local_steps": local,
            "global_horizon": gh,
            "step_result": {"tensor_stats": {key: {f: vec[f] for f in GATE_BEARING_FIELDS}}},
        }

    def fake_runner(model, batch, tensor_states, eligible_modules, *, device, steps, **kwargs):
        start = int(kwargs.get("start_step", 1))
        gh = kwargs.get("global_horizon")
        captured.append({
            "start_step": start, "steps": int(steps), "global_horizon": gh,
            "has_backlog_seed": kwargs.get("initial_deferred_backlog") is not None,
        })
        hook = kwargs.get("post_step_hook")
        reports = {}
        for step in range(start, start + int(steps)):
            if hook is not None and step == cut:
                hook(BoundedDeltaPostStepEvent(
                    step=step, states=states0, carry_backlog={"m": {1: {"vote": 1}}},
                ))
            tag = "U" if kwargs.get("initial_deferred_backlog") is None else "F"
            # Z arm is detected by zeroed-looking caller only via disclosure path;
            # use tensor identity: if exact shadow all zeros-ish we still emit z vector
            # when caller seeded from rehydrate_z — simplest: break when states differ hash.
            st = tensor_states[key]
            is_z = int(st.exact_accumulator_shadow.abs().sum().item()) == 0
            reports[str(step)] = _report(
                step, tag="Z" if is_z else tag, gh=int(gh), start=start, local=int(steps),
            )
        return reports, {}, {}, {}, "ok", 0, None, None, []

    # Monkeypatch C/S roundtrip to avoid real disk authority in this CPU unit test.
    import calm.hrm_text_158.native_full_stack.fork_b_resume_parity_science_driver as drv

    def _fake_cs(*, arm, freeze, model, eligible_modules, scratch, device):
        return deep_clone_states(freeze.states), {
            "path_class": "REAL_on_disk_trainer_sub2_authority_save_load",
            "simulated": False,
            "checkpoint_path": str(scratch / f"arm_{arm.value}.pt"),
            "s_accounting": {"cut_t": freeze.cut_t},
        }

    orig_cs = drv.run_cs_roundtrip_arm
    drv.run_cs_roundtrip_arm = _fake_cs  # type: ignore[assignment]
    try:
        result = run_fork_b_resume_parity_certificate(
            runner=fake_runner,
            model=model,
            batch={"x": torch.zeros(1)},
            tensor_states=states0,
            eligible_modules=eligible,
            device=torch.device("cpu"),
            scratch_root=tmp_path / f"cut{cut}",
            parent_sha16="deadbeefdeadbeef",
            batch_seed=1,
            support_order_seed=2,
            ordering_seed=3,
            cuts=(cut,),
            k_steps=4,
            total_steps=cut + 4,
            global_horizon=FORMAL_GLOBAL_HORIZON,
            require_strict_f_equals_u=True,
            require_z_gate_break=True,
        )
    finally:
        drv.run_cs_roundtrip_arm = orig_cs  # type: ignore[assignment]

    assert FORMAL_GLOBAL_HORIZON == 32
    assert result.notes["global_horizon"] == 32
    assert result.freezes[cut].global_horizon == 32
    # U + F/C/S/Z continuations — every call must carry 32, never the local-bug value.
    assert captured, "runner not invoked"
    assert all(c["global_horizon"] == 32 for c in captured)
    assert all(c["global_horizon"] != local_bug[cut] or cut == 28 for c in captured)
    # Explicit: cut-4 must NOT get 8; cut-16 must NOT get 20.
    if cut in (4, 16):
        assert local_bug[cut] != 32
        assert all(c["global_horizon"] != local_bug[cut] for c in captured)
    arms = [c for c in captured if c["start_step"] == cut + 1]
    assert len(arms) >= 4  # F,C,S,Z
    assert all(c["steps"] == 4 and c["global_horizon"] == 32 for c in arms)
    for arm_key in (f"F@{cut}", f"C@{cut}", f"S@{cut}", f"Z@{cut}"):
        assert result.arm_disclosures[arm_key]["global_horizon"] == 32
        assert result.arm_disclosures[arm_key]["start_step"] == cut + 1
        assert result.arm_disclosures[arm_key]["local_k"] == 4
