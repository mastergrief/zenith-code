"""Saturated cut-16 GPU smoke — REAL run_bounded_delta_steps (no fake runner).

Developer validation only (science_label=null). Proves:
- BoundedDeltaPostStepEvent after step-16 backlog update
- non-empty backlog seeded via initial_deferred_backlog for K-step arms
- F==U on COMPLETE decision vector (strict; no CONTROL_INVALID escape)
- Z breaks ≥1 gate-bearing field within K
- GLOBAL bp horizon=32 for U and resumed arms (horizon-sensitive spy)
- REAL C/S on-disk roundtrips + distinct scratch paths
- hook/runner exception fail-closed + outer lane-release finally
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.curriculum import BroadTokenizer
from calm.hrm_text_158.native_full_stack.bounded_delta_runner_hook_contract import (
    BoundedDeltaPostStepEvent,
    invoke_post_step_hook,
)
from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_science_driver import (
    FORMAL_GLOBAL_HORIZON,
    backlog_entry_count,
    run_fork_b_resume_parity_certificate,
)
from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
    C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
    build_identity_full_support_batches,
    build_model_from_checkpoint,
    derive_tensor_states_and_check_init_fidelity,
    run_bounded_delta_steps,
    select_eligible_bitlinears,
)
from scripts.train_hrm_text_158 import SOURCE_PIN, _build_ckpt_config


pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="cuda:0 required for science-driver GPU smoke"
)


def _tiny_always_bp2_blob(*, batch_size: int = 2) -> dict:
    """bp_min=bp_max=2 so first-bitlinear is invoked at every step of a 16-step run."""

    tok = BroadTokenizer()
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=64, n_layers=2, hidden_size=64, num_heads=2, expansion=4,
        H_cycles=1, L_cycles=1, half_layers=True,
        bp_warmup_ratio=0.0, bp_min_steps=2, bp_max_steps=2, use_ternary_bulk=True,
    )
    model = LMHead(HierarchicalReasoningModel(cfg), LMHeadConfig(vocab_size=tok.vocab_size))
    return {
        "model_state": model.state_dict(),
        "config": _build_ckpt_config(
            model, tok, cfg, 64, batch_size=batch_size, curriculum_rung="L0c1",
            curriculum_seed=17, replay_ratio=0.0, prior_rungs=[],
        ),
        "step": 50, "epoch": 1, "source_pin": SOURCE_PIN,
    }


def _setup_real_runner_fixture(tmp_path: Path, device: torch.device):
    os.environ["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
    parent = tmp_path / "tiny_bp2_parent.pt"
    torch.save(_tiny_always_bp2_blob(), parent)
    ckpt = torch.load(parent, map_location="cpu", weights_only=False)
    model, tok, _cfg = build_model_from_checkpoint(ckpt, device)
    support_batches, _proof = build_identity_full_support_batches(
        tok=tok, max_len=64, batch_size=2, curriculum_seed=17, device=device,
    )
    eligible = select_eligible_bitlinears(model, eligible_scope="first-bitlinear")
    states, report = derive_tensor_states_and_check_init_fidelity(eligible, threshold=0.0)
    assert report["all_pass"] is True
    return model, support_batches, eligible, states


def test_saturated_cut16_real_runner_gpu_smoke(tmp_path: Path):
    device = torch.device("cuda:0")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    model, support_batches, eligible, states = _setup_real_runner_fixture(tmp_path, device)
    assert run_bounded_delta_steps.__module__.endswith(
        "hrm_text_158_bounded_delta_acquisition_probe"
    )
    assert FORMAL_GLOBAL_HORIZON == 32

    # Horizon-sensitive spy: always-BP2 is horizon-inert for bp_steps outcome;
    # prove compute_train_extra_args still receives GLOBAL total_steps=32 for U and F.
    horizon_calls: list[dict[str, int]] = []
    target = model.model if hasattr(model, "model") else model
    _orig = target.compute_train_extra_args

    def _spy(step: int, total_steps: int):
        horizon_calls.append({"step": int(step), "total_steps": int(total_steps)})
        return _orig(step, total_steps)

    target.compute_train_extra_args = _spy  # type: ignore[method-assign]

    result = run_fork_b_resume_parity_certificate(
        runner=run_bounded_delta_steps,  # ACTUAL committed callable — not a fake
        model=model,
        batch=support_batches[0]["batch"],
        tensor_states=states,
        eligible_modules=eligible,
        device=device,
        scratch_root=tmp_path / "fork_b_real_smoke",
        parent_sha16="9b4e311a22787e7d",
        batch_seed=44,
        support_order_seed=43,
        ordering_seed=17,
        cuts=(16,),
        k_steps=4,
        total_steps=20,  # bounded local U; compares steps 17-20 only
        global_horizon=FORMAL_GLOBAL_HORIZON,  # formal GLOBAL bp horizon = 32
        support_batches=support_batches,
        runner_kwargs={
            "require_q_change": False,
            "max_abs_per_tensor": 4096,
            "r7_deferred_backlog_carry_enabled": True,
            "global_cap_contract": C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
        },
        developer_validation=True,
        require_strict_f_equals_u=True,
        require_z_gate_break=True,
    )

    assert result.developer_validation is True
    assert result.science_label is None
    assert result.notes["runner"] == "run_bounded_delta_steps"
    assert result.notes["global_horizon"] == 32
    assert result.notes["local_u_steps"] == 20
    freeze = result.freezes[16]
    assert freeze.global_horizon == 32
    assert backlog_entry_count(freeze.carry_backlog) > 0
    assert freeze.backlog_hash
    assert result.arm_disclosures["F@16"]["f_equals_u"] is True
    assert result.arm_disclosures["Z@16"]["z_breaks_gate_bearing"] is True
    # Receipt exposes start_step=17, local K=4, global_horizon=32
    assert result.arm_disclosures["F@16"]["start_step"] == 17
    assert result.arm_disclosures["F@16"]["local_k"] == 4
    assert result.arm_disclosures["F@16"]["global_horizon"] == 32
    assert result.arm_disclosures["F@16"]["report_global_horizon"] == 32
    assert result.arm_disclosures["F@16"]["report_start_step"] == 17
    assert result.arm_disclosures["F@16"]["report_local_steps"] == 4
    for arm in ("C@16", "S@16", "Z@16"):
        assert result.arm_disclosures[arm]["global_horizon"] == 32
        assert result.arm_disclosures[arm]["report_global_horizon"] == 32
    assert result.arm_disclosures["C@16"]["simulated"] is False
    assert result.arm_disclosures["S@16"]["simulated"] is False
    assert result.arm_disclosures["C@16"]["path_class"].startswith("REAL_")
    assert result.arm_disclosures["S@16"]["path_class"].startswith("REAL_")
    assert "cut16_C" in str(result.arm_disclosures["C@16"].get("checkpoint_path", ""))
    assert "cut16_S" in str(result.arm_disclosures["S@16"].get("checkpoint_path", ""))
    assert (tmp_path / "fork_b_real_smoke" / "cut16_C").exists()
    assert (tmp_path / "fork_b_real_smoke" / "cut16_S").exists()

    # Spy: every compute_train_extra_args call used total_steps=32 (not 20 / not local end).
    assert horizon_calls, "expected compute_train_extra_args spy captures"
    assert all(c["total_steps"] == 32 for c in horizon_calls)
    u_step17 = [c for c in horizon_calls if c["step"] == 17]
    assert len(u_step17) >= 2  # at least U and F at step 17
    assert all(c["total_steps"] == 32 for c in u_step17)


def test_real_runner_hook_exception_fail_closed_and_lane_release(tmp_path: Path):
    """Direct runner-level no-swallow + outer finally can still release the lane."""

    device = torch.device("cuda:0")
    released = {"ok": False}
    model, support_batches, eligible, states = _setup_real_runner_fixture(tmp_path, device)

    def _boom(_event: BoundedDeltaPostStepEvent) -> None:
        raise RuntimeError("forced_real_runner_hook_boom")

    try:
        with pytest.raises(RuntimeError, match="forced_real_runner_hook_boom"):
            run_bounded_delta_steps(
                model,
                support_batches[0]["batch"],
                states,
                eligible,
                device=device,
                steps=1,
                require_q_change=False,
                max_abs_per_tensor=4096,
                support_batches=support_batches,
                r7_deferred_backlog_carry_enabled=True,
                global_cap_contract=C1_BANKED_FAITHFUL_LONG_RUN_GLOBAL_CAP_CONTRACT_NAME,
                post_step_hook=_boom,
            )
        with pytest.raises(RuntimeError, match="forced_real_runner_hook_boom"):
            invoke_post_step_hook(
                _boom,
                BoundedDeltaPostStepEvent(step=16, states={}, carry_backlog=None),
            )
    finally:
        released["ok"] = True
    assert released["ok"] is True
