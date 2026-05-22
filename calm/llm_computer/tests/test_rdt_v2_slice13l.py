"""Slice 13l --save-at-step mid-training ckpt hook test (HRM branch).

Per codex audit msg 1779444785341-6ac9c321 scope guard 3:
"The test needs to hit the HRM branch, not only the aux/final-NLL
branches. Minimal acceptable: monkeypatch tiny GSM8k rows or
equivalent harness, run train(... use_hrm_act=True, use_halt_head=True,
use_carry=True, m_max=2, save_at_step=1 ...), assert the step file
exists and can be loaded/rebuilt with the relevant config flags
preserved."

This test runs an actual `train()` call with HRM segment-loop active
+ save_at_step=1, then verifies the resulting `_step{N:05d}.pt` file
exists and contains the expected config keys + flag values.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch


def _make_tiny_gsm8k_rows(n: int = 8) -> list[dict]:
    """Tiny GSM8k-shaped rows (question + integer expected) — minimum
    needed for tokenizer corpus + dataset construction."""
    return [
        {"question": f"What is {2*i+1} plus {2*i+1}?", "expected": 4*i + 2}
        for i in range(n)
    ]


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="HRM segment loop runs on CUDA only")
def test_save_at_step_hrm_branch_creates_step_ckpt(tmp_path, monkeypatch):
    """Run a 1-step training pass with HRM ACT + save_at_step=1 and
    confirm the step ckpt file is created with all required config
    fields. Hits the HRM segment-loop branch (NOT vanilla NLL path)."""

    sys.path.insert(0, ".")
    # Import inside the test so monkeypatch can land before module-level
    # side effects fire.
    from scripts import train_dt_gsm8k as trainer

    # Monkeypatch load_gsm8k_splits to return tiny rows in-memory rather
    # than hitting HuggingFace datasets-server.
    tiny_rows = _make_tiny_gsm8k_rows(16)
    monkeypatch.setattr(
        trainer, "load_gsm8k_splits",
        lambda val_frac=0.10: (tiny_rows, tiny_rows[:4], tiny_rows[:2]),
    )

    ckpt_root = tmp_path / "test_13l_save_at_step_best.pt"

    trainer.train(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=32, n_copy_heads=2,
        seed=42, eval_every=1, eval_cap=2,
        checkpoint_path=str(ckpt_root),
        use_chunkwise=True, n_iterations=2,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        h_cycles=2, use_h_rmsnorm=True, use_short_conv=True,
        use_h_layer_stack=True,
        use_halt_head=True, use_carry=True,
        use_pre_rmsnorm=True,
        use_hrm_act=True, m_max=2, m_min_epsilon=0.5,
        save_at_step=1,
    )

    step_path = ckpt_root.with_name(
        ckpt_root.stem + "_step00001.pt"
    )
    assert step_path.exists(), (
        f"--save-at-step=1 did not produce {step_path}; "
        f"directory contents: {list(tmp_path.iterdir())}"
    )

    ckpt = torch.load(step_path, map_location="cpu", weights_only=False)

    # Required fields
    assert "model_state" in ckpt
    assert "config" in ckpt
    assert "epoch" in ckpt
    assert "step" in ckpt
    assert ckpt["epoch"] == 1
    assert ckpt["step"] == 1

    cfg = ckpt["config"]
    # Config fields that prove the HRM-ACT + scaffold path was active:
    assert cfg["use_halt_head"] is True, (
        "use_halt_head must be in step ckpt config (proves Q-head "
        "path was active during training)"
    )
    assert cfg["use_carry"] is True, (
        "use_carry must be in step ckpt config (proves H/L carry "
        "semantics active)"
    )
    assert cfg["h_cycles"] == 2
    assert cfg["n_iterations"] == 2
    # Slice 13k flag fields preserved (would be useful for ckpts taken
    # during softmax-only / hybrid arms)
    assert "use_softmax_attn" in cfg
    assert "use_softmax_only" in cfg
    assert "use_prefix_lm" in cfg


def test_save_at_step_off_by_default_no_extra_ckpts(tmp_path, monkeypatch):
    """When --save-at-step is not set (default None), no step ckpt
    files should be created. Defensive: catches accidental always-on
    saves that would slow training + pollute the ckpt dir."""
    pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("HRM segment loop runs on CUDA only")

    sys.path.insert(0, ".")
    from scripts import train_dt_gsm8k as trainer

    tiny_rows = _make_tiny_gsm8k_rows(16)
    monkeypatch.setattr(
        trainer, "load_gsm8k_splits",
        lambda val_frac=0.10: (tiny_rows, tiny_rows[:4], tiny_rows[:2]),
    )

    ckpt_root = tmp_path / "test_13l_no_step_save_best.pt"

    trainer.train(
        epochs=1,
        batch_size=4,
        lr=1e-3,
        d_model=8, n_heads=4, n_layers=2, d_ffn=16,
        max_len=32, n_copy_heads=2,
        seed=42, eval_every=1, eval_cap=2,
        checkpoint_path=str(ckpt_root),
        use_chunkwise=True, n_iterations=2,
        use_loop_index=True, use_input_injection=True,
        use_gated_attention=True, use_z_init=True, use_lecun_init=True,
        h_cycles=2, use_h_rmsnorm=True, use_short_conv=True,
        use_h_layer_stack=True,
        use_halt_head=True, use_carry=True,
        use_pre_rmsnorm=True,
        use_hrm_act=True, m_max=2, m_min_epsilon=0.5,
        # save_at_step omitted — should default to None (no step save)
    )

    # No <stem>_step*.pt files should exist (glob on stem prefix)
    step_files = list(tmp_path.glob(f"{ckpt_root.stem}_step*.pt"))
    assert step_files == [], (
        f"save_at_step=None should produce no step ckpts; "
        f"found {step_files}"
    )
