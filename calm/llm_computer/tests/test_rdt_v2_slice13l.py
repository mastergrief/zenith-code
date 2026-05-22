"""Slice 13l --save-at-step mid-training ckpt hook tests (HRM branch).

Per codex audit msg 1779444785341-6ac9c321 scope guard 3 (HRM branch
must be hit) + msg 1779446584981 (repeatable --save-at-step spec:
single `save_at_steps` kwarg, frozenset dedupe, positive-int validation).

Tests:
- single-step save still works via list ([1])
- multi-step save creates BOTH ckpts in one trajectory ([1, 2])
- duplicates in input list dedupe (don't crash)
- save_at_steps=None creates no step ckpts
- non-positive ints raise ValueError at train() entry
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


def _common_train_kwargs(ckpt_root):
    """Shared kwargs for HRM-branch trainer invocations — tiny model
    + tiny eval + HRM ACT + halt_head + carry + Pre-RMSNorm."""
    return dict(
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
    )


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="HRM segment loop runs on CUDA only")
def test_save_at_step_hrm_branch_creates_step_ckpt(tmp_path, monkeypatch):
    """Run a 1-step training pass with HRM ACT + save_at_steps=[1] and
    confirm the step ckpt file is created with all required config
    fields. Hits the HRM segment-loop branch (NOT vanilla NLL path)."""

    sys.path.insert(0, ".")
    from scripts import train_dt_gsm8k as trainer

    tiny_rows = _make_tiny_gsm8k_rows(16)
    monkeypatch.setattr(
        trainer, "load_gsm8k_splits",
        lambda val_frac=0.10: (tiny_rows, tiny_rows[:4], tiny_rows[:2]),
    )

    ckpt_root = tmp_path / "test_13l_save_at_step_best.pt"

    trainer.train(
        **_common_train_kwargs(ckpt_root),
        save_at_steps=[1],
    )

    step_path = ckpt_root.with_name(ckpt_root.stem + "_step00001.pt")
    assert step_path.exists(), (
        f"--save-at-step=[1] did not produce {step_path}; "
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


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="HRM segment loop runs on CUDA only")
def test_save_at_steps_multi_creates_both_step_ckpts(tmp_path, monkeypatch):
    """Slice 13l multi-save: save_at_steps=[1, 2] must create BOTH
    `_step00001.pt` AND `_step00002.pt` from the SAME trajectory.
    Tiny train batches=8 batch_size=4 → 2 batches/epoch, exactly
    enough for step 1 + step 2 to both fire (false-negative-prone
    if too few batches, per codex msg 1779446584981 spec)."""
    sys.path.insert(0, ".")
    from scripts import train_dt_gsm8k as trainer

    # 8 rows / batch_size=4 → 2 batches → both step 1 + step 2 land
    tiny_rows = _make_tiny_gsm8k_rows(8)
    monkeypatch.setattr(
        trainer, "load_gsm8k_splits",
        lambda val_frac=0.10: (tiny_rows, tiny_rows[:4], tiny_rows[:2]),
    )

    ckpt_root = tmp_path / "test_13l_multi_save_best.pt"

    trainer.train(
        **_common_train_kwargs(ckpt_root),
        save_at_steps=[1, 2],
    )

    step1 = ckpt_root.with_name(ckpt_root.stem + "_step00001.pt")
    step2 = ckpt_root.with_name(ckpt_root.stem + "_step00002.pt")
    assert step1.exists() and step2.exists(), (
        f"multi-save did not produce both ckpts; "
        f"step1.exists={step1.exists()} step2.exists={step2.exists()}; "
        f"directory: {list(tmp_path.iterdir())}"
    )

    ck1 = torch.load(step1, map_location="cpu", weights_only=False)
    ck2 = torch.load(step2, map_location="cpu", weights_only=False)
    assert ck1["step"] == 1 and ck2["step"] == 2

    # Same-trajectory invariant: ckpts share epoch + come from the same
    # train() invocation (cannot equality-check model_state directly
    # since step 2 has more optimizer updates applied)
    assert ck1["epoch"] == ck2["epoch"]


@pytest.mark.skipif(not torch.cuda.is_available(),
                    reason="HRM segment loop runs on CUDA only")
def test_save_at_steps_dedup_handles_repeats(tmp_path, monkeypatch):
    """save_at_steps=[1, 1] (caller passed duplicates) should not crash
    and produce one file, not two/twice. frozenset dedupe handles this
    naturally per codex spec."""
    sys.path.insert(0, ".")
    from scripts import train_dt_gsm8k as trainer

    tiny_rows = _make_tiny_gsm8k_rows(16)
    monkeypatch.setattr(
        trainer, "load_gsm8k_splits",
        lambda val_frac=0.10: (tiny_rows, tiny_rows[:4], tiny_rows[:2]),
    )

    ckpt_root = tmp_path / "test_13l_dedup_best.pt"

    trainer.train(
        **_common_train_kwargs(ckpt_root),
        save_at_steps=[1, 1, 1],
    )

    step1 = ckpt_root.with_name(ckpt_root.stem + "_step00001.pt")
    assert step1.exists()
    # No surplus step files
    all_step = list(tmp_path.glob(f"{ckpt_root.stem}_step*.pt"))
    assert len(all_step) == 1, (
        f"dedup failed: expected 1 step file, got {len(all_step)}: {all_step}"
    )


def test_save_at_steps_rejects_non_positive():
    """train() entry validation: non-positive ints must raise ValueError.
    Runs without CUDA needed — fails before model build."""
    sys.path.insert(0, ".")
    from scripts import train_dt_gsm8k as trainer

    with pytest.raises(ValueError, match="positive ints"):
        trainer.train(
            epochs=1, batch_size=4, lr=1e-3,
            d_model=8, n_heads=4, n_layers=2, d_ffn=16,
            max_len=32, n_copy_heads=2, seed=42,
            checkpoint_path="/tmp/_unused.pt",
            save_at_steps=[0],  # invalid
        )

    with pytest.raises(ValueError, match="positive ints"):
        trainer.train(
            epochs=1, batch_size=4, lr=1e-3,
            d_model=8, n_heads=4, n_layers=2, d_ffn=16,
            max_len=32, n_copy_heads=2, seed=42,
            checkpoint_path="/tmp/_unused.pt",
            save_at_steps=[-5],  # invalid
        )


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
        **_common_train_kwargs(ckpt_root),
        # save_at_steps omitted — should default to None (no step save)
    )

    # No <stem>_step*.pt files should exist (glob on stem prefix)
    step_files = list(tmp_path.glob(f"{ckpt_root.stem}_step*.pt"))
    assert step_files == [], (
        f"save_at_steps=None should produce no step ckpts; "
        f"found {step_files}"
    )
