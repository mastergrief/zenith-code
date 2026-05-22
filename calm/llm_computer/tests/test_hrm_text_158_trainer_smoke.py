"""HRM-Text-1.58 Phase 1 Slice 2 trainer/probe smoke tests.

Per codex msg 1779452208756 guardrail: "Keep test mini-config cheap
enough for CPU; do not make 3.72M Tier A CPU training a slow unit test.
Use tiny config for unit test, and run Tier A in the actual command
receipt."

Cheap config: hidden_size=32, n_layers=2 (split 1+1), num_heads=2 (head_dim=16),
H_cycles=2, L_cycles=2, max_len=16, batch_size=2.

Tests:
1. End-to-end 1-step trainer + save_at_step=1 produces a loadable ckpt
2. Multi-save_at_step=[1,2] saves both ckpts in same trajectory
3. Probe reconstruction from ckpt + cap=2 small probe
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")


def _make_tiny_gsm8k_rows(n: int = 8) -> list[dict]:
    """Tiny GSM8k-shaped rows. Use BOTH `plus` and `times` so that the
    canonical 17×23 probe (which uses "what is 17 times 23?") finds all
    its chars covered by the tokenizer vocab."""
    return [
        {"question": f"what is {i + 1} {'plus' if i % 2 == 0 else 'times'} {i + 1}?",
         "expected": (2 * (i + 1)) if i % 2 == 0 else ((i + 1) ** 2)}
        for i in range(n)
    ]


def _splits_loader_factory(rows):
    """Monkeypatch helper: returns a (train, val, test) closure for load_gsm8k_splits."""
    def _loader(val_frac: float = 0.10):
        return rows, rows[:2], rows[:2]
    return _loader


def _common_train_kwargs(tmp_path: Path, **overrides) -> dict:
    """Cheap CPU-friendly kwargs (NOT Tier A)."""
    ckpt = tmp_path / "test_smoke_best.pt"
    base = dict(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,  # No warmup at 4-batch epoch
        hidden_size=32,
        n_layers=2,        # half_layers=True -> 1 H + 1 L
        num_heads=2,       # head_dim = 16
        expansion=4,
        H_cycles=2,
        L_cycles=2,
        half_layers=True,
        bp_warmup_ratio=0.0,
        bp_min_steps=2,
        bp_max_steps=2,
        max_len=32,
        seed=42,
        checkpoint_path=str(ckpt),
        log_every=1,
        device="cpu",
    )
    base.update(overrides)
    return base


def test_trainer_save_at_step_single(tmp_path, monkeypatch) -> None:
    """End-to-end 1-step trainer + save_at_steps=[1]: ckpt loadable."""
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows(4)
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))

    kwargs = _common_train_kwargs(tmp_path, save_at_steps=[1])
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    step1_path = ckpt_root.with_name(ckpt_root.stem + "_step00001.pt")
    assert step1_path.exists(), f"save_at_steps=[1] did not produce {step1_path}"
    assert ckpt_root.exists(), f"final ckpt {ckpt_root} not saved"

    # Load + check shape
    ckpt = torch.load(step1_path, map_location="cpu", weights_only=False)
    assert "model_state" in ckpt
    assert "config" in ckpt
    assert "source_pin" in ckpt
    assert ckpt["source_pin"]["sha"].startswith("056c4ec")
    cfg = ckpt["config"]
    assert cfg["hidden_size"] == 32
    assert cfg["H_cycles"] == 2
    assert cfg["L_cycles"] == 2
    assert cfg["half_layers"] is True
    # gsm8k tokenizer metadata
    assert "gsm8k_char_vocab" in cfg
    assert "gsm8k_normalizer_version" in cfg
    assert isinstance(cfg["gsm8k_char_vocab"], list)


def test_trainer_save_at_steps_multi(tmp_path, monkeypatch) -> None:
    """save_at_steps=[1,2] produces BOTH ckpts in same trajectory."""
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows(4)
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))

    kwargs = _common_train_kwargs(tmp_path, save_at_steps=[1, 2])
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    step1_path = ckpt_root.with_name(ckpt_root.stem + "_step00001.pt")
    step2_path = ckpt_root.with_name(ckpt_root.stem + "_step00002.pt")
    assert step1_path.exists(), f"missing step1 ckpt: {step1_path}"
    assert step2_path.exists(), f"missing step2 ckpt: {step2_path}"
    # Both ckpts come from the same trajectory: step 1 < step 2
    ck1 = torch.load(step1_path, map_location="cpu", weights_only=False)
    ck2 = torch.load(step2_path, map_location="cpu", weights_only=False)
    assert ck1["step"] == 1
    assert ck2["step"] == 2


def test_probe_loads_ckpt_and_runs_smoke(tmp_path, monkeypatch) -> None:
    """Train tiny model, save ckpt, probe reconstructs + runs cap=2.
    Probe results are reported, NOT asserted (per codex correction 1)."""
    from scripts import train_hrm_text_158 as trainer
    from scripts import probe_hrm_text_158 as probe_mod

    rows = _make_tiny_gsm8k_rows(8)
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    monkeypatch.setattr(probe_mod, "load_gsm8k_splits", _splits_loader_factory(rows))

    kwargs = _common_train_kwargs(tmp_path, save_at_steps=[1])
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    result = probe_mod.probe(
        ckpt_path=str(ckpt_root),
        eval_cap=2,
        max_gen=4,
        device="cpu",
        splits_loader=_splits_loader_factory(rows),
    )
    # Probe must return a structured dict
    assert "parsed_correct" in result
    assert "exact_correct" in result
    assert "too_long_count" in result   # per codex msg 1779452728846 gate 2
    assert "canonical_17x23" in result
    assert "rows" in result
    assert len(result["rows"]) == 2
    # Each row must have decoded + expected + parsed_ok + too_long fields
    for row in result["rows"]:
        assert "decoded" in row
        assert "expected" in row
        assert "parsed_ok" in row
        assert "too_long" in row
        # If too_long=True, parsed_ok and exact_ok MUST be False (cannot
        # silently count an unevaluable row as correct).
        if row["too_long"]:
            assert row["parsed_ok"] is False
            assert row["exact_ok"] is False
    # Canonical 17×23 row must be present + structured (correctness REPORTED, not asserted)
    canon = result["canonical_17x23"]
    assert canon["question"] == "what is 17 times 23?"
    assert canon["expected"] == 391
    assert "decoded" in canon
    assert "parsed_ok" in canon
    assert "exact_ok" in canon
    assert "too_long" in canon
    # parsed_correct + exact_correct + too_long_count are ints in [0, cap]
    assert isinstance(result["parsed_correct"], int)
    assert 0 <= result["parsed_correct"] <= 2
    assert isinstance(result["too_long_count"], int)
    assert 0 <= result["too_long_count"] <= 2
    # Sum of (correct + too_long + ordinary wrong) == cap (denominator preserved)
    ordinary_wrong = 2 - result["parsed_correct"] - result["too_long_count"]
    assert ordinary_wrong >= 0
