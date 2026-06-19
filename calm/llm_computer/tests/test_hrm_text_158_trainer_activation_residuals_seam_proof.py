"""R2-A production trainer activation/residual seam proof tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack import (
    ACTIVATION_RESIDUAL_TARGET_FAMILIES,
    PROOF_KIND_CPU_PRODUCTION_SEAM_OBSERVATION,
    build_activation_residuals_fail_closed_receipt,
    build_trainer_activation_residuals_seam_proof_receipt,
)
from calm.llm_computer.tests.test_hrm_text_158_activation_relief import (
    _activation_residual_live_tensor_proof,
)
from scripts.train_hrm_text_158 import train


def _make_tiny_gsm8k_rows(n: int = 8) -> list[dict]:
    return [
        {
            "question": f"what is {i + 1} {'plus' if i % 2 == 0 else 'times'} {i + 1}?",
            "expected": (2 * (i + 1)) if i % 2 == 0 else ((i + 1) ** 2),
        }
        for i in range(n)
    ]


def _splits_loader_factory(rows):
    def _loader(val_frac: float = 0.10):
        return rows, rows[:2], rows[:2]

    return _loader


def _common_train_kwargs(tmp_path: Path, **overrides) -> dict:
    base = dict(
        epochs=1,
        batch_size=2,
        lr=1e-3,
        weight_decay=0.0,
        warmup_ratio=0.0,
        hidden_size=32,
        n_layers=2,
        num_heads=2,
        expansion=4,
        H_cycles=2,
        L_cycles=2,
        half_layers=True,
        bp_warmup_ratio=0.0,
        bp_min_steps=2,
        bp_max_steps=5,
        max_len=32,
        seed=158,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        log_every=1,
        device="cpu",
    )
    base.update(overrides)
    return base


def test_trainer_default_off_does_not_emit_r2a_seam_proof(tmp_path, monkeypatch, capsys):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(tmp_path, save_at_steps=[1])
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    out = capsys.readouterr().out
    assert "R2-A activation/residual seam proof" not in out


def test_trainer_r2a_seam_proof_exits_before_optimizer(tmp_path, monkeypatch, capsys):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_fail_closed_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    out = capsys.readouterr().out
    assert "R2-A activation/residual seam proof" in out
    assert "EXITING before optimizer step" in out
    assert not Path(kwargs["checkpoint_path"]).exists()


def test_trainer_r2a_seam_proof_mints_cpu_receipt_json(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    receipt_path = tmp_path / "r2a_seam_receipt.json"
    monkeypatch.setenv("R2A_ACTIVATION_RESIDUALS_RECEIPT_JSON", str(receipt_path))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_fail_closed_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["proof_kind"] == PROOF_KIND_CPU_PRODUCTION_SEAM_OBSERVATION
    assert payload["live_readiness_row_flip_authorized"] is False
    assert payload["readiness_row_flip_authorized_surface_names"] == []
    fail_closed = payload["fail_closed_receipt"]
    assert fail_closed["ready_to_flip"] is False
    assert fail_closed["activations_residuals_sub2_claim"] is False
    assert fail_closed["real_sub2_or_remat_or_offload_mechanism_present"] is False
    assert fail_closed["no_hidden_bf16_authority_proven"] is False
    assert fail_closed["gpu_memory_receipt_present"] is False
    assert fail_closed["target_families"] == list(ACTIVATION_RESIDUAL_TARGET_FAMILIES)


def test_trainer_r2a_flag_mutually_exclusive_with_r1_wiring(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_residuals_fail_closed_proof=True,
        activation_relief_lossless_recompute_wiring_proof=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_validate_trainer_activation_residuals_seam_proof_receipt_accepts_cpu_blocker():
    from calm.hrm_text_158.native_full_stack import (
        validate_trainer_activation_residuals_seam_proof_receipt,
    )

    events, zL_init_observation = _activation_residual_live_tensor_proof()
    receipt = build_trainer_activation_residuals_seam_proof_receipt(
        source_commit_sha="abc123",
        proof_command_argv=("train",),
        seam_events=events,
        zL_init_observation=zL_init_observation,
    )
    validate_trainer_activation_residuals_seam_proof_receipt(receipt)
    assert receipt.fail_closed_receipt.ready_to_flip is False


def test_fixture_fail_closed_receipt_stays_non_flip():
    events, zL_init_observation = _activation_residual_live_tensor_proof()
    receipt = build_activation_residuals_fail_closed_receipt(
        seam_events=events,
        zL_init_observation=zL_init_observation,
    )
    assert receipt.ready_to_flip is False
