"""Tests for the 2C1 trainer sub-2 authority construction proof."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
import torch

import calm.hrm_text_158.native_full_stack as native_full_stack
from calm.hrm_text_158.bit_linear import BitLinear
from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
    TRAINER_SUB2_AUTHORITY_NON_CLAIMS,
    build_trainer_sub2_authority_construction_receipt,
    select_trainer_eligible_bitlinears,
    trainer_authoritative_forward_context,
    validate_trainer_sub2_authority_construction_receipt,
)


class _TinyTernary(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = BitLinear(16, 16, bias=False)
        self.tail = torch.nn.Linear(16, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.tail(self.proj(x))


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
        bp_max_steps=2,
        max_len=32,
        seed=158,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        log_every=1,
        device="cpu",
    )
    base.update(overrides)
    return base


def test_receipt_constructs_payload_counts_sub2_and_excludes_optimizer_masters():
    model = _TinyTernary()

    receipt = build_trainer_sub2_authority_construction_receipt(
        model,
        use_ternary_bulk=True,
        eligible_scope="all-bitlinear",
    )

    validate_trainer_sub2_authority_construction_receipt(receipt)
    assert receipt.pass_receipt is True
    assert receipt.trainer_entrypoint_can_construct_sub2_authority is True
    assert receipt.trainer_entrypoint_uses_candidate is False
    assert receipt.live_runtime_authority_converted is False
    assert receipt.readiness_row_flip_authorized is False
    assert receipt.learner_update_called is False
    assert receipt.optimizer_step_called is False
    assert receipt.checkpoint_written is False
    assert receipt.gpu_launched is False
    assert receipt.eligible_module_count == 1
    assert receipt.optimizer_exclusion_proof["eligible_params_in_optimizer"] == 0
    assert receipt.optimizer_exclusion_proof["eligible_optimizer_state_entries"] == 0
    assert receipt.checkpoint_payload_validated is True
    assert receipt.checkpoint_payload_summary["checkpoint_written"] is False
    assert receipt.checkpoint_payload_summary["dry_run"] is True
    assert receipt.persistent_authority_bits_per_weight < 2.0
    assert receipt.dense_int16_persistent_authority_bits_counted == 0
    assert receipt.fp_master_persistent_authority_bits_counted == 0
    assert receipt.non_claims == TRAINER_SUB2_AUTHORITY_NON_CLAIMS


def test_receipt_fails_closed_without_ternary_bulk_or_bitlinear():
    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        select_trainer_eligible_bitlinears(
            _TinyTernary(),
            use_ternary_bulk=False,
        )
    with pytest.raises(RuntimeError, match="no eligible BitLinear"):
        select_trainer_eligible_bitlinears(
            torch.nn.Linear(4, 4),
            use_ternary_bulk=True,
        )


def test_forbidden_broad_claims_fail_validation():
    receipt = build_trainer_sub2_authority_construction_receipt(
        _TinyTernary(),
        use_ternary_bulk=True,
    )

    with pytest.raises(ValueError, match="trainer-used candidate"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, trainer_entrypoint_uses_candidate=True)
        )
    with pytest.raises(ValueError, match="live runtime"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, live_runtime_authority_converted=True)
        )
    with pytest.raises(ValueError, match="readiness row"):
        validate_trainer_sub2_authority_construction_receipt(
            replace(receipt, readiness_row_flip_authorized=True)
        )


def test_forward_context_wrapper_uses_q_state_without_update_claim():
    model = _TinyTernary()
    eligible = select_trainer_eligible_bitlinears(model, use_ternary_bulk=True)
    states = native_full_stack.derive_trainer_sub2_authority_states(eligible)
    x = torch.randn(2, 16)

    with trainer_authoritative_forward_context(
        eligible,
        states,
        requires_grad=False,
    ) as handle:
        out = model.proj(x)
        expected = torch.nn.functional.linear(
            x,
            states["proj"].materialized_weight(requires_grad=False),
            None,
        )

    torch.testing.assert_close(out, expected, atol=0.0, rtol=0.0)
    assert handle.capture_enabled is False


def test_trainer_default_off_smoke_still_writes_checkpoint(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert ckpt_root.exists()
    assert ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c1_proof_exits_before_checkpoint_write(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        save_at_steps=[1],
        use_ternary_bulk=True,
        sub2_authority_construction_proof=True,
        sub2_authority_eligible_scope="first-bitlinear",
    )

    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))

    ckpt_root = Path(kwargs["checkpoint_path"])
    assert not ckpt_root.exists()
    assert not ckpt_root.with_name(ckpt_root.stem + "_step00001.pt").exists()


def test_trainer_enabled_2c1_proof_requires_ternary_bulk(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        sub2_authority_construction_proof=True,
        use_ternary_bulk=False,
    )

    with pytest.raises(RuntimeError, match="requires --use-ternary-bulk"):
        trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))


def test_exports_from_native_full_stack_facade():
    receipt = native_full_stack.build_trainer_sub2_authority_construction_receipt(
        _TinyTernary(),
        use_ternary_bulk=True,
    )

    assert receipt.pass_receipt is True
    assert "build_trainer_sub2_authority_construction_receipt" in native_full_stack.__all__
