"""R1 production trainer activation-relief wiring proof tests."""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.native_full_stack import (
    MODE_LOSSLESS_RECOMPUTE,
    PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING,
    build_trainer_backward_wiring_proof_receipt,
    validate_trainer_backward_wiring_proof_receipt,
)
from scripts.train_hrm_text_158 import SOURCE_PIN, _build_ckpt_config, train


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


def _build_parent_ckpt(tmp_path: Path, **train_kwargs) -> Path:
    from calm.llm_computer.gsm8k_tokenizer import Gsm8kTokenizer

    tok = Gsm8kTokenizer.from_corpus(_make_tiny_gsm8k_rows())
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=train_kwargs["max_len"],
        n_layers=train_kwargs["n_layers"],
        hidden_size=train_kwargs["hidden_size"],
        num_heads=train_kwargs["num_heads"],
        expansion=train_kwargs["expansion"],
        H_cycles=train_kwargs["H_cycles"],
        L_cycles=train_kwargs["L_cycles"],
        half_layers=train_kwargs["half_layers"],
        bp_warmup_ratio=train_kwargs["bp_warmup_ratio"],
        bp_min_steps=train_kwargs["bp_min_steps"],
        bp_max_steps=train_kwargs["bp_max_steps"],
    )
    hrm = HierarchicalReasoningModel(cfg)
    model = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))
    config_blob = _build_ckpt_config(
        model,
        tok,
        cfg,
        train_kwargs["max_len"],
        train_kwargs["batch_size"],
    )
    ckpt_path = tmp_path / "parent.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": config_blob,
            "step": 0,
            "epoch": 0,
            "source_pin": SOURCE_PIN,
        },
        ckpt_path,
    )
    return ckpt_path


def test_trainer_default_off_does_not_inject_activation_relief_policy(tmp_path, monkeypatch):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    captured: list[dict] = []
    original = trainer.LMHead.compute_train_extra_args

    def _capture(self, step, total_steps):
        extras = original(self, step, total_steps)
        captured.append(dict(extras))
        return extras

    monkeypatch.setattr(trainer.LMHead, "compute_train_extra_args", _capture)
    kwargs = _common_train_kwargs(tmp_path, save_at_steps=[1])
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    assert captured
    assert all("activation_relief_policy" not in extras for extras in captured)


def test_trainer_r1_wiring_proof_exits_before_optimizer_without_retained(
    tmp_path,
    monkeypatch,
    capsys,
):
    from scripts import train_hrm_text_158 as trainer

    rows = _make_tiny_gsm8k_rows()
    monkeypatch.setattr(trainer, "load_gsm8k_splits", _splits_loader_factory(rows))
    kwargs = _common_train_kwargs(
        tmp_path,
        activation_relief_lossless_recompute_wiring_proof=True,
    )
    trainer.train(**kwargs, splits_loader=_splits_loader_factory(rows))
    out = capsys.readouterr().out
    assert "R1 activation-relief wiring proof" in out
    assert "EXITING before optimizer step" in out
    assert not Path(kwargs["checkpoint_path"]).exists()


def test_trainer_r1_wiring_proof_mints_cpu_receipt_json(
    tmp_path,
    monkeypatch,
):
    from calm.llm_computer.tests.test_hrm_text_158_phase_a_wiring import (
        TINY_ARCH,
        _build_tiny_broad_ckpt_blob,
    )
    from scripts import train_hrm_text_158 as trainer

    parent_path = tmp_path / "parent_R0.pt"
    torch.save(_build_tiny_broad_ckpt_blob(), parent_path)
    receipt_path = tmp_path / "r1_wiring_receipt.json"
    monkeypatch.setenv("R1_BACKWARD_WIRING_RECEIPT_JSON", str(receipt_path))
    kwargs = dict(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=32,
        curriculum_n_heldout=8,
        load_from=str(parent_path),
        retained_support_profile=[("L0b", 0.01)],
        activation_relief_lossless_recompute_wiring_proof=True,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        epochs=1,
        batch_size=2,
        device="cpu",
        **TINY_ARCH,
    )
    trainer.train(**kwargs)
    payload = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert payload["proof_kind"] == PROOF_KIND_CPU_PRODUCTION_AUTOGAD_WIRING
    assert payload["policy_mode"] == MODE_LOSSLESS_RECOMPUTE
    assert payload["live_readiness_row_flip_authorized"] is False
    assert payload["readiness_row_flip_authorized_surface_names"] == []
    assert payload["main_path_proven"] is True
    assert payload["retained_side_path_proven"] is True
    assert payload["main_internal_payload_tensor_count"] == 0
    assert payload["retained_side_internal_payload_tensor_count"] == 0
    assert (
        payload["main_baseline_saved_tensor_count"]
        > payload["main_recompute_saved_tensor_count"]
    )
    assert (
        payload["retained_side_baseline_saved_tensor_count"]
        > payload["retained_side_recompute_saved_tensor_count"]
    )
    validate_trainer_backward_wiring_proof_receipt(
        build_trainer_backward_wiring_proof_receipt(
            source_commit_sha=payload["source_commit_sha"],
            proof_command_argv=tuple(payload["proof_command_argv"]),
            H_cycles=TINY_ARCH["H_cycles"],
            L_cycles=TINY_ARCH["L_cycles"],
            bp_steps=TINY_ARCH["bp_max_steps"],
            main_path_proof={
                "baseline_saved_tensor_count": payload["main_baseline_saved_tensor_count"],
                "recompute_saved_tensor_count": payload["main_recompute_saved_tensor_count"],
                "internal_payload_tensor_count": payload["main_internal_payload_tensor_count"],
                "recompute_checkpoint_fired": payload["main_recompute_checkpoint_fired"],
            },
            retained_side_path_proof={
                "baseline_saved_tensor_count": (
                    payload["retained_side_baseline_saved_tensor_count"]
                ),
                "recompute_saved_tensor_count": (
                    payload["retained_side_recompute_saved_tensor_count"]
                ),
                "internal_payload_tensor_count": (
                    payload["retained_side_internal_payload_tensor_count"]
                ),
                "recompute_checkpoint_fired": (
                    payload["retained_side_recompute_checkpoint_fired"]
                ),
            },
            retained_side_in_scope=True,
        )
    )


def _valid_cpu_wiring_receipt(*, retained_side_in_scope: bool = True):
    main_proof = {
        "baseline_saved_tensor_count": 20,
        "recompute_saved_tensor_count": 15,
        "internal_payload_tensor_count": 0,
        "recompute_checkpoint_fired": True,
    }
    retained_proof = {
        "baseline_saved_tensor_count": 18,
        "recompute_saved_tensor_count": 12,
        "internal_payload_tensor_count": 0,
        "recompute_checkpoint_fired": True,
    }
    return build_trainer_backward_wiring_proof_receipt(
        source_commit_sha="abc123",
        proof_command_argv=("pytest",),
        H_cycles=2,
        L_cycles=3,
        bp_steps=5,
        main_path_proof=main_proof,
        retained_side_path_proof=retained_proof if retained_side_in_scope else None,
        retained_side_in_scope=retained_side_in_scope,
        retained_side_skip_reason="" if retained_side_in_scope else "no retained supports",
    )


def test_trainer_backward_wiring_validator_rejects_false_main_checkpoint_fired():
    receipt = _valid_cpu_wiring_receipt(retained_side_in_scope=False)
    forged = replace(receipt, main_recompute_checkpoint_fired=False)
    with pytest.raises(ValueError, match="main recompute checkpoint fired"):
        validate_trainer_backward_wiring_proof_receipt(forged)


def test_trainer_backward_wiring_validator_rejects_false_retained_checkpoint_fired():
    receipt = _valid_cpu_wiring_receipt(retained_side_in_scope=True)
    forged = replace(receipt, retained_side_recompute_checkpoint_fired=False)
    with pytest.raises(ValueError, match="retained-side recompute checkpoint fired"):
        validate_trainer_backward_wiring_proof_receipt(forged)
