"""R1 production trainer activation-relief wiring proof tests."""
from __future__ import annotations

import hashlib
import inspect
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
    R1_CPU_BASE_COMMIT_SHA,
    build_trainer_backward_wiring_proof_receipt,
    validate_trainer_backward_wiring_proof_receipt,
)
from scripts.train_hrm_text_158 import SOURCE_PIN, _build_ckpt_config, train

REPO_ROOT = Path(__file__).resolve().parents[3]
W6_PARENT_PATH = (
    REPO_ROOT
    / "calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_"
    "anchorsv1r3_from_L0b_final_step01500.pt"
)


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


def test_trainer_launch_proof_flag_default_off_matches_wiring_default():
    from scripts.train_hrm_text_158 import train

    default = inspect.signature(train).parameters[
        "activation_relief_lossless_recompute_launch_proof"
    ].default
    assert default is False


def test_trainer_launch_proof_stub_removed_message_absent_when_mocked(
    tmp_path,
    monkeypatch,
    capsys,
):
    import os

    from calm.hrm_text_158.native_full_stack import activation_relief
    from scripts import train_hrm_text_158 as trainer

    monkeypatch.setattr(trainer.torch.cuda, "is_available", lambda: True)
    from calm.llm_computer.tests.test_hrm_text_158_phase_a_wiring import (
        TINY_ARCH,
        _build_tiny_broad_ckpt_blob,
    )

    parent_path = tmp_path / "parent_R0.pt"
    torch.save(_build_tiny_broad_ckpt_blob(), parent_path)
    log_path = tmp_path / "launch.log"
    log_path.write_bytes(b"RESULT=PASS\n")
    receipt_path = tmp_path / "receipt.json"
    manifest_path = tmp_path / "manifest.json"
    env_path = tmp_path / "env.json"
    manifest = {
        "r1_cpu_base_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "launch_source_commit_sha": R1_CPU_BASE_COMMIT_SHA,
        "archive_created_at_utc": "2026-06-15T00:00:00Z",
        "archive_method": "git_archive_HEAD",
    }
    env = {
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": str(REPO_ROOT),
        "CUDA_VISIBLE_DEVICES": "0",
        "R1L_LAUNCH_RECEIPT_JSON": str(receipt_path),
        "R1L_LAUNCH_LOG": str(log_path),
        "R1L_W6_PARENT_PATH": str(parent_path),
        "TORCH_CUDA_ALLOC_CONF": "",
        "CUBLAS_WORKSPACE_CONFIG": "",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    env_path.write_text(json.dumps(env), encoding="utf-8")
    monkeypatch.setenv("R1L_ANCESTRY_VERIFIED", "1")
    monkeypatch.setenv("R1L_CLEAN_RUN_DIR_SHA256", "a" * 64)
    monkeypatch.setenv("R1L_LAUNCH_MANIFEST_JSON", str(manifest_path))
    monkeypatch.setenv("R1L_LAUNCH_ENV_JSON", str(env_path))
    monkeypatch.setenv("R1L_LAUNCH_RECEIPT_JSON", str(receipt_path))
    monkeypatch.setenv("R1L_LAUNCH_LOG", str(log_path))
    monkeypatch.setenv("R1L_W6_PARENT_PATH", str(parent_path))

    def _fake_launch_proof(**_kwargs):
        from calm.hrm_text_158.native_full_stack.activation_relief import (
            build_launch_runtime_backward_validation_receipt,
        )

        return build_launch_runtime_backward_validation_receipt(
            launch_source_commit_sha=R1_CPU_BASE_COMMIT_SHA,
            launch_manifest_embedded=manifest,
            proof_env_embedded=env,
            proof_command_argv=("pytest",),
            clean_run_dir_sha256="a" * 64,
            w6_parent_path=str(parent_path),
            gpu_name="synthetic-gpu",
            gpu_uuid="gpu-uuid-test",
            driver_version="550.00",
            cuda_version="12.4",
            torch_version=torch.__version__,
            model_config_digest_sha256="b" * 64,
            proof_batch_digest_sha256="c" * 64,
            retained_support_digest_sha256="d" * 64,
            main_baseline_saved_tensor_count=20,
            main_recompute_saved_tensor_count=15,
            main_saved_tensor_payload_bytes_baseline=1000,
            main_saved_tensor_payload_bytes_recompute=800,
            retained_side_in_scope=True,
            retained_side_baseline_saved_tensor_count=18,
            retained_side_recompute_saved_tensor_count=14,
            retained_saved_tensor_payload_bytes_delta=400,
            paired_run_count=3,
            cuda_peak_allocated_bytes_baseline_median=64 * 1024 * 1024,
            cuda_peak_allocated_bytes_recompute_median=56 * 1024 * 1024,
            cuda_peak_reserved_bytes_delta_median=0,
            log_artifact_sha256=hashlib.sha256(log_path.read_bytes()).hexdigest(),
        )

    monkeypatch.setattr(activation_relief, "run_r1l_gpu_launch_proof", _fake_launch_proof)
    kwargs = dict(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=32,
        curriculum_n_heldout=8,
        load_from=str(parent_path),
        retained_support_profile=[("L0b", 0.01)],
        activation_relief_lossless_recompute_launch_proof=True,
        checkpoint_path=str(tmp_path / "should_not_write.pt"),
        epochs=1,
        batch_size=2,
        device="cpu",
        n_train_cap=8,
        **TINY_ARCH,
    )
    trainer.train(**kwargs)
    out = capsys.readouterr().out
    assert "EXITING before GPU launch mint" not in out
