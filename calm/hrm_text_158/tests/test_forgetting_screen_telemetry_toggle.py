"""Telemetry OFF/ON seam characterization (option a).

Dedicated module — keeps test_forgetting_screen_v10_contract.py free of
toggle-suite growth. Bound by +1 1784896036962 + seam defect-cycle
1784896455402.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.family_classifier import (
    ARM1,
    ARM2,
    ARM3,
)


def test_cli_telemetry_flag_defaults_on_and_no_telemetry(monkeypatch, tmp_path):
    """Live CLI main() characterization — captured args.telemetry from real parser."""
    import scripts.hrm_text_158_forgetting_mechanism_screen as cli

    captured: list = []

    def _capture_run_arm_screen(args):
        captured.append(bool(args.telemetry))
        return 0

    monkeypatch.setattr(cli, "run_arm_screen", _capture_run_arm_screen)
    ckpt = tmp_path / "dummy.pt"
    ckpt.write_bytes(b"x")
    base = [
        "hrm_text_158_forgetting_mechanism_screen.py",
        "--ckpt-path",
        str(ckpt),
        "--device",
        "cpu",
        "--skip-probes",
        "--steps",
        "1",
    ]

    monkeypatch.setattr(sys, "argv", list(base))
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", list(base) + ["--telemetry"])
    assert cli.main() == 0
    monkeypatch.setattr(sys, "argv", list(base) + ["--no-telemetry"])
    assert cli.main() == 0
    assert captured == [True, True, False]

    help_txt = __import__("subprocess").check_output(
        ["python3", "scripts/hrm_text_158_forgetting_mechanism_screen.py", "--help"],
        text=True,
    )
    assert "--telemetry" in help_txt
    assert "--no-telemetry" in help_txt


def test_telemetry_off_skips_store_passes_none_to_loop(monkeypatch, tmp_path):
    """OFF path: DeviceLifecycleStore never constructed; pressure_telemetry=None."""
    from calm.hrm_text_158.native_full_stack import screen_run_loop as srl

    constructed: list[str] = []

    class _BoomStore:
        @classmethod
        def from_arm_shapes(cls, *args, **kwargs):
            constructed.append("constructed")
            raise AssertionError("store must not be built when telemetry OFF")

    captured: dict = {}

    def _fake_load_and_patch_runtime(*, ckpt_path, device):
        q = {"layer.w": torch.zeros(4, 4, dtype=torch.int8)}
        return {
            "m": object(),
            "tok": object(),
            "eligible": ["layer.w"],
            "q_levels": q,
            "frozen_scales": {"layer.w": torch.ones((), dtype=torch.float32)},
            "max_seq_len": 32,
            "sha_before": "a" * 64,
            "scale_sha_before": "b" * 64,
            "q_sha_before": "c" * 64,
        }

    def _fake_run_train_loop(**kwargs):
        captured["pressure_telemetry"] = kwargs.get("pressure_telemetry")
        q = kwargs["q_levels"]
        z = {n: torch.zeros_like(t) for n, t in q.items()}
        return {
            "acc": z,
            "episode_start": {
                n: torch.zeros_like(t, dtype=torch.int32) for n, t in q.items()
            },
            "flip_count": {
                n: torch.zeros_like(t, dtype=torch.int32) for n, t in q.items()
            },
            "lifetimes": [],
            "credited_mass": 0,
            "n_flips": 0,
            "q_changed_count": 0,
            "n_applied_drains": 0,
            "excluded_hit_count": 0,
            "H_trajectory": [],
            "train_route_counters": {
                "n_fixed_qscale_forwards": 1,
                "n_bitlinear_dynamic_forwards": 0,
                "n_eligible_keys": 1,
                "n_credit_grads_present": 1,
            },
            "q_levels": q,
        }

    def _fake_probes():
        return {
            "acquisition": [],
            "retention": [],
            "acquisition_n": 0,
            "retention_n": 0,
            "acquisition_selection_sha256": "0" * 64,
            "identity_selection_sha256": "1" * 64,
            "math_a0_parent_support_hash": "2" * 64,
            "identity_parent_support_hash": "3" * 64,
        }

    monkeypatch.setattr(srl, "load_and_patch_runtime", _fake_load_and_patch_runtime)
    monkeypatch.setattr(srl, "run_train_loop", _fake_run_train_loop)
    monkeypatch.setattr(srl, "build_phase1_probe_sets", _fake_probes)
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk.DeviceLifecycleStore",
        _BoomStore,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.curriculum.exhaustive_supports.build_exhaustive_supports",
        lambda: {"r": [("1+1", 2)]},
    )

    ckpt = tmp_path / "p.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "off.json"
    args = argparse.Namespace(
        ckpt_path=str(ckpt),
        device="cpu",
        arm=ARM1,
        steps=2,
        batch=1,
        topk=8,
        correctness_smoke=False,
        skip_probes=True,
        telemetry=False,
        output_json=str(out),
    )
    rc = srl.run_arm_screen(args)
    assert rc == 0
    assert constructed == []
    assert captured["pressure_telemetry"] is None
    receipt = json.loads(out.read_text())
    assert receipt["telemetry"] is False
    assert "demand" not in receipt["measurements"]
    assert "deferred_survival" not in receipt["measurements"]


def test_telemetry_on_default_constructs_store(monkeypatch, tmp_path):
    """Default ON preserves DeviceLifecycleStore attach (byte-identical intent)."""
    from calm.hrm_text_158.native_full_stack import screen_run_loop as srl

    constructed: list[object] = []

    class _FakeStore:
        @classmethod
        def from_arm_shapes(cls, shapes, *, steps, device):
            obj = object()
            constructed.append(obj)
            return obj

    captured: dict = {}

    def _fake_load_and_patch_runtime(*, ckpt_path, device):
        q = {"layer.w": torch.zeros(4, 4, dtype=torch.int8)}
        return {
            "m": object(),
            "tok": object(),
            "eligible": ["layer.w"],
            "q_levels": q,
            "frozen_scales": {"layer.w": torch.ones((), dtype=torch.float32)},
            "max_seq_len": 32,
            "sha_before": "a" * 64,
            "scale_sha_before": "b" * 64,
            "q_sha_before": "c" * 64,
        }

    def _fake_run_train_loop(**kwargs):
        captured["pressure_telemetry"] = kwargs.get("pressure_telemetry")
        q = kwargs["q_levels"]
        z = {n: torch.zeros_like(t) for n, t in q.items()}
        return {
            "acc": z,
            "episode_start": {
                n: torch.zeros_like(t, dtype=torch.int32) for n, t in q.items()
            },
            "flip_count": {
                n: torch.zeros_like(t, dtype=torch.int32) for n, t in q.items()
            },
            "lifetimes": [],
            "credited_mass": 0,
            "n_flips": 0,
            "q_changed_count": 0,
            "n_applied_drains": 0,
            "excluded_hit_count": 0,
            "H_trajectory": [],
            "train_route_counters": {
                "n_fixed_qscale_forwards": 1,
                "n_bitlinear_dynamic_forwards": 0,
                "n_eligible_keys": 1,
                "n_credit_grads_present": 1,
            },
            "q_levels": q,
        }

    monkeypatch.setattr(srl, "load_and_patch_runtime", _fake_load_and_patch_runtime)
    monkeypatch.setattr(srl, "run_train_loop", _fake_run_train_loop)
    monkeypatch.setattr(
        srl,
        "build_phase1_probe_sets",
        lambda: {
            "acquisition": [],
            "retention": [],
            "acquisition_n": 0,
            "retention_n": 0,
            "acquisition_selection_sha256": "0" * 64,
            "identity_selection_sha256": "1" * 64,
            "math_a0_parent_support_hash": "2" * 64,
            "identity_parent_support_hash": "3" * 64,
        },
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.native_full_stack.pressure_metric_gpu_lifecycle_derisk.DeviceLifecycleStore",
        _FakeStore,
    )
    monkeypatch.setattr(
        "calm.hrm_text_158.curriculum.exhaustive_supports.build_exhaustive_supports",
        lambda: {"r": [("1+1", 2)]},
    )

    ckpt = tmp_path / "p.pt"
    ckpt.write_bytes(b"x")
    out = tmp_path / "on.json"
    args = argparse.Namespace(
        ckpt_path=str(ckpt),
        device="cpu",
        arm=ARM1,
        steps=2,
        batch=1,
        topk=8,
        correctness_smoke=False,
        skip_probes=True,
        output_json=str(out),
    )
    rc = srl.run_arm_screen(args)
    assert rc == 0
    assert len(constructed) == 1
    assert captured["pressure_telemetry"] is constructed[0]
    receipt = json.loads(out.read_text())
    assert receipt["telemetry"] is True


def test_telemetry_off_receipt_fails_g0_and_three_arm(tmp_path):
    """OFF receipt (no R1 surface) cannot masquerade as formal arm under v10."""
    from calm.hrm_text_158.native_full_stack.forgetting_mechanism_screen_reducers import (
        DEFAULT_PARENT_SHA256,
        V10ArmReceiptContractError,
        validate_three_mechanism_arm_receipts_v10,
    )
    from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_contract import (
        arm_metrics_for_v10_classifier,
        g0_valid_v10,
    )
    from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
        build_phase1_probe_sets,
    )
    from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
        AUTHORITY_DISPATCH,
        PLAN_SHA256,
        assemble_arm_receipt,
    )

    steps = 150
    shape = (8, 8)
    acc = {"layer.w": torch.ones(shape, dtype=torch.int16)}
    episode_start = {"layer.w": torch.ones(shape, dtype=torch.int32)}
    flip_count = {"layer.w": torch.ones(shape, dtype=torch.int32)}
    q_levels = {"layer.w": torch.zeros(shape, dtype=torch.int8)}
    frozen_scales = {"layer.w": torch.ones(shape, dtype=torch.float32)}
    ckpt = tmp_path / "parent.pt"
    ckpt.write_bytes(b"telemetry-off-fail-closed")
    parent_sha = hashlib.sha256(ckpt.read_bytes()).hexdigest()

    def _tsha(t):
        return hashlib.sha256(
            t.detach().cpu().contiguous().numpy().tobytes()
        ).hexdigest()

    scale_before = hashlib.sha256(
        b"".join(_tsha(frozen_scales[n]).encode() for n in sorted(frozen_scales))
    ).hexdigest()
    q_before = hashlib.sha256(
        b"".join(_tsha(q_levels[n]).encode() for n in sorted(q_levels))
    ).hexdigest()
    loop_out = {
        "acc": acc,
        "episode_start": episode_start,
        "flip_count": flip_count,
        "lifetimes": [1],
        "credited_mass": 1,
        "n_flips": 5000,
        "q_changed_count": 200,
        "n_applied_drains": 10000,
        "excluded_hit_count": 0,
        "H_trajectory": [
            {
                "step": steps,
                "H_bits_per_weight": 2.0,
                "support": "test",
                "denominator": "acc.numel()",
                "estimator": "shannon_unique_counts",
            }
        ],
        "train_route_counters": {
            "n_fixed_qscale_forwards": 10,
            "n_bitlinear_dynamic_forwards": 0,
            "n_eligible_keys": 32,
            "n_credit_grads_present": 32,
        },
    }
    emitted = assemble_arm_receipt(
        args=argparse.Namespace(
            arm=ARM1,
            steps=steps,
            batch=8,
            topk=1024,
            correctness_smoke=False,
            skip_probes=False,
        ),
        device="cpu",
        sha_before=parent_sha,
        scale_sha_before=scale_before,
        q_sha_before=q_before,
        frozen_scales=frozen_scales,
        q_levels=q_levels,
        ckpt_path=str(ckpt),
        probe_sets=build_phase1_probe_sets(),
        acq_step0=64,
        ret_step0=64,
        acq_final=65,
        ret_final=64,
        loop_out=loop_out,
    )
    emitted["telemetry"] = False
    assert "demand" not in emitted["measurements"]
    assert "deferred_survival" not in emitted["measurements"]
    m = arm_metrics_for_v10_classifier(emitted)
    ok, reason = g0_valid_v10(m)
    assert ok is False
    assert reason is not None
    emitted["banked_sha"] = {
        "before": DEFAULT_PARENT_SHA256,
        "after": DEFAULT_PARENT_SHA256,
        "match": True,
    }
    by_arm = {}
    for arm in (ARM1, ARM2, ARM3):
        r = json.loads(json.dumps(emitted))
        r["arm"] = arm
        by_arm[arm] = r
    with pytest.raises(V10ArmReceiptContractError):
        validate_three_mechanism_arm_receipts_v10(
            by_arm,
            expected_plan_sha256=PLAN_SHA256,
            expected_parent_sha256=DEFAULT_PARENT_SHA256,
            expected_authority_dispatch=AUTHORITY_DISPATCH,
        )
