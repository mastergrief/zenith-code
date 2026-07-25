"""PLAN_v10.1r12 producer / toggle / cost characterization (pre-commit; NO loop entry).

Companion to test_forgetting_screen_v10_1_suppression_branch.py.
Pre-commit STOP: must NOT call run_train_loop / enter q/acc loop (GPU-only rule).
Through-loop proof is packet-stage only.
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import ARM1
from calm.hrm_text_158.native_full_stack.forgetting_screen_pre_post_telemetry import (
    PrePostTransformAccumulator,
)
from calm.hrm_text_158.native_full_stack.forgetting_screen_v10_1_contract import (
    AUTHORITY_DISPATCH_V10_1,
    PLAN_V10_1_SHA256,
    SUPPRESSION_ARM,
    g0_valid_v10_1,
    suppression_diagnostic_match,
)
from calm.hrm_text_158.native_full_stack.screen_receipt_output import assemble_arm_receipt


def _qhex(ch: str = "a") -> str:
    return (ch * 64)[:64]


def _mocked_loop_out(*, pre_post_on: bool) -> dict:
    """Emitted-shape loop_out fixture — NOT produced by run_train_loop."""
    out: dict = {
        "acc": {"w": torch.zeros(4, dtype=torch.int16)},
        "episode_start": {"w": torch.zeros(4, dtype=torch.int32)},
        "flip_count": {"w": torch.zeros(4, dtype=torch.int32)},
        "lifetimes": [],
        "credited_mass": 0,
        "n_flips": 0,
        "q_changed_count": 0,
        "n_applied_drains": 0,
        "batch_rng_base": 1000,
        "excluded_hit_count": 0,
        "H_trajectory": [
            {"step": s, "H_bits_per_weight": 0.0}
            for s in (25, 50, 75, 100, 125, 150)
        ],
        "train_route_counters": {
            "n_fixed_qscale_forwards": 150,
            "n_bitlinear_dynamic_forwards": 0,
        },
        "selection_frames": [],
    }
    if pre_post_on:
        helper = PrePostTransformAccumulator(device="cpu")
        for _ in range(150):
            helper.accumulate_step(
                moves={"w": torch.tensor([1, 0, 1], dtype=torch.int8)},
                acc_pre_decay={"w": torch.tensor([1, 0, 1], dtype=torch.int16)},
                acc_post_decay={"w": torch.tensor([0, 0, 0], dtype=torch.int16)},
                n_cand_after_decay=0,
            )
        out["pre_post_transform"] = helper.finalize()
    return out


def test_receipt_boundary_present_and_absent_no_loop_entry(tmp_path):
    """Pure/mocked boundary: assemble preserves emitted evidence; NO run_train_loop."""
    import calm.hrm_text_158.native_full_stack.screen_execution_loop as sel

    assert not hasattr(test_receipt_boundary_present_and_absent_no_loop_entry, "_called_loop")
    # Guard: this module must not invoke the train loop.
    with mock.patch.object(
        sel,
        "run_train_loop",
        side_effect=AssertionError("pre-commit must not enter run_train_loop"),
    ):
        loop_out = _mocked_loop_out(pre_post_on=True)
        ppt_before = dict(loop_out["pre_post_transform"])
        h_before = list(loop_out["H_trajectory"])
        mass_before = loop_out["credited_mass"]
        flips_before = loop_out["n_flips"]

        ckpt = tmp_path / "dummy.pt"
        ckpt.write_bytes(b"x")
        geom = dict(steps=150, batch=8, topk=1024, device="cpu")
        args_on = SimpleNamespace(
            arm=SUPPRESSION_ARM,
            steps=geom["steps"],
            batch=geom["batch"],
            topk=geom["topk"],
            skip_probes=False,
            correctness_smoke=False,
            pre_post_telemetry=True,
        )
        probe_sets = {
            "acquisition_n": 1,
            "retention_n": 1,
            "acquisition_selection_sha256": "0" * 64,
            "identity_selection_sha256": "0" * 64,
            "math_a0_parent_support_hash": "0" * 16,
            "identity_parent_support_hash": "0" * 16,
        }
        q_final = {"w": torch.zeros(4, dtype=torch.int8)}
        frozen_scales = {"w": torch.ones(1)}
        from calm.hrm_text_158.native_full_stack.screen_receipt_output import _sha_tensor

        q_sha_stable = hashlib.sha256(
            b"".join(_sha_tensor(q_final[n]).encode() for n in sorted(q_final))
        ).hexdigest()
        scale_sha_stable = hashlib.sha256(
            b"".join(
                _sha_tensor(frozen_scales[n]).encode() for n in sorted(frozen_scales)
            )
        ).hexdigest()
        receipt = assemble_arm_receipt(
            args=args_on,
            device=geom["device"],
            sha_before=hashlib.sha256(b"x").hexdigest(),
            scale_sha_before=scale_sha_stable,
            q_sha_before=q_sha_stable,
            frozen_scales=frozen_scales,
            q_levels=q_final,
            ckpt_path=str(ckpt),
            probe_sets=probe_sets,
            acq_step0=64,
            ret_step0=64,
            acq_final=64,
            ret_final=64,
            loop_out=loop_out,
        )
        # Geometry identity + no evidence mutation.
        assert receipt["steps"] == geom["steps"]
        assert receipt["batch"] == geom["batch"]
        assert receipt["topk"] == geom["topk"]
        assert receipt["device"] == geom["device"]
        assert receipt["measurements"]["pre_post_transform"] == ppt_before
        assert receipt["measurements"]["H_trajectory"] == h_before
        assert receipt["measurements"]["credited_mass"] == mass_before
        assert receipt["measurements"]["n_flips"] == flips_before
        assert receipt["pre_post_telemetry"] is True
        # Receipt assembly constants must match contract rebind (requires
        # screen_receipt_output allow-list if still diverged — see SCOPE EXPANSION).
        assert receipt["plan_v10_1_sha256"] == PLAN_V10_1_SHA256
        assert receipt["authority_dispatch_v10_1"] == AUTHORITY_DISPATCH_V10_1
        # Honest CPU geometry cannot clear formal suppression (device≠cuda:0).
        ok_s, reason_s = suppression_diagnostic_match(receipt)
        assert not ok_s and reason_s == "device_not_cuda0"
        ok, _reason, meta = g0_valid_v10_1(receipt)
        assert not ok and meta["branch"] == "none"

        loop_off = _mocked_loop_out(pre_post_on=False)
        assert "pre_post_transform" not in loop_off
        args_off = SimpleNamespace(
            arm=SUPPRESSION_ARM,
            steps=geom["steps"],
            batch=geom["batch"],
            topk=geom["topk"],
            skip_probes=False,
            correctness_smoke=False,
            pre_post_telemetry=False,
        )
        receipt_off = assemble_arm_receipt(
            args=args_off,
            device=geom["device"],
            sha_before=hashlib.sha256(b"x").hexdigest(),
            scale_sha_before=scale_sha_stable,
            q_sha_before=q_sha_stable,
            frozen_scales=frozen_scales,
            q_levels=q_final,
            ckpt_path=str(ckpt),
            probe_sets=probe_sets,
            acq_step0=64,
            ret_step0=64,
            acq_final=64,
            ret_final=64,
            loop_out=loop_off,
        )
        assert receipt_off["pre_post_telemetry"] is False
        assert "pre_post_transform" not in receipt_off["measurements"]
        ok2, reason2 = suppression_diagnostic_match(receipt_off)
        assert not ok2
        assert reason2 == "device_not_cuda0" or "pre_post_telemetry" in str(reason2)


def test_module_does_not_call_run_train_loop():
    """Static guard: no run_train_loop call expression in this pre-commit module."""
    import ast

    tree = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = None
            if isinstance(func, ast.Name):
                name = func.id
            elif isinstance(func, ast.Attribute):
                name = func.attr
            assert name != "run_train_loop", "pre-commit must not call run_train_loop"


def test_cli_main_pre_post_telemetry_parse_and_receipt_propagation(tmp_path, monkeypatch):
    """Real main() argv parse + receipt.pre_post_telemetry propagation (no toy parser)."""
    cli_path = (
        Path(__file__).resolve().parents[3]
        / "scripts"
        / "hrm_text_158_forgetting_mechanism_screen.py"
    )
    spec = importlib.util.spec_from_file_location("forget_mech_cli_main", cli_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    ckpt = tmp_path / "ckpt.pt"
    ckpt.write_bytes(b"x")
    captured: dict = {}

    def fake_run_arm_screen(args):
        captured["parsed"] = bool(args.pre_post_telemetry)
        loop_out = {
            "acc": {"w": torch.zeros(2, dtype=torch.int16)},
            "episode_start": {"w": torch.zeros(2, dtype=torch.int32)},
            "flip_count": {"w": torch.zeros(2, dtype=torch.int32)},
            "lifetimes": [],
            "credited_mass": 0,
            "n_flips": 0,
            "q_changed_count": 0,
            "n_applied_drains": 0,
            "batch_rng_base": 1000,
            "excluded_hit_count": 0,
            "H_trajectory": [],
            "train_route_counters": {
                "n_fixed_qscale_forwards": 1,
                "n_bitlinear_dynamic_forwards": 0,
            },
            "selection_frames": [],
        }
        receipt = assemble_arm_receipt(
            args=args,
            device=str(args.device),
            sha_before="a" * 64,
            scale_sha_before="b" * 64,
            q_sha_before="c" * 64,
            frozen_scales={"w": torch.ones(1)},
            q_levels={"w": torch.zeros(2, dtype=torch.int8)},
            ckpt_path=str(ckpt),
            probe_sets={
                "acquisition_n": 0,
                "retention_n": 0,
                "acquisition_selection_sha256": "0" * 64,
                "identity_selection_sha256": "0" * 64,
                "math_a0_parent_support_hash": "0" * 16,
                "identity_parent_support_hash": "0" * 16,
            },
            acq_step0=None,
            ret_step0=None,
            acq_final=None,
            ret_final=None,
            loop_out=loop_out,
        )
        captured["receipt_pre_post"] = receipt["pre_post_telemetry"]
        return 0

    monkeypatch.setattr(mod, "run_arm_screen", fake_run_arm_screen)

    cases = [
        ([str(cli_path), "--ckpt-path", str(ckpt), "--skip-probes"], True),
        (
            [
                str(cli_path),
                "--ckpt-path",
                str(ckpt),
                "--skip-probes",
                "--no-pre-post-telemetry",
            ],
            False,
        ),
        (
            [
                str(cli_path),
                "--ckpt-path",
                str(ckpt),
                "--skip-probes",
                "--pre-post-telemetry",
            ],
            True,
        ),
    ]
    for argv, expected in cases:
        monkeypatch.setattr(sys, "argv", argv)
        assert mod.main() == 0
        assert captured["parsed"] is expected
        assert captured["receipt_pre_post"] is expected


def test_observer_cost_scale_smoke():
    acc = PrePostTransformAccumulator(device="cpu")
    pre = {"w": torch.randint(-2, 3, (1024,), dtype=torch.int16)}
    post = {"w": torch.zeros(1024, dtype=torch.int16)}
    moves = {"w": torch.randint(-1, 2, (1024,), dtype=torch.int8)}
    t0 = time.perf_counter()
    for _ in range(32):
        acc.accumulate_step(
            moves=moves, acc_pre_decay=pre, acc_post_decay=post, n_cand_after_decay=0
        )
    out = acc.finalize()
    dt = time.perf_counter() - t0
    assert out["steps_accumulated"] == 32
    assert dt >= 0.0
    print(f"[cost-smoke] accumulate+finalize steps=32 n=1024 dt_s={dt:.6f}")
