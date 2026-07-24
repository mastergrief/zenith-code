"""Per-index identity fields on forgetting-mechanism screen receipts.

Covers flip_count_sha256 + applied_identity_sha256 emitted by
assemble_arm_receipt (telemetry-independent). Bound by seam defect-cycle
1784897720385 (gate-2 architecture repair: keep toggle suite free of
identity-suite growth).
"""
from __future__ import annotations

import argparse
import hashlib

import torch

from calm.hrm_text_158.native_full_stack.family_classifier import ARM1


def _cpu_assemble_fixture(tmp_path, *, flip_count, selection_frames, tag: str):
    """Minimal CPU assemble_arm_receipt inputs for per-index identity tests."""
    from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
        build_phase1_probe_sets,
    )
    from calm.hrm_text_158.native_full_stack.screen_receipt_output import (
        assemble_arm_receipt,
    )

    shape = (4, 4)
    q_levels = {"layer.w": torch.zeros(shape, dtype=torch.int8)}
    frozen_scales = {"layer.w": torch.ones(shape, dtype=torch.float32)}
    ckpt = tmp_path / f"{tag}.pt"
    ckpt.write_bytes(tag.encode())
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
        "acc": {"layer.w": torch.zeros(shape, dtype=torch.int16)},
        "episode_start": {"layer.w": torch.zeros(shape, dtype=torch.int32)},
        "flip_count": flip_count,
        "lifetimes": [],
        "credited_mass": 0,
        "n_flips": 0,
        "q_changed_count": 0,
        "n_applied_drains": 0,
        "excluded_hit_count": 0,
        "H_trajectory": [
            {
                "step": 2,
                "H_bits_per_weight": 0.0,
                "support": "test",
                "denominator": "acc.numel()",
                "estimator": "shannon_unique_counts",
            }
        ],
        "train_route_counters": {
            "n_fixed_qscale_forwards": 1,
            "n_bitlinear_dynamic_forwards": 0,
            "n_eligible_keys": 1,
            "n_credit_grads_present": 1,
        },
        "selection_frames": selection_frames,
        # No pressure_telemetry — proves hashes independent of telemetry.
    }
    return assemble_arm_receipt(
        args=argparse.Namespace(
            arm=ARM1,
            steps=2,
            batch=1,
            topk=8,
            correctness_smoke=False,
            skip_probes=True,
        ),
        device="cpu",
        sha_before=parent_sha,
        scale_sha_before=scale_before,
        q_sha_before=q_before,
        frozen_scales=frozen_scales,
        q_levels=q_levels,
        ckpt_path=str(ckpt),
        probe_sets=build_phase1_probe_sets(),
        acq_step0=0,
        ret_step0=0,
        acq_final=0,
        ret_final=0,
        loop_out=loop_out,
    )


def test_per_index_identity_hashes_present_without_telemetry(tmp_path):
    """flip_count_sha256 + applied_identity_sha256 emit with no pressure_telemetry."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        ordered_selection_frame,
    )

    fc = {"layer.w": torch.zeros((4, 4), dtype=torch.int32)}
    frames = [
        ordered_selection_frame(
            step=1, ordered_flat_idx=torch.tensor([0, 1], dtype=torch.int64)
        ),
        ordered_selection_frame(
            step=2, ordered_flat_idx=torch.tensor([2], dtype=torch.int64)
        ),
    ]
    r = _cpu_assemble_fixture(
        tmp_path, flip_count=fc, selection_frames=frames, tag="pi-present"
    )
    assert "demand" not in r["measurements"]
    assert len(r["flip_count_sha256"]) == 64
    assert len(r["applied_identity_sha256"]) == 64
    assert r["applied_identity_ordering"]["sequence"] == (
        "append_per_step_in_train_loop_order"
    )


def test_per_index_identity_deterministic_cpu_fixture(tmp_path):
    """Identical flip_count + selection_frames → identical hashes across assembles."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        ordered_selection_frame,
    )

    fc = {"layer.w": torch.arange(16, dtype=torch.int32).reshape(4, 4)}
    frames = [
        ordered_selection_frame(
            step=1, ordered_flat_idx=torch.tensor([3, 1, 4], dtype=torch.int64)
        ),
        ordered_selection_frame(
            step=2, ordered_flat_idx=torch.tensor([0], dtype=torch.int64)
        ),
    ]
    a = _cpu_assemble_fixture(
        tmp_path, flip_count=fc, selection_frames=frames, tag="pi-det-a"
    )
    b = _cpu_assemble_fixture(
        tmp_path,
        flip_count={"layer.w": fc["layer.w"].clone()},
        selection_frames=list(frames),
        tag="pi-det-b",
    )
    assert a["flip_count_sha256"] == b["flip_count_sha256"]
    assert a["applied_identity_sha256"] == b["applied_identity_sha256"]


def test_per_index_identity_differs_when_applied_indices_differ(tmp_path):
    """Negative: different ordered selection indices → different applied_identity_sha256."""
    from calm.hrm_text_158.native_full_stack.pressure_metric_gpu_loop_bridge import (
        ordered_selection_frame,
    )

    fc = {"layer.w": torch.zeros((4, 4), dtype=torch.int32)}
    frames_a = [
        ordered_selection_frame(
            step=1, ordered_flat_idx=torch.tensor([0, 1], dtype=torch.int64)
        ),
    ]
    frames_b = [
        ordered_selection_frame(
            step=1, ordered_flat_idx=torch.tensor([0, 2], dtype=torch.int64)
        ),
    ]
    a = _cpu_assemble_fixture(
        tmp_path, flip_count=fc, selection_frames=frames_a, tag="pi-neg-a"
    )
    b = _cpu_assemble_fixture(
        tmp_path,
        flip_count={"layer.w": fc["layer.w"].clone()},
        selection_frames=frames_b,
        tag="pi-neg-b",
    )
    assert a["flip_count_sha256"] == b["flip_count_sha256"]
    assert a["applied_identity_sha256"] != b["applied_identity_sha256"]
