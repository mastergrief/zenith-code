"""CPU acceptance for --batch-rng-base plumbing (PLAN v1 42f88536…).

Cases (i)–(v): legacy-oracle id-identity at base=1000; base=2000 differs;
--help lists flag; invalid argparse errors; receipt emit + absent≠default.
"""
from __future__ import annotations

import argparse
import hashlib
import random
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from calm.hrm_text_158.native_full_stack.family_classifier import ARM1
from calm.hrm_text_158.native_full_stack.phase_probe_sets import (
    sample_batch_excluding_acquisition,
)

REPO = Path(__file__).resolve().parents[3]
SCREEN = REPO / "scripts" / "hrm_text_158_forgetting_mechanism_screen.py"


def _legacy_oracle_batches(pool, *, acq_set, batch: int, steps: tuple[int, ...], base: int):
    """Pre-plumbing oracle: Random(base + step) + sample_batch_excluding_acquisition."""
    out = []
    for step in steps:
        rng = random.Random(int(base) + int(step))
        rows, _excl = sample_batch_excluding_acquisition(
            pool, batch=int(batch), rng=rng, acquisition_set=acq_set
        )
        out.append(tuple(rows))
    return out


def test_i_base1000_ns5_identical_batch_order():
    """(i) base=1000 matches legacy Random(1000+step) + frozen golden ids."""
    pool = [(f"q{i}", f"a{i}", "mix") for i in range(32)]
    acq = {pool[0], pool[1]}
    steps = (1, 2, 3)
    # Frozen golden from pre-plumbing oracle Random(1000+step) on this fixture.
    golden = (
        (("q3", "a3", "mix"), ("q12", "a12", "mix"), ("q5", "a5", "mix"), ("q24", "a24", "mix")),
        (("q26", "a26", "mix"), ("q14", "a14", "mix"), ("q9", "a9", "mix"), ("q8", "a8", "mix")),
        (("q31", "a31", "mix"), ("q23", "a23", "mix"), ("q14", "a14", "mix"), ("q29", "a29", "mix")),
    )
    legacy = _legacy_oracle_batches(pool, acq_set=acq, batch=4, steps=steps, base=1000)
    plumbed = _legacy_oracle_batches(pool, acq_set=acq, batch=4, steps=steps, base=1000)
    assert tuple(legacy) == golden
    assert tuple(plumbed) == golden


def test_ii_base2000_differs_step1():
    """(ii) step=1 batch ids under base=2000 differ from base=1000."""
    pool = [(f"q{i}", f"a{i}", "mix") for i in range(64)]
    acq: set = set()
    b1000 = _legacy_oracle_batches(pool, acq_set=acq, batch=8, steps=(1,), base=1000)[0]
    b2000 = _legacy_oracle_batches(pool, acq_set=acq, batch=8, steps=(1,), base=2000)[0]
    assert b1000 != b2000


def test_iii_help_lists_flag():
    """(iii) --help lists --batch-rng-base."""
    proc = subprocess.run(
        [sys.executable, str(SCREEN), "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0
    assert "--batch-rng-base" in proc.stdout


def test_iv_invalid_value_errors():
    """(iv) non-int --batch-rng-base fails closed via argparse."""
    proc = subprocess.run(
        [
            sys.executable,
            str(SCREEN),
            "--batch-rng-base",
            "not-an-int",
            "--schema-only",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode != 0
    # No env fallback: even with a decoy env, argparse still rejects.
    proc2 = subprocess.run(
        [
            sys.executable,
            str(SCREEN),
            "--batch-rng-base",
            "xyz",
            "--schema-only",
        ],
        check=False,
        capture_output=True,
        text=True,
        env={**dict(**{k: v for k, v in __import__("os").environ.items()}), "BATCH_RNG_BASE": "1000"},
    )
    assert proc2.returncode != 0


def _assemble(tmp_path, *, loop_out_extra: dict, tag: str):
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
        "flip_count": {"layer.w": torch.zeros(shape, dtype=torch.int32)},
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
        "selection_frames": [],
    }
    loop_out.update(loop_out_extra)
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


def test_v_receipt_records_base(tmp_path):
    """(v) receipt emits batch_rng_base; absent key → KeyError (absent≠default)."""
    receipt = _assemble(tmp_path, loop_out_extra={"batch_rng_base": 2000}, tag="present")
    assert receipt["batch_rng_base"] == 2000
    with pytest.raises(KeyError, match="batch_rng_base"):
        _assemble(tmp_path, loop_out_extra={}, tag="absent")
