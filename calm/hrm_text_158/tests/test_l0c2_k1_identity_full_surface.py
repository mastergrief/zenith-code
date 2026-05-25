"""Full-density L0c2-K1 2-digit identity emission-primitive tests.

This sibling rung is a TRAIN-only 90-row acquisition surface over
`<n> equals what?`, n=10..99. It intentionally has no held-out split:
acquisition is measured by a separate 90/90 coverage audit, while trainer
validation uses replay-prior held rows as a retention/dev signal.
"""
from __future__ import annotations

import os
import re
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import calm.hrm_text_158.curriculum as curriculum_pkg  # noqa: E402
from calm.hrm_text_158 import (  # noqa: E402
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.curriculum import BroadTokenizer  # noqa: E402
from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    L0C2K1_IDENTITY_FULL_TRAIN_COUNT,
    L0C2K1_IDENTITY_HELD_COUNT,
    L0C2K1_IDENTITY_TOTAL_COUNT,
    L0C2K1_IDENTITY_TRAIN_COUNT,
    RUNG_NAMES,
    _RUNG_SPEC,
    _enumerate_partition_l0c2k1_identity,
    _l0c2k1_identity_full_enumerate,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    L0C2K1_IDENTITY_FULL_AUDIT_EXPECTED_COUNT,
    build_l0c2k1_identity_full_support,
    language_source_rung_buckets,
)
from calm.hrm_text_158.curriculum.replay import DIAGNOSIS_ONLY_RUNGS  # noqa: E402
from scripts.train_hrm_text_158 import _build_ckpt_config, SOURCE_PIN, train  # noqa: E402

FULL_RUNG = "L0c2-K1-identity-2digit-full"
DIAGNOSTIC_RUNG = "L0c2-K1-identity-2digit"

TINY_ARCH = dict(
    hidden_size=64,
    n_layers=2,
    num_heads=2,
    expansion=4,
    H_cycles=1,
    L_cycles=1,
    half_layers=True,
    bp_warmup_ratio=0.2,
    bp_min_steps=1,
    bp_max_steps=2,
    max_len=64,
)

EXPECTED_SEED17_DIAGNOSTIC_HELD_SAMPLE = [
    {"question": "51 equals what?", "expected": 51, "rung": DIAGNOSTIC_RUNG},
    {"question": "63 equals what?", "expected": 63, "rung": DIAGNOSTIC_RUNG},
    {"question": "38 equals what?", "expected": 38, "rung": DIAGNOSTIC_RUNG},
    {"question": "85 equals what?", "expected": 85, "rung": DIAGNOSTIC_RUNG},
    {"question": "17 equals what?", "expected": 17, "rung": DIAGNOSTIC_RUNG},
    {"question": "69 equals what?", "expected": 69, "rung": DIAGNOSTIC_RUNG},
    {"question": "78 equals what?", "expected": 78, "rung": DIAGNOSTIC_RUNG},
    {"question": "57 equals what?", "expected": 57, "rung": DIAGNOSTIC_RUNG},
    {"question": "93 equals what?", "expected": 93, "rung": DIAGNOSTIC_RUNG},
    {"question": "43 equals what?", "expected": 43, "rung": DIAGNOSTIC_RUNG},
    {"question": "93 equals what?", "expected": 93, "rung": DIAGNOSTIC_RUNG},
    {"question": "87 equals what?", "expected": 87, "rung": DIAGNOSTIC_RUNG},
    {"question": "28 equals what?", "expected": 28, "rung": DIAGNOSTIC_RUNG},
    {"question": "38 equals what?", "expected": 38, "rung": DIAGNOSTIC_RUNG},
    {"question": "11 equals what?", "expected": 11, "rung": DIAGNOSTIC_RUNG},
    {"question": "43 equals what?", "expected": 43, "rung": DIAGNOSTIC_RUNG},
    {"question": "11 equals what?", "expected": 11, "rung": DIAGNOSTIC_RUNG},
    {"question": "49 equals what?", "expected": 49, "rung": DIAGNOSTIC_RUNG},
    {"question": "13 equals what?", "expected": 13, "rung": DIAGNOSTIC_RUNG},
    {"question": "16 equals what?", "expected": 16, "rung": DIAGNOSTIC_RUNG},
]


def _qe(rows):
    return {(r["question"], r["expected"]) for r in rows}


def _build_tiny_parent_blob() -> dict:
    tok = BroadTokenizer()
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=TINY_ARCH["max_len"],
        n_layers=TINY_ARCH["n_layers"],
        hidden_size=TINY_ARCH["hidden_size"],
        num_heads=TINY_ARCH["num_heads"],
        expansion=TINY_ARCH["expansion"],
        H_cycles=TINY_ARCH["H_cycles"],
        L_cycles=TINY_ARCH["L_cycles"],
        half_layers=TINY_ARCH["half_layers"],
        bp_warmup_ratio=TINY_ARCH["bp_warmup_ratio"],
        bp_min_steps=TINY_ARCH["bp_min_steps"],
        bp_max_steps=TINY_ARCH["bp_max_steps"],
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))
    config_blob = _build_ckpt_config(
        m, tok, cfg, TINY_ARCH["max_len"], batch_size=4,
        curriculum_rung="L0c1", curriculum_seed=17,
        replay_ratio=0.0, prior_rungs=[],
    )
    return {
        "model_state": m.state_dict(),
        "config": config_blob,
        "step": 50,
        "epoch": 1,
        "source_pin": SOURCE_PIN,
    }


def test_full_identity_rung_registered_two_places_and_diagnosis_only():
    assert FULL_RUNG in RUNG_NAMES
    assert FULL_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[FULL_RUNG]) == {"train"}
    assert FULL_RUNG in DIAGNOSIS_ONLY_RUNGS

    train_src = Path(_REPO, "scripts", "train_hrm_text_158.py").read_text(encoding="utf-8")
    m = re.search(r'"--curriculum-rung"[^)]*?choices=\[([^\]]*)\]', train_src, re.DOTALL)
    assert m, "could not locate --curriculum-rung choices= in train script"
    choices = set(re.findall(r'"([^"]+)"', m.group(1)))
    assert choices == set(RUNG_NAMES) - {"R7"}
    assert FULL_RUNG in choices


def test_full_identity_enumerates_all_90_rows_and_has_no_held_out_surface():
    rows = _l0c2k1_identity_full_enumerate()
    assert len(rows) == L0C2K1_IDENTITY_FULL_TRAIN_COUNT == L0C2K1_IDENTITY_TOTAL_COUNT == 90
    assert len(_qe(rows)) == 90
    assert {r["n"] for r in rows} == set(range(10, 100))
    assert all(r["expected"] == r["n"] for r in rows)
    assert all(r["question"] == f"{r['n']} equals what?" for r in rows)
    assert all(" plus " not in r["question"] and " minus " not in r["question"] for r in rows)
    assert all("hold_kind" not in r for r in rows)

    bucket_counts = Counter(r["coverage_bucket"] for r in rows)
    assert bucket_counts["coverage_teen"] == 10
    for tens in range(2, 10):
        assert bucket_counts[f"coverage_tens_{tens}"] == 10

    train_rows = make_rung_examples(FULL_RUNG, 90, seed=17, split="train")
    assert len(train_rows) == 90
    assert len(_qe(train_rows)) == 90
    assert _qe(train_rows) == _qe(rows)
    assert all(r["rung"] == FULL_RUNG for r in train_rows)

    with pytest.raises(ValueError, match="TRAIN-only|held_out"):
        make_rung_examples(FULL_RUNG, 1, seed=17, split="held_out")


def test_full_identity_coverage_support_is_single_90_row_surface():
    support = build_l0c2k1_identity_full_support(17)
    assert list(support) == [FULL_RUNG]
    rows = support[FULL_RUNG]
    assert len(rows) == L0C2K1_IDENTITY_FULL_AUDIT_EXPECTED_COUNT == 90
    assert len({(q, e) for q, e, _bucket in rows}) == 90
    assert {e for _q, e, _bucket in rows} == set(range(10, 100))

    declared = set(language_source_rung_buckets(FULL_RUNG))
    present = Counter(bucket for _q, _e, bucket in rows)
    assert set(present) == declared
    assert present["coverage_teen"] == 10
    for tens in range(2, 10):
        assert present[f"coverage_tens_{tens}"] == 10
    assert all(not key.endswith("-train") and not key.endswith("-held") for key in support)


def test_existing_70_20_identity_diagnostic_rung_is_preserved():
    assert set(_RUNG_SPEC[DIAGNOSTIC_RUNG]) == {"train", "held_out"}
    train_rows, held_rows = _enumerate_partition_l0c2k1_identity(17)
    assert len(train_rows) == L0C2K1_IDENTITY_TRAIN_COUNT == 70
    assert len(held_rows) == L0C2K1_IDENTITY_HELD_COUNT == 20
    assert _qe(train_rows).isdisjoint(_qe(held_rows))
    assert {("11 equals what?", 11), ("17 equals what?", 17)} <= _qe(held_rows)
    assert make_rung_examples(
        DIAGNOSTIC_RUNG, 20, seed=17, split="held_out"
    ) == EXPECTED_SEED17_DIAGNOSTIC_HELD_SAMPLE


def test_full_identity_trainer_val_uses_prior_held_rows_not_active_held(
    monkeypatch, tmp_path: Path, capsys
):
    parent_path = tmp_path / "parent_L0c1_final.pt"
    torch.save(_build_tiny_parent_blob(), parent_path)

    calls: list[tuple[str, int, int, str]] = []
    original_make = curriculum_pkg.make_rung_examples

    def wrapped_make_rung_examples(rung: str, n: int, seed: int = 42, split: str = "train"):
        calls.append((rung, n, seed, split))
        return original_make(rung, n=n, seed=seed, split=split)

    monkeypatch.setattr(curriculum_pkg, "make_rung_examples", wrapped_make_rung_examples)

    train(
        curriculum_rung=FULL_RUNG,
        use_broad_tokenizer=True,
        curriculum_n_train=12,
        curriculum_n_heldout=6,
        replay_ratio=0.0,
        replay_rungs="R0,R1,R1b1",
        load_from=str(parent_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(tmp_path / "full_identity_best.pt"),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out

    assert (FULL_RUNG, 12, 42, "train") in calls
    assert not any(rung == FULL_RUNG and split == "held_out" for rung, _n, _seed, split in calls)
    assert [call for call in calls if call[3] == "held_out"] == [
        ("R0", 2, 42, "held_out"),
        ("R1", 2, 42, "held_out"),
        ("R1b1", 2, 42, "held_out"),
    ]
    assert "train-only active_held_out=0" in out
    assert "prior_held_val={'R0': 2, 'R1': 2, 'R1b1': 2}" in out
    assert "dry-run: EXITING before optimizer step" in out
