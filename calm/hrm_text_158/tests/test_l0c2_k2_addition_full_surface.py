"""STEP 1 L0c2-K2 addition-full acquisition + heldout-50s diagnostic support.

No model/GPU: this locks the finite support partitions so the next launch can
train only the acquisition surface while auditing the trained-OUT 50s transfer
diagnostic separately.
"""
from __future__ import annotations

from collections import Counter
import importlib.util
import os
from pathlib import Path
import re
import sys

import pytest
import torch

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    RUNG_NAMES,
    _RUNG_SPEC,
    _l0c2k2_addition_full_enumerate,
    _l0c2k2_addition_120_enumerate,
    _l0c2k2_addition_120_k5to8_enumerate,
    _l0c2k2_addition_50s_enumerate,
    _l0c2k2_addition_60s_trace_held_enumerate,
    _l0c2k2_addition_60s_trace_train_enumerate,
    _l0c2k2_addition_trace_target,
    _l0c2k2_addition_60s_transfer_held_enumerate,
    _l0c2k2_addition_60s_transfer_train_enumerate,
    make_rung_examples,
)
from calm.hrm_text_158.curriculum.language_supports import (  # noqa: E402
    LANGUAGE_ACTIVE_RUNGS,
    LANGUAGE_EXPECTED_AGGREGATE,
    L0C1_CLOSE_SIBLING_CE_INTERLEAVE_EXPECTED_COUNT,
    L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT,
    L0C2K2_ADDITION_FULL_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_120_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_120_K5TO8_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_50S_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_60S_TRACE_HELD_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_60S_TRACE_TRAIN_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_60S_TRANSFER_HELD_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_60S_TRANSFER_TRAIN_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_HELDOUT_50S_AUDIT_EXPECTED_COUNT,
    L0C2K2_ADDITION_HELDOUT_60S_AUDIT_EXPECTED_COUNT,
    build_l0c1_close_sibling_ce_interleave_support,
    build_l0c2k2_addition_full_support,
    build_l0c2k2_addition_120_support,
    build_l0c2k2_addition_120_k5to8_support,
    build_l0c2k2_addition_50s_support,
    build_l0c2k2_addition_60s_trace_held_support,
    build_l0c2k2_addition_60s_trace_train_support,
    build_l0c2k2_addition_60s_transfer_held_support,
    build_l0c2k2_addition_60s_transfer_train_support,
    build_l0c2k2_addition_heldout_50s_support,
    build_l0c2k2_addition_heldout_60s_support,
    build_language_supports,
    language_source_rung_buckets,
)
from calm.hrm_text_158.curriculum.replay import DIAGNOSIS_ONLY_RUNGS  # noqa: E402
import calm.hrm_text_158.curriculum as curriculum_pkg  # noqa: E402
from calm.hrm_text_158 import (  # noqa: E402
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.curriculum import BroadTokenizer  # noqa: E402

_TRAIN_SPEC = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_TRAIN = importlib.util.module_from_spec(_TRAIN_SPEC)
_TRAIN_SPEC.loader.exec_module(_TRAIN)

_PROBE_SPEC = importlib.util.spec_from_file_location(
    "_probe_hrm_text_158", os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
)
_PROBE = importlib.util.module_from_spec(_PROBE_SPEC)
_PROBE_SPEC.loader.exec_module(_PROBE)

FULL_RUNG = "L0c2-K2-addition-full"
K120_RUNG = "L0c2-K2-addition-120"
K120_K5TO8_RUNG = "L0c2-K2-addition-120-k5to8"
FIFTIES_RUNG = "L0c2-K2-addition-50s"
SIXTIES_TRANSFER_RUNG = "L0c2-K2-addition-60s-transfer"
SIXTIES_TRANSFER_TRAIN = "L0c2-K2-addition-60s-transfer-train"
SIXTIES_TRANSFER_HELD = "L0c2-K2-addition-60s-transfer-held"
SIXTIES_TRACE_RUNG = "L0c2-K2-addition-60s-trace"
SIXTIES_TRACE_TRAIN = "L0c2-K2-addition-60s-trace-train"
SIXTIES_TRACE_HELD = "L0c2-K2-addition-60s-trace-held"
HELDOUT_DIAG = "L0c2-K2-addition-heldout-50s"
HELDOUT_60S_DIAG = "L0c2-K2-addition-heldout-60s"
_PLUS_RE = re.compile(r"^(\d+) plus ([1-8]) equals what\?$")
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


def _flat(supports: dict[str, list[tuple[str, int, str]]], key: str) -> list[tuple[str, int, str]]:
    assert list(supports.keys()) == [key]
    return supports[key]


def _parse_rows(rows: list[tuple[str, int, str]]) -> list[tuple[int, int, int, str]]:
    out = []
    for q, expected, bucket in rows:
        m = _PLUS_RE.fullmatch(q)
        assert m, q
        a = int(m.group(1))
        k = int(m.group(2))
        assert a + k == expected
        out.append((a, k, expected, bucket))
    return out


def _parse_trace_rows(rows: list[tuple[str, str, str]]) -> list[tuple[int, int, int, str, str]]:
    out = []
    for q, expected, bucket in rows:
        m = _PLUS_RE.fullmatch(q)
        assert m, q
        a = int(m.group(1))
        k = int(m.group(2))
        answer = _PROBE._parse_trace_answer(expected)
        assert answer == a + k
        assert expected == _l0c2k2_addition_trace_target(a, k, answer)
        out.append((a, k, answer, bucket, expected))
    return out


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
    model = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))
    config_blob = _TRAIN._build_ckpt_config(
        model, tok, cfg, TINY_ARCH["max_len"], batch_size=4,
        curriculum_rung="L0c1", curriculum_seed=17,
        replay_ratio=0.0, prior_rungs=[],
    )
    return {
        "model_state": model.state_dict(),
        "config": config_blob,
        "step": 50,
        "epoch": 1,
        "source_pin": _TRAIN.SOURCE_PIN,
    }


def test_trainable_full_rung_registered_but_heldout_diagnostic_absent():
    assert FULL_RUNG in RUNG_NAMES
    assert FULL_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[FULL_RUNG]) == {"train"}
    assert FULL_RUNG in DIAGNOSIS_ONLY_RUNGS

    assert HELDOUT_DIAG not in RUNG_NAMES
    assert HELDOUT_DIAG not in _RUNG_SPEC
    assert HELDOUT_DIAG not in DIAGNOSIS_ONLY_RUNGS
    assert HELDOUT_DIAG not in _TRAIN._RETAINED_SUPPORT_REGISTRY
    assert FULL_RUNG not in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_full_rung_only():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{FULL_RUNG}"' in src
    assert HELDOUT_DIAG not in src


def test_full_generator_is_train_only_and_tags_rows():
    rows = make_rung_examples(FULL_RUNG, 240, seed=17, split="train")
    assert len(rows) == 240
    assert len({(r["question"], r["expected"]) for r in rows}) == 240
    assert all(r["rung"] == FULL_RUNG for r in rows)
    assert {r["expected"] for r in rows} == set(range(20, 50))
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(FULL_RUNG, 10, seed=17, split="held_out")
    with pytest.raises(ValueError, match="unknown rung"):
        make_rung_examples(HELDOUT_DIAG, 10, seed=17, split="train")


def test_addition_full_trainer_val_uses_prior_held_rows_not_active_held(
    monkeypatch, tmp_path: Path, capsys
):
    parent_path = tmp_path / "parent_L0c1_final.pt"
    torch.save(_build_tiny_parent_blob(), parent_path)

    calls: list[tuple[str, int, int, str]] = []
    original_make = curriculum_pkg.make_rung_examples

    def wrapped_make_rung_examples(rung: str, n: int, seed: int = 42, split: str = "train"):
        calls.append((rung, n, seed, split))
        if rung == FULL_RUNG and split == "held_out":
            raise AssertionError("trainer must not request active held_out for a train-only rung")
        return original_make(rung, n=n, seed=seed, split=split)

    monkeypatch.setattr(curriculum_pkg, "make_rung_examples", wrapped_make_rung_examples)

    _TRAIN.train(
        curriculum_rung=FULL_RUNG,
        use_broad_tokenizer=True,
        curriculum_n_train=12,
        curriculum_n_heldout=6,
        replay_ratio=0.0,
        replay_rungs="R0,R1,R1b1",
        load_from=str(parent_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(tmp_path / "addition_full_best.pt"),
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


def test_ce_interleave_dry_run_injects_rows(monkeypatch, tmp_path: Path, capsys):
    """STEP 2a [runtime] proof: a CPU dry-run with --ce-interleave-support injects
    exactly 13*REPEAT true-label CE rows AFTER the curriculum cap/log, and exits
    before the optimizer with no checkpoint written. Tiny temp parent fixture; no
    real chain-head .pt, no launch."""
    parent_path = tmp_path / "parent_L0c1_final.pt"
    torch.save(_build_tiny_parent_blob(), parent_path)
    ckpt_path = tmp_path / "ce_interleave_best.pt"
    repeat = 3

    _TRAIN.train(
        curriculum_rung=FULL_RUNG,
        use_broad_tokenizer=True,
        curriculum_n_train=12,
        curriculum_n_heldout=6,
        replay_ratio=0.0,
        replay_rungs="R0,R1,R1b1",
        ce_interleave_support=[f"{L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT}:{repeat}"],
        load_from=str(parent_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out

    # CE-interleave injected exactly 13*repeat rows, by the named support.
    assert "[hrm158] ce-interleave:" in out
    assert f"ce_rows_added={13 * repeat}" in out
    assert L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT in out
    # Ordering: curriculum cap/log fires BEFORE the CE append (acquisition mix unchanged).
    assert out.index("[hrm158] curriculum") < out.index("[hrm158] ce-interleave")
    # Dry-run exits before optimizer; no checkpoint written to repo/tmp scope.
    assert "dry-run: EXITING before optimizer step" in out
    assert not ckpt_path.exists()


def test_addition_full_support_count_echo_exclusions_and_marginals():
    rows = _flat(build_l0c2k2_addition_full_support(), FULL_RUNG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_FULL_AUDIT_EXPECTED_COUNT == 240
    assert len({(q, e) for q, e, _bucket in rows}) == 240
    assert all(" plus 0 " not in q and not q.startswith("0 plus ") for q, _e, _b in rows)
    assert all(" minus " not in q and " equals what?" in q for q, _e, _b in rows)
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)

    assert Counter(expected // 10 for _a, _k, expected, _b in parsed) == {2: 80, 3: 80, 4: 80}
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 30 for k in range(1, 9)}
    assert Counter((a % 10) + k >= 10 for a, k, _expected, _b in parsed) == {False: 132, True: 108}
    assert Counter(expected % 10 for _a, _k, expected, _b in parsed) == {ones: 24 for ones in range(10)}


def test_heldout_50s_is_audit_visible_non_gating_support_only():
    rows = _flat(build_l0c2k2_addition_heldout_50s_support(), HELDOUT_DIAG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_HELDOUT_50S_AUDIT_EXPECTED_COUNT == 80
    assert {expected for _a, _k, expected, _bucket in parsed} == set(range(50, 60))
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 10 for k in range(1, 9)}
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
    assert HELDOUT_DIAG not in LANGUAGE_ACTIVE_RUNGS


def test_language_audit_paths_cover_declared_buckets_without_changing_aggregate():
    for key, support in (
        (FULL_RUNG, build_l0c2k2_addition_full_support()),
        (HELDOUT_DIAG, build_l0c2k2_addition_heldout_50s_support()),
    ):
        rows = _flat(support, key)
        present = {bucket for _q, _e, bucket in rows}
        declared = set(language_source_rung_buckets(key))
        assert present == declared
        assert all(bucket.count(":") == 3 for bucket in present)

    assert LANGUAGE_ACTIVE_RUNGS == ("L0a", "L0b", "L0c")
    assert sum(len(v) for v in build_language_supports().values()) == 690
    assert LANGUAGE_EXPECTED_AGGREGATE == 690


def test_full_enumerator_metadata_matches_support_contract():
    rows = _l0c2k2_addition_full_enumerate()
    assert len(rows) == 240
    assert Counter(r["result_decade"] for r in rows) == {"20s": 80, "30s": 80, "40s": 80}
    assert Counter(r["addend_k"] for r in rows) == {f"k_{k}": 30 for k in range(1, 9)}
    assert Counter(r["carry"] for r in rows) == {"no_carry": 132, "carry": 108}
    assert Counter(r["result_ones"] for r in rows) == {f"ones_{n}": 24 for n in range(10)}


def test_l0c1_close_sibling_ce_support_is_true_label_distinct_namespace():
    support = build_l0c1_close_sibling_ce_interleave_support()
    rows = _flat(support, L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT)
    assert len(rows) == L0C1_CLOSE_SIBLING_CE_INTERLEAVE_EXPECTED_COUNT == 13
    assert {q for q, _e, _bucket in rows if q in {f"{n} equals what?" for n in range(10)}} == {
        f"{n} equals what?" for n in range(10)
    }
    assert ("2 equals what?", 2, "one_digit_identity") in rows
    assert ("11 equals what?", 11, "legacy_identity") in rows
    assert ("17 equals what?", 17, "legacy_identity") in rows
    assert ("99 equals what?", 99, "two_digit_sentinel") in rows
    assert all(q == f"{expected} equals what?" for q, expected, _bucket in rows)
    assert L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT not in _TRAIN._RETAINED_SUPPORT_REGISTRY
    assert set(language_source_rung_buckets(L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT)) == {
        "one_digit_identity",
        "legacy_identity",
        "two_digit_sentinel",
    }


# --------------------------------------------------------------------------- #
# L0c2-K2-addition-120 — 2x-density split (k=1..4 subset of the 240). Sibling
# coverage after the 240-row full surface missed bank (post-hoc step1500 acquire
# 204/240; diffuse residual). Same template/surface; first of codex's two 120
# atoms (k=1..4 then k=5..8).
# --------------------------------------------------------------------------- #

def test_120_rung_registered_train_only_diagnosis_only():
    assert K120_RUNG in RUNG_NAMES
    assert K120_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[K120_RUNG]) == {"train"}
    assert K120_RUNG in DIAGNOSIS_ONLY_RUNGS
    # Banked k=1..4 is now a true retained prior for the sibling k=5..8 retry.
    assert K120_RUNG in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_120_rung():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{K120_RUNG}"' in src


def test_120_generator_train_only_and_exact_240_subset():
    rows = make_rung_examples(K120_RUNG, 120, seed=17, split="train")
    assert len(rows) == 120
    assert len({(r["question"], r["expected"]) for r in rows}) == 120
    assert all(r["rung"] == K120_RUNG for r in rows)
    assert {r["expected"] for r in rows} == set(range(20, 50))
    # exact subset of the 240 (same question/expected, same template)
    full_pairs = {(r["question"], r["expected"]) for r in _l0c2k2_addition_full_enumerate()}
    assert all((r["question"], r["expected"]) in full_pairs for r in rows)
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(K120_RUNG, 10, seed=17, split="held_out")


def test_120_support_count_echo_exclusions_and_marginals():
    rows = _flat(build_l0c2k2_addition_120_support(), K120_RUNG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_120_AUDIT_EXPECTED_COUNT == 120
    assert len({(q, e) for q, e, _bucket in rows}) == 120
    assert all(" plus 0 " not in q and not q.startswith("0 plus ") for q, _e, _b in rows)
    assert all(" minus " not in q and " equals what?" in q for q, _e, _b in rows)
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
    # k=1..4 only, 30 each; decades 20/30/40s 40 each; ones 0..9 12 each
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 30 for k in range(1, 5)}
    assert Counter(expected // 10 for _a, _k, expected, _b in parsed) == {2: 40, 3: 40, 4: 40}
    assert Counter(expected % 10 for _a, _k, expected, _b in parsed) == {ones: 12 for ones in range(10)}
    # carry tracked as a bucket axis (both classes present, sums to 120)
    carry_counts = Counter((a % 10) + k >= 10 for a, k, _expected, _b in parsed)
    assert set(carry_counts) == {False, True}
    assert sum(carry_counts.values()) == 120


def test_120_enumerator_metadata_matches_support_contract():
    rows = _l0c2k2_addition_120_enumerate()
    assert len(rows) == 120
    assert Counter(r["result_decade"] for r in rows) == {"20s": 40, "30s": 40, "40s": 40}
    assert Counter(r["addend_k"] for r in rows) == {f"k_{k}": 30 for k in range(1, 5)}
    assert Counter(r["result_ones"] for r in rows) == {f"ones_{n}": 12 for n in range(10)}


def test_120_language_audit_buckets_cover_declared():
    rows = _flat(build_l0c2k2_addition_120_support(), K120_RUNG)
    present = {bucket for _q, _e, bucket in rows}
    declared = set(language_source_rung_buckets(K120_RUNG))
    assert present == declared
    assert all(bucket.count(":") == 3 for bucket in present)


def test_120_probe_flag_and_watcher_mode_wired():
    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "--l0c2k2-addition-120-audit" in psrc
    assert "args.l0c2k2_addition_120_audit" in psrc
    assert 'surface="l0c2k2addition120"' in psrc
    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        wsrc = fh.read()
    # (a) _AUDIT_MODES registration: flag + grep token present.
    assert "--l0c2k2-addition-120-audit" in wsrc
    assert "L0C2K2ADDITION120 AGGREGATE" in wsrc
    # (b) console summary-print inclusion: the new bank-gate surface MUST appear
    # in the l0c2_bands print tuple, else the live receipt line omits it.
    band_line = next(
        line for line in wsrc.splitlines()
        if "l0c2k2additionfull" in line and "l0c2k2additionheldout50s" in line
    )
    assert "l0c2k2addition120" in band_line, (
        "l0c2k2addition120 missing from watcher console summary band tuple"
    )


def test_120_ce_interleave_dry_run_injects_rows(monkeypatch, tmp_path: Path, capsys):
    """[runtime] proof: a CPU dry-run on the 120 rung with --ce-interleave-support
    injects exactly 13*REPEAT true-label CE rows after the curriculum cap/log and
    exits before the optimizer with no checkpoint written."""
    parent_path = tmp_path / "parent_L0c1_final.pt"
    torch.save(_build_tiny_parent_blob(), parent_path)
    ckpt_path = tmp_path / "k120_ce_interleave_best.pt"
    repeat = 3

    _TRAIN.train(
        curriculum_rung=K120_RUNG,
        use_broad_tokenizer=True,
        curriculum_n_train=12,
        curriculum_n_heldout=6,
        replay_ratio=0.0,
        replay_rungs="R0,R1,R1b1",
        ce_interleave_support=[f"{L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT}:{repeat}"],
        load_from=str(parent_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    assert "[hrm158] ce-interleave:" in out
    assert f"ce_rows_added={13 * repeat}" in out
    assert "dry-run: EXITING before optimizer step" in out
    assert not ckpt_path.exists()


# --------------------------------------------------------------------------- #
# L0c2-K2-addition-120-k5to8 — SECOND 2x-density atom (k=5..8 subset of the 240),
# DISJOINT from the banked k=1..4 atom. Same template/surface; codex's second of
# the two 120 atoms, to push computed transfer after k=1..4 banked (step1250) but
# heldout-50s stayed memorized (1/80).
# --------------------------------------------------------------------------- #

def test_120_k5to8_rung_registered_train_only_diagnosis_only():
    assert K120_K5TO8_RUNG in RUNG_NAMES
    assert K120_K5TO8_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[K120_K5TO8_RUNG]) == {"train"}
    assert K120_K5TO8_RUNG in DIAGNOSIS_ONLY_RUNGS
    # Banked k=5..8 is now a true retained prior for the 50s extension.
    assert K120_K5TO8_RUNG in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_120_k5to8_rung():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{K120_K5TO8_RUNG}"' in src


def test_120_k5to8_generator_train_only_and_exact_240_subset():
    rows = make_rung_examples(K120_K5TO8_RUNG, 120, seed=17, split="train")
    assert len(rows) == 120
    assert len({(r["question"], r["expected"]) for r in rows}) == 120
    assert all(r["rung"] == K120_K5TO8_RUNG for r in rows)
    assert {r["expected"] for r in rows} == set(range(20, 50))
    full_pairs = {(r["question"], r["expected"]) for r in _l0c2k2_addition_full_enumerate()}
    assert all((r["question"], r["expected"]) in full_pairs for r in rows)
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(K120_K5TO8_RUNG, 10, seed=17, split="held_out")


def test_120_k5to8_disjoint_from_k1to4_and_reconstructs_240():
    k5to8 = {(r["question"], r["expected"]) for r in _l0c2k2_addition_120_k5to8_enumerate()}
    k1to4 = {(r["question"], r["expected"]) for r in _l0c2k2_addition_120_enumerate()}
    assert len(k5to8) == 120 and len(k1to4) == 120
    assert k5to8.isdisjoint(k1to4)
    # the two 120 atoms together reconstruct exactly the 240-row full surface
    full = {(r["question"], r["expected"]) for r in _l0c2k2_addition_full_enumerate()}
    assert (k5to8 | k1to4) == full and len(full) == 240


def test_120_k5to8_support_count_echo_exclusions_and_marginals():
    rows = _flat(build_l0c2k2_addition_120_k5to8_support(), K120_K5TO8_RUNG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_120_K5TO8_AUDIT_EXPECTED_COUNT == 120
    assert len({(q, e) for q, e, _bucket in rows}) == 120
    assert all(" plus 0 " not in q and not q.startswith("0 plus ") for q, _e, _b in rows)
    assert all(" minus " not in q and " equals what?" in q for q, _e, _b in rows)
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
    # k=5..8 only, 30 each; decades 20/30/40s 40 each; ones 0..9 12 each
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 30 for k in range(5, 9)}
    assert Counter(expected // 10 for _a, _k, expected, _b in parsed) == {2: 40, 3: 40, 4: 40}
    assert Counter(expected % 10 for _a, _k, expected, _b in parsed) == {ones: 12 for ones in range(10)}
    carry_counts = Counter((a % 10) + k >= 10 for a, k, _expected, _b in parsed)
    assert set(carry_counts) == {False, True}
    assert sum(carry_counts.values()) == 120


def test_120_k5to8_enumerator_metadata_matches_support_contract():
    rows = _l0c2k2_addition_120_k5to8_enumerate()
    assert len(rows) == 120
    assert Counter(r["result_decade"] for r in rows) == {"20s": 40, "30s": 40, "40s": 40}
    assert Counter(r["addend_k"] for r in rows) == {f"k_{k}": 30 for k in range(5, 9)}
    assert Counter(r["result_ones"] for r in rows) == {f"ones_{n}": 12 for n in range(10)}


def test_120_k5to8_language_audit_buckets_cover_declared():
    rows = _flat(build_l0c2k2_addition_120_k5to8_support(), K120_K5TO8_RUNG)
    present = {bucket for _q, _e, bucket in rows}
    declared = set(language_source_rung_buckets(K120_K5TO8_RUNG))
    assert present == declared
    assert all(bucket.count(":") == 3 for bucket in present)


def test_120_k5to8_probe_flag_and_watcher_mode_wired():
    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "--l0c2k2-addition-120-k5to8-audit" in psrc
    assert "args.l0c2k2_addition_120_k5to8_audit" in psrc
    assert 'surface="l0c2k2addition120k5to8"' in psrc
    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        wsrc = fh.read()
    # (a) _AUDIT_MODES registration: flag + grep token present.
    assert "--l0c2k2-addition-120-k5to8-audit" in wsrc
    assert "L0C2K2ADDITION120K5TO8 AGGREGATE" in wsrc
    # (b) console summary-print inclusion: bank-gate surface MUST appear in the
    # l0c2_bands print tuple, else the live receipt line omits it.
    band_line = next(
        line for line in wsrc.splitlines()
        if "l0c2k2additionfull" in line and "l0c2k2additionheldout50s" in line
    )
    assert "l0c2k2addition120k5to8" in band_line, (
        "l0c2k2addition120k5to8 missing from watcher console summary band tuple"
    )


def test_50s_rung_registered_train_only_diagnosis_only_and_not_retained():
    assert FIFTIES_RUNG in RUNG_NAMES
    assert FIFTIES_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[FIFTIES_RUNG]) == {"train"}
    assert FIFTIES_RUNG in DIAGNOSIS_ONLY_RUNGS
    # Banked after the 50s slice; now eligible as closest same-template retained
    # prior for the 60s-transfer rung.
    assert FIFTIES_RUNG in _TRAIN._RETAINED_SUPPORT_REGISTRY

    assert HELDOUT_60S_DIAG not in RUNG_NAMES
    assert HELDOUT_60S_DIAG not in _RUNG_SPEC
    assert HELDOUT_60S_DIAG not in DIAGNOSIS_ONLY_RUNGS
    assert HELDOUT_60S_DIAG not in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_50s_rung_not_diagnostics():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{FIFTIES_RUNG}"' in src
    assert HELDOUT_DIAG not in src
    assert HELDOUT_60S_DIAG not in src


def test_50s_generator_train_only_exact_rows_and_disjoint_from_banked_20s_40s():
    rows = make_rung_examples(FIFTIES_RUNG, 80, seed=17, split="train")
    assert len(rows) == 80
    assert len({(r["question"], r["expected"]) for r in rows}) == 80
    assert all(r["rung"] == FIFTIES_RUNG for r in rows)
    assert {r["expected"] for r in rows} == set(range(50, 60))
    banked_20s_40s = {
        (r["question"], r["expected"])
        for r in _l0c2k2_addition_full_enumerate()
    }
    assert {(r["question"], r["expected"]) for r in rows}.isdisjoint(banked_20s_40s)
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(FIFTIES_RUNG, 10, seed=17, split="held_out")


def test_50s_support_count_echo_exclusions_and_marginals():
    rows = _flat(build_l0c2k2_addition_50s_support(), FIFTIES_RUNG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_50S_AUDIT_EXPECTED_COUNT == 80
    assert len({(q, e) for q, e, _bucket in rows}) == 80
    assert all(" plus 0 " not in q and not q.startswith("0 plus ") for q, _e, _b in rows)
    assert all(" minus " not in q and " equals what?" in q for q, _e, _b in rows)
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 10 for k in range(1, 9)}
    assert Counter(expected // 10 for _a, _k, expected, _b in parsed) == {5: 80}
    assert Counter(expected % 10 for _a, _k, expected, _b in parsed) == {ones: 8 for ones in range(10)}


def test_50s_enumerator_metadata_matches_support_contract():
    rows = _l0c2k2_addition_50s_enumerate()
    assert len(rows) == 80
    assert Counter(r["result_decade"] for r in rows) == {"50s": 80}
    assert Counter(r["addend_k"] for r in rows) == {f"k_{k}": 10 for k in range(1, 9)}
    assert Counter(r["result_ones"] for r in rows) == {f"ones_{n}": 8 for n in range(10)}


def test_heldout_60s_diagnostic_exact_disjoint_and_non_trainable():
    rows = _flat(build_l0c2k2_addition_heldout_60s_support(), HELDOUT_60S_DIAG)
    parsed = _parse_rows(rows)
    assert len(rows) == L0C2K2_ADDITION_HELDOUT_60S_AUDIT_EXPECTED_COUNT == 80
    assert {expected for _a, _k, expected, _bucket in parsed} == set(range(60, 70))
    assert Counter(k for _a, k, _expected, _b in parsed) == {k: 10 for k in range(1, 9)}
    assert Counter(expected % 10 for _a, _k, expected, _b in parsed) == {ones: 8 for ones in range(10)}
    assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
    assert HELDOUT_60S_DIAG not in LANGUAGE_ACTIVE_RUNGS
    assert HELDOUT_60S_DIAG not in RUNG_NAMES
    with pytest.raises(ValueError, match="unknown rung"):
        make_rung_examples(HELDOUT_60S_DIAG, 10, seed=17, split="train")

    fifties = {(r["question"], r["expected"]) for r in _l0c2k2_addition_50s_enumerate()}
    banked_20s_40s = {
        (r["question"], r["expected"])
        for r in _l0c2k2_addition_full_enumerate()
    }
    assert {(q, e) for q, e, _bucket in rows}.isdisjoint(fifties)
    assert {(q, e) for q, e, _bucket in rows}.isdisjoint(banked_20s_40s)


def test_legacy_heldout_50s_aliases_trainable_50s_rows_but_is_non_gating_label():
    legacy = _flat(build_l0c2k2_addition_heldout_50s_support(), HELDOUT_DIAG)
    canonical = _flat(build_l0c2k2_addition_50s_support(), FIFTIES_RUNG)
    assert len(legacy) == L0C2K2_ADDITION_HELDOUT_50S_AUDIT_EXPECTED_COUNT == 80
    assert {(q, e, b) for q, e, b in legacy} == {(q, e, b) for q, e, b in canonical}
    assert HELDOUT_DIAG not in _TRAIN._RETAINED_SUPPORT_REGISTRY

    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "legacy alias-only" in psrc
    assert "NON-GATING" in psrc


def test_50s_retained_support_registry_matches_banked_support_and_stays_disjoint_from_60s():
    retained_rows, support_hash = _TRAIN._retained_support(FIFTIES_RUNG, seed=17)
    canonical = sorted(
        _flat(build_l0c2k2_addition_50s_support(17), FIFTIES_RUNG),
        key=lambda r: (r[2], r[0], r[1]),
    )
    assert len(retained_rows) == L0C2K2_ADDITION_50S_AUDIT_EXPECTED_COUNT == 80
    assert retained_rows == canonical
    assert len(support_hash) == 16

    retained_pairs = {(q, e) for q, e, _bucket in retained_rows}
    sixties_train = {
        (q, e)
        for q, e, _bucket in _flat(
            build_l0c2k2_addition_60s_transfer_train_support(17),
            SIXTIES_TRANSFER_TRAIN,
        )
    }
    sixties_held = {
        (q, e)
        for q, e, _bucket in _flat(
            build_l0c2k2_addition_60s_transfer_held_support(17),
            SIXTIES_TRANSFER_HELD,
        )
    }
    assert retained_pairs.isdisjoint(sixties_train)
    assert retained_pairs.isdisjoint(sixties_held)
    assert SIXTIES_TRANSFER_RUNG not in _TRAIN._RETAINED_SUPPORT_REGISTRY
    assert SIXTIES_TRANSFER_TRAIN not in _TRAIN._RETAINED_SUPPORT_REGISTRY
    assert SIXTIES_TRANSFER_HELD not in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_50s_and_60s_language_audit_buckets_cover_declared():
    for key, support in (
        (FIFTIES_RUNG, build_l0c2k2_addition_50s_support()),
        (HELDOUT_60S_DIAG, build_l0c2k2_addition_heldout_60s_support()),
    ):
        rows = _flat(support, key)
        present = {bucket for _q, _e, bucket in rows}
        declared = set(language_source_rung_buckets(key))
        assert present == declared
        assert all(bucket.count(":") == 3 for bucket in present)


def test_50s_and_60s_probe_flags_and_watcher_modes_wired():
    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "--l0c2k2-addition-50s-audit" in psrc
    assert "--l0c2k2-addition-heldout-60s-audit" in psrc
    assert 'surface="l0c2k2addition50s"' in psrc
    assert 'surface="l0c2k2additionheldout60s"' in psrc

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        wsrc = fh.read()
    assert "--l0c2k2-addition-50s-audit" in wsrc
    assert "--l0c2k2-addition-heldout-60s-audit" in wsrc
    assert "L0C2K2ADDITION50S AGGREGATE" in wsrc
    assert "L0C2K2ADDITIONHELDOUT60S AGGREGATE" in wsrc
    assert "alias-only/non-gating" in wsrc
    band_line = next(
        line for line in wsrc.splitlines()
        if "l0c2k2addition50s" in line and "l0c2k2additionheldout60s" in line
    )
    assert "l0c2k2additionheldout50s" in band_line


def test_60s_transfer_rung_registered_train_only_diagnosis_only_and_not_retained():
    assert SIXTIES_TRANSFER_RUNG in RUNG_NAMES
    assert SIXTIES_TRANSFER_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[SIXTIES_TRANSFER_RUNG]) == {"train"}
    assert SIXTIES_TRANSFER_RUNG in DIAGNOSIS_ONLY_RUNGS
    assert SIXTIES_TRANSFER_RUNG not in _TRAIN._RETAINED_SUPPORT_REGISTRY

    for audit_key in (SIXTIES_TRANSFER_TRAIN, SIXTIES_TRANSFER_HELD):
        assert audit_key not in RUNG_NAMES
        assert audit_key not in _RUNG_SPEC
        assert audit_key not in DIAGNOSIS_ONLY_RUNGS
        assert audit_key not in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_60s_transfer_rung_not_audit_surfaces():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{SIXTIES_TRANSFER_RUNG}"' in src
    assert SIXTIES_TRANSFER_TRAIN not in src
    assert SIXTIES_TRANSFER_HELD not in src
    assert HELDOUT_60S_DIAG not in src


def test_60s_transfer_train_path_samples_only_train_split():
    train_pairs = {
        (q, expected)
        for q, expected, _bucket in _flat(
            build_l0c2k2_addition_60s_transfer_train_support(),
            SIXTIES_TRANSFER_TRAIN,
        )
    }
    held_pairs = {
        (q, expected)
        for q, expected, _bucket in _flat(
            build_l0c2k2_addition_60s_transfer_held_support(),
            SIXTIES_TRANSFER_HELD,
        )
    }
    rows = make_rung_examples(SIXTIES_TRANSFER_RUNG, 60, seed=17, split="train")
    sampled_pairs = {(r["question"], r["expected"]) for r in rows}
    assert len(rows) == 60
    assert len(sampled_pairs) == 60
    assert all(r["rung"] == SIXTIES_TRANSFER_RUNG for r in rows)
    assert sampled_pairs == train_pairs
    assert sampled_pairs.isdisjoint(held_pairs)
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(SIXTIES_TRANSFER_RUNG, 10, seed=17, split="held_out")


def test_60s_transfer_partition_counts_disjoint_and_recombination_coverage():
    train_rows = _flat(
        build_l0c2k2_addition_60s_transfer_train_support(),
        SIXTIES_TRANSFER_TRAIN,
    )
    held_rows = _flat(
        build_l0c2k2_addition_60s_transfer_held_support(),
        SIXTIES_TRANSFER_HELD,
    )
    train = _parse_rows(train_rows)
    held = _parse_rows(held_rows)
    train_pairs = {(q, e) for q, e, _bucket in train_rows}
    held_pairs = {(q, e) for q, e, _bucket in held_rows}

    assert len(train_rows) == L0C2K2_ADDITION_60S_TRANSFER_TRAIN_AUDIT_EXPECTED_COUNT == 60
    assert len(held_rows) == L0C2K2_ADDITION_60S_TRANSFER_HELD_AUDIT_EXPECTED_COUNT == 20
    assert len(train_pairs) == 60
    assert len(held_pairs) == 20
    assert train_pairs.isdisjoint(held_pairs)
    assert Counter(expected for _a, _k, expected, _bucket in train) == {
        result: 6 for result in range(60, 70)
    }
    assert Counter(expected for _a, _k, expected, _bucket in held) == {
        result: 2 for result in range(60, 70)
    }
    assert set(k for _a, k, _expected, _bucket in train) == set(range(1, 9))
    assert set(k for _a, k, _expected, _bucket in held) == set(range(1, 9))

    # recombination_coverage: every held result-value and k-value is also seen in train.
    recombination_coverage = (
        {expected for _a, _k, expected, _bucket in held}
        <= {expected for _a, _k, expected, _bucket in train}
        and {k for _a, k, _expected, _bucket in held}
        <= {k for _a, k, _expected, _bucket in train}
    )
    assert recombination_coverage


def test_60s_transfer_no_zero_minus_echo_carry_and_cross_support_disjoint():
    train_rows = _flat(
        build_l0c2k2_addition_60s_transfer_train_support(),
        SIXTIES_TRANSFER_TRAIN,
    )
    held_rows = _flat(
        build_l0c2k2_addition_60s_transfer_held_support(),
        SIXTIES_TRANSFER_HELD,
    )
    for rows in (train_rows, held_rows):
        parsed = _parse_rows(rows)
        assert all(1 <= k <= 8 for _a, k, _expected, _bucket in parsed)
        assert all(" plus 0 " not in q and not q.startswith("0 plus ") for q, _e, _b in rows)
        assert all(" minus " not in q and " equals what?" in q for q, _e, _b in rows)
        assert all(expected != a and expected != k for a, k, expected, _bucket in parsed)
        carry_counts = Counter((a % 10) + k >= 10 for a, k, _expected, _bucket in parsed)
        assert set(carry_counts) == {False, True}

    sixties_pairs = {(q, e) for q, e, _bucket in train_rows + held_rows}
    banked_20s_40s = {
        (r["question"], r["expected"])
        for r in _l0c2k2_addition_full_enumerate()
    }
    fifties = {
        (r["question"], r["expected"])
        for r in _l0c2k2_addition_50s_enumerate()
    }
    assert sixties_pairs.isdisjoint(banked_20s_40s)
    assert sixties_pairs.isdisjoint(fifties)


def test_60s_transfer_enumerator_metadata_matches_support_contract():
    train = _l0c2k2_addition_60s_transfer_train_enumerate()
    held = _l0c2k2_addition_60s_transfer_held_enumerate()
    assert len(train) == 60
    assert len(held) == 20
    assert Counter(r["result_decade"] for r in train) == {"60s": 60}
    assert Counter(r["result_decade"] for r in held) == {"60s": 20}
    assert Counter(r["result_ones"] for r in train) == {f"ones_{n}": 6 for n in range(10)}
    assert Counter(r["result_ones"] for r in held) == {f"ones_{n}": 2 for n in range(10)}
    assert set(Counter(r["addend_k"] for r in train).values()) == {7, 8}
    assert set(Counter(r["addend_k"] for r in held).values()) == {2, 3}


def test_60s_transfer_language_audit_buckets_cover_declared():
    for key, support in (
        (SIXTIES_TRANSFER_TRAIN, build_l0c2k2_addition_60s_transfer_train_support()),
        (SIXTIES_TRANSFER_HELD, build_l0c2k2_addition_60s_transfer_held_support()),
    ):
        rows = _flat(support, key)
        present = {bucket for _q, _e, bucket in rows}
        declared = set(language_source_rung_buckets(key))
        assert present == declared
        assert all(bucket.count(":") == 3 for bucket in present)


def test_60s_transfer_probe_flags_watcher_modes_and_legacy_label_wired():
    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "--l0c2k2-addition-60s-transfer-train-audit" in psrc
    assert "--l0c2k2-addition-60s-transfer-held-audit" in psrc
    assert 'surface="l0c2k2addition60stransfertrain"' in psrc
    assert 'surface="l0c2k2addition60stransferheld"' in psrc
    assert "TRANSFER_HELD_BANK_GATE_FOR_60S_TRANSFER" in psrc
    assert "LEGACY_50S_TRANSFER_DIAGNOSTIC_ONLY" in psrc
    assert "not future trained-out proof" in psrc

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        wsrc = fh.read()
    assert "--l0c2k2-addition-60s-transfer-train-audit" in wsrc
    assert "--l0c2k2-addition-60s-transfer-held-audit" in wsrc
    assert "L0C2K2ADDITION60STRANSFERTRAIN AGGREGATE" in wsrc
    assert "L0C2K2ADDITION60STRANSFERHELD AGGREGATE" in wsrc
    assert "TRANSFER_HELD_BANK_GATE_FOR_60S_TRANSFER" in wsrc
    assert "LEGACY_50S_TRANSFER_DIAGNOSTIC_ONLY" in wsrc
    band_line = next(
        line for line in wsrc.splitlines()
        if "l0c2k2addition60stransfertrain" in line
        and "l0c2k2addition60stransferheld" in line
    )
    assert "l0c2k2additionheldout60s" in band_line


def test_60s_trace_rung_registered_train_only_diagnosis_only_and_not_retained():
    assert SIXTIES_TRACE_RUNG in RUNG_NAMES
    assert SIXTIES_TRACE_RUNG in _RUNG_SPEC
    assert set(_RUNG_SPEC[SIXTIES_TRACE_RUNG]) == {"train"}
    assert SIXTIES_TRACE_RUNG in DIAGNOSIS_ONLY_RUNGS
    assert SIXTIES_TRACE_RUNG not in _TRAIN._RETAINED_SUPPORT_REGISTRY

    for audit_key in (SIXTIES_TRACE_TRAIN, SIXTIES_TRACE_HELD):
        assert audit_key not in RUNG_NAMES
        assert audit_key not in _RUNG_SPEC
        assert audit_key not in DIAGNOSIS_ONLY_RUNGS
        assert audit_key not in _TRAIN._RETAINED_SUPPORT_REGISTRY


def test_trainer_choices_include_60s_trace_rung_not_audit_surfaces():
    train_src = os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
    with open(train_src, "r", encoding="utf-8") as fh:
        src = fh.read()
    assert f'"{SIXTIES_TRACE_RUNG}"' in src
    assert SIXTIES_TRACE_TRAIN not in src
    assert SIXTIES_TRACE_HELD not in src


def test_60s_trace_target_format_and_arithmetic():
    assert _l0c2k2_addition_trace_target(57, 6, 63) == (
        "ones 7+6=13; write 3 carry 1; tens 5+1=6; answer 63"
    )
    rows = _l0c2k2_addition_60s_trace_train_enumerate() + \
        _l0c2k2_addition_60s_trace_held_enumerate()
    assert len(rows) == 80
    trace_re = re.compile(
        r"^ones (\d)\+([1-8])=(\d+); write (\d) carry ([01]); "
        r"tens (\d)\+([01])=(\d); answer (6\d)$"
    )
    for row in rows:
        m = trace_re.fullmatch(row["expected"])
        assert m, row["expected"]
        a = row["a"]
        k = row["k"]
        result = row["result"]
        ones_sum = (a % 10) + k
        carry = ones_sum // 10
        assert int(m.group(1)) == a % 10
        assert int(m.group(2)) == k
        assert int(m.group(3)) == ones_sum
        assert int(m.group(4)) == result % 10
        assert int(m.group(5)) == carry
        assert int(m.group(6)) == a // 10
        assert int(m.group(7)) == carry
        assert int(m.group(8)) == result // 10
        assert int(m.group(9)) == result


def test_60s_trace_train_path_samples_only_train_split():
    train_rows = _flat(
        build_l0c2k2_addition_60s_trace_train_support(),
        SIXTIES_TRACE_TRAIN,
    )
    held_rows = _flat(
        build_l0c2k2_addition_60s_trace_held_support(),
        SIXTIES_TRACE_HELD,
    )
    train_pairs = {(q, expected) for q, expected, _bucket in train_rows}
    held_pairs = {(q, expected) for q, expected, _bucket in held_rows}
    rows = make_rung_examples(SIXTIES_TRACE_RUNG, 60, seed=17, split="train")
    sampled_pairs = {(r["question"], r["expected"]) for r in rows}
    assert len(rows) == 60
    assert len(sampled_pairs) == 60
    assert all(r["rung"] == SIXTIES_TRACE_RUNG for r in rows)
    assert sampled_pairs == train_pairs
    assert sampled_pairs.isdisjoint(held_pairs)
    assert all(isinstance(r["expected"], str) and r["expected"].startswith("ones ") for r in rows)
    with pytest.raises(ValueError, match="TRAIN-only"):
        make_rung_examples(SIXTIES_TRACE_RUNG, 10, seed=17, split="held_out")


def test_60s_trace_partition_counts_disjoint_recombination_and_prompt_disjointness():
    train_rows = _flat(
        build_l0c2k2_addition_60s_trace_train_support(),
        SIXTIES_TRACE_TRAIN,
    )
    held_rows = _flat(
        build_l0c2k2_addition_60s_trace_held_support(),
        SIXTIES_TRACE_HELD,
    )
    train = _parse_trace_rows(train_rows)
    held = _parse_trace_rows(held_rows)
    train_prompts = {q for q, _e, _bucket in train_rows}
    held_prompts = {q for q, _e, _bucket in held_rows}

    assert len(train_rows) == L0C2K2_ADDITION_60S_TRACE_TRAIN_AUDIT_EXPECTED_COUNT == 60
    assert len(held_rows) == L0C2K2_ADDITION_60S_TRACE_HELD_AUDIT_EXPECTED_COUNT == 20
    assert len(train_prompts) == 60
    assert len(held_prompts) == 20
    assert train_prompts.isdisjoint(held_prompts)
    assert Counter(answer for _a, _k, answer, _bucket, _trace in train) == {
        result: 6 for result in range(60, 70)
    }
    assert Counter(answer for _a, _k, answer, _bucket, _trace in held) == {
        result: 2 for result in range(60, 70)
    }
    assert set(k for _a, k, _answer, _bucket, _trace in held) <= {
        k for _a, k, _answer, _bucket, _trace in train
    }
    assert set(answer for _a, _k, answer, _bucket, _trace in held) <= {
        answer for _a, _k, answer, _bucket, _trace in train
    }

    banked_surfaces = {
        "addition-full": _l0c2k2_addition_full_enumerate(),
        "addition-120": _l0c2k2_addition_120_enumerate(),
        "addition-120-k5to8": _l0c2k2_addition_120_k5to8_enumerate(),
        "addition-50s": _l0c2k2_addition_50s_enumerate(),
    }
    trace_prompts = train_prompts | held_prompts
    for label, rows in banked_surfaces.items():
        overlap = trace_prompts & {r["question"] for r in rows}
        assert not overlap, f"{label} prompt overlap: {sorted(overlap)[:5]}"


def test_60s_trace_language_audit_buckets_tokenizer_and_parser_contract():
    for key, support in (
        (SIXTIES_TRACE_TRAIN, build_l0c2k2_addition_60s_trace_train_support()),
        (SIXTIES_TRACE_HELD, build_l0c2k2_addition_60s_trace_held_support()),
    ):
        rows = _flat(support, key)
        present = {bucket for _q, _e, bucket in rows}
        declared = set(language_source_rung_buckets(key))
        assert present == declared
        assert all(bucket.count(":") == 3 for bucket in present)

    q, expected, _bucket = _flat(
        build_l0c2k2_addition_60s_trace_train_support(),
        SIXTIES_TRACE_TRAIN,
    )[0]
    tok = BroadTokenizer()
    ids, sep_pos = tok.encode_example(q, expected)
    assert tok.decode(ids[sep_pos + 1:-1]) == expected
    assert _PROBE._parse_trace_answer(expected) == _PROBE._parse_trace_answer(
        "prefix " + expected + " suffix"
    )
    assert _PROBE._parse_trace_answer("ones 7+6=13; write 3 carry 1") is None
    assert _PROBE._parse_integer_only("63") == 63
    assert _PROBE._parse_integer_only("answer 63") is None


def test_60s_trace_probe_flags_watcher_modes_and_collision_fields_wired():
    probe_src = os.path.join(_REPO, "scripts", "probe_hrm_text_158.py")
    with open(probe_src, "r", encoding="utf-8") as fh:
        psrc = fh.read()
    assert "--l0c2k2-addition-60s-trace-train-audit" in psrc
    assert "--l0c2k2-addition-60s-trace-held-audit" in psrc
    assert 'surface="l0c2k2addition60stracetrain"' in psrc
    assert 'surface="l0c2k2addition60straceheld"' in psrc
    assert "exact_trace" in psrc
    assert "parsed_answer" in psrc
    assert "integer_only_bleed" in psrc
    assert "trace_bleed" in psrc

    watcher_src = os.path.join(_REPO, "scripts", "parallel_audit_watcher.py")
    with open(watcher_src, "r", encoding="utf-8") as fh:
        wsrc = fh.read()
    assert "--l0c2k2-addition-60s-trace-train-audit" in wsrc
    assert "--l0c2k2-addition-60s-trace-held-audit" in wsrc
    assert "L0C2K2ADDITION60STRACETRAIN AGGREGATE" in wsrc
    assert "L0C2K2ADDITION60STRACEHELD AGGREGATE" in wsrc
    assert "TRACE_TRAIN_BANK_GATE" in wsrc
    assert "TRACE_HELD_RECOMBINATION_BANK_GATE" in wsrc
    assert '"--max-gen", "128"' in wsrc


def test_120_k5to8_ce_interleave_dry_run_injects_rows(monkeypatch, tmp_path: Path, capsys):
    """[runtime] proof: a CPU dry-run on the k5to8 rung with --ce-interleave-support
    injects exactly 13*REPEAT true-label CE rows and exits before the optimizer
    with no checkpoint written."""
    parent_path = tmp_path / "parent_L0c1_final.pt"
    torch.save(_build_tiny_parent_blob(), parent_path)
    ckpt_path = tmp_path / "k120_k5to8_ce_interleave_best.pt"
    repeat = 3

    _TRAIN.train(
        curriculum_rung=K120_K5TO8_RUNG,
        use_broad_tokenizer=True,
        curriculum_n_train=12,
        curriculum_n_heldout=6,
        replay_ratio=0.0,
        replay_rungs="R0,R1,R1b1",
        ce_interleave_support=[f"{L0C1_CLOSE_SIBLING_CE_INTERLEAVE_SUPPORT}:{repeat}"],
        load_from=str(parent_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    assert "[hrm158] ce-interleave:" in out
    assert f"ce_rows_added={13 * repeat}" in out
    assert "dry-run: EXITING before optimizer step" in out
    assert not ckpt_path.exists()


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("L0c2-K2 addition-full support tests: PASS")
