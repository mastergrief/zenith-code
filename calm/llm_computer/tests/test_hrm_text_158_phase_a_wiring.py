"""HRM-Text-1.58 Phase 3 Step 1 Phase A wiring tests.

Per codex msg 1779462307554-b57d8288 (Phase A +1 implement, receipt
requirement). Covers:

- Curriculum dry-run R0 with BroadTokenizer (trainer end-to-end without GPU)
- --load-from compat: Phase 2 vocab mismatch hard-fails through the trainer
- --load-from compat: matching broad tiny ckpt loads strict via trainer
- probe_curriculum writes RungProbeResult JSON with exact + parsed metrics,
  replay_ratio, ckpt path, finite flag

NO actual training. NO GPU required (forces device='cpu').
NO ckpt staging into the commit (test ckpts go to tmp_path).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, ".")

from calm.hrm_text_158 import (
    HierarchicalReasoningModel,
    HierarchicalReasoningModelConfig,
    LMHead,
    LMHeadConfig,
)
from calm.hrm_text_158.curriculum import BroadTokenizer
from scripts.train_hrm_text_158 import _build_ckpt_config, SOURCE_PIN, train
from scripts.probe_hrm_text_158 import probe_curriculum


# Tiny architecture spec — same fields used across all Phase A tests so the
# ckpt-compat path resolves with `validate_load_from_ckpt_compat`.
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


def _build_tiny_broad_ckpt_blob(use_ternary: bool = False) -> dict:
    """Construct a tiny HRM-Text-1.58 model + BroadTokenizer + ckpt blob,
    matching TINY_ARCH. Returns the blob ready for torch.save."""
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
        use_ternary_bulk=use_ternary,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))
    config_blob = _build_ckpt_config(
        m, tok, cfg, TINY_ARCH["max_len"], batch_size=4,
        curriculum_rung="R0", curriculum_seed=42,
        replay_ratio=0.0, prior_rungs=[],
    )
    return {
        "model_state": m.state_dict(),
        "config": config_blob,
        "step": 50,
        "epoch": 1,
        "source_pin": SOURCE_PIN,
    }


# ============================================================================ #
# Curriculum dry-run R0 with BroadTokenizer
# ============================================================================ #

def test_dry_run_r0_broad_tokenizer(tmp_path: Path, capsys) -> None:
    """train() with --curriculum-rung R0 + --use-broad-tokenizer + --dry-run
    must build corpus + model + first batch + verify forward, then exit
    BEFORE optimizer step. No ckpt written, no GPU required."""
    ckpt_path = tmp_path / "should_not_exist.pt"
    train(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=64,
        curriculum_n_heldout=16,
        replay_ratio=0.30,  # auto-overridden to 0.0 for R0 (no priors)
        dry_run=True,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    # Key receipt strings
    assert "PHASE 3 curriculum mode: rung=R0" in out
    assert "BroadTokenizer (vocab=260, normalizer_version=byte_utf8_v1)" in out
    assert "no prior rungs to replay" in out
    assert "dry-run: corpus stats" in out
    assert "dry-run: forward OK" in out
    assert "finite=True" in out
    assert "EXITING before optimizer step" in out
    # No ckpt written
    assert not ckpt_path.exists(), f"--dry-run wrote ckpt at {ckpt_path}"


def test_dry_run_r1_with_replay(tmp_path: Path, capsys) -> None:
    """R1 dry-run with replay_ratio=0.30 must mix in R0 samples.

    Per codex msg 1779463196431 rule 1: R1+ requires --load-from PATH,
    so this test must construct a matching R0-shaped parent ckpt first.
    """
    # Build matching R0 parent ckpt for --load-from compat
    parent_blob = _build_tiny_broad_ckpt_blob()
    parent_ckpt_path = tmp_path / "parent_R0_final.pt"
    torch.save(parent_blob, parent_ckpt_path)

    ckpt_path = tmp_path / "should_not_exist.pt"
    train(
        curriculum_rung="R1",
        use_broad_tokenizer=True,
        curriculum_n_train=100,
        curriculum_n_heldout=10,
        replay_ratio=0.30,
        load_from=str(parent_ckpt_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    assert "PHASE 3 curriculum mode: rung=R1" in out
    # Should NOT see the "no prior rungs" override; R1 has R0 as prior
    assert "no prior rungs to replay" not in out
    assert "replay" in out and "R0" in out


def test_r1_without_load_from_hard_fails(tmp_path: Path) -> None:
    """Codex msg 1779463196431 rule 1: --curriculum-rung R1+ without
    --load-from must hard-fail. Random-init at R1+ would silently break
    the checkpoint chain contract."""
    with pytest.raises(ValueError, match="requires --load-from"):
        train(
            curriculum_rung="R1",
            use_broad_tokenizer=True,
            curriculum_n_train=20,
            curriculum_n_heldout=4,
            load_from=None,  # explicit None to test the gate
            dry_run=True,
            device="cpu",
            checkpoint_path=str(tmp_path / "ignored.pt"),
            epochs=1,
            batch_size=4,
            **TINY_ARCH,
        )


def test_r0_without_load_from_succeeds(tmp_path: Path) -> None:
    """R0 is the only rung permitted from random init (no parent in chain)."""
    train(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=20,
        curriculum_n_heldout=4,
        load_from=None,
        dry_run=True,
        device="cpu",
        checkpoint_path=str(tmp_path / "ignored.pt"),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    # No exception = pass


def test_r2_r3_without_load_from_hard_fails(tmp_path: Path) -> None:
    """Same gate at R2, R3."""
    for rung in ("R2", "R3"):
        with pytest.raises(ValueError, match="requires --load-from"):
            train(
                curriculum_rung=rung,
                use_broad_tokenizer=True,
                curriculum_n_train=20,
                curriculum_n_heldout=4,
                load_from=None,
                dry_run=True,
                device="cpu",
                checkpoint_path=str(tmp_path / "ignored.pt"),
                epochs=1,
                batch_size=4,
                **TINY_ARCH,
            )


def test_curriculum_requires_broad_tokenizer() -> None:
    """--curriculum-rung without --use-broad-tokenizer must hard-fail."""
    with pytest.raises(ValueError, match="--use-broad-tokenizer"):
        train(
            curriculum_rung="R0",
            use_broad_tokenizer=False,
            dry_run=True,
            device="cpu",
            **TINY_ARCH,
        )


# ============================================================================ #
# --load-from compat (through the trainer, not just the unit-level validator)
# ============================================================================ #

def test_load_from_phase2_vocab_mismatch_hard_fails(tmp_path: Path) -> None:
    """A simulated Phase 2 GSM8k-vocab ckpt loaded against a Phase 3
    broad-tokenizer trainer config must hard-fail via the trainer's
    --load-from path with the vocab-mismatch message."""
    # Build a fake Phase 2 ckpt (98-char GSM8k vocab + char_v1 normalizer)
    tok_broad = BroadTokenizer()
    phase2_vocab = ["<pad>", "<bos>", "<eos>", "<sep>"] + [chr(i) for i in range(33, 127)]
    assert len(phase2_vocab) == 98

    # Construct a minimal blob WITHOUT building a model (trainer aborts on
    # validate_load_from_ckpt_compat before state_dict load anyway)
    fake_config = _build_tiny_broad_ckpt_blob()["config"].copy()
    fake_config["gsm8k_char_vocab"] = phase2_vocab
    fake_config["gsm8k_normalizer_version"] = "char_v1"  # Phase 2 normalizer
    fake_blob = {
        "model_state": {},  # never reached because compat fails first
        "config": fake_config,
        "step": 0,
        "epoch": 0,
        "source_pin": SOURCE_PIN,
    }
    phase2_ckpt_path = tmp_path / "phase2_fake.pt"
    torch.save(fake_blob, phase2_ckpt_path)

    # Trainer with --curriculum-rung R1 (so it goes through the broad tokenizer
    # path) + --load-from <phase2_fake>
    with pytest.raises(ValueError, match="vocab.*differs|vocab.*mismatch"):
        train(
            curriculum_rung="R1",
            use_broad_tokenizer=True,
            curriculum_n_train=20,
            curriculum_n_heldout=4,
            replay_ratio=0.30,
            load_from=str(phase2_ckpt_path),
            dry_run=True,  # don't actually launch
            device="cpu",
            checkpoint_path=str(tmp_path / "ignored.pt"),
            epochs=1,
            batch_size=4,
            **TINY_ARCH,
        )


def test_load_from_matching_broad_ckpt_loads_strict(tmp_path: Path, capsys) -> None:
    """A matching broad-vocab tiny ckpt must load via --load-from strict
    (state_dict shape/key checks pass), then the trainer continues into
    --dry-run."""
    blob = _build_tiny_broad_ckpt_blob()
    parent_ckpt_path = tmp_path / "parent_R0_final.pt"
    torch.save(blob, parent_ckpt_path)

    train(
        curriculum_rung="R1",
        use_broad_tokenizer=True,
        curriculum_n_train=20,
        curriculum_n_heldout=4,
        replay_ratio=0.30,
        load_from=str(parent_ckpt_path),
        dry_run=True,
        device="cpu",
        checkpoint_path=str(tmp_path / "child_R1_best.pt"),
        epochs=1,
        batch_size=4,
        **TINY_ARCH,
    )
    out = capsys.readouterr().out
    assert "--load-from compat OK" in out
    assert "loading model_state strict" in out
    assert "--load-from loaded" in out
    assert "EXITING before optimizer step" in out


def test_load_from_arch_mismatch_hard_fails(tmp_path: Path) -> None:
    """A broad-vocab ckpt with mismatched hidden_size must hard-fail
    through the trainer."""
    blob = _build_tiny_broad_ckpt_blob()
    blob["config"]["hidden_size"] = 128  # current run is hidden_size=64
    bad_ckpt_path = tmp_path / "bad_arch.pt"
    torch.save(blob, bad_ckpt_path)

    with pytest.raises(ValueError, match="hidden_size.*mismatch"):
        train(
            curriculum_rung="R1",
            use_broad_tokenizer=True,
            curriculum_n_train=20,
            curriculum_n_heldout=4,
            load_from=str(bad_ckpt_path),
            dry_run=True,
            device="cpu",
            checkpoint_path=str(tmp_path / "ignored.pt"),
            epochs=1,
            batch_size=4,
            **TINY_ARCH,
        )


# ============================================================================ #
# probe_curriculum writes RungProbeResult JSON
# ============================================================================ #

def test_real_r0_final_ckpt_records_effective_replay_ratio_zero(tmp_path: Path) -> None:
    """Codex msg 1779463196431 rule 2: a real (non-dry-run) tiny R0 training
    must persist `replay_ratio == 0.0` (effective) on the FINAL ckpt config,
    even though CLI default --replay-ratio=0.30 was passed. R0 has no prior
    rungs so replay is 0; the ckpt should report what actually ran."""
    ckpt_path = tmp_path / "tiny_R0_best.pt"  # will be renamed _final
    train(
        curriculum_rung="R0",
        use_broad_tokenizer=True,
        curriculum_n_train=8,
        curriculum_n_heldout=4,
        replay_ratio=0.30,  # CLI value; should NOT appear in final ckpt
        dry_run=False,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=2,
        log_every=1,
        **TINY_ARCH,
    )
    # Final ckpt: renamed _best -> _final per honest naming rule
    expected_path = tmp_path / "tiny_R0_final.pt"
    assert expected_path.exists(), f"Expected honest-renamed ckpt at {expected_path}"
    blob = torch.load(expected_path, map_location="cpu", weights_only=False)
    config = blob["config"]
    assert config["curriculum_rung"] == "R0"
    assert config["replay_ratio"] == 0.0, (
        f"Final ckpt replay_ratio should be effective (0.0 for R0, no priors), "
        f"got {config['replay_ratio']}"
    )
    assert config["prior_rungs"] == []
    assert config["gsm8k_normalizer_version"] == "byte_utf8_v1"


def test_real_r1_final_ckpt_records_effective_replay_ratio_nonzero(tmp_path: Path) -> None:
    """R1 (prior=R0) trained with --replay-ratio=0.30 must persist 0.30
    on the final ckpt (effective == CLI when prior rungs exist)."""
    # First: tiny R0 parent
    parent_blob = _build_tiny_broad_ckpt_blob()
    parent_path = tmp_path / "parent_R0_final.pt"
    torch.save(parent_blob, parent_path)

    ckpt_path = tmp_path / "tiny_R1_best.pt"
    train(
        curriculum_rung="R1",
        use_broad_tokenizer=True,
        curriculum_n_train=10,
        curriculum_n_heldout=4,
        replay_ratio=0.30,
        load_from=str(parent_path),
        dry_run=False,
        device="cpu",
        checkpoint_path=str(ckpt_path),
        epochs=1,
        batch_size=2,
        log_every=1,
        **TINY_ARCH,
    )
    expected_path = tmp_path / "tiny_R1_final.pt"
    assert expected_path.exists()
    blob = torch.load(expected_path, map_location="cpu", weights_only=False)
    assert blob["config"]["curriculum_rung"] == "R1"
    assert blob["config"]["replay_ratio"] == 0.30
    assert blob["config"]["prior_rungs"] == ["R0"]


def test_probe_finite_catches_nan_logits(tmp_path: Path) -> None:
    """Codex msg 1779463196431 rule 4: probe finite flag must catch
    non-finite logits via torch.isfinite(logits).all(), not the prior
    string-comparison no-op sentinel."""
    from scripts.probe_hrm_text_158 import _decode_greedy_no_cache

    # Stub model that returns NaN logits at every step
    class NaNModel:
        def __call__(self, carry, batch):
            B, L = batch["inputs"].shape
            logits = torch.full((B, L, 260), float("nan"))
            return None, logits

    tok = BroadTokenizer()
    decoded, too_long, finite = _decode_greedy_no_cache(
        NaNModel(), tok, "what is 1?",
        max_gen=4, max_seq_len=64, device="cpu",
    )
    assert too_long is False, "Prefix fits in max_seq_len"
    assert finite is False, "NaN logits MUST be caught by finite check"


def test_probe_finite_true_on_normal_logits(tmp_path: Path) -> None:
    """The dual: with a normal model returning finite logits, the
    decode loop's finite flag stays True."""
    from scripts.probe_hrm_text_158 import _decode_greedy_no_cache

    class FiniteModel:
        def __call__(self, carry, batch):
            B, L = batch["inputs"].shape
            # Sharp argmax at id=20 (an arbitrary byte token) so decode emits ids
            logits = torch.zeros((B, L, 260))
            logits[..., 20] = 10.0
            return None, logits

    tok = BroadTokenizer()
    decoded, too_long, finite = _decode_greedy_no_cache(
        FiniteModel(), tok, "what is 1?",
        max_gen=4, max_seq_len=64, device="cpu",
    )
    assert too_long is False
    assert finite is True


def test_probe_curriculum_writes_rung_probe_result_json(tmp_path: Path) -> None:
    """probe_curriculum on a tiny broad ckpt writes a RungProbeResult JSON
    with all fields: per-rung accuracy + parsed + exact + too_long + cap,
    canonical 17×23 dict, ckpt_path, n_params, step, finite, elapsed_sec."""
    # Build + save tiny broad ckpt
    blob = _build_tiny_broad_ckpt_blob()
    ckpt_path = tmp_path / "tiny_R0_final.pt"
    torch.save(blob, ckpt_path)

    json_out = tmp_path / "probe_R0.json"
    result = probe_curriculum(
        ckpt_path=str(ckpt_path),
        rungs=["R0", "R1"],
        eval_cap=8,        # tiny eval for test speed
        max_gen=4,
        device="cpu",
        output_json=str(json_out),
    )
    # JSON file written
    assert json_out.exists(), "probe_curriculum did not write --probe-output-json"
    data = json.loads(json_out.read_text())

    # Required fields per RungProbeResult schema
    assert data["rung"] == "R0"  # from ckpt config
    assert data["ckpt_path"] == str(ckpt_path)
    assert data["step"] == 50
    assert data["n_params"] > 0
    assert "R0" in data["rung_accuracy"]
    assert "R1" in data["rung_accuracy"]
    assert "R0" in data["rung_exact"]
    assert "R0" in data["rung_parsed"]
    assert "R0" in data["rung_too_long"]
    assert "R0" in data["rung_cap"]
    assert data["rung_cap"]["R0"] == 8

    # Codex msg 1779463196431 rule 3: exact is the PRIMARY metric for
    # curriculum gates. `rung_accuracy[r]` must equal `rung_exact[r] / cap`,
    # NOT `rung_parsed[r] / cap`. Otherwise "391xyz" counts as correct.
    for r in ("R0", "R1"):
        cap = data["rung_cap"][r]
        if cap > 0:
            expected_acc = data["rung_exact"][r] / cap
            assert data["rung_accuracy"][r] == pytest.approx(expected_acc), (
                f"{r}: rung_accuracy={data['rung_accuracy'][r]} should equal "
                f"exact/cap={expected_acc} (exact-primary metric per codex rule 3)"
            )
    # Canonical 17×23 fields
    assert data["canonical_17x23"]["question"] == "what is 17 times 23?"
    assert data["canonical_17x23"]["expected"] == 391
    assert "decoded" in data["canonical_17x23"]
    assert "parsed" in data["canonical_17x23"]
    assert "parsed_ok" in data["canonical_17x23"]
    assert "exact_ok" in data["canonical_17x23"]
    assert "too_long" in data["canonical_17x23"]
    # Finite + elapsed
    assert data["finite"] is True
    assert data["elapsed_sec"] >= 0

    # Same data on the returned dataclass
    assert result.rung == "R0"
    assert result.ckpt_path == str(ckpt_path)
    assert result.canonical_17x23["expected"] == 391


def test_probe_curriculum_rejects_r7() -> None:
    """probe_curriculum cannot probe R7 (GSM8k, not a synthetic rung)."""
    with pytest.raises(ValueError, match="Invalid curriculum rung"):
        probe_curriculum(
            ckpt_path="dummy",
            rungs=["R7"],
            device="cpu",
        )


def test_probe_curriculum_rejects_empty_rungs() -> None:
    """probe_curriculum requires at least one rung."""
    with pytest.raises(ValueError, match="at least one rung"):
        probe_curriculum(
            ckpt_path="dummy",
            rungs=[],
            device="cpu",
        )


# ============================================================================ #
# Ckpt config carries curriculum metadata
# ============================================================================ #

def test_ckpt_config_includes_curriculum_metadata() -> None:
    """_build_ckpt_config populates curriculum fields when called with
    curriculum_rung != None; omits them in GSM8k mode."""
    tok = BroadTokenizer()
    cfg = HierarchicalReasoningModelConfig(
        max_seq_len=64, n_layers=2, hidden_size=64, num_heads=2,
        expansion=4, H_cycles=1, L_cycles=1, half_layers=True,
        bp_warmup_ratio=0.2, bp_min_steps=1, bp_max_steps=2,
    )
    hrm = HierarchicalReasoningModel(cfg)
    m = LMHead(hrm, LMHeadConfig(vocab_size=tok.vocab_size))

    # Curriculum mode populates fields
    curr = _build_ckpt_config(
        m, tok, cfg, max_len=64, batch_size=4,
        curriculum_rung="R2", curriculum_seed=99,
        replay_ratio=0.45, prior_rungs=["R0", "R1"],
    )
    assert curr["curriculum_rung"] == "R2"
    assert curr["curriculum_seed"] == 99
    assert curr["replay_ratio"] == 0.45
    assert curr["prior_rungs"] == ["R0", "R1"]

    # GSM8k mode omits them
    gsm = _build_ckpt_config(m, tok, cfg, max_len=64, batch_size=4)
    assert "curriculum_rung" not in gsm
    assert "replay_ratio" not in gsm
    assert "prior_rungs" not in gsm


# ============================================================================ #
# --replay-rungs explicit-priors + DIAGNOSIS_ONLY_RUNGS auto-exclude
# (codex msg 1779475454122-1512da3b structural fix after R1b2a v2 lowmult
#  confounded fail at ddcc943)
# ============================================================================ #

from calm.hrm_text_158.curriculum.replay import (
    DIAGNOSIS_ONLY_RUNGS,
    _resolve_prior_rungs,
)


def test_replay_diagnosis_only_rungs_constants() -> None:
    """DIAGNOSIS_ONLY_RUNGS current membership per codex msgs
    1779475454122-1512da3b + 1779478819906-0e30503e + 1779479973262-6d7445d2 +
    1779483673737-20ff22ab + 1779488238721-49f03cc9 (R1b4v2 advance) +
    1779523412979-ff88b885 (R1b5 added to chain):
    - R1b2a: failed v1+v2 lowmult
    - R1b: legacy 3-template
    - R1b4: K=3 v1 failed at 7b53368 (measurement bug — 2-row one_digit
            heldout sampled ~22× via rng.choice). R1b4v2 is successor.
    - R2: failed v1 (0.085) AND v2 n_train=8000 (0.185)
    - R2a: failed v1 (0.045); variable-B blocker. R1b3 is the active
           constant K=2 successor.
    R1b2 stays OUT (PASSED via R1b2_v2_replay50 at c2686cc).
    R1b3 stays OUT (PASSED via v2 schedule at 175d327).
    R1b4v2 stays OUT (ADVANCED at b368b81 via seed=2 canonical head).
    R1b5 stays OUT (active-chain target K=4 added per 1779523412979;
    not yet attempted)."""
    assert "R1b2a" in DIAGNOSIS_ONLY_RUNGS
    assert "R1b" in DIAGNOSIS_ONLY_RUNGS
    assert "R1b4" in DIAGNOSIS_ONLY_RUNGS, (
        "R1b4 must be in DIAGNOSIS_ONLY_RUNGS after v1 fail (7b53368); "
        "R1b4v2 is the active one_digit-exhaustive successor"
    )
    assert "R2" in DIAGNOSIS_ONLY_RUNGS
    assert "R2a" in DIAGNOSIS_ONLY_RUNGS, (
        "R2a must be in DIAGNOSIS_ONLY_RUNGS after v1 fail (558fcc1); "
        "R1b3 is the active constant K=2 successor"
    )
    assert "R1b2" not in DIAGNOSIS_ONLY_RUNGS, (
        "R1b2 must stay OUT (PASSED at c2686cc)"
    )
    assert "R1b4v2" not in DIAGNOSIS_ONLY_RUNGS, (
        "R1b4v2 must stay OUT (ADVANCED at b368b81 via seed=2 head)"
    )
    assert "R1b5" not in DIAGNOSIS_ONLY_RUNGS, (
        "R1b5 must stay OUT (active-chain target, not yet attempted)"
    )
    # Sanity: well-known good/active rungs NEVER diagnosis-only
    for r in ("R0", "R1", "R1b1", "R1b3", "R1b4v2", "R3"):
        assert r not in DIAGNOSIS_ONLY_RUNGS


def test_resolve_prior_rungs_unset_positional_default() -> None:
    """replay_rungs_arg=None -> positional RUNG_NAMES[:cur_idx] minus
    DIAGNOSIS_ONLY_RUNGS minus R7."""
    from calm.hrm_text_158.curriculum.generators import RUNG_NAMES
    # R1b1: cur_idx=2, positional=[R0, R1]; no diagnosis-only at those idx
    assert _resolve_prior_rungs("R1b1", None) == ["R0", "R1"]
    # R1b2: cur_idx=4, positional=[R0, R1, R1b1, R1b2a]; minus diagnosis
    # ({R1b2a, R1b, R1b4, R2, R2a}) -> [R0, R1, R1b1]
    assert _resolve_prior_rungs("R1b2", None) == ["R0", "R1", "R1b1"]
    # R1b3: cur_idx=5, positional=[R0, R1, R1b1, R1b2a, R1b2]; minus
    # diagnosis -> [R0, R1, R1b1, R1b2]
    assert _resolve_prior_rungs("R1b3", None) == ["R0", "R1", "R1b1", "R1b2"]
    # R1b4: cur_idx=6, positional=[R0, R1, R1b1, R1b2a, R1b2, R1b3];
    # minus diagnosis -> [R0, R1, R1b1, R1b2, R1b3]. (R1b4 is current,
    # not in prior list; the diagnosis-only mark on R1b4 affects later
    # rungs.)
    assert _resolve_prior_rungs("R1b4", None) == ["R0", "R1", "R1b1", "R1b2", "R1b3"]
    # R1b4v2: cur_idx=7, positional=[R0, R1, R1b1, R1b2a, R1b2, R1b3,
    # R1b4]; minus diagnosis (R1b4 now in!) -> [R0, R1, R1b1, R1b2, R1b3]
    # — critically NOT including R1b4. Codex msg 1779483673737-20ff22ab
    # provenance preservation: R1b4 is diagnosis-only, never auto-replayed.
    assert _resolve_prior_rungs("R1b4v2", None) == ["R0", "R1", "R1b1", "R1b2", "R1b3"]
    # R1b5: cur_idx=8, positional includes R1b4v2 (active) but excludes
    # R1b4 (diagnosis) -> [R0, R1, R1b1, R1b2, R1b3, R1b4v2]. Codex msg
    # 1779523412979-ff88b885: R1b5 inherits seed=2 head from R1b4v2.
    assert _resolve_prior_rungs("R1b5", None) == ["R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2"]
    # R3: cur_idx=12 (after R1b5 insertion at 8 shifted R3 from 11),
    # positional includes R1b4v2 AND R1b5 (active); excludes diagnosis-only
    # R1b2a/R1b/R1b4/R2/R2a -> [R0, R1, R1b1, R1b2, R1b3, R1b4v2, R1b5]
    assert _resolve_prior_rungs("R3", None) == ["R0", "R1", "R1b1", "R1b2", "R1b3", "R1b4v2", "R1b5"]


def test_resolve_prior_rungs_explicit_override_basic() -> None:
    """Explicit --replay-rungs overrides positional. Accepted list returns."""
    out = _resolve_prior_rungs("R1b2", "R0,R1,R1b1")
    assert out == ["R0", "R1", "R1b1"]


def test_resolve_prior_rungs_explicit_excludes_default_diagnosis() -> None:
    """Explicit override CAN choose subset; e.g. just R0+R1 for R1b2."""
    out = _resolve_prior_rungs("R1b2", "R0,R1")
    assert out == ["R0", "R1"]
    # R1b1 explicitly excluded by operator choice (allowed)


def test_resolve_prior_rungs_explicit_whitespace_handling() -> None:
    """Whitespace around commas accepted; empty entries malformed."""
    out = _resolve_prior_rungs("R1b2", " R0 , R1 , R1b1 ")
    assert out == ["R0", "R1", "R1b1"]


def test_resolve_prior_rungs_explicit_empty_string_raises() -> None:
    """--replay-rungs '' must raise."""
    with pytest.raises(ValueError, match="cannot be empty"):
        _resolve_prior_rungs("R1b2", "")


def test_resolve_prior_rungs_explicit_empty_entry_raises() -> None:
    """--replay-rungs 'R0,,R1' (empty mid-list) must raise."""
    with pytest.raises(ValueError, match="empty entry|malformed"):
        _resolve_prior_rungs("R1b2", "R0,,R1")


def test_resolve_prior_rungs_unknown_rung_raises() -> None:
    """--replay-rungs entry not in RUNG_NAMES must raise."""
    with pytest.raises(ValueError, match="not in rung_names"):
        _resolve_prior_rungs("R1b2", "R0,R1,UNKNOWN")


def test_resolve_prior_rungs_current_rung_raises() -> None:
    """--replay-rungs cannot include current rung (would replay self)."""
    with pytest.raises(ValueError, match="cannot include current rung"):
        _resolve_prior_rungs("R1b2", "R0,R1b2,R1")


def test_resolve_prior_rungs_future_rung_raises() -> None:
    """--replay-rungs entry at index >= cur_idx must raise."""
    # R1b1 at index 2; R2 at index 6 (future)
    with pytest.raises(ValueError, match="future rungs cannot be replay"):
        _resolve_prior_rungs("R1b1", "R0,R1,R2")


def test_resolve_prior_rungs_r7_raises() -> None:
    """--replay-rungs R7 must raise (GSM8k served separately)."""
    with pytest.raises(ValueError, match="generator-incompatible|served separately"):
        _resolve_prior_rungs("R2", "R0,R1,R7")


def test_resolve_prior_rungs_duplicate_raises() -> None:
    """--replay-rungs duplicate entries must raise (would overweight prior)."""
    with pytest.raises(ValueError, match="duplicate"):
        _resolve_prior_rungs("R1b2", "R0,R1,R0,R1b1")


def test_resolve_prior_rungs_explicit_diagnosis_warns() -> None:
    """--replay-rungs explicitly including diagnosis-only emits WARN
    (caught via callback) but returns the list unchanged."""
    warns = []
    out = _resolve_prior_rungs(
        "R2",
        "R0,R1,R1b1,R1b2a",
        warn_callback=lambda msg: warns.append(msg),
    )
    assert out == ["R0", "R1", "R1b1", "R1b2a"]
    assert len(warns) == 1, f"expected 1 warn; got {warns}"
    assert "R1b2a" in warns[0]
    assert "diagnosis-only" in warns[0].lower()


def test_resolve_prior_rungs_unknown_curriculum_rung_raises() -> None:
    """curriculum_rung itself not in RUNG_NAMES must raise."""
    with pytest.raises(ValueError, match="not in rung_names"):
        _resolve_prior_rungs("UNKNOWN_RUNG", None)


def test_resolve_prior_rungs_r0_no_priors_unset() -> None:
    """R0 has no priors with unset replay_rungs; returns []."""
    assert _resolve_prior_rungs("R0", None) == []


def test_resolve_prior_rungs_r0_explicit_with_self_raises() -> None:
    """--replay-rungs R0 at R0 launch raises (current rung)."""
    with pytest.raises(ValueError, match="cannot include current rung"):
        _resolve_prior_rungs("R0", "R0")
