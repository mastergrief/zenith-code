"""F.3f-a — runtime `--l0c-hard-weight` override for the L0c_exhaustive_2digit
TRAIN sampler (codex msg 1779703363270, after F.3e rejected 3x for easy-stratum
starvation). Tiny code slice: helper mutates the train spec; trainer wires a CLI
flag with a rung-specific fail-fast. Default None keeps spec 3.0 (F.3d-b
unchanged). No model / no GPU for the unit tests; one subprocess fail-fast test.

The autouse fixture snapshots/restores the spec hard_weight so the mutating
tests cannot leak 1.5 into the F.3d-b 3x assertions (shared pytest process).
"""
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import random  # noqa: E402

import pytest  # noqa: E402

from calm.hrm_text_158.curriculum.generators import (  # noqa: E402
    set_l0c_exhaustive_2digit_hard_weight,
    _gen_l0c_exhaustive_2digit,
    _enumerate_partition_l0c_exhaustive,
    _l0c_is_hard,
    _RUNG_SPEC,
)


@pytest.fixture(autouse=True)
def _restore_hard_weight():
    orig = _RUNG_SPEC["L0c_exhaustive_2digit"]["train"].get("hard_weight")
    try:
        yield
    finally:
        _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]["hard_weight"] = orig


def _hard_frac(rows):
    return sum(1 for r in rows if _l0c_is_hard(r["question"], r["expected"])) / len(rows)


# --------------------------------------------------------------------------- #
# Default unchanged + helper override
# --------------------------------------------------------------------------- #

def test_default_spec_hard_weight_is_3():
    assert _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]["hard_weight"] == 3.0


def test_helper_overrides_spec_and_returns_effective():
    eff = set_l0c_exhaustive_2digit_hard_weight(1.5)
    assert eff == 1.5
    assert _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]["hard_weight"] == 1.5


def test_override_applies_to_data_generation():
    # 1.5 -> 1.5*1170/(1.5*1170+85) ~ 0.954 hard share (gentler than 3x's 0.976).
    set_l0c_exhaustive_2digit_hard_weight(1.5)
    spec = _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="train")
    frac = _hard_frac(rows)
    assert 0.94 <= frac <= 0.965, "1.5x hard-share %.4f not ~0.954" % frac


def test_override_is_gentler_than_default_3x():
    rng = random.Random(17)
    rows3 = _gen_l0c_exhaustive_2digit(rng, _RUNG_SPEC["L0c_exhaustive_2digit"]["train"], 8000, seed=17, split="train")
    set_l0c_exhaustive_2digit_hard_weight(1.5)
    rng2 = random.Random(17)
    rows15 = _gen_l0c_exhaustive_2digit(rng2, _RUNG_SPEC["L0c_exhaustive_2digit"]["train"], 8000, seed=17, split="train")
    assert _hard_frac(rows15) < _hard_frac(rows3)  # 1.5x starves easy LESS than 3x


def test_held_out_unaffected_by_override():
    # Held stays uniform over the full pool even with the train override.
    set_l0c_exhaustive_2digit_hard_weight(1.5)
    spec = _RUNG_SPEC["L0c_exhaustive_2digit"]["held_out"]
    rng = random.Random(17)
    rows = _gen_l0c_exhaustive_2digit(rng, spec, 8000, seed=17, split="held_out")
    _, held_pool = _enumerate_partition_l0c_exhaustive(17)
    pool_hard = sum(1 for r in held_pool if _l0c_is_hard(r["question"], r["expected"])) / len(held_pool)
    assert abs(_hard_frac(rows) - pool_hard) < 0.05


# --------------------------------------------------------------------------- #
# Fail-fast: --l0c-hard-weight with a non-target rung errors BEFORE ckpt load
# --------------------------------------------------------------------------- #

def test_non_target_rung_rejects_fail_fast():
    import subprocess
    p = subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts", "train_hrm_text_158.py"),
         "--curriculum-rung", "L0c", "--use-broad-tokenizer",
         "--l0c-hard-weight", "1.5",
         "--load-from", "/nonexistent_ckpt_for_l0chw_test.pt"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _REPO}, timeout=120)
    combined = p.stdout + p.stderr
    assert p.returncode != 0, combined
    assert "--l0c-hard-weight is only valid with" in combined, combined
    assert "L0c_exhaustive_2digit" in combined, combined
    # fail-fast: must error before ever loading the ckpt.
    assert "No such file" not in combined and "FileNotFound" not in combined, combined


@pytest.mark.parametrize("bad", ["0", "-1.5"])
def test_non_positive_hard_weight_rejects_fail_fast(bad):
    # Even with the TARGET rung, a non-positive weight must fail fast (0 would
    # silently make TRAIN all-easy; negative would fail late in rng.choices),
    # before any ckpt load / data-gen (codex msg 1779703935958).
    import subprocess
    p = subprocess.run(
        [sys.executable, os.path.join(_REPO, "scripts", "train_hrm_text_158.py"),
         "--curriculum-rung", "L0c_exhaustive_2digit", "--use-broad-tokenizer",
         "--l0c-hard-weight", bad,
         "--load-from", "/nonexistent_ckpt_for_l0chw_test.pt"],
        capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": _REPO}, timeout=120)
    combined = p.stdout + p.stderr
    assert p.returncode != 0, combined
    assert "--l0c-hard-weight must be > 0" in combined, combined
    assert "No such file" not in combined and "FileNotFound" not in combined, combined


if __name__ == "__main__":
    # Manual run without pytest fixtures: snapshot/restore by hand.
    _orig = _RUNG_SPEC["L0c_exhaustive_2digit"]["train"].get("hard_weight")
    try:
        for _name, _fn in sorted(globals().items()):
            if _name.startswith("test_") and callable(_fn) and _name != "test_non_target_rung_rejects_fail_fast":
                _fn()
                _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]["hard_weight"] = _orig
                print(f"  {_name}: PASS")
    finally:
        _RUNG_SPEC["L0c_exhaustive_2digit"]["train"]["hard_weight"] = _orig
    print("L0c hard-weight-override tests: PASS (run pytest for the subprocess fail-fast test)")
