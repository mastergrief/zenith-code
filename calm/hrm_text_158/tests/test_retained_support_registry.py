"""Tests for the retained-support consistency registry (codex registry slice
msg 1779656084090, Step 1 assertions).

Covers: stable count/hash/order for L0b + math_a0; math_a0 contains
`what is 10 minus 1?`@9; seed handling (L0b seed-dependent, math_a0
seed-independent); generic K-cyclic sampler + per-support seed namespaces;
ckpt metadata (retained_support_profile source-of-truth; legacy l0b fields
ONLY when L0b-only); train() guards for invalid / duplicate / conflicting /
unauthorized profiles fire before any training work. No GPU / no model load.
"""
import importlib.util
import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

_spec = importlib.util.spec_from_file_location(
    "_train_hrm_text_158", os.path.join(_REPO, "scripts", "train_hrm_text_158.py")
)
_thr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_thr)

_support = _thr._retained_support
_sampler_seed = _thr._retained_sampler_seed
_Sampler = _thr._RetainedSupportSampler
_REGISTRY = _thr._RETAINED_SUPPORT_REGISTRY

# L0b seed-17 canonical hash, pinned from the F.2f runs (support_hash logged).
_L0B_SEED17_HASH = "89174273d21845bc"


# --------------------------------------------------------------------------- #
# Registry membership + per-support snapshot count/order/hash
# --------------------------------------------------------------------------- #

def test_registry_names():
    assert _REGISTRY == (
        "L0b",
        "L0c",
        "math_a0",
        "math_r1b2_minus_one",
        "l0c_exhaustive",
        "L0c2-K1-identity-2digit-full",
        "L0c2-K2-addition-120",
    )


_IDENTITY_FULL_HASH = "bf43ff7354b64c4e"
_L0C2_K2_ADDITION_120_HASH = "8b29072411bb9c71"


def test_l0c2_k1_identity_full_retained_support_snapshot():
    rows, h = _support("L0c2-K1-identity-2digit-full", 17)
    assert len(rows) == 90
    assert h == _IDENTITY_FULL_HASH
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert rows[0] == ("10 equals what?", 10, "coverage_teen")
    assert rows[-1] == ("99 equals what?", 99, "coverage_tens_9")
    assert all(q == f"{e} equals what?" for q, e, _bucket in rows)


def test_l0c2_k1_identity_full_retained_support_seed_independent():
    assert _support("L0c2-K1-identity-2digit-full", 17)[1] == \
        _support("L0c2-K1-identity-2digit-full", 42)[1]


def test_l0c2_k2_addition_120_retained_support_snapshot():
    rows, h = _support("L0c2-K2-addition-120", 17)
    assert len(rows) == 120
    assert h == _L0C2_K2_ADDITION_120_HASH
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert rows[0] == ("19 plus 1 equals what?", 20, "20s:k_1:carry:ones_0")
    assert rows[-1] == ("45 plus 4 equals what?", 49, "40s:k_4:no_carry:ones_9")
    assert all(q.endswith(" equals what?") and " plus " in q for q, _e, _bucket in rows)


def test_l0c2_k2_addition_120_retained_support_matches_audit_and_not_k5to8():
    from calm.hrm_text_158.curriculum.generators import (
        _l0c2k2_addition_120_enumerate,
        _l0c2k2_addition_120_k5to8_enumerate,
    )
    from calm.hrm_text_158.curriculum.language_supports import (
        build_l0c2k2_addition_120_support,
    )
    rows, _ = _support("L0c2-K2-addition-120", 17)
    retained = {(q, e) for q, e, _bucket in rows}
    audit = {
        (q, e)
        for _surface, pairs in build_l0c2k2_addition_120_support(17).items()
        for q, e, _bucket in pairs
    }
    k1to4 = {(r["question"], r["expected"]) for r in _l0c2k2_addition_120_enumerate()}
    k5to8 = {
        (r["question"], r["expected"])
        for r in _l0c2k2_addition_120_k5to8_enumerate()
    }
    assert len(rows) == len(retained) == len(audit) == len(k1to4) == 120
    assert retained == audit == k1to4
    assert retained.isdisjoint(k5to8)


def test_l0c2_k2_addition_120_retained_support_seed_independent():
    assert _support("L0c2-K2-addition-120", 17)[1] == \
        _support("L0c2-K2-addition-120", 42)[1]


# --------------------------------------------------------------------------- #
# l0c_exhaustive: dormant language-density support (codex msg 1779693537447).
# Registry-addressable now; NOT in any recipe default until banked.
# --------------------------------------------------------------------------- #

_L0C_EXH_HASH = "3209aa0a6461d916"


def test_l0c_exhaustive_snapshot():
    rows, h = _support("l0c_exhaustive", 17)
    assert len(rows) == 1255, f"expected 1255 rows, got {len(rows)}"
    assert h == _L0C_EXH_HASH
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert all(q.endswith(" equals what?") for q, _e, _sr in rows)


def test_l0c_exhaustive_seed_independent():
    assert _support("l0c_exhaustive", 17)[1] == _support("l0c_exhaustive", 42)[1], \
        "exhaustive L0c is seed-independent (derived from math A0)"


def test_l0c_exhaustive_source_rungs_match_math_a0():
    from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports
    rows, _ = _support("l0c_exhaustive", 17)
    assert {sr for _q, _e, sr in rows} == set(build_exhaustive_supports().keys())


def test_l0b_support_snapshot():
    rows, h = _support("L0b", 17)
    assert len(rows) == 230
    assert h == _L0B_SEED17_HASH  # bit-identical to the pre-registry L0b helper
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert all(q.startswith("calculate ") and q.endswith(".") for q, _e, _sr in rows)


# --------------------------------------------------------------------------- #
# L0c: F.4c retained-support — protects the bounded L0c `<expr> equals what?`
# surface F.4b left unprotected (no replay, no support) which capped LANG-690.
# --------------------------------------------------------------------------- #

def test_l0c_support_snapshot():
    rows, h = _support("L0c", 17)
    assert len(rows) == 230, f"expected 230 L0c rows, got {len(rows)}"
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert all(q.endswith(" equals what?") for q, _e, _sr in rows)
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)


def test_l0c_matches_canonical_language_support_path():
    # Builder returns EXACTLY the canonical bounded L0c support (same path as
    # build_language_supports()["L0c"]), modulo the retained-support stable sort.
    from calm.hrm_text_158.curriculum.language_supports import build_language_supports
    rows, _ = _support("L0c", 17)
    canonical = [(q, e, sr) for (q, e, sr) in build_language_supports(17)["L0c"]]
    key = lambda r: (r[2], r[0], r[1])
    assert sorted(rows, key=key) == sorted(canonical, key=key)


def test_l0c_seed_dependent_and_deterministic():
    assert _support("L0c", 17)[1] != _support("L0c", 42)[1], "L0c support is seed-dependent"
    r1, h1 = _support("L0c", 17)
    r2, h2 = _support("L0c", 17)
    assert r1 == r2 and h1 == h2  # byte-identical on repeat


def test_l0c_sampler_namespace():
    # L0c (non-L0b) uses the generic "retained:L0c" namespace, distinct from others.
    assert _sampler_seed("L0c", 17) == _thr._stable_curriculum_seed(17, "retained:L0c")
    assert _sampler_seed("L0c", 17) != _sampler_seed("L0b", 17)
    assert _sampler_seed("L0c", 17) != _sampler_seed("math_a0", 17)


def test_l0c2_k1_identity_full_sampler_namespace():
    name = "L0c2-K1-identity-2digit-full"
    assert _sampler_seed(name, 17) == _thr._stable_curriculum_seed(17, f"retained:{name}")
    assert _sampler_seed(name, 17) != _sampler_seed("L0b", 17)


def test_l0c2_k2_addition_120_sampler_namespace():
    name = "L0c2-K2-addition-120"
    assert _sampler_seed(name, 17) == _thr._stable_curriculum_seed(17, f"retained:{name}")
    assert _sampler_seed(name, 17) != _sampler_seed("L0b", 17)


def test_math_a0_support_snapshot():
    rows, h = _support("math_a0", 17)
    assert len(rows) == 1255
    assert len(h) == 16 and all(c in "0123456789abcdef" for c in h)
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert all(q.startswith("what is ") and q.endswith("?") for q, _e, _sr in rows)


def test_math_a0_contains_10_minus_1_at_9():
    # The row F.2f regressed; protecting it via parent-KL is the whole point.
    rows, _ = _support("math_a0", 17)
    assert ("what is 10 minus 1?", 9, "R1b2") in rows


# --------------------------------------------------------------------------- #
# math_r1b2_minus_one: concentrated registry-derived R1b2 class overlay (F.2h)
# --------------------------------------------------------------------------- #

# Canonical hash pinned from the seed-17 build (codex msg 1779659487346).
_R1B2_HASH = "8c765badc7365890"


def test_math_r1b2_minus_one_snapshot():
    from calm.hrm_text_158.curriculum.exhaustive_supports import build_exhaustive_supports
    rows, h = _support("math_r1b2_minus_one", 17)
    # Codex: pin 99, but fail LOUDLY against the live source count if it drifts.
    src = len(build_exhaustive_supports()["R1b2"])
    assert len(rows) == src, f"support count {len(rows)} != source R1b2 count {src}"
    assert len(rows) == 99, f"expected 99 R1b2 rows, got {len(rows)}"
    assert h == _R1B2_HASH
    assert rows == sorted(rows, key=lambda r: (r[2], r[0], r[1]))
    assert all(sr == "R1b2" for _q, _e, sr in rows), "all rows must be source_rung R1b2"
    assert all(q.startswith("what is ") and "minus 1?" in q for q, _e, _sr in rows)


def test_math_r1b2_minus_one_contains_10_minus_1_at_9():
    # The exact row F.2g failed to protect; the class overlay must cover it.
    rows, _ = _support("math_r1b2_minus_one", 17)
    assert ("what is 10 minus 1?", 9, "R1b2") in rows


def test_math_r1b2_minus_one_seed_independent():
    assert _support("math_r1b2_minus_one", 17)[1] == _support("math_r1b2_minus_one", 42)[1], \
        "R1b2 class is exhaustive/seed-independent"


def test_math_r1b2_minus_one_is_subset_of_math_a0():
    # The overlay is a concentrated subset of the broad support, not new rows.
    r1b2, _ = _support("math_r1b2_minus_one", 17)
    a0, _ = _support("math_a0", 17)
    a0_set = set(a0)
    assert set(r1b2) <= a0_set, "every R1b2-class row must already be in math_a0"
    assert len(r1b2) < len(a0), "overlay must be strictly smaller (concentrated)"


def test_math_r1b2_minus_one_sampler_namespace():
    # Non-L0b support uses the "retained:<name>" namespace (not L0b's legacy ns).
    assert _sampler_seed("math_r1b2_minus_one", 17) == _thr._stable_curriculum_seed(
        17, "retained:math_r1b2_minus_one")
    assert _sampler_seed("math_r1b2_minus_one", 17) != _sampler_seed("math_a0", 17)


def test_determinism_same_name_seed():
    for name, seed in (
        ("L0b", 17),
        ("L0c", 17),
        ("math_a0", 17),
        ("L0c2-K1-identity-2digit-full", 17),
        ("L0c2-K2-addition-120", 17),
    ):
        r1, h1 = _support(name, seed)
        r2, h2 = _support(name, seed)
        assert r1 == r2 and h1 == h2


def test_l0b_seed_dependent_math_a0_seed_independent():
    assert _support("L0b", 17)[1] != _support("L0b", 42)[1], "L0b support is seed-dependent"
    assert _support("math_a0", 17)[1] == _support("math_a0", 42)[1], "math_a0 is exhaustive/seed-independent"


def test_unknown_support_name_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown retained support"):
        _support("bogus", 17)


# --------------------------------------------------------------------------- #
# Per-support sampler seed namespaces + generic K-cyclic sampler
# --------------------------------------------------------------------------- #

def test_sampler_seed_namespaces():
    # L0b keeps the legacy "l0b_consistency" namespace (bit-compat); others use "retained:<name>".
    assert _sampler_seed("L0b", 17) == _thr._stable_curriculum_seed(17, "l0b_consistency")
    assert _sampler_seed("math_a0", 17) == _thr._stable_curriculum_seed(17, "retained:math_a0")
    assert _sampler_seed("L0b", 17) != _sampler_seed("math_a0", 17)


def test_l0b_alias_sampler_matches_namespace():
    s = _thr._L0bConsistencySampler(n=230, seed=17, batch=8)
    assert s.support_seed == _sampler_seed("L0b", 17)


def test_generic_sampler_determinism_and_coverage():
    seed = _sampler_seed("math_a0", 17)
    a = _Sampler(n=1255, support_seed=seed, batch=8)
    b = _Sampler(n=1255, support_seed=seed, batch=8)
    assert a.perm == b.perm
    assert [a.next_indices() for _ in range(3)] == [b.next_indices() for _ in range(3)]
    # different support_seed -> different perm
    c = _Sampler(n=1255, support_seed=_sampler_seed("L0b", 17), batch=8)
    assert c.perm != a.perm
    # perm is a permutation; one cyclic pass covers all rows
    fresh = _Sampler(n=1255, support_seed=seed, batch=8)
    assert sorted(fresh.perm) == list(range(1255))
    seen = set()
    for _ in range((1255 + 7) // 8):
        seen.update(fresh.next_indices())
    assert seen == set(range(1255))


def test_sampler_rejects_bad_args():
    import pytest
    with pytest.raises(ValueError):
        _Sampler(n=0, support_seed=1, batch=8)
    with pytest.raises(ValueError):
        _Sampler(n=10, support_seed=1, batch=0)


# --------------------------------------------------------------------------- #
# ckpt metadata: retained_support_profile source-of-truth; legacy fields L0b-only
# --------------------------------------------------------------------------- #

class _CfgStub:
    max_seq_len = 384; n_layers = 8; hidden_size = 512; num_heads = 4
    expansion = 4; H_cycles = 2; L_cycles = 3; half_layers = True
    bp_warmup_ratio = 0.2; bp_min_steps = 2; bp_max_steps = 5
    norm_type = "rms"; norm_eps = 1e-5; rope_theta = 1e4
    attn_type = "a"; init_type = "i"; pos_emb_type = "p"; use_ternary_bulk = True


class _TokStub:
    vocab_size = 260
    normalizer_version = "byte_utf8_v1"
    def vocab_as_list(self):
        return []


def _meta(name, weight, count, h):
    return {"name": name, "weight": weight, "batch": 8, "count": count, "hash": h}


def test_ckpt_meta_l0b_only_keeps_legacy_fields():
    out = _thr._build_ckpt_config(
        None, _TokStub(), _CfgStub(), 384, 8,
        parent_consistency_weight=1.0, parent_consistency_temp=1.0,
        retained_support_meta=[_meta("L0b", 1.0, 230, _L0B_SEED17_HASH)],
        retained_l0b_only=True,
    )
    assert "retained_support_profile" in out
    assert out["retained_support_profile"][0]["name"] == "L0b"
    # L0b-only ⇒ legacy fields ALSO present for back-compat.
    assert out["l0b_consistency_weight"] == 1.0
    assert out["l0b_consistency_support_hash"] == _L0B_SEED17_HASH
    assert out["l0b_consistency_support_count"] == 230


def test_ckpt_meta_mixed_profile_no_legacy_fields():
    out = _thr._build_ckpt_config(
        None, _TokStub(), _CfgStub(), 384, 8,
        parent_consistency_weight=1.0, parent_consistency_temp=1.0,
        retained_support_meta=[_meta("L0b", 1.0, 230, "aa"), _meta("math_a0", 1.0, 1255, "bb")],
        retained_l0b_only=False,
    )
    names = [s["name"] for s in out["retained_support_profile"]]
    assert names == ["L0b", "math_a0"]
    # Mixed ⇒ do NOT pretend it is an old L0b-only checkpoint.
    assert "l0b_consistency_weight" not in out
    assert "l0b_consistency_support_hash" not in out


# --------------------------------------------------------------------------- #
# train() profile guards fire before any training work
# --------------------------------------------------------------------------- #

def test_invalid_support_name_in_profile_raises():
    import pytest
    with pytest.raises(ValueError, match="unknown retained support"):
        _thr.train(retained_support_profile=[("bogus", 1.0)])


def test_duplicate_support_names_raise():
    import pytest
    with pytest.raises(ValueError, match="duplicate retained-support names"):
        _thr.train(retained_support_profile=[("L0b", 1.0), ("L0b", 0.5)])


def test_legacy_explicit_l0b_conflict_raises():
    import pytest
    with pytest.raises(ValueError, match="conflicting L0b config"):
        _thr.train(retained_support_profile=[("L0b", 1.0)], l0b_consistency_weight=0.5)


def test_negative_profile_weight_raises():
    import pytest
    with pytest.raises(ValueError, match="must be >= 0"):
        _thr.train(retained_support_profile=[("L0b", -1.0)])


def test_profile_requires_load_from():
    import pytest
    with pytest.raises(ValueError, match="requires --load-from"):
        _thr.train(retained_support_profile=[("math_a0", 1.0)])


def test_identity_full_profile_name_reaches_load_from_guard():
    import pytest
    with pytest.raises(ValueError, match="requires --load-from"):
        _thr.train(retained_support_profile=[("L0c2-K1-identity-2digit-full", 1.0)])


def test_l0c2_k2_addition_120_profile_name_reaches_load_from_guard():
    import pytest
    with pytest.raises(ValueError, match="requires --load-from"):
        _thr.train(retained_support_profile=[("L0c2-K2-addition-120", 1.0)])


def test_profile_requires_curriculum_mode():
    import pytest
    with pytest.raises(ValueError, match="requires curriculum"):
        _thr.train(retained_support_profile=[("math_a0", 1.0)], load_from="x", curriculum_rung=None)


if __name__ == "__main__":
    for _name, _fn in sorted(globals().items()):
        if _name.startswith("test_") and callable(_fn):
            _fn()
            print(f"  {_name}: PASS")
    print("retained-support-registry tests: PASS")
