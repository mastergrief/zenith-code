"""CPU-static tests for signed_utility_receipt_pre_post_compare (PLAN v16)."""
from __future__ import annotations

import pytest

from calm.hrm_text_158.native_full_stack.signed_utility_receipt_pre_post_compare import (
    pre_post_compare,
    pre_post_compare_git,
    pre_post_compare_hash,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64
REV_A = "a" * 40
REV_B = "b" * 40
REV_C = "c" * 40
REV_D = "d" * 40

_MAL_SHA = [
    ("uppercase", "A" * 64),
    ("wrong_len", "a" * 63),
    ("non_hex", "g" * 64),
    ("empty", ""),
    ("non_string", 123),
]
_MAL_REV = [
    ("uppercase", "A" * 40),
    ("wrong_len", "a" * 39),
    ("non_hex", "g" * 40),
    ("empty", ""),
    ("non_string", object()),
]

# Explicit executable expected-source cases only (no skip-producing cartesian).
# (ex_state, meta_state) with optional unequal flag for valid×valid.
_HASH_EXPECTED_CASES = [
    ("absent", "absent", None, None),
    ("valid", "absent", SHA_A, None),
    ("absent", "valid", None, SHA_A),
    ("valid", "valid_eq", SHA_A, SHA_A),
    ("valid", "valid_ne_ex_a_meta_b", SHA_A, SHA_B),
    ("valid", "valid_ne_ex_b_meta_a", SHA_B, SHA_A),
    ("malformed", "absent", "BAD", None),
    ("absent", "malformed", None, "WORSE"),
    ("valid", "malformed", SHA_A, "WORSE"),
    ("malformed", "valid", "BAD", SHA_A),
    ("malformed", "malformed", "BAD", "WORSE"),
]


def _hash(sha=SHA_A, **extra):
    d = {"path": "/x", "sha256": sha}
    d.update(extra)
    return d


def _git(head=REV_A, upstream=REV_A, tree=REV_B, **extra):
    d = {"repo_root": "/r", "head": head, "upstream": upstream, "tree": tree}
    d.update(extra)
    return d


def _strip_hash(m):
    out = dict(m)
    out.pop("expected", None)
    return out


def _strip_git(m):
    out = dict(m)
    out.pop("expected_head", None)
    out.pop("expected_tree", None)
    return out


def test_ambiguous_generic_raises():
    with pytest.raises(TypeError, match="ambiguous_pre_post_compare_removed"):
        pre_post_compare({}, {})


@pytest.mark.parametrize("ex_state,meta_state,ex_v,meta_v", _HASH_EXPECTED_CASES)
def test_hash_expected_full_matrix(ex_state, meta_state, ex_v, meta_v):
    post = _hash(SHA_A)
    if meta_v is not None:
        post["expected"] = meta_v
    r = pre_post_compare_hash(_hash(SHA_A), post, expected=ex_v)
    any_mal = "malformed" in ex_state or "malformed" in meta_state
    if any_mal:
        assert r["expected_effective"] is None
        assert r["expected_conflict"] is False
        assert r["post_matches_expected"] is None
        if "malformed" in ex_state:
            assert "expected" in r["malformed_fields"]
        if "malformed" in meta_state:
            assert "post.expected" in r["malformed_fields"]
    elif "valid_ne" in meta_state:
        assert r["expected_conflict"] is True
        assert r["expected_effective"] is None
        assert r["post_matches_expected"] is None
    elif ex_state == "absent" and meta_state == "absent":
        assert r["expected_effective"] is None
        assert r["post_matches_expected"] is None
        assert r["expected_conflict"] is False
    else:
        assert r["expected_conflict"] is False
        assert r["expected_effective"] == SHA_A
        assert r["post_matches_expected"] is True


@pytest.mark.parametrize("field", ["expected_head", "expected_tree"])
@pytest.mark.parametrize(
    "ex_state,meta_state,ex_kind,meta_kind",
    [
        ("absent", "absent", None, None),
        ("valid", "absent", "good", None),
        ("absent", "valid", None, "good"),
        ("valid", "valid_eq", "good", "good"),
        ("valid", "valid_ne_ex_good_meta_other", "good", "other"),
        ("valid", "valid_ne_ex_other_meta_good", "other", "good"),
        ("malformed", "absent", "bad", None),
        ("absent", "malformed", None, "worse"),
        ("valid", "malformed", "good", "worse"),
        ("malformed", "valid", "bad", "good"),
        ("malformed", "malformed", "bad", "worse"),
    ],
)
def test_git_expected_full_matrix(field, ex_state, meta_state, ex_kind, meta_kind):
    good = REV_A if field == "expected_head" else REV_B
    other = REV_C
    mapping = {"good": good, "other": other, "bad": "BAD", "worse": "WORSE", None: None}
    ex_v = mapping[ex_kind]
    meta_v = mapping[meta_kind]
    post = _git()
    if meta_v is not None:
        post[field] = meta_v
    kwargs = {"expected_head": REV_A, "expected_tree": REV_B}
    kwargs[field] = ex_v
    r = pre_post_compare_git(_git(), post, **kwargs)
    any_mal = "malformed" in ex_state or "malformed" in meta_state
    if any_mal:
        assert r["post_matches_expected"] is None
        assert r["expected_conflict"] is False
    elif "valid_ne" in meta_state:
        assert r["expected_conflict"] is True
        assert r["post_matches_expected"] is None
    elif ex_state == "absent" and meta_state == "absent":
        assert r["post_matches_expected"] is None
    else:
        assert r["expected_conflict"] is False
        assert r["post_matches_expected"] is True


@pytest.mark.parametrize("kind,bad", _MAL_SHA)
@pytest.mark.parametrize("where", ["pre.sha256", "post.sha256", "expected", "post.expected"])
def test_hash_malformed_classes(kind, bad, where):
    pre, post = _hash(), _hash()
    expected = SHA_A
    if where == "pre.sha256":
        pre = _hash(sha=bad)
    elif where == "post.sha256":
        post = _hash(sha=bad)
    elif where == "expected":
        expected = bad
    else:
        post = _hash(expected=bad)
        expected = None
    r = pre_post_compare_hash(pre, post, expected=expected)
    if where == "pre.sha256":
        assert r["pre_matches_post"] is None
        assert "pre.sha256" in r["malformed_fields"]
        assert r["post_matches_expected"] is True  # pre malformed does not null post match
    elif where == "post.sha256":
        assert r["pre_matches_post"] is None
        assert r["post_matches_expected"] is None
        assert "post.sha256" in r["malformed_fields"]
    else:
        assert r["post_matches_expected"] is None
        assert ("expected" if where == "expected" else "post.expected") in r["malformed_fields"]


@pytest.mark.parametrize("kind,bad", _MAL_REV)
@pytest.mark.parametrize("side", ["pre", "post"])
@pytest.mark.parametrize("field", ["head", "upstream", "tree"])
def test_git_observed_malformed_classes(kind, bad, side, field):
    pre, post = _git(), _git()
    target = pre if side == "pre" else post
    target[field] = bad
    r = pre_post_compare_git(pre, post, expected_head=REV_A, expected_tree=REV_B)
    assert r["pre_matches_post"] is None
    assert f"{side}.{field}" in r["malformed_fields"]
    if side == "post":
        assert r["post_matches_expected"] is None
    else:
        assert r["post_matches_expected"] is True  # malformed pre independence


@pytest.mark.parametrize("kind,bad", _MAL_REV)
@pytest.mark.parametrize("where", ["expected_head", "expected_tree", "post.expected_head", "post.expected_tree"])
def test_git_expected_malformed_classes(kind, bad, where):
    post = _git()
    kwargs = {"expected_head": REV_A, "expected_tree": REV_B}
    if where.startswith("post."):
        key = where.split(".", 1)[1]
        post[key] = bad
        kwargs[key] = None
    else:
        kwargs[where] = bad
    r = pre_post_compare_git(_git(), post, **kwargs)
    assert r["post_matches_expected"] is None
    assert r["expected_conflict"] is False
    assert where in r["malformed_fields"]


@pytest.mark.parametrize("side", ["pre", "post"])
def test_hash_remove_required_observed_field(side):
    pre, post = _hash(), _hash()
    target = dict(pre if side == "pre" else post)
    del target["sha256"]
    r = pre_post_compare_hash(target if side == "pre" else pre, post if side == "pre" else target, expected=SHA_A)
    assert r["pre_matches_post"] is None
    if side == "post":
        assert r["post_matches_expected"] is None


@pytest.mark.parametrize("side", ["pre", "post"])
@pytest.mark.parametrize("field", ["head", "upstream", "tree"])
def test_git_remove_each_required_observed_field(side, field):
    pre, post = _git(), _git()
    target = dict(pre if side == "pre" else post)
    del target[field]
    r = (
        pre_post_compare_git(target, post, expected_head=REV_A, expected_tree=REV_B)
        if side == "pre"
        else pre_post_compare_git(pre, target, expected_head=REV_A, expected_tree=REV_B)
    )
    assert r["pre_matches_post"] is None
    if side == "post":
        assert r["post_matches_expected"] is None


def test_both_observed_missing_hash_and_git_null():
    r = pre_post_compare_hash({"path": "/x"}, {"path": "/y"}, expected=SHA_A)
    assert r["pre_matches_post"] is None
    assert r["pre_matches_post"] is not True
    r2 = pre_post_compare_git({"repo_root": "/r"}, {"repo_root": "/r2"}, expected_head=REV_A, expected_tree=REV_B)
    assert r2["pre_matches_post"] is None
    assert r2["pre_matches_post"] is not True


@pytest.mark.parametrize("api", ["hash", "git"])
@pytest.mark.parametrize("side", ["pre", "post"])
def test_non_mapping_both_apis_both_sides(api, side):
    if api == "hash":
        r = (
            pre_post_compare_hash("not-a-map", _hash(), expected=SHA_A)
            if side == "pre"
            else pre_post_compare_hash(_hash(), ["nope"], expected=SHA_A)
        )
        assert side in r["malformed_fields"]
    else:
        r = (
            pre_post_compare_git(42, _git(), expected_head=REV_A, expected_tree=REV_B)
            if side == "pre"
            else pre_post_compare_git(_git(), 42, expected_head=REV_A, expected_tree=REV_B)
        )
        assert side in r["malformed_fields"]
    assert r["pre_matches_post"] is None


def test_hash_explicit_args_absent_from_projections_and_exact_strip():
    pre = _hash(expected=SHA_B, keep=1)
    post = _hash(expected=SHA_C, keep=1)
    r = pre_post_compare_hash(pre, post, expected=SHA_A)
    assert "expected" not in r["pre_observed_projection"]
    assert "expected" not in r["post_observed_projection"]
    assert r["pre_observed_projection"] == _strip_hash(pre)
    assert r["post_observed_projection"] == _strip_hash(post)


def test_git_explicit_args_absent_from_projections_and_exact_strip():
    pre = _git(expected_head=REV_C, expected_tree=REV_D, keep=1)
    post = _git(expected_head=REV_A, expected_tree=REV_B, keep=1)
    r = pre_post_compare_git(pre, post, expected_head=REV_A, expected_tree=REV_B)
    assert "expected_head" not in r["pre_observed_projection"]
    assert "expected_tree" not in r["post_observed_projection"]
    assert r["pre_observed_projection"] == _strip_git(pre)
    assert r["post_observed_projection"] == _strip_git(post)


def test_hash_clean_true_and_drift_false():
    assert pre_post_compare_hash(_hash(), _hash(), expected=SHA_A)["post_matches_expected"] is True
    assert pre_post_compare_hash(_hash(), _hash(SHA_B), expected=SHA_A)["post_matches_expected"] is False
    assert pre_post_compare_hash(_hash(), _hash())["pre_matches_post"] is True
    assert pre_post_compare_hash(_hash(), _hash(SHA_B))["pre_matches_post"] is False


def test_hash_path_value_and_presence_asymmetry():
    assert pre_post_compare_hash(_hash(path="/a"), _hash(path="/b"))["pre_matches_post"] is False
    pre = _hash()
    post = _hash()
    del post["path"]
    assert pre_post_compare_hash(pre, post)["pre_matches_post"] is False
    pre2 = _hash(extra=1)
    post2 = _hash()
    assert pre_post_compare_hash(pre2, post2)["pre_matches_post"] is False
    assert pre_post_compare_hash(_hash(extra=1), _hash(extra=2))["pre_matches_post"] is False


def test_hash_nested_json_equality():
    pre = _hash(extra={"n": [1, {"k": True}]})
    assert pre_post_compare_hash(pre, _hash(extra={"n": [1, {"k": True}]}))["pre_matches_post"] is True
    assert pre_post_compare_hash(pre, _hash(extra={"n": [1, {"k": False}]}))["pre_matches_post"] is False


def test_git_nested_json_equality():
    pre = _git(blob={"a": [1, {"k": True}]})
    assert pre_post_compare_git(pre, _git(blob={"a": [1, {"k": True}]}))["pre_matches_post"] is True
    assert pre_post_compare_git(pre, _git(blob={"a": [1, {"k": False}]}))["pre_matches_post"] is False


def test_git_repo_root_error_extra_value_and_presence_asymmetry():
    assert pre_post_compare_git(_git(repo_root="/a"), _git(repo_root="/b"))["pre_matches_post"] is False
    pre = _git()
    post = _git()
    del post["repo_root"]
    assert pre_post_compare_git(pre, post)["pre_matches_post"] is False
    assert pre_post_compare_git(_git(error=None), _git(error="e"))["pre_matches_post"] is False
    pre2 = _git()
    post2 = _git(error="e")
    assert pre_post_compare_git(pre2, post2)["pre_matches_post"] is False
    assert pre_post_compare_git(_git(extra=1), _git())["pre_matches_post"] is False
    assert pre_post_compare_git(_git(extra=1), _git(extra=2))["pre_matches_post"] is False


def test_hash_authority_only_asymmetry_keeps_pre_matches_post():
    # valid↔valid unequal expected authority, equal stripped projection
    r = pre_post_compare_hash(_hash(expected=SHA_A), _hash(expected=SHA_B), expected=SHA_C)
    assert r["pre_matches_post"] is True
    assert r["expected_conflict"] is True
    assert r["post_matches_expected"] is None
    # valid↔malformed
    r2 = pre_post_compare_hash(_hash(expected=SHA_A), _hash(expected="BAD"), expected=None)
    assert r2["pre_matches_post"] is True
    assert r2["post_matches_expected"] is None
    # missing↔present
    r3 = pre_post_compare_hash(_hash(), _hash(expected=SHA_B), expected=None)
    assert r3["pre_matches_post"] is True


def test_git_authority_only_asymmetry_keeps_pre_matches_post():
    pre = _git(expected_head=REV_C, expected_tree=REV_D)
    post = _git(expected_head=REV_A, expected_tree=REV_B)
    r = pre_post_compare_git(pre, post, expected_head=REV_A, expected_tree=REV_B)
    assert r["pre_matches_post"] is True
    r2 = pre_post_compare_git(_git(expected_head=REV_C), _git(expected_head="BAD"), expected_head=None, expected_tree=REV_B)
    assert r2["pre_matches_post"] is True
    assert r2["post_matches_expected"] is None


def test_hash_retains_expected_head_tree_keys():
    r = pre_post_compare_hash(_hash(expected_head="x"), _hash(expected_head="y"))
    assert "expected_head" in r["pre_observed_projection"]
    assert r["pre_matches_post"] is False


def test_multi_malformation_sorted_unique_list():
    r = pre_post_compare_hash(_hash(sha="BAD"), _hash(sha="WORSE", expected="ALSO"), expected="NOPE")
    assert r["malformed_fields"] == sorted(set(r["malformed_fields"]))
    assert r["malformed_fields"] == ["expected", "post.expected", "post.sha256", "pre.sha256"]
    r2 = pre_post_compare_git(
        _git(head="BAD", upstream="WORSE"),
        _git(tree="ALSO"),
        expected_head="NOPE",
        expected_tree=REV_B,
    )
    assert r2["malformed_fields"] == sorted(set(r2["malformed_fields"]))
    assert "pre.head" in r2["malformed_fields"]
    assert "pre.upstream" in r2["malformed_fields"]
    assert "post.tree" in r2["malformed_fields"]
    assert "expected_head" in r2["malformed_fields"]


def test_git_head_only_and_tree_only_drift_post_matches_expected_false():
    base_pre = _git(head=REV_C, upstream=REV_A, tree=REV_B)
    # head-only drift vs expected_head=REV_A
    post_head = _git(head=REV_C, upstream=REV_A, tree=REV_B)
    r = pre_post_compare_git(base_pre, post_head, expected_head=REV_A, expected_tree=REV_B)
    assert r["pre_matches_post"] is True
    assert r["post_matches_expected"] is False
    # tree-only drift
    pre_tree = _git(head=REV_A, upstream=REV_A, tree=REV_C)
    post_tree = _git(head=REV_A, upstream=REV_A, tree=REV_C)
    r2 = pre_post_compare_git(pre_tree, post_tree, expected_head=REV_A, expected_tree=REV_B)
    assert r2["pre_matches_post"] is True
    assert r2["post_matches_expected"] is False
    # upstream-only retained
    pre_u = _git(head=REV_A, upstream=REV_C, tree=REV_B)
    post_u = _git(head=REV_A, upstream=REV_C, tree=REV_B)
    r3 = pre_post_compare_git(pre_u, post_u, expected_head=REV_A, expected_tree=REV_B)
    assert r3["post_matches_expected"] is False


def test_hash_partition_malformed_pre_keeps_post_match():
    r = pre_post_compare_hash(_hash(sha="BAD"), _hash(), expected=SHA_A)
    assert r["pre_matches_post"] is None
    assert r["post_matches_expected"] is True


def test_hash_partition_malformed_expected_keeps_pre_match():
    r = pre_post_compare_hash(_hash(), _hash(expected="BAD"), expected=None)
    assert r["pre_matches_post"] is True
    assert r["post_matches_expected"] is None


def test_git_partition_malformed_pre_keeps_post_match():
    r = pre_post_compare_git(_git(head="BAD"), _git(), expected_head=REV_A, expected_tree=REV_B)
    assert r["pre_matches_post"] is None
    assert r["post_matches_expected"] is True


def test_git_partition_malformed_expected_keeps_pre_match():
    r = pre_post_compare_git(_git(), _git(expected_head="BAD"), expected_head=None, expected_tree=REV_B)
    assert r["pre_matches_post"] is True
    assert r["post_matches_expected"] is None
