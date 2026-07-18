"""Pure receipt pre/post compare helpers (hash + git). Stdlib/typing only."""
from __future__ import annotations

import re
from typing import Any, Mapping

_SHA = re.compile(r"^[0-9a-f]{64}$")
_REV = re.compile(r"^[0-9a-f]{40}$")
_A, _V, _M = "absent", "valid", "malformed"


def _state(value: Any, kind: str) -> str:
    if value is None:
        return _A
    if not isinstance(value, str):
        return _M
    return _V if (_SHA if kind == "sha" else _REV).fullmatch(value) else _M


def _map(obj: Any) -> Mapping[str, Any] | None:
    return obj if isinstance(obj, Mapping) else None


def _get(m: Mapping[str, Any] | None, key: str) -> Any:
    return None if m is None or key not in m else m[key]


def _resolve(explicit: Any, metadata: Any, kind: str, exp_path: str, meta_path: str) -> tuple[Any, bool, list[str]]:
    es, ms = _state(explicit, kind), _state(metadata, kind)
    mal: list[str] = []
    if es == _M:
        mal.append(exp_path)
    if ms == _M:
        mal.append(meta_path)
    if mal:
        return None, False, mal
    if es == _V and ms == _V:
        return (explicit, False, []) if explicit == metadata else (None, True, [])
    if es == _V:
        return explicit, False, []
    if ms == _V:
        return metadata, False, []
    return None, False, []


def _proj(side: Any, drop: frozenset[str]) -> dict[str, Any] | None:
    m = _map(side)
    if m is None:
        return None
    out = dict(m)
    for k in drop:
        out.pop(k, None)
    return out


def _obs_ok(m: Mapping[str, Any] | None, keys: tuple[str, ...], kind: str) -> bool:
    return m is not None and all(_state(_get(m, k), kind) == _V for k in keys)


def pre_post_compare(*_a: Any, **_k: Any) -> None:
    raise TypeError("ambiguous_pre_post_compare_removed: use pre_post_compare_hash or pre_post_compare_git")


def pre_post_compare_hash(pre: Any, post: Any, expected: Any = None) -> dict[str, Any]:
    pre_m, post_m = _map(pre), _map(post)
    mal: list[str] = []
    if pre is not None and pre_m is None:
        mal.append("pre")
    if post is not None and post_m is None:
        mal.append("post")
    for side, label in ((pre_m, "pre"), (post_m, "post")):
        if side is not None and _state(_get(side, "sha256"), "sha") == _M:
            mal.append(f"{label}.sha256")
    meta = _get(post_m, "expected")
    effective, conflict, mal_e = _resolve(expected, meta, "sha", "expected", "post.expected")
    mal.extend(mal_e)
    pre_p, post_p = _proj(pre, frozenset({"expected"})), _proj(post, frozenset({"expected"}))
    pre_ok, post_ok = _obs_ok(pre_m, ("sha256",), "sha"), _obs_ok(post_m, ("sha256",), "sha")
    pre_matches = None if not (pre_ok and post_ok) else (pre_p == post_p)
    post_sha = _get(post_m, "sha256") if post_ok else None
    post_matches = None if (effective is None or conflict or not post_ok) else (post_sha == effective)
    return {
        "pre": pre, "post": post, "expected": expected, "expected_explicit": expected,
        "expected_metadata": meta, "expected_effective": effective, "expected_conflict": conflict,
        "post_sha256": post_sha, "pre_observed_projection": pre_p, "post_observed_projection": post_p,
        "pre_matches_post": pre_matches, "post_matches_expected": post_matches,
        "malformed_fields": sorted(set(mal)), "compare_kind": "hash_sha256",
    }


def pre_post_compare_git(
    pre: Any, post: Any, expected_head: Any = None, expected_tree: Any = None
) -> dict[str, Any]:
    pre_m, post_m = _map(pre), _map(post)
    mal: list[str] = []
    if pre is not None and pre_m is None:
        mal.append("pre")
    if post is not None and post_m is None:
        mal.append("post")
    for side, label in ((pre_m, "pre"), (post_m, "post")):
        if side is None:
            continue
        for fld in ("head", "upstream", "tree"):
            if _state(_get(side, fld), "rev") == _M:
                mal.append(f"{label}.{fld}")
    mh, mt = _get(post_m, "expected_head"), _get(post_m, "expected_tree")
    eh, ch, mal_h = _resolve(expected_head, mh, "rev", "expected_head", "post.expected_head")
    et, ct, mal_t = _resolve(expected_tree, mt, "rev", "expected_tree", "post.expected_tree")
    mal.extend(mal_h)
    mal.extend(mal_t)
    conflict = bool(ch or ct)
    drop = frozenset({"expected_head", "expected_tree"})
    pre_p, post_p = _proj(pre, drop), _proj(post, drop)
    keys = ("head", "upstream", "tree")
    pre_ok, post_ok = _obs_ok(pre_m, keys, "rev"), _obs_ok(post_m, keys, "rev")
    pre_matches = None if not (pre_ok and post_ok) else (pre_p == post_p)
    if eh is None or et is None or conflict or not post_ok:
        post_matches = None
    else:
        post_matches = (
            _get(post_m, "head") == eh
            and _get(post_m, "upstream") == eh
            and _get(post_m, "tree") == et
        )
    return {
        "pre": pre, "post": post, "expected_head": expected_head, "expected_tree": expected_tree,
        "expected_head_explicit": expected_head, "expected_tree_explicit": expected_tree,
        "expected_head_metadata": mh, "expected_tree_metadata": mt,
        "expected_head_effective": eh, "expected_tree_effective": et,
        "expected_conflict": conflict, "expected_conflict_head": ch, "expected_conflict_tree": ct,
        "post_head": _get(post_m, "head") if post_ok else None,
        "post_upstream": _get(post_m, "upstream") if post_ok else None,
        "post_tree": _get(post_m, "tree") if post_ok else None,
        "pre_observed_projection": pre_p, "post_observed_projection": post_p,
        "pre_matches_post": pre_matches, "post_matches_expected": post_matches,
        "malformed_fields": sorted(set(mal)), "compare_kind": "git_head_upstream_tree",
    }


__all__ = ["pre_post_compare", "pre_post_compare_hash", "pre_post_compare_git"]
