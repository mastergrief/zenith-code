"""CPU-static characterization of A1 Tier-2 lifted helpers.

Supplements — never replaces — the 120-suite / corpus. Named behaviors from
plan v3 §4 (proposed 11; collected count frozen at claude gate-1).
"""
from __future__ import annotations

import hashlib
import inspect
import sys
from pathlib import Path
from unittest import mock

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import lands_ab_packet_dry_exec as T  # noqa: E402


# ---------------------------------------------------------------- _add_ref


def test_add_ref_mutates_caller_owned_set_alias() -> None:
    owned: set = set()
    T._add_ref(owned, "Path/Foo")
    assert owned is not set()  # still the same object identity after mutation
    # alias identity: function must mutate the passed set, not return a copy
    marker = owned
    T._add_ref(owned, "other")
    assert marker is owned
    assert "other" in owned and "other".lower() in owned


def test_add_ref_adds_strip_and_lower_pair() -> None:
    s: set = set()
    T._add_ref(s, "  AbC  ")
    assert "AbC" in s
    assert "abc" in s
    assert len(s) == 2


def test_add_ref_ignores_non_str() -> None:
    s: set = {"keep"}
    T._add_ref(s, None)
    T._add_ref(s, 12)
    T._add_ref(s, b"bytes")
    T._add_ref(s, ["x"])
    assert s == {"keep"}


def test_add_ref_ignores_whitespace_only() -> None:
    s: set = set()
    T._add_ref(s, "")
    T._add_ref(s, "   ")
    T._add_ref(s, "\t\n")
    assert s == set()


def test_add_ref_idempotent_on_repeat() -> None:
    s: set = set()
    T._add_ref(s, "same")
    snap = set(s)
    T._add_ref(s, "same")
    T._add_ref(s, "same")
    assert s == snap


# ---------------------------------------------------------------- _disk_sha_ok


def test_disk_sha_ok_containment_before_disk_read(tmp_path: Path) -> None:
    """Q5: _canonical_repo_relpath runs before any disk/hash access."""
    order: list[str] = []
    real_canon = T._canonical_repo_relpath

    def wrap_canon(rel, *, repo):
        order.append("canon")
        return real_canon(rel, repo=repo)

    def boom_sha(*_a, **_k):
        order.append("sha")
        raise AssertionError("sha256_file must not run before containment fails")

    repo = tmp_path
    with mock.patch.object(T, "_canonical_repo_relpath", side_effect=wrap_canon), mock.patch.object(
        T, "sha256_file", side_effect=boom_sha
    ):
        # traversal: non-canonical → False; sha never consulted
        assert T._disk_sha_ok(repo, "artifacts/../../../etc/passwd", "deadbeef") is False
    assert order == ["canon"]
    assert "sha" not in order


def test_disk_sha_ok_false_on_non_canonical(tmp_path: Path) -> None:
    repo = tmp_path
    assert T._disk_sha_ok(repo, "../outside.json", "a" * 64) is False
    assert T._disk_sha_ok(repo, "artifacts/../../secret", "a" * 64) is False


def test_disk_sha_ok_false_on_missing_file(tmp_path: Path) -> None:
    repo = tmp_path
    (repo / "artifacts").mkdir()
    assert T._disk_sha_ok(repo, "artifacts/absent.json", "a" * 64) is False


def test_disk_sha_ok_false_on_hash_mismatch(tmp_path: Path) -> None:
    repo = tmp_path
    p = repo / "artifacts" / "x.json"
    p.parent.mkdir(parents=True)
    p.write_text('{"ok": true}\n')
    assert T._disk_sha_ok(repo, "artifacts/x.json", "0" * 64) is False


def test_disk_sha_ok_true_on_match(tmp_path: Path) -> None:
    repo = tmp_path
    p = repo / "artifacts" / "x.json"
    p.parent.mkdir(parents=True)
    body = b'{"ok": true}\n'
    p.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    assert T._disk_sha_ok(repo, "artifacts/x.json", digest) is True
    assert T._disk_sha_ok(repo, "artifacts/x.json", digest.upper()) is True


# ---------------------------------------------------------------- _check_path_sha


def test_check_path_sha_silent_success() -> None:
    man_map = {"scripts/foo.py": "abc"}
    man_path_set = {"scripts/foo.py"}
    # no raise
    T._check_path_sha(man_map, man_path_set, "scripts/foo.py", "ABC", where="pins.x")


def test_check_path_sha_path_absent_exact_message() -> None:
    man_map: dict = {}
    man_path_set: set = set()
    with pytest.raises(ValueError) as ei:
        T._check_path_sha(man_map, man_path_set, "scripts/missing.py", "a" * 64, where="pins.x")
    assert str(ei.value) == "I2 pin path absent from manifest: scripts/missing.py (pins.x)"


def test_check_path_sha_sha_mismatch_exact_message() -> None:
    man_map = {"scripts/foo.py": "abcd"}
    man_path_set = {"scripts/foo.py"}
    with pytest.raises(ValueError) as ei:
        T._check_path_sha(man_map, man_path_set, "scripts/foo.py", "ffff", where="pins.runner")
    assert str(ei.value) == (
        "I2 pin sha != manifest for scripts/foo.py at pins.runner: pin=ffff manifest=abcd"
    )


def test_check_path_sha_where_is_keyword_only() -> None:
    sig = inspect.signature(T._check_path_sha)
    assert sig.parameters["where"].kind is inspect.Parameter.KEYWORD_ONLY
    man_map = {"scripts/foo.py": "a"}
    man_path_set = {"scripts/foo.py"}
    # positional where must fail
    with pytest.raises(TypeError):
        T._check_path_sha(man_map, man_path_set, "scripts/foo.py", "a", "pins.x")  # type: ignore[misc]
    # and where appears in both error messages (already covered above via exact text)
