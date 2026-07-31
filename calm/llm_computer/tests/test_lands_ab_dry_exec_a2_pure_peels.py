"""CPU-static characterization of A2 pure zero-emission peels (P1/P2/P3)."""
from __future__ import annotations

import argparse
import inspect
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "scripts"))
import lands_ab_packet_dry_exec as T  # noqa: E402


# ---------------------------------------------------------------- P1


def test_seed_authority_mutates_caller_owned_set() -> None:
    owned: set = set()
    plan = T.EXPECTED_OPERATIVE_PLAN_REL
    sha = T.EXPECTED_OPERATIVE_PLAN_SHA256
    src = "abc123deadbeef"
    auth = {"k1": plan, "k2": sha, "k3": src, "k4": "not-verifiable"}
    T._seed_known_refs_from_authority_chain(
        owned,
        auth,
        expected_operative_plan_path=plan,
        expected_operative_plan_sha256=sha,
        src=src,
    )
    assert plan in owned and plan.lower() in owned
    assert sha in owned  # already lower
    assert src in owned
    assert "not-verifiable" not in owned
    marker = owned
    T._seed_known_refs_from_authority_chain(
        owned,
        {"x": plan},
        expected_operative_plan_path=plan,
        expected_operative_plan_sha256=sha,
        src=src,
    )
    assert marker is owned


def test_seed_authority_ignores_non_dict_and_empty() -> None:
    s: set = {"keep"}
    T._seed_known_refs_from_authority_chain(
        s,
        None,
        expected_operative_plan_path="p",
        expected_operative_plan_sha256="s",
        src="x",
    )
    T._seed_known_refs_from_authority_chain(
        s,
        "not-a-dict",
        expected_operative_plan_path="p",
        expected_operative_plan_sha256="s",
        src="x",
    )
    T._seed_known_refs_from_authority_chain(
        s,
        {"a": "", "b": "   ", "c": 12},
        expected_operative_plan_path="p",
        expected_operative_plan_sha256="s",
        src="x",
    )
    assert s == {"keep"}


# ---------------------------------------------------------------- P2


def test_expand_pins_alias_and_packet_fields(tmp_path: Path) -> None:
    known: set = set()
    man_path_set = {"scripts/foo.py"}
    man_sha = {"scripts/foo.py": "abcd"}
    pins = {
        "runner_and_harness_shas": {"scripts/foo.py": "ABCD"},
        "other": {"path": "scripts/foo.py", "sha256": "abcd"},
    }
    packet = {
        "science_source_manifest_sha256": "1111",
        "generator_script_sha256": "2222",
        "dry_exec_tool_sha256": "3333",
        "source_commit_sha": "4444",
    }
    marker = known
    T._expand_known_refs_from_validated_pins(
        known,
        pins,
        exempt_pin_paths=set(),
        man_path_set=man_path_set,
        man_sha_by_path=man_sha,
        repo=tmp_path,
        packet=packet,
    )
    assert marker is known
    assert "scripts/foo.py" in known
    assert "abcd" in known
    for v in ("1111", "2222", "3333", "4444"):
        assert v in known


def test_expand_pins_artifacts_uses_disk_sha(tmp_path: Path) -> None:
    art = tmp_path / "artifacts"
    art.mkdir()
    f = art / "x.json"
    body = b'{"ok":1}\n'
    f.write_bytes(body)
    import hashlib

    digest = hashlib.sha256(body).hexdigest()
    known: set = set()
    pins = {"p": {"path": "artifacts/x.json", "sha256": digest}}
    T._expand_known_refs_from_validated_pins(
        known,
        pins,
        exempt_pin_paths=set(),
        man_path_set=set(),
        man_sha_by_path={},
        repo=tmp_path,
        packet={},
    )
    assert "artifacts/x.json" in known
    assert digest in known


def test_expand_pins_no_emissions_in_body() -> None:
    src = inspect.getsource(T._expand_known_refs_from_validated_pins)
    tree = __import__("ast").parse(src)
    fn = tree.body[0]
    for node in __import__("ast").walk(fn):
        if isinstance(node, __import__("ast").Raise):
            raise AssertionError("P2 must not raise")
        if (
            isinstance(node, __import__("ast").Call)
            and isinstance(node.func, __import__("ast").Name)
            and node.func.id == "print"
        ):
            raise AssertionError("P2 must not print")


# ---------------------------------------------------------------- P3


def test_build_parser_defaults_and_required() -> None:
    ap = T._build_dry_exec_arg_parser()
    assert isinstance(ap, argparse.ArgumentParser)
    # required flags present
    dests = {a.dest for a in ap._actions if getattr(a, "option_strings", None)}
    assert "packet" in dests
    assert "verify_source_manifest" in dests
    assert "expected_source_commit" in dests
    # defaults
    ns = ap.parse_args(
        [
            "--packet",
            "p.json",
            "--verify-source-manifest",
            "m.json",
            "--expected-source-commit",
            "a" * 40,
        ]
    )
    assert ns.expected_operative_plan_path == T.EXPECTED_OPERATIVE_PLAN_REL
    assert ns.expected_operative_plan_sha256 == T.EXPECTED_OPERATIVE_PLAN_SHA256
    assert ns.repo_root == "."


def test_build_parser_fresh_each_call() -> None:
    a = T._build_dry_exec_arg_parser()
    b = T._build_dry_exec_arg_parser()
    assert a is not b


def test_no_nested_functiondefs_in_new_helpers() -> None:
    for name in (
        "_seed_known_refs_from_authority_chain",
        "_expand_known_refs_from_validated_pins",
        "_build_dry_exec_arg_parser",
    ):
        fn = getattr(T, name)
        tree = __import__("ast").parse(inspect.getsource(fn))
        outer = tree.body[0]
        nested = [
            n
            for n in __import__("ast").walk(outer)
            if isinstance(n, __import__("ast").FunctionDef) and n is not outer
        ]
        assert nested == [], f"{name} has nested FunctionDef"
