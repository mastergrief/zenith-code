"""O3 pure tool-module-set pin tests (Phase B). Dedicated carrier — not the 120-suite god file."""
from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

# Reuse packet/man helpers from the parent suite (no duplication).
from calm.llm_computer.tests import test_hrm_text_158_lands_ab_science_source_manifest as S

REPO = S.REPO
GEN = S.GEN
DRY = S.DRY
OWNER = REPO / "scripts" / "lands_ab_dry_exec_tool_module_set.py"
HEAD_A = (
    REPO
    / "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_LANDS_AB_science_source_manifest_HEAD_A.json"
)
OWNER_REL = "scripts/lands_ab_dry_exec_tool_module_set.py"


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _force_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)
    s_src, s_dst = src.stat(), dst.stat()
    assert (s_src.st_dev, s_src.st_ino) != (s_dst.st_dev, s_dst.st_ino), "alias"


def test_O3_TOOL_MODULE_SET_GENERATOR_UNION_AND_SINGLE_ENUM():
    owner = _load_mod(OWNER, "o3_owner")
    tool_set = set(owner.DRY_EXEC_TOOL_MODULE_SET)
    gen = _load_mod(GEN, "o3_gen")
    force = set(gen.generator_force_include_result(owner_path=OWNER))
    assert tool_set <= force
    assert tool_set == (force & tool_set)

    assert "scripts/lands_ab_packet_dry_exec.py" not in gen.MANDATORY_ALWAYS_BASE
    assert OWNER_REL not in gen.MANDATORY_ALWAYS_BASE
    assert "scripts/lands_ab_packet_dry_exec.py" in force
    assert OWNER_REL in force

    base_src = ast.parse(GEN.read_text(encoding="utf-8"))
    base_assign = None
    for node in base_src.body:
        if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "MANDATORY_ALWAYS_BASE":
            base_assign = node
            break
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "MANDATORY_ALWAYS_BASE":
                    base_assign = node
    assert base_assign is not None
    assert "lands_ab_packet_dry_exec.py" not in ast.dump(base_assign)

    dry = _load_mod(DRY, "o3_dry")
    assert set(owner.DRY_EXEC_TOOL_MODULE_SET) <= set(dry.MANDATORY_EXECUTION_SOURCE_SET)
    assert dry.DRY_EXEC_TOOL_ENTRYPOINT == "scripts/lands_ab_packet_dry_exec.py"
    tree = ast.parse(DRY.read_text(encoding="utf-8"))
    for node in tree.body:
        val = None
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if getattr(t, "id", None) == "_MANDATORY_EXECUTION_SOURCE_SET_BASE":
                    val = node
        elif isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "_MANDATORY_EXECUTION_SOURCE_SET_BASE":
            val = node
        if val is not None:
            dump = ast.dump(val)
            assert "lands_ab_packet_dry_exec.py" not in dump
            assert "lands_ab_dry_exec_tool_module_set.py" not in dump


def test_O3_TOOL_MODULE_SET_MANDATORY_AND_MAN_PARITY(tmp_path):
    owner = _load_mod(OWNER, "o3_owner")
    tool_set = set(owner.DRY_EXEC_TOOL_MODULE_SET)
    dry = _load_mod(DRY, "o3_dry")
    mandatory = set(dry.MANDATORY_EXECUTION_SOURCE_SET)
    assert tool_set <= mandatory
    assert tool_set == (mandatory & tool_set)

    o = tmp_path / "man.json"
    r = S._run_gen(o)
    assert r.returncode == 0, r.stderr
    man_paths = {e["path"] for e in S._load_man(o)["entries"]}
    assert tool_set <= man_paths
    assert tool_set == (man_paths & tool_set)

    assert HEAD_A.is_file()
    ha = json.loads(HEAD_A.read_text(encoding="utf-8"))
    ha_paths = {e["path"] for e in ha["entries"]}
    assert tool_set <= ha_paths, "HEAD_A must be regenerated before parity gate"
    assert tool_set == (ha_paths & tool_set)


def _owner_load_fail_closed_variants(tmp_path: Path) -> None:
    """A2.1 cure: owner-load fail-closed on FINAL tool bytes (N′ stays 124; in-node).

    Per-variant FRESH isolated tree. Owner is force-copied from LIVE then mutated
    (CURE 2). Sentinel is a NONEXISTENT packet path whose marker would appear only
    if packet load began after successful owner import (CURE 1).
    """
    poison_marker = "O3_POISON_NONEXISTENT_PACKET_MARKER"
    packet_load_err = "error: packet load:"
    live_pre = {
        OWNER_REL: _sha(OWNER),
        "scripts/lands_ab_packet_dry_exec.py": _sha(DRY),
    }
    variants = (
        ("absent", "absent"),
        ("malformed", "malformed"),
        ("missing_export", "missing_export"),
    )
    terminals: dict[str, str] = {}
    try:
        for name, kind in variants:
            root = tmp_path / f"iso_owner_load_{name}"
            if root.exists():
                shutil.rmtree(root)
            root.mkdir(parents=True)
            scripts = root / "scripts"
            scripts.mkdir(parents=True)
            # FRESH force-copy of FINAL tool bytes per variant.
            _force_copy(DRY, scripts / "lands_ab_packet_dry_exec.py")
            s_tool_live, s_tool_iso = DRY.stat(), (scripts / "lands_ab_packet_dry_exec.py").stat()
            assert (s_tool_live.st_dev, s_tool_live.st_ino) != (
                s_tool_iso.st_dev,
                s_tool_iso.st_ino,
            )

            # CURE 2: force-copy LIVE owner first + non-alias, THEN mutate per variant.
            owner_dst = scripts / "lands_ab_dry_exec_tool_module_set.py"
            _force_copy(OWNER, owner_dst)
            s_own_live, s_own_iso = OWNER.stat(), owner_dst.stat()
            assert (s_own_live.st_dev, s_own_live.st_ino) != (
                s_own_iso.st_dev,
                s_own_iso.st_ino,
            )
            if kind == "absent":
                owner_dst.unlink()
                assert not owner_dst.exists()
            elif kind == "malformed":
                owner_dst.write_text("def broken(:\n  this is not valid python\n", encoding="utf-8")
            elif kind == "missing_export":
                owner_dst.write_text(
                    '"""owner missing required export"""\n'
                    'DRY_EXEC_TOOL_ENTRYPOINT = "scripts/lands_ab_packet_dry_exec.py"\n',
                    encoding="utf-8",
                )
            else:
                raise AssertionError(kind)

            # CURE 1: nonexistent packet path — packet-load error text + marker must
            # be ABSENT (owner-load fails before first packet read).
            pkt_path = root / f"O3_POISON_NONEXISTENT_PACKET_{poison_marker}.json"
            assert not pkt_path.exists()
            rr = subprocess.run(
                [
                    sys.executable,
                    str(scripts / "lands_ab_packet_dry_exec.py"),
                    "--packet",
                    str(pkt_path),
                    "--verify-source-manifest",
                    "artifacts/acc_entropy/x.json",
                    "--expected-source-commit",
                    "a" * 40,
                    "--repo-root",
                    str(root),
                ],
                cwd=str(root),
                capture_output=True,
                text=True,
            )
            combined = (rr.stdout or "") + (rr.stderr or "")
            assert rr.returncode != 0, f"{name}: expected fail-closed, got rc0: {combined}"
            assert packet_load_err not in combined, (
                f"{name}: packet load began ({packet_load_err!r} present): {combined[:500]}"
            )
            assert poison_marker not in combined, (
                f"{name}: packet load began (marker present): {combined[:500]}"
            )
            if kind == "absent":
                assert (
                    "FileNotFoundError" in combined
                    or "No such file" in combined
                    or "lands_ab_dry_exec_tool_module_set" in combined
                ), combined[:500]
            elif kind == "malformed":
                assert "SyntaxError" in combined or "invalid syntax" in combined, combined[:500]
            elif kind == "missing_export":
                assert "AttributeError" in combined or "DRY_EXEC_TOOL_MODULE_SET" in combined, combined[
                    :500
                ]
            terminals[name] = f"rc={rr.returncode};class_snip={combined[:160]!r}"
        assert len(set(terminals.values())) == len(variants), terminals
    finally:
        live_post = {
            OWNER_REL: _sha(OWNER),
            "scripts/lands_ab_packet_dry_exec.py": _sha(DRY),
        }
        assert live_post == live_pre, "live tracked bytes mutated during owner-load hostiles"


def test_O3_UNDER_INCLUSION_OWNER_MISSING_FROM_MAN_FAIL(tmp_path):
    commit = "d4" * 20
    o = tmp_path / "m.json"
    r = S._run_gen(o)
    assert r.returncode == 0, r.stderr
    man = S._load_man(o)
    new_entries = [e for e in man["entries"] if e["path"] != OWNER_REL]
    assert len(new_entries) == len(man["entries"]) - 1, "owner must be in generated man"
    man["entries"] = sorted(new_entries, key=lambda e: e["path"])
    man["n_entries"] = len(man["entries"])
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_o3_underinc_{tmp_path.name}.json"
    try:
        rel.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
        pkt = S._packet_for_dry(tmp_path, rel, S._sha(rel), commit)
        rr = S._run_dry(pkt, rel, commit)
        assert rr.returncode != 0
        assert "missing mandatory source" in rr.stderr
        assert "lands_ab_dry_exec_tool_module_set" in rr.stderr or OWNER_REL in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()
    # A2.1 in-node owner-load fail-closed (same node → N′ stays 124).
    _owner_load_fail_closed_variants(tmp_path / "owner_load")


def test_O3_SIBLING_DRIFT_OWNER_DISK_SHA_MISMATCH_FAIL(tmp_path):
    """Mutate only owner bytes in isolated ws → disk sha mismatch for owner rel."""
    live_owner_sha = _sha(OWNER)
    commit = "d5" * 20

    o = tmp_path / "m_live.json"
    r = S._run_gen(o)
    assert r.returncode == 0, r.stderr
    man = S._load_man(o)
    owner_ent = next(e for e in man["entries"] if e["path"] == OWNER_REL)
    assert owner_ent["sha256"] == live_owner_sha

    root = tmp_path / "iso_repo"
    paths = [e["path"] for e in man["entries"]]
    for extra in (
        "scripts/lands_ab_packet_dry_exec.py",
        "scripts/lands_ab_science_source_manifest.py",
        OWNER_REL,
    ):
        if extra not in paths:
            paths.append(extra)
    for rel in paths:
        src = REPO / rel
        if src.is_file():
            _force_copy(src, root / rel)

    live_pre = {
        OWNER_REL: _sha(REPO / OWNER_REL),
        "scripts/lands_ab_packet_dry_exec.py": _sha(DRY),
    }
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_o3_sibman_{tmp_path.name}.json"
    try:
        iso_owner = root / OWNER_REL
        s_live, s_iso = (REPO / OWNER_REL).stat(), iso_owner.stat()
        assert (s_live.st_dev, s_live.st_ino) != (s_iso.st_dev, s_iso.st_ino)
        iso_owner.write_text(iso_owner.read_text(encoding="utf-8") + "\n# o3 sibling drift probe\n")
        assert _sha(iso_owner) != live_owner_sha

        rel.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
        pkt_path = S._packet_for_dry(tmp_path, rel, _sha(rel), commit)

        iso_man = root / "artifacts/acc_entropy" / f"_tmp_o3_sibman_{tmp_path.name}.json"
        iso_man.parent.mkdir(parents=True, exist_ok=True)
        iso_man.write_bytes(rel.read_bytes())
        j = json.loads(pkt_path.read_text())
        j["science_source_manifest_path"] = iso_man.relative_to(root).as_posix()
        j["science_source_manifest_sha256"] = _sha(iso_man)
        j["dry_exec_tool_sha256"] = _sha(root / "scripts/lands_ab_packet_dry_exec.py")
        j["generator_script_sha256"] = _sha(root / "scripts/lands_ab_science_source_manifest.py")
        # dry loads tool module set at import from its own parent dir (iso scripts/)
        iso_pkt = root / "artifacts/acc_entropy" / f"_tmp_o3_sibpkt_{tmp_path.name}.json"
        iso_pkt.write_text(json.dumps(j) + "\n")

        rr = subprocess.run(
            [
                sys.executable,
                str(root / "scripts/lands_ab_packet_dry_exec.py"),
                "--packet",
                str(iso_pkt),
                "--verify-source-manifest",
                iso_man.relative_to(root).as_posix(),
                "--expected-source-commit",
                commit,
                "--repo-root",
                str(root),
            ],
            cwd=str(root),
            capture_output=True,
            text=True,
        )
        assert rr.returncode != 0, rr.stdout + rr.stderr
        assert "disk sha mismatch" in rr.stderr
        assert OWNER_REL in rr.stderr or "lands_ab_dry_exec_tool_module_set" in rr.stderr
    finally:
        live_post = {
            OWNER_REL: _sha(REPO / OWNER_REL),
            "scripts/lands_ab_packet_dry_exec.py": _sha(DRY),
        }
        assert live_post == live_pre, "live tracked bytes mutated"
        if rel.exists():
            rel.unlink()


# --- HEAD_B dual-assert + structural validator (HEAD_B source-authority landing) ---
HEAD_B = (
    REPO
    / "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_LANDS_AB_science_source_manifest_HEAD_B.json"
)
HEAD_A_SHA256 = (
    "7d111d527452820ad05431b34dc6ef5742361196ef188916e0e7d3c58fa43560"
)
HEAD_B_TOP_KEYS = frozenset({"schema", "repo_root_note", "entries", "n_entries"})
HEAD_B_ENTRY_KEYS = frozenset({"path", "sha256"})


def validate_head_b_manifest(man: dict, *, repo: Path, head_ref: str = "HEAD") -> None:
    """Fail-closed structural + committed-blob equality. Raises AssertionError on any defect."""
    assert set(man.keys()) == HEAD_B_TOP_KEYS, f"top_keys={set(man.keys())}"
    assert man["schema"] == "LANDS_AB_science_source_manifest/v1"
    assert isinstance(man["repo_root_note"], str) and man["repo_root_note"]
    assert isinstance(man["entries"], list)
    assert isinstance(man["n_entries"], int)
    assert man["n_entries"] == len(man["entries"]), (
        f"n_entries={man['n_entries']} len_entries={len(man['entries'])}"
    )
    paths: list[str] = []
    for i, e in enumerate(man["entries"]):
        assert isinstance(e, dict), f"entry[{i}] not dict"
        assert set(e.keys()) == HEAD_B_ENTRY_KEYS, f"entry[{i}] keys={set(e.keys())}"
        path, digest = e["path"], e["sha256"]
        assert isinstance(path, str) and path
        assert not path.startswith("/") and not (len(path) >= 2 and path[1] == ":")
        assert "\\" not in path and "\x00" not in path
        assert ".." not in Path(path).parts
        assert Path(path).as_posix() == path
        assert isinstance(digest, str) and len(digest) == 64
        assert all(c in "0123456789abcdef" for c in digest), "sha256 must be 64-lowerhex"
        paths.append(path)
    assert paths == sorted(paths), "paths not sorted"
    assert len(paths) == len(set(paths)), "duplicate paths"
    for e in man["entries"]:
        blob = subprocess.check_output(
            ["git", "-C", str(repo), "show", f"{head_ref}:{e['path']}"]
        )
        got = hashlib.sha256(blob).hexdigest()
        assert got == e["sha256"], f"SHA_DRIFT path={e['path']} pin={e['sha256']} head={got}"


def test_O3_HEAD_A_HISTORICAL_IMMUTABLE():
    assert HEAD_A.is_file()
    assert _sha(HEAD_A) == HEAD_A_SHA256
    owner = _load_mod(OWNER, "o3_owner_ha")
    tool_set = set(owner.DRY_EXEC_TOOL_MODULE_SET)
    man = json.loads(HEAD_A.read_text(encoding="utf-8"))
    paths = {e["path"] for e in man["entries"]}
    assert tool_set <= paths


def test_O3_HEAD_B_LIVE_COMMITTED_SOURCE_AUTHORITY():
    assert HEAD_B.is_file(), "HEAD_B must exist for live authority test"
    man = json.loads(HEAD_B.read_text(encoding="utf-8"))
    validate_head_b_manifest(man, repo=REPO)
    owner = _load_mod(OWNER, "o3_owner_hb")
    tool_set = set(owner.DRY_EXEC_TOOL_MODULE_SET)
    paths = {e["path"] for e in man["entries"]}
    assert tool_set <= paths


def test_O3_HEAD_B_ENTRY_SHA_DRIFT_FAILS(tmp_path: Path):
    man = json.loads(HEAD_B.read_text(encoding="utf-8"))
    man = json.loads(json.dumps(man))  # deep copy via json
    man["entries"][0]["sha256"] = "0" * 64
    try:
        validate_head_b_manifest(man, repo=REPO)
        raised = False
    except AssertionError as e:
        raised = True
        assert "SHA_DRIFT" in str(e)
    assert raised


def test_O3_HEAD_B_N_ENTRIES_MISMATCH_FAILS():
    man = json.loads(json.dumps(json.loads(HEAD_B.read_text(encoding="utf-8"))))
    man["n_entries"] = len(man["entries"]) + 1
    try:
        validate_head_b_manifest(man, repo=REPO)
        raised = False
    except AssertionError:
        raised = True
    assert raised


def test_O3_HEAD_B_DUPLICATE_PATH_FAILS():
    man = json.loads(json.dumps(json.loads(HEAD_B.read_text(encoding="utf-8"))))
    man["entries"].append(dict(man["entries"][0]))
    man["n_entries"] = len(man["entries"])
    try:
        validate_head_b_manifest(man, repo=REPO)
        raised = False
    except AssertionError as e:
        raised = True
        assert "duplicate" in str(e).lower() or "paths" in str(e).lower()
    assert raised
