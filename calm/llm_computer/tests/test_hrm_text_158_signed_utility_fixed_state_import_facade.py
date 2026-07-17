"""Import-facade tests: hash/path bind + clean-revision session (PLAN v3 / d1_c4)."""
from __future__ import annotations
import hashlib, json, os, subprocess, sys, tarfile, textwrap, types
from pathlib import Path
import pytest
import calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_creditdir_import_facade as fac
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_creditdir_import_facade import (
    MODULE_IMPORT_NAMES, MODULE_REL_PATHS, ImportFacadeError,
    load_signed_utility_fixed_state_modules, signed_utility_fixed_state_session,
    verify_expected_sha256_by_module,
)
REPO, MOD = fac.REPO_ROOT, Path(fac.__file__)
FX = Path("/home/gabe/claw-code-creditdir/transient_fp_credit/fa_accounting_v2_post_seam_signed_utility_d1_r1_fixture_partition_v1.json")
EXPECTED = {k: hashlib.sha256((REPO / rel).read_bytes()).hexdigest() for k, rel in MODULE_REL_PATHS.items()}
_CALM = ("calm", "calm.hrm_text_158", "calm.hrm_text_158.native_full_stack")
def _seed_chain():
    old = {n: sys.modules[n] for n in list(sys.modules) if n in _CALM or n.startswith("_su_sent_")}
    seed, paths = {}, {}
    for n in ("_su_sent_before",) + _CALM + ("_su_sent_after",):
        m = types.ModuleType(n)
        if n in _CALM:
            paths[n] = [f"/preseed/{n}"]; m.__path__ = paths[n]  # type: ignore[attr-defined]
        sys.modules[n] = m; seed[n] = m
    order = list(sys.modules)
    def restore():
        for n in list(sys.modules):
            if n in _CALM or n.startswith("_su_sent_"): sys.modules.pop(n, None)
        sys.modules.update(old)
    return seed, paths, order, restore
def _assert_order(seed, paths, order_before):
    for n, m in seed.items():
        assert sys.modules.get(n) is m
        if n in paths: assert sys.modules[n].__path__ is paths[n]
    keyed = set(order_before)
    assert [k for k in list(sys.modules) if k in keyed] == [k for k in order_before if k in sys.modules]
def _lock_free():
    assert fac._ACTIVE is False and fac._LOCK.acquire(blocking=False); fac._LOCK.release()
def test_loc_budget():
    assert sum(1 for _ in MOD.open()) <= 360 and sum(1 for _ in Path(__file__).open()) <= 200


def test_eleven_module_import_closure_keys():
    assert set(MODULE_REL_PATHS) == {
        "reducers", "schema", "pin_validation", "phase_telemetry", "integrity_proofs",
        "partition_leakage", "arm_proofs", "legal_subset", "eval_contract",
        "authoritative_gpu", "driver", "facade",
    }
    assert list(fac._LOAD_ORDER) == [
        "reducers", "schema", "pin_validation", "phase_telemetry", "integrity_proofs",
        "partition_leakage", "arm_proofs", "legal_subset", "eval_contract",
        "authoritative_gpu", "driver", "facade",
    ]
    b = load_signed_utility_fixed_state_modules(EXPECTED)
    assert hasattr(b, "legal_subset") and hasattr(b, "partition_leakage") and hasattr(b, "arm_proofs")
    assert set(b.observed_sha256_by_module) == set(MODULE_REL_PATHS)
    with pytest.raises(ImportFacadeError, match="expected_keys_mismatch"):
        verify_expected_sha256_by_module({k: v for k, v in EXPECTED.items() if k != "integrity_proofs"})


def test_hash_and_path_bind(tmp_path: Path):
    import importlib, shutil
    b = load_signed_utility_fixed_state_modules(EXPECTED)
    assert hasattr(b.facade, "developer_check") and b.observed_sha256_by_module == EXPECTED
    bad = dict(EXPECTED); bad["reducers"] = "0" * 64
    with pytest.raises(ImportFacadeError, match="module_sha_mismatch:reducers"):
        verify_expected_sha256_by_module(bad)
    for rel in MODULE_REL_PATHS.values():
        d = tmp_path / rel; d.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(REPO / rel, d)
    exp = {k: hashlib.sha256((tmp_path / rel).read_bytes()).hexdigest() for k, rel in MODULE_REL_PATHS.items()}
    importlib.import_module(MODULE_IMPORT_NAMES["reducers"])
    b2 = load_signed_utility_fixed_state_modules(exp, repo_root=tmp_path)
    for key, rel in MODULE_REL_PATHS.items():
        assert Path(getattr(b2, key).__file__).resolve() == (tmp_path / rel).resolve()
@pytest.mark.parametrize("name,typ,match", [
    ("/abs/x.py", tarfile.REGTYPE, "tar_unsafe"), ("a/../b.py", tarfile.REGTYPE, "tar_unsafe"),
    ("link", tarfile.SYMTYPE, "tar_link_forbidden"), ("hlink", tarfile.LNKTYPE, "tar_link_forbidden"),
    ("fifo", tarfile.FIFOTYPE, "tar_special_forbidden"),
])
def test_tar_member_rejection(tmp_path: Path, name, typ, match):
    m = tarfile.TarInfo(name=name); m.type = typ
    if typ in (tarfile.SYMTYPE, tarfile.LNKTYPE): m.linkname = "t"
    with pytest.raises(ImportFacadeError, match=match):
        fac._validate_tar_member(m, tmp_path)
def test_session_order_restore_success_body_cleanup(monkeypatch):
    seed, paths, order, restore = _seed_chain()
    try:
        path_id, path_before = id(sys.path), list(sys.path)
        with signed_utility_fixed_state_session(EXPECTED) as bundle:
            assert id(sys.path) == path_id and str(bundle.snapshot_root) in sys.path
            assert bundle.facade.developer_check(json.loads(FX.read_text())).get("non_authoritative") is True
        assert list(sys.path) == path_before; _assert_order(seed, paths, order)
        with pytest.raises(RuntimeError, match="boom_inside"):
            with signed_utility_fixed_state_session(EXPECTED): raise RuntimeError("boom_inside")
        _assert_order(seed, paths, order)
        real = fac.tempfile.TemporaryDirectory
        class BoomTD(real):
            def cleanup(self): raise RuntimeError("cleanup_boom")
        monkeypatch.setattr(fac.tempfile, "TemporaryDirectory", BoomTD)
        with pytest.raises(ValueError, match="body_boom") as ei:
            with signed_utility_fixed_state_session(EXPECTED): raise ValueError("body_boom")
        assert not isinstance(ei.value, RuntimeError)
        assert any("cleanup_boom" in n for n in getattr(ei.value, "__notes__", []))
        _assert_order(seed, paths, order); _lock_free()
    finally: restore()
def test_pre_snapshot_fault_releases_lock(monkeypatch):
    real = fac._git_tree; n = {"c": 0}
    def boom(*a, **k):
        n["c"] += 1
        if n["c"] == 1: raise RuntimeError("snapshot_boom")
        return real(*a, **k)
    monkeypatch.setattr(fac, "_git_tree", boom)
    with pytest.raises(RuntimeError, match="snapshot_boom"):
        with signed_utility_fixed_state_session(EXPECTED): pass
    _lock_free()
    with signed_utility_fixed_state_session(EXPECTED) as b: assert b.facade is not None
def test_cleanup_only_propagates_and_follow_on(monkeypatch):
    real = fac.tempfile.TemporaryDirectory; n = {"c": 0}
    class BoomTD(real):
        def cleanup(self):
            n["c"] += 1
            if n["c"] == 1: raise RuntimeError("cleanup_boom")
            return super().cleanup()
    monkeypatch.setattr(fac.tempfile, "TemporaryDirectory", BoomTD)
    with pytest.raises(RuntimeError, match="cleanup_boom"):
        with signed_utility_fixed_state_session(EXPECTED): pass
    _lock_free()
    with signed_utility_fixed_state_session(EXPECTED) as b: assert b.snapshot_root.exists()
def test_setup_failure_restores_after_mutation(monkeypatch):
    seed, paths, order, restore = _seed_chain()
    try:
        path_before = list(sys.path)
        real_v, nv = fac.verify_expected_sha256_by_module, {"v": 0}
        def boom_v(*a, **k):
            nv["v"] += 1
            if nv["v"] == 1: raise RuntimeError("verify_boom")
            return real_v(*a, **k)
        monkeypatch.setattr(fac, "verify_expected_sha256_by_module", boom_v)
        with pytest.raises(RuntimeError, match="verify_boom"):
            with signed_utility_fixed_state_session(EXPECTED): pass
        assert list(sys.path) == path_before; _assert_order(seed, paths, order); _lock_free()
        real_l, hits, seen = fac._load_verified, {"n": 0}, {}
        def boom_l(name, path):
            hits["n"] += 1
            if hits["n"] == 2: raise RuntimeError("load_boom")
            return real_l(name, path)
        real_e = fac._extract_archive
        def wrap_e(repo, rev, dest):
            seen["snap"] = Path(dest).resolve(); return real_e(repo, rev, dest)
        monkeypatch.setattr(fac, "_extract_archive", wrap_e)
        monkeypatch.setattr(fac, "_load_verified", boom_l)
        with pytest.raises(RuntimeError, match="load_boom"):
            with signed_utility_fixed_state_session(EXPECTED): pass
        assert list(sys.path) == path_before; _assert_order(seed, paths, order)
        assert seen.get("snap") is not None and not seen["snap"].exists(); _lock_free()
        with signed_utility_fixed_state_session(EXPECTED) as b:
            assert b.facade is not None and b.snapshot_root.exists()
        assert not b.snapshot_root.exists()
    finally: restore()
def test_session_exposes_eight_modules():
    with signed_utility_fixed_state_session(EXPECTED) as bundle:
        assert bundle.phase_telemetry is not None
        assert bundle.integrity_proofs is not None
        assert bundle.authoritative_gpu is not None
        assert hasattr(bundle.authoritative_gpu, "run_authoritative_gpu_call_graph")
        assert set(bundle.observed_sha256_by_module) == set(MODULE_REL_PATHS)


def test_tree_pin_and_nested():
    before = list(sys.path)
    with pytest.raises(ImportFacadeError, match="tree_pin_mismatch"):
        with signed_utility_fixed_state_session(EXPECTED, expected_tree="0" * 40): pass
    assert sys.path == before
    with signed_utility_fixed_state_session(EXPECTED):
        with pytest.raises(ImportFacadeError, match="nested_session_forbidden|session_busy"):
            with signed_utility_fixed_state_session(EXPECTED): pass
def test_clean_subprocess_developer_check():
    script = (
        "import hashlib,json,os,sys,importlib.util\nfrom pathlib import Path\n"
        f"repo=Path({str(REPO)!r}).resolve()\n"
        "sys.path[:]=[p for p in sys.path if Path(p).resolve()!=repo and not str(Path(p).resolve()).startswith(str(repo)+os.sep)]\n"
        "os.environ.pop('PYTHONPATH',None)\n"
        f"mp=Path({str(MOD)!r}); spec=importlib.util.spec_from_file_location('su_if',mp)\n"
        "m=importlib.util.module_from_spec(spec); sys.modules['su_if']=m; spec.loader.exec_module(m)\n"
        "exp={k:hashlib.sha256((repo/rel).read_bytes()).hexdigest() for k,rel in m.MODULE_REL_PATHS.items()}\n"
        f"ff=json.loads(Path({str(FX)!r}).read_text())\n"
        "with m.signed_utility_fixed_state_session(exp) as b:\n"
        " p=b.facade.developer_check(ff); snap=b.snapshot_root.resolve(); d=[]\n"
        " for rel in m.DRIFTED_TRANSITIVE_PROOF_SET:\n"
        "  n='calm.hrm_text_158.native_full_stack.'+Path(rel).stem\n"
        "  if n in sys.modules:\n"
        "   fp=str(Path(sys.modules[n].__file__).resolve()); assert fp.startswith(str(snap)); d.append(fp)\n"
        " out={'non_authoritative':p.get('non_authoritative'),'snap':str(snap),'drifted':d}\n"
        "assert not Path(out['snap']).exists(); print(json.dumps(out))\n"
    )
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run([sys.executable, "-c", script], cwd="/tmp", env=env, capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    assert data["non_authoritative"] is True and data["drifted"]
