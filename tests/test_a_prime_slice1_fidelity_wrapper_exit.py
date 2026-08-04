"""F5–F10, F_marker, F_branch*, F_atomic, F_postpub — wrapper exit contract (R2d)."""
from __future__ import annotations

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
WRAPPER = REPO / "scripts/a_prime_slice1_retained_credit_fidelity_wrapper_v0.py"
REDUCER = REPO / "scripts/a_prime_slice1_retained_credit_fidelity_reducer_v0.py"
PROBE = REPO / "scripts/hrm_text_158_bounded_delta_acquisition_probe.py"


def _sha(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


_CACHED_ENV: dict | None = None


def _env(extra: dict | None = None) -> dict:
    global _CACHED_ENV
    if _CACHED_ENV is None:
        from scripts.a_prime_slice1_fidelity_pins import (
            collect_dirty, collect_git_head, compute_rollup,
        )
        rollup, n = compute_rollup(REPO)
        head = collect_git_head(REPO)
        dirty_sha, dirty_n = collect_dirty(REPO)
        e = os.environ.copy()
        e["PYTHONPATH"] = str(REPO)
        e["PYTHONDONTWRITEBYTECODE"] = "1"
        e["CUDA_VISIBLE_DEVICES"] = "0"
        e["HRM_TEXT_158_ALLOW_C2_GPU_LAUNCH"] = "1"
        e["A_PRIME_EXPECT_PROBE_SHA"] = _sha(PROBE)
        e["A_PRIME_EXPECT_REDUCER_SHA"] = _sha(REDUCER)
        e["A_PRIME_EXPECT_WRAPPER_SHA"] = _sha(WRAPPER)
        e["A_PRIME_EXPECT_ROLLUP_SHA"] = rollup
        e["A_PRIME_EXPECT_ROLLUP_N"] = str(n)
        e["A_PRIME_EXPECT_HEAD"] = head
        e["A_PRIME_EXPECT_DIRTY_SHA"] = dirty_sha
        e["A_PRIME_EXPECT_DIRTY_N"] = str(dirty_n)
        _CACHED_ENV = e
    out = dict(_CACHED_ENV)
    if extra:
        out.update(extra)
    return out


def _fresh_root(tag: str, *, mkdir: bool = False) -> Path:
    root = Path(f"/tmp/aprime_{tag}_{os.getpid()}")
    if root.exists():
        shutil.rmtree(root)
    if mkdir:
        root.mkdir(parents=True, exist_ok=True)
        (root / "command_status").mkdir()
    return root


def _write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj) + "\n")


def _preflight_argv(root: Path, env: dict, *, include_auth_shas: bool = True, dirty_n: str | None = None) -> list[str]:
    argv = [
        sys.executable, "-u", "-B", str(REDUCER),
        "--mode", "preflight", "--run-root", str(root), "--repo", str(REPO),
        "--expect-head", env["A_PRIME_EXPECT_HEAD"],
        "--expect-dirty-sha", env["A_PRIME_EXPECT_DIRTY_SHA"],
        "--expect-dirty-n", dirty_n if dirty_n is not None else env["A_PRIME_EXPECT_DIRTY_N"],
        "--expect-probe-sha", env["A_PRIME_EXPECT_PROBE_SHA"],
    ]
    if include_auth_shas:
        argv.extend([
            "--expect-reducer-sha", env["A_PRIME_EXPECT_REDUCER_SHA"],
            "--expect-wrapper-sha", env["A_PRIME_EXPECT_WRAPPER_SHA"],
        ])
    argv.extend([
        "--expect-rollup-sha", env["A_PRIME_EXPECT_ROLLUP_SHA"],
        "--expect-rollup-n", env["A_PRIME_EXPECT_ROLLUP_N"],
        "--synthetic",
    ])
    return argv


def _run_wrapper(run_root: Path, branch: str = "PAIRED_ACHIEVED_FIDELITY_AT_N", extra_args=None, env=None):
    argv = [
        sys.executable, "-u", "-B", str(WRAPPER),
        "--run-root", str(run_root), "--dry-synthetic-final", branch,
    ]
    if extra_args:
        argv.extend(extra_args)
    return subprocess.run(
        argv, cwd=str(REPO), env=env or _env(),
        capture_output=True, text=True, timeout=120,
    )


def _post_exit_check(run_root: Path):
    man_path = run_root / "terminal_manifest.json"
    if not man_path.is_file():
        return None
    man = json.loads(man_path.read_text())
    actual = (
        sorted(str(p.relative_to(run_root)) for p in (run_root / "command_status").glob("*.json"))
        if (run_root / "command_status").is_dir() else []
    )
    assert man["expected_status_set"] == actual
    for rel, exp in man["outputs"].items():
        p = run_root / rel
        assert p.is_file(), rel
        assert _sha(p) == exp, rel
    return man


def test_F5_success_post_exit_ess_hashes_marker():
    root = _fresh_root("f5")
    p = _run_wrapper(root, "PAIRED_ACHIEVED_FIDELITY_AT_N")
    assert p.returncode == 0, p.stdout + p.stderr
    assert "PHASE_MARKER RUN_ROOT=" in p.stdout
    assert "PACKET_TERMINAL PAIRED_ACHIEVED_FIDELITY_AT_N" in p.stdout
    assert p.stdout.index("PHASE_MARKER RUN_ROOT=") < p.stdout.index("PACKET_TERMINAL")
    man = _post_exit_check(root)
    assert man is not None
    assert "command_status/reduce.json" in man["expected_status_set"]
    assert man["branch"] == "PAIRED_ACHIEVED_FIDELITY_AT_N"
    assert man["terminal_authority"] == "manifest+marker"
    receipt = json.loads((root / "terminal_receipt.json").read_text())
    assert receipt["branch"] == man["branch"]
    assert p.stdout.count("PACKET_TERMINAL") == 1


def test_F6_pin_fail_instrument_gap_dual_key():
    """Clean pin-fail path: INSTRUMENT_GAP dual-key (manifest+marker), not absent escape."""
    root = _fresh_root("f6")
    env = _env()
    env["A_PRIME_EXPECT_REDUCER_SHA"] = "deadbeef" * 8
    p = _run_wrapper(root, "PAIRED_ACHIEVED_FIDELITY_AT_N", env=env)
    assert p.returncode == 2, p.stdout + p.stderr
    assert "PACKET_TERMINAL INSTRUMENT_GAP" in p.stdout
    man = _post_exit_check(root)
    assert man is not None
    assert man["branch"] == "INSTRUMENT_GAP"
    assert man["terminal_authority"] == "manifest+marker"


def test_F7_stale_root_instrument_gap_dual_key():
    """Stale root: dual-key INSTRUMENT_GAP; ess mechanically equals actual disk set."""
    root = Path(f"/tmp/aprime_f7_{os.getpid()}")
    root.mkdir(parents=True, exist_ok=True)
    (root / "junk.txt").write_text("stale\n")
    p = _run_wrapper(root, "PAIRED_ACHIEVED_FIDELITY_AT_N")
    assert p.returncode == 2, p.stdout + p.stderr
    assert "PACKET_TERMINAL INSTRUMENT_GAP" in p.stdout
    man = _post_exit_check(root)
    assert man is not None
    assert man["branch"] == "INSTRUMENT_GAP"
    actual = (
        sorted(str(x.relative_to(root)) for x in (root / "command_status").glob("*.json"))
        if (root / "command_status").is_dir() else []
    )
    assert man["expected_status_set"] == actual


def test_F8_liveness_fail_synthetic():
    root = _fresh_root("f8")
    p = _run_wrapper(root, "LIVENESS_FAIL")
    assert p.returncode == 3
    assert "PACKET_TERMINAL LIVENESS_FAIL" in p.stdout
    man = _post_exit_check(root)
    assert man["branch"] == "LIVENESS_FAIL"


def test_F9_run_root_phase_marker():
    root = _fresh_root("f9")
    p = _run_wrapper(root, "PAIRED_ACHIEVED_FIDELITY_AT_N")
    line = [ln for ln in p.stdout.splitlines() if ln.startswith("PHASE_MARKER RUN_ROOT=")][0]
    path = line.split("=", 1)[1]
    assert path.startswith("/")
    receipt = json.loads((root / "terminal_receipt.json").read_text())
    assert receipt["run_root"] == str(root.resolve())


def test_F10_no_post_publish_mismatch_or_temp():
    """Success path: post-rename inventory == post-exit; no diagnostic/temp."""
    root = _fresh_root("f10")
    snap_path = Path(f"/tmp/aprime_f10_snap_{os.getpid()}.json")
    if snap_path.exists():
        snap_path.unlink()
    env = _env({"A_PRIME_POSTPUB_SNAP_PATH": str(snap_path)})
    p = _run_wrapper(root, "PAIRED_ACHIEVED_FIDELITY_AT_N", env=env)
    assert p.returncode == 0
    assert (root / "terminal_manifest.json").is_file()
    assert not (root / "mismatch_diagnostic.json").is_file()
    assert not list(root.glob("terminal_manifest.json.tmp.*"))
    assert snap_path.is_file(), "success path must write post-rename snap when env set"
    post_rename = json.loads(snap_path.read_text())
    from scripts.a_prime_slice1_fidelity_manifest import snapshot_run_root
    assert post_rename == snapshot_run_root(root)


def test_F_marker_order_temporal():
    """Marker absent whenever verification fails (postpub inject); present after success verify."""
    root_ok = _fresh_root("fm_ok")
    p_ok = _run_wrapper(root_ok, "PAIRED_ACHIEVED_FIDELITY_AT_N")
    assert "PACKET_TERMINAL" in p_ok.stdout
    assert (root_ok / "terminal_manifest.json").is_file()
    root_q = _fresh_root("fm_q")
    p_q = _run_wrapper(
        root_q, "PAIRED_ACHIEVED_FIDELITY_AT_N", extra_args=["--inject-postpub-fail"],
    )
    assert "PACKET_TERMINAL" not in p_q.stdout
    assert (root_q / "terminal_manifest.json").is_file()  # published but verify skipped


def test_F_branch1_rc_contradicts_map():
    root = _fresh_root("fb1", mkdir=True)
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import finalize
    _write_json(
        root / "command_status" / "preflight.json",
        {"name": "preflight", "rc": 2, "timeout": False, "oom": False},
    )
    _write_json(root / "terminal_receipt.json", {
        "schema": "a_prime_slice1_terminal_receipt/v3",
        "branch": "INSTRUMENT_GAP", "run_root": str(root.resolve()),
    })
    rc = finalize(root, reduce_rc=0)  # receipt INSTRUMENT_GAP but rc=0
    assert rc != 0
    assert not (root / "terminal_manifest.json").is_file()


def test_F_branch2_unknown_branch():
    root = _fresh_root("fb2", mkdir=True)
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import finalize
    _write_json(
        root / "terminal_receipt.json",
        {"branch": "NOT_A_REAL_BRANCH", "run_root": str(root.resolve())},
    )
    rc = finalize(root, reduce_rc=0)
    assert rc != 0
    assert not (root / "terminal_manifest.json").is_file()


def test_F_branch3_candidate_divergence():
    """inject_candidate_branch perturbs CANDIDATE only; receipt authority catches mismatch."""
    root = _fresh_root("fb3", mkdir=True)
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import finalize
    _write_json(
        root / "command_status" / "preflight.json",
        {"name": "preflight", "rc": 0, "timeout": False, "oom": False},
    )
    _write_json(
        root / "terminal_receipt.json",
        {"branch": "PAIRED_ACHIEVED_FIDELITY_AT_N", "run_root": str(root.resolve())},
    )
    rc = finalize(root, reduce_rc=0, inject_candidate_branch="FIDELITY_COLLAPSE")
    assert rc != 0
    assert not (root / "terminal_manifest.json").is_file()


def test_F_branch_clean_marker_byte_equal():
    root = _fresh_root("fbc")
    p = _run_wrapper(root, "FIDELITY_COLLAPSE")
    assert p.returncode == 0
    receipt = json.loads((root / "terminal_receipt.json").read_text())
    man = json.loads((root / "terminal_manifest.json").read_text())
    assert receipt["branch"] == man["branch"]
    assert f"PACKET_TERMINAL {receipt['branch']}" in p.stdout


def test_F_atomic_temp_not_final_on_crash_window():
    root = _fresh_root("fa", mkdir=True)
    from scripts.a_prime_slice1_fidelity_manifest import (
        build_manifest_payload, write_manifest_candidate,
    )
    _write_json(root / "terminal_receipt.json", {"branch": "PAIRED_ACHIEVED_FIDELITY_AT_N"})
    payload = build_manifest_payload(
        root, branch="PAIRED_ACHIEVED_FIDELITY_AT_N", run_root_abs=str(root)
    )
    tmp = write_manifest_candidate(root, payload)
    assert tmp.exists()
    assert not (root / "terminal_manifest.json").exists()
    assert tmp.name.startswith("terminal_manifest.json.tmp.")


def test_F_postpub_state_q_inventory_bind():
    """Post-rename inventory equals post-exit inventory (zero post-publish writes)."""
    root = _fresh_root("fpq")
    snap_path = Path(f"/tmp/aprime_fpq_snap_{os.getpid()}.json")
    if snap_path.exists():
        snap_path.unlink()
    env = _env({"A_PRIME_POSTPUB_SNAP_PATH": str(snap_path)})
    p = _run_wrapper(
        root, "PAIRED_ACHIEVED_FIDELITY_AT_N",
        extra_args=["--inject-postpub-fail"], env=env,
    )
    assert p.returncode != 0
    assert "PACKET_TERMINAL" not in p.stdout
    assert (root / "terminal_manifest.json").is_file()
    assert not (root / "mismatch_diagnostic.json").is_file()
    assert not list(root.glob("terminal_manifest.json.tmp.*"))
    assert snap_path.is_file()
    post_rename = json.loads(snap_path.read_text())
    from scripts.a_prime_slice1_fidelity_manifest import snapshot_run_root
    assert post_rename == snapshot_run_root(root)


def _packet_terminal_code_violations_ast(src: str) -> list[str]:
    """AST: executable string-literal containing PACKET_TERMINAL outside docstrings."""
    tree = ast.parse(src)
    bad: list[str] = []

    def _is_doc(stmt):
        return (
            isinstance(stmt, ast.Expr)
            and isinstance(stmt.value, ast.Constant)
            and isinstance(stmt.value.value, str)
        )

    class V(ast.NodeVisitor):
        def _skip_doc(self, body):
            for i, stmt in enumerate(body):
                if i == 0 and _is_doc(stmt):
                    continue
                self.visit(stmt)

        def visit_Module(self, node):
            self._skip_doc(node.body)

        def visit_FunctionDef(self, node):
            self._skip_doc(node.body)

        def visit_AsyncFunctionDef(self, node):
            self._skip_doc(node.body)

        def visit_ClassDef(self, node):
            self._skip_doc(node.body)

        def visit_Constant(self, node):
            if isinstance(node.value, str) and "PACKET_TERMINAL" in node.value:
                bad.append(f"L{node.lineno}:const:{node.value[:40]!r}")

        def visit_JoinedStr(self, node):
            for v in node.values:
                if isinstance(v, ast.Constant) and isinstance(v.value, str):
                    if "PACKET_TERMINAL" in v.value:
                        bad.append(f"L{node.lineno}:fstring:{v.value[:40]!r}")
            self.generic_visit(node)

    V().visit(tree)
    return bad


def test_reducer_never_prints_packet_terminal():
    """Fails if executable string contains PACKET_TERMINAL outside docstrings/comments."""
    src = REDUCER.read_text(encoding="utf-8")
    assert "print_final" not in src
    assert _packet_terminal_code_violations_ast(src) == []
    for inj in [
        '\nprint("PACKET_TERMINAL X")\n',
        '\nimport sys\nsys.stdout.write("PACKET_TERMINAL Y")\n',
    ]:
        assert _packet_terminal_code_violations_ast(src + inj), f"neg cal miss: {inj!r}"


def test_run_root_mismatch_state_p():
    """receipt run_root mismatch → STATE P, no normalize rewrite."""
    root = _fresh_root("rrm", mkdir=True)
    from scripts.a_prime_slice1_retained_credit_fidelity_wrapper_v0 import finalize
    _write_json(
        root / "terminal_receipt.json",
        {"branch": "PAIRED_ACHIEVED_FIDELITY_AT_N", "run_root": "/wrong/path"},
    )
    rc = finalize(root, reduce_rc=0)
    assert rc == 2
    assert not (root / "terminal_manifest.json").is_file()
    assert (root / "mismatch_diagnostic.json").is_file()


def test_direct_preflight_requires_reducer_wrapper_sha():
    """Direct reducer preflight without authority shas fail-closed (no self-derive)."""
    root = _fresh_root("dp")
    env = _env()
    p = subprocess.run(
        _preflight_argv(root, env, include_auth_shas=False),
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 2, p.stdout + p.stderr
    assert "preflight requires" in p.stderr
    assert "--expect-reducer-sha" in p.stderr or "--expect-wrapper-sha" in p.stderr


def test_dirty_zero_is_legitimate_denominator():
    """--expect-dirty-n 0 must NOT report 'preflight requires'; mismatch → pin_errors."""
    root = _fresh_root("dz")
    env = _env()
    p = subprocess.run(
        _preflight_argv(root, env, include_auth_shas=True, dirty_n="0"),
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=120,
    )
    assert "preflight requires" not in p.stderr
    assert p.returncode == 2, p.stdout + p.stderr
    assert (root / "launch_preflight.json").is_file()
    pf = json.loads((root / "launch_preflight.json").read_text())
    assert any("dirty" in e for e in (pf.get("pin_errors") or []))
    assert pf.get("status") in ("PIN_FAIL", "SYNTHETIC")


def test_snapshot_run_root_mtime_ns_cal():
    """Known-bad cal: os.utime bump changes mtime_ns, keeps sha256; snap inequality fires."""
    from scripts.a_prime_slice1_fidelity_manifest import snapshot_run_root
    root = _fresh_root("snap_cal", mkdir=True)
    rel = "command_status/probe.json"
    path = root / rel
    path.write_bytes(b'{"name":"probe","rc":0}\n')
    snap1 = snapshot_run_root(root)
    assert set(snap1[rel]) == {"sha256", "mtime_ns"}
    st = path.stat()
    # PRIMARY: explicit ns utime (preserve atime; bump mtime by 1s)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000_000))
    snap2 = snapshot_run_root(root)
    assert snap1[rel]["sha256"] == snap2[rel]["sha256"]
    assert snap1[rel]["mtime_ns"] != snap2[rel]["mtime_ns"]
    assert snap1 != snap2


def test_synthetic_failed_preflight_reduce_bypass():
    """Failed SYNTHETIC preflight reused by direct reduce → INSTRUMENT_GAP rc2 (not science)."""
    root = _fresh_root("syn_bypass", mkdir=True)
    dense, nondense = root / "_dense_scratch", root / "_nondense_scratch"
    dense.mkdir(); nondense.mkdir()
    _write_json(root / "launch_preflight.json", {
        "status": "SYNTHETIC", "synthetic": True,
        "pin_errors": ["dirty_n_mismatch:expected=0:got=5"],
        "head_match": False, "dirty_match": True, "parent_match": True,
        "probe_pin_match": True, "reducer_pin_match": True,
        "wrapper_pin_match": True, "rollup_match": True,
    })
    p = subprocess.run(
        [
            sys.executable, "-u", "-B", str(REDUCER),
            "--mode", "reduce", "--run-root", str(root), "--synthetic",
            "--dense-scratch-root", str(dense),
            "--nondense-scratch-root", str(nondense),
        ],
        cwd=str(REPO), env=_env(), capture_output=True, text=True, timeout=60,
    )
    assert p.returncode == 2, p.stdout + p.stderr
    receipt = json.loads((root / "terminal_receipt.json").read_text())
    assert receipt["branch"] == "INSTRUMENT_GAP"
    assert "launch_preflight_inadmissible" in receipt.get("reason", "")
    assert receipt["branch"] not in ("PAIRED_ACHIEVED_FIDELITY_AT_N", "FIDELITY_COLLAPSE")
