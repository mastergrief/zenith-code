"""Candidate-workspace mechanics and t0-t3 transition proofs for the LANDS-AB corpus.

Seam owner: WORKSPACE EXECUTION -- candidate materialization/teardown, manifest
regeneration via the real generator subprocess, and the transition proofs that run the
full corpus. Split out of the battery module under claude dispatch
1785444027626-20273707 (gate-2 BLOCK 1785443944365-76c56f7d): a single >500-line file
mixed evidence registries, IO, subprocess orchestration and result folding, which is
the repository's mixed-responsibility stop condition.

Dependency direction: battery -> transition -> {harness, sources, reducers} -> stdlib.
Neither the production harness nor reducers/sources import this module.

Workspaces are hardlinks where the filesystem allows and copies otherwise -- real files
either way, NEVER symlinks: the fixture builder derives its repo root from
Path(__file__).resolve(), so a symlinked builder resolves back into the real repo and
would silently point the run at the real tool. The real tool is never opened for writing.
"""
from __future__ import annotations

import errno
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DEFAULT))

from calm.llm_computer import lands_ab_diag_corpus_sources as S  # noqa: E402

sys.path.insert(0, str(REPO_DEFAULT / "scripts"))
import lands_ab_dry_exec_diag_corpus as H  # noqa: E402

# t3 injects its marker into ONE diagnostic; ">=1 marker-bearing row" would also pass
# on a broad unintended effect, which is weaker than the claim the receipt makes.
T3_EXPECTED_ROWS = {"G02_missing_watchwrap"}


def _codes(out) -> set:
    return {f["code"] for f in out.get("preflight_failures", [])}


# ---------------------------------------------------------------- transition proofs

MANIFEST_ARGV = [
    "--entry", "scripts/lands_ab_eval_run.py",
    "--entry", "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py",
    "--entry", "calm/hrm_text_158/native_full_stack/lands_ab_eval_site_measurement.py",
    "--entry", "calm/hrm_text_158/native_full_stack/lands_ab_eval_measurement.py",
    "--root-package", "calm.hrm_text_158.native_full_stack",
    "--also-include", "calm/hrm_text_158/native_full_stack/lands_ab_eval_twin_apply.py",
    "--also-include", "calm/hrm_text_158/native_full_stack/lands_ab_eval_schema.py",
    "--also-include", "calm/hrm_text_158/native_full_stack/lands_ab_eval_branch_reducer.py",
]


MATERIALIZE_MODE = {"mode": None}


def _materialize(src: Path, dst: Path, *, force_copy: bool = False) -> None:
    """Hardlink when possible, copy otherwise. NEVER symlink.

    force_copy=True always copies (C3): required for any path a hostile will mutate so
    a same-fs hardlink cannot share the live tracked inode.

    Non-force is idempotent (materializer defect cycle): if dst already exists, accept
    only when it already represents src — same (st_dev, st_ino) hardlink, OR a
    byte-equal distinct copy. Never catch EEXIST and blind-copy over an existing inode
    (SameFileError on same-fs re-materialize). Other errors still raise.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    if force_copy:
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        shutil.copy2(src, dst)
        MATERIALIZE_MODE["mode"] = "force_copy"
    else:
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink():
                raise RuntimeError(f"workspace entry is a symlink: {dst}")
            src_st, dst_st = os.stat(src), os.stat(dst)
            same_inode = (src_st.st_dev, src_st.st_ino) == (dst_st.st_dev, dst_st.st_ino)
            if same_inode:
                MATERIALIZE_MODE["mode"] = MATERIALIZE_MODE["mode"] or "hardlink"
                return
            if src_st.st_size == dst_st.st_size and src.read_bytes() == dst.read_bytes():
                MATERIALIZE_MODE["mode"] = MATERIALIZE_MODE["mode"] or "copy"
                return
            raise RuntimeError(
                f"materialize dst exists but does not represent src: {dst} vs {src}"
            )
        try:
            os.link(src, dst)
            MATERIALIZE_MODE["mode"] = MATERIALIZE_MODE["mode"] or "hardlink"
        except OSError as exc:
            # EXDEV / unsupported link → copy. Never treat "dst exists" as copy-over.
            if getattr(exc, "errno", None) in (errno.EEXIST, errno.EISDIR):
                raise
            shutil.copy2(src, dst)
            MATERIALIZE_MODE["mode"] = "copy"
    assert not dst.is_symlink(), f"workspace entry is a symlink: {dst}"


def assert_non_aliased(ws_path: Path, live_path: Path) -> None:
    """Fail-closed: workspace file must not share the live tracked inode (C3).

    Accepts either st_nlink==1 on the workspace file OR (st_dev, st_ino) distinct
    from the live file. Assertion failure = case error.
    """
    if not ws_path.is_file():
        raise AssertionError(f"workspace path missing for non-alias check: {ws_path}")
    if not live_path.is_file():
        raise AssertionError(f"live path missing for non-alias check: {live_path}")
    ws_st = os.stat(ws_path)
    live_st = os.stat(live_path)
    same_inode = (ws_st.st_dev, ws_st.st_ino) == (live_st.st_dev, live_st.st_ino)
    if same_inode:
        raise AssertionError(
            f"workspace file aliases live inode: {ws_path} == {live_path} "
            f"dev={ws_st.st_dev} ino={ws_st.st_ino} nlink={ws_st.st_nlink}"
        )
    # Prefer nlink==1 when same filesystem family; not required if already distinct inodes.
    # Distinct (dev,ino) alone is sufficient proof of non-aliasing.


def ensure_mutable_workspace_file(repo: Path, ws: Path, rel: str) -> Path:
    """Force-copy `rel` into ws and assert non-aliasing before any hostile mutation (C3)."""
    src, dst = repo / rel, ws / rel
    if not src.is_file():
        raise FileNotFoundError(f"live source missing for mutable materialize: {rel}")
    _materialize(src, dst, force_copy=True)
    assert_non_aliased(dst, src)
    return dst


def build_candidate_workspace(repo: Path, ws: Path, tool_suffix: str,
                              lineage: bool = True,
                              mutable_paths: list | None = None) -> Path:
    """Hardlink farm + a REAL copy of the tool carrying tool_suffix.

    Only the bounded set the run touches is materialized: the manifest's entries, the
    fixture builder, the corpus fixtures, and the manifest itself. The real tool is
    never opened for writing -- the candidate is always a fresh file.

    mutable_paths: repo-relative paths that a hostile will mutate — force-copied and
    non-alias asserted (C3). Read-only paths may still hardlink.
    """
    mutable = set(mutable_paths or ())
    corpus = S.load_corpus(repo)
    man_rel = corpus["bound_to"]["base_manifest"]
    manifest = json.loads((repo / man_rel).read_text())
    wanted = [e["path"] for e in manifest["entries"]]
    wanted += [S.HARNESS_REL, man_rel,
               f"{S.FIXTURE_DIR}/ROWS.json", f"{S.FIXTURE_DIR}/{S.BASELINE_NAME}",
               f"{S.FIXTURE_DIR}/GENERATION.json",
               f"{S.FIXTURE_DIR}/MIGRATION_v0_to_v1.json",
               f"{S.FIXTURE_DIR}/A4_EMISSION_ORDINAL_BASELINE.json",
               # Parent v0 fixtures required by validate_generation_receipt disk pins
               f"{S.GENERATIONS['v0']['fixture_dir']}/ROWS.json",
               f"{S.GENERATIONS['v0']['fixture_dir']}/{S.GENERATIONS['v0']['baseline_name']}",
               "calm/llm_computer/lands_ab_diag_corpus_sources.py",
               "calm/llm_computer/lands_ab_diag_corpus_reducers.py",
               "scripts/lands_ab_dry_exec_diag_corpus.py"]
    # The fixture builder's file closure is IMPLICIT (module-level constants, not
    # manifest entries): it reads the v6 base packet it derives every fixture from,
    # and shas the manifest generator. Omitting them fails mid-run, not at preflight.
    wanted += ["artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority"
               "_LANDS_AB_EVAL_launch_packet_v6.json",
               "scripts/lands_ab_science_source_manifest.py"]
    # Deterministic dedup (materializer defect cycle): preserve first-seen order;
    # re-materializing the same rel must not hit FileExistsError on same-fs hardlinks.
    seen: set[str] = set()
    deduped: list[str] = []
    for rel in wanted:
        if rel in seen:
            continue
        seen.add(rel)
        deduped.append(rel)
    wanted = deduped
    for rel in wanted:
        if rel == S.TOOL_REL:
            continue  # candidate is written fresh below; never hardlink the real tool
        src, dst = repo / rel, ws / rel
        if not src.is_file():
            continue
        _materialize(src, dst, force_copy=(rel in mutable))
        if rel in mutable:
            assert_non_aliased(dst, src)
    # packages the closure walker needs to import
    for pkg in ("calm/__init__.py", "calm/hrm_text_158/__init__.py",
                "calm/hrm_text_158/native_full_stack/__init__.py",
                "calm/llm_computer/__init__.py", "calm/llm_computer/tests/__init__.py"):
        src, dst = repo / pkg, ws / pkg
        if src.is_file() and not dst.exists():
            _materialize(src, dst)
    # Lineage closure: the tool resolves packet-lineage artifacts to decide whether a
    # historical reference sits inside DEAD lineage. With them absent, one L1/J2
    # dead-lineage failure fires FIRST and masks 14 unrelated diagnostics -- measured,
    # not assumed (15 differing rows -> 1 once these are present).
    # Only the cases that EXECUTE the corpus need this; preflight-only cases do not,
    # and it is ~611 files. Cost discipline, not a proof change.
    if lineage:
        for src in (repo / "artifacts/acc_entropy").glob("*.json"):
            dst = ws / "artifacts/acc_entropy" / src.name
            if not dst.exists():
                _materialize(src, dst)
    cand = ws / S.TOOL_REL
    cand.parent.mkdir(parents=True, exist_ok=True)
    cand.write_text((repo / S.TOOL_REL).read_text() + tool_suffix)
    return cand


def regenerate_manifest(repo: Path, ws: Path, corpus: dict):
    out = corpus["bound_to"]["base_manifest"]
    argv = [sys.executable, str(repo / "scripts/lands_ab_science_source_manifest.py")] \
        + MANIFEST_ARGV + ["--repo-root", str(ws), "--out", str(ws / out)]
    return subprocess.run(argv, cwd=str(ws), capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": str(ws)})


def _observed(ws: Path):
    """Preflight then measure, returning (preflight_codes, observed_map, result)."""
    corpus = S.load_corpus(ws)
    baseline = S.load_baseline(S.canonical_baseline_path(ws))
    codes = {f["code"] for f in H.preflight(ws, corpus, baseline, True, "A1")}
    if codes:
        return codes, None, None
    result, first = H.measure(ws, corpus, baseline, "A1")
    return codes, first, result


def _map_diff(a: dict, b: dict) -> list:
    keys = sorted(set(a) | set(b))
    return [{"row_id": k, "control": a.get(k), "candidate": b.get(k)}
            for k in keys if a.get(k) != b.get(k)]


def run_transition_proofs(repo: Path, full: bool = True) -> dict:
    """t0 control, t1 changed-but-equivalent, t2 stale manifest, t3 changed behavior.

    The comparator is the t0 CONTROL WORKSPACE, not the committed baseline. A relocated
    workspace legitimately perturbs rows whose fixtures are repo-root-relative (measured:
    M_main_61 asserts a run_root "under repo root", which stops holding once the root
    moves, so an earlier check fires). Diffing candidate-vs-control cancels relocation
    exactly, so no row needs an exclusion and the proof still answers the real question:
    did THIS tool edit change behavior?
    """
    results = []
    tmp = Path(tempfile.mkdtemp(prefix="a0ws_"))
    try:
        # ---- t2 first: cheapest, and it validates the workspace mechanism ----
        ws2 = tmp / "t2"
        build_candidate_workspace(repo, ws2, "\n# candidate edit, manifest NOT regenerated\n")
        c2 = S.load_corpus(ws2)
        _, out2 = H.cmd_accept(ws2, c2, step="A1", quiet=True, preflight_only=True)
        results.append({"case": "t2_stale_manifest", "preregistered": ["manifest_stale"],
                        "observed": sorted(_codes(out2)), "stage": "pre-subprocess",
                        "subprocesses": out2.get("subprocesses_spawned"),
                        "ok": _codes(out2) == {"manifest_stale"}})
        if not full:
            return {"check_id": "A0_TRANSITION_PROOFS", "cases": results,
                    "workspace_removed": False, "materialize_mode": MATERIALIZE_MODE["mode"],
                    "verdict": "PASS" if all(r.get("ok") for r in results) else "FAIL"}

        # ---- t0: CONTROL. Same relocation, tool bytes identical to the repo tool ----
        ws0 = tmp / "t0"
        build_candidate_workspace(repo, ws0, "")
        regenerate_manifest(repo, ws0, S.load_corpus(ws0))
        pre0, map0, res0 = _observed(ws0)
        relocation = [f["row_id"] for f in (res0 or {}).get("baseline_ledger_failures") or []]
        results.append({
            "case": "t0_control_workspace", "preflight_codes": sorted(pre0),
            "preregistered": "clean preflight; establishes the control map. Any nonzero "
                             "baseline delta here is a RELOCATION artifact, disclosed by row.",
            "relocation_artifact_rows": relocation,
            "rows": len(map0 or {}), "ok": not pre0 and bool(map0)})

        # ---- t1: behavior-equivalent candidate + REGENERATED manifest ----
        ws1 = tmp / "t1"
        build_candidate_workspace(repo, ws1, "\n# behavior-identical comment-only candidate edit\n")
        regen1 = regenerate_manifest(repo, ws1, S.load_corpus(ws1))
        pre1, map1, res1 = _observed(ws1)
        diff1 = _map_diff(map0 or {}, map1 or {})
        results.append({
            "case": "t1_equivalent_candidate", "regen_rc": regen1.returncode,
            "preflight_codes": sorted(pre1),
            "tool_identity": (res1 or {}).get("tool_identity"),
            "preregistered": "no tool-sha failure at A1; reaches comparison; ZERO rows "
                             "differ from the control",
            "rows_differing_from_control": len(diff1), "diff_sample": diff1[:5],
            "ok": not pre1 and diff1 == []})

        # ---- t3: behavior-CHANGED candidate + regenerated manifest ----
        ws3 = tmp / "t3"
        cand3 = build_candidate_workspace(repo, ws3, "")
        cand3.write_text(cand3.read_text().replace(
            "missing bin/watch-wrap", "missing bin/watch-wrap [T3]", 1))
        regen3 = regenerate_manifest(repo, ws3, S.load_corpus(ws3))
        pre3, map3, res3 = _observed(ws3)
        diff3 = _map_diff(map0 or {}, map3 or {})
        attributable = [d for d in diff3 if "[T3]" in str((d["candidate"] or {}).get("msg_key"))]
        results.append({
            "case": "t3_changed_diagnostic", "regen_rc": regen3.returncode,
            "preflight_codes": sorted(pre3),
            "preregistered": {"differing_row_ids": sorted(T3_EXPECTED_ROWS), "count": 1,
                              "attribution": "1/1 carry the injected [T3] marker"},
            "rows_differing_from_control": len(diff3),
            "attributable_to_injection": len(attributable),
            "differing_row_ids": sorted(d["row_id"] for d in diff3),
            "set_equal": {d["row_id"] for d in diff3} == T3_EXPECTED_ROWS,
            "ok": (not pre3
                   and {d["row_id"] for d in diff3} == T3_EXPECTED_ROWS
                   and len(diff3) == 1
                   and len(attributable) == 1)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    passed = all(r.get("ok") for r in results)
    return {"check_id": "A0_TRANSITION_PROOFS", "cases": results,
            "workspace_removed": not tmp.exists(),
            "materialize_mode": MATERIALIZE_MODE["mode"],
            "verdict": "PASS" if passed else "FAIL"}


def run_materialize_same_fs_idempotence() -> dict:
    """Same-fs duplicate non-force materialize: no exception, bytes stable, not symlink.

    Proves the materializer defect-cycle cure: second _materialize of an already-linked
    or already-copied dst does not raise and does not rewrite bytes.
    """
    td = Path(tempfile.mkdtemp(prefix="mat_samefs_"))
    try:
        src = td / "src.txt"
        dst = td / "dst.txt"
        payload = b"materialize-idempotence-payload-v1\n"
        src.write_bytes(payload)
        MATERIALIZE_MODE["mode"] = None
        _materialize(src, dst, force_copy=False)
        mode1 = MATERIALIZE_MODE["mode"]
        before = dst.read_bytes()
        st1 = os.stat(dst)
        # second call — must be a no-op success
        _materialize(src, dst, force_copy=False)
        mode2 = MATERIALIZE_MODE["mode"]
        after = dst.read_bytes()
        st2 = os.stat(dst)
        ok = (
            before == after == payload
            and not dst.is_symlink()
            and (st1.st_dev, st1.st_ino) == (st2.st_dev, st2.st_ino)
            and mode1 in ("hardlink", "copy")
        )
        return {
            "case": "materialize_same_fs_idempotence",
            "ok": ok,
            "mode_first": mode1,
            "mode_second": mode2,
            "bytes_unchanged": before == after,
            "not_symlink": not dst.is_symlink(),
            "inode_stable": (st1.st_dev, st1.st_ino) == (st2.st_dev, st2.st_ino),
            "same_inode_as_src": (st1.st_dev, st1.st_ino) == (
                os.stat(src).st_dev, os.stat(src).st_ino),
        }
    finally:
        shutil.rmtree(td, ignore_errors=True)


def test_materialize_same_fs_idempotence():
    """pytest entry: same-fs duplicate non-force materialize regression."""
    report = run_materialize_same_fs_idempotence()
    assert report["ok"], report


if __name__ == "__main__":
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_DEFAULT
    mode = sys.argv[2] if len(sys.argv) > 2 else "transition"
    if mode == "materialize-idempotence":
        out = run_materialize_same_fs_idempotence()
        print(json.dumps(out, indent=1))
        raise SystemExit(0 if out.get("ok") else 1)
    out = run_transition_proofs(repo)
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if out["verdict"] == "PASS" else 1)
