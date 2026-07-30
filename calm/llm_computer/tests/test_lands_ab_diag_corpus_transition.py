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


def _materialize(src: Path, dst: Path) -> None:
    """Hardlink when possible, copy otherwise. NEVER symlink.

    The repo lives on drvfs and tempfile lives on ext4, so os.link raises EXDEV
    across them. A copy is an equally real file and preserves the only invariant
    that matters here: the fixture builder must resolve INSIDE the workspace, which
    a symlink would defeat by resolving back to the real repo.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        MATERIALIZE_MODE["mode"] = MATERIALIZE_MODE["mode"] or "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        MATERIALIZE_MODE["mode"] = "copy"
    assert not dst.is_symlink(), f"workspace entry is a symlink: {dst}"


def build_candidate_workspace(repo: Path, ws: Path, tool_suffix: str,
                              lineage: bool = True) -> Path:
    """Hardlink farm + a REAL copy of the tool carrying tool_suffix.

    Only the bounded set the run touches is materialized: the manifest's entries, the
    fixture builder, the corpus fixtures, and the manifest itself. The real tool is
    never opened for writing -- the candidate is always a fresh file.
    """
    corpus = S.load_corpus(repo)
    man_rel = corpus["bound_to"]["base_manifest"]
    manifest = json.loads((repo / man_rel).read_text())
    wanted = [e["path"] for e in manifest["entries"]]
    wanted += [S.HARNESS_REL, man_rel,
               f"{S.FIXTURE_DIR}/ROWS.json", f"{S.FIXTURE_DIR}/{S.BASELINE_NAME}"]
    # The fixture builder's file closure is IMPLICIT (module-level constants, not
    # manifest entries): it reads the v6 base packet it derives every fixture from,
    # and shas the manifest generator. Omitting them fails mid-run, not at preflight.
    wanted += ["artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority"
               "_LANDS_AB_EVAL_launch_packet_v6.json",
               "scripts/lands_ab_science_source_manifest.py"]
    for rel in wanted:
        if rel == S.TOOL_REL:
            continue  # candidate is written fresh below; never hardlink the real tool
        src, dst = repo / rel, ws / rel
        if not src.is_file():
            continue
        _materialize(src, dst)
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



if __name__ == "__main__":
    repo = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else REPO_DEFAULT
    out = run_transition_proofs(repo)
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if out["verdict"] == "PASS" else 1)
