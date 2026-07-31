"""Generation-authority negative cases for LANDS-AB diag corpus (plan v8 split).

Owns GEN_AUTH_PREREG (30 cases) and run_generation_authority_negatives orchestration.
Split into ≤150L helpers: pure / isolated / CLI / selector+live. Case counts and
expected code-sets are behavior-identical to the pre-extract battery surface.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DEFAULT))

from calm.llm_computer import lands_ab_diag_corpus_sources as S  # noqa: E402
from calm.llm_computer.tests.test_lands_ab_diag_corpus_transition import (  # noqa: E402
    assert_non_aliased, build_candidate_workspace, ensure_mutable_workspace_file)

sys.path.insert(0, str(REPO_DEFAULT / "scripts"))
import lands_ab_dry_exec_diag_corpus as H  # noqa: E402

FAST_TIER = "fast_pre_subprocess"
STAGE = "pre-subprocess"


GEN_AUTH_PREREG = {
    # N1: pure-seam receipt field mismatch (no disk mutation)
    "gen_n1_wrong_generation_field": {"generation_receipt_mismatch"},
    # N1b: GENERATION.json absent — isolated workspace only
    "gen_n1_receipt_absent": {"generation_receipt_absent"},
    # N2: explicit generation=v0 + post-split tool cannot A4_ACCEPT
    "gen_n2_v0_explicit_no_a4_accept": {
        "accept_success_false", "candidate_sha_not_baseline_at_A0"},
    # N2b DISTINCT: default selector (ACTIVE=v1, generation kw omitted) with v0 corpus
    "gen_n2b_default_active_v1_with_v0_corpus": {"accept_success_false"},
    # N3: tool mismatch in isolated workspace (D2 include form)
    "gen_n3_tool_mismatch_includes_code": {"generation_tool_mismatch"},
    # N4: mint refuse when baseline exists + byte preservation (read-only live)
    "gen_n4_baseline_exists_refuse": {
        "baseline_exists_refuse", "existing_baseline_bytes_unchanged"},
    # Explicit v0 selection remains callable and binds v0 rows/baseline
    "gen_v0_explicit_binds_v0_paths": {"v0_rows_bound", "v0_baseline_bound"},
    # Pure-seam pin mutations (receipt dict only — no disk write)
    "gen_mut_schema_rows": {"generation_pin_mismatch"},
    "gen_mut_schema_baseline": {"generation_pin_mismatch"},
    "gen_mut_baseline_name": {"generation_pin_mismatch"},
    "gen_mut_baseline_head": {"generation_pin_mismatch"},
    "gen_mut_tool_sha_pin": {"generation_pin_mismatch"},
    "gen_mut_parent_rows_sha": {"generation_pin_mismatch"},
    "gen_mut_parent_baseline_sha": {"generation_pin_mismatch"},
    "gen_mut_prep_package_sha": {"generation_pin_mismatch"},
    "gen_mut_a4_allowlist": {"generation_pin_mismatch"},
    "gen_mut_migration_sha": {"generation_pin_mismatch"},
    "gen_mut_v1_rows_sha": {"generation_pin_mismatch"},
    "gen_mut_path_escape_baseline_name": {
        "generation_path_escape", "generation_pin_mismatch"},
    "gen_mut_non_object_receipt": {"generation_receipt_mismatch"},
    # B1 hostiles: malformed nested ROWS allowlist shapes (isolated ROWS bytes)
    "gen_hostile_rows_allowlist_list": {"generation_receipt_mismatch"},
    "gen_hostile_rows_allowlist_str_per_step": {"generation_receipt_mismatch"},
    # B2 hostiles: unknown generation API + CLI
    "gen_unknown_generation_api": {"unknown_generation"},
    "gen_unknown_generation_cli": {"unknown_generation"},
    # C2 CLI ingress hostiles (actual harness subprocess on isolated ws)
    "cli_generation_absent": {"generation_receipt_absent"},
    "cli_generation_wrong_field": {"generation_receipt_mismatch"},
    "cli_generation_unparseable": {"generation_receipt_mismatch"},
    "cli_generation_non_object": {"generation_receipt_mismatch"},
    "cli_rows_unparseable": {"rows_unparseable"},
    # B3/C3 live-byte isolation proof
    "gen_auth_live_bytes_unchanged": {
        "tool_unchanged", "generation_unchanged", "rows_unchanged"},
}




def _gen_auth_codes(fails: list) -> set:
    return {f.get("code") for f in fails if f.get("code")}


def _live_authority_hashes(repo: Path) -> dict:
    """Hashes of authoritative tracked surfaces that gen-auth must not mutate."""
    return {
        "tool": S.sha256_file(repo / S.TOOL_REL),
        "generation": S.sha256_file(
            repo / S.GENERATIONS["v1"]["fixture_dir"] / "GENERATION.json"),
        "rows": S.sha256_file(repo / S.GENERATIONS["v1"]["fixture_dir"] / "ROWS.json"),
    }


INCLUDE_FORM = {
    "gen_n3_tool_mismatch_includes_code",
    "gen_n2b_default_active_v1_with_v0_corpus",
    "gen_hostile_rows_allowlist_list",
    "gen_hostile_rows_allowlist_str_per_step",
    "cli_generation_absent",
    "cli_generation_wrong_field",
    "cli_generation_unparseable",
    "cli_generation_non_object",
    "cli_rows_unparseable",
}
CLI_HOSTILE = {
    "cli_generation_absent",
    "cli_generation_wrong_field",
    "cli_generation_unparseable",
    "cli_generation_non_object",
    "cli_rows_unparseable",
}


def _record(cases, name, observed, extra=None, tier=FAST_TIER):
    want = GEN_AUTH_PREREG[name]
    if name in INCLUDE_FORM:
        ok = want.issubset(observed) and "A4_ACCEPT" not in observed
        set_eq = want.issubset(observed)
    else:
        ok = observed == want
        set_eq = observed == want
    if name in CLI_HOSTILE and extra is not None:
        no_tb = not extra.get("traceback_seen", True)
        rc_ok = extra.get("rc", 0) not in (0, None) and extra.get("rc", 0) != 0
        zero_sp = extra.get("subprocesses_spawned", -1) == 0
        ok = ok and no_tb and rc_ok and zero_sp
    entry = {
        "case": name, "observed": sorted(observed), "set_equal": set_eq,
        "preregistered": sorted(want), "tier": tier,
        "evidence_tier": "preregistered", "stage": STAGE, "subprocesses": 0,
        "ok": ok,
    }
    if extra:
        entry["detail"] = extra
    cases.append(entry)


def run_gen_auth_pure_receipt_cases(repo: Path, good: dict) -> list:
    """Pure-seam receipt mutations — no disk write of live fixtures."""
    cases = []
    bad = dict(good)
    bad["generation"] = "v0"
    fails = S.validate_generation_receipt(repo, "v1", bad)
    _record(cases, "gen_n1_wrong_generation_field", _gen_auth_codes(fails),
            {"reach": "validate_generation_receipt_pure"})
    pin_cases = [
        ("gen_mut_schema_rows", "schema_rows", "TAMPERED_SCHEMA"),
        ("gen_mut_schema_baseline", "schema_baseline", "TAMPERED_BASELINE_SCHEMA"),
        ("gen_mut_baseline_name", "baseline_name", "BASELINE_TAMPERED.json"),
        ("gen_mut_baseline_head", "baseline_head", "deadbeef"),
        ("gen_mut_tool_sha_pin", "tool_sha256_at_authoring", "0" * 64),
        ("gen_mut_parent_rows_sha", "parent_rows_sha256", "1" * 64),
        ("gen_mut_parent_baseline_sha", "parent_baseline_sha256", "2" * 64),
        ("gen_mut_prep_package_sha", "prep_package_sha256", "3" * 64),
        ("gen_mut_a4_allowlist", "ast_allowlist_A4_allowed_over_150", ["main"]),
        ("gen_mut_migration_sha", "migration_carrier_sha256", "4" * 64),
        ("gen_mut_v1_rows_sha", "v1_rows_sha256", "5" * 64),
    ]
    for name, field, value in pin_cases:
        bad = dict(good)
        bad[field] = value
        fails = S.validate_generation_receipt(repo, "v1", bad)
        _record(cases, name, _gen_auth_codes(fails), {"field": field, "pure_seam": True})
    bad = dict(good)
    bad["baseline_name"] = "../secrets/BASELINE.json"
    fails = S.validate_generation_receipt(repo, "v1", bad)
    _record(cases, "gen_mut_path_escape_baseline_name", _gen_auth_codes(fails),
            {"baseline_name": bad["baseline_name"], "pure_seam": True})
    try:
        fails = S.validate_generation_receipt(repo, "v1", ["not", "an", "object"])
        _record(cases, "gen_mut_non_object_receipt", _gen_auth_codes(fails), {"pure_seam": True})
    except Exception as exc:  # pragma: no cover
        cases.append({
            "case": "gen_mut_non_object_receipt", "observed": ["traceback"],
            "preregistered": sorted(GEN_AUTH_PREREG["gen_mut_non_object_receipt"]),
            "set_equal": False, "tier": FAST_TIER, "evidence_tier": "preregistered",
            "stage": STAGE, "subprocesses": 0, "ok": False, "detail": {"error": repr(exc)},
        })
    return cases


def run_gen_auth_isolated_carrier_cases(repo: Path, good: dict) -> list:
    """Isolated force-copied workspace hostiles (absent GENERATION, N3 tool, ROWS allowlist)."""
    cases = []
    fixture_dir = S.GENERATIONS["v1"]["fixture_dir"]
    # N1 absent
    tmp = Path(tempfile.mkdtemp(prefix="genauth_n1abs_"))
    try:
        gen_rel = f"{fixture_dir}/GENERATION.json"
        ws = tmp / "ws"
        build_candidate_workspace(repo, ws, "", lineage=False, mutable_paths=[gen_rel])
        gpath = ensure_mutable_workspace_file(repo, ws, gen_rel)
        gpath.unlink()
        fails = S.preflight_generation(ws, "v1")
        _record(cases, "gen_n1_receipt_absent", _gen_auth_codes(fails),
                {"reach": "preflight_generation_isolated_ws", "non_aliased": True})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # N3 tool mismatch
    tmp3 = Path(tempfile.mkdtemp(prefix="genauth_n3_"))
    try:
        ws3 = tmp3 / "ws"
        build_candidate_workspace(repo, ws3, "\n# n3_mismatch_probe\n", lineage=False)
        corpus_v1_ws = S.load_corpus(ws3, "v1")
        rc3, out3 = H.cmd_accept(ws3, corpus_v1_ws, step="A4", quiet=True, generation="v1")
        codes3 = _gen_auth_codes(out3.get("preflight_failures") or [])
        if out3.get("verdict") == "A4_ACCEPT":
            codes3.add("A4_ACCEPT")
        _record(cases, "gen_n3_tool_mismatch_includes_code", codes3,
                {"rc": rc3, "verdict": out3.get("verdict"), "d2_include_form": True,
                 "isolated_ws": True})
    finally:
        shutil.rmtree(tmp3, ignore_errors=True)
    # ROWS allowlist hostiles
    def _rows_hostile(name: str, mutator):
        tdir = Path(tempfile.mkdtemp(prefix="genauth_rows_"))
        try:
            rows_rel = f"{fixture_dir}/ROWS.json"
            ws = tdir / "ws"
            build_candidate_workspace(repo, ws, "", lineage=False, mutable_paths=[rows_rel])
            rows_p = ensure_mutable_workspace_file(repo, ws, rows_rel)
            assert_non_aliased(rows_p, repo / rows_rel)
            rows_obj = json.loads(rows_p.read_text())
            mutator(rows_obj)
            rows_p.write_text(json.dumps(rows_obj, indent=1) + "\n")
            fails = S.validate_generation_receipt(ws, "v1", good)
            reasons = {f.get("reason") for f in fails}
            _record(cases, name, _gen_auth_codes(fails),
                    {"reasons": sorted(x for x in reasons if x),
                     "isolated_ws": True, "force_copy": True})
        finally:
            shutil.rmtree(tdir, ignore_errors=True)

    def _mut_allowlist_list(r):
        r["ast_allowlist"] = ["not", "a", "dict"]

    def _mut_per_step_str(r):
        al = r.get("ast_allowlist")
        if not isinstance(al, dict):
            r["ast_allowlist"] = {}
            al = r["ast_allowlist"]
        al["per_step"] = "not-a-dict"

    _rows_hostile("gen_hostile_rows_allowlist_list", _mut_allowlist_list)
    _rows_hostile("gen_hostile_rows_allowlist_str_per_step", _mut_per_step_str)
    return cases


def run_gen_auth_cli_cases(repo: Path) -> list:
    """C2 CLI ingress hostiles via actual harness subprocess on isolated ws."""
    cases = []
    fixture_dir = S.GENERATIONS["v1"]["fixture_dir"]
    gen_rel = f"{fixture_dir}/GENERATION.json"
    rows_rel = f"{fixture_dir}/ROWS.json"

    def run_cli_hostile(name: str, mutator, mutable_rels: list[str]):
        tdir = Path(tempfile.mkdtemp(prefix="genauth_cli_"))
        try:
            ws = tdir / "ws"
            build_candidate_workspace(
                repo, ws, "", lineage=False, mutable_paths=mutable_rels)
            for rel in mutable_rels:
                ensure_mutable_workspace_file(repo, ws, rel)
            mutator(ws)
            argv = [
                sys.executable, str(repo / "scripts/lands_ab_dry_exec_diag_corpus.py"),
                "--repo-root", str(ws),
                "--accept", "--step", "A0", "--generation", "v1",
                "--preflight-only",
            ]
            proc = subprocess.run(
                argv, cwd=str(repo), capture_output=True, text=True,
                env={**__import__("os").environ, "PYTHONPATH": str(repo)},
            )
            out = proc.stdout or ""
            err = proc.stderr or ""
            tb = ("Traceback" in out) or ("Traceback" in err)
            codes = set()
            subprocs = -1
            try:
                payload = json.loads(out)
                codes = _gen_auth_codes(payload.get("preflight_failures") or [])
                subprocs = payload.get("subprocesses_spawned", -1)
            except json.JSONDecodeError:
                codes = {"unparseable_stdout"}
            _record(cases, name, codes, {
                "rc": proc.returncode, "traceback_seen": tb,
                "subprocesses_spawned": subprocs,
                "stdout_tail": out[-300:], "stderr_tail": err[-200:],
                "isolated_ws": True, "mutable_paths": mutable_rels,
            })
        finally:
            shutil.rmtree(tdir, ignore_errors=True)

    def _mut_gen_absent(ws):
        p = ws / gen_rel
        if p.is_file():
            p.unlink()

    def _mut_gen_wrong(ws):
        p = ensure_mutable_workspace_file(repo, ws, gen_rel)
        obj = json.loads(p.read_text())
        obj["generation"] = "v0"
        p.write_text(json.dumps(obj, indent=2) + "\n")

    def _mut_gen_unparseable(ws):
        p = ensure_mutable_workspace_file(repo, ws, gen_rel)
        p.write_text("{not valid json\n")

    def _mut_gen_non_object(ws):
        p = ensure_mutable_workspace_file(repo, ws, gen_rel)
        p.write_text(json.dumps(["not", "an", "object"]) + "\n")

    def _mut_rows_unparseable(ws):
        p = ensure_mutable_workspace_file(repo, ws, rows_rel)
        p.write_text("{not valid rows json\n")

    run_cli_hostile("cli_generation_absent", _mut_gen_absent, [gen_rel])
    run_cli_hostile("cli_generation_wrong_field", _mut_gen_wrong, [gen_rel])
    run_cli_hostile("cli_generation_unparseable", _mut_gen_unparseable, [gen_rel])
    run_cli_hostile("cli_generation_non_object", _mut_gen_non_object, [gen_rel])
    run_cli_hostile("cli_rows_unparseable", _mut_rows_unparseable, [rows_rel])
    # unknown generation CLI
    proc = subprocess.run(
        [sys.executable, str(repo / "scripts/lands_ab_dry_exec_diag_corpus.py"),
         "--repo-root", str(repo), "--accept", "--step", "A0",
         "--generation", "v99_not_a_generation", "--preflight-only"],
        cwd=str(repo), capture_output=True, text=True,
        env={**__import__("os").environ, "PYTHONPATH": str(repo)},
    )
    cli_codes = set()
    try:
        payload = json.loads(proc.stdout or "{}")
        cli_codes = _gen_auth_codes(payload.get("preflight_failures") or [])
    except json.JSONDecodeError:
        cli_codes = {"unparseable_stdout"}
    if "unknown_generation" in (proc.stdout or "") and not cli_codes:
        cli_codes.add("unknown_generation")
    _record(cases, "gen_unknown_generation_cli", cli_codes,
            {"rc": proc.returncode, "stdout_tail": (proc.stdout or "")[-300:],
             "stderr_tail": (proc.stderr or "")[-200:]})
    return cases


def run_gen_auth_selector_and_live_cases(repo: Path, live_before: dict) -> list:
    """N2/N2b/N4/v0 binds/unknown API + live-byte unchanged proof."""
    cases = []
    corpus_v0 = S.load_corpus(repo, "v0")
    rc, out = H.cmd_accept(repo, corpus_v0, step="A0", quiet=True, generation="v0")
    tokens = set()
    if rc != 0 and out.get("verdict") not in ("A4_ACCEPT", "A0_ACCEPT"):
        tokens.add("accept_success_false")
    codes = _gen_auth_codes(out.get("preflight_failures") or [])
    if "candidate_sha_not_baseline_at_A0" in codes:
        tokens.add("candidate_sha_not_baseline_at_A0")
    _record(cases, "gen_n2_v0_explicit_no_a4_accept", tokens,
            {"rc": rc, "verdict": out.get("verdict"), "codes": sorted(codes),
             "distinct_from": "gen_n2b_default_active_v1_with_v0_corpus"})
    rc2, out2 = H.cmd_accept(repo, corpus_v0, step="A0", quiet=True)
    tokens2 = set()
    if rc2 != 0 and out2.get("verdict") not in ("A4_ACCEPT", "A0_ACCEPT"):
        tokens2.add("accept_success_false")
    codes2 = _gen_auth_codes(out2.get("preflight_failures") or [])
    _record(cases, "gen_n2b_default_active_v1_with_v0_corpus", tokens2,
            {"rc": rc2, "verdict": out2.get("verdict"), "codes": sorted(codes2),
             "generation_kw": None, "active_generation": S.ACTIVE_GENERATION,
             "corpus_schema": corpus_v0.get("schema"),
             "distinct_from": "gen_n2_v0_explicit_no_a4_accept"})
    corpus_v1 = S.load_corpus(repo, "v1")
    basel = S.canonical_baseline_path(repo, "v1")
    before = S.sha256_file(basel)
    rc4 = H.cmd_mint_baseline(repo, corpus_v1, quiet=True, generation="v1")
    after = S.sha256_file(basel)
    tokens4 = set()
    if rc4 == 1:
        tokens4.add("baseline_exists_refuse")
    if after == before:
        tokens4.add("existing_baseline_bytes_unchanged")
    _record(cases, "gen_n4_baseline_exists_refuse", tokens4,
            {"rc": rc4, "sha_before": before, "sha_after": after})
    tokens_v0 = set()
    rows_v0 = S.rows_path(repo, "v0")
    base_v0 = S.canonical_baseline_path(repo, "v0")
    if rows_v0.is_file() and rows_v0.resolve().as_posix().endswith(
            "lands_ab_dry_exec_diag_corpus_v0/ROWS.json"):
        tokens_v0.add("v0_rows_bound")
    if base_v0.is_file() and base_v0.name == "BASELINE_HEAD_9f471b3.json":
        tokens_v0.add("v0_baseline_bound")
    S.load_corpus(repo, "v0")
    _record(cases, "gen_v0_explicit_binds_v0_paths", tokens_v0,
            {"rows": str(rows_v0), "baseline": str(base_v0)})
    try:
        S.load_corpus(repo, "v99_not_a_generation")
        _record(cases, "gen_unknown_generation_api", set(),
                {"error": "expected RuntimeError unknown_generation"})
    except RuntimeError as exc:
        msg = str(exc)
        tokens = {"unknown_generation"} if msg.startswith("unknown_generation") else {msg}
        _record(cases, "gen_unknown_generation_api", tokens,
                {"error": msg, "no_traceback": True})
    except Exception as exc:  # pragma: no cover
        _record(cases, "gen_unknown_generation_api", {"traceback"},
                {"error": repr(exc)})
    live_after = _live_authority_hashes(repo)
    live_tokens = set()
    if live_after["tool"] == live_before["tool"]:
        live_tokens.add("tool_unchanged")
    if live_after["generation"] == live_before["generation"]:
        live_tokens.add("generation_unchanged")
    if live_after["rows"] == live_before["rows"]:
        live_tokens.add("rows_unchanged")
    _record(cases, "gen_auth_live_bytes_unchanged", live_tokens,
            {"before": live_before, "after": live_after})
    return cases


def run_generation_authority_negatives(repo: Path) -> list:
    """Orchestrate pure/isolated/CLI/selector gen-auth negatives (behavior-identical 30 cases)."""
    fixture_dir = S.GENERATIONS["v1"]["fixture_dir"]
    gen_path = repo / fixture_dir / "GENERATION.json"
    if not gen_path.is_file():
        return [{"case": "gen_auth_setup", "ok": False, "observed": ["generation_receipt_absent"],
                 "preregistered": [], "set_equal": False, "tier": FAST_TIER,
                 "evidence_tier": "preregistered",
                 "detail": {"error": f"missing {gen_path}"}}]
    live_before = _live_authority_hashes(repo)
    good = json.loads(gen_path.read_text())
    cases = []
    cases.extend(run_gen_auth_pure_receipt_cases(repo, good))
    cases.extend(run_gen_auth_isolated_carrier_cases(repo, good))
    cases.extend(run_gen_auth_cli_cases(repo))
    cases.extend(run_gen_auth_selector_and_live_cases(repo, live_before))
    return cases
