#!/usr/bin/env python3
"""LANDS-AB dry-exec diagnostic-class corpus harness (A0) — thin orchestrator.

Executes the frozen ROWS.json corpus against scripts/lands_ab_packet_dry_exec.py and
checks the A0 acceptance rules. Also provides --assert-structure: an AST allowlist
report used by the A1..A5 seam steps.

Acceptance compares live observations against TWO independent references, reported as
SEPARATE failure ledgers:
  * prereg ledger   -- live vs ROWS.json expected_{rc,class_token,msg_key}
  * baseline ledger -- live vs the committed BASELINE map captured at HEAD 9f471b3
The baseline is an INPUT to normal acceptance. It is never written by --accept; it is
minted once, creation-only, by the separate --mint-baseline path.

Every schema, ledger and source-pin check runs BEFORE the dynamic import and BEFORE
the first subprocess, so out-of-contract bytes fail closed without spending a run.

This file owns ONLY: CLI, fixture IO, the subprocess execution loop, and receipt
emission. Pure gate/status reducers live in calm/llm_computer/lands_ab_diag_corpus_
reducers.py; source identity, artifact contracts and the single dynamic-import facade
live in calm/llm_computer/lands_ab_diag_corpus_sources.py.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from calm.llm_computer import lands_ab_diag_corpus_reducers as R  # noqa: E402
from calm.llm_computer import lands_ab_diag_corpus_sources as S  # noqa: E402

COMMIT = "0123456789abcdef0123456789abcdef01234567"
FIXTURE_PREFIX = "artifacts/acc_entropy/_tmp_host_man_"


# --------------------------------------------------------------------------- fixture IO

def write_manifest_fixture(repo: Path, base_manifest: dict, spec_manifest: dict, fixtures: list) -> str:
    rel = f"{FIXTURE_PREFIX}{uuid.uuid4().hex[:12]}.json"
    path = repo / rel
    fixtures.append(path)
    if "raw" in spec_manifest:
        path.write_text(spec_manifest["raw"])
    else:
        mutated = R.apply_ops(copy.deepcopy(base_manifest), spec_manifest["ops"])
        path.write_text(json.dumps(mutated, indent=1) + "\n")
    return rel


def build_case(ctx, spec, tmp: Path, fixtures: list):
    """Return (packet_path, manifest_arg) for one row spec."""
    base = ctx["mod"]._packet_for_dry(tmp, ctx["manifest"], ctx["mod"]._sha(ctx["manifest"]), COMMIT)
    packet = json.loads(base.read_text())
    manifest_arg = ctx["manifest_rel"]
    if spec.get("manifest") is not None:
        rel = write_manifest_fixture(ctx["repo"], ctx["base_manifest"], spec["manifest"], fixtures)
        packet["science_source_manifest_path"] = rel
        packet["science_source_manifest_sha256"] = S.sha256_file(ctx["repo"] / rel)
        manifest_arg = rel
    cli = spec.get("cli") or {}
    if cli.get("mode") == "outside_repo_file":
        outside = tmp / cli["name"]
        outside.write_text(cli.get("content", "{}\n"))
        manifest_arg = str(outside)
    elif cli.get("mode") == "absent_in_repo":
        manifest_arg = cli["rel"]
    R.apply_ops(packet, spec.get("ops"))
    pkt_path = tmp / f"pkt_{uuid.uuid4().hex[:8]}.json"
    if spec.get("packet_raw") is not None:
        pkt_path.write_text(spec["packet_raw"])
    else:
        pkt_path.write_text(json.dumps(packet) + "\n")
    return pkt_path, manifest_arg


# --------------------------------------------------------------------------- execution

def run_row(ctx, spec, tmp: Path, fixtures: list):
    pkt_path, manifest_arg = build_case(ctx, spec, tmp, fixtures)
    argv = [
        sys.executable, str(ctx["mod"].DRY),
        "--packet", str(pkt_path),
        "--verify-source-manifest", str(manifest_arg),
        "--expected-source-commit", COMMIT,
        "--repo-root", str(ctx["repo"]),
    ] + list(spec.get("extra_args") or [])
    proc = subprocess.run(argv, cwd=str(ctx["repo"]), capture_output=True, text=True)
    lines = (proc.stderr or "").strip().splitlines()
    return proc.returncode, (R.normalize(lines[0], str(ctx["repo"])) if lines else "")


def run_all(ctx, rows):
    observed = {}
    tmp = Path(tempfile.mkdtemp(prefix="a0corpus_"))
    fixtures: list = []
    try:
        for row in rows:
            rc, msg = run_row(ctx, row["mutation_spec"], tmp, fixtures)
            observed[row["row_id"]] = {"rc": rc, "msg_key": msg}
    finally:
        for path in fixtures:
            path.unlink(missing_ok=True)
        shutil.rmtree(tmp, ignore_errors=True)
    return observed


def make_ctx(repo: Path, corpus: dict):
    mod = S.import_fixture_builder(repo, corpus["bound_to"]["harness_sha256"])
    manifest = repo / corpus["bound_to"]["base_manifest"]
    return {"repo": repo, "mod": mod, "manifest": manifest,
            "manifest_rel": manifest.relative_to(repo).as_posix(),
            "base_manifest": json.loads(manifest.read_text())}


def preflight(repo: Path, corpus: dict, baseline: dict | None, need_baseline: bool):
    """Every contract check, ordered before dynamic import and before any subprocess."""
    return R.preflight_schema(corpus) + S.preflight_bindings(
        repo, corpus, baseline, require_baseline=need_baseline)


def measure(repo: Path, corpus: dict, baseline: dict | None):
    """Run the corpus twice and compute every ledger. Preflight must already have passed."""
    ctx = make_ctx(repo, corpus)
    rows = corpus["rows"]
    first = run_all(ctx, rows)
    second = run_all(ctx, rows)
    census = R.census_from_source(S.tool_source(repo))
    result = {
        "rows": len(rows),
        "prereg_ledger_failures": R.check_observed_vs_prereg(rows, first),
        "baseline_ledger_failures": R.check_observed_vs_baseline(rows, first, baseline) if baseline else None,
        "determinism_pair_diffs": R.check_determinism(first, second),
        "token_rule": R.check_tokens(rows, first),
        "vacuity": R.check_vacuity(first),
        "structure": R.ast_structure_report(S.tool_source(repo), "A0", corpus["ast_allowlist"]),
        "census": census,
        "census_identity_ok": census["total"] == corpus["census"]["total"],
    }
    return result, first


# --------------------------------------------------------------------------- commands

def cmd_assert_structure(repo: Path, corpus: dict, step: str):
    report = R.ast_structure_report(S.tool_source(repo), step, corpus["ast_allowlist"])
    report["census"] = R.census_from_source(S.tool_source(repo))
    report["census_matches_corpus"] = report["census"]["total"] == corpus["census"]["total"]
    if not report["census_matches_corpus"]:
        report["verdict"] = "FAIL"
        report["violations"].append({"symbol": "<census>", "reason": "census_drift_vs_corpus"})
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


def cmd_accept(repo: Path, corpus: dict, baseline_path: Path, quiet: bool = False):
    """Normal acceptance. The committed baseline is a REQUIRED INPUT; nothing is written."""
    baseline = S.load_baseline(baseline_path)
    fails = preflight(repo, corpus, baseline, need_baseline=True)
    if fails:
        out = {"verdict": "A0_PREFLIGHT_FAIL", "preflight_failures": fails,
               "subprocesses_spawned": 0,
               "note": "schema/ledger/source-pin contract failed before dynamic import and before the first subprocess"}
        if not quiet:
            print(json.dumps(out, indent=1))
        return 1, out
    result, _ = measure(repo, corpus, baseline)
    result["preflight_failures"] = []
    result["baseline_path"] = str(baseline_path)
    result["baseline_sha256"] = S.sha256_file(baseline_path)
    result["verdict"] = "A0_ACCEPT" if R.acceptance_verdict(result, True) else "A0_FAIL"
    if not quiet:
        print(json.dumps(result, indent=1))
    return (0 if result["verdict"] == "A0_ACCEPT" else 1), result


def cmd_mint_baseline(repo: Path, corpus: dict, baseline_path: Path, quiet: bool = False):
    """Creation-only baseline mint. Refuses if the target exists; never rewrites."""
    if baseline_path.exists():
        if not quiet:
            print(json.dumps({"verdict": "A0_MINT_REFUSED",
                              "reason": "baseline already exists; the frozen baseline is immutable",
                              "path": str(baseline_path),
                              "existing_sha256": S.sha256_file(baseline_path)}, indent=1))
        return 1
    fails = preflight(repo, corpus, None, need_baseline=False)
    if fails:
        print(json.dumps({"verdict": "A0_PREFLIGHT_FAIL", "preflight_failures": fails,
                          "subprocesses_spawned": 0}, indent=1))
        return 1
    result, first = measure(repo, corpus, None)
    if not R.acceptance_verdict(result, False):
        result["verdict"] = "A0_FAIL"
        print(json.dumps(result, indent=1))
        return 1
    bound = corpus["bound_to"]
    payload = {"schema": S.BASELINE_SCHEMA, "head": S.BASELINE_HEAD,
               "tool_sha256": bound["tool_sha256"], "harness_sha256": bound["harness_sha256"],
               "rows_sha256": S.sha256_file(S.rows_path(repo)),
               "generated_from_rows": corpus["generated_from"], "map": first}
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    json.loads(text)
    fd = os.open(str(baseline_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    result["verdict"] = "A0_BASELINE_MINTED"
    result["baseline_written"] = str(baseline_path.relative_to(repo))
    result["baseline_sha256"] = S.sha256_file(baseline_path)
    print(json.dumps(result, indent=1))
    return 0


def cmd_self_test(repo: Path, corpus: dict, baseline_path: Path):
    """Negative-case battery. No subprocess is legitimate here; spawning is a failure."""
    baseline = S.load_baseline(baseline_path)
    cases, real_run = [], subprocess.run

    def record(name, ok, detail):
        cases.append({"case": name, "ok": bool(ok), "detail": detail})

    def guard(*a, **k):
        raise AssertionError("subprocess spawned during a pre-execution negative case")

    def preflight_codes(mutate):
        bad_corpus = copy.deepcopy(corpus)
        mutate(bad_corpus)
        subprocess.run = guard
        try:
            _, out = cmd_accept(repo, bad_corpus, baseline_path, quiet=True)
        finally:
            subprocess.run = real_run
        return {f["code"] for f in out.get("preflight_failures", [])}, out

    # (a) baseline map tamper -> BASELINE ledger specifically
    if baseline:
        tampered = copy.deepcopy(baseline)
        victim = corpus["rows"][0]["row_id"]
        tampered["map"][victim] = {"rc": 99, "msg_key": "TAMPERED"}
        observed = {r["row_id"]: dict(baseline["map"][r["row_id"]]) for r in corpus["rows"]}
        b_fail = R.check_observed_vs_baseline(corpus["rows"], observed, tampered)
        p_fail = R.check_observed_vs_prereg(corpus["rows"], observed)
        record("a_baseline_tamper_isolated_to_baseline_ledger",
               len(b_fail) == 1 and b_fail[0]["row_id"] == victim and not p_fail,
               {"baseline_ledger": b_fail, "prereg_ledger_failures": len(p_fail)})
    # (b) mint refuses over an existing baseline, bytes unchanged
    tmp = Path(tempfile.mkdtemp(prefix="a0selftest_"))
    try:
        existing = tmp / S.BASELINE_NAME
        existing.write_text('{"schema":"decoy"}\n')
        before = S.sha256_file(existing)
        rc = cmd_mint_baseline(repo, corpus, existing, quiet=True)
        record("b_mint_refuses_existing_baseline",
               rc == 1 and S.sha256_file(existing) == before,
               {"rc": rc, "sha_before": before, "sha_after": S.sha256_file(existing)})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    # (c) fixture-builder hash mismatch -> fails before import/subprocess
    codes, _ = preflight_codes(lambda c: c["bound_to"].update(harness_sha256="0" * 64))
    record("c_fixture_builder_hash_mismatch", "harness_sha_drift" in codes, sorted(codes))
    # (d) duplicate row_id
    codes, _ = preflight_codes(lambda c: c["rows"][1].update(row_id=c["rows"][0]["row_id"]))
    record("d_duplicate_row_id", "row_id_not_unique" in codes, sorted(codes))
    # (e) missing required field -- pops observed_rc_at_authoring, i.e. exactly the
    # defect class the projection actually shipped and this preflight caught pre-commit
    codes, _ = preflight_codes(lambda c: c["rows"][0].pop("observed_rc_at_authoring", None))
    record("e_missing_required_field", "required_field_absent" in codes, sorted(codes))
    # (e2) popping the IDENTIFIER field must classify, not traceback
    codes, _ = preflight_codes(lambda c: c["rows"][0].pop("row_id", None))
    record("e2_missing_row_id_classifies_not_raises", "required_field_absent" in codes, sorted(codes))
    # (f) tier / Ledger-B crossover
    codes, _ = preflight_codes(lambda c: c["rows"][0].update(independent_correctness_eligible=not c["rows"][0]["independent_correctness_eligible"]))
    record("f_tier_ledger_crossover", "tier_ledger_crossover" in codes, sorted(codes))

    passed = all(c["ok"] for c in cases)
    print(json.dumps({"check_id": "A0_NEGATIVE_BATTERY", "cases": cases,
                      "verdict": "PASS" if passed else "FAIL"}, indent=1))
    return 0 if passed else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="LANDS-AB dry-exec diagnostic-class corpus (A0)")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--assert-structure", action="store_true")
    parser.add_argument("--step", default="A0")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--mint-baseline", action="store_true")
    parser.add_argument("--self-test-negatives", action="store_true")
    parser.add_argument("--baseline-path", default=None,
                        help="override the committed baseline (negative-case replay only)")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    corpus = S.load_corpus(repo)
    baseline_path = Path(args.baseline_path) if args.baseline_path else S.canonical_baseline_path(repo)
    if args.assert_structure:
        return cmd_assert_structure(repo, corpus, args.step)
    if args.self_test_negatives:
        return cmd_self_test(repo, corpus, baseline_path)
    if args.mint_baseline:
        return cmd_mint_baseline(repo, corpus, baseline_path)
    if args.accept:
        return cmd_accept(repo, corpus, baseline_path)[0]
    parser.error("choose --assert-structure, --accept, --mint-baseline or --self-test-negatives")


if __name__ == "__main__":
    raise SystemExit(main())
