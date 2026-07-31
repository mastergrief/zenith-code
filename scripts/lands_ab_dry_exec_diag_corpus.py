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


def resolved_mode(step: str) -> str:
    """The declared step IS the mode. A0 is strict; there is no independent flag."""
    return "A0_strict_candidate_equals_provenance" if step == "A0" \
        else "A1plus_candidate_currency_via_manifest"


def preflight(repo: Path, corpus: dict, baseline: dict | None, need_baseline: bool,
              step: str, baseline_path: Path | None = None,
              generation: str | None = None):
    """Every contract check, ordered before dynamic import and before any subprocess.

    Composable standalone: callers may invoke this without measure() to obtain a
    preflight-only terminal (eligible-to-proceed) verdict.
    """
    fails = []
    if baseline_path is not None:
        canonical = S.canonical_baseline_path(repo, generation)
        if baseline_path.resolve() != canonical.resolve():
            fails.append({"code": "baseline_path_not_canonical",
                          "supplied": str(baseline_path), "canonical": str(canonical),
                          "why": "a caller-supplied observation map can carry valid bindings "
                                 "and still bypass the committed baseline's authority"})
    fails += S.preflight_generation(repo, generation, check_live_tool=False)
    fails += R.preflight_schema(corpus)
    fails += R.preflight_normalization_register(corpus)
    fails += R.preflight_step_allowlist(corpus, step)
    fails += S.preflight_identity(repo, corpus, baseline, step)
    fails += S.preflight_manifest_currency(repo, corpus)
    fails += S.preflight_bindings(repo, corpus, baseline, require_baseline=need_baseline,
                                  generation=generation)
    return fails


def measure(repo: Path, corpus: dict, baseline: dict | None, step: str):
    """Run the corpus twice and compute every ledger. Preflight must already have passed."""
    ctx = make_ctx(repo, corpus)
    rows = corpus["rows"]
    first = run_all(ctx, rows)
    second = run_all(ctx, rows)
    census = R.census_from_source(S.tool_source(repo))
    result = {
        "step": step,
        "allowlist_policy_key": R.resolve_allowlist_policy_key(step),
        "resolved_mode": resolved_mode(step),
        "tool_identity": S.tool_identity(repo, corpus, baseline),
        "rows": len(rows),
        "prereg_ledger_failures": R.check_observed_vs_prereg(rows, first),
        "baseline_ledger_failures": R.check_observed_vs_baseline(rows, first, baseline) if baseline else None,
        "determinism_pair_diffs": R.check_determinism(first, second),
        "token_rule": R.check_tokens(rows, first),
        "vacuity": R.check_vacuity(first),
        "structure": R.ast_structure_report(S.tool_source(repo), step, corpus["ast_allowlist"]),
        "census": census,
        "census_identity_ok": census["total"] == corpus["census"]["total"],
    }
    return result, first


# --------------------------------------------------------------------------- commands

def cmd_assert_structure(repo: Path, corpus: dict, step: str,
                         generation: str | None = None):
    report = R.ast_structure_report(S.tool_source(repo), step, corpus["ast_allowlist"])
    report["census"] = R.census_from_source(S.tool_source(repo))
    report["census_matches_corpus"] = report["census"]["total"] == corpus["census"]["total"]
    if not report["census_matches_corpus"]:
        report["verdict"] = "FAIL"
        report["violations"].append({"symbol": "<census>", "reason": "census_drift_vs_corpus"})
    g = generation if generation is not None else S.ACTIVE_GENERATION
    report["selected_generation"] = g
    report["rows_path"] = str(S.rows_path(repo, g))
    report["rows_sha256"] = S.sha256_file(S.rows_path(repo, g))
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


def cmd_accept(repo: Path, corpus: dict, step: str = "A0", quiet: bool = False,
               baseline_path: Path | None = None, preflight_only: bool = False,
               generation: str | None = None):
    """Normal acceptance. The committed baseline is a REQUIRED INPUT; nothing is written.

    baseline_path is NOT a production knob: production callers pass None and the
    canonical repo path is resolved internally. Only the negative battery supplies an
    alternate, and it is rejected as non-canonical -- it can never mint an ACCEPT.
    """
    g = generation if generation is not None else S.ACTIVE_GENERATION
    resolved = baseline_path if baseline_path is not None else S.canonical_baseline_path(repo, g)
    baseline = S.load_baseline(resolved)
    fails = preflight(repo, corpus, baseline, need_baseline=True, step=step,
                      baseline_path=baseline_path, generation=g)
    # Accept-path live tool pin (N3): same validate_generation_receipt seam with
    # check_live_tool=True. preflight_only probes stay free of generation_tool_mismatch
    # (A1 dirty-candidate amendment). No second semantics body.
    if (not preflight_only) and S.GENERATIONS.get(g, {}).get("has_generation_json"):
        try:
            receipt = S.load_generation_receipt(repo, g)
            tool_fails = S.validate_generation_receipt(
                repo, g, receipt, check_live_tool=True)
            # Only append the live-tool code; pin/path codes already covered by preflight.
            for f in tool_fails:
                if f.get("code") == "generation_tool_mismatch":
                    fails.append(f)
        except RuntimeError as e:
            msg = str(e)
            code = ("generation_receipt_absent" if "absent" in msg
                    else "generation_receipt_mismatch")
            if not any(x.get("code") == code for x in fails):
                fails.append({"code": code, "error": msg})
    if fails:
        out = {"verdict": f"{step}_PREFLIGHT_FAIL", "step": step,
               "allowlist_policy_key": R.resolve_allowlist_policy_key(step),
               "resolved_mode": resolved_mode(step), "preflight_failures": fails,
               "subprocesses_spawned": 0,
               "note": "schema/ledger/identity/currency contract failed before dynamic import and before the first subprocess"}
        if not quiet:
            print(json.dumps(out, indent=1))
        return 1, out
    if preflight_only:
        out = {"verdict": f"{step}_PREFLIGHT_OK", "step": step,
               "allowlist_policy_key": R.resolve_allowlist_policy_key(step),
               "resolved_mode": resolved_mode(step), "preflight_failures": [],
               "subprocesses_spawned": 0,
               "tool_identity": S.tool_identity(repo, corpus, baseline),
               "note": "eligible to proceed to corpus comparison; no measurement performed"}
        if not quiet:
            print(json.dumps(out, indent=1))
        return 0, out
    result, _ = measure(repo, corpus, baseline, step)
    result["preflight_failures"] = []
    result["baseline_path"] = str(resolved)
    result["baseline_sha256"] = S.sha256_file(resolved)
    g = generation if generation is not None else S.ACTIVE_GENERATION
    result["active_generation"] = g
    result["selected_generation"] = g
    result["fixture_dir"] = S.GENERATIONS[g]["fixture_dir"]
    result["rows_sha256"] = S.sha256_file(S.rows_path(repo, g))
    try:
        result["generation_receipt_sha256"] = S.sha256_file(
            repo / S.GENERATIONS[g]["fixture_dir"] / "GENERATION.json"
        ) if S.GENERATIONS[g].get("has_generation_json") else None
    except Exception:
        result["generation_receipt_sha256"] = None
    result["verdict"] = f"{step}_ACCEPT" if R.acceptance_verdict(result, True) else f"{step}_FAIL"
    if not quiet:
        print(json.dumps(result, indent=1))
    return (0 if result["verdict"] == f"{step}_ACCEPT" else 1), result


def cmd_mint_baseline(repo: Path, corpus: dict, quiet: bool = False,
                      baseline_path: Path | None = None,
                      generation: str | None = None):
    """Creation-only baseline mint. Refuses if the target exists; never rewrites.

    Minting is an A0-only operation: a baseline captured against anything other than
    the provenance tool would not be a HEAD-9f471b3 reference at all.
    """
    g = generation if generation is not None else S.ACTIVE_GENERATION
    resolved = baseline_path if baseline_path is not None else S.canonical_baseline_path(repo, g)
    if resolved.exists():
        if not quiet:
            print(json.dumps({"verdict": "A0_MINT_REFUSED",
                              "code": "baseline_exists_refuse",
                              "reason": "baseline already exists; the frozen baseline is immutable",
                              "path": str(resolved),
                              "existing_sha256": S.sha256_file(resolved)}, indent=1))
        return 1
    fails = preflight(repo, corpus, None, need_baseline=False, step="A0", generation=g)
    if fails:
        print(json.dumps({"verdict": "A0_PREFLIGHT_FAIL", "preflight_failures": fails,
                          "subprocesses_spawned": 0}, indent=1))
        return 1
    result, first = measure(repo, corpus, None, "A0")
    if not R.acceptance_verdict(result, False):
        result["verdict"] = "A0_FAIL"
        print(json.dumps(result, indent=1))
        return 1
    bound = corpus["bound_to"]
    meta = S.generation_meta(repo, g)
    payload = {"schema": meta["baseline_schema"], "head": meta["baseline_head"],
               "tool_sha256": bound["tool_sha256"], "harness_sha256": bound["harness_sha256"],
               "rows_sha256": S.sha256_file(S.rows_path(repo, g)),
               "generated_from_rows": corpus["generated_from"], "map": first}
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    json.loads(text)
    fd = os.open(str(resolved), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    with os.fdopen(fd, "w") as handle:
        handle.write(text)
    result["verdict"] = "A0_BASELINE_MINTED"
    result["baseline_written"] = str(resolved)
    result["baseline_sha256"] = S.sha256_file(resolved)
    if not quiet:
        print(json.dumps(result, indent=1))
    return 0


def cmd_self_test(repo: Path):
    """Delegate to the battery module.

    Seam: battery / negative-case + transition-proof orchestration is owned by
    calm/llm_computer/tests/test_lands_ab_diag_corpus_battery.py, which sits ABOVE
    this harness in the import order. This entry stays thin on purpose.
    """
    from calm.llm_computer.tests import test_lands_ab_diag_corpus_battery as B
    report = B.run_negative_battery(repo)
    print(json.dumps(report, indent=1))
    return 0 if report["verdict"] == "PASS" else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="LANDS-AB dry-exec diagnostic-class corpus (A0)")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--assert-structure", action="store_true")
    parser.add_argument("--step", default="A0")
    parser.add_argument("--accept", action="store_true")
    parser.add_argument("--mint-baseline", action="store_true")
    parser.add_argument("--self-test-negatives", action="store_true")
    parser.add_argument("--preflight-only", action="store_true",
                        help="stop after the contract checks; emit an eligible-to-proceed verdict")
    parser.add_argument("--generation", default=None,
                        help="fixture generation (default: ACTIVE_GENERATION)")
    args = parser.parse_args(argv)
    repo = Path(args.repo_root).resolve()
    gen = args.generation
    # C1: classify ALL generation/ROWS load failures at ingress (single semantics body
    # in S.classify_corpus_load_error). Never traceback on damaged/absent carriers.
    try:
        corpus = S.load_corpus(repo, gen)
    except Exception as exc:
        fail = S.classify_corpus_load_error(exc)
        if fail is not None:
            if gen is not None and "generation" not in fail:
                fail = dict(fail, generation=gen)
            out = {"verdict": "PREFLIGHT_FAIL",
                   "preflight_failures": [fail],
                   "subprocesses_spawned": 0}
            print(json.dumps(out, indent=1))
            return 2
        raise
    # The committed baseline is resolved INTERNALLY for every production command.
    # There is deliberately no --baseline-path: an alternate map is reachable only
    # from the negative battery, which cannot emit an ACCEPT.
    if args.assert_structure:
        return cmd_assert_structure(repo, corpus, args.step, generation=gen)
    if args.self_test_negatives:
        return cmd_self_test(repo)
    if args.mint_baseline:
        return cmd_mint_baseline(repo, corpus, generation=gen)
    if args.accept:
        return cmd_accept(repo, corpus, step=args.step, preflight_only=args.preflight_only,
                          generation=gen)[0]
    parser.error("choose --assert-structure, --accept, --mint-baseline or --self-test-negatives")


if __name__ == "__main__":
    raise SystemExit(main())
