"""Negative-case battery and transition proofs for the LANDS-AB diagnostic corpus.

Seam owner: battery / negative-case + transition-proof ORCHESTRATION. Authorized as a
sixth tracked file by claude ruling 1785437631492-081d734a (co_lead's Phase E
no-sixth-file ruling superseded by changed line-count evidence).

Test orchestration sits ABOVE the stack, so this module may import the harness,
sources and reducers; none of them import this module (no reverse edges).

Assertion discipline: every case compares its observed failure-code set for SET
EQUALITY against a PREREGISTERED set. Membership (`expected in observed`) is
subset-blind -- it reports PASS while an unrelated live defect sits in the same
failure set, which is exactly how a real corpus defect survived n1-n8/n10 and was
caught only by the two positive controls (n9, n11).

The transition proofs build a CANDIDATE WORKSPACE of HARDLINKS. Hardlinks, not
symlinks: the fixture builder derives DRY from `Path(__file__).resolve()`, and a
symlinked builder resolves back to the real repo -- which would silently point the
whole run at the real tool. The real tool is NEVER written.

Generation v1 active: workspace materialize includes GENERATION.json via transition builder.
Gen-auth negatives live in lands_ab_diag_gen_auth_negatives (plan v8).
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

REPO_DEFAULT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_DEFAULT))

from calm.llm_computer import lands_ab_diag_corpus_reducers as R  # noqa: E402
from calm.llm_computer import lands_ab_diag_corpus_sources as S  # noqa: E402
from calm.llm_computer.tests.test_lands_ab_diag_corpus_transition import (  # noqa: E402
    _codes, build_candidate_workspace, regenerate_manifest, run_transition_proofs)
from calm.llm_computer.tests.lands_ab_diag_gen_auth_negatives import (  # noqa: E402
    GEN_AUTH_PREREG, run_generation_authority_negatives)

sys.path.insert(0, str(REPO_DEFAULT / "scripts"))
import lands_ab_dry_exec_diag_corpus as H  # noqa: E402

# ---------------------------------------------------------------- preregistration
# Reasoned BEFORE execution from the cure semantics, not transcribed from a run.
# A case whose observed set differs from this is a FINDING to report, never an
# expectation to retro-fit.
PREREG = {
    # (a)/(b) predate every correction in this slice, so they are preregistered like any
    # other case. They express OUTCOME TOKENS rather than failure codes because they
    # assert reducer/mint behavior, not preflight codes -- but the schema is identical,
    # so headline reconciliation can count them.
    "a_baseline_tamper_isolated_to_baseline_ledger": {
        "baseline_ledger_fires_once_on_tampered_row", "prereg_ledger_clean"},
    "b_mint_refuses_existing_baseline": {
        "mint_refused_rc1", "existing_baseline_bytes_unchanged"},
    "c_fixture_builder_hash_mismatch": {"harness_sha_drift", "baseline_source_pin"},
    "d_duplicate_row_id": {"row_id_not_unique", "baseline_rowid_set"},
    "e_missing_required_field": {"required_field_absent"},
    "e2_missing_row_id_classifies_not_raises": {"required_field_absent", "baseline_rowid_set"},
    "f_tier_ledger_crossover": {"tier_ledger_crossover", "ledger_row_count", "ledger_site_count"},
    "n1_alternate_baseline_rejected": {"baseline_path_not_canonical"},
    "n2_null_provenance_field": {"required_field_null_or_bad_type"},
    "n3_wrong_typed_field": {"required_field_null_or_bad_type"},
    "n5_row_residual_overlap": {"row_residual_overlap", "census_unique_site_reconciliation"},
    "n6_ledger_site_count": {"ledger_site_count"},
    "n7a_A0_candidate_only": {"candidate_sha_not_baseline_at_A0"},
    "n7b_provenance_carrier_only": {"provenance_internal_mismatch"},
    "n8_manifest_absent_or_stale": {"manifest_absent"},
    "n9_allowlist_missing_for_step": {"allowlist_missing_for_step"},
    "n10_reference_refresh_bypass_blocked": {"provenance_internal_mismatch"},
    "n11_A1_demotion_positive_control": set(),
    "p1_A5_policy_resolution": set(),
    "p2_A6_policy_resolution": set(),
    "p3_unknown_step_A7_rejected": {"allowlist_missing_for_step"},
    "p4_normalization_register_absent": {"normalization_register_absent"},
}

# NOT preregistered. Corrected AFTER observation and tiered accordingly, so a
# characterization fix can never be counted as an independent prereg claim.
# Ruling: co_lead 1785440127113-32b71379 item 4; claude dispatch 1785440183343-0ad34afb cure 3.
CORRECTED = {
    # Duplicating a residual id also drops unique combined sites 180 -> 179, so the
    # census check MUST co-fire. Identical arithmetic to n5, which was preregistered
    # correctly -- the omission was the author's; the implementation is right.
    "n4_residual_site_not_unique": {"residual_site_not_unique",
                                    "census_unique_site_reconciliation"},
}
FAST_TIER = "fast_pre_subprocess"
WORKSPACE_TIER = "workspace"
STAGE = "pre-subprocess"


REQUIRED_CASE_FIELDS = ("case", "observed", "set_equal", "tier", "evidence_tier", "ok")


def reconcile_case_schema(cases: list) -> list:
    """Every case carries BOTH dimensions and set_equal; both tiers sum to the total.

    Fail-closed by construction: a missing field is a FAULT, never a silently-passing
    default. A `.get(field, True)` here would let an unclassified case report green --
    the same affirmative-green-over-a-subset defect this slice has now found four times.
    """
    faults = []
    for case in cases:
        missing = [f for f in REQUIRED_CASE_FIELDS if f not in case]
        if "preregistered" not in case and "expected_set" not in case:
            missing.append("preregistered|expected_set")
        if missing:
            faults.append({"case": case.get("case", "<unnamed>"), "missing": missing})
    evidence = sum(1 for c in cases
                   if c.get("evidence_tier") in ("preregistered", "post_observation_corrected"))
    execution = sum(1 for c in cases if c.get("tier") in (FAST_TIER, WORKSPACE_TIER))
    if evidence != len(cases):
        faults.append({"dimension": "evidence_tier", "classified": evidence,
                       "total": len(cases)})
    if execution != len(cases):
        faults.append({"dimension": "execution_tier", "classified": execution,
                       "total": len(cases)})
    return faults



def battery_probe(repo: Path, corpus: dict, baseline: dict):
    """Factory: fast-tier probe with subprocess guard (plan v8 ≤150)."""
    real_run = subprocess.run

    def guard(*a, **k):
        raise AssertionError("subprocess spawned during a pre-execution negative case")

    def probe(mutate, step="A0", alt_baseline=None, mutate_baseline=None):
        bad = copy.deepcopy(corpus)
        mutate(bad)
        tmpdir = None
        try:
            if mutate_baseline is not None:
                edited = copy.deepcopy(baseline)
                mutate_baseline(edited)
                tmpdir = Path(tempfile.mkdtemp(prefix="a0basemut_"))
                shutil.copy2(S.rows_path(repo), tmpdir / "ROWS.json")
                (tmpdir / S.BASELINE_NAME).write_text(
                    json.dumps(edited, indent=1, sort_keys=True) + "\n")
                subprocess.run = guard
                try:
                    fails = H.preflight(repo, bad, edited, True, step)
                finally:
                    subprocess.run = real_run
                return {"preflight_failures": fails}
            subprocess.run = guard
            try:
                _, out = H.cmd_accept(repo, bad, step=step, quiet=True,
                                      baseline_path=alt_baseline, preflight_only=True)
            finally:
                subprocess.run = real_run
            return out
        finally:
            if tmpdir:
                shutil.rmtree(tmpdir, ignore_errors=True)
    return probe


def battery_record(cases: list, name: str, observed, extra=None, tier=FAST_TIER):
    """Append one prereg/corrected case entry (plan v8 ≤150)."""
    corrected = name in CORRECTED
    want = CORRECTED[name] if corrected else PREREG[name]
    entry = {"case": name, "observed": sorted(observed), "set_equal": observed == want,
             "stage": STAGE, "subprocesses": 0, "tier": tier,
             "evidence_tier": "post_observation_corrected" if corrected else "preregistered",
             "ok": observed == want}
    entry["expected_set" if corrected else "preregistered"] = sorted(want)
    if extra:
        entry["detail"] = extra
    cases.append(entry)


def run_fast_preregistered_cases(repo: Path, corpus: dict, baseline: dict) -> list:
    """Fast preregistered a..p4 cases (plan v8 ≤150)."""
    cases = []
    probe = battery_probe(repo, corpus, baseline)
    record = lambda name, observed, extra=None, tier=FAST_TIER: battery_record(
        cases, name, observed, extra, tier)

    tampered = copy.deepcopy(baseline)
    victim = corpus["rows"][0]["row_id"]
    tampered["map"][victim] = {"rc": 99, "msg_key": "TAMPERED"}
    observed_map = {r["row_id"]: dict(baseline["map"][r["row_id"]]) for r in corpus["rows"]}
    b_fail = R.check_observed_vs_baseline(corpus["rows"], observed_map, tampered)
    p_fail = R.check_observed_vs_prereg(corpus["rows"], observed_map)
    tokens_a = set()
    if len(b_fail) == 1 and b_fail[0]["row_id"] == victim:
        tokens_a.add("baseline_ledger_fires_once_on_tampered_row")
    if not p_fail:
        tokens_a.add("prereg_ledger_clean")
    record("a_baseline_tamper_isolated_to_baseline_ledger", tokens_a,
           {"baseline_ledger": b_fail, "prereg_ledger_failures": len(p_fail),
            "stage_detail": "reducer"})
    tmp = Path(tempfile.mkdtemp(prefix="a0selftest_"))
    try:
        existing = tmp / S.BASELINE_NAME
        existing.write_text('{"schema":"decoy"}\n')
        before = S.sha256_file(existing)
        rc = H.cmd_mint_baseline(repo, corpus, quiet=True, baseline_path=existing)
        after = S.sha256_file(existing)
        tokens_b = set()
        if rc == 1:
            tokens_b.add("mint_refused_rc1")
        if after == before:
            tokens_b.add("existing_baseline_bytes_unchanged")
        record("b_mint_refuses_existing_baseline", tokens_b,
               {"rc": rc, "sha_before": before, "sha_after": after})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    record("c_fixture_builder_hash_mismatch",
           _codes(probe(lambda c: c["bound_to"].update(harness_sha256="0" * 64))))
    record("d_duplicate_row_id",
           _codes(probe(lambda c: c["rows"][1].update(row_id=c["rows"][0]["row_id"]))))
    record("e_missing_required_field",
           _codes(probe(lambda c: c["rows"][0].pop("observed_rc_at_authoring", None))))
    record("e2_missing_row_id_classifies_not_raises",
           _codes(probe(lambda c: c["rows"][0].pop("row_id", None))))
    record("f_tier_ledger_crossover",
           _codes(probe(lambda c: c["rows"][0].update(
               independent_correctness_eligible=not c["rows"][0]["independent_correctness_eligible"]))))
    alt_dir = Path(tempfile.mkdtemp(prefix="a0altbase_"))
    try:
        alt = alt_dir / S.BASELINE_NAME
        alt.write_text(json.dumps(baseline, indent=1, sort_keys=True) + "\n")
        record("n1_alternate_baseline_rejected", _codes(probe(lambda c: None, alt_baseline=alt)))
    finally:
        shutil.rmtree(alt_dir, ignore_errors=True)
    record("n2_null_provenance_field",
           _codes(probe(lambda c: c["rows"][0].update(source_increment=None))))
    record("n3_wrong_typed_field",
           _codes(probe(lambda c: c["rows"][0].update(expected_rc="2"))))
    record("n4_residual_site_not_unique",
           _codes(probe(lambda c: c["residual_sites"][1].update(site_id=c["residual_sites"][0]["site_id"]))))
    record("n5_row_residual_overlap",
           _codes(probe(lambda c: c["residual_sites"][0].update(site_id=c["rows"][0]["intended_site_id"]))))
    record("n6_ledger_site_count",
           _codes(probe(lambda c: c["ledgers"]["A_equivalence"].update(
               sites=c["ledgers"]["A_equivalence"]["sites"] - 1))))
    record("n7b_provenance_carrier_only",
           _codes(probe(lambda c: None,
                        mutate_baseline=lambda b: b.update(tool_sha256="1" * 64))))
    record("n8_manifest_absent_or_stale",
           _codes(probe(lambda c: c["bound_to"].update(
               base_manifest="artifacts/acc_entropy/_a0_absent_manifest.json"))))
    record("n9_allowlist_missing_for_step", _codes(probe(lambda c: None, step="A9")))
    record("n10_reference_refresh_bypass_blocked",
           _codes(probe(lambda c: c["bound_to"].update(tool_sha256="2" * 64), step="A1")))
    for name, step in (("p1_A5_policy_resolution", "A5"), ("p2_A6_policy_resolution", "A6")):
        out = probe(lambda c: None, step=step)
        record(name, _codes(out),
               {"declared_step": step, "policy_key": R.resolve_allowlist_policy_key(step),
                "verdict": out.get("verdict")})
    record("p3_unknown_step_A7_rejected", _codes(probe(lambda c: None, step="A7")))
    record("p4_normalization_register_absent",
           _codes(probe(lambda c: c.pop("normalizations", None))))
    return cases


def run_negative_battery(repo: Path) -> dict:
    """Fast + workspace + gen-auth orchestration (plan v8 ≤150)."""
    corpus = S.load_corpus(repo)
    canonical = S.canonical_baseline_path(repo)
    baseline = S.load_baseline(canonical)
    cases = run_fast_preregistered_cases(repo, corpus, baseline)
    cases.extend(run_workspace_cases(repo, corpus))
    cases.extend(run_generation_authority_negatives(repo))
    schema_faults = reconcile_case_schema(cases)
    passed = all(c["ok"] for c in cases) and not schema_faults
    fast = [c for c in cases if c.get("tier", FAST_TIER) == FAST_TIER]
    ws = [c for c in cases if c.get("tier") == WORKSPACE_TIER]
    return {"check_id": "A0_NEGATIVE_BATTERY", "cases": cases,
            "headline": {
                "preregistered_cases": sum(1 for c in cases
                                           if c.get("evidence_tier") == "preregistered"),
                "post_observation_corrected_cases": sum(
                    1 for c in cases if c.get("evidence_tier") == "post_observation_corrected"),
                "fast_pre_subprocess_cases": len(fast),
                "workspace_cases": len(ws),
                "total_cases": len(cases),
                "fast_tier_verdict": "PASS" if all(c["ok"] for c in fast) else "FAIL",
                "workspace_tier_verdict": "PASS" if all(c["ok"] for c in ws) else "FAIL"},
            "schema_faults": schema_faults,
            "verdict": "PASS" if passed else "FAIL"}



def run_workspace_cases(repo: Path, corpus: dict) -> list:
    """n7a + n11: changed candidate bytes, REGENERATED manifest, frozen provenance.

    Both are impossible in the fast tier: making the manifest current requires the real
    generator, which is a subprocess, and weakening the fast tier's raising guard to
    admit them would be the forbidden "relax the check to admit the case" move. They
    share ONE workspace and differ only in the declared step, which is exactly the
    demotion claim -- identical inputs, A0 rejects, A1 proceeds.
    """
    out = []
    tmp = Path(tempfile.mkdtemp(prefix="a0wscase_"))
    try:
        ws = tmp / "n7a_n11"
        build_candidate_workspace(repo, ws, "\n# candidate bytes differ from provenance\n",
                                  lineage=False)
        regen = regenerate_manifest(repo, ws, S.load_corpus(ws))
        c = S.load_corpus(ws)
        b = S.load_baseline(S.canonical_baseline_path(ws))
        ident = S.tool_identity(ws, c, b)
        frozen_consistent = (ident["rows_bound_to_tool_sha256"]
                             == ident["baseline_tool_sha256"]
                             != ident["candidate_tool_sha256"])
        want7a = {"candidate_sha_not_baseline_at_A0"}
        got7a = {f["code"] for f in H.preflight(ws, c, b, True, "A0")}
        out.append({"case": "n7a_A0_candidate_only", "tier": WORKSPACE_TIER,
                    "evidence_tier": "preregistered", "preregistered": sorted(want7a),
                    "observed": sorted(got7a), "set_equal": got7a == want7a,
                    "stage": "workspace-preflight", "regen_rc": regen.returncode,
                    "provenance_frozen_and_consistent": frozen_consistent,
                    "tool_identity": ident,
                    "ok": got7a == want7a and frozen_consistent})
        want11 = set()
        got11 = {f["code"] for f in H.preflight(ws, c, b, True, "A1")}
        out.append({"case": "n11_A1_demotion_positive_control", "tier": WORKSPACE_TIER,
                    "evidence_tier": "preregistered", "preregistered": sorted(want11),
                    "observed": sorted(got11), "set_equal": got11 == want11,
                    "stage": "workspace-preflight", "eligible_to_proceed": not got11,
                    "why": "same changed candidate as n7a; ONLY the declared step differs, "
                           "so a retained candidate==provenance equality would fail here",
                    "ok": got11 == want11 and frozen_consistent})
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


def test_negative_battery():
    """pytest entry: the fast battery only. Transition proofs are opt-in (minutes)."""
    report = run_negative_battery(REPO_DEFAULT)
    assert report["verdict"] == "PASS", json.dumps(report, indent=1)


def test_generation_authority_negatives():
    """pytest entry: committed generation-authority N1–N4 + pin mutations (CURE-2)."""
    cases = run_generation_authority_negatives(REPO_DEFAULT)
    failed = [c for c in cases if not c.get("ok")]
    assert not failed, json.dumps(failed, indent=1)
    # Also require every preregistered name was exercised
    got = {c["case"] for c in cases}
    missing = set(GEN_AUTH_PREREG) - got
    assert not missing, f"missing cases: {sorted(missing)}"


if __name__ == "__main__":
    repo = Path(sys.argv[2]).resolve() if len(sys.argv) > 2 else REPO_DEFAULT
    mode = sys.argv[1] if len(sys.argv) > 1 else "negatives"
    out = run_transition_proofs(repo) if mode == "transition" else run_negative_battery(repo)
    print(json.dumps(out, indent=1))
    raise SystemExit(0 if out["verdict"] == "PASS" else 1)
