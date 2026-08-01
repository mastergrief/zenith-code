"""PLAN_v6 Phase A: science source manifest + packet dry-exec hostiles (v3 faithful)."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
GEN = REPO / "scripts" / "lands_ab_science_source_manifest.py"
DRY = REPO / "scripts" / "lands_ab_packet_dry_exec.py"
PACKET_V6 = (
    REPO
    / "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_LANDS_AB_EVAL_launch_packet_v6.json"
)

MANDATORY_SUBSTR = (
    "lands_ab_eval_cuda_sites.py",
    "lands_ab_eval_twin_apply.py",
    "lands_ab_eval_oracle_sites.py",
    "lands_ab_eval_site_measurement.py",
    "lands_ab_eval_production_post_state.py",
    "lands_ab_eval_production_binding.py",
    "lands_ab_eval_schema.py",
    "lands_ab_eval_branch_reducer.py",
    "scripts/lands_ab_eval_run.py",
    "scripts/lands_ab_plan_v4_characterization.py",
    "scripts/lands_ab_packet_dry_exec.py",
    "scripts/lands_ab_science_source_manifest.py",
    # H1 formal CUDA execution surfaces
    "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py",
    "bin/watch-wrap",
    "test_hrm_text_158_lands_ab_eval_gpu_live.py",
)

GEN_ARGV_BASE = [
    sys.executable,
    str(GEN),
    "--entry",
    "scripts/lands_ab_eval_run.py",
    "--entry",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py",
    "--entry",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_site_measurement.py",
    "--entry",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_measurement.py",
    "--entry",
    "scripts/lands_ab_packet_dry_exec.py",
    "--root-package",
    "calm.hrm_text_158.native_full_stack",
    "--also-include",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_twin_apply.py",
    "--also-include",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_schema.py",
    "--also-include",
    "calm/hrm_text_158/native_full_stack/lands_ab_eval_branch_reducer.py",
    "--repo-root",
    str(REPO),
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _run_gen(out: Path, *, extra=None) -> subprocess.CompletedProcess:
    cmd = list(GEN_ARGV_BASE) + ["--out", str(out)]
    if extra:
        cmd.extend(extra)
    return subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)


def _load_man(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def _packet_for_dry(tmp: Path, man_path: Path, man_sha: str, commit: str) -> Path:
    """Faithful positive: packet_v6 row mechanics + binding + I-series live-field rebinds."""
    base = json.loads(PACKET_V6.read_text(encoding="utf-8"))
    pkt = copy.deepcopy(base)
    pkt["schema"] = "LANDS_AB_EVAL_launch_packet/v8_dry_fixture"
    pkt["packet_revision"] = "v8"
    pkt["operative_packet_revision"] = "v8"  # L1 structured authority
    pkt["path"] = (
        "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
        "LANDS_AB_EVAL_launch_packet_v8.json"
    )
    pkt["science_source_manifest_path"] = man_path.relative_to(REPO).as_posix()
    pkt["science_source_manifest_sha256"] = man_sha
    pkt["source_commit_sha"] = commit
    pkt["generator_script_path"] = "scripts/lands_ab_science_source_manifest.py"
    pkt["generator_script_sha256"] = _sha(GEN)
    pkt["dry_exec_tool_path"] = "scripts/lands_ab_packet_dry_exec.py"
    pkt["dry_exec_tool_sha256"] = _sha(DRY)
    # I5: distinct operative plan authority
    pkt["operative_plan_id"] = (
        "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
        "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v6.json"
    )
    pkt["operative_plan_sha256"] = (
        "e421aecdf1cc4b9a94d118a0563e7d8ac8516f97da998d38af3b8c60ac88a41c"
    )
    cc = dict(pkt.get("claim_ceiling") or {})
    for k in (
        "LANDS_AB",
        "science_claim",
        "equivalent_minted",
        "full_sub2_runtime_ready_for_science",
    ):
        cc[k] = False
    pkt["claim_ceiling"] = cc
    LIVE_ORDER = [
        "BR-LANDS-AB-SCOPE-CREEP-STOP",
        "BR-LANDS-AB-FIXTURE-CONTRACT-FAIL",
        "BR-LANDS-AB-VACUOUS",
        "BR-LANDS-AB-DIVERGENT-EVENT",
        "BR-LANDS-AB-DIVERGENT-APPLY",
        "BR-LANDS-AB-DIVERGENT-ORACLE-LIVE",
        "BR-LANDS-AB-EQUIVALENT",
    ]
    pkt["PRIORITY_ORDER"] = list(LIVE_ORDER)
    pkt["branch_ids"] = list(LIVE_ORDER)
    pkt["terminal_branch_allow_set"] = list(LIVE_ORDER)
    pkt["procedural_stop_reasons"] = [
        "protocol fail",
        "pin mismatch",
        "unlisted branch outside terminal_branch_allow_set",
    ]
    ex = dict(pkt.get("executor") or {})
    ex["role"] = "claude_as_test_operator"
    ex.setdefault(
        "forbidden_for_plan_dev",
        ["formal 7-row matrix execution", "terminal receipt mint under FORMAL_RUNTIME_CREATE"],
    )
    ex["one_terminal_receipt"] = True
    pkt["executor"] = ex

    man = _load_man(man_path)
    man_map = {e["path"]: e["sha256"] for e in man["entries"]}
    pkt["preflight_checklist"] = [
        "HEAD must equal source_commit_sha " + commit,
        "origin/feature/hrm-text-1.58 equals HEAD",
        (
            "re-hash operative CONSUMER_ADAPT_RERUN PLAN_v6 -> "
            + pkt["operative_plan_sha256"]
        ),
        (
            "re-hash historical EVAL PLAN_v6 lineage pin only (non-operative) -> "
            "93645d31ea8a0cb0f89cfc4f1aedd38190a47f18433b6bc67c9b5d98da7093c5"
        ),
        "re-hash science_source_manifest -> " + man_sha,
        "re-hash runner/harness pins against bound manifest entries",
        "digest/pin mismatch at preflight or mid-run recheck = STOP",
    ]
    sc = dict(pkt.get("self_check") or {})
    sc.pop("pinned_to_a258f314", None)
    sc["pinned_to_source_commit"] = True
    sc["expected_branch_locked_structural_null"] = False
    sc["expected_branch_classifier_determined"] = True
    sc["packet_v6_dead_lineage_referenced"] = True
    pkt["self_check"] = sc
    # L1: non-current packet_vN/vN tokens only allowed inside DEAD-marked subtrees —
    # alias prose must not name prior revisions (structured authority only).
    pkt["alias_note"] = (
        "Prior packet revisions are DEAD immutable lineage "
        "(see pins.*_dead_lineage with do_not_activate). "
        "operative_packet_revision=v8 is the sole authority "
        "for task formal run under source_commit_sha=" + commit + "."
    )
    stops = list(pkt.get("stop_conditions") or [])
    cleaned = []
    for s in stops:
        sl = str(s).lower()
        if "terminal branch !=" in sl and "fixture-contract-fail" in sl:
            continue
        if "terminal_branch !=" in sl and "fixture" in sl:
            continue
        cleaned.append(s)
    cleaned.append(
        "procedural DEVIATION only: unlisted branch / protocol fail / pin mismatch "
        "(preregistered branch_ids including EQUIVALENT are valid terminals)"
    )
    pkt["stop_conditions"] = cleaned

    pins = dict(pkt.get("pins") or {})
    pins["repo_HEAD"] = commit
    # O2: rebind path→sha maps to current disk/manifest so known_refs are exact-value coherent.
    def _rebind_path_sha_map(m: object) -> dict:
        out = {}
        if not isinstance(m, dict):
            return out
        for rel, _old in m.items():
            rel_n = str(rel).replace('\\', "/").lstrip("./")
            if rel_n in man_map:
                out[rel_n] = man_map[rel_n]
                continue
            fp = REPO / rel_n
            if fp.is_file():
                out[rel_n] = _sha(fp)
            else:
                continue
        return out

    if "runner_and_harness_shas" in pins and isinstance(pins["runner_and_harness_shas"], dict):
        pins["runner_and_harness_shas"] = _rebind_path_sha_map(pins["runner_and_harness_shas"])
    if "default_source_pins_for_consumer" in pins and isinstance(
        pins["default_source_pins_for_consumer"], dict
    ):
        pins["default_source_pins_for_consumer"] = _rebind_path_sha_map(
            pins["default_source_pins_for_consumer"]
        )
    if "PLAN_v6" in pins:
        pins["historical_eval_PLAN_v6"] = pins.pop("PLAN_v6")
        if isinstance(pins["historical_eval_PLAN_v6"], dict):
            pins["historical_eval_PLAN_v6"] = dict(pins["historical_eval_PLAN_v6"])
            pins["historical_eval_PLAN_v6"]["dead_lineage"] = True
            pins["historical_eval_PLAN_v6"]["do_not_activate"] = True
    rhs_new = {}
    for rel in sorted(man_map):
        if not (
            rel.startswith("scripts/")
            or rel.startswith("bin/")
            or rel.startswith("calm/hrm_text_158/native_full_stack/lands_ab_")
            or "lands_ab_eval" in rel
            or rel.endswith("watch-wrap")
        ):
            continue
        rhs_new[rel] = man_map[rel]
    for rel in (
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_site_measurement.py",
        "calm/hrm_text_158/native_full_stack/lands_ab_eval_fixture_source.py",
        "bin/watch-wrap",
        "scripts/lands_ab_eval_run.py",
        "scripts/lands_ab_packet_dry_exec.py",
    ):
        if rel in man_map:
            rhs_new[rel] = man_map[rel]
    pins["runner_and_harness_shas"] = {
        k: v for k, v in rhs_new.items() if "dry_exec_v16" not in k and k in man_map
    }
    # J1/O1: packet_vN lineage pins must dna + independent status + path/sha256 identity
    for pk, pv in list(pins.items()):
        if "packet_v" in str(pk).lower() and isinstance(pv, dict):
            pins[pk] = dict(pv)
            pins[pk]["do_not_activate"] = True
            pins[pk]["dead_lineage"] = True
            pins[pk]["status"] = "DEAD"
            # ensure identity fields if missing but path-like data present
            if not pins[pk].get("path") and isinstance(pins[pk].get("packet_path"), str):
                pins[pk]["path"] = pins[pk]["packet_path"]
    for name, rel in (
        ("TSA", "calm/hrm_text_158/native_full_stack/trainer_sub2_authority.py"),
        ("BDL", "calm/hrm_text_158/native_full_stack/bounded_delta_learner.py"),
    ):
        if name in pins and isinstance(pins[name], dict):
            if rel in man_map:
                pins[name] = {
                    "path": rel,
                    "sha256": man_map[rel],
                    "read_only": True,
                    "rehash_pre_and_post": True,
                }
            else:
                pins.pop(name, None)
    # historical EVAL plan pin: full O1 identity
    if "historical_eval_PLAN_v6" in pins and isinstance(pins["historical_eval_PLAN_v6"], dict):
        pins["historical_eval_PLAN_v6"] = dict(pins["historical_eval_PLAN_v6"])
        pins["historical_eval_PLAN_v6"]["do_not_activate"] = True
        pins["historical_eval_PLAN_v6"]["dead_lineage"] = True
        pins["historical_eval_PLAN_v6"]["status"] = "DEAD"
        if not pins["historical_eval_PLAN_v6"].get("path"):
            pins["historical_eval_PLAN_v6"]["path"] = (
                "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
                "LANDS_AB_EVAL_PLAN_v6.json"
            )
        if not pins["historical_eval_PLAN_v6"].get("sha256"):
            pins["historical_eval_PLAN_v6"]["sha256"] = (
                "93645d31ea8a0cb0f89cfc4f1aedd38190a47f18433b6bc67c9b5d98da7093c5"
            )

    # O1/O2: fold lineage_policy into identity-bound DEAD carriers (dna+status+path+sha)
    lp = pkt.pop("lineage_policy", None)
    if isinstance(lp, dict):
        for lk, lv in lp.items():
            if isinstance(lv, dict):
                entry = dict(lv)
                entry["do_not_activate"] = True
                entry["dead_lineage"] = True
                entry["status"] = str(entry.get("status") or "DEAD")
                pins[f"lineage_policy_{lk}"] = entry
            else:
                # non-dict: attach under a known lineage path if available
                continue
    # pick a bound identity from any packet_v* pin for freeform historical notes
    bind_path, bind_sha = None, None
    for pk, pv in pins.items():
        if (
            isinstance(pv, dict)
            and "packet_v" in str(pk).lower()
            and isinstance(pv.get("path"), str)
            and isinstance(pv.get("sha256"), str)
        ):
            bind_path, bind_sha = pv["path"], pv["sha256"]
            break
    if bind_path is None:
        bind_path = (
            "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
            "LANDS_AB_EVAL_launch_packet_v1.json"
        )
        bind_sha = "a9a6ec6861e8c5406c9ebc4f59a4c8bf7c134c1cd6cd02664b185fc47cdfd7bc"

    pkt["pins"] = pins

    auth = dict(pkt.get("authority_chain") or {})
    auth["operative_plan_id"] = pkt["operative_plan_id"]
    auth["operative_plan_sha256"] = pkt["operative_plan_sha256"]
    auth["plan_path"] = pkt["operative_plan_id"]
    auth["plan_sha256"] = pkt["operative_plan_sha256"]
    auth["head_a"] = commit
    for k in ("commit_sha", "source_commit_sha", "repo_HEAD", "head", "HEAD"):
        if k in auth:
            auth[k] = commit
    # O1: nest non-current packet path/sha leaves under identity-bound DEAD object
    dead_auth_refs = {
        "do_not_activate": True,
        "dead_lineage": True,
        "status": "DEAD",
        "path": bind_path,
        "sha256": bind_sha,
    }
    for k in list(auth.keys()):
        kl = str(k).lower()
        if any(
            x in kl
            for x in (
                "packet_v1",
                "packet_v2",
                "packet_v3",
                "packet_v4",
                "packet_v5",
                "packet_v6",
                "packet_v7",
            )
        ):
            dead_auth_refs[k] = auth.pop(k)
    if len(dead_auth_refs) > 5:
        auth["historical_packet_path_refs"] = dead_auth_refs
    pkt["authority_chain"] = auth
    for k in ("commit_sha", "source_commit", "parent_commit"):
        if k in pkt and isinstance(pkt[k], str) and len(pkt[k]) == 40:
            pkt[k] = commit

    # O1/O2: freeform historical under identity-bound DEAD carrier (path+sha bind).
    dead_hist = {
        "do_not_activate": True,
        "dead_lineage": True,
        "status": "DEAD_immutable",
        "path": bind_path,
        "sha256": bind_sha,
        "proposed_staging_list_for_packet_commit": pkt.pop(
            "proposed_staging_list_for_packet_commit", None
        ),
        "v6_changes_vs_v5": pkt.pop("v6_changes_vs_v5", None),
        "v5_changes_vs_v4_historical": pkt.pop("v5_changes_vs_v4_historical", None),
        "v4_changes_vs_v3_historical": pkt.pop("v4_changes_vs_v3_historical", None),
        "v3_changes_vs_v2_historical": pkt.pop("v3_changes_vs_v2_historical", None),
    }
    pins = dict(pkt.get("pins") or {})
    pins["historical_packet_fixture_notes"] = dead_hist
    # O2: move orphan bare hex64 pin leaves (no path identity) under identity-bound DEAD carrier.
    orphan_hex = {}
    for pk, pv in list(pins.items()):
        if isinstance(pv, str) and len(pv.strip()) in (40, 64) and all(
            c in "0123456789abcdefABCDEF" for c in pv.strip()
        ):
            orphan_hex[pk] = pins.pop(pk)
    if orphan_hex:
        hist2 = dict(pins["historical_packet_fixture_notes"])
        hist2["orphan_hex_leaves"] = orphan_hex
        pins["historical_packet_fixture_notes"] = hist2
    pkt["pins"] = pins

    # Scrub live surfaces: DEAD only via identity-bound schema, never key name.
    def _scrub_live_packet_rev_tokens(obj, *, dead: bool = False):
        import re as _re

        tok = _re.compile(r"packet_v(\d+)\b|(?<![A-Za-z0-9_])v(\d+)(?![A-Za-z0-9_])", _re.I)

        def _obj_dead(o: object) -> bool:
            if not isinstance(o, dict):
                return False
            if o.get("do_not_activate") is not True:
                return False
            if not (
                o.get("dead_lineage") is True
                or o.get("historical") is True
                or o.get("superseded") is True
                or str(o.get("status") or "").lower().replace(" ", "_")
                in {"dead", "dead_immutable", "historical", "superseded"}
            ):
                return False
            # O1: identity required
            return isinstance(o.get("path"), str) and isinstance(o.get("sha256"), str)

        if isinstance(obj, dict):
            out = {}
            for k, v in obj.items():
                child_dead = dead or _obj_dead(v)
                out[k] = _scrub_live_packet_rev_tokens(v, dead=child_dead)
            return out
        if isinstance(obj, list):
            return [_scrub_live_packet_rev_tokens(v, dead=dead) for v in obj]
        if isinstance(obj, str) and not dead:
            def _repl(m):
                n = m.group(1) or m.group(2)
                return "v8" if n == "8" else f"prior_rev_{n}"

            return tok.sub(_repl, obj)
        return obj

    # Full-tree scrub of live content for non-current rev tokens.
    pkt = _scrub_live_packet_rev_tokens(pkt)

    out = tmp / "packet_fixture.json"
    out.write_text(json.dumps(pkt, indent=2, sort_keys=True) + "\n")
    return out


def _faithful_rel_man(tmp_path: Path) -> Path:
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_host_man_{tmp_path.name}.json"
    rel.write_bytes(o.read_bytes())
    return rel


def _run_dry(pkt_path: Path, man_rel: Path, commit: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(DRY),
            "--packet",
            str(pkt_path),
            "--verify-source-manifest",
            man_rel.relative_to(REPO).as_posix(),
            "--expected-source-commit",
            commit,
            "--repo-root",
            str(REPO),
        ],
        cwd=str(REPO),
        capture_output=True,
        text=True,
    )


def test_H_MANIFEST_DETERMINISTIC_SORTED(tmp_path):
    o1 = tmp_path / "m1.json"
    o2 = tmp_path / "m2.json"
    r1 = _run_gen(o1)
    r2 = _run_gen(o2)
    assert r1.returncode == 0, r1.stderr
    assert r2.returncode == 0, r2.stderr
    assert "SCIENCE_SOURCE_MANIFEST_OK" in r1.stdout
    assert o1.read_bytes() == o2.read_bytes()
    paths = [e["path"] for e in _load_man(o1)["entries"]]
    assert paths == sorted(paths)
    assert len(paths) == len(set(paths))


def test_H_MANIFEST_SCHEMA_PATH_SHA(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    man = _load_man(o)
    assert man["schema"].startswith("LANDS_AB_science_source_manifest")
    for e in man["entries"]:
        assert isinstance(e["path"], str) and e["path"]
        assert len(e["sha256"]) == 64
        assert all(c in "0123456789abcdef" for c in e["sha256"])


def test_H_MANIFEST_MANDATORY_PATHS_PRESENT(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    paths = {e["path"] for e in _load_man(o)["entries"]}
    for sub in MANDATORY_SUBSTR:
        assert any(sub in p for p in paths), f"missing mandatory {sub}"


def test_H_MANIFEST_MISSING_ENTRY_FAIL(tmp_path):
    o = tmp_path / "m.json"
    cmd = [
        sys.executable,
        str(GEN),
        "--entry",
        "scripts/does_not_exist_lands_ab.py",
        "--out",
        str(o),
        "--repo-root",
        str(REPO),
    ]
    r = subprocess.run(cmd, cwd=str(REPO), capture_output=True, text=True)
    assert r.returncode != 0


def test_H_MANIFEST_HASHES_FROM_DISK(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0
    scratch = REPO / "artifacts" / "acc_entropy" / f"_tmp_manifest_hash_probe_{tmp_path.name}.py"
    try:
        scratch.write_text("# probe v1\n")
        o1 = tmp_path / "h1.json"
        r1 = _run_gen(o1, extra=["--also-include", scratch.relative_to(REPO).as_posix()])
        assert r1.returncode == 0, r1.stderr
        h1 = {e["path"]: e["sha256"] for e in _load_man(o1)["entries"]}[
            scratch.relative_to(REPO).as_posix()
        ]
        scratch.write_text("# probe v2 changed\n")
        o2 = tmp_path / "h2.json"
        r2 = _run_gen(o2, extra=["--also-include", scratch.relative_to(REPO).as_posix()])
        assert r2.returncode == 0, r2.stderr
        h2 = {e["path"]: e["sha256"] for e in _load_man(o2)["entries"]}[
            scratch.relative_to(REPO).as_posix()
        ]
        assert h1 != h2
        assert h2 == _sha(scratch)
    finally:
        if scratch.exists():
            scratch.unlink()


def test_H_MANIFEST_ESCAPE_OR_DUP_FAIL(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(
        o,
        extra=[
            "--also-include",
            "scripts/lands_ab_eval_run.py",
            "--also-include",
            "scripts/lands_ab_eval_run.py",
        ],
    )
    assert r.returncode != 0
    r2 = _run_gen(o, extra=["--also-include", "../outside.py"])
    assert r2.returncode != 0


def test_H_MANIFEST_PACKET_DRY_EXEC_INTEGRATION(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    commit = "0" * 40
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_bad_man_{tmp_path.name}.json"
    try:
        man3 = _load_man(o)
        man3["entries"].append(
            {"path": "scripts/__missing_lands_ab_required__.py", "sha256": "a" * 64}
        )
        man3["entries"] = sorted(man3["entries"], key=lambda e: e["path"])
        rel.write_text(json.dumps(man3, indent=2, sort_keys=True) + "\n")
        pkt4 = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        rr = _run_dry(pkt4, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_MANIFEST_SUBSTITUTION_FAIL(tmp_path):
    o = tmp_path / "m.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    commit = "1" * 40
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_sub_man_{tmp_path.name}.json"
    try:
        rel.write_bytes(o.read_bytes())
        good_sha = _sha(rel)
        pkt = _packet_for_dry(tmp_path, rel, good_sha, commit)
        j = json.loads(pkt.read_text())
        j["science_source_manifest_sha256"] = "f" * 64
        pkt_bad = tmp_path / "pkt_bad_sha.json"
        pkt_bad.write_text(json.dumps(j, indent=2) + "\n")
        rr = _run_dry(pkt_bad, rel, commit)
        assert rr.returncode != 0
        j2 = json.loads(pkt.read_text())
        j2["source_commit_sha"] = "2" * 40
        pkt_bad2 = tmp_path / "pkt_bad_commit.json"
        pkt_bad2.write_text(json.dumps(j2, indent=2) + "\n")
        rr2 = _run_dry(pkt_bad2, rel, commit)
        assert rr2.returncode != 0
        rr3 = subprocess.run(
            [
                sys.executable,
                str(DRY),
                "--packet",
                str(pkt),
                "--verify-source-manifest",
                "scripts/lands_ab_eval_run.py",
                "--expected-source-commit",
                commit,
                "--repo-root",
                str(REPO),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert rr3.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_FAITHFUL_V6_POSITIVE_PASS(tmp_path):
    commit = "a" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        rr = _run_dry(pkt_path, rel, commit)
        assert rr.returncode == 0, rr.stderr
        assert "PACKET_DRY_EXEC_OK" in rr.stdout
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_CLAIM_CEILING_MISSING_FAIL(tmp_path):
    commit = "4" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        del j["claim_ceiling"]
        bad = tmp_path / "pkt_no_cc.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_PRIORITY_ORDER_TAMPER_FAIL(tmp_path):
    commit = "5" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["PRIORITY_ORDER"] = list(reversed(j["PRIORITY_ORDER"]))
        bad = tmp_path / "pkt_po.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_EXECUTOR_TEST_OPERATOR_ALIAS_FAIL(tmp_path):
    commit = "6" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["executor"] = {
            "role": "test-operator",
            "forbidden_for_plan_dev": ["formal 7-row matrix execution"],
            "one_terminal_receipt": True,
        }
        bad = tmp_path / "pkt_ex.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_SEVEN_ONLY_EXECUTION_ORDER_FAIL(tmp_path):
    commit = "7" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["execution_order"] = [
            "G_CPU_STATIC_AB",
            "G_CUDA_B1_APPLY",
            "G_CUDA_B2_APPLY",
            "G_CUDA_B3_APPLY",
            "G_CUDA_ORACLE_B1",
            "G_CUDA_ORACLE_B2",
            "G_CUDA_ORACLE_B3",
        ]
        bad = tmp_path / "pkt_eo.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_HARVEST_BOOL_BARE_FAIL(tmp_path):
    commit = "8" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["runtime_scratch"]["harvest_exactly_one_raw_obs"] = True
        bad = tmp_path / "pkt_hv.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_METADATA_ARGV_PHASE_MISMATCH_FAIL(tmp_path):
    commit = "9" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        for rc in j["row_commands"]:
            if str(rc.get("gating_row")).startswith("G_CUDA"):
                rc["invocation"]["env_required"][
                    "SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"
                ] = "/tmp/wrong_phase.jsonl"
                break
        bad = tmp_path / "pkt_ph.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_CUDA_ARBITRARY_ARGV_FAIL(tmp_path):
    commit = "b" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        for rc in j["row_commands"]:
            if str(rc.get("gating_row")).startswith("G_CUDA"):
                rc["invocation"]["argv_template"] = ["true"]
                break
        bad = tmp_path / "pkt_true.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_NO_TIMEOUT_FAIL(tmp_path):
    commit = "c" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        for rc in j["row_commands"]:
            if str(rc.get("gating_row")).startswith("G_CUDA"):
                argv = list(rc["invocation"]["argv_template"])
                if argv and argv[0] == "timeout":
                    rc["invocation"]["argv_template"] = argv[2:]
                break
        bad = tmp_path / "pkt_nt.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_SUBSTITUTED_TOOL_PATH_FAIL(tmp_path):
    commit = "d" * 40
    rel = _faithful_rel_man(tmp_path)
    alt = REPO / "artifacts/acc_entropy" / f"_tmp_alt_dry_{tmp_path.name}.py"
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        alt.write_bytes(DRY.read_bytes())
        j["dry_exec_tool_path"] = alt.relative_to(REPO).as_posix()
        j["dry_exec_tool_sha256"] = _sha(alt)
        bad = tmp_path / "pkt_tool.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()
        if alt.exists():
            alt.unlink()


def test_H_PACKET_MANIFEST_SCHEMA_MUTATION_FAIL(tmp_path):
    commit = "e" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        man = json.loads(rel.read_text())
        man["schema"] = "WRONG_SCHEMA/v0"
        rel.write_text(json.dumps(man, indent=2) + "\n")
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        rr = _run_dry(pkt_path, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_INJECTED_CHECKPOINT_WRITE_PATH_FAIL(tmp_path):
    commit = "f" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        rr_ok = _run_dry(pkt_path, rel, commit)
        assert rr_ok.returncode == 0, rr_ok.stderr
        j = json.loads(pkt_path.read_text())
        for rc in j["row_commands"]:
            if rc.get("gating_row") == "G_CPU_STATIC_AB":
                inv = rc["invocation"]
                bad_path = "calm/hrm/checkpoints/evil.pt"
                inv["raw_obs_path_template"] = bad_path
                argv = list(inv["argv_template"])
                if "--out" in argv:
                    argv[argv.index("--out") + 1] = bad_path
                inv["argv_template"] = argv
                break
        bad = tmp_path / "pkt_ckpt.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MISSING_CUDA_ENFORCER_FAIL(tmp_path):
    commit = "1" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        for rc in j["row_commands"]:
            if str(rc.get("gating_row")).startswith("G_CUDA"):
                inv = rc["invocation"]
                inv.pop("enforcer_receipt_path_template", None)
                inv.pop("terminal_collection_enforcer_receipt_path", None)
                argv = list(inv["argv_template"])
                if "--enforcer-receipt" in argv:
                    i = argv.index("--enforcer-receipt")
                    del argv[i : i + 2]
                inv["argv_template"] = argv
                break
        bad = tmp_path / "pkt_enf.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MISSING_SCRATCH_ENV_FAIL(tmp_path):
    commit = "2" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["row_commands"][0]["invocation"]["env_required"].pop("LANDS_AB_RUN_ROOT", None)
        bad = tmp_path / "pkt_env.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_ROW_TAMPER_FAIL(tmp_path):
    commit = "3" * 40
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["row_commands"] = j["row_commands"][:-1]
        bad = tmp_path / "pkt_row.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def _first_cuda(j: dict) -> dict:
    for rc in j["row_commands"]:
        if str(rc.get("gating_row")).startswith("G_CUDA"):
            return rc
    raise AssertionError("no CUDA row")


def test_H_PACKET_CUDA_CHAIN_PERMUTED_FAIL(tmp_path):
    """F1: pytest before enforcer / delimiter order must FAIL."""
    commit = "a1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        # Move second `--` + pytest block before first enforcer segment (permute chain)
        # timeout budget watch-wrap flags -- enforcer -- pytest
        # -> timeout budget watch-wrap flags -- pytest -- enforcer  (still two dashes)
        d0 = argv.index("--")
        d1 = argv.index("--", d0 + 1)
        head = argv[: d0 + 1]
        mid = argv[d0 + 1 : d1]  # enforcer segment
        tail = argv[d1 + 1 :]  # pytest segment
        rc["invocation"]["argv_template"] = head + tail + ["--"] + mid
        bad = tmp_path / "pkt_perm.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "enforcer" in rr.stderr.lower() or "pytest" in rr.stderr.lower() or "delimiter" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_WRONG_PYTEST_NODE_FAIL(tmp_path):
    """F2: cross-row pytest node substitution must FAIL."""
    commit = "a2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        gr = rc["gating_row"]
        argv = list(rc["invocation"]["argv_template"])
        pi = argv.index("pytest")
        # substitute B1 node with ORACLE B1 node
        argv[pi + 1] = (
            "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py"
            "::test_gpu_live_lands_ab_oracle_b1_events_equal"
        )
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_node.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0, (rr.stdout, rr.stderr)
        assert gr in rr.stderr or "pytest node" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MALFORMED_PHASE_BUDGETS_FAIL(tmp_path):
    """F3: four junk budgets / non-numeric / duplicates must FAIL."""
    commit = "a3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        # replace all --budget values with junk=1
        for i, a in enumerate(argv):
            if a == "--budget" and i + 1 < len(argv):
                argv[i + 1] = "junk=1.0"
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_bud.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_NONPOSITIVE_HEARTBEAT_TIMEOUT_FAIL(tmp_path):
    """F4: zero/negative heartbeat or timeout must FAIL."""
    commit = "a4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        hi = argv.index("--heartbeat")
        argv[hi + 1] = "0"
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_hb.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0

        # outer timeout 0
        pkt_path2 = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j2 = json.loads(pkt_path2.read_text())
        rc2 = _first_cuda(j2)
        argv2 = list(rc2["invocation"]["argv_template"])
        assert argv2[0] == "timeout"
        argv2[1] = "0"
        rc2["invocation"]["argv_template"] = argv2
        bad2 = tmp_path / "pkt_to.json"
        bad2.write_text(json.dumps(j2) + "\n")
        rr2 = _run_dry(bad2, rel, commit)
        assert rr2.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_RUNTIME_PATH_OUTSIDE_RUN_ROOT_FAIL(tmp_path):
    """F5: raw/phase/enforcer outside LANDS_AB_RUN_ROOT must FAIL."""
    commit = "a5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        inv = rc["invocation"]
        outside = "/tmp/not_under_run_root/phase_events_evil.jsonl"
        inv["env_required"]["SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"] = outside
        argv = list(inv["argv_template"])
        pi = argv.index("--phase-events-jsonl")
        argv[pi + 1] = outside
        inv["argv_template"] = argv
        bad = tmp_path / "pkt_outroot.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_RUNTIME_PATH_INSIDE_REPO_ARTIFACTS_FAIL(tmp_path):
    """F5: runtime sink under repo artifacts/ must FAIL."""
    commit = "a6" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        # rebind all rows' scratch/run_root into repo artifacts with nonce
        scratch = str((REPO / "artifacts/acc_entropy/_runtime_scratch_hostile").resolve())
        run_root = scratch + "/test-operator/<nonce>"
        for rc in j["row_commands"]:
            inv = rc["invocation"]
            env = inv["env_required"]
            env["LANDS_AB_RUNTIME_SCRATCH"] = scratch
            env["LANDS_AB_RUN_ROOT"] = run_root
            gr = rc["gating_row"]
            raw = f"{run_root}/lands_ab_raw_obs_{gr}_<run_local_nonce>.json"
            inv["raw_obs_path_template"] = raw
            if "out_path_template" in inv:
                inv["out_path_template"] = raw
            argv = list(inv["argv_template"])
            if "--out" in argv:
                argv[argv.index("--out") + 1] = raw
            if str(gr).startswith("G_CUDA"):
                phase = f"{run_root}/phase_events_{gr}.jsonl"
                enf = f"{run_root}/enforcer_receipt_{gr}.json"
                env["SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"] = phase
                inv["enforcer_receipt_path_template"] = enf
                if "--phase-events-jsonl" in argv:
                    argv[argv.index("--phase-events-jsonl") + 1] = phase
                if "--enforcer-receipt" in argv:
                    argv[argv.index("--enforcer-receipt") + 1] = enf
            inv["argv_template"] = argv
        bad = tmp_path / "pkt_repo_sink.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_STATIC_RUN_ROOT_NO_NONCE_FAIL(tmp_path):
    """F6: static/reused run root without nonce token must FAIL."""
    commit = "a7" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        static_root = "/tmp/lands_ab_runtime_scratch/static_no_nonce/test-operator/fixed"
        scratch = "/tmp/lands_ab_runtime_scratch/static_no_nonce"
        for rc in j["row_commands"]:
            inv = rc["invocation"]
            env = inv["env_required"]
            env["LANDS_AB_RUNTIME_SCRATCH"] = scratch
            env["LANDS_AB_RUN_ROOT"] = static_root
            gr = rc["gating_row"]
            raw = f"{static_root}/lands_ab_raw_obs_{gr}_fixed.json"
            inv["raw_obs_path_template"] = raw
            argv = list(inv["argv_template"])
            if "--out" in argv:
                argv[argv.index("--out") + 1] = raw
            if str(gr).startswith("G_CUDA"):
                phase = f"{static_root}/phase_events_{gr}.jsonl"
                enf = f"{static_root}/enforcer_receipt_{gr}.json"
                env["SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"] = phase
                inv["enforcer_receipt_path_template"] = enf
                if "--phase-events-jsonl" in argv:
                    argv[argv.index("--phase-events-jsonl") + 1] = phase
                if "--enforcer-receipt" in argv:
                    argv[argv.index("--enforcer-receipt") + 1] = enf
            inv["argv_template"] = argv
        bad = tmp_path / "pkt_static.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "nonce" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_RELATIVE_MANIFEST_ESCAPE_FAIL(tmp_path):
    """F7: relative --verify-source-manifest with `..` escape must FAIL."""
    commit = "a8" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        # craft a path that resolves outside repo via ..
        outside = Path("/tmp") / f"lands_ab_escape_man_{tmp_path.name}.json"
        outside.write_bytes(rel.read_bytes())
        # relative from REPO: e.g. artifacts/acc_entropy/../../../tmp/... is messy;
        # use an absolute outside path on CLI and ensure rejection; also packet path with ..
        rr = subprocess.run(
            [
                sys.executable,
                str(DRY),
                "--packet",
                str(pkt_path),
                "--verify-source-manifest",
                str(outside),
                "--expected-source-commit",
                commit,
                "--repo-root",
                str(REPO),
            ],
            cwd=str(REPO),
            capture_output=True,
            text=True,
        )
        assert rr.returncode != 0

        # relative escape: invent a path string with .. that leaves repo
        # from REPO, `../` + basenames — resolve to parent of REPO
        parent = REPO.parent
        esc = parent / f"_lands_ab_escape_{tmp_path.name}.json"
        esc.write_bytes(rel.read_bytes())
        try:
            rel_esc = Path("..") / esc.name
            # place file at REPO.parent / name so .. / name resolves there
            rr2 = subprocess.run(
                [
                    sys.executable,
                    str(DRY),
                    "--packet",
                    str(pkt_path),
                    "--verify-source-manifest",
                    str(rel_esc),
                    "--expected-source-commit",
                    commit,
                    "--repo-root",
                    str(REPO),
                ],
                cwd=str(REPO),
                capture_output=True,
                text=True,
            )
            assert rr2.returncode != 0
            assert "outside repo" in rr2.stderr.lower() or "manifest" in rr2.stderr.lower()
        finally:
            if esc.exists():
                esc.unlink()
            if outside.exists():
                outside.unlink()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_THIRD_DELIMITER_FAIL(tmp_path):
    """G1: extra `--` delimiter must FAIL."""
    commit = "b1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        argv.append("--")
        argv.append("true")
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_g1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "exactly two" in rr.stderr.lower() or "delimiter" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MISSING_PYTHON3_INTERPRETER_FAIL(tmp_path):
    """G2: enforcer child without leading python3 must FAIL."""
    commit = "b2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        d0 = argv.index("--")
        # remove python3 after first --
        if argv[d0 + 1] == "python3":
            del argv[d0 + 1]
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_g2.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_WATCHWRAP_FLAG_AFTER_D0_FAIL(tmp_path):
    """G3: --error placed after watch-wrap `--` must FAIL."""
    commit = "b3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        ei = argv.index("--error")
        # move --error and its value after first --
        err_flag = argv.pop(ei)
        err_val = argv.pop(ei)
        d0 = argv.index("--")
        argv.insert(d0 + 1, err_flag)
        argv.insert(d0 + 2, err_val)
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_g3.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_ENFORCER_FLAG_OUTSIDE_SEGMENT_FAIL(tmp_path):
    """G4: --budget outside (d0,d1) must FAIL."""
    commit = "b4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        # append decorative budget after second --
        argv.extend(["--budget", "forward_backward=1.0"])
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_g4.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_DECLARED_BUDGET_VALUE_MISMATCH_FAIL(tmp_path):
    """G5: declared phase_budgets_seconds != argv budgets must FAIL."""
    commit = "b5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        pb = dict(rc["invocation"]["phase_budgets_seconds"])
        pb["forward_backward"] = 999.0
        rc["invocation"]["phase_budgets_seconds"] = pb
        bad = tmp_path / "pkt_g5.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "declared" in rr.stderr.lower() or "budget" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_OUTER_TIMEOUT_BELOW_ROW_HARD_FAIL(tmp_path):
    """G6: outer timeout != row hard bound must FAIL."""
    commit = "b6" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        argv = list(rc["invocation"]["argv_template"])
        argv[1] = "1"
        rc["invocation"]["argv_template"] = argv
        bad = tmp_path / "pkt_g6.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "hard bound" in rr.stderr.lower() or "timeout" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_DIFFERENT_RUN_ROOTS_ACROSS_ROWS_FAIL(tmp_path):
    """G7: two rows with different LANDS_AB_RUN_ROOT must FAIL."""
    commit = "b7" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        # retarget only second row under a different nonce root
        rc = j["row_commands"][2]
        gr = rc["gating_row"]
        scratch = rc["invocation"]["env_required"]["LANDS_AB_RUNTIME_SCRATCH"]
        alt_root = scratch + "/other-operator/<nonce>"
        inv = rc["invocation"]
        inv["env_required"]["LANDS_AB_RUN_ROOT"] = alt_root
        raw = f"{alt_root}/lands_ab_raw_obs_{gr}_<run_local_nonce>.json"
        inv["raw_obs_path_template"] = raw
        argv = list(inv["argv_template"])
        if "--out" in argv:
            argv[argv.index("--out") + 1] = raw
        if str(gr).startswith("G_CUDA"):
            phase = f"{alt_root}/phase_events_{gr}.jsonl"
            enf = f"{alt_root}/enforcer_receipt_{gr}.json"
            inv["env_required"]["SPARSE_LIVE_CARRIER_PHASE_EVENTS_JSONL"] = phase
            inv["enforcer_receipt_path_template"] = enf
            inv["terminal_collection_enforcer_receipt_path"] = enf
            inv["phase_events_jsonl_template"] = phase
            inv["terminal_collection_phase_events_jsonl_path"] = phase
            if "--phase-events-jsonl" in argv:
                argv[argv.index("--phase-events-jsonl") + 1] = phase
            if "--enforcer-receipt" in argv:
                argv[argv.index("--enforcer-receipt") + 1] = enf
        inv["argv_template"] = argv
        bad = tmp_path / "pkt_g7.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_RETAINED_ALIAS_MISMATCH_FAIL(tmp_path):
    """G8: retained enforcer alias mismatch must FAIL."""
    commit = "b8" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        rc["invocation"]["terminal_collection_enforcer_receipt_path"] = "/tmp/wrong_alias.json"
        bad = tmp_path / "pkt_g8.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "alias" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_WRONG_ROW_RAW_BASENAME_FAIL(tmp_path):
    """G9: raw basename for wrong gating row must FAIL."""
    commit = "b9" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rc = _first_cuda(j)
        gr = rc["gating_row"]
        inv = rc["invocation"]
        run_root = inv["env_required"]["LANDS_AB_RUN_ROOT"]
        # use another row's id in basename
        other = "G_CUDA_B2_APPLY" if gr != "G_CUDA_B2_APPLY" else "G_CUDA_B3_APPLY"
        bad_raw = f"{run_root}/lands_ab_raw_obs_{other}_<run_local_nonce>.json"
        inv["raw_obs_path_template"] = bad_raw
        if "terminal_collection_raw_obs_path_template" in inv:
            inv["terminal_collection_raw_obs_path_template"] = bad_raw
        argv = list(inv["argv_template"])
        if "--out" in argv:
            argv[argv.index("--out") + 1] = bad_raw
        inv["argv_template"] = argv
        bad = tmp_path / "pkt_g9.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "harvest" in rr.stderr.lower() or "basename" in rr.stderr.lower() or gr in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def _hostile_remove_mandatory(tmp_path: Path, remove_path: str, commit: str) -> subprocess.CompletedProcess:
    """Build faithful packet+manifest with one mandatory entry removed (self-consistent)."""
    o = tmp_path / "m_full.json"
    r = _run_gen(o)
    assert r.returncode == 0, r.stderr
    man = _load_man(o)
    new_entries = [e for e in man["entries"] if e["path"] != remove_path]
    assert len(new_entries) == len(man["entries"]) - 1, f"{remove_path} not in generated manifest"
    man["entries"] = sorted(new_entries, key=lambda e: e["path"])
    man["n_entries"] = len(man["entries"])
    rel = REPO / "artifacts/acc_entropy" / f"_tmp_h_man_{tmp_path.name}.json"
    rel.write_text(json.dumps(man, indent=2, sort_keys=True) + "\n")
    pkt = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
    return _run_dry(pkt, rel, commit), rel


def test_H_PACKET_MISSING_MANDATORY_GPU_LIVE_FAIL(tmp_path):
    """H3: removing gpu_live test from manifest must FAIL missing mandatory source."""
    commit = "c1" * 20
    rr, rel = _hostile_remove_mandatory(
        tmp_path,
        "calm/llm_computer/tests/test_hrm_text_158_lands_ab_eval_gpu_live.py",
        commit,
    )
    try:
        assert rr.returncode != 0
        assert "missing mandatory source" in rr.stderr
        assert "gpu_live" in rr.stderr or "test_hrm_text_158_lands_ab_eval_gpu_live" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MISSING_MANDATORY_ENFORCER_FAIL(tmp_path):
    """H3: removing enforcer script from manifest must FAIL missing mandatory source."""
    commit = "c2" * 20
    rr, rel = _hostile_remove_mandatory(
        tmp_path,
        "scripts/sparse_live_carrier_gpu_phase_budget_enforcer.py",
        commit,
    )
    try:
        assert rr.returncode != 0
        assert "missing mandatory source" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_H_PACKET_MISSING_MANDATORY_WATCH_WRAP_FAIL(tmp_path):
    """H3: removing bin/watch-wrap from manifest must FAIL missing mandatory source."""
    commit = "c3" * 20
    rr, rel = _hostile_remove_mandatory(
        tmp_path,
        "bin/watch-wrap",
        commit,
    )
    try:
        assert rr.returncode != 0
        assert "missing mandatory source" in rr.stderr
        assert "watch-wrap" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_I1_STALE_HEAD_IN_PREFLIGHT_FAIL(tmp_path):
    """I1: live preflight commit token != source_commit_sha must FAIL."""
    commit = "a1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["preflight_checklist"] = [
            "HEAD must equal a258f3142e5e9af1ca95b6cc825d84afa3e7d637",
            "bound source_commit_sha is " + commit,
        ]
        bad = tmp_path / "pkt_i1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I1" in rr.stderr or "source_commit_sha" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_I2_STALE_RUNNER_SHA_VS_MANIFEST_FAIL(tmp_path):
    """I2: pins.runner_and_harness_shas mismatch vs bound manifest must FAIL."""
    commit = "a2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        pins = dict(j.get("pins") or {})
        rhs = dict(pins.get("runner_and_harness_shas") or {})
        target = "calm/hrm_text_158/native_full_stack/lands_ab_eval_cuda_sites.py"
        assert target in rhs
        rhs[target] = "0" * 64
        pins["runner_and_harness_shas"] = rhs
        j["pins"] = pins
        bad = tmp_path / "pkt_i2.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I2" in rr.stderr or "manifest" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_I3_V6_LIVE_ALIAS_TEXT_FAIL(tmp_path):
    """I3: alias_note claiming v6 sole live while packet_revision!=v6 must FAIL."""
    commit = "a3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["packet_revision"] = "v8"
        j["alias_note"] = "v6 is the sole live/operative launch packet for this formal matrix."
        bad = tmp_path / "pkt_i3.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I3" in rr.stderr or "alias" in rr.stderr.lower() or "live" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_I4_STOP_FORBIDS_EQUIVALENT_FAIL(tmp_path):
    """I4: stop rule forbidding EQUIVALENT while it is preregistered must FAIL."""
    commit = "a4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["stop_conditions"] = [
            "terminal branch != BR-LANDS-AB-FIXTURE-CONTRACT-FAIL is DEVIATION",
            "any other terminal is forbidden",
        ]
        bad = tmp_path / "pkt_i4.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I4" in rr.stderr or "stop" in rr.stderr.lower() or "EQUIVALENT" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_I5_AMBIGUOUS_PLAN_V6_PREFLIGHT_FAIL(tmp_path):
    """I5: bare PLAN_v6 rehash with foreign sha and no historical/eval qualifier must FAIL."""
    commit = "a5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["preflight_checklist"] = [
            "HEAD must equal source_commit_sha " + commit,
            "re-hash PLAN_v6 -> 93645d31ea8a0cb0f89cfc4f1aedd38190a47f18433b6bc67c9b5d98da7093c5",
        ]
        bad = tmp_path / "pkt_i5.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I5" in rr.stderr or "PLAN_v6" in rr.stderr or "ambiguous" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_J1_MISSING_DO_NOT_ACTIVATE_FAIL(tmp_path):
    """J1: packet lineage pin without do_not_activate must FAIL."""
    commit = "b1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        pins = dict(j.get("pins") or {})
        # find a packet_v lineage key
        key = None
        for k in pins:
            if "packet_v" in str(k).lower() and isinstance(pins[k], dict):
                key = k
                break
        assert key is not None, "fixture must carry packet_v lineage pins"
        pins[key] = dict(pins[key])
        pins[key].pop("do_not_activate", None)
        j["pins"] = pins
        bad = tmp_path / "pkt_j1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J1" in rr.stderr or "do_not_activate" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_J2_REMAINS_OPERATIVE_ALIAS_FAIL(tmp_path):
    """J2: 'packet_v6 remains the operative revision' must FAIL."""
    commit = "b2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["alias_note"] = (
            "packet_v6 remains the operative revision; packet_v8 is the current artifact"
        )
        bad = tmp_path / "pkt_j2.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J2" in rr.stderr or "I3" in rr.stderr or "operative" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_J2_STAYS_ACTIVE_ALIAS_FAIL(tmp_path):
    """K-A/L1: gate-1 Mb — non-current 'v6' outside DEAD subtree must FAIL."""
    commit = "c2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["alias_note"] = (
            "the v6 artifact stays active for launches; v8 exists on disk"
        )
        bad = tmp_path / "pkt_j2_stays_active.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "J2" in rr.stderr or "I3" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_L1_AUTHORITATIVE_ALIAS_FAIL(tmp_path):
    """L1: co_lead 'packet_v6 is authoritative' must FAIL (structural, no synonym list)."""
    commit = "d1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["alias_note"] = "packet_v6 is authoritative for launches; packet_v8 exists on disk"
        bad = tmp_path / "pkt_l1_auth.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "packet_v6" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_L1_MISSING_OPERATIVE_PACKET_REVISION_FAIL(tmp_path):
    """L1: missing operative_packet_revision must FAIL."""
    commit = "d2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j.pop("operative_packet_revision", None)
        bad = tmp_path / "pkt_l1_miss.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "operative_packet_revision" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_L2_PREREGISTRATION_PLAN_REHASH_FAIL(tmp_path):
    """L2: preregistration.live_plan_rehash bare foreign PLAN_v6 must FAIL (full-tree scan)."""
    commit = "d3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["preregistration"] = {
            "live_plan_rehash": "re-hash PLAN_v6 -> " + ("b" * 64),
        }
        bad = tmp_path / "pkt_l2_prereg.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr or "plan" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_L2_NESTED_LIST_PLAN_REHASH_FAIL(tmp_path):
    """L2: nested-list foreign PLAN_v6 under freeform live subtree must FAIL."""
    commit = "d4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        # freeform live region (not execution_order which has its own structural checks)
        j["row_metadata"] = [
            {"meta": ["re-hash PLAN_v6 -> " + ("c" * 64)]},
        ]
        bad = tmp_path / "pkt_l2_nested.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr or "plan" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_L3_DEAD_PLAN_V1_WITH_CORRECT_HASH_FAIL(tmp_path):
    """L3: DEAD PLAN_v1 as operative with its correct disk hash must FAIL."""
    commit = "d5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        plan_v1 = (
            REPO
            / "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
            "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v1.json"
        )
        assert plan_v1.is_file(), "PLAN_v1 must exist for hostile"
        v1_sha = _sha(plan_v1)
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rel_plan = plan_v1.relative_to(REPO).as_posix()
        j["operative_plan_id"] = rel_plan
        j["operative_plan_sha256"] = v1_sha
        ac = dict(j.get("authority_chain") or {})
        ac["operative_plan_id"] = rel_plan
        ac["operative_plan_sha256"] = v1_sha
        ac["plan_path"] = rel_plan
        ac["plan_sha256"] = v1_sha
        j["authority_chain"] = ac
        bad = tmp_path / "pkt_l3_planv1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L3" in rr.stderr or "J5" in rr.stderr or "operative_plan" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_M1_KEY_HINT_PACKET_V3_OVERRIDE_FAIL(tmp_path):
    """M1/co1: key containing packet_v3 alone must NOT exempt live claim."""
    commit = "e1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["packet_v3_override"] = "packet_v3 is operative for launches"
        bad = tmp_path / "pkt_m1_co1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "packet_v3" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_M1_KEY_HINT_PLAN_REHASH_FAIL(tmp_path):
    """M1/co2a: packet_v3_context plan rehash must FAIL (no key-name exemption)."""
    commit = "e2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["packet_v3_context"] = "re-hash PLAN_v2 -> " + ("b" * 64)
        bad = tmp_path / "pkt_m1_co2a.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr or "plan" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_M1_KEY_HINT_COMMIT_TOKEN_FAIL(tmp_path):
    """M1/co2b: packet_v3_context source_commit_sha foreign token must FAIL."""
    commit = "e3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["packet_v3_context"] = "source_commit_sha " + ("b" * 40)
        bad = tmp_path / "pkt_m1_co2b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I1" in rr.stderr or "commit" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_M2_DNA_ALONE_RENAMED_PIN_FAIL(tmp_path):
    """M2/M3/co3a: packet_v5_archive with only do_not_activate must FAIL J1 status."""
    commit = "e4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        pins = dict(j.get("pins") or {})
        key = next(
            k
            for k in pins
            if "packet_v5" in str(k).lower() and isinstance(pins[k], dict)
        )
        entry = dict(pins.pop(key))
        for f in ("status", "dead_lineage", "historical", "superseded"):
            entry.pop(f, None)
        entry["do_not_activate"] = True
        pins["packet_v5_archive"] = entry
        j["pins"] = pins
        bad = tmp_path / "pkt_m2_co3a.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J1" in rr.stderr or "independent" in rr.stderr.lower() or "status" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_M4_HINT_FREE_DNA_ALONE_LIVE_SCAN_FAIL(tmp_path):
    """M4/co3b: hint-free dict with only do_not_activate stays live-scanned."""
    commit = "e5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["archive_pin_alpha"] = {
            "do_not_activate": True,
            "note_plan": "PLAN_v2 rehash " + ("c" * 64),
            "source_commit_sha": "b" * 40,
        }
        bad = tmp_path / "pkt_m4_co3b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert (
            "I1" in rr.stderr
            or "J4" in rr.stderr
            or "L2" in rr.stderr
            or "commit" in rr.stderr.lower()
            or "plan" in rr.stderr.lower()
        )
    finally:
        if rel.exists():
            rel.unlink()


def test_J3_ONLY_FIXTURE_MAY_TERMINATE_FAIL(tmp_path):
    """J3: only FIXTURE may terminate; other branches forbidden DEVIATION must FAIL."""
    commit = "b3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["stop_conditions"] = [
            "Only BR-LANDS-AB-FIXTURE-CONTRACT-FAIL may terminate; all other preregistered branches are forbidden DEVIATION"
        ]
        bad = tmp_path / "pkt_j3.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J3" in rr.stderr or "I4" in rr.stderr or "stop" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_J4_AUTHORITY_CHAIN_BARE_PLAN_V6_FAIL(tmp_path):
    """J4: authority_chain bare PLAN_v6 rehash with foreign sha must FAIL."""
    commit = "b4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        auth = dict(j.get("authority_chain") or {})
        auth["live_plan_rehash"] = (
            "re-hash PLAN_v6 -> bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
        )
        j["authority_chain"] = auth
        bad = tmp_path / "pkt_j4.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "I5" in rr.stderr or "PLAN_v6" in rr.stderr or "ambiguous" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_J5_WRONG_OPERATIVE_PLAN_HASH_FAIL(tmp_path):
    """J5: operative_plan_sha256 not matching disk bytes must FAIL."""
    commit = "b5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["operative_plan_sha256"] = "c" * 64
        if isinstance(j.get("authority_chain"), dict):
            j["authority_chain"] = dict(j["authority_chain"])
            j["authority_chain"]["operative_plan_sha256"] = "c" * 64
            j["authority_chain"]["plan_sha256"] = "c" * 64
        bad = tmp_path / "pkt_j5.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J5" in rr.stderr or "operative_plan" in rr.stderr.lower() or "mismatch" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_O1_FREE_RIDER_SELF_DECLARED_DEAD_FAIL(tmp_path):
    """O1/n1: dna+status without identity-bound (path,sha) must stay live-scanned and FAIL."""
    commit = "f1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["free_rider"] = {
            "do_not_activate": True,
            "status": "historical",
            "claim": "packet_v3 is operative for launches",
            "plan": "PLAN_v2 -> " + ("c" * 64),
            "commit": "source_commit_sha " + ("b" * 40),
        }
        bad = tmp_path / "pkt_o1_n1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert (
            "I1" in rr.stderr
            or "L1" in rr.stderr
            or "J4" in rr.stderr
            or "packet_v3" in rr.stderr
            or "commit" in rr.stderr.lower()
        )
    finally:
        if rel.exists():
            rel.unlink()


def test_O2_ALT_SOURCE_SHA256_UNBOUND_HEX_FAIL(tmp_path):
    """O2/n2: whole-string foreign hex40 under a caller-supplied key must FAIL (no key-suffix carve-out)."""
    commit = "f2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["alt_source_sha256"] = "b" * 40
        bad = tmp_path / "pkt_o2_n2.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "O2" in rr.stderr or "I1" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_O2_CONTEXT_SHA256_UNBOUND_HEX_FAIL(tmp_path):
    """O2/n3: whole-string foreign hex64 under a caller-supplied key must FAIL."""
    commit = "f3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["context_sha256"] = "c" * 64
        bad = tmp_path / "pkt_o2_n3.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "O2" in rr.stderr or "I1" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_O2_BOOT_PATH_PACKET_V3_TOKEN_FAIL(tmp_path):
    """O2/n4: non-current packet_v3 token in path-shaped live leaf must FAIL L1."""
    commit = "f4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["boot_path"] = "run/packet_v3_operative.json"
        bad = tmp_path / "pkt_o2_n4.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "packet_v3" in rr.stderr or "J2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_O2_NOTE_PATH_FOREIGN_PLAN_FAIL(tmp_path):
    """O2/n5: foreign PLAN_v2 path leaf must FAIL J4/L2 (no key-suffix carve-out)."""
    commit = "f5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["note_path"] = (
            "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
            "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v2.json"
        )
        bad = tmp_path / "pkt_o2_n5.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr or "plan" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def _real_pin_identity(j):
    """A legitimate, disk-verifiable (path, sha256) pair from the faithful packet."""
    pv = j["pins"]["packet_v1_dead_lineage"]
    return pv["path"], pv["sha256"]


def test_Q1_SELF_MINTED_PIN_FABRICATED_IDENTITY_FAIL(tmp_path):
    """Q1/p1: pin at a caller-supplied key with fabricated path+sha must not exempt its subtree."""
    commit = "a1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["pins"]["my_archive"] = {
            "do_not_activate": True,
            "status": "DEAD",
            "path": "artifacts/acc_entropy/fabricated_archive.json",
            "sha256": "d" * 64,
            "hidden_claim": "packet_v3 is operative for launches",
        }
        bad = tmp_path / "pkt_q1_p1.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q3" in rr.stderr or "L1" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_Q1_NON_ALLOWLISTED_KEY_WITH_REAL_IDENTITY_FAIL(tmp_path):
    """
    Q1'/q1a: the copyable-pair hole. A non-allowlisted pins key that restates a
    LEGITIMATE disk-verified (path, sha256) pair must NOT inherit exemption —
    exemption is location-bound, not value-bound.
    """
    commit = "a2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        real_path, real_sha = _real_pin_identity(j)
        j["pins"]["my_archive"] = {
            "do_not_activate": True,
            "status": "DEAD",
            "path": real_path,
            "sha256": real_sha,
            "hidden_claim": "packet_v3 is operative for launches",
        }
        bad = tmp_path / "pkt_q1_q1a.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "J2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q1_ALLOWLISTED_KEY_MISMATCHED_DISK_SHA_FAIL(tmp_path):
    """Q1/q1b: allowlisted key shape but path/sha pairing not matching disk bytes must FAIL."""
    commit = "a3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        real_path, _ = _real_pin_identity(j)
        plan_rel = (
            "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
            "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v6.json"
        )
        plan_sha = hashlib.sha256((REPO / plan_rel).read_bytes()).hexdigest()
        j["pins"]["packet_v9_dead_lineage"] = {
            "do_not_activate": True,
            "status": "DEAD",
            "path": real_path,
            "sha256": plan_sha,
            "hidden_claim": "packet_v3 is operative for launches",
        }
        bad = tmp_path / "pkt_q1_q1b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "J2" in rr.stderr or "Q4" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q1_ALLOWLISTED_KEY_MISSING_FILE_FAIL(tmp_path):
    """Q1/q1c: allowlisted key whose declared artifact does not exist must FAIL."""
    commit = "a4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        plan_rel = (
            "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
            "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v6.json"
        )
        plan_sha = hashlib.sha256((REPO / plan_rel).read_bytes()).hexdigest()
        j["pins"]["packet_v9_dead_lineage"] = {
            "do_not_activate": True,
            "status": "DEAD",
            "path": "artifacts/acc_entropy/does_not_exist_v9.json",
            "sha256": plan_sha,
            "hidden_claim": "packet_v3 is operative for launches",
        }
        bad = tmp_path / "pkt_q1_q1c.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "L1" in rr.stderr or "J2" in rr.stderr or "Q4" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q2_FREEFORM_FIELD_CANNOT_TEACH_KNOWN_REFS_FAIL(tmp_path):
    """Q2/p2: a lineage pin's freeform field must not seed the known-reference set."""
    commit = "a5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        real_path, real_sha = _real_pin_identity(j)
        j["pins"]["packet_v9_dead_lineage"] = {
            "do_not_activate": True,
            "status": "DEAD",
            "path": real_path,
            "sha256": real_sha,
            "teach": "b" * 40,
        }
        j["deploy_commit"] = "b" * 40
        bad = tmp_path / "pkt_q2.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q4" in rr.stderr or "Q3" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_Q3_EMBEDDED_HEX_IN_PROSE_FAIL(tmp_path):
    """Q3/p4: a hex40 embedded in prose under a neutral key must FAIL (no context gate)."""
    commit = "a6" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["note"] = "deployed at " + "b" * 40 + " earlier today"
        bad = tmp_path / "pkt_q3.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q3" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_Q4_ADMITTED_PIN_NON_SCHEMA_FIELD_FAIL(tmp_path):
    """Q4: a disk-verified pin must not carry freeform live content behind its exemption."""
    commit = "a7" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["pins"]["packet_v1_dead_lineage"]["hidden_claim"] = (
            "packet_v3 is operative for launches"
        )
        bad = tmp_path / "pkt_q4.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q4" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q5_TRAVERSAL_PIN_CANNOT_INJECT_OUT_OF_REPO_REF_FAIL(tmp_path):
    """
    Q5: a pin path may satisfy startswith("artifacts/") and STILL resolve outside
    the repo via `..`. Binding such a file must not seed the known-reference set,
    or its path string laundries a foreign PLAN reference past L2/J4.
    """
    commit = "a8" * 20
    outside = tmp_path / "outside"
    outside.mkdir()
    evil = outside / (
        "optimizer_credit_state_sparse_vote_authority_"
        "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v3.json"
    )
    evil.write_text('{"note": "foreign plan"}\n')
    evil_sha = hashlib.sha256(evil.read_bytes()).hexdigest()
    rel_trav = "artifacts/" + os.path.relpath(evil, REPO / "artifacts").replace("\\", "/")
    assert rel_trav.startswith("artifacts/")
    assert ".." in rel_trav

    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["pins"]["q5_traversal_pin"] = {"path": rel_trav, "sha256": evil_sha}
        j["q5_live_foreign_plan_ref"] = rel_trav
        bad = tmp_path / "pkt_q5.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr or "Q3" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q5_ALIASED_IN_REPO_PIN_PATH_FAIL(tmp_path):
    """
    Q5 canonicality leg: a path carrying `..` that still resolves INSIDE the repo
    is an alias of a real artifact. Accepting it would let one file legitimize
    many distinct reference strings, so the alias must not be admitted.
    """
    commit = "a9" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        pin = dict(j["pins"]["packet_v1_dead_lineage"])
        real_rel = str(pin["path"]).replace("\\", "/")
        head, _, tail = real_rel.rpartition("/")
        pin["path"] = f"{head}/../{head.rsplit('/', 1)[-1]}/{tail}"
        assert ".." in pin["path"]
        j["pins"]["packet_v1_dead_lineage"] = pin
        bad = tmp_path / "pkt_q5b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        # alias is not admitted as an identity, so its subtree is live-scanned
        assert rr.returncode != 0
    finally:
        if rel.exists():
            rel.unlink()


_Q6_FOREIGN_PLAN = (
    "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
    "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v3.json"
)
# The family-frozen operative plan identity (must mirror the tool's
# EXPECTED_OPERATIVE_PLAN_REL / _SHA256 and what _packet_for_dry writes).
PLAN_V6_REL = (
    "artifacts/acc_entropy/optimizer_credit_state_sparse_vote_authority_"
    "LANDS_AB_CONSUMER_ADAPT_RERUN_PLAN_v6.json"
)
PLAN_V6_SHA256 = "e421aecdf1cc4b9a94d118a0563e7d8ac8516f97da998d38af3b8c60ac88a41c"


def test_Q6_CARRIER_CANNOT_TEACH_KNOWN_REFS_PLAN_FAIL(tmp_path):
    """
    Q6: the freeform carrier is Q4-schema-exempt, so it can hold arbitrary
    `*_path` fields. Harvesting those into known_refs would let the carrier
    exempt a FOREIGN plan reference in the LIVE tree — an escape FROM the
    carrier, not merely unscanned content inside it.
    """
    commit = "c1" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        carrier = dict(j["pins"]["historical_packet_fixture_notes"])
        carrier["injected_path"] = _Q6_FOREIGN_PLAN
        j["pins"]["historical_packet_fixture_notes"] = carrier
        j["q6_live_foreign_plan_ref"] = _Q6_FOREIGN_PLAN
        bad = tmp_path / "pkt_q6.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q6_CARRIER_CANNOT_TEACH_KNOWN_REFS_HEX_FAIL(tmp_path):
    """Q6: same escape via a typed `*_sha256` field injecting a foreign hex64."""
    commit = "c2" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        carrier = dict(j["pins"]["historical_packet_fixture_notes"])
        carrier["injected_sha256"] = "e" * 64
        j["pins"]["historical_packet_fixture_notes"] = carrier
        j["q6_live_hex"] = "e" * 64
        bad = tmp_path / "pkt_q6c.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q3" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_Q6_CARRIER_NESTED_TYPED_FIELD_FAIL(tmp_path):
    """Q6 depth leg: a typed ref nested BELOW the carrier must not seed refs either."""
    commit = "c3" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        carrier = dict(j["pins"]["historical_packet_fixture_notes"])
        carrier["nested"] = {"injected_path": _Q6_FOREIGN_PLAN}
        j["pins"]["historical_packet_fixture_notes"] = carrier
        j["q6_nested_live_ref"] = _Q6_FOREIGN_PLAN
        bad = tmp_path / "pkt_q6n.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q7_PIN_BLOCK_MSG_ID_CANNOT_BIND_HEX_FAIL(tmp_path):
    """
    Q7: `block_msg_id` is BOTH allowed by the Q4 closed schema on an ordinary
    (non-carrier) pin AND reference-shaped by key name. Trusting it by key name
    admitted an arbitrary hex64 into the known-reference set, which then exempted
    a live token. A room message id is not a content hash and must never bind one.
    Only the pin's disk-verified `path`/`sha256` may seed references.
    """
    commit = "c4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        pin = dict(j["pins"]["packet_v1_dead_lineage"])
        pin["block_msg_id"] = "f" * 64
        j["pins"]["packet_v1_dead_lineage"] = pin
        j["q7_live_hex"] = "f" * 64
        bad = tmp_path / "pkt_q7.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q3" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def _q8_authchain_case(tmp_path, commit, key, value, live_key, live_value):
    """Set authority_chain[key]=value and place the same value on a LIVE surface."""
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        ac = dict(j.get("authority_chain") or {})
        assert key in ac, f"{key} must be a pre-existing authority_chain key"
        ac[key] = value
        j["authority_chain"] = ac
        j[live_key] = live_value
        bad = tmp_path / f"pkt_q8_{key}.json"
        bad.write_text(json.dumps(j) + "\n")
        return _run_dry(bad, rel, commit)
    finally:
        if rel.exists():
            rel.unlink()


def test_Q8_AUTHCHAIN_PLAN_SHA256_CANNOT_BIND_HEX_FAIL(tmp_path):
    """
    Q8: authority_chain values must not be trusted by KEY NAME. `plan_sha256` was
    seeded unconditionally with no disk binding and no comparison against the
    operative plan, so a changed value was admitted and exempted a live token.
    """
    hexv = "b2" * 32
    rr = _q8_authchain_case(
        tmp_path, "e1" * 20, "plan_sha256", hexv, "q8_live_hex", hexv
    )
    assert rr.returncode != 0
    assert "Q3" in rr.stderr or "unbound" in rr.stderr.lower()


def test_Q8_AUTHCHAIN_PLAN_PATH_CANNOT_BIND_FOREIGN_PLAN_FAIL(tmp_path):
    """Q8: same route, path leg — a foreign PLAN reference must not be admitted."""
    rr = _q8_authchain_case(
        tmp_path,
        "e2" * 20,
        "plan_path",
        _Q6_FOREIGN_PLAN,
        "q8_live_ref",
        _Q6_FOREIGN_PLAN,
    )
    assert rr.returncode != 0
    assert "J4" in rr.stderr or "L2" in rr.stderr


def test_Q8_AUTHCHAIN_HEAD_A_CANNOT_BIND_HEX_FAIL(tmp_path):
    """Q8: same route, head_a leg — not the source commit, so it must not be admitted."""
    hexv = "c3" * 32
    rr = _q8_authchain_case(
        tmp_path, "e3" * 20, "head_a", hexv, "q8_live_head_hex", hexv
    )
    assert rr.returncode != 0
    assert "Q3" in rr.stderr or "unbound" in rr.stderr.lower()


def test_Q8_FAITHFUL_AUTHCHAIN_VERIFIABLE_VALUES_STILL_ACCEPTED(tmp_path):
    """
    Q8 positive guard: the faithful authority_chain must KEEP validating. Its
    plan_path/plan_sha256 are the family-frozen operative plan identity and head_a
    is the source commit, so all three stay independently verifiable. This fails if
    Q8 is ever tightened past value-verifiability into rejecting legitimate values.
    """
    commit = "e4" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        ac = j.get("authority_chain") or {}
        assert ac.get("plan_path") == PLAN_V6_REL
        assert ac.get("plan_sha256") == PLAN_V6_SHA256
        assert ac.get("head_a") == commit
        rr = _run_dry(pkt_path, rel, commit)
        assert rr.returncode == 0, f"faithful authority_chain regressed: {rr.stderr!r}"
    finally:
        if rel.exists():
            rel.unlink()


def test_Q9_ADAPT_PLAN_ID_DISAGREEMENT_FAIL(tmp_path):
    """
    Q9: `operative_adaptation_plan_id` was harvested unconditionally while L3/J5's
    `or` short-circuited past it whenever the standard sibling was present, so the
    alternative naming carried an identity nothing ever compared.
    """
    commit = "f5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["operative_adaptation_plan_id"] = _Q6_FOREIGN_PLAN
        bad = tmp_path / "pkt_q9a.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q9" in rr.stderr or "J4" in rr.stderr or "L2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q9_ADAPT_PLAN_SHA_DISAGREEMENT_FAIL(tmp_path):
    """Q9: same asymmetry on the sha256 leg."""
    commit = "f6" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j["operative_adaptation_plan_sha256"] = "d" * 64
        bad = tmp_path / "pkt_q9b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "Q9" in rr.stderr or "unbound" in rr.stderr.lower()
    finally:
        if rel.exists():
            rel.unlink()


def test_Q9_POSITIVE_ADAPTATION_ONLY_NAMING_ACCEPTED(tmp_path):
    """
    Q9 positive: the alternative naming is a shape this tool supports, so a packet
    using ONLY `operative_adaptation_*` with correct values must still validate.
    Guards against curing the asymmetry by breaking the alternative naming.
    """
    commit = "f7" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        j.pop("operative_plan_id", None)
        j.pop("operative_plan_sha256", None)
        j["operative_adaptation_plan_id"] = PLAN_V6_REL
        j["operative_adaptation_plan_sha256"] = PLAN_V6_SHA256
        good = tmp_path / "pkt_q9pos.json"
        good.write_text(json.dumps(j) + "\n")
        rr = _run_dry(good, rel, commit)
        assert rr.returncode == 0, f"adaptation-only naming regressed: {rr.stderr!r}"
    finally:
        if rel.exists():
            rel.unlink()


def test_Q10_PLAN_PATH_MENTION_PLUS_FOREIGN_REVISION_FAIL(tmp_path):
    """
    Q10: the skip for plan_path/plan_sha256 used SUBSTRING containment, so a value
    that merely MENTIONED the operative identity was waved through and an appended
    foreign-revision claim rode along on the mention.
    """
    commit = "f8" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        ac = dict(j["authority_chain"])
        ac["plan_path"] = PLAN_V6_REL + " superseding PLAN_v3 which stays operative"
        j["authority_chain"] = ac
        bad = tmp_path / "pkt_q10a.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q10_PLAN_SHA_MENTION_PLUS_FOREIGN_REVISION_FAIL(tmp_path):
    """Q10: same containment escape on the sha leg."""
    commit = "f9" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        ac = dict(j["authority_chain"])
        ac["plan_sha256"] = PLAN_V6_SHA256 + " and PLAN_v2 remains authoritative"
        j["authority_chain"] = ac
        bad = tmp_path / "pkt_q10b.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "J4" in rr.stderr or "L2" in rr.stderr
    finally:
        if rel.exists():
            rel.unlink()


def test_Q10_POSITIVE_OPERATIVE_REVISION_MENTION_ACCEPTED(tmp_path):
    """
    Q10 positive: naming the OPERATIVE revision in prose is legitimate — a faithful
    preflight row reads "re-hash operative CONSUMER_ADAPT_RERUN PLAN_v6 -> <sha>".
    Only a DIFFERENT revision in the remainder is disqualifying. This caught a real
    over-tightening during implementation; keep it.
    """
    commit = "fa" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rows = [r for r in (j.get("preflight_checklist") or []) if isinstance(r, str)]
        assert any(
            "plan_v6" in r.lower() and PLAN_V6_SHA256 in r.lower() for r in rows
        ), "expected a faithful row naming the operative revision alongside its sha"
        rr = _run_dry(pkt_path, rel, commit)
        assert rr.returncode == 0, f"operative-revision mention regressed: {rr.stderr!r}"
    finally:
        if rel.exists():
            rel.unlink()


# --- Q11: `self_check` `pinned_to_*` head claims -------------------------------
# The retired guard fired only on an exact hex40 token or one of two hardcoded
# literals, so the 7-12 char shorthand git actually writes for a head satisfied
# neither and passed silently -- a narrow trigger standing in for the invariant.
# Cure = RULE: every hex-shaped run (>=7) in a True `pinned_to_*` key must be a
# PREFIX of src. NON-repeating commit below so "prefix" and "substring" differ.
_Q11_COMMIT = "0123456789abcdef0123456789abcdef01234567"
_Q11_MIDSHA = "89abcdef"  # substring of _Q11_COMMIT at index 8 -- NOT a prefix


def _q11_pin_case(tmp_path, tag, key, *, commit=_Q11_COMMIT):
    """Plant a boolean `self_check` pin key and dry-exec. Returns CompletedProcess."""
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        sc = dict(j["self_check"])
        sc[key] = True
        j["self_check"] = sc
        bad = tmp_path / f"pkt_q11_{tag}.json"
        bad.write_text(json.dumps(j) + "\n")
        return _run_dry(bad, rel, commit)
    finally:
        if rel.exists():
            rel.unlink()


def test_Q11_SHORTHAND_12HEX_NONSOURCE_HEAD_FAIL(tmp_path):
    """
    Q11: the measured regression. `pinned_to_deadbeefcafe` is neither a hex40 token
    nor one of the two retired literals, so the enumeration-based guard returned 0
    on a head claim naming a commit that is not the source. 7-12 char abbreviations
    are how git writes heads, so this shape is MORE likely in a real packet than the
    hex40 form the retired check caught.
    """
    rr = _q11_pin_case(tmp_path, "sh12", "pinned_to_deadbeefcafe")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_SHORTHAND_7HEX_NONSOURCE_HEAD_FAIL(tmp_path):
    """Q11: 7 chars is git's shortest conventional abbreviation -- the rule's floor."""
    rr = _q11_pin_case(tmp_path, "sh7", "pinned_to_deadbee")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_HEX40_NONSOURCE_HEAD_FAIL(tmp_path):
    """
    Q11 CONTROL: the full-length shape the retired guard DID catch must stay caught
    under the rule. Without this, a regression that narrows the rule back toward an
    enumeration could pass on the new cases alone.
    """
    rr = _q11_pin_case(tmp_path, "hex40", "pinned_to_" + "b" * 40)
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_RETIRED_LITERALS_STILL_CAUGHT_BY_RULE(tmp_path):
    """
    Q11 CONTROL: both retired hardcoded literals must still be rejected -- BY THE
    RULE, with the enumeration deleted. This is what proves the deletion lost no
    coverage rather than merely moving the goalposts to shapes the rule happens to
    cover. (`pinned_to_a258f314` is the value the real HEAD_A packet template
    carries, which is why the literals were there.)
    """
    for tag, key in (
        ("lit_a", "pinned_to_a258f314ffff"),
        ("lit_b", "pinned_to_95097a8d"),
    ):
        rr = _q11_pin_case(tmp_path, tag, key)
        assert rr.returncode != 0, f"retired literal {key} no longer caught"
        assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_MIDSHA_RUN_IS_NOT_A_PREFIX_FAIL(tmp_path):
    """
    Q11: the retired `suffix not in src` leg skipped any suffix appearing ANYWHERE in
    src, so a run matching mid-sha was accepted. A mid-sha run is not a legitimate
    abbreviation of a head -- only a prefix is.
    """
    rr = _q11_pin_case(tmp_path, "midsha", f"pinned_to_{_Q11_MIDSHA}")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_ADJACENT_LETTER_DOES_NOT_SHIELD_FAIL(tmp_path):
    """
    Q11: the token regex deliberately carries NO lookbehind/lookahead guard, so an
    adjacent non-hex letter cannot shield a head-shaped run from the check.
    """
    rr = _q11_pin_case(tmp_path, "shield", "pinned_to_xdeadbeefcafe")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q11_HEX64_NONHEAD_GETS_HEAD_ONLY_DIAGNOSTIC(tmp_path):
    """
    Q11 head-only contract: `pinned_to_*` is head-only BY CONTRACT, so a legitimate
    NON-head sha under that prefix (e.g. a manifest sha256) is malformed by design.
    It must be rejected with its OWN diagnostic naming the contract -- a confusing
    generic rejection of a legitimate-looking value is how a later round "repairs"
    the guard by widening it. Classified by LENGTH ALONE (a head is at most 40
    chars), never by membership in the reference set: coupling the diagnostic to
    `known_refs` would reintroduce name-conferred trust from the exemption side.
    """
    rr = _q11_pin_case(tmp_path, "hex64", "pinned_to_manifest_" + "c" * 64)
    assert rr.returncode != 0
    assert "Q11" in rr.stderr
    assert "head-only" in rr.stderr, f"lost the head-only diagnostic: {rr.stderr!r}"
    assert "cannot be a commit head" in rr.stderr


def test_Q11_POSITIVE_LEGIT_SRC_PREFIX_PINS_ACCEPTED(tmp_path):
    """
    Q11 positive (over-tightening guard, paired with the tightening per standing):
    a faithful packet and every legitimate abbreviation of the SOURCE head must
    still validate. The faithful packet's only pin key is `pinned_to_source_commit`,
    whose longest contiguous [0-9a-f] run is 2 chars, so the rule cannot reach it.
    """
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), _Q11_COMMIT)
        rr = _run_dry(pkt_path, rel, _Q11_COMMIT)
        assert rr.returncode == 0, f"faithful packet regressed: {rr.stderr!r}"
        sc = json.loads(pkt_path.read_text())["self_check"]
        pins = [k for k in sc if "pinned_to" in str(k).lower()]
        assert pins == ["pinned_to_source_commit"], f"pin-key surface changed: {pins}"
    finally:
        if rel.exists():
            rel.unlink()
    for tag, n in (("p7", 7), ("p12", 12), ("p40", 40)):
        rr = _q11_pin_case(tmp_path, tag, f"pinned_to_{_Q11_COMMIT[:n]}")
        assert rr.returncode == 0, (
            f"legitimate {n}-char src-prefix pin rejected: {rr.stderr!r}"
        )


# --- Q12 / blocker B1v10-P1: the pin scan must read the WHOLE key ---------------
# The Q11 rule was correct but its INPUT was truncated: `kl.split("pinned_to",1)[-1]`
# discarded everything BEFORE the first marker, while the admission test still treated
# the key as a pin. A head-shaped run placed ahead of the marker was therefore never
# examined at ANY length -- including hex40 and hex64, which proves the Q3/I1 token
# backstop does not reach `self_check` key names either (it walks VALUES). Negative
# cases below are the four measured rc=0 rows.


def test_Q12_PREMARKER_7HEX_NONSOURCE_HEAD_FAIL(tmp_path):
    """Q12: 7-char run before the marker -- rule floor, previously unexamined."""
    rr = _q11_pin_case(tmp_path, "pre7", "abcdef1_pinned_to_source_commit")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q12_PREMARKER_12HEX_NONSOURCE_HEAD_FAIL(tmp_path):
    """Q12: 12-char git-shorthand run before the marker."""
    rr = _q11_pin_case(tmp_path, "pre12", "deadbeefcafe_pinned_to_source_commit")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q12_PREMARKER_HEX40_NONSOURCE_HEAD_FAIL(tmp_path):
    """
    Q12: a FULL-LENGTH head before the marker. This row is load-bearing beyond the
    positional defect itself: it demonstrates that no other check catches a hex40
    token in a `self_check` KEY name, so this guard is the only line of defence.
    """
    rr = _q11_pin_case(tmp_path, "pre40", "b" * 40 + "_pinned_to_source_commit")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr and "non-source head" in rr.stderr


def test_Q12_PREMARKER_HEX64_HEAD_ONLY_CONTRACT_FAIL(tmp_path):
    """Q12: >40 before the marker must still route to the head-only diagnostic."""
    rr = _q11_pin_case(tmp_path, "pre64", "c" * 64 + "_pinned_to_source_commit")
    assert rr.returncode != 0
    assert "Q11" in rr.stderr
    assert "head-only" in rr.stderr and "cannot be a commit head" in rr.stderr


def test_Q12_POSITIVE_FAITHFUL_PIN_KEY_STILL_ACCEPTED(tmp_path):
    """
    Q12 positive control: widening the scan from the post-marker remainder to the
    whole key must not reject the faithful key. `pinned_to_source_commit` has a
    longest contiguous [0-9a-f] run of 2 ("ce" in "source"), far under the 7 floor.
    """
    rr = _q11_pin_case(tmp_path, "posfaithkey", "pinned_to_source_commit")
    assert rr.returncode == 0, f"faithful pin key regressed: {rr.stderr!r}"


def test_Q12_POSITIVE_PREMARKER_LEGIT_SRC_PREFIX_ACCEPTED(tmp_path):
    """
    Q12 positive: the whole-key scan must judge a run by VALUE, not by position. A
    legitimate src prefix placed BEFORE the marker is a true head claim and must be
    accepted -- otherwise the cure would trade a fail-open for a fail-closed on
    legitimate input, which is the mirror-image defect this slice has already hit
    once (the plan_vN prose row).
    """
    rr = _q11_pin_case(
        tmp_path, "posprelegit", f"{_Q11_COMMIT[:12]}_pinned_to_source_commit"
    )
    assert rr.returncode == 0, f"legitimate pre-marker src prefix rejected: {rr.stderr!r}"


# --- Q13 / blocker B1v11-P2a+P2b: assertion fields must be JSON booleans ---------
# Both P2 gaps were IDENTITY tests (`v is True`), which silently skip every non-bool:
# the validator reads "no assertion" while a consumer testing truthiness reads one.
# Strict-bool is the only COHERENT rule, and `"false"` is the deciding case -- a
# non-empty string is TRUTHY in Python, so a truthiness rule would raise
# "claims non-source head" on a packet that literally said false while skipping the
# falsy `0`. Two identical negations, opposite treatment, decided by Python rather than
# by meaning.
_Q13_FOREIGN = "deadbeefcafe"
_Q13_MIDSHA = _Q11_COMMIT[1:15]  # genuine non-prefix substring
_Q13_NONBOOL = (
    ("int1", 1), ("float1", 1.0), ("str_true", "true"), ("str_True", "True"),
    ("str_yes", "yes"), ("str_1", "1"), ("list1", [1]), ("str_false", "false"),
    ("dict_empty", {}), ("dict_a1", {"a": 1}), ("null", None), ("int0", 0),
    ("str_empty", ""), ("list_empty", []),
)


def _q13_case(tmp_path, tag, mutate, *, commit=_Q11_COMMIT):
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        mutate(j)
        bad = tmp_path / f"pkt_q13_{tag}.json"
        bad.write_text(json.dumps(j) + "\n")
        return _run_dry(bad, rel, commit)
    finally:
        if rel.exists():
            rel.unlink()


def _q13_set_pin(key, val):
    def _m(j):
        sc = dict(j["self_check"])
        sc[key] = val
        j["self_check"] = sc
    return _m


def _q13_set_sci(val):
    def _m(j):
        j["science_claim"] = val
    return _m


def test_Q13_PIN_VALUE_NONBOOL_IS_MALFORMED_FAIL(tmp_path):
    """
    Q13a: a pin is an ASSERTION field, so a non-bool value is malformed rather than
    "not asserted". Covers the measured rc=0 rows 1 / 1.0 / "true" / "True" / "yes" /
    "1" / [1], plus the shapes that decide the rule: "false" (truthy string), {} ,
    {"a":1}, null, 0, "", [].
    """
    for tag, val in _Q13_NONBOOL:
        rr = _q13_case(
            tmp_path, f"pv_{tag}", _q13_set_pin(f"pinned_to_{_Q13_FOREIGN}", val)
        )
        assert rr.returncode != 0, f"non-bool pin value {tag} admitted"
        assert "Q13" in rr.stderr, f"{tag}: {rr.stderr!r}"


def test_Q13_PIN_VALUE_NONBOOL_PREMARKER_FAIL(tmp_path):
    """
    Q13a: the value axis must not reopen the position route cured by Q12. A pre-marker
    head-shaped run with a non-bool value was rc=0 even after the Q12 cure, because the
    identity test skipped the key before the whole-key scan ever ran.
    """
    for tag, val in (("int1", 1), ("str_true", "true"), ("float1", 1.0)):
        rr = _q13_case(
            tmp_path,
            f"pvpre_{tag}",
            _q13_set_pin(f"{_Q13_FOREIGN}_pinned_to_source_commit", val),
        )
        assert rr.returncode != 0, f"pre-marker non-bool {tag} admitted"
        assert "Q13" in rr.stderr


def test_Q13_SELF_CHECK_NONSCALAR_IS_REJECTED_FAIL(tmp_path):
    """
    Q13a depth: REJECT non-scalar, do not recurse. `sc.items()` walks one level, so a
    pin-shaped key nested under self_check was never reached. Rejecting the non-scalar
    outer value closes all nesting shapes with no traversal to write or get wrong, and
    is fail-closed. Nested self_check would be a schema change with its own gate.
    """
    cases = {
        "nest_dict": lambda j: j["self_check"].update(
            {"nested": {f"pinned_to_{_Q13_FOREIGN}": True}}
        ),
        "nest_list": lambda j: j["self_check"].update(
            {"nested": [{f"pinned_to_{_Q13_FOREIGN}": True}]}
        ),
        "nest_two": lambda j: j["self_check"].update(
            {"a": {"b": {f"pinned_to_{_Q13_FOREIGN}": True}}}
        ),
        "plain_list": lambda j: j["self_check"].update({"some_list": [1, 2]}),
    }
    for tag, mut in cases.items():
        rr = _q13_case(tmp_path, f"nest_{tag}", mut)
        assert rr.returncode != 0, f"non-scalar self_check {tag} admitted"
        assert "Q13" in rr.stderr and "must be a scalar" in rr.stderr


def test_Q13_SCIENCE_CLAIM_NONBOOL_IS_MALFORMED_FAIL(tmp_path):
    """
    Q13b: the more consequential site. The review-risk tier turns on "no science
    claim", so a packet the validator cleared while a consumer read it as claiming one
    is a divergence between the validator's guarantee and a consumer's reading.
    """
    for tag, val in (
        ("int1", 1), ("float1", 1.0), ("str_true", "true"), ("str_True", "True"),
        ("list_yes", ["yes"]), ("str_false", "false"), ("int0", 0), ("null", None),
        ("str_empty", ""), ("dict_empty", {}), ("list_empty", []),
    ):
        rr = _q13_case(tmp_path, f"sci_{tag}", _q13_set_sci(val))
        assert rr.returncode != 0, f"non-bool science_claim {tag} admitted"
        assert "Q13" in rr.stderr and "science_claim" in rr.stderr


def test_Q13_SCIENCE_CLAIM_TRUE_STILL_FORBIDDEN(tmp_path):
    """Q13b control: the original prohibition must survive the type check."""
    rr = _q13_case(tmp_path, "sci_true", _q13_set_sci(True))
    assert rr.returncode != 0
    assert "science_claim true forbidden" in rr.stderr


def test_Q13_CONTROL_CLAIM_CEILING_INT0_STILL_REJECTED(tmp_path):
    """
    Q13 same-shape control: claim_ceiling already rejects a non-bool by identity
    (fail-closed direction). Pinning it here proves the surrounding house style is
    unchanged and that the two cured sites were the fail-OPEN ones.
    """
    def _m(j):
        cc = dict(j["claim_ceiling"])
        cc["LANDS_AB"] = 0
        j["claim_ceiling"] = cc

    rr = _q13_case(tmp_path, "cc_int0", _m)
    assert rr.returncode != 0
    assert "must be false" in rr.stderr


def test_Q13_POSITIVE_LEGITIMATE_NONCLAIMS_ACCEPTED(tmp_path):
    """
    Q13 positive (the row that decides the cure is not the mirror defect): a packet is
    ENTITLED to assert nothing. `pinned_to_<foreign>: false` says "not pinned to that
    head", and `science_claim: false` / absent say "no claim". A strictness cure that
    rejected these would be worse than the gap it closes.
    """
    for tag, mut in (
        ("pin_foreign_false", _q13_set_pin(f"pinned_to_{_Q13_FOREIGN}", False)),
        ("sci_false", _q13_set_sci(False)),
        ("sci_absent", lambda j: j.pop("science_claim", None)),
    ):
        rr = _q13_case(tmp_path, f"pos_{tag}", mut)
        assert rr.returncode == 0, f"legitimate non-claim {tag} rejected: {rr.stderr!r}"


def test_Q13_POSITION_ROWS_UNCHANGED_BY_STRICT_BOOL(tmp_path):
    """
    Q13 regression guard: the Q11/Q12 position x length rows must still raise after the
    strict-bool change, post- AND pre-marker. Without this, a future edit to the value
    check could short-circuit the whole-key scan and no test would notice.
    """
    for tag, key in (
        ("post7", "pinned_to_abcdef1"),
        ("post12", f"pinned_to_{_Q13_FOREIGN}"),
        ("post40", "pinned_to_" + "b" * 40),
        ("postmid", f"pinned_to_{_Q13_MIDSHA}"),
        ("pre12", f"{_Q13_FOREIGN}_pinned_to_source_commit"),
        ("premid", f"{_Q13_MIDSHA}_pinned_to_source_commit"),
    ):
        rr = _q13_case(tmp_path, f"posn_{tag}", _q13_set_pin(key, True))
        assert rr.returncode != 0, f"position row {tag} regressed to pass"
        assert "Q11" in rr.stderr


# --- Q14 / gate-2 findings F1 + F2 (round 9) -------------------------------------
# F1: the assertion contract covered only pin-shaped keys, so a non-pin scalar reached
#     `continue` before validation; and a non-dict `self_check` skipped the whole block
#     because the isinstance entry guard had no else-branch.
# F2: `deadbeefcafe_pinned_to` -- marker at the END, no trailing underscore -- failed the
#     admission predicate and was discharged unscanned while carrying a foreign head.
#
# F2's cure is ADMISSION-based (marker at any position), NOT the originally dispatched
# marker-at-start grammar: that grammar rejects `<src_prefix>_pinned_to_source_commit`,
# a legitimate claim and a frozen acceptance positive. Ratified 1785415760948-52adea6f.


def test_Q14_SELF_CHECK_NONDICT_IS_MALFORMED_FAIL(tmp_path):
    """F1a: a non-mapping self_check must raise, not silently skip the section."""
    for tag, val in (("list", [True]), ("str", "all_good"), ("int", 1)):
        rr = _q13_case(tmp_path, f"scnd_{tag}", lambda j, _v=val: j.__setitem__("self_check", _v))
        assert rr.returncode != 0, f"non-dict self_check ({tag}) skipped the block"
        assert "Q14" in rr.stderr and "must be a JSON object" in rr.stderr


def test_Q14_NONPIN_SCALAR_ASSERTIONS_MUST_BE_BOOL_FAIL(tmp_path):
    """
    F1b: the assertion contract covers the WHOLE mapping. The retired order hit
    `continue` for every non-pin key before `_require_packet_bool` ran, so a non-bool
    on a non-pin assertion key was admitted unvalidated.
    """
    for tag, val in (
        ("str_false", "false"), ("int0", 0), ("null", None), ("int1", 1), ("str_true", "true"),
    ):
        rr = _q13_case(
            tmp_path, f"np_{tag}", _q13_set_pin("expected_branch_classifier_determined", val)
        )
        assert rr.returncode != 0, f"non-pin non-bool {tag} admitted"
        assert "Q13" in rr.stderr and "assertion field" in rr.stderr


def test_Q14_MARKER_AT_END_IS_ADMITTED_AND_SCANNED_FAIL(tmp_path):
    """
    F2: marker-at-end keys must be ADMITTED and then judged on value. Asserting the
    `I1/Q11` class specifically is what distinguishes the ratified admission cure from a
    grammar rejection -- the key reaches the whole-key scan rather than being
    pattern-rejected, so the fix landed on the mechanism that failed.
    """
    for tag, key in (
        ("hex12", f"{_Q13_FOREIGN}_pinned_to"),
        ("hex40", "b" * 40 + "_pinned_to"),
        ("hex64", "c" * 64 + "_pinned_to"),
    ):
        rr = _q13_case(tmp_path, f"mend_{tag}", _q13_set_pin(key, True))
        assert rr.returncode != 0, f"marker-at-end {tag} passed unscanned"
        assert "Q11" in rr.stderr, f"{tag} was pattern-rejected, not scanned: {rr.stderr!r}"


def test_Q14_MARKER_ONLY_KEY_NAMES_NO_REFERENT_FAIL(tmp_path):
    """
    Option 2: a key that is ONLY the marker asserts pinning with no referent. Under the
    mapping-wide assertion contract, a claim naming nothing is malformed by the same
    logic that makes a non-bool value malformed. This closes the one row the superseded
    grammar would have caught, so the admission cure is strictly no weaker than it.
    """
    for tag, key in (
        ("bare", "pinned_to"), ("trailing", "pinned_to_"), ("padded", "__pinned_to__"),
    ):
        rr = _q13_case(tmp_path, f"monly_{tag}", _q13_set_pin(key, True))
        assert rr.returncode != 0, f"marker-only key {tag} admitted"
        assert "Q14" in rr.stderr and "names no referent" in rr.stderr


def test_Q14_POSITIVE_FROZEN_POSITIVES_AND_NONPIN_BOOLS_ACCEPTED(tmp_path):
    """
    Q14 positives, incl. the row that discriminated the two candidate cures:
    `<src[:12]>_pinned_to_source_commit` is a LEGITIMATE claim naming a true prefix of
    the source head. The superseded marker-at-start grammar rejected it, which would have
    made a frozen acceptance criterion unsatisfiable. Also pins the faithful packet's
    non-pin bools, which are the live path (26 of its 27 self_check entries).
    """
    for tag, mut in (
        ("canonical", _q13_set_pin("pinned_to_source_commit", True)),
        ("nonpin_true", _q13_set_pin("some_assertion", True)),
        ("nonpin_false", _q13_set_pin("some_assertion", False)),
        ("src_prefix12_pre_marker",
         _q13_set_pin(f"{_Q11_COMMIT[:12]}_pinned_to_source_commit", True)),
        ("pin_foreign_false", _q13_set_pin(f"pinned_to_{_Q13_FOREIGN}", False)),
    ):
        rr = _q13_case(tmp_path, f"q14pos_{tag}", mut)
        assert rr.returncode == 0, f"frozen positive {tag} regressed: {rr.stderr!r}"


def test_Q7_ORDERING_SPECIFIC_CLASS_STILL_WINS(tmp_path):
    """
    Q3' ordering guard: when a mutation violates a SPECIFIC check and also carries
    a foreign token, the specific class must own the diagnosis. If the catch-all
    ever moves back ahead of the specific checks, those checks silently stop being
    exercised even though the suite stays green.
    """
    commit = "c5" * 20
    rel = _faithful_rel_man(tmp_path)
    try:
        pkt_path = _packet_for_dry(tmp_path, rel, _sha(rel), commit)
        j = json.loads(pkt_path.read_text())
        rhs = dict(j["pins"]["runner_and_harness_shas"])
        rhs[sorted(rhs)[0]] = "0" * 64
        j["pins"]["runner_and_harness_shas"] = rhs
        bad = tmp_path / "pkt_q7ord.json"
        bad.write_text(json.dumps(j) + "\n")
        rr = _run_dry(bad, rel, commit)
        assert rr.returncode != 0
        assert "I2" in rr.stderr, f"specific I2 class lost to catch-all: {rr.stderr!r}"
    finally:
        if rel.exists():
            rel.unlink()
