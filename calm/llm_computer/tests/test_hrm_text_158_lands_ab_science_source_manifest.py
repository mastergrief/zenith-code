"""PLAN_v6 Phase A: science source manifest + packet dry-exec hostiles (v3 faithful)."""
from __future__ import annotations

import copy
import hashlib
import json
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
    """Faithful positive: real packet_v6 bytes + ONLY preregistered v7 binding updates."""
    base = json.loads(PACKET_V6.read_text(encoding="utf-8"))
    pkt = copy.deepcopy(base)
    pkt["schema"] = "LANDS_AB_EVAL_launch_packet/v7_dry_fixture"
    pkt["science_source_manifest_path"] = man_path.relative_to(REPO).as_posix()
    pkt["science_source_manifest_sha256"] = man_sha
    pkt["source_commit_sha"] = commit
    pkt["generator_script_path"] = "scripts/lands_ab_science_source_manifest.py"
    pkt["generator_script_sha256"] = _sha(GEN)
    pkt["dry_exec_tool_path"] = "scripts/lands_ab_packet_dry_exec.py"
    pkt["dry_exec_tool_sha256"] = _sha(DRY)
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
    ex = dict(pkt.get("executor") or {})
    ex["role"] = "claude_as_test_operator"
    ex.setdefault(
        "forbidden_for_plan_dev",
        ["formal 7-row matrix execution", "terminal receipt mint under FORMAL_RUNTIME_CREATE"],
    )
    ex["one_terminal_receipt"] = True
    pkt["executor"] = ex
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
