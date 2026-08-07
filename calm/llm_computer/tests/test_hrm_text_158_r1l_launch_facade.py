"""Characterization battery for r1l_launch facade (CPU; 13 arms; zero-skip accept)."""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.r1l_launch.argv import (
    assert_suffix_equals_child,
    build_child_timeout_argv,
    build_run_phase_bash_c,
    build_watch_wrap_spawn_argv,
    render_argv_shell,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.classify import (
    TerminalObservation,
    classify_terminal,
    count_runner_pass,
    last_nonempty_line,
    parse_exit_rc,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.freeze_digest import (
    content_digest_from_members,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.materialize import (
    PHASE_ORDER,
    PhaseFilePreflightError,
    absolute_phase_paths,
    mint_phase_files,
    re_resolve_phase_manifest,
)

REPO = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158")
FIX = REPO / "tests/fixtures/r1l_launch"
MAN = FIX / "fixtures_manifest.json"
WATCH_WRAP = Path("/mnt/c/Users/gabes/projects/claw-code/bin/watch-wrap")
W6_PIN = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"
W6_PATH = Path(
    "/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/calm/hrm/checkpoints/"
    "hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_"
    "anchorsv1r3_from_L0b_final_step01500.pt"
)
INJECT_POINTS = (
    "attestation_exists",
    "attestation_write",
    "closure",
    "membership_extra",
    "membership_missing",
    "chmod",
    "mode_assert",
    "append",
)

ARM_RESULTS: dict[str, dict] = {}
W6_SUITE: dict[str, str] = {}
ACCEPT_STATE_PATH = Path(
    os.environ.get("R1L_FACADE_ACCEPT_STATE", str(Path("/tmp") / "r1l_facade_accept_state.json"))
)


def _load_manifest() -> dict:
    return json.loads(MAN.read_text())


def _fixture_shells() -> dict[str, str]:
    man = _load_manifest()
    out = {}
    for ph, rec in man["members"].items():
        stem = ph[:-3] if ph.endswith(".sh") else ph
        path = FIX / rec["path"]
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest() == rec["sha256"]
        assert len(data) == rec["bytes"]
        out[stem] = data.decode("utf-8")
    return out


def _record(arm: str, **kwargs):
    ARM_RESULTS[arm] = {"executed": True, "skipped": False, **kwargs}
    try:
        prev = {}
        if ACCEPT_STATE_PATH.is_file():
            prev = json.loads(ACCEPT_STATE_PATH.read_text())
        prev[arm] = ARM_RESULTS[arm]
        ACCEPT_STATE_PATH.write_text(json.dumps(prev, indent=2, sort_keys=True) + "\n")
    except Exception as exc:
        ARM_RESULTS[arm]["state_write_error"] = str(exc)


def _sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _require_w6() -> Path:
    assert W6_PATH.is_file() and not W6_PATH.is_symlink(), f"W6 missing: {W6_PATH}"
    got = _sha_file(W6_PATH)
    assert got == W6_PIN, f"W6 sha mismatch got={got} pin={W6_PIN}"
    return W6_PATH


@pytest.fixture(scope="module", autouse=True)
def _suite_w6_bracket():
    w6 = _require_w6()
    W6_SUITE["before"] = _sha_file(w6)
    assert W6_SUITE["before"] == W6_PIN
    yield
    W6_SUITE["after"] = _sha_file(w6)
    assert W6_SUITE["after"] == W6_PIN


def _make_dry_gate(td: Path):
    plan = td / "MAIN_LANE_REENTRY_STEP2_R1L_GPU_REMINT_PLAN_v13.json"
    plan.write_text('{"schema":"main_lane_reentry_step2_r1l_gpu_remint_plan/v13","dry":true}\n')
    md = td / "MAIN_LANE_REENTRY_STEP2_R1L_GPU_REMINT_PLAN_v13.md"
    md.write_text("# dry\n")
    members = {
        plan.name: {
            "sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
            "size": plan.stat().st_size,
        },
        md.name: {
            "sha256": hashlib.sha256(md.read_bytes()).hexdigest(),
            "size": md.stat().st_size,
        },
    }
    cd = content_digest_from_members({k: v["sha256"] for k, v in members.items()})
    gate = td / "gate_freeze_manifest.json"
    gate.write_text(
        json.dumps(
            {
                "schema": "r1l_gpu_plan_gate1_freeze_manifest/v1",
                "CONTENT_DIGEST": cd,
                "members": members,
                "gate1_verdict": "PASS",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    os.chmod(gate, 0o444)
    return {
        "plan": plan,
        "gate": gate,
        "plan_sha": members[plan.name]["sha256"],
        "gate_sha": hashlib.sha256(gate.read_bytes()).hexdigest(),
        "cd": cd,
    }


def _live_launch_source_sha() -> str:
    """Launch HEAD parameter for fixtures — live git HEAD of the HRM repo (40-hex)."""
    out = subprocess.check_output(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], text=True
    ).strip()
    assert len(out) == 40 and all(c in "0123456789abcdef" for c in out), out
    return out


def _base_env(root: Path, ev: Path, rlog: Path, gate_info: dict) -> dict:
    env = os.environ.copy()
    env.update(
        {
            "R1L_ROOT": str(root),
            "R1L_EV": str(ev),
            "R1L_RUNNER_LOG": str(rlog),
            "R1L_PLAN_JSON": str(gate_info["plan"]),
            "R1L_EXPECTED_PLAN_SHA256": gate_info["plan_sha"],
            "R1L_GATE1_FREEZE_MANIFEST_PATH": str(gate_info["gate"]),
            "R1L_EXPECTED_GATE1_FREEZE_MANIFEST_SHA256": gate_info["gate_sha"],
            "R1L_EXPECTED_CONTENT_DIGEST": gate_info["cd"],
            # Launch source is a parameter supplied by the materializer/test host —
            # fixtures no longer hardcode a commit (slice 1.5).
            "R1L_LAUNCH_SOURCE_COMMIT_SHA": _live_launch_source_sha(),
            "R1L_S2_MODE": "synthetic",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
        }
    )
    env.pop("R1L_S5_INJECT_FAIL", None)
    env.pop("R1L_S5_ALLOW_INJECT_BATTERY", None)
    return env


def _run_shell(shell_text: str, env: dict, timeout: int = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["bash", "-c", shell_text],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_s0b_launch_head_param_wrong_fails_right_passes():
    """Calibration: S0b live-HEAD assert still bites on wrong launch param; passes on live HEAD.

    Wrong value must fail closed (assertion not a no-op). Right value is live git HEAD.
    P1 freeze head stays distinct (0636177f…) and is not what this assert compares.
    """
    shells = _fixture_shells()
    s0b = shells["S0b"]
    live = _live_launch_source_sha()
    wrong = "0" * 40
    assert wrong != live
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gate_info = _make_dry_gate(td_path)
        # wrong launch source
        env_bad = _base_env(td_path / "root_bad", td_path / "ev_bad", td_path / "log_bad", gate_info)
        env_bad["R1L_LAUNCH_SOURCE_COMMIT_SHA"] = wrong
        r_bad = _run_shell(s0b, env_bad, timeout=120)
        blob_bad = r_bad.stdout + r_bad.stderr
        assert r_bad.returncode != 0, blob_bad[-500:]
        assert "head_mismatch" in blob_bad or "R1L_LAUNCH_SOURCE_COMMIT_SHA" in blob_bad
        # right launch source
        env_ok = _base_env(td_path / "root_ok", td_path / "ev_ok", td_path / "log_ok", gate_info)
        assert env_ok["R1L_LAUNCH_SOURCE_COMMIT_SHA"] == live
        r_ok = _run_shell(s0b, env_ok, timeout=120)
        blob_ok = r_ok.stdout + r_ok.stderr
        assert r_ok.returncode == 0, blob_ok[-800:]
        assert "S0b_AUTHORITY_AND_DATA_PREFLIGHT_OK" in blob_ok
        assert live in blob_ok
    _record(
        "s0b_launch_head_param_calibration",
        ok=True,
        live_launch_source=live,
        wrong_rejected=True,
        right_passed=True,
    )


def test_argv_from_script_derives_from_r1l_root_not_stale_literal():
    """Calibration: S2 mint + S3 equality authority both use supplied R1L_ROOT.

    Run root path deliberately avoids sha 0636177f so a stale
    r1l_launch_HEAD_0636177f_r1 literal cannot match by coincidence.
    Two wrongs that agree (hardcoded list in S2 and S3) is the defect this catches.
    """
    shells = _fixture_shells()
    w6 = _require_w6()
    w6_before = _sha_file(w6)
    assert w6_before == W6_PIN
    with tempfile.TemporaryDirectory(prefix="r1l_runroot_deadbeef_") as td:
        td_path = Path(td)
        # temp roots never contain the stale launch-root token
        assert "r1l_launch_HEAD_" not in str(td_path)
        assert "0636177f" not in str(td_path)
        gate_info = _make_dry_gate(td_path)
        root = td_path / "root_deadbeef_not_stale"
        env = _base_env(root, td_path / "ev", td_path / "runner.log", gate_info)
        env["R1L_S2_MODE"] = "synthetic"
        for ph in ("S0", "S0b", "S1"):
            r = _run_shell(shells[ph], env, timeout=600)
            assert r.returncode == 0, (ph, (r.stdout + r.stderr)[-800:])
        r2 = _run_shell(shells["S2"], env, timeout=300)
        assert r2.returncode == 0, (r2.stdout + r2.stderr)[-800:]
        receipt_path = root / "receipts" / "r1l_launch_runtime_receipt.json"
        assert receipt_path.is_file()
        receipt = json.loads(receipt_path.read_text())
        argv = list(receipt.get("proof_command_argv") or [])
        assert argv, "proof_command_argv missing"
        # minted argv must bind the supplied root — not a stale launch-root literal
        assert all("r1l_launch_HEAD_" not in str(a) for a in argv), argv[:4]
        assert str(root / "code" / "scripts" / "train_hrm_text_158.py") == argv[0]
        assert "--load-from" in argv
        load_idx = argv.index("--load-from")
        assert str(argv[load_idx + 1]).startswith(str(root)), (
            "load-from not under supplied R1L_ROOT",
            argv[load_idx + 1],
            str(root),
        )
        # S3 equality authority must accept that argv (same derivation from ROOT)
        r3 = _run_shell(shells["S3"], env, timeout=300)
        blob3 = r3.stdout + r3.stderr
        assert r3.returncode == 0, blob3[-800:]
        assert "S3_ARGV_EQUALITY_OK" in blob3
        assert "ARGV_MISMATCH" not in blob3
        log = Path(env["R1L_RUNNER_LOG"]).read_text()
        assert "R1L_TERMINAL_PASS" in log
        assert "R1L_TERMINAL_FAIL" not in log
    assert _sha_file(w6) == w6_before == W6_PIN
    _record(
        "argv_from_script_root_derived_calibration",
        ok=True,
        root_token="deadbeef",
        minted_argv0_under_root=True,
        s3_equality_ok=True,
        no_stale_launch_root_literal=True,
    )


def test_00_fixture_digest_and_drift():
    man = _load_manifest()
    assert man["FIXTURE_CONTENT_DIGEST"] == "30f545ccdc80c30d1e3ccd12c6f886b397d4a97ce6f78006ac57f62ca9a1d60f"
    assert man["source_plan_sha256"] == "0b8fa3805ee4d6e5ace8b311bc9a3928452be63fd63e7572f1d9a370df466531"
    flat = {k: v["sha256"] for k, v in man["members"].items()}
    assert content_digest_from_members(flat) == man["FIXTURE_CONTENT_DIGEST"]
    bad = dict(flat)
    bad["S0.sh"] = "0" * 64
    assert content_digest_from_members(bad) != man["FIXTURE_CONTENT_DIGEST"]
    _record("fixture_digest", ok=True, mode="fixture_sha_check")


def test_arm12_stale_literal_sweep_over_fixture_shells():
    shells = _fixture_shells()
    hits = []
    for ph, text in shells.items():
        for lit in ("success_stop_on", "failure_stop_on", "double-hash"):
            if lit in text:
                hits.append((ph, lit))
    assert not any(h[1] in ("success_stop_on", "failure_stop_on") for h in hits)
    _record("arm12_stale_sweep", ok=True, hits=hits, mode="fixture_text_scan")


# Value-keyed: launch-root path class (any hex), not a single sha and not arm12's stop-on set.
LAUNCH_ROOT_LITERAL_RE = re.compile(r"r1l_launch_HEAD_[0-9a-f]+")


def _launch_root_literal_hits(shells: dict[str, str]) -> list[tuple[str, int, str]]:
    """Enumerate (phase, line_no, token) for launch-root path literals in phase texts."""
    hits: list[tuple[str, int, str]] = []
    for ph, text in shells.items():
        for i, line in enumerate(text.splitlines(), 1):
            for m in LAUNCH_ROOT_LITERAL_RE.finditer(line):
                hits.append((ph, i, m.group(0)))
    return hits


def test_fixture_corpus_has_zero_launch_root_path_literals():
    """Regression guard: no phase may embed r1l_launch_HEAD_<hex> launch-root paths.

    Distinct from arm12 (success_stop_on / failure_stop_on / double-hash). Keyed on the
    property — any hex — so re-baking a different sha is the same defect.
    """
    shells = _fixture_shells()
    hits = _launch_root_literal_hits(shells)
    assert hits == [], hits
    # denominator: all seven phases scanned
    assert set(shells.keys()) == set(PHASE_ORDER)
    _record(
        "launch_root_literal_corpus_sweep",
        ok=True,
        hits=hits,
        phases_scanned=sorted(shells.keys()),
        mode="value_keyed_launch_root_path_scan",
    )


def test_launch_root_literal_matcher_discriminates_known_bad():
    """Both directions: matcher silent on live corpus; fires on pre-correction S2/S3 bytes."""
    # known-good (live)
    live = _fixture_shells()
    assert _launch_root_literal_hits(live) == []

    # known-bad: pre-correction fixture bytes at slice-1 HEAD (fc81ae12) still had the class
    bad_s2 = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            "fc81ae12:tests/fixtures/r1l_launch/phases/S2.sh",
        ],
        text=True,
    )
    bad_s3 = subprocess.check_output(
        [
            "git",
            "-C",
            str(REPO),
            "show",
            "fc81ae12:tests/fixtures/r1l_launch/phases/S3.sh",
        ],
        text=True,
    )
    assert LAUNCH_ROOT_LITERAL_RE.search(bad_s2), "known-bad S2 must contain the class"
    assert LAUNCH_ROOT_LITERAL_RE.search(bad_s3), "known-bad S3 must contain the class"
    bad_shells = dict(live)
    bad_shells["S2"] = bad_s2
    bad_shells["S3"] = bad_s3
    bad_hits = _launch_root_literal_hits(bad_shells)
    assert bad_hits, "matcher must fire on known-bad reintroduction"
    assert any(h[0] == "S2" for h in bad_hits)
    assert any(h[0] == "S3" for h in bad_hits)
    # property is any hex, not only 0636177f — synthetic re-bake still matches
    synthetic = "ARGV = ['/tmp/r1l_launch_HEAD_deadbeef00aaaaaaaaaaaaaaaaaaaaaaaa/code/x.py']\n"
    assert LAUNCH_ROOT_LITERAL_RE.search(synthetic)
    assert _launch_root_literal_hits({"SX": synthetic}) == [
        ("SX", 1, "r1l_launch_HEAD_deadbeef00aaaaaaaaaaaaaaaaaaaaaaaa")
    ]
    _record(
        "launch_root_literal_matcher_calibration",
        ok=True,
        live_silent=True,
        known_bad_fires=True,
        any_hex_not_single_sha=True,
        mode="matcher_both_directions",
    )


def test_arm13_launch_argv_list_authority():
    shells = _fixture_shells()
    with tempfile.TemporaryDirectory() as td:
        man = mint_phase_files(Path(td) / "phases", shells)
        paths = absolute_phase_paths(man)
        budgets = {ph: 120 for ph in PHASE_ORDER}
        body = build_run_phase_bash_c(paths, budgets, list(PHASE_ORDER))
        child = build_child_timeout_argv(outer_timeout_s=3095, kill_after_s=60, bash_c_body=body)
        mon = build_watch_wrap_spawn_argv(child, watch_wrap_path=str(WATCH_WRAP))
        assert_suffix_equals_child(mon, child)
        assert "--log" not in mon and "--stop-on" not in mon
        assert "bash -c '<" not in render_argv_shell(mon)
        re_resolve_phase_manifest(man)
    _record("arm13_argv", ok=True, mode="argv_list_construct")


def test_arm08_phase_tampered_preflight():
    shells = _fixture_shells()
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        man = mint_phase_files(phases, shells)
        p = Path(man["members"]["S3"]["path"])
        os.chmod(phases, 0o755)
        os.chmod(p, 0o644)
        p.write_bytes(p.read_bytes() + b"\n#tamper\n")
        os.chmod(p, 0o444)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(man)
        assert "hash mismatch" in str(ei.value)
    _record("arm08_tamper", ok=True, mode="real_frozen_shell_bytes_mint_mutate")


def test_arm09_phase_missing():
    shells = _fixture_shells()
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        man = mint_phase_files(phases, shells)
        p = Path(man["members"]["S1"]["path"])
        os.chmod(phases, 0o755)
        os.chmod(p, 0o644)
        p.unlink()
        with pytest.raises(PhaseFilePreflightError):
            re_resolve_phase_manifest(man)
    _record("arm09_missing", ok=True, mode="real_frozen_shell_bytes_mint_unlink")


def test_arm11_foreign_file_at_manifest_path():
    shells = _fixture_shells()
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        man = mint_phase_files(phases, shells)
        p = Path(man["members"]["S2"]["path"])
        os.chmod(phases, 0o755)
        os.chmod(p, 0o644)
        p.write_bytes(b"foreign-content-not-in-manifest\n")
        os.chmod(p, 0o444)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(man)
        assert "hash mismatch" in str(ei.value)
    _record("arm11_foreign", ok=True, mode="real_frozen_shell_bytes_mint_replace")


def test_arm10_phase_budget_breach():
    assert WATCH_WRAP.is_file()
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        man = mint_phase_files(phases, {"S0": "#!/usr/bin/env bash\nset -euo pipefail\nsleep 5\n"}, phases=("S0",))
        paths = absolute_phase_paths(man)
        body = build_run_phase_bash_c(paths, {"S0": 1}, ["S0"])
        child = build_child_timeout_argv(outer_timeout_s=10, kill_after_s=2, bash_c_body=body)
        mon = build_watch_wrap_spawn_argv(child, watch_wrap_path=str(WATCH_WRAP), heartbeat=5, replay=5)
        r = subprocess.run(mon, capture_output=True, text=True, timeout=30)
        out = r.stdout + r.stderr
        assert "PHASE_BUDGET_BREACH" in out
        assert r.returncode != 0
        obs = TerminalObservation(
            exit_rc=parse_exit_rc(out) or r.returncode,
            runner_pass_count=0,
            last_nonempty_line=None,
            actual_log_sha256=None,
            projected_log_sha256=None,
            stderr_text=out,
            phase_file_preflight_ok=True,
        )
        assert classify_terminal(obs).fail_class == "PHASE_BUDGET_BREACH"
    _record("arm10_budget_breach", ok=True, mode="reduced_child", watchwrap_rc=r.returncode)


def test_arm03_stop_on_hazard():
    assert WATCH_WRAP.is_file()
    with tempfile.TemporaryDirectory() as td:
        child = Path(td) / "c.sh"
        child.write_text("#!/usr/bin/env bash\nset -euo pipefail\necho RUNNER_PASS\nsleep 30\n")
        child.chmod(0o755)
        cmd = [
            str(WATCH_WRAP), "--heartbeat", "5", "--success", "RUNNER_PASS",
            "--stop-on", "RUNNER_PASS", "--replay", "5", "--", "bash", str(child),
        ]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
        out = r.stdout + r.stderr
        assert ("stop-triggered" in out) or ("STOP-TRIGGER" in out)
        assert r.returncode == 0
    _record("arm03_stopon", ok=True, mode="reduced_child", wrapper_rc=r.returncode)


def test_arm02_orphan_self_bound():
    assert WATCH_WRAP.is_file()
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        pidfile = td_path / "child.pid"
        child = td_path / "child.sh"
        child.write_text(
            "#!/usr/bin/env bash\nset -euo pipefail\n"
            f"echo $$ > {pidfile}\nwhile true; do sleep 0.2; done\n"
        )
        child.chmod(0o755)
        outer = subprocess.run(
            [
                "timeout", "--foreground", "-s", "KILL", "1",
                str(WATCH_WRAP), "--heartbeat", "1", "--replay", "5", "--",
                "timeout", "--signal=TERM", "--kill-after=1", "3",
                "bash", str(child),
            ],
            capture_output=True, text=True, timeout=30,
        )
        import time
        time.sleep(5)
        alive = None
        if pidfile.exists():
            try:
                os.kill(int(pidfile.read_text().strip()), 0)
                alive = True
            except (ProcessLookupError, ValueError):
                alive = False
            except PermissionError:
                alive = True
        assert alive is False
    _record("arm02_orphan", ok=True, mode="reduced_child", outer_rc=outer.returncode, child_alive=alive)


def _extract_s5_append_region(s5_text: str) -> str:
    """Contiguous frozen S5 finalize region: Sole-success comment through final exit $arc.

    Includes the tee -a RUNNER_FAIL line that a reimplementation would omit.
    """
    start = s5_text.find("# Sole success authority")
    assert start >= 0, "frozen S5 missing Sole success authority anchor"
    # last exit $arc in file (final line of finalize)
    end = s5_text.rfind("exit $arc")
    assert end >= start
    end = end + len("exit $arc")
    if end < len(s5_text) and s5_text[end] == "\n":
        end += 1
    region = s5_text[start:end]
    assert 'echo "RUNNER_PASS" >> "$LOG"' in region
    assert 'tee -a "$LOG"' in region
    assert "TERMINAL_MARKER_UNWRITABLE" in region
    assert region.rstrip().endswith("exit $arc")
    # complete if/fi pairs for append inject + arc failure
    assert region.count("if [") == region.count("fi")
    return region


def test_arm01_append_unwritable():
    shells = _fixture_shells()
    s5 = shells["S5"]
    region = _extract_s5_append_region(s5)
    region_sha = hashlib.sha256(region.encode()).hexdigest()
    # re-extract and prove stable byte identity against fixture S5
    region2 = _extract_s5_append_region(s5)
    assert region2 == region
    extraction_form = (
        "contiguous_extract_from_fixture_S5: from inject-append block through exit $arc; "
        "includes tee -a RUNNER_FAIL line; only LOG= binding substituted"
    )
    with tempfile.TemporaryDirectory() as td:
        log = Path(td) / "runner.log"
        log.write_text("pre\n")
        # Make unwritable via inject path inside extracted region (chmod a-w) —
        # also chmod now so even without inject env the append fails.
        os.chmod(log, 0o444)
        child = Path(td) / "append_child.sh"
        # Substitute only LOG binding; execute frozen region bytes.
        body = (
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            f"LOG={json.dumps(str(log))}\n"
            "set +e\n"
            f"{region}\n"
        )
        child.write_text(body)
        child.chmod(0o755)
        # Also set inject=append so extracted inject branch runs chmod a-w
        env = os.environ.copy()
        env["R1L_S5_INJECT_FAIL"] = "append"
        mon = build_watch_wrap_spawn_argv(
            ["bash", str(child)], watch_wrap_path=str(WATCH_WRAP), heartbeat=5, replay=5
        )
        r = subprocess.run(mon, capture_output=True, text=True, timeout=20, env=env)
        out = r.stdout + r.stderr
        assert r.returncode != 0
        assert "S5_TERMINAL_MARKER_UNWRITABLE" in out or "TERMINAL_MARKER_UNWRITABLE" in out
        assert count_runner_pass(log.read_text()) == 0
        # classifier may need digests absent → TERMINAL_MARKER path via stderr
        obs = TerminalObservation(
            exit_rc=parse_exit_rc(out) or r.returncode,
            runner_pass_count=0,
            last_nonempty_line=last_nonempty_line(log.read_text()),
            actual_log_sha256=None,
            projected_log_sha256=None,
            stderr_text=out,
            phase_file_preflight_ok=True,  # not under test here; marker path
        )
        # With digests None, fail-closed would be DIGEST_ABSENT if exit 0;
        # we have nonzero + marker text → TERMINAL_MARKER_UNWRITABLE first
        v = classify_terminal(obs)
        assert v.status == "FAIL"
        assert v.fail_class in ("TERMINAL_MARKER_UNWRITABLE", "WATCH_WRAP_CHILD_NONZERO_EXIT")
    _record(
        "arm01_append",
        ok=True,
        mode="reduced_child_extracted_fixture_bytes",
        extraction_form=extraction_form,
        extracted_region_sha256=region_sha,
        fixture_s5_sha=_load_manifest()["members"]["S5.sh"]["sha256"],
        includes_tee_runner_fail_line=True,
    )


def test_arm05_s0_inject_allow_refusals_execute_shell():
    shells = _fixture_shells()
    s0 = shells["S0"]
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        gate_info = _make_dry_gate(td_path)
        for label, extra in [
            ("inject", {"R1L_S5_INJECT_FAIL": "append"}),
            ("allow", {"R1L_S5_ALLOW_INJECT_BATTERY": "1"}),
            ("both", {"R1L_S5_INJECT_FAIL": "append", "R1L_S5_ALLOW_INJECT_BATTERY": "1"}),
        ]:
            root = td_path / f"root_{label}"
            env = _base_env(root, td_path / f"ev_{label}", td_path / f"log_{label}", gate_info)
            env.update(extra)
            r = _run_shell(s0, env, timeout=60)
            assert r.returncode != 0, (label, r.stdout, r.stderr)
            assert not root.exists(), label
            blob = r.stdout + r.stderr
            if label == "inject":
                assert "S0_INJECT_FAIL_SEAM_SET" in blob or r.returncode == 97
            if label == "allow":
                assert "S0_ALLOW_INJECT_BATTERY_SEAM_SET" in blob or r.returncode == 97
    _record("arm05_s0_refusals", ok=True, mode="real_frozen_shell_bytes_executed")


def test_arm07_synthetic_fail_executes_s2_shell():
    """CPU synthetic_fail path: real S0-S2 fixture shells; no GPU claim."""
    shells = _fixture_shells()
    w6 = _require_w6()
    w6_before = _sha_file(w6)
    assert w6_before == W6_PIN
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        try:
            gate_info = _make_dry_gate(td_path)
            root = td_path / "root"
            env = _base_env(root, td_path / "ev", td_path / "runner.log", gate_info)
            # S0 success (inject unset)
            r0 = _run_shell(shells["S0"], env, timeout=120)
            assert r0.returncode == 0, r0.stdout + r0.stderr
            assert root.is_dir()
            # S0b needs p1 authority files on disk — if fails, still prove S2 synthetic_fail
            # when S0-S1 can run; S0b may require creditdir pins.
            r0b = _run_shell(shells["S0b"], env, timeout=120)
            if r0b.returncode != 0:
                # still execute S2 synthetic_fail against a minimal env if possible after S0 only
                # S2 needs launch_env etc from S1 — cannot skip to S2 without S1
                _record(
                    "arm07_synthetic_fail_fixture_contract",
                    ok=False,
                    mode="real_frozen_shell_bytes_partial",
                    s0b_rc=r0b.returncode,
                    s0b_err=(r0b.stdout + r0b.stderr)[-500:],
                )
                raise AssertionError(f"S0b failed; cannot reach S2 synthetic_fail: {(r0b.stdout+r0b.stderr)[-500:]}")
            r1 = _run_shell(shells["S1"], env, timeout=300)
            assert r1.returncode == 0, (r1.stdout + r1.stderr)[-800:]
            env["R1L_S2_MODE"] = "synthetic_fail"
            r2 = _run_shell(shells["S2"], env, timeout=300)
            assert r2.returncode != 0
            log = Path(env["R1L_RUNNER_LOG"]).read_text() if Path(env["R1L_RUNNER_LOG"]).exists() else ""
            assert count_runner_pass(log) == 0
            assert "RUNNER_FAIL" in log or r2.returncode != 0
        finally:
            # teardown any W6 copy under root if present
            for p in (td_path / "root").rglob("*.pt") if (td_path / "root").exists() else []:
                try:
                    os.chmod(p, 0o644)
                    p.unlink()
                except OSError:
                    pass
            w6_after = _sha_file(w6)
            assert w6_after == w6_before == W6_PIN
    _record(
        "arm07_synthetic_fail_fixture_contract",
        ok=True,
        mode="real_frozen_shell_bytes_executed",
        claims_gpu=False,
        w6_before=w6_before,
        w6_after=w6_before,
    )


def test_arm04_inject_battery_and_arm06_s5_entry_execute_shells():
    """Real frozen shells S0-S5; inject 8/8 + S5 entry refusal; W6 pin enforced."""
    shells = _fixture_shells()
    w6 = _require_w6()
    w6_before = _sha_file(w6)
    assert w6_before == W6_PIN
    inject_results = {}
    s5_entry = {}
    ephemeral_pt_path = None
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        try:
            # --- arm 6: S0-S4 clean then S5 with inject set WITHOUT allow ---
            gate_info = _make_dry_gate(td_path)
            root = td_path / "root_entry"
            env = _base_env(root, td_path / "ev_entry", td_path / "log_entry", gate_info)
            for ph in ("S0", "S0b", "S1", "S2", "S3", "S4"):
                r = _run_shell(shells[ph], env, timeout=600)
                assert r.returncode == 0, (ph, (r.stdout + r.stderr)[-800:])
            # W6 copy should exist under root after S1
            pts = list(root.rglob("*.pt"))
            assert pts, "expected W6 copy under ROOT after S1"
            ephemeral_pt_path = pts[0]
            assert _sha_file(ephemeral_pt_path) == W6_PIN
            # S5 entry refuse
            env6 = dict(env)
            env6["R1L_S5_INJECT_FAIL"] = "closure"
            # no ALLOW
            r5e = _run_shell(shells["S5"], env6, timeout=300)
            s5_entry = {
                "rc": r5e.returncode,
                "runner_pass_count": count_runner_pass(
                    Path(env6["R1L_RUNNER_LOG"]).read_text()
                    if Path(env6["R1L_RUNNER_LOG"]).exists()
                    else ""
                ),
            }
            assert r5e.returncode != 0
            assert s5_entry["runner_pass_count"] == 0
            assert (
                "S5_INJECT_FAIL_SEAM_SET_AT_ENTRY" in (r5e.stdout + r5e.stderr)
                or r5e.returncode == 97
            )

            # --- arm 4: inject battery 8/8 with ALLOW, each on fresh chain ---
            for point in INJECT_POINTS:
                gdir = td_path / f"g_{point}"
                gdir.mkdir(parents=True, exist_ok=True)
                gate_i = _make_dry_gate(gdir)
                root_i = td_path / f"root_{point}"
                env_i = _base_env(
                    root_i, td_path / f"ev_{point}", td_path / f"log_{point}", gate_i
                )
                ok = True
                for ph in ("S0", "S0b", "S1", "S2", "S3", "S4"):
                    rr = _run_shell(shells[ph], env_i, timeout=600)
                    if rr.returncode != 0:
                        ok = False
                        inject_results[point] = {
                            "setup_fail": ph,
                            "rc": rr.returncode,
                            "err": (rr.stdout + rr.stderr)[-300:],
                        }
                        break
                if not ok:
                    continue
                # baseline pre-S5
                log_i = Path(env_i["R1L_RUNNER_LOG"])
                baseline = count_runner_pass(log_i.read_text() if log_i.exists() else "")
                assert baseline == 0
                env_i["R1L_S5_INJECT_FAIL"] = point
                env_i["R1L_S5_ALLOW_INJECT_BATTERY"] = "1"
                # append inject makes log unwritable — ensure set +e path still fails
                rs = _run_shell(shells["S5"], env_i, timeout=300)
                txt = log_i.read_text() if log_i.exists() else ""
                inject_results[point] = {
                    "rc": rs.returncode,
                    "runner_pass_count": count_runner_pass(txt),
                    "baseline_pre_s5": baseline,
                }
                assert rs.returncode != 0, point
                assert count_runner_pass(txt) == 0, point

            assert len(inject_results) == 8
            assert all(v.get("rc", 0) != 0 for v in inject_results.values())
            assert all(v.get("runner_pass_count", 1) == 0 for v in inject_results.values())

            # injected-failure teardown proof on W6 copy: force failure then finally remove
            fail_copy = td_path / "w6_fail_copy.pt"
            try:
                shutil.copy2(w6, fail_copy)
                os.chmod(fail_copy, 0o444)
                raise RuntimeError("injected_failure_after_w6_copy")
            except RuntimeError:
                pass
            finally:
                if fail_copy.exists():
                    os.chmod(fail_copy, 0o644)
                    fail_copy.unlink()
            assert not fail_copy.exists()
        finally:
            w6_after = _sha_file(w6)
            assert w6_after == w6_before == W6_PIN
            # cleanup ROOT copies if any remain
            for p in td_path.rglob("*.pt"):
                try:
                    os.chmod(p, 0o644)
                    p.unlink()
                except OSError:
                    pass

    _record(
        "arm04_inject_and_arm06_s5_entry",
        ok=True,
        mode="real_frozen_shell_bytes_executed",
        inject_results=inject_results,
        s5_entry=s5_entry,
        w6_before=w6_before,
        w6_after=w6_before,
        w6_path=str(W6_PATH),
        teardown_on_inject_w6_copy=True,
    )


def test_teardown_on_injected_failure_and_git_denom():
    w6 = _require_w6()
    before = _sha_file(w6)
    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        copy = td_path / "ephemeral_w6.pt"
        try:
            shutil.copy2(w6, copy)
            os.chmod(copy, 0o444)
            raise RuntimeError("injected_failure")
        except RuntimeError:
            pass
        finally:
            if copy.exists():
                os.chmod(copy, 0o644)
                copy.unlink()
        assert not copy.exists()
        assert _sha_file(w6) == before == W6_PIN
    allowed = [
        "calm/hrm_text_158/native_full_stack/r1l_launch",
        "calm/llm_computer/tests/test_hrm_text_158_r1l_launch_facade.py",
        "calm/hrm_text_158/tests/test_r1l_launch_classify_budget_argv.py",
        "tests/fixtures/r1l_launch",
    ]
    paths = []
    for g in allowed:
        p = REPO / g
        if p.is_file():
            paths.append(p)
        elif p.is_dir():
            paths.extend([x for x in p.rglob("*") if x.is_file() and "__pycache__" not in str(x)])
    pt_hits = [p for p in paths if p.suffix == ".pt"]
    assert not pt_hits
    _record(
        "teardown_and_git_denom",
        ok=True,
        denom=len(paths),
        pt_count=0,
        teardown_on_inject=True,
        w6_suite=W6_SUITE,
        mode="w6_copy_finally_unlink",
    )


def test_zz_accept_table_13_of_13():
    required = [
        "arm01_append",
        "arm02_orphan",
        "arm03_stopon",
        "arm04_inject_and_arm06_s5_entry",
        "arm05_s0_refusals",
        "arm07_synthetic_fail_fixture_contract",
        "arm08_tamper",
        "arm09_missing",
        "arm10_budget_breach",
        "arm11_foreign",
        "arm12_stale_sweep",
        "arm13_argv",
        "teardown_and_git_denom",
    ]
    merged = {}
    if ACCEPT_STATE_PATH.is_file():
        merged.update(json.loads(ACCEPT_STATE_PATH.read_text()))
    merged.update(ARM_RESULTS)
    missing = [a for a in required if a not in merged or not merged[a].get("executed")]
    assert not missing, missing
    skipped = [a for a, v in merged.items() if v.get("skipped")]
    assert not skipped, skipped
    # modes must not be vacuous for 4/6
    a4 = merged["arm04_inject_and_arm06_s5_entry"]
    assert a4.get("mode") == "real_frozen_shell_bytes_executed"
    assert a4.get("w6_before") == W6_PIN
    assert len(a4.get("inject_results", {})) == 8
    print("ACCEPT_TABLE", json.dumps({k: merged[k].get("mode") for k in required if k in merged}))
