"""Pure-unit characterization for r1l_launch classify/budget/argv/freeze_digest."""
from __future__ import annotations

import hashlib
import shlex

import pytest

from calm.hrm_text_158.native_full_stack.r1l_launch.argv import (
    assert_suffix_equals_child,
    build_child_timeout_argv,
    build_run_phase_bash_c,
    build_watch_wrap_spawn_argv,
    render_argv_shell,
)
from calm.hrm_text_158.native_full_stack.r1l_launch.budget import (
    MONITOR_TIMEOUT_MS_MAX,
    derive_budget_plan,
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
    FROZEN_FIXTURE_CONTENT_DIGEST,
    PHASE_ORDER,
    mint_phase_files,
    re_resolve_phase_manifest,
    PhaseFilePreflightError,
)
import json
import os
import tempfile
from pathlib import Path


def test_budget_plan_r1l_defaults():
    per = {"S0": 120, "S0b": 120, "S1": 600, "S2": 1800, "S3": 180, "S4": 120, "S5": 120}
    bp = derive_budget_plan(per, orch_margin_s=35, kill_after_s=60, monitor_timeout_ms=3_300_000)
    assert bp.total_seconds == 3060
    assert bp.outer_timeout_seconds == 3095
    assert bp.child_wall_bound_seconds == 3155
    assert bp.monitor_timeout_ms == 3_300_000
    assert bp.monitor_timeout_ms <= MONITOR_TIMEOUT_MS_MAX
    assert bp.monitor_timeout_ms > bp.child_wall_bound_seconds * 1000


def test_budget_rejects_monitor_not_above_child_wall():
    with pytest.raises(ValueError):
        derive_budget_plan({"A": 100}, orch_margin_s=0, kill_after_s=0, monitor_timeout_ms=50_000)


def test_content_digest_sorted_nul_raw32():
    members = {
        "b.sh": hashlib.sha256(b"b").hexdigest(),
        "a.sh": hashlib.sha256(b"a").hexdigest(),
    }
    # manual
    parts = []
    for name in sorted(members):
        parts.append(name.encode() + b"\0" + bytes.fromhex(members[name]))
    exp = hashlib.sha256(b"".join(parts)).hexdigest()
    assert content_digest_from_members(members) == exp


def test_classify_pass_exit0_runner_pass():
    log = "x\nRUNNER_PASS\n"
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=count_runner_pass(log),
        last_nonempty_line=last_nonempty_line(log),
        actual_log_sha256="aa",
        projected_log_sha256="aa",
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "PASS"
    assert v.fail_class is None


def test_classify_fail_closed_absent_actual_digest():
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=1,
        last_nonempty_line="RUNNER_PASS",
        actual_log_sha256=None,
        projected_log_sha256="aa",
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "TERMINAL_LOG_DIGEST_ABSENT"


def test_classify_fail_closed_absent_projected_digest():
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=1,
        last_nonempty_line="RUNNER_PASS",
        actual_log_sha256="aa",
        projected_log_sha256=None,
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "TERMINAL_LOG_DIGEST_ABSENT"


def test_classify_fail_closed_preflight_none():
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=1,
        last_nonempty_line="RUNNER_PASS",
        actual_log_sha256="aa",
        projected_log_sha256="aa",
        phase_file_preflight_ok=None,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "PHASE_FILE_PREFLIGHT_ABSENT"


def test_classify_digest_mismatch_still_fails():
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=1,
        last_nonempty_line="RUNNER_PASS",
        actual_log_sha256="aa",
        projected_log_sha256="bb",
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "TERMINAL_LOG_VERIFY_FAIL"


def test_classify_phase_budget_breach():
    obs = TerminalObservation(
        exit_rc=124,
        runner_pass_count=0,
        last_nonempty_line=None,
        actual_log_sha256=None,
        projected_log_sha256=None,
        stderr_text="PHASE_BUDGET_BREACH phase=S0 budget=1 rc=124\n",
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "PHASE_BUDGET_BREACH"


def test_classify_terminal_marker_unwritable():
    obs = TerminalObservation(
        exit_rc=1,
        runner_pass_count=0,
        last_nonempty_line=None,
        actual_log_sha256=None,
        projected_log_sha256=None,
        stderr_text="S5_TERMINAL_MARKER_UNWRITABLE append_rc=1\n",
        phase_file_preflight_ok=True,
    )
    v = classify_terminal(obs)
    assert v.status == "FAIL"
    assert v.fail_class == "TERMINAL_MARKER_UNWRITABLE"


def test_classify_preflight_fail():
    obs = TerminalObservation(
        exit_rc=0,
        runner_pass_count=1,
        last_nonempty_line="RUNNER_PASS",
        actual_log_sha256="a",
        projected_log_sha256="a",
        phase_file_preflight_ok=False,
    )
    v = classify_terminal(obs)
    assert v.fail_class == "PHASE_FILE_PREFLIGHT_FAIL"


def test_parse_exit_rc_last_wins():
    text = "[EXIT rc=0] elapsed 1s\n[EXIT rc=1] elapsed 2s\n"
    assert parse_exit_rc(text) == 1


def test_argv_spawn_suffix_equals_child_and_shlex_render():
    body = build_run_phase_bash_c(
        {"S0": "/tmp/phases/S0.sh"},
        {"S0": 120},
        ["S0"],
    )
    assert "/tmp/phases/S0.sh" in body
    assert "run_phase S0.sh" not in body
    child = build_child_timeout_argv(outer_timeout_s=3095, kill_after_s=60, bash_c_body=body)
    mon = build_watch_wrap_spawn_argv(child, watch_wrap_path="/bin/true")
    assert_suffix_equals_child(mon, child)
    assert "--log" not in mon and "--stop-on" not in mon
    rendered = render_argv_shell(mon)
    # shlex roundtrip length stability
    assert shlex.split(rendered) == mon


def test_argv_rejects_relative_phase_path():
    with pytest.raises(ValueError):
        build_run_phase_bash_c({"S0": "S0.sh"}, {"S0": 1}, ["S0"])


def test_run_phase_wrapper_emits_first_phase_fail_marker_on_ordinary_failure():
    """Facade-emitted run_phase must emit FIRST_PHASE_FAIL on stream (not mere rc).

    Broken shape ``timeout; ec=$?`` under set -e yields the same rc without the
    marker — rc-only checks are green over nothing for this defect class.
    """
    import os
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "fail.sh"
        stub.write_text("#!/usr/bin/env bash\nexit 3\n")
        os.chmod(stub, 0o755)
        body = build_run_phase_bash_c({ "X": str(stub) }, { "X": 30 }, ["X"])
        # cure shape present; broken sequential capture absent
        assert "else ec=$?; fi" in body
        assert 'bash "$script"; ec=$?' not in body
        wrapper = f"set -euo pipefail; {body}; echo REACHED"
        r = subprocess.run(["bash", "-c", wrapper], capture_output=True, text=True)
        stream = r.stderr + r.stdout
        assert r.returncode == 3
        assert "FIRST_PHASE_FAIL" in stream, stream
        assert "phase=X" in stream
        assert "REACHED" not in r.stdout


def test_run_phase_wrapper_emits_phase_budget_breach_marker_on_timeout():
    """Facade-emitted run_phase must emit PHASE_BUDGET_BREACH on stream under timeout."""
    import os
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "slow.sh"
        stub.write_text("#!/usr/bin/env bash\nsleep 30\n")
        os.chmod(stub, 0o755)
        body = build_run_phase_bash_c({ "SLOW": str(stub) }, { "SLOW": 2 }, ["SLOW"])
        assert "else ec=$?; fi" in body
        wrapper = f"set -euo pipefail; {body}; echo REACHED"
        r = subprocess.run(["bash", "-c", wrapper], capture_output=True, text=True)
        stream = r.stderr + r.stdout
        assert r.returncode in (124, 137)
        assert "PHASE_BUDGET_BREACH" in stream, stream
        assert "phase=SLOW" in stream
        assert "REACHED" not in r.stdout


def test_run_phase_wrapper_success_reaches_after_run_phase():
    """Success arm: zero rc and control continues past run_phase."""
    import os
    import subprocess

    with tempfile.TemporaryDirectory() as td:
        stub = Path(td) / "ok.sh"
        stub.write_text("#!/usr/bin/env bash\nexit 0\n")
        os.chmod(stub, 0o755)
        body = build_run_phase_bash_c({ "OK": str(stub) }, { "OK": 30 }, ["OK"])
        wrapper = f"set -euo pipefail; {body}; echo REACHED_AFTER_RUNPHASE"
        r = subprocess.run(["bash", "-c", wrapper], capture_output=True, text=True)
        assert r.returncode == 0
        assert "REACHED_AFTER_RUNPHASE" in r.stdout
        assert "FIRST_PHASE_FAIL" not in (r.stderr + r.stdout)
        assert "PHASE_BUDGET_BREACH" not in (r.stderr + r.stdout)


def test_mint_content_digest_uses_basenames_matches_frozen_fixture_identity():
    """Known-good: mint of DEAD v13 fixture shells yields FROZEN_FIXTURE_CONTENT_DIGEST."""
    fix = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/tests/fixtures/r1l_launch")
    man = json.loads((fix / "fixtures_manifest.json").read_text())
    shells = {}
    for name, rec in man["members"].items():
        stem = name[:-3] if name.endswith(".sh") else name
        shells[stem] = (fix / rec["path"]).read_text()
    with tempfile.TemporaryDirectory() as td:
        m = mint_phase_files(Path(td) / "phases", shells)
        assert m["CONTENT_DIGEST"] == FROZEN_FIXTURE_CONTENT_DIGEST
        assert m["CONTENT_DIGEST"] == "30f545ccdc80c30d1e3ccd12c6f886b397d4a97ce6f78006ac57f62ca9a1d60f"
        re_resolve_phase_manifest(m)


def test_re_resolve_rejects_incomplete_member_set():
    """Known-bad: drop one member, recompute digest over remaining, leave count=7 → FAIL."""
    fix = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/tests/fixtures/r1l_launch")
    man = json.loads((fix / "fixtures_manifest.json").read_text())
    shells = {}
    for name, rec in man["members"].items():
        stem = name[:-3] if name.endswith(".sh") else name
        shells[stem] = (fix / rec["path"]).read_text()
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        m = mint_phase_files(phases, shells)
        # mutate manifest: remove S5 member + file, recompute digest, leave count=7
        os.chmod(phases, 0o755)
        s5 = Path(m["members"]["S5"]["path"])
        os.chmod(s5, 0o644)
        s5.unlink()
        del m["members"]["S5"]
        # wrong: recompute digest over remaining but leave count=7
        from calm.hrm_text_158.native_full_stack.r1l_launch.materialize import _basename_digest_map
        m["CONTENT_DIGEST"] = content_digest_from_members(_basename_digest_map(m["members"]))
        m["count"] = 7
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        assert "member set mismatch" in str(ei.value) or "count" in str(ei.value).lower()


def test_re_resolve_rejects_extra_member():
    fix = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/tests/fixtures/r1l_launch")
    man = json.loads((fix / "fixtures_manifest.json").read_text())
    shells = { (n[:-3] if n.endswith(".sh") else n): (fix/rec["path"]).read_text()
               for n, rec in man["members"].items() }
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        m = mint_phase_files(phases, shells)
        m["members"]["SX"] = {
            "path": str(phases / "SX.sh"),
            "sha256": "0"*64,
            "mode": 0o444,
            "bytes": 0,
        }
        with pytest.raises(PhaseFilePreflightError):
            re_resolve_phase_manifest(m)


def test_re_resolve_rejects_basename_mismatch():
    fix = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/tests/fixtures/r1l_launch")
    man = json.loads((fix / "fixtures_manifest.json").read_text())
    shells = { (n[:-3] if n.endswith(".sh") else n): (fix/rec["path"]).read_text()
               for n, rec in man["members"].items() }
    with tempfile.TemporaryDirectory() as td:
        phases = Path(td) / "phases"
        m = mint_phase_files(phases, shells)
        # rename file but keep member key
        os.chmod(phases, 0o755)
        old = Path(m["members"]["S0"]["path"])
        os.chmod(old, 0o644)
        new = phases / "renamed.sh"
        old.rename(new)
        m["members"]["S0"]["path"] = str(new)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        assert "basename" in str(ei.value)


def _minted_complete_manifest(td: str) -> dict:
    """Mint a full PHASE_ORDER manifest with all members present on disk.

    Metadata mutants keep the member set complete so the member-set check cannot
    mask count/phase_order gaps (the failure mode that produced this defect cycle).
    """
    fix = Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158/tests/fixtures/r1l_launch")
    man = json.loads((fix / "fixtures_manifest.json").read_text())
    shells = {
        (n[:-3] if n.endswith(".sh") else n): (fix / rec["path"]).read_text()
        for n, rec in man["members"].items()
    }
    return mint_phase_files(Path(td) / "phases", shells)


def test_re_resolve_metadata_clean_control_passes_with_complete_members():
    """Known-good: mint emits count + phase_order; members complete → ACCEPTED."""
    with tempfile.TemporaryDirectory() as td:
        m = _minted_complete_manifest(td)
        assert m["count"] == len(PHASE_ORDER) == len(m["members"])
        assert m["phase_order"] == list(PHASE_ORDER)
        re_resolve_phase_manifest(m)


def test_re_resolve_metadata_count_absent_fails_with_complete_members():
    """Known-bad: drop count only; members complete → FAIL (not masked by set check)."""
    with tempfile.TemporaryDirectory() as td:
        m = _minted_complete_manifest(td)
        del m["count"]
        assert len(m["members"]) == len(PHASE_ORDER)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        assert "count missing" in str(ei.value)


def test_re_resolve_metadata_count_wrong_fails_with_complete_members():
    """Known-bad: count=99 with members complete → FAIL (pinned count mismatch)."""
    with tempfile.TemporaryDirectory() as td:
        m = _minted_complete_manifest(td)
        m["count"] = 99
        assert len(m["members"]) == len(PHASE_ORDER)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        msg = str(ei.value)
        assert "count mismatch" in msg
        assert "count missing" not in msg


def test_re_resolve_metadata_phase_order_absent_fails_with_complete_members():
    """Known-bad: drop phase_order only; members complete → FAIL."""
    with tempfile.TemporaryDirectory() as td:
        m = _minted_complete_manifest(td)
        del m["phase_order"]
        assert len(m["members"]) == len(PHASE_ORDER)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        assert "phase_order missing" in str(ei.value)


def test_re_resolve_metadata_phase_order_reversed_fails_with_complete_members():
    """Known-bad: reverse phase_order; members complete → FAIL (order-sensitive)."""
    with tempfile.TemporaryDirectory() as td:
        m = _minted_complete_manifest(td)
        m["phase_order"] = list(reversed(list(PHASE_ORDER)))
        assert set(m["phase_order"]) == set(PHASE_ORDER)
        assert m["phase_order"] != list(PHASE_ORDER)
        assert len(m["members"]) == len(PHASE_ORDER)
        with pytest.raises(PhaseFilePreflightError) as ei:
            re_resolve_phase_manifest(m)
        msg = str(ei.value)
        assert "phase_order mismatch" in msg
        assert "phase_order missing" not in msg
