"""CPU dryrun: tmux-detached launcher survives full kicker-subtree kill."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path

import pytest

RUN_ID = "dryrun_tmux_detach"
POSTHASH_SCHEMA = "hrm_text_158_parent_checkpoint_posthash/v0"


def _ppid(pid: int) -> int | None:
    status_path = Path(f"/proc/{pid}/status")
    if not status_path.exists():
        return None
    for line in status_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("PPid:"):
            return int(line.split()[1])
    return None


def _is_descendant(pid: int, ancestor_pid: int) -> bool:
    current = pid
    while current > 1:
        if current == ancestor_pid:
            return True
        parent = _ppid(current)
        if parent is None:
            return False
        current = parent
    return False


def _tmux_server_pid() -> int:
    out = subprocess.check_output(
        ["tmux", "display-message", "-p", "#{pid}"],
        text=True,
    ).strip()
    return int(out)


def _write_stub_launcher(launcher_path: Path, run_root: Path, launch_log: Path, session: str) -> None:
    launcher_path.write_text(
        f"""#!/bin/bash
set -euo pipefail
RUN_ROOT="{run_root}/"
LAUNCH_LOG="{launch_log}"
TMUX_SESSION_NAME="{session}"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
echo "=== LAUNCHER START pid=$$ ==="
mkdir -p "$RUN_ROOT/prelaunch" "$RUN_ROOT/phase_a"
python3 -c 'import json; from pathlib import Path; r=Path("{run_root}"); (r/"prelaunch"/"run_root_freshness_witness.json").write_text(json.dumps({{"run_root_fresh_pass":True}})+"\\n")'
python3 -c 'import json,os; from pathlib import Path; r=Path("{run_root}"); (r/"prelaunch"/"launcher_session_witness.json").write_text(json.dumps({{"launcher_pid":os.getpid(),"tmux_session":os.environ.get("TMUX_SESSION_NAME")}})+"\\n")'
for i in 1 2 3 4 5 6 7 8 9 10; do
  echo "{{\\"event\\":\\"heartbeat\\",\\"phase\\":\\"step_update\\",\\"step\\":$i}}" >> "$RUN_ROOT/phase_a/probe.stdout.log"
  sleep 2
done
echo 0 > "$RUN_ROOT/probe.exit_code.txt"
python3 -c 'import json; from pathlib import Path; (Path("{run_root}")/"prelaunch"/"parent_checkpoint_posthash.json").write_text(json.dumps({{"schema":"{POSTHASH_SCHEMA}","flush_reason":"wrapper_post_exit"}})+"\\n")'
if [ ! -f "$RUN_ROOT/probe.exit_code.txt" ] || [ ! -f "$RUN_ROOT/prelaunch/parent_checkpoint_posthash.json" ]; then
  echo TERMINAL_ARTIFACT_MISSING >&2
  exit 3
fi
echo LAUNCHER_DONE
""",
        encoding="utf-8",
    )
    launcher_path.chmod(0o755)


def _write_kick_script(
    kick_path: Path,
    *,
    session: str,
    launcher: Path,
    run_root: Path,
    launch_log: Path,
) -> None:
    kick_path.write_text(
        f"""#!/bin/bash
set -euo pipefail
SESSION="{session}"
LAUNCHER="{launcher}"
LAUNCH_LOG="{launch_log}"
tmux has-session -t "$SESSION" 2>/dev/null && exit 2
tmux new-session -d -s "$SESSION" "bash -lc 'LAUNCH_LOG=$LAUNCH_LOG TMUX_SESSION_NAME=$SESSION exec $LAUNCHER'"
exit 0
""",
        encoding="utf-8",
    )
    kick_path.chmod(0o755)


def test_tmux_detach_survives_full_kicker_subtree_kill(tmp_path: Path) -> None:
    if subprocess.run(["which", "tmux"], capture_output=True).returncode != 0:
        pytest.skip("tmux not available")

    session = f"v4_detach_test_{os.getpid()}"
    run_root = tmp_path / "run_root"
    launch_log = tmp_path / "launch.log"
    launcher = tmp_path / "launch.sh"
    kick = tmp_path / "kick.sh"

    _write_stub_launcher(launcher, run_root, launch_log, session)
    _write_kick_script(
        kick,
        session=session,
        launcher=launcher,
        run_root=run_root,
        launch_log=launch_log,
    )

    try:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
    except Exception:
        pass

    kicker = subprocess.Popen(
        ["bash", str(kick)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    time.sleep(3)
    kicker_pgid = os.getpgid(kicker.pid)
    tmux_pid = _tmux_server_pid()
    assert not _is_descendant(tmux_pid, kicker.pid), (
        f"tmux server pid={tmux_pid} must not be descendant of kicker pid={kicker.pid}"
    )

    os.killpg(kicker_pgid, signal.SIGTERM)
    try:
        kicker.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(kicker_pgid, signal.SIGKILL)
        kicker.wait(timeout=5)

    deadline = time.time() + 45
    while time.time() < deadline:
        exit_path = run_root / "probe.exit_code.txt"
        posthash_path = run_root / "prelaunch" / "parent_checkpoint_posthash.json"
        if exit_path.is_file() and posthash_path.is_file():
            log_text = launch_log.read_text(encoding="utf-8") if launch_log.is_file() else ""
            if "LAUNCHER_DONE" in log_text:
                break
        time.sleep(1)
    else:
        raise AssertionError("detached launcher did not complete terminal artifacts in time")

    assert (run_root / "probe.exit_code.txt").read_text(encoding="utf-8").strip() == "0"
    posthash = json.loads(
        (run_root / "prelaunch" / "parent_checkpoint_posthash.json").read_text(encoding="utf-8")
    )
    assert posthash.get("flush_reason") == "wrapper_post_exit"
    assert "LAUNCHER_DONE" in launch_log.read_text(encoding="utf-8")
    witness = json.loads(
        (run_root / "prelaunch" / "launcher_session_witness.json").read_text(encoding="utf-8")
    )
    assert witness.get("launcher_pid")
    assert (run_root / "prelaunch" / "run_root_freshness_witness.json").is_file()
    assert sum(1 for _ in open(run_root / "phase_a" / "probe.stdout.log")) >= 6

    subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)
