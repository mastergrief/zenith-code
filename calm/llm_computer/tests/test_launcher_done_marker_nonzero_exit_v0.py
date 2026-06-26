"""Launcher wrapper must emit durable LAUNCHER_DONE marker on nonzero child exit."""
from __future__ import annotations

import subprocess
from pathlib import Path


def _write_wrapper_script(wrapper_path: Path, run_root: Path, launch_log: Path) -> None:
    wrapper_path.write_text(
        f"""#!/bin/bash
set -euo pipefail
RUN_ROOT="{run_root}/"
LAUNCH_LOG="{launch_log}"
exec > >(tee -a "$LAUNCH_LOG") 2>&1
echo "=== LAUNCHER START pid=$$ ==="
mkdir -p "$RUN_ROOT/prelaunch" "$RUN_ROOT/phase_a"
set +e
false
EXIT_CODE=$?
set -e
echo "$EXIT_CODE" > "$RUN_ROOT/probe.exit_code.txt"
printf '%s\\n' '{{"schema":"hrm_text_158_parent_checkpoint_posthash/v0","flush_reason":"wrapper_post_exit"}}' > "$RUN_ROOT/prelaunch/parent_checkpoint_posthash.json"
if [ ! -f "$RUN_ROOT/probe.exit_code.txt" ] || [ ! -f "$RUN_ROOT/prelaunch/parent_checkpoint_posthash.json" ]; then
  echo TERMINAL_ARTIFACT_MISSING >&2
  exit 3
fi
printf '%s\\n' "LAUNCHER_DONE" > "$RUN_ROOT/prelaunch/LAUNCHER_DONE.marker"
printf '%s\\n' "LAUNCHER_DONE" >> "$LAUNCH_LOG"
echo "LAUNCHER_DONE"
exit "$EXIT_CODE"
""",
        encoding="utf-8",
    )
    wrapper_path.chmod(0o755)


def test_launcher_done_marker_before_nonzero_exit(tmp_path: Path) -> None:
    run_root = tmp_path / "run_root"
    launch_log = tmp_path / "launch.log"
    wrapper = tmp_path / "wrapper.sh"
    _write_wrapper_script(wrapper, run_root, launch_log)

    proc = subprocess.run(
        ["bash", str(wrapper)],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 1

    marker = run_root / "prelaunch" / "LAUNCHER_DONE.marker"
    assert marker.is_file()
    assert marker.read_text(encoding="utf-8").strip() == "LAUNCHER_DONE"
    assert "LAUNCHER_DONE" in launch_log.read_text(encoding="utf-8")
    assert (run_root / "probe.exit_code.txt").read_text(encoding="utf-8").strip() == "1"
    assert (run_root / "prelaunch" / "parent_checkpoint_posthash.json").is_file()
