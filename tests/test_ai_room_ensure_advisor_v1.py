"""Lock-lifecycle battery for .claude/hooks/ai-room-ensure-advisor.sh.

This file exists because L1-L4 had no owner. A frontmatter parser test is not an
owner for concurrent process locking, SIGKILL recovery, cross-channel isolation,
or spawn-failure release -- a receipt could report the frontmatter command green
while not one lock assertion ran.

Every arm drives the REAL hook as a subprocess against a temporary HOME holding
a fixture parity module. Nothing here reads or writes the live ~/.ai-room room:
no journal, no lease, no session, no spawn.

L1 same-channel concurrent -- the loser must not spawn a duplicate.
L2 corpse recovery       -- process-SYNCHRONIZED; steps 3 and 5 are the
                            calibration pair that separate "recovered from a
                            corpse" from "there was never a lock".
L3 cross-channel         -- the key is per-channel, not global.
L4 spawn failure         -- an error path must not wedge later ensures.

Run: PYTHONPATH=. python3 -m pytest tests/test_ai_room_ensure_advisor_v1.py -q
"""
from __future__ import annotations

import fcntl
import json
import os
import pathlib
import signal
import subprocess
import time

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".claude" / "hooks" / "ai-room-ensure-advisor.sh"

MARKER_TIMEOUT = 30.0  # hard bound; a timeout is a test FAILURE, never a pass
DEATH_TIMEOUT = 10.0

# A fixture stand-in for ~/.ai-room/mcp-server.py. It defines exactly the seam
# the hook calls, and nothing else, so an arm cannot accidentally reach live
# spawn machinery. Behaviour is env-driven so one module serves every arm.
FIXTURE_PARITY_MODULE = '''
import json, os, pathlib, time

class _Resolution:
    error = None
    error_detail = ""
    tmux_socket = None
    collab_instance_id = None

def init_room(channel=None, room_dir=None):
    return pathlib.Path(os.environ["HOME"]) / ".ai-room"

def ensure_room():
    return None

def _read_lease(handle):
    mode = os.environ.get("FIXTURE_ADVISOR_LEASE", "none")
    if handle != "advisor" or mode == "none":
        return None
    return {"handle": handle, "owner_pid": 4242, "heartbeat": "fixture", "_live": mode == "live"}

def _is_lease_live(lease):
    return bool(lease.get("_live"))

def _discover_live_codex_handles(exclude_handle):
    raise AssertionError("the hook must not use the codex NAME filter for this handle")

def _resolve_tmux_socket(channel, **kwargs):
    return _Resolution()

def _write_marker(text):
    path = pathlib.Path(os.environ["FIXTURE_MARKER"])
    tmp = path.with_suffix(".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)  # atomic: the parent never sees a partial marker

def _spawn_claude_core(**kwargs):
    mode = os.environ.get("FIXTURE_SPAWN_MODE", "record")
    shape = {k: repr(v) for k, v in sorted(kwargs.items())}
    if mode == "raise":
        _write_marker("RAISED " + json.dumps(shape))
        raise RuntimeError("fixture: spawn failed")
    if mode == "block":
        # Emitted only here -- i.e. only after A10 has taken the flock and the
        # hook has reached the spawn. Then hold the lock open indefinitely.
        _write_marker("LOCK_ACQUIRED " + json.dumps(shape))
        while True:
            time.sleep(3600)
    _write_marker("PROCEED " + json.dumps(shape))
    return {"ok": True, "handle": kwargs.get("handle"), "error": None}
'''


# --- harness ------------------------------------------------------------------

@pytest.fixture
def home(tmp_path):
    """A temp HOME carrying the fixture parity module."""
    h = tmp_path / "home"
    (h / ".ai-room").mkdir(parents=True)
    (h / ".ai-room" / "mcp-server.py").write_text(FIXTURE_PARITY_MODULE, encoding="utf-8")
    return h


def channel_dir(home: pathlib.Path, channel: str) -> pathlib.Path:
    d = home / ".ai-room" / "channels" / channel
    d.mkdir(parents=True, exist_ok=True)
    return d


def lock_path(home: pathlib.Path, channel: str) -> pathlib.Path:
    return channel_dir(home, channel) / ".ensure-advisor.lock"


def ensure_env(home, channel, marker, *, mode="record", lease="none", handle=None):
    env = dict(os.environ)
    env.pop("AI_ROOM_HANDLE", None)
    if handle is not None:
        env["AI_ROOM_HANDLE"] = handle
    env.update({
        "HOME": str(home),
        "AI_ROOM_CHANNEL": channel,
        "CLAUDE_PROJECT_DIR": str(REPO_ROOT),
        "AI_ROOM_ENSURE_ADVISOR_LOG": str(pathlib.Path(marker).with_suffix(".log")),
        "FIXTURE_MARKER": str(marker),
        "FIXTURE_SPAWN_MODE": mode,
        "FIXTURE_ADVISOR_LEASE": lease,
    })
    return env


def run_ensure(home, channel, marker, *, mode="record", lease="none", handle=None, timeout=60):
    return subprocess.run(
        ["bash", str(HOOK)], capture_output=True, text=True, timeout=timeout,
        env=ensure_env(home, channel, marker, mode=mode, lease=lease, handle=handle),
    )


def start_ensure(home, channel, marker, *, mode="block", lease="none"):
    """Launch in its own session so the whole tree can be signalled at once.

    Without start_new_session, SIGKILL to bash would leave the python child --
    the actual lock holder -- alive, and L2 would be measuring the wrong process.
    """
    return subprocess.Popen(
        ["bash", str(HOOK)], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=ensure_env(home, channel, marker, mode=mode, lease=lease),
    )


def read_marker(marker: pathlib.Path) -> str:
    return marker.read_text(encoding="utf-8") if marker.exists() else ""


def await_marker(marker: pathlib.Path, token: str, timeout=MARKER_TIMEOUT) -> str:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = read_marker(marker)
        if text.startswith(token):
            return text
        time.sleep(0.05)
    raise AssertionError(
        f"timed out after {timeout}s waiting for {token!r}; marker={read_marker(marker)!r}"
    )


def try_acquire(path: pathlib.Path) -> bool:
    """Independent nonblocking acquire, from this process, on a fresh FD."""
    fd = os.open(str(path), os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    else:
        fcntl.flock(fd, fcntl.LOCK_UN)
        return True
    finally:
        os.close(fd)


class holding_ensure:
    """Hold a channel's lock using a REAL ensure blocked inside the spawn seam.

    Deliberately not "open the path the test thinks the hook uses and flock it":
    that only contends if the test's guess is right, so a hook keyed on a global
    path would sail past and the arm would pass over nothing. Measured -- an
    earlier revision of L3 did exactly that and survived the global-key mutant.
    """

    def __init__(self, home, channel, marker):
        self.args = (home, channel, marker)
        self.marker = marker
        self.proc = None

    def __enter__(self):
        self.proc = start_ensure(*self.args, mode="block")
        await_marker(self.marker, "LOCK_ACQUIRED")
        return self

    def __exit__(self, *exc):
        if self.proc.poll() is None:
            os.killpg(os.getpgid(self.proc.pid), signal.SIGKILL)
        self.proc.wait(timeout=DEATH_TIMEOUT)
        return False


# --- syntax -------------------------------------------------------------------

def test_hook_parses_under_bash_n():
    proc = subprocess.run(["bash", "-n", str(HOOK)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


# --- L1: same-channel concurrent ----------------------------------------------

def test_L1_second_ensure_does_not_spawn_while_lock_held(home, tmp_path):
    marker = tmp_path / "l1.marker"
    with holding_ensure(home, "chan-a", tmp_path / "l1holder.marker"):
        proc = run_ensure(home, "chan-a", marker, mode="record")
    assert proc.returncode == 0, f"ensure must fail soft; stderr={proc.stderr!r}"
    assert not marker.exists(), (
        f"the lock loser must NOT spawn a duplicate; marker={read_marker(marker)!r}"
    )


def test_L1_positive_control_spawns_when_lock_is_free(home, tmp_path):
    """Without this, L1 is satisfied by a hook that never spawns at all."""
    marker = tmp_path / "l1pos.marker"
    proc = run_ensure(home, "chan-a", marker, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert read_marker(marker).startswith("PROCEED"), read_marker(marker)


# --- L2: corpse recovery, process-synchronized --------------------------------

def test_L2_lock_dies_with_its_process(home, tmp_path):
    lock = lock_path(home, "chan-a")
    first = tmp_path / "l2a.marker"

    # 1. Hold the lock in a child that blocks inside the spawn seam.
    child = start_ensure(home, "chan-a", first, mode="block")
    try:
        # 2. Hard-bounded wait. A timeout raises; it is never a pass.
        await_marker(first, "LOCK_ACQUIRED")

        # 3. CALIBRATION: prove the lock is genuinely held RIGHT NOW. Without
        #    this, step 6 succeeding proves nothing -- "nothing was ever locked"
        #    and "recovered from a corpse" are indistinguishable.
        assert not try_acquire(lock), (
            "the running holder's lock was acquirable; this arm would be green "
            "over nothing"
        )

        # 4. Kill the whole session -- bash AND the python child that holds it.
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        child.wait(timeout=DEATH_TIMEOUT)
        assert child.poll() is not None, "holder did not reach terminal death"
    finally:
        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait(timeout=DEATH_TIMEOUT)

    # 5. CALIBRATION: the kernel released it with the process. An O_EXCL
    #    sentinel -- the v6 defect -- fails exactly here, leaving a corpse that
    #    suppresses every later ensure while reporting success.
    deadline = time.monotonic() + DEATH_TIMEOUT
    while not try_acquire(lock) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert try_acquire(lock), "lock survived its process"

    # 6. A fresh ensure proceeds all the way to the spawn seam.
    second = tmp_path / "l2b.marker"
    proc = run_ensure(home, "chan-a", second, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert read_marker(second).startswith("PROCEED"), read_marker(second)


# --- L3: the key is channel-scoped --------------------------------------------

def test_L3_other_channel_proceeds_while_channel_a_is_locked(home, tmp_path):
    marker = tmp_path / "l3.marker"
    with holding_ensure(home, "chan-a", tmp_path / "l3holder.marker"):
        proc = run_ensure(home, "chan-b", marker, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert read_marker(marker).startswith("PROCEED"), (
        f"channel B was suppressed by channel A's lock -- the key is global; "
        f"marker={read_marker(marker)!r}"
    )


# --- L4: the error path releases ----------------------------------------------

def test_L4_spawn_failure_releases_the_lock(home, tmp_path):
    lock = lock_path(home, "chan-a")
    failing = tmp_path / "l4a.marker"
    proc = run_ensure(home, "chan-a", failing, mode="raise")
    assert proc.returncode == 0, f"a spawn failure must stay fail-soft; {proc.stderr!r}"
    assert read_marker(failing).startswith("RAISED"), read_marker(failing)

    assert try_acquire(lock), "the raising path wedged the lock"

    recovering = tmp_path / "l4b.marker"
    proc = run_ensure(home, "chan-a", recovering, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert read_marker(recovering).startswith("PROCEED"), read_marker(recovering)


# --- A-series: what the hook actually asks for --------------------------------

def test_A_spawn_shape_is_default_backend_claude_peer(home, tmp_path):
    """The model rides on the agent file, so the spawn must NOT set grok/sol/
    neural -- any of them would route the advisor to a different backend."""
    marker = tmp_path / "shape.marker"
    run_ensure(home, "chan-a", marker, mode="record")
    shape = json.loads(read_marker(marker).split(" ", 1)[1])
    assert shape["handle"] == "'advisor'"
    assert shape["agent"] == "'fable-advisor'"
    assert shape["grok"] == "False"
    assert shape["sol"] == "False"
    assert shape["neural"] == "False"
    assert shape["allow_dangerous"] == "True"


def test_A_never_evicts_a_live_advisor(home, tmp_path):
    """Asserts the REASON, not just the absence of a spawn. The hook is fail-soft,
    so "no spawn happened" is also what a crash looks like -- an arm that stops at
    `not marker.exists()` passes on a hook that threw before reaching liveness."""
    marker = tmp_path / "live.marker"
    proc = run_ensure(home, "chan-a", marker, mode="record", lease="live")
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), (
        f"a live advisor must be left alone, never respawned; {read_marker(marker)!r}"
    )
    log = marker.with_suffix(".log").read_text(encoding="utf-8")
    assert "skip: advisor lease live" in log, log
    assert "non-fatal error" not in log, log


def test_A_stale_lease_does_not_suppress_the_spawn(home, tmp_path):
    """Calibrates the arm above: it must key on lease LIVENESS, not on the lease
    file merely existing -- otherwise a dead peer's leftover lease would suppress
    the bootstrap forever, which is the corpse failure in another costume."""
    marker = tmp_path / "stale.marker"
    run_ensure(home, "chan-a", marker, mode="record", lease="stale")
    assert read_marker(marker).startswith("PROCEED"), read_marker(marker)


def test_A_codex_name_filter_is_not_the_liveness_seam(home, tmp_path):
    """MEASURED, not hypothesized: with the advisor running and its heartbeat
    fresh, _discover_live_codex_handles returned ['codex', 'codex_co_lead'] --
    advisor absent. It filters leases through `^codex(?:$|[_-].+$)`
    (config.py:141), a NAME filter this handle can never match, so it reports
    not-live unconditionally and the hook respawned over a healthy peer.

    The fixture raises if that helper is called; the hook is fail-soft and would
    swallow it, so this arm reads the log rather than the exit code."""
    marker = tmp_path / "seam.marker"
    run_ensure(home, "chan-a", marker, mode="record", lease="live")
    log = marker.with_suffix(".log").read_text(encoding="utf-8")
    assert "must not use the codex NAME filter" not in log, (
        "the hook still routes this handle's liveness through the codex name "
        f"filter; log={log!r}"
    )


def test_A13_advisor_session_does_not_self_spawn(home, tmp_path):
    """Regression for a MEASURED duplicate, not a hypothesized one: spawning the
    advisor ran the advisor's OWN SessionStart, which reached this hook at
    08:25:52 before its lease was discoverable and spawned a second advisor at
    08:25:56. The flock does not help -- the two ensures are sequential, so each
    acquires cleanly -- and liveness cannot either, because the racing session is
    exactly the one not yet live. Fails if the hook spawns while being the
    advisor."""
    marker = tmp_path / "self.marker"
    proc = run_ensure(home, "chan-a", marker, mode="record", handle="advisor")
    assert proc.returncode == 0, proc.stderr
    assert not marker.exists(), (
        f"an advisor session spawned another advisor; {read_marker(marker)!r}"
    )


def test_A13_other_handles_still_ensure(home, tmp_path):
    """Calibrates the arm above: it must key on being the advisor, not on
    AI_ROOM_HANDLE merely being set. A presence check would disable the
    bootstrap for every real session, since every peer sets that variable."""
    marker = tmp_path / "otherhandle.marker"
    run_ensure(home, "chan-a", marker, mode="record", handle="claude")
    assert read_marker(marker).startswith("PROCEED"), read_marker(marker)


# --- [data]: the live room is never touched -----------------------------------

def test_live_room_is_never_touched_by_this_suite(home, tmp_path):
    """Every arm runs under a temp HOME. If the hook resolved ~ some other way,
    a live channel lock would appear."""
    live_lock = pathlib.Path.home() / ".ai-room" / "channels" / "chan-a"
    before = live_lock.exists()
    run_ensure(home, "chan-a", tmp_path / "iso.marker", mode="record")
    assert live_lock.exists() == before, "the suite created state under the live HOME"
