"""Liveness-topology guard for .claude/hooks/ai-room-ensure-codex-roles.sh.

The hook skips spawning a peer whose PINNED handle is already live. The
canonical helper it uses for the live set, `_discover_live_codex_handles`,
filters to AUTO codex handles only, so non-auto pinned handles such as
"claude" and "gate1_audit" can never appear in it. A membership test against
that list reads those peers as dead on every SessionStart and spawns a
duplicate.

This battery binds the hook's OWN bytes: `_handle_is_live` is extracted
verbatim from the shell file between its markers and exec'd, so the function
under test is the shipped function, not a copy of it.

Run: PYTHONPATH=. python3 -m pytest tests/test_ai_room_ensure_codex_roles_v1.py -q
"""

from __future__ import annotations

import re
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
HOOK = REPO_ROOT / ".claude" / "hooks" / "ai-room-ensure-codex-roles.sh"

BEGIN = "# --- BEGIN _handle_is_live"
END = "# --- END _handle_is_live"

# The pinned handles the hook spawns. Kept as a property, not a copy of the
# hook's table: the test reads the table out of the hook below.
AUTO_RE = re.compile(r"^codex(?:$|[_-].+$)")


def _extract_handle_is_live():
    src = HOOK.read_text(encoding="utf-8")
    assert src.count(BEGIN) == 1, "BEGIN marker must appear exactly once"
    assert src.count(END) == 1, "END marker must appear exactly once"
    block = src.split(BEGIN, 1)[1].split(END, 1)[0]
    # strip the hook's indentation so the def sits at module level
    lines = [ln[4:] if ln.startswith("    ") else ln for ln in block.splitlines()]
    text = "\n".join(lines)
    assert "def _handle_is_live(" in text, "extracted block must define the function"
    ns: dict = {}
    exec(compile(text, str(HOOK), "exec"), ns)
    return ns["_handle_is_live"], text


def _pinned_handles():
    """Pinned handles the hook spawns, read out of the hook's own table."""
    src = HOOK.read_text(encoding="utf-8")
    block = src.split("CLAUDE_PEER_ROLES = [", 1)[1].split("]", 1)[0]
    return re.findall(r'\(\s*"[^"]+",\s*"([^"]+)"', block)


class _FakeModule:
    """Stands in for the ai-room parity module."""

    def __init__(self, *, live_lease_handles=(), raise_on=(), stale=()):
        self.live_lease_handles = set(live_lease_handles)
        self.raise_on = set(raise_on)
        self.stale = set(stale)
        self.reads = []

    def _is_auto_codex_handle(self, h):
        return bool(AUTO_RE.match(h))

    def _read_lease(self, h):
        self.reads.append(h)
        if h in self.raise_on:
            raise OSError("simulated unreadable lease")
        if h in self.live_lease_handles or h in self.stale:
            return {"handle": h}
        return None

    def _is_lease_live(self, lease):
        return lease["handle"] not in self.stale

    def _lease_registry_metadata_live_if_present(self, lease):
        return True


class HandleIsLiveTests(unittest.TestCase):
    def setUp(self):
        self.fn, self.text = _extract_handle_is_live()
        self.logged = []

    def log(self, msg):
        self.logged.append(msg)

    def test_hook_parses_under_bash_n(self):
        import subprocess

        proc = subprocess.run(
            ["bash", "-n", str(HOOK)], capture_output=True, text=True
        )
        self.assertEqual(proc.returncode, 0, proc.stderr)

    def test_the_topology_that_regressed_non_auto_pinned_handles(self):
        """A live non-auto pinned handle must read live even though the auto
        list cannot contain it. This is the case the green suite lacked."""
        for handle in ("claude", "gate1_audit"):
            with self.subTest(handle=handle):
                self.assertNotRegex(handle, AUTO_RE)
                mod = _FakeModule(live_lease_handles={handle})
                # live_handles is the auto-filtered list: empty for these
                self.assertTrue(
                    self.fn(handle, mod, [], self.log),
                    f"{handle} has a live lease but read as dead",
                )

    def test_calibration_the_previous_predicate_reads_them_dead(self):
        """The old skip test was bare membership in the auto-filtered list.
        Run it on the same inputs: it must read a LIVE non-auto pinned handle
        as dead. Without this, a green result above is indistinguishable from
        a test that could never have failed."""

        def previous_predicate(h, module, live_handles, log):
            return h in live_handles

        for handle in ("claude", "gate1_audit"):
            with self.subTest(handle=handle):
                mod = _FakeModule(live_lease_handles={handle})
                self.assertFalse(
                    previous_predicate(handle, mod, [], self.log),
                    "old predicate should read a live non-auto handle as dead",
                )
                self.assertTrue(
                    self.fn(handle, mod, [], self.log),
                    "new predicate must read the same input as live",
                )

    def test_absent_lease_reads_not_live_so_the_peer_is_spawned(self):
        mod = _FakeModule(live_lease_handles=set())
        self.assertFalse(self.fn("gate1_audit", mod, [], self.log))

    def test_stale_lease_reads_not_live(self):
        mod = _FakeModule(stale={"claude"})
        self.assertFalse(self.fn("claude", mod, [], self.log))

    def test_unreadable_lease_reads_not_live_and_logs(self):
        mod = _FakeModule(raise_on={"claude"})
        self.assertFalse(self.fn("claude", mod, [], self.log))
        self.assertTrue(
            any("lease read claude failed" in m for m in self.logged),
            self.logged,
        )

    def test_auto_handles_still_use_the_canonical_list_not_the_lease(self):
        mod = _FakeModule(live_lease_handles={"codex_co_lead"})
        self.assertTrue(self.fn("codex_co_lead", mod, ["codex_co_lead"], self.log))
        self.assertFalse(self.fn("codex_co_lead", mod, [], self.log))
        self.assertEqual(mod.reads, [], "auto handles must not hit the lease path")

    def test_every_pinned_handle_in_the_hook_is_covered(self):
        """Denominator check: whatever the hook's table lists, each entry
        resolves through this predicate rather than bare membership."""
        pinned = _pinned_handles()
        self.assertTrue(pinned, "hook table must be readable and non-empty")
        for handle in pinned:
            with self.subTest(handle=handle):
                mod = _FakeModule(live_lease_handles={handle})
                live_list = [handle] if AUTO_RE.match(handle) else []
                self.assertTrue(self.fn(handle, mod, live_list, self.log))

    def test_skip_sites_call_the_predicate_not_bare_membership(self):
        src = HOOK.read_text(encoding="utf-8")
        self.assertNotIn("if pinned_handle in live_handles:", src)
        self.assertNotIn("elif CO_LEAD_HANDLE in live_handles:", src)
        self.assertIn("_handle_is_live(pinned_handle,", src)
        self.assertIn("_handle_is_live(CO_LEAD_HANDLE,", src)


if __name__ == "__main__":
    unittest.main()


# =============================================================================
# Lock-lifecycle + self-spawn battery for the SURVIVING hook.
#
# The extraction tests above bind one function. They are not an owner for
# concurrent process locking, SIGKILL recovery, cross-channel isolation,
# spawn-failure release, or the self-spawn guard: this file could report green
# without one lock assertion having run. That coverage previously lived in
# tests/test_ai_room_ensure_advisor_v1.py against a hook that is now retired;
# it is restored here against ai-room-ensure-codex-roles.sh.
#
# Every arm drives the REAL hook as a subprocess under a temporary HOME holding
# a fixture parity module. Nothing here reads or writes the live ~/.ai-room:
# no journal, no lease, no session, no spawn.
#
#   L1 same-channel concurrent -- the lock loser must not spawn duplicates.
#   L2 corpse recovery         -- process-SYNCHRONIZED; the two calibration
#                                 steps separate "recovered from a corpse"
#                                 from "there was never a lock".
#   L3 cross-channel           -- the key is per-channel, not global.
#   L4 spawn failure           -- an error path must not wedge later ensures.
#   A13 self-spawn             -- a session that IS a lane must not respawn
#                                 itself, and must still ensure the others.
# =============================================================================

import fcntl
import json
import os
import signal
import subprocess
import time

import pytest

MARKER_TIMEOUT = 30.0  # hard bound; a timeout is a test FAILURE, never a pass
DEATH_TIMEOUT = 10.0

# Stands in for ~/.ai-room/mcp-server.py. It defines exactly the seam the hook
# calls and nothing else, so an arm cannot reach live spawn machinery.
FIXTURE_PARITY_MODULE = '''
import json, os, pathlib, re, time

_AUTO = re.compile(r"^codex(?:$|[_-].+$)")

def _env_set(name):
    return {h for h in os.environ.get(name, "").split(",") if h}

class _Resolution:
    error = None
    error_detail = ""
    tmux_socket = None
    collab_instance_id = None

def init_room(channel=None, room_dir=None):
    return pathlib.Path(os.environ["HOME"]) / ".ai-room"

def ensure_room():
    return None

def _is_auto_codex_handle(h):
    return bool(_AUTO.match(h))

def _read_lease(handle):
    if handle in _env_set("FIXTURE_LIVE_HANDLES"):
        return {"handle": handle, "_live": True}
    if handle in _env_set("FIXTURE_STALE_HANDLES"):
        return {"handle": handle, "_live": False}
    return None

def _is_lease_live(lease):
    return bool(lease.get("_live"))

def _lease_registry_metadata_live_if_present(lease):
    return True

def _discover_live_codex_handles(exclude_handle):
    return sorted(h for h in _env_set("FIXTURE_LIVE_HANDLES") if _is_auto_codex_handle(h))

def _resolve_tmux_socket(channel, **kwargs):
    return _Resolution()

def _write_marker(text):
    # Append: the hook spawns several lanes per run, and which lanes were
    # reached is exactly what the arms below measure.
    with open(os.environ["FIXTURE_MARKER"], "a") as fh:
        fh.write(text + "\\n")
        fh.flush()
        os.fsync(fh.fileno())

def _spawn_claude_core(**kwargs):
    mode = os.environ.get("FIXTURE_SPAWN_MODE", "record")
    shape = {k: repr(v) for k, v in sorted(kwargs.items())}
    if mode == "raise":
        _write_marker("RAISED " + json.dumps(shape))
        raise RuntimeError("fixture: spawn failed")
    if mode == "block":
        # Emitted only here -- i.e. only after this ensure has taken the flock
        # and reached the spawn. Then hold the lock open indefinitely.
        _write_marker("LOCK_ACQUIRED " + json.dumps(shape))
        while True:
            time.sleep(3600)
    _write_marker("PROCEED " + json.dumps(shape))
    return {"ok": True, "handle": kwargs.get("handle"), "error": None}
'''


@pytest.fixture
def home(tmp_path):
    """A temp HOME carrying the fixture parity module."""
    h = tmp_path / "home"
    (h / ".ai-room").mkdir(parents=True)
    (h / ".ai-room" / "mcp-server.py").write_text(
        FIXTURE_PARITY_MODULE, encoding="utf-8"
    )
    return h


def _channel_dir(home, channel):
    d = home / ".ai-room" / "channels" / channel
    d.mkdir(parents=True, exist_ok=True)
    return d


def lock_path(home, channel):
    return _channel_dir(home, channel) / ".ensure-codex-roles.lock"


def ensure_env(home, channel, marker, *, mode="record", live=(), stale=(), handle=None):
    env = dict(os.environ)
    env.pop("AI_ROOM_HANDLE", None)
    if handle is not None:
        env["AI_ROOM_HANDLE"] = handle
    env.update({
        "HOME": str(home),
        "AI_ROOM_CHANNEL": channel,
        "CLAUDE_PROJECT_DIR": str(REPO_ROOT),
        "AI_ROOM_ENSURE_CODEX_ROLES_LOG": str(Path(marker).with_suffix(".log")),
        "FIXTURE_MARKER": str(marker),
        "FIXTURE_SPAWN_MODE": mode,
        "FIXTURE_LIVE_HANDLES": ",".join(live),
        "FIXTURE_STALE_HANDLES": ",".join(stale),
    })
    return env


def run_ensure(home, channel, marker, *, mode="record", live=(), stale=(),
               handle=None, timeout=60):
    return subprocess.run(
        ["bash", str(HOOK)], capture_output=True, text=True, timeout=timeout,
        env=ensure_env(home, channel, marker, mode=mode, live=live, stale=stale,
                       handle=handle),
    )


def start_ensure(home, channel, marker, *, mode="block", live=(), stale=()):
    """Launch in its own session so the whole tree can be signalled at once.

    Without start_new_session, SIGKILL to bash would leave the python child --
    the actual lock holder -- alive, and L2 would be measuring the wrong process.
    """
    return subprocess.Popen(
        ["bash", str(HOOK)], start_new_session=True,
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env=ensure_env(home, channel, marker, mode=mode, live=live, stale=stale),
    )


def read_marker(marker):
    return marker.read_text(encoding="utf-8") if marker.exists() else ""


def spawned_handles(marker):
    """The pinned handles this run actually reached the spawn seam for."""
    out = []
    for line in read_marker(marker).splitlines():
        if " " not in line:
            continue
        token, blob = line.split(" ", 1)
        out.append(json.loads(blob)["handle"].strip("'"))
    return out


def spawn_shape(marker, handle):
    for line in read_marker(marker).splitlines():
        if " " not in line:
            continue
        shape = json.loads(line.split(" ", 1)[1])
        if shape["handle"].strip("'") == handle:
            return shape
    raise AssertionError(f"no spawn recorded for {handle}; marker={read_marker(marker)!r}")


def await_marker(marker, token, timeout=MARKER_TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        text = read_marker(marker)
        if token in text:
            return text
        time.sleep(0.05)
    raise AssertionError(
        f"timed out after {timeout}s waiting for {token!r}; marker={read_marker(marker)!r}"
    )


def try_acquire(path):
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
    path would sail past and the arm would pass over nothing.
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


# --- L1: same-channel concurrent ---------------------------------------------

def test_L1_second_ensure_does_not_spawn_while_lock_held(home, tmp_path):
    marker = tmp_path / "l1.marker"
    with holding_ensure(home, "chan-a", tmp_path / "l1holder.marker"):
        proc = run_ensure(home, "chan-a", marker, mode="record")
    assert proc.returncode == 0, f"ensure must fail soft; stderr={proc.stderr!r}"
    assert not marker.exists(), (
        f"the lock loser must NOT spawn duplicates; marker={read_marker(marker)!r}"
    )


def test_L1_positive_control_spawns_when_lock_is_free(home, tmp_path):
    """Without this, L1 is satisfied by a hook that never spawns at all."""
    marker = tmp_path / "l1pos.marker"
    proc = run_ensure(home, "chan-a", marker, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert spawned_handles(marker), read_marker(marker)


# --- L2: corpse recovery, process-synchronized -------------------------------

def test_L2_lock_dies_with_its_process(home, tmp_path):
    lock = lock_path(home, "chan-a")
    first = tmp_path / "l2a.marker"

    # 1. Hold the lock in a child that blocks inside the spawn seam.
    child = start_ensure(home, "chan-a", first, mode="block")
    try:
        # 2. Hard-bounded wait. A timeout raises; it is never a pass.
        await_marker(first, "LOCK_ACQUIRED")

        # 3. CALIBRATION: prove the lock is genuinely held RIGHT NOW. Without
        #    this, step 5 succeeding proves nothing -- "nothing was ever locked"
        #    and "recovered from a corpse" are indistinguishable.
        assert not try_acquire(lock), (
            "the running holder's lock was acquirable; this arm would be green "
            "over nothing"
        )

        # 4. Kill the whole session -- bash AND the python child holding it.
        os.killpg(os.getpgid(child.pid), signal.SIGKILL)
        child.wait(timeout=DEATH_TIMEOUT)
        assert child.poll() is not None, "holder did not reach terminal death"
    finally:
        if child.poll() is None:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait(timeout=DEATH_TIMEOUT)

    # 5. CALIBRATION: the kernel released it with the process. An O_EXCL
    #    sentinel fails exactly here, leaving a corpse that suppresses every
    #    later ensure while reporting success.
    deadline = time.monotonic() + DEATH_TIMEOUT
    while not try_acquire(lock) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert try_acquire(lock), "lock survived its process"

    # 6. A fresh ensure proceeds all the way to the spawn seam.
    second = tmp_path / "l2b.marker"
    proc = run_ensure(home, "chan-a", second, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert spawned_handles(second), read_marker(second)


# --- L3: the key is channel-scoped -------------------------------------------

def test_L3_other_channel_proceeds_while_channel_a_is_locked(home, tmp_path):
    marker = tmp_path / "l3.marker"
    with holding_ensure(home, "chan-a", tmp_path / "l3holder.marker"):
        proc = run_ensure(home, "chan-b", marker, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert spawned_handles(marker), (
        f"channel B was suppressed by channel A's lock -- the key is global; "
        f"marker={read_marker(marker)!r}"
    )


# --- L4: the error path releases ---------------------------------------------

def test_L4_spawn_failure_releases_the_lock(home, tmp_path):
    lock = lock_path(home, "chan-a")
    failing = tmp_path / "l4a.marker"
    proc = run_ensure(home, "chan-a", failing, mode="raise")
    assert proc.returncode == 0, f"a spawn failure must stay fail-soft; {proc.stderr!r}"
    assert "RAISED" in read_marker(failing), read_marker(failing)

    assert try_acquire(lock), "the raising path wedged the lock"

    recovering = tmp_path / "l4b.marker"
    proc = run_ensure(home, "chan-a", recovering, mode="record")
    assert proc.returncode == 0, proc.stderr
    assert spawned_handles(recovering), read_marker(recovering)


# --- A13: self-spawn ---------------------------------------------------------

def test_A13_a_lane_session_does_not_respawn_itself(home, tmp_path):
    """Regression for a MEASURED duplicate: a spawned peer runs its own
    SessionStart, which reaches this hook before its lease is discoverable.
    The flock does not help -- the two ensures are sequential, so each acquires
    cleanly -- and liveness cannot either, because the racing session is
    exactly the one not yet live."""
    for self_handle in _pinned_handles() + ["codex_co_lead"]:
        marker = tmp_path / f"self-{self_handle}.marker"
        proc = run_ensure(home, "chan-a", marker, mode="record", handle=self_handle)
        assert proc.returncode == 0, proc.stderr
        assert self_handle not in spawned_handles(marker), (
            f"a {self_handle} session spawned another {self_handle}; "
            f"{read_marker(marker)!r}"
        )


def test_A13_guard_is_per_handle_not_a_whole_file_exit(home, tmp_path):
    """Calibrates the arm above twice over: the guard must key on BEING that
    lane (not on AI_ROOM_HANDLE merely being set, which every real session
    sets), and it must skip only that lane -- the others still get ensured."""
    marker = tmp_path / "otherlanes.marker"
    run_ensure(home, "chan-a", marker, mode="record", handle="codex")
    reached = spawned_handles(marker)
    assert "codex" not in reached, reached
    for other in ["codex_co_lead"] + [h for h in _pinned_handles() if h != "codex"]:
        assert other in reached, f"{other} was suppressed by the self-guard; {reached}"

    unrelated = tmp_path / "unrelated.marker"
    run_ensure(home, "chan-a", unrelated, mode="record", handle="advisor")
    assert "codex" in spawned_handles(unrelated), (
        "a non-lane handle disabled the bootstrap; the guard keys on presence, "
        f"not identity; {read_marker(unrelated)!r}"
    )


# --- liveness through the REAL hook ------------------------------------------

def test_a_live_non_auto_pinned_handle_is_not_respawned(home, tmp_path):
    """The extraction tests bind `_handle_is_live` in isolation. This binds the
    hook CALLING it: a live non-auto pinned handle can never appear in the
    auto-filtered list, so a bare membership test respawns over a healthy peer."""
    marker = tmp_path / "live.marker"
    proc = run_ensure(home, "chan-a", marker, mode="record", live=("gate1_audit",))
    assert proc.returncode == 0, proc.stderr
    assert "gate1_audit" not in spawned_handles(marker), (
        f"a live peer was respawned; {read_marker(marker)!r}"
    )
    log = Path(marker).with_suffix(".log").read_text(encoding="utf-8")
    assert "already live" in log, log
    assert "FATAL parity-block error" not in log, log
    assert "WARN role-spawn block failed" not in log, log


def test_a_stale_lease_does_not_suppress_the_spawn(home, tmp_path):
    """Calibrates the arm above: it must key on lease LIVENESS, not on the
    lease merely existing -- otherwise a dead peer's leftover lease suppresses
    the bootstrap forever, which is the corpse failure in another costume."""
    marker = tmp_path / "stale.marker"
    run_ensure(home, "chan-a", marker, mode="record", stale=("gate1_audit",))
    assert "gate1_audit" in spawned_handles(marker), read_marker(marker)


def test_spawn_shapes_match_the_staged_topology(home, tmp_path):
    """The backend rides on these flags; a wrong one routes a peer to the wrong
    credentials silently."""
    marker = tmp_path / "shape.marker"
    run_ensure(home, "chan-a", marker, mode="record")
    co_lead = spawn_shape(marker, "codex_co_lead")
    assert co_lead["sol"] == "True" and co_lead["agent"] == "'co-lead'", co_lead
    expected_grok = {"claude": "True", "codex": "False", "gate1_audit": "True"}
    for handle, grok in expected_grok.items():
        shape = spawn_shape(marker, handle)
        assert shape["grok"] == grok, (handle, shape)
        assert shape["allow_dangerous"] == "True", (handle, shape)


# --- the live room is never touched ------------------------------------------

def test_live_room_is_never_touched_by_this_suite(home, tmp_path):
    """Every arm runs under a temp HOME. If the hook resolved ~ some other way,
    a live channel lock would appear."""
    live_dir = Path.home() / ".ai-room" / "channels" / "chan-a"
    before = live_dir.exists()
    run_ensure(home, "chan-a", tmp_path / "iso.marker", mode="record")
    assert live_dir.exists() == before, "the suite created state under the live HOME"
