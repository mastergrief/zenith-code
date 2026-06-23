"""Unit tests for bounded retry/backoff in hrm_text_158_r7_resource_lane_acquire.

Run: PYTHONPATH=. python3 -m unittest tests.test_r7_resource_lane_acquire_backoff -v

No MCP server / no GPU: the acquire callable and sleep are injected, so these
exercise the wait-before-acquire backoff and the fail-closed exhaustion path
purely at the import level.
"""
from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

_MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hrm_text_158_r7_resource_lane_acquire.py"
)
_spec = importlib.util.spec_from_file_location("_r7_lane_acquire", _MODULE_PATH)
assert _spec and _spec.loader
lane_acquire = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lane_acquire)


class _RecordingSleep:
    """Captures the backoff durations without actually sleeping."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(float(seconds))


class _ScriptedAcquire:
    """Returns acquired=False for the first `fail_count` calls, then True."""

    def __init__(self, fail_count: int) -> None:
        self.fail_count = fail_count
        self.attempts = 0

    def __call__(self) -> dict:
        self.attempts += 1
        if self.attempts <= self.fail_count:
            return {"acquired": False, "reason": "held"}
        return {"acquired": True, "token": "ok", "attempt": self.attempts}


class AcquireWithBackoffTests(unittest.TestCase):
    def test_free_lane_first_attempt_acquires_no_sleep(self) -> None:
        """(a) Free lane: first attempt acquires, behavior unchanged, no backoff."""
        sleep = _RecordingSleep()
        acquire = _ScriptedAcquire(fail_count=0)
        result = lane_acquire.acquire_with_backoff(acquire, sleep_fn=sleep)
        self.assertTrue(result["acquired"])
        self.assertEqual(acquire.attempts, 1)
        self.assertEqual(sleep.calls, [])  # no waiting when lane is free

    def test_transient_fail_then_success_acquires_after_backoff(self) -> None:
        """(b) Transient contention: acquires after bounded backoff."""
        sleep = _RecordingSleep()
        acquire = _ScriptedAcquire(fail_count=2)
        result = lane_acquire.acquire_with_backoff(
            acquire,
            max_retries=5,
            backoff_schedule=(10.0, 20.0, 40.0, 60.0, 60.0),
            sleep_fn=sleep,
        )
        self.assertTrue(result["acquired"])
        self.assertEqual(acquire.attempts, 3)  # fail, fail, success
        self.assertEqual(sleep.calls, [10.0, 20.0])  # one sleep per failed attempt

    def test_exhaustion_returns_last_unacquired_bounded_total_wait(self) -> None:
        """(c)/total-bound: never acquires -> last non-acquired result, <=240s waited."""
        sleep = _RecordingSleep()
        acquire = _ScriptedAcquire(fail_count=99)
        result = lane_acquire.acquire_with_backoff(
            acquire,
            max_retries=5,
            backoff_schedule=(10.0, 20.0, 40.0, 60.0, 60.0),
            sleep_fn=sleep,
        )
        self.assertFalse(result["acquired"])
        self.assertEqual(acquire.attempts, 5)  # exactly max_retries attempts
        # 4 sleeps (no trailing sleep after the final failed attempt)
        self.assertEqual(sleep.calls, [10.0, 20.0, 40.0, 60.0])
        self.assertLessEqual(sum(sleep.calls), 240.0)

    def test_backoff_schedule_shorter_than_attempts_clamps_to_last(self) -> None:
        sleep = _RecordingSleep()
        acquire = _ScriptedAcquire(fail_count=99)
        lane_acquire.acquire_with_backoff(
            acquire,
            max_retries=4,
            backoff_schedule=(5.0,),
            sleep_fn=sleep,
        )
        self.assertEqual(sleep.calls, [5.0, 5.0, 5.0])  # clamps to last entry

    def test_default_schedule_total_wait_under_cap(self) -> None:
        """Module defaults stay within the 240s bounded-wait cap."""
        total = sum(
            lane_acquire.DEFAULT_ACQUIRE_BACKOFF_SECONDS[: lane_acquire.DEFAULT_ACQUIRE_MAX_RETRIES - 1]
        )
        self.assertLessEqual(total, 240.0)

    def test_invalid_max_retries_raises(self) -> None:
        with self.assertRaises(ValueError):
            lane_acquire.acquire_with_backoff(_ScriptedAcquire(0), max_retries=0)


def _install_fake_mcp(acquire_handler) -> dict:
    """Inject a fake mcp_server_lib so the in-function import resolves to a stub.

    Returns the saved sys.modules entries so the caller can restore them.
    """
    import sys
    import types

    saved = {
        name: sys.modules.get(name)
        for name in (
            "mcp_server_lib",
            "mcp_server_lib.main",
            "mcp_server_lib.compat",
            "mcp_server_lib.facade_exports",
            "mcp_server_lib.tools",
            "mcp_server_lib.tools.resource_lanes",
        )
    }
    pkg = types.ModuleType("mcp_server_lib")
    main_mod = types.ModuleType("mcp_server_lib.main")
    main_mod.init_room = lambda **_kwargs: None
    compat_mod = types.ModuleType("mcp_server_lib.compat")
    compat_mod.init = lambda *_a, **_k: None
    facade_mod = types.ModuleType("mcp_server_lib.facade_exports")
    tools_mod = types.ModuleType("mcp_server_lib.tools")
    lanes_mod = types.ModuleType("mcp_server_lib.tools.resource_lanes")
    lanes_mod.tool_resource_lane_acquire = acquire_handler
    pkg.main = main_mod
    pkg.compat = compat_mod
    pkg.facade_exports = facade_mod
    pkg.tools = tools_mod
    tools_mod.resource_lanes = lanes_mod
    sys.modules.update(
        {
            "mcp_server_lib": pkg,
            "mcp_server_lib.main": main_mod,
            "mcp_server_lib.compat": compat_mod,
            "mcp_server_lib.facade_exports": facade_mod,
            "mcp_server_lib.tools": tools_mod,
            "mcp_server_lib.tools.resource_lanes": lanes_mod,
        }
    )
    return saved


def _restore_modules(saved: dict) -> None:
    import sys

    for name, mod in saved.items():
        if mod is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = mod


class AcquireResourceLaneFailClosedTests(unittest.TestCase):
    """(c)/(d): the real entrypoint stays fail-closed and never spawns/holds."""

    def test_exhaustion_writes_holding_then_systemexit_no_spawn(self) -> None:
        """(c): never-acquire drives the real fail-closed branch via a fake MCP."""
        sleep = _RecordingSleep()
        calls = {"n": 0}

        def always_held(_handle, _args) -> str:
            calls["n"] += 1
            return json.dumps({"acquired": False, "reason": "held"})

        saved = _install_fake_mcp(always_held)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_root = Path(tmp) / "run"
                with self.assertRaises(SystemExit) as ctx:
                    lane_acquire.acquire_resource_lane(
                        run_root,
                        mock=False,
                        max_retries=5,
                        backoff_schedule=(10.0, 20.0, 40.0, 60.0, 60.0),
                        sleep_fn=sleep,
                    )
                self.assertEqual(str(ctx.exception), "resource_lane_acquire_failed")
                self.assertEqual(calls["n"], 5)  # bounded attempts, no spawn
                self.assertEqual(sleep.calls, [10.0, 20.0, 40.0, 60.0])
                self.assertLessEqual(sum(sleep.calls), 240.0)
                out = run_root / "prelaunch" / "resource_lane_holding.json"
                self.assertTrue(out.exists())  # holding written before fail-close
                holding = json.loads(out.read_text(encoding="utf-8"))
                self.assertFalse(holding["acquire_result"]["acquired"])
        finally:
            _restore_modules(saved)

    def test_transient_then_success_acquires_via_real_entrypoint(self) -> None:
        """(b): fake MCP fails twice then grants; entrypoint returns acquired holding."""
        sleep = _RecordingSleep()
        state = {"n": 0}

        def held_then_grant(_handle, _args) -> str:
            state["n"] += 1
            if state["n"] <= 2:
                return json.dumps({"acquired": False, "reason": "held"})
            return json.dumps({"acquired": True, "token": "tok"})

        saved = _install_fake_mcp(held_then_grant)
        try:
            with tempfile.TemporaryDirectory() as tmp:
                run_root = Path(tmp) / "run"
                holding = lane_acquire.acquire_resource_lane(
                    run_root, mock=False, sleep_fn=sleep
                )
                self.assertTrue(holding["acquire_result"]["acquired"])
                self.assertEqual(state["n"], 3)
                self.assertEqual(sleep.calls, [10.0, 20.0])
        finally:
            _restore_modules(saved)

    def test_mock_path_unchanged_first_attempt_success(self) -> None:
        """(a)/(d): mock path returns acquired holding, no retry/spawn surface."""
        with tempfile.TemporaryDirectory() as tmp:
            run_root = Path(tmp) / "run"
            holding = lane_acquire.acquire_resource_lane(run_root, mock=True)
            self.assertTrue(holding["acquire_result"]["acquired"])
            self.assertTrue(holding.get("mock"))
            self.assertTrue(
                (run_root / "prelaunch" / "resource_lane_holding.json").exists()
            )

    def test_no_force_release_or_spawn_tokens_in_source(self) -> None:
        """(d): assert no force-release / subprocess-spawn was introduced."""
        src = _MODULE_PATH.read_text(encoding="utf-8")
        self.assertNotIn("force=True", src)  # no force-release escape hatch
        self.assertNotIn("tool_resource_lane_release", src)  # acquire never releases
        self.assertNotIn("subprocess", src)  # no self-spawn
        self.assertNotIn("Popen", src)
        self.assertNotIn("os.fork", src)
        # fail-closed contract preserved verbatim
        self.assertIn('raise SystemExit("resource_lane_acquire_failed")', src)


if __name__ == "__main__":
    unittest.main()
