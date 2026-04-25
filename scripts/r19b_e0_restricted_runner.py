#!/usr/bin/env python3
"""R19b/E0 restricted BigCodeBench test runner.

Local to the E0 failure-surface scout. It gives BigCodeBench enough of the
normal Python ecosystem to run unit tests without relaxing ``calm.sandbox``.
Generated code is still executed in a fresh subprocess with a sanitized
process environment, temp cwd, timeout, rlimits, and network/process guards.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import resource
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


OUTCOMES = {"passed", "failed", "env_unsupported", "format_fail", "timeout"}
DEFAULT_TIMEOUT = 30.0


_CHILD_RUNNER = r'''
from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import types
import unittest


class E0Unsupported(RuntimeError):
    pass


def _install_runtime_guards():
    # Import network-capable libraries before socket.socket is patched. Some
    # modules subclass socket during import; patching socket first turns those
    # imports into TypeError instead of clean environment-unsupported signal.
    try:
        import requests

        def _blocked_request(self, method, url, **kwargs):
            raise E0Unsupported("network_blocked:requests")

        requests.sessions.Session.request = _blocked_request
    except Exception:
        pass

    try:
        import urllib.request as _urllib_request

        def _blocked_urlopen(*args, **kwargs):
            raise E0Unsupported("network_blocked:urllib")

        _urllib_request.urlopen = _blocked_urlopen
    except Exception:
        pass

    try:
        import ftplib

        class _BlockedFTP:
            def __init__(self, *args, **kwargs):
                raise E0Unsupported("network_blocked:ftplib")

        ftplib.FTP = _BlockedFTP
        ftplib.FTP_TLS = _BlockedFTP
    except Exception:
        pass

    try:
        import socket

        def _blocked_socket(*args, **kwargs):
            raise E0Unsupported("network_blocked")

        socket.socket = _blocked_socket
        socket.create_connection = _blocked_socket
    except Exception:
        pass

    try:
        import subprocess as _subprocess

        def _blocked_subprocess(*args, **kwargs):
            raise E0Unsupported("subprocess_blocked")

        _subprocess.Popen = _blocked_subprocess
        _subprocess.run = _blocked_subprocess
        _subprocess.call = _blocked_subprocess
        _subprocess.check_call = _blocked_subprocess
        _subprocess.check_output = _blocked_subprocess
    except Exception:
        pass

    try:
        import os as _os

        def _blocked_os_process(*args, **kwargs):
            raise E0Unsupported("subprocess_blocked:os")

        _os.system = _blocked_os_process
        _os.popen = _blocked_os_process
        for _name in ("spawnl", "spawnle", "spawnlp", "spawnlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
            if hasattr(_os, _name):
                setattr(_os, _name, _blocked_os_process)
    except Exception:
        pass

def _missing_dep_reason(exc):
    name = getattr(exc, "name", None)
    if name:
        return "missing_dep:" + str(name).split(".")[0]
    text = str(exc)
    import re
    m = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
    if m:
        return "missing_dep:" + m.group(1).split(".")[0]
    return "import_error:" + text[:120]


def _unsupported_from_text(text):
    import re
    if "E0Unsupported" in text:
        m = re.search(r"E0Unsupported:\s*([^\n]+)", text)
        return (m.group(1).strip() if m else "blocked_runtime_effect")[:160]
    if "ModuleNotFoundError" in text or "ImportError" in text:
        m = re.search(r"No module named ['\"]([^'\"]+)['\"]", text)
        if m:
            return "missing_dep:" + m.group(1).split(".")[0]
        return "import_error"
    return None


def _emit(payload):
    payload.setdefault("outcome", "failed")
    payload.setdefault("unsupported_reason", None)
    payload.setdefault("exit_code", 0)
    payload.setdefault("stdout", "")
    payload.setdefault("stderr", "")
    payload.setdefault("tests_passed", 0)
    payload.setdefault("tests_total", 0)
    payload.setdefault("error_type", None)
    print("E0_RESULT_JSON " + json.dumps(payload, sort_keys=True))


def _run():
    _install_runtime_guards()
    candidate_code = Path("candidate.py").read_text(encoding="utf-8")
    test_code = Path("test_code.py").read_text(encoding="utf-8")

    module = types.ModuleType("e0_case")
    module.__file__ = str(Path.cwd() / "e0_case.py")
    module.__dict__["__name__"] = "e0_case"
    sys.modules["e0_case"] = module

    captured_out = io.StringIO()
    captured_err = io.StringIO()

    try:
        candidate_compiled = compile(candidate_code, "<candidate>", "exec")
    except SyntaxError as exc:
        _emit({"outcome": "format_fail", "error_type": type(exc).__name__, "stderr": str(exc)})
        return

    try:
        test_compiled = compile(test_code, "<test_code>", "exec")
    except SyntaxError as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": "test_compile_error:" + type(exc).__name__,
            "error_type": type(exc).__name__,
            "stderr": str(exc),
        })
        return

    try:
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            exec(candidate_compiled, module.__dict__)
    except (ModuleNotFoundError, ImportError) as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": _missing_dep_reason(exc),
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + str(exc),
        })
        return
    except E0Unsupported as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": str(exc)[:160],
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue(),
        })
        return
    except Exception as exc:
        _emit({
            "outcome": "failed",
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + str(exc),
            "tests_total": 1,
        })
        return

    try:
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            exec(test_compiled, module.__dict__)
    except (ModuleNotFoundError, ImportError) as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": _missing_dep_reason(exc),
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + str(exc),
        })
        return
    except E0Unsupported as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": str(exc)[:160],
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue(),
        })
        return
    except AssertionError as exc:
        _emit({
            "outcome": "failed",
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + str(exc),
            "tests_total": 1,
        })
        return
    except Exception as exc:
        _emit({
            "outcome": "failed",
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + str(exc),
            "tests_total": 1,
        })
        return

    suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    total = suite.countTestCases()
    if total <= 0:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": "no_tests_discovered",
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue(),
        })
        return

    stream = io.StringIO()
    try:
        with contextlib.redirect_stdout(captured_out), contextlib.redirect_stderr(captured_err):
            result = unittest.TextTestRunner(stream=stream, verbosity=0).run(suite)
    except E0Unsupported as exc:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": str(exc)[:160],
            "error_type": type(exc).__name__,
            "stdout": captured_out.getvalue(),
            "stderr": captured_err.getvalue() + stream.getvalue(),
            "tests_total": total,
        })
        return

    stderr = captured_err.getvalue() + stream.getvalue()
    unsupported = None
    for _test, tb in list(result.errors) + list(result.failures):
        unsupported = _unsupported_from_text(tb)
        if unsupported:
            break

    if unsupported:
        _emit({
            "outcome": "env_unsupported",
            "unsupported_reason": unsupported,
            "stdout": captured_out.getvalue(),
            "stderr": stderr,
            "tests_total": total,
            "error_type": "EnvironmentUnsupported",
        })
        return

    skipped = len(getattr(result, "skipped", []))
    failed = len(result.failures) + len(result.errors)
    passed = max(0, result.testsRun - failed - skipped)
    outcome = "passed" if failed == 0 and result.testsRun == total else "failed"
    first_error_type = None
    if result.errors:
        first_error_type = result.errors[0][1].splitlines()[-1].split(":", 1)[0][:80]
    elif result.failures:
        first_error_type = "AssertionError"

    _emit({
        "outcome": outcome,
        "stdout": captured_out.getvalue(),
        "stderr": stderr,
        "tests_passed": passed,
        "tests_total": total,
        "error_type": first_error_type,
    })


_run()
'''


def _normalize_deps(deps: Iterable[str] | None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for dep in deps or []:
        root = str(dep).strip().split(".", 1)[0]
        if not root or not re.match(r"^[A-Za-z_]\w*$", root):
            continue
        if root not in seen:
            seen.add(root)
            out.append(root)
    return out


def _set_limits(timeout: float) -> None:
    cpu_seconds = max(1, int(timeout) + 2)
    limits = (
        (resource.RLIMIT_CPU, (cpu_seconds, cpu_seconds + 1)),
        (resource.RLIMIT_AS, (2 * 1024 * 1024 * 1024, 2 * 1024 * 1024 * 1024)),
        (resource.RLIMIT_NOFILE, (128, 128)),
        (resource.RLIMIT_NPROC, (16, 16)),
    )
    for res, value in limits:
        try:
            resource.setrlimit(res, value)
        except Exception:
            pass


def _env_for_child(tmpdir: str, python_executable: str) -> dict[str, str]:
    py_dir = str(Path(python_executable).resolve().parent)
    return {
        "HOME": tmpdir,
        "TMPDIR": tmpdir,
        "TEMP": tmpdir,
        "TMP": tmpdir,
        "PATH": os.pathsep.join([py_dir, "/usr/bin", "/bin"]),
        "PYTHONHASHSEED": "0",
        "PYTHONNOUSERSITE": "1",
        "MPLBACKEND": "Agg",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }


def _parse_child_result(stdout: str) -> dict:
    marker = "E0_RESULT_JSON "
    for line in reversed(stdout.splitlines()):
        if line.startswith(marker):
            data = json.loads(line[len(marker):])
            if data.get("outcome") not in OUTCOMES:
                data["outcome"] = "failed"
                data["error_type"] = data.get("error_type") or "InvalidOutcome"
            return data
    return {
        "outcome": "failed",
        "unsupported_reason": None,
        "exit_code": 0,
        "stdout": stdout,
        "stderr": "child result marker missing",
        "tests_passed": 0,
        "tests_total": 0,
        "error_type": "MissingResultMarker",
    }


def run_test_restricted(
    code: str,
    test_code: str,
    deps: Iterable[str] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    python_executable: str | None = None,
) -> dict:
    """Run ``code`` against ``test_code`` in an E0-specific subprocess."""
    if not code or not code.strip():
        return {
            "outcome": "format_fail",
            "unsupported_reason": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "empty extracted code",
            "tests_passed": 0,
            "tests_total": 0,
            "error_type": "NoCode",
            "deps": _normalize_deps(deps),
        }
    if not test_code or not test_code.strip():
        return {
            "outcome": "env_unsupported",
            "unsupported_reason": "missing_test_code",
            "exit_code": None,
            "stdout": "",
            "stderr": "",
            "tests_passed": 0,
            "tests_total": 0,
            "error_type": "MissingTestCode",
            "deps": _normalize_deps(deps),
        }

    python_executable = python_executable or os.environ.get("E0_PYTHON") or sys.executable
    with tempfile.TemporaryDirectory(prefix="r19b_e0_") as tmpdir:
        tmp_path = Path(tmpdir)
        (tmp_path / "candidate.py").write_text(code, encoding="utf-8")
        (tmp_path / "test_code.py").write_text(test_code, encoding="utf-8")
        (tmp_path / "runner.py").write_text(_CHILD_RUNNER, encoding="utf-8")
        env = _env_for_child(tmpdir, python_executable)
        preexec_fn = (lambda: _set_limits(timeout)) if hasattr(os, "fork") else None
        try:
            proc = subprocess.run(
                [python_executable, "runner.py"],
                cwd=tmpdir,
                env=env,
                text=True,
                capture_output=True,
                timeout=timeout,
                preexec_fn=preexec_fn,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "outcome": "timeout",
                "unsupported_reason": None,
                "exit_code": None,
                "stdout": (exc.stdout or "") if isinstance(exc.stdout, str) else "",
                "stderr": (exc.stderr or "") if isinstance(exc.stderr, str) else "",
                "tests_passed": 0,
                "tests_total": 0,
                "error_type": "TimeoutExpired",
                "deps": _normalize_deps(deps),
            }

    result = _parse_child_result(proc.stdout)
    result["exit_code"] = proc.returncode
    if proc.stderr:
        result["stderr"] = (result.get("stderr") or "") + proc.stderr
    result["deps"] = _normalize_deps(deps)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code", required=True, help="Path to extracted candidate code")
    parser.add_argument("--test-code", required=True, help="Path to BigCodeBench test code")
    parser.add_argument("--deps", default="", help="Comma-separated dependency roots")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument("--python", default=os.environ.get("E0_PYTHON") or sys.executable)
    args = parser.parse_args()

    code = Path(args.code).read_text(encoding="utf-8")
    test_code = Path(args.test_code).read_text(encoding="utf-8")
    deps = [d.strip() for d in args.deps.split(",") if d.strip()]
    result = run_test_restricted(
        code,
        test_code,
        deps=deps,
        timeout=args.timeout,
        python_executable=args.python,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
