"""llama-server hot-swap orchestrator.

llama-server does not support runtime model reload, so swapping models
means killing the current process and starting a new one with a different
GGUF. This module manages that lifecycle.

## Design

- One server runs at a time on a fixed port (default :8080).
- ``LlamaServerManager`` tracks the subprocess it owns, or adopts an
  externally-started server by reading ``/props`` to discover its model path.
- ``swap(target)`` is a no-op when ``target`` is already loaded; otherwise
  it stops the current process and starts a new one, blocking until the
  new server reports healthy.
- Model paths are discovered on disk via ``discover_specialist_models()``,
  which scans ``~/models/`` for known domain name patterns.

## Integration point

``SpecialistCoordinator`` consults ``discover_specialist_models()`` at
construction time. If any specialists are found on disk, it switches to
hot-swap mode: one agent per domain, all pointing at :8080, with a swap
triggered before each delegated call.

## Why not Ollama?

Ollama keeps multiple models hot on disk and can serve them concurrently,
but it does not quantize the KV cache. At 8 GB VRAM our 4B model at 64K
context *requires* ``--cache-type-k q4_0 --cache-type-v q4_0`` (see the
"Why Q4 KV Cache" section of ``.claude/rules/architecture.md``). llama.cpp
hot-swap is the only way to keep specialists on the same physical GPU.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Optional

DEFAULT_PORT = 8080
DEFAULT_BINARY = Path.home() / "llama.cpp" / "build" / "bin" / "llama-server"
DEFAULT_CTX = 65536
DEFAULT_LOG = Path("/tmp/llama-server.log")
STARTUP_TIMEOUT = 60.0  # seconds; loading + KV cache allocation can take a while
HEALTH_POLL_INTERVAL = 0.5
STOP_TIMEOUT = 10.0
PORT_FREE_TIMEOUT = 10.0


class ModelSwapError(RuntimeError):
    """Raised when a swap operation cannot complete."""


class LlamaServerManager:
    """Manages a llama-server subprocess for hot-swapping GGUF models.

    The manager can either own its subprocess (``start()`` / ``stop()``) or
    adopt an externally-started server by polling ``/props``. After a swap,
    the manager always owns the new process.
    """

    def __init__(
        self,
        port: int = DEFAULT_PORT,
        binary: Path = DEFAULT_BINARY,
        ctx_size: int = DEFAULT_CTX,
        gpu_layers: int = 999,
        cache_type_k: str = "q4_0",
        cache_type_v: str = "q4_0",
        log_file: Path = DEFAULT_LOG,
    ):
        self.port = port
        self.binary = Path(binary)
        self.ctx_size = ctx_size
        self.gpu_layers = gpu_layers
        self.cache_type_k = cache_type_k
        self.cache_type_v = cache_type_v
        self.log_file = Path(log_file)
        self._process: Optional[subprocess.Popen] = None

    # ── Status ─────────────────────────────────────────────────────

    @property
    def base_url(self) -> str:
        return f"http://localhost:{self.port}"

    def is_running(self) -> bool:
        """True if a server (ours or not) responds on /health."""
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=2) as r:
                if r.status != 200:
                    return False
                data = json.loads(r.read())
                return data.get("status") == "ok"
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            return False

    def current_model(self) -> Optional[Path]:
        """Return the model path of the currently loaded server, or None.

        Uses /props which works regardless of whether this manager owns
        the process. This is the source of truth — don't cache it.
        """
        try:
            with urllib.request.urlopen(f"{self.base_url}/props", timeout=2) as r:
                if r.status != 200:
                    return None
                data = json.loads(r.read())
                model_path = data.get("model_path")
                if model_path:
                    return Path(model_path)
        except (urllib.error.URLError, TimeoutError, ConnectionError, json.JSONDecodeError):
            pass
        return None

    def owns_process(self) -> bool:
        """True if this manager started the currently-running server."""
        return self._process is not None and self._process.poll() is None

    # ── Lifecycle ──────────────────────────────────────────────────

    def start(self, model_path: Path, wait: bool = True) -> None:
        """Start llama-server with the given model. Fails if a server is already running."""
        if self.is_running():
            raise ModelSwapError(
                f"a server is already running on :{self.port} — use swap() or stop_any() first"
            )

        model = Path(model_path).expanduser()
        if not model.exists():
            raise ModelSwapError(f"model file not found: {model}")
        if not self.binary.exists():
            raise ModelSwapError(f"llama-server binary not found: {self.binary}")

        cmd = [
            str(self.binary),
            "-m", str(model),
            "--ctx-size", str(self.ctx_size),
            "--cache-type-k", self.cache_type_k,
            "--cache-type-v", self.cache_type_v,
            "-ngl", str(self.gpu_layers),
            "--port", str(self.port),
        ]

        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        log = open(self.log_file, "w")
        # start_new_session decouples the child from our SIGINT — we'll kill
        # it explicitly in stop().
        self._process = subprocess.Popen(
            cmd,
            stdout=log,
            stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

        if wait:
            self._wait_for_health()

    def stop(self, timeout: float = STOP_TIMEOUT) -> None:
        """Stop the managed server. Safe to call if nothing is running.

        Only stops the process this manager started. To stop an externally-
        started server, use ``stop_any()``.
        """
        if self._process is None:
            return
        if self._process.poll() is not None:
            self._process = None
            return

        try:
            self._process.terminate()
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            try:
                self._process.wait(timeout=5.0)
            except subprocess.TimeoutExpired:
                pass
        self._process = None
        self._wait_port_free()

    def stop_any(self, timeout: float = STOP_TIMEOUT) -> None:
        """Stop whatever server is running on our port, managed or not.

        For the externally-started case, this reads the PID from the
        listening socket via /proc/net/tcp, then sends SIGTERM. Falls back
        to polling /health until it goes away.
        """
        if self._process is not None and self._process.poll() is None:
            self.stop(timeout=timeout)
            return

        if not self.is_running():
            return

        pid = _find_listening_pid(self.port)
        if pid is None:
            raise ModelSwapError(
                f"a server is running on :{self.port} but we can't find its PID"
            )

        try:
            os.kill(pid, 15)  # SIGTERM
        except ProcessLookupError:
            return
        except PermissionError:
            raise ModelSwapError(
                f"cannot kill pid {pid} on :{self.port} — permission denied"
            )

        # Wait for the port to free
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if not self.is_running():
                return
            time.sleep(0.2)

        # Escalate to SIGKILL
        try:
            os.kill(pid, 9)
        except ProcessLookupError:
            return

        self._wait_port_free()

    # ── Swap ───────────────────────────────────────────────────────

    def swap(
        self,
        model_path: Path,
        on_event: Optional[Callable[[str, Path], None]] = None,
    ) -> float:
        """Swap to a different model. Returns elapsed seconds.

        No-op if the target model is already loaded. If an external server
        is running with a different model, it's stopped and replaced.
        The ``on_event`` callback receives ('start'|'stopped'|'ready'|'noop', path).
        """
        target = Path(model_path).expanduser().resolve()
        if not target.exists():
            raise ModelSwapError(f"target model not found: {target}")

        current = self.current_model()
        if current is not None:
            try:
                current_resolved = current.resolve()
            except OSError:
                current_resolved = current
            if current_resolved == target:
                if on_event:
                    on_event("noop", target)
                return 0.0

        start_time = time.monotonic()
        if on_event:
            on_event("start", target)

        self.stop_any()
        if on_event:
            on_event("stopped", target)

        self.start(target, wait=True)
        if on_event:
            on_event("ready", target)

        return time.monotonic() - start_time

    # ── Internal ───────────────────────────────────────────────────

    def _wait_for_health(self) -> None:
        deadline = time.monotonic() + STARTUP_TIMEOUT
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise ModelSwapError(
                    f"llama-server exited with code {self._process.returncode}. "
                    f"Check {self.log_file}"
                )
            if self.is_running():
                return
            time.sleep(HEALTH_POLL_INTERVAL)

        self.stop()
        raise ModelSwapError(
            f"llama-server failed to become healthy within {STARTUP_TIMEOUT}s. "
            f"Check {self.log_file}"
        )

    def _wait_port_free(self) -> None:
        deadline = time.monotonic() + PORT_FREE_TIMEOUT
        while time.monotonic() < deadline:
            if not self.is_running():
                return
            time.sleep(0.2)


# ── Specialist discovery ────────────────────────────────────────────


def discover_specialist_models(models_dir: Optional[Path] = None) -> dict[str, Path]:
    """Scan ``~/models/`` for specialist GGUF files, keyed by domain.

    Looks for any of these naming patterns per domain:
      - ``specialist-<domain>*.gguf``
      - ``Qwen*<domain>*.gguf``
      - ``<domain>*.gguf``

    Prefers Q5_K_M quantization when multiple matches exist. Returns an
    empty dict if no specialists are present (base-only deployment).
    """
    if models_dir is None:
        models_dir = Path.home() / "models"
    models_dir = Path(models_dir).expanduser()
    if not models_dir.exists():
        return {}

    from agents.distill.config import DOMAINS

    specialists: dict[str, Path] = {}
    for domain in DOMAINS:
        if domain == "orchestrator":
            # Orchestrator is a router; treat it like any other specialist
            pass
        patterns = [
            f"specialist-{domain}*.gguf",
            f"Qwen*{domain}*.gguf",
            f"{domain}*.gguf",
        ]
        found: list[Path] = []
        for pattern in patterns:
            found.extend(models_dir.glob(pattern))
        if not found:
            continue
        # Prefer Q5_K_M > Q4_K_M > anything else
        q5 = [m for m in found if "Q5_K_M" in m.name or "q5_k_m" in m.name]
        q4 = [m for m in found if "Q4_K_M" in m.name or "q4_k_m" in m.name]
        specialists[domain] = q5[0] if q5 else (q4[0] if q4 else found[0])

    return specialists


def default_base_model() -> Path:
    """Return the expected path of the 4B reasoning base GGUF."""
    return Path.home() / "models" / "Qwen3.5-4B.Q5_K_M.gguf"


# ── Port → PID lookup ──────────────────────────────────────────────


def _find_listening_pid(port: int) -> Optional[int]:
    """Find the PID listening on the given TCP port via /proc.

    Parses ``/proc/net/tcp`` (and ``/proc/net/tcp6``) to find the inode of
    the listening socket, then walks ``/proc/*/fd/*`` to find which process
    owns it. Returns None if not found or not accessible.
    """
    port_hex = f"{port:04X}"
    target_inodes = set()

    for tcp_file in ("/proc/net/tcp", "/proc/net/tcp6"):
        try:
            with open(tcp_file) as f:
                lines = f.readlines()[1:]
        except FileNotFoundError:
            continue

        for line in lines:
            parts = line.split()
            if len(parts) < 10:
                continue
            local_addr = parts[1]
            state = parts[3]
            if state != "0A":  # 0A = LISTEN
                continue
            if local_addr.endswith(f":{port_hex}"):
                target_inodes.add(parts[9])

    if not target_inodes:
        return None

    # Walk /proc/*/fd/ looking for the socket inode
    for pid_dir in Path("/proc").iterdir():
        if not pid_dir.name.isdigit():
            continue
        fd_dir = pid_dir / "fd"
        if not fd_dir.exists():
            continue
        try:
            for fd in fd_dir.iterdir():
                try:
                    target = os.readlink(fd)
                except (OSError, PermissionError):
                    continue
                if target.startswith("socket:["):
                    inode = target[len("socket:["):-1]
                    if inode in target_inodes:
                        return int(pid_dir.name)
        except (PermissionError, OSError):
            continue

    return None
