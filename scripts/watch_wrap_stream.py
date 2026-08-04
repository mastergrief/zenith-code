"""Shared stream drain + line policy for bin/watch-wrap.

Import-safe from any cwd. Policies here require durable test coverage.
"""
from __future__ import annotations

import os
import re
import select
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class Category:
    name: str
    pattern: Optional[re.Pattern]
    prefix: str


def build_categories(
    error: str = "",
    progress: str = "",
    success: str = "",
) -> list[Category]:
    cats: list[Category] = []
    if error:
        cats.append(Category("error", re.compile(error), "[ERR]"))
    if progress:
        cats.append(Category("progress", re.compile(progress), "[PROG]"))
    if success:
        cats.append(Category("success", re.compile(success), "[OK]"))
    return cats


def classify(line: str, cats: list[Category]) -> tuple[Optional[Category], bool]:
    """First matching category wins (registration order = precedence)."""
    for c in cats:
        if c.pattern is not None and c.pattern.search(line):
            return c, True
    if not cats:
        return None, True
    return None, False


def ingest_bytes(byte_buf: bytes, data: bytes) -> tuple[bytes, list[str]]:
    byte_buf += data
    lines: list[str] = []
    while b"\n" in byte_buf:
        line, byte_buf = byte_buf.split(b"\n", 1)
        lines.append(line.decode(errors="replace").rstrip("\r"))
    return byte_buf, lines


def flush_partial_on_eof(byte_buf: bytes) -> tuple[bytes, Optional[str]]:
    if byte_buf.strip():
        return b"", byte_buf.decode(errors="replace").rstrip("\r")
    return b"", None


def check_stop(line: str, stop_re: Optional[re.Pattern]) -> bool:
    return bool(stop_re and stop_re.search(line))


def heartbeat_due(
    now: float, last_activity: float, hb_interval: float, hb_mult: int
) -> bool:
    if not hb_interval:
        return False
    return (now - last_activity) >= hb_interval * hb_mult


def next_hb_mult(hb_mult: int, last_raw: str, hb_last_line: Optional[str]) -> int:
    return min(hb_mult * 2, 8) if last_raw == hb_last_line else 2


@dataclass
class DrainConfig:
    stop_re: Optional[re.Pattern] = None
    cats: Optional[list[Category]] = None
    heartbeat: float = 0.0
    coalesce: float = 0.0
    replay: int = 10
    select_cap: float = 60.0
    select_floor: float = 0.1


@dataclass
class DrainResult:
    """Bounded history only: replay deque. stop_triggered for harness liveness."""

    stop_triggered: bool = False
    replay: deque = field(default_factory=lambda: deque(maxlen=10))
    hb_mult: int = 1
    hb_last_line: Optional[str] = None


def drain_fd_loop(
    fd: int,
    *,
    cfg: DrainConfig,
    on_event: Callable[[str], None],
    should_break_external: Optional[Callable[[], Optional[str]]] = None,
    now_fn: Callable[[], float] = time.time,
    read_fn: Optional[Callable[[int, int], bytes]] = None,
    select_fn: Optional[Callable] = None,
) -> DrainResult:
    """Drain fd via os.read + pending_lines. NEVER TextIO.readline.

    Does NOT emit replay bodies (harness prints HEAD-parity raw indented lines).
    """
    cats = cfg.cats or []
    stop_re = cfg.stop_re
    read_fn = read_fn or os.read
    select_fn = select_fn or select.select

    last_activity = now_fn()
    coalesce_buf: list[tuple[Optional[Category], str]] = []
    coalesce_start: Optional[float] = None
    replay: deque[str] = deque(maxlen=max(1, cfg.replay))
    hb_mult = 1
    hb_last_line: Optional[str] = None
    byte_buf = b""
    pending_lines: list[str] = []
    eof = False
    stop_triggered = False

    def flush_coalesce() -> None:
        nonlocal coalesce_buf, coalesce_start
        if not coalesce_buf:
            return
        cat, last_line = coalesce_buf[-1]
        prefix = cat.prefix if cat else "[OTHER]"
        count = len(coalesce_buf)
        if count == 1:
            on_event(f"{prefix} {last_line}")
        else:
            on_event(f"[COALESCED x{count}] {prefix} {last_line}")
        coalesce_buf = []
        coalesce_start = None

    while True:
        if should_break_external is not None:
            ext = should_break_external()
            if ext is not None:
                on_event(ext)
                break

        now = now_fn()
        timeout_parts: list[float] = []
        if cfg.heartbeat:
            timeout_parts.append(cfg.heartbeat * hb_mult - (now - last_activity))
        if cfg.coalesce and coalesce_start is not None:
            timeout_parts.append(cfg.coalesce - (now - coalesce_start))
        wait = min([t for t in timeout_parts if t > 0], default=1.0)
        wait = max(cfg.select_floor, min(wait, cfg.select_cap))

        ready = False
        if not pending_lines and not eof:
            ready_list, _, _ = select_fn([fd], [], [], wait)
            now = now_fn()
            ready = bool(ready_list)
            if ready:
                try:
                    chunk = read_fn(fd, 65536)
                except OSError:
                    chunk = b""
                if not chunk:
                    eof = True
                    byte_buf, partial = flush_partial_on_eof(byte_buf)
                    if partial is not None:
                        pending_lines.append(partial)
                else:
                    byte_buf, lines = ingest_bytes(byte_buf, chunk)
                    pending_lines.extend(lines)
        else:
            now = now_fn()
            ready = True

        if (
            coalesce_start is not None
            and cfg.coalesce
            and (now - coalesce_start) >= cfg.coalesce
        ):
            flush_coalesce()

        if not ready and not pending_lines:
            if eof:
                break
            if heartbeat_due(now, last_activity, cfg.heartbeat, hb_mult):
                last_raw = replay[-1] if replay else "<no output yet>"
                suffix = f" (x{hb_mult})" if hb_mult > 1 else ""
                on_event(
                    f"[HEARTBEAT{suffix}] silent {int(now - last_activity)}s, "
                    f"last: {last_raw[:120]}"
                )
                hb_mult = next_hb_mult(hb_mult, last_raw, hb_last_line)
                hb_last_line = last_raw
                last_activity = now
            continue

        if not pending_lines:
            if eof:
                break
            continue

        line = pending_lines.pop(0)
        if not line:
            continue
        replay.append(line)
        last_activity = now
        hb_mult = 1
        hb_last_line = None

        cat, should_emit = classify(line, cats)
        if should_emit:
            if cat and cat.name == "error":
                on_event(f"{cat.prefix} {line}")
            elif cfg.coalesce:
                if coalesce_start is None:
                    coalesce_start = now
                coalesce_buf.append((cat, line))
            else:
                prefix = cat.prefix if cat else "[OTHER]"
                on_event(f"{prefix} {line}")

        if check_stop(line, stop_re):
            if coalesce_buf:
                flush_coalesce()
            on_event(f"[STOP-TRIGGER] {line}")
            stop_triggered = True
            break

    if coalesce_buf:
        flush_coalesce()

    return DrainResult(
        stop_triggered=stop_triggered,
        replay=replay,
        hb_mult=hb_mult,
        hb_last_line=hb_last_line,
    )


def emit_replay(replay: deque, replay_n: int, on_marker: Callable[[str], None]) -> None:
    """HEAD parity: marker via on_marker (gets [T+]); bodies printed RAW indented."""
    if replay and replay_n > 0:
        on_marker(f"[REPLAY last {len(replay)} lines]")
        for r in replay:
            print(f"  {r}", flush=True)
