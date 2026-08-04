"""Durable characterization for watch-wrap stream (W1–W10, N1). <500 lines."""
from __future__ import annotations

import importlib.util
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
STREAM_PATH = REPO / "scripts" / "watch_wrap_stream.py"
WATCH_WRAP = REPO / "bin" / "watch-wrap"


def load_stream():
    spec = importlib.util.spec_from_file_location("watch_wrap_stream", STREAM_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


S = load_stream()


def one_shot_io(payload: bytes):
    st = {"n": 0}

    def read_fn(_fd, _n):
        if st["n"] == 0:
            st["n"] = 1
            return payload
        return b""

    def select_fn(r, w, x, timeout):
        if st["n"] == 0:
            return (r, [], [])
        return ([], [], [])

    return read_fn, select_fn, st


def run_drain(payload, *, stop=None, cats=None, hb=0.0, coal=0.0, replay=5, **kw):
    events: list[str] = []
    rf, sf, st = one_shot_io(payload)
    cfg = S.DrainConfig(
        stop_re=re.compile(stop) if stop else None,
        cats=cats if cats is not None else [],
        heartbeat=hb,
        coalesce=coal,
        replay=replay,
    )
    res = S.drain_fd_loop(
        0, cfg=cfg, on_event=events.append, read_fn=rf, select_fn=sf, **kw
    )
    return events, res, st


def test_W1_burst_terminal_last():
    """Fails if last line stranded / no STOP-TRIGGER."""
    cats = S.build_categories(success=r"PACKET_TERMINAL")
    ev, res, _ = run_drain(
        b'{"status":"ok"}\nPACKET_TERMINAL BRANCH_X\n',
        stop=r"PACKET_TERMINAL",
        cats=cats,
    )
    assert res.stop_triggered
    assert any(e.startswith("[STOP-TRIGGER]") and "BRANCH_X" in e for e in ev)
    assert any(e.startswith("[OK]") and "BRANCH_X" in e for e in ev)


def test_W2_partial_eof():
    """Fails if partial lost or premature stop."""
    buf, lines = S.ingest_bytes(b"", b"PACKET_TERM")
    assert lines == [] and buf == b"PACKET_TERM"
    _, p = S.flush_partial_on_eof(buf)
    assert p == "PACKET_TERM"
    buf, lines = S.ingest_bytes(b"", b"a\nb\npartial")
    assert lines == ["a", "b"] and buf == b"partial"
    _, p = S.flush_partial_on_eof(b"   ")
    assert p is None


def test_W3_missing_log_rc2():
    """Fails if hang, wrong rc, or rc=0."""
    missing = REPO / "tests" / f"_miss_{os.getpid()}.log"
    if missing.exists():
        missing.unlink()
    t0 = time.time()
    p = subprocess.run(
        [sys.executable, str(WATCH_WRAP), "--log", str(missing), "--stop-on", "DONE"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    el = time.time() - t0
    assert p.returncode == 2 and "log not found" in p.stdout
    assert 9.0 <= el <= 15.0


def test_W4_stop_raw_past_filters():
    """Fails if stop requires category match."""
    cats = S.build_categories(progress=r"PROGRESS")
    ev, res, _ = run_drain(
        b"PROGRESS 1\nTERMINAL_NOW\n", stop=r"TERMINAL_NOW", cats=cats
    )
    assert res.stop_triggered
    assert any("TERMINAL_NOW" in e and e.startswith("[STOP-TRIGGER]") for e in ev)


def test_W5_replay_no_refire_and_head_parity(capsys):
    """Fails if second STOP or bodies get [T+] prefix."""
    cats = S.build_categories(success=r"PACKET_TERMINAL")
    ev, res, _ = run_drain(b"PACKET_TERMINAL X\n", stop=r"PACKET_TERMINAL", cats=cats)
    assert sum(1 for e in ev if e.startswith("[STOP-TRIGGER]")) == 1
    # harness-style emit_replay
    S.emit_replay(res.replay, 5, lambda m: print(f"[T+   0s] {m}", flush=True))
    out = capsys.readouterr().out
    assert "[REPLAY last" in out
    body_lines = [ln for ln in out.splitlines() if ln.startswith("  ")]
    assert body_lines and all(not ln.lstrip().startswith("[T+") for ln in body_lines)
    assert any(ln == "  PACKET_TERMINAL X" for ln in body_lines)


def test_W6_pending_order():
    """Fails if only first line of chunk processed."""
    st_box = {}
    rf, sf, st = one_shot_io(b"L1\nL2\nL3\n")
    st_box["selects"] = 0
    orig = sf

    def sf2(r, w, x, timeout):
        st_box["selects"] += 1
        return orig(r, w, x, timeout)

    ev: list[str] = []
    cfg = S.DrainConfig(stop_re=re.compile(r"L3"), cats=[], replay=5)
    S.drain_fd_loop(0, cfg=cfg, on_event=ev.append, read_fn=rf, select_fn=sf2)
    bodies = [e.split(" ", 1)[1] for e in ev if e.startswith("[OTHER]")]
    assert bodies == ["L1", "L2", "L3"]
    assert st_box["selects"] <= 2


def test_W7_heartbeat_loop_backoff_and_reset():
    """Fails if first post-activity beat carries (xN) or arrives on un-reset schedule.

    Ordered proof: silent backoff climbs; real_line resets; FIRST heartbeat
    AFTER [OTHER] real_line is BASE form (no (xN)) at base interval; then
    subsequent silence backs off again ((x2)).
    """
    # (ts, event) log — injected clock stamps each emission
    timed: list[tuple[float, str]] = []
    clock = {"t": 0.0}
    phase = {"p": "silent1"}  # silent → activity → silent2

    def now():
        return clock["t"]

    def on_event(msg: str) -> None:
        timed.append((clock["t"], msg))

    def read_fn(_fd, _n):
        if phase["p"] == "activity":
            phase["p"] = "silent2"
            return b"real_line\n"
        return b""

    def select_fn(r, w, x, timeout):
        clock["t"] += max(timeout, 0.1)
        if phase["p"] == "silent1" and clock["t"] >= 15.5:
            phase["p"] = "activity"
            return (r, [], [])
        if phase["p"] == "activity":
            return (r, [], [])
        return ([], [], [])

    def should_break():
        # need: pre-activity backoff beats + post-activity base + post-activity (x2)
        post_idx = next(
            (i for i, (_, e) in enumerate(timed) if e.startswith("[OTHER] real_line")),
            None,
        )
        if post_idx is None:
            return None
        post_hb = [
            (t, e)
            for t, e in timed[post_idx + 1 :]
            if e.startswith("[HEARTBEAT")
        ]
        if len(post_hb) >= 2:
            return "[PID EXIT pid=0] elapsed 0s"
        return None

    cfg = S.DrainConfig(
        cats=[], heartbeat=1.0, coalesce=0, replay=3, select_cap=1.0, select_floor=0.1
    )
    S.drain_fd_loop(
        0,
        cfg=cfg,
        on_event=on_event,
        should_break_external=should_break,
        now_fn=now,
        read_fn=read_fn,
        select_fn=select_fn,
    )
    # pre-activity: climb
    pre_hb = []
    act_i = None
    for i, (t, e) in enumerate(timed):
        if e.startswith("[OTHER] real_line"):
            act_i = i
            act_t = t
            break
        if e.startswith("[HEARTBEAT"):
            pre_hb.append(e)
    assert act_i is not None, timed
    assert any("(x2)" in e for e in pre_hb), pre_hb
    assert any("(x4)" in e or "(x8)" in e for e in pre_hb), pre_hb

    post_hb = [
        (t, e) for t, e in timed[act_i + 1 :] if e.startswith("[HEARTBEAT")
    ]
    assert len(post_hb) >= 2, (timed, post_hb)
    t0, first = post_hb[0]
    # FIRST post-activity beat: BASE form — no (xN) suffix
    assert "(x" not in first, f"first post-activity beat not base: {first}"
    assert first.startswith("[HEARTBEAT]"), first
    # occurs at base interval (1.0) after activity timestamp (± select floor)
    assert abs((t0 - act_t) - 1.0) <= 0.15, f"base interval off: act={act_t} beat={t0}"
    # THEN subsequent silence backs off again
    t1, second = post_hb[1]
    assert "(x2)" in second, f"second post-activity beat not (x2): {second}"
    assert abs((t1 - t0) - 2.0) <= 0.15, f"backoff interval off: {t0}→{t1}"


def test_W8_coalesce_mechanism():
    """Fails if flood not coalesced, window not flushed, or errors delayed."""
    # N same-category → one [COALESCED xN]
    events: list[str] = []
    clock = {"t": 0.0}
    payload = b"prog a\nprog b\nprog c\n"
    rf, sf, st = one_shot_io(payload)

    def now():
        return clock["t"]

    def select_fn(r, w, x, timeout):
        if st["n"] == 0:
            return (r, [], [])
        clock["t"] += max(timeout, 0.05)
        return ([], [], [])

    cfg = S.DrainConfig(
        stop_re=None,
        cats=S.build_categories(progress=r"prog"),
        heartbeat=0,
        coalesce=0.5,
        replay=5,
        select_cap=0.2,
        select_floor=0.05,
    )
    # force EOF after payload by second read empty + advance past coalesce window
    def read_fn(fd, n):
        data = rf(fd, n)
        if data == b"":
            clock["t"] += 1.0  # expire coalesce on next loop
        return data

    beats = {"n": 0}

    def should_break():
        # stop once coalesce flushed or time large
        if any("[COALESCED" in e for e in events) or clock["t"] > 2.0:
            return "[PID EXIT pid=0] elapsed 0s"
        return None

    S.drain_fd_loop(
        0,
        cfg=cfg,
        on_event=events.append,
        should_break_external=should_break,
        now_fn=now,
        read_fn=read_fn,
        select_fn=select_fn,
    )
    coal = [e for e in events if e.startswith("[COALESCED")]
    assert coal and "x3" in coal[0], events

    # error bypass
    ev2, _, _ = run_drain(
        b"prog a\nERR boom\nprog b\nSTOP\n",
        stop=r"STOP",
        cats=S.build_categories(error=r"ERR", progress=r"prog"),
        coal=10.0,
    )
    assert any(e.startswith("[ERR]") and "boom" in e for e in ev2)
    # precedence
    cats = S.build_categories(error=r"ERR|BOTH", progress=r"PROG|BOTH")
    cat, _ = S.classify("BOTH match", cats)
    assert cat and cat.name == "error"


def test_W9_pid_promptness():
    """Fails if [PID EXIT] delayed > 2×poll(1s)+select_cap(5s)+slack = 8s from death."""
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.2)"])
    threading.Thread(target=child.wait, daemon=True).start()
    log = REPO / "tests" / f"_pid_{os.getpid()}.log"
    log.write_text("")
    death_deadline = time.time() + 0.5  # child dies ~0.2s
    try:
        t0 = time.time()
        p = subprocess.run(
            [
                sys.executable,
                str(WATCH_WRAP),
                "--log",
                str(log),
                "--pid",
                str(child.pid),
                "--heartbeat",
                "30",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        elapsed = time.time() - t0
        assert p.returncode == 0 and f"[PID EXIT pid={child.pid}]" in p.stdout
        # bound: poll 1s + select_cap 5s + 2s slack from process start
        assert elapsed <= 8.0, f"not prompt: {elapsed}s"
    finally:
        if log.exists():
            log.unlink()


def test_W10_wrap_stop_rc0_no_orphan():
    """Fails if orphaned child or nonzero wrapper rc on clean stop (terminated_on_stop)."""
    src = "import time\nprint('PACKET_TERMINAL DONE', flush=True)\ntime.sleep(60)\n"
    p = subprocess.run(
        [
            sys.executable,
            str(WATCH_WRAP),
            "--stop-on",
            "PACKET_TERMINAL",
            "--success",
            "PACKET_TERMINAL",
            "--replay",
            "3",
            "--",
            sys.executable,
            "-c",
            src,
        ],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert p.returncode == 0
    assert "[STOP-TRIGGER] PACKET_TERMINAL DONE" in p.stdout
    assert "stop-triggered" in p.stdout
    for ln in p.stdout.splitlines():
        if ln.startswith("  PACKET_TERMINAL"):
            assert not ln.strip().startswith("[T+")
            break
    else:
        assert "[REPLAY last" in p.stdout


def test_W10b_natural_exit_keeps_nonzero_rc():
    """Fails if natural nonzero after terminal line is masked to 0.

    Child prints PACKET_TERMINAL then exits rc=7 immediately (no sleep).
    Wrapper must keep rc==7 and must NOT emit '(stop-triggered)' suffix.
    """
    src = (
        "import sys\n"
        "print('PACKET_TERMINAL DONE', flush=True)\n"
        "sys.exit(7)\n"
    )
    p = subprocess.run(
        [
            sys.executable,
            str(WATCH_WRAP),
            "--stop-on",
            "PACKET_TERMINAL",
            "--success",
            "PACKET_TERMINAL",
            "--replay",
            "2",
            "--",
            sys.executable,
            "-c",
            src,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert p.returncode == 7, p.stdout + p.stderr
    assert "[STOP-TRIGGER] PACKET_TERMINAL DONE" in p.stdout
    assert "stop-triggered" not in p.stdout
    assert "[EXIT rc=7]" in p.stdout


def test_N1_foreign_cwd():
    """Fails with ImportError from foreign cwd."""
    log = Path(f"/tmp/ww_fc_{os.getpid()}.log")
    log.write_text("")
    try:
        p = subprocess.Popen(
            [
                sys.executable,
                str(WATCH_WRAP),
                "--log",
                str(log),
                "--stop-on",
                "PACKET_TERMINAL",
                "--success",
                "PACKET_TERMINAL",
                "--replay",
                "2",
            ],
            cwd="/tmp",
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        time.sleep(0.3)
        log.write_text('{"ok":1}\nPACKET_TERMINAL X\n')
        out, err = p.communicate(timeout=10)
        assert p.returncode == 0 and "ImportError" not in out + err
        assert "[STOP-TRIGGER] PACKET_TERMINAL X" in out
    finally:
        if log.exists():
            log.unlink()


def test_burst_three_distinct_proofs():
    """Fails unless rc=0 AND STOP event AND separate category emission of terminal."""
    log = Path(f"/tmp/ww_b_{os.getpid()}.log")
    log.write_text("")
    p = subprocess.Popen(
        [
            sys.executable,
            str(WATCH_WRAP),
            "--log",
            str(log),
            "--stop-on",
            "PACKET_TERMINAL",
            "--success",
            "PACKET_TERMINAL",
            "--replay",
            "5",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    time.sleep(0.3)
    with log.open("a") as f:
        f.write('{"status":"ok"}\nPACKET_TERMINAL BRANCH_X\n')
    out, err = p.communicate(timeout=10)
    assert p.returncode == 0, out + err
    # (1) STOP event
    assert "[STOP-TRIGGER] PACKET_TERMINAL BRANCH_X" in out
    # (2) separate category emission (not only as substring of STOP line)
    assert any(
        ln.startswith("[T+") and "[OK] PACKET_TERMINAL BRANCH_X" in ln
        for ln in out.splitlines()
    ), out
    # (3) raw indented replay body (HEAD parity)
    assert any(ln == "  PACKET_TERMINAL BRANCH_X" for ln in out.splitlines()), out
    if log.exists():
        log.unlink()


def test_line_cap():
    assert WATCH_WRAP.read_text().count("\n") < 500
    assert Path(__file__).read_text().count("\n") < 500
