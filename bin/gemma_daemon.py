#!/usr/bin/env python3
"""Gemma persistent daemon — keep Gemma loaded across many experiment
rounds, avoiding the ~3 min reload cost per script run.

Launch once:

  mkfifo /tmp/gemma_in  # one-time, if missing
  setsid PYTHONPATH=. python3 -u bin/gemma_daemon.py \\
      < /tmp/gemma_in > /tmp/gemma_log 2>&1 &
  disown

Then `tail -f /tmp/gemma_log` to monitor. First load takes the usual
~3 min; after that the daemon awaits script paths on stdin.

Client usage (via `bin/gemma-run` or manually):

  # Send a script path; daemon execs it with `m`, `tok` pre-bound.
  echo "scripts/test_l24_new_probe.py" > /tmp/gemma_in
  # Watch /tmp/gemma_log for "DONE" marker.

Script contract: scripts can assume `m` (GemmaSubstrate) and `tok`
(GemmaTokenizer) are already loaded in the global namespace. They
must NOT call `GemmaSubstrate.from_gguf` or `preload_gpu` again
(defeats the point). They CAN import anything else, create CardSlots,
hooks, run captures, etc. All globals defined in the script persist
for subsequent scripts — useful for sharing captured activations
across rounds, but scripts should namespace their state via
clearly-named variables to avoid collisions.

Supervisor commands (sent as single lines on stdin):
  QUIT       — exit cleanly
  RESET_GLOBALS  — clear the namespace (keeps m, tok)
  <path>     — path to a Python script to exec
"""

from __future__ import annotations

import os
import sys
import traceback

# Path setup (match script-local invocations' PYTHONPATH=. convention)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _boot():
    """One-time Gemma load. Returns (m, tok)."""
    from calm.llm_computer.gemma_substrate import (
        GemmaSubstrate, enable_triton_tq4,
    )
    from calm.llm_computer.synth.gemma_tokenizer import GemmaTokenizer

    gguf = os.path.expanduser(
        "~/models/gemma-4-E4B-it-tq4-aligned.gguf")
    print("[daemon] loading Gemma substrate...", flush=True)
    enable_triton_tq4(True)
    # Bumped 1024 → 8192 → 32768 after SWA fix + post-forward trim.
    # SWA layers cap at window=512 storage (post-forward trim);
    # global layers (7 of 42) grow with context. Memory at 32K:
    # ~350 MB global KV + ~150 MB SWA + 5 GB weights = ~5.5 GB.
    # Headroom for cards / bigger evals on 8 GB GPU.
    m = GemmaSubstrate.from_gguf(gguf, max_len=32768)
    m.preload_gpu("cuda")
    m.warmup(seq_lens=(1, 20))
    tok = GemmaTokenizer.from_gguf(gguf)
    print(f"[daemon] ready. awaiting script paths on stdin.", flush=True)
    return m, tok


def main():
    m, tok = _boot()

    # Namespace shared across scripts. `m` and `tok` persist.
    # Scripts can add their own state; RESET_GLOBALS wipes everything
    # except m and tok.
    import builtins
    base_ns = {
        "__name__": "__daemon__",
        "__builtins__": builtins,
        "m": m,
        "tok": tok,
    }
    ns = dict(base_ns)

    while True:
        try:
            line = input()
        except EOFError:
            print("[daemon] stdin closed, exiting", flush=True)
            break
        line = line.strip()
        if not line:
            continue

        if line == "QUIT":
            print("[daemon] QUIT received, exiting", flush=True)
            break

        if line == "RESET_GLOBALS":
            ns = dict(base_ns)
            print("[daemon] globals reset (m, tok preserved)", flush=True)
            print("DONE", flush=True)
            continue

        if not os.path.isfile(line):
            print(f"[daemon] ERROR: no such file: {line!r}", flush=True)
            print("DONE", flush=True)
            continue

        print(f"[daemon] running {line}", flush=True)
        try:
            with open(line) as f:
                code = f.read()
            # __file__ gives scripts the right path for relative ops.
            ns["__file__"] = line
            exec(compile(code, line, "exec"), ns)
            print(f"[daemon] completed {line}", flush=True)
        except SystemExit as e:
            # Scripts commonly call sys.exit(0/1) — treat as completion.
            print(f"[daemon] script sys.exit({e.code})", flush=True)
        except KeyboardInterrupt:
            print(f"[daemon] KeyboardInterrupt in script", flush=True)
            break
        except Exception as e:
            print(f"[daemon] EXCEPTION in {line}: {e}", flush=True)
            traceback.print_exc()
        print("DONE", flush=True)

    print("[daemon] exit", flush=True)


if __name__ == "__main__":
    main()
