#!/usr/bin/env python3
"""Integration test for agents/model_swap.py.

Exercises a real swap cycle against the locally running llama-server:

  1. Adopt the existing server (read current model path via /props)
  2. Create a hard link to the same GGUF inode at a different path
  3. Swap to the hard link — validates kill + restart + health + path change
  4. Swap back to the original path
  5. Verify everything is where it started

Hard link trick: we only have one GGUF on disk, but a hard link gives us
two independent *canonical* paths pointing at the same inode. Unlike a
symlink, ``Path.resolve()`` won't collapse them — so the swap manager
sees them as different and performs a real kill + restart. llama-server
re-reads the bytes (same weights) under the new path.

We deliberately do NOT use a symlink: ``Path.resolve()`` normalizes
symlinks, which would make the swap a no-op — correct behavior for the
manager, wrong for the test.

This test touches the running server. It will briefly (~30-60s per swap)
make the harness unavailable. Don't run it mid-session.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Make the agents package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.model_swap import (
    LlamaServerManager,
    ModelSwapError,
    _find_listening_pid,
    default_base_model,
    discover_specialist_models,
)


def log(msg: str) -> None:
    print(f"[swap-test] {msg}", flush=True)


def main() -> int:
    mgr = LlamaServerManager()

    # ── 1. Adopt existing server ──
    log("1. Checking for running llama-server on :8080...")
    if not mgr.is_running():
        log("   No server running. Start one with `claw` first, or run:")
        log("   ~/llama.cpp/build/bin/llama-server -m ~/models/Qwen3.5-4B.Q5_K_M.gguf \\")
        log("     --ctx-size 65536 --cache-type-k q4_0 --cache-type-v q4_0 -ngl 999 --port 8080")
        return 2

    original = mgr.current_model()
    if original is None:
        log("   ERROR: server is running but /props didn't return model_path")
        return 1
    log(f"   Adopted: {original}")
    pid = _find_listening_pid(8080)
    log(f"   Listening PID (from /proc/net/tcp): {pid}")
    if pid is None:
        log("   WARN: could not resolve PID via /proc — stop_any() will fail on this test")

    if not original.exists():
        log(f"   ERROR: reported model path does not exist: {original}")
        return 1

    # ── 2. Make a hard link at a different path ──
    alt = original.parent / f"{original.stem}.swap-test{original.suffix}"
    log(f"2. Creating hard link {alt.name} -> {original.name}")
    if alt.exists() or alt.is_symlink():
        alt.unlink()
    os.link(original, alt)
    # Sanity: same inode, different paths, different resolve() results
    assert alt.stat().st_ino == original.stat().st_ino, "hard link failed"
    assert alt.resolve() != original.resolve(), "hard link paths collapsed — test broken"

    try:
        # ── 3. Swap to the alt path (real kill + restart) ──
        log("3. Swapping to alt path (real kill + restart)...")
        events: list = []
        try:
            elapsed = mgr.swap(alt, on_event=lambda kind, path: events.append((kind, path.name)))
        except ModelSwapError as e:
            log(f"   SWAP FAILED: {e}")
            log("   Events so far: " + ", ".join(f"{k}" for k, _ in events))
            return 1

        log(f"   Events: {[e[0] for e in events]}")
        log(f"   Elapsed: {elapsed:.1f}s")

        # Real swap should emit 'start', 'stopped', 'ready' — not 'noop'
        kinds = {e[0] for e in events}
        if "noop" in kinds:
            log("   ERROR: swap was a no-op when a real kill+restart was expected")
            return 1
        if not {"start", "stopped", "ready"}.issubset(kinds):
            log(f"   ERROR: missing expected events, got {kinds}")
            return 1

        if not mgr.is_running():
            log("   ERROR: server not healthy after swap")
            return 1
        if not mgr.owns_process():
            log("   ERROR: manager should own the new process after swap")
            return 1

        new_current = mgr.current_model()
        log(f"   /props model_path: {new_current}")
        if new_current is None or new_current.resolve() != alt.resolve():
            log(f"   ERROR: expected {alt}, got {new_current}")
            return 1
        log("   Swap verified — real kill+restart cycle completed")

        # ── 4. Swap back to original ──
        log("4. Swapping back to original (second real kill + restart)...")
        events2: list = []
        try:
            elapsed2 = mgr.swap(original, on_event=lambda k, p: events2.append((k, p.name)))
        except ModelSwapError as e:
            log(f"   RESTORE FAILED: {e}")
            log(f"   Server may be in an inconsistent state. Alt path: {alt}")
            return 1

        log(f"   Events: {[e[0] for e in events2]}")
        log(f"   Elapsed: {elapsed2:.1f}s")

        restored = mgr.current_model()
        log(f"   /props model_path: {restored}")
        if restored is None or restored.resolve() != original.resolve():
            log(f"   ERROR: expected {original}, got {restored}")
            return 1

        # ── 5. No-op swap to same model ──
        log("5. No-op swap to same model (should be near-zero)...")
        elapsed3 = mgr.swap(original, on_event=lambda k, p: log(f"   event: {k}"))
        log(f"   Elapsed: {elapsed3:.3f}s")
        if elapsed3 > 1.0:
            log(f"   WARN: no-op swap took {elapsed3:.3f}s — expected near-zero")

    finally:
        # ── 6. Clean up hard link ──
        log("6. Cleaning up hard link...")
        if alt.exists() or alt.is_symlink():
            alt.unlink()

    log("")
    log("ALL CHECKS PASSED")
    log(f"Total swap cycles: 2 (forward + back)")
    log(f"Forward swap: {elapsed:.1f}s")
    log(f"Reverse swap: {elapsed2:.1f}s")

    # Bonus: show specialist discovery results
    found = discover_specialist_models()
    if found:
        log(f"Discovered specialists: {list(found.keys())}")
    else:
        log("No specialist GGUFs found on disk yet (expected — not trained).")
        log(f"When trained, place them as ~/models/specialist-<domain>*.gguf")

    return 0


if __name__ == "__main__":
    sys.exit(main())
