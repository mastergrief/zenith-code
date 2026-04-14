"""Unified chat REPL — one conversation across the whole CHRLM stack.

Every turn tries the substrate-native expert stack first. Falls back to
the external LLM (Gemma via llama-server on localhost:8080) with full
conversation history for anything the substrate can't route. History is
unified: CHRLM answers are visible to Gemma on subsequent turns, and
Gemma's context flows through to CHRLM (as natural language input).

Routing order per turn:
  1. CHRLM Dispatcher  (RouterHRM → HRM specialists → LLM-Computer)
     - Math, NL word problems, GSM-style, structure extraction
  2. Discoverer         (IO-pair tasks → library lookup or synth)
  3. Compiled programs  (!isa, !library, !run <program>)
  4. Gemma fallback     (localhost:8080 chat/completions endpoint)

If Gemma is offline, the REPL still works for the substrate-native
subset and reports "LLM offline" for fallback queries.

Usage:
    PYTHONPATH=. python3 scripts/unified_chat.py

Commands:
    /exit, /quit        — leave
    /history            — show full turn history
    /status             — show backend health
    /route <text>       — show which backend would answer (no generation)
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
GEMMA_URL = "http://localhost:8080/v1/chat/completions"
GEMMA_HEALTH_URL = "http://localhost:8080/health"

SYSTEM_PROMPT = (
    "You are Zenith, a unified AI assistant. The user is talking to a "
    "hybrid system: compiled computational experts + HRM specialists + "
    "an LLM (you) sharing one conversation. When the user asks something "
    "computational, the substrate answers first; you only respond for "
    "open-ended / language-level questions. Keep responses concise."
)


@dataclass
class Turn:
    role: str           # "user" / "assistant"
    content: str
    backend: str = ""   # "chrlm" / "gemma" / "error"
    meta: dict = field(default_factory=dict)


def gemma_available(timeout: float = 2.0) -> bool:
    try:
        req = urllib.request.Request(GEMMA_HEALTH_URL)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


def gemma_chat(messages: list[dict], temperature: float = 0.7,
               max_tokens: int = 512, timeout: float = 120.0) -> str:
    body = json.dumps({
        "model": "zenith",
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode("utf-8")
    req = urllib.request.Request(
        GEMMA_URL, data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        resp = json.loads(r.read().decode("utf-8"))
    return resp["choices"][0]["message"]["content"]


def try_chrlm(dispatcher, text: str) -> Optional[dict]:
    """Return {label, expression, answer} if CHRLM handled; else None."""
    try:
        result = dispatcher.run(text)
    except Exception as e:
        return None
    if result.answer is None:
        return None
    return {
        "label": result.label,
        "expression": result.expression,
        "answer": result.answer,
    }


def render_chrlm_reply(result: dict) -> str:
    """Format CHRLM's structured answer as a natural assistant response."""
    ans = result["answer"]
    expr = result["expression"]
    label = result["label"]
    if expr and expr.strip():
        return f"{ans}  (via {label} specialist: {expr.strip('= ')})"
    return f"{ans}  (via {label} specialist)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    print("=== Unified CHRLM Chat ===")

    # Lazy-import to avoid loading torch/HRM unless REPL actually starts.
    try:
        from calm.hrm.dispatcher import DEFAULT_ROUTER_CKPT, Dispatcher
    except ImportError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    if not Path(DEFAULT_ROUTER_CKPT).exists():
        print(f"ERROR: router checkpoint missing at {DEFAULT_ROUTER_CKPT}",
              file=sys.stderr)
        return 1
    dispatcher = Dispatcher()

    gemma_up = gemma_available()
    print(f"  CHRLM: ready (router + 5 specialists lazy-loaded)")
    print(f"  Gemma (llama-server): {'up' if gemma_up else 'OFFLINE (start zenith for LLM fallback)'}")
    print("\nType questions. Commands: /exit /history /status /route <text>\n")

    history: list[Turn] = []

    while True:
        try:
            user_input = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye")
            return 0
        if not user_input:
            continue
        if user_input in ("/exit", "/quit"):
            return 0
        if user_input == "/history":
            for i, t in enumerate(history):
                tag = f"[{t.backend}]" if t.role == "assistant" else ""
                print(f"  {i+1}. {t.role} {tag}: {t.content[:200]}")
            continue
        if user_input == "/status":
            gemma_up = gemma_available()
            print(f"  CHRLM: ready")
            print(f"  Gemma: {'up' if gemma_up else 'OFFLINE'}")
            print(f"  Turns in history: {len(history)}")
            continue
        if user_input.startswith("/route "):
            text = user_input[len("/route "):].strip()
            r = try_chrlm(dispatcher, text)
            if r is not None:
                print(f"  → CHRLM [{r['label']}] would answer: {r['answer']}")
            else:
                print(f"  → Gemma (CHRLM can't route)")
            continue

        history.append(Turn(role="user", content=user_input))

        # 1. Try CHRLM
        chrlm_result = try_chrlm(dispatcher, user_input)
        if chrlm_result is not None:
            reply = render_chrlm_reply(chrlm_result)
            print(f"\n{reply}\n")
            history.append(Turn(role="assistant", content=reply,
                                backend="chrlm", meta=chrlm_result))
            continue

        # 2. Gemma fallback
        if not gemma_available():
            reply = ("I can't route this to the substrate, and the LLM backend "
                     "(Gemma) is offline. Start it with `zenith` to enable "
                     "fallback for open-ended questions.")
            print(f"\n{reply}\n")
            history.append(Turn(role="assistant", content=reply, backend="error"))
            continue

        # Build unified message history for Gemma. Include prior substrate
        # answers so Gemma sees the full conversation context.
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}]
        for t in history:
            if t.role in ("user", "assistant"):
                msgs.append({"role": t.role, "content": t.content})

        try:
            reply = gemma_chat(msgs, temperature=args.temperature,
                                max_tokens=args.max_tokens)
        except Exception as e:
            reply = f"(Gemma error: {e})"
            print(f"\n{reply}\n")
            history.append(Turn(role="assistant", content=reply, backend="error"))
            continue

        print(f"\n{reply}\n")
        history.append(Turn(role="assistant", content=reply, backend="gemma"))


if __name__ == "__main__":
    sys.exit(main() or 0)
