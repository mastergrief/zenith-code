#!/usr/bin/env python3
"""A/B evaluation of two base GGUF models on a fixed prompt suite.

Loads model A via LlamaServerManager, sends N prompts, collects responses,
swaps to model B, repeats. Writes a side-by-side markdown file for manual
review.

The prompt suite targets the four/five axes the 4B base eval focuses on:
race conditions, OOMKilled debugging, architecture design, React hooks
(weak spot), and security/SSRF (weak spot).

Usage (from repo root):
    python3 scripts/eval_base_models.py \\
        --model-a ~/models/Qwen3.5-4B.Q5_K_M.gguf \\
        --model-b ~/models/gemma-4-E4B-it-Q5_K_M.gguf \\
        --out /tmp/base_eval.md

The llama-server must be installed (bin/zenith would auto-start one but we
manage it directly here). Each prompt has a 5-minute timeout. Total run
time is ~5-15 minutes depending on model speed.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

# Make the agents package importable when running from repo root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.model_swap import LlamaServerManager, ModelSwapError


# ── Prompt suite ───────────────────────────────────────────────────

PROMPTS: list[tuple[str, str, str]] = [
    (
        "race_condition",
        "Concurrent map writes in Go (race condition debugging)",
        """I have a Go program where two goroutines write to the same map concurrently and I'm getting "fatal error: concurrent map writes". Here's the relevant code:

```go
var data = make(map[string]int)

func updateA() {
    for i := 0; i < 100; i++ {
        data["a"] = i
    }
}

func updateB() {
    for i := 0; i < 100; i++ {
        data["b"] = i
    }
}
```

How do I fix this properly? Compare the options and tell me which is best for this case.""",
    ),
    (
        "oomkilled",
        "Kubernetes OOMKilled debugging for a Node.js service",
        """My Kubernetes pod keeps getting OOMKilled. The container runs a Node.js service, memory limit is 512Mi, the pod runs fine for a few hours then dies. What are the likely causes and how do I systematically debug this?""",
    ),
    (
        "architecture",
        "Resilient job queue architecture",
        """Design a resilient background job queue with these requirements:
- ~10K jobs/day
- Jobs take 30s-5min to run
- Must survive worker restarts with no lost jobs
- Retry with exponential backoff (max 3 attempts)
- Detect stuck/zombie jobs
- Postgres is the only backing store available
What's the architecture? Be specific about tables, transactions, and the worker loop.""",
    ),
    (
        "react",
        "React cleanup and unmounted state update",
        """In a React app, I have a component that fetches user data on mount, then fetches their posts based on the user ID, then fetches comment counts for each post. When the user navigates away quickly, I get "can't perform a React state update on an unmounted component" warnings. Show me the right pattern to handle this with modern React (React 18+).""",
    ),
    (
        "security",
        "SSRF in user-URL fetching feature",
        """Our backend has an endpoint that lets users paste a URL and we fetch it to generate a link preview. Is there a security risk here? If so, what is it and how do I defend against it? Include a concrete code example of the fix.""",
    ),
]


# ── Inference ──────────────────────────────────────────────────────

LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"


def generate(
    prompt: str,
    max_tokens: int = 4096,
    timeout: int = 300,
    temperature: float = 0.7,
) -> tuple[str, float, int]:
    """Send a single-turn chat request. Returns (text, elapsed_s, output_tokens)."""
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        LLAMACPP_URL,
        data=data,
        headers={"Content-Type": "application/json"},
    )

    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            result = json.loads(r.read())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")
        raise RuntimeError(f"HTTP {e.code}: {body[:300]}")

    elapsed = time.monotonic() - start
    choice = result["choices"][0]
    msg = choice.get("message", {})

    content = msg.get("content", "") or ""
    # llama.cpp with thinking enabled may expose reasoning_content separately
    reasoning = msg.get("reasoning_content", "") or ""
    if reasoning:
        combined = f"<think>\n{reasoning.strip()}\n</think>\n\n{content.strip()}"
    else:
        combined = content

    usage = result.get("usage", {}) or {}
    out_tokens = usage.get("completion_tokens", 0)

    return combined, elapsed, out_tokens


# ── Eval loop ──────────────────────────────────────────────────────


def run_model(
    mgr: LlamaServerManager,
    label: str,
    model_path: Path,
) -> tuple[float, dict[str, dict]]:
    """Swap to model_path, run every prompt, return (swap_time, {prompt_name: result})."""
    print(f"\n=== Model {label}: {model_path.name} ===")
    try:
        swap_time = mgr.swap(
            model_path,
            on_event=lambda kind, path: print(f"  [swap] {kind}"),
        )
    except ModelSwapError as e:
        print(f"  FAILED TO LOAD: {e}")
        raise

    print(f"  loaded in {swap_time:.1f}s")

    results: dict[str, dict] = {}
    for name, title, prompt in PROMPTS:
        print(f"  [{name}] {title}")
        try:
            text, gen_time, out_tok = generate(prompt)
            tok_per_sec = out_tok / gen_time if gen_time > 0 else 0
            print(f"    {out_tok} tokens in {gen_time:.1f}s ({tok_per_sec:.1f} tok/s)")
            results[name] = {
                "text": text,
                "elapsed": gen_time,
                "output_tokens": out_tok,
                "tok_per_sec": tok_per_sec,
            }
        except Exception as e:
            print(f"    ERROR: {e}")
            results[name] = {
                "text": f"ERROR: {e}",
                "elapsed": 0.0,
                "output_tokens": 0,
                "tok_per_sec": 0.0,
            }

    return swap_time, results


def write_report(
    out_path: Path,
    model_a: Path,
    model_b: Path,
    swap_a: float,
    swap_b: float,
    results_a: dict[str, dict],
    results_b: dict[str, dict],
) -> None:
    lines: list[str] = []
    lines.append(f"# Base Model A/B Eval\n\n")
    lines.append(f"- **Model A**: `{model_a.name}` (swap load: {swap_a:.1f}s)\n")
    lines.append(f"- **Model B**: `{model_b.name}` (swap load: {swap_b:.1f}s)\n\n")

    # Speed summary
    lines.append("## Speed summary\n\n")
    lines.append("| Prompt | A tok/s | B tok/s | A time | B time |\n")
    lines.append("|---|---:|---:|---:|---:|\n")
    for name, _, _ in PROMPTS:
        ra = results_a.get(name, {})
        rb = results_b.get(name, {})
        lines.append(
            f"| {name} | {ra.get('tok_per_sec', 0):.1f} | {rb.get('tok_per_sec', 0):.1f} "
            f"| {ra.get('elapsed', 0):.1f}s | {rb.get('elapsed', 0):.1f}s |\n"
        )

    total_a = sum(r.get("elapsed", 0) for r in results_a.values())
    total_b = sum(r.get("elapsed", 0) for r in results_b.values())
    tok_a = sum(r.get("output_tokens", 0) for r in results_a.values())
    tok_b = sum(r.get("output_tokens", 0) for r in results_b.values())
    avg_a = tok_a / total_a if total_a > 0 else 0
    avg_b = tok_b / total_b if total_b > 0 else 0
    lines.append(
        f"| **total** | **{avg_a:.1f}** | **{avg_b:.1f}** "
        f"| **{total_a:.1f}s** | **{total_b:.1f}s** |\n\n"
    )

    # Side-by-side responses
    for name, title, prompt in PROMPTS:
        lines.append(f"\n---\n\n## `{name}` — {title}\n\n")
        lines.append(f"<details><summary>Prompt</summary>\n\n{prompt}\n\n</details>\n\n")

        ra = results_a.get(name, {})
        rb = results_b.get(name, {})

        lines.append(f"### Model A ({ra.get('elapsed', 0):.1f}s, {ra.get('output_tokens', 0)} tokens)\n\n")
        lines.append(f"{ra.get('text', 'NO RESPONSE')}\n\n")

        lines.append(f"### Model B ({rb.get('elapsed', 0):.1f}s, {rb.get('output_tokens', 0)} tokens)\n\n")
        lines.append(f"{rb.get('text', 'NO RESPONSE')}\n\n")

    out_path.write_text("".join(lines), encoding="utf-8")
    print(f"\nReport written to: {out_path}")
    print(f"  Model A total: {total_a:.1f}s ({tok_a} tokens, {avg_a:.1f} tok/s avg)")
    print(f"  Model B total: {total_b:.1f}s ({tok_b} tokens, {avg_b:.1f} tok/s avg)")


# ── Main ───────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-a", required=True, type=Path, help="First GGUF path")
    parser.add_argument("--model-b", required=True, type=Path, help="Second GGUF path")
    parser.add_argument("--out", type=Path, default=Path("/tmp/base_eval.md"), help="Output markdown path")
    parser.add_argument("--ctx-size", type=int, default=65536, help="Context size")
    args = parser.parse_args()

    model_a = args.model_a.expanduser().resolve()
    model_b = args.model_b.expanduser().resolve()

    for p in (model_a, model_b):
        if not p.exists():
            print(f"ERROR: {p} not found")
            return 2

    mgr = LlamaServerManager(ctx_size=args.ctx_size)

    try:
        swap_a, results_a = run_model(mgr, "A", model_a)
        swap_b, results_b = run_model(mgr, "B", model_b)
    except ModelSwapError:
        return 1

    write_report(args.out, model_a, model_b, swap_a, swap_b, results_a, results_b)

    # Leave the last model loaded so the user can continue using the harness
    print(f"\n  (llama-server left running with {model_b.name})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
