#!/usr/bin/env python3
"""Needle-in-haystack effective-context test for the locally-running llama-server.

Builds haystacks of varying sizes from the project's training data, inserts
needles, asks the model to retrieve them. Outputs a size×depth grid showing
PASS/FAIL.

Three test modes:
- ``single`` (default): one needle, simple retrieval (the easy version)
- ``multi``: N needles at evenly-spaced depths, must retrieve ALL of them
- ``distractor``: 1 PRIMARY needle + N decoys with similar shape, must
  retrieve the PRIMARY without confusion

The test measures **effective context**: at what context length does the
model stop reliably retrieving facts that are in its window? This is the
real number, not the advertised ``max_position_embeddings``.

Requires: llama-server running on :8080 with sufficient context for the
target sizes (default tests up to 100K tokens, so server needs 128K+ ctx).

Usage (from repo root):
    python3 scripts/needle_test.py                              # single, default grid
    python3 scripts/needle_test.py --mode multi --needles 5     # multi-needle
    python3 scripts/needle_test.py --mode distractor --distractors 4
    python3 scripts/needle_test.py --sizes 4000 16000 32000
    python3 scripts/needle_test.py --depths 10 50 90
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CORPUS_FILE = REPO_ROOT / "agents/distill/data/coding_reasoning_claude.jsonl"

LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"
PROPS_URL = "http://localhost:8080/props"
TOKENIZE_URL = "http://localhost:8080/tokenize"

# Default test grid
DEFAULT_SIZES = [4000, 16000, 32000, 64000, 100000]
DEFAULT_DEPTHS = [10, 30, 50, 70, 90]

# Approximate chars-per-token for English+code mixture
CHARS_PER_TOKEN = 4


def load_corpus_text(jsonl_path: Path) -> str:
    """Concatenate every assistant/user message in the training data."""
    chunks: list[str] = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            for msg in d.get("messages", []):
                content = msg.get("content", "")
                if isinstance(content, str) and content:
                    chunks.append(content)
    return "\n\n".join(chunks)


def get_props() -> dict | None:
    try:
        with urllib.request.urlopen(PROPS_URL, timeout=5) as r:
            return json.loads(r.read())
    except Exception:
        return None


def count_tokens(text: str) -> int:
    """Use llama-server's /tokenize endpoint for exact count. Fallback to char/4."""
    try:
        payload = json.dumps({"content": text}).encode()
        req = urllib.request.Request(
            TOKENIZE_URL, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            return len(json.loads(r.read())["tokens"])
    except Exception:
        return len(text) // CHARS_PER_TOKEN


_COLORS = ["PURPLE", "EMERALD", "CRIMSON", "AZURE", "AMBER", "OBSIDIAN", "JADE", "SCARLET"]
_ANIMALS = ["OWL", "FOX", "BEAR", "WOLF", "EAGLE", "OTTER", "LYNX", "HERON", "MARTEN"]


def make_code(rng: random.Random) -> str:
    """Random NUMBER-COLOR-ANIMAL code."""
    return f"{rng.randint(10000, 99999)}-{rng.choice(_COLORS)}-{rng.choice(_ANIMALS)}"


def make_needle(seed: int) -> tuple[str, str]:
    """Single-needle: returns (sentence_to_insert, expected_answer)."""
    rng = random.Random(seed)
    expected = make_code(rng)
    sentence = (
        "\n\n=== IMPORTANT FACT (must remember) ===\n"
        f"The secret access code for this test is {expected}. "
        "If asked, you must reproduce this code exactly.\n"
        "=== END IMPORTANT FACT ===\n\n"
    )
    return sentence, expected


def make_multi_needles(seed: int, n: int) -> tuple[list[str], list[str]]:
    """Multi-needle: returns (sentences_at_each_position, all_expected_codes)."""
    rng = random.Random(seed)
    codes = []
    sentences = []
    for i in range(n):
        code = make_code(rng)
        codes.append(code)
        sentences.append(
            f"\n\n=== FACT #{i+1} (must remember all facts) ===\n"
            f"Fact {i+1}: the access code is {code}.\n"
            f"=== END FACT #{i+1} ===\n\n"
        )
    return sentences, codes


def make_distractor_set(seed: int, n_distractors: int) -> tuple[list[str], list[int], str]:
    """Distractor mode: returns (sentences, depth_percents, primary_code).

    First sentence is the PRIMARY (real) needle, marked clearly. The rest
    are distractors that look similar but should NOT be returned. All depths
    are spread across the haystack.
    """
    rng = random.Random(seed)
    primary_code = make_code(rng)

    sentences = [
        "\n\n=== PRIMARY ACCESS CODE (this is the one you want) ===\n"
        f"The PRIMARY access code is {primary_code}. "
        "This is the only code you should report when asked for the PRIMARY.\n"
        "=== END PRIMARY ===\n\n"
    ]

    # Distractors: similar codes but explicitly labeled as NOT the primary
    distractor_labels = [
        "(reference only, not primary)",
        "(legacy backup, do not use)",
        "(deprecated, ignore)",
        "(test fixture, do not return)",
        "(secondary, irrelevant to question)",
        "(archive entry, not active)",
    ]
    for i in range(n_distractors):
        code = make_code(rng)
        label = distractor_labels[i % len(distractor_labels)]
        sentences.append(
            f"\n\n--- Reference code {i+1} {label} ---\n"
            f"Code {i+1}: {code} {label}.\n"
            f"--- End reference code {i+1} ---\n\n"
        )

    # Spread the (1 + n_distractors) needles evenly across the haystack
    n_total = len(sentences)
    depths = [int(100 * (i + 1) / (n_total + 1)) for i in range(n_total)]
    return sentences, depths, primary_code


def build_haystack(corpus: str, target_chars: int, needle: str, depth_pct: int) -> str:
    """Take target_chars from corpus, insert one needle at depth_pct% position."""
    if len(corpus) < target_chars:
        repeats = (target_chars // len(corpus)) + 2
        corpus = corpus * repeats
    haystack = corpus[:target_chars]
    insert_pos = int(target_chars * depth_pct / 100)
    return haystack[:insert_pos] + needle + haystack[insert_pos:]


def build_haystack_multi(corpus: str, target_chars: int, needles: list[str], depths: list[int]) -> str:
    """Insert multiple needles at multiple depths. Inserts in order of position
    (earliest first) so each subsequent insert doesn't shift earlier ones."""
    if len(corpus) < target_chars:
        repeats = (target_chars // len(corpus)) + 2
        corpus = corpus * repeats
    haystack = corpus[:target_chars]

    # Sort needles by depth so we insert from end to start (avoiding offset shift)
    indexed = sorted(enumerate(depths), key=lambda x: -x[1])
    for orig_i, depth in indexed:
        insert_pos = int(target_chars * depth / 100)
        haystack = haystack[:insert_pos] + needles[orig_i] + haystack[insert_pos:]
    return haystack


def query_model(prompt: str, max_tokens: int = 200, timeout: int = 600) -> tuple[str, str, float]:
    """Send a chat completion request.

    Returns (answer_content, reasoning_content, elapsed_seconds) — split so callers
    can decide whether to check both fields or only the final answer. For retrieval
    tests we generally want both (the answer might be in either field). For distractor
    tests we want only ``content`` — the model's thinking naturally mentions
    distractors it's considering and rejecting, and that shouldn't count as a leak.
    """
    payload = {
        "model": "local",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        LLAMACPP_URL, data=data,
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

    msg = result["choices"][0]["message"]
    content = msg.get("content", "") or ""
    reasoning = msg.get("reasoning_content", "") or ""
    return content, reasoning, elapsed


def run_grid_single(corpus: str, sizes: list[int], depths: list[int]) -> dict:
    """Single-needle test grid. Returns {(size, depth): result_dict}."""
    results: dict[tuple[int, int], dict] = {}
    seed = 42
    for size in sizes:
        for depth in depths:
            seed += 1
            needle, expected = make_needle(seed)
            target_chars = size * CHARS_PER_TOKEN
            haystack = build_haystack(corpus, target_chars, needle, depth)

            prompt = (
                "You will read a long document and then answer a question about it.\n\n"
                "=== DOCUMENT START ===\n"
                f"{haystack}\n"
                "=== DOCUMENT END ===\n\n"
                "Question: What is the secret access code mentioned in the IMPORTANT FACT block? "
                "Reply with just the code (format: NUMBER-COLOR-ANIMAL), nothing else."
            )

            print(f"  [{size:>6} tok, depth {depth:>2}%] expected={expected}", flush=True)
            try:
                content, reasoning, elapsed = query_model(prompt, max_tokens=200, timeout=2400)
            except Exception as e:
                print(f"      ERROR: {e}")
                results[(size, depth)] = {"pass": False, "elapsed": 0, "response": str(e), "expected": expected}
                continue

            combined = f"{reasoning}\n{content}".strip()
            found = expected in combined
            preview = combined.replace("\n", " ").strip()[:100]
            status = "PASS" if found else "FAIL"
            print(f"      {status} in {elapsed:>5.1f}s | {preview!r}")
            results[(size, depth)] = {
                "pass": found,
                "elapsed": elapsed,
                "response": combined,
                "expected": expected,
            }
    return results


def run_grid_multi(corpus: str, sizes: list[int], n_needles: int) -> dict:
    """Multi-needle test: N needles spread across haystack, must retrieve ALL.

    Returns {size: result_dict}. Depths are determined by n_needles (evenly spaced).
    """
    results: dict[int, dict] = {}
    seed = 1000
    for size in sizes:
        seed += 1
        needles, expected_codes = make_multi_needles(seed, n_needles)
        depths = [int(100 * (i + 1) / (n_needles + 1)) for i in range(n_needles)]
        target_chars = size * CHARS_PER_TOKEN
        haystack = build_haystack_multi(corpus, target_chars, needles, depths)

        prompt = (
            "You will read a long document containing several FACT blocks. "
            "Then list every fact in order.\n\n"
            "=== DOCUMENT START ===\n"
            f"{haystack}\n"
            "=== DOCUMENT END ===\n\n"
            f"Question: There are {n_needles} FACT blocks in the document. "
            "List the access code from each one, in order (FACT #1 first). "
            "Format: one code per line, just the codes, nothing else."
        )

        print(f"  [{size:>6} tok, {n_needles} needles at depths {depths}] expected={expected_codes}", flush=True)
        try:
            content, reasoning, elapsed = query_model(prompt, max_tokens=400, timeout=900)
        except Exception as e:
            print(f"      ERROR: {e}")
            results[size] = {"pass": False, "elapsed": 0, "found": [], "expected": expected_codes, "response": str(e)}
            continue

        response = f"{reasoning}\n{content}".strip()
        found_codes = [c for c in expected_codes if c in response]
        all_found = len(found_codes) == n_needles
        status = "PASS" if all_found else f"FAIL ({len(found_codes)}/{n_needles})"
        preview = response.replace("\n", " ").strip()[:120]
        print(f"      {status} in {elapsed:>5.1f}s | {preview!r}")
        results[size] = {
            "pass": all_found,
            "n_found": len(found_codes),
            "n_expected": n_needles,
            "found": found_codes,
            "expected": expected_codes,
            "elapsed": elapsed,
            "response": response,
        }
    return results


def run_grid_distractor(corpus: str, sizes: list[int], n_distractors: int) -> dict:
    """Distractor test: 1 PRIMARY + N distractor codes, must return ONLY primary.

    Pass criteria: primary appears in response, AND no distractors appear.
    Returns {size: result_dict}.
    """
    results: dict[int, dict] = {}
    seed = 2000
    for size in sizes:
        seed += 1
        sentences, depths, primary_code = make_distractor_set(seed, n_distractors)
        # Extract distractor codes by parsing the sentences (rough but works)
        all_codes = []
        for s in sentences:
            for word in s.split():
                if "-" in word and len(word) >= 12:
                    cleaned = word.rstrip(".,;:")
                    if all(p.isdigit() for p in cleaned.split("-")[:1]):
                        all_codes.append(cleaned)
        distractor_codes = [c for c in all_codes if c != primary_code]

        target_chars = size * CHARS_PER_TOKEN
        haystack = build_haystack_multi(corpus, target_chars, sentences, depths)

        prompt = (
            "You will read a long document containing one PRIMARY access code "
            "and several distractor reference codes. Identify only the PRIMARY.\n\n"
            "=== DOCUMENT START ===\n"
            f"{haystack}\n"
            "=== DOCUMENT END ===\n\n"
            "Question: What is the PRIMARY access code? "
            "Reply with just the primary code, nothing else. "
            "Do not include any reference, legacy, or deprecated codes."
        )

        print(f"  [{size:>6} tok, primary={primary_code}, {n_distractors} distractors]", flush=True)
        try:
            content, reasoning, elapsed = query_model(prompt, max_tokens=200, timeout=900)
        except Exception as e:
            print(f"      ERROR: {e}")
            results[size] = {"pass": False, "elapsed": 0, "primary": primary_code, "distractors": distractor_codes, "response": str(e)}
            continue

        # Primary counts as found if it appears anywhere (content OR reasoning).
        # Gemma 4 E4B sometimes puts the whole response in reasoning_content
        # on short prompts — correctness still counts.
        # Distractors leaked ONLY count if they appear in the final content
        # field. Mentions during thinking are the model correctly considering
        # and rejecting them, which is the desired reasoning behavior.
        combined = f"{content}\n{reasoning}".strip()
        primary_found = primary_code in combined
        distractors_leaked = [c for c in distractor_codes if c in content]
        distractors_in_thinking = [
            c for c in distractor_codes
            if c in reasoning and c not in content
        ]
        clean = primary_found and len(distractors_leaked) == 0
        if clean:
            status = "PASS"
            if distractors_in_thinking:
                status += f" ({len(distractors_in_thinking)} mentioned in thinking only)"
        elif primary_found and distractors_leaked:
            status = f"PARTIAL (found primary + {len(distractors_leaked)} in answer)"
        elif not primary_found:
            status = "FAIL (primary missed)"
        else:
            status = "FAIL"
        preview = content.replace("\n", " ").strip()[:120]
        print(f"      {status} in {elapsed:>5.1f}s | {preview!r}")
        results[size] = {
            "pass": clean,
            "primary_found": primary_found,
            "distractors_leaked": distractors_leaked,
            "distractors_in_thinking": distractors_in_thinking,
            "primary": primary_code,
            "distractors": distractor_codes,
            "elapsed": elapsed,
            "content": content,
            "reasoning": reasoning,
        }
    return results


def render_grid_single(results: dict, sizes: list[int], depths: list[int]) -> str:
    lines = ["\n## Single-needle recall grid (size × depth)\n"]
    lines.append("| Size (tokens) | " + " | ".join(f"{d}%" for d in depths) + " |")
    lines.append("|---:" + "|:-:" * len(depths) + "|")
    for size in sizes:
        cells = [f"**{size:,}**"]
        for depth in depths:
            r = results.get((size, depth), {})
            cells.append("✓" if r.get("pass") else "✗")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_timing_single(results: dict, sizes: list[int], depths: list[int]) -> str:
    lines = ["\n## Single-needle timing grid (seconds)\n"]
    lines.append("| Size (tokens) | " + " | ".join(f"{d}%" for d in depths) + " |")
    lines.append("|---:" + "|---:" * len(depths) + "|")
    for size in sizes:
        cells = [f"**{size:,}**"]
        for depth in depths:
            r = results.get((size, depth), {})
            cells.append(f"{r.get('elapsed', 0):.1f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_grid_multi(results: dict, sizes: list[int]) -> str:
    lines = ["\n## Multi-needle recall (must find ALL needles)\n"]
    lines.append("| Size (tokens) | found / expected | time | result |")
    lines.append("|---:|---:|---:|:-:|")
    for size in sizes:
        r = results.get(size, {})
        found = r.get("n_found", 0)
        expected = r.get("n_expected", 0)
        elapsed = r.get("elapsed", 0)
        marker = "✓" if r.get("pass") else "✗"
        lines.append(f"| **{size:,}** | {found}/{expected} | {elapsed:.1f}s | {marker} |")
    return "\n".join(lines)


def render_grid_distractor(results: dict, sizes: list[int]) -> str:
    lines = ["\n## Distractor recall (must return PRIMARY only, no decoys)\n"]
    lines.append("| Size (tokens) | primary found | distractors leaked | time | result |")
    lines.append("|---:|:-:|---:|---:|:-:|")
    for size in sizes:
        r = results.get(size, {})
        primary = "✓" if r.get("primary_found") else "✗"
        leaked = len(r.get("distractors_leaked", []))
        elapsed = r.get("elapsed", 0)
        if r.get("pass"):
            marker = "✓ clean"
        elif r.get("primary_found"):
            marker = f"⚠ leaked {leaked}"
        else:
            marker = "✗ missed"
        lines.append(f"| **{size:,}** | {primary} | {leaked} | {elapsed:.1f}s | {marker} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["single", "multi", "distractor"], default="single",
                        help="Test mode")
    parser.add_argument("--sizes", nargs="+", type=int, default=DEFAULT_SIZES)
    parser.add_argument("--depths", nargs="+", type=int, default=DEFAULT_DEPTHS,
                        help="(single mode only)")
    parser.add_argument("--needles", type=int, default=5,
                        help="(multi mode) number of needles to insert")
    parser.add_argument("--distractors", type=int, default=4,
                        help="(distractor mode) number of decoys")
    parser.add_argument("--out", type=Path, default=Path("/tmp/needle_test.md"))
    args = parser.parse_args()

    props = get_props()
    if not props:
        print("ERROR: llama-server not running on :8080")
        return 2

    model_path = props.get("model_path", "?")
    n_ctx = props.get("default_generation_settings", {}).get("n_ctx", 0)
    print(f"Model:   {model_path}")
    print(f"Context: {n_ctx} tokens")
    print(f"Mode:    {args.mode}")
    print(f"Sizes:   {args.sizes}")
    if args.mode == "single":
        print(f"Depths:  {args.depths}")
    elif args.mode == "multi":
        print(f"Needles: {args.needles}")
    elif args.mode == "distractor":
        print(f"Distractors: {args.distractors}")

    if isinstance(n_ctx, int):
        max_safe = int(n_ctx * 0.85)
        skipped = [s for s in args.sizes if s > max_safe]
        if skipped:
            print(f"\nWARNING: skipping sizes that exceed 85% of server ctx ({max_safe}): {skipped}")
            args.sizes = [s for s in args.sizes if s <= max_safe]
        if not args.sizes:
            print("ERROR: all sizes exceed safe context limit")
            return 2

    print()
    if not CORPUS_FILE.exists():
        print(f"ERROR: corpus not found: {CORPUS_FILE}")
        return 2
    corpus = load_corpus_text(CORPUS_FILE)
    print(f"Corpus loaded: {len(corpus):,} chars from {CORPUS_FILE.name}")

    print(f"\n=== Running {args.mode} needle test ===\n")

    if args.mode == "single":
        results = run_grid_single(corpus, args.sizes, args.depths)
        grid_md = render_grid_single(results, args.sizes, args.depths)
        timing_md = render_timing_single(results, args.sizes, args.depths)
        report_body = grid_md + "\n" + timing_md
        passed = sum(1 for r in results.values() if r["pass"])
        total = len(results)
    elif args.mode == "multi":
        results = run_grid_multi(corpus, args.sizes, args.needles)
        grid_md = render_grid_multi(results, args.sizes)
        report_body = grid_md
        passed = sum(1 for r in results.values() if r["pass"])
        total = len(results)
    else:  # distractor
        results = run_grid_distractor(corpus, args.sizes, args.distractors)
        grid_md = render_grid_distractor(results, args.sizes)
        report_body = grid_md
        passed = sum(1 for r in results.values() if r["pass"])
        total = len(results)

    print(report_body)

    args.out.write_text(
        f"# Needle-in-Haystack Test ({args.mode})\n\n"
        f"- **Model**: `{model_path}`\n"
        f"- **Server context**: {n_ctx} tokens\n"
        f"- **Mode**: {args.mode}\n"
        f"- **Sizes tested**: {args.sizes}\n"
        + (f"- **Depths**: {args.depths}\n" if args.mode == "single" else "")
        + (f"- **Needles per haystack**: {args.needles}\n" if args.mode == "multi" else "")
        + (f"- **Distractors per haystack**: {args.distractors}\n" if args.mode == "distractor" else "")
        + f"\n{report_body}\n\n"
        f"## Legend\n\n- ✓ = pass (all expected facts found, no leaks)\n- ✗ = fail\n",
        encoding="utf-8",
    )
    print(f"\nReport: {args.out}")
    print(f"\nOverall: {passed}/{total} PASS ({100*passed/total:.0f}%)")
    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
