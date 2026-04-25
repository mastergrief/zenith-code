"""r19b_e0_failure_surface.py — E0 Phase 1b runner.

Runs stock Gemma 4 E4B on N=30 BigCodeBench prompts (raw, native
fields preserved by Phase 0 — see `convert_bigcodebench_raw` in
`scripts/r53_fetch_corpora.py`), extracts generated code via the
existing `dt_install_eval.extract_code`, runs each against bundled
tests via codex's `r19b_e0_restricted_runner.run_test_restricted`,
and emits raw_results.json per the locked Round-3 schema.

Companion files (codex side):
- `scripts/r19b_e0_restricted_runner.py` — sanitized-env subprocess
  runner (Phase 1a). Provides `run_test_restricted(code, test_code,
  deps, timeout=30) -> dict`.
- `scripts/r19b_e0_analyze.py` — partition + filters 1-3 over
  raw_results.json (Phase 1c).

Spec: `RESEARCH/UNIVERSAL_TRANSFORMERS/03_TESTING.md` §7 + §7a + §9.
Provenance: ai-room task `1777128240830-bc475f75`.

Sampling: temperature=0.0 for E0 reproducibility (deterministic
failure surface). If you want stock-Gemma representative sampling
(T=0.7 per harness default), pass `--temperature 0.7`.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional


# Lift extractor + helpers from existing eval infrastructure (Round-3 cite:
# `dt_install_eval.py:335` `extract_code`)
from scripts.dt_install_eval import extract_code

# Codex's Phase 1a deliverable. Import is at module load time so missing
# runner fails fast with a clear traceback rather than at the first call.
from scripts.r19b_e0_restricted_runner import run_test_restricted


CORPUS_PATH = Path("agents/distill/data/bigcodebench_raw.jsonl")
LLAMACPP_URL = "http://localhost:8080/v1/chat/completions"
DEFAULT_N = 30
DEFAULT_SEED = 42
DEFAULT_OUTPUT = Path("/tmp/e0_raw_results.json")
DEFAULT_MODEL_NAME = "gemma-4-E4B-it-tq4-aligned"

SYSTEM_PROMPT = "You are a careful Python coding assistant."

# Per architecture.md MAX_TOKENS budget discipline: Gemma 4 E4B uses
# reasoning_content (thinking mode) on complex prompts and burns ~2-4K
# tokens before emitting `content`. 800 (dt_install_eval default) is
# too low — empirically produced 100% format_fail on BigCodeBench (all
# tokens consumed by reasoning, content stayed empty). 6144 matches
# harness `medium` effort default and gives clean fenced code on tested
# problems (e.g. BigCodeBench/501: 4241ch reasoning + 5477ch content
# fully completed within 2579 tokens).
GEN_MAX_TOKENS = 6144
DEFAULT_TEMPERATURE = 0.0   # deterministic for E0; override with --temperature
GEN_TIMEOUT_SECONDS = 300.0  # bumped from 180s to allow thinking budget
TEST_TIMEOUT_SECONDS = 30


def load_corpus(path: Path) -> list:
    """Read JSONL → list of dicts."""
    if not path.exists():
        raise SystemExit(
            f"Corpus not found: {path}. Run Phase 0 first:\n"
            f"  PYTHONPATH=. python3 scripts/r53_fetch_corpora.py "
            f"--sources bigcodebench_raw"
        )
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sample_corpus(records: list, n: int, seed: int) -> list:
    """Reproducible random sample (without replacement)."""
    if n > len(records):
        raise ValueError(f"n={n} > corpus size {len(records)}")
    rng = random.Random(seed)
    indices = rng.sample(range(len(records)), n)
    return [records[i] for i in indices]


def call_gemma(prompt: str, temperature: float = DEFAULT_TEMPERATURE,
               max_tokens: int = GEN_MAX_TOKENS,
               timeout: float = GEN_TIMEOUT_SECONDS) -> str:
    """Single chat-completion call against llama.cpp.

    Uses stdlib urllib per `architecture.md` §"Agent System". Returns
    the assistant message content (string). Raises on HTTP/network
    failure; caller must classify as `env_unsupported`.
    """
    payload = {
        "model": "any",  # llama.cpp picks the loaded model
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    req = urllib.request.Request(
        LLAMACPP_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def detect_extraction_method(output: str, extracted: Optional[str]) -> str:
    """Classify how `extract_code` found the code, for raw_results audit."""
    if extracted is None:
        return "none"
    end_fence = output.find("```")
    if end_fence > 0 and ("def " in output[:end_fence] or "class " in output[:end_fence]):
        return "code_fence"  # close-only (preferred path in extract_code)
    if "```" in output:
        return "code_fence"  # full fence-pair
    head = extracted.lstrip()
    if head.startswith("def "):
        return "def"
    if head.startswith("class "):
        return "class"
    return "whole_ast"


def _network_failure_record(record: dict, exc: Exception, libraries: list,
                            libraries_source: str) -> dict:
    """Build a raw_results entry when llama.cpp call fails. Classifies as
    env_unsupported so the analyzer doesn't treat it as a Gemma capability
    failure."""
    return {
        "task_id": record["task_id"],
        "prompt": record["instruct_prompt"] or record["complete_prompt"],
        "code_prompt": record["code_prompt"],
        "test_code": record["test"],
        "libraries": libraries,
        "libraries_source": libraries_source,
        "gemma_output_raw": "",
        "extracted_code": None,
        "extraction_method": "none",
        "sandbox_run": {
            "outcome": "env_unsupported",
            "unsupported_reason": f"llamacpp_call_failed:{type(exc).__name__}",
            "exit_code": None,
            "stdout": "",
            "stderr": str(exc),
            "tests_passed": 0,
            "tests_total": 0,
            "error_type": None,
        },
    }


def run_problem(record: dict, temperature: float) -> dict:
    """Execute one problem end-to-end; returns the raw_results record per
    the locked schema."""
    task_id = record["task_id"]
    prompt = record["instruct_prompt"] or record["complete_prompt"]
    test_code = record["test"]
    libs = record.get("libs") or []
    fn_name = record.get("entry_point", "")

    # libraries_source: native upstream metadata if present (Round-3 rule —
    # never parse assistant canonical/test code; runner doesn't see those).
    libraries = list(libs)
    libraries_source = "native" if libs else "none"

    # Generate
    try:
        output_raw = call_gemma(prompt, temperature=temperature)
    except (urllib.error.URLError, urllib.error.HTTPError, OSError, TimeoutError) as e:
        return _network_failure_record(record, e, libraries, libraries_source)

    # Extract
    extracted = extract_code(output_raw, fn_name) if fn_name else None
    extraction_method = detect_extraction_method(output_raw, extracted)

    # Run via codex's restricted runner
    if extracted is None:
        sandbox_run = {
            "outcome": "format_fail",
            "unsupported_reason": None,
            "exit_code": None,
            "stdout": "",
            "stderr": "extract_code returned None",
            "tests_passed": 0,
            "tests_total": 0,
            "error_type": None,
        }
    else:
        sandbox_run = run_test_restricted(
            code=extracted,
            test_code=test_code,
            deps=libraries,
            timeout=TEST_TIMEOUT_SECONDS,
        )

    return {
        "task_id": task_id,
        "prompt": prompt,
        "code_prompt": record["code_prompt"],
        "test_code": test_code,
        "libraries": libraries,
        "libraries_source": libraries_source,
        "gemma_output_raw": output_raw,
        "extracted_code": extracted,
        "extraction_method": extraction_method,
        "sandbox_run": sandbox_run,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n", type=int, default=DEFAULT_N,
                    help=f"smoke size (default {DEFAULT_N})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"sampling seed (default {DEFAULT_SEED})")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT,
                    help=f"raw_results.json path (default {DEFAULT_OUTPUT})")
    ap.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                    help=f"sampling temperature (default {DEFAULT_TEMPERATURE} for reproducibility)")
    ap.add_argument("--model-name", default=DEFAULT_MODEL_NAME,
                    help="logged in metadata; doesn't affect llama.cpp call")
    args = ap.parse_args()

    print(f"Loading corpus from {CORPUS_PATH}...", flush=True)
    records = load_corpus(CORPUS_PATH)
    print(f"  Loaded {len(records)} records", flush=True)

    sampled = sample_corpus(records, args.n, args.seed)
    print(f"  Sampled {args.n} (seed={args.seed})", flush=True)

    results = []
    t_start = time.time()
    for i, record in enumerate(sampled):
        t_problem = time.time()
        print(f"[{i+1}/{args.n}] {record['task_id']}...", flush=True, end="")
        try:
            result = run_problem(record, temperature=args.temperature)
            outcome = result["sandbox_run"]["outcome"]
            elapsed = time.time() - t_problem
            print(f" {outcome} ({elapsed:.1f}s)", flush=True)
        except Exception as e:
            elapsed = time.time() - t_problem
            print(f" RUNNER_EXCEPTION: {type(e).__name__}: {e} ({elapsed:.1f}s)", flush=True)
            result = {
                "task_id": record["task_id"],
                "runner_exception": f"{type(e).__name__}:{e}",
            }
        results.append(result)

    output = {
        "metadata": {
            "corpus": "bigcodebench_raw",
            "corpus_path": str(CORPUS_PATH),
            "n": args.n,
            "seed": args.seed,
            "temperature": args.temperature,
            "model": args.model_name,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "elapsed_seconds": time.time() - t_start,
        },
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2))
    print(
        f"Wrote {len(results)} results to {args.output} "
        f"({time.time() - t_start:.1f}s elapsed)",
        flush=True,
    )


if __name__ == "__main__":
    main()
