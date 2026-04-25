#!/usr/bin/env python3
"""Analyze R19b/E0 raw BigCodeBench smoke results.

Consumes the raw-results schema from ``scripts/r19b_e0_failure_surface.py``
and applies the automated part of the E0 gate:

1. Partition execution outcomes into solves/format/env/correctness buckets.
2. Keep only capability-headroom rows (`fails_correctness` and `partial`).
3. Drop high-confidence decode-path-addressable rows.

The fourth filter (iteratively-refineable structured output) is deliberately
manual and should be reviewed jointly with Claude.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_INPUT = Path("/tmp/e0_raw_results.json")
DEFAULT_OUTPUT = Path("/tmp/e0_classified_results.json")

THIRD_PARTY_LIBS = {
    "bs4",
    "cv2",
    "django",
    "flask",
    "lxml",
    "matplotlib",
    "mechanize",
    "networkx",
    "nltk",
    "numpy",
    "PIL",
    "pandas",
    "plotly",
    "psutil",
    "requests",
    "scipy",
    "seaborn",
    "sklearn",
    "statsmodels",
    "sympy",
    "torch",
}

EFFECT_KEYWORDS = re.compile(
    r"\b("
    r"api|archive|browser|csv|database|directory|download|email|excel|file|ftp|"
    r"http|image|json\s+file|log\s+file|network|pdf|plot|process|request|"
    r"scrape|shell|socket|sql|subprocess|tar|url|web|zip"
    r")\b",
    re.I,
)

SCALAR_PATTERNS = [
    (re.compile(r"\b(factorial|fibonacci|prime|gcd|lcm|collatz)\b", re.I), "number_theory"),
    (re.compile(r"\b(convert|conversion).*\b(binary|hex|decimal|octal)\b", re.I), "base_conversion"),
    (re.compile(r"\b(days?|weeks?|months?|years?)\s+between\b", re.I), "date_difference"),
    (re.compile(r"\b(calculate|compute|return|find|determine)\b.*\b(sum|product|average|mean|median|min|max|difference|distance|area|volume)\b", re.I), "scalar_math"),
    (re.compile(r"\b(reverse|string|palindrome|vowels?|anagram)\b", re.I), "simple_string"),
    (re.compile(r"\b(sort|deduplicate|remove duplicates|flatten)\b.*\b(list|array)\b", re.I), "simple_list"),
]


class SchemaError(ValueError):
    pass


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return [value]


def _lib_roots(row: dict[str, Any]) -> set[str]:
    roots: set[str] = set()
    for lib in _as_list(row.get("libraries")):
        if not isinstance(lib, str):
            continue
        root = lib.strip().split(".", 1)[0]
        if root:
            roots.add(root)
    return roots


def _pass_rate(run: dict[str, Any]) -> float | None:
    try:
        total = int(run.get("tests_total"))
        passed = int(run.get("tests_passed"))
    except (TypeError, ValueError):
        return None
    if total <= 0:
        return None
    return max(0.0, min(1.0, passed / total))


def _execution_partition(row: dict[str, Any]) -> tuple[str, str]:
    run = row.get("sandbox_run") or {}
    outcome = run.get("outcome")
    extraction = row.get("extraction_method")
    code = row.get("extracted_code") or ""

    if row.get("runner_exception"):
        return "environment_unsupported", f"filter1: runner exception ({row['runner_exception']})"
    if outcome == "env_unsupported":
        reason = run.get("unsupported_reason") or "unknown"
        return "environment_unsupported", f"filter1: environment unsupported ({reason})"
    if extraction == "none" or not code.strip() or outcome == "format_fail":
        return "format_fails", "filter1: no runnable extracted code"
    if outcome == "passed":
        return "solves_cleanly", "filter1: runner outcome passed"
    if outcome == "timeout":
        return "fails_correctness", "filter1: timeout counted as correctness failure"
    if outcome != "failed":
        return "environment_unsupported", f"filter1: unknown runner outcome {outcome!r}"

    rate = _pass_rate(run)
    if rate is None:
        return "fails_correctness", "filter1: failed with no partial-credit denominator"
    if rate >= 0.80:
        return "solves_cleanly", f"filter1: pass_rate={rate:.3f} >= 0.80"
    if rate >= 0.20:
        return "partial", f"filter1: 0.20 <= pass_rate={rate:.3f} < 0.80"
    return "fails_correctness", f"filter1: pass_rate={rate:.3f} < 0.20"


def _decode_path_reason(row: dict[str, Any]) -> str | None:
    """High-precision filter-3 heuristic.

    Drop only obvious safe-eval/existing-facade shapes. Ambiguous code tasks,
    especially third-party library coordination, survive to manual filter 4.
    """
    libs = _lib_roots(row)
    if libs & THIRD_PARTY_LIBS:
        return None

    prompt = str(row.get("prompt") or "")
    code_prompt = str(row.get("code_prompt") or "")
    text = f"{prompt}\n{code_prompt}"

    if EFFECT_KEYWORDS.search(text):
        return None

    # Imports like itertools/collections/statistics usually indicate a real
    # program body rather than scalar safe-eval. Let those survive unless a
    # very specific scalar pattern also matches and imports stay tiny.
    allowed_scalar_imports = {"calendar", "datetime", "math", "re", "time"}
    imported_stdlib = libs - allowed_scalar_imports
    if imported_stdlib:
        return None

    for pattern, family in SCALAR_PATTERNS:
        if pattern.search(text):
            return f"high-confidence {family} decode-path/facade shape"
    return None


def _validate_input(data: Any) -> None:
    if not isinstance(data, dict):
        raise SchemaError("input must be a JSON object")
    if not isinstance(data.get("metadata", {}), dict):
        raise SchemaError("metadata must be an object")
    if not isinstance(data.get("results"), list):
        raise SchemaError("results must be a list")


def classify_row(row: dict[str, Any], index: int) -> dict[str, Any]:
    task_id = row.get("task_id") or row.get("problem_id") or row.get("id") or f"row_{index}"
    run = row.get("sandbox_run") or {}
    partition, note = _execution_partition(row)
    notes = [note]
    pass_rate = _pass_rate(run)
    decode_reason = None

    if partition in {"fails_correctness", "partial"}:
        decode_reason = _decode_path_reason(row)
        if decode_reason:
            partition = "decode_path_addressable"
            notes.append(f"filter3: dropped ({decode_reason})")
        else:
            notes.append("filter3: survives conservative decode-path heuristic")
    else:
        notes.append("filter3: skipped because row is not capability-headroom")

    survives = partition in {"fails_correctness", "partial"}
    return {
        "task_id": task_id,
        "partition": partition,
        "survives_filters_1_3": survives,
        "pass_rate": pass_rate,
        "tests_passed": run.get("tests_passed"),
        "tests_total": run.get("tests_total"),
        "runner_outcome": run.get("outcome"),
        "unsupported_reason": run.get("unsupported_reason"),
        "error_type": run.get("error_type"),
        "libraries": sorted(_lib_roots(row)),
        "libraries_source": row.get("libraries_source"),
        "decode_path_reason": decode_reason,
        "notes": notes,
    }


def analyze(data: dict[str, Any]) -> dict[str, Any]:
    _validate_input(data)
    classified = [classify_row(row, i) for i, row in enumerate(data["results"])]
    counts = Counter(item["partition"] for item in classified)
    runner_counts = Counter(item.get("runner_outcome") for item in classified)
    survivors = [item for item in classified if item["survives_filters_1_3"]]

    metadata = dict(data.get("metadata", {}))
    metadata["analyzer"] = "scripts/r19b_e0_analyze.py"
    metadata["filters_applied"] = [
        "drop solves_cleanly, format_fails, environment_unsupported",
        "keep fails_correctness and partial",
        "drop conservative decode_path_addressable shapes",
        "manual iteratively-refineable filter is not applied here",
    ]

    return {
        "metadata": metadata,
        "summary": {
            "n_results": len(classified),
            "partition_counts": dict(sorted(counts.items())),
            "runner_outcome_counts": {
                str(k): v for k, v in sorted(runner_counts.items(), key=lambda kv: str(kv[0]))
            },
            "survivors_filters_1_3": len(survivors),
            "manual_filter_4_required": len(survivors),
        },
        "classified_results": classified,
        "survivors_filters_1_3": survivors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    data = json.loads(input_path.read_text(encoding="utf-8"))
    result = analyze(data)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
