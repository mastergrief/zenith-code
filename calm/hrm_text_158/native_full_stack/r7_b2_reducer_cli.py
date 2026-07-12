"""Thin read-only loader + CLI for the B2 sufficiency reducer pure core.

Dependency direction: this module -> r7_b2_table2_trajectory_sufficiency_reducer.
Never imported by the pure core.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from calm.hrm_text_158.native_full_stack.r7_b2_table2_trajectory_sufficiency_reducer import (
    OVERALL_INSUFFICIENT,
    OVERALL_INVALID,
    OVERALL_SUFFICIENT,
    B2ReduceResult,
    reduce_b2_trajectory,
    to_json_dict,
)


def load_sidecar_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read-only JSONL loader. Does not mutate the file."""
    p = Path(path)
    rows: list[dict[str, Any]] = []
    text = p.read_text(encoding="utf-8")
    for line_no, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"json_parse_error line={line_no}: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError(f"json_not_object line={line_no}")
        rows.append(obj)
    return rows


def load_and_reduce(
    path: Path | str,
    *,
    N: int = 32,
    W: int = 8,
    eps: float = 0.05,
    s1_min_evaluable: int = 16,
) -> B2ReduceResult:
    rows = load_sidecar_jsonl(path)
    return reduce_b2_trajectory(
        rows, N=N, W=W, eps=eps, s1_min_evaluable=s1_min_evaluable
    )


def _cli_exit_code(overall: str) -> int:
    if overall == OVERALL_INVALID:
        return 3
    if overall in (OVERALL_SUFFICIENT, OVERALL_INSUFFICIENT):
        return 0
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r7_b2_reducer_cli",
        description="Post-hoc B2 Table-2 trajectory sufficiency reducer CLI",
    )
    parser.add_argument("sidecar_path", nargs="?", help="Path to census JSONL sidecar")
    parser.add_argument("--pretty", action="store_true")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return 2 if code != 0 else 0

    if not args.sidecar_path:
        print("usage error: sidecar_path required", file=sys.stderr)
        return 2

    try:
        result = load_and_reduce(args.sidecar_path)
    except (OSError, ValueError) as exc:
        print(f"io_or_parse_error: {exc}", file=sys.stderr)
        return 2

    body = to_json_dict(result)
    if args.pretty:
        print(json.dumps(body, indent=2, sort_keys=True))
    else:
        print(json.dumps(body, separators=(",", ":"), sort_keys=True))
    return _cli_exit_code(result.verdicts.overall)


if __name__ == "__main__":
    raise SystemExit(main())
