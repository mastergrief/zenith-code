"""Thin read-only CLI for the R7 block-occupancy byte reducer pure core.

Dependency direction: this module -> r7_block_occupancy_byte_reducer.
Never imported by the pure core.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.r7_block_occupancy_byte_reducer import (
    DEPRIORITIZE_M1,
    MISSING_OBSERVABLES,
    PROCEED_TO_M1_SCREEN,
    ByteReduceResult,
    apply_ev_classifier,
    companion_from_numel_map,
    ev_to_json_dict,
    reduce_block_occupancy_bytes,
    to_json_dict,
)


def load_sidecar_jsonl(path: Path | str) -> list[dict[str, Any]]:
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


def load_companion_json(path: Path | str) -> Mapping[str, int]:
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("companion must be a JSON object of state_key -> logical_numel")
    out: dict[str, int] = {}
    for k, v in obj.items():
        if type(v) is not int or v <= 0:
            raise ValueError(f"bad companion numel for {k!r}")
        out[str(k)] = v
    return out


def _exit_for_reduce(result: ByteReduceResult) -> int:
    if result.overall == MISSING_OBSERVABLES:
        return 3
    if result.overall == "OK":
        return 0
    return 3


def _exit_for_ev(outcome: str) -> int:
    if outcome == MISSING_OBSERVABLES:
        return 3
    if outcome in (PROCEED_TO_M1_SCREEN, DEPRIORITIZE_M1):
        return 0
    return 3


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="r7_block_occupancy_byte_reducer_cli",
        description="Post-hoc R7 block-occupancy E1 byte reducer CLI",
    )
    sub = parser.add_subparsers(dest="cmd")

    p_red = sub.add_parser("reduce", help="Reduce one instrumented census sidecar")
    p_red.add_argument("sidecar_path")
    p_red.add_argument("companion_path", help="JSON map state_key -> logical_numel")
    p_red.add_argument("--N", type=int, default=32)
    p_red.add_argument("--pretty", action="store_true")

    p_ev = sub.add_parser("ev", help="Compare two seed reduces via frozen EV classifier")
    p_ev.add_argument("primary_sidecar")
    p_ev.add_argument("independent_sidecar")
    p_ev.add_argument("companion_path")
    p_ev.add_argument("--N", type=int, default=32)
    p_ev.add_argument("--pretty", action="store_true")

    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit as exc:
        code = int(exc.code) if isinstance(exc.code, int) else 2
        return 2 if code != 0 else 0

    if args.cmd is None:
        print("usage error: subcommand required (reduce|ev)", file=sys.stderr)
        return 2

    try:
        companion = companion_from_numel_map(load_companion_json(args.companion_path))
        if args.cmd == "reduce":
            rows = load_sidecar_jsonl(args.sidecar_path)
            result = reduce_block_occupancy_bytes(rows, companion=companion, N=int(args.N))
            body = to_json_dict(result)
            code = _exit_for_reduce(result)
        else:
            primary = reduce_block_occupancy_bytes(
                load_sidecar_jsonl(args.primary_sidecar),
                companion=companion,
                N=int(args.N),
            )
            independent = reduce_block_occupancy_bytes(
                load_sidecar_jsonl(args.independent_sidecar),
                companion=companion,
                N=int(args.N),
            )
            verdict = apply_ev_classifier(primary, independent)
            body = {
                "ev": ev_to_json_dict(verdict),
                "primary": to_json_dict(primary),
                "independent": to_json_dict(independent),
            }
            code = _exit_for_ev(verdict.outcome)
    except (OSError, ValueError) as exc:
        print(f"io_or_parse_error: {exc}", file=sys.stderr)
        return 2

    if args.pretty:
        print(json.dumps(body, indent=2, sort_keys=True))
    else:
        print(json.dumps(body, separators=(",", ":"), sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
