#!/usr/bin/env python3
"""Bootstrap __main__ entry: executed code-currency guard before probe harness import."""
from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    cli_argv = list(sys.argv[1:] if argv is None else argv)
    from scripts.hrm_text_158_code_currency_guard import (
        run_phase3b_probe_executed_code_currency_guard,
    )

    guard_exit = run_phase3b_probe_executed_code_currency_guard(argv=cli_argv)
    if guard_exit is None:
        from scripts.hrm_text_158_code_currency_guard import (
            maybe_enforce_phase3b_probe_import_byte_currency,
        )

        guard_exit = maybe_enforce_phase3b_probe_import_byte_currency()
    if guard_exit is not None:
        return int(guard_exit)
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import main as probe_main

    try:
        return int(probe_main(cli_argv))
    except SystemExit as exc:
        if exc.code is None:
            return 0
        if isinstance(exc.code, int):
            return int(exc.code)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
