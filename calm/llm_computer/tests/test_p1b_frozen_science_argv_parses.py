"""P1b r4: frozen science argv must parse against the real trainer CLI."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from calm.hrm_text_158.native_full_stack.p1b_supervisor_lib import (
    FROZEN_INNER_SCIENCE_ARGV,
    parse_shell_argv,
)

REPO = Path(__file__).resolve().parents[3]
TRAIN_PY = REPO / "scripts" / "train_hrm_text_158.py"


def _trainer_arg_tokens() -> list[str]:
    tokens = parse_shell_argv(FROZEN_INNER_SCIENCE_ARGV)
    # timeout --kill-after=30 900 python3 scripts/train_hrm_text_158.py <args...>
    try:
        idx = tokens.index("scripts/train_hrm_text_158.py")
    except ValueError as exc:
        raise AssertionError(f"frozen argv missing trainer script: {tokens}") from exc
    return tokens[idx + 1 :]


def _load_train_module():
    spec = importlib.util.spec_from_file_location("train_hrm_text_158_under_test", TRAIN_PY)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_frozen_inner_science_argv_parses_on_real_trainer_parser():
    """Real ArgumentParser accepts frozen tokens; args.device == cuda."""
    mod = _load_train_module()
    parser = mod.build_arg_parser()
    args = parser.parse_args(_trainer_arg_tokens())
    assert args.device == "cuda"
    assert args.use_ternary_bulk is True
    assert args.sub2_authority_live_conversion_proof is True


def test_frozen_inner_science_argv_subprocess_parse_probe():
    """End-to-end: real __main__ parse path accepts frozen argv (no train())."""
    env = os.environ.copy()
    env["R1L_ARGV_PARSE_PROBE"] = "1"
    env["PYTHONPATH"] = str(REPO)
    proc = subprocess.run(
        [sys.executable, str(TRAIN_PY), *_trainer_arg_tokens()],
        cwd=str(REPO),
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, (proc.returncode, proc.stdout, proc.stderr)
    # Last non-empty stdout line is the probe JSON
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, proc.stdout
    payload = json.loads(lines[-1])
    assert payload.get("device") == "cuda"
