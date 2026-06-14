from __future__ import annotations

import json
from pathlib import Path

from calm.hrm_text_158.native_full_stack.consensus_probe_result_writer import (
    append_probe_result_jsonl,
)


def test_append_probe_result_jsonl_receipt_from_filesystem(tmp_path: Path) -> None:
    scratch = tmp_path / "S44_ord43" / "on"
    scratch.mkdir(parents=True)
    probe_results = tmp_path / "probe_results.jsonl"

    row_false = append_probe_result_jsonl(
        probe_results,
        probe_num=1,
        label="S44_ord43",
        arm="on",
        exit_code=0,
        wall_s=100,
        heartbeats=3,
        scratch_root=scratch,
    )
    assert row_false["receipt"] is False

    (scratch / "receipt.json").write_text(
        json.dumps({"steps_completed": 10}) + "\n",
        encoding="utf-8",
    )
    (scratch / "last_active_phase.json").write_text(
        json.dumps({"active_phase": "checkpoint_payload", "active_phase_elapsed_seconds": 42.0})
        + "\n",
        encoding="utf-8",
    )

    row_true = append_probe_result_jsonl(
        probe_results,
        probe_num=2,
        label="S44_ord43",
        arm="on",
        exit_code=0,
        wall_s=200,
        heartbeats=5,
        scratch_root=scratch,
    )
    assert row_true["receipt"] is True
    assert row_true["steps_completed"] == 10
    assert row_true["last_active_phase"]["active_phase"] == "checkpoint_payload"

    lines = probe_results.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    for line in lines:
        parsed = json.loads(line)
        assert isinstance(parsed["receipt"], bool)
