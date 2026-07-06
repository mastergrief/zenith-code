"""CPU-static tests for Arc #2b Slice-5 Step-2 postrun classifier wiring."""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path

from calm.hrm_text_158.native_full_stack.arc2b_slice5_in_vivo_branch import (
    STEP2_ONLY_TERMINALS,
    build_branch_input_from_step2_gpu_run,
    classify_arc2b_slice5_in_vivo_branch,
)
from scripts.hrm_text_158_arc2b_slice5_step2_in_vivo_postrun_classifier import (
    build_receipt,
)

B1_MANIFEST = Path(
    "/home/gabe/claw-code-creditdir/transient_fp_credit/"
    "d_recompute_window_feasibility_seed43_43_2189e72017/prelaunch/"
    "calibrated_selector_manifest.json"
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _seed_manifest(run_root: Path) -> None:
    manifest_dst = run_root / "prelaunch" / "calibrated_selector_manifest.json"
    manifest_dst.parent.mkdir(parents=True, exist_ok=True)
    if B1_MANIFEST.is_file():
        shutil.copy2(B1_MANIFEST, manifest_dst)
    else:
        manifest_dst.write_text('{"selector_internal_manifest_sha256":"stub"}\n', encoding="utf-8")


def test_postrun_decay_mismatch_never_emits_mechanism_terminal() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        scratch = run_root / "d_recompute_window_diagnostic"
        _write_jsonl(
            scratch / "recompute_window_log.jsonl",
            [
                {
                    "replay_constants": {
                        "decay_numerator": 1,
                        "decay_denominator": 1,
                    }
                }
            ],
        )
        _write_jsonl(
            scratch / "live_carrier_snapshot.jsonl",
            [
                {
                    "step": 1,
                    "events_bytes": 10,
                    "backlog_bytes": 10,
                    "hot_exact_bytes": 10,
                    "metadata_bytes": 10,
                    "live_acc_carrier_bytes_total": 40,
                    "live_carrier_bytes_exact": True,
                }
            ],
        )
        (run_root / "prelaunch").mkdir(parents=True, exist_ok=True)
        _seed_manifest(run_root)
        hygiene = {
            "pass": True,
            "bounded_steps_start_count": 1,
        }
        (run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json").write_text(
            json.dumps(hygiene),
            encoding="utf-8",
        )
        (scratch / "receipt.json").write_text(
            json.dumps({"steps_completed": 1, "numel_by_key": {"a": 1_000_000}}),
            encoding="utf-8",
        )

        inputs = build_branch_input_from_step2_gpu_run(
            run_root=run_root,
            hygiene_receipt=hygiene,
            eligible_weight_numel=1_000_000,
        )
        result = classify_arc2b_slice5_in_vivo_branch(inputs)
        assert result["terminal_branch"] not in STEP2_ONLY_TERMINALS
        assert result["terminal_branch"] == "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH"


def test_postrun_build_receipt_emits_slice5_schema_fields() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        run_root = Path(tmp)
        scratch = run_root / "d_recompute_window_diagnostic"
        _write_jsonl(
            scratch / "recompute_window_log.jsonl",
            [
                {
                    "replay_constants": {
                        "decay_numerator": 1,
                        "decay_denominator": 2,
                    }
                }
            ],
        )
        _write_jsonl(
            scratch / "live_carrier_snapshot.jsonl",
            [
                {
                    "step": 1,
                    "events_bytes": 60_000,
                    "backlog_bytes": 0,
                    "hot_exact_bytes": 0,
                    "metadata_bytes": 0,
                    "live_acc_carrier_bytes_total": 60_000,
                    "live_carrier_bytes_exact": True,
                }
            ],
        )
        (run_root / "prelaunch").mkdir(parents=True, exist_ok=True)
        _seed_manifest(run_root)
        hygiene = {"pass": True, "bounded_steps_start_count": 1}
        (run_root / "prelaunch" / "post_confirmation_hygiene_receipt.json").write_text(
            json.dumps(hygiene),
            encoding="utf-8",
        )
        (scratch / "receipt.json").write_text(
            json.dumps({"steps_completed": 1, "numel_by_key": {"a": 1_000_000}}),
            encoding="utf-8",
        )

        packet = {"packet_revision": "test", "effective_acc_budget_bpw": 0.4}
        receipt = build_receipt(
            run_root=run_root,
            packet=packet,
            repo_root=Path("/mnt/c/Users/gabes/projects/claw-code-hrm-text-158"),
        )
        assert receipt["schema"] == "hrm_text_158_arc2b_slice5_in_vivo_law_validation_receipt/v1"
        assert receipt["resume_generation"] == 0
        assert receipt["terminal_branch"] in {
            "D_NEEDS_UPDATE_LAW_REDESIGN",
            "SLICE5_IN_VIVO_LAW_BOUND",
            "SLICE5_INCONCLUSIVE_SOURCE_MISMATCH",
            "SLICE5_INCONCLUSIVE_LOG_COVERAGE",
            "SLICE5_NO_VERDICT_SCHEMA",
        }
