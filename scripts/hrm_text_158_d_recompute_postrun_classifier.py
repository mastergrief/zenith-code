#!/usr/bin/env python3
"""D recompute-window feasibility postrun classifier (extracted from launch-packet heredoc)."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    iter_recompute_window_log_records,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_feasibility_analyzer import (
    analyze_recompute_window_log,
)

CLASSIFIER_RECEIPT_SCHEMA = "hrm_text_158_d_recompute_window_classifier_receipt/v1"
CLASSIFIER_INPUT_DRIFT_BLOCKED = "INPUT_DRIFT_BLOCKED"
PACKET_REVISION = "v1_rev3d_postrun_classifier_extract"
REPRODUCTION_MODE = "postrun_only_over_native_rev3c_artifacts"
DIAGNOSTIC_SUBDIR = "d_recompute_window_diagnostic"
PARENT_CHECKPOINT = Path(
    "calm/hrm/checkpoints/hrm_text_158_phase3_L0c1_seed0017_replay83_n12k_lr7p5e5_pc1p0_rsL0b1math1r1b2_1_anchorsv1r3_from_L0b_final_step01500.pt"
)
PARENT_SHA256 = "9b4e311a22787e7d4808bde7bc2953568d767a2ee8ac648942a3f5dbb7b4d5ec"

MANIFEST_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("d_recompute_window_diagnostic/receipt.json", "sha256"),
    ("d_recompute_window_diagnostic/recompute_window_log.jsonl", "sha256"),
    ("driver_summary.json", "sha256"),
    ("prelaunch/post_confirmation_hygiene_receipt.json", "sha256"),
    ("prelaunch/parent_checkpoint_rehash.json", "sha256"),
    ("prelaunch/parent_checkpoint_rehash_after_scale_smoke.json", "sha256"),
    ("prelaunch/parent_checkpoint_rehash_after_confirmation.json", "sha256"),
)


def helper_script_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_row_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _load_packet_manifest(packet_path: Path) -> dict[str, Any]:
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    manifest = packet.get("expected_native_input_manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"missing expected_native_input_manifest in packet: {packet_path}")
    return manifest


def _collect_input_artifact_hashes(run_root: Path) -> dict[str, Any]:
    hashes: dict[str, Any] = {}
    for rel_path, _kind in MANIFEST_ARTIFACTS:
        path = run_root / rel_path
        if not path.is_file():
            hashes[rel_path] = {"present": False}
            continue
        entry: dict[str, Any] = {
            "present": True,
            "sha256": _sha256_file(path),
        }
        if rel_path.endswith("recompute_window_log.jsonl"):
            entry["jsonl_row_count"] = _jsonl_row_count(path)
        hashes[rel_path] = entry
    return hashes


def _verify_input_manifest(
    *,
    run_root: Path,
    manifest: dict[str, Any],
) -> list[str]:
    failures: list[str] = []
    observed = _collect_input_artifact_hashes(run_root)
    for rel_path, _kind in MANIFEST_ARTIFACTS:
        expected = manifest.get(rel_path)
        if not isinstance(expected, dict):
            failures.append(f"missing_manifest_entry:{rel_path}")
            continue
        actual = observed.get(rel_path) or {}
        if not actual.get("present"):
            failures.append(f"missing_artifact:{rel_path}")
            continue
        if actual.get("sha256") != expected.get("sha256"):
            failures.append(f"sha256_mismatch:{rel_path}")
        if rel_path.endswith("recompute_window_log.jsonl"):
            expected_rows = expected.get("jsonl_row_count")
            actual_rows = actual.get("jsonl_row_count")
            if expected_rows is not None and actual_rows != expected_rows:
                failures.append(
                    f"jsonl_row_count_mismatch:{rel_path}:{actual_rows}!={expected_rows}"
                )
    return failures


def _state_numel(state: Any) -> int:
    if hasattr(state, "accumulators"):
        return int(state.accumulators.numel())
    return int(state.decoded_accumulators(rebuild_if_stale=True).numel())


def _build_numel_by_key(logged_keys: list[str], records: list[dict[str, Any]]) -> tuple[dict[str, int], str]:
    numel_by_key: dict[str, int] = {}
    numel_basis_source = "parent_checkpoint_tensor_state_numel"
    try:
        import torch

        from calm.hrm_text_158.native_full_stack.accumulator_real_dynamics_verdict import (
            default_vote_update_spec,
        )
        from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
            build_model_from_checkpoint,
            derive_tensor_states_and_check_init_fidelity,
            load_parent_checkpoint,
            select_eligible_bitlinears,
        )

        ckpt, _ = load_parent_checkpoint(PARENT_CHECKPOINT, expected_sha256=PARENT_SHA256)
        model, _, _ = build_model_from_checkpoint(ckpt, torch.device("cpu"))
        eligible = select_eligible_bitlinears(model, eligible_scope="all-bitlinear")
        vote_spec = default_vote_update_spec()
        tensor_states, _ = derive_tensor_states_and_check_init_fidelity(
            eligible,
            threshold=float(vote_spec.threshold_abs),
        )
        for key in logged_keys:
            if key not in tensor_states:
                raise KeyError(key)
            numel_by_key[key] = _state_numel(tensor_states[key])
    except Exception:
        numel_by_key = {}
        numel_basis_source = "lane_span_lower_bound_from_log"
        for key in logged_keys:
            key_records = [r for r in records if str(r["state_key"]) == key]
            max_lane = max(int(idx) for r in key_records for idx in r["lane_indices"])
            numel_by_key[key] = int(max_lane) + 1
    return numel_by_key, numel_basis_source


def _emit_input_drift_blocked_receipt(
    *,
    run_root: Path,
    run_id: str | None,
    input_artifact_hashes: dict[str, Any],
    drift_failures: list[str],
) -> None:
    drift_receipt = {
        "schema": CLASSIFIER_RECEIPT_SCHEMA,
        "run_id": run_id or "unknown",
        "run_root": str(run_root),
        "packet_revision": PACKET_REVISION,
        "helper_script_sha256": helper_script_sha256(),
        "primary_classifier": CLASSIFIER_INPUT_DRIFT_BLOCKED,
        "promoted_fork": None,
        "input_artifact_hashes": input_artifact_hashes,
        "fallback_fired": False,
        "reproduction_mode": REPRODUCTION_MODE,
        "drift_failures": drift_failures,
    }
    out = run_root / "classifier_receipt.json"
    out.write_text(json.dumps(drift_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    raise SystemExit(1)


def _resolve_packet_manifest(
    *,
    run_root: Path,
    packet_path: Path | None,
    skip_input_drift_check: bool,
    run_id: str | None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if skip_input_drift_check:
        return None, {}

    packet_label = str(packet_path) if packet_path is not None else "None"
    if packet_path is None or not packet_path.is_file():
        _emit_input_drift_blocked_receipt(
            run_root=run_root,
            run_id=run_id,
            input_artifact_hashes={},
            drift_failures=[f"missing_packet_manifest:{packet_label}"],
        )

    input_artifact_hashes = _collect_input_artifact_hashes(run_root)
    try:
        manifest = _load_packet_manifest(packet_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        _emit_input_drift_blocked_receipt(
            run_root=run_root,
            run_id=run_id,
            input_artifact_hashes=input_artifact_hashes,
            drift_failures=[f"invalid_packet_manifest:{packet_path}:{exc}"],
        )

    drift_failures = _verify_input_manifest(run_root=run_root, manifest=manifest)
    if drift_failures:
        _emit_input_drift_blocked_receipt(
            run_root=run_root,
            run_id=run_id,
            input_artifact_hashes=input_artifact_hashes,
            drift_failures=drift_failures,
        )
    return manifest, input_artifact_hashes


def emit_timeout_classifier_receipt(
    *,
    run_root: Path,
    run_id: str,
    timeout_seconds: float,
    packet_revision: str = PACKET_REVISION,
) -> dict[str, Any]:
    receipt = {
        "schema": CLASSIFIER_RECEIPT_SCHEMA,
        "run_id": run_id,
        "run_root": str(run_root),
        "packet_revision": packet_revision,
        "helper_script_sha256": helper_script_sha256(),
        "postrun_timeout_classification": "OBSERVER_TOO_EXPENSIVE",
        "postrun_duration_seconds": float(timeout_seconds),
        "selected_state_keys": [],
        "numel_by_key": {},
        "sampled_lane_count_by_key": {},
        "jsonl_row_count": None,
        "numel_basis_source": "postrun_timeout",
        "fallback_fired": True,
        "reproduction_mode": REPRODUCTION_MODE,
        "input_artifact_hashes": {},
        "primary_classifier": "OBSERVER_TOO_EXPENSIVE",
        "promoted_fork": None,
    }
    out = run_root / "classifier_receipt.json"
    out.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def emit_d_recompute_window_classifier_receipt(
    *,
    run_root: Path,
    packet_path: Path | None = None,
    skip_input_drift_check: bool = False,
    run_id: str | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    scratch = run_root / DIAGNOSTIC_SUBDIR
    manifest, input_artifact_hashes = _resolve_packet_manifest(
        run_root=run_root,
        packet_path=packet_path,
        skip_input_drift_check=skip_input_drift_check,
        run_id=run_id,
    )

    receipt_path = scratch / "receipt.json"
    if not receipt_path.is_file():
        raise FileNotFoundError(f"missing diagnostic receipt: {receipt_path}")
    diagnostic_receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    log_path = Path(
        diagnostic_receipt.get("d_recompute_window_log_path") or (scratch / "recompute_window_log.jsonl")
    )
    records = iter_recompute_window_log_records(log_path)
    logged_keys = sorted({str(r["state_key"]) for r in records})
    if len(logged_keys) < 2:
        raise SystemExit(f"expected >=2 instrumented state_keys, got {logged_keys}")

    numel_by_key, numel_basis_source = _build_numel_by_key(logged_keys, records)
    numel_for_bpw = int(sum(numel_by_key.values()))
    analysis = analyze_recompute_window_log(
        log_path,
        numel_for_bpw=numel_for_bpw,
        numel_basis_source=numel_basis_source,
        state_numel_by_key=numel_by_key,
    )
    inventory = analysis["log_inventory"]
    duration = time.monotonic() - started
    if run_id is None and packet_path is not None and packet_path.is_file():
        packet = json.loads(packet_path.read_text(encoding="utf-8"))
        run_id = str(packet.get("run_id") or "unknown")
    elif run_id is None and manifest is not None:
        run_id = str(manifest.get("run_id") or "unknown")
    elif run_id is None:
        run_id = "unknown"

    classifier_receipt = {
        "schema": CLASSIFIER_RECEIPT_SCHEMA,
        "run_id": run_id,
        "run_root": str(run_root),
        "log_path": str(log_path),
        "packet_revision": PACKET_REVISION,
        "helper_script_sha256": helper_script_sha256(),
        "input_artifact_hashes": input_artifact_hashes,
        "fallback_fired": False,
        "reproduction_mode": REPRODUCTION_MODE,
        "selected_state_keys": inventory["selected_state_keys"],
        "numel_by_key": inventory["per_key_numel"],
        "sampled_lane_count_by_key": inventory["sampled_lane_count_by_key"],
        "jsonl_row_count": inventory["jsonl_row_count"],
        "numel_basis_source": inventory["numel_basis_source"],
        "postrun_duration_seconds": duration,
        "postrun_timeout_classification": None,
        "analysis": analysis,
        "primary_classifier": analysis.get("primary_classifier"),
        "promoted_fork": analysis.get("promoted_fork"),
    }
    out = run_root / "classifier_receipt.json"
    out.write_text(json.dumps(classifier_receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return classifier_receipt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="D recompute-window postrun classifier")
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--packet",
        type=Path,
        default=None,
        help="Launch packet JSON with expected_native_input_manifest",
    )
    parser.add_argument("--emit-timeout-receipt", action="store_true")
    parser.add_argument("--timeout-seconds", type=float, default=900.0)
    parser.add_argument(
        "--skip-input-drift-check",
        action="store_true",
        help="FIXTURE-ONLY: bypass manifest hash binding",
    )
    args = parser.parse_args(argv)

    if args.emit_timeout_receipt:
        run_id = "2189e72014"
        if args.packet is not None and args.packet.is_file():
            packet = json.loads(args.packet.read_text(encoding="utf-8"))
            run_id = str(packet.get("run_id") or run_id)
        emit_timeout_classifier_receipt(
            run_root=args.run_root,
            run_id=run_id,
            timeout_seconds=float(args.timeout_seconds),
        )
        print(
            json.dumps(
                {
                    "primary_classifier": "OBSERVER_TOO_EXPENSIVE",
                    "fallback_fired": True,
                },
                indent=2,
            )
        )
        return 0

    try:
        receipt = emit_d_recompute_window_classifier_receipt(
            run_root=args.run_root,
            packet_path=args.packet,
            skip_input_drift_check=bool(args.skip_input_drift_check),
        )
    except SystemExit as exc:
        return int(exc.code) if isinstance(exc.code, int) else 1

    print(
        json.dumps(
            {
                "primary_classifier": receipt["primary_classifier"],
                "promoted_fork": receipt["promoted_fork"],
                "postrun_duration_seconds": receipt["postrun_duration_seconds"],
                "numel_by_key": receipt["numel_by_key"],
                "fallback_fired": receipt["fallback_fired"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
