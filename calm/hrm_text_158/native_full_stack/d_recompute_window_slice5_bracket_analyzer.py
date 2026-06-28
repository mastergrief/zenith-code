"""Offline Slice-5 bracket analyzer for real density vs sub-2 acc budget.

Read-only CPU analysis over preserved recompute-window logs. Emits a 3-way
decision before any GPU confirmation run:
  REAL_DENSITY_EXCEEDS_SUB2 | ENVELOPE_TOO_TIGHT | INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.d_recompute_window_acc_sizing import (
    effective_acc_budget_bpw,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    D_RECOMPUTE_WINDOW_LOG_FILENAME,
    D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0,
    iter_recompute_window_log_records,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_in_vivo_bound_validator import (
    _collect_logged_backlog_hot,
    _collect_logged_flip_events,
    _records_in_measurement_window,
    _verify_manifest_lanes,
    extract_logged_density_surface,
    measure_packed_payload_total_bytes,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    StratifiedSelectorManifest,
)
from calm.hrm_text_158.native_full_stack.event_coded_acc_checkpoint_codec import (
    EventCodedAccEvent,
    pack_event_coded_acc_checkpoint_v1,
)
from calm.hrm_text_158.native_full_stack.sub2_carrier_family_discriminator import (
    DECLARED_Q_BPW_BASE3,
)

SLICE5_BRACKET_SCHEMA_VERSION = "hrm_text_158_d_recompute_slice5_bracket/v0"

BRACKET_REAL_DENSITY_EXCEEDS_SUB2 = "REAL_DENSITY_EXCEEDS_SUB2"
BRACKET_ENVELOPE_TOO_TIGHT = "ENVELOPE_TOO_TIGHT"
BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT = "INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT"

HONESTY_FAIL_REASONS: frozenset[str] = frozenset(
    {
        "missing_peak_backlog_depth",
        "digest_only_v0_log",
        "empty_measurement_window",
        "manifest_lane_mismatch",
        "raw_global_cap_incomplete",
    }
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes_to_bpw(total_bytes: int, *, eligible_weight_numel: int) -> float:
    if int(eligible_weight_numel) <= 0:
        raise ValueError("eligible_weight_numel must be positive")
    return (float(int(total_bytes)) * 8.0) / float(int(eligible_weight_numel))


def _pack_and_measure(
    *,
    numel: int,
    events: Sequence[EventCodedAccEvent],
    backlog_indices: Sequence[int],
    hot_indices: Sequence[int],
    hot_values: Sequence[int],
    state_key: str,
) -> dict[str, int]:
    payload = pack_event_coded_acc_checkpoint_v1(
        logical_numel=int(numel),
        events=tuple(events),
        backlog_indices=tuple(int(index) for index in backlog_indices),
        hot_exact_indices=tuple(int(index) for index in hot_indices),
        hot_exact_values=tuple(int(value) for value in hot_values),
    )
    return measure_packed_payload_total_bytes(payload, numel=int(numel), state_key=str(state_key))


def _high_index_backlog(*, numel: int, peak_backlog_depth: int) -> tuple[int, ...]:
    peak = int(peak_backlog_depth)
    if peak <= 0:
        return ()
    return tuple(int(numel) - 1 - offset for offset in range(peak))


def _low_index_backlog(*, peak_backlog_depth: int) -> tuple[int, ...]:
    peak = int(peak_backlog_depth)
    if peak <= 0:
        return ()
    return tuple(range(peak))


def _adversarial_hot_surface(
    *,
    numel: int,
    peak_backlog_depth: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    peak = int(peak_backlog_depth)
    if peak <= 0:
        return (), ()
    hot_start = max(0, int(numel) - 2 * peak)
    hot_indices = tuple(hot_start + offset for offset in range(peak))
    hot_values = tuple(127 if offset % 2 == 0 else -127 for offset in range(peak))
    return hot_indices, hot_values


def _adversarial_events(*, numel: int, event_count: int) -> tuple[EventCodedAccEvent, ...]:
    count = int(event_count)
    if count <= 0:
        return ()
    flat_index = int(numel) - 1
    return tuple(
        EventCodedAccEvent(
            flat_index=flat_index,
            direction=1,
            residual_mag=15,
            event_type=1,
        )
        for _ in range(count)
    )


def _honesty_fail_reasons(
    *,
    surface: Any,
    window_records: Sequence[Mapping[str, Any]],
    manifest: StratifiedSelectorManifest | None,
) -> list[str]:
    reasons: list[str] = []
    if int(surface.records_in_window) <= 0 or int(surface.steps_in_window) <= 0:
        reasons.append("empty_measurement_window")
    if surface.peak_backlog_depth is None:
        reasons.append("missing_peak_backlog_depth")
    if surface.schema_version_min == D_RECOMPUTE_WINDOW_SCHEMA_VERSION_V0:
        reasons.append("digest_only_v0_log")
    if not bool(surface.raw_global_cap_complete):
        reasons.append("raw_global_cap_incomplete")
    if manifest is not None and int(_verify_manifest_lanes(window_records, manifest)) > 0:
        reasons.append("manifest_lane_mismatch")
    return reasons


def _max_step_proxy_bytes(
    *,
    records: Sequence[Mapping[str, Any]],
    numel: int,
    sizing_horizon_h: int,
    measurement_start_step: int,
) -> dict[str, int]:
    by_step_peak: dict[int, int] = {}
    for record in records:
        step = int(record["step"])
        if step < int(measurement_start_step) or step > int(sizing_horizon_h):
            continue
        backlog = record.get("backlog_depth")
        if backlog is None:
            continue
        by_step_peak[step] = max(by_step_peak.get(step, 0), int(backlog))

    if not by_step_peak:
        return {
            "total_payload_bytes": 0,
            "events_payload_bytes": 0,
            "backlog_payload_bytes": 0,
            "hot_exact_payload_bytes": 0,
            "metadata_bytes": 0,
        }

    best = {
        "total_payload_bytes": 0,
        "events_payload_bytes": 0,
        "backlog_payload_bytes": 0,
        "hot_exact_payload_bytes": 0,
        "metadata_bytes": 0,
    }
    for step, peak in by_step_peak.items():
        step_records = [dict(record) for record in records if int(record["step"]) == step]
        events = _collect_logged_flip_events(step_records)
        _, hot_indices, hot_values = _collect_logged_backlog_hot(step_records, numel=int(numel))
        measured = _pack_and_measure(
            numel=int(numel),
            events=events,
            backlog_indices=_low_index_backlog(peak_backlog_depth=int(peak)),
            hot_indices=hot_indices,
            hot_values=hot_values,
            state_key=f"slice5.proxy.step_{step}",
        )
        if int(measured["total_payload_bytes"]) > int(best["total_payload_bytes"]):
            best = dict(measured)
    return best


def _derive_budget(
    *,
    numel_by_key: Mapping[str, int],
    measured_q_scale_bpw: float | None,
) -> dict[str, Any]:
    eligible_weight_numel = int(sum(int(value) for value in numel_by_key.values()))
    if eligible_weight_numel <= 0:
        raise ValueError("eligible_weight_numel must be positive")
    q_bpw = (
        float(measured_q_scale_bpw)
        if measured_q_scale_bpw is not None
        else float(DECLARED_Q_BPW_BASE3)
    )
    budget_bpw = float(effective_acc_budget_bpw(measured_q_scale_bpw=float(q_bpw)))
    budget_bytes = float(budget_bpw) * float(eligible_weight_numel) / 8.0
    return {
        "eligible_weight_numel": int(eligible_weight_numel),
        "measured_q_scale_bpw": float(q_bpw),
        "effective_acc_budget_bpw": float(budget_bpw),
        "budget_bytes": float(budget_bytes),
    }


def _classify_bracket(
    *,
    honesty_fail: bool,
    lower_total_bytes: int,
    adversarial_upper_total_bytes: int,
    budget_bytes: float,
) -> str:
    if honesty_fail:
        return BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT
    if int(lower_total_bytes) > float(budget_bytes):
        return BRACKET_REAL_DENSITY_EXCEEDS_SUB2
    if int(adversarial_upper_total_bytes) < float(budget_bytes):
        return BRACKET_ENVELOPE_TOO_TIGHT
    return BRACKET_INSUFFICIENT_LOGS_NEED_LIVE_SNAPSHOT


def analyze_slice5_density_bracket(
    records: Sequence[Mapping[str, Any]],
    *,
    numel_by_key: Mapping[str, int],
    sizing_horizon_h: int = 200,
    measurement_start_step: int = 1,
    measured_q_scale_bpw: float | None = None,
    manifest: StratifiedSelectorManifest | None = None,
    envelope_backlog_lane_count: int | None = None,
    run_root: str | Path | None = None,
    run_id: str | None = None,
    artifact_hashes: Mapping[str, str] | None = None,
    numel_basis_source: str | None = None,
) -> dict[str, Any]:
    budget = _derive_budget(
        numel_by_key=numel_by_key,
        measured_q_scale_bpw=measured_q_scale_bpw,
    )
    eligible_weight_numel = int(budget["eligible_weight_numel"])
    budget_bytes = float(budget["budget_bytes"])

    window_records = _records_in_measurement_window(
        records,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
    )
    surface = extract_logged_density_surface(
        records,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
        manifest=manifest,
    )
    honesty_reasons = _honesty_fail_reasons(
        surface=surface,
        window_records=window_records,
        manifest=manifest,
    )

    peak_backlog_depth = surface.peak_backlog_depth
    peak = int(peak_backlog_depth or 0)

    logged_events = _collect_logged_flip_events(window_records)
    _, sampled_hot_indices, sampled_hot_values = _collect_logged_backlog_hot(
        window_records,
        numel=eligible_weight_numel,
    )

    lower_measured = _pack_and_measure(
        numel=eligible_weight_numel,
        events=logged_events,
        backlog_indices=_low_index_backlog(peak_backlog_depth=peak),
        hot_indices=sampled_hot_indices,
        hot_values=sampled_hot_values,
        state_key="slice5.lower",
    )

    observed_sample_upper_measured = _pack_and_measure(
        numel=eligible_weight_numel,
        events=logged_events,
        backlog_indices=_high_index_backlog(
            numel=eligible_weight_numel,
            peak_backlog_depth=peak,
        ),
        hot_indices=sampled_hot_indices,
        hot_values=sampled_hot_values,
        state_key="slice5.observed_sample_upper",
    )

    event_upper_count = 0
    if bool(surface.raw_global_cap_complete):
        cap_accepted = int(surface.total_global_rate_cap_accepted or 0)
        event_upper_count = max(int(surface.total_flip_events), cap_accepted)
    adversarial_events = _adversarial_events(
        numel=eligible_weight_numel,
        event_count=event_upper_count,
    )
    adversarial_hot_indices, adversarial_hot_values = _adversarial_hot_surface(
        numel=eligible_weight_numel,
        peak_backlog_depth=peak,
    )
    adversarial_upper_measured = _pack_and_measure(
        numel=eligible_weight_numel,
        events=adversarial_events,
        backlog_indices=_high_index_backlog(
            numel=eligible_weight_numel,
            peak_backlog_depth=peak,
        ),
        hot_indices=adversarial_hot_indices,
        hot_values=adversarial_hot_values,
        state_key="slice5.sample_limited_adversarial_upper",
    )

    proxy_measured = _max_step_proxy_bytes(
        records=records,
        numel=eligible_weight_numel,
        sizing_horizon_h=int(sizing_horizon_h),
        measurement_start_step=int(measurement_start_step),
    )

    lower_total_bytes = int(lower_measured["total_payload_bytes"])
    observed_sample_upper_bytes = int(observed_sample_upper_measured["total_payload_bytes"])
    adversarial_upper_total_bytes = int(adversarial_upper_measured["total_payload_bytes"])

    bracket_decision = _classify_bracket(
        honesty_fail=bool(honesty_reasons),
        lower_total_bytes=lower_total_bytes,
        adversarial_upper_total_bytes=adversarial_upper_total_bytes,
        budget_bytes=budget_bytes,
    )

    serialized_checkpoint_acc_bpw = _bytes_to_bpw(
        lower_total_bytes,
        eligible_weight_numel=eligible_weight_numel,
    )
    live_runtime_acc_working_set_bpw_proxy = _bytes_to_bpw(
        int(proxy_measured["total_payload_bytes"]),
        eligible_weight_numel=eligible_weight_numel,
    )
    backlog_density_bounded_envelope_bpw = _bytes_to_bpw(
        adversarial_upper_total_bytes,
        eligible_weight_numel=eligible_weight_numel,
    )

    return {
        "schema_version": SLICE5_BRACKET_SCHEMA_VERSION,
        "run_root": str(run_root) if run_root is not None else None,
        "run_id": run_id,
        "artifact_hashes": dict(artifact_hashes or {}),
        "numel_basis_source": numel_basis_source,
        "numel_by_key": {str(key): int(value) for key, value in numel_by_key.items()},
        "eligible_weight_numel": eligible_weight_numel,
        "measured_q_scale_bpw": float(budget["measured_q_scale_bpw"]),
        "effective_acc_budget_bpw": float(budget["effective_acc_budget_bpw"]),
        "budget_bytes": float(budget_bytes),
        "sizing_horizon_h": int(sizing_horizon_h),
        "measurement_start_step": int(measurement_start_step),
        "logged_density_surface": surface.to_dict(),
        "honesty_fail_reasons": list(honesty_reasons),
        "honesty_flags": {
            "backlog_indices_synthetic_from_count": True,
            "hot_surfaces_sample_limited": True,
            "live_carrier_bytes_exact": False,
            "recommended_law_eligible": False,
            "in_vivo_validated": False,
        },
        "lower_bound": lower_measured,
        "lower_total_bytes": lower_total_bytes,
        "observed_sample_upper": observed_sample_upper_measured,
        "observed_sample_upper_bytes": observed_sample_upper_bytes,
        "sample_limited_adversarial_upper": adversarial_upper_measured,
        "sample_limited_adversarial_upper_bytes": adversarial_upper_total_bytes,
        "event_upper_count": int(event_upper_count),
        "live_runtime_proxy": proxy_measured,
        "live_runtime_acc_working_set_bpw_proxy": float(live_runtime_acc_working_set_bpw_proxy),
        "serialized_checkpoint_acc_bpw": float(serialized_checkpoint_acc_bpw),
        "backlog_density_bounded_envelope_bpw": float(backlog_density_bounded_envelope_bpw),
        "envelope_backlog_lane_count": envelope_backlog_lane_count,
        "bracket_decision": bracket_decision,
        "provenance": {
            "task_id": "1782633464140-b85ec12a",
            "dispatch_id": "1782681356649-b19241f8",
            "gate_1_refreeze_id": "1782681467771-fc607ce8",
            "co_lead_gate_2_pass_id": "1782681537803-7f27c605",
            "allowed_files": [
                "calm/hrm_text_158/native_full_stack/d_recompute_window_slice5_bracket_analyzer.py",
                "scripts/hrm_text_158_d_recompute_slice5_bracket_analyzer.py",
                "calm/llm_computer/tests/test_d_recompute_window_slice5_bracket_analyzer_v0.py",
            ],
        },
    }


def load_classifier_receipt(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_slice5_density_bracket_from_run_root(
    run_root: Path,
    *,
    classifier_receipt_path: Path | None = None,
    log_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    run_root = Path(run_root)
    receipt_path = (
        Path(classifier_receipt_path)
        if classifier_receipt_path is not None
        else run_root / "classifier_receipt.json"
    )
    resolved_log_path = (
        Path(log_path)
        if log_path is not None
        else run_root / "d_recompute_window_diagnostic" / D_RECOMPUTE_WINDOW_LOG_FILENAME
    )
    receipt = load_classifier_receipt(receipt_path)
    records = iter_recompute_window_log_records(resolved_log_path)

    numel_by_key = {
        str(key): int(value)
        for key, value in dict(receipt.get("numel_by_key") or {}).items()
    }
    if not numel_by_key:
        raise ValueError("classifier receipt missing numel_by_key")

    manifest = None
    selector_manifest_path: Path | None = None
    postrun_input_manifest_path: Path | None = None
    manifest_candidates = []
    if manifest_path is not None:
        manifest_candidates.append(Path(manifest_path))
    else:
        manifest_candidates.extend(
            [
                run_root / "prelaunch" / "calibrated_selector_manifest.json",
                run_root / "prelaunch" / "postrun_input_manifest.json",
            ]
        )
    postrun_candidate = run_root / "prelaunch" / "postrun_input_manifest.json"
    if postrun_candidate.is_file():
        postrun_input_manifest_path = postrun_candidate

    for candidate in manifest_candidates:
        if not candidate.is_file():
            continue
        manifest_body = json.loads(candidate.read_text(encoding="utf-8"))
        manifest_payload = (
            manifest_body.get("manifest")
            or manifest_body.get("stratified_selector_manifest")
            or manifest_body
        )
        if not isinstance(manifest_payload, Mapping) or "schema_version" not in manifest_payload:
            continue
        manifest = StratifiedSelectorManifest.from_dict(manifest_payload)
        selector_manifest_path = candidate
        break

    in_vivo = dict(receipt.get("in_vivo_validation") or {})
    pre_screens = dict(in_vivo.get("pre_screens") or {})
    envelope_backlog_lane_count = pre_screens.get("envelope_backlog_lane_count")
    if envelope_backlog_lane_count is not None:
        envelope_backlog_lane_count = int(envelope_backlog_lane_count)

    horizon = receipt.get("horizon_growth") or {}
    sizing_horizon_h = int(
        receipt.get("sizing_horizon_h")
        or horizon.get("classification_horizon_h")
        or horizon.get("sizing_horizon_h")
        or 200
    )

    artifact_hashes_by_path: dict[str, str] = {
        str(receipt_path.resolve()): _sha256_file(receipt_path),
        str(resolved_log_path.resolve()): _sha256_file(resolved_log_path),
    }
    artifact_hashes = {
        "classifier_receipt_sha256": artifact_hashes_by_path[str(receipt_path.resolve())],
        "recompute_window_log_sha256": artifact_hashes_by_path[str(resolved_log_path.resolve())],
    }
    source_artifacts: dict[str, Any] = {}
    if selector_manifest_path is not None and selector_manifest_path.is_file():
        selector_sha = _sha256_file(selector_manifest_path)
        selector_resolved = str(selector_manifest_path.resolve())
        artifact_hashes_by_path[selector_resolved] = selector_sha
        source_artifacts["selector_manifest_path"] = str(selector_manifest_path)
        source_artifacts["selector_manifest_sha256"] = selector_sha
    if postrun_input_manifest_path is not None and postrun_input_manifest_path.is_file():
        postrun_sha = _sha256_file(postrun_input_manifest_path)
        postrun_resolved = str(postrun_input_manifest_path.resolve())
        artifact_hashes_by_path[postrun_resolved] = postrun_sha
        source_artifacts["postrun_input_manifest_path"] = str(postrun_input_manifest_path)
        source_artifacts["postrun_input_manifest_sha256"] = postrun_sha

    bracket = analyze_slice5_density_bracket(
        records,
        numel_by_key=numel_by_key,
        sizing_horizon_h=sizing_horizon_h,
        measurement_start_step=1,
        measured_q_scale_bpw=None,
        manifest=manifest,
        envelope_backlog_lane_count=envelope_backlog_lane_count,
        run_root=run_root,
        run_id=str(receipt.get("run_id") or ""),
        artifact_hashes=artifact_hashes,
        numel_basis_source=str(receipt.get("numel_basis_source") or ""),
    )
    bracket["artifact_hashes_by_path"] = artifact_hashes_by_path
    bracket["source_artifacts"] = source_artifacts
    return bracket
