"""Evidence contract: raw obs schema/IO + science consumer (IMPLEMENT_v3 seam e).

Science path only: explicit authorized raw paths → rehash → recompute cells → reducer once.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.lands_ab_eval_branch_reducer import (
    LandsAbReducerSchemaError,
    reduce_lands_ab_branch_strict,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_fixture_source import (
    DEFAULT_SOURCE_PINS,
    verify_source_pins,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_topology import (
    classify_phase_topology,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_schema import (
    APPLICABILITY_MAP,
    CANONICAL_CELL_KEYS,
    CLAIM_CEILING,
    DIAGNOSTIC_RECEIPT_SCHEMA,
    EVAL_RECEIPT_SCHEMA,
    GATING_ROWS,
    PHASE_ORDER,
    PLAN_V6_PATH,
    PLAN_V6_SHA256,
    RANK_SPEC_DIGEST_EXPECTED,
    RANK_SPEC_SYMBOL,
    RAW_ROW_OBSERVATION_SCHEMA,
    REQUIRED_EVAL_RECEIPT_KEYS,
    REQUIRED_RAW_ROW_KEYS,
    TASK_ID,
    cell_key,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_metric_reducer import (
    recompute_surface_cells_from_primitives,
    validate_metrics_schema,
    validate_required_key_universe,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_twin_apply import (
    key_universe_sha256,
)
from calm.hrm_text_158.native_full_stack.lands_ab_eval_runtime_io import (
    harvest_exactly_one_raw_obs,
    o_excl_write_json,
    o_excl_write_text,
    resolve_run_scratch_dir,
    runtime_scratch_raw_path,
)

# Expected device per gating row (PLAN_v6 site_device_matrix)
EXPECTED_DEVICE_BY_ROW: dict[str, str] = {
    "G_CPU_STATIC_AB": "cpu",
    "G_CUDA_B1_APPLY": "cuda",
    "G_CUDA_B2_APPLY": "cuda",
    "G_CUDA_B3_APPLY": "cuda",
    "G_CUDA_ORACLE_B1": "cuda",
    "G_CUDA_ORACLE_B2": "cuda",
    "G_CUDA_ORACLE_B3": "cuda",
}

# Authorized raw-obs field set (unknown fields rejected)
AUTHORIZED_RAW_FIELDS: frozenset[str] = frozenset(
    {
        "schema",
        "gating_row",
        "device",
        "measured_surfaces",
        "metrics",
        "key_universe",
        "key_universe_sha256",
        "rank_spec_digest",
        "rank_spec_symbol",
        "fixture_contract_raw_fail",
        "science_claim",
        "synthetic_only",
        "claim_ceiling",
        "phase_topology",
        "phase_events",
        "phase_stream_class",
        "phase_stream_anomaly",
        "phase_events_synthesized",
        "site_tag",
        "production_site",
    }
)


def make_raw_row_observation(
    *,
    gating_row: str,
    device: str,
    measured_surfaces: Mapping[str, bool],
    metrics: Mapping[str, Any],
    key_universe: Sequence[str],
    fixture_contract_raw_fail: bool,
    synthetic_only: bool,
    phase_topology: Mapping[str, Any] | None = None,
    phase_events: Sequence[Mapping[str, Any]] | None = None,
    site_tag: str | None = None,
    production_site: str | None = None,
) -> dict[str, Any]:
    if gating_row not in APPLICABILITY_MAP:
        raise ValueError(f"unknown_gating_row:{gating_row}")
    expected = set(APPLICABILITY_MAP[gating_row])
    got = set(measured_surfaces.keys())
    if got != expected:
        raise ValueError(
            f"measured_surfaces_key_mismatch row={gating_row} missing={sorted(expected-got)} "
            f"extra={sorted(got-expected)}"
        )
    for sk, val in measured_surfaces.items():
        if type(val) is not bool:
            raise ValueError(f"non_bool_measured_surface:{gating_row}/{sk}")
    keys = [str(k) for k in key_universe]
    obs: dict[str, Any] = {
        "schema": RAW_ROW_OBSERVATION_SCHEMA,
        "gating_row": gating_row,
        "device": str(device),
        "measured_surfaces": {k: bool(measured_surfaces[k]) for k in sorted(measured_surfaces)},
        "metrics": dict(metrics),
        "key_universe": keys,
        "key_universe_sha256": key_universe_sha256(keys),
        "rank_spec_digest": RANK_SPEC_DIGEST_EXPECTED,
        "rank_spec_symbol": RANK_SPEC_SYMBOL,
        "fixture_contract_raw_fail": bool(fixture_contract_raw_fail),
        "science_claim": False,
        "synthetic_only": bool(synthetic_only),
        "claim_ceiling": dict(CLAIM_CEILING),
    }
    if phase_topology is not None:
        obs["phase_topology"] = dict(phase_topology)
    if phase_events is not None:
        obs["phase_events"] = [dict(e) for e in phase_events]
    if site_tag is not None:
        obs["site_tag"] = str(site_tag)
    if production_site is not None:
        obs["production_site"] = str(production_site)
    validate_raw_row_observation(obs)
    return obs


def validate_raw_row_observation(obs: Mapping[str, Any]) -> None:
    missing = REQUIRED_RAW_ROW_KEYS - set(obs.keys())
    if missing:
        raise ValueError(f"raw_obs_missing_keys:{sorted(missing)}")
    unknown = set(obs.keys()) - AUTHORIZED_RAW_FIELDS
    if unknown:
        raise ValueError(f"raw_obs_unknown_fields:{sorted(unknown)}")
    if obs.get("schema") != RAW_ROW_OBSERVATION_SCHEMA:
        raise ValueError("raw_obs_schema_mismatch")
    if obs.get("science_claim") is True:
        raise ValueError("raw_obs_science_claim_forbidden")
    row = obs["gating_row"]
    if row not in APPLICABILITY_MAP:
        raise ValueError(f"raw_obs_unknown_row:{row}")
    ms = obs["measured_surfaces"]
    if set(ms.keys()) != set(APPLICABILITY_MAP[row]):
        raise ValueError("raw_obs_surface_set_mismatch")
    if obs.get("rank_spec_digest") != RANK_SPEC_DIGEST_EXPECTED:
        raise ValueError("raw_obs_rank_spec_digest_mismatch")




def recompute_surface_cells_from_metrics(
    *,
    gating_row: str,
    metrics: Mapping[str, Any],
    fixture_contract_raw_fail: bool,
    key_universe: Sequence[str] | None = None,
) -> dict[str, bool]:
    """Delegate to pure primitive metric reducer (per-key hashes, not summary bools)."""
    ku = list(key_universe if key_universe is not None else metrics.get("_key_universe") or [])
    if not ku and "post_q_sha256_by_key" in metrics:
        ku = sorted(str(k) for k in (metrics.get("post_q_sha256_by_key") or {}))
    if not ku and "events_equal_by_key" in metrics:
        ku = sorted(str(k) for k in (metrics.get("events_equal_by_key") or {}))
    return recompute_surface_cells_from_primitives(
        gating_row=gating_row,
        metrics=metrics,
        key_universe=ku,
        fixture_contract_raw_fail=fixture_contract_raw_fail,
    )



def load_and_validate_raw_artifact(
    *,
    path: Path,
    expected_sha256: str,
    expected_gating_row: str,
) -> dict[str, Any]:
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_sha256:
        raise ValueError(
            f"raw_sha_mismatch row={expected_gating_row} path={path} "
            f"expected={expected_sha256} actual={digest}"
        )
    obs = json.loads(raw.decode("utf-8"))
    if not isinstance(obs, dict):
        raise ValueError("raw_obs_not_object")
    validate_raw_row_observation(obs)
    if obs.get("gating_row") != expected_gating_row:
        raise ValueError(
            f"raw_obs_gating_row_mismatch: path binds {expected_gating_row} "
            f"but body has {obs.get('gating_row')}"
        )
    if bool(obs.get("synthetic_only")) is True:
        raise ValueError(f"synthetic_row_rejected:{expected_gating_row}")
    expected_dev = EXPECTED_DEVICE_BY_ROW[expected_gating_row]
    body_dev = str(obs.get("device") or "")
    if not (body_dev == expected_dev or body_dev.startswith(expected_dev)):
        raise ValueError(
            f"wrong_device row={expected_gating_row} expected={expected_dev} actual={body_dev}"
        )
    # recompute key universe sha
    recomputed_ku = key_universe_sha256(obs.get("key_universe") or [])
    if recomputed_ku != obs.get("key_universe_sha256"):
        raise ValueError(f"key_universe_sha_tamper:{expected_gating_row}")
    # recompute surface cells from per-key primitives
    metrics = dict(obs.get("metrics") or {})
    ku = [str(k) for k in (obs.get("key_universe") or [])]
    validate_metrics_schema(gating_row=expected_gating_row, metrics=metrics)
    recomputed = recompute_surface_cells_from_metrics(
        gating_row=expected_gating_row,
        metrics=metrics,
        fixture_contract_raw_fail=bool(obs.get("fixture_contract_raw_fail")),
        key_universe=ku,
    )
    claimed = {k: bool(v) for k, v in dict(obs.get("measured_surfaces") or {}).items()}
    if claimed != recomputed:
        raise ValueError(
            f"metric_cell_contradiction row={expected_gating_row} "
            f"claimed={claimed} recomputed={recomputed}"
        )
    # phase topology fold for CUDA rows — ALWAYS classify from raw events
    if expected_gating_row.startswith("G_CUDA_"):
        events = obs.get("phase_events")
        topo = obs.get("phase_topology")
        if events is None:
            # reject caller-authored phase_topology without events (C2)
            if topo is not None:
                raise ValueError(
                    f"caller_authored_phase_topology_without_events:{expected_gating_row}"
                )
            raise ValueError(f"cuda_row_missing_phase_events:{expected_gating_row}")
        if not isinstance(events, list) or len(events) == 0:
            raise ValueError(f"cuda_row_empty_phase_events:{expected_gating_row}")
        classified = classify_phase_topology(
            events, expected_node_id=expected_gating_row, require_enforcer_fields=True
        )
        if topo is not None:
            # both present: must match recomputation
            caller_gt = bool((topo or {}).get("good_topology") is True)
            recomputed_gt = bool(classified.get("good_topology") is True)
            if caller_gt != recomputed_gt or str((topo or {}).get("detail")) != str(
                classified.get("detail")
            ):
                raise ValueError(
                    f"phase_topology_mismatch_vs_events:{expected_gating_row} "
                    f"caller={topo} recomputed={classified}"
                )
        # bind validated classification + science gate on synthesized/partial streams
        from calm.hrm_text_158.native_full_stack.lands_ab_eval_phase_jsonl import (
            apply_phase_stream_science_gate,
        )

        obs = dict(obs)
        obs["phase_topology"] = classified
        obs = apply_phase_stream_science_gate(obs, gating_row=expected_gating_row)
    return obs


def build_eval_receipt_from_raw_artifacts(
    *,
    raw_artifact_paths: Mapping[str, Mapping[str, str]],
    source_pins: Mapping[str, str],
    required_key_set: Sequence[str],
    caveats: Sequence[str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """SOLE science consumer; raw_artifact_paths/source_pins/required_key_set required."""
    if not isinstance(raw_artifact_paths, Mapping) or not raw_artifact_paths:
        raise ValueError("raw_artifact_paths_required")
    if not isinstance(source_pins, Mapping) or not source_pins:
        raise ValueError("source_pins_required")
    if required_key_set is None:
        raise ValueError("required_key_set_required")

    if set(raw_artifact_paths.keys()) != set(GATING_ROWS):
        raise ValueError(
            f"raw_artifact_path_row_set_mismatch missing="
            f"{sorted(set(GATING_ROWS)-set(raw_artifact_paths))} "
            f"extra={sorted(set(raw_artifact_paths)-set(GATING_ROWS))}"
        )

    pin_report = verify_source_pins(source_pins, repo_root=repo_root)
    scope_creep = bool(pin_report["scope_creep"])

    raw_by_row: dict[str, dict[str, Any]] = {}
    artifact_meta: dict[str, dict[str, str]] = {}
    fixture_raw = False
    for row in GATING_ROWS:
        entry = raw_artifact_paths[row]
        if "path" not in entry or "sha256" not in entry:
            raise ValueError(f"raw_artifact_entry_incomplete:{row}")
        path = Path(entry["path"])
        if not path.is_absolute():
            root = repo_root or Path.cwd()
            path = root / path
        obs = load_and_validate_raw_artifact(
            path=path,
            expected_sha256=str(entry["sha256"]),
            expected_gating_row=row,
        )
        fixture_raw = fixture_raw or bool(obs.get("fixture_contract_raw_fail"))
        # topology fold may set fixture fail
        if obs.get("_topology_folded_fixture_fail"):
            fixture_raw = True
        raw_by_row[row] = obs
        artifact_meta[row] = {
            "path": str(entry["path"]),
            "sha256": str(entry["sha256"]),
        }

    # derive 17-cell matrix from RECOMPUTED surfaces (already equal to claimed)
    matrix: dict[str, bool] = {}
    for row in GATING_ROWS:
        for surf, val in raw_by_row[row]["measured_surfaces"].items():
            matrix[cell_key(row, surf)] = bool(val)
    if set(matrix.keys()) != set(CANONICAL_CELL_KEYS):
        raise ValueError("derived_matrix_not_17_cells")

    primitives = {
        "scope_creep": scope_creep,
        "fixture_contract_raw_fail": bool(fixture_raw),
        "surface_pass_by_row": matrix,
    }
    reducer_out = reduce_lands_ab_branch_strict(primitives)
    row_universes = {row: list(raw_by_row[row].get("key_universe") or []) for row in GATING_ROWS}
    per_maps_by_row: dict[str, list[dict[str, Any]]] = {}
    for row in GATING_ROWS:
        met = raw_by_row[row].get("metrics") or {}
        maps: list[dict[str, Any]] = []
        for mk in ("post_q_sha256_by_key", "post_logical_acc_sha256_by_key", "events_equal_by_key"):
            if mk in met and isinstance(met[mk], dict):
                maps.append(met[mk])
        if maps:
            per_maps_by_row[row] = maps
    keys = validate_required_key_universe(
        required_key_set=required_key_set,
        row_key_universes=row_universes,
        per_key_maps_by_row=per_maps_by_row,
    )
    receipt = {
        "schema": EVAL_RECEIPT_SCHEMA,
        "task_id": TASK_ID,
        "plan_sha256": PLAN_V6_SHA256,
        "plan_path": PLAN_V6_PATH,
        "source_pins": dict(source_pins),
        "source_pin_report": pin_report,
        "required_key_set": keys,
        "required_key_set_sha256": key_universe_sha256(keys),
        "row_key_universes": {r: list(row_universes[r]) for r in GATING_ROWS},
        "raw_row_artifacts": artifact_meta,
        "surface_pass_by_row": matrix,
        "scope_creep": scope_creep,
        "fixture_contract_raw_fail": bool(fixture_raw),
        "reducer_output": {
            "branch_id": reducer_out["branch_id"],
            "reason_codes": list(reducer_out["reason_codes"]),
            "derived": reducer_out["derived"],
            "ok": reducer_out["ok"],
        },
        "claim_ceiling": dict(CLAIM_CEILING),
        "science_claim": False,
        "synthetic_only": False,
        "caveats": list(
            caveats
            or [
                "IMPLEMENT_v3 evidence-bound science consumer; formal LANDS-AB claim requires post-push packet + go",
                "claim_ceiling.LANDS_AB=false until formal matrix run",
            ]
        ),
    }
    missing = REQUIRED_EVAL_RECEIPT_KEYS - set(receipt.keys())
    if missing:
        raise ValueError(f"eval_receipt_missing:{sorted(missing)}")
    return receipt


# Back-compat name used by older tests — now requires artifacts path mapping when used for science
def build_eval_receipt_from_raw_observations(
    *,
    raw_by_row: Mapping[str, Mapping[str, Any]] | None = None,
    raw_artifact_paths: Mapping[str, Mapping[str, str]] | None = None,
    source_pins: Mapping[str, str] | None = None,
    caveats: Sequence[str] | None = None,
    required_key_set: Sequence[str] | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Science path requires raw_artifact_paths + source_pins + required_key_set."""
    if raw_artifact_paths is None or source_pins is None or required_key_set is None:
        raise ValueError(
            "science_consumer_requires_raw_artifact_paths_and_source_pins_and_required_key_set"
        )
    if raw_by_row is not None:
        pass  # artifacts authoritative
    return build_eval_receipt_from_raw_artifacts(
        raw_artifact_paths=raw_artifact_paths,
        source_pins=source_pins,
        required_key_set=required_key_set,
        caveats=caveats,
        repo_root=repo_root,
    )


def derive_matrix_from_raw_observations(
    raw_by_row: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Diagnostic helper only — rejects synthetic_only; not science path."""
    if set(raw_by_row.keys()) != set(GATING_ROWS):
        raise ValueError(
            f"raw_row_set_mismatch missing={sorted(set(GATING_ROWS)-set(raw_by_row))} "
            f"extra={sorted(set(raw_by_row)-set(GATING_ROWS))}"
        )
    matrix: dict[str, bool] = {}
    fixture_raw = False
    for row in GATING_ROWS:
        obs = raw_by_row[row]
        validate_raw_row_observation(obs)
        if obs.get("gating_row") != row:
            raise ValueError(f"raw_obs_gating_row_mismatch:{row}")
        if bool(obs.get("synthetic_only")) is True:
            raise ValueError(f"synthetic_row_rejected:{row}")
        recomputed = recompute_surface_cells_from_metrics(
            gating_row=row,
            metrics=dict(obs.get("metrics") or {}),
            fixture_contract_raw_fail=bool(obs.get("fixture_contract_raw_fail")),
            key_universe=list(obs.get("key_universe") or []),
        )
        claimed = {k: bool(v) for k, v in dict(obs.get("measured_surfaces") or {}).items()}
        if claimed != recomputed:
            raise ValueError(f"metric_cell_contradiction:{row}")
        fixture_raw = fixture_raw or bool(obs.get("fixture_contract_raw_fail"))
        for surf, val in recomputed.items():
            matrix[cell_key(row, surf)] = bool(val)
    if set(matrix.keys()) != set(CANONICAL_CELL_KEYS):
        raise ValueError("derived_matrix_not_17_cells")
    return {
        "scope_creep": False,  # diagnostic path has no pin check — science path uses pins
        "fixture_contract_raw_fail": bool(fixture_raw),
        "surface_pass_by_row": matrix,
        "diagnostic_only": True,
        "science_claim": False,
    }


def build_eval_receipt_from_primitives(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Diagnostic-only; not science path."""
    if "branch_id" in raw:
        raise ValueError("caller_authored_branch_id_forbidden")
    if raw.get("schema") == EVAL_RECEIPT_SCHEMA:
        raise ValueError("primitives_builder_cannot_mint_eval_receipt_schema")
    reduced = reduce_lands_ab_branch_strict(raw)
    return {
        "schema": DIAGNOSTIC_RECEIPT_SCHEMA,
        "synthetic_only": True,
        "science_claim": False,
        "claim_ceiling": dict(CLAIM_CEILING),
        "reducer_output": reduced,
        "caveat": "diagnostic primitives path; not evidence-bound; cannot mint LANDS-AB",
    }
