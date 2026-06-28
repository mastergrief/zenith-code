"""Two-phase input-manifest bind contract for D recompute-window v2 postrun.

GAP-A / GAP-B fix for the postrun classifier <-> launch packet integration:

* The committed launch packet predeclares a SPEC -- an EXACT artifact allowlist
  plus run metadata, and **no produced hashes**. The spec is reviewable in the
  packet JSON and carries a ``spec_sha256`` over its own canonical body.
* After the producing run, a bind step (``build_input_manifest``) writes a
  manifest over **only** the allowlisted artifacts, recording their live hashes.
  It fails closed if any allowlisted artifact is missing, and it can never add
  arbitrary present files (it iterates the spec allowlist, not the directory).
* The classifier accepts the bound manifest **only if** it proves provenance
  from the packet spec (``spec_sha256`` recomputed from the packet + ``run_id``
  + exact allowlist), and then **re-hashes the observed artifacts live**,
  failing closed on any mismatch. Stored manifest hashes are cross-checked
  evidence, never trusted as authority.

This prevents ceremonial self-approval: a manifest listing "whatever files are
present" fails the allowlist-equality and ``spec_sha256`` checks; a manifest
with forged hashes fails the live re-hash.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

INPUT_MANIFEST_SPEC_SCHEMA = "hrm_text_158_d_recompute_input_manifest_spec/v0"
INPUT_MANIFEST_SCHEMA = "hrm_text_158_d_recompute_postrun_input_manifest/v0"
PACKET_SPEC_KEY = "expected_native_input_manifest_spec"

# GAP-B reconciled v2 artifact set -- exactly what the official v2
# launch_sequence emits and that affects the v2 verdict / launch eligibility.
# The source of truth is the committed packet spec; this constant is the
# canonical reconciliation asserted equal to the packet allowlist by tests.
# NOTE: deliberately NO root ``driver_summary.json`` -- that file is not emitted
# by any launch_sequence step, and the real meta-artifact
# (``prelaunch/packet_replay_driver_summary.json``) embeds the classifier's own
# verdict (circular), so it is never an input.
V2_RECONCILED_ALLOWLIST: tuple[str, ...] = (
    "d_recompute_window_diagnostic/receipt.json",
    "d_recompute_window_diagnostic/recompute_window_log.jsonl",
    "prelaunch/calibrated_selector_manifest.json",
    "prelaunch/calibration_prepass_receipt.json",
    "prelaunch/scale_smoke_receipt.json",
    "prelaunch/post_confirmation_hygiene_receipt.json",
    "prelaunch/parent_checkpoint_rehash.json",
    "prelaunch/parent_checkpoint_rehash_after_calibration_warmup.json",
    "prelaunch/parent_checkpoint_rehash_after_scale_smoke.json",
    "prelaunch/parent_checkpoint_rehash_after_confirmation.json",
)

LOG_ARTIFACT_REL = "d_recompute_window_diagnostic/recompute_window_log.jsonl"
DIAGNOSTIC_RECEIPT_REL = "d_recompute_window_diagnostic/receipt.json"
CALIBRATED_SELECTOR_MANIFEST_REL = "prelaunch/calibrated_selector_manifest.json"

ROW_COUNT_CHECK_EQUALS_STEPS_X_KEYS = "equals_steps_x_keys"


def reducer_sha256() -> str:
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonl_row_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _log_state_keys(path: Path) -> set[str]:
    keys: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        keys.add(str(json.loads(line)["state_key"]))
    return keys


def compute_spec_sha256(spec: Mapping[str, Any]) -> str:
    """sha256 over the canonical spec body, excluding any embedded ``spec_sha256``.

    Mirrors the selector-manifest digest convention (sort_keys + compact
    separators) so the spec self-binds independent of key insertion order.
    """
    body = {k: v for k, v in dict(spec).items() if k != "spec_sha256"}
    encoded = json.dumps(body, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_packet_spec(packet: Mapping[str, Any]) -> dict[str, Any]:
    spec = packet.get(PACKET_SPEC_KEY)
    if not isinstance(spec, dict):
        raise ValueError(f"missing {PACKET_SPEC_KEY} in packet")
    allowlist = spec.get("artifact_allowlist")
    if not isinstance(allowlist, list) or not allowlist:
        raise ValueError(f"{PACKET_SPEC_KEY}.artifact_allowlist missing or empty")
    return spec


def _selector_internal_sha256(path: Path) -> str:
    """Return the selector manifest's self-bound internal ``manifest_sha256``.

    Uses ``load_stratified_selector_manifest`` (lazy import) which recomputes the
    internal digest from content and raises on self-bind mismatch -- so a tampered
    selector body cannot keep a stale digest field.
    """
    from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
        load_stratified_selector_manifest,
    )

    manifest = load_stratified_selector_manifest(path)
    return manifest.manifest_sha256


def _selector_entry_keys(path: Path) -> set[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {str(entry["state_key"]) for entry in payload.get("entries", ())}


def _diag_selected_key_count_and_steps(run_root: Path) -> tuple[int | None, int | None]:
    receipt_path = run_root / DIAGNOSTIC_RECEIPT_REL
    if not receipt_path.is_file():
        return None, None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    steps = receipt.get("steps_completed")
    steps_int = int(steps) if steps is not None else None
    return None, steps_int


def _diag_selector_sha256(run_root: Path) -> str | None:
    receipt_path = run_root / DIAGNOSTIC_RECEIPT_REL
    if not receipt_path.is_file():
        return None
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    value = receipt.get("d_recompute_selector_manifest_sha256")
    return str(value) if value is not None else None


def collect_observed_artifact_hashes(
    run_root: Path,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash the allowlisted artifacts that are present (present=False otherwise).

    Constrained to the spec allowlist -- never walks the directory, so it cannot
    pick up non-allowlisted files. Shared by ``build_input_manifest`` (which then
    fails closed on missing) and the classifier's receipt echo.
    """
    allowlist = list(spec["artifact_allowlist"])
    selector_rel = str(spec.get("calibrated_selector_manifest_rel") or CALIBRATED_SELECTOR_MANIFEST_REL)
    log_rel = str(spec.get("log_artifact_rel") or LOG_ARTIFACT_REL)
    observed: dict[str, Any] = {}
    for rel in allowlist:
        path = run_root / rel
        if not path.is_file():
            observed[rel] = {"present": False}
            continue
        entry: dict[str, Any] = {"present": True, "sha256": _sha256_file(path)}
        if rel == log_rel:
            entry["jsonl_row_count"] = _jsonl_row_count(path)
        if rel == selector_rel:
            entry["selector_internal_manifest_sha256"] = _selector_internal_sha256(path)
        observed[rel] = entry
    return observed


def build_input_manifest(run_root: Path, packet: Mapping[str, Any]) -> dict[str, Any]:
    """Build the bound input manifest over ONLY the spec allowlist.

    Fails closed (ValueError) if any allowlisted artifact is missing -- a
    partial run can never produce a "green" manifest.
    """
    spec = load_packet_spec(packet)
    allowlist = list(spec["artifact_allowlist"])
    observed = collect_observed_artifact_hashes(run_root, spec)
    missing = [rel for rel in allowlist if not observed.get(rel, {}).get("present")]
    if missing:
        raise ValueError(f"missing_allowlisted_artifacts:{sorted(missing)}")
    artifacts = {rel: {k: v for k, v in observed[rel].items() if k != "present"} for rel in allowlist}
    return {
        "schema": INPUT_MANIFEST_SCHEMA,
        "spec_schema": str(spec.get("spec_schema") or INPUT_MANIFEST_SPEC_SCHEMA),
        "spec_sha256": compute_spec_sha256(spec),
        "run_id": str(spec.get("run_id")),
        "packet_revision": str(spec.get("packet_revision")),
        "artifact_allowlist": list(allowlist),
        "artifacts": artifacts,
        "bind_reducer_sha256": reducer_sha256(),
    }


def verify_input_manifest_against_spec(
    run_root: Path,
    packet: Mapping[str, Any],
    bound_manifest: Mapping[str, Any],
) -> list[str]:
    """Return failure codes; empty list means the bound manifest is trustworthy.

    Provenance (spec_sha256 + run_id + exact allowlist) is checked against the
    committed packet, then observed artifacts are RE-HASHED live and compared to
    the bound manifest. Stored hashes are never authority.
    """
    failures: list[str] = []
    try:
        spec = load_packet_spec(packet)
    except (ValueError, TypeError) as exc:
        return [f"missing_packet_spec:{exc}"]

    expected_spec_sha = compute_spec_sha256(spec)
    if str(bound_manifest.get("spec_sha256")) != expected_spec_sha:
        failures.append("spec_sha256_mismatch")
    if str(bound_manifest.get("run_id")) != str(spec.get("run_id")):
        failures.append("run_id_mismatch")

    # Catch a spec lifted from a different run/revision: the embedded spec's
    # run_id / packet_revision must agree with the packet's top-level fields.
    top_run_id = packet.get("run_id")
    if top_run_id is not None and str(spec.get("run_id")) != str(top_run_id):
        failures.append("packet_toplevel_run_id_vs_spec_mismatch")
    top_revision = packet.get("packet_revision")
    if top_revision is not None and str(spec.get("packet_revision")) != str(top_revision):
        failures.append("packet_toplevel_packet_revision_vs_spec_mismatch")

    allowlist = list(spec["artifact_allowlist"])
    allowlist_set = set(allowlist)
    bound_allowlist = set(bound_manifest.get("artifact_allowlist") or [])
    if bound_allowlist != allowlist_set:
        failures.append("bound_allowlist_mismatch")
    artifacts = bound_manifest.get("artifacts") or {}
    if not isinstance(artifacts, Mapping):
        return failures + ["bound_artifacts_malformed"]
    bound_keys = set(artifacts.keys())
    extra = sorted(bound_keys - allowlist_set)
    if extra:
        failures.append(f"manifest_extra_paths:{extra}")
    missing_keys = sorted(allowlist_set - bound_keys)
    if missing_keys:
        failures.append(f"manifest_missing_paths:{missing_keys}")

    selector_rel = str(spec.get("calibrated_selector_manifest_rel") or CALIBRATED_SELECTOR_MANIFEST_REL)
    log_rel = str(spec.get("log_artifact_rel") or LOG_ARTIFACT_REL)

    live_row_count: int | None = None
    for rel in allowlist:
        path = run_root / rel
        if not path.is_file():
            failures.append(f"missing_artifact:{rel}")
            continue
        bound_entry = artifacts.get(rel)
        if not isinstance(bound_entry, Mapping):
            failures.append(f"missing_manifest_entry:{rel}")
            continue
        if str(bound_entry.get("sha256")) != _sha256_file(path):
            failures.append(f"sha256_mismatch:{rel}")
        if rel == log_rel:
            live_row_count = _jsonl_row_count(path)
            if bound_entry.get("jsonl_row_count") != live_row_count:
                failures.append(f"row_count_mismatch:{rel}")
        if rel == selector_rel:
            try:
                live_internal = _selector_internal_sha256(path)
            except (ValueError, KeyError, OSError) as exc:
                failures.append(f"selector_internal_sha_self_bind_failed:{exc}")
            else:
                if str(bound_entry.get("selector_internal_manifest_sha256")) != live_internal:
                    failures.append("selector_internal_sha_mismatch_vs_bound")
                diag_selector = _diag_selector_sha256(run_root)
                if diag_selector is not None and live_internal != diag_selector:
                    failures.append("selector_internal_sha_mismatch_vs_diag_receipt")

    # Independent row-count sanity: rows == steps_completed * selected_key_count.
    row_check = spec.get("row_count_check")
    if isinstance(row_check, Mapping) and row_check.get("mode") == ROW_COUNT_CHECK_EQUALS_STEPS_X_KEYS:
        if live_row_count is None:
            log_path = run_root / log_rel
            if log_path.is_file():
                live_row_count = _jsonl_row_count(log_path)
        _, steps = _diag_selected_key_count_and_steps(run_root)
        selector_path = run_root / selector_rel
        selected_key_count = None
        if selector_path.is_file():
            selected_key_count = len(_selector_entry_keys(selector_path))
        # Fail closed: the row-count source must be present, never silently skip.
        if steps is None:
            failures.append("row_count_source_missing:steps_completed")
        elif live_row_count is not None and selected_key_count:
            expected_rows = int(steps) * int(selected_key_count)
            if live_row_count != expected_rows:
                failures.append(
                    f"row_count_check_failed:{live_row_count}!={steps}x{selected_key_count}"
                )

    # Selector/log key-set alignment (5A2 in-vivo alignment checklist).
    if spec.get("selector_log_key_alignment_check"):
        selector_path = run_root / selector_rel
        log_path = run_root / log_rel
        if selector_path.is_file() and log_path.is_file():
            if _selector_entry_keys(selector_path) != _log_state_keys(log_path):
                failures.append("selector_log_state_key_set_mismatch")

    return failures
