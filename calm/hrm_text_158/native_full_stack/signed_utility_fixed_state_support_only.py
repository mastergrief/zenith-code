"""Support-only characterization orchestrator (D2c9). Stops before apply/parity/NLL."""
from __future__ import annotations
import hashlib, json, re, subprocess
from pathlib import Path
from typing import Any, Mapping
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_authoritative_gpu import (
    AuthoritativeGpuError, AuthoritativeGpuHooks, build_live_hooks,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_legal_subset import (
    ESTIMAND_NAME, LegalSubsetError, MAX_AUTHORITATIVE_RESULT_BYTES, MAX_COMPACT_TELEMETRY_BYTES,
    assert_compact_json_nbytes, characterize_plans_bidirectional_legal,
    enforce_legal_subset_support_floors, payload_has_raw_index_arrays,
)
from calm.hrm_text_158.native_full_stack.signed_utility_fixed_state_pin_validation import (
    PinValidationError, WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256, WATCH_WRAP_HRM158_SHA256,
    rehash_path, require_formal_source_pin_basenames, require_head_equals_upstream_pin,
    validate_proof_packet_source_pins,
)
SUPPORT_ONLY_CALL_GRAPH = (
    "parse_packet_live_rehash_pins", "parent_sha_pre_materialize", "rebuild_support_batches_bind_leakage",
    "fork_clones_from_materialized_base", "capture_backward_vote", "characterize_plans_bidirectional_legal",
    "emit_complete_compact_characterization", "enforce_legal_subset_support_floors", "emit_support_only_terminal")
_HEAD_RE = re.compile(r"^[0-9a-f]{40}$")
SUPPORT_ELIGIBLE = "SUPPORT_ELIGIBLE"
SUPPORT_DEGENERATE_BELOW_FLOOR = "SUPPORT_DEGENERATE_BELOW_FLOOR"
SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE = "SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE"
SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE = "SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE"
TERMINAL_TAXONOMY = (
    SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE,
    SUPPORT_DEGENERATE_BELOW_FLOOR, SUPPORT_ELIGIBLE)
class SupportOnlyError(RuntimeError): pass
def require_exact_40hex_commit(repo_root: str | Path, expected_head: str) -> str:
    head = str(expected_head)
    if not _HEAD_RE.fullmatch(head): raise PinValidationError(f"expected_head_not_40hex:{head}")
    try:
        subprocess.check_output(
            ["git", "rev-parse", "--verify", f"{head}^{{commit}}"], cwd=str(repo_root), text=True,
            stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as exc:
        raise PinValidationError(f"expected_head_not_commit:{head}:{exc.returncode}") from exc
    return require_head_equals_upstream_pin(repo_root, head)
def _require_path_pin(pin: Any, *, label: str) -> str:
    if not isinstance(pin, Mapping) or "absolute_path" not in pin or "sha256" not in pin:
        raise PinValidationError(f"{label}_pin_requires_absolute_path_and_sha256")
    path = Path(str(pin["absolute_path"]))
    if not path.is_file(): raise PinValidationError(f"{label}_pin_path_missing:{path}")
    digest = rehash_path(path)
    if digest != str(pin["sha256"]): raise PinValidationError(f"{label}_pin_sha_mismatch:{digest}!={pin['sha256']}")
    return digest
def validate_launch_surface_pins(packet: Mapping[str, Any]) -> dict[str, str]:
    cli_sha = _require_path_pin(packet.get("cli_pin"), label="cli")
    ww_sha = _require_path_pin(packet.get("watch_wrap_pin"), label="watch_wrap")
    if ww_sha == WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256:
        raise PinValidationError("watch_wrap_must_be_hrm158_a19f1c5f_not_claw_code_ba54e8dd")
    if ww_sha != WATCH_WRAP_HRM158_SHA256: raise PinValidationError(f"watch_wrap_unexpected_sha:{ww_sha}")
    return {"cli": cli_sha, "watch_wrap": ww_sha}
def _source_sha_map(packet: Mapping[str, Any]) -> dict[str, str | None]:
    pins = packet.get("source_pins"); out: dict[str, str | None] = {}
    if not isinstance(pins, Mapping): return out
    for key, pin in pins.items():
        if key == "head" or not isinstance(pin, Mapping) or "absolute_path" not in pin: continue
        try: out[str(key)] = rehash_path(pin["absolute_path"])
        except Exception: out[str(key)] = None  # noqa: BLE001
    return out
def _launch_sha_map(packet: Mapping[str, Any]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for label, key in (("cli", "cli_pin"), ("watch_wrap", "watch_wrap_pin")):
        pin = packet.get(key)
        if not isinstance(pin, Mapping) or "absolute_path" not in pin:
            out[label] = None; continue
        try: out[label] = rehash_path(pin["absolute_path"])
        except Exception: out[label] = None  # noqa: BLE001
    return out
def _validate_packet(packet: Mapping[str, Any]) -> None:
    validate_proof_packet_source_pins(packet); require_formal_source_pin_basenames(packet)
    validate_launch_surface_pins(packet)
    mode = str(packet.get("pin_mode") or "formal")
    if mode == "cpu_static_di": return
    root, expected = packet.get("repo_root"), packet.get("expected_head")
    if not root or not expected: raise PinValidationError("formal_head_or_repo_root_missing")
    require_exact_40hex_commit(root, str(expected))
def _safe_char_diag(characterization: Any) -> dict[str, Any] | None:
    if characterization is None: return None
    try:
        raw = json.dumps(characterization, allow_nan=False, separators=(",", ":"), sort_keys=True).encode()
        digest = hashlib.sha256(raw).hexdigest()
    except Exception as exc:  # noqa: BLE001
        return {"characterization_invalid": f"serialize:{type(exc).__name__}:{exc}", "sha256_unavailable": True}
    if payload_has_raw_index_arrays(characterization):
        return {"characterization_invalid": "raw_index_arrays", "characterization_sha256": digest, "sha256_unavailable": False}
    if len(raw) > MAX_COMPACT_TELEMETRY_BYTES:
        return {"characterization_invalid": f"oversized:{len(raw)}", "characterization_sha256": digest, "sha256_unavailable": False}
    return dict(characterization)
def _terminal(*, classifier, route, reason=None, parent_pre=None, parent_post=None,
              source_pre=None, source_post=None, launch_pre=None, launch_post=None,
              characterization=None, extra=None) -> dict[str, Any]:
    if classifier not in TERMINAL_TAXONOMY: raise SupportOnlyError(f"illegal_classifier:{classifier}")
    payload: dict[str, Any] = {
        "schema": "support_only_terminal_v1", "estimand": ESTIMAND_NAME, "classifier": classifier,
        "route": list(route), "claim_ceiling": "support_eligibility_only",
        "parent_sha_pre": parent_pre, "parent_sha_post": parent_post,
        "source_sha_pre": dict(source_pre or {}), "source_sha_post": dict(source_post or {}),
        "launch_surface_sha_pre": dict(launch_pre or {}), "launch_surface_sha_post": dict(launch_post or {}),
    }
    if reason is not None: payload["reason"] = reason
    safe = _safe_char_diag(characterization)
    if safe is not None: payload["characterization"] = safe
    if extra: payload.update(dict(extra))
    if payload_has_raw_index_arrays(payload): raise SupportOnlyError("raw_index_arrays_forbidden")
    assert_compact_json_nbytes(payload, limit=MAX_AUTHORITATIVE_RESULT_BYTES, label="support_terminal")
    return payload
def _rehash_all(packet, parent_path=None):
    parent_post = None
    if parent_path is not None:
        try: parent_post = rehash_path(parent_path)
        except Exception: parent_post = None  # noqa: BLE001
    return parent_post, _source_sha_map(packet), _launch_sha_map(packet)
def _immutability_ok(parent_pre, parent_post, source_pre, source_post, launch_pre, launch_post) -> str | None:
    if parent_pre is not None and parent_post != parent_pre: return "parent_sha_drift"
    if dict(source_post or {}) != dict(source_pre or {}): return "source_sha_drift"
    if dict(launch_post or {}) != dict(launch_pre or {}): return "launch_surface_sha_drift"
    return None
def run_support_only_characterization(packet: Mapping[str, Any], *, hooks: AuthoritativeGpuHooks | None = None) -> dict[str, Any]:
    if not isinstance(packet, Mapping): raise SupportOnlyError("packet_not_mapping")
    route: list[str] = []; parent_pre = None; parent_path = None; characterization = None
    source_pre = _source_sha_map(packet); launch_pre = _launch_sha_map(packet)
    source_post, launch_post = dict(source_pre), dict(launch_pre)
    def term(*, classifier, reason=None, characterization=None, extra=None, parent_post=None):
        return _terminal(
            classifier=classifier, route=route, reason=reason, parent_pre=parent_pre, parent_post=parent_post,
            source_pre=source_pre, source_post=source_post, launch_pre=launch_pre, launch_post=launch_post,
            characterization=characterization, extra=extra)
    def finish(classifier, *, reason=None, characterization=None, extra=None, revalidate_pins=False):
        nonlocal source_post, launch_post
        parent_post, source_post, launch_post = _rehash_all(packet, parent_path)
        drift = _immutability_ok(parent_pre, parent_post, source_pre, source_post, launch_pre, launch_post)
        if drift is not None:
            return term(classifier=SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason=drift,
                        characterization=characterization, extra=extra, parent_post=parent_post)
        if revalidate_pins:
            try:
                validate_proof_packet_source_pins(packet); validate_launch_surface_pins(packet)
            except PinValidationError as exc:
                return term(classifier=SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason=str(exc),
                            characterization=characterization, extra=extra, parent_post=parent_post)
        return term(classifier=classifier, reason=reason, characterization=characterization,
                    extra=extra, parent_post=parent_post)
    try:
        _validate_packet(packet); route.append("parse_packet_live_rehash_pins")
        source_pre = _source_sha_map(packet); launch_pre = _launch_sha_map(packet)
        source_post, launch_post = dict(source_pre), dict(launch_pre)
    except PinValidationError as exc:
        source_post, launch_post = _source_sha_map(packet), _launch_sha_map(packet)
        return term(classifier=SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason=str(exc))
    mode = str(packet.get("pin_mode") or "formal")
    if hooks is not None and mode != "cpu_static_di":
        return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="hooks_require_pin_mode_cpu_static_di")
    try:
        h = hooks if hooks is not None else build_live_hooks(packet)
        parent = packet.get("parent_checkpoint") or {}
        if "absolute_path" not in parent or "sha256" not in parent:
            return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="parent_checkpoint_missing")
        parent_path = parent["absolute_path"]; parent_pre = rehash_path(parent_path)
        if parent_pre != str(parent["sha256"]):
            return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="parent_sha_mismatch",
                          extra={"expected": parent["sha256"]})
        route.append("parent_sha_pre_materialize")
        bundle = h.materialize(packet); batches = h.rebuild_support_batches(bundle); leak = h.leakage_report(batches)
        route.append("rebuild_support_batches_bind_leakage")
        if leak.get("pass") is not True:
            return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="leakage_overlap",
                          extra={"leakage": leak}, revalidate_pins=True)
        arms = h.fork_arm_states(bundle)
        if "capture_disposable" not in arms or "prod" not in arms:
            return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="fork_arms_missing", revalidate_pins=True)
        route.append("fork_clones_from_materialized_base")
        plans, _cs, holder_calls = h.capture_plans(bundle, arms)
        if holder_calls != 1:
            return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason="holder_call_count",
                          extra={"calls": holder_calls}, revalidate_pins=True)
        route.append("capture_backward_vote")
        try:
            _f, characterization = characterize_plans_bidirectional_legal(arms["prod"], plans)
        except LegalSubsetError as exc:
            return finish(SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE, reason=str(exc), revalidate_pins=True)
        route.append("characterize_plans_bidirectional_legal")
        try:
            assert_compact_json_nbytes(characterization, limit=MAX_COMPACT_TELEMETRY_BYTES, label="characterization")
            if payload_has_raw_index_arrays(characterization): raise SupportOnlyError("raw_index_arrays_forbidden")
            json.dumps(characterization, allow_nan=False, separators=(",", ":"), sort_keys=True)
        except Exception as exc:  # noqa: BLE001
            return finish(SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE,
                          reason=f"characterization_invalid:{type(exc).__name__}:{exc}",
                          characterization=characterization, revalidate_pins=True)
        route.append("emit_complete_compact_characterization")
        try: enforce_legal_subset_support_floors(characterization)
        except LegalSubsetError as exc:
            route.append("enforce_legal_subset_support_floors")
            return finish(SUPPORT_DEGENERATE_BELOW_FLOOR, reason=str(exc),
                          characterization=characterization, revalidate_pins=True)
        route.append("enforce_legal_subset_support_floors"); route.append("emit_support_only_terminal")
        return finish(SUPPORT_ELIGIBLE, characterization=characterization, extra={"leakage_pass": True},
                      revalidate_pins=True)
    except (PinValidationError, AuthoritativeGpuError, SupportOnlyError) as exc:
        return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason=str(exc),
                      characterization=characterization, revalidate_pins=False)
    except Exception as exc:  # noqa: BLE001
        return finish(SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE, reason=f"{type(exc).__name__}:{exc}",
                      characterization=characterization, revalidate_pins=False)
__all__ = [
    "SUPPORT_ASYMMETRIC_OR_CHARACTERIZATION_FAILURE", "SUPPORT_DEGENERATE_BELOW_FLOOR", "SUPPORT_ELIGIBLE",
    "SUPPORT_INTEGRITY_OR_EXECUTION_FAILURE", "SUPPORT_ONLY_CALL_GRAPH", "TERMINAL_TAXONOMY",
    "SupportOnlyError", "require_exact_40hex_commit", "run_support_only_characterization",
    "validate_launch_surface_pins",
]
