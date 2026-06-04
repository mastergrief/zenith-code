"""Front-C identity-artifact adapter and CPU/static inventory checks.

This module does not emit live learner identities by itself. It validates the
compact identity artifacts that a future lightweight emission run must write and
proves when saved B2 audit roots are count/hash-only rather than identity-ready.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from calm.hrm_text_158.native_full_stack.front_c_harness import (
    base3_q_ledger_from_front_c_artifact,
    front_c_report_from_mapping,
)
from calm.hrm_text_158.native_full_stack.front_c_projection import (
    FRONT_C_SCHEMA_VERSION,
    FrontCDecisionPath,
    FrontCDecisionSurfaceStep,
    FrontCProjectionReport,
    normalize_front_c_decision_path,
    normalize_front_c_decision_surface_step,
    validate_front_c_projection_report,
)
from calm.hrm_text_158.native_full_stack.q_entropy_packing import Base3QEntropyLedgerRow


FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION = (
    "hrm_text_158_front_c/v0.identity_emitter_adapter"
)
FRONT_C_IDENTITY_EXTRACTABLE = "identity_extractable"
FRONT_C_COUNT_ONLY = "count_only"
FRONT_C_AMBIGUOUS_SPLIT_CONTRACT = "ambiguous_split_identity_contract"
FRONT_C_SYNTHETIC_FIXTURE_ARTIFACT = "synthetic_fixture"
FRONT_C_RUN_DERIVED_ARTIFACT = "run_derived"
FRONT_C_CANONICAL_STATE_KEY_SEMANTICS = "canonical_tensor_parameter_key"
FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS = "local_to_tensor_flat_index"
FRONT_C_STATE_LAYOUT_HASH_SEMANTICS = "stable_layout_key_map_not_mutable_qacc_values"
FRONT_C_DENSE_DECISION_SOURCE = "global_rate_cap_dense_int16_reference"
FRONT_C_SPARSE_DECISION_SOURCE = "front_c_sparse_encode_decode_reference"
FRONT_C_PATH_B_CARRY_FORWARD_FOLDS = (
    "P1: populate real overhead-inclusive metadata bits, not placeholder zeros",
    "P2: independently derive sparse_decision_path from the sparse representation",
    "P3: emit multiple ordered timeline rows so union/churn is meaningful",
    "H0/F: include step-0 prior audit after q/acc conversion and before mutation",
)

_REQUIRED_ARTIFACT_KEYS = (
    "timeline",
    "dense_decision_path",
    "sparse_decision_path",
    "state_metadata",
    "decision_path_derivation",
)
_IDENTITY_SIGNAL_KEYS = frozenset(
    {
        "timeline",
        "dense_decision_path",
        "sparse_decision_path",
        "q_flip_directions",
        "global_rate_cap_accepted_indices",
        "global_rate_cap_deferred_indices",
        "post_veto_applied_indices",
        "replay_ce_veto_indices",
        "pc_aux_veto_indices",
        "applied_directions",
        "accepted_under_global_cap_keys",
        "deferred_under_global_cap_keys",
        "backlog_keys",
        "replay_veto_decision_keys",
    }
)
_COUNT_HASH_SIGNAL_SUFFIXES = ("_count", "_sha256")
_COUNT_HASH_SIGNAL_KEYS = frozenset(
    {
        "row_ids",
        "row_index",
        "checkpoint_payload_summary",
        "tensor_summary_count",
        "authoritative_state_sha256",
        "updater_config_sha256",
    }
)


@dataclass(frozen=True)
class FrontCArtifactInventoryReport:
    """Inventory verdict for a saved audit root or raw artifact payload."""

    schema_version: str
    status: str
    identity_extractable: bool
    inspected_file_count: int
    matched_artifact_path: str
    observed_identity_keys: tuple[str, ...]
    observed_count_or_hash_keys: tuple[str, ...]
    missing_required_keys: tuple[str, ...]
    aggregate_only_required_keys: tuple[str, ...]
    external_q_ledger_required: bool
    rejection_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FrontCIdentityArtifactValidation:
    """Hard validation receipt for a future Front-C identity artifact."""

    schema_version: str
    projection_schema_version: str
    status: str
    timeline_step_count: int
    eligible_weight_count: int
    q_ledger_eligible_weight_count: int
    state_key_count: int
    synthetic_fixture: bool
    independent_sparse_derivation: bool
    claimed_front_c_viable: bool
    non_claims: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["non_claims"] = list(self.non_claims)
        return payload


@dataclass(frozen=True)
class FrontCConflictOverlapDiagnostic:
    """Diagnostic-only overlap classifier for conflict-vs-coupling."""

    schema_version: str
    label: str
    target_helping_count: int
    prior_serving_count: int
    same_identity_opposite_direction_overlap_count: int
    same_identity_same_direction_overlap_count: int
    accepted_prior_harming_count: int
    classification: str
    diagnostic_only: bool
    same_identity_opposite_direction_overlap_keys: tuple[dict[str, int | str], ...]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["same_identity_opposite_direction_overlap_keys"] = [
            dict(item) for item in self.same_identity_opposite_direction_overlap_keys
        ]
        return payload


def _as_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _as_sequence(value: Any, *, name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{name} must be a sequence")
    return value


def _identity_dict(identity: tuple[str, int]) -> dict[str, int | str]:
    state_key, flat_index = identity
    return {"state_key": state_key, "flat_index": int(flat_index)}


def _shape_numel(shape: Sequence[Any], *, state_key: str) -> int:
    dims = tuple(int(dim) for dim in shape)
    if not dims:
        raise ValueError(f"state {state_key!r} logical_shape must be non-empty")
    numel = 1
    for dim in dims:
        if dim <= 0:
            raise ValueError(f"state {state_key!r} logical_shape dims must be > 0")
        numel *= dim
    return int(numel)


def _walk_keys(value: Any, out: set[str]) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            out.add(str(key))
            _walk_keys(child, out)
    elif isinstance(value, list):
        for child in value[:32]:
            _walk_keys(child, out)


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def classify_front_c_identity_payload(
    payload: Mapping[str, Any],
    *,
    inspected_file_count: int = 1,
    matched_artifact_path: str = "",
) -> FrontCArtifactInventoryReport:
    """Classify one payload as identity-ready or count/hash-only."""

    keys: set[str] = set()
    _walk_keys(payload, keys)
    missing = tuple(key for key in _REQUIRED_ARTIFACT_KEYS if key not in payload)
    observed_identity = tuple(sorted(keys & _IDENTITY_SIGNAL_KEYS))
    observed_count_hash = tuple(
        sorted(
            key
            for key in keys
            if key in _COUNT_HASH_SIGNAL_KEYS or key.endswith(_COUNT_HASH_SIGNAL_SUFFIXES)
        )
    )
    extractable = not missing
    return FrontCArtifactInventoryReport(
        schema_version=FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION,
        status=FRONT_C_IDENTITY_EXTRACTABLE if extractable else FRONT_C_COUNT_ONLY,
        identity_extractable=extractable,
        inspected_file_count=int(inspected_file_count),
        matched_artifact_path=matched_artifact_path if extractable else "",
        observed_identity_keys=observed_identity,
        observed_count_or_hash_keys=observed_count_hash,
        missing_required_keys=missing,
        aggregate_only_required_keys=(),
        external_q_ledger_required=bool(extractable and "q_ledger" not in payload),
        rejection_reason=(
            ""
            if extractable
            else "Front-C artifact is not identity-extractable: missing "
            f"{missing}; counts/hashes/eval row IDs are diagnostic-only"
        ),
    )


def classify_front_c_saved_audit_root(root: str | Path) -> FrontCArtifactInventoryReport:
    """Inspect a saved audit root without claiming identities exist."""

    root_path = Path(root)
    if not root_path.exists():
        raise ValueError(f"saved audit root does not exist: {root_path}")
    if root_path.is_file():
        payload = _load_json_mapping(root_path)
        return classify_front_c_identity_payload(
            payload,
            matched_artifact_path=str(root_path),
        )

    summary_paths = sorted(root_path.glob("audits/step_*/summary.json"))
    if not summary_paths:
        summary_paths = sorted(root_path.glob("step_*/summary.json"))
    if not summary_paths:
        raise ValueError(f"saved audit root contains no step summary JSON files: {root_path}")

    keys: set[str] = set()
    matched_payload: Mapping[str, Any] | None = None
    matched_path: Path | None = None
    for path in summary_paths:
        payload = _load_json_mapping(path)
        _walk_keys(payload, keys)
        if matched_payload is None and all(key in payload for key in _REQUIRED_ARTIFACT_KEYS):
            matched_payload = payload
            matched_path = path
    if matched_payload is not None and matched_path is not None:
        return classify_front_c_identity_payload(
            matched_payload,
            inspected_file_count=len(summary_paths),
            matched_artifact_path=str(matched_path),
        )

    observed_identity = tuple(sorted(keys & _IDENTITY_SIGNAL_KEYS))
    observed_count_hash = tuple(
        sorted(
            key
            for key in keys
            if key in _COUNT_HASH_SIGNAL_KEYS or key.endswith(_COUNT_HASH_SIGNAL_SUFFIXES)
        )
    )
    aggregate_required = tuple(key for key in _REQUIRED_ARTIFACT_KEYS if key in keys)
    status = FRONT_C_AMBIGUOUS_SPLIT_CONTRACT if aggregate_required else FRONT_C_COUNT_ONLY
    return FrontCArtifactInventoryReport(
        schema_version=FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION,
        status=status,
        identity_extractable=False,
        inspected_file_count=len(summary_paths),
        matched_artifact_path="",
        observed_identity_keys=observed_identity,
        observed_count_or_hash_keys=observed_count_hash,
        missing_required_keys=tuple(key for key in _REQUIRED_ARTIFACT_KEYS if key not in keys),
        aggregate_only_required_keys=aggregate_required,
        external_q_ledger_required=False,
        rejection_reason=(
            "saved audit root is not identity-extractable: required Front-C "
            "identity keys were observed only in aggregate across files"
            if aggregate_required
            else "saved audit root is not identity-extractable: no complete Front-C "
            "timeline/dense/sparse/state/derivation identity artifact was found"
        ),
    )


def require_front_c_identity_extractable_saved_audit_root(
    root: str | Path,
) -> FrontCArtifactInventoryReport:
    report = classify_front_c_saved_audit_root(root)
    if not report.identity_extractable:
        raise ValueError(report.rejection_reason)
    return report


def _normalize_timeline(payload: Mapping[str, Any]) -> tuple[FrontCDecisionSurfaceStep, ...]:
    timeline = _as_sequence(payload.get("timeline"), name="timeline")
    if not timeline:
        raise ValueError("timeline must contain at least one row")
    return tuple(normalize_front_c_decision_surface_step(step) for step in timeline)


def _normalize_paths(payload: Mapping[str, Any]) -> tuple[FrontCDecisionPath, FrontCDecisionPath]:
    return (
        normalize_front_c_decision_path(payload.get("dense_decision_path")),
        normalize_front_c_decision_path(payload.get("sparse_decision_path")),
    )


def _all_identities(
    timeline: Sequence[FrontCDecisionSurfaceStep],
    dense: FrontCDecisionPath,
    sparse: FrontCDecisionPath,
) -> tuple[tuple[str, int], ...]:
    identities: set[tuple[str, int]] = set()
    for step in timeline:
        identities |= set(step.decision_relevant_exact_keys)
    for path in (dense, sparse):
        identities |= set(path.q_flip_directions)
        identities |= set(path.accepted_under_global_cap_keys)
        identities |= set(path.deferred_under_global_cap_keys)
        identities |= set(path.backlog_keys)
        identities |= set(path.replay_veto_decision_keys)
    return tuple(sorted(identities))


def _validate_state_metadata(
    payload: Mapping[str, Any],
    *,
    timeline: Sequence[FrontCDecisionSurfaceStep],
    dense: FrontCDecisionPath,
    sparse: FrontCDecisionPath,
    eligible_weight_count: int,
) -> Mapping[str, Mapping[str, Any]]:
    metadata = _as_mapping(payload.get("state_metadata"), name="state_metadata")
    if metadata.get("state_key_semantics") != FRONT_C_CANONICAL_STATE_KEY_SEMANTICS:
        raise ValueError("state_metadata.state_key_semantics must be canonical_tensor_parameter_key")
    if metadata.get("flat_index_semantics") != FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS:
        raise ValueError("state_metadata.flat_index_semantics must be local_to_tensor_flat_index")
    if metadata.get("state_hash_semantics") != FRONT_C_STATE_LAYOUT_HASH_SEMANTICS:
        raise ValueError(
            "state_metadata.state_hash_semantics must be stable layout/key-map, "
            "not mutable q/acc value hashes"
        )
    if not str(metadata.get("state_layout_metadata_sha256", "")):
        raise ValueError("state_metadata must include state_layout_metadata_sha256")

    state_entries = _as_sequence(metadata.get("states"), name="state_metadata.states")
    if not state_entries:
        raise ValueError("state_metadata.states must be non-empty")
    states: dict[str, Mapping[str, Any]] = {}
    total_eligible = 0
    for raw_state in state_entries:
        state = _as_mapping(raw_state, name="state_metadata.states[]")
        state_key = str(state.get("state_key", ""))
        if not state_key:
            raise ValueError("state_metadata state_key must be non-empty")
        if state_key in states:
            raise ValueError(f"duplicate state_key in state_metadata: {state_key}")
        logical_shape = _as_sequence(state.get("logical_shape"), name=f"{state_key}.logical_shape")
        shape_numel = _shape_numel(logical_shape, state_key=state_key)
        state_eligible = int(state.get("eligible_weight_count", -1))
        if state_eligible != shape_numel:
            raise ValueError(
                f"state {state_key!r} eligible_weight_count must match logical_shape numel"
            )
        if not str(state.get("state_layout_sha256", "")):
            raise ValueError(f"state {state_key!r} must include state_layout_sha256")
        states[state_key] = state
        total_eligible += state_eligible

    if total_eligible != int(eligible_weight_count):
        raise ValueError(
            "state_metadata eligible_weight_count must match timeline eligible_weight_count; "
            f"state={total_eligible} timeline={eligible_weight_count}"
        )

    step_hashes = _as_mapping(
        metadata.get("step_state_layout_sha256"),
        name="state_metadata.step_state_layout_sha256",
    )
    for step in timeline:
        step_key = str(int(step.step))
        per_step = _as_mapping(
            step_hashes.get(step_key),
            name=f"state_metadata.step_state_layout_sha256[{step_key}]",
        )
        for state_key, state in states.items():
            if per_step.get(state_key) != state.get("state_layout_sha256"):
                raise ValueError(
                    "state_metadata step_state_layout_sha256 drift or missing layout hash for "
                    f"step={step_key} state_key={state_key}"
                )

    for state_key, flat_index in _all_identities(timeline, dense, sparse):
        if state_key not in states:
            raise ValueError(f"identity state_key {state_key!r} is missing from state_metadata")
        limit = int(states[state_key]["eligible_weight_count"])
        if int(flat_index) >= limit:
            raise ValueError(
                f"identity flat_index out of range for state_key {state_key!r}: "
                f"{flat_index} >= {limit}"
            )
    return states


def _paths_match(dense: FrontCDecisionPath, sparse: FrontCDecisionPath) -> bool:
    return (
        dict(dense.q_flip_directions) == dict(sparse.q_flip_directions)
        and tuple(dense.accepted_under_global_cap_keys) == tuple(sparse.accepted_under_global_cap_keys)
        and tuple(dense.deferred_under_global_cap_keys) == tuple(sparse.deferred_under_global_cap_keys)
        and tuple(dense.backlog_keys) == tuple(sparse.backlog_keys)
        and tuple(dense.replay_veto_decision_keys) == tuple(sparse.replay_veto_decision_keys)
    )


def _validate_decision_path_derivation(
    payload: Mapping[str, Any],
    *,
    dense: FrontCDecisionPath,
    sparse: FrontCDecisionPath,
    claimed_front_c_viable: bool,
) -> tuple[bool, bool]:
    derivation = _as_mapping(
        payload.get("decision_path_derivation"),
        name="decision_path_derivation",
    )
    artifact_class = str(derivation.get("artifact_class", ""))
    synthetic = artifact_class == FRONT_C_SYNTHETIC_FIXTURE_ARTIFACT
    independent = bool(derivation.get("independent_sparse_derivation", False))

    if synthetic:
        if claimed_front_c_viable:
            raise ValueError("synthetic Front-C fixtures cannot claim live viability")
        if _paths_match(dense, sparse) and not bool(derivation.get("synthetic_fixture_non_claim")):
            raise ValueError(
                "synthetic dense==sparse fixtures must be explicitly labeled as non-claim"
            )
        return synthetic, independent

    if artifact_class != FRONT_C_RUN_DERIVED_ARTIFACT:
        raise ValueError("decision_path_derivation artifact_class must be synthetic_fixture or run_derived")
    if not independent:
        raise ValueError("decision_path_derivation must prove independent_sparse_derivation")
    if derivation.get("dense_source") != FRONT_C_DENSE_DECISION_SOURCE:
        raise ValueError("decision_path_derivation dense_source is not the dense int16 reference")
    if derivation.get("sparse_source") != FRONT_C_SPARSE_DECISION_SOURCE:
        raise ValueError("decision_path_derivation sparse_source is not the Front-C sparse reference")
    if not str(derivation.get("source_artifact_id", "")):
        raise ValueError("decision_path_derivation must include source_artifact_id")
    state_hash = str(
        _as_mapping(payload["state_metadata"], name="state_metadata").get(
            "state_layout_metadata_sha256",
            "",
        )
    )
    if derivation.get("state_layout_metadata_sha256") != state_hash:
        raise ValueError(
            "decision_path_derivation state_layout_metadata_sha256 must match state_metadata"
        )
    if _paths_match(dense, sparse) and not independent:
        raise ValueError("live dense==sparse paths require independent derivation receipt")
    return synthetic, independent


def validate_front_c_identity_artifact(
    payload: Mapping[str, Any],
    *,
    q_ledger_row: Base3QEntropyLedgerRow | None = None,
    claimed_front_c_viable: bool = False,
) -> FrontCIdentityArtifactValidation:
    """Validate a compact Front-C identity artifact before the scaffold consumes it."""

    payload = _as_mapping(payload, name="payload")
    timeline = _normalize_timeline(payload)
    dense, sparse = _normalize_paths(payload)
    eligible_values = {int(step.eligible_weight_count) for step in timeline}
    if len(eligible_values) != 1:
        raise ValueError("timeline eligible_weight_count must be stable across rows")
    eligible = next(iter(eligible_values))
    q_row = q_ledger_row if q_ledger_row is not None else base3_q_ledger_from_front_c_artifact(payload)
    if int(q_row.eligible_weight_count) != int(eligible):
        raise ValueError(
            "q ledger eligible weight count must match timeline eligible_weight_count; "
            f"q={q_row.eligible_weight_count} timeline={eligible}"
        )
    states = _validate_state_metadata(
        payload,
        timeline=timeline,
        dense=dense,
        sparse=sparse,
        eligible_weight_count=eligible,
    )
    synthetic, independent = _validate_decision_path_derivation(
        payload,
        dense=dense,
        sparse=sparse,
        claimed_front_c_viable=claimed_front_c_viable,
    )
    return FrontCIdentityArtifactValidation(
        schema_version=FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION,
        projection_schema_version=FRONT_C_SCHEMA_VERSION,
        status=FRONT_C_IDENTITY_EXTRACTABLE,
        timeline_step_count=len(timeline),
        eligible_weight_count=eligible,
        q_ledger_eligible_weight_count=int(q_row.eligible_weight_count),
        state_key_count=len(states),
        synthetic_fixture=synthetic,
        independent_sparse_derivation=independent,
        claimed_front_c_viable=bool(claimed_front_c_viable),
        non_claims=(
            "CPU/static identity adapter only; no live emission run",
            "counts/hashes are diagnostics only and cannot compute union/churn",
            "Front-C viability requires run-derived identities and independent sparse path derivation",
        ),
    )


def front_c_report_from_identity_artifact(
    payload: Mapping[str, Any],
    *,
    q_ledger_row: Base3QEntropyLedgerRow | None = None,
    claimed_front_c_viable: bool = False,
) -> FrontCProjectionReport:
    """Validate then hand a future identity artifact to the locked Front-C harness."""

    validation = validate_front_c_identity_artifact(
        payload,
        q_ledger_row=q_ledger_row,
        claimed_front_c_viable=claimed_front_c_viable,
    )
    q_row = q_ledger_row if q_ledger_row is not None else base3_q_ledger_from_front_c_artifact(payload)
    report = front_c_report_from_mapping(payload, q_ledger_row=q_row)
    validate_front_c_projection_report(report, claimed_front_c_viable=claimed_front_c_viable)
    if claimed_front_c_viable and validation.synthetic_fixture:
        raise ValueError("synthetic identity artifacts cannot claim Front-C viability")
    return report


def _direction_map(values: Any, *, label: str) -> dict[tuple[str, int], int]:
    return dict(FrontCDecisionPath(label=label, q_flip_directions=values).q_flip_directions)


def _identity_tuple(values: Any, *, label: str) -> tuple[tuple[str, int], ...]:
    return FrontCDecisionPath(label=label, accepted_under_global_cap_keys=values).accepted_under_global_cap_keys


def classify_front_c_conflict_overlap(
    *,
    target_helping_q_directions: Any,
    prior_serving_q_directions: Any,
    accepted_prior_harming_keys: Any = (),
) -> FrontCConflictOverlapDiagnostic:
    """Classify conflict-vs-coupling as a diagnostic-only readout."""

    target = _direction_map(target_helping_q_directions, label="target_helping")
    prior = _direction_map(prior_serving_q_directions, label="prior_serving")
    accepted_prior_harming = _identity_tuple(
        accepted_prior_harming_keys,
        label="accepted_prior_harming",
    )
    common = sorted(set(target) & set(prior))
    opposite = tuple(identity for identity in common if int(target[identity]) != int(prior[identity]))
    same = tuple(identity for identity in common if int(target[identity]) == int(prior[identity]))
    if opposite:
        classification = "representational_conflict_or_isolation"
    elif accepted_prior_harming:
        classification = "selection_or_cap_pressure"
    else:
        classification = "no_conflict_signal"
    return FrontCConflictOverlapDiagnostic(
        schema_version=FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION,
        label="front_c_conflict_vs_coupling_overlap_diagnostic",
        target_helping_count=len(target),
        prior_serving_count=len(prior),
        same_identity_opposite_direction_overlap_count=len(opposite),
        same_identity_same_direction_overlap_count=len(same),
        accepted_prior_harming_count=len(accepted_prior_harming),
        classification=classification,
        diagnostic_only=True,
        same_identity_opposite_direction_overlap_keys=tuple(_identity_dict(item) for item in opposite),
    )


__all__ = [
    "FRONT_C_AMBIGUOUS_SPLIT_CONTRACT",
    "FRONT_C_CANONICAL_STATE_KEY_SEMANTICS",
    "FRONT_C_COUNT_ONLY",
    "FRONT_C_DENSE_DECISION_SOURCE",
    "FRONT_C_IDENTITY_EMITTER_SCHEMA_VERSION",
    "FRONT_C_IDENTITY_EXTRACTABLE",
    "FRONT_C_LOCAL_FLAT_INDEX_SEMANTICS",
    "FRONT_C_PATH_B_CARRY_FORWARD_FOLDS",
    "FRONT_C_RUN_DERIVED_ARTIFACT",
    "FRONT_C_SPARSE_DECISION_SOURCE",
    "FRONT_C_STATE_LAYOUT_HASH_SEMANTICS",
    "FRONT_C_SYNTHETIC_FIXTURE_ARTIFACT",
    "FrontCArtifactInventoryReport",
    "FrontCConflictOverlapDiagnostic",
    "FrontCIdentityArtifactValidation",
    "classify_front_c_conflict_overlap",
    "classify_front_c_identity_payload",
    "classify_front_c_saved_audit_root",
    "front_c_report_from_identity_artifact",
    "require_front_c_identity_extractable_saved_audit_root",
    "validate_front_c_identity_artifact",
]
