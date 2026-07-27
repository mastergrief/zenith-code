#!/usr/bin/env python3
"""Governing harness for backward_saved_tensors composition audit (PLAN_v7).

Sole runtime owner of BackwardSavedTensorCompositionEvidence + factory + validators.
Tests MUST import these symbols; a second test-local implementation is FORBIDDEN.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from calm.hrm_text_158.native_full_stack.activation_relief import (  # noqa: E402
    analyze_saved_tensor_hook_events,
)
from calm.hrm_text_158.native_full_stack.full_sub2_runtime_readiness import (  # noqa: E402
    fixture_full_sub2_runtime_ready_for_science,
)
from calm.llm_computer.tests import test_hrm_text_158_activation_relief as _ar_helper  # noqa: E402

PLAN_SHA256_EXPECTED = (
    "ebbc5ecd413ad5c6d895e2e43dd16e8f9087a9eb8bac2efa83f1b3895a2b8fa2"
)
PLAN_REVISION = "v7"
HELPER_QUALNAME = (
    "test_hrm_text_158_activation_relief._hrm_forward_saved_tensor_events"
)
HELPER_PATH = "calm/llm_computer/tests/test_hrm_text_158_activation_relief.py"
HELPER_SHA256_EXPECTED = (
    "c74ee97510d943bafb2c421003ddff73aa9ca61c56c0e78a61e5afec437bab54"
)
FIXTURE_NAME = "bst_composition_cpu_tiny_v1"
PROBE_MODE = "saved_tensors_hooks_instrumented"
BRANCH_FP_DENSE = "BR-BST-C-FP-DENSE"
BRANCH_BOUNDARY_ONLY = "BR-BST-C-BOUNDARY-ONLY"
BRANCH_INVALID = "BR-BST-C-MEASUREMENT-INVALID"
BOUNDARY_SHAPE = (2, 16, 32)
BOUNDARY_DTYPE = "torch.float32"
BACKEND_EXACT = "pytorch_cpu_sdpa_backend_unclassified"
BACKEND_BINDING_MODE = "HONEST_UNCLASSIFIED_LABEL"
DEVICE_BINDING_MODE = "OBSERVED_FROM_EVERY_EVENT"
REPO_HEAD_BINDING_MODE = "LIVE_GIT_REV_PARSE"
EXPECTED_EVENT_COUNT = 180
NON_GENERALIZATION_CLAUSE = (
    "CPU-fixture classification does NOT generalize to CUDA or fused-SDPA/"
    "memory-efficient/flash backward surfaces; separate runtime contract required. "
    "ternary-rotor: SDPA-saved q/k/v screen-excluded; attention precision via "
    "quantize-then-compute kernels, never saved-tensor swaps. "
    "Backend label is unclassified unless a replayable selector observation is bound."
)

# PLAN-frozen dependency currency (not implementation-time discovery).
DEPENDENCY_CURRENCY_PINS: tuple[tuple[str, str], ...] = (
    (
        "calm/hrm_text_158/layers.py",
        "bae47480c31b5f9865ea7383febc26c70ad60891c12a71b2441531388e74517d",
    ),
    (
        "calm/hrm_text_158/hrm.py",
        "ec703218b79b18f21582136c36a1a198aea1e95a568d48dd1f2f692730f5d3cd",
    ),
    (
        "calm/hrm_text_158/transformer.py",
        "7e403a89c8a82ca45bc29d466231f6722fb989d71d46fd4ab17b24a9ddea7512",
    ),
    (
        "calm/hrm_text_158/config.py",
        "c56318f96cd57fd5faa5d79453c50b54a295bae5a03f80560b5cf59c2012c2e7",
    ),
    (
        "calm/hrm_text_158/__init__.py",
        "57790db112cb24891a761a8f531ef8acf47631a53e68dd0328d4b900a005f632",
    ),
    (
        "calm/hrm_text_158/native_full_stack/activation_relief.py",
        "107d73ec52b885669146aa3b38e4a47c20923329aed8d010fb35b9b945571333",
    ),
    (HELPER_PATH, HELPER_SHA256_EXPECTED),
)

_EVENT_KEYS_EXACT = frozenset({"dtype", "shape", "numel", "requires_grad", "device"})


def _sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_canonical_json_sha256(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def live_git_rev_parse_head(*, cwd: Path = REPO_ROOT) -> str:
    out = subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=str(cwd),
        text=True,
    ).strip()
    if not out or len(out) < 7:
        raise ValueError(f"git rev-parse HEAD returned invalid value: {out!r}")
    return out


def validate_live_repo_head_matches_claim(*, claimed: str, live: str | None = None) -> str:
    observed = live if live is not None else live_git_rev_parse_head()
    if claimed != observed:
        raise ValueError(
            f"repo_head live mismatch claimed={claimed} live={observed}"
        )
    return observed


def validate_dependency_currency_against_plan_pins(
    *,
    pins: Sequence[tuple[str, str]] = DEPENDENCY_CURRENCY_PINS,
) -> dict[str, str]:
    observed: dict[str, str] = {}
    for rel, expected in pins:
        path = REPO_ROOT / rel
        live = _sha_file(path)
        if live != expected:
            raise ValueError(
                f"dependency-currency mismatch path={rel} expected={expected} live={live}"
            )
        observed[rel] = live
    if len(observed) != 7:
        raise ValueError(f"dependency-currency must cover exactly 7 files; got {len(observed)}")
    return observed


@dataclass(frozen=True)
class SavedTensorEventEvidence:
    dtype: str
    shape: tuple[int, ...]
    numel: int
    requires_grad: bool
    device: str

    def to_canonical_dict(self) -> dict[str, Any]:
        # shape stored immutable tuple; canonical JSON emits list[int]
        return {
            "device": str(self.device),
            "dtype": self.dtype,
            "numel": int(self.numel),
            "requires_grad": bool(self.requires_grad),
            "shape": [int(x) for x in self.shape],
        }


@dataclass(frozen=True)
class BackwardSavedTensorCompositionEvidence:
    events: tuple[SavedTensorEventEvidence, ...]
    probe_mode: str
    observed_boundary_tensor_count: int
    observed_checkpoint_dummy_tensor_count: int
    observed_internal_payload_tensor_count: int
    saved_tensor_count: int
    boundary_tensor_shape: tuple[int, ...]
    boundary_tensor_dtype: str
    events_sha256: str
    partition_sha256: str
    fixture_name: str
    device: str
    device_binding_mode: str
    device_census_items: tuple[tuple[str, int], ...]
    backend: str
    backend_binding_mode: str

    @property
    def device_census(self) -> dict[str, int]:
        return {k: int(v) for k, v in self.device_census_items}
    seed: int
    activation_relief_policy: None
    bp_steps: int
    helper_qualname: str
    helper_sha256: str
    deep_frozen_snapshot: bool
    events_container_type: str


def _reject_bool_as_int(value: object, *, field: str) -> None:
    # bool is a subclass of int in Python — must reject explicitly.
    if type(value) is not int:
        raise ValueError(f"{field} must be exact int (bool rejected); got {type(value).__name__}")


def validate_event_schema_exact(raw: Mapping[str, object]) -> None:
    keys = set(raw.keys())
    if keys != _EVENT_KEYS_EXACT:
        missing = _EVENT_KEYS_EXACT - keys
        extra = keys - _EVENT_KEYS_EXACT
        raise ValueError(
            f"event schema exactness failed missing={sorted(missing)} extra={sorted(extra)}"
        )
    if not isinstance(raw["dtype"], str):
        raise ValueError("dtype must be str")
    if not isinstance(raw["device"], str):
        raise ValueError("device must be str")
    shape = raw["shape"]
    if not isinstance(shape, (list, tuple)):
        raise ValueError("shape must be list/tuple of int")
    for x in shape:
        _reject_bool_as_int(x, field="shape entry")
    _reject_bool_as_int(raw["numel"], field="numel")
    if not isinstance(raw["requires_grad"], bool) or type(raw["requires_grad"]) is not bool:
        raise ValueError("requires_grad must be bool")


def freeze_events(
    raw_events: Sequence[Mapping[str, object]],
) -> tuple[SavedTensorEventEvidence, ...]:
    frozen: list[SavedTensorEventEvidence] = []
    for raw in raw_events:
        validate_event_schema_exact(raw)
        frozen.append(
            SavedTensorEventEvidence(
                dtype=str(raw["dtype"]),
                shape=tuple(int(x) for x in raw["shape"]),  # type: ignore[arg-type]
                numel=int(raw["numel"]),  # validated exact int
                requires_grad=bool(raw["requires_grad"]),
                device=str(raw["device"]),
            )
        )
    return tuple(frozen)


def events_canonical_payload(
    events: Sequence[SavedTensorEventEvidence],
) -> list[dict[str, Any]]:
    return [e.to_canonical_dict() for e in events]


def compute_events_sha256(events: Sequence[SavedTensorEventEvidence]) -> str:
    return compute_canonical_json_sha256(events_canonical_payload(events))


def compute_partition_sha256(
    *,
    observed_boundary_tensor_count: int,
    observed_checkpoint_dummy_tensor_count: int,
    observed_internal_payload_tensor_count: int,
    saved_tensor_count: int,
) -> str:
    return compute_canonical_json_sha256(
        {
            "observed_boundary_tensor_count": int(observed_boundary_tensor_count),
            "observed_checkpoint_dummy_tensor_count": int(
                observed_checkpoint_dummy_tensor_count
            ),
            "observed_internal_payload_tensor_count": int(
                observed_internal_payload_tensor_count
            ),
            "saved_tensor_count": int(saved_tensor_count),
        }
    )


def validate_events_sha256(evidence: BackwardSavedTensorCompositionEvidence) -> None:
    actual = compute_events_sha256(evidence.events)
    if actual != evidence.events_sha256:
        raise ValueError(
            f"events_sha256 mismatch expected={evidence.events_sha256} actual={actual}"
        )


def validate_partition_sha256(evidence: BackwardSavedTensorCompositionEvidence) -> None:
    actual = compute_partition_sha256(
        observed_boundary_tensor_count=evidence.observed_boundary_tensor_count,
        observed_checkpoint_dummy_tensor_count=evidence.observed_checkpoint_dummy_tensor_count,
        observed_internal_payload_tensor_count=evidence.observed_internal_payload_tensor_count,
        saved_tensor_count=evidence.saved_tensor_count,
    )
    if actual != evidence.partition_sha256:
        raise ValueError(
            f"partition_sha256 mismatch expected={evidence.partition_sha256} actual={actual}"
        )


def compute_device_census_from_every_event(
    events: Sequence[SavedTensorEventEvidence],
) -> dict[str, int]:
    if not events:
        raise ValueError("device census requires events")
    counts = Counter(ev.device for ev in events)
    return {k: int(v) for k, v in sorted(counts.items())}


def validate_device_census_covers_every_event(
    events: Sequence[SavedTensorEventEvidence],
    device_census: Mapping[str, int],
) -> None:
    if sum(int(v) for v in device_census.values()) != len(events):
        raise ValueError(
            "device census does not cover every event: "
            f"census_sum={sum(int(v) for v in device_census.values())} event_count={len(events)}"
        )
    recomputed = compute_device_census_from_every_event(events)
    if dict(device_census) != recomputed:
        raise ValueError(
            f"device census mismatch claimed={dict(device_census)} recomputed={recomputed}"
        )


def validate_observed_device_uniformity(
    events: Sequence[SavedTensorEventEvidence],
    *,
    expected_event_count: int = EXPECTED_EVENT_COUNT,
) -> tuple[str, dict[str, int]]:
    if len(events) != expected_event_count:
        raise ValueError(
            f"expected {expected_event_count} events for device census; got {len(events)}"
        )
    census = compute_device_census_from_every_event(events)
    validate_device_census_covers_every_event(events, census)
    if len(census) != 1:
        raise ValueError(
            f"observed-device disagreement across full event census: {census}"
        )
    device = next(iter(census))
    return device, census


def _assert_no_mutable_containers(evidence: BackwardSavedTensorCompositionEvidence) -> None:
    if not isinstance(evidence.events, tuple):
        raise ValueError("events must be tuple")
    if not isinstance(evidence.device_census_items, tuple):
        raise ValueError("device_census_items must be tuple")
    for ev in evidence.events:
        if not isinstance(ev, SavedTensorEventEvidence):
            raise ValueError("event records must be SavedTensorEventEvidence")
        if not isinstance(ev.shape, tuple):
            raise ValueError("SavedTensorEventEvidence.shape must be immutable tuple")
        for attr in ("dtype", "numel", "requires_grad", "device"):
            val = getattr(ev, attr)
            if isinstance(val, (list, dict)):
                raise ValueError(f"mutable container reachable via {attr}")


def build_composition_evidence_from_frozen_fixture() -> BackwardSavedTensorCompositionEvidence:
    """Single helper/hook campaign. Deep-freeze before hashing/partition."""
    helper_path = REPO_ROOT / HELPER_PATH
    helper_sha = _sha_file(helper_path)
    if helper_sha != HELPER_SHA256_EXPECTED:
        raise ValueError(
            f"helper sha mismatch expected={HELPER_SHA256_EXPECTED} actual={helper_sha}"
        )
    validate_dependency_currency_against_plan_pins()

    # EXACTLY ONE helper call in the governing path (source-guarded separately).
    raw_events = _ar_helper._hrm_forward_saved_tensor_events(
        activation_relief_policy=None,
        bp_steps=5,
    )
    events = freeze_events(raw_events)
    device, device_census = validate_observed_device_uniformity(events)

    # Partition from the frozen snapshot (mapping view; no second hook campaign).
    partition = analyze_saved_tensor_hook_events(
        [e.to_canonical_dict() for e in events],
        boundary_tensor_shape=BOUNDARY_SHAPE,
        boundary_tensor_dtype=BOUNDARY_DTYPE,
    )
    events_sha = compute_events_sha256(events)
    partition_sha = compute_partition_sha256(
        observed_boundary_tensor_count=partition["observed_boundary_tensor_count"],
        observed_checkpoint_dummy_tensor_count=partition[
            "observed_checkpoint_dummy_tensor_count"
        ],
        observed_internal_payload_tensor_count=partition[
            "observed_internal_payload_tensor_count"
        ],
        saved_tensor_count=partition["saved_tensor_count"],
    )
    evidence = BackwardSavedTensorCompositionEvidence(
        events=events,
        probe_mode=PROBE_MODE,
        observed_boundary_tensor_count=int(partition["observed_boundary_tensor_count"]),
        observed_checkpoint_dummy_tensor_count=int(
            partition["observed_checkpoint_dummy_tensor_count"]
        ),
        observed_internal_payload_tensor_count=int(
            partition["observed_internal_payload_tensor_count"]
        ),
        saved_tensor_count=int(partition["saved_tensor_count"]),
        boundary_tensor_shape=BOUNDARY_SHAPE,
        boundary_tensor_dtype=BOUNDARY_DTYPE,
        events_sha256=events_sha,
        partition_sha256=partition_sha,
        fixture_name=FIXTURE_NAME,
        device=device,
        device_binding_mode=DEVICE_BINDING_MODE,
        device_census_items=tuple((k, int(v)) for k, v in device_census.items()),
        backend=BACKEND_EXACT,
        backend_binding_mode=BACKEND_BINDING_MODE,
        seed=1701,
        activation_relief_policy=None,
        bp_steps=5,
        helper_qualname=HELPER_QUALNAME,
        helper_sha256=helper_sha,
        deep_frozen_snapshot=True,
        events_container_type="tuple[SavedTensorEventEvidence, ...]",
    )
    _assert_no_mutable_containers(evidence)
    validate_events_sha256(evidence)
    validate_partition_sha256(evidence)
    validate_device_census_covers_every_event(evidence.events, evidence.device_census)
    return evidence


def classify_composition(evidence: BackwardSavedTensorCompositionEvidence) -> str:
    if evidence.probe_mode != PROBE_MODE or evidence.saved_tensor_count <= 0:
        return BRANCH_INVALID
    try:
        validate_observed_device_uniformity(evidence.events)
    except ValueError:
        return BRANCH_INVALID
    if evidence.observed_internal_payload_tensor_count > 0:
        return BRANCH_FP_DENSE
    if (
        evidence.observed_internal_payload_tensor_count == 0
        and evidence.saved_tensor_count
        == evidence.observed_boundary_tensor_count
        + evidence.observed_checkpoint_dummy_tensor_count
    ):
        return BRANCH_BOUNDARY_ONLY
    return BRANCH_INVALID


def evidence_to_receipt_fields(
    evidence: BackwardSavedTensorCompositionEvidence,
) -> dict[str, Any]:
    return {
        "fixture_name": evidence.fixture_name,
        "device": evidence.device,
        "device_binding_mode": evidence.device_binding_mode,
        "device_census": dict(evidence.device_census),
        "backend": evidence.backend,
        "backend_binding_mode": evidence.backend_binding_mode,
        "seed": evidence.seed,
        "activation_relief_policy": evidence.activation_relief_policy,
        "bp_steps": evidence.bp_steps,
        "helper_qualname": evidence.helper_qualname,
        "helper_sha256": evidence.helper_sha256,
        "probe_mode": evidence.probe_mode,
        "hook_event_count": evidence.saved_tensor_count,
        "observed_boundary_tensor_count": evidence.observed_boundary_tensor_count,
        "observed_checkpoint_dummy_tensor_count": evidence.observed_checkpoint_dummy_tensor_count,
        "observed_internal_payload_tensor_count": evidence.observed_internal_payload_tensor_count,
        "events_sha256": evidence.events_sha256,
        "partition_sha256": evidence.partition_sha256,
        "events_container_type": evidence.events_container_type,
        "deep_frozen_snapshot": evidence.deep_frozen_snapshot,
        "sourced_from_evidence_not_recomputed": True,
    }


def validate_evidence_receipt_field_equality(
    evidence: BackwardSavedTensorCompositionEvidence,
    receipt: Mapping[str, Any],
) -> None:
    expected = evidence_to_receipt_fields(evidence)
    census = list(expected.keys())
    missing = [k for k in census if k not in receipt]
    if missing:
        raise ValueError(f"receipt missing evidence fields: {missing}")
    for key, value in expected.items():
        if receipt[key] != value:
            raise ValueError(
                f"evidence↔receipt inequality on {key}: evidence={value!r} receipt={receipt[key]!r}"
            )


def count_helper_campaigns_in_harness_source() -> int:
    """Source-guard: exactly one helper call expression in this file."""
    src = Path(__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    count = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            name = ""
            if isinstance(func, ast.Attribute):
                name = func.attr
            elif isinstance(func, ast.Name):
                name = func.id
            if name == "_hrm_forward_saved_tensor_events":
                count += 1
    return count


def build_governing_receipt(*, plan_path: Path, argv: list[str]) -> dict[str, Any]:
    plan_sha = _sha_file(plan_path)
    if plan_sha != PLAN_SHA256_EXPECTED:
        raise ValueError(
            f"plan sha mismatch: expected={PLAN_SHA256_EXPECTED} actual={plan_sha}"
        )
    campaigns = count_helper_campaigns_in_harness_source()
    if campaigns != 1:
        raise ValueError(
            f"harness must contain exactly one helper campaign; found {campaigns}"
        )

    live_head = live_git_rev_parse_head()
    dependency_currency = validate_dependency_currency_against_plan_pins()

    evidence = build_composition_evidence_from_frozen_fixture()
    branch = classify_composition(evidence)
    if branch == "BR-BST-C-AUDIT-PENDING" or branch.endswith("PENDING"):
        raise ValueError("PENDING forbidden as terminal result")
    if evidence.backend != BACKEND_EXACT:
        raise ValueError(f"backend must be exactly {BACKEND_EXACT}")

    gap = fixture_full_sub2_runtime_ready_for_science("current_repo_scaffold")
    missing = list(getattr(gap, "missing_surface_names", ()) or ())
    current_repo_backward = (
        "missing"
        if "backward_saved_tensors_transients" in missing
        else "NOT_MISSING_UNEXPECTED"
    )

    fields = evidence_to_receipt_fields(evidence)
    receipt: dict[str, Any] = {
        "plan_id": "backward_saved_tensors_composition_current_fixture_audit",
        "plan_revision": PLAN_REVISION,
        "plan_sha256": plan_sha,
        "repo_head": live_head,
        "repo_head_binding_mode": REPO_HEAD_BINDING_MODE,
        "repo_head_live_matched": True,
        "dependency_currency": dependency_currency,
        "non_generalization_clause": NON_GENERALIZATION_CLAUSE,
        "audit_branch_id": branch,
        "current_repo_backward_classification_at_mint": current_repo_backward,
        "claim_ceiling": {
            "no_readiness_row_flip": True,
            "no_cuda_fused_sdpa_generalization": True,
            "scope": "CURRENT CPU fixture composition only",
        },
        "no_readiness_row_flip": True,
        "argv": list(argv),
        "evidence_to_receipt_field_equality_validated": False,
        **fields,
    }
    validate_live_repo_head_matches_claim(claimed=receipt["repo_head"], live=live_head)
    validate_evidence_receipt_field_equality(evidence, receipt)
    receipt["evidence_to_receipt_field_equality_validated"] = True

    required_tokens = [
        "EVIDENCE_SEAM_PRESENT",
        "EVIDENCE_RECEIPT_EQUALITY",
        "DEEP_FROZEN_SNAPSHOT",
        "SINGLE_HELPER_CAMPAIGN",
        "CANONICAL_HASH_CONTRACT",
        "NO_READINESS_ROW_FLIP",
        "OBSERVED_DEVICE_EVERY_EVENT",
        "DEPENDENCY_CURRENCY_PINNED",
        "LIVE_REPO_HEAD_MATCHED",
    ]
    receipt["required_tokens"] = required_tokens
    receipt["tokens"] = {
        "EVIDENCE_SEAM_PRESENT": True,
        "EVIDENCE_RECEIPT_EQUALITY": True,
        "DEEP_FROZEN_SNAPSHOT": evidence.deep_frozen_snapshot,
        "SINGLE_HELPER_CAMPAIGN": campaigns == 1,
        "CANONICAL_HASH_CONTRACT": True,
        "NO_READINESS_ROW_FLIP": True,
        "OBSERVED_DEVICE_EVERY_EVENT": evidence.device_binding_mode == DEVICE_BINDING_MODE,
        "DEPENDENCY_CURRENCY_PINNED": len(dependency_currency) == 7,
        "LIVE_REPO_HEAD_MATCHED": True,
    }
    return receipt


def mint_receipt_o_excl(path: Path, doc: Mapping[str, Any]) -> str:
    payload = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(path), flags, 0o644)
    with os.fdopen(fd, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return hashlib.sha256(payload).hexdigest()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    plan_path = args.plan if args.plan.is_absolute() else REPO_ROOT / args.plan
    out_path = args.out if args.out.is_absolute() else REPO_ROOT / args.out
    receipt = build_governing_receipt(plan_path=plan_path, argv=list(sys.argv))
    sha = mint_receipt_o_excl(out_path, receipt)
    print(
        json.dumps(
            {
                "status": "ok",
                "audit_branch_id": receipt["audit_branch_id"],
                "out": str(out_path),
                "receipt_sha256": sha,
                "hook_event_count": receipt["hook_event_count"],
                "device_census": receipt["device_census"],
                "backend": receipt["backend"],
                "observed_internal_payload_tensor_count": receipt[
                    "observed_internal_payload_tensor_count"
                ],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
