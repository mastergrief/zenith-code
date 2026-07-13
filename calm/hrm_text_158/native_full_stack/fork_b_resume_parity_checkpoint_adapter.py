"""Fork B REAL checkpoint roundtrip + path-disclosure adapter.

SOLE owner of trainer_sub2_authority save→disk→load and path disclosure.
Isolated from pure reducers/classifier.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import (
    BoundedDeltaTensorState,
)

PATH_CLASS_REAL = "REAL_on_disk_trainer_sub2_authority_save_load"
PATH_CLASS_IN_MEMORY_UNINTERRUPTED = "in_memory_uninterrupted"
PATH_CLASS_IN_MEMORY_FULL_STATE = "in_memory_test_only_full_state"
PATH_CLASS_IN_MEMORY_ZEROS = "in_memory_zeros_injection"


def emit_fork_b_cli_scaffold_receipt(
    *,
    arm: Any,
    cut_t: int,
    artifact_dir: Any,
    parent_receipt_keys: Sequence[Any],
    schema: str | None = None,
) -> dict[str, Any]:
    """Thin CLI scaffold write; no science label. Lives with adapter (IO owner)."""

    from calm.hrm_text_158.native_full_stack.fork_b_resume_parity_contracts import (
        CERTIFICATE_SCHEMA,
        DENSE_SHADOW_FIELD_PERSISTENT_BPW,
        SCHEMA_ID,
        ArmId,
        PreScienceClass,
    )

    out_dir = Path(artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    schema_id = SCHEMA_ID if schema is None else str(schema)
    receipt: dict[str, Any] = {
        "schema": CERTIFICATE_SCHEMA,
        "non_target_snapshot_schema": schema_id,
        "arm": arm.value if isinstance(arm, ArmId) else str(arm),
        "cut_t": int(cut_t),
        "run_local_test_evidence_only": True,
        "is_checkpoint_authority": False,
        "contributes_persistent_bpw": False,
        "science_label": None,
        "pre_science": PreScienceClass.MISSING_OBSERVABLE.value,
        "note": (
            "CLI scaffold receipt — developer wiring only; reduced 5-arm GPU smoke "
            "is the focused validation; formal science requires later +1 launch"
        ),
        "parent_receipt_key_count": len(tuple(parent_receipt_keys)),
        "dense_shadow_field_persistent_bpw": DENSE_SHADOW_FIELD_PERSISTENT_BPW,
    }
    path = out_dir / f"fork_b_arm_{receipt['arm']}_cut_{int(cut_t)}_scaffold.json"
    path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    receipt["artifact_path"] = str(path)
    return receipt


def real_trainer_sub2_authority_checkpoint_roundtrip(
    *,
    model: torch.nn.Module,
    eligible_modules: Mapping[str, Any],
    tensor_states: Mapping[str, BoundedDeltaTensorState],
    checkpoint_path: Path | str,
    step: int = 0,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    """REAL 2C4a save→disk→load via trainer_sub2_authority (NOT in-memory strip).

    Uses ``build_trainer_sub2_authority_checkpoint_blob`` + ``torch.save`` then
    ``torch.load`` + ``load_trainer_sub2_authority_checkpoint_blob``. Proves the
    dense shadow is stripped by the authority path (shadow is None on load).
    Returns loaded states plus on-disk digests for smoke/receipt disclosure.
    """

    from calm.hrm_text_158.native_full_stack.trainer_sub2_authority import (
        build_trainer_sub2_authority_checkpoint_blob,
        load_trainer_sub2_authority_checkpoint_blob,
    )

    path = Path(checkpoint_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()

    pre_shadow_presence = {
        str(key): state.exact_accumulator_shadow is not None
        for key, state in sorted(tensor_states.items())
    }
    blob = build_trainer_sub2_authority_checkpoint_blob(
        model,
        eligible_modules=eligible_modules,
        tensor_states=tensor_states,
        step=int(step),
    )
    sidecar = dict(blob.get("trainer_sub2_authority") or {})
    if bool(sidecar.get("dense_int16_persistent_accumulator_saved")):
        raise RuntimeError("authority blob must not persist dense int16 accumulators")
    tensor_payloads = dict(sidecar.get("tensor_payloads") or {})
    for key, payload in tensor_payloads.items():
        if bool(payload.get("exact_accumulator_shadow_saved")):
            raise RuntimeError(
                f"authority blob must not save exact_accumulator_shadow for {key}"
            )

    torch.save(blob, path)
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"REAL checkpoint write failed: {path}")
    on_disk_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    on_disk_bytes = int(path.stat().st_size)

    loaded_blob = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(loaded_blob, dict):
        raise RuntimeError("loaded checkpoint is not a dict blob")
    loaded_states = load_trainer_sub2_authority_checkpoint_blob(
        model,
        loaded_blob,
        eligible_modules=eligible_modules,
        device=device,
    )
    post_shadow_presence = {
        str(key): state.exact_accumulator_shadow is not None
        for key, state in sorted(loaded_states.items())
    }
    if any(post_shadow_presence.values()):
        raise RuntimeError(
            "REAL authority load must strip exact_accumulator_shadow "
            f"(got presence={post_shadow_presence})"
        )
    return {
        "path_class": PATH_CLASS_REAL,
        "simulated": False,
        "checkpoint_path": str(path),
        "on_disk_sha256": on_disk_sha256,
        "on_disk_bytes": on_disk_bytes,
        "blob_sha256": str(blob.get("checkpoint_blob_sha256") or ""),
        "sidecar_sha256": str(sidecar.get("authoritative_state_payload_sha256") or ""),
        "pre_save_shadow_present": pre_shadow_presence,
        "post_load_shadow_present": post_shadow_presence,
        "dense_int16_persistent_accumulator_saved": bool(
            sidecar.get("dense_int16_persistent_accumulator_saved")
        ),
        "loaded_states": loaded_states,
        "loaded_blob": loaded_blob,
    }


def build_path_disclosure_record(
    arm_id: str,
    *,
    roundtrip_result: Mapping[str, Any] | None = None,
    checkpoint_roundtrip: bool | None = None,
    path_class: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build per-arm path-disclosure record (adapter-owned; not a science label)."""

    if roundtrip_result is not None:
        record: dict[str, Any] = {
            "arm": str(arm_id),
            "path_class": str(roundtrip_result.get("path_class") or PATH_CLASS_REAL),
            "simulated": bool(roundtrip_result.get("simulated", False)),
            "checkpoint_roundtrip": True,
            "on_disk_sha256": roundtrip_result.get("on_disk_sha256"),
            "on_disk_bytes": roundtrip_result.get("on_disk_bytes"),
            "checkpoint_path": roundtrip_result.get("checkpoint_path"),
            "post_load_shadow_stripped": not any(
                (roundtrip_result.get("post_load_shadow_present") or {}).values()
            ),
            "dense_int16_persistent_accumulator_saved": bool(
                roundtrip_result.get("dense_int16_persistent_accumulator_saved")
            ),
        }
    else:
        record = {
            "arm": str(arm_id),
            "path_class": str(path_class or PATH_CLASS_IN_MEMORY_UNINTERRUPTED),
            "simulated": False,
            "checkpoint_roundtrip": bool(checkpoint_roundtrip)
            if checkpoint_roundtrip is not None
            else False,
        }
    if extra:
        record.update(dict(extra))
    return record
