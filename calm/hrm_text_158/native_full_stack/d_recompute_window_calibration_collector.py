"""Session-boundary collector for D recompute-window calibration warmup observations."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import torch

from calm.hrm_text_158.native_full_stack.d_recompute_window_emit import (
    ReplayConstants,
    _shadow_numel,
)
from calm.hrm_text_158.native_full_stack.d_recompute_window_stratified_selector import (
    CalibrationWarmupStep,
    MAX_REPRESENTATIVE_KEYS,
    MIN_REPRESENTATIVE_KEYS,
    StratifiedSelectorManifest,
    build_stratified_selector_manifest,
    extract_calibration_observation,
    select_instrumentation_state_keys_from_manifest,
)

CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA = (
    "hrm_text_158_d_recompute_calibration_warmup_observations/v0"
)
CALIBRATION_SURFACE_REPRESENTATIVE = "representative_stratified_v1"

DEFAULT_REPRESENTATIVE_MANIFEST_SPEC: dict[str, Any] = {
    "calibration_surface": CALIBRATION_SURFACE_REPRESENTATIVE,
    "min_keys": MIN_REPRESENTATIVE_KEYS,
    "max_keys": MAX_REPRESENTATIVE_KEYS,
}


def parent_checkpoint_sha256(parent_path: Path | str) -> str:
    target = Path(parent_path)
    return hashlib.sha256(target.read_bytes()).hexdigest()


class CalibrationWarmupCollector:
    """Accumulates per-step calibration observations; writes once at session end."""

    def __init__(
        self,
        *,
        output_path: Path,
        pre_warmup_parent_sha256: str,
        policy: Mapping[str, Any],
        representative_manifest_spec: Mapping[str, Any] | None = None,
    ) -> None:
        self.output_path = Path(output_path)
        self.pre_warmup_parent_sha256 = str(pre_warmup_parent_sha256)
        self.policy = dict(policy)
        self._representative_manifest_spec = {
            **DEFAULT_REPRESENTATIVE_MANIFEST_SPEC,
            **dict(representative_manifest_spec or {}),
        }
        self._representative_manifest: StratifiedSelectorManifest | None = None
        self._steps: list[CalibrationWarmupStep] = []
        self._numel_by_key: dict[str, int] = {}

    def _resolve_representative_state_keys(
        self,
        pre_update_states: Mapping[str, Any],
        *,
        replay_constants: ReplayConstants,
    ) -> list[str]:
        if self._representative_manifest is None:
            manifest_spec = {
                **self._representative_manifest_spec,
                "calibration_policy": self.policy,
            }
            self._representative_manifest = build_stratified_selector_manifest(
                pre_update_states,
                manifest_spec=manifest_spec,
                replay_constants=replay_constants,
            )
        return select_instrumentation_state_keys_from_manifest(
            pre_update_states,
            self._representative_manifest,
        )

    def record_step(
        self,
        *,
        step: int,
        pre_update_states: Mapping[str, Any],
        votes_by_key: Mapping[str, torch.Tensor],
        replay_constants: ReplayConstants,
    ) -> None:
        observations: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {}
        for state_key in self._resolve_representative_state_keys(
            pre_update_states,
            replay_constants=replay_constants,
        ):
            state = pre_update_states[state_key]
            vote_tensor = votes_by_key[state_key]
            acc_values, vote_values = extract_calibration_observation(
                state_key,
                state,
                vote_tensor,
                replay_constants=replay_constants,
            )
            observations[state_key] = (acc_values, vote_values)
            self._numel_by_key[state_key] = int(_shadow_numel(state))
        if observations:
            self._steps.append(
                CalibrationWarmupStep(step=int(step), observations=observations)
            )

    def write(self) -> Path:
        if not self._steps:
            raise ValueError("calibration warmup collector has no recorded steps")
        if self._representative_manifest is None:
            raise ValueError(
                "calibration warmup collector missing representative manifest surface"
            )
        steps_payload = []
        for sample in self._steps:
            step_observations: dict[str, Any] = {}
            for state_key, (acc_values, vote_values) in sample.observations.items():
                step_observations[state_key] = {
                    "acc_values": list(acc_values),
                    "vote_values": list(vote_values),
                }
            steps_payload.append(
                {"step": int(sample.step), "observations": step_observations}
            )
        manifest = self._representative_manifest
        payload = {
            "schema_version": CALIBRATION_WARMUP_OBSERVATIONS_SCHEMA,
            "policy": self.policy,
            "pre_warmup_banked_state_sha256": self.pre_warmup_parent_sha256,
            "calibration_surface": {
                "surface_id": CALIBRATION_SURFACE_REPRESENTATIVE,
                "selected_key_count": int(manifest.selected_key_count),
                "coverage_tier": str(manifest.coverage_tier),
                "representative_manifest_sha256": str(manifest.manifest_sha256),
                "manifest_spec": dict(manifest.manifest_spec),
            },
            "tensor_state_numel_by_key": dict(sorted(self._numel_by_key.items())),
            "steps": steps_payload,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return self.output_path
