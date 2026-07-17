"""Packet/source pin validation for fixed-state signed-utility diagnostic (PLAN v5)."""
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path
from typing import Any, Mapping

WATCH_WRAP_HRM158_SHA256 = "a19f1c5fe88fb3dcbf00ab442047576708f75272210e9a0cc94ed9369bf45d4b"
WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256 = "ba54e8dde1c7948d6733c5bcce77d7ae8e6b3c9102935d31fe70f01d784aee4d"
FORMAL_SOURCE_PIN_BASENAMES = (
    "signed_utility_fixed_state_phase_telemetry.py",
    "signed_utility_fixed_state_integrity_proofs.py",
    "signed_utility_fixed_state_partition_leakage.py",
    "signed_utility_fixed_state_arm_proofs.py",
    "signed_utility_fixed_state_legal_subset.py",
    "signed_utility_fixed_state_eval_contract.py",
    "signed_utility_fixed_state_authoritative_gpu.py",
    "signed_utility_fixed_state_creditdir_import_facade.py",
    "signed_utility_fixed_state_driver.py",
    "signed_utility_fixed_state_facade.py",
    "signed_utility_fixed_state_schema.py",
    "signed_utility_fixed_state_pin_validation.py",
    "signed_utility_fixed_state_reducers.py",
)


class PinValidationError(RuntimeError):
    pass


def rehash_path(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require_head_equals_upstream_pin(repo_root: str | Path, expected_head: str) -> str:
    root = Path(repo_root)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    upstream = subprocess.check_output(
        ["git", "rev-parse", "@{u}"], cwd=str(root), text=True
    ).strip()
    if head != expected_head:
        raise PinValidationError(f"head_mismatch:{head}!={expected_head}")
    if head != upstream:
        raise PinValidationError(f"head_ne_upstream:{head}!={upstream}")
    return head


def validate_proof_packet_source_pins(packet: Mapping[str, Any]) -> dict[str, str]:
    pins = packet.get("source_pins")
    if not isinstance(pins, Mapping) or not pins:
        raise PinValidationError("source_pins_missing")
    observed: dict[str, str] = {}
    for key, pin in pins.items():
        if key == "head":
            continue
        if not isinstance(pin, Mapping):
            raise PinValidationError(f"pin_not_mapping:{key}")
        if "absolute_path" not in pin or "sha256" not in pin:
            raise PinValidationError(f"pin_requires_absolute_path_and_sha256:{key}")
        path = Path(str(pin["absolute_path"]))
        if not path.is_file():
            raise PinValidationError(f"pin_path_missing:{key}:{path}")
        digest = rehash_path(path)
        expected = str(pin["sha256"])
        if digest != expected:
            raise PinValidationError(f"pin_sha_mismatch:{key}:{digest}!={expected}")
        observed[key] = digest
        if key in {"watch_wrap", "bin/watch-wrap", "watch-wrap"} or path.name == "watch-wrap":
            if digest == WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256:
                raise PinValidationError("watch_wrap_must_be_hrm158_a19f1c5f_not_claw_code_ba54e8dd")
            if digest != WATCH_WRAP_HRM158_SHA256:
                raise PinValidationError(f"watch_wrap_unexpected_sha:{digest}")
    return observed


def require_formal_source_pin_basenames(packet: Mapping[str, Any]) -> list[str]:
    """Fail closed unless formal packet pins include the twelve formal source-pin basenames."""
    pins = packet.get("source_pins")
    if not isinstance(pins, Mapping) or not pins:
        raise PinValidationError("source_pins_missing")
    observed = []
    for pin in pins.values():
        if not isinstance(pin, Mapping) or "absolute_path" not in pin:
            continue
        observed.append(Path(str(pin["absolute_path"])).name)
    missing = [b for b in FORMAL_SOURCE_PIN_BASENAMES if b not in observed]
    if missing:
        raise PinValidationError(f"formal_source_pins_missing:{missing}")
    return list(FORMAL_SOURCE_PIN_BASENAMES)


__all__ = [
    "FORMAL_SOURCE_PIN_BASENAMES",
    "PinValidationError",
    "WATCH_WRAP_CLAW_CODE_FORBIDDEN_SHA256",
    "WATCH_WRAP_HRM158_SHA256",
    "rehash_path",
    "require_formal_source_pin_basenames",
    "require_head_equals_upstream_pin",
    "validate_proof_packet_source_pins",
]
