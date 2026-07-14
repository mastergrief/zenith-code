"""Run-arms live materialization facade (identity fail-closed + A-CFG).

Import-only reuse of probe helpers. No probe body rewrite.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping

from calm.hrm_text_158.native_full_stack.bounded_delta_learner import file_sha256
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
    ELIGIBLE_SCOPE,
    PARENT_SHA256_FULL,
)
from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_science_driver import (
    CadenceSaverFn,
    default_cadence_saver,
)

SCHEMA = "forgotten_accum_training_equivalence_materialization/v1"
SOURCE_CHECKPOINT = "checkpoint"
SOURCE_DEFAULT_APPLIED = "DEFAULT_APPLIED"

# Fields consumed by probe.model_config_from_checkpoint_config (probe.py:3791).
_MODEL_CONFIG_REQUIRED_KEYS = (
    "max_seq_len",
    "n_layers",
    "hidden_size",
    "num_heads",
    "expansion",
    "H_cycles",
    "L_cycles",
    "half_layers",
    "bp_warmup_ratio",
    "bp_min_steps",
    "bp_max_steps",
)
_MODEL_CONFIG_DEFAULTABLE = {
    "norm_type": "pre",
    "norm_eps": 1e-6,
    "rope_theta": 10000.0,
    "attn_type": "prefixlm",
    "init_type": "lecun_normal",
    "pos_emb_type": "rope",
    "use_ternary_bulk": False,
}


class IdentityRefuse(ValueError):
    """Identity fail-closed before model construction / runner / GPU."""

    def __init__(self, message: str):
        super().__init__(f"IDENTITY_REFUSE: {message}")


@dataclass
class RunArmsLiveBundle:
    model: Any
    tok: Any
    cfg: Any
    batch: Mapping[str, Any]
    tensor_states: Mapping[str, Any]
    eligible_modules: Mapping[str, Any]
    device: Any
    parent_path: Path
    parent_sha256_expected: str
    parent_sha256_pre_load: str
    parent_sha256_post_load: str
    identity_inventory: dict[str, Any]
    cadence_saver: CadenceSaverFn
    config: dict[str, Any]
    fidelity_report: dict[str, Any]


def inventory_model_config_field_provenance(
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    """Per-field provenance for every key consumed by model_config_from_checkpoint_config."""

    out: dict[str, dict[str, Any]] = {}
    for key in _MODEL_CONFIG_REQUIRED_KEYS:
        if key not in config:
            raise IdentityRefuse(f"required model config key missing: {key}")
        out[key] = {"value": config[key], "source": SOURCE_CHECKPOINT}
    for key, default in _MODEL_CONFIG_DEFAULTABLE.items():
        if key in config:
            out[key] = {"value": config[key], "source": SOURCE_CHECKPOINT}
        else:
            out[key] = {"value": default, "source": SOURCE_DEFAULT_APPLIED}
    return out


def assert_no_default_applied(provenance: Mapping[str, Mapping[str, Any]]) -> None:
    defaults = sorted(
        k for k, meta in provenance.items() if meta.get("source") == SOURCE_DEFAULT_APPLIED
    )
    if defaults:
        raise IdentityRefuse(
            "DEFAULT_APPLIED fields not permitted this slice: " + ",".join(defaults)
        )


def assert_use_ternary_bulk_true(config: Mapping[str, Any]) -> None:
    if "use_ternary_bulk" not in config:
        raise IdentityRefuse("use_ternary_bulk missing from checkpoint config")
    if config["use_ternary_bulk"] is not True:
        raise IdentityRefuse(
            f"use_ternary_bulk must be explicit True; got {config['use_ternary_bulk']!r}"
        )


def verify_parent_bytes_sha256(path: Path | str, expected_sha256: str) -> str:
    """Hash parent bytes BEFORE any deserialize. Returns actual sha."""

    actual = file_sha256(Path(path))
    if actual != str(expected_sha256):
        raise IdentityRefuse(
            f"parent sha256 mismatch pre-load: expected {expected_sha256}, got {actual}"
        )
    return actual


def load_verified_parent_checkpoint(
    path: Path | str, *, expected_sha256: str
) -> tuple[dict[str, Any], str, str]:
    """Hash-before-deserialize then load. Returns (ckpt, pre_sha, post_sha)."""

    from scripts.hrm_text_158_bounded_delta_acquisition_probe import load_parent_checkpoint

    path = Path(path)
    pre = verify_parent_bytes_sha256(path, expected_sha256)
    # load_parent_checkpoint re-hashes and deserializes; keep fail-closed contract.
    ckpt, loaded_sha = load_parent_checkpoint(path, expected_sha256=expected_sha256)
    post = file_sha256(path)
    if loaded_sha != expected_sha256 or post != expected_sha256 or pre != expected_sha256:
        raise IdentityRefuse(
            "parent sha256 pre/post law failed: "
            f"pre={pre} loaded={loaded_sha} post={post} expected={expected_sha256}"
        )
    return ckpt, pre, post


def derive_all_bitlinear_name_set(model: Any) -> set[str]:
    """Independent eligible-name derivation under frozen all-bitlinear contract."""

    from calm.hrm_text_158.bit_linear import BitLinear

    return {
        name
        for name, module in model.named_modules()
        if isinstance(module, BitLinear)
    }


def names_sha256(names: list[str] | set[str]) -> str:
    blob = "\0".join(sorted(str(n) for n in names)).encode()
    return hashlib.sha256(blob).hexdigest()


def config_digest(config: Mapping[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(config, sort_keys=True, default=str).encode()
    ).hexdigest()


def tensor_state_summary(tensor_states: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for name in sorted(tensor_states):
        st = tensor_states[name]
        q = getattr(st, "q_levels", None)
        device = str(q.device) if q is not None and hasattr(q, "device") else None
        dtype = str(q.dtype) if q is not None and hasattr(q, "dtype") else None
        shape = list(q.shape) if q is not None and hasattr(q, "shape") else None
        rows.append(
            {"name": str(name), "shape": shape, "dtype": dtype, "device": device}
        )
    return rows


def assert_complete_eligible_inventory(
    *,
    model: Any,
    eligible_modules: Mapping[str, Any],
    tensor_states: Mapping[str, Any],
    eligible_scope: str = ELIGIBLE_SCOPE,
) -> dict[str, Any]:
    if eligible_scope != "all-bitlinear":
        raise IdentityRefuse(f"eligible_scope must be all-bitlinear; got {eligible_scope!r}")
    independent = derive_all_bitlinear_name_set(model)
    selected = set(str(k) for k in eligible_modules.keys())
    state_keys = set(str(k) for k in tensor_states.keys())
    if not selected:
        raise IdentityRefuse("eligible_module_count must be > 0")
    if selected != independent:
        raise IdentityRefuse(
            "COMPLETE-INVENTORY eligible name mismatch: "
            f"selected={sorted(selected)} independent={sorted(independent)}"
        )
    if state_keys != selected:
        raise IdentityRefuse(
            "tensor-state keys must equal selected eligible set: "
            f"states={sorted(state_keys)} selected={sorted(selected)}"
        )
    return {
        "eligible_module_count": len(selected),
        "eligible_module_names": sorted(selected),
        "eligible_module_names_sha256": names_sha256(selected),
        "independent_eligible_count": len(independent),
        "independent_eligible_names_sha256": names_sha256(independent),
        "tensor_state_key_count": len(state_keys),
    }


def materialize_identity_inventory(
    *,
    parent_path: Path,
    parent_sha256_expected: str,
    parent_sha256_pre_load: str,
    parent_sha256_post_load: str,
    config: Mapping[str, Any],
    provenance: Mapping[str, Mapping[str, Any]],
    eligible_inventory: Mapping[str, Any],
    tensor_states: Mapping[str, Any],
    device: Any,
) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "parent_path": str(parent_path),
        "parent_sha256_expected": str(parent_sha256_expected),
        "parent_sha256_pre_load": str(parent_sha256_pre_load),
        "parent_sha256_post_load_echo": str(parent_sha256_post_load),
        "model_config_digest": config_digest(config),
        "model_config_field_provenance": dict(provenance),
        "model_state_key_count": None,  # filled by caller when known
        "eligible_module_count": int(eligible_inventory["eligible_module_count"]),
        "eligible_module_names_sha256": str(
            eligible_inventory["eligible_module_names_sha256"]
        ),
        "eligible_module_names": list(eligible_inventory["eligible_module_names"]),
        "tensor_state_summary": tensor_state_summary(tensor_states),
        "device": str(device),
    }


def bind_cadence_saver(
    *,
    source_pin: str,
    config: Mapping[str, Any],
    eligible_scope: str = ELIGIBLE_SCOPE,
) -> CadenceSaverFn:
    """Thin binder — reuses science_driver.default_cadence_saver semantics."""

    def _saver(*, path: Path, model: Any, event: Any, **_kw: Any) -> Path:
        return default_cadence_saver(
            path=path,
            model=model,
            event=event,
            config=dict(config),
            source_pin=str(source_pin),
            eligible_scope=str(eligible_scope),
        )

    return _saver


def materialize_run_arms_live_bundle(
    *,
    parent_path: Path | str,
    expected_parent_sha256: str,
    device: str | Any = "cpu",
    eligible_scope: str = ELIGIBLE_SCOPE,
    batch_size: int = 1,
    curriculum_seed: int = 43,
) -> RunArmsLiveBundle:
    """Full identity-gated materialization. Fail-closed before runner/GPU work."""

    import torch
    from scripts.hrm_text_158_bounded_delta_acquisition_probe import (
        build_identity_full_support_batches,
        build_model_from_checkpoint,
        derive_tensor_states_and_check_init_fidelity,
        select_eligible_bitlinears,
    )

    parent_path = Path(parent_path)
    ckpt, pre_sha, post_sha = load_verified_parent_checkpoint(
        parent_path, expected_sha256=str(expected_parent_sha256)
    )
    config = dict(ckpt["config"])
    provenance = inventory_model_config_field_provenance(config)
    assert_no_default_applied(provenance)
    assert_use_ternary_bulk_true(config)

    torch_device = torch.device(str(device))
    model, tok, cfg = build_model_from_checkpoint(ckpt, torch_device)
    model_state_key_count = len(dict(ckpt.get("model_state") or {}))

    eligible = select_eligible_bitlinears(model, eligible_scope=str(eligible_scope))
    states, fidelity = derive_tensor_states_and_check_init_fidelity(
        eligible, threshold=0.0
    )
    if not bool(fidelity.get("all_pass", False)):
        raise IdentityRefuse("tensor-state init fidelity failed")
    eligible_inv = assert_complete_eligible_inventory(
        model=model,
        eligible_modules=eligible,
        tensor_states=states,
        eligible_scope=str(eligible_scope),
    )
    support_batches, _proof = build_identity_full_support_batches(
        tok=tok,
        max_len=int(getattr(cfg, "max_seq_len", 64) or 64),
        batch_size=int(batch_size),
        curriculum_seed=int(curriculum_seed),
        device=torch_device,
    )
    if not support_batches:
        raise IdentityRefuse("empty support batches after materialization")
    batch = support_batches[0]["batch"]

    inventory = materialize_identity_inventory(
        parent_path=parent_path,
        parent_sha256_expected=str(expected_parent_sha256),
        parent_sha256_pre_load=pre_sha,
        parent_sha256_post_load=post_sha,
        config=config,
        provenance=provenance,
        eligible_inventory=eligible_inv,
        tensor_states=states,
        device=torch_device,
    )
    inventory["model_state_key_count"] = int(model_state_key_count)
    inventory["build_identity_full_support_batches"] = (
        "scripts/hrm_text_158_bounded_delta_acquisition_probe.py:778"
    )

    saver = bind_cadence_saver(
        source_pin=str(expected_parent_sha256),
        config=config,
        eligible_scope=str(eligible_scope),
    )
    return RunArmsLiveBundle(
        model=model,
        tok=tok,
        cfg=cfg,
        batch=batch,
        tensor_states=states,
        eligible_modules=eligible,
        device=torch_device,
        parent_path=parent_path,
        parent_sha256_expected=str(expected_parent_sha256),
        parent_sha256_pre_load=pre_sha,
        parent_sha256_post_load=post_sha,
        identity_inventory=inventory,
        cadence_saver=saver,
        config=config,
        fidelity_report=fidelity,
    )


def run_readonly_actual_parent_preflight(
    *,
    repo_root: Path | str,
    parent_relpath: str,
    expected_parent_sha256: str = PARENT_SHA256_FULL,
    device: str = "cpu",
) -> dict[str, Any]:
    """READ-ONLY CPU identity preflight against the ACTUAL pinned parent."""

    root = Path(repo_root)
    parent = root / parent_relpath
    bundle = materialize_run_arms_live_bundle(
        parent_path=parent,
        expected_parent_sha256=expected_parent_sha256,
        device=device,
    )
    return {
        "status": "OK",
        "parent_path": str(parent.resolve()),
        "parent_sha256": expected_parent_sha256,
        "identity_inventory": bundle.identity_inventory,
        "eligible_module_count": len(bundle.eligible_modules),
        "fidelity_all_pass": bool(bundle.fidelity_report.get("all_pass")),
    }


def assert_formal_canonical_params(*, t_cut: int, runway_steps: int, W: int) -> None:
    from calm.hrm_text_158.native_full_stack.forgotten_accum_training_equivalence_contracts import (
        RUNWAY_STEPS,
        T_CUT,
        W_REWARM_STEPS,
    )

    if int(t_cut) != int(T_CUT):
        raise ValueError(f"PREFLIGHT_REFUSE: formal t_cut must be {T_CUT}")
    if int(runway_steps) != int(RUNWAY_STEPS):
        raise ValueError(f"PREFLIGHT_REFUSE: formal runway_steps must be {RUNWAY_STEPS}")
    if int(W) != int(W_REWARM_STEPS):
        raise ValueError(f"PREFLIGHT_REFUSE: formal W must be {W_REWARM_STEPS}")


__all__ = [
    "SCHEMA",
    "SOURCE_CHECKPOINT",
    "SOURCE_DEFAULT_APPLIED",
    "IdentityRefuse",
    "RunArmsLiveBundle",
    "inventory_model_config_field_provenance",
    "assert_no_default_applied",
    "assert_use_ternary_bulk_true",
    "verify_parent_bytes_sha256",
    "load_verified_parent_checkpoint",
    "derive_all_bitlinear_name_set",
    "assert_complete_eligible_inventory",
    "materialize_identity_inventory",
    "bind_cadence_saver",
    "materialize_run_arms_live_bundle",
    "run_readonly_actual_parent_preflight",
    "assert_formal_canonical_params",
]
